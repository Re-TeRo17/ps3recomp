/*
 * ps3recomp - platform-neutral RSX draw engine (see rsx_draw_engine.h)
 *
 * The orchestration carried across from libs/video/rsx_live_draw.c, the
 * NV4097 -> D3D12 engine a title has shipped on. Line numbers in the comments
 * below are that file's, so the two can be read side by side; the file itself
 * is vendored and stays exactly as it is.
 *
 * What is deliberately left out, because it is not renderer behaviour a title
 * needs: the shader disk cache, movie mode and its compositor, the a010
 * probe, and every YZ_PERF_PROFILE block.
 */
#include "rsx_draw_engine.h"
#include "rsx_draw_snap.h"
#include "rsx_capture.h"

#include "rsx_fp_decompiler.h"
#include "rsx_primitives.h"
#include "rsx_rt.h"
#include "rsx_restart_cuts.h"
#include "rsx_vertex_formats.h"
#include <math.h>
#include <time.h>
#ifdef _WIN32
#include <intrin.h>
#endif
#include "rsx_vp_decompiler.h"
extern volatile int g_rsx_pass_shadow;
extern volatile u32 g_rsx_pass_cw, g_rsx_pass_ch;
#ifndef CUT88_DEF_BMUL
#define CUT88_DEF_BMUL 2.0f   /* cut88 (01/11): D3D12 slope-scaled bias of the Z16 shadow passes x3, bias x2 (RSX_SHADOW_BIAS_FIX=0 = the game's raw values) */
#define CUT88_DEF_SMUL 3.0f
#endif
static unsigned s_vp_branch_bits_reg;   /* last SET_TRANSFORM_BRANCH_BITS (0x1FF8) as written by the dispatcher (D side) */
static unsigned s_vp_bb_draw;           /* the value the draw being run sees (job snapshot under RSX_DRAW_SPLIT=2) */

#include <stdio.h>
#include <stdlib.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <string.h>

/* Environment switches are read where they are used, some of them per call:
 * answer from the pointer-keyed cache (ps3emu/env_cache.h). */
#include "ps3emu/vm_watch.h"
#include "ps3emu/beacon.h"
#include "ps3emu/env_cache.h"
#define getenv(name) ps3_env(name)

/* Guest memory and the RSX offset resolvers, declared the way every backend
 * that reads guest data declares them (rsx_vertex_fetch.h). */
extern u8* vm_base;
u32 cellGcmResolveLocated(int local, u32 offset);
/* Non-zero only when the offset's page is in the IO table, so a caller can
 * tell IO-mapped main memory from "assume VRAM". */
u32 cellGcmResolveIO(u32 offset);
/* The guest VM's OOB guard: the size a host backed, or 0 for "the whole 32-bit
 * space is backed and no check is needed" (runtime/memory/vm.h). */
extern u32 ppu_vm_size;

/* ------------------------------------------------------------------------- */

#define ENG_MAX_SURFACES   512   /* surf54 31/10: was 192 (full after 5 min in the arena); was 64: Stratus City keeps ~60 small live targets (32x32 / 64x64 pools), the table filled at ~35 s */
#define ENG_MAX_ZDEPTHS    128   /* was 64, same reason (per-object depth maps) */
#define ENG_MAX_TEXTURES   RSX_DRAW_ENGINE_TEXTURE_CACHE
#define ENG_MAX_VTEXTURES  64
#define ENG_MAX_PIPELINES  16384
/* Open-addressed indexes over the texture and pipeline caches (entry + 1, 0
 * empty), at most half full. */
#define ENG_TEX_INDEX      16384u
#define ENG_PIPE_INDEX     32768u   /* open addressing: keep it 2x ENG_MAX_PIPELINES (a full index never ends a miss) */
#define ENG_MAX_BATCHES    256
#define ENG_VERT_STRIDE    (RSX_DSP_NUM_VERTEX_ATTR * 16u)
#define ENG_INVALID        0xFFFFFFFFu
/* Guest textures larger than this are a misdecoded register, not an image;
 * rsx_live_draw.c rejects the same 4096 bound (2033). */
#define ENG_MAX_TEX_DIM    4096u
/* A surface declaration beyond this is a guest pointer read as clip
 * dimensions. rsx_live_draw.c's surface-guard (2625) preserves the existing
 * target rather than destroying a live one. */
#define ENG_MAX_SURFACE_DIM 8192u

#define ENG_VP_CB_BYTES    ((RSX_DSP_NUM_CONSTANTS + 2 + 4) * 16u)   /* +4: RSX_SSR previous-frame matrix */
#define ENG_FP_MAX_BYTES   0x10000u

#define M_ALPHA_TEST_ENABLE   0x0304
#define M_ALPHA_FUNC          0x0308
#define M_ALPHA_REF           0x030C
#define M_BLEND_ENABLE        0x0310
#define M_BLEND_SFACTOR       0x0314
#define M_BLEND_DFACTOR       0x0318
#define M_BLEND_EQUATION      0x0320
#define M_COLOR_MASK          0x0324
#define M_STENCIL_TEST_ENABLE 0x0328
#define M_STENCIL_MASK        0x032C
#define M_STENCIL_FUNC        0x0330
#define M_STENCIL_FUNC_REF    0x0334
#define M_STENCIL_FUNC_MASK   0x0338
#define M_STENCIL_OP_FAIL     0x033C
#define M_STENCIL_OP_ZFAIL    0x0340
#define M_STENCIL_OP_ZPASS    0x0344
#define M_TWO_SIDED_STENCIL   0x0348
#define M_BACK_STENCIL_FUNC   0x0350
#define M_BACK_STENCIL_OP_FAIL  0x035C
#define M_BACK_STENCIL_OP_ZFAIL 0x0360
#define M_BACK_STENCIL_OP_ZPASS 0x0364
#define M_SCISSOR_HORIZONTAL  0x08C0
#define M_SCISSOR_VERTICAL    0x08C4
#define M_DEPTH_FUNC          0x0A6C
#define M_DEPTH_WRITE         0x0A70
#define M_DEPTH_TEST_ENABLE   0x0A74
#define M_ZSTENCIL_CLEAR      0x1D8C

/* ---- module state -------------------------------------------------------- */

#define ENG_SURF_PARK 3   /* parked size/format variants kept per surface offset */

typedef struct {
    u32 w, h;
    rsx_be_format fmt;
    u32 handle;
} eng_surface_variant;

typedef struct {
    u32 location, offset;
    u32 w, h;
    rsx_be_format fmt;
    u32 handle;
    /* Targets this offset used before its size/format moved, oldest first.
     * Still live (never released while parked), so a draw recorded against
     * one stays valid; released only on eviction or shutdown, through the
     * backend's own deferred release. */
    eng_surface_variant park[ENG_SURF_PARK];
    u32 park_n;
    u8 aa;          /* SET_SURFACE_FORMAT antialias mode it was last drawn with */
    u8 blit_only;   /* made only by NV3089 copies, never drawn into */
    u32 alias_off;  /* guest address of the clip origin (offset + clip_y*pitch + clip_x*4) when the pass draws a window inside the buffer; 0 = none */
    u32 last_frame; /* g.frames when last drawn into or sampled */
} eng_surface;

/* An eng_surface slot released for reuse: matches no location. */
#define ENG_FREE_LOCATION 0xFFFFFFFFu

typedef struct {
    u32 location, offset;
    u32 w, h;
    u32 handle;
    u32 snapshot;          /* sampleable copy, or 0                        */
    int cleared;
    int had_write;
    int snapshot_valid;
    u32 packed;            /* RGBA8 copy as the D24S8 bytes read as colour */
    int packed_valid;
    u32 snap_w, snap_h;    /* the region each copy covers (see below)      */
    u32 packed_w, packed_h;
    u32 last_frame;        /* g.frames when last bound or sampled          */
} eng_zdepth;

typedef struct {
    u32 location, offset, format, width, height, pitch, remap, cubemap;
    u32 handle;
    u64 content_hash;
    u32 last_hash_frame;
    u64 last_use_serial;
    u64 watch_stamp;       /* vm_watch_arm's stamp when the bytes were last hashed */
    u32 watched;           /* every page under it was protected then              */
    u32 standin_gen;       /* s_standin_gen when the port was last asked          */
    const void* standin;   /* the stand-in's bytes it was uploaded from, or NULL  */
    u32 span;              /* eng_texture_span for span_levels - 1 mip levels     */
    u32 span_levels;       /* the levels it was worked out for, + 1; 0 none       */
    u32 reups;             /* surf54: times its bytes changed and it was decoded again (RSX_TEX_REUP_LOG) */
} eng_texture;

struct eng_pipe_job;
typedef struct { u64 key; u32 handle; u8 fixed; u64 vp_fnv; u64 vp_hlsl; u64 fp_hlsl; u64 fp_fnv;
                 u8 pending; u8 wpos; struct eng_pipe_job* job; u8 rt_static; } eng_pipeline;
/* The pipeline the last eng_pipeline_get returned (RT_AO: is its vertex program a static world one). */
static const eng_pipeline* s_cur_pipe;

typedef struct { u32 first, count; } eng_batch;

typedef struct {
    u32 location, offset, pitch, width, height;
    int valid;
} eng_display_buffer;

static struct {
    const rsx_draw_backend* be;
    int ready;
    int default_on;
    u32 width, height;

    rsx_dispatch rsx;
    rsx_dispatch* rv;        /* 27/10 walkF2: register view the draw path READS: &rsx (live, default) or the snapshot view
                              * (RSX_DRAW_SPLIT=1, only inside sink_end). Writes by the dispatcher always go to rsx. */

    eng_surface surfaces[ENG_MAX_SURFACES];
    u32 n_surfaces;
    eng_zdepth zdepths[ENG_MAX_ZDEPTHS];
    u32 n_zdepths;
    eng_texture textures[ENG_MAX_TEXTURES];
    u32 n_textures;
    eng_pipeline pipelines[ENG_MAX_PIPELINES];
    u32 n_pipelines;
    u32 tex_index[ENG_TEX_INDEX];
    u32 pipe_index[ENG_PIPE_INDEX];
    eng_display_buffer display_buffers[8];

    u32 frames;
    u64 texture_use_serial;
    u32 guest_draws;
    u32 last_guest_draws;
    u32 last_present_surface;
    u32 last_flip_buffer;
    u32 q_cur;               /* open occlusion query (backend counter), or 0 */
    u32 q_attempts, q_draws; /* draws the game issued / the backend counted */
    u32 q_unflushed;         /* reports handed to the backend since its last submit */
    u32 sink_flips;          /* flips decoded from the FIFO (0xE944), in draw order */

    /* staging: decoded texture levels, the constant blocks, the index list */
    u8* tex_staging;
    u32 tex_staging_cap;
    u8* vp_cb;
    u32 vp_cb_gen;           /* rsx.const_gen vp_cb was filled at        */
    int vp_cb_valid;
    int vp_cb_bound;         /* the backend's last staged block is vp_cb */
    u8* fp_cb;
    u32 fp_cb_cap;
    u32* indices;
    u32 index_cap;

    rsx_fp_constant_block fp_constants;
} g;
#ifdef _WIN32
#define ENG_TLS __declspec(thread)
#else
#define ENG_TLS _Thread_local
#endif
static ENG_TLS rsx_dispatch* t_rv;   /* RSX_DRAW_WORKERS: a prep worker (or B2 stealing a prep) reads its own register view */
static int s_wk_n;               /* RSX_DRAW_WORKERS: prep worker threads (0 = off); B2 steals preps too */
static volatile int s_arm_lock;   /* serialises vm_watch_arm while prep workers run */
#define RV (t_rv ? t_rv : g.rv)   /* every read of the register file by the draw/clear/flip path goes through RV (see rsx_draw_snap.h) */

/* ---- walker diagnostics and opt-in fast paths (26/10, branch walker-perf) ----
 * RSX_WPROF=1: rdtsc time of the phases of a draw (sink_end_impl), of the FIFO pass
 * and of the D3D12 record/encode, printed every 240 presents as [wprof] (ms per frame).
 * RSX_WALKER_FAST=1: (a) a per-texture-unit memo that skips the scan of g.surfaces when
 * no render target sits at the unit's address (invalidated whenever a surface is created
 * or freed), (b) no wall-clock reads around every draw (the [hitch] draw ms reads 0).
 * Both are off by default; behaviour with them off is unchanged. */
int g_wp_fl = -1; unsigned long long g_wp_c[6], g_wp_meths, g_wp_runs, g_wp_runw, g_wp_cmds;
unsigned long long g_wp_fifo_cyc, g_wp_enc_cyc, g_wp_fifo_calls;   /* cellGcmSys.c / rsx_d3d12_engine.c */
static int s_wp_on = -1, s_wk_fast = -1;
static unsigned long long s_wp_cyc[64], s_wp_draws, s_wp_nscan, s_wp_nmiss, s_wp_n2[48];
int g_wp2_on; unsigned long long g_wp2_samp_cyc;   /* prof102: set when RSX_WPROF=2; the D3D12 backend adds its sampler-slot time */
static unsigned s_tdec_pass, s_wp_dk;   /* prof102: which pass decodes a texture unit; sample counter */
#define WP2_CNT(i) do { if (s_wp_on > 1) s_wp_n2[i]++; } while (0)
/* RSX_WPROF=2 (metro131): finer rdtsc buckets [wprof5]; cycle slots 8.. , counters s_wp_n2. */
#define WP2_BEGIN(v) unsigned long long v = s_wp_on > 1 ? wp_tsc() : 0
#define WP2_END(v, k) do { if (s_wp_on > 1) s_wp_cyc[k] += wp_tsc() - (v); } while (0)
static u32 s_zgen = 1;                                       /* walk37: bumped when a depth target is created or moved to another address */
static struct { u32 loc, off, gen; } s_tex_znomatch[16];      /* per texture unit: no depth target at loc/off as of s_zgen */
static u32 s_surf_gen = 1;                                   /* bumped when the surface set changes */
static struct { u32 loc, off, gen; } s_tex_nomatch[16];       /* per texture unit: no surface at loc/off as of gen */
/* RSX_SURF_HASH=1: first live surface at (location, offset), memoised per surface-set generation. */
static int s_surf_hash_on = -1;
static struct { u32 loc, off, gen, idx; } s_sh[512];
static u32 eng_surf_first(u32 loc, u32 off);
static inline unsigned long long wp_tsc(void)
{
#if defined(_WIN32) || defined(__x86_64__) || defined(__i386__)
# if defined(_WIN32)
    return __rdtsc();
# else
    unsigned lo, hi; __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi)); return ((unsigned long long)hi << 32) | lo;
# endif
#else
    return 0;
#endif
}
/* metro131 RSX_TEX_DEC_MEMO=1 (default OFF): the draw-side view's texture unit u, decoded once per change of its registers (snap_apply marks them in
 * g_snap_tex_dirty) instead of by every consumer of every draw. Only for the B2 view; anything else decodes as before. */
static inline void eng_get_tex_impl(const rsx_dispatch* rsx, u32 u, rsx_dsp_texture* t);
static inline void eng_get_tex(const rsx_dispatch* rsx, u32 u, rsx_dsp_texture* t)
{
    if (s_wp_on > 1) {   /* prof102: decode calls per pass; 1 call in 16 is timed (x16) */
        s_wp_n2[10 + s_tdec_pass]++;
        if ((++s_wp_dk & 15u) == 0) { const unsigned long long a_ = wp_tsc(); eng_get_tex_impl(rsx, u, t); s_wp_cyc[40 + s_tdec_pass] += (wp_tsc() - a_) * 16ull; return; }
    }
    eng_get_tex_impl(rsx, u, t);
}
/* opt102 RSX_TEXMASK_MEMO (1.1: default ON, =0 off, =2: check). (a) Per-unit decode memo: the unit's 9 raw register words (8-word block 0x1A00 + u*0x20 and
 * the size1 word 0x1840 + u*4) are compared with the ones the memoised rsx_dsp_texture was decoded from; equal = same struct, no decode.
 * Pure function of those words, so valid for any view (per thread). (b) eng_texmasks_get below: the masks of eng_pipeline_get. */
static int s_tm_mode = -1;                       /* 0 off, 1 on, 2 on + compare with a fresh computation */
static ENG_TLS int s_tm_raw;                     /* set while a check computes the old way: no memo at all */
static unsigned long long s_tm_dec_n, s_tm_dec_bad, s_tm_mask_n, s_tm_mask_hit, s_tm_mask_bad, s_tm_vt_n, s_tm_vt_bad;
static ENG_TLS struct { u32 raw[9]; int valid; rsx_dsp_texture t; } s_tdm[RSX_DSP_NUM_TEXTURES];
static inline int eng_tm_mode(void)
{
    if (s_tm_mode < 0) { const char* e = getenv("RSX_TEXMASK_MEMO"); s_tm_mode = (e && e[0]) ? (e[0] == '0' ? 0 : e[0] == '2' ? 2 : 1) : 1; }   /* 1.1: default ON */
    return s_tm_mode;
}
static void eng_tm_report(void)
{
    fprintf(stderr, "[texmask-check] mode %d: mask memo reads %llu (hits %llu, mismatches %llu); unit decode checks %llu (mismatches %llu); vtex checks %llu (mismatches %llu) frame %u\n",
            s_tm_mode, s_tm_mask_n, s_tm_mask_hit, s_tm_mask_bad, s_tm_dec_n, s_tm_dec_bad, s_tm_vt_n, s_tm_vt_bad, g.frames);
}
static inline void eng_get_tex_impl(const rsx_dispatch* rsx, u32 u, rsx_dsp_texture* t)
{
    if (eng_tm_mode() && !s_tm_raw && u < RSX_DSP_NUM_TEXTURES && !g_rsx_dsp_trace) {
        u32 raw[9];
        memcpy(raw, &rsx->regs[(0x1A00u + u * 0x20u) >> 2], 8 * sizeof(u32));
        raw[8] = rsx->regs[(0x1840u + u * 4u) >> 2];
        if (s_tdm[u].valid && !memcmp(raw, s_tdm[u].raw, sizeof raw)) {
            if (s_tm_mode == 2) {
                rsx_dsp_texture f; rsx_dsp_get_texture(rsx, u, &f);
                s_tm_dec_n++;
                if (memcmp(&f, &s_tdm[u].t, sizeof f)) { if (s_tm_dec_bad++ < 16) fprintf(stderr, "[texmask-check] MISMATCH unit decode %u frame %u\n", u, g.frames); }
            }
            *t = s_tdm[u].t;
            return;
        }
        rsx_dsp_get_texture(rsx, u, t);
        memcpy(s_tdm[u].raw, raw, sizeof raw); s_tdm[u].t = *t; s_tdm[u].valid = 1;
        return;
    }
    rsx_dsp_get_texture(rsx, u, t);
}
/* ---- RSX_B2_PROF=1 (03/11 b2prof) ----------------------------------------------------------------------------------------------
 * Exclusive rdtsc buckets of the work thread B2 does per job (b2_exec -> sink_end_impl and what it calls, the waits of the D3D12
 * engine included). The time since the last mark goes to the current bucket: RSX_B2P_SET changes the base bucket (and clears the
 * nesting stack), ENTER/LEAVE nest a callee. Only thread B2 accumulates (t_b2p_thr). Printed every 2 s as [b2prof] by b2_report;
 * each mark's own cost is calibrated at start and taken out of the bucket it opens. Off: g_b2p_on == 0, every hook is one test. */
int g_b2p_on;
static ENG_TLS int t_b2p_thr;
static unsigned long long s_b2p_cyc[B2P_N], s_b2p_nm[B2P_N], s_b2p_cnt[B2C_N], s_b2p_last;
static int s_b2p_cur, s_b2p_sp, s_b2p_stk[24];
static inline void b2p_mark(void) { const unsigned long long t = wp_tsc(); s_b2p_cyc[s_b2p_cur] += t - s_b2p_last; s_b2p_last = t; }
void rsx_b2p_set(int k) { if (!t_b2p_thr) return; b2p_mark(); s_b2p_sp = 0; s_b2p_cur = k; s_b2p_nm[k]++; }
void rsx_b2p_enter(int k) { if (!t_b2p_thr) return; b2p_mark(); if (s_b2p_sp < 24) s_b2p_stk[s_b2p_sp++] = s_b2p_cur; s_b2p_cur = k; s_b2p_nm[k]++; }
void rsx_b2p_leave(void) { if (!t_b2p_thr) return; b2p_mark(); if (s_b2p_sp > 0) s_b2p_cur = s_b2p_stk[--s_b2p_sp]; s_b2p_nm[s_b2p_cur]++; }
void rsx_b2p_add(int c, unsigned long long v) { if (t_b2p_thr) s_b2p_cnt[c] += v; }
static void b2p_job_begin(void) { b2p_mark(); s_b2p_sp = 0; s_b2p_cur = B2P_JOBPRE; s_b2p_nm[B2P_JOBPRE]++; }
static void b2p_job_end(void) { b2p_mark(); s_b2p_sp = 0; s_b2p_cur = B2P_IDLE; s_b2p_nm[B2P_IDLE]++; }
static double s_b2p_cost = -1.0;   /* cycles of one mark (calibrated on B2 at start) */
static void b2p_calibrate(void)
{
    t_b2p_thr = 1;
    s_b2p_last = wp_tsc(); s_b2p_cur = B2P_IDLE;
    const unsigned long long a = wp_tsc();
    for (int i = 0; i < 20000; i++) { rsx_b2p_enter(B2P_TAIL); rsx_b2p_leave(); }
    const unsigned long long b = wp_tsc();
    s_b2p_cost = (double)(b - a) / 40000.0;
    memset(s_b2p_cyc, 0, sizeof s_b2p_cyc); memset(s_b2p_nm, 0, sizeof s_b2p_nm); memset(s_b2p_cnt, 0, sizeof s_b2p_cnt);
    s_b2p_last = wp_tsc(); s_b2p_cur = B2P_IDLE; s_b2p_sp = 0;
}

static void eng_wk_env(void)
{
    { const char* e2 = getenv("RSX_SURF_HASH"); s_surf_hash_on = (!e2 || !e2[0] || e2[0] != '0') ? 1 : 0;   /* default ON (28/10) */ }
    const char* e = getenv("RSX_WPROF"); s_wp_on = (e && e[0] && e[0] != '0') ? (e[0] == '2' ? 2 : 1) : 0; g_wp2_on = s_wp_on > 1;
    e = getenv("RSX_WALKER_FAST"); s_wk_fast = (!e || !e[0] || e[0] != '0') ? 1 : 0;   /* default ON (28/10) */
}

/* The decompilers' HLSL. The VP decompiler builds bodies up to 192 KB. */
static char s_vs_hlsl[256 * 1024];
static char s_ps_hlsl[256 * 1024];

/* The draw accumulator: one BEGIN_END group's batches, references and cuts. */
static struct {
    eng_batch arr[ENG_MAX_BATCHES];
    u32 n_arr;
    eng_batch idx[ENG_MAX_BATCHES];
    u32 n_idx;
    u32 n_packets;

    rsx_vertex_ref* refs;
    u32 n_refs, cap_refs, n_source_refs;
    int refs_remapped;
    rsx_vertex_remap ref_remap;

    u32* cuts;
    u32 n_cuts, cap_cuts;

    u8* verts;
    u64 verts_cap;
    u32 n_verts;
    const u8* out_verts;     /* what the draw uploads: verts, or a cached copy */

    rsx_vertex_layout_plan layout;
    rsx_vertex_fetch_plan fetch_plan;
    int fetch_ok;

    /* INLINE_ARRAY stream for this group, when the vertices came down the
     * FIFO rather than from a vertex array. */
    const u8* inl;
    u32 inl_bytes;
} dc;

/* ---- guest memory -------------------------------------------------------- */

/* rsx_dsp_* locations are RSX_LOCATION_LOCAL = 0 / _MAIN = 1;
 * cellGcmResolveLocated's argument is the opposite sense (1 = local), and
 * getting that backwards resolves every VRAM object through the IO table. */
static rsx_vertex_guest_ptr_fn s_guest_reader;
static void* s_guest_user;

void rsx_draw_engine_set_guest_memory(rsx_vertex_guest_ptr_fn reader, void* user)
{
    s_guest_reader = reader;
    s_guest_user = user;
}

static rsx_guest_watch_fn s_guest_watch;
void rsx_draw_engine_set_guest_watch(rsx_guest_watch_fn fn)
{
    s_guest_watch = fn;
}

static const u8* eng_guest_ptr(void* user, u32 location, u32 offset,
                               u32 min_bytes)
{
    (void)user;
    if (!min_bytes || min_bytes > (256u << 20)) return NULL;
    if (s_guest_reader) return s_guest_reader(s_guest_user, location, offset, min_bytes);
    if (!vm_base) return NULL;
    const u32 ea = cellGcmResolveLocated(location == RSX_LOCATION_LOCAL, offset);
    if (!ea) return NULL;
    if (ppu_vm_size && (u64)ea + min_bytes > ppu_vm_size) return NULL;
    if (g_rsx_capture_on) rsx_capture_read(location, offset, min_bytes);
    return vm_base + ea;
}

/* tools/rsx_replay: start from a captured register file, transform program
 * and constants, as rsx_dispatch_seed_* lay them in. */
void rsx_draw_engine_seed_state(const u32* regs, u32 nregs, const u32* vp, u32 nvp,
                                const u32* constants, u32 nconst_words)
{
    if (regs) rsx_dispatch_seed_registers(&g.rsx, regs, nregs);
    if (vp) rsx_dispatch_seed_transform_program(&g.rsx, vp, nvp);
    if (constants) rsx_dispatch_seed_transform_constants(&g.rsx, constants, nconst_words);
}

static u64 eng_fnv1a(const void* data, u32 n, u64 hash)
{
    const u8* p = (const u8*)data;
    for (u32 i = 0; i < n; i++) { hash ^= p[i]; hash *= 1099511628211ull; }
    return hash;
}

/* ---- TOD_PSO_NORMALIZE (perf115) ------------------------------------------------------------------------------
 * Idea and constants adapted from RACiT's runtime patch (MIT, (c) 2026 HighDrexler; ps3recomp-0001-racacit-runtime-fixes
 * P:8542-8560): what a DISABLED stage would use is not state. The title leaves junk in the blend factor / compare /
 * stencil registers while the stage is off, and the pipeline key hashes every one of those fields, so each junk value
 * is a new pipeline (a driver compile = a hitch, and a bigger PSO cache) that builds the identical D3D12 state.
 * Off: the field reads as the reset value its disabled stage implies. Safe because rsx_d3d12_engine.c (PSO builder) and
 * every reader in this file already gate on the enable bit (blend factors on blend_enable, depth_func/depth_write on
 * depth_test, cull_face on cull_enable, the s_ and bs_ fields on stencil_enable / stencil_two_sided, alpha_func on alpha_test_enable).
 * front_face is kept (the rasteriser reads it regardless of culling).
 * Default ON since 1.0 (pirate base: 225 -> 173 pipelines, same image); TOD_PSO_NORMALIZE=0 turns it off.
 * TOD_PSO_NORMALIZE_STATS=1 (works on or off) counts, over the session, the
 * distinct pipeline keys with and without the normalisation: "[pso-norm] frames=N distinct raw=A norm=B (-X%)" every 600
 * frames and at exit. In stats mode the per-draw key memo is bypassed (slower; diagnostic only). */
static int eng_pso_norm_mode(void)
{
    static int m = -1;
    if (m < 0) { const char* e = getenv("TOD_PSO_NORMALIZE"); m = (e && e[0] == '0') ? 0 : 1; }
    return m;
}
static int eng_pso_norm_stats(void)
{
    static int m = -1;
    if (m < 0) { const char* e = getenv("TOD_PSO_NORMALIZE_STATS"); m = (e && e[0] && e[0] != '0') ? 1 : 0; }
    return m;
}
static void eng_pso_normalize_state(rsx_be_render_state* rs)
{
    if (!rs->blend_enable) {
        rs->sf_rgb = rs->sf_a = 0x0001u; rs->df_rgb = rs->df_a = 0x0000u;     /* ONE, ZERO */
        rs->eq_rgb = rs->eq_a = 0x8006u;                                      /* FUNC_ADD */
    }
    if (!rs->alpha_test_enable) rs->alpha_func = 0x0207u;                     /* ALWAYS */
    if (!rs->depth_test) { rs->depth_func = 0x0207u; rs->depth_write = 0; }   /* no test, no write (the builder already says so) */
    if (!rs->cull_enable) rs->cull_face = 0x0405u;                            /* BACK; front_face stays */
    if (!rs->stencil_enable) {
        rs->stencil_two_sided = 0;
        rs->s_func = 0x0207u; rs->s_func_mask = rs->s_write_mask = 0xFFu;
        rs->s_fail = rs->s_zfail = rs->s_zpass = 0x1E00u;                     /* KEEP */
    }
    if (!rs->stencil_two_sided) { rs->bs_func = 0; rs->bs_fail = rs->bs_zfail = rs->bs_zpass = 0; }
}
/* stats: the render-state hashes of the LAST decode, without / with the normalisation (the key site combines them with
 * the rest of the key). */
static u64 s_pson_h_raw, s_pson_h_norm;
static void eng_pso_norm_decode_tail(rsx_be_render_state* rs)
{
    const int on = eng_pso_norm_mode();
    if (eng_pso_norm_stats()) {
        rsx_be_render_state a = *rs, b = *rs;
        eng_pso_normalize_state(&b);
        s_pson_h_raw  = rsx_draw_engine_hash_render_state(&a, 0x9E3779B97F4A7C15ull);
        s_pson_h_norm = rsx_draw_engine_hash_render_state(&b, 0x9E3779B97F4A7C15ull);
    }
    if (on) eng_pso_normalize_state(rs);
}
#define ENG_PSON_SLOTS 65536u
static u64 s_pson_set_raw[ENG_PSON_SLOTS], s_pson_set_norm[ENG_PSON_SLOTS];
static u32 s_pson_n_raw, s_pson_n_norm, s_pson_next_frame;
static int s_pson_atexit;
static int eng_pson_insert(u64* set, u32* n, u64 k)
{
    if (!k) k = 1;
    for (u32 i = (u32)(k * 0x9E3779B97F4A7C15ull >> 48) & (ENG_PSON_SLOTS - 1u), c = 0; c < ENG_PSON_SLOTS; i = (i + 1u) & (ENG_PSON_SLOTS - 1u), c++) {
        if (set[i] == k) return 0;
        if (!set[i]) { if (*n >= ENG_PSON_SLOTS - ENG_PSON_SLOTS / 8u) return 0; set[i] = k; (*n)++; return 1; }
    }
    return 0;
}
static void eng_pson_print(u32 frames)
{
    fprintf(stderr, "[pso-norm] frames=%u distinct pipeline keys: raw=%u normalized=%u (%.1f%% fewer) mode=%s\n", frames, s_pson_n_raw, s_pson_n_norm,
            s_pson_n_raw ? 100.0 * (1.0 - (double)s_pson_n_norm / (double)s_pson_n_raw) : 0.0, eng_pso_norm_mode() ? "ON" : "off");
}
static u32 s_pson_last_frames;
static void eng_pson_atexit(void) { eng_pson_print(s_pson_last_frames); }
/* key_pre = the key before the render state was folded in. */
static void eng_pson_key(u64 key_pre, u32 frames)
{
    u64 kr = eng_fnv1a(&s_pson_h_raw, sizeof s_pson_h_raw, key_pre), kn = eng_fnv1a(&s_pson_h_norm, sizeof s_pson_h_norm, key_pre);
    eng_pson_insert(s_pson_set_raw, &s_pson_n_raw, kr);
    eng_pson_insert(s_pson_set_norm, &s_pson_n_norm, kn);
    s_pson_last_frames = frames;
    if (!s_pson_atexit) { s_pson_atexit = 1; atexit(eng_pson_atexit); }
    if (frames >= s_pson_next_frame) { s_pson_next_frame = frames + 600u; eng_pson_print(frames); }
}

/* ---- render state -------------------------------------------------------- */

void rsx_draw_engine_decode_render_state(const rsx_dispatch* rsx,
                                         rsx_be_render_state* rs)
{
    memset(rs, 0, sizeof(*rs));
    rs->alpha_test_enable = rsx_dsp_reg(rsx, M_ALPHA_TEST_ENABLE) & 1;
    rs->alpha_func   = rsx_dsp_reg(rsx, M_ALPHA_FUNC);
    rs->alpha_ref_raw = rsx_dsp_reg(rsx, M_ALPHA_REF);
    rsx_dsp_surface sf;
    rsx_dsp_get_surface(rsx, &sf);
    rs->alpha_ref_format = sf.color_format;
    rs->rt_fp16 = sf.color_format == RSX_SURFACE_FMT_F_W16Z16Y16X16;
    rs->blend_enable = rsx_dsp_reg(rsx, M_BLEND_ENABLE) & 1;
    const u32 sfac = rsx_dsp_reg(rsx, M_BLEND_SFACTOR);
    const u32 dfac = rsx_dsp_reg(rsx, M_BLEND_DFACTOR);
    const u32 eq   = rsx_dsp_reg(rsx, M_BLEND_EQUATION);
    rs->sf_rgb = sfac & 0xFFFF; rs->sf_a = sfac >> 16;
    rs->df_rgb = dfac & 0xFFFF; rs->df_a = dfac >> 16;
    rs->eq_rgb = eq & 0xFFFF;   rs->eq_a = eq >> 16;
    rs->depth_test  = rsx_dsp_reg(rsx, M_DEPTH_TEST_ENABLE) & 1;
    rs->depth_write = rsx_dsp_reg(rsx, M_DEPTH_WRITE) & 1;
    rs->depth_func  = rsx_dsp_reg(rsx, M_DEPTH_FUNC);
    rs->cull_enable = rsx_dsp_reg(rsx, 0x183C) & 1;
    rs->cull_face   = rsx_dsp_reg(rsx, 0x1830);
    rs->front_face  = rsx_dsp_reg(rsx, 0x1834);
    /* RSX_CULL_OFF=1 (diagnostic, 30/10 water): no face culling on any draw. RSX_FRONT_FLIP=1: swap the front face. */
    { static int co = -1, ff = -1; if (co < 0) { co = getenv("RSX_CULL_OFF") ? 1 : 0; ff = getenv("RSX_FRONT_FLIP") ? 1 : 0; }
      if (co) rs->cull_enable = 0;
      if (ff) rs->front_face = (rs->front_face == 0x0900u) ? 0x0901u : 0x0900u; }
    /* RSX_TRANS_NODEPTH=1 (diagnostic, 30/10 water): blended draws that do not write depth also skip the
     * depth test -- with pick=x:y, every transparent draw covering a pixel shows up, hidden or not. */
    { static int tn = -1; if (tn < 0) { const char* e = getenv("RSX_TRANS_NODEPTH"); tn = e ? (atoi(e) == 2 ? 2 : 1) : 0; }
      if (tn == 2 || (tn && rs->blend_enable && !rs->depth_write)) rs->depth_test = 0; }   /* =2: every draw */
    /* Polygon offset (depth bias): ToD draws its shadow maps with it; without
     * it a caster shadows itself on any surface that is not flat (acne). */
    /* Both are floats the game varies nearly per caster: keyed exactly they made
     * ~1300 pipelines out of one shader pair (5300 in 2 minutes of metropolis).
     * Rounded to 2 mantissa bits (4 steps an octave, within ~12%) they key a
     * handful. RSX_POLY_OFFSET_EXACT=1 keeps the exact values. */
    /* 26/10: 2 mantissa bits (up to ~12% off) made the slope bias of the shadow casters too coarse:
     * stripes on the characters in the in-engine cutscenes. A/B l03_s02_know_rt: RSX_POLY_OFFSET=0 clean, EXACT=1 clean, MANT=8 NOT clean, RSX_POLY_SIGN=- (rsx_d3d12_engine.c) clean.
     * RSX_POLY_MANT=<bits> sets the precision (default 2 = unchanged); RSX_POLY_OFFSET_EXACT=1 = 23. 8 bits did NOT cure it (25/10 test), see RSX_POLY_SIGN.
     * RSX_POLY_LOG=1 prints the distinct raw values (first 64) and the distinct count per precision. */
    { static int mant = -1, plog = 0;
      if (mant < 0) { const char* m = getenv("RSX_POLY_MANT"); mant = m ? atoi(m) : 2;
          if (getenv("RSX_POLY_OFFSET_EXACT")) mant = 23; if (mant < 1) mant = 1; if (mant > 23) mant = 23;
          plog = getenv("RSX_POLY_LOG") ? 1 : 0; }
      rs->poly_offset_fill  = rsx_dsp_reg(rsx, 0x0A68) & 1;
      u32 sc = rs->poly_offset_fill ? rsx_dsp_reg(rsx, 0x0A78) : 0;
      u32 bi = rs->poly_offset_fill ? rsx_dsp_reg(rsx, 0x0A7C) : 0;
      if (plog && rs->poly_offset_fill) {
          static u64 seen[4096]; static unsigned ns, nlog;
          u64 k = ((u64)sc << 32) | bi; unsigned i;
          for (i = 0; i < ns; i++) if (seen[i] == k) break;
          if (i == ns && ns < 4096) { seen[ns++] = k;
              if (nlog++ < 64) { float fs, fb; memcpy(&fs, &sc, 4); memcpy(&fb, &bi, 4);
                  fprintf(stderr, "[polyoff] #%u scale=%g (0x%08X) bias=%g (0x%08X)\n", ns, fs, sc, fb, bi); }
              if (ns % 100 == 0) fprintf(stderr, "[polyoff] distinct raw pairs: %u\n", ns); }
      }
      if (mant < 23) { const u32 sh = 23u - (u32)mant; const u32 half = 1u << (sh - 1);
          sc = (sc + half) & ~((1u << sh) - 1u); bi = (bi + half) & ~((1u << sh) - 1u); }
      /* cut88: Z16 shadow passes (g_rsx_pass_shadow: the zeta of the pass is Z16) get their slope/bias multiplied HERE, at draw time:
       * the D3D12 PSO is created/loaded from the cache without knowing the pass, so a multiplier in the PSO builder never acted.
       * RSX_POLY_Z16_MUL (bias) / RSX_POLY_Z16_SMUL (slope). Defaults set below. RSX_CUT88_LOG=1 prints the distinct pairs. */
      if (rs->poly_offset_fill && g_rsx_pass_shadow) {
          static float zm = -1.f, zs = -1.f; static int lg = -1;
          if (zm < 0.f) { const char* fx = getenv("RSX_SHADOW_BIAS_FIX"); const int fxoff = fx && fx[0] == '0'; const char* e = getenv("RSX_POLY_Z16_MUL"); zm = e ? (float)atof(e) : fxoff ? 1.f : CUT88_DEF_BMUL; if (zm < 0.f) zm = 1.f;
              e = getenv("RSX_POLY_Z16_SMUL"); zs = e ? (float)atof(e) : fxoff ? 1.f : CUT88_DEF_SMUL; if (zs < 0.f) zs = 1.f; lg = getenv("RSX_CUT88_LOG") ? 1 : 0; }
          float fs, fb; memcpy(&fs, &sc, 4); memcpy(&fb, &bi, 4);
          if (lg) { static u32 seen[64][3]; static int n; int i; for (i = 0; i < n; i++) if (seen[i][0] == sc && seen[i][1] == bi && seen[i][2] == g_rsx_pass_cw) break;
              if (i == n && n < 64) { seen[n][0] = sc; seen[n][1] = bi; seen[n][2] = g_rsx_pass_cw; n++;
                  fprintf(stderr, "[cut88 poly] shadow pass %ux%u raw scale=%g bias=%g -> x%g/x%g\n", (unsigned)g_rsx_pass_cw, (unsigned)g_rsx_pass_ch, (double)fs, (double)fb, (double)zs, (double)zm); } }
          fs *= zs; fb *= zm; memcpy(&sc, &fs, 4); memcpy(&bi, &fb, 4);
      }
      rs->poly_offset_scale = sc; rs->poly_offset_bias = bi; }
    /* The RAW register: a game-written 0 is a real "write no colour channel"
     * (a depth-prime pass), and rsx_dispatch_init seeds the nv40 reset value
     * so never-written reads as all-on. */
    rs->color_mask  = rsx_dsp_reg(rsx, M_COLOR_MASK);
    rs->stencil_enable    = rsx_dsp_reg(rsx, M_STENCIL_TEST_ENABLE) & 1;
    rs->stencil_two_sided = rsx_dsp_reg(rsx, M_TWO_SIDED_STENCIL) & 1;
    rs->s_func       = rsx_dsp_reg(rsx, M_STENCIL_FUNC);
    rs->s_func_mask  = rsx_dsp_reg(rsx, M_STENCIL_FUNC_MASK) & 0xFF;
    rs->s_write_mask = rsx_dsp_reg(rsx, M_STENCIL_MASK) & 0xFF;
    rs->s_fail       = rsx_dsp_reg(rsx, M_STENCIL_OP_FAIL);
    rs->s_zfail      = rsx_dsp_reg(rsx, M_STENCIL_OP_ZFAIL);
    rs->s_zpass      = rsx_dsp_reg(rsx, M_STENCIL_OP_ZPASS);
    rs->bs_func      = rsx_dsp_reg(rsx, M_BACK_STENCIL_FUNC);
    rs->bs_fail      = rsx_dsp_reg(rsx, M_BACK_STENCIL_OP_FAIL);
    rs->bs_zfail     = rsx_dsp_reg(rsx, M_BACK_STENCIL_OP_ZFAIL);
    rs->bs_zpass     = rsx_dsp_reg(rsx, M_BACK_STENCIL_OP_ZPASS);
    /* RSX_FORCE=<list>: experiment overrides. "nostencil" disables the
     * stencil test, "alpharef0" zeroes the alpha-test reference, "noalpha"
     * disables the alpha test. */
    { static int f = -1, ns = 0, ar = 0, na = 0;
      if (f < 0) { const char* e = getenv("RSX_FORCE"); f = e ? 1 : 0;
          if (e) { ns = strstr(e, "nostencil") != NULL; ar = strstr(e, "alpharef0") != NULL; na = strstr(e, "noalpha") != NULL; } }
      if (f) { if (ns) rs->stencil_enable = 0; if (ar) rs->alpha_ref_raw = 0; if (na) rs->alpha_test_enable = 0; } }
    /* perf115: TOD_PSO_NORMALIZE (default off) / TOD_PSO_NORMALIZE_STATS. Last, so the RSX_FORCE / TRANS_NODEPTH switches count as state. */
    eng_pso_norm_decode_tail(rs);
}

u64 rsx_draw_engine_hash_render_state(const rsx_be_render_state* rs, u64 hash)
{
#define ENG_HASH_FIELD(name) hash = eng_fnv1a(&rs->name, sizeof(rs->name), hash)
    /* The alpha REFERENCE is deliberately absent: it lives in the fragment
     * constant block, so a title animating a fade must not compile a new
     * pipeline per frame. Enable and compare mode still select a variant.
     * Every field is listed by name so struct padding can never be identity
     * (rsx_live_draw.c:3151-3185). */
    ENG_HASH_FIELD(alpha_test_enable);
    ENG_HASH_FIELD(alpha_func);
    ENG_HASH_FIELD(blend_enable);
    ENG_HASH_FIELD(sf_rgb);
    ENG_HASH_FIELD(df_rgb);
    ENG_HASH_FIELD(sf_a);
    ENG_HASH_FIELD(df_a);
    ENG_HASH_FIELD(eq_rgb);
    ENG_HASH_FIELD(eq_a);
    ENG_HASH_FIELD(depth_test);
    ENG_HASH_FIELD(depth_write);
    ENG_HASH_FIELD(depth_func);
    ENG_HASH_FIELD(cull_enable);
    ENG_HASH_FIELD(cull_face);
    ENG_HASH_FIELD(front_face);
    ENG_HASH_FIELD(poly_offset_fill);
    ENG_HASH_FIELD(poly_offset_scale);
    ENG_HASH_FIELD(poly_offset_bias);
    ENG_HASH_FIELD(color_mask);
    ENG_HASH_FIELD(rt_fp16);
    ENG_HASH_FIELD(stencil_enable);
    ENG_HASH_FIELD(stencil_two_sided);
    ENG_HASH_FIELD(s_func);
    ENG_HASH_FIELD(s_func_mask);
    ENG_HASH_FIELD(s_write_mask);
    ENG_HASH_FIELD(s_fail);
    ENG_HASH_FIELD(s_zfail);
    ENG_HASH_FIELD(s_zpass);
    ENG_HASH_FIELD(bs_func);
    ENG_HASH_FIELD(bs_fail);
    ENG_HASH_FIELD(bs_zfail);
    ENG_HASH_FIELD(bs_zpass);
#undef ENG_HASH_FIELD
    return hash;
}

/* ---- topology expansion -------------------------------------------------- */

/* Every expansion is bounded by the restart cuts, so a strip that the guest
 * broke with the sentinel index gets no connecting triangle across the break.
 * Missing that decode is what upstream's aa7fb63 records as "the exploded
 * spiky mesh that occluded the scene". */
u32 rsx_draw_engine_topology_index_count(u32 primitive, u32 source_refs,
                                         const u32* cuts, u32 cut_count)
{
    const u32 segments = cut_count + 1;
    switch (primitive) {
    case RSX_PRIMITIVE_TRIANGLES:
        return source_refs - source_refs % 3u;
    case RSX_PRIMITIVE_TRIANGLE_STRIP:
    case RSX_PRIMITIVE_TRIANGLE_FAN:
    case RSX_PRIMITIVE_POLYGON: {
        u32 total = 0;
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            (void)begin;
            if (count >= 3) total += (count - 2) * 3;
        }
        return total;
    }
    case RSX_PRIMITIVE_QUAD_STRIP: {
        /* Two vertices per quad after the first pair, so a segment of n
         * carries (n - 2) / 2 quads and an odd trailing vertex is dropped. */
        u32 total = 0;
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            (void)begin;
            if (count >= 4) total += ((count - 2u) / 2u) * 6u;
        }
        return total;
    }
    case RSX_PRIMITIVE_QUADS:
        return (source_refs / 4u) * 6u;
    default:
        return 0;
    }
}

static u32 eng_topology_vertex(const u32* occurrence_to_unique, u32 occurrence)
{
    return occurrence_to_unique ? occurrence_to_unique[occurrence] : occurrence;
}

void rsx_draw_engine_write_topology_indices(u32 primitive, u32 source_refs,
                                            const u32* cuts, u32 cut_count,
                                            const u32* occurrence_to_unique,
                                            u32* indices)
{
    u32 write = 0;
    const u32 segments = cut_count + 1;
    switch (primitive) {
    case RSX_PRIMITIVE_TRIANGLES: {
        const u32 count = source_refs - source_refs % 3u;
        for (u32 o = 0; o < count; o++)
            indices[write++] = eng_topology_vertex(occurrence_to_unique, o);
        break;
    }
    case RSX_PRIMITIVE_TRIANGLE_STRIP:
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            if (count < 3) continue;
            /* Odd triangles swap their first two vertices, so the whole strip
             * keeps one winding. */
            for (u32 i = 0; i + 2 < count; i++) {
                indices[write++] = eng_topology_vertex(
                    occurrence_to_unique, begin + i + (i & 1u));
                indices[write++] = eng_topology_vertex(
                    occurrence_to_unique, begin + i + 1u - (i & 1u));
                indices[write++] = eng_topology_vertex(
                    occurrence_to_unique, begin + i + 2u);
            }
        }
        break;
    /* A polygon is a convex fan around its own first vertex, which is what
     * both of this tree's other renderers make of one. */
    case RSX_PRIMITIVE_TRIANGLE_FAN:
    case RSX_PRIMITIVE_POLYGON:
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            if (count < 3) continue;
            for (u32 i = 1; i + 1 < count; i++) {
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + i);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + i + 1u);
            }
        }
        break;
    case RSX_PRIMITIVE_QUAD_STRIP:
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            if (count < 4) continue;
            /* Every quad is split along the diagonal between the two pairs,
             * so each one keeps the strip's winding without alternating the
             * way a triangle strip has to; a cut simply starts the pairing
             * again from the segment's own first vertex. This is the
             * expansion the vtable path's emit_vertices performs. */
            for (u32 q = 0; q + 3 < count; q += 2) {
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 1u);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 2u);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 1u);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 3u);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 2u);
            }
        }
        break;
    case RSX_PRIMITIVE_QUADS:
        for (u32 q = 0; q < source_refs / 4u; q++) {
            const u32 base = q * 4u;
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base + 1u);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base + 2u);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base + 2u);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base + 3u);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base);
        }
        break;
    default:
        break;
    }
}

/* Does this primitive become an indexed triangle list, and is the index buffer
 * needed at all? rsx_vertex_topology_plan answers that for the reference
 * engine, whose own expansion stops at quads. Quad strips and polygons are
 * this engine's, so their two cases are decided here rather than in the
 * shared helper, which the vendored rsx_live_draw.c also calls. */
static int eng_topology_rebuild(u32 primitive, int refs_remapped, int* indexed)
{
    if (primitive == RSX_PRIMITIVE_QUAD_STRIP ||
        primitive == RSX_PRIMITIVE_POLYGON) {
        (void)refs_remapped;
        *indexed = 1;
        return 1;
    }
    return rsx_vertex_topology_plan(primitive, refs_remapped, indexed);
}

/* ---- surfaces ------------------------------------------------------------ */

static rsx_be_format eng_surface_format(u32 color_format)
{
    switch (color_format & 0x1Fu) {
    case RSX_SURFACE_FMT_F_W16Z16Y16X16: return RSX_BE_FMT_R16G16B16A16F;
    case 0x0C: return RSX_BE_FMT_R32G32B32A32F;
    case 0x0D: return RSX_BE_FMT_R32F;
    default:   return RSX_BE_FMT_R8G8B8A8;
    }
}

/* Decode the guest's own bytes behind a colour surface into host R,G,B,A rows
 * so a freshly created target starts with what the title CPU-initialised it
 * to. Returns the staging pointer, or NULL when the bytes are unreadable or
 * the format is not one the decoder handles. */
static const void* eng_surface_seed(u32 location, u32 offset, u32 w, u32 h,
                                    rsx_be_format fmt, u32* out_row_bytes)
{
    *out_row_bytes = 0;
    if (fmt != RSX_BE_FMT_R8G8B8A8) return NULL;
    /* Only a surface in IO-mapped MAIN memory is seeded. RSX local memory is
     * reserved and not backed by this tree's guest VM -- runtime/memory/vm.h
     * maps the whole space PROT_NONE and commits main memory and the stacks,
     * leaving a runner to commit what its own title needs -- so reading a VRAM
     * surface's backing would fault long before it could help. Both halves are
     * checked: the DMA context says which space the offset is in, and the IO
     * table says the page is really mapped. */
    if (location != RSX_LOCATION_MAIN ||
        (!s_guest_reader && !cellGcmResolveIO(offset))) return NULL;
    rsx_tex_layout tl;
    /* A8R8G8B8 with the LN bit: a surface is linear, never swizzled. */
    rsx_texture_layout(0x85u | 0x20u, w, h, &tl);
    if (!tl.face_bytes || !tl.dst_row_bytes) return NULL;
    const u8* src = eng_guest_ptr(NULL, location, offset, tl.face_bytes);
    if (!src) return NULL;
    const u32 bytes = tl.dst_row_bytes * tl.rows;
    if (g.tex_staging_cap < bytes) {
        u8* n = (u8*)realloc(g.tex_staging, bytes);
        if (!n) return NULL;
        g.tex_staging = n;
        g.tex_staging_cap = bytes;
    }
    rsx_texture_decode(g.tex_staging, tl.dst_row_bytes, src, w, h, &tl,
                       rsx_texture_argb_is_rgba());
    *out_row_bytes = tl.dst_row_bytes;
    return g.tex_staging;
}

/* ---- surf54 (31/10): what a frame spent, for the [hitch] line --------------------------------------------------------
 * Only with RSX_HITCH_LOG>0 (the [hitch] line is what reads it): the sum of the draws' time (rdtsc around each draw,
 * two reads of ~25 cycles), the render targets made and released (count and ms), the NV3089 copies, and the table's
 * refusals. The backend adds its lock waits (g_rsx_frame_*). All reset at every present. */
static double eng_now_ms(void);
static struct {
    int on;                                   /* -1 undecided */
    unsigned long long draw_cyc; u32 draws;
    u32 rt_c_n, rt_d_n, zd_c_n, zd_d_n, rt_recycled, refused, blit_n;
    double rt_c_ms, rt_d_ms, zd_c_ms, blit_ms;
    unsigned long long tsc0, fifo0; double ms0;
} s_hp = { -1 };
static u32 s_surf_refused_total;
double g_rsx_frame_lock_ms, g_rsx_frame_pclock_ms, g_rsx_frame_pcsave_ms, g_rsx_frame_join_ms;
unsigned g_rsx_frame_lock_n, g_rsx_frame_pclock_n, g_rsx_frame_pcsave_n;
int (*g_rsx_vram_hook)(unsigned long long* used, unsigned long long* budget) = 0;   /* set by the D3D12 backend */
static inline unsigned long long wp_tsc(void);
static inline int hp_on(void)
{
    if (s_hp.on < 0) {
        const char* e = getenv("RSX_HITCH_LOG");
        s_hp.on = (e && atof(e) > 0.0) ? 1 : 0;
        if (s_hp.on) { s_hp.ms0 = eng_now_ms(); s_hp.tsc0 = wp_tsc(); }
    }
    return s_hp.on;
}
/* rdtsc cycles to ms, the clock rate measured against QueryPerformanceCounter since the first call. */
static double hp_cyc_ms(unsigned long long cyc)
{
    double hz = 3.0e9;
    const double secs = (eng_now_ms() - s_hp.ms0) / 1000.0;
    if (secs > 0.05) hz = (double)(wp_tsc() - s_hp.tsc0) / secs;
    return (double)cyc * 1000.0 / hz;
}
/* Surface pool counters (RSX_RT_STATS=1) and its switch (RSX_RT_POOL=0 off). */
static u32 s_rt_created, s_rt_destroyed, s_rt_reused;

/* Usable table sizes. RSX_SURF_LIMIT=<n> (diagnostic) caps them, e.g. 64 for the old size. */
static u32 eng_surf_cap(void)
{
    static u32 cap;
    if (!cap) {
        const char* e = getenv("RSX_SURF_LIMIT");
        const u32 v = e ? (u32)atoi(e) : 0;
        cap = (v >= 16 && v < ENG_MAX_SURFACES) ? v : ENG_MAX_SURFACES;
    }
    return cap;
}
static u32 eng_zdepth_cap(void)
{
    static u32 cap;
    if (!cap) {
        const char* e = getenv("RSX_SURF_LIMIT");
        const u32 v = e ? (u32)atoi(e) : 0;
        cap = (v >= 16 && v < ENG_MAX_ZDEPTHS) ? v : ENG_MAX_ZDEPTHS;
    }
    return cap;
}

static void eng_rt_destroy(u32 handle)
{
    const double hp_t0 = hp_on() ? eng_now_ms() : 0.0;
    g.be->color_target_release(g.be->user, handle);
    s_rt_destroyed++;
    if (hp_t0 > 0.0) { s_hp.rt_d_ms += eng_now_ms() - hp_t0; s_hp.rt_d_n++; }
}

static void eng_rt_stats(void)
{
    static int on = -1; static time_t last;
    if (on < 0) on = getenv("RSX_RT_STATS") ? 1 : 0;
    if (!on) return;
    const time_t now = time(NULL);
    if (now == last) return;
    if (last)
        fprintf(stderr, "[rt-stats] created %u destroyed %u reused %u (pool %s)\n",
                s_rt_created, s_rt_destroyed, s_rt_reused,
                getenv("RSX_RT_POOL") && getenv("RSX_RT_POOL")[0] == '0' ? "off" : "on");
    last = now;
    s_rt_created = s_rt_destroyed = s_rt_reused = 0;
}

/* Would a freshly created target for this surface be seeded from guest bytes?
 * (the conditions of eng_surface_seed). Such a target is never pooled: a reused
 * one would keep its old contents instead of the guest's. */
static int eng_surface_seedable(u32 location, u32 offset, rsx_be_format fmt)
{
    return fmt == RSX_BE_FMT_R8G8B8A8 && location == RSX_LOCATION_MAIN &&
           (s_guest_reader || cellGcmResolveIO(offset));
}

static void eng_surface_free(u32 slot);

/* The colour target for this (location, offset), created or reallocated when
 * the size or format moved. Returns a slot index, or ENG_INVALID.
 * rsx_live_draw.c's surface_get (2605-2690). */
static u32 eng_surface_get(u32 location, u32 offset, u32 want_w, u32 want_h,
                          rsx_be_format want_fmt)
{
    if (!want_w) want_w = g.width;
    if (!want_h) want_h = g.height;

    u32 slot = ENG_MAX_SURFACES;
    for (u32 i = 0; i < g.n_surfaces; i++)
        if (g.surfaces[i].location == location && g.surfaces[i].offset == offset) {
            if (g.surfaces[i].w == want_w && g.surfaces[i].h == want_h &&
                g.surfaces[i].fmt == want_fmt) {
                g.surfaces[i].last_frame = g.frames;
                return i;
            }
            slot = i;
            break;
        }

    /* Never destroy a usable render target because one malformed command
     * decoded a guest pointer as clip dimensions. */
    if (want_w > ENG_MAX_SURFACE_DIM || want_h > ENG_MAX_SURFACE_DIM) {
        static u32 logs = 0;
        if (logs++ < 8)
            fprintf(stderr, "[rsx engine] rejected implausible surface 0x%X %ux%u;"
                            " keeping the %s target\n", offset, want_w, want_h,
                    slot < ENG_MAX_SURFACES ? "existing" : "absent");
        return slot < ENG_MAX_SURFACES ? slot : ENG_INVALID;
    }
    if (slot == ENG_MAX_SURFACES) {
        for (u32 i = 0; i < g.n_surfaces && slot == ENG_MAX_SURFACES; i++)
            if (!g.surfaces[i].handle && g.surfaces[i].location == ENG_FREE_LOCATION) slot = i;
        /* Nearly full: ToD draws its distortion patches into small targets
         * packed at offsets that move every frame (0x0169xxxx: 64x64, 96x64
         * ...), and kept they fill the table within minutes. Move a small
         * target of the same size and format that nothing drew or sampled
         * for four frames to the new offset instead of making one. Releasing
         * and re-creating them (19a09f26, reverted) churned D3D12 objects
         * faster than they retire and ran the object table dry; this makes
         * and frees nothing. A fresh target is zero, so clear it to match. */
        if (slot == ENG_MAX_SURFACES && g.n_surfaces >= eng_surf_cap() - 8) {
            /* surf54 31/10: the three recoveries below used to run only for a request of at most 256x256, so a 512x512 one (the
             * pause menu's blurred picture) found the table "full" with 11 of 192 slots used and was refused ten thousand
             * times: that effect was not drawn. Now any size. */
            for (u32 i = 0; want_w * want_h <= 256u * 256u && i < g.n_surfaces; i++) {
                eng_surface* t = &g.surfaces[i];
                if (!t->handle || t->location != location || t->w != want_w || t->h != want_h ||
                    t->fmt != want_fmt || t->park_n || i == g.last_present_surface ||
                    t->last_frame + 4u > g.frames)
                    continue;
                { static u32 nm = 0;
                  if (nm++ < 8)
                      fprintf(stderr, "[rsx engine] surface table nearly full: s%u 0x%08X -> 0x%08X (%ux%u, idle since frame %u)\n",
                              i, t->offset, offset, want_w, want_h, t->last_frame); }
                t->offset = offset; t->aa = 0; t->blit_only = 0; t->last_frame = g.frames;
                { static const float zero[4] = {0, 0, 0, 0};
                  g.be->clear_color(g.be->user, t->handle, zero); }
                return i;
            }
            /* No idle target of this size: the smoke and enemy-shot patches are
             * copy-only (NV3089) targets whose size changes every frame (16x1 ..
             * 16x16, 32xN), so a same-size match rarely exists and the table
             * filled anyway -- copies failed and the effect came out black.
             * Free one idle copy-only target of any size and take its slot.
             * Copy-only only: real targets (per-object shadow maps) are never
             * released here, which is what made 19a09f26 churn objects. */
            for (u32 i = 0; i < g.n_surfaces; i++) {
                eng_surface* t = &g.surfaces[i];
                if (!t->handle || !t->blit_only || t->park_n || i == g.last_present_surface ||
                    t->last_frame + 4u > g.frames)
                    continue;
                { static u32 nf = 0;
                  if (nf++ < 8)
                      fprintf(stderr, "[rsx engine] surface table nearly full: freed copy-only s%u 0x%08X %ux%u for 0x%08X %ux%u\n",
                              i, t->offset, t->w, t->h, offset, want_w, want_h); }
                eng_surface_free(i);
                slot = i;
                break;
            }
            /* Still nothing (22/10 4K game: the table filled with real 64x64
             * per-object shadow maps idle for hundreds of frames): free the
             * small real target idle the longest -- 16+ frames, never the
             * scene or anything just drawn. Rare by construction (table nearly
             * full and no move or copy-only candidate), so no churn. */
            if (slot == ENG_MAX_SURFACES) {
                u32 best = ENG_MAX_SURFACES, best_frame = 0xFFFFFFFFu;
                for (u32 i = 0; i < g.n_surfaces; i++) {
                    eng_surface* t = &g.surfaces[i];
                    if (!t->handle || i == g.last_present_surface || t->w * t->h > 512u * 512u ||
                        t->last_frame + 16u > g.frames)
                        continue;
                    if (t->last_frame < best_frame) { best_frame = t->last_frame; best = i; }
                }
                if (best != ENG_MAX_SURFACES) {
                    { static u32 nr = 0;
                      if (nr++ < 8)
                          fprintf(stderr, "[rsx engine] surface table nearly full: freed idle s%u 0x%08X %ux%u (idle since frame %u) for 0x%08X %ux%u\n",
                                  best, g.surfaces[best].offset, g.surfaces[best].w, g.surfaces[best].h,
                                  best_frame, offset, want_w, want_h); }
                    eng_surface_free(best);
                    slot = best;
                }
            }
        }
        if (slot == ENG_MAX_SURFACES) {
            if (g.n_surfaces >= eng_surf_cap()) {
                static u32 logs = 0, total = 0;
                total++; s_surf_refused_total++; s_hp.refused++;
                if (logs >= 8 && (total & 0x3FFu) == 0) {   /* the count, so a long run shows whether it never stops */
                    u32 act = 0;
                    for (u32 i = 0; i < g.n_surfaces; i++) if (g.surfaces[i].last_frame + 16u > g.frames) act++;
                    fprintf(stderr, "[rsx engine] surface table full: %u refusals so far (%u of %u slots used in the last 16 frames)\n",
                            total, act, ENG_MAX_SURFACES);
                }
                if (logs++ < 8)
                    fprintf(stderr, "[rsx engine] surface table full (%u): no target for %u:0x%08X %ux%u\n",
                            ENG_MAX_SURFACES, location, offset, want_w, want_h);
                if (logs == 1)   /* who holds the slots, once */
                    for (u32 i = 0; i < g.n_surfaces; i++)
                        fprintf(stderr, "[rsx engine]   s%u %u:0x%08X %ux%u fmt %d aa %u%s parked %u\n", i,
                                g.surfaces[i].location, g.surfaces[i].offset, g.surfaces[i].w, g.surfaces[i].h,
                                (int)g.surfaces[i].fmt, g.surfaces[i].aa,
                                g.surfaces[i].blit_only ? " blit" : "", g.surfaces[i].park_n);
                return ENG_INVALID;
            }
            slot = g.n_surfaces;
            if (slot >= 48 && (slot & 15u) == 0)   /* high-water marks of the table, so a run shows how close it came */
                fprintf(stderr, "[rsx engine] surface table reached %u of %u (frame %u)\n", slot + 1u, eng_surf_cap(), g.frames);
        }
    }

    static int pool_on = -1;
    if (pool_on < 0) { const char* e = getenv("RSX_RT_POOL"); pool_on = !(e && e[0] == '0'); }
    const int poolable = pool_on && !eng_surface_seedable(location, offset, want_fmt);
    eng_rt_stats();

    /* A size this offset already used: swap that parked target back in. A fresh
     * unseeded target is all zero, so clear the reused one to match. */
    if (poolable && slot < g.n_surfaces && g.surfaces[slot].handle) {
        eng_surface* ps = &g.surfaces[slot];
        for (u32 k = 0; k < ps->park_n; k++) {
            if (ps->park[k].w != want_w || ps->park[k].h != want_h ||
                ps->park[k].fmt != want_fmt) continue;
            const eng_surface_variant back = ps->park[k];
            ps->park[k].w = ps->w; ps->park[k].h = ps->h;
            ps->park[k].fmt = ps->fmt; ps->park[k].handle = ps->handle;
            ps->w = back.w; ps->h = back.h; ps->fmt = back.fmt;
            ps->handle = back.handle;
            { static const float zero[4] = {0, 0, 0, 0};
              g.be->clear_color(g.be->user, ps->handle, zero); }
            s_rt_reused++;
            ps->last_frame = g.frames;
            return slot;
        }
    }

    u32 seed_row = 0;
    const void* seed = eng_surface_seed(location, offset, want_w, want_h,
                                        want_fmt, &seed_row);
    const double hp_t0 = hp_on() ? eng_now_ms() : 0.0;
    const u32 handle = g.be->color_target_create(g.be->user, want_fmt,
                                                 want_w, want_h, seed, seed_row);
    if (hp_t0 > 0.0) { s_hp.rt_c_ms += eng_now_ms() - hp_t0; s_hp.rt_c_n++; }
    if (!handle)
        return (slot < g.n_surfaces && g.surfaces[slot].handle) ? slot : ENG_INVALID;

    eng_surface* s = &g.surfaces[slot];
    s_rt_created++;
    if (s->handle) {
        if (poolable && slot < g.n_surfaces) {
            /* Park the old target; the oldest parked one makes room. */
            if (s->park_n == ENG_SURF_PARK) {
                eng_rt_destroy(s->park[0].handle);
                for (u32 k = 1; k < ENG_SURF_PARK; k++) s->park[k - 1] = s->park[k];
                s->park_n--;
            }
            s->park[s->park_n].w = s->w; s->park[s->park_n].h = s->h;
            s->park[s->park_n].fmt = s->fmt; s->park[s->park_n].handle = s->handle;
            s->park_n++;
        } else
            eng_rt_destroy(s->handle);
    }
    if (s->location != location || s->offset != offset) { s->aa = 0; s->blit_only = 0; s->alias_off = 0; }
    s->location = location; s->offset = offset;
    s->w = want_w; s->h = want_h; s->fmt = want_fmt;
    s->handle = handle;
    s_surf_gen++;
    s->last_frame = g.frames;
    if (slot == g.n_surfaces) g.n_surfaces++;
    { static u32 logs = 0; if (logs++ < 16)
        fprintf(stderr, "[rsx engine] surface %u:0x%08X %ux%u fmt %d%s\n",
                location, offset, want_w, want_h, (int)want_fmt,
                seed ? " (seeded from guest memory)" : ""); }
    return slot;
}

/* Release a slot and its parked variants for reuse (the backend defers the
 * release past any draw already recorded against them). */
static void eng_surface_free(u32 slot)
{
    eng_surface* s = &g.surfaces[slot];
    for (u32 k = 0; k < s->park_n; k++) eng_rt_destroy(s->park[k].handle);
    if (s->handle) eng_rt_destroy(s->handle);
    s->handle = 0; s->park_n = 0;
    s->location = ENG_FREE_LOCATION; s->offset = 0;
    s->w = s->h = 0; s->fmt = RSX_BE_FMT_NONE; s->aa = 0; s->blit_only = 0; s->alias_off = 0;
    s_surf_gen++;
}

/* A 1:1 NV3089 blit whose source lies in a colour target this engine drew:
 * the target exists only on the GPU, so the guest-memory copy moves zeros.
 * Ratchet ToD's smoke copies the 32x34 patch of the scene behind each puff
 * (pitch 8192, out of the 960x704 scene target) to 0x01673E80 and samples it
 * for its distortion: from zeros every puff came out black. Do the copy on
 * the GPU into a target at the destination, which a texture unit naming that
 * address then binds. The source rect is sw x sh guest texels (the puffs
 * copy it downscaled: 128x38 samples into 32x19). Returns 1 when done.
 * RSX_NV3089_GPU=0 disables. */
int rsx_draw_engine_blit(u32 src_loc, u32 src_off, u32 src_pitch, u32 sw, u32 sh,
                         u32 dst_loc, u32 dst_off, u32 dst_pitch, u32 w, u32 h)
{
    static int en = -1;
    if (en < 0) { const char* e = getenv("RSX_NV3089_GPU"); en = !(e && e[0] == '0'); }
    if (!en || !g.be || !g.be->color_copy_rect || !src_pitch || !dst_pitch || !sw || !sh || !w || !h) return 0;
    rsx_draw_engine_b2_drain();   /* RSX_DRAW_SPLIT=2: reads/writes g and the targets */
    /* The closest target base at or below the source, with the rect inside it.
     * A multisampled target (ToD's 960x704 scene: 2x diagonal, pitch 8192)
     * holds two samples per pixel across in guest memory (and two down for
     * the 4x modes), so the blit's rect is in samples: the host target is
     * resolved, so divide. The mode is the one the target was last drawn
     * with -- guessing it from whether the rect fits copied the left half of
     * the screen from twice its x. */
    int si = -1; u32 gx = 0, gy = 0, kx = 1, ky = 1;
    for (u32 i = 0; i < g.n_surfaces; i++) {
        const eng_surface* t = &g.surfaces[i];
        if (!t->handle || t->location != src_loc || src_off < t->offset) continue;
        const u32 kx_t = t->aa >= 3 ? 2u : 1u, ky_t = t->aa >= 4 ? 2u : 1u;
        const u32 d = src_off - t->offset, x = (d % src_pitch) / 4u / kx_t, y = d / src_pitch / ky_t;
        if (x + (sw + kx_t - 1u) / kx_t > t->w || y + (sh + ky_t - 1u) / ky_t > t->h) continue;
        if (si >= 0 && g.surfaces[si].offset > t->offset) continue;
        si = (int)i; gx = x; gy = y; kx = kx_t; ky = ky_t;
    }
    static int lg = -1; static unsigned nlog;
    if (lg < 0) lg = getenv("RSX_NV3089_GPU_LOG") ? 1 : 0;
#define ENG_BLIT_FAIL(why) do { if (lg && nlog++ < 256) \
        fprintf(stderr, "[rsx engine] nv3089 gpu blit SKIP %s: %ux%u <- %u:0x%08X (pitch %u, %ux%u) -> %u:0x%08X s%d\n", \
                why, w, h, src_loc, src_off, src_pitch, sw, sh, dst_loc, dst_off, si); return 0; } while (0)
    if (si < 0) ENG_BLIT_FAIL("no source target");
    if (g.surfaces[si].fmt != RSX_BE_FMT_R8G8B8A8) ENG_BLIT_FAIL("source format");
    if (dst_loc == src_loc && dst_off == g.surfaces[si].offset) ENG_BLIT_FAIL("into itself");
    const u32 src_handle = g.surfaces[si].handle;
    /* The puffs pack their patches one after another into a scratch buffer
     * (0x01673E80, 0x01674A00, 0x01675180 ...: the offsets move with the
     * patch sizes every frame), so each copy names a new destination. A
     * target only ever made by these copies and overlapping this one holds
     * bytes the guest has since overwritten: release it. Kept, they filled
     * the 64-slot table within seconds, after which every copy failed and
     * the smoke behind the ships came out black. */
    int had_real = 0, exact = -1, recycle = -1;
    /* surf54 31/10: the copy overwrites the whole w x h destination, so a copy-only target of exactly this size that nothing
     * has drawn or sampled for two frames (or one stale and overlapping this one) is moved to the new offset instead of making
     * a D3D12 resource (the smoke and shot patches come in a handful of sizes: the arena made and dropped them all game long).
     * RSX_RT_RECYCLE=1 on (default off). */
    static int rcy = -1; if (rcy < 0) { const char* e = getenv("RSX_RT_RECYCLE"); rcy = (e && e[0] == '1'); }   /* integ113: default OFF for now (=1 on). hitch104 02/11 had default ON (=0 off). The user's 36-min play log (pirate base / Apogee: 14.7k [hitch] rows) had 43 s in colour-target creation (up to 60 NV3089 smoke/puff patches per frame, 0.37 ms each = 5-18 ms/frame, and 1-2 depth targets 1.2 ms each); the arena (surf54) had too few copies to show it */
    const u32 d_end = dst_off + h * dst_pitch;
    for (u32 i = 0; i < g.n_surfaces; i++) {
        eng_surface* t = &g.surfaces[i];
        if (!t->handle || t->location != dst_loc) continue;
        if (t->offset == dst_off) { had_real = !t->blit_only; exact = (int)i; continue; }
        if (!t->blit_only || (int)i == si) continue;
        const u32 t_end = t->offset + t->h * t->w * 4u;
        if (t->offset >= d_end || t_end <= dst_off) continue;
        if (rcy && recycle < 0 && !t->park_n && t->w == w && t->h == h && t->fmt == RSX_BE_FMT_R8G8B8A8) { recycle = (int)i; continue; }
        eng_surface_free(i);
    }
    if (rcy && recycle < 0 && exact < 0) {
        u32 best_frame = 0xFFFFFFFFu;
        for (u32 i = 0; i < g.n_surfaces; i++) {
            const eng_surface* t = &g.surfaces[i];
            if (!t->handle || !t->blit_only || t->park_n || (int)i == si || i == g.last_present_surface || t->location != dst_loc ||
                t->w != w || t->h != h || t->fmt != RSX_BE_FMT_R8G8B8A8 || t->last_frame + 2u > g.frames)
                continue;
            if (t->last_frame < best_frame) { best_frame = t->last_frame; recycle = (int)i; }
        }
    }
    u32 di;
    if (recycle >= 0 && exact < 0) {
        eng_surface* t = &g.surfaces[recycle];
        t->location = dst_loc; t->offset = dst_off; t->aa = 0; t->alias_off = 0; t->last_frame = g.frames;
        s_surf_gen++; s_hp.rt_recycled++; s_rt_reused++;
        di = (u32)recycle;
    } else {
        if (recycle >= 0) eng_surface_free((u32)recycle);   /* a same-size target is already at the destination: the old behaviour */
        di = eng_surface_get(dst_loc, dst_off, w, h, RSX_BE_FMT_R8G8B8A8);
    }
    if (di == ENG_INVALID || (int)di == si) ENG_BLIT_FAIL("no destination");
#undef ENG_BLIT_FAIL
    if (!had_real) g.surfaces[di].blit_only = 1;
    const double hp_t0 = hp_on() ? eng_now_ms() : 0.0;
    const int ok = g.be->color_copy_rect(g.be->user, src_handle, gx, gy, (sw + kx - 1u) / kx,
                                         (sh + ky - 1u) / ky, g.surfaces[di].handle, 0, 0, w, h);
    if (hp_t0 > 0.0) { s_hp.blit_ms += eng_now_ms() - hp_t0; s_hp.blit_n++; }
    { if (lg && nlog++ < 256)
          fprintf(stderr, "[rsx engine] nv3089 gpu blit %ux%u <- s%d(+%u,%u %ux%u /%ux%u) -> %u:0x%08X (pitch %u) s%u of %u ok=%d\n",
                  w, h, si, gx, gy, sw, sh, kx, ky, dst_loc, dst_off, dst_pitch, di, g.n_surfaces, ok); }
    return ok;
}

/* The colour targets SET_SURFACE_COLOR_TARGET names, as slots into
 * g.surfaces, with A first. One for an ordinary draw; an MRT set adds B, C
 * and D, each created at target A's size and format, which is the rule the
 * vtable path's target_for binds by. Resolution stops at the first member
 * that cannot be created, as that path stops at the first gap. Returns how
 * many were resolved, or 0. */
/* 26/10 gfx-quick: the pass about to be recorded is a per-object shadow map when its zeta is Z16 (ToD's
 * scene depth is Z24S8; the maps are sampled as DEPTH16 with compare, fmt 0x92). The d3d12 backend reads
 * these when it creates the colour target, records a clear and scales the viewport (RSX_SHADOW_SCALE). */
volatile int g_rsx_pass_shadow;
volatile u32 g_rsx_pass_cw, g_rsx_pass_ch;
/* shad114 RSX_DBIAS_EPILOGUE (default ON, =0 off): polygon offset reworked as RACiT does it (idea: RACiT, MIT, (c) 2026 HighDrexler).
 * The CONSTANT part (reg 0x0A7C, raw float, no cut88 multiplier, original sign) goes into the VS epilogue's window-z offset
 * (xf[6] += bias * 2^-16 for a Z16 zeta, 2^-24 otherwise): D3D12's DepthBias on a D32F depth buffer is scaled by
 * 2^(exponent(z)-23) per primitive, i.e. ~256x too small for a Z16 map at z~0.5 and wrong everywhere else. The SLOPE part
 * (reg 0x0A78 raw) is multiplied by the target's host/guest pixel ratio (s_scale, times the shadow-map extra k for Z16 shadow passes)
 * and quantised in log2 steps (8 per octave) so the PSO count does not explode; the PSO then gets DepthBias=0 and sign +
 * (RSX_POLY_SIGN and the cut88 multipliers are ignored). Off: byte-identical to before. RSX_POLY_OFFSET=0 keeps it off.
 * g_rsx_poly_target_scale_fn is installed by the D3D12 backend (NULL elsewhere = 1.0). */
float (*g_rsx_poly_target_scale_fn)(void) = NULL;
int rsx_dbias_epilogue_on(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("RSX_DBIAS_EPILOGUE"); const char* po = getenv("RSX_POLY_OFFSET");
        on = (!(e && e[0] == '0') && !(po && po[0] == '0')) ? 1 : 0; }   /* default ON since the 02/11 Cobalia check (ground shadow back, body clean); =0 off */
    return on;
}
static void eng_note_pass(const rsx_dsp_surface* sf)
{
    g_rsx_pass_shadow = sf->depth_format == 1 && sf->zeta_location != 0xFFFFFFFFu;
    g_rsx_pass_cw = sf->clip_w; g_rsx_pass_ch = sf->clip_h;
}
static u32 eng_current_target_set(u32 slots[RSX_BE_MAX_COLOR_TARGETS])
{
    rsx_dsp_surface sf;
    rsx_dsp_get_surface(RV, &sf);
    eng_note_pass(&sf);
    /* SET_SURFACE_COLOR_TARGET 2 selects B alone; every other value starts at
     * A, and the MRT selectors add B, C and D on top of it. */
    const u32 sel = (sf.color_target == CELL_GCM_SURFACE_TARGET_1) ? 1u : 0u;
    const rsx_be_format fmt = eng_surface_format(sf.color_format);
    const u32 first = eng_surface_get(sf.color_location[sel], sf.color_offset[sel],
                                      sf.clip_w, sf.clip_h, fmt);
    if (first == ENG_INVALID) return 0;
    slots[0] = first;
    /* What an NV3089 copy out of it needs to know, and that it is drawn. */
    g.surfaces[first].aa = (u8)((rsx_dsp_reg(RV, 0x0208) >> 12) & 0xFu);
    g.surfaces[first].blit_only = 0;
    /* R1 black (27/10): Ratchet ToD draws the 1280x704 picture into a 1280x720 display buffer with the clip
     * window at y=8 and later samples that window as a texture at offset + 8*pitch (0xA000 / 0x442000): a
     * post pass (the R1 slow-time effect, the bloom taps) reading what the scene pass just wrote. The surface is
     * keyed by the colour offset alone, so that texture matched nothing and came from guest memory: zeros, black.
     * Remember where the clip origin lies so the texture lookup can match it. RSX_NO_CLIP_ALIAS=1 turns it off. */
    if (fmt == RSX_BE_FMT_R8G8B8A8 && sf.color_pitch[sel]) {
        static int off = -1; if (off < 0) off = getenv("RSX_NO_CLIP_ALIAS") ? 1 : 0;
        const u32 al = (!off && (sf.clip_x || sf.clip_y))
            ? sf.color_offset[sel] + sf.clip_y * sf.color_pitch[sel] + sf.clip_x * 4u : 0u;
        if (g.surfaces[first].alias_off != al) {
            g.surfaces[first].alias_off = al; s_surf_gen++;
            static u32 nl; if (nl++ < 16)
                fprintf(stderr, "[rsx engine] s%u off=0x%08X clip=%u,%u+%ux%u pitch=%u -> texture alias at 0x%08X\n",
                        first, sf.color_offset[sel], sf.clip_x, sf.clip_y, sf.clip_w, sf.clip_h, sf.color_pitch[sel], al);
        }
    }
    u32 n = 1;

    static const u32 mrt_from[3] = {
        CELL_GCM_SURFACE_TARGET_MRT1, CELL_GCM_SURFACE_TARGET_MRT2,
        CELL_GCM_SURFACE_TARGET_MRT3
    };
    for (u32 i = 0; i < 3; i++) {
        if (sf.color_target < mrt_from[i]) break;
        const u32 slot = eng_surface_get(sf.color_location[i + 1],
                                         sf.color_offset[i + 1],
                                         sf.clip_w, sf.clip_h, fmt);
        if (slot == ENG_INVALID) break;
        /* A set naming one buffer twice would attach the same target twice,
         * which no host API allows. */
        int seen = 0;
        for (u32 k = 0; k < n; k++) if (slots[k] == slot) seen = 1;
        if (seen) break;
        g.surfaces[slot].blit_only = 0;
        slots[n++] = slot;
    }
    return n;
}

static u32 eng_current_surface(void)
{
    u32 slots[RSX_BE_MAX_COLOR_TARGETS];
    return eng_current_target_set(slots) ? slots[0] : ENG_INVALID;
}

/* ---- per-zeta depth ------------------------------------------------------ */

/* One depth target per guest zeta address. Sharing a single resource across a
 * title's shadow, scene and post passes cross-contaminates later depth tests;
 * upstream's ef3271b records that as "the black player-character mass". */
static u32 eng_zdepth_get(u32 location, u32 offset, u32 rt_w, u32 rt_h)
{
    /* The attachment must cover the whole canvas: an early pass declaring a
     * smaller clip than the live viewport otherwise gets a target the host
     * API rejects when a later, larger pass binds it. */
    u32 want_w = rt_w > g.width ? rt_w : g.width;
    u32 want_h = rt_h > g.height ? rt_h : g.height;
    /* 30/10: but a small pass (ToD's per-object shadow maps, 64x64 .. 320x320 at
     * offsets that move every frame) only needs its clip times the extra factor
     * the backend may draw it at (RSX_SHADOW_SCALE / RSX_BLOOM_SCALE, default 2,
     * capped at the canvas there too). Canvas-sized, each one cost 63 MB at 4K
     * (D32S8): 54 of them held 3.4 GB and MSAA x8 ran the 3070 out of VRAM,
     * the GPU stalling for whole frames in cutscenes. A later, larger pass
     * still grows the target below. RSX_ZDEPTH_FIT=0: canvas-sized again. */
    { static u32 fit = 0xFFFFFFFFu;
      if (fit == 0xFFFFFFFFu) {
          const char* e = getenv("RSX_ZDEPTH_FIT");
          fit = 0;
          if (!(e && e[0] == '0')) {
              const char* s = getenv("RSX_SHADOW_SCALE"); const char* b = getenv("RSX_BLOOM_SCALE");
              const int ks = s && *s ? atoi(s) : 2, kb = b && *b ? atoi(b) : 2;
              const int k = ks > kb ? ks : kb;
              fit = k >= 4 ? 4u : 2u;
          } }
      if (fit && rt_w && rt_h) {
          const u32 fw = rt_w * fit, fh = rt_h * fit;
          if (fw < want_w) want_w = fw;
          if (fh < want_h) want_h = fh;
      } }
    if (want_w > ENG_MAX_SURFACE_DIM || want_h > ENG_MAX_SURFACE_DIM)
        return ENG_INVALID;

    u32 slot = ENG_MAX_ZDEPTHS;
    for (u32 i = 0; i < g.n_zdepths; i++) {
        eng_zdepth* z = &g.zdepths[i];
        if (z->location == location && z->offset == offset) {
            if (z->w >= want_w && z->h >= want_h) { z->last_frame = g.frames; return i; }
            slot = i;
            if (want_w < z->w) want_w = z->w;
            if (want_h < z->h) want_h = z->h;
            break;
        }
    }
    /* Nearly full: ToD gives each caster its own small shadow map (64x64,
     * 96x128 ...) packed at offsets that move every frame (0x0169xxxx), and
     * each offset got a canvas-sized depth target of its own: the 64 slots
     * filled, new maps got none, and the shadows of Ratchet and the enemies
     * flickered. Move a depth target idle for four frames (g.frames counts
     * repeated presents too, so two could steal last frame's map), large enough, to
     * the new offset; nothing is made or freed. Its copies are stale. */
    if (slot == ENG_MAX_ZDEPTHS && g.n_zdepths >= eng_zdepth_cap() - 4) {
        for (u32 i = 0; i < g.n_zdepths; i++) {
            eng_zdepth* z = &g.zdepths[i];
            if (!z->handle || z->location != location || z->w < want_w || z->h < want_h ||
                z->last_frame + 4u > g.frames)
                continue;
            { static u32 nm = 0;
              if (nm++ < 8)
                  fprintf(stderr, "[rsx engine] zeta table nearly full: z%u 0x%08X -> 0x%08X (idle since frame %u)\n",
                          i, z->offset, offset, z->last_frame); }
            z->offset = offset; z->last_frame = g.frames; s_zgen++;
            z->cleared = 0; z->had_write = 0; z->snapshot_valid = 0; z->packed_valid = 0;
            return i;
        }
    }
    if (slot == ENG_MAX_ZDEPTHS) {
        if (g.n_zdepths >= eng_zdepth_cap()) {
            static u32 nf = 0;
            if (nf++ < 8) fprintf(stderr, "[rsx engine] zeta table full (%u): no depth for %u:0x%08X\n",
                                  ENG_MAX_ZDEPTHS, location, offset);
            return ENG_INVALID;
        }
        slot = g.n_zdepths;
        for (u32 i = 0; i < g.n_zdepths; i++)   /* surf54: a slot eng_zdepth_free released */
            if (!g.zdepths[i].handle && g.zdepths[i].location == ENG_FREE_LOCATION) { slot = i; break; }
    }

    const double hp_t0 = hp_on() ? eng_now_ms() : 0.0;
    const u32 handle = g.be->depth_target_create(g.be->user, want_w, want_h);
    if (hp_t0 > 0.0) { s_hp.zd_c_ms += eng_now_ms() - hp_t0; s_hp.zd_c_n++; }
    if (!handle)
        return (slot < g.n_zdepths && g.zdepths[slot].handle) ? slot : ENG_INVALID;

    eng_zdepth* z = &g.zdepths[slot];
    if (z->handle) { g.be->depth_target_release(g.be->user, z->handle); s_hp.zd_d_n++; }
    if (z->snapshot && g.be->texture_release) g.be->texture_release(g.be->user, z->snapshot);
    if (z->packed && g.be->texture_release) g.be->texture_release(g.be->user, z->packed);
    z->packed = 0; z->packed_valid = 0;
    z->location = location; z->offset = offset; s_zgen++;
    z->w = want_w; z->h = want_h;
    z->handle = handle;
    z->snapshot = 0;
    z->cleared = 0;
    z->had_write = 0;
    z->snapshot_valid = 0;
    z->last_frame = g.frames;
    if (slot == g.n_zdepths) g.n_zdepths++;
    return slot;
}

/* surf54: release a depth target (and its copies) and leave the slot free for eng_zdepth_get. */
static void eng_zdepth_free(u32 slot)
{
    eng_zdepth* z = &g.zdepths[slot];
    if (z->handle) { g.be->depth_target_release(g.be->user, z->handle); s_hp.zd_d_n++; }
    if (z->snapshot && g.be->texture_release) g.be->texture_release(g.be->user, z->snapshot);
    if (z->packed && g.be->texture_release) g.be->texture_release(g.be->user, z->packed);
    memset(z, 0, sizeof *z);
    z->location = ENG_FREE_LOCATION;
    s_zgen++;
}

/* A depth target read back as a texture, but only once the pass has executed
 * a depth-WRITING draw: a write-enable bit alone does not prove the pass
 * produced a usable depth map, and a clear-only zeta falls through to guest
 * memory instead (rsx_live_draw.c:6016-6018, 6265-6269). */
/* A depth target is allocated at least as large as the screen
 * (eng_zdepth_get), so a 512x512 shadow map lives in the top-left corner of a
 * 1280x720 texture. A copy for sampling must be the size the title's texture
 * unit declares -- the region it addresses with 0..1 coordinates -- or every
 * lookup lands in the wrong texels: Drakengard 3's character shadow
 * projection read its 512x512 map as if it were 1280x720 and found nothing,
 * so no character cast a shadow. The copy is cropped texel for texel. */
static void eng_zdepth_copy_size(const eng_zdepth* z, u32 tw, u32 th, u32* w, u32* h)
{
    *w = (tw && tw < z->w) ? tw : z->w;
    *h = (th && th < z->h) ? th : z->h;
}

static u32 eng_zdepth_snapshot(u32 slot, u32 tw, u32 th)
{
    eng_zdepth* z = &g.zdepths[slot];
    if (!z->handle || !z->had_write) return 0;
    u32 cw, ch; eng_zdepth_copy_size(z, tw, th, &cw, &ch);
    if (z->snapshot_valid && z->snapshot && z->snap_w == cw && z->snap_h == ch) return z->snapshot;
    if (!g.be->depth_snapshot) return 0;
    /* The previous image of this zeta is stale (a clear invalidated it), so
     * hand it back before resolving a new one. It never was: every shadow-map
     * clear leaked one backend texture -- a frame's worth of them per frame --
     * until the backend's 4096-entry object table filled, a few minutes into
     * play, and from then on EVERY new texture failed to create. In
     * Drakengard 3 that is the in-game cutscene sampling three missing video
     * planes (solid green), and the player's body vanishing for good once its
     * textures were next re-uploaded. Draws already recorded against the old
     * snapshot keep it: the backend retires the handle and recycles it only
     * after the frame is encoded. */
    if (z->snapshot && g.be->texture_release) g.be->texture_release(g.be->user, z->snapshot);
    z->snapshot = 0;
    const u32 tex = g.be->depth_snapshot(g.be->user, z->handle, cw, ch);
    if (!tex) return 0;
    z->snapshot = tex;
    z->snapshot_valid = 1;
    z->snap_w = cw; z->snap_h = ch;
    return tex;
}

static u32 eng_zdepth_packed(u32 slot, u32 tw, u32 th)
{
    eng_zdepth* z = &g.zdepths[slot];
    if (!z->handle || !z->had_write) return 0;
    u32 cw, ch; eng_zdepth_copy_size(z, tw, th, &cw, &ch);
    if (z->packed_valid && z->packed && z->packed_w == cw && z->packed_h == ch) return z->packed;
    if (!g.be->depth_snapshot_rgba8) return 0;
    if (z->packed && g.be->texture_release) g.be->texture_release(g.be->user, z->packed);
    z->packed = 0;
    const u32 tex = g.be->depth_snapshot_rgba8(g.be->user, z->handle, cw, ch);
    if (!tex) return 0;
    z->packed = tex;
    z->packed_valid = 1;
    z->packed_w = cw; z->packed_h = ch;
    return tex;
}

/* ---- textures ------------------------------------------------------------ */

static rsx_be_format eng_texfmt(rsx_texfmt f)
{
    switch (f) {
    case RSX_TEXFMT_R8:              return RSX_BE_FMT_R8;
    case RSX_TEXFMT_R8G8:            return RSX_BE_FMT_R8G8;
    case RSX_TEXFMT_R8G8B8A8:        return RSX_BE_FMT_R8G8B8A8;
    case RSX_TEXFMT_BC1:             return RSX_BE_FMT_BC1;
    case RSX_TEXFMT_BC2:             return RSX_BE_FMT_BC2;
    case RSX_TEXFMT_BC3:             return RSX_BE_FMT_BC3;
    case RSX_TEXFMT_R16:             return RSX_BE_FMT_R16;
    case RSX_TEXFMT_R16G16:          return RSX_BE_FMT_R16G16;
    case RSX_TEXFMT_R16G16F:         return RSX_BE_FMT_R16G16F;
    case RSX_TEXFMT_R16G16B16A16F:   return RSX_BE_FMT_R16G16B16A16F;
    case RSX_TEXFMT_R32F:            return RSX_BE_FMT_R32F;
    case RSX_TEXFMT_R32G32B32A32F:   return RSX_BE_FMT_R32G32B32A32F;
    default:                         return RSX_BE_FMT_R8;
    }
}

/* How many guest bytes a texture occupies: the whole mip chain, times six for
 * a cube map. That is what the content hash has to cover, or a title
 * animating one face or one lower level keeps its stale upload. */
static u32 eng_texture_span(u32 fmt, u32 w, u32 h, u32 levels, u32 pitch, int cube)
{
    if (cube) return rsx_texture_cube_face_stride(fmt, w, h, levels, pitch) * 6u;
    rsx_tex_level lv[RSX_MAX_TEXTURE_LEVELS];
    const u32 n = rsx_texture_mip_chain(fmt, w, h, levels, pitch, lv);
    if (!n) return 0;
    return lv[n - 1].offset + lv[n - 1].tl.face_bytes;
}

/* RSX_ENGINE_SLOW=1: the per-draw paths as they were before the indexes,
 * the single fragment-program pass and the per-draw program memo -- for
 * comparing the two in one build (rsx_replay --bench, DOD3_AB). */
int g_rsx_eng_slow = -1;
static inline int eng_slow(void)
{
    if (g_rsx_eng_slow < 0) g_rsx_eng_slow = getenv("RSX_ENGINE_SLOW") ? 1 : 0;
    return g_rsx_eng_slow;
}

static inline u32 eng_tex_key_hash(u32 location, u32 offset, u32 fmt, u32 w, u32 h,
                                   u32 pitch, u32 remap, u32 cube)
{
    const u64 a = ((u64)offset << 32) ^ ((u64)location << 28) ^ ((u64)cube << 27) ^ fmt;
    const u64 b = ((u64)w << 48) ^ ((u64)h << 32) ^ ((u64)pitch << 20) ^ remap;
    u64 x = a * 0x9E3779B97F4A7C15ull ^ b * 0xC2B2AE3D27D4EB4Full;
    x ^= x >> 31;
    return (u32)x;
}

static eng_texture* eng_tex_find(u32 location, u32 offset, u32 fmt, u32 w, u32 h,
                                 u32 pitch, u32 remap, u32 cube)
{
    const u32 m = ENG_TEX_INDEX - 1u;
    for (u32 i = eng_tex_key_hash(location, offset, fmt, w, h, pitch, remap, cube) & m;; i = (i + 1u) & m) {
        const u32 v = g.tex_index[i];
        if (!v) return NULL;
        eng_texture* e = &g.textures[v - 1u];
        if (e->location == location && e->offset == offset && e->format == fmt &&
            e->width == w && e->height == h && e->pitch == pitch && e->remap == remap &&
            e->cubemap == cube)
            return e;
    }
}

static void eng_tex_index_add(u32 index)
{
    const eng_texture* e = &g.textures[index];
    const u32 m = ENG_TEX_INDEX - 1u;
    u32 i = eng_tex_key_hash(e->location, e->offset, e->format, e->width, e->height,
                             e->pitch, e->remap, e->cubemap) & m;
    while (g.tex_index[i]) i = (i + 1u) & m;
    g.tex_index[i] = index + 1u;
}

/* An evicted entry took another key: linear probing has no cheap delete, so
 * the index is rebuilt (only when the cache is full and turning over). */
static void eng_tex_index_rebuild(void)
{
    memset(g.tex_index, 0, sizeof g.tex_index);
    for (u32 i = 0; i < g.n_textures; i++) eng_tex_index_add(i);
}

static inline u32 eng_pipe_key_slot(u64 key)
{
    return (u32)(key ^ (key >> 29) ^ (key >> 47)) & (ENG_PIPE_INDEX - 1u);
}

static eng_pipeline* eng_pipe_find(u64 key)
{
    for (u32 i = eng_pipe_key_slot(key);; i = (i + 1u) & (ENG_PIPE_INDEX - 1u)) {
        const u32 v = g.pipe_index[i];
        if (!v) return NULL;
        if (g.pipelines[v - 1u].key == key) return &g.pipelines[v - 1u];
    }
}

static void eng_pipe_index_add(u32 index)
{
    u32 i = eng_pipe_key_slot(g.pipelines[index].key);
    while (g.pipe_index[i]) i = (i + 1u) & (ENG_PIPE_INDEX - 1u);
    g.pipe_index[i] = index + 1u;
}

/* Fragment texture unit u is switched on (CONTROL0 bit 31) and the draw's
 * fragment program samples it (eng_fp_reads, below): the loops over all
 * sixteen units skip the others before decoding them. */
static u32 eng_fp_reads(void);
static inline int eng_tex_unit_on(u32 u, u32 reads)
{
    return (int)(rsx_dsp_reg(RV, 0x1A00u + u * 0x20u + 0x0Cu) >> 31) & (int)((reads >> u) & 1u);
}

/* One hash per cached texture per presented frame. FNV-style over the whole
 * span is deliberately cheap: this is a mutation detector, not a content id
 * (rsx_live_draw.c:2002-2026). Eight independent lanes, folded at the end: a
 * single lane is one dependent 64-bit multiply per 8 bytes, latency-bound at a
 * few GB/s, and it was 13% of the draw engine's time in Drakengard 3's battle
 * areas. The lanes run in parallel and cover every byte as before. */
static u64 eng_texture_content_hash_impl(u32 location, u32 offset, u32 span, int* readable);
static u64 eng_texture_content_hash(u32 location, u32 offset, u32 span, int* readable)
{
    RSX_B2P_ENTER(B2P_TEXHASH); RSX_B2P_ADD(B2C_TEXHASH, 1);
    const u64 v = eng_texture_content_hash_impl(location, offset, span, readable);
    RSX_B2P_LEAVE();
    return v;
}
static u64 eng_texture_content_hash_impl(u32 location, u32 offset, u32 span,
                                         int* readable)
{
    const u8* src = span ? eng_guest_ptr(NULL, location, offset, span) : NULL;
    if (!src) { *readable = 0; return 0; }
    BEACON_WAIT("tex_hash");
    const u64 P = 1099511628211ull;
    u64 h[8];
    for (int k = 0; k < 8; k++) h[k] = 1469598103934665603ull + (u64)k;
    u32 i = 0;
    for (; i + 64 <= span; i += 64) {
        u64 w[8];
        memcpy(w, src + i, sizeof w);
        for (int k = 0; k < 8; k++) h[k] = (h[k] ^ w[k]) * P;
    }
    u64 hash = 1469598103934665603ull;
    for (int k = 0; k < 8; k++) hash = (hash ^ h[k]) * P;
    for (; i < span; i++) { hash ^= src[i]; hash *= P; }
    BEACON_POP();
    *readable = 1;
    return hash;
}

static int eng_staging_reserve(u32 bytes)
{
    if (g.tex_staging_cap >= bytes) return 1;
    u8* n = (u8*)realloc(g.tex_staging, bytes);
    if (!n) return 0;
    g.tex_staging = n;
    g.tex_staging_cap = bytes;
    return 1;
}

/* Decode a guest texture out of guest memory and hand every face and level to
 * the backend. Returns the backend handle, or 0. */
/* What the walker did in the current frame, for the stutter line in
 * eng_present (DOD3_STUTTER_MS). */
static struct { u32 tex; u64 tex_bytes; u32 vc_store; u64 vc_bytes; u32 pipes; double pipe_ms;
                u32 vc_evicts; double vc_evict_ms; u32 vc_evicted; u32 draws; double draw_ms; u32 tex_skipped;
                double tex_ms, tex_create_ms, tex_decode_ms, tex_stage_ms, vc_wrap_ms; } s_fstat;
static double eng_now_ms(void)
{
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* Uploads of an 8-bit texture at least 256 wide: a decoded video's plane
 * (Bink hands the RSX its Y, Cb and Cr that way). A host can tell from the
 * count moving that a movie is playing. */
volatile unsigned g_rsx_video_plane_uploads;

/* The draws of the last frame presented: a few dozen for a menu, a loading
 * screen or a movie, hundreds and up with a scene on screen. */
volatile unsigned g_rsx_frame_draws;

/* rsx_canvas_remap_set (rsx_draw_engine.h): the scale and move for the
 * marked draws, in the target's pixels. */
static volatile float s_canvas_remap[3];
void rsx_canvas_remap_set(u32 mark, float k, float x, float y)
{
    s_canvas_remap[0] = k;
    s_canvas_remap[1] = x;
    s_canvas_remap[2] = y;
    g_rsx_canvas_mark = mark;
}

/* rsx_present_layout (rsx_draw_engine.h). */
static volatile float s_content_aspect, s_screen_aspect, s_window_aspect;
static unsigned s_frame_draws;   /* the frame's so far */
static int s_frame_video;        /* the frame has drawn a movie's picture */
void rsx_present_set_content_aspect(float aspect) { s_content_aspect = aspect; }
void rsx_present_set_screen_aspect(float aspect) { s_screen_aspect = aspect; }
/* TOD_ASPECT (ultrawide): the whole frame is shown stretched to this shape (width / height) instead of its own;
 * the game then renders its 3D view with the same aspect, so the squeezed 16:9 frame is stretched back. 0 = off. */
static volatile float s_frame_shape;
void rsx_present_set_frame_shape(float aspect) { s_frame_shape = aspect; }
float rsx_present_window_aspect(void) { return s_window_aspect; }

/* The largest a-shaped rectangle in (x, y, w, h), in its middle. */
static void present_fit(float a, float x, float y, float w, float h, float r[4])
{
    if (w > h * a) { r[2] = h * a; r[3] = h; }
    else { r[2] = w; r[3] = w / a; }
    r[0] = x + (w - r[2]) * 0.5f;
    r[1] = y + (h - r[3]) * 0.5f;
}

int rsx_present_layout(u32 fw, u32 fh, u32 ww, u32 wh, float vp[4], int rect[4])
{
    vp[0] = vp[1] = 0.0f; vp[2] = (float)ww; vp[3] = (float)wh;
    rect[0] = rect[1] = 0; rect[2] = (int)ww; rect[3] = (int)wh;
    if (!fw || !fh || !ww || !wh) return 0;
    const float fa = (float)fw / (float)fh, wa = (float)ww / (float)wh;
    s_window_aspect = wa;
    /* The screen: the shape the player has, in the middle of the window. */
    const float sa = s_screen_aspect > 0.0f ? s_screen_aspect : wa;
    float scr[4];
    present_fit(sa, 0.0f, 0.0f, (float)ww, (float)wh, scr);
    if (s_frame_shape > 0.0f && !s_frame_video) {
        /* ultrawide: the whole frame, stretched to the chosen shape, in the middle of the screen (a movie keeps its own, below) */
        float dw[4];
        present_fit(s_frame_shape, scr[0], scr[1], scr[2], scr[3], dw);
        vp[0] = dw[0]; vp[1] = dw[1]; vp[2] = dw[2]; vp[3] = dw[3];
        rect[0] = (int)(dw[0] + 0.5f); rect[1] = (int)(dw[1] + 0.5f);
        rect[2] = (int)(dw[0] + dw[2] + 0.5f) - rect[0]; rect[3] = (int)(dw[1] + dw[3] + 0.5f) - rect[1];
        if (rect[2] > (int)ww) rect[2] = (int)ww;
        if (rect[3] > (int)wh) rect[3] = (int)wh;
        return rect[0] || rect[1] || rect[2] != (int)ww || rect[3] != (int)wh;
    }
    /* The picture: the view the title boxed to the content's shape, its
     * bars left out, or the whole frame ... */
    float ca = s_content_aspect;
    const float cr = ca / fa;
    if (!(ca > 0.0f) || (cr > 0.99f && cr < 1.01f) || s_frame_video) ca = fa;
    { /* RSX_PRESENT_LAYOUT_LOG=1: each change of what is shown */
        static int log = -1;
        static float last_ca = -1.0f, last_sa = -1.0f;
        static unsigned n;
        if (log < 0) log = getenv("RSX_PRESENT_LAYOUT_LOG") ? 1 : 0;
        n++;
        if (log && (ca != last_ca || sa != last_sa))
            fprintf(stderr, "[present] #%u: showing %.4f (content %.4f, movie %d), screen %.4f, window %ux%u\n",
                    n, ca, s_content_aspect, s_frame_video, sa, ww, wh);
        last_ca = ca; last_sa = sa;
    }
    float pic[4], dst[4];
    present_fit(ca, 0.0f, 0.0f, (float)fw, (float)fh, pic);
    /* ... as big as the screen takes it, in its middle. */
    present_fit(ca, scr[0], scr[1], scr[2], scr[3], dst);
    const float k = dst[2] / pic[2];
    vp[0] = dst[0] - pic[0] * k; vp[1] = dst[1] - pic[1] * k;
    vp[2] = (float)fw * k; vp[3] = (float)fh * k;
    rect[0] = (int)(dst[0] + 0.5f); rect[1] = (int)(dst[1] + 0.5f);
    rect[2] = (int)(dst[0] + dst[2] + 0.5f) - rect[0]; rect[3] = (int)(dst[1] + dst[3] + 0.5f) - rect[1];
    if (rect[2] > (int)ww) rect[2] = (int)ww;
    if (rect[3] > (int)wh) rect[3] = (int)wh;
    return rect[0] || rect[1] || rect[2] != (int)ww || rect[3] != (int)wh;
}

/* The port's stand-ins for the title's textures (rsx_draw_engine.h). */
static rsx_texture_standin_fn s_standin_fn;
static void* s_standin_user;
static volatile u32 s_standin_gen;
static const void* s_upload_standin;   /* what the last eng_texture_upload used */

void rsx_draw_engine_set_texture_standin(rsx_texture_standin_fn fn, void* user)
{
    s_standin_fn = fn;
    s_standin_user = user;
    s_standin_gen++;
}

void rsx_draw_engine_texture_standins_changed(void)
{
    s_standin_gen++;
}

static int eng_standin_ask(u32 fmt, u32 w, u32 h, u32 levels, const u8* src,
                           u32 span, int cube, rsx_texture_standin* sub)
{
    memset(sub, 0, sizeof *sub);
    return s_standin_fn && !cube &&
           s_standin_fn(s_standin_user, fmt, w, h, levels, src, span, sub) &&
           sub->bytes && sub->w && sub->h &&
           sub->w <= ENG_MAX_TEX_DIM && sub->h <= ENG_MAX_TEX_DIM;
}

static u32 eng_texture_upload_impl(u32 location, u32 offset, u32 fmt, u32 w, u32 h, u32 levels, u32 pitch, int cube, u32 remap);
static u32 eng_texture_upload(u32 location, u32 offset, u32 fmt, u32 w, u32 h,
                              u32 levels, u32 pitch, int cube, u32 remap)
{
    RSX_B2P_ENTER(B2P_TEXUP); RSX_B2P_ADD(B2C_TEXUP, 1);
    const u32 v = eng_texture_upload_impl(location, offset, fmt, w, h, levels, pitch, cube, remap);
    RSX_B2P_LEAVE();
    return v;
}
static u32 eng_texture_upload_impl(u32 location, u32 offset, u32 fmt, u32 w, u32 h,
                                   u32 levels, u32 pitch, int cube, u32 remap)
{
    const u32 span = eng_texture_span(fmt, w, h, levels, pitch, cube);
    const u8* src = eng_guest_ptr(NULL, location, offset, span);
    if (!src) return 0;
    s_fstat.tex++; s_fstat.tex_bytes += span;
    if ((fmt & 0x9Fu) == 0x81u /* CELL_GCM_TEXTURE_B8 */ && w >= 256) g_rsx_video_plane_uploads++;

    /* The port's stand-in, if it has one for these bytes: from here on the
     * texture is the stand-in's, under the guest's place in the cache. */
    s_upload_standin = NULL;
    { rsx_texture_standin sub;
      if (eng_standin_ask(fmt, w, h, levels, src, span, cube, &sub)) {
          s_upload_standin = src = sub.bytes;
          fmt = sub.fmt; w = sub.w; h = sub.h;
          levels = sub.levels ? sub.levels : 1; pitch = sub.pitch;
      } }

    rsx_tex_level lv[RSX_MAX_TEXTURE_LEVELS];
    const u32 nlv = rsx_texture_mip_chain(fmt, w, h, levels, pitch, lv);
    if (!nlv || !lv[0].tl.face_bytes) return 0;
    const u32 faces = cube ? 6u : 1u;
    const u32 face_stride = cube
        ? rsx_texture_cube_face_stride(fmt, w, h, levels, pitch) : 0u;

    /* TEX_DUMP_DIR=<dir>: log every upload, and write each 8-bit (B8) upload at
     * least 256 wide -- a Bink video plane -- as a PPM, so "is the decoder
     * producing pixels" can be answered by looking at the image. */
    { static const char* dd = (const char*)1; static int ul = -1;
      if (dd == (const char*)1) dd = getenv("TEX_DUMP_DIR");
      if (ul < 0) ul = getenv("TEX_UP_LOG") ? 1 : 0;   /* the log line alone, no PPMs */
      if (dd || ul) {
          static int n = 0, saved = 0;
          if ((n < 400 && ++n) || (fmt & 0x9Fu) == 0x81u /* CELL_GCM_TEXTURE_B8, any layout flags */) {
              /* A B8 plane's mean, so "are the decoder's pixels there" is in
               * the log line itself: a Bink plane of a real picture averages
               * tens to a hundred-odd; all-zero planes shade the YUV->RGB
               * quad solid green (0,135,0). */
              unsigned long b8mean = 0;
              if ((fmt & 0x9Fu) == 0x81u && w && h) {
                  const u32 row = pitch ? pitch : w; unsigned long sum = 0;
                  for (u32 y = 0; y < h; y += 4) for (u32 x = 0; x < w; x += 4) sum += src[y * row + x];
                  b8mean = sum / (((h + 3) / 4) * ((w + 3) / 4)); }
              fprintf(stderr, "[tex-up] f%u fmt=0x%02X %ux%u pitch=%u levels=%u loc=%u off=0x%08X ea=0x%08X remap=0x%04X cube=%d mean=%lu\n",
                      g.frames, fmt, w, h, pitch, levels, location, offset,
                      cellGcmResolveLocated(location == RSX_LOCATION_LOCAL, offset), remap, cube, b8mean);
              /* TEX_UP_PEEK=<hex offset>: for that upload, the mean and first bytes of the SAME offset in BOTH
               * memories (the descriptor's location and the other one): tells a wrong location bit from an
               * empty texture (Fastoon's environment cube map 0x02660B00 uploads all-zero, 30/10). */
              { static long long pk = -2; if (pk == -2) { const char* e = getenv("TEX_UP_PEEK"); pk = e ? (long long)strtoul(e, NULL, 16) : -1; }
                if (pk >= 0 && (long long)offset == pk) {
                    for (u32 L = 0; L < 2; L++) {
                        const u8* p = eng_guest_ptr(NULL, L, offset, 4096);
                        unsigned long sum = 0, nz = 0;
                        if (p) for (u32 i = 0; i < 4096; i++) { sum += p[i]; nz += p[i] != 0; }
                        fprintf(stderr, "[tex-peek] off=0x%08X location %u (%s): %s mean=%lu nonzero=%lu/4096 bytes:", offset, L,
                                L == RSX_LOCATION_LOCAL ? "VRAM" : "main", p ? "ok" : "UNMAPPED", p ? sum / 4096 : 0, nz);
                        if (p) for (u32 i = 0; i < 16; i++) fprintf(stderr, " %02X", p[i]);
                        fputc('\n', stderr);
                    } } }
          }
          static int b8n = 0;
          /* TEX_DUMP_FROM=<frame>: spend the PPM budget from that frame on. */
          static long dfrom = -1; if (dfrom < 0) { const char* e = getenv("TEX_DUMP_FROM"); dfrom = e ? atol(e) : 0; }
          if (dd && (long)g.frames >= dfrom && (fmt & 0x9Fu) == 0x81u /* CELL_GCM_TEXTURE_B8, any layout flags */ && w >= 256 &&
              (b8n++ % 150) == 0 && saved < 80) {   /* one plane every ~1.5 s of video */
              char path[512]; snprintf(path, sizeof path, "%s/b8_%03d_%ux%u_%08X.ppm", dd, saved++, w, h, offset);
              FILE* f = fopen(path, "wb");
              if (f) { fprintf(f, "P6\n%u %u\n255\n", w, h);
                  const u32 row = pitch ? pitch : w;
                  unsigned long sum = 0;
                  for (u32 y = 0; y < h; y++) for (u32 x = 0; x < w; x++) {
                      u8 v = src[y * row + x]; sum += v; fputc(v, f); fputc(v, f); fputc(v, f); }
                  fclose(f);
                  fprintf(stderr, "[tex-up] saved %s mean=%lu\n", path, sum / ((unsigned long)w * h)); }
          } } }

    const double t_create = eng_now_ms();
    BEACON_WAIT("tex_create");
    const u32 handle = g.be->texture_create(g.be->user, eng_texfmt(lv[0].tl.fmt),
                                            w, h, nlv, faces, remap, fmt);
    BEACON_POP();
    s_fstat.tex_create_ms += eng_now_ms() - t_create;
    if (!handle) return 0;
    BEACON_WAIT("tex_upload");

    for (u32 f = 0; f < faces; f++) {
        const u8* face = src + (size_t)f * face_stride;
        for (u32 m = 0; m < nlv; m++) {
            const rsx_tex_layout* tl = &lv[m].tl;
            if (!eng_staging_reserve(tl->dst_row_bytes * tl->rows)) {
                g.be->texture_release(g.be->user, handle);
                BEACON_POP();
                return 0;
            }
            const double t_d0 = eng_now_ms();
            rsx_texture_decode(g.tex_staging, tl->dst_row_bytes,
                               face + lv[m].offset, lv[m].w, lv[m].h, tl,
                               rsx_texture_argb_is_rgba());
            const double t_d1 = eng_now_ms();
            g.be->texture_upload(g.be->user, handle, f, m, lv[m].w, lv[m].h,
                                 g.tex_staging, tl->dst_row_bytes, tl->rows);
            s_fstat.tex_decode_ms += t_d1 - t_d0; s_fstat.tex_stage_ms += eng_now_ms() - t_d1;
        }
    }
    BEACON_POP();
    s_fstat.tex_ms += eng_now_ms() - t_create;
    return handle;
}

/* The cache slot for a texture unit's bytes, keyed on where they are and what
 * the registers say they are. An already-cached entry is re-hashed at most
 * once per presented frame and re-decoded when the guest changed it, which is
 * what makes an animated UI or a video texture update at all. A miss with the
 * cache full evicts the least recently used entry rather than returning
 * nothing: returning white "made the recovered orphanage render as flat
 * green/black geometry" (rsx_live_draw.c:2326-2334). */
/* The write-watch (ps3emu/vm_watch.h) in place of a hash per texture per
 * frame: when every page under a texture is protected and none was written
 * since the last hash, the bytes are known unchanged. RSX_TEX_WATCH=0, or a
 * host without it, hashes as before. -1 undecided; a host may switch it. */
/* RSX_AA: fxaa (1) -- FXAA on the presented frame (rsx_present_passes.h);
 * msaa2 / msaa4 / msaa8 (2, 4, 8) -- the 3D passes multisampled (the
 * engines' eng_ms_*). Read on first use; the port's settings page sets it
 * while running. */
int g_rsx_aa = -1;
int rsx_aa_mode(void)
{
    if (g_rsx_aa < 0) {
        const char* e = getenv("RSX_AA");
        g_rsx_aa = 0;
        if (e && (!strcmp(e, "fxaa") || !strcmp(e, "FXAA") || !strcmp(e, "1"))) g_rsx_aa = 1;
        else if (e && !strcmp(e, "msaa2")) g_rsx_aa = 2;
        else if (e && !strcmp(e, "msaa4")) g_rsx_aa = 4;
        else if (e && !strcmp(e, "msaa8")) g_rsx_aa = 8;
    }
    return g_rsx_aa;
}

/* Set (by the port's settings page) when RSX_VSYNC / RSX_DISPLAY /
 * RSX_WINDOW have changed; the backend applies them at its next message
 * pump and clears it. */
volatile int g_rsx_display_reload;
int g_eng_tex_watch = -1;
static unsigned long long s_tex_skips_total;   /* hashes the watch saved, for [frametime] */
static int eng_tex_watch_on(void)
{
    if (g_eng_tex_watch < 0)
        g_eng_tex_watch = s_guest_reader ? (s_guest_watch ? 1 : 0) : (vm_watch_available() ? 1 : 0);
    return g_eng_tex_watch > 0;
}
/* Arm the watch over a texture's bytes; 1 when the stamp can be trusted. */
static int eng_tex_watch_arm_impl(u32 location, u32 offset, u32 span, u64* stamp);
static int eng_tex_watch_arm(u32 location, u32 offset, u32 span, u64* stamp)
{
    RSX_B2P_ENTER(B2P_TEXWATCH); RSX_B2P_ADD(B2C_TEXWATCH, 1);
    const int v = eng_tex_watch_arm_impl(location, offset, span, stamp);
    RSX_B2P_LEAVE();
    return v;
}
static int eng_tex_watch_arm_impl(u32 location, u32 offset, u32 span, u64* stamp)
{
    u32 unwatched = 1;
    if (s_guest_reader) {
        *stamp = s_guest_watch ? s_guest_watch(s_guest_user, location, offset, span, &unwatched) : 0;
        return s_guest_watch && unwatched == 0;
    }
    const u32 ea = cellGcmResolveLocated(location == RSX_LOCATION_LOCAL, offset);
    if (!ea) { *stamp = 0; return 0; }
    if (s_wk_n) { while (__atomic_exchange_n(&s_arm_lock, 1, __ATOMIC_ACQUIRE)) _mm_pause(); }
    *stamp = vm_watch_arm(ea, span, &unwatched);
    if (s_wk_n) __atomic_store_n(&s_arm_lock, 0, __ATOMIC_RELEASE);
    return unwatched == 0;
}

static u32 eng_texture_slot(u32 location, u32 offset, u32 fmt, u32 w, u32 h,
                            u32 levels, u32 pitch, int cube, u32 remap)
{
    if (!w || !h || w > ENG_MAX_TEX_DIM || h > ENG_MAX_TEX_DIM) return 0;
    const int slow = eng_slow();
    u32 span = 0;
    eng_texture* e = NULL;
    if (slow) {
        span = eng_texture_span(fmt, w, h, levels, pitch, cube);
        if (!span) return 0;
        for (u32 i = 0; i < g.n_textures && !e; i++) {
            eng_texture* c = &g.textures[i];
            if (c->location == location && c->offset == offset &&
                c->format == fmt && c->width == w && c->height == h &&
                c->pitch == pitch && c->remap == remap &&
                c->cubemap == (u32)(cube != 0))
                e = c;
        }
    } else {
        e = eng_tex_find(location, offset, fmt, w, h, pitch, remap, (u32)(cube != 0));
        if (e) {
            /* The span depends on the level count, which is not part of the
             * key: work it out again only when that changes. */
            if (e->span_levels != levels + 1u) {
                e->span = eng_texture_span(fmt, w, h, levels, pitch, cube);
                e->span_levels = levels + 1u;
            }
            span = e->span;
        } else {
            span = eng_texture_span(fmt, w, h, levels, pitch, cube);
        }
        if (!span) return 0;
    }

    if (e) {
        /* The port's stand-ins changed: ask about this texture again, and
         * upload it again only if the answer is another one. */
        const u32 standin_gen = s_standin_gen;
        if (e->handle && e->standin_gen != standin_gen) {
            e->standin_gen = standin_gen;
            const u8* src = eng_guest_ptr(NULL, location, offset, span);
            rsx_texture_standin sub;
            const void* now = src && eng_standin_ask(fmt, w, h, levels, src, span, cube, &sub)
                              ? (const void*)sub.bytes : NULL;
            if (src && now != e->standin) {
                const u32 fresh = eng_texture_upload(location, offset, fmt, w, h,
                                                     levels, pitch, cube, remap);
                if (fresh) {
                    g.be->texture_release(g.be->user, e->handle);
                    e->handle = fresh;
                    e->standin = s_upload_standin;
                }
            }
        }
        if (e->handle && e->last_hash_frame != g.frames) {
            e->last_hash_frame = g.frames;
            int need_hash = 1;
            /* Not while capturing: the hash's read is what records the
             * texture's pages in the capture. */
            if (eng_tex_watch_on() && !g_rsx_capture_on) {
                /* Armed before the hash, so a write during or after it
                 * changes next frame's stamp. */
                u64 st = 0;
                const int watched = eng_tex_watch_arm(location, offset, span, &st);
                if (watched && e->watched && st == e->watch_stamp) need_hash = 0;
                e->watch_stamp = st; e->watched = (u32)watched;
                if (!need_hash) { s_fstat.tex_skipped++; s_tex_skips_total++; }
                /* RSX_TEX_WATCH_CHECK=1: hash anyway, and report a texture
                 * whose bytes changed while the watch said they had not -- a
                 * write it missed (a kernel write nobody touched, a page
                 * re-opened behind its back). The texture is still updated. */
                { static int chk = -1;
                  if (chk < 0) chk = getenv("RSX_TEX_WATCH_CHECK") ? 1 : 0;
                  if (chk && !need_hash) {
                      int rd = 0;
                      const u64 h2 = eng_texture_content_hash(location, offset, span, &rd);
                      static unsigned long long n_chk, n_bad;
                      n_chk++;
                      if (rd && h2 != e->content_hash) {
                          if (n_bad++ < 20)
                              fprintf(stderr, "[tex-watch] MISSED a write: loc %u off 0x%08X %ux%u fmt 0x%02X span %u (frame %u)\n",
                                      location, offset, w, h, fmt, span, g.frames);
                          need_hash = 1;
                      }
                      if ((n_chk & 0xFFFFF) == 0)
                          fprintf(stderr, "[tex-watch] check: %llu skipped hashes verified, %llu missed\n", n_chk, n_bad);
                  } }
            }
            int readable = 0;
            const u64 hash = need_hash ? eng_texture_content_hash(location, offset, span, &readable) : 0;
            if (need_hash && readable && hash != e->content_hash) {
                /* RSX_TEX_REUP_LOG=1: which textures the game rewrites (and the walker decodes again) over and over -- the log
                 * line says when one is seen for the 2nd, 3rd, 100th, 1000th and 10000th time. */
                { static int rl = -1; if (rl < 0) rl = getenv("RSX_TEX_REUP_LOG") ? 1 : 0;
                  if (rl) { const u32 n = ++e->reups;
                      if (n == 2 || n == 3 || n == 100 || n == 1000 || n == 10000)
                          fprintf(stderr, "[tex-reup] f%u %u:0x%08X fmt 0x%02X %ux%u pitch %u levels %u span %u: rewritten %u times\n",
                                  g.frames, location, offset, fmt, w, h, pitch, levels, span, n); } }
                const u32 fresh = eng_texture_upload(location, offset, fmt, w, h,
                                                     levels, pitch, cube, remap);
                if (fresh) {
                    g.be->texture_release(g.be->user, e->handle);
                    e->handle = fresh;
                    e->content_hash = hash;
                    e->standin = s_upload_standin;
                }
            }
        }
        e->last_use_serial = ++g.texture_use_serial;
        return e->handle;
    }

    u32 index;
    u32 evicted = 0;
    if (g.n_textures < ENG_MAX_TEXTURES) {
        index = g.n_textures++;
    } else {
        index = 0;
        for (u32 i = 1; i < g.n_textures; i++)
            if (g.textures[i].last_use_serial < g.textures[index].last_use_serial)
                index = i;
        evicted = g.textures[index].handle;
    }

    {   /* the new entry: its own block, the hit's pointer above is NULL here */
    eng_texture e;
    memset(&e, 0, sizeof(e));
    e.location = location; e.offset = offset; e.format = fmt;
    e.width = w; e.height = h; e.pitch = pitch;
    e.remap = remap; e.cubemap = (u32)(cube != 0);
    e.span = span; e.span_levels = levels + 1u;
    e.last_hash_frame = g.frames;
    e.last_use_serial = ++g.texture_use_serial;
    if (eng_tex_watch_on()) { u64 st = 0; e.watched = (u32)eng_tex_watch_arm(location, offset, span, &st); e.watch_stamp = st; }
    { int readable = 0;
      e.content_hash = eng_texture_content_hash(location, offset, span, &readable); }
    e.standin_gen = s_standin_gen;
    e.handle = eng_texture_upload(location, offset, fmt, w, h, levels, pitch,
                                  cube, remap);
    e.standin = s_upload_standin;
    if (e.handle && evicted) g.be->texture_release(g.be->user, evicted);
    if (e.handle || !evicted) {
        g.textures[index] = e;
        if (evicted) eng_tex_index_rebuild(); else eng_tex_index_add(index);
    }
    return e.handle;
    }
}

/* SET_TEXTURE_FILTER's min field is [18:16] (1 NEAREST, 2 LINEAR, then 3..6,
 * the four combinations of a nearest/linear minification with a
 * nearest/linear mip filter) and mag [26:24]; SET_TEXTURE_CONTROL0 carries
 * max LOD at [18:7] and min LOD at [30:19], both 4.8 fixed point. RSX's LOD
 * bias, SET_TEXTURE_FILTER [12:0] (signed 4.8, as RPCS3 decodes it), is applied
 * from 26/10 together with the user's RSX_TEX_LOD_BIAS (tod.ini, default 0;
 * negative = sharper); only on a sampler that reads mips. */
static void eng_decode_sampler(u32 filter, u32 wrap, u32 control0, u32 border,
                               rsx_be_sampler_desc* out)
{
    out->border_argb = border;
    { static float user = 1e9f;
      if (user > 1e8f) { const char* e = getenv("RSX_TEX_LOD_BIAS"); user = e && *e ? (float)atof(e) : 0.0f;
                         if (user < -4.0f) user = -4.0f; if (user > 4.0f) user = 4.0f; }
      int raw = (int)(filter & 0x1FFFu); if (raw & 0x1000) raw -= 0x2000;   /* signed 13 bit, 1/256 */
      float b = (float)raw / 256.0f + user;
      if (b < -7.5f) b = -7.5f; if (b > 7.5f) b = 7.5f;
      out->lod_bias = b; }
    const u32 minf = (filter >> 16) & 7u;
    const u32 magf = (filter >> 24) & 7u;
    out->min_linear  = (u8)(minf == 2 || minf == 4 || minf == 6);
    out->mag_linear  = (u8)(magf == 2);
    out->mip_present = (u8)(minf >= 3);
    out->mip_linear  = (u8)(minf == 5 || minf == 6);
    out->wrap_s = (u8)(wrap & 0xFu);
    out->wrap_t = (u8)((wrap >> 8) & 0xFu);
    out->wrap_r = (u8)((wrap >> 16) & 0xFu);
    out->min_lod = (float)((control0 >> 19) & 0xFFFu) / 256.0f;
    out->max_lod = out->mip_present ? (float)((control0 >> 7) & 0xFFFu) / 256.0f
                                    : 0.0f;
    if (out->max_lod < out->min_lod) out->max_lod = out->min_lod;
    if (!out->mip_present) out->lod_bias = 0.0f;
}

/* Texel conversions per unit for rsx_fp_set_texel_ops: TEXTURE_ADDRESS gamma
 * (bits 20-23, R G B A) and UNSIGNED_REMAP_BIASED expansion (bits 12-15 == 1),
 * on the formats the RSX applies them to (RPCS3 get_format_features: gamma
 * and expansion on the 8-bit and compressed colour formats, expansion alone
 * on depth, X16, Y16_X16 and HILO8, neither on float formats). Expansion
 * skips channels the remap fills with a constant and channels gamma already
 * converts. RSX_NO_TEXEL_OPS=1 turns both off, RSX_NO_TEX_GAMMA=1 gamma only. */
static u32 eng_texel_ops(u32 ops[16])
{
    s_tdec_pass = 2;
    static int off = -1, no_gamma = -1;
    if (off < 0) { off = getenv("RSX_NO_TEXEL_OPS") ? 1 : 0; no_gamma = getenv("RSX_NO_TEX_GAMMA") ? 1 : 0; }
    u32 mask = 0;
    memset(ops, 0, 16 * sizeof ops[0]);
    if (off) return 0;
    const u32 reads = eng_fp_reads();
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES && u < 16; u++) {
        if (!eng_tex_unit_on(u, reads)) continue;
        rsx_dsp_texture t;
        eng_get_tex(RV, u, &t);
        if (!t.enabled) continue;
        const u32 base = t.format & RSX_TEX_FMT_BASE_MASK & ~(u32)RSX_TEX_FMT_UNNORM;
        int can_gamma = 0, can_expand = 0, wide = 0;
        switch (base) {
        case 0x81: case 0x82: case 0x83: case 0x84: case 0x85: case 0x86: case 0x87:
        case 0x88: case 0x8B: case 0x8D: case 0x8E: case 0x8F: case 0x97: case 0x9D: case 0x9E:
            can_gamma = can_expand = 1; break;
        case 0x90: case 0x91: case 0x92: case 0x93: case 0x98:
            can_expand = 1; break;
        case 0x94: case 0x95:
            can_expand = 1; wide = 1; break;
        default: break;
        }
        u32 gamma = can_gamma && !no_gamma ? (t.wrap >> 20) & 0xFu : 0u;
        u32 expand = 0;
        if (can_expand && ((t.wrap >> 12) & 0xFu) == 1u) {
            const u32 r = t.remap & 0xFFFFu;
            const u32 op_r = (r >> 10) & 3u, op_g = (r >> 12) & 3u, op_b = (r >> 14) & 3u, op_a = (r >> 8) & 3u;
            expand = (op_r == 2u ? 1u : 0u) | (op_g == 2u ? 2u : 0u) | (op_b == 2u ? 4u : 0u) | (op_a == 2u ? 8u : 0u);
            expand &= ~gamma;
        }
        if (!gamma && !expand) continue;
        ops[u] = gamma | (expand << 4) | (wide ? 0x100u : 0u);
        mask |= 1u << u;
    }
    return mask;
}

/* A fragment program's identity across draws: its structure without the
 * inline constants (which titles patch per draw), as the pipeline log prints
 * it ("fp-struct="). Per-title corrections name programs by it. */
static u64 eng_fp_struct_id(const u8* fp_uc, u32 fp_size)
{
    return fp_uc ? rsx_fp_structural_hash(fp_uc, fp_size, 1469598103934665603ull) : 0;
}

/* RSX_FP_SAT_ALPHA=<fp-struct>[,<fp-struct>...]: clamp those programs' first
 * colour export alpha to [0, 1]. A per-title correction, set by the title's
 * runner: Drakengard 3's point-light shaft mask writes
 * max(distance / radius, behind-the-light)^4 unclamped into an FP16 target,
 * and its composite multiplies the scene by about 0.3 + 1.05 * mask^2 -- the
 * smoky doorway and the windows of the village interior came out 15-70x too
 * bright. Nothing in the microcode or the FP16 path bounds it, and RPCS3 does
 * not either; clamped, the room matches the original by eye. */
static int eng_fp_sat_alpha(const u8* fp_uc, u32 fp_size)
{
    static u64 ids[16]; static int n = -1;
    if (n < 0) {
        n = 0;
        const char* e = getenv("RSX_FP_SAT_ALPHA");
        while (e && *e && n < 16) {
            char* end; const u64 v = strtoull(e, &end, 16);
            if (end == e) break;
            ids[n++] = v;
            e = (*end == ',') ? end + 1 : end;
        }
    }
    if (!n) return 0;
    const u64 id = eng_fp_struct_id(fp_uc, fp_size);
    for (int i = 0; i < n; i++) if (ids[i] == id) return 1;
    return 0;
}

/* ---- pipelines ----------------------------------------------------------- */

static u32 eng_vtex_mask_raw(void)
{
    u32 mask = 0;
    for (u32 u = 0; u < RSX_DSP_NUM_VERTEX_TEXTURES; u++) {
        rsx_dsp_vertex_texture vt;
        rsx_dsp_get_vertex_texture(RV, u, &vt);
        if (vt.enabled && vt.width && vt.height) mask |= 1u << u;
    }
    return mask;
}
/* opt102: the mask is a function of the 4 vertex-texture blocks (4 x 8 words at 0x900) only. */
static u32 eng_vtex_mask(void)
{
    if (!eng_tm_mode() || s_tm_raw || g_rsx_dsp_trace) return eng_vtex_mask_raw();
    static ENG_TLS u32 key[32]; static ENG_TLS u32 val; static ENG_TLS int valid;
    const u32* r = &RV->regs[0x900u >> 2];
    if (valid && !memcmp(key, r, sizeof key)) {
        if (s_tm_mode == 2) { s_tm_vt_n++; if (eng_vtex_mask_raw() != val && s_tm_vt_bad++ < 16) fprintf(stderr, "[texmask-check] MISMATCH vtex mask frame %u\n", g.frames); }
        return val;
    }
    val = eng_vtex_mask_raw(); memcpy(key, r, sizeof key); valid = 1;
    return val;
}

/* Shadow-map units: an enabled depth-format texture (DEPTH24_D8, DEPTH16 and
 * their float variants) whose TEXTURE_ADDRESS carries a compare function
 * (bits 28-31, CELL_GCM_TEXTURE_ZFUNC_*; 0 = NEVER = no comparison). Such a
 * unit returns the comparison result, not the depth (see
 * rsx_fp_set_shadow_units). RSX_NO_SHADOW_CMP=1 turns it off. */
static u32 eng_shadow_units(u8 funcs[16])
{
    s_tdec_pass = 1;
    static int off = -1; if (off < 0) off = getenv("RSX_NO_SHADOW_CMP") ? 1 : 0;
    u32 mask = 0;
    memset(funcs, 0, 16);
    if (off) return 0;
    const u32 reads = eng_fp_reads();
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES && u < 16; u++) {
        if (!eng_tex_unit_on(u, reads)) continue;
        rsx_dsp_texture t;
        eng_get_tex(RV, u, &t);
        if (!t.enabled) continue;
        const u32 base = t.format & RSX_TEX_FMT_BASE_MASK & ~(u32)RSX_TEX_FMT_UNNORM;
        if (base < 0x90u || base > 0x93u) continue;
        const u32 zf = (t.wrap >> 28) & 0xFu;
        if (!zf || zf > 7u) continue;
        mask |= 1u << u; funcs[u] = (u8)zf;
        /* shad114: what the game asks for at the map's edge -- wrap S/T (guest codes: 1 repeat, 2 mirror, 3 clamp-to-edge, 4 border,
         * 5 clamp, ...) of every shadow unit, logged once per unit and value (max 32 lines). */
        { static u32 seen_w[16]; static int nlog;
          if (seen_w[u] != (t.wrap | 0x80000000u) && nlog < 32) { seen_w[u] = t.wrap | 0x80000000u; nlog++;
              fprintf(stderr, "[shad114 shadow-wrap] unit %u: loc %u off 0x%08X %ux%u fmt 0x%X zfunc %u wrapS %u wrapT %u (raw wrap 0x%08X) border 0x%08X frame %u\n",
                      u, t.location, t.offset, t.width, t.height, t.format, zf, t.wrap & 0xFu, (t.wrap >> 8) & 0xFu, t.wrap, t.border_color, g.frames); } }
    }
    return mask;
}

/* ---- RSX_SSR: reflections in screen space ---------------------------------------------------------------
 * The 313 fragment programs that sample the 32x32 cube map on unit 7 (fake reflections) try a ray through the
 * previous frame's scene colour and depth first (rsx_envL in rsx_fp_decompiler.c), the cube map is the fall-back.
 * The two images are snapshots taken once a frame, when a pass other than the scene first samples the scene
 * target (the bloom chain), so the HUD is not in them; reflective draws of the next frame bind them on two
 * fragment units the program does not read. Default off (RSX_SSR=1, or the F1 page). */
int g_rsx_ssr = -1;
static u32 s_ssr_col, s_ssr_dep;            /* snapshot handles of the last frame's scene (0: none yet) */
static u32 s_ssr_tgt = ~0u, s_ssr_zs = ~0u; /* surface / zeta slot of the reflective draws */
static u32 s_ssr_refl_frame = ~0u, s_ssr_snap_frame = ~0u;
static int s_ssr_cu = -1, s_ssr_du = -1;    /* units bound for the draw being set up (-1: none) */
static u32 s_ssr_stat_bound, s_ssr_stat_snaps;
static float s_ssr_cam_cur[16], s_ssr_cam_prev[16]; static int s_ssr_cam_have;   /* c[8..11] of the last reflective draw / of the frame the stored scene belongs to */
static float s_ssr_minup = 0.55f; static int s_ssr_axis = 1, s_ssr_par_init;   /* Y is up in this game's world space (measured with the debug view) */
static void eng_ssr_params(void)
{
    if (s_ssr_par_init) return;
    s_ssr_par_init = 1;
    const char* e = getenv("RSX_SSR_MINUP"); if (e && e[0]) s_ssr_minup = (float)atof(e);
    e = getenv("RSX_SSR_UPAXIS"); if (e && e[0]) s_ssr_axis = atoi(e);
}
static int eng_ssr_on(void)
{
    if (g_rsx_ssr < 0) { const char* e = getenv("RSX_SSR"); g_rsx_ssr = (e && e[0] && e[0] != '0') ? 1 : 0;
                         if (g_rsx_ssr) fprintf(stderr, "[ssr] RSX_SSR on\n"); }
    return g_rsx_ssr;
}
/* RSX_SSR_DEBUG=1: reflective pixels are painted, green = screen-space hit, red = cube-map fall-back.
 * RSX_SSR_VIEWS=<start>,<window>[,<parity>] (diagnostic A/B, with PS3RECOMP_FRAME_DUMP): from frame <start> the frame cycles
 * through seven windows of <window> frames: SSR off, on, debug, off, on, debug, off. The last frame of windows 0..2 of every
 * second 7-window cycle (parity 1 = odd cycles, 0 = even) is dumped as <dump>.<off|on|dbg>.<frame>.ppm. */
unsigned g_rsx_ssr_dump_tag, g_rsx_ssr_dump_frame;
static int s_ssr_dbg_env = -1, s_ssr_vstart = -2, s_ssr_vwin = 20, s_ssr_vpar = 1;
static int s_ssr_act, s_ssr_dbg, s_ssr_mode_frame = -1;
static void eng_ssr_frame_mode(void)
{
    if (s_ssr_mode_frame == (int)g.frames) return;
    s_ssr_mode_frame = (int)g.frames;
    if (s_ssr_vstart == -2) {
        const char* e = getenv("RSX_SSR_DEBUG"); s_ssr_dbg_env = (e && e[0] && e[0] != '0') ? 1 : 0;
        const char* v = getenv("RSX_SSR_VIEWS"); s_ssr_vstart = -1;
        if (v && *v) { int a = -1, b = 20, c = 1; if (sscanf(v, "%d,%d,%d", &a, &b, &c) >= 1) { s_ssr_vstart = a; s_ssr_vwin = b > 0 ? b : 20; s_ssr_vpar = c; } }
    }
    s_ssr_act = eng_ssr_on(); s_ssr_dbg = s_ssr_dbg_env;
    g_rsx_ssr_dump_tag = 0;
    if (s_ssr_act && s_ssr_vstart >= 0 && (int)g.frames >= s_ssr_vstart) {
        static const int vmap[7] = { 0, 1, 2, 0, 1, 2, 0 };
        const int idx = (int)g.frames - s_ssr_vstart, w = (idx / s_ssr_vwin) % 7, view = vmap[w];
        s_ssr_act = view != 0; s_ssr_dbg = view == 2;
        if ((idx % s_ssr_vwin) == s_ssr_vwin - 1 && w <= 2 && (((idx / (7 * s_ssr_vwin)) & 1) == (s_ssr_vpar & 1))) {
            g_rsx_ssr_dump_tag = (unsigned)view + 1; g_rsx_ssr_dump_frame = g.frames; }
    }
}
/* The scene target was just sampled by another pass: this frame's scene is complete (the HUD is not drawn yet).
 * The depth is the SSR's own copy (not the zeta's shared snapshot, which another reader of the depth releases in the
 * same frame): it lives until the next snapshot replaces it. */
static void eng_ssr_snapshot(u32 surface_slot)
{
    if (!eng_ssr_on() || surface_slot != s_ssr_tgt || s_ssr_refl_frame != g.frames || s_ssr_snap_frame == g.frames) return;
    if (s_ssr_zs >= g.n_zdepths || !g.be->color_snapshot || !g.be->depth_snapshot) return;
    eng_zdepth* z = &g.zdepths[s_ssr_zs];
    if (!z->handle || !z->had_write) return;
    s_ssr_snap_frame = g.frames;
    const u32 c = g.be->color_snapshot(g.be->user, g.surfaces[surface_slot].handle);
    const u32 d = g.be->depth_snapshot(g.be->user, z->handle, z->w, z->h);
    if (!c || !d) return;
    if (s_ssr_dep && g.be->texture_release) g.be->texture_release(g.be->user, s_ssr_dep);
    s_ssr_col = c; s_ssr_dep = d; s_ssr_stat_snaps++;
    memcpy(s_ssr_cam_prev, s_ssr_cam_cur, sizeof s_ssr_cam_prev); s_ssr_cam_have = 1;
    { static int n; if (n < 3 && ++n)
        fprintf(stderr, "[ssr] frame %u: scene s%u + zeta %u snapshots taken (col %u, depth %u)\n", g.frames, surface_slot, s_ssr_zs, c, d); }
}

static u32 eng_cube_mask(void)
{
    s_tdec_pass = 0;
    u32 mask = 0;
    const u32 reads = eng_fp_reads();
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++) {
        if (!eng_tex_unit_on(u, reads)) continue;
        rsx_dsp_texture t;
        eng_get_tex(RV, u, &t);
        /* A cube face is square by construction, so a non-square image
         * cannot be one; the fragment program is compiled against this mask
         * and a texturecube slot filled with a 2D texture is a validation
         * failure, not a wrong pixel. */
        if (t.enabled && t.cubemap && t.width && t.width == t.height)
            mask |= 1u << u;
    }
    return mask;
}

/* Are both of the guest's own programs resident? One answer, used by the
 * layout and by the pipeline, because they must not disagree: the built-in
 * program declares all sixteen inputs, so narrowing the layout for it would
 * leave the pipeline's vertex descriptor short of what the shader reads. */
/* What every draw would otherwise work out again from the resident vertex
 * program -- its size, the inputs it reads, its hash -- for as long as no
 * program word is uploaded (rsx.vp_gen) and the start slot stays put.
 * RSX_VP_INFO_NOCACHE=1 recomputes them on every draw. */
static ENG_TLS struct {
    int valid;
    u32 gen, start, instrs;
    int mask_valid; u32 input_mask;
    int hash_valid; u64 hash;
} s_vpi;

static int eng_vp_info_current(u32 start)
{
    static int off = -1;
    if (off < 0) off = getenv("RSX_VP_INFO_NOCACHE") ? 1 : 0;
    if (!off && s_vpi.valid && s_vpi.gen == RV->vp_gen && s_vpi.start == start)
        return 1;
    s_vpi.valid = 0;
    s_vpi.mask_valid = s_vpi.hash_valid = 0;
    return 0;
}

/* The answer for the draw being issued: the layout, the pipeline and the
 * snapshot check each ask, and nothing a draw does changes it. s_draw_open
 * is set by sink_end around the draw, so a question asked outside one is
 * answered afresh. */
static ENG_TLS int s_draw_open;
static struct { int valid, ret; const u8* vp; u32 vp_instrs; const u8* fp; u32 fp_size;
                int tex_valid; u32 tex;
                int scanned, scan_ok; u64 fp_key; } s_progs;

/* Per-program cache of the fragment program's size and scan (rsx_fp_program_size + rsx_fp_scan),
 * which used to run for EVERY draw. Direct-mapped by (location, offset); an entry keeps a copy of
 * the program's bytes and is valid only while the guest bytes still compare equal (memcmp of the
 * program, tens to hundreds of bytes), so a program the game rewrites in place is rescanned.
 * Both the size and the scan depend on nothing but those bytes. RSX_FPCACHE=0 turns it off. */
#define ENG_FPC_N     256u
#define ENG_FPC_BYTES 2048u
typedef struct {
    u32 loc, off, size, ncon; int valid, scan_ok; u64 key; u32 tex;
    u8 bytes[ENG_FPC_BYTES];
    u32 con[ENG_FPC_BYTES / 16][4];
} EngFpc;
static ENG_TLS EngFpc* s_fpc;
static ENG_TLS EngFpc* s_fpc_hit;    /* set by the program lookup on a validated hit */
static ENG_TLS EngFpc* s_fpc_fill;   /* set on a miss worth caching: filled once the scan has run */
static int s_fpc_on = -1;
static unsigned long long s_fpc_hits, s_fpc_miss;

static int eng_guest_programs_uncached(const u8** out_vp, u32* out_vp_instrs,
                                       const u8** out_fp, u32* out_fp_size);
static int eng_guest_programs_impl(const u8** out_vp, u32* out_vp_instrs, const u8** out_fp, u32* out_fp_size);
static int eng_guest_programs(const u8** out_vp, u32* out_vp_instrs,
                              const u8** out_fp, u32* out_fp_size)
{
    const int b2p_work = g_b2p_on && (!s_progs.valid || !s_progs.scanned);   /* only the call that does the lookup/scan is a bucket */
    if (b2p_work) rsx_b2p_enter(B2P_GPROG);
    const int v = eng_guest_programs_impl(out_vp, out_vp_instrs, out_fp, out_fp_size);
    if (b2p_work) rsx_b2p_leave();
    return v;
}
static int eng_guest_programs_impl(const u8** out_vp, u32* out_vp_instrs,
                                   const u8** out_fp, u32* out_fp_size)
{
    if (!s_draw_open || eng_slow())
        return eng_guest_programs_uncached(out_vp, out_vp_instrs, out_fp, out_fp_size);
    if (!s_progs.valid) {
        s_progs.ret = eng_guest_programs_uncached(&s_progs.vp, &s_progs.vp_instrs,
                                                  &s_progs.fp, &s_progs.fp_size);
        s_progs.valid = 1;
        /* The fragment program's one pass, here where the draw first asks:
         * its constants (into g.fp_constants, which only this draw reads),
         * its pipeline key and the units it samples. */
        s_progs.scanned = 0;
        if (s_progs.fp && s_progs.fp_size) {
            u32 tex = ~0u;
            const unsigned long long _a = s_wp_on > 0 ? wp_tsc() : 0;
            EngFpc* hit = s_fpc_hit; EngFpc* fill = s_fpc_fill;
            s_fpc_hit = s_fpc_fill = NULL;
            if (hit) {
                s_progs.scan_ok = hit->scan_ok; s_progs.fp_key = hit->key; tex = hit->tex;
                g.fp_constants.count = hit->ncon;
                if (hit->ncon) memcpy(g.fp_constants.values, hit->con, hit->ncon * 16u);
                s_fpc_hits++;
                if (s_fpc_on == 2) {   /* opt102 check: scan anyway (into a scratch block) and compare with what the hit gave */
                    static unsigned long long chk, bad;
                    rsx_fp_constant_block cb; u64 k2 = 0; u32 t2 = ~0u;
                    const int ok2 = rsx_fp_scan(s_progs.fp, s_progs.fp_size, &cb, &k2, &t2) >= 0;
                    const u32 sz2 = rsx_fp_program_size(s_progs.fp, ENG_FP_MAX_BYTES);
                    int diff = ok2 != hit->scan_ok || sz2 != s_progs.fp_size;
                    if (ok2 && hit->scan_ok) diff |= k2 != hit->key || t2 != hit->tex || cb.count != hit->ncon || (hit->ncon && memcmp(cb.values, hit->con, hit->ncon * 16u));
                    chk++;
                    if (diff && bad++ < 16) fprintf(stderr, "[fpscan-check] MISMATCH frame %u ok %d/%d size %u/%u key %llx/%llx tex %x/%x ncon %u/%u\n", g.frames, hit->scan_ok, ok2, s_progs.fp_size, sz2,
                                                    (unsigned long long)hit->key, (unsigned long long)k2, hit->tex, t2, hit->ncon, cb.count);
                    if ((chk & 0xFFFFu) == 0) fprintf(stderr, "[fpscan-check] %llu hits compared, %llu mismatches (misses %llu) frame %u\n", chk, bad, s_fpc_miss, g.frames);
                }
            } else {
                s_progs.scan_ok = rsx_fp_scan(s_progs.fp, s_progs.fp_size, &g.fp_constants, &s_progs.fp_key, &tex) >= 0;
                if (fill) {
                    const u32 n = s_progs.scan_ok ? g.fp_constants.count : 0;
                    if (n <= ENG_FPC_BYTES / 16) {
                        fill->ncon = n; fill->scan_ok = s_progs.scan_ok; fill->key = s_progs.fp_key; fill->tex = tex;
                        if (n) memcpy(fill->con, g.fp_constants.values, n * 16u);
                        fill->valid = 1;
                    }
                    s_fpc_miss++;
                }
            }
            if (s_wp_on > 0) { const unsigned long long _d = wp_tsc() - _a; s_wp_cyc[6] += _d; s_wp_nscan++;
                if (s_wp_on > 1) { static const u8* pfp2; if (s_progs.fp == pfp2) { s_wp_cyc[62] += _d; s_wp_n2[32]++; } pfp2 = s_progs.fp; if (hit) s_wp_n2[33]++; } }
            s_progs.scanned = 1;
            if (tex != ~0u) { s_progs.tex = tex; s_progs.tex_valid = 1; }
        }
    }
    if (out_vp) *out_vp = s_progs.vp;
    if (out_vp_instrs) *out_vp_instrs = s_progs.vp_instrs;
    if (out_fp) *out_fp = s_progs.fp;
    if (out_fp_size) *out_fp_size = s_progs.fp_size;
    return s_progs.ret;
}

/* fp-struct of the draw's current fragment program (0 = none): ties the "reads zeta" / "own target" log
 * lines to the program that reads them (tools_sweep/shader_coverage.py joins it with the pipeline log). */
static u64 eng_cur_fp_struct(void)
{
    const u8* vp = NULL; const u8* fp = NULL; u32 vn = 0, fs = 0;
    if (!eng_guest_programs(&vp, &vn, &fp, &fs) || !fp) return 0;
    return eng_fp_struct_id(fp, fs);
}

/* The texture units the draw's fragment program samples. A unit can stay
 * switched on with a stale binding from an earlier pass (a shadow pass keeps
 * its map bound); binding it anyway cost a cache check and a hash or an
 * upload, a depth target's resolve on an MSAA run, and a pipeline variant
 * for its type. Units not sampled are left unbound and out of the program's
 * texture masks. The built-in program, and RSX_BIND_UNREAD=1 or
 * RSX_ENGINE_SLOW=1, take every unit that is on, as before. */
int g_rsx_bind_unread = -1;   /* DOD3_AB=texreads switches it in a run */
static u32 eng_fp_reads(void)
{
    if (g_rsx_bind_unread < 0) g_rsx_bind_unread = getenv("RSX_BIND_UNREAD") ? 1 : 0;
    if (g_rsx_bind_unread || eng_slow()) return 0xFFFFu;
    if (s_draw_open && s_progs.tex_valid) return s_progs.tex;
    const u8* vp = NULL; const u8* fp = NULL; u32 vn = 0, fs = 0;
    if (s_draw_open) {   /* the lookup's scan may have answered */
        eng_guest_programs(&vp, &vn, &fp, &fs);
        if (s_progs.tex_valid) return s_progs.tex;
    }
    const u32 m = eng_guest_programs(&vp, &vn, &fp, &fs) && fp ? rsx_fp_texture_mask(fp, fs) : 0xFFFFu;
    if (s_draw_open) { s_progs.tex = m; s_progs.tex_valid = 1; }
    return m;
}

static int eng_guest_programs_uncached(const u8** out_vp, u32* out_vp_instrs,
                                       const u8** out_fp, u32* out_fp_size)
{
    const u8* vp_uc = NULL;
    u32 vp_instrs = 0;
    const u32 start = rsx_dsp_vp_start(RV);
    if (start < RSX_DSP_VP_INSTR) {
        vp_uc = (const u8*)(RV->vp + start * 4);
        if (eng_vp_info_current(start)) {
            vp_instrs = s_vpi.instrs;
        } else {
            vp_instrs = rsx_vp_program_size_instrs(
                vp_uc, (RSX_DSP_VP_INSTR - start) * 16u);
            s_vpi.valid = 1;
            s_vpi.gen = RV->vp_gen;
            s_vpi.start = start;
            s_vpi.instrs = vp_instrs;
        }
    }

    /* A zero SET_SHADER_PROGRAM is "none", not "the program at offset 0": the
     * register carries the location in its low two bits, so a real program can
     * never read back as zero. */
    const u8* fp_uc = NULL;
    u32 fp_size = 0;
    if (rsx_dsp_reg(RV, 0x08E4)) {
        u32 fp_loc = 0;
        const u32 fp_off = rsx_dsp_fragment_program(RV, &fp_loc);
        fp_uc = eng_guest_ptr(NULL, fp_loc, fp_off, 16);
        s_fpc_hit = s_fpc_fill = NULL;
        EngFpc* ce = NULL;
        if (fp_uc && s_draw_open && !eng_slow()) {
            if (s_fpc_on < 0) {   /* 1.1: ON by default (=0 off). opt102: RSX_FPSCAN_MEMO=1 (=2: also scan and compare on every hit) or the older RSX_FPCACHE=1 */
                const char* e = getenv("RSX_FPSCAN_MEMO"); const char* e0 = getenv("RSX_FPCACHE");
                s_fpc_on = (e && e[0]) ? (e[0] == '2' ? 2 : e[0] == '0' ? 0 : 1) : (e0 && e0[0] == '0') ? 0 : 1; }   /* 1.1: default ON */
            if (s_fpc_on) {
                if (!s_fpc) s_fpc = (EngFpc*)calloc(ENG_FPC_N, sizeof(EngFpc));
                if (s_fpc) {
                    ce = &s_fpc[((fp_loc * 0x9E3779B1u) ^ (fp_off >> 4) ^ (fp_off * 0x85EBCA6Bu >> 7)) & (ENG_FPC_N - 1u)];
                    if (ce->valid && ce->loc == fp_loc && ce->off == fp_off) {
                        const u8* p = eng_guest_ptr(NULL, fp_loc, fp_off, ce->size);
                        if (p && !memcmp(p, ce->bytes, ce->size)) {
                            fp_uc = p; fp_size = ce->size; s_fpc_hit = ce;
                        }
                    }
                }
            }
        }
        if (!s_fpc_hit) {
            fp_size = fp_uc ? rsx_fp_program_size(fp_uc, ENG_FP_MAX_BYTES) : 0;
            if (fp_size) fp_uc = eng_guest_ptr(NULL, fp_loc, fp_off, fp_size);
            if (!fp_uc) fp_size = 0;
            if (ce && fp_size && fp_size <= ENG_FPC_BYTES) {
                ce->valid = 0; ce->loc = fp_loc; ce->off = fp_off; ce->size = fp_size;
                memcpy(ce->bytes, fp_uc, fp_size);
                s_fpc_fill = ce;
            }
        }
    }

    if (out_vp) *out_vp = vp_uc;
    if (out_vp_instrs) *out_vp_instrs = vp_instrs;
    if (out_fp) *out_fp = fp_uc;
    if (out_fp_size) *out_fp_size = fp_size;
    return vp_instrs && fp_size;
}

/* The layout the active vertex program actually reads. An uncertain analysis
 * falls back to all sixteen registers (rsx_live_draw.c:3625-3643). */
static void eng_vertex_layout_impl(rsx_vertex_layout_plan* layout);
static void eng_vertex_layout(rsx_vertex_layout_plan* layout)
{
    RSX_B2P_ENTER(B2P_LAYOUT);
    eng_vertex_layout_impl(layout);
    RSX_B2P_LEAVE();
}
static void eng_vertex_layout_impl(rsx_vertex_layout_plan* layout)
{
    u32 mask = 0xFFFFu;
    const u8* uc = NULL;
    u32 instrs = 0;
    if (eng_guest_programs(&uc, &instrs, NULL, NULL)) {
        if (s_vpi.valid && s_vpi.mask_valid) {
            mask = s_vpi.input_mask;
        } else {
            rsx_vp_input_analysis analysis = { 0xFFFFu, 0 };
            if (rsx_vp_analyze_inputs(uc, instrs * 16u, &analysis) == (int)instrs &&
                analysis.exact && analysis.input_mask)
                mask = analysis.input_mask;
            if (s_vpi.valid) { s_vpi.input_mask = mask; s_vpi.mask_valid = 1; }
        }
    }
    rsx_vertex_layout_plan_init(layout, mask);
}

/* What a draw runs before a title has loaded programs of its own.
 *
 * The reference engine has no fallback and drops such a draw, because a title
 * always has both programs resident by the time it draws anything. A general
 * toolkit does not get to assume that: the host harness's fixed-function modes
 * are exactly this case, and so is the first frame of a title that clears and
 * flips before its first SET_TRANSFORM_PROGRAM. It is written in the same HLSL
 * shape the decompilers emit -- ATTRn inputs, the COLOR0/TEXCOORD varyings, the
 * VPConst block and the viewport epilogue -- so it goes through the one
 * pipeline_create the backend already implements rather than needing a second
 * entry point for a built-in shader.
 *
 * Vertex constant rows 0..3 are the transform when the guest programmed them
 * and identity when it did not: an all-zero matrix would collapse every vertex
 * onto the origin. */
static const char* const kEngFixedVS =
"struct VSInput {\n"
"    float4 a0:ATTR0;  float4 a1:ATTR1;  float4 a2:ATTR2;  float4 a3:ATTR3;\n"
"    float4 a4:ATTR4;  float4 a5:ATTR5;  float4 a6:ATTR6;  float4 a7:ATTR7;\n"
"    float4 a8:ATTR8;  float4 a9:ATTR9;  float4 a10:ATTR10; float4 a11:ATTR11;\n"
"    float4 a12:ATTR12; float4 a13:ATTR13; float4 a14:ATTR14; float4 a15:ATTR15;\n"
"};\n"
"struct VSOutput {\n"
"    float4 pos:SV_Position; float4 col0:COLOR0; float4 col1:COLOR1;\n"
"    float4 fog:FOG;\n"
"    float4 t0:TEXCOORD0; float4 t1:TEXCOORD1; float4 t2:TEXCOORD2; float4 t3:TEXCOORD3;\n"
"    float4 t4:TEXCOORD4; float4 t5:TEXCOORD5; float4 t6:TEXCOORD6; float4 t7:TEXCOORD7;\n"
"    float4 t8:TEXCOORD8; float4 t9:TEXCOORD9;\n"
"};\n"
"cbuffer VPConst : register(b0) {\n"
"    float4 vp_c[512];\n"
"    float4 vp_posscale;\n"
"    float4 vp_posoffset;\n"
"};\n"
"VSOutput main(VSInput input) {\n"
"    float4 _p = float4(dot(vp_c[0], input.a0), dot(vp_c[1], input.a0),\n"
"                       dot(vp_c[2], input.a0), dot(vp_c[3], input.a0));\n"
"    float _m = dot(abs(vp_c[0]), 1.0) + dot(abs(vp_c[1]), 1.0)\n"
"             + dot(abs(vp_c[2]), 1.0) + dot(abs(vp_c[3]), 1.0);\n"
"    if (_m == 0.0) _p = input.a0;\n"
"    VSOutput Out;\n"
"    Out.pos = float4(_p.xyz * vp_posscale.xyz + _p.w * vp_posoffset.xyz, _p.w);\n"
"    Out.col0 = input.a3; Out.col1 = float4(0,0,0,1); Out.fog = (float4)0;\n"
"    Out.t0 = input.a8; Out.t1 = (float4)0; Out.t2 = (float4)0; Out.t3 = (float4)0;\n"
"    Out.t4 = (float4)0; Out.t5 = (float4)0; Out.t6 = (float4)0; Out.t7 = (float4)0;\n"
"    Out.t8 = (float4)0; Out.t9 = (float4)0;\n"
"    return Out;\n"
"}\n";

static const char* const kEngFixedPS =
"struct PSInput {\n"
"    float4 position : SV_POSITION; float4 col0 : COLOR0; float4 col1 : COLOR1;\n"
"    float4 fog : FOG;\n"
"    float4 tc0:TEXCOORD0; float4 tc1:TEXCOORD1; float4 tc2:TEXCOORD2; float4 tc3:TEXCOORD3;\n"
"    float4 tc4:TEXCOORD4; float4 tc5:TEXCOORD5; float4 tc6:TEXCOORD6; float4 tc7:TEXCOORD7;\n"
"    float4 tc8:TEXCOORD8; float4 tc9:TEXCOORD9;\n"
"};\n"
"float4 main(PSInput input) : SV_TARGET { return input.col0; }\n";

/* The pipeline for this draw, built once per distinct key. A negative result
 * is cached too (handle 0), so a program pair that will not translate is not
 * retried on every draw of every frame. */
/* ---- asynchronous pipeline builds ------------------------------------------
 *
 * A pipeline the cache has not seen means translating its programs (HLSL ->
 * SPIR-V -> MSL) and compiling them, and the first time on a machine that is
 * 200-450 ms a pipeline: Drakengard 3 froze for 0.7 s entering an area with a
 * handful of new effects. The walker now hands the build to a worker and
 * waits for it out of a budget of RSX_ASYNC_WAIT_MS a frame (default 2) -- a
 * cached shader is ready well inside that, so nothing changes for it -- and
 * a slow one is skipped until it is ready: the effect appears a few frames
 * late, the first time, instead of the game stopping. RSX_ASYNC_SHADERS=0 builds on the walker as
 * before. Metal and D3D12: both backends' pipeline_create are thread-safe. */
#if defined(__APPLE__) || defined(_WIN32)
#if defined(_WIN32)
/* The pthread names the worker was written with, over Win32: an SRW lock,
 * condition variables and CreateThread. The timed wait takes a deadline the
 * caller computed with timespec_get, so it is converted to a span here. */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
typedef SRWLOCK pj_mutex_t;
typedef CONDITION_VARIABLE pj_cond_t;
typedef HANDLE pj_thread_t;
#define PJ_MUTEX_INIT SRWLOCK_INIT
#define PJ_COND_INIT  CONDITION_VARIABLE_INIT
static void pj_lock(pj_mutex_t* m)   { AcquireSRWLockExclusive(m); }
static void pj_unlock(pj_mutex_t* m) { ReleaseSRWLockExclusive(m); }
static void pj_wait(pj_cond_t* c, pj_mutex_t* m) { SleepConditionVariableSRW(c, m, INFINITE, 0); }
static int  pj_timedwait(pj_cond_t* c, pj_mutex_t* m, const struct timespec* dl)
{
    struct timespec now; timespec_get(&now, TIME_UTC);
    long long ms = (long long)(dl->tv_sec - now.tv_sec) * 1000LL + (dl->tv_nsec - now.tv_nsec) / 1000000LL;
    if (ms <= 0) return 1;
    return SleepConditionVariableSRW(c, m, (DWORD)ms, 0) ? 0 : 1;
}
static void pj_signal(pj_cond_t* c)    { WakeConditionVariable(c); }
static void pj_broadcast(pj_cond_t* c) { WakeAllConditionVariable(c); }
#else
#include <pthread.h>
typedef pthread_mutex_t pj_mutex_t;
typedef pthread_cond_t  pj_cond_t;
typedef pthread_t       pj_thread_t;
#define PJ_MUTEX_INIT PTHREAD_MUTEX_INITIALIZER
#define PJ_COND_INIT  PTHREAD_COND_INITIALIZER
static void pj_lock(pj_mutex_t* m)   { pthread_mutex_lock(m); }
static void pj_unlock(pj_mutex_t* m) { pthread_mutex_unlock(m); }
static void pj_wait(pj_cond_t* c, pj_mutex_t* m) { pthread_cond_wait(c, m); }
static int  pj_timedwait(pj_cond_t* c, pj_mutex_t* m, const struct timespec* dl) { return pthread_cond_timedwait(c, m, dl); }
static void pj_signal(pj_cond_t* c)    { pthread_cond_signal(c); }
static void pj_broadcast(pj_cond_t* c) { pthread_cond_broadcast(c); }
#endif
typedef struct eng_pipe_job {
    char* vs; char* ps;
    rsx_be_render_state rs;
    rsx_vertex_layout_plan layout;
    u32 stride; rsx_be_format rt_fmt; u32 rt_count;
    u32 handle; int done;
    u32 frame;
    u64 key;                 /* the engine's pipeline key (for the manifest) */
    u8 borrowed;             /* vs/ps point into the manifest buffer: not freed */
    u8 record;               /* append to the manifest once built */
    u8 started, bumped;      /* a worker took it / it was moved to the queue head */
    struct eng_pipe_job* next;
} eng_pipe_job;
static pj_mutex_t s_pj_mu = PJ_MUTEX_INIT;
static pj_cond_t  s_pj_work = PJ_COND_INIT, s_pj_done = PJ_COND_INIT;
static eng_pipe_job *s_pj_head, *s_pj_tail;
static int s_pj_started, s_pj_quit;
#define PJ_MAX_THREADS 8
static pj_thread_t s_pj_threads[PJ_MAX_THREADS];
static int s_pj_nthreads;
/* RSX_ASYNC_THREADS=<n>: build workers (default: the cores less two, 2..4).
 * Both backends compile outside their locks, so builds overlap. */
static int eng_pipe_nthreads(void)
{
    const char* e = getenv("RSX_ASYNC_THREADS");
    int n = e ? atoi(e) : 0;
    if (n <= 0) {
#if defined(_WIN32)
        SYSTEM_INFO si; GetSystemInfo(&si); n = (int)si.dwNumberOfProcessors - 2;
#else
        n = (int)sysconf(_SC_NPROCESSORS_ONLN) - 2;
#endif
    }
    if (n < 2) n = 2;
    if (!e && n > 4) n = 4;     /* default 2..4; RSX_ASYNC_THREADS goes up to PJ_MAX_THREADS */
    if (n > PJ_MAX_THREADS) n = PJ_MAX_THREADS;
    return n;
}

static int eng_async_on(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("RSX_ASYNC_SHADERS"); on = !(e && e[0] == '0'); }
    return on;
}
static void eng_man_append(const eng_pipe_job* j);   /* the pipeline manifest, below */
/* 31/10 (pre52) pre-warm statistics: the manifest's jobs, built in the background at start-up. */
static volatile long s_pw_total, s_pw_done, s_pw_fail, s_pw_bumped;     /* queued / finished / failed / a draw had to wait for it */
static volatile long long s_pw_cpu_us;                                   /* summed build time of the finished ones */
static struct timespec s_pw_t0;                                          /* when the queue was handed to the workers */
static int s_pw_cap = 2;                                                 /* RSX_PSO_PREWARM_THREADS: workers a not-yet-wanted pre-warm job may occupy at once */
static int s_pw_running;                                                 /* pre-warm jobs being built now (under s_pj_mu) */
static int s_pw_prio_lowest = 1;                                         /* RSX_PSO_PREWARM_PRIO=0: below-normal like every build */
#if defined(_WIN32)
static DWORD WINAPI eng_pipe_worker(LPVOID arg)
#else
static void* eng_pipe_worker(void* arg)
#endif
{
    (void)arg;
#if defined(__APPLE__)
    pthread_setname_np("rsx pipeline build");
#endif
#if defined(_WIN32)
    /* Builds are background work: below the game's own threads, so a burst of
     * new effects does not take cores from the SPU/PPU/RSX threads. */
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
    for (;;) {
        pj_lock(&s_pj_mu);
        /* The first job that may run now: any build a draw asked for, or a pre-warm one while fewer than s_pw_cap of
         * them are running (the rest of the workers stay free for what the game needs at once; 31/10 pre52: four
         * threads of FXC -O3 took the game to 5 fps for a minute). At shutdown anything goes (dropped, not built). */
        eng_pipe_job *j = NULL, *jprev = NULL;
        for (;;) {
            jprev = NULL;
            for (eng_pipe_job* c = s_pj_head; c; jprev = c, c = c->next)
                if (s_pj_quit || !c->borrowed || c->bumped || s_pw_running < s_pw_cap) { j = c; break; }
            if (j || (s_pj_quit && !s_pj_head)) break;
            pj_wait(&s_pj_work, &s_pj_mu);
        }
        if (!j) { pj_unlock(&s_pj_mu); return 0; }
        if (jprev) jprev->next = j->next; else s_pj_head = j->next;
        if (s_pj_tail == j) s_pj_tail = jprev;
        j->next = NULL;
        const int pw_job = j->borrowed && !j->bumped;
        if (pw_job) s_pw_running++;
        j->started = 1;
        const int quit = s_pj_quit;
        pj_unlock(&s_pj_mu);
#if defined(_WIN32)
        /* A pre-warm job nobody asked for yet runs at the lowest priority; one a draw is waiting for, like any build. */
        { const int pw = pw_job && s_pw_prio_lowest;
          SetThreadPriority(GetCurrentThread(), pw ? THREAD_PRIORITY_LOWEST : THREAD_PRIORITY_BELOW_NORMAL); }
#endif
        struct timespec t0, t1;
        timespec_get(&t0, TIME_UTC);
        /* At shutdown the queued precompiles are dropped, not built. */
        const u32 h = quit ? 0u : g.be->pipeline_create(g.be->user, j->vs, j->ps, &j->rs,
                                            &j->layout, j->stride, j->rt_fmt, j->rt_count);
        timespec_get(&t1, TIME_UTC);
        if (h && j->record) eng_man_append(j);
        const double ms = (double)(t1.tv_sec - t0.tv_sec) * 1e3 + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
        if (j->borrowed && !quit) {
            __atomic_fetch_add(&s_pw_cpu_us, (long long)(ms * 1000.0), __ATOMIC_RELAXED);
            if (!h) __atomic_fetch_add(&s_pw_fail, 1, __ATOMIC_RELAXED);
            const long dn = __atomic_add_fetch(&s_pw_done, 1, __ATOMIC_RELAXED);
            if (dn == s_pw_total) {
                struct timespec tn; timespec_get(&tn, TIME_UTC);
                fprintf(stderr, "[pso-prewarm] done: %ld pipelines built in %.1f s wall (%.1f s of build time, %ld failed, %ld a draw waited for)\n",
                        dn, (double)(tn.tv_sec - s_pw_t0.tv_sec) + (double)(tn.tv_nsec - s_pw_t0.tv_nsec) / 1e9,
                        (double)s_pw_cpu_us / 1e6, s_pw_fail, s_pw_bumped);
            }
        }
        if (ms >= 20.0)
            fprintf(stderr, "[pipe-slow] async build took %.1f ms (asked for in frame %u, ready in frame %u)%s\n",
                    ms, j->frame, g.frames, h ? "" : " -- FAILED");
        /* 27/10: a borrowed job keeps its pointers (into the manifest buffer, which stays): eng_pipe_poll reads them */
        if (!j->borrowed) { free(j->vs); free(j->ps); j->vs = j->ps = NULL; }
        pj_lock(&s_pj_mu);
        j->handle = h;
        __atomic_store_n(&j->done, 1, __ATOMIC_RELEASE);
        if (pw_job) { s_pw_running--; pj_broadcast(&s_pj_work); }   /* a pre-warm slot is free again */
        pj_broadcast(&s_pj_done);
        pj_unlock(&s_pj_mu);
    }
}
static void eng_pipe_start_workers_locked(void)
{
    if (s_pj_started) return;
    s_pj_started = 1; s_pj_quit = 0;
    const int want = eng_pipe_nthreads();
    s_pj_nthreads = 0;
    for (int t = 0; t < want; t++) {
#if defined(_WIN32)
        s_pj_threads[t] = CreateThread(NULL, 64u << 20, eng_pipe_worker, NULL, 0, NULL);
        if (!s_pj_threads[t]) break;
#else
        pthread_attr_t at; pthread_attr_init(&at);
        pthread_attr_setstacksize(&at, 64u << 20);   /* glslang and spirv-opt recurse deeply */
        const int rc = pthread_create(&s_pj_threads[t], &at, eng_pipe_worker, NULL);
        pthread_attr_destroy(&at);
        if (rc != 0) break;
#endif
        s_pj_nthreads++;
    }
    if (!s_pj_nthreads) s_pj_started = -1;
}
/* The wait is a budget for the frame, not for each build: 38 new
 * effects at once had waited 8 ms apiece, 300 ms in one frame.
 * Called with s_pj_mu held; the wait releases it. */
static double s_pj_waited_ms;     /* waited on the walker since the last present */
static void eng_pipe_wait_locked(eng_pipe_job* j)
{
    static long wait_ms = -1;
    static u32 budget_frame;
    static long left_us;
    if (wait_ms < 0) { const char* e = getenv("RSX_ASYNC_WAIT_MS"); wait_ms = e ? atol(e) : 2; }
    if (budget_frame != g.frames) { budget_frame = g.frames; left_us = wait_ms * 1000L; }
    if (left_us > 0 && !j->done) {
        struct timespec t0, dl; timespec_get(&t0, TIME_UTC);
        dl = t0;
        dl.tv_nsec += (left_us % 1000000L) * 1000L; dl.tv_sec += left_us / 1000000L + dl.tv_nsec / 1000000000L;
        dl.tv_nsec %= 1000000000L;
        while (!j->done) if (pj_timedwait(&s_pj_done, &s_pj_mu, &dl) != 0) break;
        struct timespec t1; timespec_get(&t1, TIME_UTC);
        const long used = (long)((t1.tv_sec - t0.tv_sec) * 1000000L + (t1.tv_nsec - t0.tv_nsec) / 1000L);
        left_us -= used;
        s_pj_waited_ms += (double)used / 1000.0;
    }
}
/* Queue a build and wait for it a little; the job is the entry's until done.
 * A build the walker needs now goes ahead of any precompile backlog. */
static eng_pipe_job* eng_pipe_submit(u64 key, const char* vs, const char* ps, const rsx_be_render_state* rs,
                                     const rsx_vertex_layout_plan* layout, u32 stride,
                                     rsx_be_format rt_fmt, u32 rt_count)
{
    eng_pipe_job* j = (eng_pipe_job*)calloc(1, sizeof *j);
    if (!j) return NULL;
    j->vs = strdup(vs); j->ps = strdup(ps);
    if (!j->vs || !j->ps) { free(j->vs); free(j->ps); free(j); return NULL; }
    j->rs = *rs; j->layout = *layout; j->stride = stride; j->rt_fmt = rt_fmt; j->rt_count = rt_count;
    j->frame = g.frames;
    j->key = key; j->record = 1; j->bumped = 1;
    BEACON_WAIT("pipe_submit");
    pj_lock(&s_pj_mu);
    eng_pipe_start_workers_locked();
    if (s_pj_started < 0) { pj_unlock(&s_pj_mu); BEACON_POP(); free(j->vs); free(j->ps); free(j); return NULL; }
    j->next = s_pj_head; s_pj_head = j; if (!s_pj_tail) s_pj_tail = j;
    pj_signal(&s_pj_work);
    eng_pipe_wait_locked(j);
    pj_unlock(&s_pj_mu);
    BEACON_POP();
    return j;
}
static void eng_pipe_stop(void)
{
    pj_lock(&s_pj_mu);
    const int started = s_pj_started > 0;
    s_pj_quit = 1;
    pj_broadcast(&s_pj_work);
    pj_unlock(&s_pj_mu);
    for (int t = 0; started && t < s_pj_nthreads; t++) {
#if defined(_WIN32)
        WaitForSingleObject(s_pj_threads[t], INFINITE); CloseHandle(s_pj_threads[t]); s_pj_threads[t] = NULL;
#else
        pthread_join(s_pj_threads[t], NULL);
#endif
    }
    s_pj_nthreads = 0;
    s_pj_started = 0;
}
/* ---- pipeline manifest: precompile at start-up ------------------------------
 *
 * The first draw with a pipeline the run has not built yet queues a build and
 * is skipped until it is ready (the object is missing for a few frames): the
 * key and the ammo crate flickered at the start of every level, because each
 * level's pipelines are first met there. The state objects themselves come
 * cheap from the disk caches (dxbc\ and the pipeline library), but only once
 * the draw asks for them.
 *
 * Every pipeline built is appended to <cache dir>\pipes.bin: the engine's
 * key, both HLSL texts, the render state and the vertex layout, in the order
 * first met. At the next start the file is read once and the first
 * RSX_PSO_PRECOMPILE_MAX (default 3000) records are queued to the build
 * workers before the first draw asks for anything -- behind any pipeline a draw
 * needs at once, which goes to the head of the queue (and may be waited for,
 * see eng_pipe_poll_wait). A draw then finds its pipeline in the table, ready.
 *
 * What makes a record valid: the header names the exe (size and time -- the
 * decompilers and the key are code), the size of the render-state and layout
 * structs (a field added to either changes it) and a version. Any mismatch:
 * the file is ignored and rewritten. RSX_PSO_MANIFEST=0 turns it all off
 * (and RSX_PSO_CACHE=0), RSX_PSO_PRECOMPILE=0 keeps recording but queues
 * nothing, RSX_PSO_CACHE_DIR moves the file. */
#define MAN_MAGIC   0x4D504950u   /* 'PIPM' */
#define MAN_VERSION 2u    /* 2 (31/10 pre52): exe_id is the codegen id; was the exe's size/time */
#define MAN_KEYREV  1u    /* bump by hand when the pipeline key, the render-state decode or the record layout changes meaning */
typedef struct { u32 magic, version, sz_rec, sz_rs, sz_layout, pad; u64 exe_id; } man_hdr;
typedef struct {
    u64 key; u32 vs_len, ps_len, stride, rt_fmt, rt_count, pad;   /* lengths include the NUL */
    rsx_be_render_state rs; rsx_vertex_layout_plan layout;
} man_rec;
#ifdef RSX_CODEGEN_ID
#define RSX_CG_STR2(x) #x
#define RSX_CG_STR(x) RSX_CG_STR2(x)
/* Readable in the exe (tools_sweepun2_update_new.ps1 looks for it): the id this build's cache must carry. */
const char g_rsx_codegen_stamp[] = "RSX-CG:" RSX_CG_STR(RSX_CODEGEN_ID) ":K" RSX_CG_STR(MAN_KEYREV) ":";
#else
const char g_rsx_codegen_stamp[] = "RSX-CG:none:";
#endif
static pj_mutex_t s_man_mu = PJ_MUTEX_INIT;
static char s_man_path[MAX_PATH + 16];
static int s_man_on = -1, s_man_valid;       /* usable / the file on disk has our header */
static u64 s_man_exe_id;
static FILE* s_man_fp;
static unsigned long long s_man_bytes;
static u64* s_man_keys; static u32 s_man_nkeys, s_man_cap;   /* keys already in the file (open addressing, 0 = empty) */
static unsigned s_man_appended, s_man_loaded, s_man_queued;

static int eng_man_enabled(void)
{
    if (s_man_on >= 0) return s_man_on;
    s_man_on = 0;
    const char* e = getenv("RSX_PSO_MANIFEST"); if (e && e[0] == '0') return 0;
    e = getenv("RSX_PSO_CACHE"); if (e && e[0] == '0') return 0;
#if defined(_WIN32)
    char exe[MAX_PATH];
    const DWORD n = GetModuleFileNameA(NULL, exe, sizeof exe);
    if (!n || n >= sizeof exe) return 0;
    const char* ov = getenv("RSX_PSO_CACHE_DIR");
    if (ov && *ov) snprintf(s_man_path, sizeof s_man_path, "%s\\pipes.bin", ov);
    else {
        char* sl = strrchr(exe, '\\'); if (!sl) sl = strrchr(exe, '/');
        char dir[MAX_PATH];
        snprintf(dir, sizeof dir, "%.*s", sl ? (int)(sl - exe) : 0, exe);
        snprintf(s_man_path, sizeof s_man_path, "%s\\cache\\pipes.bin", dir);
    }
#ifdef RSX_CODEGEN_ID
    /* 31/10 (pre52): the id is what decides how a pipeline is built (hash of the vertex/fragment decompiler sources,
     * made by CMake, plus MAN_KEYREV), not the exe's size and time: every rebuild used to make the manifest and the
     * pipeline library "another build" and the game started cold (the 31/10 18:00 play). */
    { u64 h = 1469598103934665603ull; const u64 cg = (u64)(RSX_CODEGEN_ID) ^ (u64)rsx_fp_codegen_salt(); const u32 kr = MAN_KEYREV;
      h = eng_fnv1a(&cg, sizeof cg, h); h = eng_fnv1a(&kr, sizeof kr, h);
      s_man_exe_id = h; }
#else
    WIN32_FILE_ATTRIBUTE_DATA d;
    if (!GetFileAttributesExA(exe, GetFileExInfoStandard, &d)) return 0;
    u64 h = 1469598103934665603ull;
    h = eng_fnv1a(&d.nFileSizeLow, 4, h); h = eng_fnv1a(&d.nFileSizeHigh, 4, h);
    h = eng_fnv1a(&d.ftLastWriteTime, sizeof d.ftLastWriteTime, h);
    s_man_exe_id = h;
#endif
    s_man_on = 1;
#endif
    return s_man_on;
}
static int eng_man_known(u64 key)
{
    if (!s_man_cap) return 0;
    for (u32 i = (u32)(key ^ (key >> 31)) & (s_man_cap - 1u);; i = (i + 1u) & (s_man_cap - 1u)) {
        if (!s_man_keys[i]) return 0;
        if (s_man_keys[i] == key) return 1;
    }
}
static void eng_man_note(u64 key)
{
    if (!key || eng_man_known(key)) return;
    if ((s_man_nkeys + 1u) * 2u > s_man_cap) {          /* grow, rehash */
        const u32 oc = s_man_cap; u64* ok = s_man_keys;
        s_man_cap = oc ? oc * 2u : 4096u;
        s_man_keys = (u64*)calloc(s_man_cap, sizeof(u64));
        if (!s_man_keys) { s_man_keys = ok; s_man_cap = oc; return; }
        for (u32 i = 0; i < oc; i++) if (ok[i]) {
            u32 k = (u32)(ok[i] ^ (ok[i] >> 31)) & (s_man_cap - 1u);
            while (s_man_keys[k]) k = (k + 1u) & (s_man_cap - 1u);
            s_man_keys[k] = ok[i];
        }
        free(ok);
    }
    u32 k = (u32)(key ^ (key >> 31)) & (s_man_cap - 1u);
    while (s_man_keys[k]) k = (k + 1u) & (s_man_cap - 1u);
    s_man_keys[k] = key; s_man_nkeys++;
}
/* A pipeline a worker has built: append it, unless the file has it. */
static void eng_man_append(const eng_pipe_job* j)
{
    if (!j->key || !eng_man_enabled()) return;
    pj_lock(&s_man_mu);
    if (!eng_man_known(j->key) && s_man_bytes < (192ull << 20)) {
        if (!s_man_fp) {
            if (s_man_valid) s_man_fp = fopen(s_man_path, "ab");
            else {
                /* 31/10 (pre52): the file of another format is kept as pipes.bin.prev, not destroyed */
                { char prev[MAX_PATH + 24]; snprintf(prev, sizeof prev, "%s.prev", s_man_path);
                  remove(prev); rename(s_man_path, prev); }
                s_man_fp = fopen(s_man_path, "wb");
                if (s_man_fp) {
                    man_hdr h = { MAN_MAGIC, MAN_VERSION, (u32)sizeof(man_rec), (u32)sizeof(rsx_be_render_state),
                                  (u32)sizeof(rsx_vertex_layout_plan), 0, s_man_exe_id };
                    if (fwrite(&h, sizeof h, 1, s_man_fp) == 1) { s_man_valid = 1; s_man_bytes = sizeof h; }
                    else { fclose(s_man_fp); s_man_fp = NULL; }
                }
            }
        }
        if (s_man_fp) {
            man_rec r; memset(&r, 0, sizeof r);
            r.key = j->key; r.vs_len = (u32)strlen(j->vs) + 1u; r.ps_len = (u32)strlen(j->ps) + 1u;
            r.stride = j->stride; r.rt_fmt = (u32)j->rt_fmt; r.rt_count = j->rt_count;
            r.rs = j->rs; r.layout = j->layout;
            if (fwrite(&r, sizeof r, 1, s_man_fp) == 1 && fwrite(j->vs, 1, r.vs_len, s_man_fp) == r.vs_len &&
                fwrite(j->ps, 1, r.ps_len, s_man_fp) == r.ps_len) {
                fflush(s_man_fp);
                s_man_bytes += sizeof r + r.vs_len + r.ps_len;
                eng_man_note(j->key); s_man_appended++;
            } else { fclose(s_man_fp); s_man_fp = NULL; s_man_bytes = ~0ull; }
        }
    }
    pj_unlock(&s_man_mu);
}
/* Once, before the first pipeline lookup: read the file, queue the builds. */
static void eng_pipe_preload(void)
{
    static int done;
    if (done) return;
    done = 1;
    if (!eng_async_on() || !eng_man_enabled()) return;
    FILE* f = fopen(s_man_path, "rb");
    if (!f) return;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    man_hdr h; char* buf = NULL;
    if (n > (long)sizeof h && fread(&h, sizeof h, 1, f) == 1 && h.magic == MAN_MAGIC && h.version == MAN_VERSION &&
        h.sz_rec == sizeof(man_rec) && h.sz_rs == sizeof(rsx_be_render_state) && h.sz_layout == sizeof(rsx_vertex_layout_plan) &&
        h.exe_id == s_man_exe_id && (buf = (char*)malloc((size_t)(n - (long)sizeof h) + 1u)) != NULL &&
        fread(buf, 1, (size_t)(n - (long)sizeof h), f) == (size_t)(n - (long)sizeof h)) {
        s_man_valid = 1; s_man_bytes = (unsigned long long)n;
    } else {
        free(buf); buf = NULL;
        fprintf(stderr, "[pso-manifest] '%s' is for another build or damaged (file id %016llx v%u, expected %016llx v%u): ignored, will be rewritten (old one kept as .prev)\n",
                s_man_path, (unsigned long long)h.exe_id, h.version, (unsigned long long)s_man_exe_id, MAN_VERSION);
    }
    fclose(f);
    if (!buf) return;
    const size_t len = (size_t)(n - (long)sizeof h);
    long maxn = 12000;
    { const char* e = getenv("RSX_PSO_PRECOMPILE_MAX"); if (e && *e) maxn = atol(e); }
    /* RSX_PSO_PREWARM=0 (or the older RSX_PSO_PRECOMPILE=0): the manifest is read and recorded, nothing is built ahead. */
    int queue = maxn > 0 && !(getenv("RSX_PSO_PRECOMPILE") && getenv("RSX_PSO_PRECOMPILE")[0] == '0');
    { const char* e = getenv("RSX_PSO_PREWARM"); if (e && e[0] == '0') queue = 0; }
    { const char* e = getenv("RSX_PSO_PREWARM_PRIO"); if (e && e[0] == '0') s_pw_prio_lowest = 0; }
    { const char* e = getenv("RSX_PSO_PREWARM_THREADS"); if (e && atoi(e) > 0) s_pw_cap = atoi(e); }
    eng_pipe_job *head = NULL, *tail = NULL;
    size_t off = 0;
    while (off + sizeof(man_rec) <= len) {
        man_rec r; memcpy(&r, buf + off, sizeof r);
        const size_t need = sizeof r + (size_t)r.vs_len + (size_t)r.ps_len;
        if (!r.vs_len || !r.ps_len || need > len - off) break;      /* a record cut short by a kill */
        char* vs = buf + off + sizeof r; char* ps = vs + r.vs_len;
        off += need;
        if (vs[r.vs_len - 1] != 0 || ps[r.ps_len - 1] != 0) break;
        s_man_loaded++;
        eng_man_note(r.key);
        if (!queue || (long)s_man_queued >= maxn || g.n_pipelines + 64u >= ENG_MAX_PIPELINES || eng_pipe_find(r.key)) continue;
        eng_pipe_job* j = (eng_pipe_job*)calloc(1, sizeof *j);
        if (!j) break;
        j->vs = vs; j->ps = ps; j->borrowed = 1; j->key = r.key;
        j->rs = r.rs; j->layout = r.layout; j->stride = r.stride; j->rt_fmt = (rsx_be_format)r.rt_fmt; j->rt_count = r.rt_count;
        j->frame = g.frames;
        eng_pipeline* p = &g.pipelines[g.n_pipelines];
        memset(p, 0, sizeof *p);
        p->key = r.key; p->pending = 1; p->job = j;
        { u64 vh = 1469598103934665603ull, ph = 1469598103934665603ull;
          for (const char* c = vs; *c; c++) vh = (vh ^ (unsigned char)*c) * 1099511628211ull;
          for (const char* c = ps; *c; c++) ph = (ph ^ (unsigned char)*c) * 1099511628211ull;
          p->vp_hlsl = vh; p->fp_hlsl = ph; }
        eng_pipe_index_add(g.n_pipelines);
        g.n_pipelines++;
        s_man_queued++;
        if (tail) tail->next = j; else head = j;
        tail = j;
    }
    /* The buffer stays: the queued jobs read their texts from it. */
    if (head) {
        s_pw_total = (long)s_man_queued; timespec_get(&s_pw_t0, TIME_UTC);
        pj_lock(&s_pj_mu);
        eng_pipe_start_workers_locked();
        if (s_pj_started > 0) {
            if (s_pj_tail) s_pj_tail->next = head; else s_pj_head = head;
            s_pj_tail = tail;
            pj_broadcast(&s_pj_work);
        }
        pj_unlock(&s_pj_mu);
    }
    fprintf(stderr, "[pso-manifest] '%s': %u records, %u queued for pre-warm (%.1f MB), id %016llx (%s)\n",
            s_man_path, s_man_loaded, s_man_queued, (double)n / 1048576.0, (unsigned long long)s_man_exe_id, g_rsx_codegen_stamp);
}
/* A job a draw needs before a worker reached it: to the head of the queue. */
static void eng_pipe_bump(eng_pipe_job* j)
{
    if (j->bumped) return;
    pj_lock(&s_pj_mu);
    j->bumped = 1;
    if (j->borrowed && !j->started) __atomic_fetch_add(&s_pw_bumped, 1, __ATOMIC_RELAXED);
    if (!j->started && s_pj_head != j) {
        eng_pipe_job* prev = s_pj_head;
        while (prev && prev->next != j) prev = prev->next;
        if (prev) {
            prev->next = j->next;
            if (s_pj_tail == j) s_pj_tail = prev;
            j->next = s_pj_head; s_pj_head = j;
        }
        pj_broadcast(&s_pj_work);   /* a worker idle for the pre-warm cap may take it now */
    }
    pj_unlock(&s_pj_mu);
}
/* An entry whose build is in flight: its handle once done (and the job
 * freed), else 0. */
static u32 eng_pipe_poll(eng_pipeline* p)
{
    if (!p->pending) return p->handle;
    if (!__atomic_load_n(&p->job->done, __ATOMIC_ACQUIRE)) return 0;
    p->handle = p->job->handle;
    p->pending = 0;
    /* A precompile that failed is not the draw's answer: forget the entry so
     * the next draw builds the pipeline the usual way. */
    if (!p->handle && p->job->borrowed) p->key = 0;
    /* 27/10: an entry from the manifest was memset 0 in eng_pipe_preload: set what the creation path sets
     * (wpos, and the RT scene's class of the vertex program, from the manifest's own VS/FS text) or RT_AO is
     * blind to every static draw of a second run. Every manifest entry is a non-fixed, decompiled pipeline. */
    if (p->handle && p->job->borrowed && p->job->ps && p->job->vs) {
        p->wpos = (u8)(strstr(p->job->ps, "input.position") != NULL);
        p->rt_static = g_rsx_rt_on ? (u8)rsx_rt_register_vs(p->vp_hlsl, p->job->vs) : 0;
    }
    free(p->job); p->job = NULL;
    return p->handle;
}
/* The same for a draw that needs the pipeline now: if it is still being
 * built, wait for it out of the frame's budget (RSX_ASYNC_WAIT_MS) before
 * the draw is skipped. */
static unsigned s_pj_skips, s_pj_skips_total;      /* draws skipped for a pending build, this frame / ever */
static u32 eng_pipe_poll_wait(eng_pipeline* p)
{
    if (!p->pending) return p->handle;
    if (!__atomic_load_n(&p->job->done, __ATOMIC_ACQUIRE)) {
        eng_pipe_bump(p->job);
        pj_lock(&s_pj_mu);
        eng_pipe_wait_locked(p->job);
        pj_unlock(&s_pj_mu);
        if (!__atomic_load_n(&p->job->done, __ATOMIC_ACQUIRE)) { s_pj_skips++; s_pj_skips_total++; return 0; }
    }
    return eng_pipe_poll(p);
}
/* At every present: RSX_PSO_LOG=1 names the frames that lost draws. */
static void eng_pipe_frame_end(void)
{
    if (s_pj_skips) {
        static int lg = -1; if (lg < 0) lg = getenv("RSX_PSO_LOG") ? 1 : 0;
        if (lg) fprintf(stderr, "[pso-skip] frame %u: %u draws skipped, pipeline still building (waited %.1f ms; %u skipped in all)\n",
                        g.frames, s_pj_skips, s_pj_waited_ms, s_pj_skips_total);
    }
    s_pj_skips = 0; s_pj_waited_ms = 0.0;
}
#else
static int eng_async_on(void) { return 0; }
static void eng_pipe_stop(void) {}
static u32 eng_pipe_poll(eng_pipeline* p) { return p->handle; }
static u32 eng_pipe_poll_wait(eng_pipeline* p) { return p->handle; }
static void eng_pipe_preload(void) {}
static void eng_pipe_frame_end(void) {}
#endif

/* 01/11 (bloom90): the size a texel-addressed (UNNORM) unit must be normalised by. The title samples its bloom
 * chain (160x96 -> 80x48 targets) through descriptors declared 160x88 / 80x44, but the resource bound is the whole
 * target; dividing the texel coordinates by the DECLARED height then reads row y*96/88 of a 96-row target, so every
 * level sat lower than its source and the bloom was drawn shifted upward (doubled halos). On the RSX the coordinates
 * are texel rows of the surface itself, so the divisor is the bound surface's guest size (not the backend's scaled
 * size: the normalised coordinate stays scale-invariant). Same match as sink_bind_textures' scan (exact offset only:
 * a clip-origin alias keeps the declared size). RSX_UNNORM_SURF_DIM=0 restores the declared size. Returns 1 if set. */
static int eng_unnorm_surf_dim(const rsx_dsp_texture* t, u32* w, u32* h)
{
    static int on = -1; if (on < 0) { const char* e = getenv("RSX_UNNORM_SURF_DIM"); on = !(e && e[0] == '0'); }
    if (!on) return 0;
    const u32 dfmt = t->format & RSX_TEX_FMT_BASE_MASK & ~(u32)RSX_TEX_FMT_UNNORM;
    if (dfmt >= 0x86u && dfmt <= 0x93u) return 0;   /* DXT and depth formats are never matched to a colour target */
    u32 i = 0;
    if (s_surf_hash_on) i = eng_surf_first(t->location, t->offset);
    for (; i < g.n_surfaces; i++) {
        if (!g.surfaces[i].handle || g.surfaces[i].location != t->location ||
            (g.surfaces[i].offset != t->offset && !(g.surfaces[i].alias_off && g.surfaces[i].alias_off == t->offset)))
            continue;
        if (g.surfaces[i].offset != t->offset || !g.surfaces[i].w || !g.surfaces[i].h) return 0;
        *w = g.surfaces[i].w; *h = g.surfaces[i].h;
        return 1;
    }
    return 0;
}

/* opt102 RSX_TEXMASK_MEMO: every texture-derived input of the pipeline key that is a pure function of the texture registers and the
 * program's read mask: cube / vtex / shadow (+ funcs) / texel-op (+ ops) masks and the candidate list of the unnorm loop (enabled, UNNORM format,
 * not cube, sized; the bound-surface adjustment that follows depends on the surface set and runs every draw). The memo key is the 176 register
 * words of texture units, size1 and vertex textures plus the FP read mask; equal key = equal masks. Per thread. */
typedef struct { u32 reads, cube, vtex, shadow, texop, ucand; u8 sfun[16]; u32 ops[16]; u32 udim[16][2]; } EngTexMasks;
#define ENG_TM_KEYW 176u
static ENG_TLS struct { int valid; u32 key[ENG_TM_KEYW]; EngTexMasks m; } s_tmk;
static void eng_texmasks_compute(EngTexMasks* m)
{
    memset(m, 0, sizeof *m);
    m->reads = eng_fp_reads();
    m->cube = eng_cube_mask();
    m->vtex = eng_vtex_mask_raw();
    m->shadow = eng_shadow_units(m->sfun);
    m->texop = eng_texel_ops(m->ops);
    static int off = -1; if (off < 0) off = getenv("RSX_NO_UNNORM") ? 1 : 0;
    for (u32 u = 0; !off && u < RSX_DSP_NUM_TEXTURES && u < 16; u++) {
        if (!eng_tex_unit_on(u, m->reads)) continue;
        rsx_dsp_texture t; eng_get_tex(RV, u, &t);
        if (!t.enabled || !(t.format & RSX_TEX_FMT_UNNORM) || ((m->cube >> u) & 1u)) continue;
        if (!t.width || !t.height) continue;
        m->ucand |= 1u << u; m->udim[u][0] = t.width; m->udim[u][1] = t.height;
    }
}
/* Returns the masks for the draw: the memo (hit), or a fresh computation stored in the memo. NULL = memo off, caller computes as before. */
static const EngTexMasks* eng_texmasks_get(void)
{
    if (!eng_tm_mode() || g_rsx_dsp_trace || eng_slow()) return NULL;
    const u32 reads = eng_fp_reads();
    const rsx_dispatch* rv = RV;
    const u32* a = &rv->regs[0x1A00u >> 2]; const u32* b = &rv->regs[0x1840u >> 2]; const u32* c = &rv->regs[0x900u >> 2];
    s_tm_mask_n++;
    if (s_tmk.valid && s_tmk.m.reads == reads && !memcmp(s_tmk.key, a, 128 * 4) && !memcmp(s_tmk.key + 128, b, 16 * 4) && !memcmp(s_tmk.key + 144, c, 32 * 4)) {
        s_tm_mask_hit++;
        if (s_tm_mode == 2) {
            EngTexMasks f; s_tm_raw = 1; eng_texmasks_compute(&f); s_tm_raw = 0;
            if (memcmp(&f, &s_tmk.m, sizeof f) && s_tm_mask_bad++ < 16)
                fprintf(stderr, "[texmask-check] MISMATCH masks frame %u: cube %X/%X vtex %X/%X shadow %X/%X texop %X/%X ucand %X/%X\n", g.frames,
                        s_tmk.m.cube, f.cube, s_tmk.m.vtex, f.vtex, s_tmk.m.shadow, f.shadow, s_tmk.m.texop, f.texop, s_tmk.m.ucand, f.ucand);
        }
        if ((s_tm_mask_n & 0xFFFFu) == 0) eng_tm_report();
        return &s_tmk.m;
    }
    eng_texmasks_compute(&s_tmk.m);
    memcpy(s_tmk.key, a, 128 * 4); memcpy(s_tmk.key + 128, b, 16 * 4); memcpy(s_tmk.key + 144, c, 32 * 4);
    s_tmk.valid = 1;
    if ((s_tm_mask_n & 0xFFFFu) == 0) eng_tm_report();
    return &s_tmk.m;
}

static u32 eng_pipeline_get(const rsx_vertex_layout_plan* layout,
                            const rsx_be_render_state* rs,
                            rsx_be_format rt_fmt, u32 rt_count, int* out_fixed)
{
    eng_pipe_preload();
    WP2_BEGIN(_w17);
    *out_fixed = 1;
    const u8* vp_uc = NULL;
    const u8* fp_uc = NULL;
    u32 vp_instrs = 0, fp_size = 0;
    /* No resident program pair means the built-in one; see kEngFixedVS. */
    const int fixed = !eng_guest_programs(&vp_uc, &vp_instrs, &fp_uc, &fp_size);

    const u32 fp_ctrl  = rsx_dsp_shader_control(RV);
    const EngTexMasks* tmm = eng_texmasks_get();   /* opt102: NULL = RSX_TEXMASK_MEMO off */
    WP2_BEGIN(_w30);
    const u32 cube_mask = tmm ? tmm->cube : eng_cube_mask();
    WP2_END(_w30, 30);
    WP2_BEGIN(_w29);
    const u32 vtex_mask = tmm ? tmm->vtex : eng_vtex_mask();
    WP2_END(_w29, 29);
    const u32 ssr_cfg = (s_ssr_cu >= 0 && s_ssr_du >= 0 && cube_mask) ? (0x10000u | (s_ssr_dbg ? 0x20000u : 0u) | ((u32)s_ssr_cu << 8) | (u32)s_ssr_du) : 0;
    u8 shadow_funcs[16];
    WP2_BEGIN(_w31);
    const u32 shadow_mask = tmm ? (memcpy(shadow_funcs, tmm->sfun, 16), tmm->shadow) : eng_shadow_units(shadow_funcs);
    WP2_END(_w31, 31);
    u32 texel_ops[16];
    WP2_BEGIN(_w32);
    const u32 texop_mask = tmm ? (memcpy(texel_ops, tmm->ops, sizeof texel_ops), tmm->texop) : eng_texel_ops(texel_ops);
    WP2_END(_w32, 32);
    WP2_BEGIN(_w33);
    /* Texel-addressed units (RSX_TEX_FMT_UNNORM): the decompiler scales their
     * coordinates by 1/size. The D3D12 path patched this in; this engine never
     * did, so every post-process pass that samples a render target in texels
     * -- downsampling, blur, bloom, the luminance chain -- read coordinates
     * hundreds of times out of range and got one clamped edge texel. The
     * near-black luminance then drove Drakengard 3's exposure up until the
     * village interior washed out to white. RSX_NO_UNNORM=1 turns it off. */
    u32 unnorm_mask = 0, unnorm_dim[16][2];
    memset(unnorm_dim, 0, sizeof unnorm_dim);
    { static int off = -1; if (off < 0) off = getenv("RSX_NO_UNNORM") ? 1 : 0;
      const u32 reads = eng_fp_reads();
      s_tdec_pass = 3;
      for (u32 u = 0; !off && u < RSX_DSP_NUM_TEXTURES && u < 16; u++) {
          rsx_dsp_texture t;
          if (tmm) {   /* opt102: only the memoised candidates; same body as below */
              if (!((tmm->ucand >> u) & 1u)) continue;
              eng_get_tex(RV, u, &t);
          } else {
          if (!eng_tex_unit_on(u, reads)) continue;
          eng_get_tex(RV, u, &t);
          if (!t.enabled || !(t.format & RSX_TEX_FMT_UNNORM) || ((cube_mask >> u) & 1u)) continue;
          if (!t.width || !t.height) continue;
          }
          unnorm_mask |= 1u << u; unnorm_dim[u][0] = t.width; unnorm_dim[u][1] = t.height;
          { u32 sw = 0, sh = 0;
            if (eng_unnorm_surf_dim(&t, &sw, &sh) && (sw > t.width || sh > t.height)) {
                /* Per axis, only a target LARGER than the declared size: a declared size larger than the target is
                 * the 2x/4x horizontal MSAA scene (1920 declared in samples, 960 pixels) whose coordinates already
                 * span the whole width. */
                if (sw < t.width) sw = t.width;
                if (sh < t.height) sh = t.height;
                static u32 nl_; if (nl_ < 12 && ++nl_)
                    fprintf(stderr, "[rsx engine] unnorm unit %u: declared %ux%u, bound target %ux%u -> normalised by the target\n", u, t.width, t.height, sw, sh);
                unnorm_dim[u][0] = sw; unnorm_dim[u][1] = sh; } }
      } }

    WP2_END(_w17, 17);
    WP2_BEGIN(_w18);
    WP2_END(_w33, 33);
    const int slow = eng_slow();
    u64 fp_key = 0;
    if (fixed) g.fp_constants.count = 0;   /* a guest program's are already in (the lookup's scan) */
    if (!fixed) {
        if (slow) {
            if (rsx_fp_collect_constants(fp_uc, fp_size, &g.fp_constants) < 0) return 0;
        } else if (s_draw_open && s_progs.valid && s_progs.scanned && s_progs.fp == fp_uc &&
                   s_progs.fp_size == fp_size) {
            if (!s_progs.scan_ok) return 0;
            fp_key = s_progs.fp_key;
        } else if (rsx_fp_scan(fp_uc, fp_size, &g.fp_constants, &fp_key, NULL) < 0) {
            return 0;
        }
    }

    /* Identity: the vertex program's own bytes, the fragment program's
     * STRUCTURE (its inline constants are hoisted into the buffered block, so
     * a constant change must not be a new pipeline), the export-width bit of
     * SHADER_CONTROL, the cube and vertex-texture masks, the input layout,
     * and the structural render state. */
    /* Consecutive draws nearly always repeat every input of the key: reuse
     * the last key while they all compare equal (after NoahBPeterson's fork,
     * where hashing them was 2.7% of the RSX thread). The render state is
     * zeroed before it is decoded, so comparing its bytes compares fields;
     * the arrays are zeroed before they are filled. */
    static struct {
        int valid, fixed;
        u64 vp_key, fp_key;
        u32 fp_ctrl, cube_mask, vtex_mask, shadow_mask, unnorm_mask, texop_mask, ssr_cfg;
        u8 shadow_funcs[16]; u32 unnorm_dim[16][2]; u32 texel_ops[16];
        u32 layout_mask, layout_stride, rt_count;
        rsx_be_format rt_fmt;
        rsx_be_render_state rs;
        u32 vp_bb;
        u64 key;
    } km;
    u64 key;
    const u64 in_vp = fixed ? 0 : (s_vpi.valid && s_vpi.hash_valid ? s_vpi.hash : 0);
    /* BRB/CLB programs: the branch bits are part of the identity. Whether the
     * program has them is remembered per program hash (one scan). */
    u32 vp_bb = 0;
    if (!fixed) {
        static u64 bb_hash; static int bb_uses;
        if (!in_vp || in_vp != bb_hash) { bb_uses = rsx_vp_uses_branch_bits(vp_uc, vp_instrs * 16u); bb_hash = in_vp; }
        if (bb_uses) vp_bb = s_vp_bb_draw;
    }
    const int memo = !slow && !fixed && in_vp && fp_key && !eng_pso_norm_stats();   /* stats need every key computed */
    WP2_BEGIN(_w39);
    const int _kmh = (memo && km.valid && !km.fixed && km.vp_key == in_vp && km.fp_key == fp_key &&
        km.fp_ctrl == (fp_ctrl & 0x40u) && km.cube_mask == cube_mask && km.vtex_mask == vtex_mask && km.ssr_cfg == ssr_cfg &&
        km.shadow_mask == shadow_mask && km.unnorm_mask == unnorm_mask && km.texop_mask == texop_mask &&
        !memcmp(km.shadow_funcs, shadow_funcs, sizeof km.shadow_funcs) &&
        !memcmp(km.unnorm_dim, unnorm_dim, sizeof km.unnorm_dim) &&
        !memcmp(km.texel_ops, texel_ops, sizeof km.texel_ops) &&
        km.layout_mask == layout->mask && km.layout_stride == layout->stride &&
        km.rt_count == rt_count && km.rt_fmt == rt_fmt && km.vp_bb == vp_bb && !memcmp(&km.rs, rs, sizeof *rs));
    WP2_END(_w39, 39);
    if (_kmh) {
        key = km.key;
        WP2_CNT(22);
        goto key_done;
    }
    WP2_BEGIN(_w61);
    key = 1469598103934665603ull;
    if (fixed) {
        static const u32 fixed_tag = 0x4E464958u;   /* "XIFN" */
        key = eng_fnv1a(&fixed_tag, sizeof fixed_tag, key);
    } else {
        if (s_vpi.valid && s_vpi.hash_valid) {
            key = s_vpi.hash;
        } else {
            key = eng_fnv1a(vp_uc, vp_instrs * 16u, key);
            if (s_vpi.valid) { s_vpi.hash = key; s_vpi.hash_valid = 1; }
        }
        if (slow) {
            key = rsx_fp_structural_hash(fp_uc, fp_size, key);
            if (!key) return 0;
        } else {
            key = eng_fnv1a(&fp_key, sizeof fp_key, key);
        }
        const u32 fp_ctrl_key = fp_ctrl & 0x40u;
        key = eng_fnv1a(&fp_ctrl_key, sizeof fp_ctrl_key, key);
        key = eng_fnv1a(&cube_mask, sizeof cube_mask, key);
        key = eng_fnv1a(&vtex_mask, sizeof vtex_mask, key);
        if (ssr_cfg) { eng_ssr_params(); const u32 sv = 3; key = eng_fnv1a(&ssr_cfg, sizeof ssr_cfg, key); key = eng_fnv1a(&sv, sizeof sv, key); key = eng_fnv1a(&s_ssr_minup, sizeof s_ssr_minup, key); key = eng_fnv1a(&s_ssr_axis, sizeof s_ssr_axis, key); }
        if (vp_bb) key = eng_fnv1a(&vp_bb, sizeof vp_bb, key);
        if (shadow_mask) {
            key = eng_fnv1a(&shadow_mask, sizeof shadow_mask, key);
            key = eng_fnv1a(shadow_funcs, sizeof shadow_funcs, key);
            if (rsx_fp_shadow_oob_lit()) { const u32 ov = 0x00B0B011u; key = eng_fnv1a(&ov, sizeof ov, key); }   /* shad114: other HLSL */
        }
        if (unnorm_mask) {
            key = eng_fnv1a(&unnorm_mask, sizeof unnorm_mask, key);
            key = eng_fnv1a(unnorm_dim, sizeof unnorm_dim, key);
        }
        if (texop_mask) {
            key = eng_fnv1a(&texop_mask, sizeof texop_mask, key);
            key = eng_fnv1a(texel_ops, sizeof texel_ops, key);
        }
    }
    key = eng_fnv1a(&layout->mask, sizeof layout->mask, key);
    key = eng_fnv1a(&layout->stride, sizeof layout->stride, key);
    key = eng_fnv1a(&rt_fmt, sizeof rt_fmt, key);
    /* How many colour attachments the pass will have is pipeline identity:
     * a host API matches the two against each other. */
    key = eng_fnv1a(&rt_count, sizeof rt_count, key);
    const u64 key_pre_rs = key;
    key = rsx_draw_engine_hash_render_state(rs, key);
    if (eng_pso_norm_stats()) eng_pson_key(key_pre_rs, g.frames);
    km.valid = memo;
    WP2_END(_w61, 61);
    if (memo) {
        km.fixed = 0; km.vp_key = in_vp; km.fp_key = fp_key; km.fp_ctrl = fp_ctrl & 0x40u;
        km.cube_mask = cube_mask; km.vtex_mask = vtex_mask; km.shadow_mask = shadow_mask; km.ssr_cfg = ssr_cfg;
        km.unnorm_mask = unnorm_mask; km.texop_mask = texop_mask;
        memcpy(km.shadow_funcs, shadow_funcs, sizeof km.shadow_funcs);
        memcpy(km.unnorm_dim, unnorm_dim, sizeof km.unnorm_dim);
        memcpy(km.texel_ops, texel_ops, sizeof km.texel_ops);
        km.layout_mask = layout->mask; km.layout_stride = layout->stride;
        km.rt_count = rt_count; km.rt_fmt = rt_fmt; memcpy(&km.rs, rs, sizeof *rs); km.vp_bb = vp_bb;
        km.key = key;
    }
key_done:;
    if (s_wp_on > 1) { WP2_BEGIN(_w47); static u64 pk; static int pkv; static const u8* pfp; static u64 pfk; s_wp_n2[20]++; if (pkv && pk == key) s_wp_n2[21]++; pk = key; pkv = 1;
        s_wp_n2[25]++; if (fp_uc == pfp) s_wp_n2[23]++; if (fp_key == pfk) s_wp_n2[24]++; pfp = fp_uc; pfk = fp_key; WP2_END(_w47, 47); }
    WP2_END(_w18, 18);
    WP2_BEGIN(_w19);

    if (slow) {
        for (u32 i = 0; i < g.n_pipelines; i++)
            if (g.pipelines[i].key == key) {
                *out_fixed = g.pipelines[i].fixed;
                s_cur_pipe = &g.pipelines[i];
                return eng_pipe_poll_wait(&g.pipelines[i]);
            }
    } else {
        eng_pipeline* p = eng_pipe_find(key);
        if (p) {
            *out_fixed = p->fixed;
            s_cur_pipe = p;
            const u32 _r = eng_pipe_poll_wait(p);
            WP2_END(_w19, 19);
            return _r;
        }
    }
    if (g.n_pipelines >= ENG_MAX_PIPELINES) {
        static int n = 0;
        if (n < 4 && ++n) fprintf(stderr, "[rsx engine] pipeline table full (%u): draws needing a new pipeline drop\n", ENG_MAX_PIPELINES);
        return 0;
    }

    RSX_B2P_ENTER(B2P_PIPECREATE); RSX_B2P_ADD(B2C_PIPECREATE, 1);   /* left by the next RSX_B2P_SET in sink_end_impl */
    u32 handle = 0;
    int vi = 1, fi = 1;
    u32 nconst = 0;
    struct timespec t_create0, t_create1;
    timespec_get(&t_create0, TIME_UTC);
    if (fixed) {
        snprintf(s_vs_hlsl, sizeof s_vs_hlsl, "%s", kEngFixedVS);
        snprintf(s_ps_hlsl, sizeof s_ps_hlsl, "%s", kEngFixedPS);
    } else {
        rsx_vp_set_branch_bits(vp_bb);
        vi = rsx_vp_decompile_compact_ex(vp_uc, vp_instrs * 16u, vtex_mask,
                                         layout->mask, s_vs_hlsl,
                                         sizeof s_vs_hlsl);
        rsx_vp_set_branch_bits(0);
        rsx_fp_set_shadow_units(shadow_mask, shadow_funcs);
        rsx_fp_set_unnorm_units(unnorm_mask, unnorm_dim);
        rsx_fp_set_texel_ops(texop_mask ? texel_ops : NULL);
        rsx_fp_set_saturate_alpha(eng_fp_sat_alpha(fp_uc, fp_size));
        { static int wf = -1; if (wf < 0) wf = getenv("RSX_WPOS_FLIP") ? atoi(getenv("RSX_WPOS_FLIP")) : 1;
          rsx_fp_set_wpos_flip(wf); }
        if (ssr_cfg) { rsx_fp_set_ssr_debug((ssr_cfg >> 17) & 1u); rsx_fp_set_ssr_params(s_ssr_minup, s_ssr_axis); } rsx_fp_set_ssr((int)((ssr_cfg >> 8) & 0xFFu), (int)(ssr_cfg & 0xFFu));
        fi = rsx_fp_decompile_buffered_ex(fp_uc, fp_size, fp_ctrl, cube_mask,
                                          s_ps_hlsl, sizeof s_ps_hlsl, &nconst);
        rsx_fp_set_ssr(-1, -1);
        rsx_fp_set_shadow_units(0, NULL);
        rsx_fp_set_unnorm_units(0, NULL);
        rsx_fp_set_texel_ops(NULL);
        rsx_fp_set_saturate_alpha(0);
        rsx_fp_set_wpos_flip(0);
        if (unnorm_mask) { static int n = 0; if (n < 6 && ++n)
            fprintf(stderr, "[rsx engine] unnormalised units 0x%X (unit %u %ux%u)\n", unnorm_mask,
                    __builtin_ctz(unnorm_mask), unnorm_dim[__builtin_ctz(unnorm_mask)][0], unnorm_dim[__builtin_ctz(unnorm_mask)][1]); }
        if (shadow_mask) { static int n = 0; if (n < 6 && ++n)
            fprintf(stderr, "[rsx engine] shadow-compare units 0x%X (func of first: %u)\n",
                    shadow_mask, shadow_funcs[__builtin_ctz(shadow_mask)]); }
        if (fi > 0 && nconst != g.fp_constants.count) fi = -1;
        if (fi > 0 && rs->alpha_test_enable &&
            rsx_fp_apply_alpha_test_buffered(s_ps_hlsl, sizeof s_ps_hlsl,
                                             rs->alpha_func) < 0)
            fi = -1;
    }
    /* RSX_FP_SHOW_HASH=<hex fp-hlsl hash> + RSX_FP_SHOW=<hlsl expr>: diagnostic. That program's colour 0
     * output becomes float4((expr).xyz, 1) -- reads an intermediate register directly (white-eyes study). */
    int fp_show_hit = 0;
    { const char* sh = getenv("RSX_FP_SHOW"); const char* hh = getenv("RSX_FP_SHOW_HASH");
      if (sh && hh && !fixed && fi > 0) {
          unsigned long long ph0 = 1469598103934665603ull;
          for (const char* c = s_ps_hlsl; *c; c++) ph0 = (ph0 ^ (unsigned char)*c) * 1099511628211ull;
          /* the hash may be the fp-hlsl text hash or the microcode fp-hash (stable across builds); comma list */
          unsigned long long uh = 1469598103934665603ull; int hit = 0;
          for (u32 i = 0; fp_uc && i < fp_size; i++) uh = (uh ^ fp_uc[i]) * 1099511628211ull;
          for (const char* q = hh; q && *q; ) { unsigned long long w = strtoull(q, 0, 16);
              if (w == ph0 || w == uh) hit = 1; q = strchr(q, ','); if (q) q++; }
          /* RSX_FP_SHOW_AFTER=<text>: show the expression right after the first line of main() that contains
           * <text> (an early return), so a register can be read mid-program, before later code overwrites it. */
          const char* aft = getenv("RSX_FP_SHOW_AFTER");
          if (hit && aft && *aft) { hit = 0;
              char* mb = strstr(s_ps_hlsl, "PSOutput main("); char* ln = mb ? strstr(mb, aft) : 0; char* eol = ln ? strchr(ln, '\n') : 0;
              if (eol) { char ins[700]; const int nl = snprintf(ins, sizeof ins,
                      "\n    { PSOutput _so = (PSOutput)0; _so.t0 = float4((%s).xyz, 1.0); return _so; }", sh);
                  const size_t tail = strlen(eol) + 1;
                  if (nl > 0 && (size_t)(eol - s_ps_hlsl) + (size_t)nl + tail < sizeof s_ps_hlsl) {
                      memmove(eol + nl, eol, tail); memcpy(eol, ins, (size_t)nl);
                      fprintf(stderr, "[FPSHOW] after '%s' -> %s\n", aft, sh); } } }
          if (hit) fp_show_hit = 1;
          if (hit) {
              char* rp = strstr(s_ps_hlsl, "_out.t0 = "); char* semi = rp ? strchr(rp, ';') : 0;
              if (semi) { char rep[600]; snprintf(rep, sizeof rep, "_out.t0 = float4((%s).xyz, 1.0)", sh);
                  size_t nl = strlen(rep), tail = strlen(semi) + 1;
                  if ((size_t)(rp - s_ps_hlsl) + nl + tail < sizeof s_ps_hlsl) {
                      memmove(rp + nl, semi, tail); memcpy(rp, rep, nl);
                      fprintf(stderr, "[FPSHOW] fp-hlsl=%s -> %s\n", hh, sh); } }
              /* holo107: a single-target program ends in "return (_o == _o) ? _o : (float4)0;" -- same diagnostic there. */
              else { const char* pat = "return (_o == _o) ? _o : (float4)0;"; char* rq = strstr(s_ps_hlsl, pat);
                  if (rq) { char rep[600]; snprintf(rep, sizeof rep, "return float4((%s).xyz, 1.0);", sh);
                      size_t pl = strlen(pat), nl = strlen(rep), tail = strlen(rq + pl) + 1;
                      if ((size_t)(rq - s_ps_hlsl) + nl + tail < sizeof s_ps_hlsl) {
                          memmove(rq + nl, rq + pl, tail); memcpy(rq, rep, nl);
                          fprintf(stderr, "[FPSHOW] fp-hlsl=%s (single target) -> %s\n", hh, sh); } } } } } }
    /* RSX_FX_MAGENTA=<vp instrs>: pipelines whose vertex program has that many
     * instructions draw solid magenta with no blend, depth, alpha test or
     * cull -- whether an effect's geometry reaches the screen at all. */
    rsx_be_render_state fx_rs;
    /* RSX_FP_SHOW_NODEPTH=1: the RSX_FP_SHOW program also draws with no depth test, no blend, no cull --
     * how much of the screen it would cover if nothing hid it. */
    if (fp_show_hit && getenv("RSX_FP_SHOW_NODEPTH")) {
        fx_rs = *rs; fx_rs.depth_test = 0; fx_rs.depth_write = 0; fx_rs.blend_enable = 0; fx_rs.cull_enable = 0;
        fx_rs.alpha_test_enable = 0; fx_rs.stencil_enable = 0; rs = &fx_rs; }
    { static int want = -2;
      if (want == -2) { const char* e = getenv("RSX_FX_MAGENTA"); want = e ? atoi(e) : -1; }
      if (want > 0 && !fixed && vi > 0 && fi > 0 && vp_instrs == (u32)want) {
          char* o = s_ps_hlsl; size_t cap = sizeof s_ps_hlsl;
          int n = snprintf(o, cap, "struct PSInput { float4 position : SV_POSITION; };\n"
                                   "struct PSOutput {");
          for (u32 t = 0; t < (rt_count ? rt_count : 1u); t++)
              n += snprintf(o + n, cap - n, " float4 t%u : SV_TARGET%u;", t, t);
          n += snprintf(o + n, cap - n, " };\nPSOutput main(PSInput input) { PSOutput _o;");
          for (u32 t = 0; t < (rt_count ? rt_count : 1u); t++)
              n += snprintf(o + n, cap - n, " _o.t%u = float4(1,0,1,1);", t);
          snprintf(o + n, cap - n, " return _o; }\n");
          fx_rs = *rs; fx_rs.blend_enable = 0; fx_rs.depth_test = 0; fx_rs.depth_write = 0;
          fx_rs.alpha_test_enable = 0; fx_rs.cull_enable = 0; fx_rs.color_mask = 0x01010101u;
          fx_rs.stencil_enable = 0;
          rs = &fx_rs;
      } }
    struct eng_pipe_job* job = NULL;
#if defined(__APPLE__) || defined(_WIN32)
    if (vi > 0 && fi > 0 && !fixed && eng_async_on()) {
        job = eng_pipe_submit(key, s_vs_hlsl, s_ps_hlsl, rs, layout, layout->stride, rt_fmt, rt_count);
        if (job && __atomic_load_n(&job->done, __ATOMIC_ACQUIRE)) {
            handle = job->handle; free(job); job = NULL;
        } else if (job) {
            handle = 0;                          /* not ready: this draw is skipped */
            s_pj_skips++; s_pj_skips_total++;
        }
    }
    if (!job && handle == 0)
#endif
    if (vi > 0 && fi > 0)
        handle = g.be->pipeline_create(g.be->user, s_vs_hlsl, s_ps_hlsl, rs,
                                       layout, layout->stride, rt_fmt, rt_count);
    timespec_get(&t_create1, TIME_UTC);
    /* RSX_PIPE_LOG=1 lifts the cap: the draw trace names pipelines by handle,
     * and the fragment program's FNV hash is what the shader dump files carry. */
    { static u32 logs = 0; static int uncapped = -1;
      if (uncapped < 0) uncapped = getenv("RSX_PIPE_LOG") ? 1 : 0;
      if (uncapped || logs++ < 32) {
        unsigned long long fh = 1469598103934665603ull;
        for (u32 i = 0; fp_uc && i < fp_size; i++) fh = (fh ^ fp_uc[i]) * 1099511628211ull;
        /* The HLSL hashes are the names the shader dump files carry
         * (PS3RECOMP_METAL_SHADER_DUMP writes vp_<hash>.msl / fp_<hash>.msl). */
        unsigned long long vh = 1469598103934665603ull, ph = 1469598103934665603ull;
        for (const char* c = s_vs_hlsl; *c; c++) vh = (vh ^ (unsigned char)*c) * 1099511628211ull;
        for (const char* c = s_ps_hlsl; *c; c++) ph = (ph ^ (unsigned char)*c) * 1099511628211ull;
        fprintf(stderr, "[rsx engine] pipeline %016llx: %s vp %d, fp %d,"
                        " %u constants -> %s handle=%u fp-hash=%016llx fp-size=%u blend=%u(%X/%X) vp-hlsl=%016llx fp-hlsl=%016llx vp-start=%u vp-instrs=%u"
                        " prec(h/x12/x9)=%u/%u/%u exptex=%u fp-struct=%016llx\n",
                (unsigned long long)key, fixed ? "built-in" : "guest",
                vi, fi, nconst, handle ? "ok" : job ? "compiling (draw skipped until it is)" : "FAILED (draw dropped)",
                handle, fh, fp_size,
                rs->blend_enable, rs->sf_rgb, rs->df_rgb, vh, ph, rsx_dsp_vp_start(RV), vp_instrs,
                g_rsx_fp_stats.prec[1], g_rsx_fp_stats.prec[2], g_rsx_fp_stats.prec[3], g_rsx_fp_stats.exp_tex,
                (unsigned long long)eng_fp_struct_id(fp_uc, fp_size));
        /* RSX_HLSL_DUMP_DIR=<dir>: write this pipeline's translated shaders as
         * <dir>/<key>_vs.hlsl and _ps.hlsl (the key is the one logged above). */
        { const char* dd = getenv("RSX_HLSL_DUMP_DIR");
          if (dd && *dd) {
              char path[512]; FILE* hf;
              snprintf(path, sizeof path, "%s/%016llx_vs.hlsl", dd, (unsigned long long)key);
              if ((hf = fopen(path, "wb"))) { fputs(s_vs_hlsl, hf); fclose(hf); }
              snprintf(path, sizeof path, "%s/%016llx_ps.hlsl", dd, (unsigned long long)key);
              if ((hf = fopen(path, "wb"))) { fputs(s_ps_hlsl, hf); fclose(hf); }
          } }
        /* RSX_VP_HEX=<vp-hlsl hash>: that program's microcode, four words per
         * instruction, for decoding a field by hand against the decompiler. */
        { static unsigned long long want = 0; static int got = -1;
          if (got < 0) { const char* e = getenv("RSX_VP_HEX"); want = e ? strtoull(e, 0, 16) : 0; got = 0; }
          if (want && vh == want && !got && vp_uc) { got = 1;
              fprintf(stderr, "[vp-hex] vp-hlsl=%016llx start=%u instrs=%u\n", vh, rsx_dsp_vp_start(RV), vp_instrs);
              for (u32 i = 0; i < vp_instrs; i++) {
                  const u32* w = (const u32*)(vp_uc + i * 16);
                  fprintf(stderr, "[vp-hex] %3u: %08X %08X %08X %08X\n", i, w[0], w[1], w[2], w[3]); } } } } }

    g.pipelines[g.n_pipelines].key = key;
    g.pipelines[g.n_pipelines].handle = handle;
    g.pipelines[g.n_pipelines].fixed = (u8)fixed;
    g.pipelines[g.n_pipelines].pending = job ? 1 : 0;
    g.pipelines[g.n_pipelines].job = job;
    { u64 vf = 1469598103934665603ull, vh2 = 1469598103934665603ull;
      for (u32 i = 0; vp_uc && i < vp_instrs * 16u; i++) vf = (vf ^ vp_uc[i]) * 1099511628211ull;
      for (const char* c = s_vs_hlsl; *c; c++) vh2 = (vh2 ^ (unsigned char)*c) * 1099511628211ull;
      g.pipelines[g.n_pipelines].vp_fnv = vf;
      g.pipelines[g.n_pipelines].vp_hlsl = fixed ? 0 : vh2;
      u64 fh2 = 1469598103934665603ull;
      for (const char* c = s_ps_hlsl; *c; c++) fh2 = (fh2 ^ (unsigned char)*c) * 1099511628211ull;
      g.pipelines[g.n_pipelines].fp_hlsl = fixed ? 0 : fh2;
      { u64 ff = 1469598103934665603ull; for (u32 i = 0; fp_uc && i < fp_size; i++) ff = (ff ^ fp_uc[i]) * 1099511628211ull;
        g.pipelines[g.n_pipelines].fp_fnv = fixed ? 0 : ff; }   /* the microcode fp-hash of the pipeline log (stable across builds) */
      g.pipelines[g.n_pipelines].rt_static = (g_rsx_rt_on && !fixed && vi > 0) ? (u8)rsx_rt_register_vs(vh2, s_vs_hlsl) : 0;
      g.pipelines[g.n_pipelines].wpos = (u8)(!fixed && strstr(s_ps_hlsl, "input.position") != NULL);
      s_cur_pipe = &g.pipelines[g.n_pipelines]; }
    /* A pipeline built mid-game stalls the FIFO walker for as long as it
     * takes (decompile, translate, Metal compile): name every slow one. */
    { static double total_ms = 0;
      const double ms = (double)(t_create1.tv_sec - t_create0.tv_sec) * 1e3 +
                        (double)(t_create1.tv_nsec - t_create0.tv_nsec) / 1e6;
      total_ms += ms;
      s_fstat.pipes++; s_fstat.pipe_ms += ms;
      if (ms >= 20.0)
          fprintf(stderr, "[pipe-slow] frame %u: new pipeline took %.1f ms (vp-hlsl %016llx fp-hlsl %016llx); %u built, %.0f ms in all\n",
                  g.frames, ms, (unsigned long long)g.pipelines[g.n_pipelines].vp_hlsl,
                  (unsigned long long)g.pipelines[g.n_pipelines].fp_hlsl, g.n_pipelines + 1, total_ms); }
    eng_pipe_index_add(g.n_pipelines);
    g.n_pipelines++;
    *out_fixed = fixed;
    return handle;
}

/* ---- draw accumulation --------------------------------------------------- */

static void dc_reset(void)
{
    dc.n_arr = dc.n_idx = dc.n_packets = 0;
    dc.n_refs = dc.n_source_refs = dc.n_verts = dc.n_cuts = 0;
    dc.out_verts = NULL;
    dc.refs_remapped = 0;
    dc.fetch_ok = 1;
    dc.inl = NULL;
    dc.inl_bytes = 0;
}

/* Room for `need` references in dc.refs, grown as dc_push_ref grows it. */
static int dc_reserve_refs(u64 need)
{
    if (need <= dc.cap_refs) return 1;
    if (need > 0xFFFFFFFFull) return 0;
    u32 next = dc.cap_refs ? dc.cap_refs : 4096u;
    while (next < need) next = next * 2u > next ? next * 2u : (u32)need;
    rsx_vertex_ref* r = (rsx_vertex_ref*)realloc(dc.refs, (size_t)next * sizeof(*r));
    if (!r) return 0;
    dc.refs = r;
    dc.cap_refs = next;
    return 1;
}

/* A mapped run of big-endian indices into dc.refs (room reserved), a restart
 * index becoming a cut at the reference position it falls on. Out of line
 * and one loop per width and restart mode: inlined into the fetch, the width
 * was tested and each index assembled a byte at a time on every index (after
 * NoahBPeterson's fork of ps3recomp, where it was 11.6% of the RSX thread). */
#define DC_DECODE(T, BSWAP, RESTART)                                              \
    for (u32 i = 0; i < count; i++) {                                             \
        T raw; memcpy(&raw, run + (size_t)i * sizeof(T), sizeof(T));             \
        const u32 index = (u32)BSWAP(raw);                                        \
        if (RESTART && index == restart_val) {                                    \
            if (!rsx_restart_cut_push(&dc.cuts, &dc.n_cuts, &dc.cap_cuts, n)) {   \
                dc.n_refs = n; return 0;                                          \
            }                                                                     \
            continue;                                                             \
        }                                                                         \
        out[n].vertex_id = index;                                                 \
        out[n].base_index = base_index;                                           \
        n++;                                                                      \
    }
static __attribute__((noinline)) int dc_decode_indices(const u8* run, u32 count, int is_u32,
                                                       int restart_en, u32 restart_val,
                                                       u32 base_index)
{
    rsx_vertex_ref* const out = dc.refs;
    u32 n = dc.n_refs;
    if (is_u32) {
        if (restart_en) { DC_DECODE(u32, __builtin_bswap32, 1) }
        else            { DC_DECODE(u32, __builtin_bswap32, 0) }
    } else {
        if (restart_en) { DC_DECODE(u16, __builtin_bswap16, 1) }
        else            { DC_DECODE(u16, __builtin_bswap16, 0) }
    }
    dc.n_refs = n;
    return 1;
}
#undef DC_DECODE

static int dc_push_ref(u32 vertex_id, u32 base_index)
{
    if (dc.n_refs >= dc.cap_refs) {
        const u32 next = dc.cap_refs ? dc.cap_refs * 2u : 4096u;
        rsx_vertex_ref* r = (rsx_vertex_ref*)realloc(dc.refs,
                                                     (size_t)next * sizeof(*r));
        if (!r) return 0;
        dc.refs = r;
        dc.cap_refs = next;
    }
    dc.refs[dc.n_refs].vertex_id = vertex_id;
    dc.refs[dc.n_refs].base_index = base_index;
    dc.n_refs++;
    return 1;
}

static int dc_reserve_verts(u64 bytes)
{
    if (bytes <= dc.verts_cap) return 1;
    u64 next = dc.verts_cap ? dc.verts_cap : (1u << 20);
    while (next < bytes) next *= 2u;
    u8* v = (u8*)realloc(dc.verts, (size_t)next);
    if (!v) return 0;
    dc.verts = v;
    dc.verts_cap = next;
    return 1;
}

static int dc_reserve_indices(u32 count)
{
    if (count <= g.index_cap) return 1;
    u32 next = g.index_cap ? g.index_cap : 4096u;
    while (next < count) next *= 2u;
    u32* p = (u32*)realloc(g.indices, (size_t)next * sizeof(u32));
    if (!p) return 0;
    g.indices = p;
    g.index_cap = next;
    return 1;
}

/* Turn this group's batches into a reference list, recording a CUT rather
 * than fetching a phantom vertex wherever the restart sentinel appears, then
 * collapse repeated references and fetch each unique vertex once.
 * rsx_live_draw.c's fetch_batches_hoisted (4165-4292). */
/* RSX_VFETCH_CHECK totals, printed at exit. */
static unsigned long *s_vfc_draws, *s_vfc_bad, *s_vfc_bytes;
static void eng_vfetch_check_summary(void)
{
    if (s_vfc_draws)
        fprintf(stderr, "[vfetch-check] total: %lu draws compared, %lu mismatched (%lu bytes)\n",
                *s_vfc_draws, *s_vfc_bad, *s_vfc_bytes);
}

/* ---- vertex cache -------------------------------------------------------- *
 *
 * Most of what a frame draws is the same static meshes from the same guest
 * vertex and index buffers, frame after frame, and converting them again was
 * most of the draw engine's time: a Drakengard 3 battle frame decodes ~280k
 * vertices (19 MB) from ~560k index references. A draw is keyed on everything
 * that decides its conversion other than memory -- the vertex layout, every
 * fetched attribute's descriptor and default, the index array and the batch
 * ranges -- and an entry is reused only when a hash of the bytes it read (the
 * index ranges, then each attribute's vertex span) still matches. The hash is
 * taken BEFORE the conversion reads memory, so a guest write racing it can
 * only cause a miss later, never a stale hit. RSX_VCACHE=0 turns it off;
 * RSX_VCACHE_CHECK=1 converts every hit again and compares. */
/* metro131 RSX_VC_CHUNK: what an entry's last trusted validation saw of the write counters (vm_watch_chunk_gens) of the 1 MB chunks its ranges
 * (vertex spans + this key's index ranges) lie in. Unchanged counters and the same chunks = nothing in those chunks was written since. */
#define ENG_VR_MAX 8
typedef struct { u32 n; u32 c0[ENG_VR_MAX], c1[ENG_VR_MAX], g0[ENG_VR_MAX], g1[ENG_VR_MAX]; } EngVR;
typedef struct { u32 location, start, len; } EngVCSpan;
typedef struct {
    u64 key, content;                       /* key 0: empty slot            */
    u32 last_frame;
    u32 n_verts, stride, n_source_refs, n_draw;
    int indexed;
    u32 n_spans;
    EngVCSpan span[RSX_DSP_NUM_VERTEX_ATTR];
    u8* verts;
    u32* indices;
    size_t bytes;
    u32 buf;                                /* backend buffer, or 0          */
    u32 ib_off;                             /* the indices' offset in it     */
    u8* cpu_verts; u32* cpu_indices;        /* RT_AO: the CPU copy kept next to the GPU buffer */
    u64 watch_stamp;                        /* write-watch stamp at the last hash */
    int watched;                            /* 1: every page was watched then */
    u64 gen;                                /* store serial: a prep result names the entry it validated by it */
    u32 vr_ok;                              /* metro131: vr below is a trusted validation's record */
    EngVR vr;
    u32 ep_ok, ep;                          /* prof102: global write epoch at the last trusted validation */
} EngVCEntry;
typedef struct {
    int ok;
    u64 content;
    u32 n_spans;
    EngVCSpan span[RSX_DSP_NUM_VERTEX_ATTR];
} EngVCFill;
/* What the vertex-cache key and the lookup read of the draw: the batches. Normally dc's; a prep worker points t_di at its job's copy. */
typedef struct { const eng_batch* arr; u32 n_arr; const eng_batch* idx; u32 n_idx; const u8* inl; u32 inl_bytes; } DrawIn;
static ENG_TLS const DrawIn* t_di;
static inline DrawIn di_get(void)
{
    if (t_di) return *t_di;
    DrawIn d = { dc.arr, dc.n_arr, dc.idx, dc.n_idx, dc.inl, dc.inl_bytes };
    return d;
}
/* RSX_VC_SLOTS=<n> (power of two, 4096..65536; default 32768 since 1.0, 8192 before): the table's size. drop132: in Metropolis a frame draws ~5300
 * distinct cacheable entries, more than half of 8192, so eng_vc_evict's keep_none branch dropped the WHOLE cache every ~9
 * frames and the next frame converted all of it again (5330 stores / 72 MB, +35 ms, and +8 ms on the frame after: a 65 ms
 * hitch roughly every 300 ms). 32768 keeps the live set and evicts rarely. */
static u32 s_vc_slots = 32768u;   /* 1.0: default 32768 (drop132 quiet A/B, Metropolis: frames >50 ms -78%); RSX_VC_SLOTS=8192 = old */
#define ENG_VC_SLOTS  s_vc_slots
#define ENG_VC_BUDGET ((size_t)384 << 20)
#define ENG_VC_GPU_MIN ((size_t)64 << 10)    /* entries this big get a buffer */
static EngVCEntry* s_vc;
static u64* s_vck;   /* walk37: the entries' keys alone (64 KB, stays in L2): the probe no longer touches a ~300-byte entry per step */
static u32 s_vc_count;
static u64 s_vc_gen;
#ifdef _WIN32
static SRWLOCK s_vc_lock = SRWLOCK_INIT;   /* shared: a prep worker looking an entry up; exclusive: store / evict / free (B2) */
#endif
static size_t s_vc_bytes;
static struct { unsigned long long hit, miss, stored, uncacheable, check_bad, stale, hit_watch; } s_vcstat;

static inline u64 eng_vc_mum(u64 a, u64 b)
{
    const __uint128_t r = (__uint128_t)a * b;
    return (u64)r ^ (u64)(r >> 64);
}

/* Four independent 128-bit-multiply lanes over 64-byte blocks; every byte
 * reaches every bit of the result, unlike the texture mutation hash. */
static u64 eng_vc_hash(const u8* p, size_t n, u64 seed)
{
    const u64 s0 = 0xa0761d6478bd642full, s1 = 0xe7037ed1a0b428dbull,
              s2 = 0x8ebc6af09c88c6e3ull, s3 = 0x589965cc75374cc3ull;
    u64 a = seed ^ s0, b = seed ^ s1, c = seed ^ s2, d = seed ^ s3;
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        u64 w[8];
        memcpy(w, p + i, sizeof w);
        a = eng_vc_mum(w[0] ^ s0, w[1] ^ a);
        b = eng_vc_mum(w[2] ^ s1, w[3] ^ b);
        c = eng_vc_mum(w[4] ^ s2, w[5] ^ c);
        d = eng_vc_mum(w[6] ^ s3, w[7] ^ d);
    }
    for (; i + 8 <= n; i += 8) {
        u64 w; memcpy(&w, p + i, 8);
        a = eng_vc_mum(w ^ s1, a ^ s2);
    }
    if (i < n) {
        u64 w = 0; memcpy(&w, p + i, n - i);
        b = eng_vc_mum(w ^ s3, b ^ s0);
    }
    const u64 h = eng_vc_mum(a ^ s1, b ^ s2) ^ eng_vc_mum(c ^ s3, d ^ s0);
    return eng_vc_mum(h ^ (u64)n, s1 ^ seed);
}

/* RSX_FETCH_PLAN_ONCE=0: build the fetch plan again in dc_fetch (old path). */
static ENG_TLS const rsx_vertex_fetch_plan* s_fp;   /* the plan eng_vc_key built (storage: s_vkm.plan); copied only on a miss */
static ENG_TLS int s_fp_valid;
static int eng_fp_once(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("RSX_FETCH_PLAN_ONCE"); on = !(e && e[0] == '0'); }
    return on;
}

static int eng_vc_enabled(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("RSX_VCACHE"); on = !(e && e[0] == '0'); }
    return on;
}
static unsigned long long s_vc_checked, s_vc_check_bad_total;
static void eng_vc_check_summary(void)
{
    fprintf(stderr, "[vcache-check] %llu hits converted again, %llu mismatched\n",
            s_vc_checked, s_vc_check_bad_total);
}
static int eng_vc_checking(void)
{
    static int on = -1;
    if (on < 0) { on = getenv("RSX_VCACHE_CHECK") ? 1 : 0; if (on) atexit(eng_vc_check_summary); }
    return on;
}

/* Everything other than memory that decides the draw's conversion; 0 for a
 * draw the cache does not take (an inline stream, or no batches). */
static u32 s_vcdbg_n, s_vcdbg_m[256];
#define ENG_VK_MAX (8 + RSX_DSP_NUM_VERTEX_ATTR * 13 + 2 * ENG_MAX_BATCHES * 2 + 16)

/* RSX_VCKEY_MEMO=1: the part of the key that comes from registers (the fetch plan, the head of the key and the
 * index-array words) is built again only when one of those registers changed (g_rsx_dsp_vtx_gen, see
 * rsx_dispatch.c) or the draw's prim / rebuild / layout differ. 2: as 1, and every hit is also rebuilt and
 * compared (a [vkmemo] line with the counts every 240 presents; any mismatch is printed). */
extern u32 g_rsx_dsp_vtx_gen;
static ENG_TLS u32 s_draw_vtx_gen;   /* g_rsx_dsp_vtx_gen as of the END of the draw being run (from the DrawSnap under RSX_DRAW_SPLIT=2) */
static ENG_TLS struct { int valid; u32 gen, prim, rebuild, lmask, lstride, lcount, n_head, ia[6];
                u64 hh;   /* metro131 RSX_VCKEY_SPLIT: eng_vc_hash of head[0..n_head) */
                u32 head[ENG_VK_MAX]; rsx_vertex_fetch_plan plan; } s_vkm;
/* metro131 RSX_VCKEY_SPLIT=1 (default OFF): the key is the hash of the per-draw words (batches, index array) seeded with the hash of the
 * register-derived head, which the memo keeps -- the head's ~300 bytes are neither copied nor hashed again for every draw that
 * repeats it. A different function of the same inputs: keys only have to agree within one run. Needs RSX_VCKEY_MEMO=1. */
static int eng_vkm_mode(void);
static int eng_vkey_split(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("RSX_VCKEY_SPLIT"); on = (e && e[0] && e[0] != '0' && eng_vkm_mode() == 1 && !getenv("RSX_VCACHE_BURST_LOG")) ? 1 : 0; }
    return on;
}
static unsigned long long s_vkm_hit, s_vkm_miss, s_vkm_bad;
static int eng_vkm_mode(void)
{
    static int mode = -1;
    if (mode < 0) { const char* e = getenv("RSX_VCKEY_MEMO"); mode = (e && e[0]) ? atoi(e) : 1; if (mode < 0) mode = 0;   /* default ON (28/10) */ }
    return mode;
}

/* RSX_VCKEY_NODEF (default ON since 1.1, =0 off): leave out of the key what the conversion cannot read: the constant default value of an
 * attribute that has an array (type and size set; the default is only used for a disabled attribute or an unreadable
 * source), the divider bits / frequency that rsx_vertex_element_index ignores (frequency <= 1), and the array fields of a
 * disabled attribute. A guest that writes a constant into a vertex
 * attribute register (VERTEX_DATA4F) and keeps it there otherwise changed the key of every following draw of the same
 * meshes (Metropolis bursts: ~5000 re-conversions per frame for 12-25 frames). */
static int eng_vckey_nodef(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("RSX_VCKEY_NODEF"); on = !(e && e[0] == '0'); }   /* 1.1: default ON, =0 turns it off */
    return on;
}

/* The register-derived words of the key: the head, up to the draw's own ranges, and the index-array tail. */
static u32 eng_vc_head(const rsx_vertex_layout_plan* layout, u32 prim, int rebuild, rsx_vertex_fetch_plan* plan,
                       u32* m, u32 ia_out[6])
{
    rsx_vertex_fetch_plan_init(plan, RV, layout, eng_guest_ptr, NULL);
    u32 n = 0;
    m[n++] = prim; m[n++] = (u32)rebuild;
    m[n++] = layout->mask; m[n++] = layout->stride; m[n++] = layout->count;
    const int nodef = eng_vckey_nodef();
    u32 dmask = plan->divider_mask;
    if (nodef) {   /* a divider bit counts only for an attribute that is fetched with a frequency above 1 (rsx_vertex_element_index) */
        dmask = 0;
        for (u32 slot = 0; slot < layout->count; slot++) {
            const u32 attr = layout->attrs[slot];
            const rsx_vertex_fetch_attr* f = &plan->attr[attr];
            if (f->desc.type && f->desc.size && f->desc.frequency > 1u) dmask |= plan->divider_mask & (1u << attr);
        }
    }
    m[n++] = plan->base_offset; m[n++] = dmask;
    for (u32 slot = 0; slot < layout->count; slot++) {
        const u32 attr = layout->attrs[slot];
        const rsx_vertex_fetch_attr* f = &plan->attr[attr];
        m[n++] = attr;
        if (nodef && !(f->desc.type && f->desc.size)) {   /* disabled attribute: every vertex reads its default, nothing else counts */
            m[n++] = f->desc.type; m[n++] = f->desc.size;
            memset(&m[n], 0, 6 * 4u); n += 6;
            memcpy(&m[n], f->default_value, 16); n += 4;
            continue;
        }
        m[n++] = f->desc.type; m[n++] = f->desc.size; m[n++] = f->desc.stride;
        m[n++] = nodef ? (f->desc.frequency > 1u ? f->desc.frequency : 0u) : f->desc.frequency;   /* 0 and 1 fetch alike */
        m[n++] = f->desc.offset; m[n++] = f->desc.location;
        m[n++] = f->elem_size; m[n++] = f->stride;
        if (nodef) memset(&m[n], 0, 16); else memcpy(&m[n], f->default_value, 16);   /* an array attribute does not read its default */
        n += 4;
    }
    rsx_dsp_index_array ia;
    rsx_dsp_get_index_array(RV, &ia);
    ia_out[0] = ia.offset; ia_out[1] = ia.location; ia_out[2] = ia.is_u32;
    ia_out[3] = rsx_dsp_vertex_data_base_index(RV);
    ia_out[4] = (u32)rsx_dsp_restart_index_enabled(RV, ia.is_u32);
    ia_out[5] = rsx_dsp_restart_index(RV);
    return n;
}

static void eng_vkm_report(void)
{
    static u32 last; if (g.frames - last < 240) return;
    last = g.frames;
    fprintf(stderr, "[vkmemo] %llu hits, %llu rebuilds, %llu audit mismatches\n", s_vkm_hit, s_vkm_miss, s_vkm_bad);
}

static u64 eng_vc_key(const rsx_vertex_layout_plan* layout, u32 prim, int rebuild)
{
    /* A capture records the guest pages a draw reads (eng_guest_ptr); a
     * cache hit reads none, so while one runs every draw converts afresh. */
    const DrawIn di = di_get();
    if (!eng_vc_enabled() || g_rsx_capture_on || di.inl || di.inl_bytes) return 0;
    if (!di.n_arr && !di.n_idx) return 0;
    u32 m[ENG_VK_MAX];
    u32 n = 0;
    u32 ia[6];
    const rsx_vertex_fetch_plan* plan;
    const int memo = eng_vkm_mode();
    const int split = eng_vkey_split();
    u64 kseed = 0x5643414348450001ull;
    if (memo && s_vkm.valid && s_vkm.gen == s_draw_vtx_gen && s_vkm.prim == prim && s_vkm.rebuild == (u32)rebuild &&
        s_vkm.lmask == layout->mask && s_vkm.lstride == layout->stride && s_vkm.lcount == layout->count) {
        s_vkm_hit++;
        n = s_vkm.n_head;
        if (split) { n = 0; kseed = s_vkm.hh; }
        else memcpy(m, s_vkm.head, n * 4u);
        memcpy(ia, s_vkm.ia, sizeof ia);
        plan = &s_vkm.plan;
        if (memo >= 2) {
            u32 m2[ENG_VK_MAX], ia2[6]; rsx_vertex_fetch_plan p2;
            const u32 n2 = eng_vc_head(layout, prim, rebuild, &p2, m2, ia2);
            if (n2 != n || memcmp(m2, m, n * 4u) || memcmp(ia2, ia, sizeof ia) || memcmp(&p2, plan, sizeof p2)) {
                if (s_vkm_bad++ < 8)
                    fprintf(stderr, "[vkmemo] MISMATCH at frame %u: head %u/%u words, ia %d, plan %d\n", g.frames, n, n2,
                            memcmp(ia2, ia, sizeof ia) != 0, memcmp(&p2, plan, sizeof p2) != 0);
            }
            eng_vkm_report();
        }
    } else {
        s_vkm.valid = 0;
        WP2_BEGIN(_w50);
        n = eng_vc_head(layout, prim, rebuild, &s_vkm.plan, m, ia);   /* built in place: no 1.4 KB copy per draw */
        WP2_END(_w50, 50);
        plan = &s_vkm.plan;
        if (memo) {
            s_vkm_miss++;
            s_vkm.valid = 1; s_vkm.gen = s_draw_vtx_gen; s_vkm.prim = prim; s_vkm.rebuild = (u32)rebuild;
            s_vkm.lmask = layout->mask; s_vkm.lstride = layout->stride; s_vkm.lcount = layout->count;
            s_vkm.n_head = n; memcpy(s_vkm.head, m, n * 4u); memcpy(s_vkm.ia, ia, sizeof ia);
            if (split) s_vkm.hh = eng_vc_hash((const u8*)m, (size_t)n * 4u, 0x5643414348450001ull);
        }
        if (split) { kseed = s_vkm.hh; n = 0; }
    }
    if (eng_fp_once()) { s_fp = plan; s_fp_valid = 1; }
    m[n++] = di.n_arr;
    for (u32 b = 0; b < di.n_arr; b++) { m[n++] = di.arr[b].first; m[n++] = di.arr[b].count; }
    m[n++] = di.n_idx;
    for (u32 b = 0; b < di.n_idx; b++) { m[n++] = di.idx[b].first; m[n++] = di.idx[b].count; }
    if (di.n_idx) {
        m[n++] = ia[0]; m[n++] = ia[1]; m[n++] = ia[2];
        m[n++] = ia[3];
        m[n++] = ia[4];
        m[n++] = ia[5];
    }
    { static int dbg = -1; if (dbg < 0) dbg = getenv("RSX_VCACHE_BURST_LOG") ? 1 : 0;   /* only the burst log reads it */
      if (dbg) { s_vcdbg_n = n < 256 ? n : 256; memcpy(s_vcdbg_m, m, s_vcdbg_n * 4u); } }
    WP2_BEGIN(_w51);
    const u64 k = eng_vc_hash((const u8*)m, (size_t)n * 4u, kseed);
    WP2_END(_w51, 51);
    return k ? k : 1;
}

/* RSX_VCACHE_BURST_LOG=<threshold>: debug aid. Remembers the key fields of every
 * stored conversion by content hash (bytes only); in a frame with more than
 * <threshold> misses it prints, for a sample of misses, which key field differs
 * from the last conversion of the same bytes, and a histogram of those fields. */
static int eng_vc_hash_indices(u64* h);
static int eng_vc_hash_spans(const EngVCSpan* sp, u32 n, u64* h);
static void eng_vcdbg_miss(const EngVCFill* f)
{
    static long thr = -2;
    if (thr == -2) { const char* e = getenv("RSX_VCACHE_BURST_LOG"); thr = e ? atol(e) : -1; }
    if (thr < 0) return;
    /* RSX_VCACHE_BURST_FROM=<frame>: report only bursts from that frame on
     * (skips the level-load cluster). */
    static long from = -2;
    if (from == -2) { const char* e = getenv("RSX_VCACHE_BURST_FROM"); from = e ? atol(e) : 0; }
    typedef struct { u64 h; u32 n; u32 m[256]; } Ent;
    static Ent* tab; static const u32 TS = 1u << 15;
    static u32 cur_frame = ~0u, nmiss, nsame, nnew, nsamp; static u32 hist[300]; static char samp[16][400];
    static int bursts;
    if (!tab) tab = (Ent*)calloc(TS, sizeof(Ent));
    if (!tab) return;
    if (cur_frame != g.frames) {
        if (cur_frame != ~0u && (long)nmiss > thr && (long)cur_frame >= from && bursts < 6) {
            bursts++;
            fprintf(stderr, "[vc-burst] frame %u misses %u (same-fields-diff-content %u, new-content %u)\n", cur_frame, nmiss, nsame, nnew);
            for (u32 i = 0; i < 300; i++) if (hist[i]) fprintf(stderr, "[vc-burst]   field[%u] differs in %u misses\n", i, hist[i]);
            for (u32 i = 0; i < nsamp; i++) fprintf(stderr, "%s\n", samp[i]);
        }
        cur_frame = g.frames; nmiss = nsame = nnew = nsamp = 0; memset(hist, 0, sizeof hist);
    }
    nmiss++;
    u64 h = 0;
    if (!eng_vc_hash_indices(&h) || !eng_vc_hash_spans(f->span, f->n_spans, &h)) return;
    if (!h) h = 1;
    u32 i = (u32)(h >> 7) & (TS - 1u);
    Ent* e = NULL;
    for (u32 p = 0; p < TS; p++, i = (i + 1u) & (TS - 1u)) { if (!tab[i].h || tab[i].h == h) { e = &tab[i]; break; } }
    if (!e) return;
    if (e->h == h) {
        const u32 n = e->n < s_vcdbg_n ? e->n : s_vcdbg_n;
        if (e->n != s_vcdbg_n) hist[299]++;
        int first = -1;
        for (u32 k = 0; k < n; k++) if (e->m[k] != s_vcdbg_m[k]) {
            hist[k < 299 ? k : 298]++;
            if (first < 0) first = (int)k;
        }
        if (first < 0) nsame++;
        else if (nsamp < 16) {
            char* o = samp[nsamp++]; size_t sz = 400; int w = snprintf(o, sz, "[vc-burst] n=%u prim %u stride %u diffs:", s_vcdbg_n, s_vcdbg_m[0], s_vcdbg_m[3]);
            for (u32 k = 0; k < n && w < (int)sz - 40; k++) if (e->m[k] != s_vcdbg_m[k])
                w += snprintf(o + w, sz - w, " [%u]%08X->%08X", k, e->m[k], s_vcdbg_m[k]);
        }
    } else nnew++;
    e->h = h; e->n = s_vcdbg_n; memcpy(e->m, s_vcdbg_m, s_vcdbg_n * 4u);
}

/* The index bytes the draw reads, batch by batch; 0 when a batch is not one
 * contiguous guest range. */
static int eng_vc_hash_indices(u64* h)
{
    const DrawIn di = di_get();
    if (!di.n_idx) return 1;
    rsx_dsp_index_array ia;
    rsx_dsp_get_index_array(RV, &ia);
    const u32 esz = ia.is_u32 ? 4u : 2u;
    const u32 ia_loc = ia.location ? RSX_LOCATION_MAIN : RSX_LOCATION_LOCAL;
    for (u32 b = 0; b < di.n_idx; b++) {
        const u64 start = (u64)ia.offset + (u64)di.idx[b].first * esz;
        const u64 bytes = (u64)di.idx[b].count * esz;
        if (!bytes) continue;
        if (start + bytes > 0x100000000ull) return 0;
        const u8* p = eng_guest_ptr(NULL, ia_loc, (u32)start, (u32)bytes);
        if (!p) return 0;
        *h = eng_vc_hash(p, (size_t)bytes, *h);
    }
    return 1;
}

static int eng_vc_hash_spans(const EngVCSpan* sp, u32 n, u64* h)
{
    for (u32 i = 0; i < n; i++) {
        const u8* p = eng_guest_ptr(NULL, sp[i].location, sp[i].start, sp[i].len);
        if (!p) return 0;
        *h = eng_vc_hash(p, sp[i].len, *h);
    }
    return 1;
}

static EngVCEntry* eng_vc_slot(u64 key)
{
    if (!s_vc) {
        { const char* e = getenv("RSX_VC_SLOTS"); const unsigned long v = e ? strtoul(e, 0, 0) : 0ul;
          if (v >= 4096ul && v <= 65536ul && (v & (v - 1ul)) == 0ul) s_vc_slots = (u32)v; }
        s_vc = (EngVCEntry*)calloc(ENG_VC_SLOTS, sizeof *s_vc);
        if (!s_vc) return NULL;
        free(s_vck); s_vck = (u64*)calloc(ENG_VC_SLOTS, sizeof *s_vck);
    }
    u32 i = (u32)key & (ENG_VC_SLOTS - 1u);
    if (s_vck) {
        for (u32 probe = 0; probe < ENG_VC_SLOTS; probe++) {
            const u64 k = s_vck[i];
            if (!k || k == key) return &s_vc[i];
            i = (i + 1u) & (ENG_VC_SLOTS - 1u);
        }
        return NULL;
    }
    for (u32 probe = 0; probe < ENG_VC_SLOTS; probe++) {
        EngVCEntry* e = &s_vc[i];
        if (!e->key || e->key == key) return e;
        i = (i + 1u) & (ENG_VC_SLOTS - 1u);
    }
    return NULL;
}

/* Arm the texture write-watch (vm_watch) over everything validating an entry
 * would hash -- its vertex spans and this draw's index ranges -- and return
 * the combined stamp; *trusted is 1 when every page is watched. Armed before
 * any hash, so a write during or after one changes the next stamp. */
int g_eng_vc_watch = 1;   /* DOD3_AB=vcwatch switches it in a run */
static u64 eng_vc_arm(const EngVCEntry* e, int* trusted)
{
    *trusted = 0;
    if (!g_eng_vc_watch || !eng_tex_watch_on() || g_rsx_capture_on) return 0;
    int ok = 1;
    u64 st = 0x5643574154434801ull;
    for (u32 i = 0; i < e->n_spans; i++) {
        u64 x = 0;
        if (!eng_tex_watch_arm(e->span[i].location, e->span[i].start, e->span[i].len, &x)) ok = 0;
        st = eng_vc_mum(st ^ x, 0x9E3779B97F4A7C15ull) + e->span[i].start;
    }
    const DrawIn di = di_get();
    if (di.n_idx) {
        rsx_dsp_index_array ia;
        rsx_dsp_get_index_array(RV, &ia);
        const u32 esz = ia.is_u32 ? 4u : 2u;
        const u32 ia_loc = ia.location ? RSX_LOCATION_MAIN : RSX_LOCATION_LOCAL;
        for (u32 b = 0; b < di.n_idx; b++) {
            const u64 start = (u64)ia.offset + (u64)di.idx[b].first * esz;
            const u64 bytes = (u64)di.idx[b].count * esz;
            if (!bytes) continue;
            if (start + bytes > 0x100000000ull) return 0;
            u64 x = 0;
            if (!eng_tex_watch_arm(ia_loc, (u32)start, (u32)bytes, &x)) ok = 0;
            st = eng_vc_mum(st ^ x, 0x9E3779B97F4A7C15ull) + start;
        }
    }
    *trusted = ok;
    return st;
}

/* A valid entry for the key: present, and the bytes it read unchanged --
 * known from the write-watch when every page it reads is watched and none
 * was written since the last check, otherwise by hashing them again (the
 * texture cache does the same; RSX_TEX_WATCH=0 turns both off). */
/* metro131 RSX_VC_CHUNK=1 (default OFF): skip the page walk of eng_vc_arm for an entry whose chunks saw no write since its last trusted validation. */
static int s_vcchunk_on = -1;
static volatile long* s_cgen;
static unsigned long long s_vcchunk_hit, s_vcchunk_chk, s_vcchunk_bad;
static int eng_vcchunk_on(void)
{
    if (s_vcchunk_on < 0) {
        const char* e = getenv("RSX_VC_CHUNK");
        s_vcchunk_on = (e && e[0] && e[0] != '0') ? 1 : 0;
        if (s_vcchunk_on) {
            s_cgen = s_guest_reader ? NULL : vm_watch_chunk_gens();
            if (!s_cgen) s_vcchunk_on = 0;
        }
    }
    return s_vcchunk_on && g_eng_vc_watch && eng_tex_watch_on() && !g_rsx_capture_on;
}
static int eng_vr_add(EngVR* o, int local, u32 offset, u32 len)
{
    if (!len) return 1;
    const u32 ea = cellGcmResolveLocated(local, offset);
    if (!ea) return 0;
    const u64 end = (u64)ea + len - 1u;
    if (end > 0xFFFFFFFFull || o->n >= ENG_VR_MAX) return 0;
    const u32 c0 = ea >> 20, c1 = (u32)(end >> 20);
    if (c1 - c0 > 1u) return 0;
    o->c0[o->n] = c0; o->c1[o->n] = c1; o->g0[o->n] = (u32)s_cgen[c0]; o->g1[o->n] = (u32)s_cgen[c1]; o->n++;
    return 1;
}
/* The counters as they are now for the ranges eng_vc_arm would arm (same ranges, same resolve). 0 = cannot describe them (too many, or a span crossing 2 chunk boundaries). */
static int eng_vr_snap(const EngVCEntry* e, EngVR* o)
{
    o->n = 0;
    for (u32 i = 0; i < e->n_spans; i++)
        if (!eng_vr_add(o, e->span[i].location == RSX_LOCATION_LOCAL, e->span[i].start, e->span[i].len)) return 0;
    const DrawIn di = di_get();
    if (di.n_idx) {
        rsx_dsp_index_array ia;
        rsx_dsp_get_index_array(RV, &ia);
        const u32 esz = ia.is_u32 ? 4u : 2u;
        const int local = ia.location ? 0 : 1;
        for (u32 b = 0; b < di.n_idx; b++) {
            const u64 start = (u64)ia.offset + (u64)di.idx[b].first * esz;
            const u64 bytes = (u64)di.idx[b].count * esz;
            if (!bytes) continue;
            if (start + bytes > 0x100000000ull) return 0;
            if (!eng_vr_add(o, local, (u32)start, (u32)bytes)) return 0;
        }
    }
    return 1;
}
static inline int eng_vr_eq(const EngVR* a, const EngVR* b)
{
    if (a->n != b->n) return 0;
    for (u32 i = 0; i < a->n; i++)
        if (a->c0[i] != b->c0[i] || a->c1[i] != b->c1[i] || a->g0[i] != b->g0[i] || a->g1[i] != b->g1[i]) return 0;
    return 1;
}

static EngVCEntry* eng_vc_lookup_impl(u64 key);
static EngVCEntry* eng_vc_lookup(u64 key)
{
    RSX_B2P_ENTER(B2P_VCLOOKUP);
    EngVCEntry* v = eng_vc_lookup_impl(key);
    RSX_B2P_LEAVE();
    return v;
}
static EngVCEntry* eng_vc_lookup_impl(u64 key)
{
    EngVCEntry* e = eng_vc_slot(key);
    if (!e || e->key != key) return NULL;
    EngVR vsnap; int have_snap = 0;
    if (eng_vcchunk_on()) {
        /* Read before the arm below: a write between the two bumps a counter, so the next check fails and takes the full path. */
        have_snap = eng_vr_snap(e, &vsnap);
        if (have_snap && e->vr_ok && e->watched && eng_vr_eq(&e->vr, &vsnap)) {
            static int vchk = -1;
            if (vchk < 0) vchk = getenv("RSX_VC_CHUNK_CHECK") ? 1 : 0;   /* hash anyway and report a hit the counters got wrong */
            if (vchk) {
                u64 hc = key; s_vcchunk_chk++;
                if (!eng_vc_hash_indices(&hc) || !eng_vc_hash_spans(e->span, e->n_spans, &hc) || hc != e->content) {
                    if (s_vcchunk_bad++ < 16) fprintf(stderr, "[vc-chunk] MISMATCH at frame %u: key %016llx (%u bytes of vertices)\n", g.frames, (unsigned long long)key, (unsigned)e->bytes);
                    e->watched = 0; e->vr_ok = 0;
                    return NULL;
                }
                if ((s_vcchunk_chk & 0xFFFFF) == 0) fprintf(stderr, "[vc-chunk] check: %llu shortcut hits verified, %llu bad\n", s_vcchunk_chk, s_vcchunk_bad);
            }
            s_vcchunk_hit++;
            s_vcstat.hit_watch++;
            e->last_frame = g.frames;
            return e;
        }
    }
    int trusted = 0;
    EngVR csnap; int chave = 0; u32 ep0 = 0; const u32 lf0 = e->last_frame;
    if (s_wp_on > 1) {
        WP2_BEGIN(_w55);
        s_wp_n2[3]++;
        { static int cg_tried; if (!cg_tried) { cg_tried = 1; if (!s_cgen && !s_guest_reader) s_cgen = vm_watch_chunk_gens(); } }
        if (s_cgen) { if (have_snap) { csnap = vsnap; chave = 1; } else chave = eng_vr_snap(e, &csnap); ep0 = (u32)s_cgen[4096]; }
        WP2_END(_w55, 55);
    }
    const unsigned long long _a11 = s_wp_on > 1 ? wp_tsc() : 0;
    const u64 st = eng_vc_arm(e, &trusted);
    const unsigned long long _armc = s_wp_on > 1 ? wp_tsc() - _a11 : 0;
    if (s_wp_on > 1) s_wp_cyc[11] += _armc;
    if (trusted && e->watched && st == e->watch_stamp) {
        /* RSX_VC_WATCH_CHECK=1: hash anyway and count the hits the watch got
         * wrong (a write it never saw) as check mismatches. */
        static int chk = -1;
        if (chk < 0) chk = getenv("RSX_VC_WATCH_CHECK") ? 1 : 0;
        if (chk) {
            u64 hc = key;
            if (!eng_vc_hash_indices(&hc) || !eng_vc_hash_spans(e->span, e->n_spans, &hc) || hc != e->content) {
                s_vcstat.check_bad++;
                e->watched = 0;
                e->vr_ok = 0;
                return NULL;
            }
        }
        s_vcstat.hit_watch++;
        e->last_frame = g.frames;
        if (have_snap) { e->vr = vsnap; e->vr_ok = 1; }
        if (s_wp_on > 1) {
            WP2_BEGIN(_w47);
            s_wp_n2[4]++; s_wp_cyc[59] += _armc;
            const int cs = chave && e->vr_ok && !have_snap && eng_vr_eq(&e->vr, &csnap);
            const int es = e->ep_ok && e->ep == ep0;
            if (cs) { s_wp_n2[5]++; s_wp_cyc[53] += _armc; }
            if (es) { s_wp_n2[6]++; s_wp_cyc[54] += _armc; }
            if (lf0 == g.frames) s_wp_n2[7]++; else if (cs) s_wp_n2[8]++;
            if (chave && !have_snap) { e->vr = csnap; e->vr_ok = 1; }
            e->ep = ep0; e->ep_ok = 1;
            WP2_END(_w47, 47);
        }
        return e;
    }
    u64 h = key;
    WP2_BEGIN(_w12);
    const int _hok = eng_vc_hash_indices(&h) && eng_vc_hash_spans(e->span, e->n_spans, &h) && h == e->content;
    WP2_END(_w12, 12);
    if (s_wp_on > 1) { s_wp_n2[0]++; if (_hok) s_wp_n2[1]++; if (!trusted) s_wp_n2[2]++; }
    if (!_hok) {
        s_vcstat.stale++;
        e->watched = 0;
        e->vr_ok = 0; e->ep_ok = 0;
        return NULL;
    }
    e->watch_stamp = st;
    e->watched = trusted;
    if (have_snap && trusted) { e->vr = vsnap; e->vr_ok = 1; } else e->vr_ok = 0;
    if (s_wp_on > 1) {
        if (trusted && chave && !have_snap) { e->vr = csnap; e->vr_ok = 1; }
        if (trusted) { e->ep = ep0; e->ep_ok = 1; } else e->ep_ok = 0;
    }
    e->last_frame = g.frames;
    return e;
}

/* RSX_DRAW_WORKERS: the lookup a prep worker (or B2 stealing a prep) does for a draw, off the commit thread. Read-mostly: the shared lock keeps
 * the table and its entries alive against eng_vc_store/evict (exclusive, B2); it never allocates the table, never touches last_frame (B2 sets
 * it when it uses the entry) and the validated entry is named by its store serial, which B2 compares before using it. Returns 1 = valid hit
 * (*egen = the entry's serial), 0 = no entry or stale (B2 then runs the whole serial path). */
static int eng_vc_lookup_pre(u64 key, u64* egen)
{
    int ret = 0;
#ifdef _WIN32
    AcquireSRWLockShared(&s_vc_lock);
#endif
    EngVCEntry* e = NULL;
    if (s_vc) {
        u32 i = (u32)key & (ENG_VC_SLOTS - 1u);
        for (u32 probe = 0; probe < ENG_VC_SLOTS; probe++) {
            EngVCEntry* c = &s_vc[i];
            if (!c->key) break;
            if (c->key == key) { e = c; break; }
            i = (i + 1u) & (ENG_VC_SLOTS - 1u);
        }
    }
    if (e) {
        int trusted = 0;
        const u64 st = eng_vc_arm(e, &trusted);
        const int w = __atomic_load_n(&e->watched, __ATOMIC_ACQUIRE);
        const u64 ws = __atomic_load_n(&e->watch_stamp, __ATOMIC_RELAXED);
        if (trusted && w && st == ws) {
            s_vcstat.hit_watch++;
            *egen = e->gen; ret = 1;
        } else {
            u64 h = key;
            if (!eng_vc_hash_indices(&h) || !eng_vc_hash_spans(e->span, e->n_spans, &h) || h != e->content) {
                s_vcstat.stale++;
                __atomic_store_n(&e->watched, 0, __ATOMIC_RELEASE);
            } else {
                __atomic_store_n(&e->watch_stamp, st, __ATOMIC_RELAXED);
                __atomic_store_n(&e->watched, trusted, __ATOMIC_RELEASE);
                *egen = e->gen; ret = 1;
            }
        }
    }
#ifdef _WIN32
    ReleaseSRWLockShared(&s_vc_lock);
#endif
    return ret;
}

/* An entry reused (its second draw) and at least RSX_VCACHE_PROMOTE bytes
 * (default 4 KB, 0 never) gets a buffer of its own then, as one of
 * ENG_VC_GPU_MIN does when it is stored: from there on it is drawn in place
 * instead of copied into the per-submit arena at every draw. Doing it at the
 * second use leaves the entries drawn once as cheap as before. */
#define ENG_VC_PROMOTE_MIN ((size_t)4 << 10)
static size_t eng_vc_promote_min(void)
{
    static size_t v = (size_t)-1;
    if (v == (size_t)-1) { const char* e = getenv("RSX_VCACHE_PROMOTE");
                           v = e ? (size_t)strtoull(e, 0, 0) : ENG_VC_PROMOTE_MIN; }
    return v;
}
static void eng_vc_promote(EngVCEntry* e)
{
    if (!g.be->buffer_wrap || !g.be->draw_buffer || !g.be->buffer_release) return;
    const size_t vb = (size_t)e->n_verts * e->stride;
    const size_t ib = e->indexed ? (size_t)e->n_draw * sizeof(u32) : 0;
#ifdef _WIN32
    const size_t page = 4096u;
#else
    const size_t page = (size_t)getpagesize();
#endif
    const u32 ib_off = (u32)((vb + 255u) & ~(size_t)255u);
    const size_t total = ((size_t)ib_off + ib + page - 1u) & ~(page - 1u);
    if (total > 0xFFFFFFFFu) return;
    void* mem = NULL;
#ifdef _WIN32
    mem = _aligned_malloc(total, page);
    if (!mem) return;
#else
    if (posix_memalign(&mem, page, total) != 0) return;
#endif
    memcpy(mem, e->verts, vb);
    if (ib) memcpy((u8*)mem + ib_off, e->indices, ib);
    const int keep_cpu = g_rsx_rt_on;   /* RT_AO reads the vertices again (static geometry for the acceleration structures) */
    const double t_wrap = eng_now_ms();
    const u32 buf = g.be->buffer_wrap(g.be->user, mem, (u32)total);
    s_fstat.vc_wrap_ms += eng_now_ms() - t_wrap;
    if (!buf) {
#ifdef _WIN32
        _aligned_free(mem);
#else
        free(mem);
#endif
        return;
    }
    /* As eng_vc_store keeps them: the buffer's memory, which a backend that
     * copies (D3D12) has already let go of -- nothing reads them for an
     * entry with a buffer. */
    if (keep_cpu) { e->cpu_verts = e->verts; e->cpu_indices = e->indices; }
    else { free(e->verts); free(e->indices); }
    e->verts = (u8*)mem;
    e->indices = ib ? (u32*)((u8*)mem + ib_off) : NULL;
    e->buf = buf;
    e->ib_off = ib_off;
}

static void eng_vc_free(EngVCEntry* e)
{
    if (e->buf) g.be->buffer_release(g.be->user, e->buf);   /* frees later */
    else { free(e->verts); free(e->indices); }
    free(e->cpu_verts); free(e->cpu_indices);
    s_vc_bytes -= e->bytes;
    memset(e, 0, sizeof *e);
}

/* perf101 03/11: RSX_VC_DEFER_FREE (default OFF until the 3-run A/B is complete; =1 on). eng_vc_evict used to free every dropped entry (verts/indices heap blocks,
 * backend buffer releases) inside the one walker frame that hit the table limit: 11-58 ms hitch every ~190 frames in Metropolis
 * (log\perf101 M2: 32 evicts, 779 ms), a frame of +15..40 ms each. Now the dropped entries' pointers go to a list that the following
 * stores drain a few at a time (under the same lock that evict ran in); the table itself is rebuilt at once as before. */
typedef struct { void* v; void* x; void* cv; void* cx; u32 buf; } EngVCDead;
static EngVCDead* s_vc_dead;
static u32 s_vc_dead_n, s_vc_dead_cap;
static int eng_vc_defer_on(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("RSX_VC_DEFER_FREE"); on = (e && e[0] == '1'); }
    return on;
}
static void eng_vc_dead_one(const EngVCDead* d)
{
    if (d->buf) g.be->buffer_release(g.be->user, d->buf);
    else { free(d->v); free(d->x); }
    free(d->cv); free(d->cx);
}
static void eng_vc_dead_drain(u32 max)
{
    while (s_vc_dead_n && max--) eng_vc_dead_one(&s_vc_dead[--s_vc_dead_n]);
}
/* Like eng_vc_free, but the memory is released later by eng_vc_dead_drain (falls back to freeing now if the list cannot grow). */
static void eng_vc_free_later(EngVCEntry* e)
{
    if (s_vc_dead_n == s_vc_dead_cap) {
        const u32 nc = s_vc_dead_cap ? s_vc_dead_cap * 2u : 4096u;
        EngVCDead* nd = (EngVCDead*)realloc(s_vc_dead, (size_t)nc * sizeof *nd);
        if (!nd) { eng_vc_free(e); return; }
        s_vc_dead = nd; s_vc_dead_cap = nc;
    }
    EngVCDead* d = &s_vc_dead[s_vc_dead_n++];
    d->v = e->verts; d->x = e->indices; d->cv = e->cpu_verts; d->cx = e->cpu_indices; d->buf = e->buf;
    s_vc_bytes -= e->bytes;
    memset(e, 0, sizeof *e);
}

/* Over budget, or the table filling up: keep what the last two frames drew,
 * re-inserted so the probe chains stay intact, and drop the rest -- or
 * everything, when even the recent draws would leave it nearly full again. */
static void eng_vc_evict(void)
{
    const double t0 = eng_now_ms();
    const u32 before = s_vc_count;
    size_t recent = 0;
    u32 n_recent = 0;
    const int defer = eng_vc_defer_on();
    u32 n_gpu = 0;
    eng_vc_dead_drain(0xFFFFFFFFu);   /* a previous evict's leftovers first */
    for (u32 i = 0; i < ENG_VC_SLOTS; i++)
        if (s_vc[i].key && s_vc[i].last_frame + 1u >= g.frames) { recent += s_vc[i].bytes; n_recent++; }
    const int keep_none = recent > ENG_VC_BUDGET / 2u || n_recent > ENG_VC_SLOTS / 2u;
    const double t_scan = eng_now_ms() - t0;
    EngVCEntry* old = s_vc;
    u64* old_k = s_vck;
    s_vc = (EngVCEntry*)calloc(ENG_VC_SLOTS, sizeof *s_vc);
    if (!s_vc) { s_vc = old; return; }
    s_vck = (u64*)calloc(ENG_VC_SLOTS, sizeof *s_vck);
    if (!s_vck) { free(s_vc); s_vc = old; s_vck = old_k; return; }
    s_vc_count = 0;
    for (u32 i = 0; i < ENG_VC_SLOTS; i++) {
        EngVCEntry* e = &old[i];
        if (!e->key) continue;
        if (keep_none || e->last_frame + 1u < g.frames) { if (e->buf) n_gpu++; if (defer) eng_vc_free_later(e); else eng_vc_free(e); continue; }
        EngVCEntry* d = eng_vc_slot(e->key);
        if (!d) { if (defer) eng_vc_free_later(e); else eng_vc_free(e); continue; }
        *d = *e;
        s_vck[d - s_vc] = d->key;
        s_vc_count++;
    }
    free(old); free(old_k);
    s_fstat.vc_evicts++; s_fstat.vc_evict_ms += eng_now_ms() - t0; s_fstat.vc_evicted += before - s_vc_count;
    { static int lg = -1; if (lg < 0) { const char* e = getenv("RSX_VC_LOG"); lg = e && e[0] == '1'; }
      if (lg) fprintf(stderr, "[vcache] frame %u: evict %u -> %u entries (%s), %.1f ms, slots %u, %zu MB live\n", g.frames, before, s_vc_count,
                      keep_none ? "dropped ALL" : "kept the last 2 frames", eng_now_ms() - t0, ENG_VC_SLOTS, s_vc_bytes >> 20); }
}

static void eng_vc_store_x(u64 key, const EngVCFill* f, const u8* verts,
                         u32 n_verts, u32 stride, u32 n_source_refs,
                         int indexed, const u32* indices, u32 n_draw)
{
    if (s_vc_dead_n) eng_vc_dead_drain(48);   /* perf101: the entries a recent evict dropped, a few per store */
    if (s_vc_bytes > ENG_VC_BUDGET || s_vc_count >= ENG_VC_SLOTS * 3u / 4u)
        eng_vc_evict();
    EngVCEntry* e = eng_vc_slot(key);
    if (!e) return;
    if (e->key) { eng_vc_free(e); s_vc_count--; if (s_vck) s_vck[e - s_vc] = 0; }
    const size_t vb = (size_t)n_verts * stride;
    const size_t ib = indexed ? (size_t)n_draw * sizeof(u32) : 0;
    u8* v = NULL;
    u32* x = NULL;
    u32 buf = 0, ib_off = 0;
    /* A large entry lives in a buffer of its own that the GPU reads in place,
     * where a small one is copied into the per-submit arena with the rest. */
    /* RSX_VCACHE_GPU_MIN=<bytes>: the smallest entry given a buffer of its own. */
    static size_t gpu_min = 0;
    if (!gpu_min) { const char* e = getenv("RSX_VCACHE_GPU_MIN");
                    gpu_min = e ? (size_t)strtoull(e, 0, 0) : ENG_VC_GPU_MIN; if (!gpu_min) gpu_min = 1; }
    if (vb + ib >= gpu_min && g.be->buffer_wrap && g.be->draw_buffer &&
        g.be->buffer_release) {
        /* Page-aligned, a whole number of pages: what buffer_wrap takes. On
         * Windows the memory comes from _aligned_malloc and the backend frees
         * it with _aligned_free; elsewhere posix_memalign and free. */
#ifdef _WIN32
        const size_t page = 4096u;
#else
        const size_t page = (size_t)getpagesize();
#endif
        ib_off = (u32)((vb + 255u) & ~(size_t)255u);
        const size_t total = ((size_t)ib_off + ib + page - 1u) & ~(page - 1u);
        void* mem = NULL;
#ifdef _WIN32
        if (total <= 0xFFFFFFFFu) mem = _aligned_malloc(total, page);
        if (mem) {
#else
        if (total <= 0xFFFFFFFFu && posix_memalign(&mem, page, total) == 0) {
#endif
            memcpy(mem, verts, vb);
            if (ib) memcpy((u8*)mem + ib_off, indices, ib);
            { const double t_wrap = eng_now_ms();
              buf = g.be->buffer_wrap(g.be->user, mem, (u32)total);
              s_fstat.vc_wrap_ms += eng_now_ms() - t_wrap; }
            if (buf) { v = (u8*)mem; x = ib ? (u32*)((u8*)mem + ib_off) : NULL; }
#ifdef _WIN32
            else _aligned_free(mem);
#else
            else free(mem);
#endif
        }
    }
    if (!buf) {
        ib_off = 0;
        v = (u8*)malloc(vb ? vb : 1);
        x = ib ? (u32*)malloc(ib) : NULL;
        if (!v || (ib && !x)) { free(v); free(x); return; }
        memcpy(v, verts, vb);
        if (ib) memcpy(x, indices, ib);
    }
    e->key = key;
    if (s_vck) s_vck[e - s_vc] = key;
    e->gen = ++s_vc_gen;
    e->content = f->content;
    e->watch_stamp = 0; e->watched = 0; e->vr_ok = 0; e->ep_ok = 0;   /* validated by hashing until a lookup arms the watch */
    e->last_frame = g.frames;
    e->n_verts = n_verts; e->stride = stride;
    e->n_source_refs = n_source_refs;
    e->indexed = indexed; e->n_draw = n_draw;
    e->n_spans = f->n_spans;
    memcpy(e->span, f->span, sizeof e->span);
    e->verts = v; e->indices = x;
    e->buf = buf; e->ib_off = ib_off;
    /* RT_AO: a large entry lives in a GPU buffer and its CPU memory is gone (D3D12 copies it): keep a copy for the acceleration structure. */
    if (g_rsx_rt_on && buf) {
        e->cpu_verts = (u8*)malloc(vb ? vb : 1);
        if (e->cpu_verts) memcpy(e->cpu_verts, verts, vb);
        if (ib && e->cpu_verts) { e->cpu_indices = (u32*)malloc(ib); if (e->cpu_indices) memcpy(e->cpu_indices, indices, ib); else { free(e->cpu_verts); e->cpu_verts = NULL; } }
    }
    e->bytes = vb + ib;
    s_vc_bytes += e->bytes;
    s_vc_count++;
    s_vcstat.stored++;
    s_fstat.vc_store++; s_fstat.vc_bytes += vb + ib;
}

static void eng_vc_store(u64 key, const EngVCFill* f, const u8* verts,
                         u32 n_verts, u32 stride, u32 n_source_refs,
                         int indexed, const u32* indices, u32 n_draw)
{
#ifdef _WIN32
    if (s_wk_n) AcquireSRWLockExclusive(&s_vc_lock);
#endif
    eng_vc_store_x(key, f, verts, n_verts, stride, n_source_refs, indexed, indices, n_draw);
#ifdef _WIN32
    if (s_wk_n) ReleaseSRWLockExclusive(&s_vc_lock);
#endif
}

/* The prepared plan's vertex spans, merged where interleaved attributes
 * overlap, hashed into the fill. 0 when an attribute has no single span. */
static int eng_vc_fill_spans(EngVCFill* f, const rsx_vertex_fetch_plan* plan)
{
    EngVCSpan sp[RSX_DSP_NUM_VERTEX_ATTR];
    u32 n = 0;
    for (u32 slot = 0; slot < plan->layout.count; slot++) {
        const rsx_vertex_fetch_attr* a = &plan->attr[plan->layout.attrs[slot]];
        if (!a->desc.type || !a->desc.size) continue;
        if (!a->stable_base || !a->elem_size) return 0;
        const u64 prefix = (u64)plan->base_offset + a->desc.offset;
        const u64 start = prefix + (u64)a->stable_first * a->stride;
        const u64 end = prefix + (u64)a->stable_last * a->stride + a->elem_size;
        if (end <= start || end > 0x100000000ull) return 0;
        EngVCSpan s = { a->desc.location, (u32)start, (u32)(end - start) };
        u32 k = n;
        while (k > 0 && (sp[k - 1].location > s.location ||
                         (sp[k - 1].location == s.location && sp[k - 1].start > s.start))) {
            sp[k] = sp[k - 1]; k--;
        }
        sp[k] = s; n++;
    }
    f->n_spans = 0;
    for (u32 i = 0; i < n; i++) {
        EngVCSpan* last = f->n_spans ? &f->span[f->n_spans - 1] : NULL;
        if (last && last->location == sp[i].location &&
            sp[i].start <= last->start + last->len) {
            const u32 end = sp[i].start + sp[i].len;
            if (end > last->start + last->len) last->len = end - last->start;
        } else {
            f->span[f->n_spans++] = sp[i];
        }
    }
    return eng_vc_hash_spans(f->span, f->n_spans, &f->content);
}

static void dc_fetch(const rsx_vertex_layout_plan* layout, int allow_remap,
                     EngVCFill* vc_fill)
{
    /* The index bytes are hashed before they are read; see the cache note. */
    WP2_BEGIN(_w34);
    if (vc_fill && vc_fill->ok && !eng_vc_hash_indices(&vc_fill->content))
        vc_fill->ok = 0;
    WP2_END(_w34, 34);

    WP2_BEGIN(_w35);
    for (u32 b = 0; b < dc.n_arr && dc.fetch_ok; b++)
        for (u32 i = 0; i < dc.arr[b].count && dc.fetch_ok; i++)
            if (!dc_push_ref(dc.arr[b].first + i, 0)) dc.fetch_ok = 0;

    if (dc.n_idx && dc.fetch_ok) {
        const u32 base_index = rsx_dsp_vertex_data_base_index(RV);
        rsx_dsp_index_array ia;
        rsx_dsp_get_index_array(RV, &ia);
        const u32 esz = ia.is_u32 ? 4u : 2u;
        const int restart_en = rsx_dsp_restart_index_enabled(RV, ia.is_u32);
        const u32 restart_val = rsx_dsp_restart_index(RV);
        /* The index array's location field is a DMA selector, not the
         * rsx_dsp location enum: 0 selects local memory. */
        const u32 ia_loc = ia.location ? RSX_LOCATION_MAIN : RSX_LOCATION_LOCAL;

        for (u32 b = 0; b < dc.n_idx && dc.fetch_ok; b++) {
            const u32 first = dc.idx[b].first, count = dc.idx[b].count;
            const u64 start = (u64)ia.offset + (u64)first * esz;
            const u64 bytes = (u64)count * esz;
            const u8* run = NULL;
            if (count && start <= 0xFFFFFFFFull && bytes <= 0xFFFFFFFFull &&
                start + bytes <= 0x100000000ull)
                run = eng_guest_ptr(NULL, ia_loc, (u32)start, (u32)bytes);
            if (run && count && !eng_slow()) {
                if (!dc_reserve_refs((u64)dc.n_refs + count) ||
                    !dc_decode_indices(run, count, ia.is_u32, restart_en, restart_val, base_index))
                    dc.fetch_ok = 0;
                continue;
            }
            for (u32 i = 0; i < count && dc.fetch_ok; i++) {
                const u8* p = run ? run + (size_t)i * esz
                    : eng_guest_ptr(NULL, ia_loc, ia.offset + (first + i) * esz, esz);
                if (!p) { dc.fetch_ok = 0; break; }
                const u32 index = ia.is_u32
                    ? (((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | p[3])
                    : (u32)((p[0] << 8) | p[1]);
                if (restart_en && index == restart_val) {
                    if (!rsx_restart_cut_push(&dc.cuts, &dc.n_cuts, &dc.cap_cuts,
                                              dc.n_refs))
                        dc.fetch_ok = 0;
                    continue;
                }
                if (!dc_push_ref(index, base_index)) dc.fetch_ok = 0;
            }
        }
    }
    WP2_END(_w35, 35);
    if (!dc.fetch_ok) return;

    WP2_BEGIN(_w36);
    dc.n_source_refs = dc.n_refs;
    /* Only a topology this engine rebuilds through an index buffer may share
     * repeated references. A line strip or a point list is drawn in the order
     * it arrived, so collapsing duplicates would reorder it. */
    if (allow_remap && dc.n_refs > 1) {
        u32 unique = dc.n_refs;
        if (rsx_vertex_remap_build(&dc.ref_remap, dc.refs, dc.n_refs, &unique)) {
            dc.refs_remapped = unique < dc.n_refs;
            dc.n_refs = unique;
        }
    }

    WP2_END(_w36, 36);
    WP2_BEGIN(_w37);
    dc.layout = *layout;
    /* Same registers, layout and guest_ptr as the plan eng_vc_key just built
     * for this draw: reuse it (RSX_FETCH_PLAN_ONCE=0 rebuilds). */
    if (s_fp_valid && s_fp->layout.mask == layout->mask && s_fp->layout.stride == layout->stride &&
        s_fp->layout.count == layout->count)
        dc.fetch_plan = *s_fp;
    else
        rsx_vertex_fetch_plan_init(&dc.fetch_plan, RV, layout, eng_guest_ptr, NULL);
    s_fp_valid = 0;
    rsx_vertex_fetch_plan_set_inline(&dc.fetch_plan, RV, dc.inl, dc.inl_bytes);
    rsx_vertex_fetch_plan_prepare(&dc.fetch_plan, dc.refs, dc.n_refs);
    if (vc_fill && vc_fill->ok && !eng_vc_fill_spans(vc_fill, &dc.fetch_plan))
        vc_fill->ok = 0;
    WP2_END(_w37, 37);
    if (!dc_reserve_verts((u64)dc.n_refs * layout->stride)) {
        dc.fetch_ok = 0;
        return;
    }
    WP2_BEGIN(_w38);
    /* rsx_vertex_fetch_all decodes attribute by attribute (one format
     * dispatch per attribute, not per component per vertex); it was the
     * largest share of the draw engine's time. RSX_VFETCH_ONE=1 keeps the
     * per-vertex rsx_vertex_fetch_one path; RSX_VFETCH_CHECK=1 runs both on
     * every draw and reports any byte that differs. */
    static int vf_mode = -1;
    if (vf_mode < 0) vf_mode = getenv("RSX_VFETCH_ONE") ? 1 : getenv("RSX_VFETCH_CHECK") ? 2 : 0;
    if (vf_mode == 1) {
        for (u32 i = 0; i < dc.n_refs; i++) {
            u8* dst = layout->stride ? dc.verts + (u64)i * layout->stride : NULL;
            if (!rsx_vertex_fetch_one(&dc.fetch_plan, &dc.refs[i], dst)) {
                dc.fetch_ok = 0;
                return;
            }
        }
    } else {
        const int ok = rsx_vertex_fetch_all(&dc.fetch_plan, dc.refs, dc.n_refs, dc.verts);
        WP2_END(_w38, 38);
        if (vf_mode == 2) {
            static u8* ref_buf; static u64 ref_cap;
            static unsigned long draws, bad_draws, bad_bytes;
            static int summary_armed = 0;
            if (!summary_armed) { summary_armed = 1; atexit(eng_vfetch_check_summary); }
            s_vfc_draws = &draws; s_vfc_bad = &bad_draws; s_vfc_bytes = &bad_bytes;
            const u64 bytes = (u64)dc.n_refs * layout->stride;
            int ref_ok = 1;
            if (bytes > ref_cap) { free(ref_buf); ref_buf = (u8*)malloc(bytes); ref_cap = ref_buf ? bytes : 0; }
            if (ref_buf || !bytes) {
                for (u32 i = 0; i < dc.n_refs && ref_ok; i++)
                    if (!rsx_vertex_fetch_one(&dc.fetch_plan, &dc.refs[i],
                                              layout->stride ? ref_buf + (u64)i * layout->stride : NULL))
                        ref_ok = 0;
                draws++;
                unsigned long nb = 0;
                if (ok == ref_ok && ok)
                    for (u64 k = 0; k < bytes; k++) nb += dc.verts[k] != ref_buf[k];
                if (ok != ref_ok || nb) {
                    bad_draws++; bad_bytes += nb;
                    if (bad_draws <= 8)
                        fprintf(stderr, "[vfetch-check] f%u MISMATCH: fetch_all ok=%d fetch_one ok=%d, %lu of %llu bytes differ (verts=%u stride=%u)\n",
                                g.frames, ok, ref_ok, nb, (unsigned long long)bytes, dc.n_refs, layout->stride);
                }
                if ((draws % 20000) == 0)
                    fprintf(stderr, "[vfetch-check] %lu draws compared, %lu mismatched (%lu bytes)\n", draws, bad_draws, bad_bytes);
            }
        }
        if (!ok) {
            dc.fetch_ok = 0;
            return;
        }
    }
    dc.n_verts = dc.n_refs;
}

/* ---- sink ---------------------------------------------------------------- */

/* D-side accumulator (RSX_DRAW_SPLIT=2, walkF3): the batches the dispatcher hands over between BEGIN and END. `dc` is the
 * draw side's working set (it also holds the fetch/vcache scratch); sink_end copies dd -> dc (or dd -> a B2 job). */
static struct {
    eng_batch arr[ENG_MAX_BATCHES]; u32 n_arr;
    eng_batch idx[ENG_MAX_BATCHES]; u32 n_idx;
    u32 n_packets;
    const u8* inl; u32 inl_bytes;
} dd;

static void sink_begin(void* user, const rsx_dispatch* r, u32 prim)
{
    (void)user; (void)r; (void)prim;
    dd.n_arr = dd.n_idx = dd.n_packets = 0;
    dd.inl = NULL; dd.inl_bytes = 0;
}

static void sink_draw_arrays(void* user, const rsx_dispatch* r, u32 first, u32 count)
{
    (void)user; (void)r;
    dd.n_packets++;
    if (dd.n_arr >= ENG_MAX_BATCHES) return;
    dd.arr[dd.n_arr].first = first;
    dd.arr[dd.n_arr].count = count;
    dd.n_arr++;
}

/* INLINE_ARRAY delivers one finished stream rather than a (first, count)
 * range; queue it as an ordinary consecutive batch and let the fetch plan
 * read the vertices out of it (rsx_vertex_fetch_plan_set_inline). */
static void sink_inline_array(void* user, const rsx_dispatch* r,
                              const u8* data, u32 bytes)
{
    (void)user;
    const u32 stride = rsx_vertex_inline_layout(r, NULL);
    if (!stride || bytes < stride) return;
    dd.inl = data;
    dd.inl_bytes = bytes;
    dd.n_packets++;
    if (dd.n_arr >= ENG_MAX_BATCHES) return;
    dd.arr[dd.n_arr].first = 0;
    dd.arr[dd.n_arr].count = bytes / stride;
    dd.n_arr++;
}

static void sink_draw_index(void* user, const rsx_dispatch* r, u32 first, u32 count)
{
    (void)user; (void)r;
    dd.n_packets++;
    if (dd.n_idx >= ENG_MAX_BATCHES) return;
    dd.idx[dd.n_idx].first = first;
    dd.idx[dd.n_idx].count = count;
    dd.n_idx++;
}

/* dd (or a job's copy) -> dc, the working set of sink_end_impl. */
static void dc_load(const eng_batch* arr, u32 na, const eng_batch* idx, u32 ni, u32 np, const u8* inl, u32 inl_bytes)
{
    dc_reset();
    dc.n_arr = na; if (na) memcpy(dc.arr, arr, (size_t)na * sizeof(eng_batch));
    dc.n_idx = ni; if (ni) memcpy(dc.idx, idx, (size_t)ni * sizeof(eng_batch));
    dc.n_packets = np;
    dc.inl = inl; dc.inl_bytes = inl_bytes;
}

/* Which texture units this draw binds, and from where: a unit naming a
 * registered colour surface samples that live target rather than guest
 * memory (every post-process, shadow and reflection in a PS3 title is a
 * render-to-texture read back through a texture unit), a unit naming a
 * tracked zeta in DEPTH24_D8 samples its snapshot, and everything else is a
 * guest upload (rsx_live_draw.c:5986-6062). */
static int s_zeta_alias_log;   /* print the next pipeline's fragment constants */
static void eng_surface_dump_now(void);   /* every registered surface, now */
static const char* s_dump_tag = "";        /* file-name suffix for the dumps above */
/* RSX_SNAP_STATS=1: every 300 frames, how many draws sampled the target they
 * write (each a full copy of it), how many texture-cache invalidations
 * (NV4097 0x1FD8) the title sent, and how many of those copies came with no
 * invalidation since the previous copy of the same target. */
static int s_snap_stats = -1;
static unsigned long s_snap_n, s_snap_noinval, s_inval_n[4], s_snap_unused;
static u32 s_snap_last_surface = ~0u; static unsigned long s_inval_at_last_snap = ~0ul, s_inval_total;
static void snap_stats_frame(void)
{
    static u32 last;
    if (g.frames - last < 300) return;
    last = g.frames;
    fprintf(stderr, "[snap-stats] 300 frames: %lu own-target copies (%lu with no texture-cache invalidation since the last), "
            "%lu own-target units the program does not sample (no copy); invalidations arg1 %lu arg2 %lu arg3 %lu other %lu\n",
            s_snap_n, s_snap_noinval, s_snap_unused, s_inval_n[1], s_inval_n[2], s_inval_n[3], s_inval_n[0]);
    s_snap_n = s_snap_noinval = s_snap_unused = 0; s_inval_n[0] = s_inval_n[1] = s_inval_n[2] = s_inval_n[3] = 0;
}
static u32 eng_surf_first(u32 loc, u32 off)
{
    const u32 h = ((loc * 0x9E3779B1u) ^ (off * 0x85EBCA6Bu)) >> 23;   /* 9 bits */
    if (s_sh[h].gen == s_surf_gen && s_sh[h].loc == loc && s_sh[h].off == off) return s_sh[h].idx;
    u32 i = 0;
    for (; i < g.n_surfaces; i++)
        if (g.surfaces[i].handle && g.surfaces[i].location == loc &&
            (g.surfaces[i].offset == off || (g.surfaces[i].alias_off && g.surfaces[i].alias_off == off))) break;
    /* 30/10: the alias test too (clip-origin textures, R1 slow-time / bloom), as in the scan of
     * sink_bind_textures: without it this memo, on by default since 28/10, skipped the target and the
     * R1 effect went back to sampling guest memory, black. */
    s_sh[h].loc = loc; s_sh[h].off = off; s_sh[h].gen = s_surf_gen; s_sh[h].idx = i;
    return i;
}

static u32 sink_bind_textures(const u32* target_slots, u32 n_targets,
                              u32 current_zslot,
                              u32 textures[RSX_BE_MAX_TEXTURES],
                              rsx_be_sampler_desc samplers[RSX_BE_MAX_TEXTURES])
{
    u32 mask = 0;
    const u32 reads = eng_fp_reads();
    s_tdec_pass = 4;
    static rsx_dsp_texture s_pt[RSX_BE_MAX_TEXTURES]; static u32 s_pon; u32 cur_on = 0, same_u = 0;   /* prof102 event counts */
    for (u32 u = 0; u < RSX_BE_MAX_TEXTURES; u++) {
        if (!eng_tex_unit_on(u, reads)) continue;   /* off, or not sampled */
        rsx_dsp_texture t;
        RSX_B2P_ENTER(B2P_TEXDEC);
        eng_get_tex(RV, u, &t);
        if (!t.enabled) { RSX_B2P_LEAVE(); continue; }
        mask |= 1u << u; RSX_B2P_ADD(B2C_TEXUNIT, 1);
        if (s_wp_on > 1) { WP2_BEGIN(_w47); cur_on |= 1u << u; s_wp_n2[17]++; if (((s_pon >> u) & 1u) && !memcmp(&s_pt[u], &t, sizeof t)) { same_u |= 1u << u; s_wp_n2[18]++; } s_pt[u] = t; WP2_END(_w47, 47); }
        { WP2_BEGIN(_w46); eng_decode_sampler(t.filter, t.wrap, t.control0, t.border_color, &samplers[u]); WP2_END(_w46, 46); }
        RSX_B2P_LEAVE();   /* B2P_TEXDEC, entered before eng_get_tex */

        int sampled = -1, unused_own = 0;
        /* A depth format is the zeta, never a colour target at the same
         * address (ToD's shadow casters bind a never-written R5G6B5 target
         * there with colour writes off): the zeta path below binds it. */
        const u32 dfmt = t.format & RSX_TEX_FMT_BASE_MASK & ~(u32)RSX_TEX_FMT_UNNORM;
        const int depth_fmt = dfmt >= 0x90u && dfmt <= 0x93u;
        /* Nor is a DXT format ever a target (the RSX cannot render one): ToD
         * binds a 16x16 DXT1 at local offset 0 on unit 2 of some materials,
         * where the scene target s0 lives -- matched by address alone, the
         * whole previous frame got sampled there, a blurred ghost of the
         * scene. Not proven to matter (22/10: the brown smudge next to Ratchet
         * stayed), so off by default: RSX_DXT_SURFACE=0 skips the match. */
        static int dxt_surf = -1; if (dxt_surf < 0) { const char* e = getenv("RSX_DXT_SURFACE"); dxt_surf = !(e && e[0] == '0'); }
        const int dxt_fmt = !dxt_surf && dfmt >= 0x86u && dfmt <= 0x88u;
        if (dxt_fmt) { static u32 nd; if (nd++ < 4) fprintf(stderr, "[rsx engine] unit %u: DXT fmt 0x%02X at %u:0x%08X %ux%u not matched to a target\n", u, t.format, t.location, t.offset, t.width, t.height); }
        /* RSX_WALKER_FAST: no render target at this address as of the last surface
         * change (the usual case: an ordinary texture) -> skip the scan. */
        if (s_wk_fast < 0) eng_wk_env();
        const int nm_hit = s_wk_fast && s_tex_nomatch[u].gen == s_surf_gen &&
                           s_tex_nomatch[u].loc == t.location && s_tex_nomatch[u].off == t.offset;
        const int scan = !depth_fmt && !dxt_fmt && !nm_hit;
        u32 i = 0;
        if (scan && s_surf_hash_on) i = eng_surf_first(t.location, t.offset);
        for (; scan && i < g.n_surfaces; i++) {
            if (!g.surfaces[i].handle || g.surfaces[i].location != t.location ||
                (g.surfaces[i].offset != t.offset &&
                 !(g.surfaces[i].alias_off && g.surfaces[i].alias_off == t.offset)))
                continue;
            /* Not one this draw writes: reading a target a pass is writing
             * is undefined on every API, and an MRT set writes more than
             * one of them. */
            int own = 0;
            for (u32 k = 0; k < n_targets; k++) if (target_slots[k] == i) own = 1;
            if (own) {
                /* A unit left enabled from an earlier pass that this
                 * program never samples: nothing to copy, and a full copy
                 * of a 4K target is ~0.1 ms of GPU. (Drakengard 3's ~95
                 * own-target copies a frame in its destruction scenes are
                 * not this case -- each of those programs samples the scene
                 * at an offset, heat haze, and needs the copy.)
                 * RSX_SNAP_ALL=1 copies regardless. */
                { static int all = -1; if (all < 0) all = getenv("RSX_SNAP_ALL") ? 1 : 0;
                  const u8* vpu; u32 vpn; const u8* fpu = NULL; u32 fpn = 0;
                  const int got = !all && eng_guest_programs(&vpu, &vpn, &fpu, &fpn);
                  if (got && fpu &&
                      !((rsx_fp_texture_mask(fpu, fpn) >> u) & 1u)) {
                      if (s_snap_stats > 0) s_snap_unused++;
                      unused_own = 1;
                      break;
                  } }
                /* The draw samples the target it writes. Drakengard 3's
                 * post-process runs in place in the back buffer it then
                 * flips; binding the live target is undefined, and the old
                 * fallback -- the guest's stale bytes -- came out white on
                 * every such frame. Sample a copy taken at this point. */
                if (s_snap_stats < 0) s_snap_stats = getenv("RSX_SNAP_STATS") ? 1 : 0;
                if (s_snap_stats) {
                    s_snap_n++;
                    if (s_snap_last_surface == i && s_inval_at_last_snap == s_inval_total) s_snap_noinval++;
                    s_snap_last_surface = i; s_inval_at_last_snap = s_inval_total;
                    snap_stats_frame();
                }
                const u32 snap = g.be->color_snapshot
                    ? g.be->color_snapshot(g.be->user, g.surfaces[i].handle) : 0;
                if (snap) {
                    const u32 view = g.be->surface_view
                        ? g.be->surface_view(g.be->user, snap, t.remap & 0xFFFFu, t.format) : 0;
                    textures[u] = view ? view : snap;
                    { static int _n = 0; if (_n < 4 && ++_n)
                        fprintf(stderr, "[rsx engine] unit %u samples its own target s%u: bound a copy (fp-struct=%016llx)\n", u, i, (unsigned long long)eng_cur_fp_struct()); }
                }
                break;
            }
            sampled = (int)i;
            eng_ssr_snapshot(i);
            if (g_rsx_rt_on) rsx_rt_note_sampled(g.surfaces[i].handle, g.frames);
            g.surfaces[i].last_frame = g.frames;
            break;
        }
        if (s_wk_fast && scan && i == g.n_surfaces) {   /* the scan ran to the end: nothing at this address */
            s_tex_nomatch[u].loc = t.location; s_tex_nomatch[u].off = t.offset; s_tex_nomatch[u].gen = s_surf_gen;
        }
        if (unused_own) continue;   /* bound to nothing: the program never reads it */
        if (textures[u]) continue;
        if (sampled >= 0) {
            const u32 view = g.be->surface_view
                ? g.be->surface_view(g.be->user, g.surfaces[sampled].handle,
                                     t.remap & 0xFFFFu, t.format)
                : 0;
            textures[u] = view ? view : g.surfaces[sampled].handle;
            continue;
        }
        const u32 base_fmt = t.format & RSX_TEX_FMT_BASE_MASK & ~(u32)RSX_TEX_FMT_UNNORM;
        /* A non-depth format over a tracked zeta: the title reads its depth
         * buffer's bytes as colour (UE3 decodes D24 from A8R8G8B8 with a dot
         * product). Report each (offset, format) once. */
        /* walk37: an ordinary texture matches no depth target, and with ~100 of them in a battle the scan below ran per unit per
         * draw: remembered per unit until a depth target is created or moved (RSX_ZNOMATCH=0 = scan always). */
        static int znm = -1; if (znm < 0) { const char* e = getenv("RSX_ZNOMATCH"); znm = !(e && e[0] == '0'); }
        const int z_nm_hit = znm && s_tex_znomatch[u & 15].gen == s_zgen && s_tex_znomatch[u & 15].loc == t.location &&
                             s_tex_znomatch[u & 15].off == t.offset;
        int z_found = 0;
        if (base_fmt != RSX_TEX_FMT_DEPTH24_D8 && base_fmt != 0x92u && !z_nm_hit) {
            for (u32 i = 0; i < g.n_zdepths; i++)
                if (g.zdepths[i].location == t.location && g.zdepths[i].offset == t.offset) {
                    z_found = 1;
                    static u32 seen[32][2]; static u32 nseen = 0; int dup = 0;
                    rsx_dsp_surface dsf; rsx_dsp_get_surface(RV, &dsf);
                    const u32 skey = t.format | (u << 16) | (dsf.clip_w << 20);
                    for (u32 k = 0; k < nseen; k++) if (seen[k][0] == t.offset && seen[k][1] == skey) dup = 1;
                    /* Bind the depth packed as those bytes, through the unit's
                     * own remap, when it is A8R8G8B8. The packed image is a
                     * copy, so this also works for the zeta the draw has bound:
                     * UE3's shadow projection keeps the scene depth attached
                     * for its stencil test while it reads that depth back,
                     * and skipping the zeta-in-use case handed it the guest
                     * bytes (all zero), so it rebuilt every receiver at the
                     * near plane and no character cast a shadow.
                     * RSX_NO_DEPTH_AS_COLOR=1 leaves the guest bytes. */
                    { static int off = -1; if (off < 0) off = getenv("RSX_NO_DEPTH_AS_COLOR") ? 1 : 0;
                      if (!off && base_fmt == 0x85u) {
                          u32 ptw = t.width, pth = t.height;
                          /* 2x/4x horizontal MSAA: the title declares the depth image in
                           * samples (1920 wide) while the scene is clip_w (960) pixels wide
                           * and the program addresses it as pixel*2 / 1920: the copy must be
                           * the pixel-wide region so 0..1 spans the rendered image. */
                          { static int nm = -1; if (nm < 0) nm = getenv("RSX_DEPTH_NO_MSAA_FIX") ? 1 : 0;
                            if (!nm && dsf.clip_w && ptw > dsf.clip_w && (ptw % dsf.clip_w) == 0 && ptw / dsf.clip_w <= 4) ptw = dsf.clip_w;
                            if (!nm && dsf.clip_h && pth > dsf.clip_h && (pth % dsf.clip_h) == 0 && pth / dsf.clip_h <= 4) pth = dsf.clip_h; }
                          const u32 pk = eng_zdepth_packed(i, ptw, pth);
                          if (pk) {
                              const u32 view = g.be->surface_view
                                  ? g.be->surface_view(g.be->user, pk, t.remap & 0xFFFFu, t.format) : 0;
                              textures[u] = view ? view : pk;
                          }
                      } }
                    if (!dup && nseen < 32) { seen[nseen][0] = t.offset; seen[nseen][1] = skey; nseen++;
                        fprintf(stderr, "[rsx engine] unit %u reads zeta %u (loc %u off 0x%08X %ux%u) as format 0x%02X %ux%u remap 0x%04X -- depth as colour (bound clip %ux%u) fp-struct=%016llx\n",
                                u, i, t.location, t.offset, g.zdepths[i].w, g.zdepths[i].h, t.format, t.width, t.height, t.remap & 0xFFFFu, dsf.clip_w, dsf.clip_h,
                                (unsigned long long)eng_cur_fp_struct());
                        s_zeta_alias_log = 1; }
                    break;
                }
            if (!z_found && znm) { s_tex_znomatch[u & 15].loc = t.location; s_tex_znomatch[u & 15].off = t.offset; s_tex_znomatch[u & 15].gen = s_zgen; }
        }
        if (textures[u]) continue;
        /* Every depth format (DEPTH24_D8, its FLOAT, DEPTH16, its FLOAT) reads
         * the rendered zeta: Ratchet ToD's per-object 64x64 shadow maps are
         * DEPTH16, and from the guest bytes (never written by the GPU) every
         * compare came out "in shadow" -- a dark square under each caster. */
        if (base_fmt >= 0x90u && base_fmt <= 0x93u) {
            int found = 0;
            for (u32 i = 0; i < g.n_zdepths; i++)
                if (g.zdepths[i].location == t.location &&
                    g.zdepths[i].offset == t.offset && current_zslot != i) {
                    found = 1;
                    g.zdepths[i].last_frame = g.frames;
                    const u32 snap = eng_zdepth_snapshot(i, t.width, t.height);
                    if (snap) textures[u] = snap;
                    else { static int n = 0; if (n < 16 && ++n)
                        fprintf(stderr, "[rsx engine] depth texture unit %u (loc %u off 0x%08X %ux%u): tracked zeta %u has no snapshot (handle %u had_write %u cleared %d zeta %ux%u frame %u)\n",
                                u, t.location, t.offset, t.width, t.height, i, g.zdepths[i].handle, g.zdepths[i].had_write,
                                g.zdepths[i].cleared, g.zdepths[i].w, g.zdepths[i].h, g.frames); }
                    break;
                }
            if (!found) { static int n = 0; if (n < 6 && ++n) {
                fprintf(stderr, "[rsx engine] depth texture unit %u (loc %u off 0x%08X %ux%u zfunc %u): no tracked zeta matches; zetas:",
                        u, t.location, t.offset, t.width, t.height, (t.wrap >> 28) & 0xFu);
                for (u32 i = 0; i < g.n_zdepths; i++) fprintf(stderr, " %u:%u/0x%08X/%ux%u%s", i, g.zdepths[i].location, g.zdepths[i].offset, g.zdepths[i].w, g.zdepths[i].h, current_zslot == i ? "(bound)" : "");
                fputc('\n', stderr); } }
            if (textures[u]) continue;
        }
        RSX_B2P_ENTER(B2P_TEXSLOT); RSX_B2P_ADD(B2C_TEXSLOT, 1);
        { WP2_BEGIN(_w48);
        textures[u] = eng_texture_slot(t.location, t.offset, t.format,
                                       t.width, t.height, t.mipmaps, t.pitch,
                                       (int)t.cubemap, t.remap & 0xFFFFu);
        WP2_END(_w48, 48); }
        RSX_B2P_LEAVE();
        /* RSX_DROP_STATS=1: a sampled unit left with no texture reads the
         * null texture (transparent black), so a particle or icon drawn with
         * it vanishes without a trace. Count them by format, with an example
         * of each, every 300 frames. */
        if (!textures[u]) {
            static int ds = -1; if (ds < 0) ds = getenv("RSX_DROP_STATS") ? 1 : 0;
            if (ds) {
                static u32 cnt[256], ex_off[256], ex_wh[256], ex_loc[256], last_frame;
                const u32 f8 = t.format & 0xFFu;
                if (!cnt[f8]++) { ex_off[f8] = t.offset; ex_wh[f8] = (t.width << 16) | t.height; ex_loc[f8] = t.location; }
                if (g.frames - last_frame >= 300) {
                    last_frame = g.frames;
                    for (u32 k = 0; k < 256; k++) if (cnt[k]) {
                        fprintf(stderr, "[drop-stats] null texture fmt=0x%02X x%u (e.g. loc %u off 0x%08X %ux%u)\n",
                                k, cnt[k], ex_loc[k], ex_off[k], ex_wh[k] >> 16, ex_wh[k] & 0xFFFFu);
                        cnt[k] = 0;
                    }
                }
            }
        }
    }
    if (s_wp_on > 1) { WP2_BEGIN(_w47); s_wp_n2[9]++; if (cur_on == s_pon && same_u == cur_on && cur_on) s_wp_n2[16]++; s_pon = cur_on; WP2_END(_w47, 47); }
    return mask;
}

static u32 sink_bind_vertex_textures(
    u32 vtex_mask, u32 textures[RSX_BE_MAX_VERTEX_TEXTURES],
    rsx_be_sampler_desc samplers[RSX_BE_MAX_VERTEX_TEXTURES])
{
    u32 bound = 0;
    for (u32 u = 0; u < RSX_BE_MAX_VERTEX_TEXTURES; u++) {
        if (!((vtex_mask >> u) & 1u)) continue;
        rsx_dsp_vertex_texture vt;
        rsx_dsp_get_vertex_texture(RV, u, &vt);
        eng_decode_sampler(vt.filter, vt.wrap, vt.control0, vt.border_color, &samplers[u]);
        /* A vertex unit has no crossbar, so its remap is the identity. */
        textures[u] = eng_texture_slot(vt.location, vt.offset, vt.format,
                                       vt.width, vt.height, vt.mipmaps,
                                       vt.pitch, 0, 0xAAE4u);
        if (textures[u]) bound |= 1u << u;
    }
    return bound;
}

/* RSX_DRAW_STATS=1: every 120 frames, how many draws reached the backend and
 * how many were dropped at each silent return below (a dropped draw is an
 * object missing from the frame, and until now nothing said so).
 * RSX_DRAW_TRACE_FRAME=<n>: every draw of frame n in full -- primitive,
 * counts, the enabled vertex attributes (type/size/stride/location/offset),
 * the index array, textures and the outcome. */
static struct { unsigned long issued, drop_topo, drop_fetch, drop_targets, drop_pipeline, drop_empty;
                unsigned long long refs, verts, vbytes; } s_dstat;
static unsigned s_surf_draws[ENG_MAX_SURFACES];   /* draws per surface since the last present */
/* GPU-skinned colour draws since the last present (bone indices in attribute
 * 7 as unnormalised bytes, colour writes on) and their vertex total: whether a
 * character that vanished from the picture is still being submitted. */
static unsigned s_skin_draws, s_skin_verts;
static int s_dstat_on = -1; static long s_dtrace_frame = -2;
static u32 s_trace_tex[RSX_BE_MAX_TEXTURES];   /* what sink_bind_textures bound for the draw being traced */
static void eng_draw_stats_tick(void)
{
    if (s_dstat_on < 0) s_dstat_on = getenv("RSX_DRAW_STATS") ? 1 : 0;
    if (s_dtrace_frame == -2) { const char* e = getenv("RSX_DRAW_TRACE_FRAME"); s_dtrace_frame = e ? atol(e) : -1; }
}
/* Arm the four-frame draw trace `after` presents from now (a game-side event,
 * e.g. a menu opening, whose frame differs from run to run). */
/* RSX_PICK state (set from the environment, or by rsx_draw_engine_pick_after). */
static int s_pick_x = -2, s_pick_y = 0; static long s_pick_frame = -1;
/* Pick pixel (x, y) of the 1280x720 picture `after` presents from now. */
void rsx_draw_engine_pick_after(int x, int y, long after)
{
    s_pick_x = x; s_pick_y = y; s_pick_frame = (long)g.frames + after;
    fprintf(stderr, "[pick] armed at f%u: pixel %d,%d of frame %ld\n", g.frames, x, y, s_pick_frame);
}
void rsx_draw_engine_trace_after(long after)
{
    eng_draw_stats_tick();
    s_dtrace_frame = (long)g.frames + after;
    fprintf(stderr, "[draw-trace] armed at f%u: tracing frames %ld..%ld\n", g.frames, s_dtrace_frame, s_dtrace_frame + 3);
}
uint32_t g_rsx_engine_frame = 0;   /* the present count, for frame-gated logs elsewhere */
uint32_t g_rsx_engine_hitches = 0; /* presents more than 25 ms after the one before */
/* Set by the host: called with a frame's time when RSX_HITCH_LOG reports it
 * (main.cpp hands it to the sampling profiler's slow-frame report). */
void (*g_rsx_hitch_hook)(double frame_ms) = 0;
double g_rsx_frame_gpu_wait_ms = 0.0;   /* a backend's blocking GPU waits since the last present */
double g_rsx_frame_pso_ms = 0.0;        /* a backend's pipeline states built on the walker since then */
static void eng_draw_stats_report(void)
{
    if (s_dstat_on != 1 || (g.frames % 120u) != 0u) return;
    fprintf(stderr, "[draw-stats] frame %u: issued %lu, dropped topo=%lu fetch=%lu targets=%lu pipeline=%lu empty=%lu; "
            "refs %llu, vertices %llu (%llu KB); vcache hit %llu (%llu by the write-watch) miss %llu uncacheable %llu stored %llu (stale %llu), %u entries %zu MB, check mismatches %llu\n",
            g.frames, s_dstat.issued, s_dstat.drop_topo, s_dstat.drop_fetch, s_dstat.drop_targets,
            s_dstat.drop_pipeline, s_dstat.drop_empty, s_dstat.refs, s_dstat.verts, s_dstat.vbytes >> 10,
            s_vcstat.hit, s_vcstat.hit_watch, s_vcstat.miss, s_vcstat.uncacheable, s_vcstat.stored, s_vcstat.stale, s_vc_count, s_vc_bytes >> 20,
            s_vcstat.check_bad);
    if (s_vcchunk_on > 0) fprintf(stderr, "[m131] vc chunk shortcut: %llu hits (cum.), %llu verified, %llu bad\n", s_vcchunk_hit, s_vcchunk_chk, s_vcchunk_bad);
    memset(&s_vcstat, 0, sizeof s_vcstat);
    memset(&s_dstat, 0, sizeof s_dstat);
}
/* Retroactive trace: with RSX_TRACE_ON_WHITE set, every draw and clear is
 * also formatted into a ring of the last 512 records, and a present that
 * reads back white dumps the ring -- the draws that PRODUCED the white
 * buffer, which a trace armed after the fact can never show. */
#define ENG_RING_N 512
static char s_ring[ENG_RING_N][400];
static u32  s_ring_head = 0, s_ring_used = 0;
static int  s_ring_on = -1;
static void eng_ring_push(const char* line)
{
    size_t n = strlen(line); if (n >= sizeof s_ring[0]) n = sizeof s_ring[0] - 1;
    memcpy(s_ring[s_ring_head], line, n); s_ring[s_ring_head][n] = 0;
    s_ring_head = (s_ring_head + 1) % ENG_RING_N; if (s_ring_used < ENG_RING_N) s_ring_used++;
}
static void eng_ring_dump(const char* why)
{
    fprintf(stderr, "[ring] ---- %s: last %u records ----\n", why, s_ring_used);
    for (u32 i = 0; i < s_ring_used; i++)
        fprintf(stderr, "[ring] %s\n", s_ring[(s_ring_head + ENG_RING_N - s_ring_used + i) % ENG_RING_N]);
    fprintf(stderr, "[ring] ---- end ----\n");
}
static void eng_ring_draw(const char* outcome, u32 prim, int indexed, u32 n_draw, u32 pipeline, u32 tex_mask)
{
    if (s_ring_on < 0) s_ring_on = (getenv("RSX_TRACE_ON_WHITE") || getenv("RSX_TRACE_ON_COLOR")) ? 1 : 0;
    if (!s_ring_on) return;
    rsx_be_render_state trs; rsx_draw_engine_decode_render_state(RV, &trs);
    u32 tt[RSX_BE_MAX_COLOR_TARGETS]; const u32 tn = eng_current_target_set(tt);
    rsx_dsp_surface tsf; rsx_dsp_get_surface(RV, &tsf);
    char line[400]; int o = snprintf(line, sizeof line,
        "f%u %s prim=%u verts=%u idx=%d n=%u pipe=%u tex=0x%X target=s%d(fmt%u) zeta=0x%08X clip=%ux%u blend=%u(%X/%X) alpha=%u(f%X ref=0x%X fmt%u -> %.4g) depth=%u/%u/%X cull=%u mask=0x%X",
        g.frames, outcome, prim, dc.n_verts, indexed, n_draw, pipeline, tex_mask,
        tn ? (int)tt[0] : -1, tn ? (u32)g.surfaces[tt[0]].fmt : 0, tsf.zeta_offset, tsf.clip_w, tsf.clip_h,
        trs.blend_enable, trs.sf_rgb, trs.df_rgb, trs.alpha_test_enable, trs.alpha_func, trs.alpha_ref_raw, trs.alpha_ref_format, (double)rsx_fp_alpha_ref(trs.alpha_ref_raw, trs.alpha_ref_format), trs.depth_test, trs.depth_write, trs.depth_func,
        trs.cull_enable, trs.color_mask);
    if (dc.n_verts && dc.n_verts <= 4 && dc.out_verts && dc.layout.stride && o > 0 && o < (int)sizeof line - 40) {
        const float* v = (const float*)dc.out_verts;
        for (u32 k = 0; k < 8 && k < dc.layout.stride / 4 && o < (int)sizeof line - 12; k++)
            o += snprintf(line + o, sizeof line - o, "%s%.3g", k ? " " : " v0=", v[k]);
    }
    eng_ring_push(line);
}
static void eng_draw_trace(const char* outcome, u32 prim, int indexed, u32 n_draw, u32 pipeline, u32 tex_mask)
{
    eng_ring_draw(outcome, prim, indexed, n_draw, pipeline, tex_mask);
    /* RSX_TRACE_SKINNED_FROM=<frame>: from that frame on, the first 40 draws
     * into a full-size target that carry an unnormalised-byte attribute (a
     * GPU-skinned mesh's bone indices), whatever frame they fall in. */
    int skinned_pick = 0;
    { static long from = -2; static int left = 40;
      if (from == -2) { const char* e = getenv("RSX_TRACE_SKINNED_FROM"); from = e ? atol(e) : -1; }
      if (from >= 0 && (long)g.frames >= from && left > 0) {
          u32 tt[RSX_BE_MAX_COLOR_TARGETS]; const u32 tn = eng_current_target_set(tt);
          if (tn && g.surfaces[tt[0]].w >= 1280) {
              /* Bone indices ride in attribute 7 as unnormalised bytes; a rigid
               * mesh's packed tangents are unnormalised bytes too, in 2 and 5. */
              rsx_dsp_vertex_attr a; rsx_dsp_get_vertex_attr(RV, 7, &a);
              if (a.type == 7) { skinned_pick = 1; left--; } } } }
    /* Four consecutive frames: a title that presents each rendered frame
     * twice has every other frame empty of draws. */
    if (!skinned_pick && (s_dtrace_frame < 0 || (long)g.frames < s_dtrace_frame || (long)g.frames >= s_dtrace_frame + 4)) return;
    { static int nos6 = -1; if (nos6 < 0) nos6 = getenv("RSX_TRACE_ON_HUD") ? 1 : 0;
      if (nos6) { u32 tt6[RSX_BE_MAX_COLOR_TARGETS]; if (eng_current_target_set(tt6) && tt6[0] == 6) return; } }
    rsx_dsp_index_array ia; rsx_dsp_get_index_array(RV, &ia);
    rsx_be_render_state trs; rsx_draw_engine_decode_render_state(RV, &trs);
    u32 ttargets[RSX_BE_MAX_COLOR_TARGETS]; const u32 tnt = eng_current_target_set(ttargets);
    fprintf(stderr, "[draw-trace] f%u %s prim=%u packets=%u refs=%u verts=%u indexed=%d n=%u pipe=%u tex=0x%X idx(loc=%u off=0x%08X u32=%d)"
            " target=s%d(%ux%u fmt%u) blend=%u(%X/%X eq%X) alpha=%u(f%X ref=0x%X fmt%u -> %.4g) depth=%u/%u cull=%u mask=0x%X",
            g.frames, outcome, prim, dc.n_packets, dc.n_source_refs, dc.n_verts, indexed, n_draw, pipeline, tex_mask,
            ia.location, ia.offset, ia.is_u32,
            tnt ? (int)ttargets[0] : -1, tnt ? g.surfaces[ttargets[0]].w : 0, tnt ? g.surfaces[ttargets[0]].h : 0,
            tnt ? (u32)g.surfaces[ttargets[0]].fmt : 0,
            trs.blend_enable, trs.sf_rgb, trs.df_rgb, trs.eq_rgb, trs.alpha_test_enable, trs.alpha_func, trs.alpha_ref_raw, trs.alpha_ref_format, (double)rsx_fp_alpha_ref(trs.alpha_ref_raw, trs.alpha_ref_format), trs.depth_test, trs.depth_write,
            trs.cull_enable, trs.color_mask);
    /* The first decoded vertex of a small draw: a full-screen quad's position
     * and its vertex colour are what decide whether it covers the frame. */
    if (dc.n_verts && dc.n_verts <= 8 && dc.out_verts && dc.layout.stride) {
        /* Every vertex of a small draw, first 8 floats each: a quad's four
         * corners and colours say whether it covers the frame and with what. */
        for (u32 n = 0; n < dc.n_verts; n++) {
            const float* v = (const float*)(dc.out_verts + (size_t)n * dc.layout.stride);
            fprintf(stderr, " v%u=[", n);
            for (u32 k = 0; k < dc.layout.stride / 4 && k < 8; k++) fprintf(stderr, "%s%.3g", k ? " " : "", v[k]);
            fprintf(stderr, "]");
        }
    }
    fprintf(stderr, " attrs:");
    for (u32 i = 0; i < 16; i++) {
        rsx_dsp_vertex_attr a; rsx_dsp_get_vertex_attr(RV, i, &a);
        if (!a.type) continue;
        fprintf(stderr, " %u:t%u/%u/s%u/L%u/0x%08X", i, a.type, a.size, a.stride, a.location, a.offset);
    }
    /* For a quad-sized draw (a movie, a post-process or UI pass), each bound
     * texture: location, offset, format, size, and the mean of its first
     * row -- whether the planes a video quad samples hold any picture. */
    static int tex_all = -1; if (tex_all < 0) tex_all = getenv("RSX_TRACE_TEXUNITS") ? 1 : 0;
    if (dc.n_verts && (dc.n_verts <= 8 || tex_all)) {
        fprintf(stderr, " texunits:");
        for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++) {
            if (!(tex_mask & (1u << u))) continue;
            rsx_dsp_texture t; eng_get_tex(RV, u, &t);
            const u32 row = t.pitch ? t.pitch : t.width;
            const u8* p = (t.width && row) ? eng_guest_ptr(NULL, t.location, t.offset, row) : NULL;
            unsigned long sum = 0; for (u32 k = 0; p && k < row && k < 4096; k++) sum += p[k];
            int surf = -1, zeta = -1;
            for (u32 k = 0; k < g.n_surfaces; k++) if (g.surfaces[k].handle && g.surfaces[k].location == t.location && g.surfaces[k].offset == t.offset) surf = (int)k;
            for (u32 k = 0; k < g.n_zdepths; k++) if (g.zdepths[k].handle && g.zdepths[k].location == t.location && g.zdepths[k].offset == t.offset) zeta = (int)k;
            if (surf >= 0) fprintf(stderr, " [surface s%d fmt%u]", surf, (u32)g.surfaces[surf].fmt);
            if (zeta >= 0) fprintf(stderr, " [zeta z%d]", zeta);
            fprintf(stderr, " %u:L%u/0x%08X/f%02X/%ux%u/p%u/mips%u/filt%08X/ctl%08X/remap%04X/addr%08X/ea%08X/m%lu", u, t.location, t.offset, t.format, t.width, t.height, t.pitch, t.mipmaps, t.filter, t.control0, t.remap & 0xFFFFu, t.wrap,
                    cellGcmResolveLocated(t.location == RSX_LOCATION_LOCAL, t.offset),
                    p ? sum / (row < 4096 ? row : 4096) : 9999ul);
        }
    }
    fputc('\n', stderr);
    /* What each active unit really got: a registered colour surface (s), a
     * tracked zeta (z), or a texture from guest memory (cache); and the
     * fragment program's hash. */
    { u64 fh = 0;
      for (u32 i = 0; i < g.n_pipelines; i++) if (g.pipelines[i].handle == pipeline) { fh = g.pipelines[i].fp_hlsl; break; }
      fprintf(stderr, "[draw-trace]   fp-hash=%016llx\n", (unsigned long long)fh);
      for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES && u < RSX_BE_MAX_TEXTURES; u++) {
          if (!(tex_mask & (1u << u))) continue;
          rsx_dsp_texture t; eng_get_tex(RV, u, &t);
          int surf = -1, zeta = -1;
          for (u32 k = 0; k < g.n_surfaces; k++) if (g.surfaces[k].handle && g.surfaces[k].location == t.location && g.surfaces[k].offset == t.offset) { surf = (int)k; break; }
          for (u32 k = 0; k < g.n_zdepths; k++) if (g.zdepths[k].handle && g.zdepths[k].location == t.location && g.zdepths[k].offset == t.offset) { zeta = (int)k; break; }
          char what[96];
          if (surf >= 0) snprintf(what, sizeof what, "s%d(%ux%u fmt%u%s)%s", surf, g.surfaces[surf].w, g.surfaces[surf].h,
                                  (u32)g.surfaces[surf].fmt, g.surfaces[surf].blit_only ? " blit-only" : "",
                                  s_trace_tex[u] == g.surfaces[surf].handle ? "" : " [handle is not the target's: view/copy/cache]");
          else if (zeta >= 0) snprintf(what, sizeof what, "z%d(%ux%u)", zeta, g.zdepths[zeta].w, g.zdepths[zeta].h);
          else snprintf(what, sizeof what, "cache(guest memory)");
          fprintf(stderr, "[draw-trace]   tex unit %u: %u:0x%08X %ux%u fmt %02X pitch %u -> bound %s (handle %u)%s\n",
                  u, t.location, t.offset, t.width, t.height, t.format, t.pitch, what, s_trace_tex[u],
                  s_trace_tex[u] ? "" : " NOTHING");
      } }
    /* A GPU-skinned draw (an unnormalised-byte attribute carries its bone
     * indices): the constants Drakengard 3's skinning program reads -- bone
     * rows from c207, the unpack factors in c463, position scale/offset in
     * c465/c466 -- as the dispatcher holds them for this draw. */
    { int skinned = 0;
      for (u32 i = 0; i < 16 && !skinned; i++) { rsx_dsp_vertex_attr a; rsx_dsp_get_vertex_attr(RV, i, &a); if (a.type == 7) skinned = 1; }
      if (skinned) {
          static const u32 slots[] = { 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 429, 430, 431, 432, 462, 463, 464, 465, 466, 467 };
          { const u8* vu = NULL; const u8* fu = NULL; u32 vi2 = 0, fs2 = 0;
            u64 vf = 1469598103934665603ull;
            if (eng_guest_programs(&vu, &vi2, &fu, &fs2))
                for (u32 i = 0; i < vi2 * 16u; i++) vf = (vf ^ vu[i]) * 1099511628211ull;
            u64 pf = 0, ph2 = 0;
            for (u32 i = 0; i < g.n_pipelines; i++) if (g.pipelines[i].handle == pipeline) { pf = g.pipelines[i].vp_fnv; ph2 = g.pipelines[i].vp_hlsl; break; }
            fprintf(stderr, "[draw-trace]   vp resident=%016llx (start=%u instrs=%u) pipeline's=%016llx %s vp-hlsl=%016llx\n",
                    (unsigned long long)vf, rsx_dsp_vp_start(RV), vi2, (unsigned long long)pf, vf == pf ? "same" : "DIFFERENT",
                    (unsigned long long)ph2); }
          fprintf(stderr, "[draw-trace]   constants:");
          for (u32 k = 0; k < sizeof slots / sizeof slots[0]; k++) {
              const float* c = (const float*)RV->constants[slots[k]];
              fprintf(stderr, " c%u=(%.3g %.3g %.3g %.3g)", slots[k], c[0], c[1], c[2], c[3]); }
          fputc('\n', stderr); } }
    /* RSX_TRACE_VPC=<lo>,<hi>: the vertex-program constants lo..hi of every traced draw. */
    { static int lo = -2, hi = -1;
      if (lo == -2) { const char* e = getenv("RSX_TRACE_VPC"); lo = -1; if (e && sscanf(e, "%d,%d", &lo, &hi) != 2) lo = -1; }
      if (lo >= 0) {
          fprintf(stderr, "[draw-trace]   vpc:");
          for (int k = lo; k <= hi && k < RSX_DSP_NUM_CONSTANTS; k++) {
              const float* c = (const float*)RV->constants[k];
              fprintf(stderr, " c%d=(%.4g %.4g %.4g %.4g)", k, c[0], c[1], c[2], c[3]); }
          fputc('\n', stderr); } }
}
typedef struct { u32 hit; u64 key, egen; } EngPre;   /* a prep worker's result for one draw (RSX_DRAW_WORKERS) */
static const EngPre* s_cur_pre;   /* the job B2 is committing, or NULL */
static void sink_end_impl(void* user, const rsx_dispatch* r);
static int eng_b2_on(void);
static void eng_b2_enqueue(void);
static int eng_b2_post_query(u32 m, u32 arg);

/* Timed, for the slow-frame report: how much of a frame the walker spent
 * issuing draws (conversion, hashing, cache, records). */
static void sink_end(void* user, const rsx_dispatch* r)
{
    if (s_wk_fast < 0) eng_wk_env();
    if (eng_b2_on()) { eng_b2_enqueue(); return; }   /* RSX_DRAW_SPLIT=2: the draw runs on thread B2 */
    const double t0 = s_wk_fast ? 0.0 : eng_now_ms();   /* RSX_WALKER_FAST: two clock reads less per draw */
    static unsigned hp_k_; const unsigned long long hp_c0 = (hp_on() && ((++hp_k_) & 7u) == 0u) ? wp_tsc() : 0ull;   /* reg60: 1 draw in 8 is timed (the rest costs nothing) */   /* surf54: the draws' time of the frame, for the [hitch] line */
    s_progs.valid = 0;
    s_progs.tex_valid = 0;
    s_draw_open = 1;
    s_draw_vtx_gen = g_rsx_dsp_vtx_gen;
    s_vp_bb_draw = s_vp_branch_bits_reg;
    dc_load(dd.arr, dd.n_arr, dd.idx, dd.n_idx, dd.n_packets, dd.inl, dd.inl_bytes);
    const int sm = snap_mode();
    if (!sm) sink_end_impl(user, r);   /* default: the unchanged path (RV == &g.rsx) */
    else {
        /* RSX_DRAW_SPLIT=1 / RSX_SNAP_CHECK=1 / RSX_DSP_READ_LOG=1 (walkF2): the D side fills a DrawSnap from the live
         * register file, the draw side (sink_end_impl, still on THIS thread) reads the snapshot view through RV. */
        rsx_dispatch* view = &g.rsx;
        DrawSnap* sn = NULL;
        if (sm & 3) { sn = snap_fill(&g.rsx); view = snap_view(); snap_apply(sn, view); }
        snap_trace_begin(sn, view, &g.rsx);
        g.rv = view;
        sink_end_impl(user, r);
        g.rv = &g.rsx;
        snap_trace_end(view, &g.rsx, g.frames);
    }
    s_draw_open = 0;
    s_fstat.draws++; if (!s_wk_fast) s_fstat.draw_ms += eng_now_ms() - t0;
    if (hp_c0) { s_hp.draw_cyc += (wp_tsc() - hp_c0) * 8ull; s_hp.draws += 8; }
    s_frame_draws++;
}

/* 2^(ex - 15) for a half float's exponent field 1..30, as the float with that
 * exponent: no libm, which the standalone draw-engine test is linked without
 * on Linux, and exact where pow(2.0, n) is only promised to be close. */
static inline float eng_half_pow2(unsigned ex)
{
    union { u32 u; float f; } p;
    p.u = (ex + 112u) << 23;
    return p.f;
}

static void sink_end_impl(void* user, const rsx_dispatch* r)
{
    (void)user; (void)r;
    if (!g.ready || !dc.n_packets) return;
    RSX_B2P_SET(B2P_SETUP);
    eng_draw_stats_tick();
    if (s_wp_on < 0) eng_wk_env();
    unsigned long long wp_last = s_wp_on > 0 ? wp_tsc() : 0;
#define WP_MARK(k) do { if (s_wp_on > 0) { const unsigned long long _t = wp_tsc(); s_wp_cyc[k] += _t - wp_last; wp_last = _t; } } while (0)

    const u32 prim = RV->current_primitive;
    /* Everything that becomes triangles -- lists, strips, fans, quads, quad
     * strips and polygons -- is REBUILT into one triangle list through an
     * index buffer. A host strip topology cannot express a restart cut, which
     * is the whole reason the reference engine rebuilds them, and rebuilding
     * is also what lets repeated references share an uploaded vertex. Points
     * and lines pass through as the guest issued them, and anything left that
     * would need an expansion this engine does not have is dropped rather
     * than drawn as something else. */
    rsx_topology topology = RSX_TOPOLOGY_TRIANGLES;
    int scratch = 0;
    const int rebuild = eng_topology_rebuild(prim, 0, &scratch) != 0;
    if (!rebuild) {
        if (rsx_primitive_needs_expansion(prim)) { s_dstat.drop_topo++; eng_draw_trace("DROP-topology", prim, 0, 0, 0, 0); return; }
        topology = rsx_primitive_topology(prim);
        if (topology == RSX_TOPOLOGY_UNSUPPORTED) { s_dstat.drop_topo++; eng_draw_trace("DROP-topology", prim, 0, 0, 0, 0); return; }
    }
    /* RSX_DROP_STATS=1: POINTS draws, and how many with point sprites on
     * (SET_POINT_SPRITE_CONTROL 0x1EE8 bit 0) -- the host draws every point
     * as one pixel, so a sprite particle all but vanishes. */
    if (topology == RSX_TOPOLOGY_POINTS) {
        static int ds = -1; if (ds < 0) ds = getenv("RSX_DROP_STATS") ? 1 : 0;
        if (ds) {
            static u32 n_pts, n_spr, last_frame, last_size;
            n_pts++;
            if (rsx_dsp_reg(RV, 0x1EE8) & 1u) { n_spr++; last_size = rsx_dsp_reg(RV, 0x1EE0); }
            if (g.frames - last_frame >= 300) {
                last_frame = g.frames;
                fprintf(stderr, "[drop-stats] POINTS draws: %u, with point sprites: %u (last point size 0x%08X)\n",
                        n_pts, n_spr, last_size);
                n_pts = n_spr = 0;
            }
        }
    }

    rsx_vertex_layout_plan layout;
    { WP2_BEGIN(_w8); eng_vertex_layout(&layout); WP2_END(_w8, 8); }

    /* A cache hit supplies the converted vertices and the rebuilt index list;
     * a miss converts and, once the indices are written, stores both. Under
     * RSX_VCACHE_CHECK a hit is converted anyway and compared at that point. */
    RSX_B2P_SET(B2P_VCKEY);
    s_fp_valid = 0;
    u64 vc_key = 0;
    EngVCEntry* vce = NULL;
    if (s_cur_pre && s_cur_pre->hit && !eng_vc_checking()) {
        /* A prep worker validated this entry (lookup + content check); use it if it is still the one it saw. */
        EngVCEntry* pe = s_vc ? eng_vc_slot(s_cur_pre->key) : NULL;
        if (pe && pe->key == s_cur_pre->key && pe->gen == s_cur_pre->egen) {
            vc_key = s_cur_pre->key; vce = pe; pe->last_frame = g.frames;
        }
    }
    if (!vce) {
        WP2_BEGIN(_w9);
        vc_key = eng_vc_key(&layout, prim, rebuild);
        WP2_END(_w9, 9);
        WP2_BEGIN(_w10);
        vce = vc_key ? eng_vc_lookup(vc_key) : NULL;
        WP2_END(_w10, 10);
    }
    EngVCEntry* vc_check = NULL;
    if (vce && eng_vc_checking()) { vc_check = vce; vce = NULL; }
    EngVCFill vcf;
    vcf.ok = vc_key != 0;
    vcf.content = vc_key;
    vcf.n_spans = 0;
    const u8* draw_verts = NULL;
    const u32* draw_indices = NULL;
    int indexed = 0;
    u32 n_draw = 0;
    if (vce) {
        s_vcstat.hit++; RSX_B2P_ADD(B2C_VCHIT, 1);
        if (!vce->buf && !eng_slow()) {
            const size_t pm = eng_vc_promote_min();
            if (pm && vce->bytes >= pm) eng_vc_promote(vce);
        }
        dc.fetch_ok = 1;
        dc.n_verts = vce->n_verts;
        dc.n_source_refs = vce->n_source_refs;
        indexed = vce->indexed;
        n_draw = vce->n_draw;
        draw_verts = vce->verts;
        draw_indices = vce->indices;
        dc.out_verts = draw_verts;
        dc.layout = layout;
    } else {
        if (vc_key) s_vcstat.miss++; else s_vcstat.uncacheable++;
        RSX_B2P_ADD(B2C_VCMISS, 1);
        /* RSX_VCACHE_LOG_MISS=<frame>: every miss of that frame. */
        { static long lf = -2; if (lf == -2) { const char* e = getenv("RSX_VCACHE_LOG_MISS"); lf = e ? atol(e) : -1; }
          if (lf >= 0 && (long)g.frames == lf) {
              rsx_dsp_index_array ia; rsx_dsp_get_index_array(RV, &ia);
              char buf[512]; int o = snprintf(buf, sizeof buf, "[vc-miss] prim %u arr %u idx %u", prim, dc.n_arr, dc.n_idx);
              if (dc.n_idx) o += snprintf(buf + o, sizeof buf - o, " ia %u:%08X first %u count %u base %u", ia.location, ia.offset, dc.idx[0].first, dc.idx[0].count, rsx_dsp_vertex_data_base_index(RV));
              if (dc.n_arr) o += snprintf(buf + o, sizeof buf - o, " arr first %u count %u", dc.arr[0].first, dc.arr[0].count);
              o += snprintf(buf + o, sizeof buf - o, " mask %04X baseoff %08X", layout.mask, rsx_dsp_vertex_data_base_offset(RV));
              for (u32 a = 0; a < 16 && o < (int)sizeof buf - 40; a++) if ((layout.mask >> a) & 1u) {
                  rsx_dsp_vertex_attr va; rsx_dsp_get_vertex_attr(RV, a, &va);
                  o += snprintf(buf + o, sizeof buf - o, " a%u=%u:%08X/%u/t%u", a, va.location, va.offset, va.stride, va.type); }
              fprintf(stderr, "%s\n", buf); } }
        RSX_B2P_SET(B2P_VCFETCH);
        const unsigned long long _fa = s_wp_on > 0 ? wp_tsc() : 0;
        dc_fetch(&layout, rebuild, vc_key ? &vcf : NULL);
        if (s_wp_on > 0) { s_wp_cyc[7] += wp_tsc() - _fa; s_wp_nmiss++; }
        draw_verts = dc.verts;
        dc.out_verts = draw_verts;
    }
    if (s_dstat_on == 1 && dc.fetch_ok) {
        s_dstat.refs += dc.n_source_refs; s_dstat.verts += dc.n_verts;
        s_dstat.vbytes += (unsigned long long)dc.n_verts * layout.stride;
    }
    if (!dc.n_verts || !dc.fetch_ok) { s_dstat.drop_fetch++; eng_draw_trace(dc.fetch_ok ? "DROP-noverts" : "DROP-fetch", prim, 0, 0, 0, 0); return; }

    if (!vce) {
        n_draw = dc.n_source_refs;
        if (rebuild) {
            if (!eng_topology_rebuild(prim, dc.refs_remapped, &indexed)) { s_dstat.drop_topo++; eng_draw_trace("DROP-rebuild", prim, 0, 0, 0, 0); return; }
            n_draw = indexed
                ? rsx_draw_engine_topology_index_count(prim, dc.n_source_refs,
                                                       dc.cuts, dc.n_cuts)
                : dc.n_source_refs - dc.n_source_refs % 3u;
        }
    }
    if (!n_draw) { s_dstat.drop_empty++; eng_draw_trace("DROP-empty", prim, indexed, 0, 0, 0); return; }
    if (g.q_cur) g.q_attempts++;
    WP_MARK(0);   /* layout, vertex-cache key/lookup or conversion */

    RSX_B2P_SET(B2P_RSTATE);
    rsx_be_render_state rs;
    WP2_BEGIN(_w13);
    rsx_draw_engine_decode_render_state(RV, &rs);
    WP2_END(_w13, 13);
    RSX_B2P_SET(B2P_TARGETS);
    /* RSX_SKIP_BLEND=<sf>/<df> (hex blend factors): drop every blended draw
     * using that pair -- an experiment switch to tell whether one blend mode's
     * draws are what wrecks a frame. */
    { static long sk_sf = -2, sk_df = 0;
      if (sk_sf == -2) { const char* e = getenv("RSX_SKIP_BLEND"); sk_sf = -1;
          if (e) { char* d; sk_sf = strtol(e, &d, 16); if (*d == '/') sk_df = strtol(d + 1, 0, 16); } }
      if (sk_sf >= 0 && rs.blend_enable && (long)rs.sf_rgb == sk_sf && (long)rs.df_rgb == sk_df) {
          static unsigned long nskip = 0; if (++nskip <= 4 || (nskip % 1000) == 0)
              fprintf(stderr, "[rsx engine] RSX_SKIP_BLEND: dropped draw #%lu (blend %lX/%lX)\n", nskip, sk_sf, sk_df);
          return; } }

    u32 targets[RSX_BE_MAX_COLOR_TARGETS];
    WP2_BEGIN(_w14);
    const u32 n_targets = eng_current_target_set(targets);
    WP2_END(_w14, 14);
    if (!n_targets) { s_dstat.drop_targets++; eng_draw_trace("DROP-targets", prim, indexed, n_draw, 0, 0); return; }
    const u32 target = targets[0];

    rsx_dsp_surface sf;
    rsx_dsp_viewport vp;
    rsx_dsp_get_surface(RV, &sf);
    rsx_dsp_get_viewport(RV, &vp);
    WP2_BEGIN(_w52);
    const u32 zslot = eng_zdepth_get(sf.zeta_location, sf.zeta_offset,
                                     sf.clip_w, sf.clip_h);
    WP2_END(_w52, 52);
    if (s_wp_on > 1) { WP2_BEGIN(_w47); static u32 pt[RSX_BE_MAX_COLOR_TARGETS], pn, pz = ~0u; s_wp_n2[31]++; if (pn == n_targets && pz == zslot && !memcmp(pt, targets, n_targets * sizeof *targets)) s_wp_n2[30]++; pn = n_targets; pz = zslot; memcpy(pt, targets, n_targets * sizeof *targets); WP2_END(_w47, 47); }
    /* shad114 RSX_DBIAS_EPILOGUE: slope = raw game slope x the target's pixel ratio, quantised; constant bias moves to the VS epilogue
     * (below). Done HERE (not in the render-state decode) because the pass is known now (eng_current_target_set -> eng_note_pass ran). */
    const int dbe = rsx_dbias_epilogue_on() && rs.poly_offset_fill;
    if (dbe) {
        u32 scr = rsx_dsp_reg(RV, 0x0A78); float fs; memcpy(&fs, &scr, 4);
        if (fs == fs && fs > -1e6f && fs < 1e6f && fs != 0.0f) {
            if (g_rsx_poly_target_scale_fn) fs *= g_rsx_poly_target_scale_fn();
            const float a = fabsf(fs);
            if (a > 1e-12f) { const float q = exp2f(floorf(log2f(a) * 8.0f + 0.5f) / 8.0f); fs = fs < 0.0f ? -q : q; }
        }
        memcpy(&scr, &fs, 4);
        rs.poly_offset_scale = scr;
        rs.poly_offset_bias = 0;
    }

    /* Textures before any pipeline or target binding: a cache refresh can
     * replace a resource, and the backend is free to submit while doing it. */
    RSX_B2P_SET(B2P_TEXBIND);
    u32 textures[RSX_BE_MAX_TEXTURES];
    rsx_be_sampler_desc samplers[RSX_BE_MAX_TEXTURES];
    u32 vtextures[RSX_BE_MAX_VERTEX_TEXTURES];
    rsx_be_sampler_desc vsamplers[RSX_BE_MAX_VERTEX_TEXTURES];
    memset(textures, 0, sizeof textures);
    memset(samplers, 0, sizeof samplers);
    memset(vtextures, 0, sizeof vtextures);
    memset(vsamplers, 0, sizeof vsamplers);
    WP2_BEGIN(_w15);
    u32 tex_mask = sink_bind_textures(targets, n_targets, zslot,
                                      textures, samplers);
    WP2_END(_w15, 15);
    /* RSX_SSR: a draw with a cube map on unit 7 gets the last frame's scene colour and depth on two free units. */
    s_ssr_cu = s_ssr_du = -1;
    eng_ssr_frame_mode();
    if (eng_ssr_on() && ((tex_mask >> 7) & 1u) && eng_cube_mask() != 0 && zslot < g.n_zdepths && n_targets == 1) {
        s_ssr_tgt = target; s_ssr_zs = zslot; s_ssr_refl_frame = g.frames;
        memcpy(s_ssr_cam_cur, RV->constants[8], sizeof s_ssr_cam_cur);
        if (s_ssr_col && s_ssr_dep && s_ssr_act) {
            int cu = -1, du = -1;
            for (int u = 15; u >= 0 && du < 0; u--) {
                if (u == 7 || ((tex_mask >> u) & 1u)) continue;
                if (cu < 0) cu = u; else du = u;
            }
            if (cu >= 0 && du >= 0) {
                memset(&samplers[cu], 0, sizeof samplers[cu]);
                samplers[cu].min_linear = samplers[cu].mag_linear = 1;
                samplers[cu].wrap_s = samplers[cu].wrap_t = samplers[cu].wrap_r = 3;   /* guest code 3 = clamp to edge */
                samplers[du] = samplers[cu];
                textures[cu] = s_ssr_col; textures[du] = s_ssr_dep;
                tex_mask |= (1u << cu) | (1u << du);
                s_ssr_cu = cu; s_ssr_du = du; s_ssr_stat_bound++;
            }
        }
    }
    WP2_BEGIN(_w16);
    RSX_B2P_SET(B2P_VTEX);
    const u32 vtex_mask = sink_bind_vertex_textures(eng_vtex_mask(), vtextures,
                                                    vsamplers);
    WP2_END(_w16, 16);
    if (s_dtrace_frame >= 0) memcpy(s_trace_tex, textures, sizeof s_trace_tex);   /* for eng_draw_trace */
    WP_MARK(1);   /* render state, targets, zeta, texture binding */
    RSX_B2P_SET(B2P_PIPE);

    /* A guest program pair that will not translate has no fallback: the draw
     * is dropped, as the reference engine drops it. Substituting the built-in
     * program would draw the geometry in the wrong colours, which is harder to
     * see than a missing object and hides the translation failure. */
    int pipeline_is_fixed = 1;
    /* The pipeline's colour attachment format must be the format of the
     * texture that is really bound, which is the one the surface was
     * registered with. A pass that declares another format for the same
     * offset -- Drakengard 3's depth pre-pass binds a display buffer -- would
     * otherwise get a pipeline whose attachment format disagrees with the
     * attachment, and what Metal then renders is undefined. */
    rsx_be_format rt_fmt = eng_surface_format(sf.color_format);
    if (g.surfaces[target].fmt != RSX_BE_FMT_NONE && g.surfaces[target].fmt != rt_fmt) {
        static unsigned long n_mismatch = 0;
        if (++n_mismatch <= 12 || (n_mismatch % 5000) == 0)
            fprintf(stderr, "[rsx engine] surface s%u (off=0x%08X) declared fmt %u but registered as %u -- using the registered one (#%lu, mask=0x%X)\n",
                    target, g.surfaces[target].offset, (unsigned)rt_fmt, (unsigned)g.surfaces[target].fmt,
                    n_mismatch, rs.color_mask);
        rt_fmt = g.surfaces[target].fmt;
    }
    const u32 pipeline = eng_pipeline_get(&layout, &rs, rt_fmt,
                                          n_targets, &pipeline_is_fixed);
    if (!pipeline) { s_dstat.drop_pipeline++; eng_draw_trace("DROP-pipeline", prim, indexed, n_draw, 0, tex_mask); return; }
    WP_MARK(2);   /* pipeline lookup */
    RSX_B2P_SET(B2P_IDX);
    /* RSX_SURF_DUMP_AT_FP=<hash>: the first time (from RSX_SURF_DUMP_FROM) a
     * draw with that fragment program is about to run, finish the GPU work
     * recorded so far and dump every surface -- what the draw will read. */
    /* RSX_SURF_DUMP_AT_FP_EVERY=1: once per frame instead of once per run. */
    { static int armed = -1, every = 0; static unsigned long long want = 0; static u32 last_frame = ~0u;
      if (armed < 0) { const char* e = getenv("RSX_SURF_DUMP_AT_FP"); want = e ? strtoull(e, 0, 16) : 0; armed = want ? 1 : 0;
                       every = getenv("RSX_SURF_DUMP_AT_FP_EVERY") ? 1 : 0; }
      if (armed == 1 && (!every || last_frame != g.frames)) {
          const char* e = getenv("RSX_SURF_DUMP_FROM"); const long from = e ? atol(e) : 0;
          u64 fh = 0;
          for (u32 i = 0; i < g.n_pipelines; i++) if (g.pipelines[i].handle == pipeline) { fh = g.pipelines[i].fp_hlsl; break; }
          if (fh == want && (long)g.frames >= from) {
              if (every) last_frame = g.frames; else armed = 0;
              if (g.be->submit_and_wait) g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_GUEST_REFERENCE);
              fprintf(stderr, "[surf-dump] before the first draw with fp-hlsl %016llx in frame %u\n", want, g.frames);
              s_dump_tag = "_pre"; eng_surface_dump_now(); s_dump_tag = "";
          } } }
    /* RSX_SKIP_FP=<hash>[,<hash>...]: drop every draw whose fragment program
     * translates to that HLSL hash (the fp-hlsl the pipeline log prints) --
     * to find which pass produces an artefact. */
    { static int n = -1; static unsigned long long want[16];
      if (n < 0) { n = 0; const char* e = getenv("RSX_SKIP_FP");
          while (e && *e && n < 16) { char* d; want[n++] = strtoull(e, &d, 16); e = (*d == ',') ? d + 1 : NULL; } }
      if (n) { u64 fh = 0;
          for (u32 i = 0; i < g.n_pipelines; i++) if (g.pipelines[i].handle == pipeline) { fh = g.pipelines[i].fp_hlsl; break; }
          for (int k = 0; k < n; k++) if (fh && fh == want[k]) return; } }

    WP2_BEGIN(_w56);
    if (indexed && !vce) {
        if (!dc_reserve_indices(n_draw)) return;
        rsx_draw_engine_write_topology_indices(
            prim, dc.n_source_refs, dc.cuts, dc.n_cuts,
            dc.refs_remapped ? dc.ref_remap.occurrence_to_unique : NULL,
            g.indices);
        draw_indices = g.indices;
    }
    WP2_END(_w56, 56);
    if (vc_check) {
        const int same = vc_check->n_verts == dc.n_verts && vc_check->stride == layout.stride &&
            vc_check->indexed == indexed && vc_check->n_draw == n_draw &&
            !memcmp(vc_check->verts, dc.verts, (size_t)dc.n_verts * layout.stride) &&
            (!indexed || !memcmp(vc_check->indices, g.indices, (size_t)n_draw * sizeof(u32)));
        s_vc_checked++;
        if (!same) s_vc_check_bad_total++;
        { static u32 lastrep; if (g.frames - lastrep >= 120) { lastrep = g.frames; eng_vc_check_summary(); } }   /* periodic: a killed run has no atexit */
        if (!same && ++s_vcstat.check_bad <= 8)
            fprintf(stderr, "[vcache-check] f%u MISMATCH: cached %u verts/%u draw (indexed %d), converted %u/%u (%d)\n",
                    g.frames, vc_check->n_verts, vc_check->n_draw, vc_check->indexed,
                    dc.n_verts, n_draw, indexed);
    } else if (!vce && vcf.ok && dc.fetch_ok) {
        eng_vcdbg_miss(&vcf);
        WP2_BEGIN(_w57);
        eng_vc_store(vc_key, &vcf, dc.verts, dc.n_verts, layout.stride,
                     dc.n_source_refs, indexed, g.indices, n_draw);
        WP2_END(_w57, 57);
    }

    WP_MARK(3);   /* index build, vertex-cache store */
    RSX_B2P_SET(B2P_CONST);
    /* The transform constant block, then the viewport epilogue the vertex
     * program's tail multiplies by: RSX window coordinates mapped into the
     * host's clip space (rsx_live_draw.c:6143-6156). */
    const float W = sf.clip_w ? (float)sf.clip_w : (float)g.width;
    const float H = sf.clip_h ? (float)sf.clip_h : (float)g.height;
    float xf[8] = { 1, 1, 1, 0, 0, 0, 0, 0 };
    if (vp.scale[0] != 0.0f || vp.translate[0] != 0.0f) {
        xf[0] = vp.scale[0] / (W * 0.5f);
        xf[1] = -(vp.scale[1] / (H * 0.5f));
        xf[2] = vp.scale[2];
        xf[4] = (vp.translate[0] - W * 0.5f) / (W * 0.5f);
        xf[5] = -((vp.translate[1] - H * 0.5f) / (H * 0.5f));
        xf[6] = vp.translate[2];
        if (dbe) {   /* shad114 RSX_DBIAS_EPILOGUE: constant polygon offset, units of the guest depth format, added to window z */
            const u32 br = rsx_dsp_reg(RV, 0x0A7C); float fb; memcpy(&fb, &br, 4);
            if (fb == fb && fb > -1e9f && fb < 1e9f)
                xf[6] += fb * (sf.depth_format == 1 ? (1.0f / 65536.0f) : (1.0f / 16777216.0f));
        }
        /* A draw a host marked (rsx_canvas_remap_set): its window
         * coordinates scaled by k and moved, and nothing else the program
         * computes changed. */
        const u32 mark = g_rsx_canvas_mark, at = RV->canvas_mark_at;
        if (mark && at && RV->constants[(at - 1u) >> 2][(at - 1u) & 3u] == mark) {
            const float k = s_canvas_remap[0], mx = s_canvas_remap[1], my = s_canvas_remap[2];
            xf[0] = k * vp.scale[0] / (W * 0.5f);
            xf[1] = -(k * vp.scale[1] / (H * 0.5f));
            xf[4] = (mx + k * vp.translate[0] - W * 0.5f) / (W * 0.5f);
            xf[5] = -((my + k * vp.translate[1] - H * 0.5f) / (H * 0.5f));
        }
    }
    WP2_BEGIN(_w20);
    /* The 8 KB block is copied and staged again only when a constant or the
     * viewport changed since the last draw; otherwise the backend rebinds
     * what it already staged. RSX_VS_CB_ALWAYS=1 stages it every draw. */
    float ssr_tail[16];   /* RSX_SSR: the matrix of the frame the stored scene colour/depth belongs to (this frame's if none yet) */
    memcpy(ssr_tail, s_ssr_cam_have ? (const void*)s_ssr_cam_prev : (const void*)RV->constants[8], sizeof ssr_tail);
    static int vs_cb_always = -1;
    if (vs_cb_always < 0) vs_cb_always = getenv("RSX_VS_CB_ALWAYS") ? 1 : 0;
    const int vs_cb_same = !vs_cb_always && g.vp_cb_valid &&
        g.vp_cb_gen == RV->const_gen &&
        !memcmp(g.vp_cb + RSX_DSP_NUM_CONSTANTS * 16u, xf, sizeof xf) &&
        !memcmp(g.vp_cb + (RSX_DSP_NUM_CONSTANTS + 2u) * 16u, ssr_tail, sizeof ssr_tail);
    if (!vs_cb_same) {
        /* Only the slots written since the last copy are moved (const_lo..
         * const_hi); RSX_VP_CONST_FULL=1, a first fill, or a generation bump
         * with no recorded range copies everything as before. */
        static int vp_full = -1;
        if (vp_full < 0) vp_full = getenv("RSX_VP_CONST_FULL") ? 1 : 0;
        const u32 lo = RV->const_lo, hi = RV->const_hi;
        const int range = lo <= hi && hi < RSX_DSP_NUM_CONSTANTS;
        if (vp_full || vs_cb_always || !g.vp_cb_valid ||
            (g.vp_cb_gen != RV->const_gen && !range))
            memcpy(g.vp_cb, RV->constants, RSX_DSP_NUM_CONSTANTS * 16u);
        else if (g.vp_cb_gen != RV->const_gen)
            memcpy(g.vp_cb + lo * 16u, RV->constants[lo], (hi - lo + 1u) * 16u);
        RV->const_lo = RSX_DSP_NUM_CONSTANTS; RV->const_hi = 0;
        memcpy(g.vp_cb + RSX_DSP_NUM_CONSTANTS * 16u, xf, sizeof xf);
        memcpy(g.vp_cb + (RSX_DSP_NUM_CONSTANTS + 2u) * 16u, ssr_tail, sizeof ssr_tail);
        g.vp_cb_gen = RV->const_gen;
        g.vp_cb_valid = 1;
        g.vp_cb_bound = 0;
    }

    WP2_END(_w20, 20);
    WP2_BEGIN(_w21);
    /* The buffered fragment constants, then fp_alpha: the layout
     * rsx_fp_decompile_buffered_ex compiled the shader against. */
    const u32 nslots = g.fp_constants.count ? g.fp_constants.count : 1u;
    const u32 fp_bytes = (nslots + 1u) * 16u;
    if (g.fp_cb_cap < fp_bytes) {
        u8* n = (u8*)realloc(g.fp_cb, fp_bytes);
        if (!n) return;
        g.fp_cb = n;
        g.fp_cb_cap = fp_bytes;
    }
    WP2_BEGIN(_w58);
    memset(g.fp_cb, 0, fp_bytes);
    if (g.fp_constants.count)
        memcpy(g.fp_cb, g.fp_constants.values, g.fp_constants.count * 16u);
    WP2_END(_w58, 58);
    /* RSX_FPCONST_AT_FP=<hash>: the fragment constants of the first 3 draws
     * (from RSX_SURF_DUMP_FROM) whose program translates to that hash. */
    { static int left = -1; static unsigned long long want = 0;
      if (left < 0) { const char* e = getenv("RSX_FPCONST_AT_FP"); want = e ? strtoull(e, 0, 16) : 0; left = want ? 3 : 0; }
      if (left > 0) {
          const char* e = getenv("RSX_SURF_DUMP_FROM"); const long from = e ? atol(e) : 0;
          u64 fh = 0, fu = 0;
          for (u32 i = 0; i < g.n_pipelines; i++) if (g.pipelines[i].handle == pipeline) { fh = g.pipelines[i].fp_hlsl; fu = g.pipelines[i].fp_fnv; break; }
          if ((fh == want || fu == want) && (long)g.frames >= from) { left--;
              fprintf(stderr, "[fpconst] f%u fp %016llx:", g.frames, want);
              for (u32 k = 0; k < g.fp_constants.count; k++) {
                  const float* c = (const float*)g.fp_constants.values[k];
                  fprintf(stderr, " c%u=(%.6g %.6g %.6g %.6g)", k, c[0], c[1], c[2], c[3]); }
              fputc('\n', stderr); } } }
    /* RSX_VPCONST_AT_FP=<fp-hlsl hash>: vertex constants 5/11/16/22 + all fragment constants of draws whose
     * program has that hash, one line per distinct set (first 80). */
    { static int left = -1; static unsigned long long want = 0; static float prev[16];
      if (left < 0) { const char* e = getenv("RSX_VPCONST_AT_FP"); want = e ? strtoull(e, 0, 16) : 0; left = want ? 80 : 0; }
      if (left > 0) {
          u64 fh = 0;
          for (u32 i = 0; i < g.n_pipelines; i++) if (g.pipelines[i].handle == pipeline) { fh = g.pipelines[i].fp_hlsl; break; }
          if (fh == want) {
              const float* v = (const float*)g.vp_cb; float cur[16];
              memcpy(cur + 0, v + 5 * 4, 16); memcpy(cur + 4, v + 11 * 4, 16);
              memcpy(cur + 8, v + 16 * 4, 16); memcpy(cur + 12, v + 22 * 4, 16);
              if (memcmp(cur, prev, sizeof cur)) { left--; memcpy(prev, cur, sizeof cur);
                  fprintf(stderr, "[vpconst] f%u c5=(%g %g %g %g) c11=(%g %g %g %g) c16=(%g %g %g %g) c22=(%g %g %g %g) fp:",
                          g.frames, cur[0],cur[1],cur[2],cur[3], cur[4],cur[5],cur[6],cur[7],
                          cur[8],cur[9],cur[10],cur[11], cur[12],cur[13],cur[14],cur[15]);
                  for (u32 k = 0; k < g.fp_constants.count; k++) {
                      const float* c = (const float*)g.fp_constants.values[k];
                      fprintf(stderr, " k%u=(%g %g %g %g)", k, c[0], c[1], c[2], c[3]); }
                  fputc('\n', stderr); } } } }
    /* RSX_FPCONST_ALL=<fp-hlsl hash>: one line per draw of that program (first 6000): frame, reference
     * count, and every fragment constant -- are the repeated passes really the same light? */
    { static int left = -1; static unsigned long long want[16]; static int nwant, wpos_all;
      if (left < 0) { const char* e = getenv("RSX_FPCONST_ALL");
          if (e && !strcmp(e, "wpos")) { wpos_all = 1; nwant = 0; e = NULL; left = 1500; }
          while (e && *e && nwant < 16) { want[nwant++] = strtoull(e, 0, 16); e = strchr(e, ','); if (e) e++; }
          if (!wpos_all) left = nwant ? 6000 : 0; }
      if (left > 0) {
          u64 fh = 0; int wp = 0;
          for (u32 i = 0; i < g.n_pipelines; i++) if (g.pipelines[i].handle == pipeline) { fh = g.pipelines[i].fp_hlsl; wp = g.pipelines[i].wpos; break; }
          int hit = wpos_all && wp; for (int w = 0; w < nwant; w++) if (fh == want[w]) hit = 1;
          if (hit) { left--;
              fprintf(stderr, "[fpall] f%u fp=%016llx aa=%u clip=%ux%u refs=%u:", g.frames, (unsigned long long)fh,
                      (rsx_dsp_reg(RV, 0x0208) >> 12) & 0xFu, sf.clip_w, sf.clip_h, (unsigned)dc.n_refs);
              for (u32 k = 0; k < g.fp_constants.count && k < 16; k++) {
                  const float* c = (const float*)g.fp_constants.values[k];
                  fprintf(stderr, " k%u=(%g %g %g %g)", k, c[0], c[1], c[2], c[3]); }
              fputc('\n', stderr); } } }
    if (s_zeta_alias_log) { s_zeta_alias_log = 0;
        fprintf(stderr, "[rsx engine]   its fragment constants:");
        for (u32 k = 0; k < g.fp_constants.count && k < 6; k++) {
            const float* c = (const float*)g.fp_constants.values[k];
            fprintf(stderr, " c%u=(%.6g %.6g %.6g %.6g)", k, c[0], c[1], c[2], c[3]); }
        fputc('\n', stderr); }
    { float* alpha = (float*)(g.fp_cb + nslots * 16u);
      alpha[0] = rsx_fp_alpha_ref(rs.alpha_ref_raw, rs.alpha_ref_format);
      alpha[1] = sf.clip_h ? (float)sf.clip_h : (float)g.height; alpha[3] = 0.0f;
      /* alpha[2]: 1 = snap the scaled WPOS (rsx_d3d12_engine.c RSX_WPOSF); RSX_WPOS_SNAP_FILE=<path>: 0 while that file exists (same-scene A/B) */
      { static const char* flag = (const char*)-1; static int off;
        if (flag == (const char*)-1) flag = getenv("RSX_WPOS_SNAP_FILE");
        if (flag && !(g.frames & 7u)) { FILE* ff = fopen(flag, "rb"); off = ff != NULL; if (ff) fclose(ff); }
        alpha[2] = off ? 0.0f : 1.0f; } }

    /* The depth attachment is cleared the first time it is bound, so a title
     * that never clears its shadow zeta still tests against something. */
    const u32 depth_handle = (zslot != ENG_INVALID) ? g.zdepths[zslot].handle : 0;
    if (zslot != ENG_INVALID && !g.zdepths[zslot].cleared) {
        g.be->clear_depth_stencil(g.be->user, depth_handle,
                                  RSX_BE_CLEAR_DEPTH | RSX_BE_CLEAR_STENCIL,
                                  1.0f, 0);
        g.zdepths[zslot].cleared = 1;
        g.zdepths[zslot].had_write = 0;
    }

    /* RT_AO: the first draw into another target that samples the scene colour (the bloom chain)
     * ends the scene pass: the AO goes in before it. */
    if (g_rsx_rt_on && g.be->rt_ao_inject && n_targets == 1 &&
        rsx_rt_should_inject(g.frames, g.surfaces[targets[0]].handle)) {
        u32 rch, rdh;
        const u32 rjob = rsx_rt_ao_job(&rch, &rdh);
        rsx_rt_freeze_camera(rjob);
        g.be->rt_ao_inject(g.be->user, rch, rdh, rjob);
    }
    WP2_END(_w21, 21);
    WP2_BEGIN(_w22);
    RSX_B2P_SET(B2P_BIND);
    u32 handles[RSX_BE_MAX_COLOR_TARGETS];
    for (u32 i = 0; i < n_targets; i++) handles[i] = g.surfaces[targets[i]].handle;
    { WP2_BEGIN(_w24); g.be->bind_targets(g.be->user, handles, n_targets, depth_handle); WP2_END(_w24, 24); }
    g.be->bind_pipeline(g.be->user, pipeline);
    { WP2_BEGIN(_w25);
    if (vs_cb_always || !g.vp_cb_bound || !g.be->reuse_vs_constants ||
        !g.be->reuse_vs_constants(g.be->user))
        g.be->bind_vs_constants(g.be->user, g.vp_cb, ENG_VP_CB_BYTES);
    WP2_END(_w25, 25); }
    g.vp_cb_bound = 1;
    if (s_wp_on > 1) { WP2_BEGIN(_w47); static u8 pcb[4096]; static u32 pbytes; s_wp_n2[26]++; if (fp_bytes == pbytes && fp_bytes <= sizeof pcb && !memcmp(pcb, g.fp_cb, fp_bytes)) s_wp_n2[27]++; if (fp_bytes <= sizeof pcb) { memcpy(pcb, g.fp_cb, fp_bytes); pbytes = fp_bytes; } else pbytes = 0; s_wp_n2[29]++; if (vs_cb_same) s_wp_n2[19]++; WP2_END(_w47, 47); }
    { WP2_BEGIN(_w26); g.be->bind_ps_constants(g.be->user, g.fp_cb, fp_bytes); WP2_END(_w26, 26); }
    { WP2_BEGIN(_w27); g.be->bind_textures(g.be->user, textures, samplers, tex_mask); WP2_END(_w27, 27); }
    { WP2_BEGIN(_w28); g.be->bind_vertex_textures(g.be->user, vtextures, vsamplers, vtex_mask); WP2_END(_w28, 28); }
    g.be->set_viewport(g.be->user, 0.0f, 0.0f, W, H);
    /* A movie's picture: a quad sampling two or more of its planes (8-bit,
     * 256 wide and up, as g_rsx_video_plane_uploads counts them). The
     * present shows the frame it is in whole (rsx_present_layout). */
    if (!s_frame_video && tex_mask && dc.n_verts && dc.n_verts <= 8) {
        u32 planes = 0;
        s_tdec_pass = 5;
        for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++) {
            if (!(tex_mask & (1u << u))) continue;
            rsx_dsp_texture t;
            eng_get_tex(RV, u, &t);
            if ((t.format & 0x9Fu) == 0x81u && t.width >= 256) planes++;
        }
        if (planes >= 2) s_frame_video = 1;
    }

    /* The guest scissor intersected with the surface. The nv40 reset is a
     * full 4096x4096 window and a never-written register reads 0 here, which
     * the w == 0 test treats as no scissor, so only a real game scissor
     * narrows anything (rsx_live_draw.c:6180-6197). */
    {
        u32 sx = 0, sy = 0;
        u32 sw = g.surfaces[target].w, sh = g.surfaces[target].h;
        const u32 h = rsx_dsp_reg(RV, M_SCISSOR_HORIZONTAL);
        const u32 v = rsx_dsp_reg(RV, M_SCISSOR_VERTICAL);
        const u32 gx = h & 0xFFFFu, gw = h >> 16;
        const u32 gy = v & 0xFFFFu, gh = v >> 16;
        if (gw > 0 && gh > 0) {
            u32 right = sx + sw, bottom = sy + sh;
            if (gx > sx) sx = gx;
            if (gy > sy) sy = gy;
            if (gx + gw < right)  right = gx + gw;
            if (gy + gh < bottom) bottom = gy + gh;
            sw = right > sx ? right - sx : 0;
            sh = bottom > sy ? bottom - sy : 0;
        }
        g.be->set_scissor(g.be->user, sx, sy, sw, sh);
    }
    /* Dynamic, and deliberately outside the pipeline key. */
    g.be->set_stencil_ref(g.be->user,
                          rsx_dsp_reg(RV, M_STENCIL_FUNC_REF) & 0xFFu);
    /* NV4097_SET_BLEND_COLOR (0x31C): A8R8G8B8 for an 8-bit target; for an
     * FP16 one it holds R16|G16<<16 and SET_BLEND_COLOR2 (0x300) B16|A16<<16. */
    if (g.be->set_blend_color) {
        const u32 c = rsx_dsp_reg(RV, 0x031C);
        rsx_dsp_surface bsf; rsx_dsp_get_surface(RV, &bsf);
        float bc[4];
        if (bsf.color_format == RSX_SURFACE_FMT_F_W16Z16Y16X16) {
            const u32 c2 = rsx_dsp_reg(RV, 0x0300);
            const u16 h[4] = { (u16)c, (u16)(c >> 16), (u16)c2, (u16)(c2 >> 16) };
            for (int k = 0; k < 4; k++) {
                const u32 e = (h[k] >> 10) & 0x1F, m = h[k] & 0x3FF;
                float v = e == 0 ? m / 16777216.0f : (e == 31 ? 65504.0f : ldexpf(1.0f + m / 1024.0f, (int)e - 15));
                bc[k] = (h[k] & 0x8000) ? -v : v;
            }
        } else {
            bc[0] = ((c >> 16) & 0xFF) / 255.0f; bc[1] = ((c >> 8) & 0xFF) / 255.0f;
            bc[2] = (c & 0xFF) / 255.0f;         bc[3] = (c >> 24) / 255.0f;
        }
        /* CONSTANT_ALPHA / ONE_MINUS_CONSTANT_ALPHA (0x8003/0x8004) weight every channel by the constant's
         * alpha; D3D12's BLEND_FACTOR weights each channel by its own component. The game's water uses
         * CONSTANT_ALPHA with a constant like (0,0,0,opacity), so the colour channels got weight 0 and every
         * water surface came out fully transparent (30/10). When the colour factors use the constant's alpha
         * (and not its colour), the constant handed to D3D12 is that alpha in all four components. */
        { const u32 sf = rsx_dsp_reg(RV, M_BLEND_SFACTOR) & 0xFFFFu, df = rsx_dsp_reg(RV, M_BLEND_DFACTOR) & 0xFFFFu;
          const int ca = sf == 0x8003u || sf == 0x8004u || df == 0x8003u || df == 0x8004u;
          const int cc = sf == 0x8001u || sf == 0x8002u || df == 0x8001u || df == 0x8002u;
          if (ca && !cc) bc[0] = bc[1] = bc[2] = bc[3]; }
        g.be->set_blend_color(g.be->user, bc);
    }

    RSX_B2P_SET(B2P_DRAWREC);
    const u32 uploaded = indexed ? dc.n_verts : n_draw;
    /* RT_AO: transparent draws of the scene (particles, steam, water, light passes) are drawn after the AO, whatever their place in
     * the stream: the AO is a function of the opaque depth and must not tint what is in front of it. Fur shells stay (they are
     * the character: see rt_ao_mark). Needs depth test (a draw without it, a UI layer, keeps its place). */
    if (g_rsx_rt_on && g.be->rt_ao_defer && rs.blend_enable && rs.depth_test && !rs.depth_write &&
        n_targets == 1 && depth_handle && g.surfaces[target].handle == rsx_rt_scene_color() &&
        !(vce && s_cur_pipe && (s_cur_pipe->rt_static & 2) && topology == RSX_TOPOLOGY_TRIANGLES && !pipeline_is_fixed))
        g.be->rt_ao_defer(g.be->user);
    WP2_END(_w22, 22);
    WP2_BEGIN(_w23);
    if (g.q_cur && g.be->query_set) { g.be->query_set(g.be->user, g.q_cur); g.q_draws++; }
    if (vce && vce->buf)
        g.be->draw_buffer(g.be->user, topology, vce->buf, 0, uploaded,
                          layout.stride, vce->ib_off, indexed ? n_draw : 0);
    else
        g.be->draw(g.be->user, topology, draw_verts, uploaded, layout.stride,
                   indexed ? draw_indices : NULL, indexed ? n_draw : 0);
    WP2_END(_w23, 23);
    WP_MARK(4);   /* constants, backend binds, record */
    RSX_B2P_SET(B2P_TAIL);
    /* Counted only when the draw ran the guest's OWN programs, which is what
     * the test hook means: a draw through the built-in pair is a draw, not
     * evidence that the decompile-translate-compile path worked. */
    if (!pipeline_is_fixed) g.guest_draws++;
    if (g_rsx_rt_on && g_rsx_rt.trace && n_targets == 1 && g.surfaces[target].w >= 640)
        rsx_rt_trace(g.frames, rs.blend_enable ? 'B' : !rs.depth_write ? 'N' : (s_cur_pipe && (s_cur_pipe->rt_static & 1)) ? (vce ? 'S' : 's') : (s_cur_pipe && (s_cur_pipe->rt_static & 2)) ? 'D' : 'O',
                     s_cur_pipe ? s_cur_pipe->vp_hlsl : 0, n_draw / 3u, rs.alpha_test_enable);
    /* RT_AO: an opaque static world draw (vertex cache hit) feeds the ray-tracing scene. */
    /* RT_AO: where the opaque scene geometry ends (the AO record is inserted after the last such draw, so that
     * particles, additive passes and other blended draws are not darkened). */
    if (g_rsx_rt_on && g.be->rt_ao_mark && rs.depth_test && rs.depth_write && !rs.blend_enable && rs.color_mask &&
        n_targets == 1 && depth_handle && g.surfaces[target].handle == rsx_rt_scene_color())
        g.be->rt_ao_mark(g.be->user);
    /* Fur shells (skinned triangles, blended over the character's opaque mesh, no depth write) are part of the character: the
     * AO goes after them too, or it darkens the base mesh alone and the strands stand out as lines. Particles (quads) are not. */
    else if (g_rsx_rt_on && g.be->rt_ao_mark && vce && s_cur_pipe && (s_cur_pipe->rt_static & 2) && !pipeline_is_fixed &&
             topology == RSX_TOPOLOGY_TRIANGLES && rs.depth_test && !rs.depth_write && rs.blend_enable && rs.color_mask &&
             n_targets == 1 && depth_handle && g.surfaces[target].handle == rsx_rt_scene_color())
        g.be->rt_ao_mark(g.be->user);
    /* RT_AO_COV: why draws into the scene target do or do not feed the ray-tracing scene (counted per frame, logged). */
    if (g_rsx_rt_on && n_targets == 1 && depth_handle && g.surfaces[target].w >= 640 && g.surfaces[target].handle == rsx_rt_scene_color())
        rsx_rt_cov((pipeline_is_fixed || !s_cur_pipe || !s_cur_pipe->rt_static) ? 2 : topology != RSX_TOPOLOGY_TRIANGLES ? 7 :
                   !rs.depth_test ? 6 : rs.blend_enable ? 5 : !rs.depth_write ? (rs.color_mask ? 3 : 4) : !vce ? 1 :
                   (vce->cpu_verts || (!vce->buf && vce->verts)) ? 0 : 8, n_draw / 3u);
    /* A depth-only pre-pass (colour mask 0) is geometry like any other: it feeds the scene too. */
    if (g_rsx_rt_on && vce && s_cur_pipe && s_cur_pipe->rt_static && !pipeline_is_fixed &&
        topology == RSX_TOPOLOGY_TRIANGLES && rs.depth_test && rs.depth_write && !rs.blend_enable &&
        n_targets == 1 && depth_handle && g.surfaces[target].w >= 640) {
        const u8* rv = vce->cpu_verts ? vce->cpu_verts : (vce->buf ? NULL : vce->verts);
        const u32* ri = vce->cpu_verts ? vce->cpu_indices : (vce->buf ? NULL : vce->indices);
        if (rv && (!indexed || ri)) {
            RsxRtDraw rd;
            rd.vc_key = vce->key; rd.vc_content = vce->content; rd.vs_id = s_cur_pipe->vp_hlsl;
            rd.verts = rv; rd.n_verts = uploaded; rd.stride = layout.stride;
            rd.indices = indexed ? ri : NULL; rd.n_draw = n_draw;
            rd.cb = (const float*)g.vp_cb; rd.frame = g.frames;
            rd.color_h = g.surfaces[targets[0]].handle; rd.depth_h = depth_handle; rd.alpha_test = rs.alpha_test_enable;
            rsx_rt_note_draw(&rd);
        }
    }
    /* RSX_NAN_SCAN=1 (30/10, water): per draw, the uploaded vertices (all float4 slots) checked for NaN, Inf and
     * |x| > 1e30 -- values the SPU never produces (no NaN/Inf there) but a recompiled SPU job could -- and the
     * position slot for all-zero vertices. One line per draw with any, first 400 lines. */
    { static int on = -1; static int left = 400; static long from = 0;
      if (on < 0) { const char* e = getenv("RSX_NAN_SCAN"); on = e ? atoi(e) : 0; if (e && on <= 0) on = 1;
          const char* f2 = getenv("RSX_NAN_FROM"); from = f2 ? atol(f2) : 0; }   /* RSX_NAN_SCAN=<min % of bad values>, RSX_NAN_FROM=<frame> */
      /* only the freshly converted buffer (dc.verts): a cached copy's CPU side may already be freed */
      if (on && left > 0 && (long)g.frames >= from && dc.out_verts && dc.out_verts == dc.verts && dc.n_verts && dc.layout.stride &&
          (u64)dc.n_verts * dc.layout.stride <= dc.verts_cap) {
          const u32 nf = dc.layout.stride / 4u; const float* f = (const float*)dc.out_verts;
          const int ps = dc.layout.slot_by_attr[0];
          u32 bad = 0, badpos = 0, zpos = 0, slots = 0;
          for (u32 v = 0; v < dc.n_verts; v++) {
              const float* q = f + (size_t)v * nf;
              for (u32 k = 0; k < nf; k++) {
                  u32 w; memcpy(&w, &q[k], 4);
                  if ((w & 0x7F800000u) == 0x7F800000u || fabsf(q[k]) > 1e30f) { bad++; slots |= 1u << (k / 4u);
                      if (ps >= 0 && k / 4u == (u32)ps) badpos++; } }
              if (ps >= 0 && q[ps * 4] == 0.0f && q[ps * 4 + 1] == 0.0f && q[ps * 4 + 2] == 0.0f) zpos++;
          }
          if ((bad && (u64)bad * 100u >= (u64)on * dc.n_verts) || (ps >= 0 && zpos * 10 >= dc.n_verts * 9u && dc.n_verts > 8)) { left--;
              u64 fh = 0; for (u32 i = 0; i < g.n_pipelines; i++) if (g.pipelines[i].handle == pipeline) { fh = g.pipelines[i].fp_fnv; break; }
              fprintf(stderr, "[nan] f%u pipe %u fp-ucode %016llx verts %u stride %u: bad floats %u (in position %u, slots 0x%X), zero positions %u\n",
                      g.frames, pipeline, (unsigned long long)fh, dc.n_verts, dc.layout.stride, bad, badpos, slots, zpos); } } }
    s_dstat.issued++;
    if (target < ENG_MAX_SURFACES) s_surf_draws[target]++;
    if (rs.color_mask) { rsx_dsp_vertex_attr a7; rsx_dsp_get_vertex_attr(RV, 7, &a7);
        if (a7.type == 7) { s_skin_draws++; s_skin_verts += dc.n_verts; } }
    eng_draw_trace(pipeline_is_fixed ? "OK-fixed" : "OK", prim, indexed, n_draw, pipeline, tex_mask);
    /* RSX_PICK=<x>,<y>,<frame>: after every draw of that frame into a
     * full-size target, finish the GPU work and read pixel (x, y) of the
     * draw's first target back -- which draws touch a pixel, and what each
     * one leaves there. Slow (a GPU sync per draw); for tools/rsx_replay. */
    { static u32 last[4]; static u32 last_surf = ENG_INVALID; static u32 idx;
      int px = s_pick_x, py = s_pick_y; long pf = s_pick_frame;
      if (px == -2) { const char* e = getenv("RSX_PICK"); px = -1;
          if (e && sscanf(e, "%d,%d,%ld", &px, &py, &pf) != 3) px = -1;
          s_pick_x = px; s_pick_y = py; s_pick_frame = pf; }
      /* A frame that issued no draw (the present count can step by two)
       * moves the pick to the next one that does. */
      { static long hit = -1;
        if (px >= 0 && (long)g.frames > pf && hit != pf && pf >= 0) s_pick_frame = pf = (long)g.frames;
        if (px >= 0 && (long)g.frames == pf) hit = pf; }
      if (px >= 0 && (long)g.frames == pf && target < g.n_surfaces) {
          const eng_surface* ts = &g.surfaces[target];
          idx++;
          /* A smaller target (a downsample) is read at the same place in
           * the picture: the pick scaled to its size. */
          const u32 qx = ts->w >= 1280 ? (u32)px : (u32)((u64)px * ts->w / 1280u);
          const u32 qy = ts->h >= 720 ? (u32)py : (u32)((u64)py * ts->h / 720u);
          if (qx < ts->w && qy < ts->h) {
              if (g.be->submit_and_wait) g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_GUEST_REFERENCE);
              u8 b[16]; memset(b, 0, sizeof b);
              const u32 bpp = ts->fmt == RSX_BE_FMT_R16G16B16A16F ? 8u : 4u;
              g.be->readback(g.be->user, ts->handle, qx, qy, 1, 1, b, bpp);
              u32 cur[4] = {0, 0, 0, 0}; memcpy(cur, b, bpp);
              if (target != last_surf || memcmp(cur, last, sizeof cur)) {
                  float c[4] = {0, 0, 0, 0};
                  if (bpp == 8) for (int k = 0; k < 4; k++) {
                      const u16 v = ((const u16*)b)[k]; const u32 ex = (v >> 10) & 0x1F, mant = v & 0x3FF;
                      float o = ex == 0 ? mant / 1024.0f / 16384.0f : ex == 31 ? 65504.0f : (1.0f + mant / 1024.0f) * eng_half_pow2(ex);
                      c[k] = (v & 0x8000) ? -o : o; }
                  else for (int k = 0; k < 4; k++) c[k] = b[k] / 255.0f;
                  u64 fh = 0;
                  for (u32 i = 0; i < g.n_pipelines; i++) if (g.pipelines[i].handle == pipeline) { fh = g.pipelines[i].fp_hlsl; break; }
                  rsx_dsp_surface psf; rsx_dsp_get_surface(RV, &psf);
                  fprintf(stderr, "[pick] f%u draw %u pipe %u fp %016llx -> s%u(rsxfmt 0x%X engfmt %u) = (%.4g %.4g %.4g %.4g) blend=%u(%X/%X) depth=%u/%u/%X mask=0x%X tex=0x%X verts=%u\n",
                          g.frames, idx, pipeline, (unsigned long long)fh, target, psf.color_format, (u32)ts->fmt, c[0], c[1], c[2], c[3],
                          rs.blend_enable, rs.sf_rgb, rs.df_rgb, rs.depth_test, rs.depth_write, rs.depth_func, rs.color_mask, tex_mask, dc.n_verts);
                  memcpy(last, cur, sizeof cur); last_surf = target;
              }
          }
      } }

    if (zslot != ENG_INVALID && rs.depth_test && rs.depth_write) {
        /* New depth: any snapshot taken before this draw is stale. Only a
         * depth CLEAR used to invalidate them, so a title that never clears
         * a depth buffer between reads (Drakengard 3's battle areas) read a
         * snapshot from long before: the character shadow projection
         * rebuilt every receiver at the near plane, landed outside its
         * shadow map, and no character cast a shadow. */
        g.zdepths[zslot].had_write = 1;
        g.zdepths[zslot].snapshot_valid = 0;
        g.zdepths[zslot].packed_valid = 0;
    }
    WP_MARK(5);   /* the rest (stats, pick, depth bookkeeping) */
    s_wp_draws++;
#undef WP_MARK
}

static void sink_clear(void* user, const rsx_dispatch* r, u32 mask)
{
    (void)user; (void)r;
    if (!g.ready) return;
    rsx_draw_engine_b2_drain();   /* RSX_DRAW_SPLIT=2: ordered after the queued draws, reads the live registers (RV == g.rsx here) */
    u32 targets[RSX_BE_MAX_COLOR_TARGETS];
    const u32 n_targets = eng_current_target_set(targets);
    if (s_ring_on < 0) s_ring_on = (getenv("RSX_TRACE_ON_WHITE") || getenv("RSX_TRACE_ON_COLOR")) ? 1 : 0;
    if (s_ring_on) { char line[160];
        snprintf(line, sizeof line, "f%u CLEAR mask=0x%X color=0x%08X target=s%d", g.frames, mask,
                 rsx_dsp_clear_color(RV), n_targets ? (int)targets[0] : -1);
        eng_ring_push(line); }
    if (!n_targets) return;

    if (mask & (RSX_CLEAR_COLOR_R | RSX_CLEAR_COLOR_G |
                RSX_CLEAR_COLOR_B | RSX_CLEAR_COLOR_A)) {
        const u32 c = rsx_dsp_clear_color(RV);
        const float rgba[4] = {
            (float)((c >> 16) & 0xFF) / 255.0f,
            (float)((c >>  8) & 0xFF) / 255.0f,
            (float)( c        & 0xFF) / 255.0f,
            (float)((c >> 24) & 0xFF) / 255.0f,
        };
        /* Every target the set names, not only A: a deferred pass clears its
         * whole G-buffer in one CLEAR_SURFACE, and B, C and D would otherwise
         * keep whatever the last frame left in them. */
        for (u32 i = 0; i < n_targets; i++)
            g.be->clear_color(g.be->user, g.surfaces[targets[i]].handle, rgba);
    }
    if (mask & (RSX_CLEAR_DEPTH | RSX_CLEAR_STENCIL)) {
        rsx_dsp_surface sf;
        rsx_dsp_get_surface(RV, &sf);
        eng_note_pass(&sf);
        const u32 zslot = eng_zdepth_get(sf.zeta_location, sf.zeta_offset,
                                         sf.clip_w, sf.clip_h);
        if (zslot == ENG_INVALID) return;
        u32 flags = 0;
        if (mask & RSX_CLEAR_DEPTH)   flags |= RSX_BE_CLEAR_DEPTH;
        if (mask & RSX_CLEAR_STENCIL) flags |= RSX_BE_CLEAR_STENCIL;
        /* ZSTENCIL_CLEAR_VALUE is Z24S8: depth24 << 8 | stencil8, with the
         * nv40 reset 0xFFFFFF00 seeded by rsx_dispatch_init, so a stream that
         * never writes it still clears to 1.0 / 0. */
        const u32 zs = rsx_dsp_reg(RV, M_ZSTENCIL_CLEAR);
        const float zval = zs ? (float)(zs >> 8) / 16777215.0f : 1.0f;
        /* CLEAR_SURFACE honours the scissor. Drakengard 3 packs its shadow
         * maps into one 512x512 depth target and clears each region just
         * before drawing it; cleared whole, the last region's clear wiped the
         * character shadows drawn earlier in the frame whenever the
         * projections came after it, and they flickered in and out. The
         * scissor is the draws' (guest scissor within the surface);
         * RSX_CLEAR_NO_SCISSOR=1 clears the whole target as before. */
        int partial = 0;
        u32 cx = 0, cy = 0, cw = 0, ch = 0;
        { static int off = -1; if (off < 0) off = getenv("RSX_CLEAR_NO_SCISSOR") ? 1 : 0;
          const u32 h = rsx_dsp_reg(RV, M_SCISSOR_HORIZONTAL);
          const u32 v = rsx_dsp_reg(RV, M_SCISSOR_VERTICAL);
          const u32 gx = h & 0xFFFFu, gw = h >> 16, gy = v & 0xFFFFu, gh = v >> 16;
          const u32 sw = sf.clip_w ? sf.clip_w : g.zdepths[zslot].w;
          const u32 sh = sf.clip_h ? sf.clip_h : g.zdepths[zslot].h;
          if (!off && gw > 0 && gh > 0 && g.be->clear_depth_stencil_rect) {
              u32 right = sw, bottom = sh;
              cx = gx; cy = gy;
              if (gx + gw < right)  right = gx + gw;
              if (gy + gh < bottom) bottom = gy + gh;
              cw = right > cx ? right - cx : 0;
              ch = bottom > cy ? bottom - cy : 0;
              /* Whole-target clears keep the load-action path. */
              partial = !(cx == 0 && cy == 0 && cw >= g.zdepths[zslot].w && ch >= g.zdepths[zslot].h);
              if (partial && (!cw || !ch)) return;   /* scissored away entirely */
          } }
        if (partial)
            g.be->clear_depth_stencil_rect(g.be->user, g.zdepths[zslot].handle, flags,
                                           zval, (u8)(zs & 0xFFu), cx, cy, cw, ch);
        else
            g.be->clear_depth_stencil(g.be->user, g.zdepths[zslot].handle, flags,
                                      zval, (u8)(zs & 0xFFu));
        g.zdepths[zslot].cleared = 1;
        if (mask & RSX_CLEAR_DEPTH) {
            /* A clear invalidates the older published depth image; the next
             * texture consumer resolves the newly written pass exactly once.
             * A partial clear leaves the rest of the target's depth in place,
             * so it still counts as written. */
            /* shad114 (idea: RACiT, MIT, (c) 2026 HighDrexler): a depth clear IS a
             * write -- on the RSX it fills the buffer, and a depth read of that
             * address sees the clear value (far plane). Treating a non-partial clear
             * as "no write" sent an empty 64x64 character shadow map to the guest
             * bytes (zero = nearest): everything in shadow, the dark box under
             * Ratchet. RSX_CLEAR_COUNTS_WRITE=0 restores the old behaviour. */
            { static int cw = -1; if (cw < 0) { const char* e = getenv("RSX_CLEAR_COUNTS_WRITE"); cw = !(e && e[0] == '0'); }
              if (cw) g.zdepths[zslot].had_write = 1;
              else if (!partial) g.zdepths[zslot].had_write = 0; }
            g.zdepths[zslot].snapshot_valid = 0;
            g.zdepths[zslot].packed_valid = 0;
        }
    }
}

/* Resolve a flip's buffer id to a REGISTERED surface rather than whatever is
 * currently bound: the live target at that instant is often an offscreen
 * shadow or post-process surface, and copying it presents black despite the
 * scene's draws having executed (rsx_live_draw.c:7557-7570). */
static u32 eng_present_surface(u32 buffer_id)
{
    if (buffer_id < 8 && g.display_buffers[buffer_id].valid) {
        const eng_display_buffer* d = &g.display_buffers[buffer_id];
        for (u32 i = 0; i < g.n_surfaces; i++)
            if (g.surfaces[i].handle && g.surfaces[i].location == d->location &&
                g.surfaces[i].offset == d->offset)
                return i;
    }
    /* No registered surface for this display buffer: the bound target is presented instead. If that is an
     * offscreen pass it comes out black -- a black flash. RSX_PRESENT_LOG=1 says when it happens. */
    { static int lg = -1; static u32 n;
      if (lg < 0) { const char* e = getenv("RSX_PRESENT_LOG"); lg = e && *e && *e != '0'; }
      if (lg && (n < 24 || (n & 255u) == 0)) {
          const u32 cur = eng_current_surface();
          const eng_display_buffer* d = buffer_id < 8 ? &g.display_buffers[buffer_id] : NULL;
          fprintf(stderr, "[present-fallback] #%u frame %u buffer %u (%s 0x%08X %ux%u): no surface; bound target s%d %ux%u 0x%08X, last presented s%u\n",
                  n + 1u, g.frames, buffer_id, d && d->valid ? "registered" : "unregistered",
                  d ? d->offset : 0u, d ? d->width : 0u, d ? d->height : 0u,
                  cur == ENG_INVALID ? -1 : (int)cur, cur == ENG_INVALID ? 0u : g.surfaces[cur].w,
                  cur == ENG_INVALID ? 0u : g.surfaces[cur].h, cur == ENG_INVALID ? 0u : g.surfaces[cur].offset,
                  g.last_present_surface);
      }
      n++; }
    return eng_current_surface();
}

/* RSX_SURF_DUMP_DIR=<dir> [RSX_SURF_DUMP_FROM=<frame> RSX_SURF_DUMP_EVERY=<n>
 * RSX_SURF_DUMP_COUNT=<k>]: after a present, read every registered colour
 * surface back and write it as a PPM (an FP16 target clamped to 0..1), with a
 * log line carrying its mean and maximum. The whole post-process chain of one
 * frame side by side: which stage goes white is then a matter of looking. */
static void eng_surface_dump_frame(void)
{
    static const char* dir = (const char*)1; static long from, every, count, done;
    if (dir == (const char*)1) {
        dir = getenv("RSX_SURF_DUMP_DIR");
        const char* e;
        from  = (e = getenv("RSX_SURF_DUMP_FROM"))  ? atol(e) : 0;
        every = (e = getenv("RSX_SURF_DUMP_EVERY")) ? atol(e) : 240; if (every <= 0) every = 240;
        count = (e = getenv("RSX_SURF_DUMP_COUNT")) ? atol(e) : 8;
    }
    /* RSX_SURF_DUMP_BRIGHT=<mean>: instead of a frame schedule, dump when the
     * presented frame's mean brightness (0-255) is at least <mean>, from
     * RSX_SURF_DUMP_FROM on, at least RSX_SURF_DUMP_EVERY frames apart -- the
     * scene you are after, wherever the run's timing put it. */
    static long bright = -2, last_dump = -1000000;
    if (bright == -2) { const char* e = getenv("RSX_SURF_DUMP_BRIGHT"); bright = e ? atol(e) : -1; }
    if (!dir || !*dir || done >= count || (long)g.frames < from) return;
    if (bright >= 0) {
        if ((long)g.frames - last_dump < every) return;
        const eng_surface* ps = &g.surfaces[g.last_present_surface];
        if (ps->fmt != RSX_BE_FMT_R8G8B8A8 || !ps->w || !ps->h) return;
        u8* buf = (u8*)malloc((size_t)ps->w * ps->h * 4);
        if (!buf) return;
        g.be->readback(g.be->user, ps->handle, 0, 0, ps->w, ps->h, buf, ps->w * 4);
        unsigned long long sum = 0; u32 n = 0;
        for (u32 q = 0; q < ps->w * ps->h; q += 61) { sum += buf[q * 4] + buf[q * 4 + 1] + buf[q * 4 + 2]; n += 3; }
        free(buf);
        if (!n || (long)(sum / n) < bright) return;
        last_dump = (long)g.frames;
        fprintf(stderr, "[surf-dump] frame %u presented mean %llu >= %ld: dumping\n", g.frames, sum / n, bright);
    } else if (((long)g.frames - from) % every) return;
    done++;
    eng_surface_dump_now();
}

static void eng_surface_dump_now(void)
{
    static const char* dir = (const char*)1;
    if (dir == (const char*)1) dir = getenv("RSX_SURF_DUMP_DIR");
    if (!dir || !*dir) return;
    for (u32 i = 0; i < g.n_surfaces; i++) {
        const eng_surface* sf = &g.surfaces[i];
        if (!sf->handle || !sf->w || !sf->h) continue;
        const int fp16 = sf->fmt == RSX_BE_FMT_R16G16B16A16F;
        const int f32  = sf->fmt == RSX_BE_FMT_R32F;
        if (!fp16 && !f32 && sf->fmt != RSX_BE_FMT_R8G8B8A8) continue;
        const u32 bpp = fp16 ? 8u : 4u;
        u8* buf = (u8*)malloc((size_t)sf->w * sf->h * bpp);
        if (!buf) continue;
        memset(buf, 0, (size_t)sf->w * sf->h * bpp);
        g.be->readback(g.be->user, sf->handle, 0, 0, sf->w, sf->h, buf, sf->w * bpp);
        double sum[3] = {0, 0, 0}, mx[3] = {0, 0, 0}; unsigned long nan = 0;
        double asum = 0, amin = 1e30, amax = -1e30;
        char path[1024];
        snprintf(path, sizeof path, "%s/f%06u%s_s%02u_%ux%u_%s.ppm", dir, g.frames, s_dump_tag, i, sf->w, sf->h,
                 fp16 ? "fp16" : f32 ? "r32f" : "rgba8");
        FILE* f = fopen(path, "wb");
        if (f) fprintf(f, "P6\n%u %u\n255\n", sf->w, sf->h);
        for (u32 p = 0; p < sf->w * sf->h; p++) {
            float c[3];
            if (fp16) {
                const u16* h = (const u16*)(buf + (size_t)p * 8);
                float a4[4];
                for (int k = 0; k < 4; k++) {
                    u16 v = h[k]; u32 sgn = (v >> 15) & 1, ex = (v >> 10) & 0x1F, mant = v & 0x3FF; float out;
                    if (ex == 0) out = (float)mant / 1024.0f / 16384.0f;
                    else if (ex == 31) { out = mant ? 0.0f : 1e30f; if (mant) nan++; }
                    else out = (1.0f + mant / 1024.0f) * eng_half_pow2(ex);
                    a4[k] = sgn ? -out : out;
                }
                c[0] = a4[0]; c[1] = a4[1]; c[2] = a4[2];
                asum += a4[3]; if (a4[3] < amin) amin = a4[3]; if (a4[3] > amax) amax = a4[3];
            } else if (f32) {
                float v; memcpy(&v, buf + (size_t)p * 4, 4); c[0] = c[1] = c[2] = v;
            } else {
                c[0] = buf[p * 4] / 255.0f; c[1] = buf[p * 4 + 1] / 255.0f; c[2] = buf[p * 4 + 2] / 255.0f;
            }
            u8 rgb[3];
            for (int k = 0; k < 3; k++) {
                if (c[k] == c[k]) { sum[k] += c[k]; if (c[k] > mx[k]) mx[k] = c[k]; }
                float t = c[k] < 0 ? 0 : c[k] > 1 ? 1 : c[k]; rgb[k] = (u8)(t * 255.0f + 0.5f);
            }
            if (f) fwrite(rgb, 1, 3, f);
        }
        if (f) fclose(f);
        const double n = (double)sf->w * sf->h;
        fprintf(stderr, "[surf-dump] f%u s%u %ux%u %s off=0x%08X mean=(%.3f %.3f %.3f) max=(%.2f %.2f %.2f) nan=%lu",
                g.frames, i, sf->w, sf->h, fp16 ? "fp16" : f32 ? "r32f" : "rgba8", sf->offset,
                sum[0] / n, sum[1] / n, sum[2] / n, mx[0], mx[1], mx[2], nan);
        if (fp16) fprintf(stderr, " alpha mean=%.4g min=%.4g max=%.4g", asum / n, amin, amax);
        fprintf(stderr, " -> %s\n", path);
        /* An FP16 target's alpha as its own image, scaled to its maximum:
         * UE3 on PS3 keeps scene depth there. */
        if (fp16 && amax > 0) {
            char apath[1100]; snprintf(apath, sizeof apath, "%s.alpha.ppm", path);
            FILE* af = fopen(apath, "wb");
            if (af) { fprintf(af, "P6\n%u %u\n255\n", sf->w, sf->h);
                for (u32 p = 0; p < sf->w * sf->h; p++) {
                    const u16 v = ((const u16*)(buf + (size_t)p * 8))[3];
                    u32 ex = (v >> 10) & 0x1F, mant = v & 0x3FF; float out;
                    if (ex == 0) out = (float)mant / 1024.0f / 16384.0f; else if (ex == 31) out = 0; else out = (1.0f + mant / 1024.0f) * eng_half_pow2(ex);
                    if (v & 0x8000) out = -out;
                    float t = out / (float)amax; t = t < 0 ? 0 : t > 1 ? 1 : t; u8 g8 = (u8)(t * 255.0f + 0.5f); u8 px3[3] = { g8, g8, g8 };
                    fwrite(px3, 1, 3, af); }
                fclose(af); } }
        free(buf);
    }
}
static const char* s_present_src = "host";   /* "fifo": the title's own flip method */
/* The RSX shows at most one flip per vblank: a title that runs ahead (its
 * cellGcmSetWaitFlip only queues a wait for the RSX, see cellGcmSys.c) is
 * held here, and its next label/report waits pace the PPU -- 60 Hz, like the
 * console, instead of the title's own 83 fps limiter. GCM_FLIP_PACE_HZ=<hz>
 * (default 60; 0 = off). */
static int s_pace_held; static double s_pace_wait_ms;   /* TOD_PIPE_DIAG */
static void rsx_flip_pace(void)
{
#ifdef _WIN32
    s_pace_held = 0; s_pace_wait_ms = 0.0;
    static double hz = -1.0;
    if (hz < 0.0) { const char* e = getenv("GCM_FLIP_PACE_HZ"); hz = e ? atof(e) : 60.0;
                    const char* b = getenv("GCM_WAITFLIP_BLOCK"); if (b && *b && *b != '0') hz = 0.0; }
    if (hz <= 0.0) return;
    static LARGE_INTEGER f; static long long next;
    static HANDLE timer;
    if (!f.QuadPart) { QueryPerformanceFrequency(&f);
        timer = CreateWaitableTimerExW(NULL, NULL, 0x00000002 /* HIGH_RESOLUTION */, TIMER_ALL_ACCESS); }
    const long long period = (long long)((double)f.QuadPart / hz);
    LARGE_INTEGER now; QueryPerformanceCounter(&now);
    { static long calls, waits; static long long t0; calls++; if (next && now.QuadPart < next) waits++;
      if (!t0) t0 = now.QuadPart;
      if (getenv("GCM_FLIP_PACE_LOG") && now.QuadPart - t0 > f.QuadPart * 5) {
          fprintf(stderr, "[flip-pace] %ld flips, %ld held (5 s)\n", calls, waits); calls = waits = 0; t0 = now.QuadPart; } }
    if (next && now.QuadPart < next) {
        const long long left = next - now.QuadPart;
        s_pace_held = 1;
        const long long margin = f.QuadPart / 2000;                /* spin the last 0.5 ms */
        if (timer && left > margin) {
            LARGE_INTEGER due; due.QuadPart = -((left - margin) * 10000000ll / f.QuadPart);
            SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE);
            WaitForSingleObject(timer, 100);
        }
        do { YieldProcessor(); QueryPerformanceCounter(&now); } while (now.QuadPart < next);
        { LARGE_INTEGER e; QueryPerformanceCounter(&e); s_pace_wait_ms = (double)(e.QuadPart - (next - left)) * 1000.0 / (double)f.QuadPart; }
        next += period;
    } else
        next = now.QuadPart + period;                               /* behind: no catch-up burst */
#endif
}

/* RSX_WPROF=1: every 240 presents, ms per present of each walker phase (rdtsc, converted by
 * the QueryPerformanceCounter span between reports). draw = sink_end_impl phases 0-5; fifo =
 * whole cellGcm_rsx_process_fifo (so fifo - draw = FIFO parsing, method dispatch, clears);
 * enc = eng_encode_records (D3D12 command list recording, in the present). */
static void eng_wprof_report(void)
{
    if (s_wp_on < 0) eng_wk_env();
    if (s_wp_on <= 0) return;
    static unsigned n; static unsigned long long t0tsc, t0q;
    static unsigned long long l_nscan, l_nmiss, l_cyc[64], l_fifo, l_enc, l_draws, l_calls;
    static unsigned long long l_c[6], l_meths, l_runs, l_runw, l_cmds;
    unsigned long long q = 0;
#ifdef _WIN32
    { LARGE_INTEGER c; QueryPerformanceCounter(&c); q = (unsigned long long)c.QuadPart; }
#endif
    if (!t0tsc) { t0tsc = wp_tsc(); t0q = q; return; }
    if (++n < 240) return;
    n = 0;
    unsigned long long qf = 1;
#ifdef _WIN32
    { LARGE_INTEGER f; QueryPerformanceFrequency(&f); qf = (unsigned long long)f.QuadPart; }
#endif
    const unsigned long long t1 = wp_tsc();
    const double secs = (double)(q - t0q) / (double)qf;
    const double hz = secs > 0.0 ? (double)(t1 - t0tsc) / secs : 1.0;
    const double k = 1000.0 / hz / 240.0;     /* cycles -> ms per present */
    fprintf(stderr, "[wprof] %.1f present/s, %.0f draws/present: draw ms/present: fetch+vcache %.2f, state+tex %.2f, pipeline %.2f, "
            "index+vcstore %.2f, consts+binds+record %.2f, tail %.2f | fifo pass %.2f (%.1f calls) | encode %.2f\n",
                        240.0 / secs, (double)(s_wp_draws - l_draws) / 240.0,
            (s_wp_cyc[0] - l_cyc[0]) * k, (s_wp_cyc[1] - l_cyc[1]) * k, (s_wp_cyc[2] - l_cyc[2]) * k,
            (s_wp_cyc[3] - l_cyc[3]) * k, (s_wp_cyc[4] - l_cyc[4]) * k, (s_wp_cyc[5] - l_cyc[5]) * k,
            (g_wp_fifo_cyc - l_fifo) * k, (double)(g_wp_fifo_calls - l_calls) / 240.0, (g_wp_enc_cyc - l_enc) * k);
    fprintf(stderr, "[wprof2] per present: cmds %.0f, methods %.0f (engine_method ms %.2f incl. draws), upload runs %.0f (%.0f words, %.2f ms), process_method %.2f ms, present total %.2f ms\n",
            (g_wp_cmds - l_cmds) / 240.0, (g_wp_meths - l_meths) / 240.0, (g_wp_c[0] - l_c[0]) * k, (g_wp_runs - l_runs) / 240.0, (g_wp_runw - l_runw) / 240.0, (g_wp_c[1] - l_c[1]) * k, (g_wp_c[2] - l_c[2]) * k, (g_wp_c[3] - l_c[3]) * k);
    for (int i = 0; i < 6; i++) l_c[i] = g_wp_c[i];
    l_meths = g_wp_meths; l_runs = g_wp_runs; l_runw = g_wp_runw; l_cmds = g_wp_cmds;
    fprintf(stderr, "[wprof3] per present: fp_scan %.2f ms (%.0f scans), dc_fetch miss %.2f ms (%.0f misses)\n", (s_wp_cyc[6] - l_cyc[6]) * k, (double)(s_wp_nscan - l_nscan) / 240.0, (s_wp_cyc[7] - l_cyc[7]) * k, (double)(s_wp_nmiss - l_nmiss) / 240.0);
    l_nscan = s_wp_nscan; l_nmiss = s_wp_nmiss;
    {   extern double g_pw_ms[12]; static double l_pw[12];
        fprintf(stderr, "[wprof4] per present ms: submit{pre %.2f allocwait %.2f encode %.2f post %.2f submit %.2f present %.2f inflight %.2f retire %.2f} | drain %.2f pace %.2f be_present %.2f\n",
                (g_pw_ms[0] - l_pw[0]) / 240.0, (g_pw_ms[1] - l_pw[1]) / 240.0, (g_pw_ms[2] - l_pw[2]) / 240.0, (g_pw_ms[3] - l_pw[3]) / 240.0,
                (g_pw_ms[4] - l_pw[4]) / 240.0, (g_pw_ms[5] - l_pw[5]) / 240.0, (g_pw_ms[6] - l_pw[6]) / 240.0, (g_pw_ms[7] - l_pw[7]) / 240.0,
                (g_pw_ms[9] - l_pw[9]) * k / 1.0, (g_pw_ms[10] - l_pw[10]) * k, (g_pw_ms[11] - l_pw[11]) * k);
        for (int i = 0; i < 12; i++) l_pw[i] = g_pw_ms[i]; }
    if (s_wp_on > 1) {
        static unsigned long long l_n2[48];
        const double* kk = &k;
        fprintf(stderr, "[wprof5] per present ms: layout %.2f key %.2f lookup %.2f (arm %.2f hashfallback %.2f; hashed %.0f ok %.0f untrusted %.0f) | state: decode %.2f targets %.2f bindtex %.2f vtex+ssr %.2f | "
                "pipeline: pre %.2f key %.2f find %.2f | p4: vpcb %.2f fpcb %.2f binds %.2f draw %.2f\n",
                (s_wp_cyc[8] - l_cyc[8]) * *kk, (s_wp_cyc[9] - l_cyc[9]) * *kk, (s_wp_cyc[10] - l_cyc[10]) * *kk, (s_wp_cyc[11] - l_cyc[11]) * *kk, (s_wp_cyc[12] - l_cyc[12]) * *kk,
                (double)(s_wp_n2[0] - l_n2[0]) / 240.0, (double)(s_wp_n2[1] - l_n2[1]) / 240.0, (double)(s_wp_n2[2] - l_n2[2]) / 240.0,
                (s_wp_cyc[13] - l_cyc[13]) * *kk, (s_wp_cyc[14] - l_cyc[14]) * *kk, (s_wp_cyc[15] - l_cyc[15]) * *kk, (s_wp_cyc[16] - l_cyc[16]) * *kk,
                (s_wp_cyc[17] - l_cyc[17]) * *kk, (s_wp_cyc[18] - l_cyc[18]) * *kk, (s_wp_cyc[19] - l_cyc[19]) * *kk,
                (s_wp_cyc[20] - l_cyc[20]) * *kk, (s_wp_cyc[21] - l_cyc[21]) * *kk, (s_wp_cyc[22] - l_cyc[22]) * *kk, (s_wp_cyc[23] - l_cyc[23]) * *kk);
        fprintf(stderr, "[wprof6] per present ms: p4 binds: targets %.2f vscb %.2f pscb %.2f tex %.2f vtex %.2f | pipeline pre: cube+vtex %.2f shadow %.2f texop %.2f unnorm %.2f\n",
                (s_wp_cyc[24] - l_cyc[24]) * *kk, (s_wp_cyc[25] - l_cyc[25]) * *kk, (s_wp_cyc[26] - l_cyc[26]) * *kk, (s_wp_cyc[27] - l_cyc[27]) * *kk, (s_wp_cyc[28] - l_cyc[28]) * *kk,
                (s_wp_cyc[30] - l_cyc[30]) * *kk, (s_wp_cyc[31] - l_cyc[31]) * *kk, (s_wp_cyc[32] - l_cyc[32]) * *kk, (s_wp_cyc[33] - l_cyc[33]) * *kk);
        fprintf(stderr, "[wprof7] per present ms, dc_fetch (miss path): hash-indices %.2f decode-refs %.2f remap %.2f plan+prepare+spans %.2f fetch_all %.2f\n",
                (s_wp_cyc[34] - l_cyc[34]) * *kk, (s_wp_cyc[35] - l_cyc[35]) * *kk, (s_wp_cyc[36] - l_cyc[36]) * *kk, (s_wp_cyc[37] - l_cyc[37]) * *kk, (s_wp_cyc[38] - l_cyc[38]) * *kk);
        { extern unsigned long long g_wp2_samp_cyc; static unsigned long long l_samp, l_hits;
          fprintf(stderr, "[wprof8]");
          for (int i = 0; i < 64; i++) if (s_wp_cyc[i] != l_cyc[i]) fprintf(stderr, " c%d=%.4f", i, (s_wp_cyc[i] - l_cyc[i]) * *kk);
          fprintf(stderr, " c49=%.4f", (g_wp2_samp_cyc - l_samp) * *kk);
          for (int i = 0; i < 48; i++) fprintf(stderr, " n%d=%.2f", i, (double)(s_wp_n2[i] - l_n2[i]) / 240.0);
          fprintf(stderr, " fpchit=%.2f\n", (double)(s_fpc_hits - l_hits) / 240.0);
          l_samp = g_wp2_samp_cyc; l_hits = s_fpc_hits; }
        for (int i = 0; i < 48; i++) l_n2[i] = s_wp_n2[i];
    }
    for (int i = 0; i < 64; i++) l_cyc[i] = s_wp_cyc[i];
    l_fifo = g_wp_fifo_cyc; l_enc = g_wp_enc_cyc; l_draws = s_wp_draws; l_calls = g_wp_fifo_calls;
    t0tsc = t1; t0q = q;
}

/* surf54: figures for the [hitch] line (only called when a line is printed). */
static u32 eng_hp_surf_used(void) { u32 n = 0; for (u32 i = 0; i < g.n_surfaces; i++) if (g.surfaces[i].handle) n++; return n; }
static u32 eng_hp_surf_active(void) { u32 n = 0; for (u32 i = 0; i < g.n_surfaces; i++) if (g.surfaces[i].handle && g.surfaces[i].last_frame + 16u > g.frames) n++; return n; }
static unsigned long long eng_hp_vram(int budget)
{
    unsigned long long u = 0, b = 0;
    if (g_rsx_vram_hook && g_rsx_vram_hook(&u, &b)) return (budget ? b : u) >> 20;
    return 0;
}
/* surf54: every 64 presents, the copy-only targets (NV3089 patches) idle for 600 presents are released: they are not
 * recreated unless the game copies to that offset again (and then the recycling above usually finds one), and a table of
 * 512 slots must not fill with them again in a long game. RSX_RT_SWEEP=0 off. */
static void eng_surface_sweep(void)
{
    static int on = -1; if (on < 0) { const char* e = getenv("RSX_RT_SWEEP"); on = !(e && e[0] == '0'); }
    if (!on || (g.frames & 63u) != 0u) return;   /* reg60: was every 16 presents; the DXGI budget query is a driver call on the walker */
    static u32 last_run = ~0u; if (last_run == g.frames) return; last_run = g.frames;
    /* Video memory pressure (the D3D12 backend's hook; DXGI's own budget, which falls when another process uses the GPU): at
     * 85% of the budget or over, everything idle for 10 s goes -- parked variants after 1 s, copy-only targets after 2 s, real
     * targets and depth targets after 10 s -- because the 31/10 game's 2 s freezes and 10 fps stretches came with more video memory
     * in use than the budget (4318 of 3434 MB in the arena run). RSX_RT_SWEEP=0 off. */
    unsigned long long used = 0, budget = 0;
    const int have = g_rsx_vram_hook && g_rsx_vram_hook(&used, &budget) && budget;
    const int hard = have && used * 100ull > budget * 85ull;
    const int soft = have && used * 10ull > budget * 6ull;
    { static int lastp = -1; if (hard != lastp) { lastp = hard;
        fprintf(stderr, "[rt-sweep] frame %u: video memory %llu of %llu MB: %s\n", g.frames, used >> 20, budget >> 20, hard ? "over 85%, releasing idle targets" : "back under 85%"); } }
    if ((g.frames & 63u) == 0u || hard)
    for (u32 i = 0; i < g.n_surfaces; i++) {
        const eng_surface* t = &g.surfaces[i];
        if (t->handle && t->blit_only && i != g.last_present_surface && t->last_frame + (hard ? 120u : 600u) < g.frames)
            eng_surface_free(i);
    }
    if (hard) {
        for (u32 i = 0; i < g.n_surfaces; i++) {
            const eng_surface* t = &g.surfaces[i];
            if (t->handle && i != g.last_present_surface && t->last_frame + 600u < g.frames) eng_surface_free(i);
        }
        for (u32 i = 0; i < g.n_zdepths; i++) {
            const eng_zdepth* z = &g.zdepths[i];
            if (z->handle && z->last_frame + 600u < g.frames) eng_zdepth_free(i);
        }
    }
    /* The parked size variants of a target (up to 3 per offset, kept for a size that comes back) are what grows the colour
     * targets with the minutes: 568 objects, 1.15 GB after 2 minutes of the arena at 4K, depth targets another 1.2 GB, 5 of 7.25 GB
     * in use -- and the 2 s freezes of the 31/10 game were seen with the budget down to 3-4 GB. Parks idle for 20 s are released
     * (5 s when the video memory in use is above 60% of the budget). They are not recreated unless that size comes back. */
    {
      const u32 idle = hard ? 60u : soft ? 300u : 1200u;
      for (u32 i = 0; i < g.n_surfaces; i++) {
          eng_surface* t = &g.surfaces[i];
          if (!t->handle || !t->park_n || i == g.last_present_surface || t->last_frame + idle >= g.frames) continue;
          for (u32 k = 0; k < t->park_n; k++) eng_rt_destroy(t->park[k].handle);
          t->park_n = 0;
      } }
}
static void eng_present_impl(u32 buffer_id);
static void eng_present(u32 buffer_id)
{
    eng_surface_sweep();
    extern double g_pw_ms[12];
    if (g_wp_fl > 0) { const unsigned long long a0_ = wp_tsc(); rsx_draw_engine_b2_drain(); g_pw_ms[9] += (double)(wp_tsc() - a0_); }   /* cycles here, scaled in the report */
    else rsx_draw_engine_b2_drain();   /* RSX_DRAW_SPLIT=2: every queued draw is in the records before the frame is presented */
    if (g_wp_fl > 0) { const unsigned long long a_ = wp_tsc(); eng_present_impl(buffer_id); g_wp_c[3] += wp_tsc() - a_; }
    else eng_present_impl(buffer_id);
}
static void eng_present_impl(u32 buffer_id)
{
    if (!g.ready) return;
    const u32 target = eng_present_surface(buffer_id);
    if (target == ENG_INVALID) return;
    { extern volatile int g_tl_on; extern void tl_rsx(int, int, uint64_t, uint64_t); extern void tl_present_tick(void);   /* TOD_TIMELINE */
      tl_present_tick(); if (g_tl_on) tl_rsx(3, 'B', 0, 0); }
    { extern double g_pw_ms[12]; const unsigned long long a_ = g_wp_fl > 0 ? wp_tsc() : 0; rsx_flip_pace(); if (g_wp_fl > 0) g_pw_ms[10] += (double)(wp_tsc() - a_); }
    { extern volatile int g_tl_on; extern void tl_rsx(int, int, uint64_t, uint64_t); if (g_tl_on) tl_rsx(3, 'E', 0, 0); }
    { extern void dt_diag_report(unsigned); static unsigned fl; dt_diag_report(++fl); }   /* TOD_DT_DIAG */
    { /* TOD_PIPE_DIAG=1: one line per host flip. gap = ms since the previous flip; held/pwait = the 60 Hz pace;
       * idle = walker thread asleep waiting for work; busy = gap - idle - pwait; q = PPU flips issued minus
       * flips the walker has reached (frames of slack between PPU and walker); ppu_age = ms since the PPU's
       * last flip request; gpu/lim = the guest's RSX-sync usleeps / frame-limiter usleeps in the gap
       * (calls/ms); ring = ms in the ring-recycle wait. */
      static int diag = -1; static long long last; static unsigned n, last_n_ppu;
      static long long last_idle, last_gu[2], last_ring; static long last_gn[2];
      if (diag < 0) { const char* e = getenv("TOD_PIPE_DIAG"); diag = (e && e[0] == '1'); }
      if (diag) {
          extern volatile long long g_pipe_usl_us[2], g_pipe_ring_us, g_pipe_ppu_flip_us, g_pipe_idle_us;
          extern volatile long g_pipe_usl_n[2]; extern volatile unsigned g_pipe_ppu_flips;
          extern unsigned long long ps3_qpc_us(void);
          const long long now_us = (long long)ps3_qpc_us(); n++;
          const long long idle = g_pipe_idle_us, gu0 = g_pipe_usl_us[0], gu1 = g_pipe_usl_us[1], ring = g_pipe_ring_us;
          const long gn0 = g_pipe_usl_n[0], gn1 = g_pipe_usl_n[1];
          if (last) {
              const double gap = (now_us - last) / 1000.0, id = (idle - last_idle) / 1000.0;
              fprintf(stderr, "[pipe] n=%u t=%.1f gap=%.1f held=%d pwait=%.1f idle=%.1f busy=%.1f q=%d ppu_age=%.1f gpu=%ld/%.1f lim=%ld/%.1f ring=%.1f\n",
                      n, now_us / 1000.0, gap, s_pace_held, s_pace_wait_ms, id, gap - id - s_pace_wait_ms,
                      (int)(g_pipe_ppu_flips - n), (now_us - g_pipe_ppu_flip_us) / 1000.0,
                      gn0 - last_gn[0], (gu0 - last_gu[0]) / 1000.0, gn1 - last_gn[1], (gu1 - last_gu[1]) / 1000.0,
                      (ring - last_ring) / 1000.0);
          }
          last = now_us; last_idle = idle; last_gu[0] = gu0; last_gu[1] = gu1; last_gn[0] = gn0; last_gn[1] = gn1; last_ring = ring;
      } }
    if (getenv("PS3RECOMP_METAL_FRAME_DUMP") && g.frames % 120 == 0) {
        u32 current = eng_current_surface();
        fprintf(stderr, "[rsx capture] frame=%u buffer=%u selected=%u current=%u draws=%u surfaces=%u\n",
                g.frames, buffer_id, target, current, g.guest_draws, g.n_surfaces);
        for (u32 i = 0; i < g.n_surfaces; ++i)
            fprintf(stderr, "[rsx capture] surface %u loc=%u offset=%08X size=%ux%u handle=%u\n",
                    i, g.surfaces[i].location, g.surfaces[i].offset,
                    g.surfaces[i].w, g.surfaces[i].h, g.surfaces[i].handle);
    }
    { static u32 nss; if (eng_ssr_on() && (++nss % 600) == 0) fprintf(stderr, "[ssr] frame %u: %u draws bound, %u snapshots, mode act %d dbg %d\n", g.frames, s_ssr_stat_bound, s_ssr_stat_snaps, s_ssr_act, s_ssr_dbg); }
    eng_ssr_frame_mode();
    g.last_present_surface = target;
    { extern double g_pw_ms[12]; const unsigned long long a_ = g_wp_fl > 0 ? wp_tsc() : 0; g.be->present(g.be->user, g.surfaces[target].handle); if (g_wp_fl > 0) g_pw_ms[11] += (double)(wp_tsc() - a_); }
    g_rsx_frame_draws = s_frame_draws; s_frame_draws = 0;
    s_frame_video = 0;
    /* DOD3_STUTTER_MS=<n>: a frame longer than n ms says what the walker did
     * in it -- texture uploads, vertex conversions stored, pipelines built --
     * beside the guest threads' report ([stutter], ppu_hle.cpp). */
    { static int thr = -1; static double last;
      if (thr < 0) { const char* e = getenv("DOD3_STUTTER_MS"); thr = e ? atoi(e) : 0; }
      if (thr > 0) {
          struct timespec ts; timespec_get(&ts, TIME_UTC);
          const double now = (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
          if (last > 0 && now - last > thr)
              { extern double g_jc_walk_ms; extern unsigned g_jc_walks;
              fprintf(stderr, "[stutter-rsx] frame %u took %.0f ms: %u textures uploaded (%llu KB), "
                      "%u vertex conversions stored (%llu KB), %u pipelines built (%.0f ms), "
                      "%u cache evictions (%u entries, %.1f ms), %u draws in %.1f ms of walker; "
                      "%u job-chain walks, %.1f ms, on the render thread\n",
                      g.frames, now - last, s_fstat.tex, (unsigned long long)(s_fstat.tex_bytes >> 10),
                      s_fstat.vc_store, (unsigned long long)(s_fstat.vc_bytes >> 10),
                      s_fstat.pipes, s_fstat.pipe_ms, s_fstat.vc_evicts, s_fstat.vc_evicted, s_fstat.vc_evict_ms,
                      s_fstat.draws, s_fstat.draw_ms, g_jc_walks, g_jc_walk_ms); }
          last = now;
          memset(&s_fstat, 0, sizeof s_fstat);
          { extern double g_jc_walk_ms; extern unsigned g_jc_walks; g_jc_walk_ms = 0; g_jc_walks = 0; }
      } }
    /* Every 5 s: presents per second and frame times, worst included -- the
     * numbers a player feels. RSX_FRAMETIME=0 turns the line off. */
    { static int on = -1; static double win_start, last, worst, first; static u32 n, slow;
      if (on < 0) { const char* e = getenv("RSX_FRAMETIME"); on = !(e && e[0] == '0'); }
      if (on) {
          struct timespec ts; timespec_get(&ts, TIME_UTC);
          const double now = (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
          if (last > 0.0) {
              const double ft = (now - last) * 1000.0;
              if (ft > worst) worst = ft;
              if (ft > 34.0) slow++;
              if (ft > 25.0) g_rsx_engine_hitches++;
              /* RSX_HITCH_LOG=<ms>: each present that came later than that,
               * with its frame number -- nothing else switched on, so the
               * log does not make hitches of its own. */
              { static double hl = -1.0;
                if (hl < 0.0) { const char* e = getenv("RSX_HITCH_LOG"); hl = e ? atof(e) : 0.0; }
                if (hl > 0.0 && ft > hl && g_rsx_hitch_hook) g_rsx_hitch_hook(ft);
                /* With the walker's blocking GPU waits in the frame (the
                 * D3D12 backend adds them up; 0 elsewhere). */
                if (hl > 0.0 && ft > hl)
                    fprintf(stderr, "[hitch] frame %u: %.1f ms (walker blocked on the GPU %.1f ms of it; "
                            "%u textures uploaded %llu KB in %.1f ms (%.1f creating them, %.1f decoding, %.1f staging), "
                            "%u vertex conversions %llu KB (%.1f ms into buffers), %u pipelines built %.1f ms, "
                            "%u draws %.1f ms, pipeline states %.1f ms; "
                            "walker: draws %.1f ms in %u, fifo pass %.1f ms; "
                            "targets: colour +%u %.1f ms -%u %.1f ms (recycled %u), depth +%u %.1f ms -%u, nv3089 copies %u %.1f ms; "
                            "table: %u of %u slots (%u active), refused %u now %u total, zeta %u of %u; "
                            "locks: pc_lock %.1f ms (%u), pso-save %.1f ms (%u), other %.1f ms (%u), enc-join %.1f ms; vram %llu of %llu MB)\n",
                            g.frames, ft, g_rsx_frame_gpu_wait_ms,
                            s_fstat.tex, (unsigned long long)(s_fstat.tex_bytes >> 10), s_fstat.tex_ms, s_fstat.tex_create_ms,
                            s_fstat.tex_decode_ms, s_fstat.tex_stage_ms,
                            s_fstat.vc_store, (unsigned long long)(s_fstat.vc_bytes >> 10), s_fstat.vc_wrap_ms,
                            s_fstat.pipes, s_fstat.pipe_ms, s_fstat.draws, s_fstat.draw_ms, g_rsx_frame_pso_ms,
                            hp_cyc_ms(s_hp.draw_cyc), s_hp.draws, hp_cyc_ms(g_wp_fifo_cyc - s_hp.fifo0),
                            s_hp.rt_c_n, s_hp.rt_c_ms, s_hp.rt_d_n, s_hp.rt_d_ms, s_hp.rt_recycled,
                            s_hp.zd_c_n, s_hp.zd_c_ms, s_hp.zd_d_n, s_hp.blit_n, s_hp.blit_ms,
                            eng_hp_surf_used(), eng_surf_cap(), eng_hp_surf_active(), s_hp.refused, s_surf_refused_total,
                            g.n_zdepths, eng_zdepth_cap(),
                            g_rsx_frame_pclock_ms, g_rsx_frame_pclock_n, g_rsx_frame_pcsave_ms, g_rsx_frame_pcsave_n,
                            g_rsx_frame_lock_ms, g_rsx_frame_lock_n, g_rsx_frame_join_ms,
                            eng_hp_vram(0), eng_hp_vram(1));
                g_rsx_frame_gpu_wait_ms = 0.0; g_rsx_frame_pso_ms = 0.0;
                if (hl > 0.0) {   /* surf54: the per-frame accumulators the line above reads */
                    s_hp.draw_cyc = 0; s_hp.draws = 0; s_hp.fifo0 = g_wp_fifo_cyc;
                    s_hp.rt_c_n = s_hp.rt_d_n = s_hp.zd_c_n = s_hp.zd_d_n = s_hp.rt_recycled = s_hp.refused = s_hp.blit_n = 0;
                    s_hp.rt_c_ms = s_hp.rt_d_ms = s_hp.zd_c_ms = s_hp.blit_ms = 0.0;
                    g_rsx_frame_pclock_ms = g_rsx_frame_pcsave_ms = g_rsx_frame_lock_ms = g_rsx_frame_join_ms = 0.0;
                    g_rsx_frame_pclock_n = g_rsx_frame_pcsave_n = g_rsx_frame_lock_n = 0;
                }
                /* The stutter report resets these per frame; without it, here. */
                { static int st = -1; if (st < 0) st = getenv("DOD3_STUTTER_MS") ? 1 : 0;
                  if (!st) memset(&s_fstat, 0, sizeof s_fstat); } }
              n++;
          } else {
              win_start = first = now;
          }
          last = now;
          if (now - win_start >= 5.0 && n) {
              const double span = now - win_start;
              fprintf(stderr, "[frametime] t=%.0fs %.1f fps, mean %.1f ms, worst %.1f ms, %u of %u frames over 34 ms",
                      now - first, n / span, span * 1000.0 / n, worst, slow, n);
              /* The texture write-watch's work in the window, when it runs. */
              if (eng_tex_watch_on()) {
                  static uint64_t f0;
                  static unsigned long long s0;
                  uint64_t f = 0, pp = 0;          /* vm_watch_stats takes uint64_t: unsigned long on Linux */
                  vm_watch_stats(&f, &pp);
                  fprintf(stderr, "; watch: %llu faults, %llu hashes saved, %llu pages protected in all",
                          (unsigned long long)(f - f0), s_tex_skips_total - s0, (unsigned long long)pp);
                  f0 = f; s0 = s_tex_skips_total;
              }
              fputc('\n', stderr);
              win_start = now; worst = 0.0; n = 0; slow = 0;
          }
      } }
    g.q_unflushed = 0;   /* the present's command buffer carries them */
    eng_surface_dump_frame();
    /* RSX_PRESENT_LOG=<from frame>: one line per present from that frame on:
     * the guest's buffer id, the surface it resolved to, the draws each
     * 1280x720 colour surface received since the last present, and the mean
     * brightness of what was presented. The dump strides of 120 and 240
     * frames are multiples of a three-buffer rotation, so they could not
     * tell "one buffer in three is white" from "a white episode". */
    { static long from = -2; if (from == -2) { const char* e = getenv("RSX_PRESENT_LOG"); from = e ? atol(e) : -1; }
      if (from >= 0 && (long)g.frames >= from) {
          double mean = -1.0;
          const eng_surface* ps = &g.surfaces[target];
          if (ps->fmt == RSX_BE_FMT_R8G8B8A8 && ps->w && ps->h) {
              u8* buf = (u8*)malloc((size_t)ps->w * ps->h * 4);
              if (buf) { g.be->readback(g.be->user, ps->handle, 0, 0, ps->w, ps->h, buf, ps->w * 4);
                  unsigned long long sum = 0; u32 n = 0;
                  for (u32 p = 0; p < ps->w * ps->h; p += 61) { sum += buf[p * 4] + buf[p * 4 + 1] + buf[p * 4 + 2]; n += 3; }
                  mean = n ? (double)sum / n : 0.0; free(buf); } }
          fprintf(stderr, "[present] f=%u buffer=%u -> s%u(off=%08X) %s mean=%.0f skin=%u/%u draws:", g.frames, buffer_id, target, ps->offset, s_present_src, mean, s_skin_draws, s_skin_verts);
          for (u32 i = 0; i < g.n_surfaces; i++)
              if (s_surf_draws[i]) fprintf(stderr, " s%u=%u", i, s_surf_draws[i]);
          fputc('\n', stderr);
      }
      /* RSX_TRACE_ON_HUD=1: arm the four-frame draw trace twice, on the first present whose
       * target took <=8 draws (HUD hidden) and on the first later one that took >=20 (HUD up). */
      { static int hud = -1, stage = 1; static u32 af = 0; if (hud < 0) hud = getenv("RSX_TRACE_ON_HUD") ? 1 : 0;
        if (hud && stage < 3 && g.frames > 9000 && s_surf_draws[7] >= 15) {
            const u32 nd = s_surf_draws[0] + s_surf_draws[1];
            if ((stage == 0 && nd <= 8) || (stage == 1 && nd >= 20) || (stage == 2 && nd >= 20 && g.frames > af + 120)) {
                s_dtrace_frame = (long)g.frames + 1; stage++; af = g.frames;
                fprintf(stderr, "[hudtrace] stage %d armed at f%u (target s%u drew %u)\n", stage, g.frames, target, nd); } } }
      memset(s_surf_draws, 0, sizeof s_surf_draws); s_skin_draws = s_skin_verts = 0; }
    /* RSX_TRACE_ON_WHITE=1: read the presented frame back; when it comes out
     * (nearly) all white, trace the next four frames draw by draw. */
    if (s_ring_on > 0) { char line[120];
        snprintf(line, sizeof line, "f%u PRESENT buffer=%u -> s%u(off=%08X)", g.frames, buffer_id, target, g.surfaces[target].offset);
        eng_ring_push(line); }
    { static long on = -1; static int cr = -1, cg = 0, cb = 0, ctol = 12;
      if (on < 0) { const char* e = getenv("RSX_TRACE_ON_WHITE"); on = e ? atol(e) : 0;
          /* RSX_TRACE_ON_COLOR=r,g,b[,tol]: the same trigger for a frame whose
           * mean colour is within tol of r,g,b. The all-green Bink cutscene
           * is (0,135,0): the YUV->RGB of three planes that read as zero. */
          if ((e = getenv("RSX_TRACE_ON_COLOR"))) {
              if (sscanf(e, "%d,%d,%d,%d", &cr, &cg, &cb, &ctol) >= 3) { if (!on) on = 1; } else cr = -1; } }
      /* The value is the first frame to watch from (a title's own fade-from-
       * white at a level start is not the bug); each episode traces four
       * frames and the watch re-arms 600 frames later. */
      static int armed = 0; static u32 armed_at = 0;
      if (armed && g.frames > armed_at + 600u) armed = 0;
      if (on > 0 && (long)g.frames >= on && !armed && g.surfaces[target].fmt == RSX_BE_FMT_R8G8B8A8 && g.surfaces[target].w && g.surfaces[target].h) {
          const u32 w = g.surfaces[target].w, h = g.surfaces[target].h;
          u8* buf = (u8*)malloc((size_t)w * h * 4);
          if (buf) {
              g.be->readback(g.be->user, g.surfaces[target].handle, 0, 0, w, h, buf, w * 4);
              unsigned long long sr = 0, sg = 0, sb = 0; const u32 step = 61;
              u32 n = 0;
              for (u32 p = 0; p < w * h; p += step) { sr += buf[p * 4]; sg += buf[p * 4 + 1]; sb += buf[p * 4 + 2]; n++; }
              const double mr = n ? (double)sr / n : 0.0, mg = n ? (double)sg / n : 0.0, mb = n ? (double)sb / n : 0.0;
              const double mean = (mr + mg + mb) / 3.0;
              const int white = mean > 250.0;
              const int colour = cr >= 0 && mr >= cr - ctol && mr <= cr + ctol && mg >= cg - ctol && mg <= cg + ctol &&
                                 mb >= cb - ctol && mb <= cb + ctol;
              if (white || colour) {
                  eng_ring_dump(white ? "white present" : "colour present");
                  armed = 1; armed_at = g.frames; s_dtrace_frame = (long)g.frames + 1;
                  fprintf(stderr, "[draw-trace] frame %u presented %s (mean %.0f,%.0f,%.0f): tracing frames %ld..%ld\n",
                          g.frames, white ? "WHITE" : "the watched colour", mr, mg, mb, s_dtrace_frame, s_dtrace_frame + 3);
              }
              free(buf);
          }
      } }
    /* RSX_TRACE_BUSY=<draws>[,<from frame>]: trace the four frames after the
     * first one (from that frame on) that issued more than <draws> draws --
     * a gameplay frame, wherever the run's timing put it. Once per run. */
    { static long busy = -2, bfrom = 0; static int done = 0;
      if (busy == -2) { const char* e = getenv("RSX_TRACE_BUSY"); busy = -1;
          if (e) { char* d; busy = strtol(e, &d, 0); if (*d == ',') bfrom = strtol(d + 1, 0, 0); } }
      if (busy > 0 && (g.frames % 250) == 0)
          fprintf(stderr, "[draw-trace] busy watch: frame %u guest_draws=%u\n", g.frames, g.guest_draws);
      if (busy > 0 && !done && (long)g.frames >= bfrom && (long)g.guest_draws > busy) {
          done = 1; s_dtrace_frame = (long)g.frames + 1;
          fprintf(stderr, "[draw-trace] frame %u issued %u draws: tracing frames %ld..%ld\n",
                  g.frames, g.guest_draws, s_dtrace_frame, s_dtrace_frame + 3); } }
    /* Frame capture (rsx_capture.h): starts, counts and ends at presents. */
    { double cmean = -1.0;
      if (rsx_capture_wants_mean(g.frames)) {
          const eng_surface* ps = &g.surfaces[target];
          if (ps->fmt == RSX_BE_FMT_R8G8B8A8 && ps->w && ps->h) {
              u8* buf = (u8*)malloc((size_t)ps->w * ps->h * 4);
              if (buf) { g.be->readback(g.be->user, ps->handle, 0, 0, ps->w, ps->h, buf, ps->w * 4);
                  unsigned long long sum = 0; u32 n = 0;
                  for (u32 p = 0; p < ps->w * ps->h; p += 61) { sum += buf[p * 4] + buf[p * 4 + 1] + buf[p * 4 + 2]; n += 3; }
                  cmean = n ? (double)sum / n : 0.0; free(buf); } } }
      u32 db[8][6];
      for (u32 i = 0; i < 8; i++) {
          const eng_display_buffer* d = &g.display_buffers[i];
          db[i][0] = (u32)d->valid; db[i][1] = d->location; db[i][2] = d->offset;
          db[i][3] = d->pitch; db[i][4] = d->width; db[i][5] = d->height; }
      rsx_capture_present(g.frames, cmean, g.rsx.regs, RSX_DSP_NUM_REGS, g.rsx.vp, RSX_DSP_VP_WORDS,
                          &g.rsx.constants[0][0], RSX_DSP_NUM_CONSTANTS * 4u, &db[0][0], 8); }
    { extern volatile uint32_t g_ppu_frame_counter; g_ppu_frame_counter = g.frames + 1u; }
    g.frames++;
    eng_pipe_frame_end();
    g_rsx_engine_frame = g.frames;
    eng_draw_stats_report();
    eng_wprof_report();
    g.last_guest_draws = g.guest_draws;
    g.guest_draws = 0;
}

static void sink_flip(void* user, const rsx_dispatch* r, u32 arg)
{
    (void)user; (void)r;
    rsx_draw_engine_b2_drain();
    g.rsx.canvas_mark_at = 0;   /* a mark the frame's last draws left: not the next frame's */
    g.last_flip_buffer = arg & 7u;
    g.sink_flips++;
    s_present_src = "fifo";
    eng_present(g.last_flip_buffer);
    s_present_src = "host";
}

/* A title whose FIFO carries its flips -- the 0xE944 method, or the
 * 0xFEADxxxx word libgcm's flip and prepare-flip commands write -- presents
 * through sink_flip or rsx_draw_engine_fifo_flip, in order with its draws.
 * Once that has happened the host clock's presents are not harmless repeats:
 * by the time the clock runs, the drain may have gone through several more
 * frames of commands, and the buffer it names then holds a later frame's
 * clear. Drakengard 3 clears each display buffer to white for its
 * light-attenuation pass before drawing into it, so those late presents
 * flashed white, cyan (a shadow mask) and half-built frames between the
 * real ones, and its characters and distant buildings seemed to flicker. */
static int eng_host_present_allowed(void)
{
    if (!g.sink_flips) return 1;
    static int said = 0;
    if (!said++) fprintf(stderr, "[rsx engine] host presents ignored from now on: this title flips through its FIFO\n");
    return 0;
}

/* ---- public API ---------------------------------------------------------- */

void rsx_draw_engine_set_backend(const rsx_draw_backend* backend)
{
    g.be = backend;
}

void rsx_draw_engine_set_default(int on)
{
    g.default_on = on;
}

extern int g_rsx_query_nosync;
static int eng_b2_post_query(u32 m, u32 arg);
/* Occlusion-query methods (CLEAR_REPORT_VALUE 0x17C8, GET_REPORT 0x1800, the 0x1D70 fence): touch g.q_* and the backend, so with
 * RSX_DRAW_SPLIT=2 they run on B2, in order with the draws (control job), or on the caller after a drain. */
static void eng_query_method(u32 m, u32 arg)
{
    if (m == 0x17C8u && arg == 1u && g.be->query_begin) {
        g.q_cur = g.be->query_begin(g.be->user);
        g.q_attempts = g.q_draws = 0;
    } else if (m == 0x1800u && (arg >> 24) == 1u && g.q_cur) {
        const u32 index = (arg & 0xFFFFFFu) / 16u;
        if (g.q_draws && g.q_draws >= g.q_attempts) { g.be->query_report(g.be->user, g.q_cur, index); g.q_unflushed++; }
        else rsx_draw_engine_query_result(index, 0xFFFFu);
        g.q_cur = 0;
        if (g.be->query_set) g.be->query_set(g.be->user, 0);
    } else if (m == 0x1D70u && g.q_unflushed && g.be->submit_and_wait) {
        /* BACK_END_WRITE_SEMAPHORE_RELEASE after occlusion queries: the title
         * reads a fence the GPU writes once it is past the query block, then
         * trusts the report values. The walker writes the fence as soon as it
         * reaches it, long before the counts come back, so Drakengard 3 read
         * last frame's values -- another object's, since report indices are a
         * per-frame pool -- and culled whatever drew a 0: characters and props
         * flickered out for single frames. Finish the GPU work up to here and
         * deliver the counts before the fence lands. RSX_QUERY_NOSYNC=1 off. */
        if (g_rsx_query_nosync < 0) g_rsx_query_nosync = getenv("RSX_QUERY_NOSYNC") ? 1 : 0;
        if (!g_rsx_query_nosync) g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_QUERY_FENCE);
        g.q_unflushed = 0;
    }
}

/* ======================================================================================================
 * RSX_DRAW_SPLIT=2 (27/10 walkF3, WALKER_PARALLELO.md phase 3 step 1): thread B2 runs sink_end_impl on a queue of
 * DrawSnaps; the dispatcher (D, the walker thread) only snapshots. Rules:
 *   - D owns g.rsx, dd, g_rsx_dsp_vtx_gen, s_vp_branch_bits_reg. B2 owns EVERYTHING else of `g`, the backend, dc.
 *   - Anything on D (or any other thread) that touches `g`/the backend, or that makes guest memory observable as
 *     "consumed" (labels, get, ref, flip, 2D copies, blit, readback, queries), first waits for B2 to be idle
 *     (rsx_draw_engine_b2_drain) or, for fences, for the draws queued before them (rsx_draw_engine_b2_wait).
 *   - Per draw the job carries: DrawSnap (whitelisted registers, VP copy, constants delta, vtx_gen), a copy of the
 *     batches, the inline-array bytes and s_vp_branch_bits_reg.
 * ====================================================================================================== */
#ifdef _WIN32
#define B2_SLOTS 256
typedef struct EngJob {
    DrawSnap snap;
    eng_batch arr[ENG_MAX_BATCHES];
    eng_batch idx[ENG_MAX_BATCHES];
    u32 n_arr, n_idx, n_packets, inl_bytes;
    unsigned vp_branch;
    u8 inl[RSX_DSP_INLINE_MAX_BYTES];
    u32 kind, c_a, c_b;   /* 0 = draw; 1 = ordered guest write32 (c_a = address, c_b = value); 2 = query method (c_a = m, c_b = arg) */
    volatile LONG pstate;   /* prep: 3 filling, 0 ready, 1 claimed, 2 done (RSX_DRAW_WORKERS) */
    unsigned long long seq_id;
    EngPre pre;
} EngJob;
static EngJob* s_b2_jobs;
static unsigned long long s_b2_prod, s_b2_cons;   /* __atomic; prod written by D only, cons by B2 only */
static HANDLE s_b2_ev, s_b2_thr;
static int s_b2_sleeping, s_b2_stop, s_b2_started;
static __declspec(thread) int t_on_b2;
static int s_b2_on = -1;
/* statistics (RSX_WPROF=1) */
static volatile long long s_b2_busy_q, s_b2_idle_q, s_b2_full_q, s_b2_drain_q;
static volatile unsigned s_b2_njobs, s_b2_ndrain, s_b2_nblocked, s_b2_nfull, s_b2_nwait, s_b2_nwait_blk;
static volatile unsigned s_b2_maxq;
static volatile unsigned s_b2_nctrl;
static DWORD s_b2_prod_tid;       /* thread that queues draws: only it may queue control jobs (SPSC) */
static int s_b2_ctrl_on = -1;     /* RSX_B2_CTRL=0: control jobs off (drain + direct write as in walkF3) */
#define B2_SITES 24
static struct { const char* f; int line; unsigned n, nblk; double us; } s_b2_site[B2_SITES];

static long long b2_qpc(void) { LARGE_INTEGER q; QueryPerformanceCounter(&q); return q.QuadPart; }

static void b2_exec(EngJob* j)
{
    if (j->kind == 1) { extern void vm_write32(unsigned, unsigned); RSX_B2P_SET(B2P_CTRLW); vm_write32(j->c_a, j->c_b); s_b2_nctrl++; return; }
    if (j->kind == 2) { RSX_B2P_SET(B2P_CTRLQ); eng_query_method(j->c_a, j->c_b); s_b2_nctrl++; return; }
    s_b2_njobs++;
    const double t0 = s_wk_fast ? 0.0 : eng_now_ms();
    static unsigned hp_k_; const unsigned long long hp_c0 = (hp_on() && ((++hp_k_) & 7u) == 0u) ? wp_tsc() : 0ull;   /* reg60: 1 draw in 8 is timed (the rest costs nothing) */
    s_progs.valid = 0;
    s_progs.tex_valid = 0;
    s_draw_open = 1;
    rsx_dispatch* view = snap_view();
    snap_apply(&j->snap, view);
    dc_load(j->arr, j->n_arr, j->idx, j->n_idx, j->n_packets, j->inl_bytes ? j->inl : NULL, j->inl_bytes);
    s_vp_bb_draw = j->vp_branch;
    s_draw_vtx_gen = j->snap.vtx_gen;
    g.rv = view;
    sink_end_impl(NULL, NULL);
    g.rv = &g.rsx;
    s_draw_open = 0;
    s_fstat.draws++; if (!s_wk_fast) s_fstat.draw_ms += eng_now_ms() - t0;
    if (hp_c0) { s_hp.draw_cyc += (wp_tsc() - hp_c0) * 8ull; s_hp.draws += 8; }
    s_frame_draws++;
}

/* ---- RSX_DRAW_WORKERS=N (27/10 walkF3c): prep stage of the 2-stage pipeline -------------------------------------------------------
 * N threads (lane w takes jobs w, w+N, ...) run, for each draw job, the part of sink_end_impl that depends on nothing but the job's snapshot
 * and guest memory: vertex layout, the vertex-cache key and the cache lookup/validation (shared lock, see eng_vc_lookup_pre). Result: EngPre in
 * the job. B2 (commit, in queue order) uses it when the entry is still the one validated, otherwise runs its normal serial path; a job no
 * worker has started when B2 reaches it is prepped by B2 itself (so N workers that lag never stall it). Per-thread state: t_rv/t_di, s_vpi,
 * s_vkm, s_fp, s_draw_vtx_gen (all ENG_TLS). Everything that touches the backend, records, surfaces, pipelines stays on B2. */
typedef struct { rsx_dispatch* view; u32 vp_ver; unsigned lane; } WkCtx;
static WkCtx* s_wk_ctx;                 /* [s_wk_n + 1]; the last one is B2's own (stealing) */
static HANDLE s_wk_thr[8], s_wk_sem;
static volatile LONG s_wk_sleepers;
static volatile unsigned s_wk_nsteal, s_wk_njob;

static void wk_prep(WkCtx* x, EngJob* j)
{
    j->pre.hit = 0;
    if (j->kind != 0) return;
    if (eng_vc_checking() || eng_slow() || !eng_vc_enabled() || g_rsx_capture_on) return;
    if (!x->view) {
        x->view = snap_view_new(); x->vp_ver = 0xFFFFFFFFu;
        if (!x->view) return;
    }
    snap_apply_prep(&j->snap, x->view, &x->vp_ver);
    DrawIn di = { j->arr, j->n_arr, j->idx, j->n_idx, j->inl_bytes ? j->inl : NULL, j->inl_bytes };
    t_rv = x->view; t_di = &di;
    const int was_open = s_draw_open; const u32 was_gen = s_draw_vtx_gen;
    s_draw_open = 0; s_draw_vtx_gen = j->snap.vtx_gen;
    const u32 prim = RV->current_primitive;
    int scratch = 0;
    const int rebuild = eng_topology_rebuild(prim, 0, &scratch) != 0;
    int ok = 1;
    if (!rebuild) {
        if (rsx_primitive_needs_expansion(prim)) ok = 0;
        else if (rsx_primitive_topology(prim) == RSX_TOPOLOGY_UNSUPPORTED) ok = 0;
    }
    if (ok) {
        rsx_vertex_layout_plan layout;
        eng_vertex_layout(&layout);
        const u64 key = eng_vc_key(&layout, prim, rebuild);
        u64 eg = 0;
        if (key && eng_vc_lookup_pre(key, &eg)) { j->pre.hit = 1; j->pre.key = key; j->pre.egen = eg; }
    }
    s_draw_open = was_open; s_draw_vtx_gen = was_gen;
    t_rv = NULL; t_di = NULL;
}

static DWORD WINAPI wk_main(LPVOID arg)
{
    WkCtx* x = (WkCtx*)arg;
    const unsigned long long N = (unsigned long long)s_wk_n;
    unsigned long long n = x->lane;
    int idle = 0;
    for (;;) {
        if (__atomic_load_n(&s_b2_stop, __ATOMIC_ACQUIRE)) break;
        const unsigned long long cons = __atomic_load_n(&s_b2_cons, __ATOMIC_ACQUIRE);
        if (n < cons) n += ((cons - n + N - 1) / N) * N;
        const unsigned long long prod = __atomic_load_n(&s_b2_prod, __ATOMIC_ACQUIRE);
        if (n >= prod) {
            if (++idle < 4000) { YieldProcessor(); continue; }
            InterlockedIncrement(&s_wk_sleepers);
            _ReadWriteBarrier(); MemoryBarrier();
            if (__atomic_load_n(&s_b2_prod, __ATOMIC_ACQUIRE) <= n && !__atomic_load_n(&s_b2_stop, __ATOMIC_ACQUIRE))
                WaitForSingleObject(s_wk_sem, 1);
            InterlockedDecrement(&s_wk_sleepers);
            idle = 0;
            continue;
        }
        idle = 0;
        EngJob* j = &s_b2_jobs[n % B2_SLOTS];
        if (__atomic_load_n(&j->pstate, __ATOMIC_ACQUIRE) == 0 && InterlockedCompareExchange(&j->pstate, 1, 0) == 0) {
            if (j->seq_id == n) { wk_prep(x, j); s_wk_njob++; __atomic_store_n(&j->pstate, 2, __ATOMIC_RELEASE); }
            else __atomic_store_n(&j->pstate, 0, __ATOMIC_RELEASE);   /* the slot was recycled for a later job: not ours */
        }
        n += N;
    }
    return 0;
}

/* B2, before committing job j: its prep is done (by a worker), or B2 does it now. */
static void wk_commit_wait(EngJob* j)
{
    for (;;) {
        const LONG st = __atomic_load_n(&j->pstate, __ATOMIC_ACQUIRE);
        if (st == 2) return;
        if (st == 0 && InterlockedCompareExchange(&j->pstate, 1, 0) == 0) {
            wk_prep(&s_wk_ctx[s_wk_n], j); s_wk_nsteal++;
            __atomic_store_n(&j->pstate, 2, __ATOMIC_RELEASE);
            return;
        }
        YieldProcessor();
    }
}

static void wk_start(void)
{
    const char* e = getenv("RSX_DRAW_WORKERS");
    int n = e ? atoi(e) : 0;
    if (n < 0) n = 0;
    if (n > 8) n = 8;
    if (!n) return;
    s_wk_ctx = (WkCtx*)calloc((size_t)n + 1, sizeof(WkCtx));
    s_wk_sem = CreateSemaphoreA(NULL, 0, 1 << 20, NULL);
    if (!s_wk_ctx || !s_wk_sem) return;
    s_wk_n = n;
    for (int i = 0; i < n; i++) {
        s_wk_ctx[i].lane = (unsigned)i;
        s_wk_thr[i] = CreateThread(NULL, 0, wk_main, &s_wk_ctx[i], 0, NULL);
        if (s_wk_thr[i]) SetThreadPriority(s_wk_thr[i], THREAD_PRIORITY_ABOVE_NORMAL);
    }
    fprintf(stderr, "[b2] RSX_DRAW_WORKERS=%d prep workers started\n", n);
}

static void wk_stop(void)
{
    if (!s_wk_n) return;
    if (s_wk_sem) ReleaseSemaphore(s_wk_sem, 16, NULL);
    for (int i = 0; i < s_wk_n; i++) if (s_wk_thr[i]) { WaitForSingleObject(s_wk_thr[i], 3000); CloseHandle(s_wk_thr[i]); s_wk_thr[i] = NULL; }
}

static const char* const s_b2p_name[B2P_N] = { "idle", "jobpre", "setup", "layout", "vckey", "vclookup", "vcfetch", "rstate", "targets", "texbind",
    "texdec", "texslot", "texhash", "texwatch", "texup", "vtex", "pipe", "pipecreate", "gprog", "idx", "const", "bind", "sampler", "stage", "drawrec",
    "tail", "w_enc", "w_fence", "w_submit", "ctrl_wr", "ctrl_q" };
/* nj == 0: only takes the baseline snapshot. Called from b2_report (thread B2), once per printed window. */
static void b2p_report(unsigned nj, unsigned frames, long long now_q, long long qf, double busy_ms)
{
    static unsigned long long t0tsc, l_cyc[B2P_N], l_nm[B2P_N], l_cnt[B2C_N], l_cpu; static long long t0q;
    FILETIME ct, et, kt, ut; GetThreadTimes(GetCurrentThread(), &ct, &et, &kt, &ut);
    const unsigned long long cpu = ((((unsigned long long)kt.dwHighDateTime << 32) | kt.dwLowDateTime) + (((unsigned long long)ut.dwHighDateTime << 32) | ut.dwLowDateTime));
    const unsigned long long cpu_k = (((unsigned long long)kt.dwHighDateTime << 32) | kt.dwLowDateTime);
    static unsigned long long l_cpu_k;
    const unsigned long long t1 = wp_tsc();
    if (nj && t0tsc) {
        const double secs = (double)(now_q - t0q) / (double)qf;
        const double hz = (double)(t1 - t0tsc) / secs;
        const double us_cyc = 1e6 / hz;
        double net[B2P_N], sum = 0.0, raw = 0.0, ovh = 0.0; unsigned long long marks = 0;
        for (int k = 0; k < B2P_N; k++) {
            const double dc = (double)(s_b2p_cyc[k] - l_cyc[k]), dm = (double)(s_b2p_nm[k] - l_nm[k]);
            const double n = dc - dm * s_b2p_cost;
            net[k] = n > 0.0 ? n * us_cyc : 0.0;                     /* us in the window */
            marks += (unsigned long long)dm;
            if (k != B2P_IDLE) { sum += net[k]; raw += dc * us_cyc; ovh += dm * s_b2p_cost * us_cyc; }
        }
        fprintf(stderr, "[b2prof] %.2f s, %u draws, %u frames (%.0f draws/frame): busy(qpc) %.1f ms | buckets non-idle raw %.1f ms, net of mark cost %.1f ms | idle bucket %.1f ms | "
                "thread CPU %.1f ms (kernel %.1f) | marks/draw %.1f x %.0f cyc = %.2f us/draw (est.)\n",
                secs, nj, frames, frames ? (double)nj / frames : 0.0, busy_ms, raw / 1000.0, sum / 1000.0, net[B2P_IDLE] / 1000.0,
                (double)(cpu - l_cpu) / 10000.0, (double)(cpu_k - l_cpu_k) / 10000.0, (double)marks / nj, s_b2p_cost, ovh / nj);
        fprintf(stderr, "[b2prof-us/draw]");
        for (int k = 1; k < B2P_N; k++) fprintf(stderr, " %s %.3f", s_b2p_name[k], net[k] / nj);
        fputc(10, stderr);
        fprintf(stderr, "[b2prof-ms/frame]");
        for (int k = 1; k < B2P_N; k++) fprintf(stderr, " %s %.3f", s_b2p_name[k], frames ? net[k] / 1000.0 / frames : 0.0);
        fputc(10, stderr);
        const unsigned long long* c = s_b2p_cnt;
        fprintf(stderr, "[b2prof-cnt/draw] vchit %.3f vcmiss %.4f texunit %.3f texslot %.3f texhash %.4f texwatch %.4f texup %.5f pipecreate %.5f stage %.3f stage_bytes %.1f | window: enc-join found-pending %llu blocked %llu, fence-blocked %llu, mid-frame submits %llu\n",
                (double)(c[B2C_VCHIT] - l_cnt[B2C_VCHIT]) / nj, (double)(c[B2C_VCMISS] - l_cnt[B2C_VCMISS]) / nj, (double)(c[B2C_TEXUNIT] - l_cnt[B2C_TEXUNIT]) / nj,
                (double)(c[B2C_TEXSLOT] - l_cnt[B2C_TEXSLOT]) / nj, (double)(c[B2C_TEXHASH] - l_cnt[B2C_TEXHASH]) / nj, (double)(c[B2C_TEXWATCH] - l_cnt[B2C_TEXWATCH]) / nj,
                (double)(c[B2C_TEXUP] - l_cnt[B2C_TEXUP]) / nj, (double)(c[B2C_PIPECREATE] - l_cnt[B2C_PIPECREATE]) / nj, (double)(c[B2C_STAGE] - l_cnt[B2C_STAGE]) / nj,
                (double)(c[B2C_STAGEB] - l_cnt[B2C_STAGEB]) / nj, c[B2C_ENCJOIN] - l_cnt[B2C_ENCJOIN], c[B2C_ENCBLK] - l_cnt[B2C_ENCBLK],
                c[B2C_FENCEBLK] - l_cnt[B2C_FENCEBLK], c[B2C_SUBMIT] - l_cnt[B2C_SUBMIT]);
    }
    memcpy(l_cyc, s_b2p_cyc, sizeof l_cyc); memcpy(l_nm, s_b2p_nm, sizeof l_nm); memcpy(l_cnt, s_b2p_cnt, sizeof l_cnt);
    l_cpu = cpu; l_cpu_k = cpu_k; t0tsc = t1; t0q = now_q;
}

static void b2_report(void)
{
    static long long last_q; static unsigned last_jobs; static unsigned last_frames;
    LARGE_INTEGER f; QueryPerformanceFrequency(&f);
    const long long now = b2_qpc();
    if (!last_q) { last_q = now; last_jobs = s_b2_njobs; last_frames = g.frames; if (g_b2p_on) b2p_report(0, 0, now, f.QuadPart, 0.0); return; }
    if ((now - last_q) * 1000 / f.QuadPart < 2000) return;
    const double ms = 1000.0 / (double)f.QuadPart;
    const unsigned nj = s_b2_njobs - last_jobs;
    fprintf(stderr, "[b2] %.1f s: %u draws, busy %.1f ms (%.2f us/draw), idle %.1f ms | D: queue-full waits %u (%.1f ms), drains %u (blocked %u, %.1f ms), seq-waits %u (blocked %u) | max queue %u\n",
            (double)(now - last_q) / (double)f.QuadPart, nj, (double)s_b2_busy_q * ms, nj ? (double)s_b2_busy_q * ms * 1000.0 / nj : 0.0,
            (double)s_b2_idle_q * ms, s_b2_nfull, (double)s_b2_full_q * ms, s_b2_ndrain, s_b2_nblocked, (double)s_b2_drain_q * ms,
            s_b2_nwait, s_b2_nwait_blk, s_b2_maxq);
    if (g_b2p_on) b2p_report(nj, g.frames - last_frames, now, f.QuadPart, (double)s_b2_busy_q * ms);
    last_frames = g.frames;
    last_q = now; last_jobs = s_b2_njobs;
    s_b2_busy_q = s_b2_idle_q = s_b2_full_q = s_b2_drain_q = 0;
    s_b2_nfull = s_b2_ndrain = s_b2_nblocked = s_b2_nwait = s_b2_nwait_blk = s_b2_maxq = 0;
    if (s_wk_n) { fprintf(stderr, "[b2] prep: workers %d, prepped by workers %u, by B2 (stolen) %u\n", s_wk_n, s_wk_njob, s_wk_nsteal); s_wk_njob = s_wk_nsteal = 0; }
    fprintf(stderr, "[b2] control jobs %u; drain sites:", s_b2_nctrl); s_b2_nctrl = 0;
    for (int i = 0; i < B2_SITES; i++) if (s_b2_site[i].n) {
        const char* fn = s_b2_site[i].f; for (const char* c = fn; *c; c++) if (*c == '/' || *c == '\\') fn = c + 1;
        fprintf(stderr, " %s:%d x%u (blk %u, %.1f ms)", fn, s_b2_site[i].line, s_b2_site[i].n, s_b2_site[i].nblk, s_b2_site[i].us / 1000.0);
        s_b2_site[i].n = s_b2_site[i].nblk = 0; s_b2_site[i].us = 0;
    }
    fputc(10, stderr);
}

static DWORD WINAPI b2_main(LPVOID arg)
{
    (void)arg;
    t_on_b2 = 1;
    { extern void ps3_sampler_tag_thread(void); if (getenv("PS3_SAMPLE_B2")) ps3_sampler_tag_thread(); }   /* PS3_SAMPLE=<ms> PS3_SAMPLE_SPU=1: profile this thread */
    if (g_b2p_on) b2p_calibrate();
    const int prof = s_wp_on > 0 || g_b2p_on;
    long long idle0 = 0;
    for (;;) {
        const unsigned long long c = s_b2_cons;
        unsigned long long p = __atomic_load_n(&s_b2_prod, __ATOMIC_ACQUIRE);
        if (c == p) {
            if (__atomic_load_n(&s_b2_stop, __ATOMIC_ACQUIRE)) break;
            if (prof && !idle0) idle0 = b2_qpc();
            for (int i = 0; i < 3000 && c == p; i++) { YieldProcessor(); p = __atomic_load_n(&s_b2_prod, __ATOMIC_ACQUIRE); }
            if (c == p) {
                __atomic_store_n(&s_b2_sleeping, 1, __ATOMIC_SEQ_CST);
                __atomic_thread_fence(__ATOMIC_SEQ_CST);
                if (__atomic_load_n(&s_b2_prod, __ATOMIC_ACQUIRE) == c && !__atomic_load_n(&s_b2_stop, __ATOMIC_ACQUIRE))
                    WaitForSingleObject(s_b2_ev, 2);
                __atomic_store_n(&s_b2_sleeping, 0, __ATOMIC_SEQ_CST);
                if (prof) b2_report();
                continue;
            }
        }
        long long t0 = 0;
        if (prof) { t0 = b2_qpc(); if (idle0) { s_b2_idle_q += t0 - idle0; idle0 = 0; } }
        { EngJob* jj = &s_b2_jobs[c % B2_SLOTS];
          if (g_b2p_on) b2p_job_begin();
          if (s_wk_n && jj->kind == 0) { wk_commit_wait(jj); s_cur_pre = &jj->pre; }
          b2_exec(jj);
          if (g_b2p_on) b2p_job_end();
          s_cur_pre = NULL; }
        if (prof) { s_b2_busy_q += b2_qpc() - t0; }
        __atomic_store_n(&s_b2_cons, c + 1, __ATOMIC_RELEASE);
    }
    return 0;
}

static int eng_b2_on(void)
{
    if (s_b2_on < 0) s_b2_on = (snap_mode() & 8) ? 1 : 0;
    return s_b2_on;
}

static void b2_start(void)
{
    { const char* pe = getenv("RSX_B2_PROF"); g_b2p_on = (pe && pe[0] && pe[0] != '0') ? 1 : 0; }   /* before the thread exists */
    s_b2_jobs = (EngJob*)calloc(B2_SLOTS, sizeof(EngJob));
    s_b2_ev = CreateEventA(NULL, FALSE, FALSE, NULL);
    wk_start();
    s_b2_started = 1;
    if (!s_b2_jobs || !s_b2_ev || !(s_b2_thr = CreateThread(NULL, 0, b2_main, NULL, 0, NULL))) {
        fprintf(stderr, "[b2] could not start the draw executor thread -- RSX_DRAW_SPLIT=2 disabled\n");
        s_b2_on = 0; s_b2_started = 0;
        return;
    }
    SetThreadPriority(s_b2_thr, THREAD_PRIORITY_ABOVE_NORMAL);
    SetThreadDescription(s_b2_thr, L"rsx draw B2");
    fprintf(stderr, "[b2] draw executor thread started (%d slots, job %u bytes)\n", B2_SLOTS, (unsigned)sizeof(EngJob));
}

static void b2_wait_cons(unsigned long long target, int is_drain)
{
    if (__atomic_load_n(&s_b2_cons, __ATOMIC_ACQUIRE) >= target) return;
    const long long q0 = s_wp_on > 0 ? b2_qpc() : 0;
    LARGE_INTEGER f; QueryPerformanceFrequency(&f);
    const long long qs = b2_qpc();
    int warned = 0;
    unsigned spins = 0;
    while (__atomic_load_n(&s_b2_cons, __ATOMIC_ACQUIRE) < target) {
        if (__atomic_load_n(&s_b2_sleeping, __ATOMIC_SEQ_CST)) SetEvent(s_b2_ev);
        if (++spins < 4000) YieldProcessor(); else SwitchToThread();
        if (!warned && (spins & 0xFFFF) == 0 && (b2_qpc() - qs) > 10 * f.QuadPart) {
            warned = 1;
            fprintf(stderr, "[b2] WARNING: %s waiting >10 s for B2 (cons %llu prod %llu target %llu)\n", is_drain ? "drain" : "seq-wait",
                    s_b2_cons, s_b2_prod, target);
        }
    }
    if (s_wp_on > 0) {
        if (is_drain) { s_b2_nblocked++; s_b2_drain_q += b2_qpc() - q0; } else s_b2_nwait_blk++;
    }
}

static void eng_b2_enqueue(void)
{
    if (!s_b2_started) b2_start();
    if (!g.ready || !dd.n_packets) return;
    if (!s_b2_started) return;
    s_b2_prod_tid = GetCurrentThreadId();
    const unsigned long long p = s_b2_prod;
    if (p - __atomic_load_n(&s_b2_cons, __ATOMIC_ACQUIRE) >= B2_SLOTS) {
        const long long q0 = s_wp_on > 0 ? b2_qpc() : 0;
        unsigned spins = 0;
        while (p - __atomic_load_n(&s_b2_cons, __ATOMIC_ACQUIRE) >= B2_SLOTS) {
            if (__atomic_load_n(&s_b2_sleeping, __ATOMIC_SEQ_CST)) SetEvent(s_b2_ev);
            if (++spins < 2000) YieldProcessor(); else SwitchToThread();
        }
        if (s_wp_on > 0) { s_b2_nfull++; s_b2_full_q += b2_qpc() - q0; }
    }
    EngJob* j = &s_b2_jobs[p % B2_SLOTS];
    __atomic_store_n(&j->pstate, 3, __ATOMIC_RELEASE);   /* filling: no prep worker may claim it */
    j->kind = 0;
    snap_fill_into(&j->snap, &g.rsx);
    j->n_arr = dd.n_arr; if (dd.n_arr) memcpy(j->arr, dd.arr, (size_t)dd.n_arr * sizeof(eng_batch));
    j->n_idx = dd.n_idx; if (dd.n_idx) memcpy(j->idx, dd.idx, (size_t)dd.n_idx * sizeof(eng_batch));
    j->n_packets = dd.n_packets;
    j->inl_bytes = 0;
    if (dd.inl && dd.inl_bytes) {
        j->inl_bytes = dd.inl_bytes < RSX_DSP_INLINE_MAX_BYTES ? dd.inl_bytes : RSX_DSP_INLINE_MAX_BYTES;
        memcpy(j->inl, dd.inl, j->inl_bytes);
    }
    j->vp_branch = s_vp_branch_bits_reg;
    j->seq_id = p;
    __atomic_store_n(&j->pstate, s_wk_n ? 0 : 2, __ATOMIC_RELEASE);
    __atomic_store_n(&s_b2_prod, p + 1, __ATOMIC_RELEASE);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    if (__atomic_load_n(&s_b2_sleeping, __ATOMIC_SEQ_CST)) SetEvent(s_b2_ev);
    if (s_wk_n && s_wk_sleepers > 0) ReleaseSemaphore(s_wk_sem, 1, NULL);
    if (s_wp_on > 0) { const unsigned q = (unsigned)(p + 1 - s_b2_cons); if (q > s_b2_maxq) s_b2_maxq = q; }
}

/* Queue a control job (kind 1: write32, kind 2: query method) behind the draws queued so far. Only from the thread that queues
 * draws, with B2 running and RSX_B2_CTRL != 0; returns 0 otherwise and the caller does drain + direct. */
static int b2_post_ctrl(u32 kind, u32 a, u32 b)
{
    if (!s_b2_started || t_on_b2) return 0;
    if (s_b2_ctrl_on < 0) { const char* e = getenv("RSX_B2_CTRL"); s_b2_ctrl_on = (e && e[0] == '0') ? 0 : 1; }
    if (!s_b2_ctrl_on || s_b2_prod_tid != GetCurrentThreadId()) return 0;
    const unsigned long long p = s_b2_prod;
    if (p - __atomic_load_n(&s_b2_cons, __ATOMIC_ACQUIRE) >= B2_SLOTS) {
        unsigned spins = 0;
        while (p - __atomic_load_n(&s_b2_cons, __ATOMIC_ACQUIRE) >= B2_SLOTS) {
            if (__atomic_load_n(&s_b2_sleeping, __ATOMIC_SEQ_CST)) SetEvent(s_b2_ev);
            if (++spins < 2000) YieldProcessor(); else SwitchToThread();
        }
    }
    EngJob* j = &s_b2_jobs[p % B2_SLOTS];
    __atomic_store_n(&j->pstate, 3, __ATOMIC_RELEASE);
    j->kind = kind; j->c_a = a; j->c_b = b;
    j->seq_id = p;
    __atomic_store_n(&j->pstate, 2, __ATOMIC_RELEASE);
    __atomic_store_n(&s_b2_prod, p + 1, __ATOMIC_RELEASE);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    if (__atomic_load_n(&s_b2_sleeping, __ATOMIC_SEQ_CST)) SetEvent(s_b2_ev);
    return 1;
}
static int eng_b2_post_query(u32 m, u32 arg) { return b2_post_ctrl(2, m, arg); }
void rsx_draw_engine_b2_ordered_write32(unsigned addr, unsigned val)
{
    extern void vm_write32(unsigned, unsigned);
    if (b2_post_ctrl(1, addr, val)) return;
    rsx_draw_engine_b2_drain();
    vm_write32(addr, val);
}

/* All draws queued so far have been executed (and so have read their guest memory). */
void rsx_draw_engine_b2_drain_at(const char* file, int line)
{
    if (!s_b2_started || t_on_b2) return;
    if (s_wp_on > 0) {
        s_b2_ndrain++;
        int k = 0;
        while (k < B2_SITES - 1 && s_b2_site[k].f && !(s_b2_site[k].line == line && s_b2_site[k].f == file)) k++;
        s_b2_site[k].f = file; s_b2_site[k].line = line; s_b2_site[k].n++;
        const unsigned long long prod = __atomic_load_n(&s_b2_prod, __ATOMIC_ACQUIRE);
        if (__atomic_load_n(&s_b2_cons, __ATOMIC_ACQUIRE) < prod) {
            const long long q0 = b2_qpc();
            b2_wait_cons(prod, 1);
            LARGE_INTEGER f; QueryPerformanceFrequency(&f);
            s_b2_site[k].nblk++; s_b2_site[k].us += (double)(b2_qpc() - q0) * 1e6 / (double)f.QuadPart;
        }
        return;
    }
    b2_wait_cons(__atomic_load_n(&s_b2_prod, __ATOMIC_ACQUIRE), 1);
}
unsigned long long rsx_draw_engine_b2_seq(void) { return s_b2_started ? __atomic_load_n(&s_b2_prod, __ATOMIC_ACQUIRE) : 0; }
void rsx_draw_engine_b2_wait(unsigned long long seq)
{
    if (!s_b2_started || t_on_b2 || !seq) return;
    if (s_wp_on > 0) s_b2_nwait++;
    b2_wait_cons(seq, 0);
}
static void eng_b2_stop(void)
{
    if (!s_b2_started) return;
    rsx_draw_engine_b2_drain();
    __atomic_store_n(&s_b2_stop, 1, __ATOMIC_RELEASE);
    SetEvent(s_b2_ev);
    WaitForSingleObject(s_b2_thr, 5000);
    CloseHandle(s_b2_thr); s_b2_thr = NULL;
    wk_stop();
    s_b2_started = 0;
}
#else
static int eng_b2_on(void) { return 0; }
static void eng_b2_enqueue(void) {}
static int eng_b2_post_query(u32 m, u32 arg) { (void)m; (void)arg; return 0; }
void rsx_draw_engine_b2_ordered_write32(unsigned addr, unsigned val) { extern void vm_write32(unsigned, unsigned); vm_write32(addr, val); }
void rsx_draw_engine_b2_drain_at(const char* file, int line) { (void)file; (void)line; }
unsigned long long rsx_draw_engine_b2_seq(void) { return 0; }
void rsx_draw_engine_b2_wait(unsigned long long seq) { (void)seq; }
static void eng_b2_stop(void) {}
#endif

/* RSX_QUERY_NOSYNC (see the fence in rsx_draw_engine_method): -1 undecided.
 * A global so a host can switch it in a run. */
int g_rsx_query_nosync = -1;

int rsx_draw_engine_enabled(void)
{
    static int cached = -1;
    static int env_seen = 0;
    static int env_on = 0;
    if (!env_seen) {
        const char* e = getenv("PS3RECOMP_RSX_ENGINE");
        env_seen = 1;
        if (e && *e) env_on = (strcmp(e, "dispatch") == 0) ? 1 : -1;
    }
    /* The environment is authoritative in both directions; without it the
     * backend's own default decides. Cached after the first backend has
     * registered, so the FIFO walker pays one compare. */
    const int on = env_on > 0 ? 1 : (env_on < 0 ? 0 : g.default_on);
    if (!g.be) return 0;
    if (cached < 0) cached = on;
    return cached;
}

int rsx_draw_engine_init(u32 width, u32 height)
{
    if (!g.be) return -1;
    g.width  = width  ? width  : 1280;
    g.height = height ? height : 720;
    rsx_rt_config_load();
    g.vp_cb = (u8*)calloc(1, ENG_VP_CB_BYTES);
    if (!g.vp_cb) return -1;
    g.vp_cb_valid = 0;
    g.vp_cb_bound = 0;
    if (g.be->init && g.be->init(g.be->user, g.width, g.height) != 0) {
        free(g.vp_cb); g.vp_cb = NULL;
        return -1;
    }
    rsx_dispatch_sink sink;
    memset(&sink, 0, sizeof sink);
    sink.clear            = sink_clear;
    sink.begin            = sink_begin;
    sink.end              = sink_end;
    sink.draw_arrays      = sink_draw_arrays;
    sink.draw_index_array = sink_draw_index;
    sink.inline_array     = sink_inline_array;
    sink.flip             = sink_flip;
    rsx_dispatch_init(&g.rsx, &sink);
    g.rv = &g.rsx;
    snap_init();
    g.ready = 1;
    eng_pipe_preload();   /* the manifest goes to the build workers now, not at the first draw */
    fprintf(stderr, "[rsx engine] register-file draw engine up (%ux%u)\n",
            g.width, g.height);
    return 0;
}

void rsx_draw_engine_shutdown(void)
{
    if (!g.be) return;
    eng_b2_stop();
    eng_pipe_stop();
    for (u32 i = 0; i < g.n_pipelines; i++)
        if (g.pipelines[i].pending) eng_pipe_poll(&g.pipelines[i]);
    if (g.ready && g.be->submit_and_wait)
        g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_SHUTDOWN);
    for (u32 i = 0; i < g.n_textures; i++)
        if (g.textures[i].handle) g.be->texture_release(g.be->user, g.textures[i].handle);
    for (u32 i = 0; i < g.n_pipelines; i++)
        if (g.pipelines[i].handle) g.be->pipeline_release(g.be->user, g.pipelines[i].handle);
    for (u32 i = 0; i < g.n_surfaces; i++)
        if (g.surfaces[i].handle) {
            g.be->color_target_release(g.be->user, g.surfaces[i].handle);
            for (u32 k = 0; k < g.surfaces[i].park_n; k++)
                g.be->color_target_release(g.be->user, g.surfaces[i].park[k].handle);
        }
    for (u32 i = 0; i < g.n_zdepths; i++)
        if (g.zdepths[i].handle) g.be->depth_target_release(g.be->user, g.zdepths[i].handle);
    if (s_vc) {
        for (u32 i = 0; i < ENG_VC_SLOTS; i++) if (s_vc[i].key) eng_vc_free(&s_vc[i]);
        free(s_vc); s_vc = NULL; free(s_vck); s_vck = NULL; s_vc_count = 0; s_vc_bytes = 0;
    }
    if (g.ready && g.be->shutdown) g.be->shutdown(g.be->user);

    rsx_draw_engine_set_guest_memory(NULL, NULL);
    free(dc.refs); free(dc.cuts); free(dc.verts);
    rsx_vertex_remap_destroy(&dc.ref_remap);
    memset(&dc, 0, sizeof dc);
    free(g.tex_staging); free(g.vp_cb); free(g.fp_cb); free(g.indices);
    const rsx_draw_backend* be = g.be;
    memset(&g, 0, sizeof g);
    g.be = be;
    s_surf_gen++; s_zgen++;
}

void rsx_draw_engine_method(u32 method, u32 arg)
{
    if (!g.ready) return;
    /* The subchannel is a binding slot, not an engine selector, and a title
     * whose SPU-built command lists bind NV4097 elsewhere would otherwise
     * store its state in the wrong register bank.  Standard NV4097 methods
     * (< 0x2000) strip the subchannel with & 0x1FFC.  Driver methods
     * (0xE9xx, 0xEBxx) encode the subchannel bits as part of the address;
     * preserve them with the wider mask. */
    u32 m = (method >= 0xE000u) ? (method & 0xFFFCu) : (method & 0x1FFCu);
    /* RSX_METHOD_HIST=1: every 120 presents, the methods the engine saw in that
     * window (method id x count, top 24) -- "is the title submitting anything,
     * and what kind of work" without a full method trace. */
    { static int hist_on = -1; static unsigned hist[0x800]; static u32 hist_last = ~0u;
      if (hist_on < 0) hist_on = (getenv("RSX_METHOD_HIST") || getenv("RSX_MCNT")) ? 1 : 0;
      if (hist_on) {
          if (g.frames != hist_last && (g.frames % 120u) == 0u) {
              hist_last = g.frames;
              fprintf(stderr, "[mhist] frame %u:", g.frames);
              for (int k = 0; k < 30; k++) {
                  unsigned best = 0, bi = 0;
                  for (unsigned i = 0; i < 0x800; i++) if (hist[i] > best) { best = hist[i]; bi = i; }
                  if (!best) break;
                  fprintf(stderr, " %04X x%u", bi * 4u, best); hist[bi] = 0;
              }
              fputc('\n', stderr);
              memset(hist, 0, sizeof hist);
          }
          hist[(m >> 2) & 0x7FFu]++;
      } }
    /* RSX_METHOD_TRACE_FRAME=<f>: every method the engine sees during present
     * <f> (one frame), as "[mtrace] f<frame> <method> <arg>". */
    { static long tf = -2; if (tf == -2) { const char* e = getenv("RSX_METHOD_TRACE_FRAME"); tf = e ? atol(e) : -1; }
      if (tf >= 0 && (long)g.frames == tf) fprintf(stderr, "[mtrace] f%u %04X %08X\n", g.frames, m, arg); }
    if (m == 0x1FD8u && s_snap_stats > 0) { s_inval_n[arg < 4 ? arg : 0]++; s_inval_total++; }
    /* Recorded after the dispatch, behind the pages it read (rsx_capture.c),
     * and only when the capture was already running when it began. */
    const int cap = g_rsx_capture_on;
    /* Occlusion-query methods in the draw trace: CLEAR_REPORT_VALUE (0x17C8),
     * GET_REPORT (0x1800), SET_ZPASS_PIXEL_COUNT_ENABLE (0x1D84). */
    /* Occlusion queries: CLEAR_REPORT_VALUE(ZPASS) opens a counter for the
     * draws that follow, GET_REPORT(ZPASS) closes it and asks the backend for
     * its total under that report index (rsx_draw_engine_query_result). The
     * title never sets ZPASS_PIXEL_COUNT_ENABLE and still expects counts, so
     * an open query counts regardless. A query whose draws never reached the
     * backend -- dropped, not drawn -- reports "visible": a bug of ours must
     * not hide the object. */
    if (m == 0x1FF8u) s_vp_branch_bits_reg = arg;   /* SET_TRANSFORM_BRANCH_BITS (BRB/CLB) */
    if (m == 0x17C8u || (m == 0x1800u && (arg >> 24) == 1u) || m == 0x1D70u) {
        if (!eng_b2_post_query(m, arg)) { rsx_draw_engine_b2_drain(); eng_query_method(m, arg); }   /* RSX_DRAW_SPLIT=2: g.q_*, query_begin/report, the fence */
    }
    if ((m == 0x17C8u || m == 0x1800u || m == 0x1D84u) && s_dtrace_frame >= 0 &&
        (long)g.frames >= s_dtrace_frame && (long)g.frames < s_dtrace_frame + 4)
        fprintf(stderr, "[draw-trace] f%u QUERY %s 0x%08X\n", g.frames,
                m == 0x17C8u ? "CLEAR_REPORT_VALUE" : m == 0x1800u ? "GET_REPORT" : "ZPASS_ENABLE", arg);
    rsx_dispatch_method(&g.rsx, m, arg);
    if (cap) rsx_capture_method(method, arg);
}

/* RSX_FIFO_FAST=1: the runs of texture-unit and vertex-array registers go over whole as well
 * (rsx_dispatch_plain_run). Not with the per-method diagnostics, which would no longer see those words. */
static int fifo_fast_plain_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("RSX_FIFO_FAST");
        on = (e && e[0] && e[0] != '0' && !getenv("RSX_METHOD_HIST") && !getenv("RSX_METHOD_TRACE_FRAME") &&
              !getenv("RSX_CONST_LOG")) ? 1 : 0;
    }
    return on;
}

int rsx_draw_engine_upload_window(u32 method, u32 count)
{
    static int mc = -1; if (mc < 0) mc = getenv("RSX_MCNT") ? 1 : 0;   /* RSX_MCNT: count every word in rsx_draw_engine_method */
    if (!count || mc) return 0;
    const u32 m = method & 0x1FFCu, end = m + count * 4u;
    if ((m >= 0x1F00u && end <= 0x1F80u) || (m >= 0x0B80u && end <= 0x0C00u)) return 1;
    return fifo_fast_plain_on() &&
           ((m >= 0x1A00u && end <= 0x1C00u) || (m >= 0x1680u && end <= 0x16C0u) || (m >= 0x1740u && end <= 0x1780u));
}

void rsx_draw_engine_upload_run(u32 method, u32 count, const u8* be)
{
    if (!g.ready) return;
    /* The capture records method by method, behind the pages it read. */
    if (!g_rsx_capture_on && rsx_dispatch_upload_run(&g.rsx, method & 0x1FFCu, count, be)) return;
    if (!g_rsx_capture_on && fifo_fast_plain_on() && rsx_dispatch_plain_run(&g.rsx, method & 0x1FFCu, count, be)) return;
    for (u32 i = 0; i < count; i++) {
        const u8* p = be + i * 4u;
        rsx_draw_engine_method(method + i * 4u,
                               ((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | (u32)p[3]);
    }
}

void cellGcm_set_report_value(u32 index, u32 value);   /* cellGcmSys.c */
void rsx_draw_engine_query_result(u32 report_index, u64 count)
{
    static int log = -1; if (log < 0) log = getenv("RSX_QUERY_LOG") ? 1 : 0;
    if (log) fprintf(stderr, "[query] f%u report %u = %llu\n", g.frames, report_index, (unsigned long long)count);
    cellGcm_set_report_value(report_index, count > 0xFFFFFFFFull ? 0xFFFFFFFFu : (u32)count);
}

void rsx_draw_engine_set_display_buffer(u32 buffer_id, u32 location, u32 offset,
                                        u32 pitch, u32 width, u32 height)
{
    if (buffer_id >= 8) return;
    rsx_draw_engine_b2_drain();
    rsx_capture_display_buffer(buffer_id, location, offset, pitch, width, height);
    eng_display_buffer* d = &g.display_buffers[buffer_id];
    d->location = location;
    d->offset = offset;
    d->pitch = pitch;
    d->width = width;
    d->height = height;
    d->valid = width && height;
}

void rsx_draw_engine_sync_queries(void)
{
    if (!g.ready) return;
    rsx_draw_engine_b2_drain();
    if (!g.q_unflushed || !g.be->submit_and_wait) return;
    { extern volatile int g_tl_on; extern void tl_f38(int, uint32_t, uint64_t, uint64_t); if (g_tl_on) tl_f38(8, 0, 0, 0); }   /* fence38 */
    g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_QUERY_FENCE);
    { extern volatile int g_tl_on; extern void tl_f38(int, uint32_t, uint64_t, uint64_t); if (g_tl_on) tl_f38(9, 0, 0, 0); }
    g.q_unflushed = 0;
}

void rsx_draw_engine_flush(void)
{
    rsx_draw_engine_b2_drain();
    if (g.ready && g.be->submit_and_wait)
        g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_GUEST_REFERENCE);
}

void rsx_draw_engine_present(void)
{
    /* A host that drives the flip itself gets whichever buffer the guest's
     * own last flip named, and display buffer 0 before there has been one:
     * a runner that both consumes 0xE944 and calls here would otherwise
     * re-present a double-buffered title's OTHER scanout. Presenting twice is
     * harmless either way -- the engine never clears the surface, so a second
     * present just blits the same image again. */
    if (!eng_host_present_allowed()) return;
    eng_present(g.last_flip_buffer);
}

void rsx_draw_engine_present_buffer(u32 buffer_id)
{
    if (!eng_host_present_allowed()) return;
    g.last_flip_buffer = buffer_id & 7u;
    eng_present(g.last_flip_buffer);
}

int rsx_draw_engine_fifo_flip(u32 buffer_id)
{
    extern void (*g_gcm_trace_hook)(u32 type, u32 a, u32 b);
    if (g_gcm_trace_hook) g_gcm_trace_hook(8, buffer_id, g.frames);
    if (!g.ready) return 0;
    rsx_draw_engine_b2_drain();
    g.last_flip_buffer = buffer_id & 7u;
    g.sink_flips++;
    rsx_capture_flip(buffer_id);
    s_present_src = "fifo";
    eng_present(g.last_flip_buffer);
    s_present_src = "host";
    return 1;
}

u32 rsx_draw_engine_guest_draws(void)
{
    return g.last_guest_draws;
}

u32 rsx_draw_engine_readback_center(void)
{
    if (!g.ready || !g.be->readback) return 0;
    rsx_draw_engine_b2_drain();
    const u32 slot = g.last_present_surface;
    if (slot >= g.n_surfaces) return 0;
    const eng_surface* s = &g.surfaces[slot];
    if (!s->handle || s->fmt != RSX_BE_FMT_R8G8B8A8 || !s->w || !s->h) return 0;
    u8 px[4] = { 0, 0, 0, 0 };
    g.be->readback(g.be->user, s->handle, s->w / 2u, s->h / 2u, 1, 1, px, 4);
    /* Decoded rows are R,G,B,A; the guest's own clear colour is A8R8G8B8, and
     * a host comparing the two wants them in the same order. */
    return ((u32)px[3] << 24) | ((u32)px[0] << 16) |
           ((u32)px[1] << 8)  |  (u32)px[2];
}

/* RSX_FIFO_RUN: a run of `count` big-endian words, either consecutive methods (incr) or one method repeated. */
void rsx_draw_engine_method_run(u32 method, u32 count, int incr, const u8* be)
{
    if (!g.ready) return;
    /* Plain state words (rsx_dispatch_state_run) are stored in a loop; anything the per-method diagnostics or a
     * capture want to see, or any special method, goes one by one through rsx_draw_engine_method. */
    static int diag = -1;
    if (diag < 0) diag = (getenv("RSX_METHOD_HIST") || getenv("RSX_MCNT") || getenv("RSX_METHOD_TRACE_FRAME") || getenv("RSX_FIFO_RUN_SLOW")) ? 1 : 0;
    u32 i = 0;
    while (i < count) {
        if (!diag && !g_rsx_capture_on) {
            i += rsx_dispatch_state_run(&g.rsx, incr ? method + i * 4u : method, count - i, incr, be + i * 4u);
            if (i >= count) break;
        }
        const u8* p = be + i * 4u;
        rsx_draw_engine_method(incr ? method + i * 4u : method,
                               ((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | (u32)p[3]);
        i++;
    }
}
