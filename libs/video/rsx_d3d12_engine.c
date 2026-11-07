/*
 * ps3recomp - the register-file draw engine's D3D12 backend (Windows)
 *
 * rsx_draw_engine.c (libs/video/rsx_draw_engine.h) decodes the NV4097
 * register file and drives a backend through rsx_draw_backend: resources by
 * handle, pipelines from the decompilers' HLSL plus render state, and one
 * record per draw or clear. This file is that interface over Direct3D 12,
 * written against rsx_metal_backend.m's s_engine_backend, whose record
 * stream and object tables it mirrors so the two can be read side by side.
 *
 * Shape:
 *   - bind_* calls accumulate into LB->pending; draw/clear/snapshot calls
 *     append an EngRecord. Nothing touches a command list until a submit.
 *   - Vertices, indices and constants are copied into one UPLOAD-heap arena
 *     per submit (the "stage"); texture rows go through the same arena and a
 *     copy record ordered with the draws.
 *   - A submit opens a command list on one of ENG_FRAMES allocator slots,
 *     walks the records emitting barriers from per-object state tracking,
 *     resolves the occlusion queries and signals a fence. Each slot owns a
 *     region of the shader-visible SRV heap and a sampler heap of its own,
 *     so a slot is reused only once its fence has passed.
 *   - A windowed present keeps at most ENG_INFLIGHT submits queued; every
 *     other submit (query fence, guest reference, headless) waits.
 *
 * Handles are 1-based indices into one object table (textures, colour and
 * depth targets, snapshots and views alike), because the engine binds all of
 * them through bind_textures. Pipelines have their own table.
 *
 * PS3RECOMP_D3D12_HEADLESS=1 renders without a window (tools/rsx_replay);
 * PS3RECOMP_RSX_ENGINE=vtable keeps the old rsx_state path in
 * rsx_d3d12_backend.c instead of this one.
 */
#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <intrin.h>
#include <windows.h>
#include "ps3emu/beacon.h"
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <io.h>
#include <direct.h>

#include "rsx_draw_engine.h"
#define RSX_RT_D3D12
#include "rsx_rt.h"
#include "rsx_texture_layout.h"
#include "rsx_d3d12_engine.h"
#include "rsx_host_mouse.h"
#include "rsx_present_pace.h"

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")

/* COM through C: obj->lpVtbl->Method(obj, ...). */
#define CALL(o, m, ...) ((o)->lpVtbl->m((o), __VA_ARGS__))
#define CALL0(o, m)     ((o)->lpVtbl->m((o)))
#define RELEASE(p)      do { if (p) { (p)->lpVtbl->Release(p); (p) = NULL; } } while (0)

#define ENG_MAX_OBJECTS   16384       /* was 4096: RSX_BIND_UNREAD=1 filled it in metropolis */
#define ENG_MAX_PIPES     16384   /* 4096 filled in a 4K MSAA metropolis session (22/10): every later pipeline dropped its draws -- floors, weapon effects */
#define ENG_MAX_RECORDS   65536       /* a busy frame (Ratchet ToD metropolis) has over 11000 */
#define ENG_MAX_VIEWS     256
#define ENG_MAX_BLOBS     16384   /* 27/10: was 4096; the PSO manifest precompile alone queues up to 3000 vs+fp (+ WPOS variants) */
#define ENG_MAX_SAMPLERS  256
#define ENG_MAX_BUFS      16384
/* Where a reservation in the stage starts. Texture uploads are copied out of
 * the stage with CopyTextureRegion, whose placed footprint has to start on
 * D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT (512) -- constant buffers need only
 * 256. A device that reports UnrestrictedBufferTextureCopyPitchSupported
 * (Windows 11's runtime with a current driver) takes any offset, so 256 went
 * unnoticed there; on Windows 10 the copy made the whole command list fail
 * to close (E_INVALIDARG), and the engine stopped drawing for the rest of
 * the session on the first loading screen. */
#define ENG_ALIGN         512u
#define ENG_STAGE_MAX     (384u << 20)
#define ENG_STAGE_CHUNK   (16u << 20)
#define ENG_FRAMES        4           /* allocator slots */
#define ENG_INFLIGHT      2           /* windowed presents queued at most */
#define ENG_SRV_PER_SLOT  65536u      /* shader-visible SRV descriptors per slot (all slots: 262144, under the 1M tier-1 limit) */
#define ENG_SMP_PER_SLOT  2048u       /* the D3D12 sampler-heap maximum */
#define ENG_SRV_TABLE     (RSX_BE_MAX_TEXTURES + RSX_BE_MAX_VERTEX_TEXTURES) /* 20 */
#define ENG_QUERIES_PER_SLOT 16384u
#define ENG_VIS_SLOTS     16384u
#define ENG_MAX_REPORTS   4096
#define ENG_SRV_NULL      ENG_MAX_OBJECTS   /* CPU SRV heap slot for the null texture */
#define ENG_DEPTH_FMT     DXGI_FORMAT_D32_FLOAT_S8X24_UINT
#define ENG_DEPTH_RES_FMT DXGI_FORMAT_R32G8X24_TYPELESS
#define ENG_DEPTH_SRV_FMT DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS
#define ENG_SHADER_READ   (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)

#include "rsx_logv2.h"
static int eng_v2_vram(int full, char* out, size_t n, unsigned long long* used, unsigned long long* budget);
typedef enum { OBJ_NONE = 0, OBJ_TEXTURE, OBJ_COLOR, OBJ_DEPTH, OBJ_SNAPSHOT, OBJ_VIEW } EngObjKind;

typedef struct {
    ID3D12Resource* res;        /* NULL for a view: see alias            */
    EngObjKind kind;
    u32 alias;                  /* OBJ_VIEW: the surface handle it views */
    DXGI_FORMAT fmt;
    u32 w, h, mips, faces;      /* host pixels                           */
    u32 gw, gh;                 /* guest pixels; w,h = gw,gh * the scale  */
    D3D12_RESOURCE_STATES state;
    int has_rtv, has_dsv;       /* RTV/DSV written at slot handle-1      */
    int retired;
    u32 heap;                   /* a placed texture: its heap + 1, and    */
    u64 heap_off, heap_size;    /* its range there (see the texture heaps) */
    int dsinit;                 /* RSX_DS_HEAP: a fresh placed snapshot: its first DEPTH_RESOLVE record brings an aliasing barrier + a zero clear first */
    /* MSAA (see eng_ms_*): the multisampled twin a 3D pass draws into, made
     * at the pass's first use; `res` stays the single-sampled one that every
     * read uses. ms_dirty: the twin has draws `res` lacks, in host pixels
     * [ms_x0,ms_x1) x [ms_y0,ms_y1). ms_stale: `res` was written another
     * way (a pass without depth, an upload), so the twin is refreshed from
     * it (a depth twin: cleared) before its next pass. */
    ID3D12Resource* ms;
    D3D12_RESOURCE_STATES ms_state;
    int ms_dirty, ms_stale;
    int ms_znewer;              /* a depth twin has passes `res` lacks */
    float xk;                   /* 26/10 gfx-quick: extra resolution multiplier (RSX_SHADOW_SCALE / RSX_BLOOM_SCALE), 0 = none */
    int xbloom;                 /* xk is the bloom one: its base is max(RSX_SCALE,1); the shadow one's is RSX_SCALE */
    u8 zshadow;                 /* a depth target: the last pass bound to it was a Z16 shadow pass (g_rsx_pass_shadow); RSX_MS_NO_SHADOW */
    float pf;                   /* a depth target: factor of the last pass drawn into it with a shadow colour target, 0 = default */
    float xx;                   /* 02/11 aa110 (RSX_HSTRETCH_FIX): extra HORIZONTAL-only multiplier of a 960x704 scene colour target (the game's own 960->1280 stretch pass then becomes 1:1); a depth target: that of the last pass drawn into it. 0 = none */
    u32 ms_x0, ms_y0, ms_x1, ms_y1;
    u32 ms_w, ms_h;             /* rissa127 RSX_MS_TWIN_FIT: size of the twin (it can be larger than w x h); set with every o->ms */
    /* 04/11 vram106 (RSX_MS_TWIN_LOG / RSX_MS_TWIN_POOL): use of the twin, in encode frames (s_enc_frame). */
    u32 mt_last, mt_born;       /* frame of the last pass drawn into the twin / of its creation */
    u32 mt_fdraws;              /* draws into the twin this frame */
    u32 ms_few; int ms_never;   /* msaa127 RSX_MS_FEWDRAW: used frames in a row with <= 1 draw; 1 = this target gets no twin any more */
    u32 mt_wdraws, mt_wframes, mt_wpasses;   /* this log window: draws, frames used, passes */
    u32 mt_wdo, mt_wcol, mt_wshadow;         /* window draws: depth-only pass / with colour / with a shadow-scale colour target */
    u32 mt_ex, mt_ey;           /* the largest scissor extent drawn into the twin (host pixels) */
    u32 mt_pool;                /* RSX_MS_TWIN_POOL: the twin has been taken over from another owner this many times */
    u32 mt_ford, mt_lord;       /* ordinals (draws of the frame so far) of the first / last draw into the twin this frame */
    u32 mt_robbed;              /* RSX_MS_TWIN_POOL: s_enc_frame + 1 when this object's twin was taken over by another one */
    u32 v2_site, v2_born, v2_gaddr, v2_ginfo;   /* logv2: allocation site (index in s_v2s), present frame of creation, guest address / info of the caller */
} EngObj;
static void eng_ms_drop(EngObj* o);   /* vram106: a twin released with its target */

/* ---- internal resolution --------------------------------------------------
 *
 * RSX_SCALE=<factor> (1 = the title's own 1280x720; 2 = 2560x1440; 3 = 4K).
 * Render targets, depth targets and their snapshots are created `factor`
 * times the size the guest asked for; viewports, scissors and clear
 * rectangles are scaled to match; everything a shader samples is addressed
 * in normalised coordinates and needs nothing. What crosses back into the
 * guest's world is scaled down: a readback is resampled to the guest size,
 * an occlusion count is divided by factor^2, and a fragment program's WPOS
 * is divided back to guest pixels (a title computes screen UVs from it).
 * Guest textures (uploaded pixels) are never scaled.
 *
 * RSX_DISPLAY=windowed|borderless|fullscreen: a bordered window of
 * RSX_WINDOW=<w>x<h> (default 1280x720); a borderless window the size of
 * the primary display; or exclusive full screen at the display's current
 * mode. (RSX_FULLSCREEN=1 is the old spelling of borderless.) The present blit resamples the surface to
 * the window whatever the two sizes are.
 *
 * RSX_SCALE_MIN=<n> (default 64): a target smaller than n in either
 * dimension stays at the guest size. Drakengard 3 builds its 256x16
 * colour-grading table in a render target, texel by texel from WPOS; scaled,
 * the table came out wrong (its whites 75% grey) and tone mapping then put
 * olive blotches on dark surfaces. So the scale is per target, and so is
 * everything that follows from it, from the target a pass draws into:
 * viewports, scissors and clear rectangles (recorded in guest pixels, scaled
 * at encoding), occlusion counts, depth-snapshot sizes, and WPOS -- a pass
 * into an unscaled target uses a second pipeline state whose fragment program
 * does not divide it (EngPipeline.ps_plain, built on first use). As the
 * Metal engine does (rsx_metal_backend.m, eng_obj_fx). */
static float s_scale = 1.0f;
static u32 s_scale_min = 64;
extern volatile int g_rsx_display_reload;   /* rsx_draw_engine.c */
static u32 s_win_w, s_win_h;                 /* the window and swap chain */
typedef enum { DISP_WINDOWED = 0, DISP_BORDERLESS, DISP_FULLSCREEN } EngDisplayMode;
static EngDisplayMode s_display = DISP_WINDOWED;
/* 26/10 gfx-quick. RSX_SHADOW_SCALE (1/2/4, default 2): per-object shadow maps (the pass has a Z16 zeta,
 * g_rsx_pass_shadow, set by the draw engine) get their colour target and viewport xk times larger than the
 * internal resolution; the canvas-sized depth they draw into already holds them (checked in eng_shadow_f).
 * RSX_BLOOM_SCALE (1/2/4, default 2): the 320x176 / 160x96 / 80x48 bloom chain, likewise. 1 = as before. */
static int s_shadow_k = 2, s_bloom_k = 2;
extern volatile int g_rsx_pass_shadow;          /* rsx_draw_engine.c */
extern volatile u32 g_rsx_pass_cw, g_rsx_pass_ch;
extern unsigned g_rsx_gov_ev[6];   /* rsx_draw_engine.c: governor frees by kind, printed in the [rt-heap] vram guard line */
extern float (*g_rsx_poly_target_scale_fn)(void);   /* rsx_draw_engine.c, shad114 RSX_DBIAS_EPILOGUE */
extern int rsx_dbias_epilogue_on(void);
static u32 s_width, s_height;
static int eng_extra_on(void) { return s_shadow_k > 1 || s_bloom_k > 1; }
static u32 scf_dim(u32 v, float f) { u32 r = (u32)((float)v * f + 0.5f); return v && !r ? 1u : r; }
static u32 scf_pos(u32 v, float f) { return (u32)((float)v * f + 0.5f); }
static void scf_rect(u32* x, u32* y, u32* w, u32* h, float f)
{
    const u32 x0 = scf_pos(*x, f), y0 = scf_pos(*y, f), x1 = scf_pos(*x + *w, f), y1 = scf_pos(*y + *h, f);
    *x = x0; *y = y0; *w = x1 - x0; *h = y1 - y0;
}
/* The extra multiplier for a shadow pass of this guest size (the current pass: g_rsx_pass_shadow), 1 if none.
 * It must fit the canvas-sized depth (guest canvas x RSX_SCALE) and stay under 2048 host pixels. */
static int eng_shadow_k(u32 w, u32 h)
{
    if (s_shadow_k < 2 || !g_rsx_pass_shadow || !w || !h) return 1;
    int k = s_shadow_k;
    while (k > 1 && ((u64)w * k > s_width || (u64)h * k > s_height ||
                     (float)w * s_scale * (float)k > 2048.0f || (float)h * s_scale * (float)k > 2048.0f)) k >>= 1;
    return k;
}
/* shad114 RSX_DBIAS_EPILOGUE: host pixels per guest pixel of the CURRENT pass's target (the slope-scaled bias is per host pixel):
 * s_scale, times the extra shadow-map k for Z16 shadow passes. */
static float eng_poly_target_scale(void)
{
    const int ks = eng_shadow_k(g_rsx_pass_cw, g_rsx_pass_ch);
    return s_scale * (float)(ks > 1 ? ks : 1);
}
static int eng_bloom_k(u32 w, u32 h)
{
    if (s_bloom_k < 2) return 1;
    return ((w == 320 && h == 176) || (w == 160 && h == 96) || (w == 80 && h == 48)) ? s_bloom_k : 1;
}
static u32 sc_dim(u32 v) { u32 r = (u32)((float)v * s_scale + 0.5f); return v && !r ? 1u : r; }
static u32 sc_pos(u32 v) { return (u32)((float)v * s_scale + 0.5f); }
/* [x, x+w) in guest pixels -> host pixels, as the two ends rounded, so
 * neighbouring rectangles still meet. */
static void sc_rect(u32* x, u32* y, u32* w, u32* h)
{
    const u32 x0 = sc_pos(*x), y0 = sc_pos(*y), x1 = sc_pos(*x + *w), y1 = sc_pos(*y + *h);
    *x = x0; *y = y0; *w = x1 - x0; *h = y1 - y0;
}
/* Whether a target of this guest size is created at the internal resolution. */
static int eng_scales(u32 w, u32 h) { return s_scale != 1.0f && w >= s_scale_min && h >= s_scale_min; }
static void eng_read_scale(void)
{
    const char* m = getenv("RSX_SCALE_MIN");
    if (m && *m) s_scale_min = (u32)atoi(m);
    { const char* k = getenv("RSX_SHADOW_SCALE"); if (k && *k) { const int v = atoi(k); s_shadow_k = v >= 4 ? 4 : v >= 2 ? 2 : 1; } else s_shadow_k = 2; }
    { const char* k = getenv("RSX_BLOOM_SCALE");  if (k && *k) { const int v = atoi(k); s_bloom_k  = v >= 4 ? 4 : v >= 2 ? 2 : 1; } else s_bloom_k = 2; }
    const char* e = getenv("RSX_SCALE");
    s_scale = 1.0f;
    if (e && *e) {
        const float f = (float)atof(e);
        if (f >= 0.5f && f <= 8.0f) s_scale = f;
        else fprintf(stderr, "[RSX d3d12] RSX_SCALE=%s ignored (0.5 to 8)\n", e);
    }
    g_rsx_poly_target_scale_fn = eng_poly_target_scale;   /* shad114 RSX_DBIAS_EPILOGUE */
}

typedef struct {
    ID3DBlob* vs; ID3DBlob* ps;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc;   /* everything but the topology class */
    D3D12_INPUT_ELEMENT_DESC il[RSX_DSP_NUM_VERTEX_ATTR];
    ID3D12PipelineState* pso[3];               /* by topology class: point, line, triangle */
    int failed[3];
    int live;
    /* A fragment program that reads WPOS, at RSX_SCALE != 1: its text as
     * the decompiler gave it (WPOS not divided), and that build's states,
     * for passes into a target kept at the guest size. NULL otherwise. */
    char* ps_plain;
    ID3DBlob* ps1;
    ID3D12PipelineState* pso1[3];
    int failed1[3];
    /* The same geometry with the snapshot-copy pixel shader (see
     * eng_encode_copy_draw), by topology class, for one target format. */
    ID3D12PipelineState* copy_pso[3];
    DXGI_FORMAT copy_fmt[3];
    int copy_failed[3];
    /* The same reading the target's multisampled twin (eng_ms_*): for a
     * snapshot taken while the twin holds draws the target lacks, instead
     * of a resolve of the whole twin first. */
    ID3D12PipelineState* copy_pso_ms[3];
    DXGI_FORMAT copy_fmt_ms[3];
    int copy_failed_ms[3];
    /* The MSAA build (s_ms_n samples), for passes into MSAA twins. */
    ID3D12PipelineState* psoms[3];
    int failedms[3];
} EngPipeline;
static const char kSnapCopyHLSL[] =
    "Texture2D rsx_tex[16] : register(t0);\n"
    "struct PSInput { float4 position : SV_POSITION; };\n"
    "float4 main(PSInput input) : SV_Target0 { return rsx_tex[0].Load(int3((int2)input.position.xy, 0)); }\n";
static ID3DBlob* s_snap_copy_ps;
/* The same from a multisampled twin, the samples averaged: what a resolve
 * of the pixel would give. Built for the sample count in use (eng_ms_apply). */
static const char kSnapCopyMsHLSL[] =
    "Texture2DMS<float4, %u> rsx_tex_ms : register(t0);\n"
    "struct PSInput { float4 position : SV_POSITION; };\n"
    "float4 main(PSInput input) : SV_Target0 {\n"
    "  int2 p = (int2)input.position.xy; float4 c = 0;\n"
    "  [unroll] for (int s = 0; s < %u; s++) c += rsx_tex_ms.Load(p, s);\n"
    "  return c / %u; }\n";
static ID3DBlob* s_snap_copy_ms_ps;
/* RSX_MS_TWIN=0 (read on first use; the host A/B, DOD3_AB=mstwin, flips it):
 * snapshots resolve the whole twin first and passes without a real depth
 * target draw single-sampled, as before. */
int g_rsx_ms_twin = -1;
static u64 s_snap_incr_n, s_snap_full_n, s_snap_copy_draws;   /* RSX_GPU_TIME */
int g_rsx_snap_incr = -1;   /* RSX_SNAP_INCR (see eng_encode_records) */


typedef struct { u64 hash; ID3DBlob* blob; } EngBlob;
typedef struct { u64 key; u32 border; } EngSampler;   /* the descriptor lives at CPU sampler slot i+1 */
typedef struct { u32 surface, remap, format, view; } EngView;

typedef enum {
    ENG_REC_DRAW, ENG_REC_CLEAR_COLOR, ENG_REC_CLEAR_DS, ENG_REC_DEPTH_RESOLVE,
    ENG_REC_COLOR_COPY, ENG_REC_CLEAR_DS_RECT, ENG_REC_UPLOAD,
    ENG_REC_RT_AO,       /* RT_AO: rt[0] = scene colour, depth = scene depth, rt_job */
    ENG_REC_COLOR_RECT   /* depth = source target, resolve_dst = destination, clear_rect = sx,sy,sw,sh, sc = dx,dy,dw,dh */
} EngRecKind;

typedef struct {
    EngRecKind kind;
    u32 rt[RSX_BE_MAX_COLOR_TARGETS];
    u32 nrt;
    u32 depth;
    u32 pipeline;
    rsx_topology topology;
    u32 vb_off, stride, vertex_count;
    u32 ib_off, index_count;
    u32 vs_cb_off, vs_cb_bytes;
    u32 ps_cb_off, ps_cb_bytes;
    u32 tex[RSX_BE_MAX_TEXTURES];
    int samp[RSX_BE_MAX_TEXTURES];
    u32 vtex[RSX_BE_MAX_VERTEX_TEXTURES];
    int vsamp[RSX_BE_MAX_VERTEX_TEXTURES];
    float vp[4];
    u32   sc[4];
    u32   stencil_ref;
    float blend[4];
    float clear_rgba[4];
    u32   clear_flags;
    float clear_depth;
    u8    clear_stencil;
    u32   resolve_dst;
    u32   resolve_packed;
    u32   vis;
    u32   vbuf;
    u32   clear_rect[4];
    /* ENG_REC_UPLOAD: stage offset, subresource and footprint. */
    u32   up_off, up_sub, up_w, up_h, up_pitch;
    u32   rt_job;
    u8    rt_defer;   /* RT_AO: a transparent draw, moved after the AO record */
} EngRecord;

/* One UPLOAD-heap chunk, mapped for its whole life. What a submit stages is
 * a chain of them (see staging). */
typedef struct {
    ID3D12Resource* res;
    u8* mapped;
    D3D12_GPU_VIRTUAL_ADDRESS va;
    u32 cap;
    u32 base, used;   /* in a chain: where it starts in the records' offsets, and its fill */
    u64 fence;        /* the submit that last used it */
    u32 seq;          /* and that submit's number */
    int in_use;       /* 1: handed to a submit that may still be running; 2: in the chain being built */
} EngStage;
#define ENG_STAGE_POOL 64
#define ENG_STAGE_KEEP 8      /* chunks kept however long they go unused */
#define ENG_STAGE_IDLE 600u   /* submits a chunk beyond those is kept unused */


typedef struct { u32 slot, index; } EngVisReport;

/* A submit the GPU may still be running. */
typedef struct {
    u64 fence;
    u32 q_slot, q_count;             /* its queries, in allocator slot q_slot's region */
    u32* q_vis;                      /* vis counter per query */
    EngVisReport* reports; u32 n_reports;
    int live;
} EngSubmit;
#define ENG_MAX_SUBMITS 16

/* ---- state ---------------------------------------------------------------- */

static int s_active, s_headless, s_ready;
static HWND s_hwnd; static int s_window_closed;
/* 04/11 menu110: exclusive full screen. DXGI Present on the encode/present thread then SetWindowLong()s the window, owned by the
 * window (pump) thread, and waits for it to answer the sent message. If the window thread is itself blocked waiting for that
 * present (enc_join, present_join, the B2 drain), nobody answers: deadlock for ever (menu109 run a1, menu110 run a1 after
 * SetFullscreenState was moved to the window thread). So when the window thread waits for the present path it answers the
 * messages SENT to its window (not the posted ones: no handler runs that was not already running while it was blocked). */
static DWORD s_win_tid;
#ifndef PM_QS_SENDMESSAGE
#define PM_QS_SENDMESSAGE (QS_SENDMESSAGE << 16)
#endif
void rsx_d3d12_engine_service_sent(void)
{
    if (!s_win_tid || GetCurrentThreadId() != s_win_tid) return;
    MSG m;
    while (PeekMessageA(&m, NULL, 0, 0, PM_REMOVE | PM_QS_SENDMESSAGE)) { }
}
static void eng_wait_serv(HANDLE h)
{
    if (!s_win_tid || GetCurrentThreadId() != s_win_tid) { WaitForSingleObject(h, INFINITE); return; }
    for (;;) {
        if (MsgWaitForMultipleObjectsEx(1, &h, INFINITE, QS_SENDMESSAGE, 0) == WAIT_OBJECT_0) return;
        rsx_d3d12_engine_service_sent();
    }
}
/* Exclusive full screen is dropped by DXGI when the window loses focus and
 * does not come back on its own: set by WM_ACTIVATEAPP, acted on before the
 * next present (SetFullscreenState and the ResizeBuffers it requires must
 * not run inside the message handler). */
static int s_fs_restore;
static volatile LONG s_fs_left;   /* the window thread left exclusive full screen: the submit thread must ResizeBuffers before the next Present (else DXGI_ERROR_INVALID_CALL, black screen) */
static ID3D12Device* s_dev;
static u64 s_pc_adapter_id;   /* vendor/device/revision: the pipeline library is per GPU */
static IDXGIAdapter3* s_adapter3;   /* RSX_VRAM_LOG: the device's video memory use and budget */
static void eng_pc_save(int final);
static int eng_vram_query(unsigned long long* used, unsigned long long* budget);
static ID3D12CommandQueue* s_queue;
static IDXGISwapChain3* s_swap;
static ID3D12Resource* s_backbuf[3];
static ID3D12DescriptorHeap* s_backbuf_rtv_heap;
static ID3D12Resource* s_offscreen;              /* headless present target */
static ID3D12Fence* s_fence; static HANDLE s_fence_event; static u64 s_fence_value;
static ID3D12CommandAllocator* s_alloc[ENG_FRAMES]; static u64 s_alloc_fence[ENG_FRAMES];
static ID3D12GraphicsCommandList* s_list; static int s_list_open; static u32 s_slot;
static ID3D12DescriptorHeap* s_srv_cpu;          /* ENG_MAX_OBJECTS + 1 */
static ID3D12DescriptorHeap* s_rtv_cpu;
static ID3D12DescriptorHeap* s_dsv_cpu;
static ID3D12DescriptorHeap* s_smp_cpu;          /* ENG_MAX_SAMPLERS + 1 */
static ID3D12DescriptorHeap* s_srv_gpu;          /* ENG_FRAMES * ENG_SRV_PER_SLOT */
static ID3D12DescriptorHeap* s_smp_gpu[ENG_FRAMES];
/* Sampler heaps beyond the slot's own: a full sampler heap (2048, the D3D12
 * maximum = ~100 distinct 20-sampler tables) switches the open list to a
 * spare one instead of submitting and waiting for the GPU to go idle. A spare
 * is reused once the submit that last used it has completed. Ratchet ToD's
 * metropolis filled the slot heap 2-3 times a frame: ~140 ms/s of walker
 * stalls with the GPU idle. RSX_SMP_SWITCH=0: the old flush-and-wait. */
#define ENG_SMP_SPARES 64
static ID3D12DescriptorHeap* s_smp_spare[ENG_SMP_SPARES];
static u64 s_smp_spare_fence[ENG_SMP_SPARES];
static u8  s_smp_spare_inlist[ENG_SMP_SPARES];
static u32 s_smp_spare_n;
static ID3D12DescriptorHeap* s_smp_cur;      /* the sampler heap bound on the open list */
static u32 s_srv_step, s_rtv_step, s_dsv_step, s_smp_step;
static u32 s_srv_used, s_smp_used;               /* within the current slot's region */
static ID3D12QueryHeap* s_qheap; static ID3D12Resource* s_qread; static u32 s_q_used;
/* RSX_GPU_TIME=1: a timestamp at the start and end of every command list.
 * Every 5 s: submits a second, the GPU's busy time, the mean GPU time of a
 * submit the walker waits on and of one it does not, and how long a list sat
 * in the queue before the GPU started it -- is a synchronous wait GPU work,
 * or latency? */
static ID3D12QueryHeap* s_tsheap; static ID3D12Resource* s_tsread; static u64 s_ts_freq;
volatile double g_rsx_gpu_busy_ms_total;   /* usr123: cumulative GPU time of all submitted lists (ms), for the overlay's GPU load (TOD_GPU_LOAD=0 turns the timestamps off) */
volatile int g_rsx_gpu_load_ok;
static FILE* s_gts_fp; static unsigned s_gts_n; static int s_gt_report = 1;   /* RSX_GPU_TS_DUMP */
static struct { u64 fence; int kind; LARGE_INTEGER cpu_exec; } s_ts_slot[ENG_FRAMES];
static int s_submit_kind;   /* 0 present / other, 1 waited on */
static u64 s_gt_draws, s_gt_bar_calls, s_gt_bars, s_gt_pso, s_gt_queries, s_gt_copies;
static u64 s_gt_nopso;   /* draws skipped: no pipeline (compile failed or still compiling) */

/* RSX_GPU_TIME=2: also a timestamp wherever the recorded work changes target
 * (a "pass": the draws into one set of targets, a clear, a copy, a depth
 * resolve, the uploads), and every 5 s the passes that took the most GPU
 * time, by kind, target size and format. Where the GPU time of a frame goes
 * at 4K, which scales differently from the walker's. */
#define ENG_PASS_MAX 1024u
typedef struct { u8 kind; u8 nrt; u16 w, h; u32 fmt; u32 draws; u32 garea; } EngPassDesc;
static EngPassDesc s_pass[ENG_FRAMES][ENG_PASS_MAX]; static u32 s_pass_n[ENG_FRAMES];
static ID3D12QueryHeap* s_pheap; static ID3D12Resource* s_pread;
static ID3D12RootSignature* s_rootsig;
static ID3D12RootSignature* s_helper_rootsig;
static ID3D12PipelineState* s_blit_pso, *s_depth_pso, *s_depth_pack_pso;
static ID3D12PipelineState* s_depth_ms_pso, *s_depth_pack_ms_pso;   /* MSAA depth reads */
static ID3DBlob* s_helper_vs_blob; static ID3DBlob* s_helper_blit_blob; /* kept: MSAA refresh pipelines */
static ID3D12PipelineState* s_zcopy_ms_pso;   /* MSAA depth twin -> its single-sampled target */
static ID3D12PipelineState* s_present_pso, *s_fxaa_pso;   /* rsx_present_passes.h */
int rsx_aa_mode(void);                                    /* rsx_draw_engine.c */
static ID3D12Resource* s_null_tex;
static SRWLOCK s_pipe_lock = SRWLOCK_INIT;
static int s_vsync = 1;
/* Windowed and borderless presents go through the compositor, which holds a
 * flip-model swap chain to the display's refresh even at sync interval 0
 * unless the chain allows tearing. With vsync off and the system supporting
 * it (DXGI_FEATURE_PRESENT_ALLOW_TEARING), the chain is created with the
 * flag and presents carry it, so the frame rate is not capped at the
 * refresh. Exclusive full screen does not need it. */
static int s_tearing = 0;

static EngObj s_obj[ENG_MAX_OBJECTS];
static u32 s_obj_count, s_obj_free[ENG_MAX_OBJECTS], s_obj_free_count;
static EngPipeline s_pipe[ENG_MAX_PIPES]; static u32 s_pipe_count;
static EngBlob s_blob[ENG_MAX_BLOBS]; static u32 s_blob_count;
static EngSampler s_samp[ENG_MAX_SAMPLERS]; static u32 s_samp_count;
/* 27/10: open-addressing hash tables over the two arrays above (index+1, 0 = empty), instead of a linear scan per
 * lookup. The blob one is guarded by s_blob_lock like the array; the sampler one by the walker thread, like s_samp. */
#define ENG_BLOB_HASH 65536u    /* >= 4 x ENG_MAX_BLOBS */
#define ENG_SAMP_HASH 1024u     /* >= 4 x ENG_MAX_SAMPLERS */
static u32 s_blob_ht[ENG_BLOB_HASH];
static u16 s_samp_ht[ENG_SAMP_HASH];
static u32 eng_h32(u64 h) { h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 32; return (u32)h; }
static EngView s_view[ENG_MAX_VIEWS]; static u32 s_view_count;
/* The records of the frame the walker is building. RSX_ENCODE_THREAD=1 keeps a second array and swaps
 * them at each present (eng_submit_async): the thread that encodes owns the one it was handed, the walker
 * the other. Off, s_rec is the first array for good. */
static EngRecord s_rec_store0[ENG_MAX_RECORDS]; static EngRecord* s_rec_store1;
static EngRecord* s_rec = s_rec_store0;
static u32 s_rec_count, s_dropped;
static u32 s_rt_mark = ~0u;   /* RT_AO: record count after the last opaque scene draw */

/* ---- RSX_LANES=1 (lane107, TAPPA 1 of PIANO_CORSIE.md): records go into one LIST PER BLOCK instead of straight into s_rec. ----------------
 * The walker numbers the blocks (K draws of one pass, boundaries get their own); thread B2 / the walker announce the block they are
 * producing for with rsx_d3d12_lanes_block(); every record append (eng_rec_new) lands in that block's list. Before s_rec is consumed
 * (submit, RT AO insertion) eng_lanes_flatten() concatenates the lists in block order into s_rec[s_flat..]: the encode sees the same array.
 * s_rec_count stays the TOTAL record count (flat + listed) so every "any records?" test and s_rt_mark mean what they meant.
 * RSX_LANES_CHECK=1 (with RSX_LANES=1): the records are ALSO written straight into s_rec (today's path, what the encode uses); each list gets a
 * copy of its records, and at flatten the concatenation of the lists is compared with s_rec byte for byte ([lanes-check] lines). */
/* Per-lane backend state (lane108, TAPPA 2): what the draw calls of ONE lane (the g.be bind and draw entries sink_end_impl calls) build up between two
 * draws. One instance today (B2 owns it); tappa 3 = one per lane, LB the thread's own pointer.
 *   pending     the record being assembled (bind_targets .. draw)        vs_cb_*   "the vertex constants staged last draw are still valid" memo
 *   ll_cur      index of the record list of the block this lane is producing for
 *   chk_*       RSX_LANES_CHECK: the last appended record is copied into its list lazily
 * NOT per lane yet (shared, one writer): the upload arena chain (s_chain/s_stage, eng_stage_reserve): the lane's chunk is the tail of the chain. */
typedef struct {
    EngRecord pending;
    u32 vs_cb_seq, vs_cb_off, vs_cb_bytes;
    u32 ll_cur;
    int chk_pend; u32 chk_list, chk_idx, chk_g;
    /* lane109 (RSX_LANES>=2): a lane thread runs a chunk speculatively (spec = 1): shared state is only read, a write is an abort */
    int spec;
    u32 vis_cur;                                       /* the query the lane's draws belong to (the walker's s_vis_cur, per lane) */
    u32 q_first, q_n, q_k;                             /* the query slots the coordinator reserved for the chunk */
    struct { u32 slot, index; } *vrep; u32 vrep_n, vrep_cap;   /* the chunk's query reports, appended to s_vis_pending at close */
    u8* ar_mapped; u32 ar_base, ar_used, ar_used0, ar_cap, ar_seq;   /* the lane's region of the upload arena */
} EngLaneBe;
static EngLaneBe s_lbe0 = { .vs_cb_seq = ~0u, .ll_cur = ~0u };
static __declspec(thread) EngLaneBe* t_lb;
#define LB (t_lb ? t_lb : &s_lbe0)
typedef struct { u32 blk, pass, kind; EngRecord* v; u32 n, cap; int hold; } EngLaneList;   /* hold: a lane's result not committed yet (flatten drops it) */
extern void rsx_lane_abort(int why);
static u32 s_ll_epoch;                  /* bumped by every flatten: a chunk list from an older epoch is void */
static int s_lanes_mt;                  /* RSX_LANES>=2: lane threads exist */
#define ENG_LANE_MAXLISTS 16384
static int s_lanes = -1, s_lanes_check;
static EngLaneList* s_ll; static u32 s_ll_n;
static u32 s_flat;                       /* s_rec[0, s_flat) is final; [s_flat, s_rec_count) is in the lists */
/* (s_chk_*: EngLaneBe.chk_*) */   /* check: the last appended record is copied into its list lazily (the caller fills it after the append) */
static u64 s_lc_rec, s_lc_diff, s_lc_flat, s_lc_recs_sum; static u32 s_lc_first_said, s_lc_blk_err;
static ULONGLONG s_lc_tick; static u32 s_lc_frame0;
extern uint32_t g_rsx_engine_frame;
static void eng_lanes_flatten(void);
static void eng_lanes_init(void)
{
    const char* e = getenv("RSX_LANES"); s_lanes = (e && atoi(e) >= 1) ? 1 : 0;
    const char* c = getenv("RSX_LANES_CHECK"); s_lanes_check = (s_lanes && c && atoi(c) == 1) ? 1 : 0;   /* =2 is lane109's check (DE side) */
    s_lanes_mt = (e && atoi(e) >= 2) ? 1 : 0;
    if (s_lanes) {
        s_ll = (EngLaneList*)calloc(ENG_LANE_MAXLISTS, sizeof *s_ll);
        if (!s_ll) s_lanes = s_lanes_check = 0;
        fprintf(stderr, "[lanes] RSX_LANES=1 backend: record lists per block%s\n", s_lanes_check ? " + shadow check (RSX_LANES_CHECK=1)" : "");
    }
}
static void eng_lanes_sync_last(void)
{
    if (!LB->chk_pend) return;
    LB->chk_pend = 0;
    s_ll[LB->chk_list].v[LB->chk_idx] = s_rec[LB->chk_g];
}
void rsx_d3d12_lanes_block(unsigned blk, unsigned pass, unsigned kind)
{
    if (s_lanes < 0) eng_lanes_init();
    if (!s_lanes) return;
    if (LB->ll_cur != ~0u && s_ll[LB->ll_cur].blk == blk) return;
    if (s_lanes_check) eng_lanes_sync_last();
    if (s_ll_n >= ENG_LANE_MAXLISTS) eng_lanes_flatten();
    EngLaneList* L = &s_ll[s_ll_n];
    L->blk = blk; L->pass = pass; L->kind = kind; L->n = 0; L->hold = 0;
    LB->ll_cur = s_ll_n++;
}
static EngRecord* eng_rec_new(void)
{
    if (s_lanes < 0) eng_lanes_init();
    if (!s_lanes) return &s_rec[s_rec_count++];
    if (LB->spec) {   /* lane109: the chunk's own list (the coordinator made it); s_rec_count is the coordinator's, added at commit */
        EngLaneList* SL = &s_ll[LB->ll_cur];
        if (SL->n >= SL->cap) {
            const u32 nc = SL->cap ? SL->cap * 2u : 256u;
            EngRecord* nv = (EngRecord*)realloc(SL->v, (size_t)nc * sizeof *nv);
            if (!nv) { fprintf(stderr, "[lanes] out of memory\n"); abort(); }
            SL->v = nv; SL->cap = nc;
        }
        return &SL->v[SL->n++];
    }
    if (s_lanes_check) eng_lanes_sync_last();
    if (LB->ll_cur == ~0u) rsx_d3d12_lanes_block(0xFFFFFFFFu, 0, 0);
    EngLaneList* L = &s_ll[LB->ll_cur];
    if (L->n >= L->cap) {
        const u32 nc = L->cap ? L->cap * 2u : 64u;
        EngRecord* nv = (EngRecord*)realloc(L->v, (size_t)nc * sizeof *nv);
        if (!nv) { fprintf(stderr, "[lanes] out of memory\n"); abort(); }
        L->v = nv; L->cap = nc;
    }
    if (s_lanes_check) {
        LB->chk_pend = 1; LB->chk_list = LB->ll_cur; LB->chk_idx = L->n; LB->chk_g = s_rec_count;
        L->n++;
        return &s_rec[s_rec_count++];
    }
    EngRecord* r = &L->v[L->n++];
    s_rec_count++;
    return r;
}
static void eng_rec_reset(void)
{
    s_rec_count = 0; s_flat = 0; s_ll_n = 0; LB->ll_cur = ~0u; LB->chk_pend = 0;
}
static const struct { u32 off; const char* nm; } s_lfield[] = {
    { (u32)offsetof(EngRecord, kind), "kind" }, { (u32)offsetof(EngRecord, rt), "rt" }, { (u32)offsetof(EngRecord, nrt), "nrt" },
    { (u32)offsetof(EngRecord, depth), "depth" }, { (u32)offsetof(EngRecord, pipeline), "pipeline" }, { (u32)offsetof(EngRecord, topology), "topology" },
    { (u32)offsetof(EngRecord, vb_off), "vb_off" }, { (u32)offsetof(EngRecord, stride), "stride" }, { (u32)offsetof(EngRecord, vertex_count), "vertex_count" },
    { (u32)offsetof(EngRecord, ib_off), "ib_off" }, { (u32)offsetof(EngRecord, index_count), "index_count" },
    { (u32)offsetof(EngRecord, vs_cb_off), "vs_cb_off" }, { (u32)offsetof(EngRecord, vs_cb_bytes), "vs_cb_bytes" },
    { (u32)offsetof(EngRecord, ps_cb_off), "ps_cb_off" }, { (u32)offsetof(EngRecord, ps_cb_bytes), "ps_cb_bytes" },
    { (u32)offsetof(EngRecord, tex), "tex" }, { (u32)offsetof(EngRecord, samp), "samp" }, { (u32)offsetof(EngRecord, vtex), "vtex" },
    { (u32)offsetof(EngRecord, vsamp), "vsamp" }, { (u32)offsetof(EngRecord, vp), "vp" }, { (u32)offsetof(EngRecord, sc), "sc" },
    { (u32)offsetof(EngRecord, stencil_ref), "stencil_ref" }, { (u32)offsetof(EngRecord, blend), "blend" },
    { (u32)offsetof(EngRecord, clear_rgba), "clear_rgba" }, { (u32)offsetof(EngRecord, clear_flags), "clear_flags" },
    { (u32)offsetof(EngRecord, clear_depth), "clear_depth" }, { (u32)offsetof(EngRecord, clear_stencil), "clear_stencil" },
    { (u32)offsetof(EngRecord, resolve_dst), "resolve_dst" }, { (u32)offsetof(EngRecord, resolve_packed), "resolve_packed" },
    { (u32)offsetof(EngRecord, vis), "vis" }, { (u32)offsetof(EngRecord, vbuf), "vbuf" }, { (u32)offsetof(EngRecord, clear_rect), "clear_rect" },
    { (u32)offsetof(EngRecord, up_off), "up_off" }, { (u32)offsetof(EngRecord, up_sub), "up_sub" }, { (u32)offsetof(EngRecord, up_w), "up_w" },
    { (u32)offsetof(EngRecord, up_h), "up_h" }, { (u32)offsetof(EngRecord, up_pitch), "up_pitch" }, { (u32)offsetof(EngRecord, rt_job), "rt_job" },
    { (u32)offsetof(EngRecord, rt_defer), "rt_defer" },
};
static const char* eng_lane_field(u32 byte)
{
    const char* nm = "pad";
    for (u32 i = 0; i < sizeof s_lfield / sizeof s_lfield[0]; i++) if (s_lfield[i].off <= byte) nm = s_lfield[i].nm;
    return nm;
}
/* The lists, concatenated in block order, become s_rec[s_flat..s_rec_count). Runs wherever s_rec is about to be consumed. */
static void eng_lanes_flatten(void)
{
    if (s_lanes <= 0) return;
    if (s_lanes_check) eng_lanes_sync_last();
    u32 w = s_flat; u32 diff = 0; u32 prev_blk = 0; int have_prev = 0;
    const u32 fr_rec0 = s_rec_count - s_flat;
    for (u32 i = 0; i < s_ll_n; i++) {
        EngLaneList* L = &s_ll[i];
        if (L->hold) { L->n = 0; L->hold = 0; continue; }   /* an uncommitted lane result: void (its chunk is redone, see s_ll_epoch) */
        if (!L->n) continue;
        if (L->blk != 0xFFFFFFFFu) {
            if (have_prev && L->blk < prev_blk) { s_lc_blk_err++; diff++; if (!s_lc_first_said) { s_lc_first_said = 1; fprintf(stderr, "[lanes-check] FIRST DIFF: block order: list %u block %u after block %u\n", i, L->blk, prev_blk); } }
            prev_blk = L->blk; have_prev = 1;
        }
        if (s_lanes_check) {
            for (u32 k = 0; k < L->n; k++, w++) {
                if (w >= s_rec_count) { diff++; continue; }
                if (memcmp(&s_rec[w], &L->v[k], sizeof(EngRecord))) {
                    diff++;
                    if (!s_lc_first_said) {
                        s_lc_first_said = 1;
                        const u8* a = (const u8*)&s_rec[w]; const u8* b = (const u8*)&L->v[k]; u32 o = 0;
                        while (o < sizeof(EngRecord) && a[o] == b[o]) o++;
                        fprintf(stderr, "[lanes-check] FIRST DIFF: frame %u record %u (block %u pass %u kind %u, index %u in the block) field %s (+%u) kind=%d: direct 0x%02X list 0x%02X\n",
                                g_rsx_engine_frame, w, L->blk, L->pass, L->kind, k, eng_lane_field(o), o, (int)s_rec[w].kind, a[o], b[o]);
                    }
                }
            }
        } else {
            memcpy(&s_rec[w], L->v, (size_t)L->n * sizeof(EngRecord)); w += L->n;
        }
        L->n = 0;
    }
    if (w != s_rec_count) {
        diff++;
        if (!s_lc_first_said) { s_lc_first_said = 1; fprintf(stderr, "[lanes-check] FIRST DIFF: count: lists %u vs s_rec_count %u (flat %u)\n", w, s_rec_count, s_flat); }
    }
    s_flat = s_rec_count; s_ll_n = 0; LB->ll_cur = ~0u; s_ll_epoch++;
    s_lc_flat++; s_lc_rec += fr_rec0; s_lc_diff += diff; s_lc_recs_sum += fr_rec0;
    const ULONGLONG now = GetTickCount64();
    if (now - s_lc_tick >= 2000) {
        const u32 fr = g_rsx_engine_frame - s_lc_frame0;
        if (s_lc_tick) {
            if (s_lanes_check) fprintf(stderr, "[lanes-check] frame %u rec=%llu diff=%llu (flattens %llu, block-order errors %u)\n", g_rsx_engine_frame, s_lc_rec, s_lc_diff, s_lc_flat, s_lc_blk_err);
            fprintf(stderr, "[lanes] records: %.1f per frame over %u frames (%llu flattens, total %llu)\n", fr ? (double)s_lc_recs_sum / fr : 0.0, fr, s_lc_flat, s_lc_rec);
        }
        s_lc_tick = now; s_lc_frame0 = g_rsx_engine_frame; s_lc_recs_sum = 0;
    }
}
static u32 s_pres_n;           /* presents so far (RT_AO views, A/B dumps) */
static u8 s_rt_defer_next;     /* RT_AO: the next draw record is a transparent one */
/* (LB->pending: EngLaneBe.pending) */
static ID3D12Resource* s_buf[ENG_MAX_BUFS]; static u32 s_buf_bytes[ENG_MAX_BUFS];
static u32 s_buf_count, s_buf_free[ENG_MAX_BUFS], s_buf_free_count; static u8 s_buf_retired[ENG_MAX_BUFS];
static EngStage s_stage[ENG_STAGE_POOL];
static int s_chain[ENG_STAGE_POOL]; static u32 s_chain_n;   /* the chunks of the submit being built */
/* The chain eng_stage_at searches: the records being encoded name offsets of THEIR submit's chain. It is
 * s_chain itself unless a frame is encoded on the encode thread, whose chain was frozen at the hand-over. */
static const int* s_view_chain = s_chain; static const u32* s_view_chain_n = &s_chain_n;
static EngSubmit s_sub[ENG_MAX_SUBMITS];
static u32 s_submit_seq;
static int s_enc_pending;        /* RSX_ENCODE_THREAD (eng_submit_async): walker thread only, a job is out */
static void eng_enc_join_why(const char* why, int is_handover);
/* Wait for the encode thread before touching anything it may own; counted by `why` (a literal) in the log. */
static __declspec(thread) int t_no_place;   /* RSX_DS_HEAP: set around eng_fallback_depth: it runs inside the encode (the synchronous one on the walker has t_on_enc = 0), where no zero-clear record can be added */
static __declspec(thread) int t_on_enc;   /* set on the encode thread: it never joins itself */
/* lane110 RSX_ENCPROF=1: where the encode thread's time goes per frame: job wall, thread CPU, Present + pacing, fence waits (a GPU wait is not CPU work) */
static int s_encprof = -1; static double s_ep_wall, s_ep_cpu, s_ep_present, s_ep_fence; static unsigned s_ep_n;
static int ep_on(void) { if (s_encprof < 0) { const char* e = getenv("RSX_ENCPROF"); s_encprof = (e && e[0] && e[0] != '0') ? 1 : 0; } return s_encprof; }
static double ep_now(void) { static double inv; LARGE_INTEGER q; if (!inv) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); inv = 1000.0 / (double)f.QuadPart; } QueryPerformanceCounter(&q); return (double)q.QuadPart * inv; }
static double ep_cpu(void) { FILETIME c, e, k, u; if (!GetThreadTimes(GetCurrentThread(), &c, &e, &k, &u)) return 0.0;
    return (double)((((ULONGLONG)k.dwHighDateTime << 32) | k.dwLowDateTime) + (((ULONGLONG)u.dwHighDateTime << 32) | u.dwLowDateTime)) / 10000.0; }
#define ENG_JOIN(why) do { if (s_enc_pending && !t_on_enc) { if (LB->spec) rsx_lane_abort(RSX_LAB_JOIN); eng_enc_join_why((why), 0); } } while (0)
static u32 s_enc_seq;            /* submit number of the frame being encoded (RT AO reads it) */
static int s_frame_has_rt;       /* RSX_ENCODE_THREAD: a RT AO record is in the frame being built; it goes the synchronous way */
/* (the vertex-constant memo s_vs_cb_*: EngLaneBe) */
static u64 s_vis_count[ENG_VIS_SLOTS]; static u32 s_vis_next, s_vis_cur;
static EngVisReport s_vis_pending[ENG_MAX_REPORTS]; static u32 s_vis_npending;
static u32 s_q_vis[ENG_QUERIES_PER_SLOT];     /* vis counter per query of the list being encoded */
/* Set on a query whose pass drew into a scaled target: its count is in host
 * pixels and is divided by the scale squared when read. */
#define ENG_Q_SCALED 0x80000000u
/* ... and in an MSAA pass a count is of samples: log2 of the sample count
 * in bits 28-29, divided out when read. */
#define ENG_Q_MS_SHIFT 28
#define ENG_Q_INDEX 0x0FFFFFFFu
#define ENG_Q_XX 0x40000000u   /* 02/11 aa110: the pass drew into the horizontally stretched scene target: 4/3 more pixels */
static u64 eng_q_count(u32 qv, u64 c)
{
    if ((qv & ENG_Q_SCALED) && s_scale != 1.0f) c = (u64)((double)c / ((double)s_scale * (double)s_scale) + 0.5);
    if ((qv & ENG_Q_XX) && (qv & ENG_Q_SCALED)) c = (u64)((double)c * 0.75 + 0.5);
    const u32 ms = (qv >> ENG_Q_MS_SHIFT) & 3u;
    if (ms) c = (c + (1u << ms) / 2) >> ms;
    return c;
}
static u32 s_clear_argb;
static u32 s_fallback_depth[8]; static u32 s_fallback_n;
static u32 s_present_center;                  /* last presented centre pixel, headless */
static u32 s_stat_tex, s_stat_snap, s_stat_snap_reused, s_stat_buf, s_stat_rt;   /* created since the last RSX_OBJ_STATS line */

/* Released depth-snapshot textures, kept for the next request of the same
 * size and format: a committed D3D12 resource costs the walker a kernel
 * round trip to create and another to free, and this title asks for a few
 * snapshots a frame. */
#define ENG_SNAP_POOL 512          /* array size; the working cap is s_snap_pool_cap (RSX_SNAP_POOL, default 256; 64 before 31/10) */
static struct { ID3D12Resource* res; DXGI_FORMAT fmt; u32 w, h; u64 fence; D3D12_RESOURCE_STATES state; u32 heap; u64 heap_off, heap_size; } s_snap_pool[ENG_SNAP_POOL];   /* heap*: RSX_DS_HEAP, a placed snapshot keeps its range while pooled */
static u32 s_snap_pool_n;

/* ---- small helpers -------------------------------------------------------- */

static u64 fnv1a64(const void* data, u32 n, u64 h)
{
    const u8* p = (const u8*)data;
    for (u32 i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

static DXGI_FORMAT eng_dxgi(rsx_be_format f)
{
    switch (f) {
    case RSX_BE_FMT_R8:              return DXGI_FORMAT_R8_UNORM;
    case RSX_BE_FMT_R8G8:            return DXGI_FORMAT_R8G8_UNORM;
    case RSX_BE_FMT_R8G8B8A8:        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case RSX_BE_FMT_BC1:             return DXGI_FORMAT_BC1_UNORM;
    case RSX_BE_FMT_BC2:             return DXGI_FORMAT_BC2_UNORM;
    case RSX_BE_FMT_BC3:             return DXGI_FORMAT_BC3_UNORM;
    case RSX_BE_FMT_R16:             return DXGI_FORMAT_R16_UNORM;
    case RSX_BE_FMT_R16G16:          return DXGI_FORMAT_R16G16_UNORM;
    case RSX_BE_FMT_R16G16F:         return DXGI_FORMAT_R16G16_FLOAT;
    case RSX_BE_FMT_R16G16B16A16F:   return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case RSX_BE_FMT_R32F:            return DXGI_FORMAT_R32_FLOAT;
    case RSX_BE_FMT_R32G32B32A32F:   return DXGI_FORMAT_R32G32B32A32_FLOAT;
    default:                         return DXGI_FORMAT_R8_UNORM;
    }
}

static u32 eng_bpp(DXGI_FORMAT f)
{
    switch (f) {
    case DXGI_FORMAT_R8_UNORM: return 1;
    case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R16_UNORM: return 2;
    case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R32_FLOAT: return 4;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return 8;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return 16;
    default: return 4;
    }
}

/* The crossbar's selectors arrive in A,R,G,B order; RSX_REMAP_ZERO/ONE are
 * 4 and 5, which are D3D12's FORCE_VALUE_0/1. */
static UINT eng_mapping(u32 remap, u32 rsx_fmt)
{
    u8 sel[4];
    rsx_texture_component_remap(remap, rsx_fmt & 0x9Fu, sel);
    return D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(sel[1], sel[2], sel[3], sel[0]);
}

static D3D12_COMPARISON_FUNC gcm_cmp(u32 f)
{
    switch (f) {
    case 0x0200: return D3D12_COMPARISON_FUNC_NEVER;
    case 0x0201: return D3D12_COMPARISON_FUNC_LESS;
    case 0x0202: return D3D12_COMPARISON_FUNC_EQUAL;
    case 0x0203: return D3D12_COMPARISON_FUNC_LESS_EQUAL;
    case 0x0204: return D3D12_COMPARISON_FUNC_GREATER;
    case 0x0205: return D3D12_COMPARISON_FUNC_NOT_EQUAL;
    case 0x0206: return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
    default:     return D3D12_COMPARISON_FUNC_ALWAYS;
    }
}
static D3D12_BLEND gcm_blend_factor(u32 f, int alpha)
{
    switch (f & 0xFFFFu) {
    case 0x0000: return D3D12_BLEND_ZERO;
    case 0x0001: return D3D12_BLEND_ONE;
    case 0x0300: return alpha ? D3D12_BLEND_SRC_ALPHA : D3D12_BLEND_SRC_COLOR;
    case 0x0301: return alpha ? D3D12_BLEND_INV_SRC_ALPHA : D3D12_BLEND_INV_SRC_COLOR;
    case 0x0302: return D3D12_BLEND_SRC_ALPHA;
    case 0x0303: return D3D12_BLEND_INV_SRC_ALPHA;
    case 0x0304: return D3D12_BLEND_DEST_ALPHA;
    case 0x0305: return D3D12_BLEND_INV_DEST_ALPHA;
    case 0x0306: return alpha ? D3D12_BLEND_DEST_ALPHA : D3D12_BLEND_DEST_COLOR;
    case 0x0307: return alpha ? D3D12_BLEND_INV_DEST_ALPHA : D3D12_BLEND_INV_DEST_COLOR;
    case 0x0308: return D3D12_BLEND_SRC_ALPHA_SAT;
    case 0x8001: return D3D12_BLEND_BLEND_FACTOR;
    case 0x8002: return D3D12_BLEND_INV_BLEND_FACTOR;
    case 0x8003: return D3D12_BLEND_BLEND_FACTOR;
    case 0x8004: return D3D12_BLEND_INV_BLEND_FACTOR;
    default:     return D3D12_BLEND_ONE;
    }
}
static D3D12_BLEND_OP gcm_blend_op(u32 e)
{
    switch (e & 0xFFFFu) {
    case 0x8007: return D3D12_BLEND_OP_MIN;
    case 0x8008: return D3D12_BLEND_OP_MAX;
    case 0x800A: return D3D12_BLEND_OP_SUBTRACT;
    case 0x800B: return D3D12_BLEND_OP_REV_SUBTRACT;
    default:     return D3D12_BLEND_OP_ADD;
    }
}
static D3D12_STENCIL_OP gcm_stencil_op(u32 op)
{
    switch (op) {
    case 0x0000: return D3D12_STENCIL_OP_ZERO;
    case 0x1E01: return D3D12_STENCIL_OP_REPLACE;
    case 0x1E02: return D3D12_STENCIL_OP_INCR_SAT;
    case 0x1E03: return D3D12_STENCIL_OP_DECR_SAT;
    case 0x150A: return D3D12_STENCIL_OP_INVERT;
    case 0x8507: return D3D12_STENCIL_OP_INCR;
    case 0x8508: return D3D12_STENCIL_OP_DECR;
    default:     return D3D12_STENCIL_OP_KEEP;
    }
}
static D3D12_TEXTURE_ADDRESS_MODE gcm_wrap(u32 w)
{
    switch (w & 0xFu) {
    case 1:  return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    case 2:  return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
    case 4:  return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    case 6: case 7: case 8: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
    default: return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    }
}
static D3D_PRIMITIVE_TOPOLOGY topo_d3d(rsx_topology t)
{
    switch (t) {
    case RSX_TOPOLOGY_POINTS:         return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
    case RSX_TOPOLOGY_LINES:          return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
    case RSX_TOPOLOGY_LINE_STRIP:     return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
    case RSX_TOPOLOGY_TRIANGLE_STRIP: return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
    default:                          return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    }
}
static int topo_class(rsx_topology t)   /* index into EngPipeline.pso */
{
    switch (t) {
    case RSX_TOPOLOGY_POINTS: return 0;
    case RSX_TOPOLOGY_LINES: case RSX_TOPOLOGY_LINE_STRIP: return 1;
    default: return 2;
    }
}

static D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle(ID3D12DescriptorHeap* heap, u32 step, u32 slot)
{
    D3D12_CPU_DESCRIPTOR_HANDLE h;
    CALL(heap, GetCPUDescriptorHandleForHeapStart, &h);
    h.ptr += (SIZE_T)slot * step;
    return h;
}
static D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle(ID3D12DescriptorHeap* heap, u32 step, u32 slot)
{
    D3D12_GPU_DESCRIPTOR_HANDLE h;
    CALL(heap, GetGPUDescriptorHandleForHeapStart, &h);
    h.ptr += (UINT64)slot * step;
    return h;
}
static D3D12_CPU_DESCRIPTOR_HANDLE obj_srv(u32 handle) { return cpu_handle(s_srv_cpu, s_srv_step, handle ? handle - 1 : ENG_SRV_NULL); }
static D3D12_CPU_DESCRIPTOR_HANDLE obj_rtv(u32 handle) { return cpu_handle(s_rtv_cpu, s_rtv_step, handle - 1); }
static D3D12_CPU_DESCRIPTOR_HANDLE obj_dsv(u32 handle) { return cpu_handle(s_dsv_cpu, s_dsv_step, handle - 1); }
/* The MSAA twins' views, at the same slots in heaps of their own. */
static ID3D12DescriptorHeap* s_rtv_ms; static ID3D12DescriptorHeap* s_dsv_ms; static ID3D12DescriptorHeap* s_srv_ms;
static D3D12_CPU_DESCRIPTOR_HANDLE obj_rtv_ms(u32 handle) { return cpu_handle(s_rtv_ms, s_rtv_step, handle - 1); }
static D3D12_CPU_DESCRIPTOR_HANDLE obj_dsv_ms(u32 handle) { return cpu_handle(s_dsv_ms, s_dsv_step, handle - 1); }
static D3D12_CPU_DESCRIPTOR_HANDLE obj_srv_ms(u32 handle) { return cpu_handle(s_srv_ms, s_srv_step, handle - 1); }

static EngObj* eng_obj(u32 handle)
{
    if (!handle || handle > s_obj_count) return NULL;
    EngObj* o = &s_obj[handle - 1];
    return o->kind ? o : NULL;
}
/* The object that owns the memory: a view resolves to its surface. */
static EngObj* eng_owner(u32 handle)
{
    EngObj* o = eng_obj(handle);
    if (o && o->kind == OBJ_VIEW) o = eng_obj(o->alias);
    return o && o->res ? o : NULL;
}
/* Whether a target (or the surface a view aliases) is at the internal
 * resolution: then its pass's guest-pixel coordinates are scaled. */
static int eng_obj_scaled(u32 handle)
{
    const EngObj* o = eng_owner(handle);
    return o && o->gw && o->gh && (o->w != o->gw || o->h != o->gh);
}
/* The object's host/guest factor: 1 if kept at the guest size. */
static float eng_xk_f(const EngObj* o)
{
    return o->xbloom ? (s_scale > 1.0f ? s_scale : 1.0f) * o->xk : s_scale * o->xk;
}
static float eng_obj_f(u32 handle)
{
    const EngObj* o = eng_owner(handle);
    if (!o || !o->gw || !o->gh || (o->w == o->gw && o->h == o->gh)) return 1.0f;
    return o->xk > 0.0f ? eng_xk_f(o) : s_scale;
}

/* 02/11 aa110: ToD renders its 3D scene into a 960x704 target (2x horizontal MSAA, samples side by side: the zeta is read as
 * 1920x704) and its own pass stretches it to 1280 wide. At RSX_SCALE=3 that is 2880 px upscaled 4/3 to 3840: soft and stair-stepped
 * whatever RSX_AA says. RSX_HSTRETCH_FIX (default 1; 0 = old): that target gets 4/3 more host pixels along x only (3840x2112 at x3),
 * so the stretch is 1:1. Viewports/scissors scale x by f*xx, WPOS by fp_alpha.z+2 (see eng_fp_blob), depth snapshots by z->xx. */
#define ENG_HSTRETCH_K (4.0f / 3.0f)
static int eng_hstretch_on(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("RSX_HSTRETCH_FIX"); on = !(e && e[0] == '0'); }
    return on;
}
static float eng_obj_xx(u32 handle)
{
    const EngObj* o = eng_owner(handle);
    return o && o->xx > 0.0f ? o->xx : 1.0f;
}

static ID3D12Resource* make_buffer(D3D12_HEAP_TYPE type, u64 bytes, D3D12_RESOURCE_STATES st)
{
    D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = type;
    D3D12_RESOURCE_DESC bd = {0};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = bytes; bd.Height = 1;
    bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* r = NULL;
    if (FAILED(CALL(s_dev, CreateCommittedResource, &hp, D3D12_HEAP_FLAG_NONE, &bd, st, NULL,
                    &IID_ID3D12Resource, (void**)&r)))
        return NULL;
    return r;
}

static ID3D12Resource* make_texture(DXGI_FORMAT fmt, u32 w, u32 h, u32 mips, u32 faces,
                                    D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES st,
                                    const D3D12_CLEAR_VALUE* cv)
{
    D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td = {0};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = w; td.Height = h; td.DepthOrArraySize = (UINT16)(faces ? faces : 1);
    td.MipLevels = (UINT16)(mips ? mips : 1); td.Format = fmt; td.SampleDesc.Count = 1;
    td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN; td.Flags = flags;
    ID3D12Resource* r = NULL;
    HRESULT hr = CALL(s_dev, CreateCommittedResource, &hp, D3D12_HEAP_FLAG_NONE, &td, st, cv,
                      &IID_ID3D12Resource, (void**)&r);
    if (FAILED(hr)) {
        static int n = 0;
        { static unsigned nv; nv++; if (nv <= 200 || (nv & (nv - 1)) == 0) V2EV("oom", "\"what\":\"texture\",\"w\":%u,\"h\":%u,\"fmt\":%d,\"hr\":\"0x%08lX\",\"count\":%u,\"at\":\"%s:%d\"", w, h, (int)fmt, (long)hr, nv, __func__, __LINE__); }
        if (n < 16 && ++n) fprintf(stderr, "[rsx engine/d3d12] texture %ux%u fmt %d failed: 0x%08lX\n", w, h, (int)fmt, (long)hr);
        return NULL;
    }
    return r;
}

/* ---- memory blocks ----------------------------------------------------------
 * A committed resource is a heap of its own, and the driver's allocation
 * behind each one is what a level's first frame waited on: several hundred
 * textures and as many vertex buffers at 0.3 to 3 ms apiece. Both now come
 * out of blocks made a few at a time, sub-allocated first fit with a sorted
 * free list (as the memory blocks of rsx_vulkan_engine.c). */
typedef struct { u64 off, size; } MemRange;
typedef struct { MemRange* fr; u32 nfr, frcap, live; } MemList;   /* a block's free ranges, and how many it has out */

static int mem_room(MemList* m)
{
    if (m->nfr < m->frcap) return 1;
    const u32 cap = m->frcap ? m->frcap * 2 : 16;
    MemRange* fr = (MemRange*)realloc(m->fr, cap * sizeof(MemRange));
    if (!fr) return 0;
    m->fr = fr; m->frcap = cap;
    return 1;
}

/* top = 0: first fit from the lowest hole, at its start. top = 1 (RT heaps, small targets): the LAST fitting hole, carved from its END, so the small
 * ones fill the top of a heap and the big ones the bottom. The start is rounded DOWN to the alignment; the lead/tail split below is the same. */
static int mem_take(MemList* m, u64 size, u64 align, u64* out, int top)
{
    for (u32 n = 0; n < m->nfr; n++) {
        const u32 i = top ? m->nfr - 1 - n : n;
        const u64 end = m->fr[i].off + m->fr[i].size;
        if (end < size) continue;
        const u64 start = top ? ((end - size) & ~(align - 1)) : ((m->fr[i].off + align - 1) & ~(align - 1));
        if (start < m->fr[i].off || start + size > end) continue;
        const u64 lead = start - m->fr[i].off, tail = end - (start + size);
        if (lead && tail) {   /* split in two */
            if (!mem_room(m)) return 0;
            memmove(&m->fr[i + 2], &m->fr[i + 1], (m->nfr - i - 1) * sizeof(MemRange));
            m->fr[i].size = lead;
            m->fr[i + 1].off = start + size; m->fr[i + 1].size = tail;
            m->nfr++;
        } else if (lead) { m->fr[i].size = lead; }
        else if (tail) { m->fr[i].off = start + size; m->fr[i].size = tail; }
        else { memmove(&m->fr[i], &m->fr[i + 1], (m->nfr - i - 1) * sizeof(MemRange)); m->nfr--; }
        *out = start;
        m->live++;
        return 1;
    }
    return 0;
}

static void mem_give(MemList* m, u64 off, u64 size)
{
    u32 i = 0;
    while (i < m->nfr && m->fr[i].off < off) i++;
    if (m->live) m->live--;
    if (!mem_room(m)) return;   /* the range is lost to its block, no more */
    memmove(&m->fr[i + 1], &m->fr[i], (m->nfr - i) * sizeof(MemRange));
    m->fr[i].off = off; m->fr[i].size = size; m->nfr++;
    if (i + 1 < m->nfr && m->fr[i].off + m->fr[i].size == m->fr[i + 1].off) {
        m->fr[i].size += m->fr[i + 1].size;
        memmove(&m->fr[i + 1], &m->fr[i + 2], (m->nfr - i - 2) * sizeof(MemRange)); m->nfr--;
    }
    if (i > 0 && m->fr[i - 1].off + m->fr[i - 1].size == m->fr[i].off) {
        m->fr[i - 1].size += m->fr[i].size;
        memmove(&m->fr[i], &m->fr[i + 1], (m->nfr - i - 1) * sizeof(MemRange)); m->nfr--;
    }
}

/* A new block: all of it free. */
static void mem_fresh(MemList* m, u64 size)
{
    m->nfr = 0; m->live = 0;
    mem_give(m, 0, size);
}

/* ---- texture heaps ----------------------------------------------------------
 * A texture is placed in a 64 MB heap: placing one in a heap that exists
 * takes some tens of microseconds. One over half a heap, or that the device
 * will not place, is committed as before. RSX_TEX_HEAPS=0 commits them all.
 *
 * A placed texture holds what its range held before until it is written;
 * every texture here has all its levels uploaded before a draw reads it. */
#define TEX_HEAP_BYTES ((u64)64 << 20)
#define TEX_HEAPS      256
typedef struct { ID3D12Heap* heap; MemList m; } TexHeap;
static TexHeap s_theap[TEX_HEAPS]; static u32 s_ntheap;
int g_eng_tex_heaps = 1;

/* The texture, placed; NULL when it is to be committed instead. */
static ID3D12Resource* make_placed_texture(DXGI_FORMAT fmt, u32 w, u32 h, u32 mips, u32 faces,
                                           D3D12_RESOURCE_STATES st, u32* heap, u64* off, u64* size)
{
    *heap = 0;
    if (!g_eng_tex_heaps) return NULL;
    D3D12_RESOURCE_DESC td = {0};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = w; td.Height = h; td.DepthOrArraySize = (UINT16)(faces ? faces : 1);
    td.MipLevels = (UINT16)(mips ? mips : 1); td.Format = fmt; td.SampleDesc.Count = 1;
    td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    D3D12_RESOURCE_ALLOCATION_INFO ai = {0};
    CALL(s_dev, GetResourceAllocationInfo, &ai, 0, 1, &td);
    if (!ai.SizeInBytes || ai.SizeInBytes > TEX_HEAP_BYTES / 2 || !ai.Alignment) return NULL;
    u32 i; u64 o = 0;
    for (i = 0; i < s_ntheap; i++)
        if (s_theap[i].heap && mem_take(&s_theap[i].m, ai.SizeInBytes, ai.Alignment, &o, 0)) break;
    if (i == s_ntheap) {
        for (i = 0; i < s_ntheap; i++) if (!s_theap[i].heap) break;
        if (i == TEX_HEAPS) return NULL;
        TexHeap* t = &s_theap[i];
        D3D12_HEAP_DESC hd = {0};
        hd.SizeInBytes = TEX_HEAP_BYTES; hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        hd.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
        if (FAILED(CALL(s_dev, CreateHeap, &hd, &IID_ID3D12Heap, (void**)&t->heap))) { t->heap = NULL; return NULL; }
        if (i == s_ntheap) s_ntheap++;
        mem_fresh(&t->m, TEX_HEAP_BYTES);
        if (!mem_take(&t->m, ai.SizeInBytes, ai.Alignment, &o, 0)) return NULL;
    }
    ID3D12Resource* r = NULL;
    if (FAILED(CALL(s_dev, CreatePlacedResource, s_theap[i].heap, o, &td, st, NULL,
                    &IID_ID3D12Resource, (void**)&r))) {
        mem_give(&s_theap[i].m, o, ai.SizeInBytes);
        return NULL;
    }
    *heap = i + 1; *off = o; *size = ai.SizeInBytes;
    return r;
}

/* ---- RT heaps (RSX_RT_HEAP=1, 2 = check mode; default 0 = off, as before) -----------------------------------
 * The single-sample colour targets of eng_color_target_create are placed in heaps of RSX_RT_HEAP_MB (256) MB made with
 * D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES, sub-allocated first fit by the same MemList as the texture heaps: a creation
 * is then a CreatePlacedResource (tens of microseconds) instead of a CreateCommittedResource (0.87 ms measured, Rykan V).
 * EngObj.heap codes the owner: 1..TEX_HEAPS = texture heap, RTH_TAG + 1 + i = RT heap i (heap_size != 0 for both, which is
 * what keeps a placed object out of the RSX_REAP reaper). A target over half a heap, a full table, a heap that would take the
 * VRAM past RSX_RT_HEAP_GUARD (92) percent of the budget, or a full record array is committed as before.
 *
 * A placed target holds the bytes its range held before: eng_color_target_create records a zero clear (ENG_REC_CLEAR_COLOR with
 * clear_flags bit 0: an aliasing barrier first) before anything else names the handle, so it reads as the zero a committed one is.
 * The range goes back (eng_collect_retired -> theap_done -> rtheap_done) only after the retire fence is done and no record waits.
 *
 * RSX_RT_HEAP_MAX (default 8, hard cap RT_HEAPS_MAX 16): past that many live heaps a target that fits no free range is committed as before
 * (counter "heaps full": the same ~0.9 ms CreateCommittedResource as with RSX_RT_HEAP=0, never an error). Eight 256 MB heaps = 2 GB of
 * the 8 GB card; raise it only if "heaps full" keeps counting in the 5 s log lines. For the 4K MSAA x8 play (488 targets = 1398 MB live)
 * 8 x 256 MB is enough; RSX_RT_HEAP_MB=512 would halve the CreateHeap count but each one costs more (CreateHeap grows with the size)
 * and VRAM moves in 512 MB steps; with RSX_RT_HEAP_SPARE=1 the cost is off the walker, so 256 stays.
 *
 * RSX_RT_HEAP_SPARE=1 (default 0; needs RSX_RT_HEAP>=1): a service thread ("rsx rt heap maker") creates ONE empty spare heap ahead of time
 * (CreateHeap is free-threaded) and keeps it in s_sp_heap with its MemList already mem_fresh. make_placed_rt takes the spare (a swap under
 * a light lock) instead of calling CreateHeap on the walker, and wakes the thread to make the next one. A miss (spare not ready: first
 * target of the level, or two heaps needed in less than ~20 ms, or the VRAM guard holds it back) falls back to CreateHeap in line as
 * before and is counted. The spare is not in s_nrth / the RT heap stats until taken; it counts against RSX_RT_HEAP_MAX when taken
 * (the thread does not make one when the live heaps already reach the max). Guard: the thread only makes the spare when used + 2 heaps
 * stays under RSX_RT_HEAP_GUARD percent (the spare and the heap that will be needed next), re-checked every second while it is empty. */
#define RTH_TAG       0x10000u
#define RT_HEAPS_MAX  16
typedef struct { ID3D12Heap* heap; MemList m; } RtHeap;
static RtHeap s_rth[RT_HEAPS_MAX]; static u32 s_nrth;
static int s_rth_mode = -1; static u64 s_rth_bytes; static u32 s_rth_max, s_rth_guard;
static int s_rth_spare;                       /* RSX_RT_HEAP_SPARE */
static u64 s_rth_small = (u64)8 << 20;        /* RSX_RT_HEAP_SMALL_MB (8; 0 = off): targets under this size are placed from the top of a heap */
static volatile LONG s_sp_demand;             /* set by the walker on "no hole" / heaps full / vram guard; cleared by the maker when it makes an RT spare */
static volatile LONG s_rth_alive;             /* live heaps (s_rth[].heap != NULL), kept for the maker thread; written by the walker only */
static SRWLOCK s_sp_lock = SRWLOCK_INIT;      /* guards s_sp_heap, s_sp_m, s_sp_make_ms */
static ID3D12Heap* s_sp_heap; static MemList s_sp_m; static double s_sp_make_ms;
static HANDLE s_sp_evt, s_sp_th; static volatile LONG s_sp_stop;
static u64 s_sp_hit, s_sp_miss;               /* walker only */ static u64 s_rt_ref_frag, s_rt_ref_full; static volatile LONG s_mk_guard[2];   /* guard refusals of a NEW heap on the walker: existing heaps hold enough free bytes in total (fragmented) / do not; the maker thread's refusals [RT, DS] (the guard never stops a placement into an existing heap: the mem_take loop runs first) */
static unsigned long long s_mp_budget_ovr;    /* tentative definition: the real one (RSX_VRAM_BUDGET_MB) is further down */
typedef struct { u32 heap; u64 off, size, fence; } RtRel;
static RtRel s_rth_ring[64]; static u32 s_rth_ring_pos, s_rth_ring_n;   /* mode 2: the last 64 ranges given back */
static struct {
    u64 placed, fb_total, fb_big, fb_full, fb_vram, fb_rec, fb_fail, live_n, live_bytes, heaps_made, heaps_freed, alias, alias_near;
    u64 w_placed, w_comm; double w_placed_ms, w_comm_ms, heap_ms_tot, last_made_ms, last_log_ms;
    /* RSX_RT_HEAP_NOJOIN / phase split of eng_color_target_create (5 s window): join, alloc (placed or committed), descriptors (obj_add + SRV + RTV), zero record, seed upload */
    u64 ph_n, ph_nojoin; double ph_join, ph_alloc, ph_desc, ph_rec, ph_seed, ph_max;
} s_rts;
static double rt_now(void) { static double k = 0; LARGE_INTEGER q; if (k <= 0) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); k = 1000.0 / (double)f.QuadPart; } QueryPerformanceCounter(&q); return (double)q.QuadPart * k; }
static int rt_heap_mode(void)
{
    if (s_rth_mode < 0) {
        const char* e = getenv("RSX_RT_HEAP"); const int v = e ? atoi(e) : 0;
        s_rth_mode = v < 0 ? 0 : v > 2 ? 2 : v;
        if (s_rth_mode) {
            const char* m = getenv("RSX_RT_HEAP_MB"); long mb = m ? atol(m) : 256; if (mb < 64) mb = 64; if (mb > 1024) mb = 1024;
            const char* x = getenv("RSX_RT_HEAP_MAX"); long mx = x ? atol(x) : 8; if (mx < 1) mx = 1; if (mx > RT_HEAPS_MAX) mx = RT_HEAPS_MAX;
            const char* g = getenv("RSX_RT_HEAP_GUARD"); long gp = g ? atol(g) : 92; if (gp < 50) gp = 50; if (gp > 100) gp = 100;
            s_rth_bytes = (u64)mb << 20; s_rth_max = (u32)mx; s_rth_guard = (u32)gp;
            const char* sp = getenv("RSX_RT_HEAP_SPARE"); s_rth_spare = (sp && sp[0] == '1');
            { const char* sm = getenv("RSX_RT_HEAP_SMALL_MB"); long smb = sm ? atol(sm) : 8; if (smb < 0) smb = 0; s_rth_small = (u64)smb << 20; }
            fprintf(stderr, "[rt-heap] on: heaps of %ld MB (max %ld, VRAM guard %ld%%)%s%s\n", mb, mx, gp, s_rth_spare ? ", spare heap made off the walker" : "", s_rth_mode == 2 ? ", check mode (alias counters; D3D12_DBG=1 for the debug layer)" : "");
        }
    }
    return s_rth_mode;
}
/* ---- DS heaps (RSX_DS_HEAP=1, 2 = check mode; default 0 = off, as before) ------------------------------------------
 * The RT-heap idea for the two kinds of target that were still committed on the walker / thread B2: the DEPTH targets (eng_depth_target_create_host,
 * D32S8 as R32G8X24_TYPELESS, up to 90 MB at 4K scale 3) and the DEPTH SNAPSHOTS (eng_depth_snapshot_make: R32F / RGBA8 copies of the zeta, 24-47 MB).
 * They are placed in heaps of RSX_DS_HEAP_MB (512) MB (ALLOW_ONLY_RT_DS_TEXTURES, as the RT heaps; a separate array so the 90 MB zeta do not fragment
 * the colour heaps), at most RSX_DS_HEAP_MAX (4, hard cap RT_HEAPS_MAX), the same VRAM guard (RSX_DS_HEAP_GUARD, 92 %), the same trim (an empty heap past
 * the first goes after 10 s without a new one) and the same spare (RSX_DS_HEAP_SPARE, or RSX_RT_HEAP_SPARE=1: the "rsx rt heap maker" thread makes one
 * spare DS heap as well). EngObj.heap = DSH_TAG + 1 + i (heap_size != 0, so RSX_REAP releases it inline); the range goes back in theap_done -> dsheap_done.
 * What a placed target holds before its first write is undefined, and a committed one is zero, so:
 *  - a depth target gets an ENG_REC_CLEAR_DS record (depth 0, stencil 0: what a committed zeta is born with) with DSH_INIT_FLAG in clear_flags = an aliasing
 *    barrier first (eng_ds_zero_record), recorded at once after the creation, ahead of every other use of the handle;
 *  - a snapshot is written whole by its DEPTH_RESOLVE pass, but a render target must be initialised after an aliasing barrier: EngObj.dsinit makes the
 *    first DEPTH_RESOLVE that names it do barrier + ClearRenderTargetView(0) before the pass (a zero clear, not Discard: the contents stay what a committed
 *    one has even if the pass is skipped). A snapshot that goes to the pool keeps its range (s_snap_pool[].heap*); the range goes back only when the pool
 *    drops it (eng_snap_pool_free_entry: eviction, rescale, shutdown), always past its fence (pool entries are, by construction). A pool hit needs no new
 *    barrier: the resource is the same one that was initialised. */
#define DSH_TAG        0x20000u
#define DSH_INIT_FLAG  0x40000000u   /* ENG_REC_CLEAR_DS clear_flags: aliasing barrier first */
static RtHeap s_dsh[RT_HEAPS_MAX]; static u32 s_ndsh;
static int s_dsh_mode = -1; static u64 s_dsh_bytes; static u32 s_dsh_max, s_dsh_guard;
static int s_dsh_spare;                       /* RSX_DS_HEAP_SPARE, default = RSX_RT_HEAP_SPARE */
static volatile LONG s_dsh_alive;             /* live DS heaps, for the maker thread */
static ID3D12Heap* s_sp_dheap; static MemList s_sp_dm; static double s_sp_dmake_ms;   /* the DS spare: guarded by s_sp_lock like the RT one */
static u64 s_sp_dhit, s_sp_dmiss;
static SRWLOCK s_dsh_lk = SRWLOCK_INIT;       /* s_dsh, s_ndsh, s_dss, the ring: depth targets (walker) and snapshots (walker or B2) both place here */
static RtRel s_dsh_ring[64]; static u32 s_dsh_ring_pos, s_dsh_ring_n;
static struct {
    u64 placed[2], fb_total[2], w_placed[2], w_comm[2];   /* [0] depth targets, [1] snapshots */
    u64 fb_big, fb_full, fb_vram, fb_rec, fb_fail, live_n, live_bytes, heaps_made, heaps_freed, alias, alias_near, pool_ret;
    double w_placed_ms[2], w_comm_ms[2], heap_ms_tot, last_made_ms, last_log_ms;
} s_dss;
static int ds_heap_mode(void)
{
    if (s_dsh_mode < 0) {
        const char* e = getenv("RSX_DS_HEAP"); const int v = e ? atoi(e) : 0;
        const int m = v < 0 ? 0 : v > 2 ? 2 : v;
        if (m) {
            const char* mb_ = getenv("RSX_DS_HEAP_MB"); long mb = mb_ ? atol(mb_) : 512; if (mb < 128) mb = 128; if (mb > 2048) mb = 2048;
            const char* x = getenv("RSX_DS_HEAP_MAX"); long mx = x ? atol(x) : 4; if (mx < 1) mx = 1; if (mx > RT_HEAPS_MAX) mx = RT_HEAPS_MAX;
            const char* g = getenv("RSX_DS_HEAP_GUARD"); long gp = g ? atol(g) : 92; if (gp < 50) gp = 50; if (gp > 100) gp = 100;
            const char* sp = getenv("RSX_DS_HEAP_SPARE"); if (!sp) sp = getenv("RSX_RT_HEAP_SPARE");
            s_dsh_bytes = (u64)mb << 20; s_dsh_max = (u32)mx; s_dsh_guard = (u32)gp; s_dsh_spare = (sp && sp[0] == '1');
            fprintf(stderr, "[ds-heap] on: heaps of %ld MB (max %ld, VRAM guard %ld%%) for depth targets and zeta snapshots%s%s\n", mb, mx, gp, s_dsh_spare ? ", spare heap made off the walker" : "", m == 2 ? ", check mode (alias counters)" : "");
        }
        s_dsh_mode = m;   /* last: the other threads read the settings above only when the mode is set */
    }
    return s_dsh_mode;
}
/* The spare-heap maker (RSX_RT_HEAP_SPARE=1). Sleeps on s_sp_evt (1 s timeout while the spare is missing, to re-check the VRAM guard). */
/* One spare for the RT heaps (ds = 0) or the DS heaps (ds = 1). */
static void rtheap_make_one(int ds)
{
    const u64 bytes = ds ? s_dsh_bytes : s_rth_bytes;
    const u32 guard = ds ? s_dsh_guard : s_rth_guard;
    AcquireSRWLockShared(&s_sp_lock); const int have = (ds ? s_sp_dheap : s_sp_heap) != NULL; ReleaseSRWLockShared(&s_sp_lock);
    if (have || (u32)(ds ? s_dsh_alive : s_rth_alive) >= (ds ? s_dsh_max : s_rth_max)) return;
    if (!ds && !s_sp_demand) return;   /* RT spare only when the walker needed a heap (or fell back) since the last one: an idle spare holds no VRAM */
    if (s_adapter3) {   /* eng_vram_query keeps unsynchronised statics: ask DXGI directly */
        DXGI_QUERY_VIDEO_MEMORY_INFO vi = {0};
        if (SUCCEEDED(CALL(s_adapter3, QueryVideoMemoryInfo, 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &vi))) {
            unsigned long long budget = s_mp_budget_ovr ? s_mp_budget_ovr : vi.Budget;
            if (budget && vi.CurrentUsage + bytes > budget / 100 * guard) { InterlockedIncrement(&s_mk_guard[ds ? 1 : 0]); return; }
        }
    }
    D3D12_HEAP_DESC hd = {0};
    hd.SizeInBytes = bytes; hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES;
    ID3D12Heap* h = NULL; const double t0 = rt_now();
    if (FAILED(CALL(s_dev, CreateHeap, &hd, &IID_ID3D12Heap, (void**)&h)) || !h) { Sleep(250); return; }
    const double ms = rt_now() - t0;
    MemList m; memset(&m, 0, sizeof m); mem_fresh(&m, bytes);
    AcquireSRWLockExclusive(&s_sp_lock);
    if (s_sp_stop || (ds ? s_sp_dheap : s_sp_heap)) { ReleaseSRWLockExclusive(&s_sp_lock); RELEASE(h); free(m.fr); return; }
    if (ds) { s_sp_dheap = h; s_sp_dm = m; s_sp_dmake_ms += ms; }
    else    { s_sp_heap = h;  s_sp_m = m;  s_sp_make_ms += ms; InterlockedExchange(&s_sp_demand, 0); }
    ReleaseSRWLockExclusive(&s_sp_lock);
}
static DWORD WINAPI rtheap_maker(LPVOID arg)
{
    (void)arg;
    for (;;) {
        WaitForSingleObject(s_sp_evt, 1000);
        if (s_sp_stop) break;
        if (s_rth_spare) rtheap_make_one(0);
        if (s_sp_stop) break;
        if (s_dsh_spare && s_dsh_mode > 0) rtheap_make_one(1);
    }
    return 0;
}
static volatile LONG s_sp_starting;   /* walker and B2 can both make the first target: one maker thread only (cleared by rtheap_spare_stop or a failed start) */
static void rtheap_spare_start(void)
{
    if (s_sp_th || (!s_rth_spare && !s_dsh_spare)) return;
    if (InterlockedCompareExchange(&s_sp_starting, 1, 0)) return;
    s_sp_stop = 0;
    s_sp_evt = CreateEventW(NULL, FALSE, TRUE, NULL);   /* auto-reset, signalled: make the first heap now */
    if (!s_sp_evt) { s_rth_spare = 0; s_dsh_spare = 0; s_sp_starting = 0; return; }
    s_sp_th = CreateThread(NULL, 1u << 20, rtheap_maker, NULL, 0, NULL);
    if (!s_sp_th) { CloseHandle(s_sp_evt); s_sp_evt = NULL; s_rth_spare = 0; s_dsh_spare = 0; s_sp_starting = 0; return; }
    SetThreadDescription(s_sp_th, L"rsx rt heap maker");
}
static void rtheap_spare_stop(void)
{
    if (s_sp_th) {
        s_sp_stop = 1; SetEvent(s_sp_evt);
        WaitForSingleObject(s_sp_th, 10000);
        CloseHandle(s_sp_th); s_sp_th = NULL;
    }
    if (s_sp_evt) { CloseHandle(s_sp_evt); s_sp_evt = NULL; }
    RELEASE(s_sp_heap); free(s_sp_m.fr); memset(&s_sp_m, 0, sizeof s_sp_m);
    s_sp_make_ms = 0.0; s_sp_hit = s_sp_miss = 0; s_rth_alive = 0;
    RELEASE(s_sp_dheap); free(s_sp_dm.fr); memset(&s_sp_dm, 0, sizeof s_sp_dm);   /* RSX_DS_HEAP spare */
    s_sp_dmake_ms = 0.0; s_sp_dhit = s_sp_dmiss = 0; s_dsh_alive = 0;
    s_sp_starting = 0;
}
/* heaps alive, their bytes, and the bytes of live (or retired, not yet given back) placed targets inside them */
static u64 rtheap_stats(u32* n, u64* live)
{
    u32 c = 0; for (u32 i = 0; i < s_nrth; i++) if (s_rth[i].heap) c++;
    *n = c; *live = s_rts.live_bytes;
    return (u64)c * s_rth_bytes;
}
/* The target's resource at a range of an RT heap; NULL when it is to be committed instead (the reason is counted). */
static ID3D12Resource* make_placed_rt(DXGI_FORMAT df, u32 w, u32 h, const D3D12_CLEAR_VALUE* cv, u32* heap, u64* off, u64* size)
{
    *heap = 0;
    rtheap_spare_start();   /* first use: the thread starts making the first heap now */
    D3D12_RESOURCE_DESC td = {0};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = w; td.Height = h; td.DepthOrArraySize = 1; td.MipLevels = 1; td.Format = df; td.SampleDesc.Count = 1;
    td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN; td.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_RESOURCE_ALLOCATION_INFO ai = {0};
    CALL(s_dev, GetResourceAllocationInfo, &ai, 0, 1, &td);
    if (!ai.SizeInBytes || !ai.Alignment) { s_rts.fb_fail++; return NULL; }
    if (ai.SizeInBytes > s_rth_bytes / 2) { s_rts.fb_big++; return NULL; }
    u32 i; u64 o = 0;
    const int top = ai.SizeInBytes < s_rth_small;   /* RSX_RT_HEAP_SMALL_MB: small targets from the top of a heap, big ones from the bottom */
    for (i = 0; i < s_nrth; i++)
        if (s_rth[i].heap && mem_take(&s_rth[i].m, ai.SizeInBytes, ai.Alignment, &o, top)) break;
    if (i == s_nrth) {
        InterlockedExchange(&s_sp_demand, 1);   /* no hole in the heaps that exist: the maker may build a spare */
        for (i = 0; i < s_nrth; i++) if (!s_rth[i].heap) break;
        if (i >= s_rth_max) { s_rts.fb_full++; return NULL; }
        RtHeap* t = &s_rth[i];
        int from_spare = 0;
        if (s_rth_spare && s_sp_th) {   /* the spare: a swap, no CreateHeap on the walker */
            AcquireSRWLockExclusive(&s_sp_lock);
            if (s_sp_heap) { free(t->m.fr); t->heap = s_sp_heap; t->m = s_sp_m; s_sp_heap = NULL; memset(&s_sp_m, 0, sizeof s_sp_m); from_spare = 1; }
            ReleaseSRWLockExclusive(&s_sp_lock);
            SetEvent(s_sp_evt);   /* make the next one (also on a miss: it may have been held back by the guard) */
        }
        if (from_spare) {
            s_sp_hit++; s_rts.heaps_made++; s_rts.last_made_ms = rt_now();
            InterlockedIncrement(&s_rth_alive);
            if (s_rts.heaps_made <= 16) fprintf(stderr, "[rt-heap] new heap %u of %llu MB: taken from the spare (no CreateHeap on the walker)\n", i, (unsigned long long)(s_rth_bytes >> 20));
            if (i == s_nrth) s_nrth++;
        } else {
        if (s_rth_spare) s_sp_miss++;
        { unsigned long long used = 0, budget = 0;
          if (eng_vram_query(&used, &budget) && budget && used + s_rth_bytes > budget / 100 * s_rth_guard) { s_rts.fb_vram++; InterlockedExchange(&s_sp_demand, 1); { u64 fb = 0; for (u32 k = 0; k < s_nrth; k++) if (s_rth[k].heap) for (u32 j = 0; j < s_rth[k].m.nfr; j++) fb += s_rth[k].m.fr[j].size; if (fb >= ai.SizeInBytes) s_rt_ref_frag++; else s_rt_ref_full++; } return NULL; } }
        D3D12_HEAP_DESC hd = {0};
        hd.SizeInBytes = s_rth_bytes; hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        hd.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES;
        const double h0 = rt_now();
        if (FAILED(CALL(s_dev, CreateHeap, &hd, &IID_ID3D12Heap, (void**)&t->heap))) { t->heap = NULL; s_rts.fb_fail++; return NULL; }
        const double hms = rt_now() - h0;
        s_rts.heap_ms_tot += hms; s_rts.heaps_made++; s_rts.last_made_ms = h0;
        InterlockedIncrement(&s_rth_alive);
        if (s_rts.heaps_made <= 16) fprintf(stderr, "[rt-heap] new heap %u of %llu MB: CreateHeap %.2f ms\n", i, (unsigned long long)(s_rth_bytes >> 20), hms);
        if (i == s_nrth) s_nrth++;
        mem_fresh(&t->m, s_rth_bytes);
        }
        if (!mem_take(&t->m, ai.SizeInBytes, ai.Alignment, &o, top)) return NULL;
    }
    if (s_rth_mode == 2)   /* check mode: a range that overlaps one given back in the last 64 gives, and how many submits ago */
        for (u32 k = 0; k < s_rth_ring_n; k++) {
            const RtRel* e = &s_rth_ring[k];
            if (e->heap == i && o < e->off + e->size && e->off < o + ai.SizeInBytes) { s_rts.alias++; if (s_fence_value - e->fence <= 2) s_rts.alias_near++; }
        }
    ID3D12Resource* r = NULL;
    if (FAILED(CALL(s_dev, CreatePlacedResource, s_rth[i].heap, o, &td, D3D12_RESOURCE_STATE_RENDER_TARGET, cv,
                    &IID_ID3D12Resource, (void**)&r))) {
        mem_give(&s_rth[i].m, o, ai.SizeInBytes);
        s_rts.fb_fail++;
        return NULL;
    }
    s_rts.live_n++; s_rts.live_bytes += ai.SizeInBytes;
    *heap = RTH_TAG + i + 1; *off = o; *size = ai.SizeInBytes;
    return r;
}
/* An RT-heap range goes back (the resource is already released and its fence is done). Heaps are only let go by rtheap_trim. */
static void rtheap_done(u32 heap, u64 off, u64 size)
{
    const u32 i = heap - RTH_TAG - 1;
    if (i >= s_nrth || !s_rth[i].heap) return;
    mem_give(&s_rth[i].m, off, size);
    if (s_rts.live_n) s_rts.live_n--;
    s_rts.live_bytes = s_rts.live_bytes >= size ? s_rts.live_bytes - size : 0;
    if (s_rth_mode == 2) {
        RtRel* e = &s_rth_ring[s_rth_ring_pos++ & 63u];
        e->heap = i; e->off = off; e->size = size; e->fence = s_fence_value;
        if (s_rth_ring_n < 64) s_rth_ring_n++;
    }
}
/* Empty heaps past the first are released, but only when none was made in the last 10 s (a spike must not make and free a heap in turn). */
static void rtheap_trim(double now)
{
    if (now - s_rts.last_made_ms < 10000.0) return;
    /* Rule: keep one empty heap, plus one more while the spare is not ready (so the next need never has to create one on the walker);
     * the spare itself is never touched here. With the spare ready, the next need is served by it and the maker builds another. */
    u32 keep = 1;
    if (s_rth_spare && s_sp_th) { AcquireSRWLockShared(&s_sp_lock); if (!s_sp_heap) keep = 2; ReleaseSRWLockShared(&s_sp_lock); }
    u32 empty = 0;
    for (u32 i = 0; i < s_nrth; i++) if (s_rth[i].heap && !s_rth[i].m.live) empty++;
    for (u32 i = s_nrth; i-- > 0 && empty > keep;)
        if (s_rth[i].heap && !s_rth[i].m.live) { RELEASE(s_rth[i].heap); s_rth[i].m.nfr = 0; s_rts.heaps_freed++; empty--; InterlockedDecrement(&s_rth_alive); }
}
static void rtheap_log(double now)
{
    if (now - s_rts.last_log_ms < 5000.0) return;
    s_rts.last_log_ms = now;
    rtheap_trim(now);
    u32 n = 0; u64 live = 0; const u64 tot = rtheap_stats(&n, &live);
    fprintf(stderr, "[rt-heap] heaps %u (%llu MB), placed targets %llu (live %llu = %llu MB), fallback committed %llu (big %llu, heaps full %llu, vram guard %llu, records %llu, failed %llu), "
            "create ms avg placed %.3f (%llu in 5 s) / committed %.3f (%llu in 5 s), heaps made %llu freed %llu (CreateHeap %.1f ms total)\n",
            n, (unsigned long long)(tot >> 20), (unsigned long long)s_rts.placed, (unsigned long long)s_rts.live_n, (unsigned long long)(live >> 20),
            (unsigned long long)s_rts.fb_total, (unsigned long long)s_rts.fb_big, (unsigned long long)s_rts.fb_full, (unsigned long long)s_rts.fb_vram,
            (unsigned long long)s_rts.fb_rec, (unsigned long long)s_rts.fb_fail,
            s_rts.w_placed ? s_rts.w_placed_ms / (double)s_rts.w_placed : 0.0, (unsigned long long)s_rts.w_placed,
            s_rts.w_comm ? s_rts.w_comm_ms / (double)s_rts.w_comm : 0.0, (unsigned long long)s_rts.w_comm,
            (unsigned long long)s_rts.heaps_made, (unsigned long long)s_rts.heaps_freed, s_rts.heap_ms_tot);
    fprintf(stderr, "[rt-heap] vram guard: new heap refused on the walker %llu (existing heaps held enough free bytes but no hole: %llu, not enough free bytes: %llu), refused in the maker thread RT %ld DS %ld; placements into existing heaps are never guarded; governor frees colour committed %u placed %u (skipped placed %u), depth committed %u placed %u (skipped placed %u)\n",
            (unsigned long long)s_rts.fb_vram, (unsigned long long)s_rt_ref_frag, (unsigned long long)s_rt_ref_full, (long)s_mk_guard[0], (long)s_mk_guard[1],
            g_rsx_gov_ev[0], g_rsx_gov_ev[1], g_rsx_gov_ev[4], g_rsx_gov_ev[2], g_rsx_gov_ev[3], g_rsx_gov_ev[5]);
    if (s_rth_spare) {
        AcquireSRWLockShared(&s_sp_lock); const double mk = s_sp_make_ms; const int rdy = s_sp_heap != NULL; ReleaseSRWLockShared(&s_sp_lock);
        fprintf(stderr, "[rt-heap] spare hits %llu miss %llu, spare make ms %.1f (off the walker), spare %s\n",
                (unsigned long long)s_sp_hit, (unsigned long long)s_sp_miss, mk, rdy ? "ready" : "empty");
    }
    { /* per heap: largest free hole (MB) / number of free ranges; the free bytes the guard line calls "enough but no hole" are in these */
        char hb[512]; int hn = 0; hb[0] = 0;
        for (u32 k = 0; k < s_nrth && hn < 440; k++) {
            if (!s_rth[k].heap) { hn += snprintf(hb + hn, sizeof hb - (size_t)hn, " -"); continue; }
            u64 big = 0, fr = 0; for (u32 j = 0; j < s_rth[k].m.nfr; j++) { fr += s_rth[k].m.fr[j].size; if (s_rth[k].m.fr[j].size > big) big = s_rth[k].m.fr[j].size; }
            hn += snprintf(hb + hn, sizeof hb - (size_t)hn, " %llu/%u(free %llu)", (unsigned long long)(big >> 20), s_rth[k].m.nfr, (unsigned long long)(fr >> 20));
        }
        AcquireSRWLockShared(&s_sp_lock); const int rdy2 = s_sp_heap != NULL; ReleaseSRWLockShared(&s_sp_lock);
        fprintf(stderr, "[rt-heap] holes (largest MB/free ranges(free MB), per heap):%s; spare ready %s; small-target top placement under %llu MB; spare demand %ld\n",
                hb, rdy2 ? "yes" : "no", (unsigned long long)(s_rth_small >> 20), (long)s_sp_demand);
    }
    if (s_rth_mode == 2)
        fprintf(stderr, "[rt-heap] check: placements overlapping a range given back in the last 64 gives: %llu, of them within 2 submits: %llu\n",
                (unsigned long long)s_rts.alias, (unsigned long long)s_rts.alias_near);
    if (s_rts.ph_n)
        fprintf(stderr, "[rt-heap] create phases (%llu targets in 5 s, %llu without join, ms per target): join %.3f alloc %.3f descriptors %.3f zero-record %.3f seed %.3f total %.3f (worst %.2f)\n",
                (unsigned long long)s_rts.ph_n, (unsigned long long)s_rts.ph_nojoin, s_rts.ph_join / (double)s_rts.ph_n, s_rts.ph_alloc / (double)s_rts.ph_n,
                s_rts.ph_desc / (double)s_rts.ph_n, s_rts.ph_rec / (double)s_rts.ph_n, s_rts.ph_seed / (double)s_rts.ph_n,
                (s_rts.ph_join + s_rts.ph_alloc + s_rts.ph_desc + s_rts.ph_rec + s_rts.ph_seed) / (double)s_rts.ph_n, s_rts.ph_max);
    s_rts.ph_n = s_rts.ph_nojoin = 0; s_rts.ph_join = s_rts.ph_alloc = s_rts.ph_desc = s_rts.ph_rec = s_rts.ph_seed = s_rts.ph_max = 0.0;
    s_rts.w_placed = s_rts.w_comm = 0; s_rts.w_placed_ms = s_rts.w_comm_ms = 0.0;
}
/* ---- DS heap placement (see the comment at s_dsh) ---- */
static u64 dsheap_stats(u32* n, u64* live)
{
    u32 c = 0; for (u32 i = 0; i < s_ndsh; i++) if (s_dsh[i].heap) c++;
    *n = c; *live = s_dss.live_bytes;
    return (u64)c * s_dsh_bytes;
}
/* A range of a DS heap for ai; the caller holds s_dsh_lk. 0 = none (the reason is counted): the caller commits the resource instead. */
static int dsheap_take(const D3D12_RESOURCE_ALLOCATION_INFO* ai, u32* hi, u64* ho)
{
    u32 i; u64 o = 0;
    for (i = 0; i < s_ndsh; i++)
        if (s_dsh[i].heap && mem_take(&s_dsh[i].m, ai->SizeInBytes, ai->Alignment, &o, 0)) break;
    if (i == s_ndsh) {
        for (i = 0; i < s_ndsh; i++) if (!s_dsh[i].heap) break;
        if (i >= s_dsh_max) { s_dss.fb_full++; return 0; }
        RtHeap* t = &s_dsh[i];
        int from_spare = 0;
        if (s_dsh_spare && s_sp_th) {   /* the spare: a swap, no CreateHeap on the walker */
            AcquireSRWLockExclusive(&s_sp_lock);
            if (s_sp_dheap) { free(t->m.fr); t->heap = s_sp_dheap; t->m = s_sp_dm; s_sp_dheap = NULL; memset(&s_sp_dm, 0, sizeof s_sp_dm); from_spare = 1; }
            ReleaseSRWLockExclusive(&s_sp_lock);
            SetEvent(s_sp_evt);   /* make the next one (also on a miss: the guard may have held it back) */
        }
        if (from_spare) {
            s_sp_dhit++; s_dss.heaps_made++; s_dss.last_made_ms = rt_now();
            InterlockedIncrement(&s_dsh_alive);
            if (s_dss.heaps_made <= 16) fprintf(stderr, "[ds-heap] new heap %u of %llu MB: taken from the spare (no CreateHeap on the walker)\n", i, (unsigned long long)(s_dsh_bytes >> 20));
            if (i == s_ndsh) s_ndsh++;
        } else {
            if (s_dsh_spare) s_sp_dmiss++;
            { unsigned long long used = 0, budget = 0;
              if (eng_vram_query(&used, &budget) && budget && used + s_dsh_bytes > budget / 100 * s_dsh_guard) { s_dss.fb_vram++; return 0; } }
            D3D12_HEAP_DESC hd = {0};
            hd.SizeInBytes = s_dsh_bytes; hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
            hd.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES;
            const double h0 = rt_now();
            if (FAILED(CALL(s_dev, CreateHeap, &hd, &IID_ID3D12Heap, (void**)&t->heap))) { t->heap = NULL; s_dss.fb_fail++; return 0; }
            const double hms = rt_now() - h0;
            s_dss.heap_ms_tot += hms; s_dss.heaps_made++; s_dss.last_made_ms = h0;
            InterlockedIncrement(&s_dsh_alive);
            if (s_dss.heaps_made <= 16) fprintf(stderr, "[ds-heap] new heap %u of %llu MB: CreateHeap %.2f ms\n", i, (unsigned long long)(s_dsh_bytes >> 20), hms);
            if (i == s_ndsh) s_ndsh++;
            mem_fresh(&t->m, s_dsh_bytes);
        }
        if (!mem_take(&t->m, ai->SizeInBytes, ai->Alignment, &o, 0)) return 0;
    }
    *hi = i; *ho = o;
    return 1;
}
/* The depth target (snap = 0) or zeta snapshot (snap = 1) as a placed resource; NULL when it is to be committed instead.
 * The caller initialises it: depth target = eng_ds_zero_record, snapshot = EngObj.dsinit. */
static ID3D12Resource* make_placed_ds(int snap, DXGI_FORMAT df, u32 w, u32 h, D3D12_RESOURCE_FLAGS fl, D3D12_RESOURCE_STATES st,
                                      const D3D12_CLEAR_VALUE* cv, u32* heap, u64* off, u64* size)
{
    *heap = 0; (void)snap;
    rtheap_spare_start();   /* first use: the thread starts making the first spare now */
    D3D12_RESOURCE_DESC td = {0};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = w; td.Height = h; td.DepthOrArraySize = 1; td.MipLevels = 1; td.Format = df; td.SampleDesc.Count = 1;
    td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN; td.Flags = fl;
    D3D12_RESOURCE_ALLOCATION_INFO ai = {0};
    CALL(s_dev, GetResourceAllocationInfo, &ai, 0, 1, &td);   /* real size and alignment (64 KB for single-sample; asked, not assumed) */
    AcquireSRWLockExclusive(&s_dsh_lk);   /* note: dsheap_take may call CreateHeap / eng_vram_query with this lock held (spare off or empty): other DS callers wait for it */
    ID3D12Resource* r = NULL;
    if (!ai.SizeInBytes || !ai.Alignment) s_dss.fb_fail++;
    else if (ai.SizeInBytes > s_dsh_bytes / 2) s_dss.fb_big++;
    else {
        u32 i = 0; u64 o = 0;
        if (dsheap_take(&ai, &i, &o)) {
            if (s_dsh_mode == 2)   /* check mode: a range overlapping one given back in the last 64 gives, and how many submits ago */
                for (u32 k = 0; k < s_dsh_ring_n; k++) {
                    const RtRel* e = &s_dsh_ring[k];
                    if (e->heap == i && o < e->off + e->size && e->off < o + ai.SizeInBytes) { s_dss.alias++; if (s_fence_value - e->fence <= 2) s_dss.alias_near++; }
                }
            if (FAILED(CALL(s_dev, CreatePlacedResource, s_dsh[i].heap, o, &td, st, cv, &IID_ID3D12Resource, (void**)&r))) {
                mem_give(&s_dsh[i].m, o, ai.SizeInBytes); s_dss.fb_fail++; r = NULL;
            } else {
                s_dss.live_n++; s_dss.live_bytes += ai.SizeInBytes;
                *heap = DSH_TAG + i + 1; *off = o; *size = ai.SizeInBytes;
            }
        }
    }
    ReleaseSRWLockExclusive(&s_dsh_lk);
    return r;
}
/* A DS-heap range goes back (its resource is released and its fence is done). Heaps are only let go by dsheap_trim. */
static void dsheap_done(u32 heap, u64 off, u64 size)
{
    const u32 i = heap - DSH_TAG - 1;
    AcquireSRWLockExclusive(&s_dsh_lk);
    if (i < s_ndsh && s_dsh[i].heap) {
        mem_give(&s_dsh[i].m, off, size);
        if (s_dss.live_n) s_dss.live_n--;
        s_dss.live_bytes = s_dss.live_bytes >= size ? s_dss.live_bytes - size : 0;
        if (s_dsh_mode == 2) {
            RtRel* e = &s_dsh_ring[s_dsh_ring_pos++ & 63u];
            e->heap = i; e->off = off; e->size = size; e->fence = s_fence_value;
            if (s_dsh_ring_n < 64) s_dsh_ring_n++;
        }
    }
    ReleaseSRWLockExclusive(&s_dsh_lk);
}
/* Empty heaps past the first are released, only when none was made in the last 10 s. The caller holds s_dsh_lk. */
static void dsheap_trim(double now)
{
    if (now - s_dss.last_made_ms < 10000.0) return;
    u32 keep = 1;
    if (s_dsh_spare && s_sp_th) { AcquireSRWLockShared(&s_sp_lock); if (!s_sp_dheap) keep = 2; ReleaseSRWLockShared(&s_sp_lock); }
    u32 empty = 0;
    for (u32 i = 0; i < s_ndsh; i++) if (s_dsh[i].heap && !s_dsh[i].m.live) empty++;
    for (u32 i = s_ndsh; i-- > 0 && empty > keep;)
        if (s_dsh[i].heap && !s_dsh[i].m.live) { RELEASE(s_dsh[i].heap); s_dsh[i].m.nfr = 0; s_dss.heaps_freed++; empty--; InterlockedDecrement(&s_dsh_alive); }
}
/* One create counted (snap: 0 depth target, 1 snapshot; placed or committed; d = ms of the allocation), and the 5 s log line. */
static void ds_note(int snap, int placed, double d)
{
    const double now = rt_now();
    AcquireSRWLockExclusive(&s_dsh_lk);
    if (placed) { s_dss.placed[snap]++; s_dss.w_placed[snap]++; s_dss.w_placed_ms[snap] += d; }
    else        { s_dss.fb_total[snap]++; s_dss.w_comm[snap]++; s_dss.w_comm_ms[snap] += d; }
    if (now - s_dss.last_log_ms >= 5000.0) {
        s_dss.last_log_ms = now;
        dsheap_trim(now);
        u32 n = 0, pp = 0; u64 live = 0; const u64 tot = dsheap_stats(&n, &live);
        for (u32 i = 0; i < s_snap_pool_n; i++) if (s_snap_pool[i].heap_size) pp++;
        fprintf(stderr, "[ds-heap] heaps %u (%llu MB), placed depth %llu snapshots %llu (live %llu = %llu MB, %u of them in the snapshot pool), fallback committed depth %llu snapshots %llu "
                "(big %llu, heaps full %llu, vram guard %llu, records %llu, failed %llu), create ms avg depth placed %.3f (%llu in 5 s) / committed %.3f (%llu), "
                "snapshot placed %.3f (%llu) / committed %.3f (%llu), heaps made %llu freed %llu (CreateHeap %.1f ms total)\n",
                n, (unsigned long long)(tot >> 20), (unsigned long long)s_dss.placed[0], (unsigned long long)s_dss.placed[1], (unsigned long long)s_dss.live_n, (unsigned long long)(live >> 20), pp,
                (unsigned long long)s_dss.fb_total[0], (unsigned long long)s_dss.fb_total[1],
                (unsigned long long)s_dss.fb_big, (unsigned long long)s_dss.fb_full, (unsigned long long)s_dss.fb_vram, (unsigned long long)s_dss.fb_rec, (unsigned long long)s_dss.fb_fail,
                s_dss.w_placed[0] ? s_dss.w_placed_ms[0] / (double)s_dss.w_placed[0] : 0.0, (unsigned long long)s_dss.w_placed[0],
                s_dss.w_comm[0] ? s_dss.w_comm_ms[0] / (double)s_dss.w_comm[0] : 0.0, (unsigned long long)s_dss.w_comm[0],
                s_dss.w_placed[1] ? s_dss.w_placed_ms[1] / (double)s_dss.w_placed[1] : 0.0, (unsigned long long)s_dss.w_placed[1],
                s_dss.w_comm[1] ? s_dss.w_comm_ms[1] / (double)s_dss.w_comm[1] : 0.0, (unsigned long long)s_dss.w_comm[1],
                (unsigned long long)s_dss.heaps_made, (unsigned long long)s_dss.heaps_freed, s_dss.heap_ms_tot);
        if (s_dsh_spare) {
            AcquireSRWLockShared(&s_sp_lock); const double mk = s_sp_dmake_ms; const int rdy = s_sp_dheap != NULL; ReleaseSRWLockShared(&s_sp_lock);
            fprintf(stderr, "[ds-heap] spare hits %llu miss %llu, spare make ms %.1f (off the walker), spare %s\n",
                    (unsigned long long)s_sp_dhit, (unsigned long long)s_sp_dmiss, mk, rdy ? "ready" : "empty");
        }
        if (s_dsh_mode == 2)
            fprintf(stderr, "[ds-heap] check: placements overlapping a range given back in the last 64 gives: %llu, of them within 2 submits: %llu\n",
                    (unsigned long long)s_dss.alias, (unsigned long long)s_dss.alias_near);
        s_dss.w_placed[0] = s_dss.w_placed[1] = s_dss.w_comm[0] = s_dss.w_comm[1] = 0;
        s_dss.w_placed_ms[0] = s_dss.w_placed_ms[1] = s_dss.w_comm_ms[0] = s_dss.w_comm_ms[1] = 0.0;
    }
    ReleaseSRWLockExclusive(&s_dsh_lk);
}
/* The zero a committed zeta is born with (depth 0, stencil 0), recorded ahead of every other use of the handle; DSH_INIT_FLAG = aliasing barrier first
 * (see ENG_REC_CLEAR_DS in eng_encode_records). */
static void eng_ds_zero_record(u32 handle)
{
    EngRecord* r = eng_rec_new();
    memset(r, 0, sizeof *r);
    r->kind = ENG_REC_CLEAR_DS; r->depth = handle;
    r->clear_flags = RSX_BE_CLEAR_DEPTH | RSX_BE_CLEAR_STENCIL | DSH_INIT_FLAG;
    r->clear_depth = 0.0f; r->clear_stencil = 0;
}
/* depthpool: the same zero for a REUSED zeta (RSX_DEPTH_POOL): its resource never went away, so no aliasing barrier (no DSH_INIT_FLAG). */
static void eng_ds_clear_record(u32 handle)
{
    EngRecord* r = eng_rec_new();
    memset(r, 0, sizeof *r);
    r->kind = ENG_REC_CLEAR_DS; r->depth = handle;
    r->clear_flags = RSX_BE_CLEAR_DEPTH | RSX_BE_CLEAR_STENCIL;
    r->clear_depth = 0.0f; r->clear_stencil = 0;
}
/* The zero a committed target is born with: recorded ahead of every other use of the handle. clear_flags bit 0 = aliasing barrier first
 * (see ENG_REC_CLEAR_COLOR in eng_encode_records). Not eng_clear_color: that one also sets the debug colour s_clear_argb. */
static void eng_rt_zero_record(u32 handle)
{
    EngRecord* r = eng_rec_new();
    memset(r, 0, sizeof *r);
    r->kind = ENG_REC_CLEAR_COLOR; r->rt[0] = handle; r->nrt = 1; r->clear_flags = 1u;
}

/* A placed texture's range goes back to its heap, once the texture itself is
 * released. One empty heap is kept for the next textures; a second is let go. */
static void theap_done(u32 heap, u64 off, u64 size)
{
    if (!heap) return;
    if (heap >= DSH_TAG) { dsheap_done(heap, off, size); return; }   /* RSX_DS_HEAP: a placed depth target or zeta snapshot */
    if (heap >= RTH_TAG) { rtheap_done(heap, off, size); return; }   /* RSX_RT_HEAP: a placed colour target */
    TexHeap* t = &s_theap[heap - 1];
    mem_give(&t->m, off, size);
    if (t->m.live) return;
    for (u32 i = 0; i < s_ntheap; i++)
        if (&s_theap[i] != t && s_theap[i].heap && !s_theap[i].m.live) { RELEASE(t->heap); t->m.nfr = 0; break; }
}

/* A lock the walker may wait for (the build threads hold these): taken at once
 * when free; otherwise the wait is a beacon phase and logged when it is long. */
/* surf54 31/10: the walker's waits, for the [hitch] line (rsx_draw_engine.c): pc_lock (the PSO library: the load of a state and
 * the Serialize of a save) apart from the rest. */
extern double g_rsx_frame_lock_ms, g_rsx_frame_pclock_ms, g_rsx_frame_pcsave_ms, g_rsx_frame_join_ms;
extern unsigned g_rsx_frame_lock_n, g_rsx_frame_pclock_n, g_rsx_frame_pcsave_n;
extern int (*g_rsx_vram_hook)(unsigned long long* used, unsigned long long* budget);
extern unsigned long long (*g_rsx_parked_hook)(void);
static unsigned long long eng_parked_twin_bytes(void);
extern int (*g_rsx_zeta_direct_hook)(u32 depth, u32 gw, u32 gh);   /* RSX_ZETA_SRV (rsx_draw_engine.c) */
extern int rsx_d3d12_zeta_direct_ok(u32 depth, u32 gw, u32 gh);
extern unsigned long long (*g_rsx_target_info_hook)(u32 handle, int* has_twin);   /* rissa128 RSX_SURF_POOL2 (rsx_draw_engine.c) */
static unsigned long long eng_target_info(u32 handle, int* has_twin);
extern int (*g_rsx_placed_hook)(u32 handle); extern unsigned g_rsx_gov_ev[6];   /* the governor's view of RSX_RT_HEAP / RSX_DS_HEAP (rsx_draw_engine.c) */
static int eng_gov_placed(u32 handle) { const EngObj* o = eng_obj(handle); if (!o || (o->kind != OBJ_COLOR && o->kind != OBJ_DEPTH) || o->heap < RTH_TAG) return 0; return o->ms ? 2 : 1; }
extern unsigned long long (*g_rsx_doomed_hook)(void);   /* rissa128 RSX_REAP: bytes queued for release (rsx_draw_engine.c, the governor) */
unsigned long long eng_doomed_bytes(void);
extern const char* (*g_rsx_cache_dir_hook)(void);
extern unsigned long long (*g_rsx_cache_sig_hook)(void);   /* pso110 */
extern int (*g_rsx_cache_warm_hook)(void);
extern void (*g_rsx_cache_flush_hook)(void);
static u32 s_ms_n;   /* pso112: MSAA samples (defined below), read by the cache signature */
static unsigned long long eng_cache_sig_hook(void);
static int eng_cache_warm_hook(void);
static void eng_cache_flush_hook(void);
static unsigned long long s_pc_umd;   /* the adapter's user-mode driver version (a new driver makes the pipeline library invalid) */
extern int (*g_rsx_level_hook)(char* out, size_t n);
extern u8* vm_base;
static int eng_level_hook(char* out, size_t n);   /* hitch108: rsx_draw_engine.c asks for the PSO cache directory (per-level surface lists live there) */
static const char* eng_cache_dir_hook(void);
static u32 s_mp_over85[3], s_mp_made[3];   /* hitch108: [0] colour target [1] depth target [2] MSAA twin: created / created over 85% of the budget (RSX_VRAM108_LOG=1) */
static void eng_mp_count(int kind);
static DWORD s_walker_tid;
static void eng_lock(SRWLOCK* l, const char* name)
{
    if (TryAcquireSRWLockExclusive(l)) return;
    LARGE_INTEGER q0, q1, qf; QueryPerformanceCounter(&q0);
    BEACON_WAIT(name);
    AcquireSRWLockExclusive(l);
    BEACON_POP();
    if (GetCurrentThreadId() == s_walker_tid) {
        QueryPerformanceCounter(&q1); QueryPerformanceFrequency(&qf);
        const double ms = (double)(q1.QuadPart - q0.QuadPart) * 1000.0 / (double)qf.QuadPart;
        if (name[0] == 'p' && name[1] == 'c') { g_rsx_frame_pclock_ms += ms; g_rsx_frame_pclock_n++; }
        else { g_rsx_frame_lock_ms += ms; g_rsx_frame_lock_n++; }
    }
}

/* Bounded fence wait: a device removal leaves the fence unsignaled forever,
 * and an INFINITE wait here would freeze the whole emulation (see
 * rsx_d3d12_backend.c's wait_for_gpu). */
/* RSX_SYNC_STATS=1: every 5 s, the waits for the GPU that actually blocked,
 * by the source line that waited, with their count and time -- the walker's
 * stalls on the real GPU, which the render thread waits behind. */
static int s_sync_stats = -1;
static struct { int line; unsigned long long n; double ms; } s_sync_rec[32];
static void sync_stats_add(int line, double ms)
{
    int i;
    for (i = 0; i < 32 && s_sync_rec[i].line && s_sync_rec[i].line != line; i++) {}
    if (i == 32) return;
    s_sync_rec[i].line = line; s_sync_rec[i].n++; s_sync_rec[i].ms += ms;
    static ULONGLONG last;
    const ULONGLONG now = GetTickCount64();
    if (!last) last = now;
    if (now - last >= 5000) {
        const double secs = (now - last) / 1000.0;
        last = now;
        fprintf(stderr, "[sync-stats] GPU waits that blocked, per second:");
        for (int k = 0; k < 32 && s_sync_rec[k].line; k++) {
            if (s_sync_rec[k].n)
                fprintf(stderr, " line %d %.0f/s %.1f ms/s;", s_sync_rec[k].line,
                        s_sync_rec[k].n / secs, s_sync_rec[k].ms / secs);
            s_sync_rec[k].n = 0; s_sync_rec[k].ms = 0;
        }
        fputc('\n', stderr);
    }
}

extern double g_rsx_frame_gpu_wait_ms;   /* rsx_draw_engine.c: blocking GPU waits since the last present */
static void fence_wait_at2(u64 v, int line);
static void fence_wait_at(u64 v, int line)
{
    if (!v || CALL0(s_fence, GetCompletedValue) >= v) return;
    RSX_B2P_ENTER(B2P_WFENCE); RSX_B2P_ADD(B2C_FENCEBLK, 1);
    BEACON_WAIT("fence_wait");
    if (t_on_enc && ep_on()) { const double t0 = ep_now(); fence_wait_at2(v, line); s_ep_fence += ep_now() - t0; }
    else fence_wait_at2(v, line);
    BEACON_POP();
    RSX_B2P_LEAVE();
}
static void fence_wait_at2(u64 v, int line)
{
    if (s_sync_stats < 0) s_sync_stats = getenv("RSX_SYNC_STATS") ? 1 : 0;
    static int hl = -1;
    if (hl < 0) hl = getenv("RSX_HITCH_LOG") ? 1 : 0;
    LARGE_INTEGER qf, q0, q1;
    if (s_sync_stats || hl) { QueryPerformanceFrequency(&qf); QueryPerformanceCounter(&q0); }
    CALL(s_fence, SetEventOnCompletion, v, s_fence_event);
    for (int tries = 0; tries < 5; tries++) {
        if (WaitForSingleObject(s_fence_event, 2000) != WAIT_TIMEOUT) {
            if (s_sync_stats || hl) {
                QueryPerformanceCounter(&q1);
                const double ms = (double)(q1.QuadPart - q0.QuadPart) * 1000.0 / (double)qf.QuadPart;
                if (s_sync_stats) sync_stats_add(line, ms);
                g_rsx_frame_gpu_wait_ms += ms;
            }
            return;
        }
        HRESULT rr = CALL0(s_dev, GetDeviceRemovedReason);
        fprintf(stderr, "[rsx engine/d3d12] fence %llu stuck %ds (completed %llu, removed=0x%08lX)\n",
                (unsigned long long)v, 2 * (tries + 1),
                (unsigned long long)CALL0(s_fence, GetCompletedValue), (long)rr);
        if (rr != S_OK) { extern void eng_dred_dump(const char*); eng_dred_dump("fence"); s_ready = 0; return; }
    }
}

/* D3D12_DRED=1: Device Removed Extended Data. Enabled before the device is
 * created; on the first removal it prints, per command list, the GPU op that
 * was executing (auto-breadcrumbs, with neighbours) and the page-fault VA, so a
 * GPU hang names its command (draw / copy / clear / resolve). */
/* Last draws recorded on the GPU lists (D3D12_DRED): pipeline handle (bit 31 = copy
 * path), index and vertex counts, so a hung DrawIndexedInstanced can be named. */
typedef struct { u32 pipe, idx, vtx; } EngDrawRec;
#define ENG_DRAW_RING 8192u
static EngDrawRec s_draw_ring[ENG_DRAW_RING];
static u64 s_draw_ring_n;
static void eng_draw_ring_add(u32 pipe, u32 idx, u32 vtx)
{
    EngDrawRec* d = &s_draw_ring[s_draw_ring_n++ % ENG_DRAW_RING];
    d->pipe = pipe; d->idx = idx; d->vtx = vtx;
}

static void eng_dred_enable(void)
{
    if (!getenv("D3D12_DRED")) return;
    ID3D12DeviceRemovedExtendedDataSettings* st = NULL;
    if (SUCCEEDED(D3D12GetDebugInterface(&IID_ID3D12DeviceRemovedExtendedDataSettings, (void**)&st)) && st) {
        st->lpVtbl->SetAutoBreadcrumbsEnablement(st, D3D12_DRED_ENABLEMENT_FORCED_ON);
        st->lpVtbl->SetPageFaultEnablement(st, D3D12_DRED_ENABLEMENT_FORCED_ON);
        st->lpVtbl->Release(st);
        fprintf(stderr, "[dred] auto-breadcrumbs and page faults enabled\n");
    } else fprintf(stderr, "[dred] settings interface unavailable\n");
}
void eng_dred_dump(const char* where)
{
    if (s_dev) V2EV("device", "\"where\":\"%s\",\"removed\":\"0x%08lX\"", where, (long)CALL0(s_dev, GetDeviceRemovedReason));
    static int done;
    if (done || !s_dev || !getenv("D3D12_DRED")) return;
    done = 1;
    ID3D12DeviceRemovedExtendedData1* dred = NULL;
    if (FAILED(s_dev->lpVtbl->QueryInterface(s_dev, &IID_ID3D12DeviceRemovedExtendedData1, (void**)&dred)) || !dred) {
        fprintf(stderr, "[dred] (%s) query failed\n", where); return;
    }
    D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 bc = {0};
    if (SUCCEEDED(dred->lpVtbl->GetAutoBreadcrumbsOutput1(dred, &bc))) {
        /* Name the hung draw: draws recorded after it = the draw ops from the
         * stuck op to the end of its list plus every draw of the lists after it
         * (nodes are in submission order); the ring holds them in record order. */
        u64 after = 0; int stuck_seen = 0;
        for (const D3D12_AUTO_BREADCRUMB_NODE1* n = bc.pHeadAutoBreadcrumbNode; n; n = n->pNext) {
            const u32 last = n->pLastBreadcrumbValue ? *n->pLastBreadcrumbValue : 0;
            if (last >= n->BreadcrumbCount && !stuck_seen) continue;
            const u32 from = stuck_seen ? 0 : last;
            for (u32 i = from; i < n->BreadcrumbCount; i++)
                if (n->pCommandHistory[i] == D3D12_AUTO_BREADCRUMB_OP_DRAWINSTANCED ||
                    n->pCommandHistory[i] == D3D12_AUTO_BREADCRUMB_OP_DRAWINDEXEDINSTANCED) after++;
            stuck_seen = 1;
        }
        if (after && after <= s_draw_ring_n && after < ENG_DRAW_RING - 8) {
            const u64 hung = s_draw_ring_n - after;
            fprintf(stderr, "[dred] hung draw = record %llu of %llu (pipeline handle; bit31 = copy path):\n",
                    (unsigned long long)hung, (unsigned long long)s_draw_ring_n);
            for (u64 i = hung >= 4 ? hung - 4 : 0; i <= hung + 4 && i < s_draw_ring_n; i++) {
                const EngDrawRec* d = &s_draw_ring[i % ENG_DRAW_RING];
                fprintf(stderr, "[dred]   %s#%llu pipe=%u%s idx=%u vtx=%u\n", i == hung ? "*" : " ",
                        (unsigned long long)i, d->pipe & 0x7FFFFFFFu, (d->pipe >> 31) ? " (copy)" : "", d->idx, d->vtx);
            }
        }
        unsigned k = 0;
        for (const D3D12_AUTO_BREADCRUMB_NODE1* n = bc.pHeadAutoBreadcrumbNode; n && k < 32; n = n->pNext, k++) {
            const u32 last = n->pLastBreadcrumbValue ? *n->pLastBreadcrumbValue : 0;
            if (last >= n->BreadcrumbCount) continue;           /* this list finished */
            fprintf(stderr, "[dred] (%s) list %u: %u ops, stopped at op %u:", where, k, n->BreadcrumbCount, last);
            const u32 a = last > 6 ? last - 6 : 0;
            u32 b = last + 6; if (b >= n->BreadcrumbCount) b = n->BreadcrumbCount - 1;
            for (u32 i = a; i <= b; i++)
                fprintf(stderr, " %s%u", i == last ? "*" : "", (unsigned)n->pCommandHistory[i]);
            fprintf(stderr, "\n");
            /* ops of the whole list, as a histogram (D3D12_AUTO_BREADCRUMB_OP numbers) */
            unsigned hist[64] = {0};
            for (u32 i = 0; i < n->BreadcrumbCount; i++) if ((unsigned)n->pCommandHistory[i] < 64) hist[n->pCommandHistory[i]]++;
            fprintf(stderr, "[dred]   op histogram:");
            for (unsigned o = 0; o < 64; o++) if (hist[o]) fprintf(stderr, " %u:%u", o, hist[o]);
            fprintf(stderr, "\n");
        }
    }
    D3D12_DRED_PAGE_FAULT_OUTPUT1 pf = {0};
    if (SUCCEEDED(dred->lpVtbl->GetPageFaultAllocationOutput1(dred, &pf)))
        fprintf(stderr, "[dred] page fault VA=0x%016llX (existing alloc %p, freed %p)\n",
                (unsigned long long)pf.PageFaultVA, (void*)pf.pHeadExistingAllocationNode,
                (void*)pf.pHeadRecentFreedAllocationNode);
    dred->lpVtbl->Release(dred);
    fflush(stderr);
}
#define fence_wait(v) fence_wait_at((v), __LINE__)
static int fence_done(u64 v) { return !v || CALL0(s_fence, GetCompletedValue) >= v; }
static u64 fence_signal(void)
{
    const u64 v = ++s_fence_value;
    CALL(s_queue, Signal, s_fence, v);
    return v;
}

/* ---- object table ---------------------------------------------------------- */

static volatile u64 s_lane_swc;   /* lane109: counts object creations/releases: the coordinator compares it around a serial run to know if shared state moved */
/* lane111 RSX_JOIN_LITE=1: creating a depth/colour snapshot touches only a fresh object slot, the snapshot pool and the walker's own record list, none of which the encode thread of the previous
 * frame reads or writes, so those two entries do not join it (the join was the next wait after obj_release: log lane111/s1 d3d12 'joins forced: depth_snapshot x240 (1240 ms)'). The one thing the encode thread
 * shares with them is eng_obj_add itself (eng_aa_target, when the window size changes): a spin lock covers the slot allocation. */
static int s_join_lite = -1;
static int join_lite_on(void) { if (s_join_lite < 0) { const char* e = getenv("RSX_JOIN_LITE"); s_join_lite = (e && e[0] == '1') ? 1 : 0; } return s_join_lite; }
static volatile LONG s_obj_lock;
/* logv2: allocation sites. A site = the backend call (function:line of the eng_obj_add) + the draw-engine caller (V2CTX at the create call).
 * Per site: created / destroyed objects and MSAA twins in 8 one-second slots (the 5 s windows), and totals. Live bytes are summed at the scan. */
typedef struct { const char* fn; const char* cfn; int line, cline; u32 tag[8], cr[8], de[8], tcr[8], tde[8]; unsigned long long cr_tot, de_tot, tcr_tot, tde_tot; u32 gaddr0, ginfo0; } V2Site;
#define V2_MAXSITE 192
static V2Site s_v2s[V2_MAXSITE]; static u32 s_v2s_n = 1; static volatile LONG s_v2s_lock;   /* index 0 = "other" (table full) */
static u32 v2_site_get(const char* fn, int line, const v2_ctx_t* c)
{
    const char* cfn = c ? c->fn : NULL; const int cl = c ? c->line : 0;
    const u32 n0 = s_v2s_n;
    for (u32 i = 1; i < n0; i++) if (s_v2s[i].fn == fn && s_v2s[i].line == line && s_v2s[i].cfn == cfn && s_v2s[i].cline == cl) return i;
    while (InterlockedExchange(&s_v2s_lock, 1)) YieldProcessor();
    u32 r = 0;
    for (u32 i = 1; i < s_v2s_n; i++) if (s_v2s[i].fn == fn && s_v2s[i].line == line && s_v2s[i].cfn == cfn && s_v2s[i].cline == cl) { r = i; break; }
    if (!r && s_v2s_n < V2_MAXSITE) {
        V2Site* s = &s_v2s[s_v2s_n]; s->fn = fn; s->line = line; s->cfn = cfn; s->cline = cl; s->gaddr0 = c ? c->gaddr : 0; s->ginfo0 = c ? c->ginfo : 0;
        r = s_v2s_n; s_v2s_n++;
    }
    InterlockedExchange(&s_v2s_lock, 0);
    return r;
}
static void v2_site_bump(u32 si, int which)
{
    V2Site* s = &s_v2s[si]; const u32 sec = g_v2_up_s, sl = sec & 7u;
    if (s->tag[sl] != sec) { s->tag[sl] = sec; s->cr[sl] = s->de[sl] = s->tcr[sl] = s->tde[sl] = 0; }
    switch (which) { case 0: s->cr[sl]++; s->cr_tot++; break; case 1: s->de[sl]++; s->de_tot++; break; case 2: s->tcr[sl]++; s->tcr_tot++; break; default: s->tde[sl]++; s->tde_tot++; break; }
}
static u32 eng_obj_add_unlocked(const char* sfn, int sline, ID3D12Resource* res, EngObjKind kind, DXGI_FORMAT fmt, u32 w, u32 h, u32 mips, u32 faces, D3D12_RESOURCE_STATES st);
static u32 eng_obj_add_s(const char* sfn, int sline, ID3D12Resource* res, EngObjKind kind, DXGI_FORMAT fmt, u32 w, u32 h,
                       u32 mips, u32 faces, D3D12_RESOURCE_STATES st)
{
    if (!join_lite_on()) return eng_obj_add_unlocked(sfn, sline, res, kind, fmt, w, h, mips, faces, st);
    while (InterlockedExchange(&s_obj_lock, 1)) YieldProcessor();
    const u32 r = eng_obj_add_unlocked(sfn, sline, res, kind, fmt, w, h, mips, faces, st);
    InterlockedExchange(&s_obj_lock, 0);
    return r;
}
#define eng_obj_add(...) eng_obj_add_s(__func__, __LINE__, __VA_ARGS__)
static u32 eng_obj_add_unlocked(const char* sfn, int sline, ID3D12Resource* res, EngObjKind kind, DXGI_FORMAT fmt, u32 w, u32 h,
                       u32 mips, u32 faces, D3D12_RESOURCE_STATES st)
{
    if (!res && kind != OBJ_VIEW) return 0;
    u32 slot;
    if (s_obj_free_count) slot = s_obj_free[--s_obj_free_count];
    else {
        if (s_obj_count >= ENG_MAX_OBJECTS) {
            static unsigned long n = 0;
            if (n++ % 10000 == 0)
                fprintf(stderr, "[rsx engine/d3d12] object table full (%u): texture creation FAILS (#%lu)\n", s_obj_count, n);
            if (res) res->lpVtbl->Release(res);
            return 0;
        }
        slot = s_obj_count++;
    }
    EngObj* o = &s_obj[slot];
    memset(o, 0, sizeof *o);
    o->res = res; o->kind = kind; o->fmt = fmt; o->w = w; o->h = h;
    o->gw = w; o->gh = h;       /* a scaled object sets its guest size after */
    o->mips = mips ? mips : 1; o->faces = faces ? faces : 1; o->state = st;
    if (kind != OBJ_VIEW && v2_enabled()) {
        v2_ctx_t c; const int have = v2_ctx_take(&c);
        o->v2_site = v2_site_get(sfn, sline, have ? &c : NULL); o->v2_born = g_v2_frame;
        if (have) { o->v2_gaddr = c.gaddr; o->v2_ginfo = c.ginfo; }
        v2_site_bump(o->v2_site, 0);
    }
    return slot + 1;
}

/* Retired objects are released only once no submitted list can still name
 * them: every submit the GPU may be running must have completed. Records
 * resolve handles at encode time, so nothing is recycled while records wait. */
static u32 eng_snap_pool_cap(void)
{
    static u32 cap;
    if (!cap) { const char* e = getenv("RSX_SNAP_POOL"); int v = e ? atoi(e) : 256; if (v < 1) v = 1; if (v > ENG_SNAP_POOL) v = ENG_SNAP_POOL; cap = (u32)v; }
    return cap;
}
/* The pool also has a byte budget (upper bound, 8 B/px; RSX_SNAP_POOL_MB, default 384 since 31/10 walk36: 4K MSAA8 on 8 GB reached 100% of the VRAM budget with 1024): a pool of 256 mixed 4K targets must not eat VRAM. */
static u64 eng_px_bytes(DXGI_FORMAT f);
static int eng_snap_pool_fits(u64 add)
{
    static u64 budget;
    if (!budget) { const char* e = getenv("RSX_SNAP_POOL_MB"); long v = e ? atol(e) : 384; if (v < 1) v = 1; budget = (u64)v << 20; }
    if (add < (8u << 20)) return 1;   /* small ones (shadow maps, 64x64..): only the entry cap limits them */
    u64 sum = add;
    /* rissa128: real bytes (R32F and RGBA8 are 4 B/px; counted at 8 the 384 MB budget held ~192 MB and the pool sat full at 216 MB in a brawl, with the misses
     * = CreateCommittedResource of 24-47 MB copies, 1-10 ms each on the walker). RSX_SNAP_POOL_8BPP=1 restores the old accounting. */
    static int old = -1; if (old < 0) { const char* e = getenv("RSX_SNAP_POOL_8BPP"); old = e && e[0] == '1'; }
    for (u32 i = 0; i < s_snap_pool_n; i++) { const u64 b = (u64)s_snap_pool[i].w * s_snap_pool[i].h * (old ? 8u : (u32)eng_px_bytes(s_snap_pool[i].fmt)); if (b >= (8u << 20)) sum += b; }
    return sum <= budget;
}
/* walk36: a snapshot that comes back while the pool is full (entry cap or byte budget) used to be destroyed, whatever else the
 * pool held. ToD's scene depth snapshots (2880x2112 packed, 1152x1344 ...) are made EVERY frame; with the pool already filled by
 * other sizes they were created (1,3 ms) and destroyed each frame, 2/3 of the draw thread's busy time in combat. Now the idle entry of ANOTHER
 * size that has waited longest (smallest fence) is destroyed to make room. RSX_SNAP_EVICT=0: old behaviour. Pool entries are
 * past their fence (fence_done) by construction, so releasing one is safe. */
/* A pool entry's resource is released, and a placed one's range goes back (pool entries are past their fence, or the GPU is idle: rescale, shutdown). The caller removes the entry. */
static void eng_snap_pool_free_entry(u32 i)
{
    RELEASE(s_snap_pool[i].res);
    if (s_snap_pool[i].heap) { theap_done(s_snap_pool[i].heap, s_snap_pool[i].heap_off, s_snap_pool[i].heap_size); s_dss.pool_ret++; }
}
static int eng_snap_pool_make_room(DXGI_FORMAT fmt, u32 w, u32 h)
{
    static int on = -1; if (on < 0) { const char* e = getenv("RSX_SNAP_EVICT"); on = !(e && e[0] == '0'); }
    for (;;) {
        if (s_snap_pool_n < eng_snap_pool_cap() && eng_snap_pool_fits((u64)w * h * eng_px_bytes(fmt))) return 1;
        if (!on) return 0;
        int v = -1;
        for (u32 i = 0; i < s_snap_pool_n; i++) {
            if (s_snap_pool[i].fmt == fmt && s_snap_pool[i].w == w && s_snap_pool[i].h == h) continue;
            if (v < 0 || s_snap_pool[i].fence < s_snap_pool[(u32)v].fence) v = (int)i;
        }
        if (v < 0) return 0;
        eng_snap_pool_free_entry((u32)v);
        s_snap_pool[(u32)v] = s_snap_pool[--s_snap_pool_n];
        { static unsigned ne; ne++; (void)ne; }
    }
}
static u32 s_retired[ENG_MAX_OBJECTS]; static u32 s_retired_count;
static u64 s_retired_fence[ENG_MAX_OBJECTS];

/* rissa128 RSX_REAP=1 (default off): the real Release of retired committed resources (colour/depth targets, MSAA twins) runs OFF the walker.
 * Why: the governor's frees only queue handles (eng_obj_release); the Release itself happens in eng_collect_retired at the NEXT submit, on the walker,
 * in no measured phase. On 8569 governor frames with a [hitch] line, a frame that queued >=200 objects was followed by a 100-118 ms frame with an
 * empty fifo 7% of the time (39 of 555), under 50 objects never (0 of 2573). Here the walker only moves the pointer to a list (the fence is done, the
 * GPU is past it) and the encode thread releases them after SetEvent(s_enc_done), RSX_REAP_MS (1.5) ms per frame, widened with the backlog (6 ms max).
 * Placed resources (theap) stay inline: theap_done must not overlap a live placed resource. Always on: the collect's own time and count, for log v2. */
#define ENG_DOOM_MAX 8192
static ID3D12Resource* s_doom[ENG_DOOM_MAX]; static unsigned long long s_doom_bytes_arr[ENG_DOOM_MAX]; static u32 s_doom_n;
static unsigned long long s_doom_bytes; static SRWLOCK s_doom_lk = SRWLOCK_INIT;
static unsigned long long s_reap_calls, s_reap_inline; static double s_reap_ms_tot; static u32 s_reap_frame_n; static double s_reap_frame_ms;
static u32 s_rel_frame_n; static double s_rel_frame_ms;   /* eng_collect_retired this frame: objects released (or queued) and its wall time */
static int reap_on(void) { static int v = -1; if (v < 0) { const char* e = getenv("RSX_REAP"); v = e && e[0] == '1'; } return v; }
static double reap_budget_ms(void) { static double v = 0; if (v <= 0) { const char* e = getenv("RSX_REAP_MS"); v = e && atof(e) > 0 ? atof(e) : 1.5; } return v; }
static double reap_now(void) { static double k = 0; LARGE_INTEGER q; if (k <= 0) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); k = 1000.0 / (double)f.QuadPart; } QueryPerformanceCounter(&q); return (double)q.QuadPart * k; }
static u64 eng_mt_bytes(const EngObj* o);
static u64 eng_px_bytes(DXGI_FORMAT f);
/* the real Release of *pp is queued, or done at once when off or the list is full */
static void eng_res_free(ID3D12Resource** pp, unsigned long long bytes)
{
    ID3D12Resource* r = *pp; if (!r) return; *pp = NULL;
    if (reap_on()) {
        AcquireSRWLockExclusive(&s_doom_lk);
        if (s_doom_n < ENG_DOOM_MAX) { s_doom[s_doom_n] = r; s_doom_bytes_arr[s_doom_n] = bytes; s_doom_n++; s_doom_bytes += bytes; r = NULL; }
        ReleaseSRWLockExclusive(&s_doom_lk);
    }
    if (r) { r->lpVtbl->Release(r); s_reap_inline++; }
}
static void eng_reap_drain(double budget_ms)   /* encode thread after SetEvent(s_enc_done); the walker after the collect in the sync path; teardown */
{
    if (!s_doom_n) return;
    const double t0 = reap_now(); u32 n = 0;
    const double b = budget_ms * (1.0 + (double)(s_doom_n / 128u)); const double cap = b > 6.0 && budget_ms < 1e6 ? 6.0 : b;   /* a backlog widens the budget, 6 ms at most */
    for (;;) {
        ID3D12Resource* r = NULL;
        AcquireSRWLockExclusive(&s_doom_lk);
        if (s_doom_n) { s_doom_n--; r = s_doom[s_doom_n]; s_doom_bytes -= s_doom_bytes_arr[s_doom_n]; }
        ReleaseSRWLockExclusive(&s_doom_lk);
        if (!r) break;
        r->lpVtbl->Release(r); s_reap_calls++; n++;
        if (reap_now() - t0 >= cap) break;
    }
    const double d = reap_now() - t0; s_reap_ms_tot += d; s_reap_frame_n += n; s_reap_frame_ms += d;
}
unsigned long long eng_doomed_bytes(void) { return s_doom_bytes; }
/* log v2: zeta copies this frame: resources created (with their ms; big = >= 1 Mpx host) and taken from the snapshot pool; zeroed after the read */
static u32 s_v2_snap_new, s_v2_snap_hit, s_v2_snap_big; static double s_v2_snap_ms;
void rsx_d3d12_snap_frame(unsigned* nw, double* ms, unsigned* big, unsigned* hit, unsigned* pool_n)
{
    *nw = s_v2_snap_new; *ms = s_v2_snap_ms; *big = s_v2_snap_big; *hit = s_v2_snap_hit; *pool_n = s_snap_pool_n;
    s_v2_snap_new = s_v2_snap_hit = s_v2_snap_big = 0; s_v2_snap_ms = 0.0;
}
/* log v2 "zmix": copies run on the ENCODE thread (DEPTH_RESOLVE records) and the GPU ms of those passes (RSX_GPU_TIME=2); read and zeroed by the walker.
 * s_zm: 0 small 1 medium 2 big (guest area of the request), 3 exact (host size == source), 4 source has a twin, 5 r32f, 6 rgba8, 7 r32f+exact+no twin, 8 r32f+exact+no twin possible */
static volatile LONG s_zm[9]; static double s_zm_gpu_ms[2];
void rsx_d3d12_zmix_frame(unsigned out[9], double gpu[2])
{
    for (int i = 0; i < 9; i++) out[i] = (unsigned)InterlockedExchange(&s_zm[i], 0);
    gpu[0] = s_zm_gpu_ms[0]; gpu[1] = s_zm_gpu_ms[1]; s_zm_gpu_ms[0] = s_zm_gpu_ms[1] = 0.0;
}
/* RSX_ZETA_SRV (draw engine): may a texture unit read this zeta directly through its depth SRV instead of a copy (eng_depth_snapshot_make)? Yes when the copy would be
 * exactly the zeta's host size (so 0..1 spans the same texels) and no MSAA twin can ever exist (eng_ms_ok needs guest >= 960x540). Called on the walker AND in lanes: it only
 * reads fields the walker/lanes write (kind, res, gw/gh, w/h, pf, xx), never joins the encode thread and creates nothing. gw, gh = the guest size of the request. */
int rsx_d3d12_zeta_direct_ok(u32 depth, u32 gw, u32 gh)
{
    const EngObj* z = eng_obj(depth);
    if (!z || z->kind != OBJ_DEPTH || !z->res || z->retired) return 0;
    if (z->gw >= 960 && z->gh >= 540) return 0;   /* a twin can be made for it (eng_ms_ok) */
    u32 w = gw, h = gh;                           /* the sizing of eng_depth_snapshot_make */
    if (z->pf > 0.0f) { w = scf_dim(w, z->pf); h = scf_dim(h, z->pf); }
    else if (z->xx > 0.0f && gw == 960 && gh == 704) return 0;
    else if (eng_obj_scaled(depth)) { w = sc_dim(w); h = sc_dim(h); }
    return w == z->w && h == z->h;
}
/* log v2 (draw engine, once per present): this frame's collect [n, ms], the reaper's [n, ms], the backlog and the bytes waiting; zeroed after the read */
void rsx_d3d12_rel_frame(unsigned* rel_n, double* rel_ms, unsigned* reap_n, double* reap_ms, unsigned* backlog, unsigned long long* doomed)
{
    *rel_n = s_rel_frame_n; *rel_ms = s_rel_frame_ms; *reap_n = s_reap_frame_n; *reap_ms = s_reap_frame_ms; *backlog = s_doom_n; *doomed = s_doom_bytes;
    s_rel_frame_n = 0; s_rel_frame_ms = 0.0; s_reap_frame_n = 0; s_reap_frame_ms = 0.0;
}

/* The submit an object or buffer released now may still be drawn by: the
 * last one made, or -- with records waiting that may name it -- the next. */
static u64 eng_retire_fence(void) { return s_fence_value + (s_rec_count ? 1 : 0); }

/* depthpool RSX_DEPTH_POOL=1 (default off): a plain depth target that is retired is PARKED instead of released, and the next eng_depth_target_create_host of the same
 * host size takes it back (a new handle: object, DSV and SRV are rebuilt, the resource, its DS-heap range and its state are kept). Log v2 up=166: 43 depth creations in
 * one burst cost 676 ms (zeta_create 43 ms in one frame, VidMm paging), the live depth count swinging 26 -> 84 -> 26 in seconds.
 * Parking happens in eng_collect_retired, i.e. only once the retire fence is done (the GPU has finished with it) and on the walker (the only caller). The take runs
 * on the walker (eng_depth_target_create after ENG_JOIN) or on B2 (also serialised by that join); never on the encode thread (no record can be added there, and it
 * would race the walker's collect). s_dp_lk makes the pool itself safe anyway. A reused depth holds old data: a plain depth clear (0 / 0, no aliasing barrier: the
 * resource stayed alive) is recorded ahead of every use, like the zero a placed one is born with.
 * RSX_DEPTH_POOL_MB (512): byte cap (oldest evicted). Trimmed to nothing but the newest while VRAM use is over 85% of the budget (eng_vram_query: the governor's numbers),
 * and entries idle for more than 600 submits are released. At most 4 releases per submit. Counted in [vram] / log v2 "cat" as depth_pool; [depth-pool] line every ~10 s. */
#define ENG_DP_MAX 64
#define ENG_DP_AGE 10000u   /* ms: idle entries leave after 10 s (b56 counted submits: several per frame, 1960 trims of 2147 parks at Rykan V) */
typedef struct { ID3D12Resource* res; u32 w, h; D3D12_RESOURCE_STATES state; u32 heap; u64 heap_off, heap_size; u32 last; u64 bytes; } EngDPEnt;
static EngDPEnt s_dp[ENG_DP_MAX]; static u32 s_dp_n; static u64 s_dp_bytes; static u32 s_dp_tick;
static SRWLOCK s_dp_lk = SRWLOCK_INIT;
static unsigned long long s_dp_hit, s_dp_miss, s_dp_trim, s_dp_park;
static int dpool_on(void) { static int v = -1; if (v < 0) { const char* e = getenv("RSX_DEPTH_POOL"); v = e && e[0] == '1'; } return v; }
static u64 dpool_cap(void) { static u64 v; if (!v) { const char* e = getenv("RSX_DEPTH_POOL_MB"); long m = e ? atol(e) : 512; if (m < 1) m = 1; v = (u64)m << 20; } return v; }
/* the oldest entry leaves the pool when it is idle too long, the pool (plus `add` bytes) is over its cap, or `over` (VRAM over 85%) */
static int eng_dpool_pop_old(u64 add, int over, EngDPEnt* out)
{
    int got = 0;
    AcquireSRWLockExclusive(&s_dp_lk);
    if (s_dp_n) {
        u32 v = 0;
        for (u32 i = 1; i < s_dp_n; i++) if ((s32)(s_dp[i].last - s_dp[v].last) < 0) v = i;
        if ((u32)(s_dp_tick - s_dp[v].last) > ENG_DP_AGE || s_dp_bytes + add > dpool_cap() || over) {
            *out = s_dp[v]; s_dp[v] = s_dp[--s_dp_n]; s_dp_bytes -= out->bytes; got = 1;
        }
    }
    ReleaseSRWLockExclusive(&s_dp_lk);
    return got;
}
static void eng_dpool_drop(EngDPEnt* e)   /* the real release, outside the lock */
{
    if (reap_on() && !e->heap_size) eng_res_free(&e->res, e->bytes); else RELEASE(e->res);
    theap_done(e->heap, e->heap_off, e->heap_size);
    s_dp_trim++;
}
/* the retired depth `o` (fence done) goes to the pool: its resource and range move there. 0 = not parked (the caller releases it as usual) */
static int eng_dpool_park(EngObj* o)
{
    const u64 bytes = (u64)o->w * o->h * eng_px_bytes(ENG_DEPTH_FMT);
    if (!o->res || bytes > dpool_cap()) return 0;
    EngDPEnt e;
    for (int k = 0; k < 4 && eng_dpool_pop_old(bytes, 0, &e); k++) eng_dpool_drop(&e);
    int ok = 0;
    AcquireSRWLockExclusive(&s_dp_lk);
    if (s_dp_n < ENG_DP_MAX && s_dp_bytes + bytes <= dpool_cap()) {
        EngDPEnt* p = &s_dp[s_dp_n++];
        p->res = o->res; p->w = o->w; p->h = o->h; p->state = o->state;
        p->heap = o->heap; p->heap_off = o->heap_off; p->heap_size = o->heap_size; p->last = s_dp_tick; p->bytes = bytes;
        s_dp_bytes += bytes; s_dp_park++; ok = 1;
    }
    ReleaseSRWLockExclusive(&s_dp_lk);
    if (ok) { o->res = NULL; o->heap = 0; o->heap_off = o->heap_size = 0; }
    return ok;
}
/* the newest parked depth of exactly w x h; 0 = none */
static int eng_dpool_take(u32 w, u32 h, ID3D12Resource** res, D3D12_RESOURCE_STATES* st, u32* heap, u64* off, u64* size)
{
    int got = 0;
    AcquireSRWLockExclusive(&s_dp_lk);
    int v = -1;
    for (u32 i = 0; i < s_dp_n; i++)
        if (s_dp[i].w == w && s_dp[i].h == h && (v < 0 || (s32)(s_dp[i].last - s_dp[(u32)v].last) > 0)) v = (int)i;
    if (v >= 0) {
        *res = s_dp[v].res; *st = s_dp[v].state; *heap = s_dp[v].heap; *off = s_dp[v].heap_off; *size = s_dp[v].heap_size;
        s_dp_bytes -= s_dp[v].bytes; s_dp[v] = s_dp[--s_dp_n]; s_dp_hit++; got = 1;
    } else s_dp_miss++;
    ReleaseSRWLockExclusive(&s_dp_lk);
    return got;
}
/* every parked depth released: rescale, shutdown (GPU idle) */
static void eng_dpool_flush_all(void)
{
    EngDPEnt e;
    AcquireSRWLockExclusive(&s_dp_lk);
    const u32 n = s_dp_n;
    EngDPEnt all[ENG_DP_MAX]; memcpy(all, s_dp, n * sizeof(EngDPEnt)); s_dp_n = 0; s_dp_bytes = 0;
    ReleaseSRWLockExclusive(&s_dp_lk);
    for (u32 i = 0; i < n; i++) { e = all[i]; RELEASE(e.res); theap_done(e.heap, e.heap_off, e.heap_size); }
}
/* once per submit on the walker (end of eng_collect_retired) */
static void eng_dpool_trim(void)
{
    s_dp_tick = (u32)GetTickCount64();   /* ms clock for ENG_DP_AGE */
    if (s_dp_n) {
        u64 used = 0, budget = 0; const int have = eng_vram_query(&used, &budget) && budget;
        EngDPEnt e;
        for (int k = 0; k < 4; k++) {
            const int over = have && used * 100ull > budget * 85ull;
            if (!eng_dpool_pop_old(0, over, &e)) break;
            if (used >= e.bytes) used -= e.bytes;
            eng_dpool_drop(&e);
        }
    }
    static ULONGLONG last;
    const ULONGLONG now = GetTickCount64();
    if (!last) last = now;
    if (now - last >= 10000) {
        last = now;
        fprintf(stderr, "[depth-pool] parked %u = %llu MB, hits %llu, misses %llu, parks %llu, trims %llu\n", s_dp_n, (unsigned long long)(s_dp_bytes >> 20),
                s_dp_hit, s_dp_miss, s_dp_park, s_dp_trim);
    }
}

static u32 s_rel_pn; static void eng_rel_flush(void); static __declspec(thread) int t_rel_hold;
static void eng_fb_victims_release(void);   /* vram126 */
static void eng_collect_retired(void)
{
    if (s_rel_pn && !t_on_enc && !s_enc_pending) eng_rel_flush();
    eng_fb_victims_release();
    const double rt0 = s_retired_count ? reap_now() : 0.0; u32 rn = 0;
    u32 k = 0;
    for (u32 i = 0; i < s_retired_count; i++) {
        const u32 h = s_retired[i];
        if (s_rec_count || !fence_done(s_retired_fence[i])) { s_retired[k] = h; s_retired_fence[k] = s_retired_fence[i]; k++; continue; }
        rn++;
        EngObj* o = &s_obj[h - 1];
        if (o->kind == OBJ_SNAPSHOT && o->has_rtv && !o->dsinit && eng_snap_pool_make_room(o->fmt, o->w, o->h)) {   /* dsinit: never initialised (its record was dropped): not pooled */
            s_snap_pool[s_snap_pool_n].res = o->res; s_snap_pool[s_snap_pool_n].fmt = o->fmt;
            s_snap_pool[s_snap_pool_n].w = o->w; s_snap_pool[s_snap_pool_n].h = o->h;
            s_snap_pool[s_snap_pool_n].fence = s_retired_fence[i];
            s_snap_pool[s_snap_pool_n].state = o->state;
            s_snap_pool[s_snap_pool_n].heap = o->heap; s_snap_pool[s_snap_pool_n].heap_off = o->heap_off; s_snap_pool[s_snap_pool_n].heap_size = o->heap_size;
            o->heap = 0; o->heap_off = o->heap_size = 0;   /* RSX_DS_HEAP: a placed snapshot's range stays with the pooled resource (theap_done below is then a no-op) */
            s_snap_pool_n++;
            o->res = NULL;
        }
        if (o->kind == OBJ_DEPTH && o->res && o->fmt == ENG_DEPTH_FMT && !o->dsinit && dpool_on() && !t_on_enc && eng_dpool_park(o)) { /* parked: o->res / heap now belong to the pool */ }
        if (o->v2_site) v2_site_bump(o->v2_site, 1);
        if (reap_on() && !o->heap_size && o->res) eng_res_free(&o->res, (unsigned long long)o->w * o->h * eng_px_bytes(o->fmt));   /* committed: off the walker */
        else RELEASE(o->res);
        eng_ms_drop(o);
        theap_done(o->heap, o->heap_off, o->heap_size);
        memset(o, 0, sizeof *o);
        s_obj_free[s_obj_free_count++] = h - 1;
    }
    s_retired_count = k;
    if (rn) { s_rel_frame_n += rn; s_rel_frame_ms += reap_now() - rt0; }
    if (dpool_on() && !t_on_enc) eng_dpool_trim();
}

/* lane111 RSX_REL_DEFER=1: while the encode thread owns the object table, an eng_obj_release from the walker/B2 does NOT join (the join made the first
 * release of every frame wait ~5.5 ms for the encode of the previous frame, log lane110/p2 "joins forced: obj_release x238 (1321 ms)"): the handle is queued
 * and the real release (retired flag, view removal, retire list) runs at the next join/hand-over/collect, when the encode is done. The retire fence is
 * s_fence_value + (records waiting at the CALL ? 1 : 0), evaluated at processing time = what the immediate path computes right after its join. */
typedef struct { u32 handle; u32 rec; } EngRelPend;
#define ENG_REL_PEND_MAX 2048
static EngRelPend s_rel_pend[ENG_REL_PEND_MAX]; static unsigned long long s_rel_n_defer, s_rel_n_flush;   /* t_rel_hold: a lane coordinator join with lanes possibly in flight: no flush there */
static volatile u32 s_zown_gen;   /* lane111: moves when every object may have been replaced (shutdown, rescale): the coordinator's ahead-made snapshot objects are dropped */
static int s_rel_defer = -1;
static int rel_defer_on(void) { if (s_rel_defer < 0) { const char* e = getenv("RSX_REL_DEFER"); s_rel_defer = (e && e[0] == '1') ? 1 : 0; } return s_rel_defer; }
static void eng_obj_release_do(u32 handle, u32 rec)
{
    EngObj* o = eng_obj(handle);
    if (!o || o->retired) return;
    o->retired = 1;
    /* Any view cut from this object stops being valid with it. */
    for (u32 i = 0; i < s_view_count; i++)
        if (s_view[i].surface == handle) {
            eng_obj_release_do(s_view[i].view, rec);
            s_view[i] = s_view[--s_view_count];
            i--;
        }
    if (s_retired_count < ENG_MAX_OBJECTS) {
        s_retired[s_retired_count] = handle;
        s_retired_fence[s_retired_count] = s_fence_value + (rec ? 1 : 0);
        s_retired_count++;
    }
}
static void eng_rel_flush(void)
{
    if (!s_rel_pn) return;
    const u32 n = s_rel_pn; s_rel_pn = 0;
    for (u32 i = 0; i < n; i++) eng_obj_release_do(s_rel_pend[i].handle, s_rel_pend[i].rec);
    s_rel_n_flush += n;
}
static void eng_obj_release(void* user, u32 handle)
{
    s_lane_swc++;
    (void)user;
    if (s_enc_pending && !t_on_enc && !LB->spec && rel_defer_on() && s_rel_pn < ENG_REL_PEND_MAX) {
        s_rel_pend[s_rel_pn].handle = handle; s_rel_pend[s_rel_pn].rec = s_rec_count != 0;
        s_rel_pn++; s_rel_n_defer++;
        return;
    }
    ENG_JOIN("obj_release");
    if (s_rel_pn && !t_on_enc && !t_rel_hold && !LB->spec) eng_rel_flush();
    eng_obj_release_do(handle, s_rec_count != 0);
}

/* Write the object's SRV at its CPU slot. */
static void eng_write_srv(u32 handle, ID3D12Resource* res, DXGI_FORMAT fmt, u32 mips, u32 faces, UINT mapping)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC sd = {0};
    sd.Format = fmt;
    sd.Shader4ComponentMapping = mapping;
    if (faces == 6) {
        sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
        sd.TextureCube.MipLevels = mips;
    } else {
        sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        sd.Texture2D.MipLevels = mips;
    }
    CALL(s_dev, CreateShaderResourceView, res, &sd, obj_srv(handle));
}

/* ---- staging ---------------------------------------------------------------
 * What a submit's records read -- constants, vertices without a buffer of
 * their own, texture rows -- is staged in UPLOAD-heap chunks, chained as they
 * fill; a record holds offsets into the chain. Nothing staged is ever read
 * back: the memory is write-combined, and one arena grown by copying it into
 * a larger one took 0.4 to 0.9 s over the 110 MB a level's first frame
 * stages (and left every arena of the pool that size afterwards). */

static void eng_submit(u32 present_surface, int wait);

static void eng_stage_free(EngStage* s)
{
    if (s->res) { CALL(s->res, Unmap, 0, NULL); RELEASE(s->res); }
    memset(s, 0, sizeof *s);
}

/* A free chunk of `need` bytes or more: the smallest there is, else a new
 * one, in an empty slot or in place of the smallest free one. -1 when every
 * chunk is in use. */
static int eng_stage_pick(u32 need)
{
    int best = -1, empty = -1, least = -1;
    for (int i = 0; i < ENG_STAGE_POOL; i++) {
        EngStage* s = &s_stage[i];
        if (s->in_use == 1 && fence_done((u64)InterlockedCompareExchange64((volatile LONG64*)&s->fence, 0, 0))) s->in_use = 0;
        if (s->in_use) continue;
        if (!s->res) { if (empty < 0) empty = i; }
        else if (s->cap >= need) { if (best < 0 || s->cap < s_stage[best].cap) best = i; }
        else if (least < 0 || s->cap < s_stage[least].cap) least = i;
    }
    if (best >= 0) return best;
    if (empty < 0) {
        if (least < 0) return -1;
        eng_stage_free(&s_stage[least]);
        empty = least;
    }
    u32 cap = ENG_STAGE_CHUNK;
    while (cap < need) cap *= 2u;
    EngStage* s = &s_stage[empty];
    s->res = make_buffer(D3D12_HEAP_TYPE_UPLOAD, cap, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (!s->res) return -1;
    D3D12_RANGE nr = {0, 0};
    if (FAILED(CALL(s->res, Map, 0, &nr, (void**)&s->mapped))) { RELEASE(s->res); return -1; }
    s->va = CALL0(s->res, GetGPUVirtualAddress);
    s->cap = cap;
    return empty;
}

/* Buffer <-> texture copies (CopyTextureRegion from a placed footprint)
 * want the footprint's offset on D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT
 * (512) and its row pitch on D3D12_TEXTURE_DATA_PITCH_ALIGNMENT (256),
 * unless the device reports UnrestrictedBufferTextureCopyPitchSupported
 * (D3D12_OPTIONS13: the runtimes of Agility SDK 1.602 on, which Windows
 * 11's own is and Windows 10's is not). On a device without it a copy that
 * breaks the rule makes its whole command list fail to close.
 *
 * RSX_D3D12_STRICT_COPY=1 holds a device that relaxes the rule to it all the
 * same: a list with such a copy is failed as that runtime fails it, so that
 * a machine with the relaxation can show what one without it does.
 * RSX_D3D12_STAGE_ALIGN=<n> sets the stage's alignment (512; 256 was the
 * old one, which broke the rule). */
static u32 s_stage_align = ENG_ALIGN;
static int s_copy_relaxed = 1, s_copy_strict_test, s_list_bad_copy;
static u32 s_bad_copies;

static void eng_check_copy(u64 offset, u32 row_pitch, const char* what)
{
    if (s_copy_relaxed && !s_copy_strict_test) return;
    if (!(offset & 511u) && !(row_pitch & 255u)) return;
    s_list_bad_copy = 1;
    if (s_bad_copies++ < 8)
        fprintf(stderr, "[rsx engine/d3d12] %s copy at offset %llu, row pitch %u: off the alignment this device requires\n",
                what, (unsigned long long)offset, row_pitch);
}

static int eng_stage_reserve(u32 bytes, u32* out_off)
{
    if (bytes > ENG_STAGE_MAX) return 0;
    for (int tries = 0; tries < 2; tries++) {
        EngStage* s = s_chain_n ? &s_stage[s_chain[s_chain_n - 1]] : NULL;
        const u32 start = s ? (s->used + s_stage_align - 1u) & ~(s_stage_align - 1u) : 0;
        if (s && (u64)start + bytes <= s->cap) {
            *out_off = s->base + start;
            s->used = start + bytes;
            return 1;
        }
        /* The chain's next chunk. */
        const u32 base = s ? s->base + s->cap : 0;
        int i = -1;
        if ((u64)base + bytes <= ENG_STAGE_MAX && s_chain_n < ENG_STAGE_POOL) i = eng_stage_pick(bytes);
        if (i < 0) {
            /* More than a submit stages, or every chunk in flight: submit
             * and wait rather than drop the rest of the frame. */
            if (s_enc_pending && !t_on_enc) { eng_enc_join_why("stage_full", 0); continue; }   /* a held chunk may be free by now */
            if (!s_rec_count) return 0;
            RSX_B2P_ENTER(B2P_WSUBMIT); RSX_B2P_ADD(B2C_SUBMIT, 1);
            eng_submit(0, 1);
            RSX_B2P_LEAVE();
            continue;
        }
        s = &s_stage[i];
        s->base = base; s->used = bytes; s->in_use = 2;
        s_chain[s_chain_n++] = i;
        *out_off = base;
        return 1;
    }
    return 0;
}

/* Where to write a reservation just made: it is in the chain's last chunk. */
static u8* eng_stage_ptr(u32 off)
{
    const EngStage* s = &s_stage[s_chain[s_chain_n - 1]];
    return s->mapped + (off - s->base);
}

/* lane109: a lane's own region of the arena. Regions are cut from the tail chunk of the chain under s_lane_cs (the coordinator's serial path never
 * runs at the same time as a lane: it only runs with every lane idle); inside a region the lane bumps alone. A region that does not fit the tail
 * chunk is NOT replaced here (that may submit, wait or create a buffer): the chunk is given back (abort) and the coordinator runs it serially. */
#define ENG_LANE_REGION (256u << 10)
static CRITICAL_SECTION s_lane_cs; static int s_lane_cs_init;
static int eng_lane_region(EngLaneBe* b, u32 bytes)
{
    int ok = 0;
    EnterCriticalSection(&s_lane_cs);
    EngStage* s = s_chain_n ? &s_stage[s_chain[s_chain_n - 1]] : NULL;
    if (s && s->mapped) {
        const u32 start = (s->used + s_stage_align - 1u) & ~(s_stage_align - 1u);
        u32 take = bytes > ENG_LANE_REGION ? bytes : ENG_LANE_REGION;
        if ((u64)start + take > s->cap) take = bytes;
        if ((u64)start + take <= s->cap) {
            b->ar_base = s->base + start; b->ar_mapped = s->mapped + start; b->ar_cap = take; b->ar_used = 0; b->ar_used0 = 0; b->ar_seq = s_submit_seq;
            s->used = start + take; ok = 1;
        }
    }
    LeaveCriticalSection(&s_lane_cs);
    return ok;
}
static int eng_lane_stage_copy(const void* src, u32 bytes, u32* out_off)
{
    EngLaneBe* b = LB;
    if (bytes > ENG_STAGE_MAX) return 0;
    u32 a = (b->ar_used + s_stage_align - 1u) & ~(s_stage_align - 1u);
    if (!b->ar_cap || b->ar_seq != s_submit_seq || (u64)a + bytes > b->ar_cap) {
        if (!eng_lane_region(b, bytes)) return 0;
        a = 0;
    }
    *out_off = b->ar_base + a;
    memcpy(b->ar_mapped + a, src, bytes);
    b->ar_used = a + bytes;
    return 1;
}

static int eng_stage_copy(const void* src, u32 bytes, u32* out_off)
{
    if (!bytes) { *out_off = 0; return 1; }
    if (LB->spec) { if (!eng_lane_stage_copy(src, bytes, out_off)) rsx_lane_abort(RSX_LAB_STAGE); return 1; }
    RSX_B2P_ENTER(B2P_STAGE); RSX_B2P_ADD(B2C_STAGE, 1); RSX_B2P_ADD(B2C_STAGEB, bytes);
    if (!eng_stage_reserve(bytes, out_off)) { RSX_B2P_LEAVE(); return 0; }
    memcpy(eng_stage_ptr(*out_off), src, bytes);
    RSX_B2P_LEAVE();
    return 1;
}


static u32 s_vis_job_lo;   /* tentative: defined with its neighbours below */
/* ---- lane109: what the lane coordinator (rsx_draw_engine.c) asks of the backend ---------------------------------------------------------- */
void rsx_d3d12_lane_init(void)
{
    if (!s_lane_cs_init) { InitializeCriticalSection(&s_lane_cs); s_lane_cs_init = 1; }
    if (s_lanes < 0) eng_lanes_init();
}
int rsx_d3d12_lane_ok(void) { return s_lanes > 0 && s_lanes_mt && s_ll; }
u64 rsx_d3d12_lane_sig(void) { return s_lane_swc * 1000003ull + s_obj_count * 31ull + s_view_count * 17ull + s_samp_count * 13ull + s_buf_count * 7ull + s_submit_seq * 100003ull + s_pipe_count; }
u32 rsx_d3d12_lane_list_new(unsigned blk, unsigned pass)
{
    if (s_ll_n >= ENG_LANE_MAXLISTS - 8u) return ~0u;
    EngLaneList* L = &s_ll[s_ll_n];
    L->blk = blk; L->pass = pass; L->kind = 0; L->n = 0; L->hold = 1;
    return s_ll_n++;
}
u32 rsx_d3d12_lane_epoch(void) { return s_ll_epoch; }
void rsx_d3d12_lane_flatten(void) { eng_lanes_flatten(); }
/* the coordinator's own records go to list idx (serial run of a chunk); ~0u = the next record opens a list by itself */
void rsx_d3d12_lane_list_select(u32 idx) { if (idx != ~0u) s_ll[idx].hold = 0; LB->ll_cur = idx; }
void rsx_d3d12_lane_list_commit(u32 idx) { EngLaneList* L = &s_ll[idx]; s_rec_count += L->n; L->hold = 0; }
void rsx_d3d12_lane_list_discard(u32 idx) { s_ll[idx].n = 0; }
u32 rsx_d3d12_lane_list_n(u32 idx) { return s_ll[idx].n; }
void* rsx_d3d12_lane_be_new(void) { EngLaneBe* b = (EngLaneBe*)calloc(1, sizeof *b); if (b) { b->vs_cb_seq = ~0u; b->ll_cur = ~0u; } return b; }
void rsx_d3d12_lane_be_bind(void* b) { t_lb = (EngLaneBe*)b; }
void rsx_d3d12_lane_be_begin(void* bp, u32 list, u32 qfirst, u32 qn)
{
    EngLaneBe* b = (EngLaneBe*)bp;
    b->spec = 1; b->ll_cur = list; b->vs_cb_seq = ~0u; b->vs_cb_off = b->vs_cb_bytes = 0; b->vis_cur = 0;
    b->q_first = qfirst; b->q_n = qn; b->q_k = 0; b->vrep_n = 0; b->ar_used0 = b->ar_used;
    memset(&b->pending, 0, sizeof b->pending);
}
void rsx_d3d12_lane_be_end(void* bp, int keep) { EngLaneBe* b = (EngLaneBe*)bp; b->spec = 0; if (!keep) b->ar_used = b->ar_used0; }
/* the chunk's query reports change hands: taken from the lane (its array becomes the chunk's), appended to s_vis_pending at close, in order */
u32 rsx_d3d12_lane_vrep_take(void* bp, void** arr)
{
    EngLaneBe* b = (EngLaneBe*)bp;
    const u32 n = b->vrep_n;
    if (!n) { *arr = NULL; return 0; }
    *arr = b->vrep; b->vrep = NULL; b->vrep_n = b->vrep_cap = 0;
    return n;
}
void rsx_d3d12_lane_vis_append(void* arr, u32 n)
{
    const struct { u32 slot, index; }* v = arr;
    for (u32 i = 0; i < n; i++)
        if (s_vis_npending < ENG_MAX_REPORTS) { s_vis_pending[s_vis_npending].slot = v[i].slot; s_vis_pending[s_vis_npending].index = v[i].index; s_vis_npending++; }
}
void rsx_d3d12_lane_be_reset_serial(void) { s_lbe0.vs_cb_seq = ~0u; s_lbe0.vs_cb_off = s_lbe0.vs_cb_bytes = 0; }
/* n query_begin slots in job order, counters reset, the encode join the serial begin would have done */
u32 rsx_d3d12_lane_query_reserve(u32 n)
{
    if (!s_qheap || !n) return s_vis_next;
    if (s_enc_pending && !t_on_enc) {
        static int always = -1; if (always < 0) { const char* e = getenv("RSX_ENC_QJOIN"); always = (e && e[0] == '1'); }
        if (always || (u32)(s_vis_next + n - s_vis_job_lo) >= ENG_VIS_SLOTS) eng_enc_join_why("query_begin", 0);
    }
    const u32 first = s_vis_next;
    for (u32 k = 0; k < n; k++) s_vis_count[(first + k) % ENG_VIS_SLOTS] = 0;
    s_vis_next += n;
    return first;
}
int rsx_d3d12_lane_qheap(void) { return s_qheap != NULL; }
u32 rsx_d3d12_lane_arena_free(void)
{
    if (!s_chain_n) return 0;
    const EngStage* st = &s_stage[s_chain[s_chain_n - 1]];
    const u32 start = (st->used + s_stage_align - 1u) & ~(s_stage_align - 1u);
    return start < st->cap ? st->cap - start : 0;
}
/* a new tail chunk (the serial path's own step; may submit or wait): only with every lane idle */
int rsx_d3d12_lane_arena_extend(void)
{
    u32 off = 0;
    return eng_stage_reserve(rsx_d3d12_lane_arena_free() + 1u, &off);
}
/* rissa128 RSX_LANES_ARENA_MB: a new tail chunk WITHOUT waiting for the lanes. Only the cheap case: a free (or new) chunk is chained; no submit, no wait
 * (returns 0 and the coordinator takes the idle path). Safe with lanes in flight: the chunk is picked outside the chain (no lane uses it), and the chain
 * itself (s_chain, s_chain_n, the tail's used) is read by a lane only in eng_lane_region, under s_lane_cs, where it is published here. Old chunks never move. */
int rsx_d3d12_lane_arena_extend_nowait(void)
{
    if (!s_chain_n || s_chain_n >= ENG_STAGE_POOL || !s_lane_cs_init) return 0;
    const EngStage* t = &s_stage[s_chain[s_chain_n - 1]];
    const u32 base = t->base + t->cap;
    if ((u64)base + ENG_STAGE_CHUNK > ENG_STAGE_MAX) return 0;
    const int i = eng_stage_pick(ENG_STAGE_CHUNK);
    if (i < 0) return 0;
    EnterCriticalSection(&s_lane_cs);
    EngStage* s = &s_stage[i];
    s->base = base; s->used = 0; s->in_use = 2;
    s_chain[s_chain_n++] = i;
    LeaveCriticalSection(&s_lane_cs);
    return 1;
}
/* the join with the encode of the previous frame, once, before any lane reads the object tables */
void rsx_d3d12_lane_prepare(void) { t_rel_hold = 1; ENG_JOIN("lanes"); t_rel_hold = 0; }
int rsx_d3d12_lane_enc_pending(void) { return s_enc_pending && !t_on_enc; }

static const u8* eng_stage_rd(u32 off, u32 bytes)
{
    for (u32 k = 0; k < s_chain_n; k++) {
        const EngStage* st = &s_stage[s_chain[k]];
        if (off >= st->base && (u64)off + bytes <= (u64)st->base + st->cap && st->mapped) return st->mapped + (off - st->base);
    }
    return NULL;
}
/* Compares two draw records without the arena offsets and the buffer: the fields, then the BYTES the offsets name (vertices, indices, both constant
 * blocks). Returns NULL = equal, else the name of the first field that differs. *unver = 1 when a draw is equal but one side draws from a
 * promoted vertex-cache buffer and the other from the arena (the GPU buffer cannot be read back here). */
typedef struct EngUnmap { ID3D12Resource* res[12]; int n; } EngUnmap;
static unsigned long long s_cmp_buf_content;   /* draws whose vertex/index bytes were verified by content although buffer ids / arena differed */
unsigned long long rsx_d3d12_lane_cmp_content(void) { return s_cmp_buf_content; }
static unsigned long long s_cmp_zown, s_cmp_zown_tex;   /* lane112: packed own-zeta resolve records whose lane and serial destinations were paired, and texture slots matched through that pairing */
unsigned long long rsx_d3d12_lane_cmp_zown(unsigned long long* tex) { *tex = s_cmp_zown_tex; return s_cmp_zown; }
static const u8* eng_rec_data(u32 vbuf, u32 off, u32 bytes, EngUnmap* um);
static const char* eng_rec_diff(const EngRecord* a, const EngRecord* b, int* unver)
{
    if (a->kind != b->kind) return "kind";
#define CM(f) do { if (memcmp(&a->f, &b->f, sizeof a->f)) return #f; } while (0)
    CM(rt); CM(nrt); CM(depth); CM(pipeline); CM(topology);
    if (a->kind != ENG_REC_DRAW) { CM(clear_rgba); CM(clear_flags); CM(clear_depth); CM(resolve_dst); CM(resolve_packed); CM(clear_rect); return NULL; }
    CM(stride); CM(vertex_count); CM(index_count); CM(vs_cb_bytes); CM(ps_cb_bytes);
    CM(tex); CM(samp); CM(vtex); CM(vsamp); CM(vp); CM(sc); CM(stencil_ref); CM(blend); CM(vis); CM(rt_defer);
#undef CM
    /* lane110: the vertices and indices are compared BY CONTENT whatever they live in (arena or promoted buffer): the same buffer id and offsets
     * = the same bytes; different ids (or buffer vs arena) = the bytes are read (the buffers are upload-heap memory, CPU-readable) and compared. */
    {
        const int same_id = a->vbuf && a->vbuf == b->vbuf && a->vb_off == b->vb_off && a->ib_off == b->ib_off;
        if (!same_id) {
            const u32 vb = a->vertex_count * a->stride, ib = a->index_count * 4u;
            EngUnmap um; um.n = 0;
            const u8 *pa = eng_rec_data(a->vbuf, a->vb_off, vb, &um), *pb = eng_rec_data(b->vbuf, b->vb_off, vb, &um);
            const char* r = NULL;
            if (pa && pb) { if (memcmp(pa, pb, vb)) {
                r = "vertex bytes";
                static int said; if (said++ < 10) {
                    u32 nd = 0, first = 0; for (u32 q = 0; q < vb; q++) if (pa[q] != pb[q]) { if (!nd) first = q; nd++; }
                    char hx[200]; int o = 0; for (u32 q = first; q < first + 12 && q < vb; q++) o += snprintf(hx + o, sizeof hx - o, "%02x/%02x ", pa[q], pb[q]);
                    fprintf(stderr, "[lanes-check2] vertex detail: stride %u verts %u, %u of %u bytes differ, first at vertex %u byte %u (lane/serial): %s | indices %s\n", a->stride, a->vertex_count, nd, vb,
                            first / a->stride, first % a->stride, hx, (a->index_count && eng_rec_data(a->vbuf, a->ib_off, a->index_count * 4u, &um) && eng_rec_data(b->vbuf, b->ib_off, a->index_count * 4u, &um) && !memcmp(eng_rec_data(a->vbuf, a->ib_off, a->index_count * 4u, &um), eng_rec_data(b->vbuf, b->ib_off, a->index_count * 4u, &um), a->index_count * 4u)) ? "equal" : "differ/none");
                } } } else *unver = 1;
            if (!r && ib) {
                const u8 *ia = eng_rec_data(a->vbuf, a->ib_off, ib, &um), *ic = eng_rec_data(b->vbuf, b->ib_off, ib, &um);
                if (ia && ic) { if (memcmp(ia, ic, ib)) r = "index bytes"; } else *unver = 1;
            }
            for (int k = 0; k < um.n; k++) CALL(um.res[k], Unmap, 0, NULL);
            if (r) return r;
            if (a->vbuf || b->vbuf) { if (!*unver) s_cmp_buf_content++; }
        }
    }
    if (a->vs_cb_bytes) { const u8 *pa = eng_stage_rd(a->vs_cb_off, a->vs_cb_bytes), *pb = eng_stage_rd(b->vs_cb_off, b->vs_cb_bytes);
        if (pa && pb && memcmp(pa, pb, a->vs_cb_bytes)) return "vs constants bytes"; }
    if (a->ps_cb_bytes) { const u8 *pa = eng_stage_rd(a->ps_cb_off, a->ps_cb_bytes), *pb = eng_stage_rd(b->ps_cb_off, b->ps_cb_bytes);
        if (pa && pb && memcmp(pa, pb, a->ps_cb_bytes)) return "ps constants bytes"; }
    return NULL;
}
/* RSX_LANES_CHECK=2: the lane's list (shadow) against the serial run's list (real) of the same chunk. Returns the number of differing records
 * (a count mismatch = 1 + |difference|); *n_rec = records compared, *n_unver = equal-but-unverifiable draws; text of the first difference in why. */
u32 rsx_d3d12_lane_compare(u32 shadow, u32 real, u32* n_rec, u32* n_unver, char* why, size_t whyn, const u32* lk_at, const u64* lk_key, u32 n_lk, const u64* st_key, u32 n_st, u32* n_race)
{
    *n_race = 0;
    const EngLaneList* A = &s_ll[shadow]; const EngLaneList* B = &s_ll[real];
    u32 bad = 0; *n_rec = 0; *n_unver = 0;
    u32 vm_a[64], vm_b[64], vm_n = 0;   /* the query slots of a chunk run twice are numbered apart: compared as a bijection, not as numbers */
    if (A->n != B->n) { snprintf(why, whyn, "record count: lane %u, serial %u", A->n, B->n); return 1u + (A->n > B->n ? A->n - B->n : B->n - A->n); }
    /* lane112 RSX_LANES_ZOWN under CHECK=2: the packed copy of the pass's own zeta is a different object on each side (the lane's was made ahead, the serial run's is made at the draw), so the
     * destination of a packed resolve record and the views that read it are compared as a bijection: lane dst X = serial dst Y, and a texture slot holding (a view of) X on one side
     * must hold (a view of) Y on the other. */
    u32 zm_a[16], zm_b[16], zm_n = 0;
    for (u32 i = 0; i < A->n; i++) {
        int unver = 0;
        EngRecord ra = A->v[i], rb = B->v[i];
        if (ra.kind == ENG_REC_DEPTH_RESOLVE && rb.kind == ENG_REC_DEPTH_RESOLVE && ra.resolve_packed && rb.resolve_packed && ra.resolve_dst != rb.resolve_dst && zm_n < 16) {
            zm_a[zm_n] = ra.resolve_dst; zm_b[zm_n] = rb.resolve_dst; zm_n++; s_cmp_zown++;
            rb.resolve_dst = ra.resolve_dst;
        } else if (ra.kind == ENG_REC_DRAW && rb.kind == ENG_REC_DRAW && zm_n) {
            for (u32 k = 0; k < ENG_SRV_TABLE; k++) {
                u32* ha = k < RSX_BE_MAX_TEXTURES ? &ra.tex[k] : &ra.vtex[k - RSX_BE_MAX_TEXTURES];
                u32* hb = k < RSX_BE_MAX_TEXTURES ? &rb.tex[k] : &rb.vtex[k - RSX_BE_MAX_TEXTURES];
                if (*ha == *hb) continue;
                const EngObj* oa = eng_obj(*ha); const EngObj* ob = eng_obj(*hb);
                const u32 na = (oa && oa->alias) ? oa->alias : *ha, nb = (ob && ob->alias) ? ob->alias : *hb;
                for (u32 q = 0; q < zm_n; q++) if (zm_a[q] == na && zm_b[q] == nb) { *hb = *ha; s_cmp_zown_tex++; break; }
            }
        }
        if (ra.kind == ENG_REC_DRAW) {
            if ((ra.vis != 0) != (rb.vis != 0)) { ra.vis = 1; rb.vis = 2; }
            else if (ra.vis) {
                u32 q = 0; while (q < vm_n && vm_a[q] != ra.vis) q++;
                if (q == vm_n) { if (vm_n < 64) { vm_a[vm_n] = ra.vis; vm_b[vm_n] = rb.vis; vm_n++; } else q = 0; }
                if (q < vm_n && vm_b[q] == rb.vis) rb.vis = ra.vis;
            }
        }
        const char* d = eng_rec_diff(&ra, &rb, &unver);
        (*n_rec)++; if (unver) (*n_unver)++;
        if (d && (!strcmp(d, "vertex bytes") || !strcmp(d, "index bytes")) && ra.kind == ENG_REC_DRAW) {
            /* the bytes differ: legitimate when the serial run's vertex-cache lookup of THIS draw (the first draw record at or after a stale event) found its entry stale,
             * i.e. the guest memory changed between the lane run and the serial run */
            int race = 0;
            for (u32 s = 0; s < n_lk && !race; s++) {   /* the lookup whose draw is record i (the first draw record at or after it) */
                if (lk_at[s] > i) continue;
                u32 k = lk_at[s]; while (k < i && B->v[k].kind != ENG_REC_DRAW) k++;
                if (k != i) continue;
                for (u32 t = 0; t < n_st; t++) if (st_key[t] == lk_key[s]) { race = 1; break; }
            }
            if (race) { (*n_race)++; continue; }
        }
        if (d) { if (!bad) snprintf(why, whyn, "record %u of %u (kind %d): field %s differs (lane pipe %u tex0 %u vis %u vbuf %u vb_off %u ib_off %u nv %u idx %u, serial pipe %u tex0 %u vis %u vbuf %u vb_off %u ib_off %u)", i, A->n, (int)A->v[i].kind, d,
                                    A->v[i].pipeline, A->v[i].tex[0], A->v[i].vis, A->v[i].vbuf, A->v[i].vb_off, A->v[i].ib_off, A->v[i].vertex_count, A->v[i].index_count, B->v[i].pipeline, B->v[i].tex[0], B->v[i].vis, B->v[i].vbuf, B->v[i].vb_off, B->v[i].ib_off); bad++; }
    }
    return bad;
}

/* The chunk a record's offset is in. */
static const EngStage* eng_stage_at(u32 off)
{
    for (u32 k = *s_view_chain_n; k-- > 0;) {
        const EngStage* s = &s_stage[s_view_chain[k]];
        if (off >= s->base) return s;
    }
    return NULL;
}

/* The chain is the submit's: its chunks are free again when that is done. */
/* hold: the chain goes to the encode thread, whose fence is not known yet (a mid-frame flush there
 * makes more than one). The chunks are held with a fence that never completes until
 * eng_stage_set_fence gives the real one, after the last list is executed. */
static void eng_stage_submitted_x(u64 fence, int hold)
{
    for (u32 k = 0; k < s_chain_n; k++) {
        EngStage* s = &s_stage[s_chain[k]];
        s->fence = hold ? ~0ull : fence; s->seq = s_submit_seq; s->in_use = (hold || fence) ? 1 : 0;
    }
    s_chain_n = 0;
    /* A load leaves more chunks than a frame of play stages, and larger
     * ones: what no submit has used for a while is let go, one a submit. */
    u32 have = 0; int old = -1;
    for (int i = 0; i < ENG_STAGE_POOL; i++) {
        const EngStage* s = &s_stage[i];
        if (!s->res) continue;
        have++;
        if (!s->in_use && s_submit_seq - s->seq > ENG_STAGE_IDLE && (old < 0 || s->cap > s_stage[old].cap)) old = i;
    }
    if (old >= 0 && (have > ENG_STAGE_KEEP || s_stage[old].cap > ENG_STAGE_CHUNK)) eng_stage_free(&s_stage[old]);
}
static void eng_stage_submitted(u64 fence) { eng_stage_submitted_x(fence, 0); }

/* Encode thread, after its last ExecuteCommandLists: the chunks of the frame are free when `fence` is done
 * (0: the submit failed, nothing of theirs will run). The fence goes first, then in_use. */
static void eng_stage_set_fence(const int* chain, u32 n, u64 fence)
{
    for (u32 k = 0; k < n; k++) {
        EngStage* s = &s_stage[chain[k]];
        InterlockedExchange64((volatile LONG64*)&s->fence, (LONG64)fence);
        if (!fence) s->in_use = 0;
    }
}

/* ---- buffers the engine keeps across frames (vertex cache) ------------------ */

/* Retired vertex buffers are kept and reused, by size class (powers of two
 * from 64 KiB). Creating a committed resource is a kernel allocation
 * (NtGdiDdDDICreateAllocation) and releasing one a kernel free, each a
 * millisecond or more on the thread that walks the FIFO; Drakengard 3's
 * vertex cache turns over a few hundred of these a minute, and the slow
 * frames at 60 fps caught the walker inside those calls. The pool is bounded;
 * past it, buffers are released as before. */
#define ENG_BUF_POOL        256
#define ENG_BUF_POOL_BYTES  ((u64)512 << 20)
static struct { ID3D12Resource* res; u32 cap; } s_buf_pool[ENG_BUF_POOL];
static u32 s_buf_pool_n; static u64 s_buf_pool_bytes;
static u32 s_buf_cap[ENG_MAX_BUFS];
static u32 buf_class(u32 bytes) { u32 c = 65536u; while (c < bytes) c <<= 1; return c; }
int g_eng_buf_pool = 1;   /* RSX_BUF_POOL=0 turns the pool off; a host may switch it in a run */

/* Better than a buffer each, pooled or not: a buffer is a range of a 32 MB
 * UPLOAD-heap page that stays mapped, so making one is the copy and no call
 * into the driver at all. (A level's first frame stores some hundreds: 150
 * to 370 ms of committed buffers, measured in a replay of one.) Only an
 * entry over half a page still gets a buffer of its own, and the pool above.
 * RSX_VB_PAGES=0: a buffer each, as before. */
#define VB_PAGE_BYTES ((u32)32 << 20)
#define VB_PAGES      64
typedef struct { ID3D12Resource* res; u8* mapped; D3D12_GPU_VIRTUAL_ADDRESS va; MemList m; } VbPage;
static VbPage s_vpage[VB_PAGES];
int g_eng_vb_pages = 1;
static u32 s_buf_page[ENG_MAX_BUFS];   /* the page + 1 a buffer is a range of, else 0 */
static u32 s_buf_off[ENG_MAX_BUFS];    /* and where in it; s_buf_cap is the range's size */
/* the bytes of a record's vertices (off = vb_off) or indices (off = ib_off): in the arena (vbuf 0) or in a promoted buffer (CPU view of its upload memory) */
static const u8* eng_rec_data(u32 vbuf, u32 off, u32 bytes, EngUnmap* um)
{
    if (!bytes) return NULL;
    if (!vbuf) return eng_stage_rd(off, bytes);
    if (vbuf > s_buf_count) return NULL;
    const u32 s = vbuf - 1;
    if (!s_buf[s] || (u64)off + bytes > s_buf_bytes[s]) return NULL;
    if (s_buf_page[s]) { const VbPage* pg = &s_vpage[s_buf_page[s] - 1]; return pg->mapped ? pg->mapped + s_buf_off[s] + off : NULL; }
    if (um->n >= 12) return NULL;
    void* m = NULL;
    if (FAILED(CALL(s_buf[s], Map, 0, NULL, &m)) || !m) return NULL;
    um->res[um->n++] = s_buf[s];
    return (const u8*)m + off;
}

static u32 vb_page_take(u32 size, u32* off)
{
    u64 o = 0;
    int empty = -1;
    for (u32 i = 0; i < VB_PAGES; i++) {
        if (!s_vpage[i].res) { if (empty < 0) empty = (int)i; continue; }
        if (mem_take(&s_vpage[i].m, size, 256, &o, 0)) { *off = (u32)o; return i + 1; }
    }
    if (empty < 0) return 0;
    VbPage* pg = &s_vpage[empty];
    pg->res = make_buffer(D3D12_HEAP_TYPE_UPLOAD, VB_PAGE_BYTES, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (!pg->res) return 0;
    D3D12_RANGE nr = {0, 0};
    if (FAILED(CALL(pg->res, Map, 0, &nr, (void**)&pg->mapped))) { RELEASE(pg->res); return 0; }
    pg->va = CALL0(pg->res, GetGPUVirtualAddress);
    mem_fresh(&pg->m, VB_PAGE_BYTES);
    if (!mem_take(&pg->m, size, 256, &o, 0)) return 0;
    *off = (u32)o;
    return (u32)empty + 1;
}

/* One empty page is kept for the next buffers; a second is let go. */
static void vb_page_done(u32 page, u32 off, u32 size)
{
    VbPage* pg = &s_vpage[page - 1];
    mem_give(&pg->m, off, size);
    if (pg->m.live) return;
    for (u32 i = 0; i < VB_PAGES; i++)
        if (&s_vpage[i] != pg && s_vpage[i].res && !s_vpage[i].m.live) {
            CALL(pg->res, Unmap, 0, NULL); RELEASE(pg->res);
            pg->mapped = NULL; pg->m.nfr = 0;
            break;
        }
}

static u32 eng_buffer_wrap(void* user, void* data, u32 bytes)
{
    ENG_JOIN("buffer_wrap");
    (void)user;
    if (!s_dev || !data || !bytes) return 0;
    u32 slot;
    if (s_buf_free_count) slot = s_buf_free[s_buf_free_count - 1];
    else if (s_buf_count < ENG_MAX_BUFS) slot = s_buf_count;
    else return 0;
    /* D3D12 cannot read the engine's memory in place; it is copied into
     * upload-heap memory and, ownership being ours, freed here. */
    const u32 size = (bytes + 4095u) & ~4095u;
    u32 off = 0;
    const u32 page = g_eng_vb_pages && size <= VB_PAGE_BYTES / 2 ? vb_page_take(size, &off) : 0;
    if (page) {
        memcpy(s_vpage[page - 1].mapped + off, data, bytes);
        _aligned_free(data);
        s_stat_buf++;
        if (s_buf_free_count) s_buf_free_count--; else s_buf_count++;
        s_buf[slot] = s_vpage[page - 1].res; s_buf_bytes[slot] = bytes; s_buf_retired[slot] = 0;
        s_buf_page[slot] = page; s_buf_off[slot] = off; s_buf_cap[slot] = size;
        return slot + 1;
    }
    const u32 cap = buf_class(bytes);
    ID3D12Resource* b = NULL;
    for (u32 i = 0; i < s_buf_pool_n; i++) {
        if (s_buf_pool[i].cap != cap) continue;
        b = s_buf_pool[i].res;
        s_buf_pool_bytes -= cap;
        s_buf_pool[i] = s_buf_pool[--s_buf_pool_n];
        break;
    }
    if (!b) b = make_buffer(D3D12_HEAP_TYPE_UPLOAD, cap, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (!b) return 0;
    s_buf_cap[slot] = cap;
    void* m = NULL; D3D12_RANGE nr = {0, 0};
    if (FAILED(CALL(b, Map, 0, &nr, &m))) { RELEASE(b); return 0; }
    memcpy(m, data, bytes);
    CALL(b, Unmap, 0, NULL);
    _aligned_free(data);
    s_stat_buf++;
    if (s_buf_free_count) s_buf_free_count--; else s_buf_count++;
    s_buf[slot] = b; s_buf_bytes[slot] = bytes; s_buf_retired[slot] = 0;
    s_buf_page[slot] = 0; s_buf_off[slot] = 0;
    return slot + 1;
}

static u64 s_buf_retired_fence[ENG_MAX_BUFS];
static void eng_collect_retired_buffers(void)
{
    if (s_rec_count) return;
    for (u32 i = 0; i < s_buf_count; i++) {
        if (!s_buf_retired[i] || !fence_done(s_buf_retired_fence[i])) continue;
        s_buf_retired[i] = 0;
        if (s_buf_page[i]) {
            vb_page_done(s_buf_page[i], s_buf_off[i], s_buf_cap[i]);
            s_buf[i] = NULL; s_buf_page[i] = 0;
        } else if (g_eng_buf_pool && s_buf_pool_n < ENG_BUF_POOL && s_buf_pool_bytes + s_buf_cap[i] <= ENG_BUF_POOL_BYTES) {
            s_buf_pool[s_buf_pool_n].res = s_buf[i]; s_buf_pool[s_buf_pool_n].cap = s_buf_cap[i];
            s_buf_pool_n++; s_buf_pool_bytes += s_buf_cap[i];
            s_buf[i] = NULL;
        } else {
            RELEASE(s_buf[i]);
        }
        s_buf_free[s_buf_free_count++] = i;
    }
}
static void eng_buffer_release(void* user, u32 h)
{
    ENG_JOIN("buffer_release");
    (void)user;
    if (!h || h > s_buf_count || !s_buf[h - 1] || s_buf_retired[h - 1]) return;
    s_buf_retired[h - 1] = 1;
    s_buf_retired_fence[h - 1] = eng_retire_fence();
}

/* ---- window ---------------------------------------------------------------- */

static RsxD3D12Overlay s_overlay;
void rsx_d3d12_set_overlay(const RsxD3D12Overlay* o) { if (o) s_overlay = *o; else memset(&s_overlay, 0, sizeof s_overlay); }

static LRESULT CALLBACK eng_wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    if (s_overlay.message) {
        intptr_t r = 0;
        if (s_overlay.message((void*)hwnd, msg, (uintptr_t)wp, (intptr_t)lp, &r)) return (LRESULT)r;
    }
    /* The mouse, for a port played with it (rsx_host_mouse.h); and Escape,
     * which closes the window unless such a port wants the key. */
    { intptr_t r = 0;
      if (rsx_host_mouse_message(hwnd, msg, (uintptr_t)wp, (intptr_t)lp, &r)) return (LRESULT)r; }
    switch (msg) {
    case WM_CLOSE:   s_window_closed = 1; DestroyWindow(hwnd); return 0;
    case WM_ACTIVATEAPP:
        V2EV("focus", "\"state\":\"%s\"", wp ? "gained" : "lost");
        if (wp && s_display == DISP_FULLSCREEN) s_fs_restore = 1;
        /* Alt-Tab out of exclusive full screen: the swap chain leaves it HERE, on the window thread (DXGI no longer watches
         * the window, see MakeWindowAssociation). Before, DXGI's own hook ran inside this message and took the swap-chain lock
         * that Present (encode thread) held while it waited for this very thread: deadlock, "no flip" for ever. */
        if (!wp && s_display == DISP_FULLSCREEN && s_swap && !s_headless) {
            BOOL fs_now = FALSE;
            if (SUCCEEDED(CALL(s_swap, GetFullscreenState, &fs_now, NULL)) && fs_now) {
                CALL(s_swap, SetFullscreenState, FALSE, NULL);
                InterlockedExchange(&s_fs_left, 1);
                fprintf(stderr, "[RSX d3d12] focus lost: left exclusive full screen\n");
            }
            ShowWindow(hwnd, SW_MINIMIZE);
        }
        break;
    case WM_SIZE: V2EV("window", "\"state\":\"%s\",\"w\":%d,\"h\":%d", wp == SIZE_MINIMIZED ? "minimized" : wp == SIZE_MAXIMIZED ? "maximized" : "restored", (int)LOWORD(lp), (int)HIWORD(lp)); break;
    case WM_DESTROY: PostQuitMessage(0); return 0;
    case WM_KEYDOWN:
        if (wp == VK_ESCAPE) { s_window_closed = 1; DestroyWindow(hwnd); }
        return 0;
    }
    return DefWindowProcA(hwnd, msg, wp, lp);
}

static HWND eng_create_window(u32 width, u32 height, const char* title)
{
    WNDCLASSEXA wc = {0};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = eng_wndproc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    /* The executable's icon resource (id 101 in the port's app.rc), when it
     * has one; the default application icon otherwise. */
    wc.hIcon = LoadIconA(GetModuleHandle(NULL), MAKEINTRESOURCEA(101));
    wc.hIconSm = wc.hIcon;
    wc.lpszClassName = "ps3recomp_d3d12_engine";
    RegisterClassExA(&wc);
    SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED);
    if (s_display != DISP_WINDOWED) {
        /* A popup over the display: for borderless that is the whole of it
         * (no mode switch, so alt-tab and the flip model behave as in a
         * window); for exclusive full screen DXGI takes it from here. */
        return CreateWindowExA(0, "ps3recomp_d3d12_engine", title ? title : "ps3recomp (D3D12)",
                               WS_POPUP | WS_VISIBLE, 0, 0, (int)width, (int)height,
                               NULL, NULL, GetModuleHandle(NULL), NULL);
    }
    RECT wr = {0, 0, (LONG)width, (LONG)height};
    AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);
    return CreateWindowExA(0, "ps3recomp_d3d12_engine", title ? title : "ps3recomp (D3D12)",
                           WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
                           wr.right - wr.left, wr.bottom - wr.top,
                           NULL, NULL, GetModuleHandle(NULL), NULL);
}

/* ---- helper shaders: present blit, depth resolve, depth pack --------------- */

#include "rsx_present_passes.h"

static const char kHelperHLSL[] =
    "struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
    "VSOut vs_main(uint vid : SV_VertexID) {\n"
    "    float2 p = float2((vid << 1) & 2, vid & 2);\n"
    "    VSOut o; o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0); o.uv = float2(p.x, 1.0 - p.y); return o;\n"
    "}\n"
    "Texture2D<float4> src : register(t0);\n"
    "SamplerState smp : register(s0);\n"
    "float4 ps_blit(VSOut i) : SV_Target { return src.Sample(smp, i.uv); }\n"
    /* Depth copies are texel for texel from the top-left corner: a copy
     * smaller than its depth target is the region the title's texture covers. */
    "Texture2D<float> dsrc : register(t0);\n"
    "float ps_depth(VSOut i) : SV_Target { return dsrc.Load(int3(i.pos.xy, 0)); }\n"
    /* The depth as the RSX's D24S8 word reads through an A8R8G8B8 texture:
     * A = depth[23:16], R = depth[15:8], G = depth[7:0], B = stencil (0). */
    /* holo124 (02/11): d = 1.0 (the cleared depth: sky, empty space) gave 1.0 * 16777215 + 0.5 = 16777215.5, which float32
     * rounds to 16777216 = 0x1000000; the masks then turned it into (A,R,G) = (0,0,0), i.e. depth 0 = nearest. Every
     * soft-depth fade (Apogee hologram platforms, lasers, fog billboards) saw the sky as a wall at the camera and went
     * to alpha 0. The word is clamped to 0xFFFFFF (D24 1.0), as the RSX stores it. RSX_PACK_FIX=0 picks the *_old
     * entries (the old code). */
    "float4 ps_pack(VSOut i) : SV_Target {\n"
    "    uint v = min(uint(saturate(dsrc.Load(int3(i.pos.xy, 0))) * 16777215.0 + 0.5), 16777215u);\n"
    "    return float4(float((v >> 8) & 255u), float(v & 255u), 0.0, float((v >> 16) & 255u)) / 255.0;\n"
    "}\n"
    "float4 ps_pack_old(VSOut i) : SV_Target {\n"
    "    uint v = uint(saturate(dsrc.Load(int3(i.pos.xy, 0))) * 16777215.0 + 0.5);\n"
    "    return float4(float((v >> 8) & 255u), float(v & 255u), 0.0, float((v >> 16) & 255u)) / 255.0;\n"
    "}\n"
    /* The same from an MSAA depth twin: its first sample. */
    "Texture2DMS<float> dms : register(t0);\n"
    "float ps_depth_ms(VSOut i) : SV_Target { return dms.Load(int2(i.pos.xy), 0); }\n"
    /* ... and as depth: the twin's first sample into the single-sampled
     * depth target, for a pass that is not multisampled. */
    "float ps_zcopy_ms(VSOut i) : SV_Depth { return dms.Load(int2(i.pos.xy), 0); }\n"
    "float4 ps_pack_ms(VSOut i) : SV_Target {\n"
    "    uint v = min(uint(saturate(dms.Load(int2(i.pos.xy), 0)) * 16777215.0 + 0.5), 16777215u);\n"
    "    return float4(float((v >> 8) & 255u), float(v & 255u), 0.0, float((v >> 16) & 255u)) / 255.0;\n"
    "}\n"
    "float4 ps_pack_ms_old(VSOut i) : SV_Target {\n"
    "    uint v = uint(saturate(dms.Load(int2(i.pos.xy), 0)) * 16777215.0 + 0.5);\n"
    "    return float4(float((v >> 8) & 255u), float(v & 255u), 0.0, float((v >> 16) & 255u)) / 255.0;\n"
    "}\n";

static ID3DBlob* eng_compile(const char* src, u32 len, const char* entry, const char* target, const char* what)
{
    ID3DBlob* b = NULL; ID3DBlob* e = NULL;
    HRESULT hr = D3DCompile(src, len, what, NULL, NULL, entry, target,
                            D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_IEEE_STRICTNESS, 0, &b, &e);
    if (FAILED(hr)) {
        static int n = 0;
        if (n < 32 && ++n)
            fprintf(stderr, "[rsx engine/d3d12] %s compile failed: %.900s\n", what,
                    e ? (const char*)CALL0(e, GetBufferPointer) : "no diagnostic");
        RELEASE(e);
        return NULL;
    }
    RELEASE(e);
    return b;
}

static ID3D12PipelineState* eng_helper_pso(ID3DBlob* vs, ID3DBlob* ps, DXGI_FORMAT rt)
{
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {0};
    pd.pRootSignature = s_helper_rootsig;
    pd.VS.pShaderBytecode = CALL0(vs, GetBufferPointer); pd.VS.BytecodeLength = CALL0(vs, GetBufferSize);
    pd.PS.pShaderBytecode = CALL0(ps, GetBufferPointer); pd.PS.BytecodeLength = CALL0(ps, GetBufferSize);
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pd.SampleMask = UINT_MAX;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1; pd.RTVFormats[0] = rt;
    pd.SampleDesc.Count = 1;
    ID3D12PipelineState* pso = NULL;
    HRESULT hr = CALL(s_dev, CreateGraphicsPipelineState, &pd, &IID_ID3D12PipelineState, (void**)&pso);
    if (FAILED(hr)) fprintf(stderr, "[rsx engine/d3d12] helper pipeline failed: 0x%08lX\n", (long)hr);
    return pso;
}

static int eng_make_root_signatures(void)
{
    /* The draw root signature, matching the decompilers' register plan:
     *   b0 VS constants (VPConst), b1 PS constants (PSConstants),
     *   t0-t15 / s0-s15 fragment units, t16-t19 / s0-s3 vertex units. */
    D3D12_DESCRIPTOR_RANGE r_ps_srv = {0}, r_ps_smp = {0}, r_vs_srv = {0}, r_vs_smp = {0};
    r_ps_srv.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;     r_ps_srv.NumDescriptors = RSX_BE_MAX_TEXTURES;
    r_ps_smp.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER; r_ps_smp.NumDescriptors = RSX_BE_MAX_TEXTURES;
    r_vs_srv.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;     r_vs_srv.NumDescriptors = RSX_BE_MAX_VERTEX_TEXTURES; r_vs_srv.BaseShaderRegister = 16;
    r_vs_smp.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER; r_vs_smp.NumDescriptors = RSX_BE_MAX_VERTEX_TEXTURES;
    D3D12_ROOT_PARAMETER p[6] = {0};
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[0].Descriptor.ShaderRegister = 0; p[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;   /* RSX_SSR: fragment programs read c[8..11] too */
    p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[1].DescriptorTable.NumDescriptorRanges = 1; p[1].DescriptorTable.pDescriptorRanges = &r_ps_srv; p[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[2].DescriptorTable.NumDescriptorRanges = 1; p[2].DescriptorTable.pDescriptorRanges = &r_ps_smp; p[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[3].DescriptorTable.NumDescriptorRanges = 1; p[3].DescriptorTable.pDescriptorRanges = &r_vs_srv; p[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    p[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[4].Descriptor.ShaderRegister = 1; p[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    p[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[5].DescriptorTable.NumDescriptorRanges = 1; p[5].DescriptorTable.pDescriptorRanges = &r_vs_smp; p[5].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    D3D12_ROOT_SIGNATURE_DESC rd = {0};
    rd.NumParameters = 6; rd.pParameters = p;
    rd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ID3DBlob* sig = NULL; ID3DBlob* err = NULL;
    if (FAILED(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err))) {
        fprintf(stderr, "[rsx engine/d3d12] root signature: %s\n", err ? (const char*)CALL0(err, GetBufferPointer) : "?");
        RELEASE(err); return -1;
    }
    HRESULT hr = CALL(s_dev, CreateRootSignature, 0, CALL0(sig, GetBufferPointer), CALL0(sig, GetBufferSize),
                      &IID_ID3D12RootSignature, (void**)&s_rootsig);
    RELEASE(sig);
    if (FAILED(hr)) return -1;

    /* The helper passes: one texture; s0 point/clamp, s1 linear/clamp (the
     * present passes, rsx_present_passes.h). */
    D3D12_DESCRIPTOR_RANGE hr_srv = {0};
    hr_srv.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; hr_srv.NumDescriptors = 1;
    D3D12_ROOT_PARAMETER hp[1] = {0};
    hp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; hp[0].DescriptorTable.NumDescriptorRanges = 1;
    hp[0].DescriptorTable.pDescriptorRanges = &hr_srv; hp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_STATIC_SAMPLER_DESC ss = {0};
    ss.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    ss.AddressU = ss.AddressV = ss.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    ss.MaxLOD = D3D12_FLOAT32_MAX; ss.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_STATIC_SAMPLER_DESC hs[2] = { ss, ss };
    hs[1].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR; hs[1].ShaderRegister = 1;
    D3D12_ROOT_SIGNATURE_DESC hd = {0};
    hd.NumParameters = 1; hd.pParameters = hp; hd.NumStaticSamplers = 2; hd.pStaticSamplers = hs;
    if (FAILED(D3D12SerializeRootSignature(&hd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err))) { RELEASE(err); return -1; }
    hr = CALL(s_dev, CreateRootSignature, 0, CALL0(sig, GetBufferPointer), CALL0(sig, GetBufferSize),
              &IID_ID3D12RootSignature, (void**)&s_helper_rootsig);
    RELEASE(sig);
    return FAILED(hr) ? -1 : 0;
}

static ID3D12DescriptorHeap* eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE type, u32 n, int visible)
{
    D3D12_DESCRIPTOR_HEAP_DESC d = {0};
    d.Type = type; d.NumDescriptors = n;
    d.Flags = visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    ID3D12DescriptorHeap* h = NULL;
    if (FAILED(CALL(s_dev, CreateDescriptorHeap, &d, &IID_ID3D12DescriptorHeap, (void**)&h))) return NULL;
    return h;
}

/* ---- device ---------------------------------------------------------------- */

/* The swap chain over s_hwnd at s_win_w x s_win_h for s_display: flip
 * model, three buffers; tearing allowed whenever the system supports it
 * outside exclusive full screen, so v-sync can be switched off while
 * running. At start-up and again when the display settings change. */
static int eng_swapchain_create(IDXGIFactory4* factory)
{
    HRESULT hr;
    DXGI_SWAP_CHAIN_DESC1 sd = {0};
    sd.Width = s_win_w; sd.Height = s_win_h; sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.SampleDesc.Count = 1; sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 3; sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    s_tearing = 0;
    if (s_display != DISP_FULLSCREEN) {
        IDXGIFactory5* f5 = NULL;
        if (SUCCEEDED(CALL(factory, QueryInterface, &IID_IDXGIFactory5, (void**)&f5))) {
            BOOL allow = FALSE;
            if (SUCCEEDED(CALL(f5, CheckFeatureSupport, DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allow, sizeof allow)) && allow)
                s_tearing = 1;
            RELEASE(f5);
        }
        if (s_tearing) sd.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
    }
    IDXGISwapChain1* sc1 = NULL;
    DXGI_SWAP_CHAIN_FULLSCREEN_DESC fsd = {0};
    fsd.Windowed = TRUE;
    int want_fs = 0;
    if (s_display == DISP_FULLSCREEN) {
        /* Exclusive: the swap chain owns the output. 04/11 menu110: it is made WINDOWED with ALLOW_MODE_SWITCH and put in
         * full screen below with SetFullscreenState, here, on the window thread, while the encode and present threads are
         * idle. Made with Windowed=FALSE, DXGI did the switch on the first Present, on the presenting thread: that thread
         * SetWindowLong()s a window owned by the pump thread, which was waiting in enc_join for that very present (menu109
         * run a1: deadlock, "no flip" for ever). ALLOW_MODE_SWITCH lets the mode differ from the desktop's (RSX_WINDOW). */
        sd.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        want_fs = 1;
    }
    hr = CALL(factory, CreateSwapChainForHwnd, (IUnknown*)s_queue, s_hwnd, &sd, &fsd, NULL, &sc1);
    if (FAILED(hr) && s_display == DISP_FULLSCREEN) {
        want_fs = 0;
        /* The output refused the mode: fall back to a borderless window
         * of the same size rather than no window at all. */
        fprintf(stderr, "[rsx engine/d3d12] exclusive full screen refused (0x%08lX); borderless instead\n", (long)hr);
        s_display = DISP_BORDERLESS;
        sd.Flags &= ~(UINT)DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        {
            IDXGIFactory5* f5 = NULL; BOOL allow = FALSE;
            if (SUCCEEDED(CALL(factory, QueryInterface, &IID_IDXGIFactory5, (void**)&f5))) {
                if (SUCCEEDED(CALL(f5, CheckFeatureSupport, DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allow, sizeof allow)) && allow)
                    s_tearing = 1;
                RELEASE(f5);
            }
            if (s_tearing) sd.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
        }
        hr = CALL(factory, CreateSwapChainForHwnd, (IUnknown*)s_queue, s_hwnd, &sd, NULL, NULL, &sc1);
    }
    if (FAILED(hr)) { fprintf(stderr, "[rsx engine/d3d12] swap chain failed: 0x%08lX\n", (long)hr); return -1; }
    /* NO_WINDOW_CHANGES: DXGI must not hook the window's messages (its focus-loss handling deadlocks with Present on another
     * thread); the focus loss and return are handled by eng_wndproc and eng_fullscreen_restore. */
    CALL(factory, MakeWindowAssociation, s_hwnd, s_display == DISP_FULLSCREEN ? (DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_WINDOW_CHANGES) : DXGI_MWA_NO_ALT_ENTER);   /* exclusive only: in borderless NO_WINDOW_CHANGES made the taskbar flicker through (user, 07/11 b50-b52) */
    hr = CALL(sc1, QueryInterface, &IID_IDXGISwapChain3, (void**)&s_swap);
    RELEASE(sc1);
    if (FAILED(hr)) return -1;
    if (!s_backbuf_rtv_heap) s_backbuf_rtv_heap = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 3, 0);
    if (!s_backbuf_rtv_heap) return -1;
    s_rtv_step = CALL(s_dev, GetDescriptorHandleIncrementSize, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    for (u32 i = 0; i < 3; i++) {
        if (FAILED(CALL(s_swap, GetBuffer, i, &IID_ID3D12Resource, (void**)&s_backbuf[i]))) return -1;
        CALL(s_dev, CreateRenderTargetView, s_backbuf[i], NULL, cpu_handle(s_backbuf_rtv_heap, s_rtv_step, i));
    }
    if (want_fs) {
        /* The switch itself, on this (window) thread, nothing presenting. A refusal leaves a borderless window. */
        HRESULT fh = CALL(s_swap, SetFullscreenState, TRUE, NULL);
        if (SUCCEEDED(fh)) {
            for (u32 i = 0; i < 3; i++) RELEASE(s_backbuf[i]);
            fh = CALL(s_swap, ResizeBuffers, 0, 0, 0, DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH);
            if (FAILED(fh)) { fprintf(stderr, "[rsx engine/d3d12] ResizeBuffers after full screen failed: 0x%08lX\n", (long)fh); return -1; }
            for (u32 i = 0; i < 3; i++) {
                if (FAILED(CALL(s_swap, GetBuffer, i, &IID_ID3D12Resource, (void**)&s_backbuf[i]))) return -1;
                CALL(s_dev, CreateRenderTargetView, s_backbuf[i], NULL, cpu_handle(s_backbuf_rtv_heap, s_rtv_step, i));
            }
            fprintf(stderr, "[RSX d3d12] exclusive full screen entered on the window thread\n");
        } else {
            fprintf(stderr, "[rsx engine/d3d12] SetFullscreenState(TRUE) refused (0x%08lX); borderless instead\n", (long)fh);
            s_display = DISP_BORDERLESS;
        }
    }
    return 0;
}

/* D3D12_DBG=1: the debug layer, and its errors and warnings on stderr at
 * each present (without this they reach an attached debugger only): the
 * first three of each kind, so that one repeated every frame does not bury
 * the rest. */
static ID3D12InfoQueue* s_infoq;
static int eng_debug_fresh(int id)
{
    static struct { int id; unsigned n; } seen[64];
    for (int i = 0; i < 64; i++) {
        if (seen[i].n && seen[i].id != id) continue;
        seen[i].id = id;
        if (seen[i].n >= 3) return 0;
        seen[i].n++;
        return 1;
    }
    return 0;
}
static void eng_debug_drain(void)
{
    if (!s_infoq) return;
    const UINT64 n = CALL0(s_infoq, GetNumStoredMessages);
    for (UINT64 i = 0; i < n; i++) {
        SIZE_T len = 0;
        CALL(s_infoq, GetMessage, i, NULL, &len);
        D3D12_MESSAGE* m = len ? (D3D12_MESSAGE*)malloc(len) : NULL;
        if (m && SUCCEEDED(CALL(s_infoq, GetMessage, i, m, &len)) && m->Severity <= D3D12_MESSAGE_SEVERITY_WARNING &&
            eng_debug_fresh((int)m->ID)) {
            fprintf(stderr, "[d3d12 debug] %s\n", m->pDescription);
        }
        free(m);
    }
    if (n) CALL0(s_infoq, ClearStoredMessages);
}

static int eng_init_device(u32 width, u32 height)
{
    HRESULT hr;
    eng_dred_enable();
    if (getenv("D3D12_DBG")) {
        ID3D12Debug* dbg = NULL;
        if (SUCCEEDED(D3D12GetDebugInterface(&IID_ID3D12Debug, (void**)&dbg)) && dbg) {
            CALL0(dbg, EnableDebugLayer);
            /* D3D12_GBV=1: GPU-based validation as well (descriptors, resource
             * states seen by shaders), much slower */
            ID3D12Debug1* dbg1 = NULL;
            if (getenv("D3D12_GBV") && SUCCEEDED(CALL(dbg, QueryInterface, &IID_ID3D12Debug1, (void**)&dbg1)) && dbg1) {
                CALL(dbg1, SetEnableGPUBasedValidation, TRUE); RELEASE(dbg1);
                fprintf(stderr, "[rsx engine/d3d12] GPU-based validation enabled\n");
            }
            RELEASE(dbg);
            fprintf(stderr, "[rsx engine/d3d12] debug layer enabled\n");
        }
    }
    IDXGIFactory4* factory = NULL;
    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory4, (void**)&factory))) return -1;
    {
        IDXGIFactory6* f6 = NULL;
        if (SUCCEEDED(CALL(factory, QueryInterface, &IID_IDXGIFactory6, (void**)&f6))) {
            DXGI_GPU_PREFERENCE pref = getenv("CELLMARK_IGPU") ? DXGI_GPU_PREFERENCE_MINIMUM_POWER
                                                                : DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE;
            IDXGIAdapter1* ad = NULL;
            for (UINT i = 0; CALL(f6, EnumAdapterByGpuPreference, i, pref, &IID_IDXGIAdapter1, (void**)&ad) != DXGI_ERROR_NOT_FOUND; i++) {
                DXGI_ADAPTER_DESC1 desc; CALL(ad, GetDesc1, &desc);
                if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                    SUCCEEDED(D3D12CreateDevice((IUnknown*)ad, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, (void**)&s_dev))) {
                    s_pc_adapter_id = ((u64)desc.VendorId << 40) ^ ((u64)desc.DeviceId << 16) ^ (u64)desc.Revision ^ ((u64)desc.SubSysId << 24);
                    fprintf(stderr, "[rsx engine/d3d12] adapter: %ls (%llu MB)\n", desc.Description,
                            (unsigned long long)(desc.DedicatedVideoMemory >> 20));
                    { LARGE_INTEGER umd; umd.QuadPart = 0; if (SUCCEEDED(CALL(ad, CheckInterfaceSupport, &IID_IDXGIDevice, &umd))) s_pc_umd = (unsigned long long)umd.QuadPart; }
                    if (FAILED(CALL(ad, QueryInterface, &IID_IDXGIAdapter3, (void**)&s_adapter3))) s_adapter3 = NULL;
                    else { g_rsx_target_info_hook = eng_target_info; g_rsx_placed_hook = eng_gov_placed; g_rsx_zeta_direct_hook = rsx_d3d12_zeta_direct_ok;g_rsx_doomed_hook = eng_doomed_bytes; g_rsx_vram_hook = eng_vram_query; g_rsx_parked_hook = eng_parked_twin_bytes; g_v2_vram_hook = eng_v2_vram; g_rsx_cache_dir_hook = eng_cache_dir_hook; g_rsx_level_hook = eng_level_hook;
                           g_rsx_cache_sig_hook = eng_cache_sig_hook; g_rsx_cache_warm_hook = eng_cache_warm_hook; g_rsx_cache_flush_hook = eng_cache_flush_hook; }
                    RELEASE(ad); break;
                }
                RELEASE(ad);
            }
            RELEASE(f6);
        }
    }
    if (!s_dev && FAILED(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, (void**)&s_dev))) {
        RELEASE(factory); return -1;
    }
    if (getenv("D3D12_DBG") && FAILED(CALL(s_dev, QueryInterface, &IID_ID3D12InfoQueue, (void**)&s_infoq))) s_infoq = NULL;
    {   /* D3D12_FEATURE_DATA_D3D12_OPTIONS13, spelt out for SDKs older than it */
        struct { BOOL unrestricted_copy_pitch, unrestricted_vertex_alignment, flip_y, flip_z, copy_between_dimensions,
                      alpha_blend_factor; } o13;
        memset(&o13, 0, sizeof o13);
        s_copy_relaxed = SUCCEEDED(CALL(s_dev, CheckFeatureSupport, (D3D12_FEATURE)42, &o13, sizeof o13)) && o13.unrestricted_copy_pitch;
        const char* e = getenv("RSX_D3D12_STRICT_COPY");
        s_copy_strict_test = e && e[0] == '1';
        e = getenv("RSX_D3D12_STAGE_ALIGN");
        if (e && atoi(e) >= 256 && !(atoi(e) & (atoi(e) - 1))) s_stage_align = (u32)atoi(e);
        fprintf(stderr, "[rsx engine/d3d12] relaxed copy alignment: %s%s, stage alignment %u\n",
                s_copy_relaxed ? "yes" : "no", s_copy_strict_test ? " (held to the strict rule: RSX_D3D12_STRICT_COPY)" : "",
                s_stage_align);
    }
    D3D12_COMMAND_QUEUE_DESC qd = {0}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(CALL(s_dev, CreateCommandQueue, &qd, &IID_ID3D12CommandQueue, (void**)&s_queue))) { RELEASE(factory); return -1; }
    rsx_rt_config_load();
    rsx_rt_d3d12_init(s_dev, s_queue);   /* DXR probe (logged); creates nothing unless RT_AO=1 */

    if (!s_headless && eng_swapchain_create(factory) != 0) { RELEASE(factory); return -1; }
    RELEASE(factory);

    if (FAILED(CALL(s_dev, CreateFence, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void**)&s_fence))) return -1;
    s_fence_event = CreateEventA(NULL, FALSE, FALSE, NULL);
    for (u32 i = 0; i < ENG_FRAMES; i++)
        if (FAILED(CALL(s_dev, CreateCommandAllocator, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator, (void**)&s_alloc[i]))) return -1;
    if (FAILED(CALL(s_dev, CreateCommandList, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, s_alloc[0], NULL,
                    &IID_ID3D12GraphicsCommandList, (void**)&s_list))) return -1;
    CALL0(s_list, Close); s_list_open = 0;

    s_srv_step = CALL(s_dev, GetDescriptorHandleIncrementSize, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    s_rtv_step = CALL(s_dev, GetDescriptorHandleIncrementSize, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    s_dsv_step = CALL(s_dev, GetDescriptorHandleIncrementSize, D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    s_smp_step = CALL(s_dev, GetDescriptorHandleIncrementSize, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    s_srv_cpu = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, ENG_MAX_OBJECTS + 1, 0);
    s_rtv_cpu = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, ENG_MAX_OBJECTS, 0);
    s_dsv_cpu = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, ENG_MAX_OBJECTS, 0);
    s_rtv_ms = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, ENG_MAX_OBJECTS, 0);
    s_dsv_ms = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, ENG_MAX_OBJECTS, 0);
    s_srv_ms = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, ENG_MAX_OBJECTS, 0);
    s_smp_cpu = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, ENG_MAX_SAMPLERS + 1, 0);
    s_srv_gpu = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, ENG_FRAMES * ENG_SRV_PER_SLOT, 1);
    if (!s_srv_cpu || !s_rtv_cpu || !s_dsv_cpu || !s_smp_cpu || !s_srv_gpu) return -1;
    for (u32 i = 0; i < ENG_FRAMES; i++)
        if (!(s_smp_gpu[i] = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, ENG_SMP_PER_SLOT, 1))) return -1;

    /* Sampler slot 0: the default for an unbound unit. */
    {
        D3D12_SAMPLER_DESC sd = {0};
        sd.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sd.AddressU = sd.AddressV = sd.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sd.MaxLOD = D3D12_FLOAT32_MAX; sd.MaxAnisotropy = 1;
        CALL(s_dev, CreateSampler, &sd, cpu_handle(s_smp_cpu, s_smp_step, 0));
    }
    /* The null texture: 1x1 transparent black at CPU SRV slot ENG_SRV_NULL. */
    s_null_tex = make_texture(DXGI_FORMAT_R8G8B8A8_UNORM, 1, 1, 1, 1, D3D12_RESOURCE_FLAG_NONE, ENG_SHADER_READ, NULL);
    if (!s_null_tex) return -1;
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {0};
        sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM; sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sd.Texture2D.MipLevels = 1;
        CALL(s_dev, CreateShaderResourceView, s_null_tex, &sd, obj_srv(0));
    }
    if (!getenv("RSX_NO_QUERIES")) {
        D3D12_QUERY_HEAP_DESC qh = {0};
        qh.Type = D3D12_QUERY_HEAP_TYPE_OCCLUSION; qh.Count = ENG_FRAMES * ENG_QUERIES_PER_SLOT;
        if (FAILED(CALL(s_dev, CreateQueryHeap, &qh, &IID_ID3D12QueryHeap, (void**)&s_qheap))) s_qheap = NULL;
        if (s_qheap) s_qread = make_buffer(D3D12_HEAP_TYPE_READBACK, (u64)ENG_FRAMES * ENG_QUERIES_PER_SLOT * 8u, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!s_qread) { RELEASE(s_qheap); }
    }
    { const char* gl = getenv("TOD_GPU_LOAD"); const int gpu_load = !(gl && gl[0] == '0');
    if (getenv("RSX_GPU_TIME") || getenv("RSX_GPU_TS_DUMP") || gpu_load) {
        D3D12_QUERY_HEAP_DESC th = {0};
        s_gt_report = getenv("RSX_GPU_TIME") != NULL;
        if (getenv("RSX_GPU_TS_DUMP") && *getenv("RSX_GPU_TS_DUMP")) {
            s_gts_fp = fopen(getenv("RSX_GPU_TS_DUMP"), "w");
            if (s_gts_fp) fprintf(s_gts_fp, "fence,kind,cpu_exec_us,gpu_start_us,gpu_end_us\n");
        }
        th.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; th.Count = ENG_FRAMES * 2;
        if (SUCCEEDED(CALL(s_dev, CreateQueryHeap, &th, &IID_ID3D12QueryHeap, (void**)&s_tsheap)))
            s_tsread = make_buffer(D3D12_HEAP_TYPE_READBACK, (u64)ENG_FRAMES * 16u, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!s_tsread || FAILED(CALL(s_queue, GetTimestampFrequency, &s_ts_freq)) || !s_ts_freq) { RELEASE(s_tsheap); RELEASE(s_tsread); }
        g_rsx_gpu_load_ok = s_tsheap != NULL;
        if (s_tsheap && getenv("RSX_GPU_TIME") && atoi(getenv("RSX_GPU_TIME")) >= 2) {
            D3D12_QUERY_HEAP_DESC ph = {0};
            ph.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; ph.Count = ENG_FRAMES * (ENG_PASS_MAX + 1u);
            if (SUCCEEDED(CALL(s_dev, CreateQueryHeap, &ph, &IID_ID3D12QueryHeap, (void**)&s_pheap)))
                s_pread = make_buffer(D3D12_HEAP_TYPE_READBACK, (u64)ENG_FRAMES * (ENG_PASS_MAX + 1u) * 8u, D3D12_RESOURCE_STATE_COPY_DEST);
            if (!s_pread) RELEASE(s_pheap);
        }
    } }
    if (eng_make_root_signatures() != 0) return -1;
    const int pack_old = (getenv("RSX_PACK_FIX") && getenv("RSX_PACK_FIX")[0] == '0');   /* holo124: 0 = the old depth-as-colour pack (d = 1.0 -> 0) */
    s_snap_copy_ps = eng_compile(kSnapCopyHLSL, sizeof kSnapCopyHLSL - 1, "main", "ps_5_0", "snapshot copy ps");
    {
        ID3DBlob* vs = eng_compile(kHelperHLSL, sizeof kHelperHLSL - 1, "vs_main", "vs_5_0", "helper vs");
        ID3DBlob* pb = eng_compile(kHelperHLSL, sizeof kHelperHLSL - 1, "ps_blit", "ps_5_0", "helper blit");
        ID3DBlob* pd = eng_compile(kHelperHLSL, sizeof kHelperHLSL - 1, "ps_depth", "ps_5_0", "helper depth");
        ID3DBlob* pp = eng_compile(kHelperHLSL, sizeof kHelperHLSL - 1, pack_old ? "ps_pack_old" : "ps_pack", "ps_5_0", "helper pack");
        if (vs && pb) s_blit_pso = eng_helper_pso(vs, pb, DXGI_FORMAT_R8G8B8A8_UNORM);
        { ID3DBlob* pvs = eng_compile(kPresentHLSL, sizeof kPresentHLSL - 1, "vs_main", "vs_5_0", "present vs");
          ID3DBlob* pa = eng_compile(kPresentHLSL, sizeof kPresentHLSL - 1, "ps_blit_area", "ps_5_0", "present blit");
          ID3DBlob* pf = eng_compile(kPresentHLSL, sizeof kPresentHLSL - 1, "ps_fxaa", "ps_5_0", "present fxaa");
          if (pvs && pa) s_present_pso = eng_helper_pso(pvs, pa, DXGI_FORMAT_R8G8B8A8_UNORM);
          if (pvs && pf) s_fxaa_pso = eng_helper_pso(pvs, pf, DXGI_FORMAT_R8G8B8A8_UNORM);
          RELEASE(pvs); RELEASE(pa); RELEASE(pf); }
        if (vs && pd) s_depth_pso = eng_helper_pso(vs, pd, DXGI_FORMAT_R32_FLOAT);
        if (vs && pp) s_depth_pack_pso = eng_helper_pso(vs, pp, DXGI_FORMAT_R8G8B8A8_UNORM);
        { ID3DBlob* pdm = eng_compile(kHelperHLSL, sizeof kHelperHLSL - 1, "ps_depth_ms", "ps_5_0", "helper depth ms");
          ID3DBlob* ppm = eng_compile(kHelperHLSL, sizeof kHelperHLSL - 1, pack_old ? "ps_pack_ms_old" : "ps_pack_ms", "ps_5_0", "helper pack ms");
          if (vs && pdm) s_depth_ms_pso = eng_helper_pso(vs, pdm, DXGI_FORMAT_R32_FLOAT);
          if (vs && ppm) s_depth_pack_ms_pso = eng_helper_pso(vs, ppm, DXGI_FORMAT_R8G8B8A8_UNORM);
          ID3DBlob* pz = eng_compile(kHelperHLSL, sizeof kHelperHLSL - 1, "ps_zcopy_ms", "ps_5_0", "helper zcopy ms");
          if (vs && pz) {
              D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {0};
              pd.pRootSignature = s_helper_rootsig;
              pd.VS.pShaderBytecode = CALL0(vs, GetBufferPointer); pd.VS.BytecodeLength = CALL0(vs, GetBufferSize);
              pd.PS.pShaderBytecode = CALL0(pz, GetBufferPointer); pd.PS.BytecodeLength = CALL0(pz, GetBufferSize);
              pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
              pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
              pd.SampleMask = UINT_MAX;
              pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
              pd.NumRenderTargets = 0;
              pd.DSVFormat = ENG_DEPTH_FMT;
              pd.DepthStencilState.DepthEnable = TRUE;
              pd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
              pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
              pd.SampleDesc.Count = 1;
              if (FAILED(CALL(s_dev, CreateGraphicsPipelineState, &pd, &IID_ID3D12PipelineState, (void**)&s_zcopy_ms_pso)))
                  s_zcopy_ms_pso = NULL;
          }
          RELEASE(pdm); RELEASE(ppm); RELEASE(pz); }
        s_helper_vs_blob = vs; s_helper_blit_blob = pb;   /* kept */
        RELEASE(pd); RELEASE(pp);
        if (!s_blit_pso) return -1;
    }
    if (s_headless) {
        s_offscreen = make_texture(DXGI_FORMAT_R8G8B8A8_UNORM, width, height, 1, 1,
                                   D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, NULL);
        if (!s_offscreen) return -1;
        s_backbuf_rtv_heap = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, 0);
        CALL(s_dev, CreateRenderTargetView, s_offscreen, NULL, cpu_handle(s_backbuf_rtv_heap, s_rtv_step, 0));
    }
    return 0;
}

/* ---- the engine backend: lifecycle ----------------------------------------- */

static int eng_init(void* user, u32 width, u32 height)
{
    (void)user; (void)width; (void)height;
    return s_dev ? 0 : -1;
}

static void eng_shutdown(void* user)
{
    ENG_JOIN("shutdown"); s_zown_gen++;
    (void)user;
    rtheap_spare_stop();   /* RSX_RT_HEAP_SPARE: the maker thread must be gone before the device goes */
    if (s_queue && s_fence) fence_wait(fence_signal());
    eng_pc_save(1);
    while (s_doom_n) eng_reap_drain(1e9);   /* rissa128 RSX_REAP: everything still queued */
    for (u32 i = 0; i < s_obj_count; i++) { RELEASE(s_obj[i].res); memset(&s_obj[i], 0, sizeof s_obj[i]); }
    s_obj_count = s_obj_free_count = s_retired_count = 0; s_rel_pn = 0;
    eng_dpool_flush_all();   /* depthpool: before their DS heaps (the objects above are gone) */
    for (u32 i = 0; i < s_snap_pool_n; i++) eng_snap_pool_free_entry(i);   /* RSX_DS_HEAP: placed pooled snapshots go before their heaps */
    s_snap_pool_n = 0;
    for (u32 i = 0; i < s_ntheap; i++) { RELEASE(s_theap[i].heap); free(s_theap[i].m.fr); memset(&s_theap[i], 0, sizeof s_theap[i]); }
    s_ntheap = 0;
    for (u32 i = 0; i < s_nrth; i++) { RELEASE(s_rth[i].heap); free(s_rth[i].m.fr); memset(&s_rth[i], 0, sizeof s_rth[i]); }   /* RSX_RT_HEAP: every object was released above */
    s_nrth = 0; s_rth_ring_n = s_rth_ring_pos = 0; memset(&s_rts, 0, sizeof s_rts);
    for (u32 i = 0; i < s_ndsh; i++) { RELEASE(s_dsh[i].heap); free(s_dsh[i].m.fr); memset(&s_dsh[i], 0, sizeof s_dsh[i]); }   /* RSX_DS_HEAP: objects and pool entries were released above */
    s_ndsh = 0; s_dsh_ring_n = s_dsh_ring_pos = 0; memset(&s_dss, 0, sizeof s_dss);
    for (u32 i = 0; i < s_pipe_count; i++) {
        RELEASE(s_pipe[i].vs); RELEASE(s_pipe[i].ps);
        for (int k = 0; k < 3; k++) RELEASE(s_pipe[i].pso[k]);
    }
    s_pipe_count = 0;
    for (u32 i = 0; i < s_blob_count; i++) RELEASE(s_blob[i].blob);
    memset(s_blob_ht, 0, sizeof s_blob_ht); memset(s_samp_ht, 0, sizeof s_samp_ht);
    s_blob_count = s_samp_count = s_view_count = 0; eng_rec_reset(); s_rt_mark = ~0u; s_rt_defer_next = 0;
    for (u32 i = 0; i < s_buf_count; i++) { if (s_buf_page[i]) s_buf[i] = NULL; else RELEASE(s_buf[i]); }
    memset(s_buf_page, 0, sizeof s_buf_page);
    for (u32 i = 0; i < VB_PAGES; i++) {
        if (s_vpage[i].res) { CALL(s_vpage[i].res, Unmap, 0, NULL); RELEASE(s_vpage[i].res); }
        free(s_vpage[i].m.fr); memset(&s_vpage[i], 0, sizeof s_vpage[i]);
    }
    for (u32 i = 0; i < s_buf_pool_n; i++) RELEASE(s_buf_pool[i].res);
    s_buf_pool_n = 0; s_buf_pool_bytes = 0;
    s_buf_count = s_buf_free_count = 0; memset(s_buf_retired, 0, sizeof s_buf_retired);
    for (int i = 0; i < ENG_STAGE_POOL; i++) eng_stage_free(&s_stage[i]);
    s_chain_n = 0;
    for (int i = 0; i < ENG_MAX_SUBMITS; i++) { free(s_sub[i].q_vis); free(s_sub[i].reports); memset(&s_sub[i], 0, sizeof s_sub[i]); }
    LB->vs_cb_seq = ~0u; s_vis_npending = 0; s_fallback_n = 0;
    for (u32 i = 0; i < s_snap_pool_n; i++) RELEASE(s_snap_pool[i].res);
    s_snap_pool_n = 0;
    s_active = 0;
}

/* ---- resources ------------------------------------------------------------- */

static u32 eng_texture_create(void* user, rsx_be_format fmt, u32 w, u32 h,
                              u32 mips, u32 faces, u32 remap, u32 rsx_fmt)
{
    ENG_JOIN("tex_create");
    (void)user;
    if (!s_dev || !w || !h) return 0;
    const DXGI_FORMAT df = eng_dxgi(fmt);
    u32 heap = 0; u64 heap_off = 0, heap_size = 0;
    ID3D12Resource* t = make_placed_texture(df, w, h, mips, faces, D3D12_RESOURCE_STATE_COPY_DEST, &heap, &heap_off, &heap_size);
    if (!t) t = make_texture(df, w, h, mips, faces, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, NULL);
    s_stat_tex++;
    const u32 handle = eng_obj_add(t, OBJ_TEXTURE, df, w, h, mips, faces, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!handle) { if (t) theap_done(heap, heap_off, heap_size); return 0; }   /* eng_obj_add released it */
    s_obj[handle - 1].heap = heap; s_obj[handle - 1].heap_off = heap_off; s_obj[handle - 1].heap_size = heap_size;
    eng_write_srv(handle, t, df, mips ? mips : 1, faces, eng_mapping(remap, rsx_fmt));
    return handle;
}

/* Rows go into the stage; the copy is a record, ordered with the draws. */
static void eng_upload_rows(u32 handle, u32 sub, u32 w, u32 h, const void* src, u32 row_bytes, u32 rows)
{
    if (s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return; }
    const u32 pitch = (row_bytes + 255u) & ~255u;
    u32 off;
    if (!eng_stage_reserve(pitch * rows, &off)) return;
    /* The reserve may have submitted, so the pointer is taken after it. */
    u8* dst = eng_stage_ptr(off);
    if (pitch == row_bytes) memcpy(dst, src, (size_t)row_bytes * rows);
    else for (u32 y = 0; y < rows; y++) memcpy(dst + (size_t)y * pitch, (const u8*)src + (size_t)y * row_bytes, row_bytes);
    EngRecord* r = eng_rec_new();
    memset(r, 0, sizeof *r);
    r->kind = ENG_REC_UPLOAD;
    r->depth = handle;
    r->up_off = off; r->up_sub = sub; r->up_w = w; r->up_h = h; r->up_pitch = pitch;
}

static void eng_texture_upload(void* user, u32 handle, u32 face, u32 mip,
                               u32 w, u32 h, const void* src, u32 row_bytes, u32 rows)
{
    (void)user;
    EngObj* o = eng_obj(handle);
    if (!o || !o->res || !src || !row_bytes || !rows) return;
    eng_upload_rows(handle, face * o->mips + mip, w, h, src, row_bytes, rows);
}

/* RSX_RT_HEAP_NOJOIN=1 (default 0; needs RSX_RT_HEAP>=1 and RSX_JOIN_LITE=1): an unseeded colour target does not join the encode thread (same argument as the colour snapshot under JOIN_LITE) */
static int rt_nojoin_on(void)
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("RSX_RT_HEAP_NOJOIN"); v = (e && e[0] == '1') ? 1 : 0; }
    return v && rt_heap_mode() && join_lite_on();
}
static u32 eng_color_target_create(void* user, rsx_be_format fmt, u32 w, u32 h,
                                   const void* seed, u32 seed_row_bytes)
{
    const int ph_on = rt_heap_mode() != 0;
    const double ph0 = ph_on ? rt_now() : 0.0;
    const int nojoin = !seed && rt_nojoin_on();
    if (!nojoin) ENG_JOIN("color_target_create");
    const double ph1 = ph_on ? rt_now() : 0.0;
    (void)user;
    if (!s_dev || !w || !h) return 0;
    eng_mp_count(0);
    const DXGI_FORMAT df = eng_dxgi(fmt);
    /* 26/10 gfx-quick: shadow-map colour targets and the bloom chain get an extra multiplier. */
    float xk = 0.0f; int xbloom = 0;
    { const int ks = eng_shadow_k(w, h), kb = eng_bloom_k(w, h);
      if (ks > 1) xk = (float)ks; else if (kb > 1) { xk = (float)kb; xbloom = 1; }
      if (xk > 0.0f) { static int n = 0; if (n++ < 12) fprintf(stderr, "[RSX d3d12] %s target %ux%u -> x%g extra\n", xbloom ? "bloom" : "shadow", w, h, (double)xk); } }
    const int scaled = xk > 0.0f || eng_scales(w, h);
    const float cf = xk > 0.0f ? (xbloom ? (s_scale > 1.0f ? s_scale : 1.0f) * xk : s_scale * xk) : s_scale;
    /* 02/11 aa110: the 960x704 scene target gets 4/3 more along x only, if the canvas-sized depth it draws into is wide enough. */
    const float xxk = (xk == 0.0f && scaled && w == 960 && h == 704 && eng_hstretch_on() && (u32)(960.0f * cf * ENG_HSTRETCH_K + 0.5f) <= (u32)((float)s_width * s_scale + 0.5f)) ? ENG_HSTRETCH_K : 0.0f;
    const u32 sw = scaled ? scf_dim(w, xxk > 0.0f ? cf * xxk : cf) : w, sh = scaled ? scf_dim(h, cf) : h;
    if (xxk > 0.0f) { static int n = 0; if (n++ < 4) fprintf(stderr, "[RSX d3d12] hstretch: scene target %ux%u -> host %ux%u (x extra %.4g, RSX_HSTRETCH_FIX=0 = off)" "\n", w, h, sw, sh, (double)xxk); }
    D3D12_CLEAR_VALUE cv = {0}; cv.Format = df;
    /* RSX_RT_HEAP: placed in an RT heap when it can be (needs room for its zero-clear record, and the seed's upload record, in the array). */
    const int rtm = rt_heap_mode();
    u32 rt_heap = 0; u64 rt_off = 0, rt_size = 0;
    ID3D12Resource* t = NULL;
    const double rt_t0 = rtm ? rt_now() : 0.0;
    if (rtm) {
        if (s_rec_count + 8 >= ENG_MAX_RECORDS) s_rts.fb_rec++;
        else t = make_placed_rt(df, sw, sh, &cv, &rt_heap, &rt_off, &rt_size);
    }
    if (!t) t = make_texture(df, sw, sh, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                             D3D12_RESOURCE_STATE_RENDER_TARGET, &cv);
    if (rtm) {
        const double now = rt_now(), d = now - rt_t0;
        if (rt_heap) { s_rts.placed++; s_rts.w_placed++; s_rts.w_placed_ms += d; }
        else { s_rts.fb_total++; s_rts.w_comm++; s_rts.w_comm_ms += d; }
        rtheap_log(now);
    }
    const double ph2 = ph_on ? rt_now() : 0.0;
    const u32 handle = eng_obj_add(t, OBJ_COLOR, df, sw, sh, 1, 1, D3D12_RESOURCE_STATE_RENDER_TARGET);
    s_stat_rt++;
    if (!handle) { if (rt_heap) rtheap_done(rt_heap, rt_off, rt_size); return 0; }   /* eng_obj_add released the resource */
    s_obj[handle - 1].gw = w; s_obj[handle - 1].gh = h;
    s_obj[handle - 1].xk = xk; s_obj[handle - 1].xbloom = xbloom; s_obj[handle - 1].xx = xxk;
    s_obj[handle - 1].heap = rt_heap; s_obj[handle - 1].heap_off = rt_off; s_obj[handle - 1].heap_size = rt_size;
    eng_write_srv(handle, t, df, 1, 1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
    CALL(s_dev, CreateRenderTargetView, t, NULL, obj_rtv(handle));
    s_obj[handle - 1].has_rtv = 1;
    const double ph3 = ph_on ? rt_now() : 0.0;
    if (rt_heap) eng_rt_zero_record(handle);   /* before the seed's upload record: a placed target is not zero */
    const double ph4 = ph_on ? rt_now() : 0.0;
    if (seed && seed_row_bytes) {
        if (sw == w && sh == h) eng_upload_rows(handle, 0, w, h, seed, seed_row_bytes, h);
        else {
            /* The guest's bytes, resampled (nearest) to the scaled target. */
            const u32 bpp = eng_bpp(df);
            u8* big = (u8*)malloc((size_t)sw * bpp * sh);
            if (big) {
                for (u32 y = 0; y < sh; y++) {
                    const u8* srow = (const u8*)seed + (size_t)((u64)y * h / sh) * seed_row_bytes;
                    u8* drow = big + (size_t)y * sw * bpp;
                    for (u32 x = 0; x < sw; x++)
                        memcpy(drow + (size_t)x * bpp, srow + (size_t)((u64)x * w / sw) * bpp, bpp);
                }
                eng_upload_rows(handle, 0, sw, sh, big, sw * bpp, sh);
                free(big);
            }
        }
    }
    if (ph_on) {
        const double ph5 = rt_now();
        s_rts.ph_n++; s_rts.ph_nojoin += nojoin;
        s_rts.ph_join += ph1 - ph0; s_rts.ph_alloc += ph2 - ph1; s_rts.ph_desc += ph3 - ph2; s_rts.ph_rec += ph4 - ph3; s_rts.ph_seed += ph5 - ph4;
        if (ph5 - ph0 > s_rts.ph_max) s_rts.ph_max = ph5 - ph0;
    }
    return handle;
}

/* 03/11 encjoin: RSX_SURFVIEW_LAZYJOIN=1 looks the view up BEFORE the join. s_view[], s_view_count and the surface's o->res are
 * written only by walker-thread entries that join first (eng_obj_release, eng_rescale, eng_collect_retired, creation below); the
 * encode thread never writes them (it only reads o->res; eng_aa_target there releases the host-only AA object, which has no view). */
static int eng_surfview_lazy(void)
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("RSX_SURFVIEW_LAZYJOIN"); v = (e && e[0] && e[0] != '0') ? 1 : 0; }
    return v;
}

static u32 eng_surface_view(void* user, u32 surface, u32 remap, u32 rsx_format)
{
    if (eng_surfview_lazy()) {
        EngObj* lo = eng_obj(surface);
        if (lo && lo->res)
            for (u32 i = 0; i < s_view_count; i++)
                if (s_view[i].surface == surface && s_view[i].remap == remap && s_view[i].format == rsx_format)
                    return s_view[i].view;
    }
    ENG_JOIN("surface_view");
    (void)user;
    EngObj* o = eng_obj(surface);
    if (!o || !o->res) return 0;
    for (u32 i = 0; i < s_view_count; i++)
        if (s_view[i].surface == surface && s_view[i].remap == remap && s_view[i].format == rsx_format)
            return s_view[i].view;
    if (LB->spec) rsx_lane_abort(RSX_LAB_VIEWNEW);   /* lane109: a new view is a shared write */
    if (s_view_count >= ENG_MAX_VIEWS) return 0;
    const u32 handle = eng_obj_add(NULL, OBJ_VIEW, o->fmt, o->w, o->h, 1, 1, o->state);
    if (!handle) return 0;
    s_obj[handle - 1].alias = surface;
    eng_write_srv(handle, o->res, o->fmt, 1, 1, eng_mapping(remap, rsx_format));
    s_view[s_view_count].surface = surface; s_view[s_view_count].remap = remap;
    s_view[s_view_count].format = rsx_format; s_view[s_view_count].view = handle;
    s_view_count++;
    return handle;
}

/* lane111 RSX_LANES_ZOWN=1 (needs RSX_JOIN_LITE=1 and RSX_SURFVIEW_LAZYJOIN=1): the packed copy of the zeta the running pass writes, as a RECORD with an object made ahead by the coordinator
 * (rsx_d3d12_zown_make, no record, no join) instead of an object created inside the lane (which aborted the chunk: depth-packed-own-zeta). */
u32 rsx_d3d12_zown_gen(void) { return s_zown_gen; }
static u32 eng_depth_snapshot_make_fwd(u32 depth, u32 w, u32 h, int packed, int emit);
u32 rsx_d3d12_zown_make(u32 depth, u32 w, u32 h, u32 remap, u32 rsx_format, u32* view_out)
{
    if (!join_lite_on() || !eng_surfview_lazy()) return 0;
    const u32 dst = eng_depth_snapshot_make_fwd(depth, w, h, 1, 0);
    if (!dst) return 0;
    EngObj* o = eng_obj(dst);
    if (!o || !o->res || s_view_count >= ENG_MAX_VIEWS) { eng_obj_release(NULL, dst); return 0; }
    const u32 handle = eng_obj_add(NULL, OBJ_VIEW, o->fmt, o->w, o->h, 1, 1, o->state);
    if (!handle) { eng_obj_release(NULL, dst); return 0; }
    s_obj[handle - 1].alias = dst;
    eng_write_srv(handle, o->res, o->fmt, 1, 1, eng_mapping(remap, rsx_format));
    s_view[s_view_count].surface = dst; s_view[s_view_count].remap = remap; s_view[s_view_count].format = rsx_format; s_view[s_view_count].view = handle;
    _ReadWriteBarrier();
    s_view_count++;
    *view_out = handle;
    return dst;
}
void rsx_d3d12_zown_record(u32 depth, u32 dst)
{
    EngRecord* rec = eng_rec_new();
    memset(rec, 0, sizeof *rec);
    rec->kind = ENG_REC_DEPTH_RESOLVE;
    rec->depth = depth;
    rec->resolve_dst = dst;
    rec->resolve_packed = 1;
}

static u32 eng_depth_target_create_host(u32 w, u32 h, u32 gw, u32 gh)
{
    if (!s_dev || !w || !h) return 0;
    eng_mp_count(1);
    D3D12_CLEAR_VALUE cv = {0}; cv.Format = ENG_DEPTH_FMT; cv.DepthStencil.Depth = 1.0f;
    /* RSX_DS_HEAP: placed in a DS heap when it can be. Not on the encode thread (the fallback depth: a record cannot be added there) and only with room for the zero-clear record. */
    u32 ds_heap = 0; u64 ds_off = 0, ds_size = 0;
    ID3D12Resource* t = NULL;
    /* depthpool RSX_DEPTH_POOL: a parked depth of this host size, with a zero/clear record ahead of its use (so only where a record can be added: not the encode thread) */
    int reused = 0; D3D12_RESOURCE_STATES pst = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    if (dpool_on() && !t_on_enc && !t_no_place && s_rec_count + 4 < ENG_MAX_RECORDS)
        reused = eng_dpool_take(w, h, &t, &pst, &ds_heap, &ds_off, &ds_size);
    const int dsm = !reused && ds_heap_mode() && !t_on_enc && !t_no_place;
    const double ds_t0 = dsm ? rt_now() : 0.0;
    if (dsm) {
        if (s_rec_count + 4 >= ENG_MAX_RECORDS) InterlockedIncrement64((volatile LONG64*)&s_dss.fb_rec);
        else t = make_placed_ds(0, ENG_DEPTH_RES_FMT, w, h, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE, &cv, &ds_heap, &ds_off, &ds_size);
    }
    if (!t) t = make_texture(ENG_DEPTH_RES_FMT, w, h, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                             D3D12_RESOURCE_STATE_DEPTH_WRITE, &cv);
    if (dsm) ds_note(0, ds_heap != 0, rt_now() - ds_t0);
    const u32 handle = eng_obj_add(t, OBJ_DEPTH, ENG_DEPTH_FMT, w, h, 1, 1, reused ? pst : D3D12_RESOURCE_STATE_DEPTH_WRITE);
    if (!handle) { if (ds_heap) dsheap_done(ds_heap, ds_off, ds_size); return 0; }   /* eng_obj_add released the resource */
    s_obj[handle - 1].gw = gw; s_obj[handle - 1].gh = gh;
    s_obj[handle - 1].heap = ds_heap; s_obj[handle - 1].heap_off = ds_off; s_obj[handle - 1].heap_size = ds_size;
    D3D12_DEPTH_STENCIL_VIEW_DESC dd = {0};
    dd.Format = ENG_DEPTH_FMT; dd.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    CALL(s_dev, CreateDepthStencilView, t, &dd, obj_dsv(handle));
    s_obj[handle - 1].has_dsv = 1;
    /* Sampled by the resolve passes as its depth plane. */
    eng_write_srv(handle, t, ENG_DEPTH_SRV_FMT, 1, 1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
    if (reused) eng_ds_clear_record(handle);   /* depthpool: a reused zeta holds the old frame */
    else if (ds_heap) eng_ds_zero_record(handle);   /* a placed zeta is not zero */
    return handle;
}
static u32 eng_depth_target_create(void* user, u32 w, u32 h)
{
    ENG_JOIN("depth_target_create");
    (void)user;
    if (!eng_scales(w, h)) return eng_depth_target_create_host(w, h, w, h);
    return eng_depth_target_create_host(sc_dim(w), sc_dim(h), w, h);
}

/* A depth target for a draw pass whose guest never declared a zeta: every
 * pipeline declares the depth format, so a pass always needs one bound. */
/* vram126 RSX_FBDEPTH_LRU=1 (default off): with 8 sizes cached, every further size made a new depth target on EVERY call and
 * kept it nowhere -- never released. ToD's depthless passes change size all the time: at Rykan V 4K MSAA x8 3039 such targets
 * (8.5 GB with their MSAA twins, site eng_depth_target_create_host caller "-:0") piled up in 2.5 minutes (log v2 06/11 zovl_off).
 * Now the least recently used entry is replaced. This runs on the encode thread, whose list may still name the old target:
 * it is queued and released on the walker after the join (eng_fb_victims_release), retired one submit later than usual. */
static u32 s_enc_frame;   /* defined below (4541): tentative */
static u32 s_fallback_last[8];
static u32 s_fb_victim[64]; static volatile u32 s_fb_victim_n; static unsigned long long s_fb_victim_lost;
static int fbdepth_lru_on(void) { static int v = -1; if (v < 0) { const char* e = getenv("RSX_FBDEPTH_LRU"); v = e && e[0] == '1'; } return v; }
static u32 eng_fallback_depth(u32 w, u32 h)
{
    for (u32 i = 0; i < s_fallback_n; i++) {
        EngObj* o = eng_obj(s_fallback_depth[i]);
        if (o && o->w == w && o->h == h) { s_fallback_last[i] = s_enc_frame; return s_fallback_depth[i]; }
    }
    t_no_place++;
    const u32 d = eng_depth_target_create_host(w, h, w, h);   /* w, h are host pixels here */
    t_no_place--;
    if (d && s_fallback_n < 8) { s_fallback_last[s_fallback_n] = s_enc_frame; s_fallback_depth[s_fallback_n++] = d; }
    else if (d && fbdepth_lru_on()) {
        u32 v = 0;
        for (u32 i = 1; i < s_fallback_n; i++) if (s_fallback_last[i] < s_fallback_last[v]) v = i;
        if (s_fb_victim_n < 64) s_fb_victim[s_fb_victim_n++] = s_fallback_depth[v];
        else s_fb_victim_lost++;   /* full queue: that one leaks, as before */
        s_fallback_depth[v] = d; s_fallback_last[v] = s_enc_frame;
        { static unsigned nl; if (nl++ < 8) fprintf(stderr, "[fbdepth] cache full: %ux%u replaces entry %u (queued for release)\n", w, h, v); }
    }
    return d;
}
/* Walker side, after the join (the encode thread is idle): release the replaced fallback depth targets. */
static void eng_fb_victims_release(void)
{
    if (!s_fb_victim_n || t_on_enc || s_enc_pending) return;
    const u32 n = s_fb_victim_n; s_fb_victim_n = 0;
    for (u32 i = 0; i < n; i++) eng_obj_release_do(s_fb_victim[i], 1);
    { static unsigned long long tot; tot += n; static unsigned nl; if (nl++ < 4 || (nl & 1023u) == 0) fprintf(stderr, "[fbdepth] released %u replaced fallback depth targets (%llu so far, %llu lost)\n", n, tot, s_fb_victim_lost); }
}

/* Copy of a colour target as it stands at this point in the stream. One copy
 * texture per source is kept and refreshed by each request. */
static u32 eng_color_snapshot(void* user, u32 surface)
{
    if (!join_lite_on()) ENG_JOIN("color_snapshot");
    (void)user;
    EngObj* t = eng_obj(surface);
    if (!t || !t->res) return 0;
    if (s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return 0; }
    static struct { u32 surface, snap; } pool[32]; static u32 npool;
    u32 dst = 0, slot = npool;
    for (u32 i = 0; i < npool; i++) if (pool[i].surface == surface) { slot = i; break; }
    if (slot < npool) {
        EngObj* d = eng_obj(pool[slot].snap);
        if (d && d->res && d->w == t->w && d->h == t->h && d->fmt == t->fmt) dst = pool[slot].snap;
    }
    if (!dst) {
        /* A render target as well, so the snapshot can be brought up to date
         * by drawing into it (eng_encode_copy_draw) rather than copied whole. */
        ID3D12Resource* r = make_texture(t->fmt, t->w, t->h, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                         D3D12_RESOURCE_STATE_COPY_DEST, NULL);
        dst = eng_obj_add(r, OBJ_SNAPSHOT, t->fmt, t->w, t->h, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!dst) return 0;
        s_obj[dst - 1].gw = t->gw; s_obj[dst - 1].gh = t->gh;
        eng_write_srv(dst, r, t->fmt, 1, 1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
        CALL(s_dev, CreateRenderTargetView, r, NULL, obj_rtv(dst));
        s_obj[dst - 1].has_rtv = 1;
        if (slot == npool) { if (npool >= 32) return 0; npool++; }
        pool[slot].surface = surface; pool[slot].snap = dst;
    }
    EngRecord* r = eng_rec_new();
    memset(r, 0, sizeof *r);
    r->kind = ENG_REC_COLOR_COPY;
    r->depth = surface;
    r->resolve_dst = dst;
    return dst;
}

/* An NV3089 blit out of a colour target (eng_color_copy_rect): recorded, and
 * drawn at encode time by the helper blit with its viewport placed so the
 * source's (sx,sy) lands on the destination's (dx,dy) -- each target at its
 * own scale, so a scaled scene copies into an unscaled 32x34 texture. */
static int eng_color_copy_rect(void* user, u32 src, u32 sx, u32 sy, u32 sw, u32 sh,
                               u32 dst, u32 dx, u32 dy, u32 dw, u32 dh)
{
    ENG_JOIN("color_copy_rect");
    (void)user;
    EngObj* s = eng_obj(src); EngObj* d = eng_obj(dst);
    if (!s || !s->res || !d || !d->res || !d->has_rtv || !s_blit_pso || !sw || !sh || !dw || !dh) return 0;
    if (s->fmt != DXGI_FORMAT_R8G8B8A8_UNORM || d->fmt != DXGI_FORMAT_R8G8B8A8_UNORM) return 0;
    if (s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return 0; }
    EngRecord* r = eng_rec_new();
    memset(r, 0, sizeof *r);
    r->kind = ENG_REC_COLOR_RECT;
    r->depth = src; r->resolve_dst = dst;
    r->clear_rect[0] = sx; r->clear_rect[1] = sy; r->clear_rect[2] = sw; r->clear_rect[3] = sh;
    r->sc[0] = dx; r->sc[1] = dy; r->sc[2] = dw; r->sc[3] = dh;
    return 1;
}

static int cut88_uniq(const u32* v, int n);
static u32 eng_depth_snapshot_make(u32 depth, u32 w, u32 h, int packed, int emit);
static u32 eng_depth_snapshot_common(u32 depth, u32 w, u32 h, int packed) { return eng_depth_snapshot_make(depth, w, h, packed, 1); }
static u32 eng_depth_snapshot_make_fwd(u32 depth, u32 w, u32 h, int packed, int emit) { return eng_depth_snapshot_make(depth, w, h, packed, emit); }
static u32 eng_depth_snapshot_make(u32 depth, u32 w, u32 h, int packed, int emit)
{
    if (!join_lite_on()) ENG_JOIN("depth_snapshot");
    EngObj* z = eng_obj(depth);
    if (!z || !z->res || !(packed ? s_depth_pack_pso : s_depth_pso)) return 0;
    if (emit && s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return 0; }
    const u32 gw = w, gh = h;
    { u32 v[8] = { w, h, z->w, z->h, (u32)(z->pf * 100.f), packed, 0, 0 }; if (cut88_uniq(v, 8)) fprintf(stderr, "[cut88 snap] request guest %ux%u, depth host %ux%u pf=%g packed=%d scaled=%d\n", w, h, z->w, z->h, (double)z->pf, packed, (int)eng_obj_scaled(depth)); }
    if (z->pf > 0.0f) { w = scf_dim(w, z->pf); h = scf_dim(h, z->pf); }   /* a shadow map drawn at RSX_SHADOW_SCALE */
    else if (z->xx > 0.0f && gw == 960 && gh == 704 && eng_obj_scaled(depth)) { w = scf_dim(w, s_scale * z->xx); h = sc_dim(h); }   /* 02/11 aa110: the 960x704 scene drawn 4/3 wider */
    else if (eng_obj_scaled(depth)) { w = sc_dim(w); h = sc_dim(h); }
    const DXGI_FORMAT df = packed ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_R32_FLOAT;
    ID3D12Resource* r = NULL;
    D3D12_RESOURCE_STATES st = D3D12_RESOURCE_STATE_RENDER_TARGET;
    u32 sn_heap = 0; u64 sn_off = 0, sn_size = 0; int sn_fresh = 0;   /* RSX_DS_HEAP: the range of a placed snapshot (from the pool, or new and then needing its init) */
    for (u32 i = 0; i < s_snap_pool_n; i++) {
        if (s_snap_pool[i].fmt != df || s_snap_pool[i].w != w || s_snap_pool[i].h != h) continue;
        r = s_snap_pool[i].res;
        st = s_snap_pool[i].state;
        sn_heap = s_snap_pool[i].heap; sn_off = s_snap_pool[i].heap_off; sn_size = s_snap_pool[i].heap_size;
        s_snap_pool[i] = s_snap_pool[--s_snap_pool_n];
        s_stat_snap_reused++; s_v2_snap_hit++;
        break;
    }
    if (!r) {
        D3D12_CLEAR_VALUE cv = {0}; cv.Format = df;
        LARGE_INTEGER t0, t1, fq; QueryPerformanceCounter(&t0);
        const double mk0 = reap_now();
        const int dsm = ds_heap_mode() && !t_on_enc;
        if (dsm) {
            r = make_placed_ds(1, df, w, h, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, &cv, &sn_heap, &sn_off, &sn_size);
            sn_fresh = r != NULL;
        }
        if (!r) r = make_texture(df, w, h, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                 D3D12_RESOURCE_STATE_RENDER_TARGET, &cv);
        if (dsm) ds_note(1, sn_fresh, reap_now() - mk0);
        s_stat_snap++; s_v2_snap_new++; s_v2_snap_ms += reap_now() - mk0; if ((u64)w * h >= (1u << 20)) s_v2_snap_big++;
        { static int tr = -1; static unsigned nt; if (tr < 0) { const char* e = getenv("RSX_SNAP_TRACE"); tr = (e && e[0] && e[0] != '0'); }
          if (tr && nt < 400) {   /* walk36: what is created, how long it takes, and whether the pool had that size at all */
              QueryPerformanceCounter(&t1); QueryPerformanceFrequency(&fq);
              u32 same = 0; for (u32 i = 0; i < s_snap_pool_n; i++) if (s_snap_pool[i].fmt == df && s_snap_pool[i].w == w && s_snap_pool[i].h == h) same++;
              fprintf(stderr, "[snap-new] %ux%u %s %.2f ms, pool %u (same size %u) guest %ux%u\n", w, h, packed ? "rgba8" : "r32f",
                      (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)fq.QuadPart, s_snap_pool_n, same, gw, gh);
              nt++; } }
    }
    const u32 dst = eng_obj_add(r, OBJ_SNAPSHOT, df, w, h, 1, 1, st);
    if (!dst) { theap_done(sn_heap, sn_off, sn_size); return 0; }   /* eng_obj_add released the resource (a placed one: its range goes back; theap_done(0) is a no-op) */
    s_obj[dst - 1].gw = gw; s_obj[dst - 1].gh = gh;
    s_obj[dst - 1].heap = sn_heap; s_obj[dst - 1].heap_off = sn_off; s_obj[dst - 1].heap_size = sn_size;
    s_obj[dst - 1].dsinit = sn_fresh;   /* the first DEPTH_RESOLVE into it does the aliasing barrier + zero clear */
    eng_write_srv(dst, r, df, 1, 1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
    CALL(s_dev, CreateRenderTargetView, r, NULL, obj_rtv(dst));
    s_obj[dst - 1].has_rtv = 1;
    if (!emit) return dst;   /* lane111 RSX_LANES_ZOWN: the object only; the lane that uses it writes the record (rsx_d3d12_zown_record) */
    EngRecord* rec = eng_rec_new();
    memset(rec, 0, sizeof *rec);
    rec->kind = ENG_REC_DEPTH_RESOLVE;
    rec->depth = depth;
    rec->resolve_dst = dst;
    rec->resolve_packed = packed;
    return dst;
}
static u32 eng_depth_snapshot(void* user, u32 depth, u32 w, u32 h)
{ (void)user; return eng_depth_snapshot_common(depth, w, h, 0); }
static u32 eng_depth_snapshot_rgba8(void* user, u32 depth, u32 w, u32 h)
{ (void)user; return eng_depth_snapshot_common(depth, w, h, 1); }

/* ---- occlusion queries ----------------------------------------------------- */

/* RSX_ENCODE_THREAD (walk36): the job out (the previous frame) can touch s_vis_count only for the slots of ITS queries
 * (the mid-frame flush of eng_hard_flush adds to them, synchronously), i.e. begin numbers [s_vis_job_lo, s_vis_job_hi).
 * query_begin resets the slot of begin number s_vis_next, which is the same slot only 16384 begins later: it has to
 * join only if the job's range is that long. Before, every frame's first query_begin waited for the whole encode
 * (1.7 ms per present on the draw thread). RSX_ENC_QJOIN=1 = join always (old behaviour). */
static u32 s_vis_job_lo, s_vis_job_hi, s_vis_frame_lo;
static u32 eng_query_begin(void* user)
{
    if (LB->spec) {   /* lane109: the slots were reserved (and their counters reset) by the coordinator, in job order */
        if (!s_qheap) return 0;
        if (LB->q_k >= LB->q_n) rsx_lane_abort(RSX_LAB_QUERY);
        return (LB->q_first + LB->q_k++) % ENG_VIS_SLOTS + 1;
    }
    if (s_enc_pending && !t_on_enc) {
        static int always = -1; if (always < 0) { const char* e = getenv("RSX_ENC_QJOIN"); always = (e && e[0] == '1'); }
        if (always || (u32)(s_vis_next + 1u - s_vis_job_lo) >= ENG_VIS_SLOTS) eng_enc_join_why("query_begin", 0);
    }
    (void)user;
    if (!s_qheap) return 0;
    const u32 slot = s_vis_next++ % ENG_VIS_SLOTS;
    s_vis_count[slot] = 0;
    return slot + 1;
}
static void eng_query_set(void* user, u32 query) { (void)user; if (LB->spec) LB->vis_cur = query; else s_vis_cur = query; }
static void eng_query_report(void* user, u32 query, u32 report_index)
{
    /* s_vis_pending is the walker's own (a job gets a copy at the hand-over): no join (RSX_ENC_QJOIN=1: old behaviour) */
    { static int always = -1; if (always < 0) { const char* e = getenv("RSX_ENC_QJOIN"); always = (e && e[0] == '1'); }
      if (always) ENG_JOIN("query_report"); }
    (void)user;
    if (!query || !s_qheap) return;
    if (LB->spec) {
        EngLaneBe* b = LB;
        if (b->vrep_n >= b->vrep_cap) {
            const u32 nc = b->vrep_cap ? b->vrep_cap * 2u : 256u;
            void* nv = realloc(b->vrep, (size_t)nc * sizeof *b->vrep);
            if (!nv) rsx_lane_abort(RSX_LAB_QUERY);
            b->vrep = nv; b->vrep_cap = nc;
        }
        b->vrep[b->vrep_n].slot = query - 1; b->vrep[b->vrep_n].index = report_index; b->vrep_n++;
        return;
    }
    if (s_vis_npending >= ENG_MAX_REPORTS) return;
    s_vis_pending[s_vis_npending].slot = query - 1;
    s_vis_pending[s_vis_npending].index = report_index;
    s_vis_npending++;
}

/* ---- pipelines ------------------------------------------------------------- */

static void dump_shader(const char* name, const void* text, size_t n)
{
    static const char* dir = (const char*)1;
    if (dir == (const char*)1) { dir = getenv("PS3RECOMP_SHADER_DUMP"); if (dir && !*dir) dir = NULL; }
    if (!dir) return;
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fwrite(text, 1, n, f);
    fclose(f);
}

/* ---- persistent shader / pipeline-state cache (RSX_PSO_CACHE=0 disables) ----
 *
 * Every run paid FXC (D3DCompile -O3: 20-300 ms a program) and the driver's
 * state-object compile (tens of ms each, seconds for a heavy effect with the
 * CPU busy): ~160 pipelines at level load, 6-9 s before a new combat effect
 * showed. Two layers under <exe dir>\cache\ (RSX_PSO_CACHE_DIR overrides;
 * PS3RECOMP_DXBC_CACHE still overrides the bytecode directory):
 *   dxbc\<what>_<key>.dxbc  compiled bytecode; key = HLSL text + stage + FXC
 *                           build; header {magic,version,size,checksum}, so
 *                           a truncated or damaged file is recompiled.
 *   pso_<gpu>.lib           ID3D12PipelineLibrary, names = hash of the full
 *                           pipeline description; the file header has the
 *                           format version and the exe's size/time (root
 *                           signature and state code come with the exe), and
 *                           the runtime itself refuses another driver/GPU.
 * Any mismatch or failure falls back to compiling, as before. */
#define PC_MAGIC_DXBC 0x43425844u   /* 'DXBC' */
#define PC_MAGIC_LIB  0x4C4F5350u   /* 'PSOL' */
#define PC_VERSION    2u   /* pipeline library; 2 (31/10 pre52): ids[0] is the codegen id, not the exe size/time. The dxbc files keep format 1 */
static INIT_ONCE s_pc_once = INIT_ONCE_STATIC_INIT;
static int s_pc_on;
static char s_pc_dir[MAX_PATH];
static u64 s_pc_fxc_id, s_pc_exe_id;
static ID3D12PipelineLibrary* s_pc_lib;
static void* s_pc_lib_mem;
static SRWLOCK s_pc_lock = SRWLOCK_INIT;
static volatile LONG s_pc_dirty;         /* states stored since the last save */
static volatile LONG s_pc_saving;
static ULONGLONG s_pc_last_save;
static volatile LONG s_pc_dxbc_hit, s_pc_dxbc_miss, s_pc_pso_hit, s_pc_pso_miss;
static volatile LONG64 s_pc_dxbc_miss_us, s_pc_pso_miss_us;

static u64 eng_file_id(const char* path)
{
    WIN32_FILE_ATTRIBUTE_DATA d;
    if (!GetFileAttributesExA(path, GetFileExInfoStandard, &d)) return 0;
    u64 h = 1469598103934665603ull;
    h = fnv1a64(&d.nFileSizeLow, 4, h); h = fnv1a64(&d.nFileSizeHigh, 4, h);
    h = fnv1a64(&d.ftLastWriteTime, sizeof d.ftLastWriteTime, h);
    return h;
}

static double eng_now_us(void)
{
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e6 / (double)f.QuadPart;
}

static BOOL CALLBACK eng_pc_init_cb(PINIT_ONCE o, PVOID a, PVOID* c)
{
    (void)o; (void)a; (void)c;
    const char* e = getenv("RSX_PSO_CACHE");
    if (e && e[0] == '0') { fprintf(stderr, "[pso-cache] disabled (RSX_PSO_CACHE=0)\n"); return TRUE; }
    const char* ov = getenv("RSX_PSO_CACHE_DIR");
    const char* shared = getenv("RSX_PSO_CACHE_SHARED");   /* pso110: one cache per user, shared by every build/version (%LOCALAPPDATA%\ToDRecomp\cache) */
    if (ov && *ov) snprintf(s_pc_dir, sizeof s_pc_dir, "%s", ov);
    else if (shared && shared[0] == '1' && getenv("LOCALAPPDATA")) {
        char base[MAX_PATH]; snprintf(base, sizeof base, "%s\\ToDRecomp", getenv("LOCALAPPDATA")); _mkdir(base);
        snprintf(s_pc_dir, sizeof s_pc_dir, "%s\\cache", base);
    } else {
        char exe[MAX_PATH]; DWORD n = GetModuleFileNameA(NULL, exe, sizeof exe);
        if (!n || n >= sizeof exe) return TRUE;
        char* sl = strrchr(exe, '\\'); if (!sl) sl = strrchr(exe, '/'); if (sl) *sl = 0;
        snprintf(s_pc_dir, sizeof s_pc_dir, "%s\\cache", exe);
    }
    _mkdir(s_pc_dir);
    char sub[MAX_PATH + 8]; snprintf(sub, sizeof sub, "%s\\dxbc", s_pc_dir); _mkdir(sub);
    { DWORD at = GetFileAttributesA(sub);
      if (at == INVALID_FILE_ATTRIBUTES || !(at & FILE_ATTRIBUTE_DIRECTORY)) {
          fprintf(stderr, "[pso-cache] cannot use '%s': disabled\n", s_pc_dir); return TRUE; } }
    /* The compiler build: its file size/time (a new FXC can emit other code). */
    { HMODULE m = LoadLibraryA("d3dcompiler_47.dll"); char mp[MAX_PATH];
      if (m && GetModuleFileNameA(m, mp, sizeof mp)) s_pc_fxc_id = eng_file_id(mp); }
#ifdef RSX_CODEGEN_ID
    /* 31/10 (pre52): the library is valid for any exe that builds pipelines the same way (the runtime itself
     * refuses a state whose description differs): a rebuild no longer throws the cache away. */
    { extern unsigned long long rsx_fp_codegen_salt(void); s_pc_exe_id = (u64)(RSX_CODEGEN_ID) ^ (u64)rsx_fp_codegen_salt(); }
#else
    { char exe[MAX_PATH]; if (GetModuleFileNameA(NULL, exe, sizeof exe)) s_pc_exe_id = eng_file_id(exe); }
#endif
    /* pso110: RSX_PSO_LIB_ANYGEN=1: the library does not carry the decompilers' id. A state object is stored under the hash of its
     * whole description (shader bytes, root signature, formats) and the runtime refuses one whose description differs, so a build that
     * translates differently only misses on the programs it changed instead of throwing the whole library away. */
    { const char* ag = getenv("RSX_PSO_LIB_ANYGEN"); if (ag && ag[0] == '1') s_pc_exe_id = 0x4E454741444F4C31ull; }
    s_pc_on = 1;

    /* The pipeline library, if the runtime has it and the file is ours. */
    ID3D12Device1* d1 = NULL;
    const char* pe = getenv("RSX_PSO_LIB");
    if (s_dev && !(pe && pe[0] == '0') && SUCCEEDED(CALL(s_dev, QueryInterface, &IID_ID3D12Device1, (void**)&d1))) {
        char lp[MAX_PATH + 40]; snprintf(lp, sizeof lp, "%s\\pso_%016llx.lib", s_pc_dir, (unsigned long long)s_pc_adapter_id);
        void* mem = NULL; size_t blen = 0;
        FILE* f = fopen(lp, "rb");
        if (f) {
            u32 hd[4]; u64 ids[2];
            fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
            if (n > (long)(sizeof hd + sizeof ids) && fread(hd, sizeof hd, 1, f) == 1 && fread(ids, sizeof ids, 1, f) == 1 &&
                hd[0] == PC_MAGIC_LIB && hd[1] == PC_VERSION && ids[0] == s_pc_exe_id && ids[1] == s_pc_adapter_id &&
                (long)hd[2] == n - (long)(sizeof hd + sizeof ids)) {
                blen = hd[2]; mem = malloc(blen);
                if (!mem || fread(mem, 1, blen, f) != blen ||
                    (u32)fnv1a64(mem, (u32)blen, 1469598103934665603ull) != hd[3]) { free(mem); mem = NULL; blen = 0; }
            }
            fclose(f);
        }
        HRESULT hr = E_FAIL;
        if (mem) {
            hr = CALL(d1, CreatePipelineLibrary, mem, blen, &IID_ID3D12PipelineLibrary, (void**)&s_pc_lib);
            if (FAILED(hr)) { free(mem); mem = NULL; s_pc_lib = NULL; }   /* other driver/adapter, or damaged */
        }
        if (!s_pc_lib) hr = CALL(d1, CreatePipelineLibrary, NULL, 0, &IID_ID3D12PipelineLibrary, (void**)&s_pc_lib);
        if (FAILED(hr)) s_pc_lib = NULL;
        s_pc_lib_mem = mem;   /* the library reads from it for its whole life */
        RELEASE(d1);
    }
    s_pc_last_save = GetTickCount64();
    fprintf(stderr, "[pso-cache] dir='%s' dxbc=on pso-library=%s%s id %016llx\n", s_pc_dir, s_pc_lib ? "on" : "off",
            s_pc_lib_mem ? " (warm)" : s_pc_lib ? " (cold)" : "", (unsigned long long)s_pc_exe_id);
    return TRUE;
}
static int eng_pc_enabled(void) { InitOnceExecuteOnce(&s_pc_once, eng_pc_init_cb, NULL, NULL); return s_pc_on; }
static const char* eng_cache_dir_hook(void) { return eng_pc_enabled() && s_pc_dir[0] ? s_pc_dir : NULL; }
/* hitch108: the loader keeps "<level>/level_cached" at 0x101E7E50 of guest memory (EBOOT .bss; see src/discord/discord_level.cpp) */
static int eng_level_hook(char* out, size_t n)
{
    if (!vm_base) return 0;
    const u8* p = vm_base + 0x101E7E50u;
    MEMORY_BASIC_INFORMATION mbi;
    if (!VirtualQuery(p, &mbi, sizeof mbi) || mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) ||
        (const u8*)mbi.BaseAddress + mbi.RegionSize < p + 48) return 0;
    char buf[48]; memcpy(buf, p, 47); buf[47] = 0;
    char* sl = strstr(buf, "/level_cached");
    if (!sl || sl == buf || strcmp(sl, "/level_cached")) return 0;
    *sl = 0;
    if (strlen(buf) + 1 > n) return 0;
    memcpy(out, buf, strlen(buf) + 1);
    return 1;
}

/* Writes the library out: at exit, and every 30 s while states are being added
 * (the game is usually ended by killing it, not by a clean exit). */
static ULONGLONG s_pc_last_store;   /* tick of the last state stored */
static void eng_pc_save(int final)   /* 0: at most every 30 s, 1: at exit, 2: idle save (no throttle) */
{
    if (!s_pc_lib || !s_pc_dirty) { if (final == 3) InterlockedExchange(&s_pc_saving, 0); return; }
    /* 27/10: one writer at a time, at exit too (it used to skip the CAS and could race an idle save on the same
     * .tmp). 3: the idle save, whose flag eng_pc_idle_save already took before queueing the work item. */
    if (final == 1) {
        for (int w = 0; InterlockedCompareExchange(&s_pc_saving, 1, 0) != 0; w++) {
            if (w > 1000) return;   /* ~10 s: a save that long is not coming back; leave its file alone */
            Sleep(10);
        }
        if (!s_pc_dirty) { InterlockedExchange(&s_pc_saving, 0); return; }
    } else if (final != 3) {
        if (final == 0 && GetTickCount64() - s_pc_last_save < 30000ull) return;
        if (InterlockedCompareExchange(&s_pc_saving, 1, 0) != 0) return;
    }
    void* buf = NULL; SIZE_T n = 0; int ok = 0;
    LARGE_INTEGER sq0, sq1, sqf; QueryPerformanceCounter(&sq0);
    eng_lock(&s_pc_lock, "pc_lock");
    n = CALL0(s_pc_lib, GetSerializedSize);
    buf = n ? malloc(n) : NULL;
    if (buf && SUCCEEDED(CALL(s_pc_lib, Serialize, buf, n))) ok = 1;
    s_pc_dirty = 0; s_pc_last_save = GetTickCount64();
    ReleaseSRWLockExclusive(&s_pc_lock);
    QueryPerformanceCounter(&sq1); QueryPerformanceFrequency(&sqf);
    g_rsx_frame_pcsave_ms += (double)(sq1.QuadPart - sq0.QuadPart) * 1000.0 / (double)sqf.QuadPart; g_rsx_frame_pcsave_n++;   /* surf54: how long the library stayed shut */
    if (ok) {
        char lp[MAX_PATH + 40], tp[MAX_PATH + 48];
        snprintf(lp, sizeof lp, "%s\\pso_%016llx.lib", s_pc_dir, (unsigned long long)s_pc_adapter_id);
        snprintf(tp, sizeof tp, "%s.%lu.tmp", lp, (unsigned long)GetCurrentThreadId());
        FILE* f = fopen(tp, "wb");
        if (f) {
            u32 hd[4] = { PC_MAGIC_LIB, PC_VERSION, (u32)n, (u32)fnv1a64(buf, (u32)n, 1469598103934665603ull) };
            u64 ids[2] = { s_pc_exe_id, s_pc_adapter_id };
            int w = fwrite(hd, sizeof hd, 1, f) == 1 && fwrite(ids, sizeof ids, 1, f) == 1 && fwrite(buf, 1, n, f) == n;
            w = (fclose(f) == 0) && w;
            if (w) { remove(lp); rename(tp, lp); } else remove(tp);
            fprintf(stderr, "[pso-cache] %s %llu KB; pso hit %ld miss %ld (%.0f ms compiling), dxbc hit %ld miss %ld (%.0f ms compiling)\n",
                    w ? "saved library" : "library save FAILED", (unsigned long long)(n >> 10),
                    (long)s_pc_pso_hit, (long)s_pc_pso_miss, (double)s_pc_pso_miss_us / 1000.0,
                    (long)s_pc_dxbc_hit, (long)s_pc_dxbc_miss, (double)s_pc_dxbc_miss_us / 1000.0);
        }
    }
    free(buf);
    InterlockedExchange(&s_pc_saving, 0);
}

static unsigned long long eng_cache_sig_hook(void)
{
    if (!eng_pc_enabled()) return 0;
    u64 h = 1469598103934665603ull;
    h = fnv1a64(&s_pc_adapter_id, sizeof s_pc_adapter_id, h); h = fnv1a64(&s_pc_umd, sizeof s_pc_umd, h); h = fnv1a64(&s_pc_fxc_id, sizeof s_pc_fxc_id, h);
    { const u32 ms = s_ms_n; h = fnv1a64(&ms, sizeof ms, h); }   /* pso112: the MSAA twins are prewarmed for the current sample count only: another count = another prewarm marker */
    return h ? h : 1;
}
static int eng_cache_warm_hook(void) { return eng_pc_enabled() && s_pc_lib_mem != NULL; }
static void eng_cache_flush_hook(void) { if (eng_pc_enabled()) eng_pc_save(1); }

/* The name a state object has in the library: every field of its description
 * that is not a pointer, the shader bytes, the input layout. */
static u64 eng_pso_key(const D3D12_GRAPHICS_PIPELINE_STATE_DESC* d)
{
    D3D12_GRAPHICS_PIPELINE_STATE_DESC c = *d;
    c.pRootSignature = NULL; c.VS.pShaderBytecode = NULL; c.PS.pShaderBytecode = NULL;
    c.DS.pShaderBytecode = NULL; c.HS.pShaderBytecode = NULL; c.GS.pShaderBytecode = NULL;
    c.StreamOutput.pSODeclaration = NULL; c.StreamOutput.pBufferStrides = NULL;
    c.InputLayout.pInputElementDescs = NULL; c.CachedPSO.pCachedBlob = NULL;
    u64 h = fnv1a64(&c, sizeof c, 1469598103934665603ull);
    if (d->VS.pShaderBytecode) h = fnv1a64(d->VS.pShaderBytecode, (u32)d->VS.BytecodeLength, h);
    if (d->PS.pShaderBytecode) h = fnv1a64(d->PS.pShaderBytecode, (u32)d->PS.BytecodeLength, h);
    for (UINT i = 0; i < d->InputLayout.NumElements; i++) {
        const D3D12_INPUT_ELEMENT_DESC* e = &d->InputLayout.pInputElementDescs[i];
        u32 v[6] = { e->SemanticIndex, (u32)e->Format, e->InputSlot, e->AlignedByteOffset, (u32)e->InputSlotClass, e->InstanceDataStepRate };
        h = fnv1a64(v, sizeof v, h);
    }
    return h;
}

/* CreateGraphicsPipelineState through the library: a hit costs a load, a miss
 * the driver compile and a store. */
static HRESULT eng_create_gfx_pso(const D3D12_GRAPHICS_PIPELINE_STATE_DESC* pd, ID3D12PipelineState** out)
{
    const double t0 = eng_now_us();
    if (!eng_pc_enabled() || !s_pc_lib)
        return CALL(s_dev, CreateGraphicsPipelineState, pd, &IID_ID3D12PipelineState, (void**)out);
    WCHAR name[24]; swprintf(name, 24, L"%016llx", (unsigned long long)eng_pso_key(pd));
    eng_lock(&s_pc_lock, "pc_lock");
    HRESULT hr = CALL(s_pc_lib, LoadGraphicsPipeline, name, pd, &IID_ID3D12PipelineState, (void**)out);
    ReleaseSRWLockExclusive(&s_pc_lock);
    if (SUCCEEDED(hr)) { InterlockedIncrement(&s_pc_pso_hit); return hr; }
    *out = NULL;
    hr = CALL(s_dev, CreateGraphicsPipelineState, pd, &IID_ID3D12PipelineState, (void**)out);
    if (SUCCEEDED(hr)) {
        eng_lock(&s_pc_lock, "pc_lock");
        if (SUCCEEDED(CALL(s_pc_lib, StorePipeline, name, *out))) InterlockedIncrement(&s_pc_dirty);
        ReleaseSRWLockExclusive(&s_pc_lock);
        InterlockedIncrement(&s_pc_pso_miss);
        InterlockedAdd64(&s_pc_pso_miss_us, (LONG64)(eng_now_us() - t0));
        s_pc_last_store = GetTickCount64();
        eng_pc_save(0);
    }
    return hr;
}

/* The game is usually ended by killing it, and the 30 s rule above only
 * runs when yet another state arrives: a start that built everything in its
 * first 20 s saved nothing, and the next one built it all again. So, from
 * the present: once nothing has been stored for 3 s, the library goes to
 * disk (on a pool thread). */
static DWORD WINAPI eng_pc_idle_thread(LPVOID a) { (void)a; eng_pc_save(3); return 0; }
static void eng_pc_idle_save(void)
{
    if (!s_pc_dirty || !s_pc_lib) return;
    if (GetTickCount64() - s_pc_last_store < 3000ull || s_pc_saving) return;
    if (InterlockedCompareExchange(&s_pc_saving, 1, 0) != 0) return;     /* taken here: one work item at a time */
    if (!QueueUserWorkItem(eng_pc_idle_thread, NULL, WT_EXECUTELONGFUNCTION)) InterlockedExchange(&s_pc_saving, 0);
}

/* Compiled DXBC, keyed on the HLSL the decompilers emitted: one program is
 * compiled once however many pipeline variants reference it.
 * PS3RECOMP_DXBC_CACHE=<dir> keeps the bytecode on disk between runs. */
static SRWLOCK s_blob_lock = SRWLOCK_INIT;
/* 31/10 (pre52): texts being compiled right now. Four build threads pre-warming the manifest (and the walker) met the
 * same vertex program in many pipelines and each ran its own FXC -O3 (up to seconds) on the same text; the others now
 * wait for the first. Under s_blob_lock. */
static CONDITION_VARIABLE s_blob_cv = CONDITION_VARIABLE_INIT;
static u64 s_inflight[32]; static int s_inflight_n;
static void eng_inflight_leave_locked(u64 hash)
{
    for (int i = 0; i < s_inflight_n; i++) if (s_inflight[i] == hash) { s_inflight[i] = s_inflight[--s_inflight_n]; break; }
    WakeAllConditionVariable(&s_blob_cv);
}
static ID3DBlob* eng_shader(const char* hlsl, int stage, const char* what)
{
    const u32 len = (u32)strlen(hlsl);
    const u64 hash = fnv1a64(hlsl, len, 1469598103934665603ull);
    /* The cache under its own lock and the compile under none. A D3DCompile
     * is 20-300 ms, and with it inside s_pipe_lock every state object the
     * walker had to build for a draw queued behind the build thread: frames
     * of 100-300 ms the first time through an area, on a machine without the
     * bytecode cache. Two threads compiling the same text keep the first
     * one in. */
    eng_lock(&s_blob_lock, "blob_lock");
    for (u32 h = eng_h32(hash) & (ENG_BLOB_HASH - 1u); s_blob_ht[h]; h = (h + 1u) & (ENG_BLOB_HASH - 1u))
        if (s_blob[s_blob_ht[h] - 1u].hash == hash) { ID3DBlob* b = s_blob[s_blob_ht[h] - 1u].blob; ReleaseSRWLockExclusive(&s_blob_lock); return b; }
    ReleaseSRWLockExclusive(&s_blob_lock);

    char name[96];
    snprintf(name, sizeof name, "%s_%016llx.hlsl", what, (unsigned long long)hash);
    dump_shader(name, hlsl, len);

    static const char* cache_dir = (const char*)1;
    static char cache_buf[MAX_PATH + 8];
    if (cache_dir == (const char*)1) {
        const char* ov = getenv("PS3RECOMP_DXBC_CACHE");
        if (ov && *ov) { _mkdir(ov); snprintf(cache_buf, sizeof cache_buf, "%s", ov); cache_dir = cache_buf; }
        else {
            cache_dir = NULL;
            if (eng_pc_enabled()) { snprintf(cache_buf, sizeof cache_buf, "%s\\dxbc", s_pc_dir); cache_dir = cache_buf; }
        }
    }
    char cpath[1024] = "";
    { static int dl = 0; if (dl < 3 && getenv("RSX_PSO_LOG") && ++dl) fprintf(stderr, "[pso-cache] eng_shader stage %d: cache_dir=%s\n", stage, cache_dir ? cache_dir : "(none)"); }
    ID3DBlob* blob = NULL;
    const double tc0 = eng_now_us();
    if (cache_dir) {
        u64 key = fnv1a64("dxbc-v2", 7, hash ^ (u64)(stage + 1) * 0x9E3779B97F4A7C15ull);
        key = fnv1a64(&s_pc_fxc_id, sizeof s_pc_fxc_id, key);   /* compiler build; the flags are fixed in eng_compile */
        snprintf(cpath, sizeof cpath, "%s/%s_%016llx.dxbc", cache_dir, what, (unsigned long long)key);
        FILE* cf = fopen(cpath, "rb");
        if (cf) {
            u32 hd[4];   /* magic, version, size, checksum */
            fseek(cf, 0, SEEK_END); long n = ftell(cf); fseek(cf, 0, SEEK_SET);
            if (n > (long)sizeof hd && fread(hd, sizeof hd, 1, cf) == 1 && hd[0] == PC_MAGIC_DXBC && hd[1] == 1u &&
                (long)hd[2] == n - (long)sizeof hd && hd[2] >= 4 && SUCCEEDED(D3DCreateBlob((SIZE_T)hd[2], &blob))) {
                if (fread(CALL0(blob, GetBufferPointer), 1, hd[2], cf) != hd[2] ||
                    (u32)fnv1a64(CALL0(blob, GetBufferPointer), hd[2], 1469598103934665603ull) != hd[3] ||
                    memcmp(CALL0(blob, GetBufferPointer), "DXBC", 4) != 0) RELEASE(blob);
            }
            fclose(cf);
        }
        if (blob) InterlockedIncrement(&s_pc_dxbc_hit);
    }
    int inflight = 0;
    if (!blob) {
        eng_lock(&s_blob_lock, "blob_lock");
        for (;;) {
            for (u32 h = eng_h32(hash) & (ENG_BLOB_HASH - 1u); s_blob_ht[h]; h = (h + 1u) & (ENG_BLOB_HASH - 1u))
                if (s_blob[s_blob_ht[h] - 1u].hash == hash) { ID3DBlob* b = s_blob[s_blob_ht[h] - 1u].blob; ReleaseSRWLockExclusive(&s_blob_lock); return b; }
            int other = 0; for (int i = 0; i < s_inflight_n; i++) if (s_inflight[i] == hash) other = 1;
            if (!other) break;
            SleepConditionVariableSRW(&s_blob_cv, &s_blob_lock, INFINITE, 0);
        }
        if (s_inflight_n < 32) { s_inflight[s_inflight_n++] = hash; inflight = 1; }
        ReleaseSRWLockExclusive(&s_blob_lock);
    }
    if (!blob) {
        snprintf(name, sizeof name, "%s %016llx", what, (unsigned long long)hash);
        blob = eng_compile(hlsl, len, "main", stage == 0 ? "vs_5_0" : "ps_5_0", name);
        if (blob) {
            InterlockedIncrement(&s_pc_dxbc_miss);
            InterlockedAdd64(&s_pc_dxbc_miss_us, (LONG64)(eng_now_us() - tc0));
        }
        if (blob && cpath[0]) {
            char tmp[1100]; snprintf(tmp, sizeof tmp, "%s.%lu.tmp", cpath, (unsigned long)GetCurrentThreadId());
            FILE* cf = fopen(tmp, "wb");
            { static int dl = 0; if (dl < 3 && getenv("RSX_PSO_LOG") && ++dl) fprintf(stderr, "[pso-cache] dxbc write %s -> %s\n", tmp, cf ? "open" : "FAILED to open"); }
            if (cf) {
                const u32 sz = (u32)CALL0(blob, GetBufferSize);
                u32 hd[4] = { PC_MAGIC_DXBC, 1u, sz, (u32)fnv1a64(CALL0(blob, GetBufferPointer), sz, 1469598103934665603ull) };
                int w = fwrite(hd, sizeof hd, 1, cf) == 1 && fwrite(CALL0(blob, GetBufferPointer), 1, sz, cf) == sz;
                w = (fclose(cf) == 0) && w;
                if (w) { remove(cpath); rename(tmp, cpath); } else remove(tmp);
            }
        }
    }
    eng_lock(&s_blob_lock, "blob_lock");
    u32 slot = eng_h32(hash) & (ENG_BLOB_HASH - 1u);
    for (; s_blob_ht[slot]; slot = (slot + 1u) & (ENG_BLOB_HASH - 1u))
        if (s_blob[s_blob_ht[slot] - 1u].hash == hash) {   /* compiled by another thread meanwhile */
            ID3DBlob* b = s_blob[s_blob_ht[slot] - 1u].blob;
            if (inflight) eng_inflight_leave_locked(hash);
            ReleaseSRWLockExclusive(&s_blob_lock);
            RELEASE(blob);
            return b;
        }
    if (s_blob_count >= ENG_MAX_BLOBS) {
        static int full = 0; if (full < 3 && ++full) fprintf(stderr, "[rsx engine/d3d12] shader cache full (%u): new programs fail\n", ENG_MAX_BLOBS);
        if (inflight) eng_inflight_leave_locked(hash);
        ReleaseSRWLockExclusive(&s_blob_lock); RELEASE(blob); return NULL;
    }
    s_blob[s_blob_count].hash = hash;
    s_blob[s_blob_count].blob = blob;
    s_blob_count++;
    s_blob_ht[slot] = s_blob_count;
    if (inflight) eng_inflight_leave_locked(hash);
    ReleaseSRWLockExclusive(&s_blob_lock);
    return blob;
}

/* A fragment program at the current internal resolution. */
static ID3DBlob* eng_fp_blob(const char* ps_hlsl)
{
    /* With the internal resolution raised, a fragment program's WPOS (the
     * decompiler's `input.position`) arrives in host pixels; the title
     * computes screen UVs and offsets from it in its own. Divide it back:
     * every use of the input is wrapped, and the text is what the shader
     * cache is keyed on, so the two builds never collide. */
    ID3DBlob* ps = NULL;
    const char* wp = (s_scale != 1.0f || eng_extra_on() || eng_hstretch_on()) ? strstr(ps_hlsl, "input.position") : NULL;
    if (wp) {
        const char* needle = "input.position";
        /* The RSX hands the program the guest pixel centre (n + 0.5): a title
         * that derives an ordered dither / jitter from frac(WPOS) (ToD's
         * 4-tap shadow filter on characters) sees a constant fraction along x
         * and a row parity along y. Scaled by 1/s the position lands at
         * arbitrary fractions, so the pattern toggles per host pixel (the
         * grid on Ratchet's fur and Clank). Default: snap to the centre of
         * the guest pixel. RSX_WPOS_SNAP=0: the old smooth scaled position. */
        static int snapm = -1; if (snapm < 0) { const char* e = getenv("RSX_WPOS_SNAP"); snapm = (e && e[0] == '0') ? 0 : (e && e[0] == '2') ? 2 : (e && e[0] == '1') ? 1 : (e && e[0] == '3') ? 3 : 4; }
        /* Mode 3 (default, host-pixel dither) and 1 (guest-pixel snap): only programs that take frac() of a fragment
         * position AND filter shadows (the character jitter) get the snapped
         * position; screen-space post-process keeps the smooth one. 2: all.
         * fp_alpha.z is the per-draw switch (RSX_WPOS_SNAP_FILE turns it off
         * while that file exists, for same-scene A/B). */
        const int snap = snapm == 2 || (snapm >= 1 && strstr(ps_hlsl, "frac(") && strstr(ps_hlsl, "rsx_shadow5("));
        /* Mode 3: those programs get the HOST pixel centre (floor(p)+0.5,
         * not divided by the scale), so row/column parity alternates every
         * host pixel and the dither is as fine as the original's at 720p.
         * Their WPOS feeds only the frac() dither bits (checked in the HLSL
         * of the 12 programs: one read, followed by the frac()s). */
        const int host_snap = snapm >= 3;
        /* Mode 4 (default): same programs, but the position is a CONSTANT: the 4-tap shadow filter's row dither
         * (frac(WPOS) picks an offset of the shadow taps) then has no pattern. At host resolution the row parity
         * of mode 3 shows as 2 px horizontal stripes on lit metal/skin (Clank's head). Those programs read WPOS only
         * for the frac() bits (checked in the HLSL), so a constant phase just fixes the tap offset.
         * RSX_WPOS_SNAP=3 restores the host-pixel parity. */
        const int const_snap = snapm == 4;
        const char* repl = snap ? "RSX_WPOSF(input.position)" : "(input.position*RSX_WPOS)";
        const size_t nl = strlen(needle), rl = strlen(repl), src_len = strlen(ps_hlsl);
        size_t n = 0;
        for (const char* q = wp; (q = strstr(q, needle)) != NULL; q += nl) n++;
        char head[760];
        snprintf(head, sizeof head, "#define RSX_WPOSK (fp_alpha.w > 0.0 ? fp_alpha.w : 1.0)\n"
                 "#define RSX_ANI (fp_alpha.z > 1.5)\n"
                 "#define RSX_SNAPSW ((fp_alpha.z - (RSX_ANI ? 2.0 : 0.0)) > 0.5)\n"
                 "#define RSX_WPOS float4(%.8f * RSX_WPOSK * (RSX_ANI ? 0.75 : 1.0), %.8f * RSX_WPOSK, 1.0, 1.0)\n"
                 "#define RSX_WPOSF(p) float4(RSX_SNAPSW ? %s : (p).xy * RSX_WPOS.xy, (p).zw)\n",
                 1.0 / (double)s_scale, 1.0 / (double)s_scale, const_snap ? "float2(64.5, 64.5)" : host_snap ? "floor((p).xy) + 0.5" : "floor((p).xy * RSX_WPOS.xy) + 0.5");
        char* text = (char*)malloc(src_len + n * (rl - nl) + strlen(head) + 1);
        if (text) {
            char* d = text + sprintf(text, "%s", head);
            for (const char* q = ps_hlsl;;) {
                const char* hit = strstr(q, needle);
                if (!hit) { strcpy(d, q); break; }
                memcpy(d, q, (size_t)(hit - q)); d += hit - q;
                memcpy(d, repl, rl); d += rl;
                q = hit + nl;
            }
            ps = eng_shader(text, 1, "fp");
            free(text);
        }
    }
    if (!ps) ps = eng_shader(ps_hlsl, 1, "fp");
    return ps;
}

/* The entry, from shaders already compiled (eng_pipeline_create). */
static u32 eng_pipeline_create_locked(ID3DBlob* vs, ID3DBlob* ps, const char* ps_hlsl,
                                      const rsx_be_render_state* rs,
                                      const rsx_vertex_layout_plan* layout,
                                      u32 vertex_stride, rsx_be_format rt_fmt, u32 rt_count)
{
    if (!s_dev || !vertex_stride || !vs || !ps) return 0;
    if (!rt_count) rt_count = 1;
    if (rt_count > RSX_BE_MAX_COLOR_TARGETS) rt_count = RSX_BE_MAX_COLOR_TARGETS;
    if (s_pipe_count >= ENG_MAX_PIPES) {
        static int n = 0;
        if (n < 4 && ++n) fprintf(stderr, "[rsx engine/d3d12] pipeline table full (%u): new pipelines fail, their draws drop\n", ENG_MAX_PIPES);
        return 0;
    }
    const int wp = strstr(ps_hlsl, "input.position") != NULL;

    EngPipeline* p = &s_pipe[s_pipe_count];
    memset(p, 0, sizeof *p);
    CALL0(vs, AddRef); CALL0(ps, AddRef);
    p->vs = vs; p->ps = ps;
    if (wp) p->ps_plain = _strdup(ps_hlsl);   /* at any scale: a rescale rebuilds from it */
    /* Input slot i carries attribute attrs[i] at i*16: the layout SLOT, as
     * rsx_metal_backend.m's eng_pipeline_create explains. */
    for (u32 slot = 0; slot < layout->count && slot < RSX_DSP_NUM_VERTEX_ATTR; slot++) {
        D3D12_INPUT_ELEMENT_DESC* e = &p->il[slot];
        e->SemanticName = "ATTR"; e->SemanticIndex = layout->attrs[slot];
        e->Format = DXGI_FORMAT_R32G32B32A32_FLOAT; e->InputSlot = 0;
        e->AlignedByteOffset = slot * 16u;
        e->InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
    }
    D3D12_GRAPHICS_PIPELINE_STATE_DESC* pd = &p->desc;
    pd->pRootSignature = s_rootsig;
    pd->VS.pShaderBytecode = CALL0(vs, GetBufferPointer); pd->VS.BytecodeLength = CALL0(vs, GetBufferSize);
    pd->PS.pShaderBytecode = CALL0(ps, GetBufferPointer); pd->PS.BytecodeLength = CALL0(ps, GetBufferSize);
    pd->InputLayout.pInputElementDescs = layout->count ? p->il : NULL;
    pd->InputLayout.NumElements = layout->count;
    pd->SampleMask = UINT_MAX;
    pd->SampleDesc.Count = 1;
    pd->RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd->RasterizerState.DepthClipEnable = TRUE;
    /* CULL_FACE FRONT=0x0404 BACK=0x0405 FRONT_AND_BACK=0x0408 (front here,
     * as in the Metal engine); FRONT_FACE CW=0x0900 CCW=0x0901. */
    pd->RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    if (rs->cull_enable && rs->cull_face)
        pd->RasterizerState.CullMode = (rs->cull_face == 0x0404u || rs->cull_face == 0x0408u) ? D3D12_CULL_MODE_FRONT
                                     : (rs->cull_face == 0x0405u ? D3D12_CULL_MODE_BACK : D3D12_CULL_MODE_NONE);
    pd->RasterizerState.FrontCounterClockwise = (rs->front_face == 0x0901u) ? TRUE : FALSE;
    /* POLYGON_OFFSET: glPolygonOffset(factor, units) = D3D slope-scaled bias
     * and integer bias in depth-format units. RSX_POLY_OFFSET=0 ignores it. */
    { static int po = -1; if (po < 0) { const char* e = getenv("RSX_POLY_OFFSET"); po = !(e && e[0] == '0'); }
      if (po && rs->poly_offset_fill) {
          float sc, bi; memcpy(&sc, &rs->poly_offset_scale, 4); memcpy(&bi, &rs->poly_offset_bias, 4);
          if (rsx_dbias_epilogue_on()) {   /* shad114: slope already scaled+quantised by the draw engine, constant bias is in the VS epilogue; sign + */
              if (sc == sc && sc > -1e6f && sc < 1e6f) {
                  pd->RasterizerState.SlopeScaledDepthBias = sc;
                  pd->RasterizerState.DepthBias = 0;
              }
          } else
          if (sc == sc && bi == bi && sc > -1e6f && sc < 1e6f && bi > -1e9f && bi < 1e9f) {
              { static float sg = 0.f; if (sg == 0.f) { const char* e = getenv("RSX_POLY_SIGN"); sg = (e && e[0] == 45) ? -1.f : 1.f; /* integ113 02/11: default + (original RSX sign; characters' ground shadows return). The - default of 26/10 removed the character shadows; RSX_POLY_SIGN=- = that behaviour */ } sc *= sg; bi *= sg; }
              pd->RasterizerState.SlopeScaledDepthBias = sc;
              pd->RasterizerState.DepthBias = (INT)bi;
              { static int pl = -1, pn = 0; if (pl < 0) pl = getenv("RSX_PSO_BIAS_LOG") ? 1 : 0;
                if (pl && pn < 400 && ++pn)
                    fprintf(stderr, "[cut88 pso] poly_offset scale=%g bias=%g (raw %08x/%08x) pass_shadow=%d pass=%ux%u dsv=D32F_S8 cull=%d\n",
                            (double)sc, (double)bi, (unsigned)rs->poly_offset_scale, (unsigned)rs->poly_offset_bias,
                            (int)g_rsx_pass_shadow, (unsigned)g_rsx_pass_cw, (unsigned)g_rsx_pass_ch, (int)pd->RasterizerState.CullMode); } }
      } }
    /* nv40 COLOR_MASK byte layout: B=[0:7] G=[8:15] R=[16:23] A=[24:31]. */
    UINT8 wm = 0;
    if ((rs->color_mask >>  0) & 0xFF) wm |= D3D12_COLOR_WRITE_ENABLE_BLUE;
    if ((rs->color_mask >>  8) & 0xFF) wm |= D3D12_COLOR_WRITE_ENABLE_GREEN;
    if ((rs->color_mask >> 16) & 0xFF) wm |= D3D12_COLOR_WRITE_ENABLE_RED;
    if ((rs->color_mask >> 24) & 0xFF) wm |= D3D12_COLOR_WRITE_ENABLE_ALPHA;
    D3D12_RENDER_TARGET_BLEND_DESC b = {0};
    b.RenderTargetWriteMask = wm;
    if (rs->blend_enable) {
        b.BlendEnable    = TRUE;
        b.SrcBlend       = gcm_blend_factor(rs->sf_rgb, 0);
        b.DestBlend      = gcm_blend_factor(rs->df_rgb, 0);
        b.BlendOp        = gcm_blend_op(rs->eq_rgb);
        b.SrcBlendAlpha  = gcm_blend_factor(rs->sf_a, 1);
        b.DestBlendAlpha = gcm_blend_factor(rs->df_a, 1);
        b.BlendOpAlpha   = gcm_blend_op(rs->eq_a);
    } else {
        b.SrcBlend = D3D12_BLEND_ONE; b.DestBlend = D3D12_BLEND_ZERO; b.BlendOp = D3D12_BLEND_OP_ADD;
        b.SrcBlendAlpha = D3D12_BLEND_ONE; b.DestBlendAlpha = D3D12_BLEND_ZERO; b.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    }
    /* Every attachment of an MRT set takes target A's blend and colour mask. */
    pd->NumRenderTargets = rt_count;
    for (u32 r = 0; r < rt_count; r++) {
        pd->RTVFormats[r] = eng_dxgi(rt_fmt);
        pd->BlendState.RenderTarget[r] = b;
    }
    pd->DSVFormat = ENG_DEPTH_FMT;
    pd->DepthStencilState.DepthEnable = rs->depth_test ? TRUE : FALSE;
    pd->DepthStencilState.DepthWriteMask = (rs->depth_test && rs->depth_write) ? D3D12_DEPTH_WRITE_MASK_ALL
                                                                               : D3D12_DEPTH_WRITE_MASK_ZERO;
    pd->DepthStencilState.DepthFunc = rs->depth_test ? gcm_cmp(rs->depth_func) : D3D12_COMPARISON_FUNC_ALWAYS;
    if (rs->stencil_enable) {
        pd->DepthStencilState.StencilEnable    = TRUE;
        pd->DepthStencilState.StencilReadMask  = (UINT8)(rs->s_func_mask & 0xFFu);
        pd->DepthStencilState.StencilWriteMask = (UINT8)(rs->s_write_mask & 0xFFu);
        pd->DepthStencilState.FrontFace.StencilFunc        = gcm_cmp(rs->s_func);
        pd->DepthStencilState.FrontFace.StencilFailOp      = gcm_stencil_op(rs->s_fail);
        pd->DepthStencilState.FrontFace.StencilDepthFailOp = gcm_stencil_op(rs->s_zfail);
        pd->DepthStencilState.FrontFace.StencilPassOp      = gcm_stencil_op(rs->s_zpass);
        if (rs->stencil_two_sided) {
            pd->DepthStencilState.BackFace.StencilFunc        = gcm_cmp(rs->bs_func);
            pd->DepthStencilState.BackFace.StencilFailOp      = gcm_stencil_op(rs->bs_fail);
            pd->DepthStencilState.BackFace.StencilDepthFailOp = gcm_stencil_op(rs->bs_zfail);
            pd->DepthStencilState.BackFace.StencilPassOp      = gcm_stencil_op(rs->bs_zpass);
        } else {
            /* Two-sided off: nv40 applies the front state to both faces. */
            pd->DepthStencilState.BackFace = pd->DepthStencilState.FrontFace;
        }
    }
    p->live = 1;
    return ++s_pipe_count;
}

/* The engine may build pipelines on a worker thread as well as the walker;
 * the blob cache and the table are shared, so one at a time. */
static ID3D12PipelineState* eng_pso(u32 pipeline, int cls);   /* below */
static u32 s_ms_n;   /* samples in use (defined below) */
static ID3D12PipelineState* eng_pso_ms(u32 pipeline, int cls);   /* below */
static ID3D12PipelineState* eng_pso_for(u32 pipeline, int cls, int unscaled);   /* below */
static u32 eng_pipeline_create(void* user, const char* vs_hlsl, const char* ps_hlsl,
                               const rsx_be_render_state* rs,
                               const rsx_vertex_layout_plan* layout,
                               u32 vertex_stride, rsx_be_format rt_fmt, u32 rt_count)
{
    (void)user;
    if (!s_dev) return 0;
    /* The shaders first, under no lock (eng_shader); the table under it. */
    ID3DBlob* vs = eng_shader(vs_hlsl, 0, "vp");
    ID3DBlob* ps = vs ? eng_fp_blob(ps_hlsl) : NULL;
    if (!vs || !ps) return 0;
    eng_lock(&s_pipe_lock, "pipe_lock");
    const u32 h = eng_pipeline_create_locked(vs, ps, ps_hlsl, rs, layout, vertex_stride, rt_fmt, rt_count);
    ReleaseSRWLockExclusive(&s_pipe_lock);
    /* The pipeline state for triangles too, here: this runs on the draw
     * engine's build thread (RSX_ASYNC_SHADERS), and a state object is the
     * driver's compile -- milliseconds, tens of them for a shader it has not
     * cached -- which on the walker, at the pipeline's first draw, was a
     * hitch the first time anything new came on screen. Points and lines,
     * rare, are still built on first use. RSX_PSO_PREWARM=0: all on first use. */
    { static int warm = -1;
      if (warm < 0) { const char* e = getenv("RSX_PSO_PREWARM"); warm = !(e && e[0] == '0'); }
      if (h && warm) eng_pso(h, 2);
      /* 31/10 (pre52): and the variants a 3D pass at the user's settings really draws with, so no first draw builds
       * one on the walker: the MSAA state (3D passes into the multisampled twins) and the build whose WPOS is
       * left undivided (passes into targets kept at the guest size). RSX_PSO_PREWARM_VARIANTS=0: first use. */
      static int var = -1;
      if (var < 0) { const char* e = getenv("RSX_PSO_PREWARM_VARIANTS"); var = !(e && e[0] == '0'); }
      if (h && warm && var) {
          if (s_ms_n > 1) eng_pso_ms(h, 2);
          eng_pso_for(h, 2, 1);
      } }
    return h;
}

static void eng_pipeline_release(void* user, u32 pipeline)
{
    (void)user;
    eng_lock(&s_pipe_lock, "pipe_lock");
    if (pipeline && pipeline <= s_pipe_count) {
        /* The PSOs stay until shutdown: a submitted list may still name them,
         * and a released pipeline is one the engine evicted, which is rare. */
        s_pipe[pipeline - 1].live = 0;
    }
    ReleaseSRWLockExclusive(&s_pipe_lock);
}

/* The PSO for a pipeline and a topology class, built on first use: D3D12
 * bakes the class into the state object where Metal does not. */
extern double g_rsx_frame_pso_ms;   /* rsx_draw_engine.c: pipeline states built on the walker this frame */
static ID3D12PipelineState* eng_pso(u32 pipeline, int cls)
{
    if (!pipeline || pipeline > s_pipe_count) return NULL;
    EngPipeline* p = &s_pipe[pipeline - 1];
    if (!p->vs || p->failed[cls]) return NULL;
    if (p->pso[cls]) return p->pso[cls];
    LARGE_INTEGER pq0, pq1, pqf; QueryPerformanceFrequency(&pqf); QueryPerformanceCounter(&pq0);
    /* Built under no lock -- the driver's compile, tens of ms for a state
     * object it has not cached -- and published under it: the entry's desc
     * and layout never change once it is in the table, and of two threads
     * building the same one the first to publish wins. */
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = p->desc;
    pd.InputLayout.pInputElementDescs = pd.InputLayout.NumElements ? p->il : NULL;
    pd.PrimitiveTopologyType = cls == 0 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT
                             : cls == 1 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
                                        : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    ID3D12PipelineState* pso = NULL;
    HRESULT hr = eng_create_gfx_pso(&pd, &pso);
    eng_lock(&s_pipe_lock, "pipe_lock");
    if (p->pso[cls]) RELEASE(pso);
    else if (FAILED(hr)) {
        static int n = 0;
        if (n < 16 && ++n) fprintf(stderr, "[rsx engine/d3d12] pipeline state failed: 0x%08lX (removed 0x%08lX)\n",
                              (long)hr, (long)CALL0(s_dev, GetDeviceRemovedReason)); eng_dred_dump("present");
        p->failed[cls] = 1;
    } else p->pso[cls] = pso;
    ReleaseSRWLockExclusive(&s_pipe_lock);
    QueryPerformanceCounter(&pq1);
    const double pso_ms = (double)(pq1.QuadPart - pq0.QuadPart) * 1000.0 / (double)pqf.QuadPart;
    const int on_walker = GetCurrentThreadId() == s_walker_tid;
    if (on_walker) g_rsx_frame_pso_ms += pso_ms;
    /* RSX_PSO_LOG=1: every pipeline state built, how long, and where. */
    { static int lg = -1; if (lg < 0) lg = getenv("RSX_PSO_LOG") ? 1 : 0;
      if (lg) fprintf(stderr, "[pso] pipeline %u class %d: %.2f ms on the %s\n", pipeline, cls, pso_ms,
                      on_walker ? "walker" : "build thread"); }
    return p->pso[cls];
}

/* eng_pso, or for a pass into a target kept at the guest size the build whose
 * fragment program leaves WPOS undivided (a pipeline that does not read WPOS
 * has one build for both). */
static ID3D12PipelineState* eng_pso_for(u32 pipeline, int cls, int unscaled)
{
    if (!unscaled || !pipeline || pipeline > s_pipe_count) return eng_pso(pipeline, cls);
    EngPipeline* p = &s_pipe[pipeline - 1];
    if (!p->ps_plain || (s_scale == 1.0f && !eng_extra_on())) return eng_pso(pipeline, cls);
    if (p->pso1[cls]) return p->pso1[cls];
    if (!p->vs || p->failed1[cls]) return NULL;
    LARGE_INTEGER pq0, pq1, pqf; QueryPerformanceFrequency(&pqf); QueryPerformanceCounter(&pq0);
    /* As eng_pso: compiled and built under no lock, published under it. The
     * blob cache hands out one blob per text, so two threads meet on ps1. */
    ID3DBlob* ps1 = p->ps1 ? p->ps1 : eng_shader(p->ps_plain, 1, "fp");
    ID3D12PipelineState* pso = NULL;
    HRESULT hr = E_FAIL;
    if (ps1) {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = p->desc;
        pd.InputLayout.pInputElementDescs = pd.InputLayout.NumElements ? p->il : NULL;
        pd.PS.pShaderBytecode = CALL0(ps1, GetBufferPointer); pd.PS.BytecodeLength = CALL0(ps1, GetBufferSize);
        pd.PrimitiveTopologyType = cls == 0 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT
                                 : cls == 1 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
                                            : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        hr = eng_create_gfx_pso(&pd, &pso);
    }
    eng_lock(&s_pipe_lock, "pipe_lock");
    if (!p->ps1 && ps1) { p->ps1 = ps1; CALL0(ps1, AddRef); }
    if (p->pso1[cls]) RELEASE(pso);
    else if (FAILED(hr)) p->failed1[cls] = 1;
    else p->pso1[cls] = pso;
    ReleaseSRWLockExclusive(&s_pipe_lock);
    QueryPerformanceCounter(&pq1);
    const double pso_ms = (double)(pq1.QuadPart - pq0.QuadPart) * 1000.0 / (double)pqf.QuadPart;
    if (GetCurrentThreadId() == s_walker_tid) g_rsx_frame_pso_ms += pso_ms;
    { static int lg = -1; if (lg < 0) lg = getenv("RSX_PSO_LOG") ? 1 : 0;
      if (lg) fprintf(stderr, "[pso] pipeline %u class %d, WPOS undivided: %.2f ms\n", pipeline, cls, pso_ms); }
    return p->pso1[cls];
}

/* ---- per-draw binding ------------------------------------------------------ */

/* RSX_ANISO=<1..16> (default 16): anisotropic filtering for the textures the
 * title filters linearly and mipmaps -- the world's ground and walls, which
 * the RSX's own trilinear blurs at a glancing angle. Only for a guest
 * texture with a real mip chain bound to a linear, mipmapped sampler: render
 * targets, their views and snapshots (shadow maps, packed depth, post-process
 * inputs) carry data that spreading samples corrupts -- 16x over them put red
 * and cyan speckle on the ground and the cloth. 1 is off. */
int g_rsx_aniso;   /* 0 until read from RSX_ANISO; the Graphics Settings page sets it live */
static int eng_aniso(void)
{
    int a = g_rsx_aniso;
    if (a <= 0) {
        const char* e = getenv("RSX_ANISO");
        a = (e && *e) ? atoi(e) : 16;
        g_rsx_aniso = a;
    }
    return a < 1 ? 1 : a > 16 ? 16 : a;
}
int g_eng_aniso_on = 1;   /* the host A/B (DOD3_AB=aniso) flips it */

/* A guest texture with a mip chain (not a target, view or snapshot). */
static int eng_aniso_tex(u32 handle)
{
    const EngObj* o = eng_obj(handle);
    return o && o->kind == OBJ_TEXTURE && o->mips > 1;
}
static int eng_sampler_slot_impl(const rsx_be_sampler_desc* d, int aniso_ok);
static int eng_sampler_slot(const rsx_be_sampler_desc* d, int aniso_ok)
{
    RSX_B2P_ENTER(B2P_SAMPLER);
    const int v = eng_sampler_slot_impl(d, aniso_ok);
    RSX_B2P_LEAVE();
    return v;
}
static int eng_sampler_slot_impl(const rsx_be_sampler_desc* d, int aniso_ok)
{
    aniso_ok = aniso_ok && g_eng_aniso_on && eng_aniso() > 1 && d->min_linear && d->mag_linear && d->mip_present;
    const u64 key = ((u64)(aniso_ok ? eng_aniso() : 0) << 56) | (u64)d->min_linear | ((u64)d->mag_linear << 1)
                  | ((u64)d->mip_linear << 2) | ((u64)d->mip_present << 3)
                  | ((u64)d->wrap_s << 4) | ((u64)d->wrap_t << 8) | ((u64)d->wrap_r << 12)
                  | ((u64)(u32)(d->min_lod * 256.0f) << 16)
                  | ((u64)(u32)(d->max_lod * 256.0f) << 32)
                  | ((u64)(u8)(int8_t)(int)(d->lod_bias * 16.0f + (d->lod_bias < 0 ? -0.5f : 0.5f)) << 48);
    /* The border colour matters only to the border modes (4 BORDER, 5 CLAMP,
     * 7 MIRROR_ONCE_BORDER): elsewhere it is left out of the key. */
    const int uses_border = d->wrap_s == 4 || d->wrap_t == 4 || d->wrap_r == 4 ||
                            d->wrap_s == 7 || d->wrap_t == 7 || d->wrap_r == 7;
    const u32 border = uses_border ? d->border_argb : 0u;
    const u32 sh0 = eng_h32(key ^ ((u64)border * 0x9E3779B97F4A7C15ull)) & (ENG_SAMP_HASH - 1u);
    u32 sh = sh0;
    for (; s_samp_ht[sh]; sh = (sh + 1u) & (ENG_SAMP_HASH - 1u)) {
        const u32 i = (u32)s_samp_ht[sh] - 1u;
        if (s_samp[i].key == key && s_samp[i].border == border) return (int)i;
    }
    if (LB->spec) rsx_lane_abort(RSX_LAB_SAMPNEW);   /* lane109: a new sampler is a shared write */
    if (s_samp_count >= ENG_MAX_SAMPLERS) return -1;
    D3D12_SAMPLER_DESC sd = {0};
    const D3D12_FILTER_TYPE mnf = d->min_linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
    const D3D12_FILTER_TYPE mgf = d->mag_linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
    const D3D12_FILTER_TYPE mpf = d->mip_linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
    sd.Filter = D3D12_ENCODE_BASIC_FILTER(mnf, mgf, mpf, D3D12_FILTER_REDUCTION_TYPE_STANDARD);
    sd.AddressU = gcm_wrap(d->wrap_s); sd.AddressV = gcm_wrap(d->wrap_t); sd.AddressW = gcm_wrap(d->wrap_r);
    sd.MinLOD = d->min_lod;
    sd.MaxLOD = d->mip_present ? d->max_lod : 0.0f;
    if (sd.MaxLOD < sd.MinLOD) sd.MaxLOD = sd.MinLOD;
    sd.MipLODBias = d->mip_present ? (float)(int)(d->lod_bias * 16.0f + (d->lod_bias < 0 ? -0.5f : 0.5f)) / 16.0f : 0.0f;
    sd.MaxAnisotropy = 1;
    if (aniso_ok) {
        sd.Filter = D3D12_FILTER_ANISOTROPIC;
        sd.MaxAnisotropy = (UINT)eng_aniso();
    }
    sd.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    /* Border: the guest's colour (was always transparent black, so a shadow
     * projector's texture read black outside its map -- the dark square
     * under Ratchet in ToD). RSX_BORDER_BLACK=1: the old behaviour. */
    { static int black = -1; if (black < 0) black = getenv("RSX_BORDER_BLACK") ? 1 : 0;
      if (!black) {
          sd.BorderColor[0] = (float)((border >> 16) & 0xFFu) / 255.0f;
          sd.BorderColor[1] = (float)((border >>  8) & 0xFFu) / 255.0f;
          sd.BorderColor[2] = (float)( border        & 0xFFu) / 255.0f;
          sd.BorderColor[3] = (float)((border >> 24) & 0xFFu) / 255.0f;
      } }
    const int slot = (int)s_samp_count++;
    CALL(s_dev, CreateSampler, &sd, cpu_handle(s_smp_cpu, s_smp_step, (u32)slot + 1));
    s_samp[slot].key = key; s_samp[slot].border = border;
    s_samp_ht[sh] = (u16)(slot + 1);   /* sh: the first empty cell of this key's probe */
    return slot;
}

static int cut88_uniq(const u32* v, int n)
{
    static u32 seen[256][8]; static int ns = 0; static int on = -1;
    if (on < 0) on = getenv("RSX_CUT88_LOG") ? 1 : 0;
    if (!on) return 0;
    for (int i = 0; i < ns; i++) { int same = 1; for (int k = 0; k < n; k++) if (seen[i][k] != v[k]) { same = 0; break; } if (same) return 0; }
    if (ns >= 256) return 0; for (int k = 0; k < n; k++) seen[ns][k] = v[k]; ns++; return 1;
}
static void eng_bind_targets(void* user, const u32* surfaces, u32 count, u32 depth)
{
    (void)user;
    if (count > RSX_BE_MAX_COLOR_TARGETS) count = RSX_BE_MAX_COLOR_TARGETS;
    for (u32 i = 0; i < RSX_BE_MAX_COLOR_TARGETS; i++)
        LB->pending.rt[i] = (i < count) ? surfaces[i] : 0;
    LB->pending.nrt = count;
    LB->pending.depth = depth;
    /* 26/10 gfx-quick: remember at which factor a shadow colour target drew into this depth, so the
     * snapshot a later pass samples is cropped at the same size (eng_depth_snapshot_common). */
    if (depth && count) {
        EngObj* z = eng_owner(depth); const EngObj* c = eng_owner(surfaces[0]);
        if (z && z->kind == OBJ_DEPTH) z->pf = (c && c->xk > 0.0f && !c->xbloom) ? eng_xk_f(c) : 0.0f;
        if (z && z->kind == OBJ_DEPTH) z->zshadow = g_rsx_pass_shadow ? 1 : 0;   /* RSX_MS_NO_SHADOW */
        if (z && z->kind == OBJ_DEPTH && c && c->xx > 0.0f) z->xx = c->xx;   /* sticky: a 960x704 request/clear rect on this depth is the stretched scene's */
        if (z && z->kind == OBJ_DEPTH && c && g_rsx_pass_shadow) { u32 v[8] = { c->gw, c->gh, c->w, c->h, z->w, z->h, (u32)(z->pf * 100.f), (u32)(c->xk * 100.f) };
            if (cut88_uniq(v, 8)) fprintf(stderr, "[cut88 bind] shadow pass colour guest %ux%u host %ux%u xk=%g, depth host %ux%u, pf=%g\n", c->gw, c->gh, c->w, c->h, (double)c->xk, z->w, z->h, (double)z->pf); }
    }
}
static void eng_bind_pipeline(void* user, u32 pipeline) { (void)user; LB->pending.pipeline = pipeline; }

static void eng_bind_vs_constants(void* user, const void* data, u32 bytes)
{
    (void)user;
    if (!eng_stage_copy(data, bytes, &LB->pending.vs_cb_off)) bytes = 0;
    LB->pending.vs_cb_bytes = bytes;
    LB->vs_cb_seq = s_submit_seq;        /* read after the copy: it may have submitted */
    LB->vs_cb_off = LB->pending.vs_cb_off;
    LB->vs_cb_bytes = bytes;
}
static int eng_reuse_vs_constants(void* user)
{
    (void)user;
    if (LB->vs_cb_seq != s_submit_seq || !LB->vs_cb_bytes) return 0;
    LB->pending.vs_cb_off = LB->vs_cb_off;
    LB->pending.vs_cb_bytes = LB->vs_cb_bytes;
    return 1;
}
static void eng_bind_ps_constants(void* user, const void* data, u32 bytes)
{
    (void)user;
    /* 26/10 gfx-quick: a pass into a target with an extra multiplier: the shader divides WPOS by
     * RSX_SCALE, fp_alpha.w (spare, 0 otherwise) carries the rest of the ratio (see eng_fp_blob). */
    void* patched = NULL;
    static __declspec(thread) float tl_patch[1024];   /* 27/10: no malloc/free per draw (a PS constant block is a few hundred bytes) */
    if (bytes >= 16 && eng_extra_on()) {
        const float f = eng_obj_f(LB->pending.nrt ? LB->pending.rt[0] : LB->pending.depth);
        if (f != 1.0f && f != s_scale) {
            patched = bytes <= sizeof tl_patch ? (void*)tl_patch : malloc(bytes);
            if (patched) { memcpy(patched, data, bytes); ((float*)patched)[bytes / 4 - 1] = s_scale / f; data = patched; }
        }
    }
    /* 02/11 aa110: a pass into the horizontally stretched scene target: fp_alpha.z + 2 tells the WPOS wrapper to scale x by 3/4 more */
    if (bytes >= 16 && LB->pending.nrt && eng_obj_xx(LB->pending.rt[0]) != 1.0f) {
        if (!patched) { patched = bytes <= sizeof tl_patch ? (void*)tl_patch : malloc(bytes); if (patched) { memcpy(patched, data, bytes); data = patched; } }
        if (patched) ((float*)patched)[bytes / 4 - 2] += 2.0f;
    }
    if (!eng_stage_copy(data, bytes, &LB->pending.ps_cb_off)) bytes = 0;
    if (patched && patched != (void*)tl_patch) free(patched);
    LB->pending.ps_cb_bytes = bytes;
}
static void eng_bind_textures(void* user, const u32* textures, const rsx_be_sampler_desc* samplers, u32 mask)
{
    (void)user;
    extern int g_wp2_on; extern unsigned long long g_wp2_samp_cyc;   /* prof102: RSX_WPROF=2 times the sampler-slot lookups of 1 draw in 16 (x16) */
    static unsigned s_wk;
    const int timed = g_wp2_on && ((++s_wk) & 15u) == 0;
    for (u32 u = 0; u < RSX_BE_MAX_TEXTURES; u++) {
        LB->pending.tex[u]  = ((mask >> u) & 1u) ? textures[u] : 0;
        if (timed && ((mask >> u) & 1u)) { const unsigned long long a_ = __rdtsc(); LB->pending.samp[u] = eng_sampler_slot(&samplers[u], eng_aniso_tex(textures[u])); g_wp2_samp_cyc += (__rdtsc() - a_) * 16ull; continue; }
        LB->pending.samp[u] = ((mask >> u) & 1u) ? eng_sampler_slot(&samplers[u], eng_aniso_tex(textures[u])) : -1;
    }
}
static void eng_bind_vertex_textures(void* user, const u32* textures, const rsx_be_sampler_desc* samplers, u32 mask)
{
    (void)user;
    for (u32 u = 0; u < RSX_BE_MAX_VERTEX_TEXTURES; u++) {
        LB->pending.vtex[u]  = ((mask >> u) & 1u) ? textures[u] : 0;
        LB->pending.vsamp[u] = ((mask >> u) & 1u) ? eng_sampler_slot(&samplers[u], 0) : -1;
    }
}
/* Viewport and scissor are kept in guest pixels; eng_vp_sc scales them for
 * the target the draw lands in. */
static void eng_set_viewport(void* user, float x, float y, float w, float h)
{ (void)user; LB->pending.vp[0] = x; LB->pending.vp[1] = y; LB->pending.vp[2] = w; LB->pending.vp[3] = h; }
static void eng_set_scissor(void* user, u32 x, u32 y, u32 w, u32 h)
{ (void)user; LB->pending.sc[0] = x; LB->pending.sc[1] = y; LB->pending.sc[2] = w; LB->pending.sc[3] = h; }
static void eng_set_stencil_ref(void* user, u32 ref) { (void)user; LB->pending.stencil_ref = ref; }
static void eng_set_blend_color(void* user, const float rgba[4]) { (void)user; memcpy(LB->pending.blend, rgba, sizeof LB->pending.blend); }

static void eng_draw(void* user, rsx_topology topology, const void* vertices, u32 vertex_count,
                     u32 stride, const u32* indices, u32 index_count)
{
    (void)user;
    if (!vertex_count || !stride) return;
    if (s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return; }
    u32 vb_off = 0, ib_off = 0;
    if (!eng_stage_copy(vertices, vertex_count * stride, &vb_off)) return;
    if (indices && index_count && !eng_stage_copy(indices, index_count * (u32)sizeof(u32), &ib_off)) return;
    EngRecord* r = eng_rec_new();
    *r = LB->pending;
    r->rt_defer = s_rt_defer_next; s_rt_defer_next = 0;
    r->kind = ENG_REC_DRAW; r->topology = topology; r->vbuf = 0;
    r->vb_off = vb_off; r->stride = stride; r->vertex_count = vertex_count;
    r->ib_off = ib_off; r->index_count = indices ? index_count : 0;
    r->vis = LB->spec ? LB->vis_cur : s_vis_cur;
}
static void eng_draw_buffer(void* user, rsx_topology topology, u32 buffer, u32 vb_off,
                            u32 vertex_count, u32 stride, u32 ib_off, u32 index_count)
{
    (void)user;
    if (!vertex_count || !stride || !buffer || buffer > s_buf_count || !s_buf[buffer - 1]) return;
    if (s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return; }
    EngRecord* r = eng_rec_new();
    *r = LB->pending;
    r->rt_defer = s_rt_defer_next; s_rt_defer_next = 0;
    r->kind = ENG_REC_DRAW; r->topology = topology; r->vbuf = buffer;
    r->vb_off = vb_off; r->stride = stride; r->vertex_count = vertex_count;
    r->ib_off = ib_off; r->index_count = index_count;
    r->vis = LB->spec ? LB->vis_cur : s_vis_cur;
}

/* RT_AO: the record count right after the last opaque scene draw. The AO record is inserted there (the records
 * after it, the blended tail of the frame, are few), so that particles and additive passes are left alone. */
static void eng_rt_ao_mark(void* user) { (void)user; s_rt_mark = s_rec_count; }
static void eng_rt_ao_defer(void* user) { (void)user; s_rt_defer_next = 1; }
static void eng_rt_ao_inject(void* user, u32 surface, u32 depth, u32 job)
{
    (void)user;
    s_frame_has_rt = 1;   /* this frame is encoded on the walker thread (RT AO state is not the encode thread's) */
    if (s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return; }
    eng_lanes_flatten();   /* RSX_LANES: the insertion below indexes s_rec */
    EngRecord* r;
    { static unsigned said_ins; if (said_ins < 8 && ++said_ins)
        fprintf(stderr, "[rt] AO record: mark %u of %u records (%s)\n", s_rt_mark == ~0u ? 0u : s_rt_mark, s_rec_count, s_rt_mark != ~0u && s_rt_mark > 0 && s_rt_mark < s_rec_count ? "inserted after the last opaque draw" : "appended"); }
    if (s_rt_mark != ~0u && s_rt_mark > 0 && s_rt_mark < s_rec_count) {
        const u32 mark = s_rt_mark;
        memmove(&s_rec[mark + 1], &s_rec[mark], (size_t)(s_rec_count - mark) * sizeof s_rec[0]);
        s_rec_count++;
        /* Transparent draws of [0, mark) go after the AO record (stable order); RT_AO_VIEWS view 3 (RT off) keeps the stream as is. */
        u32 w = mark, nd = 0;
        EngRecord* tmp = NULL;
        if (rsx_rt_view_of(s_pres_n) != 3) {
            for (u32 i = 0; i < mark; i++) if (s_rec[i].rt_defer) { if (w == mark) w = i; nd++; }
            if (nd && (tmp = (EngRecord*)malloc((size_t)nd * sizeof *tmp)) != NULL) {
                u32 k = 0, o = w;
                for (u32 i = w; i < mark; i++) { if (s_rec[i].rt_defer) tmp[k++] = s_rec[i]; else s_rec[o++] = s_rec[i]; }
                w = o;                                   /* the AO record goes here: [w, mark] holds it and the deferred ones */
                memcpy(&s_rec[w + 1], tmp, (size_t)nd * sizeof *tmp);
                free(tmp);
            } else w = mark;
        }
        r = &s_rec[w];
        for (u32 i = w + 1; i <= mark; i++) s_rec[i].rt_defer = 0;
    } else r = &s_rec[s_rec_count++];
    s_rt_mark = ~0u; s_flat = s_rec_count;
    memset(r, 0, sizeof *r);
    r->kind = ENG_REC_RT_AO; r->rt[0] = surface; r->nrt = 1; r->depth = depth; r->rt_job = job;
}
static void eng_clear_color(void* user, u32 surface, const float rgba[4])
{
    (void)user;
    s_clear_argb = ((u32)(rgba[3] * 255.0f + 0.5f) << 24) | ((u32)(rgba[0] * 255.0f + 0.5f) << 16) |
                   ((u32)(rgba[1] * 255.0f + 0.5f) <<  8) |  (u32)(rgba[2] * 255.0f + 0.5f);
    if (s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return; }
    EngRecord* r = eng_rec_new();
    memset(r, 0, sizeof *r);
    r->kind = ENG_REC_CLEAR_COLOR; r->rt[0] = surface; r->nrt = 1;
    memcpy(r->clear_rgba, rgba, sizeof r->clear_rgba);
}
static void eng_clear_depth_stencil(void* user, u32 depth, u32 flags, float depth_value, u8 stencil)
{
    (void)user;
    if (s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return; }
    EngRecord* r = eng_rec_new();
    memset(r, 0, sizeof *r);
    r->kind = ENG_REC_CLEAR_DS; r->depth = depth; r->clear_flags = flags;
    r->clear_depth = depth_value; r->clear_stencil = stencil;
}
static void eng_clear_depth_stencil_rect(void* user, u32 depth, u32 flags, float depth_value, u8 stencil,
                                         u32 x, u32 y, u32 w, u32 h)
{
    (void)user;
    if (!w || !h) return;
    if (s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return; }
    { const int ks = eng_shadow_k(g_rsx_pass_cw, g_rsx_pass_ch);
      if (ks > 1 && eng_obj_scaled(depth)) scf_rect(&x, &y, &w, &h, s_scale * (float)ks);
      else if (eng_obj_scaled(depth)) {
          /* 02/11 aa110: the depth of the horizontally stretched scene: a guest rect is x*scale*xx wide on the host (else the right quarter keeps stale depth and nothing passes the test there) */
          const EngObj* zo = eng_owner(depth);
          if (zo && zo->xx > 0.0f && x + w <= 960 && y + h <= 704) { const u32 x0 = scf_pos(x, s_scale * zo->xx), x1 = scf_pos(x + w, s_scale * zo->xx); const u32 y0 = sc_pos(y), y1 = sc_pos(y + h); x = x0; w = x1 - x0; y = y0; h = y1 - y0; }
          else sc_rect(&x, &y, &w, &h); } }
    EngRecord* r = eng_rec_new();
    memset(r, 0, sizeof *r);
    r->kind = ENG_REC_CLEAR_DS_RECT; r->depth = depth; r->clear_flags = flags;
    r->clear_depth = depth_value; r->clear_stencil = stencil;
    r->clear_rect[0] = x; r->clear_rect[1] = y; r->clear_rect[2] = w; r->clear_rect[3] = h;
}

/* ---- encoding -------------------------------------------------------------- */

static D3D12_RESOURCE_BARRIER s_bar[32]; static u32 s_nbar;
static void bar_flush(void)
{
    if (s_nbar) { CALL(s_list, ResourceBarrier, s_nbar, s_bar); s_gt_bar_calls++; s_gt_bars += s_nbar; s_nbar = 0; }
}
static void res_transition(ID3D12Resource* res, D3D12_RESOURCE_STATES* state, D3D12_RESOURCE_STATES to)
{
    if (*state == to) return;
    if (s_nbar == 32) bar_flush();
    D3D12_RESOURCE_BARRIER* b = &s_bar[s_nbar++];
    memset(b, 0, sizeof *b);
    b->Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b->Transition.pResource = res;
    b->Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b->Transition.StateBefore = *state;
    b->Transition.StateAfter = to;
    *state = to;
}
static void obj_transition(u32 handle, D3D12_RESOURCE_STATES to)
{
    EngObj* o = eng_owner(handle);
    if (o) res_transition(o->res, &o->state, to);
}

/* What the open list has bound, so a run of similar draws rebinds little. */
static ID3D12PipelineState* s_cur_pso; static int s_cur_helper_rs;
static u32 s_cur_rt[RSX_BE_MAX_COLOR_TARGETS], s_cur_nrt, s_cur_depth; static int s_cur_targets_valid;
static D3D_PRIMITIVE_TOPOLOGY s_cur_topo;
static D3D12_CPU_DESCRIPTOR_HANDLE s_srv_last_src[ENG_SRV_TABLE]; static D3D12_GPU_DESCRIPTOR_HANDLE s_srv_last_gpu; static int s_srv_last_valid;
typedef struct { int slots[ENG_SRV_TABLE]; D3D12_GPU_DESCRIPTOR_HANDLE gpu; } EngSmpTable;
static EngSmpTable s_smp_tab[ENG_SMP_PER_SLOT / ENG_SRV_TABLE]; static u32 s_smp_ntab;
static ID3D12Resource* s_zero_cb;            /* bound where a draw stages no constants */
static D3D12_RESOURCE_STATES s_offscreen_state = D3D12_RESOURCE_STATE_RENDER_TARGET;

static void eng_reset_bind_state(void)
{
    s_cur_pso = NULL; s_cur_helper_rs = 0; s_cur_targets_valid = 0;
    s_cur_topo = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
    s_srv_last_valid = 0; s_smp_ntab = 0;
    s_srv_used = s_smp_used = 0; s_q_used = 0;
}

static const char* pass_kind_name(int k)
{
    static const char* const n[] = { "draws", "clear colour", "clear depth", "copy", "depth resolve", "uploads" };
    return k >= 0 && k < 6 ? n[k] : "?";
}
static void eng_pass_collect(u32 slot, int report)
{
    enum { N = 256 };
    static struct { u8 kind, nrt; u16 w, h; u32 fmt; double ms; u64 n, draws; } st[N];
    static u32 nst;
    if (s_pheap && s_pass_n[slot]) {
        const u32 base = slot * (ENG_PASS_MAX + 1u), np = s_pass_n[slot];
        u64* p = NULL; D3D12_RANGE rg = { base * 8u, (base + np + 1u) * 8u };
        if (SUCCEEDED(CALL(s_pread, Map, 0, &rg, (void**)&p))) {
            for (u32 i = 0; i < np; i++) {
                const u64 a = p[base + i], b = p[base + i + 1];
                const double ms = b > a ? (double)(b - a) * 1000.0 / (double)s_ts_freq : 0.0;
                const EngPassDesc* d = &s_pass[slot][i];
                if (d->kind == 4) s_zm_gpu_ms[d->garea <= 512u * 512u ? 0 : 1] += ms;   /* log v2 zmix */
                u32 k;
                for (k = 0; k < nst; k++)
                    if (st[k].kind == d->kind && st[k].w == d->w && st[k].h == d->h && st[k].fmt == d->fmt && st[k].nrt == d->nrt) break;
                if (k == nst) { if (nst == N) continue; nst++; st[k].kind = d->kind; st[k].w = d->w; st[k].h = d->h; st[k].fmt = d->fmt; st[k].nrt = d->nrt; st[k].ms = 0; st[k].n = st[k].draws = 0; }
                st[k].ms += ms; st[k].n++; st[k].draws += d->draws;
            }
            D3D12_RANGE wr = {0, 0}; CALL(s_pread, Unmap, 0, &wr);
        }
        s_pass_n[slot] = 0;
    }
    if (report && nst) {
        double tot = 0; for (u32 k = 0; k < nst; k++) tot += st[k].ms;
        fprintf(stderr, "[gpu-pass] %.1f ms of passes in 5 s; the largest:\n", tot);
        for (int shown = 0; shown < 12; shown++) {
            int best = -1;
            for (u32 k = 0; k < nst; k++) if (st[k].n && (best < 0 || st[k].ms > st[best].ms)) best = (int)k;
            if (best < 0 || st[best].ms <= 0) break;
            fprintf(stderr, "[gpu-pass]   %5.1f%% %8.1f ms  %-13s %4ux%-4u fmt %3u x%u  %6llu passes, %7llu draws\n",
                    tot > 0 ? 100.0 * st[best].ms / tot : 0.0, st[best].ms, pass_kind_name(st[best].kind),
                    st[best].w, st[best].h, st[best].fmt, st[best].nrt,
                    (unsigned long long)st[best].n, (unsigned long long)st[best].draws);
            st[best].n = 0;   /* shown */
        }
        nst = 0;
    }
}


static void eng_gpu_time_collect(u32 slot)
{
    static double busy_ms, wait_ms, other_ms, queue_ms; static unsigned n, n_wait, n_other;
    static LARGE_INTEGER last; LARGE_INTEGER qf, now;
    if (!s_tsread || !s_ts_slot[slot].fence) return;
    QueryPerformanceFrequency(&qf); QueryPerformanceCounter(&now);
    u64* p = NULL; D3D12_RANGE rg = { slot * 16u, slot * 16u + 16u };
    if (SUCCEEDED(CALL(s_tsread, Map, 0, &rg, (void**)&p))) {
        const u64 t0 = p[slot * 2], t1 = p[slot * 2 + 1];
        D3D12_RANGE wr = {0, 0}; CALL(s_tsread, Unmap, 0, &wr);
        const double gpu_ms = t1 > t0 ? (double)(t1 - t0) * 1000.0 / (double)s_ts_freq : 0.0;
        /* Where the GPU clock was when the CPU executed the list: one
         * calibration a submit is cheap next to the submit itself. */
        u64 g_now = 0, c_now = 0;
        g_rsx_gpu_busy_ms_total += gpu_ms;
        if (s_gt_report && SUCCEEDED(CALL(s_queue, GetClockCalibration, &g_now, &c_now))) {
            const double cpu_ago_ms = (double)((long long)c_now - s_ts_slot[slot].cpu_exec.QuadPart) * 1000.0 / (double)qf.QuadPart;
            const double gpu_ago_ms = (double)((long long)g_now - (long long)t0) * 1000.0 / (double)s_ts_freq;
            const double q = cpu_ago_ms - gpu_ago_ms;   /* exec -> GPU start */
            if (q > 0 && q < 1000) queue_ms += q;
        }
        busy_ms += gpu_ms; n++;
        /* RSX_GPU_TS_DUMP=<csv>: where each list really started and finished on the GPU, on the QPC clock (us), for the
         * pacing analysis (an.py / pace115). cpu_exec = when the CPU handed it over. */
        if (s_gts_fp && t1 >= t0 && t0) {
            u64 g2 = 0, c2 = 0;
            if (SUCCEEDED(CALL(s_queue, GetClockCalibration, &g2, &c2))) {
                const double us = 1e6 / (double)qf.QuadPart;
                const double e1 = ((double)c2 - ((double)(long long)(g2 - t1)) * (double)qf.QuadPart / (double)s_ts_freq) * us;
                const double e0 = ((double)c2 - ((double)(long long)(g2 - t0)) * (double)qf.QuadPart / (double)s_ts_freq) * us;
                fprintf(s_gts_fp, "%llu,%d,%.1f,%.1f,%.1f\n", (unsigned long long)s_ts_slot[slot].fence, s_ts_slot[slot].kind,
                        (double)s_ts_slot[slot].cpu_exec.QuadPart * us, e0, e1);
                if ((++s_gts_n & 255) == 0) fflush(s_gts_fp);
            }
        }
        eng_pass_collect(slot, 0);
        if (s_ts_slot[slot].kind) { wait_ms += gpu_ms; n_wait++; } else { other_ms += gpu_ms; n_other++; }
    }
    s_ts_slot[slot].fence = 0;
    if (!last.QuadPart) last = now;
    const double secs = (double)(now.QuadPart - last.QuadPart) / (double)qf.QuadPart;
    if (secs >= 5.0 && !s_gt_report) { busy_ms = wait_ms = other_ms = queue_ms = 0; n = n_wait = n_other = 0; last = now; }
    else if (secs >= 5.0) {
        fprintf(stderr, "[gpu-time] %.0f submits/s, GPU busy %.1f ms/s; waited-on submits %.0f/s, %.2f ms GPU each; others %.0f/s, %.2f ms each; exec->GPU start %.2f ms mean\n",
                n / secs, busy_ms / secs, n_wait / secs, n_wait ? wait_ms / n_wait : 0.0,
                n_other / secs, n_other ? other_ms / n_other : 0.0, n ? queue_ms / n : 0.0);
        fprintf(stderr, "[gpu-time] per second: %.0f draws, %.0f PSO changes, %.0f ResourceBarrier calls (%.0f transitions), %.0f occlusion queries, %.0f texture uploads\n",
                s_gt_draws / secs, s_gt_pso / secs, s_gt_bar_calls / secs, s_gt_bars / secs, s_gt_queries / secs, s_gt_copies / secs);
        fprintf(stderr, "[gpu-time] draws skipped for want of a pipeline: %.0f/s\n", s_gt_nopso / secs);
        s_gt_nopso = 0;
        fprintf(stderr, "[gpu-time] own-target snapshots: %.0f/s brought up to date by %.0f copy draws/s, %.0f/s copied whole\n",
                s_snap_incr_n / secs, s_snap_copy_draws / secs, s_snap_full_n / secs);
        s_snap_incr_n = s_snap_full_n = s_snap_copy_draws = 0;
        s_gt_draws = s_gt_pso = s_gt_bar_calls = s_gt_bars = s_gt_queries = s_gt_copies = 0;
        eng_pass_collect(slot, 1);
        busy_ms = wait_ms = other_ms = queue_ms = 0; n = n_wait = n_other = 0; last = now;
    }
}

/* Bumped by every new command list: a caller that took descriptor tables
 * can tell whether a flush started another list under it. */
static u32 s_list_gen;
static void eng_list_begin(void)
{
    s_list_gen++;
    eng_gpu_time_collect(s_slot);
    CALL0(s_alloc[s_slot], Reset);
    CALL(s_list, Reset, s_alloc[s_slot], NULL);
    s_list_open = 1;
    if (s_tsheap) CALL(s_list, EndQuery, s_tsheap, D3D12_QUERY_TYPE_TIMESTAMP, s_slot * 2u);
    s_smp_cur = s_smp_gpu[s_slot];
    ID3D12DescriptorHeap* heaps[2] = { s_srv_gpu, s_smp_cur };
    CALL(s_list, SetDescriptorHeaps, 2, heaps);
    CALL(s_list, SetGraphicsRootSignature, s_rootsig);
    eng_reset_bind_state();
}

static void eng_use_main_rootsig(void)
{
    if (!s_cur_helper_rs) return;
    CALL(s_list, SetGraphicsRootSignature, s_rootsig);
    s_cur_helper_rs = 0; s_cur_pso = NULL; s_cur_targets_valid = 0;
}
static void eng_use_helper_rootsig(void)
{
    if (s_cur_helper_rs) return;
    CALL(s_list, SetGraphicsRootSignature, s_helper_rootsig);
    s_cur_helper_rs = 1; s_cur_pso = NULL; s_cur_targets_valid = 0;
}

/* The list's descriptor rings ran dry mid-frame: submit what is encoded,
 * wait, and continue on the same slot with fresh rings. */
static void eng_submit_list(u64* out_fence);
static void eng_hard_flush(void)
{
    u64 f = 0;
    eng_submit_list(&f);
    fence_wait(f);
    /* What was retired with records waiting was given the submit those
     * records would go in (eng_retire_fence): that is this partial one now,
     * and the rest of the records, still naming it, go in the next. Freed at
     * this fence, a texture went while the GPU still sampled it -- a device
     * hang (DEVICE_HUNG, no page fault) in Ratchet ToD's metropolis. */
    if (f) {
        for (u32 i = 0; i < s_retired_count; i++) if (s_retired_fence[i] >= f) s_retired_fence[i] = f + 1;
        for (u32 i = 0; i < s_buf_count; i++) if (s_buf_retired[i] && s_buf_retired_fence[i] >= f) s_buf_retired_fence[i] = f + 1;
    }
    /* The queries of the partial list, synchronously. */
    if (s_q_used && s_qread) {
        D3D12_RANGE rg = { (SIZE_T)s_slot * ENG_QUERIES_PER_SLOT * 8u, ((SIZE_T)s_slot * ENG_QUERIES_PER_SLOT + s_q_used) * 8u };
        u64* p = NULL;
        if (SUCCEEDED(CALL(s_qread, Map, 0, &rg, (void**)&p))) {
            for (u32 i = 0; i < s_q_used; i++) s_vis_count[s_q_vis[i] & ENG_Q_INDEX] += eng_q_count(s_q_vis[i], p[s_slot * ENG_QUERIES_PER_SLOT + i]);
            D3D12_RANGE wr = {0, 0}; CALL(s_qread, Unmap, 0, &wr);
        }
    }
    static unsigned n = 0;
    if (n++ < 8) fprintf(stderr, "[rsx engine/d3d12] descriptor ring full: mid-frame flush (#%u)\n", n);
    eng_list_begin();
}

static D3D12_GPU_DESCRIPTOR_HANDLE eng_srv_table(const D3D12_CPU_DESCRIPTOR_HANDLE* src, u32 n)
{
    if (s_srv_used + n > ENG_SRV_PER_SLOT) { static unsigned k; if (k++ % 64 == 0) fprintf(stderr, "[rsx engine/d3d12] SRV ring full (#%u)\n", k); eng_hard_flush(); }
    const u32 base = s_slot * ENG_SRV_PER_SLOT + s_srv_used;
    s_srv_used += n;
    D3D12_CPU_DESCRIPTOR_HANDLE dst = cpu_handle(s_srv_gpu, s_srv_step, base);
    UINT dst_size = n;
    CALL(s_dev, CopyDescriptors, 1, &dst, &dst_size, n, src, NULL, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return gpu_handle(s_srv_gpu, s_srv_step, base);
}

/* Bind a free spare sampler heap on the open list (see s_smp_spare). Every
 * descriptor table bound so far is invalid after SetDescriptorHeaps; the
 * draw path rebinds all of them once s_cur_targets_valid is clear. */
static int eng_smp_switch(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("RSX_SMP_SWITCH"); on = (e && *e == '0') ? 0 : 1; }
    if (!on || !s_list_open) return 0;
    const u64 done = CALL0(s_fence, GetCompletedValue);
    int pick = -1;
    for (u32 i = 0; i < s_smp_spare_n; i++)
        if (!s_smp_spare_inlist[i] && s_smp_spare_fence[i] <= done) { pick = (int)i; break; }
    if (pick < 0) {
        if (s_smp_spare_n >= ENG_SMP_SPARES) return 0;
        ID3D12DescriptorHeap* h = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, ENG_SMP_PER_SLOT, 1);
        if (!h) return 0;
        pick = (int)s_smp_spare_n++;
        s_smp_spare[pick] = h;
        { static unsigned n; if (n++ < 8) fprintf(stderr, "[rsx engine/d3d12] sampler heap spare #%u created\n", s_smp_spare_n); }
    }
    s_smp_spare_inlist[pick] = 1;
    s_smp_cur = s_smp_spare[pick];
    ID3D12DescriptorHeap* heaps[2] = { s_srv_gpu, s_smp_cur };
    CALL(s_list, SetDescriptorHeaps, 2, heaps);
    s_smp_used = 0; s_smp_ntab = 0;
    s_cur_targets_valid = 0;
    return 1;
}

static D3D12_GPU_DESCRIPTOR_HANDLE eng_smp_table(const int* slots)
{
    /* RSX_SAMPLER_HASH=0: the old linear memcmp scan. The hash maps the tuple to
     * an index in s_smp_tab; it is rebuilt-empty whenever s_smp_ntab went back
     * below the number of entries it holds (ring reset / heap switch). */
    static int hash_on = -1;
    if (hash_on < 0) { const char* e = getenv("RSX_SAMPLER_HASH"); hash_on = !(e && e[0] == '0'); }
    u32 hslot = 0;
    static u16 hidx[1024];          /* entry index + 1, 0 = empty */
    static u32 hn;                  /* entries of s_smp_tab inserted so far */
    if (hash_on) {
        if (s_smp_ntab < hn || hn >= 512) { memset(hidx, 0, sizeof hidx); hn = 0; }
        while (hn < s_smp_ntab) {   /* entries added by the other path / before enabling */
            u32 hh = 2166136261u;
            for (u32 i = 0; i < ENG_SRV_TABLE; i++) { hh ^= (u32)s_smp_tab[hn].slots[i]; hh *= 16777619u; }
            hh ^= hh >> 15;
            u32 p = hh & 1023u;
            while (hidx[p]) p = (p + 1) & 1023u;
            hidx[p] = (u16)(hn + 1); hn++;
        }
        u32 hh = 2166136261u;
        for (u32 i = 0; i < ENG_SRV_TABLE; i++) { hh ^= (u32)slots[i]; hh *= 16777619u; }
        hh ^= hh >> 15;
        hslot = hh & 1023u;
        while (hidx[hslot]) {
            const u32 e = (u32)hidx[hslot] - 1u;
            if (memcmp(s_smp_tab[e].slots, slots, sizeof(int) * ENG_SRV_TABLE) == 0) return s_smp_tab[e].gpu;
            hslot = (hslot + 1) & 1023u;
        }
        /* Miss: the new entry is appended below and picked up by the catch-up loop next call. */
    } else {
        for (u32 i = 0; i < s_smp_ntab; i++)
            if (memcmp(s_smp_tab[i].slots, slots, sizeof(int) * ENG_SRV_TABLE) == 0) return s_smp_tab[i].gpu;
    }
    (void)hslot;
    if (s_smp_used + ENG_SRV_TABLE > ENG_SMP_PER_SLOT && !eng_smp_switch()) {
        static unsigned k; if (k++ % 64 == 0) fprintf(stderr, "[rsx engine/d3d12] sampler ring full (#%u)\n", k);
        eng_hard_flush();
    }
    const u32 base = s_smp_used;
    s_smp_used += ENG_SRV_TABLE;
    D3D12_CPU_DESCRIPTOR_HANDLE src[ENG_SRV_TABLE];
    for (u32 i = 0; i < ENG_SRV_TABLE; i++) src[i] = cpu_handle(s_smp_cpu, s_smp_step, slots[i] >= 0 ? (u32)slots[i] + 1 : 0);
    D3D12_CPU_DESCRIPTOR_HANDLE dst = cpu_handle(s_smp_cur, s_smp_step, base);
    UINT dst_size = ENG_SRV_TABLE;
    CALL(s_dev, CopyDescriptors, 1, &dst, &dst_size, ENG_SRV_TABLE, src, NULL, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    /* A flush/heap switch above may have emptied the list: the hash then holds stale entries. */
    if (s_smp_ntab == 0 && hn) { memset(hidx, 0, sizeof hidx); hn = 0; }
    EngSmpTable* t = &s_smp_tab[s_smp_ntab++];
    memcpy(t->slots, slots, sizeof t->slots);
    t->gpu = gpu_handle(s_smp_cur, s_smp_step, base);
    return t->gpu;
}

/* A helper pass over a whole colour target: dst bound, src sampled at t0. */
/* The source drawn over a w x h target; or, with at (rsx_present_layout's
 * viewport and rectangle), over that viewport and only within that rectangle. */
static void eng_fullscreen_pass_at(ID3D12PipelineState* pso, D3D12_CPU_DESCRIPTOR_HANDLE rtv, u32 w, u32 h,
                                   D3D12_CPU_DESCRIPTOR_HANDLE src_srv, const float* at_vp, const int* at_rect)
{
    /* The table first: a full ring flushes the list (eng_hard_flush), and
     * the new one has none of the state set before it -- a draw with no
     * pipeline or target on it hangs the GPU. */
    const D3D12_GPU_DESCRIPTOR_HANDLE tab = eng_srv_table(&src_srv, 1);
    eng_use_helper_rootsig();
    CALL(s_list, OMSetRenderTargets, 1, &rtv, FALSE, NULL);
    D3D12_VIEWPORT vp = { 0.0f, 0.0f, (float)w, (float)h, 0.0f, 1.0f };
    D3D12_RECT sc = { 0, 0, (LONG)w, (LONG)h };
    if (at_vp && at_rect) {
        vp.TopLeftX = at_vp[0]; vp.TopLeftY = at_vp[1]; vp.Width = at_vp[2]; vp.Height = at_vp[3];
        sc.left = at_rect[0]; sc.top = at_rect[1]; sc.right = at_rect[0] + at_rect[2]; sc.bottom = at_rect[1] + at_rect[3];
    }
    CALL(s_list, RSSetViewports, 1, &vp);
    CALL(s_list, RSSetScissorRects, 1, &sc);
    CALL(s_list, SetPipelineState, pso);
    CALL(s_list, IASetPrimitiveTopology, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    s_cur_topo = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    CALL(s_list, SetGraphicsRootDescriptorTable, 0, tab);
    CALL(s_list, DrawInstanced, 3, 1, 0, 0);
    s_cur_pso = NULL; s_cur_targets_valid = 0;
}

static void eng_fullscreen_pass(ID3D12PipelineState* pso, D3D12_CPU_DESCRIPTOR_HANDLE rtv, u32 w, u32 h,
                                D3D12_CPU_DESCRIPTOR_HANDLE src_srv)
{
    eng_fullscreen_pass_at(pso, rtv, w, h, src_srv, NULL, NULL);
}

/* A record's viewport and scissor (guest pixels) for a target of tw x th
 * host pixels, scaled if the target is. Empty ones cover the target. */
static void eng_vp_sc(const EngRecord* r, float f, float fx, u32 tw, u32 th, D3D12_VIEWPORT* vp, D3D12_RECT* sc)
{
    const int scaled = f != 1.0f;
    vp->TopLeftX = r->vp[0] * f * fx; vp->TopLeftY = r->vp[1] * f;
    vp->Width = r->vp[2] * f * fx;    vp->Height = r->vp[3] * f;
    vp->MinDepth = 0.0f; vp->MaxDepth = 1.0f;
    if (vp->Width <= 0.0f || vp->Height <= 0.0f) { vp->TopLeftX = vp->TopLeftY = 0.0f; vp->Width = (float)tw; vp->Height = (float)th; }
    u32 x = r->sc[0], y = r->sc[1], w = r->sc[2], h = r->sc[3];
    if (w && h && scaled) { if (fx != 1.0f) { const u32 x0 = scf_pos(x, f * fx), x1 = scf_pos(x + w, f * fx); const u32 y0 = scf_pos(y, f), y1 = scf_pos(y + h, f); x = x0; w = x1 - x0; y = y0; h = y1 - y0; } else scf_rect(&x, &y, &w, &h, f); }
    if (w && h) { sc->left = (LONG)x; sc->top = (LONG)y; sc->right = (LONG)(x + w); sc->bottom = (LONG)(y + h); }
    else { sc->left = 0; sc->top = 0; sc->right = (LONG)tw; sc->bottom = (LONG)th; }
}

static ID3D12PipelineState* eng_pso_for(u32 pipeline, int cls, int unscaled);   /* below */

/* Where a draw's vertices and indices are: in a buffer of their own or
 * staged. vleft: the bytes from its first vertex to the end of what holds
 * them. */
static int eng_draw_buffers(const EngRecord* r, D3D12_GPU_VIRTUAL_ADDRESS* vb, D3D12_GPU_VIRTUAL_ADDRESS* ib, u32* vleft)
{
    if (r->vbuf) {
        const u32 b = r->vbuf - 1;
        if (!s_buf[b]) return 0;
        const D3D12_GPU_VIRTUAL_ADDRESS va = s_buf_page[b] ? s_vpage[s_buf_page[b] - 1].va + s_buf_off[b]
                                                           : CALL0(s_buf[b], GetGPUVirtualAddress);
        const u32 bytes = s_buf_bytes[b];
        *vb = va + r->vb_off; *ib = va + r->ib_off;
        *vleft = r->vb_off < bytes ? bytes - r->vb_off : 0;
        return 1;
    }
    const EngStage* s = eng_stage_at(r->vb_off);
    const EngStage* si = r->index_count ? eng_stage_at(r->ib_off) : s;
    if (!s || !si) return 0;
    const u32 lo = r->vb_off - s->base;
    *vb = s->va + lo; *ib = si->va + (r->ib_off - si->base);
    *vleft = lo < s->used ? s->used - lo : 0;
    return 1;
}

/* A draw's staged constants, or the zeroes where it staged none. */
static D3D12_GPU_VIRTUAL_ADDRESS eng_cb_va(u32 off, u32 bytes)
{
    const EngStage* s = bytes ? eng_stage_at(off) : NULL;
    return s ? s->va + (off - s->base) : CALL0(s_zero_cb, GetGPUVirtualAddress);
}

static void eng_encode_upload(const EngRecord* r)
{
    EngObj* o = eng_owner(r->depth);
    const EngStage* stage = eng_stage_at(r->up_off);
    if (!o || !stage) return;
    res_transition(o->res, &o->state, D3D12_RESOURCE_STATE_COPY_DEST);
    bar_flush();
    D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
    dst.pResource = o->res; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dst.SubresourceIndex = r->up_sub;
    src.pResource = stage->res; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Offset = r->up_off - stage->base;
    src.PlacedFootprint.Footprint.Format = o->fmt;
    u32 w = r->up_w, h = r->up_h;
    if (o->fmt == DXGI_FORMAT_BC1_UNORM || o->fmt == DXGI_FORMAT_BC2_UNORM || o->fmt == DXGI_FORMAT_BC3_UNORM) {
        w = (w + 3u) & ~3u; h = (h + 3u) & ~3u;
    }
    src.PlacedFootprint.Footprint.Width = w;
    src.PlacedFootprint.Footprint.Height = h;
    src.PlacedFootprint.Footprint.Depth = 1;
    src.PlacedFootprint.Footprint.RowPitch = r->up_pitch;
    eng_check_copy(src.PlacedFootprint.Offset, r->up_pitch, "texture upload");
    CALL(s_list, CopyTextureRegion, &dst, 0, 0, 0, &src, NULL);
    s_gt_copies++;
}

/* ---- MSAA ----------------------------------------------------------------------
 *
 * RSX_AA=msaa2|msaa4|msaa8 (the Graphics Settings page's Anti-Aliasing). A
 * pass is drawn multisampled when it has a real depth target and every
 * attachment is a full-size target (guest 960x540 or more): the 3D scene,
 * not the post-process chain or the UI. Its attachments get multisampled
 * twins at that first pass (EngObj.ms); every read of a target -- a texture,
 * a snapshot, a copy, the present, a readback -- still reads the
 * single-sampled resource, which the twin is resolved into (only the region
 * its draws touched) before anything reads it: when a pass reads it as a
 * texture, before any other record touches it, and at the end of a list.
 * Depth is not resolved: the depth-to-colour copies read a twin's first
 * sample (ps_depth_ms). A pass without depth into a target with a twin
 * draws single-sampled and marks the twin stale; the next 3D pass refreshes
 * it from the resolved one. Occlusion counts are divided by the samples. */
static u32 s_ms_n = 1;          /* samples in use: 1, 2, 4, 8 */
static ID3D12GraphicsCommandList1* s_list1;   /* ResolveSubresourceRegion */
static u32 s_ms_open[64]; static u32 s_ms_open_n;   /* twins with draws not yet resolved */
static int eng_ms_want(void)
{
    const int m = rsx_aa_mode();
    return m == 2 || m == 4 || m == 8 ? m : 1;
}
static int eng_ms_format_ok(DXGI_FORMAT f, u32 n)
{
    D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS q = {0};
    q.Format = f; q.SampleCount = n;
    return SUCCEEDED(CALL(s_dev, CheckFeatureSupport, D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &q, sizeof q)) && q.NumQualityLevels > 0;
}
static int eng_ms_ok(const EngObj* o)
{
    if (s_ms_n < 2 || !o || o->retired || !o->res || o->ms_never) return 0;
    if (o->kind != OBJ_COLOR && o->kind != OBJ_DEPTH) return 0;
    if (o->gw < 960 || o->gh < 540) return 0;
    static struct { DXGI_FORMAT f; u32 n; int ok; } cache[16]; static u32 nc;
    const DXGI_FORMAT f = o->kind == OBJ_DEPTH ? ENG_DEPTH_FMT : o->fmt;
    for (u32 i = 0; i < nc; i++) if (cache[i].f == f && cache[i].n == s_ms_n) return cache[i].ok;
    const int ok = eng_ms_format_ok(f, s_ms_n);
    if (nc < 16) { cache[nc].f = f; cache[nc].n = s_ms_n; cache[nc].ok = ok; nc++; }
    return ok;
}
/* ---- rissa127 (06/11): RSX_MS_TWIN_FIT=1 (default off) -------------------------------------------------------------------------
 * In a fight at Rykan V the game moves its 3D scene to new scratch addresses and sizes (guest 960/992/1024 x 704/736/800, depth
 * 1280x800/736): 9 new twins (200-560 MB each) in one second, video memory over the budget, frames of 100-380 ms; the old scene's
 * twins (a 720 MB depth among them) stay idle. With FIT a twin may be LARGER than its target: the pool (RSX_MS_TWIN_POOL) and the
 * parked list (RSX_MS_TWIN_PARK) hand over a twin whose size covers the target and whose bytes are at most RSX_MS_TWIN_FIT_PCT %
 * (170) of an exact one; new twins are made with sides rounded up to RSX_MS_TWIN_FIT_ROUND (256) host pixels, so the sizes the
 * game alternates share one twin. Draws land in the twin's top-left w x h (as in the target); resolves are region resolves. */
static int s_ms_fit = -1; static u32 s_ms_fit_pct = 170, s_ms_fit_round = 256;
static void eng_ms_fit_setup(void)
{
    if (s_ms_fit >= 0) return;
    const char* e = getenv("RSX_MS_TWIN_FIT"); s_ms_fit = (e && e[0] == '1') ? 1 : 0;
    e = getenv("RSX_MS_TWIN_FIT_PCT"); if (e && atoi(e) >= 100) s_ms_fit_pct = (u32)atoi(e);
    e = getenv("RSX_MS_TWIN_FIT_ROUND"); if (e && atoi(e) >= 1) s_ms_fit_round = (u32)atoi(e);
    if (s_ms_fit) fprintf(stderr, "[ms-twin] fit on: a twin up to %u%% of the bytes may serve a smaller target, new twins rounded up to %u px\n", s_ms_fit_pct, s_ms_fit_round);
}
static u32 ms_tw(const EngObj* o) { return o->ms_w ? o->ms_w : o->w; }
static u32 ms_th(const EngObj* o) { return o->ms_h ? o->ms_h : o->h; }
static int eng_ms_fits(const EngObj* o, u32 tw, u32 th)
{
    if (tw == o->w && th == o->h) return 1;
    eng_ms_fit_setup();
    if (!s_ms_fit || tw < o->w || th < o->h) return 0;
    return (u64)tw * th * 100ull <= (u64)o->w * o->h * s_ms_fit_pct;
}
/* ---- 04/11 vram106: twin accounting (RSX_MS_TWIN_LOG=1), twin pool (RSX_MS_TWIN_POOL=1, =2 checks) ----------
 * At 4K x8 the twins are ~4.6 of ~7.6 GB of the engine's objects (a depth twin of 3840x3072 is 720 MB); a twin made while the
 * video memory is over DXGI's budget blocks the encode thread for 0.5-5 s (CreateCommittedResource under eviction). Pool: a
 * target that needs a twin first takes over that of another target of the same size and format which no draw of the rest of
 * the frame names (looked up in the records before the encode; only in the last batch of a frame, the present's); the donor's
 * twin is flushed first (colour: resolved into its target; depth: copied into it if newer), so the donor, at its next pass, gets
 * a twin that starts stale -- cleared (depth) / refreshed from its target (colour), as any twin made late. Twins idle for
 * RSX_MS_TWIN_IDLE frames (default 300) are flushed and released (after the GPU is done with them). Encode thread only. */
static u32 s_enc_frame;                     /* frames (presents) encoded */
static u32 s_mt_alive[128]; static u32 s_mt_alive_n;   /* handles that have (had) a twin */
static int s_mt_log = -1, s_mt_pool = -1, s_mt_pct, s_mt_final;
static u32 s_mt_idle_n = 300;
static u32 s_mt_ri, s_mt_base;              /* record being encoded (batch-local); draws before this batch in the frame */
static const EngRecord* s_mt_rec; static u32 s_mt_rn;
static u32 s_mt_lastidx[ENG_MAX_OBJECTS];   /* per object: 1 + index of the last draw record of this batch that names it */
static struct { u32 w, h, last; } s_mt_fbl[8]; static u32 s_mt_fbl_n;   /* draws without a depth target, by pass size (fallback depth) */
static int s_mt_scanned;
static struct { ID3D12Resource* res; u64 fence; } s_mt_defer[64]; static u32 s_mt_defer_n;
static u32 s_mt_lastkey_d, s_mt_lastkey_c;
static struct {
    u32 frames, made[2], dropped[2], taken[2], idlerel[2], flush_sync, flush_res, viol_later, viol_check, refused;   /* [0] colour [1] depth */
    double make_ms[2], make_ms_max;
    u32 maxalive[2], maxused[2], maxneed[2]; u64 maxalive_mb, maxused_mb, maxneed_mb, alive_sum_mb;
    u32 passes, draws;
    struct { u32 kind, w, h, fmt, alive, used, need; } key[16]; u32 nkey;
    ULONGLONG t0;
} s_mtw;
static u64 eng_px_bytes(DXGI_FORMAT f);
static int eng_vram_query(unsigned long long* used, unsigned long long* budget);
static void eng_mt_setup(void)
{
    if (s_mt_log >= 0) return;
    const char* e = getenv("RSX_MS_TWIN_LOG"); s_mt_log = (e && e[0] && e[0] != '0') ? 1 : 0;
    e = getenv("RSX_MS_TWIN_POOL"); s_mt_pool = e ? atoi(e) : 0; if (s_mt_pool < 0 || s_mt_pool > 2) s_mt_pool = 0;
    e = getenv("RSX_MS_TWIN_IDLE"); if (e && atoi(e) > 0) s_mt_idle_n = (u32)atoi(e);
    e = getenv("RSX_MS_TWIN_POOL_PCT"); s_mt_pct = e ? atoi(e) : 0;   /* take over only while the video memory is above this % of the budget (0: always) */
    if (s_mt_pool || s_mt_log) fprintf(stderr, "[ms-twin] pool %s, idle release after %u frames, video-memory threshold %d%%, log %d\n",
                                       s_mt_pool == 2 ? "CHECK" : s_mt_pool ? "on" : "off", s_mt_idle_n, s_mt_pct, s_mt_log);
}
static u64 eng_mt_bytes(const EngObj* o)
{
    return (u64)ms_tw(o) * ms_th(o) * eng_px_bytes(o->kind == OBJ_DEPTH ? ENG_DEPTH_RES_FMT : o->fmt) * s_ms_n;
}
static void eng_mt_alive_add(u32 h)
{
    for (u32 i = 0; i < s_mt_alive_n; i++) if (s_mt_alive[i] == h) return;
    if (s_mt_alive_n < 128) s_mt_alive[s_mt_alive_n++] = h;
}
/* ---- hitch108 (04/11): twins allocated once per size, never created in the frame under memory pressure ----------------------
 * RSX_MS_TWIN_PARK=1 (default off): a twin whose target goes away (retire, rescale) is parked, not released, and the next target
 * of the same kind/size/format takes it (stale: a colour twin refreshed from its target, a depth twin cleared, as any late twin)
 * instead of CreateCommittedResource -- the 720-810 MB depth twins of handles that keep changing, in the user's long space combat
 * game. Parked bytes are capped by RSX_MS_TWIN_PARK_MB (2048); a parked twin unused for RSX_MS_TWIN_PARK_AGE frames (1800) goes.
 * RSX_MS_TWIN_CEDE_PCT=<n> (default off; 85 is the rule): with the video memory above n% of the budget no twin is created in the
 * frame: a pass that needs one and finds none parked is drawn single-sampled (it ceded), never a new resource.
 * RSX_MS_TWIN_DEPTH_MAX=<n> (default 0 = no cap): at most n live depth twins; one more cedes the same way.
 * RSX_VRAM_BUDGET_MB=<n> (default off): the budget eng_vram_query reports (the sweeps, the % rules, the [vram] log) is n MB: to
 * reproduce over-budget play with a small game. RSX_VRAM108_LOG=1: one [vram108] line per [vram] line with the creations made over 85%. */
static int s_mp_on = -1; static u32 s_mp_cap_mb = 2048, s_mp_cede_pct, s_mp_depth_max, s_mp_age = 1800; static int s_mp_nopark;
static SRWLOCK s_mp_lock = SRWLOCK_INIT;
static struct { ID3D12Resource* res; int depth; u32 w, h, n; DXGI_FORMAT fmt; D3D12_RESOURCE_STATES state; u32 frame; u64 bytes; } s_mp[16];
static u32 s_mp_n; static u64 s_mp_bytes;
static struct { u32 parked, reused, ceded_pct, ceded_depth, evicted, aged; } s_mps;
static int s_mp_log;
static unsigned long long s_mp_budget_ovr;
static void eng_mp_setup(void)
{
    if (s_mp_on >= 0) return;
    const char* e = getenv("RSX_MS_TWIN_PARK"); s_mp_on = (e && e[0] && e[0] != '0') ? 1 : 0;
    e = getenv("RSX_MS_TWIN_PARK_MB"); if (e && atoi(e) > 0) s_mp_cap_mb = (u32)atoi(e);
    e = getenv("RSX_MS_TWIN_PARK_AGE"); if (e && atoi(e) > 0) s_mp_age = (u32)atoi(e);
    e = getenv("RSX_MS_TWIN_CEDE_PCT"); s_mp_cede_pct = e ? (u32)atoi(e) : 0; if (s_mp_cede_pct > 100) s_mp_cede_pct = 0;
    e = getenv("RSX_MS_TWIN_DEPTH_MAX"); s_mp_depth_max = e ? (u32)atoi(e) : 0;
    e = getenv("RSX_VRAM108_LOG"); s_mp_log = (e && e[0] && e[0] != '0') ? 1 : 0;
    e = getenv("RSX_VRAM_BUDGET_MB"); s_mp_budget_ovr = (e && atoi(e) > 0) ? (unsigned long long)atoi(e) << 20 : 0ull;
    if (s_mp_on || s_mp_cede_pct || s_mp_depth_max || s_mp_budget_ovr)
        fprintf(stderr, "[ms-twin108] park %s (cap %u MB, age %u frames), cede above %u%% of the budget, depth twin cap %u, budget override %llu MB\n",
                s_mp_on ? "on" : "off", s_mp_cap_mb, s_mp_age, s_mp_cede_pct, s_mp_depth_max, s_mp_budget_ovr >> 20);
}
static void eng_mp_count(int kind)
{
    eng_mp_setup();
    if (!s_mp_log) return;
    unsigned long long u = 0, b = 0;
    s_mp_made[kind]++;
    if (eng_vram_query(&u, &b) && b && u * 100ull > b * 85ull) s_mp_over85[kind]++;
}
static int eng_ms_park(EngObj* o)   /* the twin of o goes to the parked list (called with the object still describing it) */
{
    const int depth = o->kind == OBJ_DEPTH;
    const u64 b = (u64)ms_tw(o) * ms_th(o) * eng_px_bytes(depth ? ENG_DEPTH_RES_FMT : o->fmt) * s_ms_n;
    if ((b >> 20) > s_mp_cap_mb) return 0;
    AcquireSRWLockExclusive(&s_mp_lock);
    while (s_mp_n && (s_mp_n >= 16 || ((s_mp_bytes + b) >> 20) > s_mp_cap_mb)) {   /* the oldest makes room */
        u32 k = 0; for (u32 i = 1; i < s_mp_n; i++) if (s_mp[i].frame < s_mp[k].frame) k = i;
        RELEASE(s_mp[k].res); s_mp_bytes -= s_mp[k].bytes; s_mp[k] = s_mp[--s_mp_n]; s_mps.evicted++;
    }
    s_mp[s_mp_n].res = o->ms; s_mp[s_mp_n].depth = depth; s_mp[s_mp_n].w = ms_tw(o); s_mp[s_mp_n].h = ms_th(o); s_mp[s_mp_n].n = s_ms_n;
    s_mp[s_mp_n].fmt = depth ? ENG_DEPTH_FMT : o->fmt; s_mp[s_mp_n].state = o->ms_state; s_mp[s_mp_n].frame = s_enc_frame; s_mp[s_mp_n].bytes = b;
    s_mp_n++; s_mp_bytes += b; s_mps.parked++;
    o->ms = NULL;
    ReleaseSRWLockExclusive(&s_mp_lock);
    return 1;
}
static void eng_mp_flush(void)   /* every parked twin released (a new sample count, shutdown) */
{
    AcquireSRWLockExclusive(&s_mp_lock);
    for (u32 i = 0; i < s_mp_n; i++) RELEASE(s_mp[i].res);
    s_mp_n = 0; s_mp_bytes = 0;
    ReleaseSRWLockExclusive(&s_mp_lock);
}
static int s_mp_press = -1;    /* RSX_MS_TWIN_PARK_PRESS: % of the budget above which the parked twins are freed (90; >=100 = never on pressure) */
static int eng_mp_press(void)
{
    if (s_mp_press < 0) { const char* e = getenv("RSX_MS_TWIN_PARK_PRESS"); s_mp_press = (e && atoi(e) > 0) ? atoi(e) : 90; }
    return s_mp_press;
}
static void eng_mp_age(void)   /* each frame end (encode thread): parked twins unused for long go */
{
    if (!s_mp_n) return;
    /* rissa127: with RSX_MS_TWIN_FIT, video memory above RSX_MS_TWIN_PARK_PRESS % (90) of the budget frees the parked twins first,
     * oldest first (through the deferred list: a twin parked this frame may still be in flight). Checked every 15 frames. */
    { static u32 chk; const int pp = eng_mp_press();
      unsigned long long u = 0, b = 0;
      if (s_ms_fit > 0 && (++chk % 15) == 0 && eng_vram_query(&u, &b) && b && u * 100ull > b * (unsigned long long)pp) {
          AcquireSRWLockExclusive(&s_mp_lock);
          while (s_mp_n && s_mt_defer_n < 64 && u * 100ull > b * (unsigned long long)pp) {
              u32 k = 0; for (u32 i = 1; i < s_mp_n; i++) if (s_mp[i].frame < s_mp[k].frame) k = i;
              s_mt_defer[s_mt_defer_n].res = s_mp[k].res; s_mt_defer[s_mt_defer_n].fence = s_fence_value + 1; s_mt_defer_n++;
              u = u > s_mp[k].bytes ? u - s_mp[k].bytes : 0;
              s_mp_bytes -= s_mp[k].bytes; s_mp[k] = s_mp[--s_mp_n]; s_mps.evicted++;
              { static u32 nl; if (nl++ < 16) fprintf(stderr, "[ms-twin] memory pressure: parked twin freed\n"); }
          }
          ReleaseSRWLockExclusive(&s_mp_lock);
      } }
    AcquireSRWLockExclusive(&s_mp_lock);
    for (u32 i = 0; i < s_mp_n; ) {
        if (s_enc_frame - s_mp[i].frame >= s_mp_age) { RELEASE(s_mp[i].res); s_mp_bytes -= s_mp[i].bytes; s_mp[i] = s_mp[--s_mp_n]; s_mps.aged++; }
        else i++;
    }
    ReleaseSRWLockExclusive(&s_mp_lock);
}
/* RSX_VRAM_GOV_PARKED: only the bytes that memory pressure really reclaims count as 'not pressure'. With RSX_MS_TWIN_FIT off or
 * RSX_MS_TWIN_PARK_PRESS>=100 the parked twins stay resident until they age out, so they are pressure like any other target (review rissa128). */
static unsigned long long eng_parked_twin_bytes(void) { return (s_ms_fit > 0 && eng_mp_press() < 100) ? s_mp_bytes : 0; }
/* rissa128 RSX_SURF_POOL2: host bytes of a colour target (host w x h x format) and whether an MSAA twin hangs on it, for the draw engine's pool cap.
 * Read-only, called from the walker thread: o->w/h/fmt never change after creation; o->ms is only a hint (a stale answer costs at worst one target
 * pooled too much or released too early). */
static unsigned long long eng_target_info(u32 handle, int* has_twin)
{
    if (has_twin) *has_twin = 0;
    const EngObj* o = eng_obj(handle);
    if (!o || o->kind != OBJ_COLOR) return 0;
    if (has_twin) *has_twin = o->ms != NULL;
    return (unsigned long long)o->w * o->h * eng_px_bytes(o->fmt);
}
static void eng_ms_drop(EngObj* o)     /* a twin released with its target (retire, rescale, new sample count) */
{
    if (!o->ms) return;
    s_mtw.dropped[o->kind == OBJ_DEPTH]++;
    if (o->v2_site) v2_site_bump(o->v2_site, 3);
    eng_mp_setup();
    if (s_mp_on && !s_mp_nopark && (o->kind == OBJ_COLOR || o->kind == OBJ_DEPTH) && eng_ms_park(o)) return;
    if (reap_on()) eng_res_free(&o->ms, eng_mt_bytes(o)); else RELEASE(o->ms);   /* rissa128 RSX_REAP: a 250-720 MB twin released off the walker */
}
static void eng_ms_resolve(u32 h);
static void eng_ms_depth_sync(u32 h);
/* The twin's views (descriptors at the owner's slots). */
static void eng_ms_views(u32 h, EngObj* o)
{
    const int depth = o->kind == OBJ_DEPTH;
    if (depth) {
        D3D12_DEPTH_STENCIL_VIEW_DESC dd = {0};
        dd.Format = ENG_DEPTH_FMT; dd.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
        CALL(s_dev, CreateDepthStencilView, o->ms, &dd, obj_dsv_ms(h));
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {0};
        sd.Format = ENG_DEPTH_SRV_FMT; sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        CALL(s_dev, CreateShaderResourceView, o->ms, &sd, obj_srv_ms(h));
    } else {
        D3D12_RENDER_TARGET_VIEW_DESC rd = {0};
        rd.Format = o->fmt; rd.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
        CALL(s_dev, CreateRenderTargetView, o->ms, &rd, obj_rtv_ms(h));
        /* Read by the snapshot copy draws (eng_encode_copy_draw). */
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {0};
        sd.Format = o->fmt; sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        CALL(s_dev, CreateShaderResourceView, o->ms, &sd, obj_srv_ms(h));
    }
}
/* Does a draw record of the rest of this batch (from index `from`) name the object? The scan's answer, by brute force (check mode). */
static int eng_mt_later_use(u32 dh, u32 from)
{
    for (u32 i = from; i < s_mt_rn; i++) {
        const EngRecord* r = &s_mt_rec[i];
        if (r->kind != ENG_REC_DRAW) continue;
        for (u32 k = 0; k < r->nrt && k < RSX_BE_MAX_COLOR_TARGETS; k++) { EngObj* q = eng_owner(r->rt[k]); if (q && (u32)(q - s_obj) + 1 == dh) return 1; }
        EngObj* z = eng_owner(r->depth); if (z && (u32)(z - s_obj) + 1 == dh) return 1;
    }
    return 0;
}
/* The twin of another target of the same size and format that nothing in the rest of the frame names. */
static int eng_ms_pool_take(u32 h)
{
    EngObj* o = &s_obj[h - 1];
    if (!s_mt_pool || !s_mt_final || !s_mt_scanned) return 0;
    const int depth = o->kind == OBJ_DEPTH;
    u32 best = 0, bage = 0; u64 barea = 0;
    for (u32 q = 0; q < s_mt_alive_n; q++) {
        const u32 dh = s_mt_alive[q];
        if (dh == h) continue;
        EngObj* d = eng_obj(dh);
        if (!d || !d->ms || d->kind != o->kind || !eng_ms_fits(o, ms_tw(d), ms_th(d)) || (!depth && d->fmt != o->fmt)) continue;
        if (s_mt_lastidx[dh - 1] > s_mt_ri) continue;                    /* drawn into by this record or a later one */
        int fb = 0;
        for (u32 i = 0; i < s_fallback_n; i++) if (s_fallback_depth[i] == dh) fb = 1;
        if (fb) { int later = 0; for (u32 i = 0; i < s_mt_fbl_n; i++) if (s_mt_fbl[i].w == d->w && s_mt_fbl[i].h == d->h && s_mt_fbl[i].last > s_mt_ri) later = 1; if (later) continue; }
        if (s_mt_pool == 2 && !fb && eng_mt_later_use(dh, s_mt_ri)) {
            s_mtw.viol_check++;
            static int n; if (n++ < 8) fprintf(stderr, "[ms-twin] CHECK: the lookahead missed a later draw into h%u (twin not taken)\n", dh);
            continue;
        }
        const u32 age = s_enc_frame - (d->mt_last > d->mt_born ? d->mt_last : d->mt_born);
        const u64 area = (u64)ms_tw(d) * ms_th(d);   /* rissa127: the smallest twin that fits, then the oldest */
        if (!best || area < barea || (area == barea && age >= bage)) { best = dh; bage = age; barea = area; }
    }
    if (!best) return 0;
    if (s_mt_pct) {
        unsigned long long u = 0, b = 0;
        if (!eng_vram_query(&u, &b) || !b || u * 100ull < b * (unsigned long long)s_mt_pct) { s_mtw.refused++; return 0; }
    }
    EngObj* d = &s_obj[best - 1];
    if (d->kind == OBJ_COLOR && d->ms_dirty) { eng_ms_resolve(best); s_mtw.flush_res++; }
    if (d->kind == OBJ_DEPTH && d->ms_znewer) { eng_ms_depth_sync(best); s_mtw.flush_sync++; }
    o->ms = d->ms; o->ms_state = d->ms_state; o->ms_w = ms_tw(d); o->ms_h = ms_th(d);
    d->ms = NULL; d->ms_dirty = d->ms_stale = d->ms_znewer = 0; d->mt_robbed = s_enc_frame + 1;
    eng_ms_views(h, o);
    o->ms_dirty = 0; o->ms_stale = 1; o->ms_znewer = 0;
    o->mt_born = o->mt_last = s_enc_frame; o->mt_pool++;
    eng_mt_alive_add(h);
    s_mtw.taken[depth]++;
    return 1;
}
/* The twin and its views. A colour twin starts stale (refreshed from the
 * resolved target), a depth twin cleared, before its first pass. */
static int eng_ms_make(u32 h)
{
    EngObj* o = eng_obj(h);
    if (!o) return 0;
    if (o->ms) return 1;
    eng_mt_setup();
    if (s_mt_pool && eng_ms_pool_take(h)) return 1;
    const int depth = o->kind == OBJ_DEPTH;
    eng_mp_setup();
    if (s_mp_on && s_mp_n) {   /* hitch108: a parked twin of this kind, size and format: no new resource */
        int got = -1;
        AcquireSRWLockExclusive(&s_mp_lock);
        for (u32 i = s_mp_n; i-- > 0; )   /* rissa127: the smallest that fits (exact size only, without RSX_MS_TWIN_FIT) */
            if (s_mp[i].depth == depth && eng_ms_fits(o, s_mp[i].w, s_mp[i].h) && s_mp[i].n == s_ms_n && (depth || s_mp[i].fmt == o->fmt) &&
                (got < 0 || (u64)s_mp[i].w * s_mp[i].h < (u64)s_mp[got].w * s_mp[got].h)) got = (int)i;
        if (got >= 0) {
            o->ms_w = s_mp[got].w; o->ms_h = s_mp[got].h;
            o->ms = s_mp[got].res; o->ms_state = s_mp[got].state; s_mp_bytes -= s_mp[got].bytes; s_mp[got] = s_mp[--s_mp_n]; s_mps.reused++;
        }
        ReleaseSRWLockExclusive(&s_mp_lock);
        if (got >= 0) {
            eng_ms_views(h, o);
            o->ms_dirty = 0; o->ms_stale = 1; o->ms_znewer = 0;
            o->mt_born = o->mt_last = s_enc_frame;
            eng_mt_alive_add(h);
            return 1;
        }
    }
    if (s_mp_cede_pct || s_mp_depth_max) {   /* hitch108: no creation in the frame when the memory is short or the depth twins are at their cap: draw single-sampled */
        if (s_mp_depth_max && depth) {
            u32 live = 0; for (u32 i = 0; i < s_obj_count; i++) if (s_obj[i].ms && s_obj[i].kind == OBJ_DEPTH) live++;
            if (live >= s_mp_depth_max) { s_mps.ceded_depth++; return 0; }
        }
        if (s_mp_cede_pct) {
            unsigned long long u = 0, b = 0;
            if (eng_vram_query(&u, &b) && b && u * 100ull > b * (unsigned long long)s_mp_cede_pct) { s_mps.ceded_pct++; return 0; }
        }
    }
    eng_mp_count(2);
    LARGE_INTEGER mt_a, mt_b, mt_f; QueryPerformanceCounter(&mt_a);
    D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td = {0};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    u32 tw = o->w, th = o->h;   /* rissa127 FIT: sides rounded up */
    eng_ms_fit_setup();
    if (s_ms_fit && s_ms_fit_round > 1) { tw = (tw + s_ms_fit_round - 1) / s_ms_fit_round * s_ms_fit_round; th = (th + s_ms_fit_round - 1) / s_ms_fit_round * s_ms_fit_round; }
    if (tw > 16384) tw = o->w;
    if (th > 16384) th = o->h;
    o->ms_w = tw; o->ms_h = th;
    td.Width = tw; td.Height = th; td.DepthOrArraySize = 1; td.MipLevels = 1;
    td.Format = depth ? ENG_DEPTH_RES_FMT : o->fmt; td.SampleDesc.Count = s_ms_n;
    td.Flags = depth ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL : D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE cv = {0};
    cv.Format = depth ? ENG_DEPTH_FMT : o->fmt;
    if (depth) cv.DepthStencil.Depth = 1.0f;
    const D3D12_RESOURCE_STATES st = depth ? D3D12_RESOURCE_STATE_DEPTH_WRITE : D3D12_RESOURCE_STATE_RENDER_TARGET;
    if (FAILED(CALL(s_dev, CreateCommittedResource, &hp, D3D12_HEAP_FLAG_NONE, &td, st, &cv,
                    &IID_ID3D12Resource, (void**)&o->ms))) {
        static int n = 0;
        { static unsigned nv; nv++; if (nv <= 200 || (nv & (nv - 1)) == 0) V2EV("oom", "\"what\":\"msaa_twin\",\"w\":%u,\"h\":%u,\"samples\":%u,\"count\":%u,\"site\":\"%s:%d\"", o->w, o->h, s_ms_n, nv, o->v2_site && s_v2s[o->v2_site].fn ? s_v2s[o->v2_site].fn : "?", o->v2_site ? s_v2s[o->v2_site].line : 0); }
        if (n < 4 && ++n) fprintf(stderr, "[rsx engine/d3d12] MSAA x%u target %ux%u failed\n", s_ms_n, o->w, o->h);
        o->ms = NULL;
        return 0;
    }
    QueryPerformanceCounter(&mt_b); QueryPerformanceFrequency(&mt_f);
    { const double ms = (double)(mt_b.QuadPart - mt_a.QuadPart) * 1000.0 / (double)mt_f.QuadPart;
      s_mtw.made[depth]++; s_mtw.make_ms[depth] += ms; if (ms > s_mtw.make_ms_max) s_mtw.make_ms_max = ms;
      if (s_mt_log && ms >= 20.0) fprintf(stderr, "[ms-twin] enc frame %u: %s twin h%u %ux%u made in %.1f ms\n", s_enc_frame, depth ? "depth" : "colour", h, o->w, o->h, ms); }
    if (o->v2_site) v2_site_bump(o->v2_site, 2);
    o->ms_state = st;
    eng_ms_views(h, o);
    o->ms_dirty = 0; o->ms_stale = 1;
    o->mt_born = o->mt_last = s_enc_frame;
    eng_mt_alive_add(h);
    return 1;
}
/* The twin's draws into the single-sampled target: the region they touched. */
static void eng_ms_resolve(u32 h)
{
    EngObj* o = eng_obj(h);
    if (!o || !o->ms || !o->ms_dirty || o->kind != OBJ_COLOR) { if (o) o->ms_dirty = 0; return; }
    u32 x0 = o->ms_x0, y0 = o->ms_y0, x1 = o->ms_x1 < o->w ? o->ms_x1 : o->w, y1 = o->ms_y1 < o->h ? o->ms_y1 : o->h;
    o->ms_dirty = 0;
    if (x0 >= x1 || y0 >= y1) return;
    res_transition(o->ms, &o->ms_state, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
    res_transition(o->res, &o->state, D3D12_RESOURCE_STATE_RESOLVE_DEST);
    bar_flush();
    if (!s_list1) CALL(s_list, QueryInterface, &IID_ID3D12GraphicsCommandList1, (void**)&s_list1);
    const int bigger = ms_tw(o) != o->w || ms_th(o) != o->h;   /* rissa127 FIT: a full resolve needs equal sizes */
    if (bigger && !s_list1) return;
    if (s_list1 && (bigger || x0 || y0 || x1 < o->w || y1 < o->h)) {
        D3D12_RECT rc = { (LONG)x0, (LONG)y0, (LONG)x1, (LONG)y1 };
        CALL(s_list1, ResolveSubresourceRegion, o->res, 0, x0, y0, o->ms, 0, &rc, o->fmt, D3D12_RESOLVE_MODE_AVERAGE);
    } else {
        CALL(s_list, ResolveSubresource, o->res, 0, o->ms, 0, o->fmt);
    }
}
static void eng_ms_resolve_all(void)
{
    for (u32 i = 0; i < s_ms_open_n; i++) eng_ms_resolve(s_ms_open[i]);
    s_ms_open_n = 0;
}
static void eng_ms_touch(u32 h, const D3D12_RECT* sc)
{
    EngObj* o = eng_obj(h);
    if (!o || !o->ms) return;
    const u32 x0 = sc->left > 0 ? (u32)sc->left : 0, y0 = sc->top > 0 ? (u32)sc->top : 0;
    const u32 x1 = (u32)sc->right, y1 = (u32)sc->bottom;
    if (!o->ms_dirty) {
        o->ms_x0 = x0; o->ms_y0 = y0; o->ms_x1 = x1; o->ms_y1 = y1; o->ms_dirty = 1;
        if (s_ms_open_n < 64) s_ms_open[s_ms_open_n++] = h; else eng_ms_resolve(h);
    } else {
        if (x0 < o->ms_x0) o->ms_x0 = x0;
        if (y0 < o->ms_y0) o->ms_y0 = y0;
        if (x1 > o->ms_x1) o->ms_x1 = x1;
        if (y1 > o->ms_y1) o->ms_y1 = y1;
    }
}
/* A stale colour twin is drawn over with its resolved target (a blit per
 * format and sample count); a stale depth twin cleared. */
static ID3D12PipelineState* eng_ms_refresh_pso(DXGI_FORMAT f)
{
    static struct { DXGI_FORMAT f; u32 n; ID3D12PipelineState* pso; } cache[16]; static u32 nc;
    for (u32 i = 0; i < nc; i++) if (cache[i].f == f && cache[i].n == s_ms_n) return cache[i].pso;
    if (!s_helper_vs_blob || !s_helper_blit_blob) return NULL;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {0};
    pd.pRootSignature = s_helper_rootsig;
    pd.VS.pShaderBytecode = CALL0(s_helper_vs_blob, GetBufferPointer); pd.VS.BytecodeLength = CALL0(s_helper_vs_blob, GetBufferSize);
    pd.PS.pShaderBytecode = CALL0(s_helper_blit_blob, GetBufferPointer); pd.PS.BytecodeLength = CALL0(s_helper_blit_blob, GetBufferSize);
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pd.SampleMask = UINT_MAX;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1; pd.RTVFormats[0] = f;
    pd.SampleDesc.Count = s_ms_n;
    ID3D12PipelineState* pso = NULL;
    if (FAILED(CALL(s_dev, CreateGraphicsPipelineState, &pd, &IID_ID3D12PipelineState, (void**)&pso))) pso = NULL;
    if (nc < 16) { cache[nc].f = f; cache[nc].n = s_ms_n; cache[nc].pso = pso; nc++; }
    return pso;
}
static void eng_ms_refresh(u32 h)
{
    EngObj* o = eng_obj(h);
    if (!o || !o->ms || !o->ms_stale) return;
    o->ms_stale = 0;
    if (o->kind == OBJ_DEPTH) {
        res_transition(o->ms, &o->ms_state, D3D12_RESOURCE_STATE_DEPTH_WRITE);
        bar_flush();
        CALL(s_list, ClearDepthStencilView, obj_dsv_ms(h), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, NULL);
        return;
    }
    ID3D12PipelineState* pso = eng_ms_refresh_pso(o->fmt);
    if (!pso) return;
    res_transition(o->res, &o->state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    res_transition(o->ms, &o->ms_state, D3D12_RESOURCE_STATE_RENDER_TARGET);
    bar_flush();
    eng_fullscreen_pass(pso, obj_rtv_ms(h), o->w, o->h, obj_srv(h));
}
/* A single-sampled pass bound to a depth target whose twin has the 3D
 * passes' depth -- Drakengard 3 draws a 640x360 pass against its 1280x720
 * depth -- gets the twin's first sample in the target first (depth only:
 * the stencil the target has stays). */
static void eng_ms_depth_sync(u32 h)
{
    EngObj* o = eng_obj(h);
    if (!o || !o->ms || !o->ms_znewer || !s_zcopy_ms_pso) return;
    o->ms_znewer = 0;
    const D3D12_CPU_DESCRIPTOR_HANDLE srv = obj_srv_ms(h);
    const D3D12_GPU_DESCRIPTOR_HANDLE tab = eng_srv_table(&srv, 1);   /* first: see eng_fullscreen_pass_at */
    res_transition(o->ms, &o->ms_state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    res_transition(o->res, &o->state, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    bar_flush();
    eng_use_helper_rootsig();
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv = obj_dsv(h);
    CALL(s_list, OMSetRenderTargets, 0, NULL, FALSE, &dsv);
    D3D12_VIEWPORT vp = { 0.0f, 0.0f, (float)o->w, (float)o->h, 0.0f, 1.0f };
    D3D12_RECT sc = { 0, 0, (LONG)o->w, (LONG)o->h };
    CALL(s_list, RSSetViewports, 1, &vp);
    CALL(s_list, RSSetScissorRects, 1, &sc);
    CALL(s_list, SetPipelineState, s_zcopy_ms_pso);
    CALL(s_list, IASetPrimitiveTopology, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    s_cur_topo = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    CALL(s_list, SetGraphicsRootDescriptorTable, 0, tab);
    CALL(s_list, DrawInstanced, 3, 1, 0, 0);
    s_cur_pso = NULL; s_cur_targets_valid = 0;
}

/* ---- vram106: use accounting, lookahead, idle release, [ms-twin] log (see eng_ms_make) ---- */
static void eng_mt_note(const EngRecord* r, u32 nrt, EngObj* z)
{
    if (!s_mt_log && !s_mt_pool) {   /* uni127: log v2 "twins" used-in-1s count reads mt_last, which was only kept with the twin log/pool on (always 0) */
        z->mt_last = s_enc_frame;
        for (u32 k = 0; k < nrt && k < RSX_BE_MAX_COLOR_TARGETS; k++) { EngObj* c = eng_owner(r->rt[k]); if (c) {
            if (c->mt_last != s_enc_frame) c->mt_fdraws = 0;   /* msaa127: draws this frame (RSX_MS_FEWDRAW) */
            c->mt_fdraws++; c->mt_last = s_enc_frame; } }
        return;
    }
    const u32 ord = s_mt_base + s_mt_ri;
    const u32 kd = (u32)(z - s_obj) + 1, kc = nrt ? r->rt[0] : 0;
    const int newpass = kd != s_mt_lastkey_d || kc != s_mt_lastkey_c;
    s_mt_lastkey_d = kd; s_mt_lastkey_c = kc;
    s_mtw.draws++; if (newpass) s_mtw.passes++;
    EngObj* c0 = nrt ? eng_owner(r->rt[0]) : NULL;
    const int shadow = c0 && c0->xk > 0.0f;
    EngObj* list[1 + RSX_BE_MAX_COLOR_TARGETS]; u32 n = 0;
    list[n++] = z;
    for (u32 k = 0; k < nrt && k < RSX_BE_MAX_COLOR_TARGETS; k++) { EngObj* c = eng_owner(r->rt[k]); if (c) list[n++] = c; }
    for (u32 i = 0; i < n; i++) {
        EngObj* o = list[i];
        if (o->mt_robbed == s_enc_frame + 1) {
            s_mtw.viol_later++; o->mt_robbed = 0;
            static int nv; if (nv++ < 8) fprintf(stderr, "[ms-twin] CHECK VIOLATION: h%u drew again in frame %u after its twin was taken over\n", (u32)(o - s_obj) + 1, s_enc_frame);
        }
        if (!o->mt_fdraws || o->mt_last != s_enc_frame) { o->mt_ford = ord; o->mt_fdraws = 0; }
        o->mt_lord = ord; o->mt_last = s_enc_frame; o->mt_fdraws++;
        if (newpass) o->mt_wpasses++;
        o->mt_wdraws++;
        if (!nrt) o->mt_wdo++; else if (shadow) o->mt_wshadow++; else o->mt_wcol++;
    }
}
static void eng_mt_extent(EngObj* z, const EngRecord* r, u32 nrt, const D3D12_RECT* sc)
{
    if (!s_mt_log) return;
    const u32 x1 = sc->right > 0 ? (u32)sc->right : 0, y1 = sc->bottom > 0 ? (u32)sc->bottom : 0;
    if (x1 > z->mt_ex) z->mt_ex = x1;
    if (y1 > z->mt_ey) z->mt_ey = y1;
    if (nrt) { EngObj* c = eng_owner(r->rt[0]); if (c) { if (x1 > c->mt_ex) c->mt_ex = x1; if (y1 > c->mt_ey) c->mt_ey = y1; } }
}
/* Before a frame's last batch is encoded: the last draw record that names each target. */
static void eng_mt_scan(const EngRecord* rec, u32 n)
{
    memset(s_mt_lastidx, 0, (size_t)(s_obj_count < ENG_MAX_OBJECTS ? s_obj_count : ENG_MAX_OBJECTS) * sizeof s_mt_lastidx[0]);
    s_mt_fbl_n = 0;
    for (u32 i = 0; i < n; i++) {
        const EngRecord* r = &rec[i];
        if (r->kind != ENG_REC_DRAW) continue;
        EngObj* c0 = NULL;
        for (u32 k = 0; k < r->nrt && k < RSX_BE_MAX_COLOR_TARGETS; k++) {
            EngObj* c = eng_owner(r->rt[k]);
            if (c) { s_mt_lastidx[(u32)(c - s_obj)] = i + 1; if (!c0) c0 = c; }
        }
        EngObj* z = eng_owner(r->depth);
        if (z && z->has_dsv) s_mt_lastidx[(u32)(z - s_obj)] = i + 1;
        else if (c0) {   /* the fallback depth of this size */
            u32 q; for (q = 0; q < s_mt_fbl_n; q++) if (s_mt_fbl[q].w == c0->w && s_mt_fbl[q].h == c0->h) break;
            if (q == s_mt_fbl_n && q < 8) { s_mt_fbl[q].w = c0->w; s_mt_fbl[q].h = c0->h; s_mt_fbl_n++; }
            if (q < 8) s_mt_fbl[q].last = i + 1;
        }
    }
}
/* A twin released for being idle: flushed into its target, freed once the GPU is past the lists that name it. */
static void eng_mt_release_idle(u32 h)
{
    EngObj* o = &s_obj[h - 1];
    if (s_mt_defer_n >= 64) return;
    if (o->kind == OBJ_COLOR && o->ms_dirty) { eng_ms_resolve(h); s_mtw.flush_res++; }
    if (o->kind == OBJ_DEPTH && o->ms_znewer) { eng_ms_depth_sync(h); s_mtw.flush_sync++; }
    /* rissa127: with RSX_MS_TWIN_PARK + RSX_MS_TWIN_FIT an idle twin is parked, not freed: at Rykan V the two 720 MB depth twins
     * (guest 1280x1024) went idle for 300 frames and came back seconds later, each re-creation a 130-510 ms frame. */
    eng_mp_setup(); eng_ms_fit_setup();
    if (s_mp_on && s_ms_fit && !s_mp_nopark && eng_ms_park(o)) { o->ms_dirty = o->ms_stale = o->ms_znewer = 0; s_mtw.idlerel[o->kind == OBJ_DEPTH]++; return; }
    s_mt_defer[s_mt_defer_n].res = o->ms; s_mt_defer[s_mt_defer_n].fence = s_fence_value + 1; s_mt_defer_n++;
    o->ms = NULL; o->ms_dirty = o->ms_stale = o->ms_znewer = 0;
    s_mtw.idlerel[o->kind == OBJ_DEPTH]++;
}
static void eng_mt_frame_end(void)
{
    eng_mp_age();   /* hitch108 */
    /* msaa127 RSX_MS_FEWDRAW=1 (default off): a colour twin that got at most one draw in each of RSX_MS_FEWDRAW_N used frames in a
     * row (default 120) is resolved, released, and its target gets no twin again. At Rykan V 4K x8 the two 1280x720 display
     * buffers' twins (2 x 253 MB) got one full-screen draw a frame (the scene is drawn into their 1280x704 window, a target of
     * its own): multisampling a full-screen quad smooths nothing. */
    { static int fd = -1; static u32 fdn = 120;
      if (fd < 0) { const char* e = getenv("RSX_MS_FEWDRAW"); fd = e && e[0] == '1'; e = getenv("RSX_MS_FEWDRAW_N"); if (e && atoi(e) > 0) fdn = (u32)atoi(e); }
      if (fd)
          for (u32 q = 0; q < s_mt_alive_n; q++) {
              const u32 h = s_mt_alive[q]; EngObj* o = eng_obj(h);
              if (!o || !o->ms || o->retired || o->kind != OBJ_COLOR || o->mt_last != s_enc_frame) continue;
              if (o->mt_fdraws > 1) { o->ms_few = 0; continue; }
              if (++o->ms_few < fdn) continue;
              if (s_mt_defer_n >= 64) break;
              eng_mt_release_idle(h); o->ms_never = 1;
              { static u32 nl; if (nl++ < 16) fprintf(stderr, "[msaa] few-draw twin released: h%u %ux%u (<=1 draw a frame for %u used frames); no MSAA for it from now\n", h, o->w, o->h, fdn); }
          } }
    if (!s_mt_log && !s_mt_pool) {
        for (u32 i = 0; i < s_mt_defer_n; i++)   /* msaa127: twins released above are freed here too */
            if (fence_done(s_mt_defer[i].fence)) { RELEASE(s_mt_defer[i].res); s_mt_defer[i--] = s_mt_defer[--s_mt_defer_n]; }
        s_enc_frame++; s_mt_base = 0; return;
    }
    for (u32 i = 0; i < s_mt_defer_n; i++)
        if (fence_done(s_mt_defer[i].fence)) { RELEASE(s_mt_defer[i].res); s_mt_defer[i--] = s_mt_defer[--s_mt_defer_n]; }
    u32 al[2] = {0, 0}, us[2] = {0, 0}; u64 al_mb = 0, us_mb = 0;
    struct { u32 kind, w, h, fmt, alive, used, need; } fk[16]; u32 nfk = 0;
    u32 used_h[64]; u32 nused = 0;
    for (u32 q = 0; q < s_mt_alive_n; q++) {
        const u32 h = s_mt_alive[q];
        EngObj* o = eng_obj(h);
        if (!o || !o->ms) { s_mt_alive[q--] = s_mt_alive[--s_mt_alive_n]; continue; }
        const u32 k = o->kind == OBJ_DEPTH; const u64 mb = eng_mt_bytes(o);
        al[k]++; al_mb += mb;
        const int used = o->mt_last == s_enc_frame && o->mt_fdraws;
        u32 f; for (f = 0; f < nfk; f++) if (fk[f].kind == k && fk[f].w == o->w && fk[f].h == o->h && fk[f].fmt == (k ? 0u : (u32)o->fmt)) break;
        if (f == nfk && f < 16) { fk[f].kind = k; fk[f].w = o->w; fk[f].h = o->h; fk[f].fmt = k ? 0u : (u32)o->fmt; fk[f].alive = fk[f].used = fk[f].need = 0; nfk++; }
        if (f < 16) fk[f].alive++;
        if (used) { us[k]++; us_mb += mb; if (f < 16) fk[f].used++; o->mt_wframes++; if (nused < 64) used_h[nused++] = h; }
    }
    /* Twins a perfect sharing would need: at each first draw, the used twins of the key whose draws span that point. */
    u64 need_mb = 0; u32 need[2] = {0, 0};
    for (u32 f = 0; f < nfk; f++) {
        u32 mx = 0;
        for (u32 a = 0; a < nused; a++) {
            EngObj* oa = &s_obj[used_h[a] - 1];
            if ((oa->kind == OBJ_DEPTH) != (int)fk[f].kind || oa->w != fk[f].w || oa->h != fk[f].h || (!fk[f].kind && (u32)oa->fmt != fk[f].fmt)) continue;
            u32 c = 0;
            for (u32 b = 0; b < nused; b++) {
                EngObj* ob = &s_obj[used_h[b] - 1];
                if ((ob->kind == OBJ_DEPTH) != (int)fk[f].kind || ob->w != fk[f].w || ob->h != fk[f].h || (!fk[f].kind && (u32)ob->fmt != fk[f].fmt)) continue;
                if (ob->mt_ford <= oa->mt_ford && ob->mt_lord >= oa->mt_ford) c++;
            }
            if (c > mx) mx = c;
        }
        fk[f].need = mx; need[fk[f].kind] += mx;
        { EngObj tmp = {0}; tmp.kind = fk[f].kind ? OBJ_DEPTH : OBJ_COLOR; tmp.w = fk[f].w; tmp.h = fk[f].h; tmp.fmt = (DXGI_FORMAT)fk[f].fmt; need_mb += eng_mt_bytes(&tmp) * mx; }
        u32 w;
        for (w = 0; w < s_mtw.nkey; w++) if (s_mtw.key[w].kind == fk[f].kind && s_mtw.key[w].w == fk[f].w && s_mtw.key[w].h == fk[f].h && s_mtw.key[w].fmt == fk[f].fmt) break;
        if (w == s_mtw.nkey && w < 16) { s_mtw.key[w].kind = fk[f].kind; s_mtw.key[w].w = fk[f].w; s_mtw.key[w].h = fk[f].h; s_mtw.key[w].fmt = fk[f].fmt; s_mtw.key[w].alive = s_mtw.key[w].used = s_mtw.key[w].need = 0; s_mtw.nkey++; }
        if (w < 16) {
            if (fk[f].alive > s_mtw.key[w].alive) s_mtw.key[w].alive = fk[f].alive;
            if (fk[f].used > s_mtw.key[w].used) s_mtw.key[w].used = fk[f].used;
            if (fk[f].need > s_mtw.key[w].need) s_mtw.key[w].need = fk[f].need;
        }
    }
    for (int k = 0; k < 2; k++) {
        if (al[k] > s_mtw.maxalive[k]) s_mtw.maxalive[k] = al[k];
        if (us[k] > s_mtw.maxused[k]) s_mtw.maxused[k] = us[k];
        if (need[k] > s_mtw.maxneed[k]) s_mtw.maxneed[k] = need[k];
    }
    if (al_mb > s_mtw.maxalive_mb) s_mtw.maxalive_mb = al_mb;
    if (us_mb > s_mtw.maxused_mb) s_mtw.maxused_mb = us_mb;
    if (need_mb > s_mtw.maxneed_mb) s_mtw.maxneed_mb = need_mb;
    s_mtw.alive_sum_mb += al_mb; s_mtw.frames++;
    /* idle release */
    if (s_mt_pool)
        for (u32 q = 0; q < s_mt_alive_n; q++) {
            const u32 h = s_mt_alive[q]; EngObj* o = eng_obj(h);
            if (!o || !o->ms || o->retired) continue;
            const u32 last = o->mt_last > o->mt_born ? o->mt_last : o->mt_born;
            if (s_enc_frame - last >= s_mt_idle_n) eng_mt_release_idle(h);
        }
    /* window log */
    const ULONGLONG now = GetTickCount64();
    if (!s_mtw.t0) s_mtw.t0 = now;
    if (s_mt_log && now - s_mtw.t0 >= 5000) {
        fprintf(stderr, "[ms-twin] enc frame %u, %.1f s, %u frames: alive now %u depth + %u colour = %llu MB (window max %llu MB, mean %llu MB); "
                "used in one frame, max: %u depth + %u colour, %llu MB; twins a perfect sharing would need, max: %u depth + %u colour, %llu MB; "
                "made %u depth + %u colour (%.1f ms/each, worst %.1f ms), released with their target %u+%u, idle-released %u+%u, taken over %u+%u (refused by %%threshold %u), "
                "flush: %u depth copies %u resolves; draws/frame %.0f, passes/frame %.1f; CHECK violations %u + lookahead misses %u\n",
                s_enc_frame, (double)(now - s_mtw.t0) / 1000.0, s_mtw.frames, al[1], al[0], (unsigned long long)(al_mb >> 20),
                (unsigned long long)(s_mtw.maxalive_mb >> 20), (unsigned long long)((s_mtw.frames ? s_mtw.alive_sum_mb / s_mtw.frames : 0) >> 20),
                s_mtw.maxused[1], s_mtw.maxused[0], (unsigned long long)(s_mtw.maxused_mb >> 20),
                s_mtw.maxneed[1], s_mtw.maxneed[0], (unsigned long long)(s_mtw.maxneed_mb >> 20),
                s_mtw.made[1], s_mtw.made[0],
                (s_mtw.made[0] + s_mtw.made[1]) ? (s_mtw.make_ms[0] + s_mtw.make_ms[1]) / (s_mtw.made[0] + s_mtw.made[1]) : 0.0, s_mtw.make_ms_max,
                s_mtw.dropped[1], s_mtw.dropped[0], s_mtw.idlerel[1], s_mtw.idlerel[0], s_mtw.taken[1], s_mtw.taken[0], s_mtw.refused,
                s_mtw.flush_sync, s_mtw.flush_res,
                s_mtw.frames ? (double)s_mtw.draws / s_mtw.frames : 0.0, s_mtw.frames ? (double)s_mtw.passes / s_mtw.frames : 0.0,
                s_mtw.viol_later, s_mtw.viol_check);
        for (u32 w = 0; w < s_mtw.nkey; w++)
            fprintf(stderr, "[ms-twin]   size %ux%u %s: alive max %u, used per frame max %u, needed with sharing max %u\n",
                    s_mtw.key[w].w, s_mtw.key[w].h, s_mtw.key[w].kind ? "depth" : "colour", s_mtw.key[w].alive, s_mtw.key[w].used, s_mtw.key[w].need);
        for (u32 q = 0; q < s_mt_alive_n; q++) {
            const u32 h = s_mt_alive[q]; EngObj* o = eng_obj(h);
            if (!o || !o->ms) continue;
            const u32 uf = o->mt_wframes ? o->mt_wframes : 1;
            fprintf(stderr, "[ms-twin]   h%u %s %ux%u (guest %ux%u, fmt %u) %llu MB: age %u idle %u frames; used in %u of %u frames, %.0f draws %.1f passes per used frame "
                    "(depth-only draws %u, shadow-colour %u, colour %u), drawn extent %ux%u, taken over %u times%s\n",
                    h, o->kind == OBJ_DEPTH ? "depth" : "colour", o->w, o->h, o->gw, o->gh, (u32)o->fmt, (unsigned long long)(eng_mt_bytes(o) >> 20),
                    s_enc_frame - o->mt_born, s_enc_frame - o->mt_last, o->mt_wframes, s_mtw.frames, (double)o->mt_wdraws / uf, (double)o->mt_wpasses / uf,
                    o->mt_wdo, o->mt_wshadow, o->mt_wcol, o->mt_ex, o->mt_ey, o->mt_pool, o->retired ? " (retired)" : "");
            o->mt_wdraws = o->mt_wframes = o->mt_wpasses = o->mt_wdo = o->mt_wcol = o->mt_wshadow = 0;
        }
        const ULONGLONG keep_t = now;
        memset(&s_mtw, 0, sizeof s_mtw); s_mtw.t0 = keep_t;
    }
    for (u32 q = 0; q < s_mt_alive_n; q++) { EngObj* o = eng_obj(s_mt_alive[q]); if (o) o->mt_fdraws = 0; }
    s_enc_frame++; s_mt_base = 0;
}

static ID3D12PipelineState* eng_pso_ms(u32 pipeline, int cls)
{
    if (!pipeline || pipeline > s_pipe_count) return NULL;
    EngPipeline* p = &s_pipe[pipeline - 1];
    if (!p->vs || p->failedms[cls]) return NULL;
    if (p->psoms[cls]) return p->psoms[cls];
    /* As eng_pso. A sample count that changed meanwhile (eng_ms_apply, which
     * drops these) makes the build stale: dropped, not published. */
    const u32 n = s_ms_n;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = p->desc;
    pd.InputLayout.pInputElementDescs = pd.InputLayout.NumElements ? p->il : NULL;
    pd.PrimitiveTopologyType = cls == 0 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT
                             : cls == 1 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
                                        : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.SampleDesc.Count = n; pd.SampleDesc.Quality = 0;
    pd.RasterizerState.MultisampleEnable = TRUE;
    ID3D12PipelineState* pso = NULL;
    /* 31/10 (pre52): through the pipeline library like the 1x states, and timed: the MSAA variant is the one a
     * 3D pass really draws with, and it used to be built here, on the walker, at the pipeline's first draw
     * (the "first time" hitches with 0 pipelines built in the hitch line). */
    LARGE_INTEGER pq0, pq1, pqf; QueryPerformanceFrequency(&pqf); QueryPerformanceCounter(&pq0);
    const HRESULT hr = eng_create_gfx_pso(&pd, &pso);
    QueryPerformanceCounter(&pq1);
    { const double pso_ms = (double)(pq1.QuadPart - pq0.QuadPart) * 1000.0 / (double)pqf.QuadPart;
      const int on_walker = GetCurrentThreadId() == s_walker_tid;
      if (on_walker) g_rsx_frame_pso_ms += pso_ms;
      static int lg = -1; if (lg < 0) lg = getenv("RSX_PSO_LOG") ? 1 : 0;
      if (lg) fprintf(stderr, "[pso] pipeline %u class %d MSAA x%u: %.2f ms on the %s\n", pipeline, cls, n, pso_ms,
                      on_walker ? "walker" : "build thread"); }
    eng_lock(&s_pipe_lock, "pipe_lock");
    if (p->psoms[cls] || n != s_ms_n) RELEASE(pso);
    else if (FAILED(hr)) {
        static int nf = 0;
        if (nf < 16 && ++nf) fprintf(stderr, "[rsx engine/d3d12] MSAA x%u pipeline state failed: 0x%08lX (pipeline %u class %d)\n",
                               n, (long)hr, pipeline, cls);
        p->failedms[cls] = 1;
    }
    else p->psoms[cls] = pso;
    ReleaseSRWLockExclusive(&s_pipe_lock);
    return p->psoms[cls];
}
/* A new sample count (or off): the twins and the MSAA builds dropped; twins
 * are made again at their next MSAA pass. Lists are drained first. */
static void eng_ms_apply(int want)
{
    fence_wait(fence_signal());
    s_mp_nopark = 1;
    for (u32 i = 0; i < s_obj_count; i++) {
        EngObj* o = &s_obj[i];
        eng_ms_drop(o);
        o->ms_dirty = o->ms_stale = o->ms_znewer = 0;
    }
    s_ms_open_n = 0;
    s_mp_nopark = 0; eng_mp_flush();   /* hitch108 */
    eng_lock(&s_pipe_lock, "pipe_lock");
    for (u32 i = 0; i < s_pipe_count; i++)
        for (int c = 0; c < 3; c++) {
            RELEASE(s_pipe[i].psoms[c]); s_pipe[i].failedms[c] = 0;
            RELEASE(s_pipe[i].copy_pso_ms[c]); s_pipe[i].copy_failed_ms[c] = 0;
        }
    ReleaseSRWLockExclusive(&s_pipe_lock);
    u32 n = (u32)want;
    while (n > 1 && !(eng_ms_format_ok(DXGI_FORMAT_R8G8B8A8_UNORM, n) && eng_ms_format_ok(DXGI_FORMAT_R16G16B16A16_FLOAT, n) &&
                      eng_ms_format_ok(ENG_DEPTH_FMT, n))) n >>= 1;
    s_ms_n = n ? n : 1;
    RELEASE(s_snap_copy_ms_ps);
    if (s_ms_n > 1) {
        char src[512];
        const int len = snprintf(src, sizeof src, kSnapCopyMsHLSL, s_ms_n, s_ms_n, s_ms_n);
        s_snap_copy_ms_ps = eng_compile(src, (u32)len, "main", "ps_5_0", "snapshot copy ps (msaa)");
    }
    fprintf(stderr, "[RSX d3d12] MSAA %s\n", s_ms_n > 1 ? (s_ms_n == 2 ? "x2" : s_ms_n == 4 ? "x4" : "x8") : "off");
}

static void eng_encode_draw(const EngRecord* r)
{
    /* Attachments: target A's size rules, as the Metal engine. */
    D3D12_CPU_DESCRIPTOR_HANDLE rtv[RSX_BE_MAX_COLOR_TARGETS];
    u32 nrt = 0, tw = 0, th = 0;
    for (u32 k = 0; k < r->nrt && k < RSX_BE_MAX_COLOR_TARGETS; k++) {
        EngObj* o = eng_owner(r->rt[k]);
        if (!o || !o->has_rtv) break;
        if (nrt && (o->w != tw || o->h != th)) break;
        if (!nrt) { tw = o->w; th = o->h; }
        rtv[nrt++] = obj_rtv(r->rt[k]);
    }
    if (nrt != r->nrt) return;   /* the pipeline declares r->nrt attachments */
    u32 depth = r->depth;
    EngObj* z = eng_owner(depth);
    if (z && !z->has_dsv) { z = NULL; depth = 0; }
    if (!nrt && !z) return;
    const int real_depth = z != NULL;
    if (!z) { depth = eng_fallback_depth(tw, th); z = eng_owner(depth); if (!z) return; }
    if (!nrt) { tw = z->w; th = z->h; }
    /* The pass's scale is its first colour target's, else its depth's. */
    const float pass_f = eng_obj_f(nrt ? r->rt[0] : depth);
    const int scaled = pass_f != 1.0f;

    /* MSAA: a 3D pass (a real depth target) whose attachments are all
     * full-size targets draws into their multisampled twins. */
    int ms = s_ms_n > 1 && real_depth && eng_ms_ok(z);
    { static int nsm = -1; if (nsm < 0) { const char* e = getenv("RSX_MS_NO_SHADOW"); nsm = e ? atoi(e) : 0; }   /* default 0 = off; bits: 1 = Z16 shadow passes single-sampled; 2 = only square power-of-2 colour targets; 4 = depth-only passes */
      if (ms && nsm) {   /* 1 = the walker flagged a Z16 shadow pass (never seen at bind time in ToD: the flag is the walker's, the bind is the lane's); 2 = the only colour target is a square power-of-2 map; 4 = a pass with no colour target at all; 8 = the depth target's size differs from the colour targets' (ToD's sun shadow map: depth 1280x1024 = 3840x3072 at scale 3, whose x8 twin is 810 MB and is resolved every frame for nothing, bound together with a masked colour target of another size); 16 = log each distinct pass shape once ([ms-pass]) */
          if (nsm & 16) {
              static u32 seen[24][5]; static int nseen;
              int k; for (k = 0; k < nseen; k++) if (seen[k][0] == tw && seen[k][1] == th && seen[k][2] == z->w && seen[k][3] == z->h && seen[k][4] == nrt) break;
              if (k == nseen && nseen < 24) { seen[nseen][0] = tw; seen[nseen][1] = th; seen[nseen][2] = z->w; seen[nseen][3] = z->h; seen[nseen][4] = nrt; nseen++;
                  fprintf(stderr, "[ms-pass] shape %d: colour %ux%u x%u, depth %ux%u (guest %ux%u), zshadow %d\n", nseen, tw, th, nrt, z->w, z->h, z->gw, z->gh, z->zshadow); }
          }
          if (nrt) {
              const EngObj* c0 = eng_owner(r->rt[0]);
              const int sq = c0 && nrt == 1 && c0->gw == c0->gh && c0->gw >= 64 && (c0->gw & (c0->gw - 1)) == 0;
              /* bit 8: the depth target is larger than the colour target (wider, or taller by more than 32 guest rows): ToD binds one 1280x1024 scratch depth to every offscreen pass (sun map 1024^2, reflections, cube faces...), while the main scene is colour 1280x704 + depth 1280x720 (16 rows apart: keep its MSAA). Run n8 (07/11): "z->w != tw || z->h != th" also matched the main pass and turned MSAA off for the whole scene. */
              /* Compared in HOST pixels: targets of different guest sizes land on the same host size (a 960x528 target at scale 4 is 3840x2112 like the 1280x704 one at scale 3; run n8 b48 lost MSAA on 619 such passes with a guest comparison). Tolerance 64 px wide, 128 px tall (the main pass's 48 rows). */
              const int odd = c0 && (z->w > tw + 64 || z->h > th + 128);
              if (((nsm & 1) && z->zshadow) || ((nsm & 2) && sq) || ((nsm & 8) && odd)) {
                  static u32 seen_w, seen_h;
                  if ((nsm & 8) && odd && (z->w != seen_w || z->h != seen_h)) { seen_w = z->w; seen_h = z->h; fprintf(stderr, "[ms-noshadow] depth %ux%u (guest %ux%u) with colour %ux%u (guest %ux%u) x%u single-sampled (RSX_MS_NO_SHADOW bit 8)\n", z->w, z->h, z->gw, z->gh, tw, th, c0->gw, c0->gh, nrt); }
                  ms = 0;
              }
          } else if (nsm & 4) {
              static u32 seen_w, seen_h;
              if (z->w != seen_w || z->h != seen_h) { seen_w = z->w; seen_h = z->h; fprintf(stderr, "[ms-noshadow] depth-only pass into %ux%u (guest %ux%u) single-sampled (RSX_MS_NO_SHADOW bit 4)\n", z->w, z->h, z->gw, z->gh); }
              ms = 0;
          }
      } }
    for (u32 k = 0; ms && k < nrt; k++) ms = eng_ms_ok(eng_owner(r->rt[k]));
    if (ms) {
        ms = eng_ms_make(depth);
        for (u32 k = 0; ms && k < nrt; k++) ms = eng_ms_make(r->rt[k]);
    }
    /* A pass without a real depth target (post-processing, the UI) into
     * targets that already have twins draws into the twins too, against the
     * fallback depth's twin. Single-sampled, it had the twin resolved into
     * the target first and left the twin stale, refreshed by a full-screen
     * draw at the next 3D pass: a 4K resolve and a 4K draw for each of the
     * ~15 such passes Drakengard 3 interleaves with its 3D passes. */
    if (g_rsx_ms_twin < 0) { const char* e = getenv("RSX_MS_TWIN"); g_rsx_ms_twin = !(e && e[0] == '0'); }
    if (!ms && g_rsx_ms_twin > 0 && s_ms_n > 1 && !real_depth && nrt && eng_ms_ok(z)) {
        ms = 1;
        for (u32 k = 0; ms && k < nrt; k++) { EngObj* o = eng_owner(r->rt[k]); ms = o && o->ms != NULL; }
        if (ms) ms = eng_ms_make(depth);
    }
    { /* RSX_MS_LOG=1: per-target census of draws drawn multisampled or not (diagnosis 02/11: is Ratchet in an MSAA pass?) */
      static int mlg = -1; if (mlg < 0) mlg = getenv("RSX_MS_LOG") ? 1 : 0;
      if (mlg && s_ms_n > 1) {
        static struct { u32 w, h, rtfmt, nrt; int real, ms; unsigned long n; } tab[48]; static u32 nt; static unsigned long tot; static u32 lastf;
        EngObj* o0 = nrt ? eng_owner(r->rt[0]) : NULL;
        const u32 fm = o0 ? (u32)o0->fmt : 0; u32 i;
        for (i = 0; i < nt; i++) if (tab[i].w == tw && tab[i].h == th && tab[i].rtfmt == fm && tab[i].nrt == nrt && tab[i].real == real_depth && tab[i].ms == ms) break;
        if (i == nt && nt < 48) { tab[nt].w = tw; tab[nt].h = th; tab[nt].rtfmt = fm; tab[nt].nrt = nrt; tab[nt].real = real_depth; tab[nt].ms = ms; tab[nt].n = 0; nt++; }
        if (i < nt) tab[i].n++;
        if (++tot % 40000 == 0) {
          fprintf(stderr, "[ms-log] after %lu draws (samples x%u): target census\n", tot, s_ms_n);
          for (u32 k = 0; k < nt; k++) fprintf(stderr, "[ms-log]   %ux%u fmt%u nrt=%u real_depth=%d MS=%d : %lu draws\n", tab[k].w, tab[k].h, tab[k].rtfmt, tab[k].nrt, tab[k].real, tab[k].ms, tab[k].n);
        }
        (void)lastf;
      }
    }
    ID3D12PipelineState* pso = ms ? eng_pso_ms(r->pipeline, topo_class(r->topology))
                                  : eng_pso_for(r->pipeline, topo_class(r->topology), !scaled);
    if (!pso) { s_gt_nopso++; return; }
    if (ms) {
        for (u32 k = 0; k < nrt; k++) {
            if (eng_obj(r->rt[k])->ms_stale) eng_ms_refresh(r->rt[k]);
            rtv[k] = obj_rtv_ms(r->rt[k]);
        }
        if (z->ms_stale) eng_ms_refresh(depth);
        eng_mt_note(r, nrt, z);
        if (real_depth) z->ms_znewer = 1;   /* the fallback depth's twin holds nothing anyone reads back */
        for (u32 k = 0; k < nrt; k++) { EngObj* o = eng_obj(r->rt[k]); res_transition(o->ms, &o->ms_state, D3D12_RESOURCE_STATE_RENDER_TARGET); }
        res_transition(z->ms, &z->ms_state, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    } else {
        /* Drawn single-sampled into a target with a twin: the target must
         * have the twin's draws first, and the twin is stale after. */
        for (u32 k = 0; k < nrt; k++) {
            EngObj* o = eng_owner(r->rt[k]);
            if (o && o->ms) { if (o->ms_dirty) eng_ms_resolve(r->rt[k]); o->ms_stale = 1; }
        }
        if (z->ms && z->ms_znewer) eng_ms_depth_sync(depth);
        for (u32 k = 0; k < nrt; k++) obj_transition(r->rt[k], D3D12_RESOURCE_STATE_RENDER_TARGET);
        obj_transition(depth, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    }

    /* Textures, with a unit that names one of this draw's own targets bound
     * to the null texture rather than left in a read/write hazard. */
    D3D12_CPU_DESCRIPTOR_HANDLE srv[ENG_SRV_TABLE];
    for (u32 u = 0; u < ENG_SRV_TABLE; u++) {
        const u32 h = u < RSX_BE_MAX_TEXTURES ? r->tex[u] : r->vtex[u - RSX_BE_MAX_TEXTURES];
        EngObj* o = eng_owner(h);
        int hazard = 0;
        if (o) {
            for (u32 k = 0; k < nrt; k++) if (eng_owner(r->rt[k]) == o) hazard = 1;
            if (z == o) hazard = 1;
        }
        if (!o || hazard) { srv[u] = obj_srv(0); continue; }
        if (o->ms && o->ms_dirty) eng_ms_resolve((u32)(o - s_obj) + 1);
        /* A depth target read as a texture (ToD's shadow maps: the caster
         * pass is a 3D pass, so it drew into the twin) has to have the twin's
         * depth first; read bare it held none of it -- shadows broken all
         * over and the shadow-receiving enemies black. */
        if (o->ms && o->kind == OBJ_DEPTH && o->ms_znewer) eng_ms_depth_sync((u32)(o - s_obj) + 1);
        obj_transition(h, ENG_SHADER_READ);
        srv[u] = obj_srv(h);
    }
    bar_flush();
    eng_use_main_rootsig();

    /* The descriptor tables first: a full ring flushes the list
     * (eng_hard_flush), and on the new one nothing is bound and the old
     * tables' slots are reused -- so both are taken again until neither
     * flushed, then everything is bound on the list that draws. */
    D3D12_GPU_DESCRIPTOR_HANDLE st, sm;
    int smp[ENG_SRV_TABLE];
    for (u32 u = 0; u < RSX_BE_MAX_TEXTURES; u++) smp[u] = r->samp[u];
    for (u32 u = 0; u < RSX_BE_MAX_VERTEX_TEXTURES; u++) smp[RSX_BE_MAX_TEXTURES + u] = r->vsamp[u];
    for (int tries = 0;; tries++) {
        const u32 gen = s_list_gen;
        if (s_srv_last_valid && memcmp(s_srv_last_src, srv, sizeof srv) == 0) st = s_srv_last_gpu;
        else {
            st = eng_srv_table(srv, ENG_SRV_TABLE);
            memcpy(s_srv_last_src, srv, sizeof srv); s_srv_last_gpu = st; s_srv_last_valid = 1;
        }
        sm = eng_smp_table(smp);
        if (gen == s_list_gen || tries) break;
    }
    D3D12_GPU_DESCRIPTOR_HANDLE vt = st; vt.ptr += (UINT64)RSX_BE_MAX_TEXTURES * s_srv_step;
    D3D12_GPU_DESCRIPTOR_HANDLE vsm = sm; vsm.ptr += (UINT64)RSX_BE_MAX_TEXTURES * s_smp_step;

    /* What the previous draw left bound, so a draw sets only what differs.
     * Everything that binds outside this function clears s_cur_targets_valid
     * (helper passes, root signature switches, a new list), and that clears
     * this too. RSX_D3D12_REBIND=1 sets everything on every draw. */
    static struct {
        int valid;
        D3D12_VIEWPORT vp; D3D12_RECT sc; u32 sref; float blend[4];
        D3D12_VERTEX_BUFFER_VIEW vbv; D3D12_INDEX_BUFFER_VIEW ibv;
        D3D12_GPU_VIRTUAL_ADDRESS cb0, cb4;
        UINT64 t1, t2, t3, t5;
    } b;
    static int rebind = -1;
    if (rebind < 0) rebind = getenv("RSX_D3D12_REBIND") ? 1 : 0;
    if (!s_cur_targets_valid || rebind) b.valid = 0;

    static int s_cur_ms;
    int same_targets = s_cur_targets_valid && s_cur_nrt == nrt && s_cur_depth == depth && s_cur_ms == ms;
    for (u32 k = 0; same_targets && k < nrt; k++) same_targets = s_cur_rt[k] == r->rt[k];
    if (!same_targets) {
        s_cur_ms = ms;
        D3D12_CPU_DESCRIPTOR_HANDLE dsv = ms ? obj_dsv_ms(depth) : obj_dsv(depth);
        CALL(s_list, OMSetRenderTargets, nrt, nrt ? rtv : NULL, FALSE, &dsv);
        for (u32 k = 0; k < RSX_BE_MAX_COLOR_TARGETS; k++) s_cur_rt[k] = k < nrt ? r->rt[k] : 0;
        s_cur_nrt = nrt; s_cur_depth = depth; s_cur_targets_valid = 1;
    }
    if (pso != s_cur_pso) { CALL(s_list, SetPipelineState, pso); s_cur_pso = pso; s_gt_pso++; }
    const D3D_PRIMITIVE_TOPOLOGY topo = topo_d3d(r->topology);
    if (topo != s_cur_topo) { CALL(s_list, IASetPrimitiveTopology, topo); s_cur_topo = topo; }

    D3D12_VIEWPORT vp; D3D12_RECT sc;
    eng_vp_sc(r, pass_f, eng_obj_xx(nrt ? r->rt[0] : depth), tw, th, &vp, &sc);
    { static int dbg2 = -1; static unsigned long n_all, n_narrow; static int nn;
      if (dbg2 < 0) dbg2 = getenv("RSX_HS_LOG") ? 1 : 0;
      if (dbg2 && eng_obj_xx(nrt ? r->rt[0] : depth) != 1.0f) { n_all++;
        if (vp.Width < (float)tw - 1.0f || sc.right < (LONG)tw) { n_narrow++; if (nn < 6 && ++nn) fprintf(stderr, "[hs-log] NARROW draw: guest vp %g %g %g %g sc %u %u %u %u -> vp %g %g %g %g sc %ld..%ld (tw %u) ms=%d\n", (double)r->vp[0], (double)r->vp[1], (double)r->vp[2], (double)r->vp[3], r->sc[0], r->sc[1], r->sc[2], r->sc[3], (double)vp.TopLeftX, (double)vp.TopLeftY, (double)vp.Width, (double)vp.Height, (long)sc.left, (long)sc.right, tw, ms); }
        if (n_all % 20000 == 0) fprintf(stderr, "[hs-log] xx-target draws %lu, narrow %lu\n", n_all, n_narrow); } }
    { static int dbg3 = -1; static int n3; if (dbg3 < 0) dbg3 = getenv("RSX_HS_LOG") ? 1 : 0;
      if (dbg3 && n3 < 4 && eng_obj_xx(nrt ? r->rt[0] : depth) != 1.0f && ++n3)
          fprintf(stderr, "[hs-log] depth of the scene pass: host %ux%u guest %ux%u real_depth=%d ms=%d\n", z->w, z->h, z->gw, z->gh, real_depth, ms); }
    { static int dbg = -1; static int nd; if (dbg < 0) dbg = getenv("RSX_HS_LOG") ? 1 : 0;
      if (dbg && nd < 12 && eng_obj_xx(nrt ? r->rt[0] : depth) != 1.0f && ++nd)
          fprintf(stderr, "[hs-log] draw: guest vp %g %g %g %g sc %u %u %u %u -> host vp %g %g %g %g sc %ld %ld %ld %ld (pass_f %g, tw %u th %u)\n",
                  (double)r->vp[0], (double)r->vp[1], (double)r->vp[2], (double)r->vp[3], r->sc[0], r->sc[1], r->sc[2], r->sc[3],
                  (double)vp.TopLeftX, (double)vp.TopLeftY, (double)vp.Width, (double)vp.Height, (long)sc.left, (long)sc.top, (long)sc.right, (long)sc.bottom, (double)pass_f, tw, th); }
    if (!b.valid || memcmp(&vp, &b.vp, sizeof vp)) { CALL(s_list, RSSetViewports, 1, &vp); b.vp = vp; }
    if (!b.valid || memcmp(&sc, &b.sc, sizeof sc)) { CALL(s_list, RSSetScissorRects, 1, &sc); b.sc = sc; }
    if (!b.valid || r->stencil_ref != b.sref) { CALL(s_list, OMSetStencilRef, r->stencil_ref); b.sref = r->stencil_ref; }
    if (!b.valid || memcmp(r->blend, b.blend, sizeof b.blend)) { CALL(s_list, OMSetBlendFactor, r->blend); memcpy(b.blend, r->blend, sizeof b.blend); }

    D3D12_GPU_VIRTUAL_ADDRESS vb_va, ib_va; u32 vleft;
    if (!eng_draw_buffers(r, &vb_va, &ib_va, &vleft)) { b.valid = 0; return; }
    D3D12_VERTEX_BUFFER_VIEW vbv = { vb_va, vleft, r->stride };
    if (vbv.SizeInBytes > (u64)r->vertex_count * r->stride && !r->index_count) vbv.SizeInBytes = r->vertex_count * r->stride;
    if (!b.valid || memcmp(&vbv, &b.vbv, sizeof vbv)) { CALL(s_list, IASetVertexBuffers, 0, 1, &vbv); b.vbv = vbv; }
    if (r->index_count) {
        D3D12_INDEX_BUFFER_VIEW ibv = { ib_va, r->index_count * 4u, DXGI_FORMAT_R32_UINT };
        if (!b.valid || memcmp(&ibv, &b.ibv, sizeof ibv)) { CALL(s_list, IASetIndexBuffer, &ibv); b.ibv = ibv; }
    }
    const D3D12_GPU_VIRTUAL_ADDRESS cb0 = eng_cb_va(r->vs_cb_off, r->vs_cb_bytes);
    const D3D12_GPU_VIRTUAL_ADDRESS cb4 = eng_cb_va(r->ps_cb_off, r->ps_cb_bytes);
    if (!b.valid || cb0 != b.cb0) { CALL(s_list, SetGraphicsRootConstantBufferView, 0, cb0); b.cb0 = cb0; }
    if (!b.valid || cb4 != b.cb4) { CALL(s_list, SetGraphicsRootConstantBufferView, 4, cb4); b.cb4 = cb4; }
    if (!b.valid || st.ptr != b.t1)  { CALL(s_list, SetGraphicsRootDescriptorTable, 1, st);  b.t1 = st.ptr; }
    if (!b.valid || vt.ptr != b.t3)  { CALL(s_list, SetGraphicsRootDescriptorTable, 3, vt);  b.t3 = vt.ptr; }
    if (!b.valid || sm.ptr != b.t2)  { CALL(s_list, SetGraphicsRootDescriptorTable, 2, sm);  b.t2 = sm.ptr; }
    if (!b.valid || vsm.ptr != b.t5) { CALL(s_list, SetGraphicsRootDescriptorTable, 5, vsm); b.t5 = vsm.ptr; }
    b.valid = 1;

    const int query = s_qheap && r->vis && s_q_used < ENG_QUERIES_PER_SLOT;
    const u32 qi = s_slot * ENG_QUERIES_PER_SLOT + s_q_used;
    if (query) { CALL(s_list, BeginQuery, s_qheap, D3D12_QUERY_TYPE_OCCLUSION, qi); s_gt_queries++; }
    s_gt_draws++;
    eng_draw_ring_add((u32)r->pipeline, r->index_count, r->vertex_count);
    if (r->index_count) CALL(s_list, DrawIndexedInstanced, r->index_count, 1, 0, 0, 0);
    else                CALL(s_list, DrawInstanced, r->vertex_count, 1, 0, 0);
    if (query) {
        CALL(s_list, EndQuery, s_qheap, D3D12_QUERY_TYPE_OCCLUSION, qi);
        const u32 lg = !ms ? 0u : s_ms_n == 2 ? 1u : s_ms_n == 4 ? 2u : 3u;
        s_q_vis[s_q_used++] = (r->vis - 1) | (scaled && pass_f == s_scale ? ENG_Q_SCALED : 0u) | (eng_obj_xx(nrt ? r->rt[0] : depth) != 1.0f ? ENG_Q_XX : 0u) | (lg << ENG_Q_MS_SHIFT);
    }
    if (ms) for (u32 k = 0; k < nrt; k++) eng_ms_touch(r->rt[k], &sc);
    if (ms) eng_mt_extent(z, r, nrt, &sc);
}

/* ---- incremental own-target snapshots --------------------------------------
 *
 * A draw that samples the colour target it renders into samples a snapshot
 * of it (eng_color_snapshot), refreshed at that point in the stream by a full
 * copy -- ~0.1 ms of GPU for a 4K RGBA16F target, and Drakengard 3's heat-haze
 * particles ask for ~95 a frame. Between two refreshes the target only
 * changes where the draws into it in between wrote, and those are recorded:
 * draw each of them again, same vertex shader and geometry, viewport and
 * scissor, with a pixel shader that copies the target's pixel into the
 * snapshot, and the snapshot equals the target everywhere again -- exactly,
 * at the cost of the draws' own footprints. Culling and depth are off in the
 * copy, so it covers at least every pixel the draw could have written; more
 * is harmless, those pixels are copies of an unchanged target. Anything else
 * that writes the target (a clear, a copy, an upload, a resolve, a draw with
 * it as a second attachment), more than 16 draws, or a refresh with no
 * earlier one in the same submit takes the full copy. RSX_SNAP_INCR=0: full
 * copies always. */

static ID3D12PipelineState* eng_copy_pso_build(u32 pipeline, int cls, DXGI_FORMAT fmt, int ms)
{
    ID3DBlob* ps_blob = ms ? s_snap_copy_ms_ps : s_snap_copy_ps;
    if (!ps_blob || !pipeline || pipeline > s_pipe_count) return NULL;
    EngPipeline* p = &s_pipe[pipeline - 1];
    ID3D12PipelineState** slot = ms ? &p->copy_pso_ms[cls] : &p->copy_pso[cls];
    DXGI_FORMAT* slot_fmt = ms ? &p->copy_fmt_ms[cls] : &p->copy_fmt[cls];
    int* failed = ms ? &p->copy_failed_ms[cls] : &p->copy_failed[cls];
    if (!p->vs || *failed) return NULL;
    if (*slot && *slot_fmt == fmt) return *slot;
    /* As eng_pso: built under no lock, published under it. */
    ID3D12PipelineState* pso = NULL;
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = p->desc;
        pd.InputLayout.pInputElementDescs = pd.InputLayout.NumElements ? p->il : NULL;
        pd.PS.pShaderBytecode = CALL0(ps_blob, GetBufferPointer);
        pd.PS.BytecodeLength = CALL0(ps_blob, GetBufferSize);
        pd.NumRenderTargets = 1;
        for (u32 r = 0; r < 8; r++) pd.RTVFormats[r] = DXGI_FORMAT_UNKNOWN;
        pd.RTVFormats[0] = fmt;
        memset(&pd.BlendState, 0, sizeof pd.BlendState);
        D3D12_RENDER_TARGET_BLEND_DESC* b = &pd.BlendState.RenderTarget[0];
        b->SrcBlend = D3D12_BLEND_ONE; b->DestBlend = D3D12_BLEND_ZERO; b->BlendOp = D3D12_BLEND_OP_ADD;
        b->SrcBlendAlpha = D3D12_BLEND_ONE; b->DestBlendAlpha = D3D12_BLEND_ZERO; b->BlendOpAlpha = D3D12_BLEND_OP_ADD;
        b->LogicOp = D3D12_LOGIC_OP_NOOP;
        b->RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        memset(&pd.DepthStencilState, 0, sizeof pd.DepthStencilState);
        pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        const D3D12_DEPTH_STENCILOP_DESC keep = { D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                                                  D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS };
        pd.DepthStencilState.FrontFace = keep; pd.DepthStencilState.BackFace = keep;
        pd.DSVFormat = DXGI_FORMAT_UNKNOWN;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.PrimitiveTopologyType = cls == 0 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT
                                 : cls == 1 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
                                            : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        HRESULT hr = CALL(s_dev, CreateGraphicsPipelineState, &pd, &IID_ID3D12PipelineState, (void**)&pso);
        eng_lock(&s_pipe_lock, "pipe_lock");
        if (*slot && *slot_fmt == fmt) { RELEASE(pso); pso = *slot; }
        else if (FAILED(hr)) {
            static int n = 0;
            if (n < 8 && ++n) fprintf(stderr, "[rsx engine/d3d12] snapshot-copy pipeline failed: 0x%08lX\n", (long)hr);
            *failed = 1; pso = NULL;
        } else if (ms && ps_blob != s_snap_copy_ms_ps) {
            RELEASE(pso);   /* the sample count changed meanwhile: stale */
        } else {
            /* A format change leaves the old one to any list still naming it. */
            *slot = pso; *slot_fmt = fmt;
        }
        ReleaseSRWLockExclusive(&s_pipe_lock);
    }
    return pso;
}

/* Copy pipelines are built on a thread of their own: a pipeline state is
 * milliseconds of driver compile, and on the walker each first use was a
 * hitch. Until one is ready its refreshes take the full copy. With
 * RSX_ASYNC_SHADERS=0 (the replays) they are built in place. */
static SRWLOCK s_cp_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE s_cp_cv = CONDITION_VARIABLE_INIT;
static struct { u32 pipeline; int cls; DXGI_FORMAT fmt; int ms; } s_cp_req[256];
static u32 s_cp_head, s_cp_tail;      /* ring of requests */
static int s_cp_thread_started;
static DWORD WINAPI eng_copy_pso_worker(LPVOID arg)
{
    (void)arg;
    for (;;) {
        eng_lock(&s_cp_lock, "cp_lock");
        while (s_cp_head == s_cp_tail) SleepConditionVariableSRW(&s_cp_cv, &s_cp_lock, INFINITE, 0);
        const u32 i = s_cp_head++ % 256u;
        const u32 pl = s_cp_req[i].pipeline; const int cls = s_cp_req[i].cls; const DXGI_FORMAT fmt = s_cp_req[i].fmt;
        const int ms = s_cp_req[i].ms;
        ReleaseSRWLockExclusive(&s_cp_lock);
        eng_copy_pso_build(pl, cls, fmt, ms);
    }
    return 0;
}
static ID3D12PipelineState* eng_copy_pso(u32 pipeline, int cls, DXGI_FORMAT fmt, int ms)
{
    if (!(ms ? s_snap_copy_ms_ps : s_snap_copy_ps) || !pipeline || pipeline > s_pipe_count) return NULL;
    EngPipeline* p = &s_pipe[pipeline - 1];
    if (!p->vs || (ms ? p->copy_failed_ms[cls] : p->copy_failed[cls])) return NULL;
    if (ms) { if (p->copy_pso_ms[cls] && p->copy_fmt_ms[cls] == fmt) return p->copy_pso_ms[cls]; }
    else    { if (p->copy_pso[cls] && p->copy_fmt[cls] == fmt) return p->copy_pso[cls]; }
    static int async = -1;
    if (async < 0) { const char* e = getenv("RSX_ASYNC_SHADERS"); async = !(e && e[0] == '0'); }
    if (!async) return eng_copy_pso_build(pipeline, cls, fmt, ms);
    /* Ask once per (pipeline, class, format, variant): requests[] remembers. */
    static struct { u32 pipeline; int cls; DXGI_FORMAT fmt; int ms; } asked[1024];
    static u32 nasked;
    for (u32 i = 0; i < nasked; i++)
        if (asked[i].pipeline == pipeline && asked[i].cls == cls && asked[i].fmt == fmt && asked[i].ms == ms) return NULL;
    if (nasked < 1024) { asked[nasked].pipeline = pipeline; asked[nasked].cls = cls; asked[nasked].fmt = fmt; asked[nasked].ms = ms; nasked++; }
    eng_lock(&s_cp_lock, "cp_lock");
    if (!s_cp_thread_started) {
        HANDLE th = CreateThread(NULL, 1u << 20, eng_copy_pso_worker, NULL, 0, NULL);
        if (th) { SetThreadDescription(th, L"rsx copy pipelines"); CloseHandle(th); s_cp_thread_started = 1; }
    }
    if (s_cp_thread_started && s_cp_tail - s_cp_head < 256u) {
        const u32 i = s_cp_tail++ % 256u;
        s_cp_req[i].pipeline = pipeline; s_cp_req[i].cls = cls; s_cp_req[i].fmt = fmt; s_cp_req[i].ms = ms;
        WakeConditionVariable(&s_cp_cv);
    }
    ReleaseSRWLockExclusive(&s_cp_lock);
    return NULL;
}

/* Draw r's geometry into snapshot S with the copy shader reading target T. */
static int eng_encode_copy_draw(const EngRecord* r, u32 T, u32 S)
{
    EngObj* t = eng_owner(T); EngObj* sn = eng_owner(S);
    if (!t || !sn || !sn->has_rtv) return 0;
    /* From the twin while it holds draws the target lacks. A snapshot used
     * to resolve the whole twin first: at 4K, ~25 snapshots a frame, that
     * resolve was the largest single item of the MSAA frame's GPU time. */
    const int ms = t->ms && t->ms_dirty;
    ID3D12PipelineState* pso = eng_copy_pso(r->pipeline, topo_class(r->topology), sn->fmt, ms);
    if (!pso) return 0;
    D3D12_GPU_VIRTUAL_ADDRESS vb_va, ib_va; u32 vleft;
    if (!eng_draw_buffers(r, &vb_va, &ib_va, &vleft)) return 0;
    /* Tables first: a full descriptor ring flushes the list and rebinds. */
    D3D12_CPU_DESCRIPTOR_HANDLE srv[ENG_SRV_TABLE];
    for (u32 u = 0; u < ENG_SRV_TABLE; u++) srv[u] = obj_srv(0);
    srv[0] = ms ? obj_srv_ms((u32)(t - s_obj) + 1) : obj_srv(T);
    u32 vt_h[RSX_BE_MAX_VERTEX_TEXTURES] = {0};
    for (u32 u = 0; u < RSX_BE_MAX_VERTEX_TEXTURES; u++) {
        EngObj* o = eng_owner(r->vtex[u]);
        if (o && o != t && o != sn) { srv[RSX_BE_MAX_TEXTURES + u] = obj_srv(r->vtex[u]); vt_h[u] = r->vtex[u]; }
    }
    int smp[ENG_SRV_TABLE];
    for (u32 u = 0; u < RSX_BE_MAX_TEXTURES; u++) smp[u] = r->samp[u];
    for (u32 u = 0; u < RSX_BE_MAX_VERTEX_TEXTURES; u++) smp[RSX_BE_MAX_TEXTURES + u] = r->vsamp[u];
    /* A flush by the sampler table reuses the SRV table's slots: both again. */
    D3D12_GPU_DESCRIPTOR_HANDLE st, sm;
    for (int tries = 0;; tries++) {
        const u32 gen = s_list_gen;
        st = eng_srv_table(srv, ENG_SRV_TABLE);
        sm = eng_smp_table(smp);
        if (gen == s_list_gen || tries) break;
    }

    if (ms) res_transition(t->ms, &t->ms_state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    else    res_transition(t->res, &t->state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    res_transition(sn->res, &sn->state, D3D12_RESOURCE_STATE_RENDER_TARGET);
    for (u32 u = 0; u < RSX_BE_MAX_VERTEX_TEXTURES; u++) if (vt_h[u]) obj_transition(vt_h[u], ENG_SHADER_READ);
    bar_flush();
    eng_use_main_rootsig();
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = obj_rtv(S);
    CALL(s_list, OMSetRenderTargets, 1, &rtv, FALSE, NULL);
    CALL(s_list, SetPipelineState, pso);
    CALL(s_list, IASetPrimitiveTopology, topo_d3d(r->topology));
    D3D12_VIEWPORT vp; D3D12_RECT sc;
    eng_vp_sc(r, eng_obj_f(T), eng_obj_xx(T), t->w, t->h, &vp, &sc);
    CALL(s_list, RSSetViewports, 1, &vp);
    CALL(s_list, RSSetScissorRects, 1, &sc);
    D3D12_VERTEX_BUFFER_VIEW vbv = { vb_va, vleft, r->stride };
    if (vbv.SizeInBytes > (u64)r->vertex_count * r->stride && !r->index_count) vbv.SizeInBytes = r->vertex_count * r->stride;
    CALL(s_list, IASetVertexBuffers, 0, 1, &vbv);
    if (r->index_count) {
        D3D12_INDEX_BUFFER_VIEW ibv = { ib_va, r->index_count * 4u, DXGI_FORMAT_R32_UINT };
        CALL(s_list, IASetIndexBuffer, &ibv);
    }
    const D3D12_GPU_VIRTUAL_ADDRESS zero_va = CALL0(s_zero_cb, GetGPUVirtualAddress);
    CALL(s_list, SetGraphicsRootConstantBufferView, 0, eng_cb_va(r->vs_cb_off, r->vs_cb_bytes));
    CALL(s_list, SetGraphicsRootConstantBufferView, 4, zero_va);
    D3D12_GPU_DESCRIPTOR_HANDLE vt = st; vt.ptr += (UINT64)RSX_BE_MAX_TEXTURES * s_srv_step;
    D3D12_GPU_DESCRIPTOR_HANDLE vsm = sm; vsm.ptr += (UINT64)RSX_BE_MAX_TEXTURES * s_smp_step;
    CALL(s_list, SetGraphicsRootDescriptorTable, 1, st);
    CALL(s_list, SetGraphicsRootDescriptorTable, 3, vt);
    CALL(s_list, SetGraphicsRootDescriptorTable, 2, sm);
    CALL(s_list, SetGraphicsRootDescriptorTable, 5, vsm);
    eng_draw_ring_add(0x80000000u | (u32)r->pipeline, r->index_count, r->vertex_count);
    if (r->index_count) CALL(s_list, DrawIndexedInstanced, r->index_count, 1, 0, 0, 0);
    else                CALL(s_list, DrawInstanced, r->vertex_count, 1, 0, 0);
    /* Everything the draw path caches is stale now. */
    s_cur_pso = NULL; s_cur_targets_valid = 0; s_srv_last_valid = 0;
    s_cur_topo = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
    s_snap_copy_draws++;
    return 1;
}

static int src_ok_for_sync(u32 T, u32 S)
{
    EngObj* t = eng_owner(T); EngObj* sn = eng_owner(S);
    return t && sn && sn->has_rtv && t->w == sn->w && t->h == sn->h && t->fmt == sn->fmt;
}

static void eng_pass_mark(const EngRecord* r, u64* cur_key)
{
    int kind; u32 h1, h2 = 0;
    switch (r->kind) {
    case ENG_REC_DRAW:          kind = 0; h1 = r->nrt ? r->rt[0] : 0; h2 = r->depth; break;
    case ENG_REC_CLEAR_COLOR:   kind = 1; h1 = r->rt[0]; break;
    case ENG_REC_CLEAR_DS:
    case ENG_REC_CLEAR_DS_RECT: kind = 2; h1 = r->depth; break;
    case ENG_REC_COLOR_COPY:    kind = 3; h1 = r->resolve_dst; break;
    case ENG_REC_DEPTH_RESOLVE: kind = 4; h1 = r->resolve_dst; break;
    default:                    kind = 5; h1 = 0; break;
    }
    const u64 key = ((u64)kind << 56) | ((u64)(h1 & 0xFFFFFFu) << 24) | (h2 & 0xFFFFFFu);
    u32* n = &s_pass_n[s_slot];
    if (key != *cur_key && *n < ENG_PASS_MAX) {
        CALL(s_list, EndQuery, s_pheap, D3D12_QUERY_TYPE_TIMESTAMP, s_slot * (ENG_PASS_MAX + 1u) + *n);
        EngPassDesc* d = &s_pass[s_slot][*n];
        memset(d, 0, sizeof *d);
        d->kind = (u8)kind; d->nrt = (u8)(kind == 0 ? r->nrt : 0);
        EngObj* o = eng_owner(h1 ? h1 : h2);
        if (o) { d->w = (u16)o->w; d->h = (u16)o->h; d->fmt = (u32)o->fmt; d->garea = o->gw * o->gh; }
        (*n)++;
        *cur_key = key;
    }
    if (kind == 0 && *n) s_pass[s_slot][*n - 1].draws++;
}

/* RT_AO: trace + filter from the depth, then multiply the scene colour (or its MSAA twin).
 * RT_AO_AB=<n> (diagnostic, with PS3RECOMP_FRAME_DUMP): the AO is left out of the frame whose present
 * number is 1 mod n, and eng_dump_frame writes the frames 0 and 1 mod n: an A/B pair one frame apart. */
static void eng_encode_rt_ao(const EngRecord* r)
{
    { static unsigned ab = ~0u; if (ab == ~0u) { const char* e = getenv("RT_AO_AB"); ab = e ? (unsigned)atoi(e) : 0; }
      if (ab >= 2 && (s_pres_n % ab) == 1) return; }
    { const int vw = rsx_rt_view_of(s_pres_n); rsx_rt_set_view(vw); if (vw == 3) return; }   /* RT_AO_VIEWS: view 3 = RT off */
    EngObj* c = eng_owner(r->rt[0]); EngObj* z = eng_owner(r->depth);
    if (!c || !z || !c->has_rtv || !z->res) return;
    static unsigned said;
    if (said < 3 && ++said)
        fprintf(stderr, "[rt] AO pass #%u: colour %ux%u fmt %d ms %s, depth %ux%u ms %s\n", said, c->w, c->h, (int)c->fmt,
                c->ms ? "yes" : "no", z->w, z->h, z->ms ? "yes" : "no");
    if (z->ms && z->ms_znewer) eng_ms_depth_sync(r->depth);
    res_transition(z->res, &z->state, ENG_SHADER_READ);
    bar_flush();
    RsxRtAoIn in;
    memset(&in, 0, sizeof in);
    in.list = s_list; in.srv_table = eng_srv_table; in.depth_res = z->res; in.depth_srv = obj_srv(r->depth);
    in.w = c->w; in.h = c->h; in.job = r->rt_job; in.slot = s_slot; in.seq = s_enc_seq;
    ID3D12Resource* ao = NULL; D3D12_CPU_DESCRIPTOR_HANDLE ao_srv;
    if (!rsx_rt_d3d12_ao(&in, &ao, &ao_srv)) return;
    /* The colour the scene passes drew: the MSAA twin while it holds draws the resolved target lacks. */
    const u32 h = r->rt[0];
    int use_ms = c->ms && c->ms_dirty;
    if (use_ms) res_transition(c->ms, &c->ms_state, D3D12_RESOURCE_STATE_RENDER_TARGET);
    else { if (c->ms) { c->ms_stale = 1; } res_transition(c->res, &c->state, D3D12_RESOURCE_STATE_RENDER_TARGET); }
    bar_flush();
    RsxRtCompIn ci;
    memset(&ci, 0, sizeof ci);
    ci.list = s_list; ci.srv_table = eng_srv_table;
    ci.rtv = use_ms ? obj_rtv_ms(h) : obj_rtv(h);
    ci.ao_srv = ao_srv; ci.w = c->w; ci.h = c->h; ci.samples = use_ms ? s_ms_n : 1; ci.fmt = (int)c->fmt;
    rsx_rt_d3d12_composite(&ci);
    rsx_rt_d3d12_ao_end(s_list, s_slot);
    /* Our passes bound a root signature, targets and pipeline of their own. */
    CALL(s_list, SetGraphicsRootSignature, s_rootsig);
    s_cur_helper_rs = 0; s_cur_pso = NULL; s_cur_targets_valid = 0; s_srv_last_valid = 0;
    s_cur_topo = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
}

static void eng_encode_records(const EngRecord* rec, u32 rec_n, u32 seq)
{
    u64 pass_key = ~0ull;
    eng_mt_setup();
    s_mt_rec = rec; s_mt_rn = rec_n; s_mt_scanned = 0;
    if (s_mt_pool && s_mt_final && s_ms_n > 1) { eng_mt_scan(rec, rec_n); s_mt_scanned = 1; }
    if (g_rsx_rt_on) rsx_rt_d3d12_process(s_list, s_slot, seq);
    /* g_rsx_snap_incr: -1 until read from RSX_SNAP_INCR; the host A/B (DOD3_AB=snapincr) flips it. */
    if (g_rsx_snap_incr < 0) { const char* e = getenv("RSX_SNAP_INCR"); g_rsx_snap_incr = !(e && e[0] == '0'); }
    const int incr = g_rsx_snap_incr;
    /* Per target with a snapshot: whether the snapshot was brought up to date
     * earlier in this submit, and the draws into the target since. */
    enum { SY_MAX = 8, SY_DRAWS = 16 };
    struct { u32 surf, snap, n; int valid; u32 idx[SY_DRAWS]; } sy[SY_MAX];
    u32 nsy = 0;
    for (u32 i = 0; i < rec_n; i++) {
        const EngRecord* r = &rec[i];
        s_mt_ri = i;
        if (s_pheap) eng_pass_mark(r, &pass_key);
        if (r->kind == ENG_REC_COLOR_COPY) {
            const u32 T = r->depth, S = r->resolve_dst;
            /* The copy draws read the twin while it is the newer of the two
             * (eng_encode_copy_draw); only the full copy needs the target
             * itself brought up to date. */
            EngObj* tobj = eng_owner(T);
            const int tms = g_rsx_ms_twin > 0 && tobj && tobj->ms && tobj->ms_dirty;
            if (!tms && tobj && tobj->ms && tobj->ms_dirty) eng_ms_resolve((u32)(tobj - s_obj) + 1);   /* RSX_MS_TWIN=0: as before */
            u32 k = 0;
            while (k < nsy && sy[k].surf != T) k++;
            int done = 0;
            EngObj* sobj = eng_owner(S);
            { static int dbg = -1; if (dbg < 0) dbg = getenv("RSX_SNAP_DEBUG") ? 1 : 0;
              static unsigned nd = 0;
              if (dbg && nd++ < 60)
                  fprintf(stderr, "[snap-debug] rec %u T=%u S=%u: entry %s valid %d snap-match %d rtv %d pending %u\n",
                          i, T, S, k < nsy ? "yes" : "no", k < nsy ? sy[k].valid : -1, k < nsy ? (sy[k].snap == S) : -1,
                          sobj ? sobj->has_rtv : -1, k < nsy ? sy[k].n : 0); }
            int ready = 1;
            if (incr && k < nsy && sy[k].valid && sy[k].snap == S && sobj && sobj->has_rtv) {
                /* Every copy pipeline ready, or the full copy this time. */
                for (u32 j = 0; j < sy[k].n; j++)
                    if (!eng_copy_pso(rec[sy[k].idx[j]].pipeline, topo_class(rec[sy[k].idx[j]].topology), sobj->fmt, tms)) { ready = 0; break; }
            }
            if (incr && ready && k < nsy && sy[k].valid && sy[k].snap == S && sobj && sobj->has_rtv) {
                done = 1;
                for (u32 j = 0; j < sy[k].n; j++)
                    if (!eng_encode_copy_draw(&rec[sy[k].idx[j]], T, S)) { done = 0; break; }
            }
            if (done) {
                s_snap_incr_n++;
                static int said = 0;
                if (said < 2 && ++said)
                    fprintf(stderr, "[rsx engine/d3d12] own-target snapshot brought up to date by %u copy draw(s) instead of a full copy\n",
                            sy[k].n);
            }
            else {
                EngObj* src = eng_owner(T); EngObj* dst = sobj;
                if (src && dst) {
                    if (tms) eng_ms_resolve((u32)(src - s_obj) + 1);
                    res_transition(src->res, &src->state, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    res_transition(dst->res, &dst->state, D3D12_RESOURCE_STATE_COPY_DEST);
                    bar_flush();
                    CALL(s_list, CopyResource, dst->res, src->res);
                    s_snap_full_n++;
                }
            }
            if (k == nsy && nsy < SY_MAX) nsy++;
            if (k < nsy) { sy[k].surf = T; sy[k].snap = S; sy[k].n = 0; sy[k].valid = (src_ok_for_sync(T, S)); }
            continue;
        }
        /* Writes to a tracked target: a draw as attachment A is replayable,
         * anything else is not. */
        for (u32 k = 0; k < nsy; k++) {
            if (!sy[k].valid) continue;
            const u32 T = sy[k].surf, S = sy[k].snap;
            switch (r->kind) {
            case ENG_REC_DRAW:
                for (u32 a = 0; a < r->nrt && a < RSX_BE_MAX_COLOR_TARGETS; a++) {
                    if (r->rt[a] == S) sy[k].valid = 0;
                    if (r->rt[a] != T) continue;
                    if (a == 0 && sy[k].n < SY_DRAWS) sy[k].idx[sy[k].n++] = i;
                    else sy[k].valid = 0;
                }
                break;
            case ENG_REC_CLEAR_COLOR:
                if (r->rt[0] == T || r->rt[0] == S) sy[k].valid = 0;
                break;
            case ENG_REC_DEPTH_RESOLVE:
            case ENG_REC_COLOR_RECT:
                if (r->resolve_dst == T || r->resolve_dst == S) sy[k].valid = 0;
                break;
            case ENG_REC_UPLOAD:
                if (r->depth == T || r->depth == S) sy[k].valid = 0;
                break;
            default:
                break;
            }
        }
        switch (r->kind) {
        case ENG_REC_DRAW:
            eng_encode_draw(r);
            break;
        case ENG_REC_UPLOAD: {
            EngObj* o = eng_owner(r->depth);
            if (o && o->ms) { if (o->ms_dirty) eng_ms_resolve((u32)(o - s_obj) + 1); o->ms_stale = 1; }
            eng_encode_upload(r);
            break; }
        case ENG_REC_RT_AO:
            eng_encode_rt_ao(r);
            break;
        case ENG_REC_CLEAR_COLOR: {
            EngObj* o = eng_owner(r->rt[0]);
            if (!o || !o->has_rtv) break;
            if (r->clear_flags & 1u) {   /* RSX_RT_HEAP: first use of a placed target's range (eng_rt_zero_record) */
                if (s_nbar == 32) bar_flush();
                D3D12_RESOURCE_BARRIER* ab = &s_bar[s_nbar++];
                memset(ab, 0, sizeof *ab);
                ab->Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
                ab->Aliasing.pResourceBefore = NULL; ab->Aliasing.pResourceAfter = o->res;
            }
            obj_transition(r->rt[0], D3D12_RESOURCE_STATE_RENDER_TARGET);
            if (o->ms) res_transition(o->ms, &o->ms_state, D3D12_RESOURCE_STATE_RENDER_TARGET);
            bar_flush();
            CALL(s_list, ClearRenderTargetView, obj_rtv(r->rt[0]), r->clear_rgba, 0, NULL);
            if (o->ms) {   /* both cleared: nothing to resolve or refresh */
                const u32 oh = (u32)(o - s_obj) + 1;
                CALL(s_list, ClearRenderTargetView, obj_rtv_ms(oh), r->clear_rgba, 0, NULL);
                o->ms_dirty = 0; o->ms_stale = 0;
            }
            break; }
        case ENG_REC_CLEAR_DS:
        case ENG_REC_CLEAR_DS_RECT: {
            EngObj* z = eng_owner(r->depth);
            if (!z || !z->has_dsv) break;
            D3D12_CLEAR_FLAGS f = 0;
            if (r->clear_flags & RSX_BE_CLEAR_DEPTH)   f |= D3D12_CLEAR_FLAG_DEPTH;
            if (r->clear_flags & RSX_BE_CLEAR_STENCIL) f |= D3D12_CLEAR_FLAG_STENCIL;
            if (!f) break;
            if (r->clear_flags & DSH_INIT_FLAG) {   /* RSX_DS_HEAP: first use of a placed zeta's range (eng_ds_zero_record) */
                if (s_nbar == 32) bar_flush();
                D3D12_RESOURCE_BARRIER* ab = &s_bar[s_nbar++];
                memset(ab, 0, sizeof *ab);
                ab->Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
                ab->Aliasing.pResourceBefore = NULL; ab->Aliasing.pResourceAfter = z->res;
            }
            obj_transition(r->depth, D3D12_RESOURCE_STATE_DEPTH_WRITE);
            if (z->ms) res_transition(z->ms, &z->ms_state, D3D12_RESOURCE_STATE_DEPTH_WRITE);
            bar_flush();
            const u32 zh = (u32)(z - s_obj) + 1;
            if (r->kind == ENG_REC_CLEAR_DS_RECT) {
                u32 x = r->clear_rect[0], y = r->clear_rect[1], w = r->clear_rect[2], h = r->clear_rect[3];
                if (x >= z->w || y >= z->h) break;
                if (x + w > z->w) w = z->w - x;
                if (y + h > z->h) h = z->h - y;
                D3D12_RECT rc = { (LONG)x, (LONG)y, (LONG)(x + w), (LONG)(y + h) };
                CALL(s_list, ClearDepthStencilView, obj_dsv(r->depth), f, r->clear_depth, r->clear_stencil, 1, &rc);
                if (z->ms) CALL(s_list, ClearDepthStencilView, obj_dsv_ms(zh), f, r->clear_depth, r->clear_stencil, 1, &rc);
            } else {
                CALL(s_list, ClearDepthStencilView, obj_dsv(r->depth), f, r->clear_depth, r->clear_stencil, 0, NULL);
                if (z->ms) {
                    CALL(s_list, ClearDepthStencilView, obj_dsv_ms(zh), f, r->clear_depth, r->clear_stencil, 0, NULL);
                    if (f == (D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL)) z->ms_stale = 0;
                    if (f & D3D12_CLEAR_FLAG_DEPTH) z->ms_znewer = 0;
                }
            }
            break; }
        case ENG_REC_COLOR_COPY: {
            EngObj* src = eng_owner(r->depth); EngObj* dst = eng_owner(r->resolve_dst);
            if (!src || !dst) break;
            if (src->ms && src->ms_dirty) eng_ms_resolve((u32)(src - s_obj) + 1);
            res_transition(src->res, &src->state, D3D12_RESOURCE_STATE_COPY_SOURCE);
            res_transition(dst->res, &dst->state, D3D12_RESOURCE_STATE_COPY_DEST);
            bar_flush();
            CALL(s_list, CopyResource, dst->res, src->res);
            break; }
        case ENG_REC_COLOR_RECT: {
            EngObj* src = eng_owner(r->depth); EngObj* dst = eng_owner(r->resolve_dst);
            if (!src || !dst || !dst->has_rtv || !src->gw || !src->gh || !dst->gw || !dst->gh) break;
            if (src->ms && src->ms_dirty) eng_ms_resolve((u32)(src - s_obj) + 1);
            if (dst->ms) { if (dst->ms_dirty) eng_ms_resolve((u32)(dst - s_obj) + 1); dst->ms_stale = 1; }
            /* Destination host pixels per source guest pixel, each axis. */
            const float kx = (float)dst->w / (float)dst->gw, ky = (float)dst->h / (float)dst->gh;
            const float ax = kx * (float)r->sc[2] / (float)r->clear_rect[2];
            const float ay = ky * (float)r->sc[3] / (float)r->clear_rect[3];
            const float vp[4] = { (float)r->sc[0] * kx - (float)r->clear_rect[0] * ax,
                                  (float)r->sc[1] * ky - (float)r->clear_rect[1] * ay,
                                  (float)src->gw * ax, (float)src->gh * ay };
            const int rc[4] = { (int)((float)r->sc[0] * kx + 0.5f), (int)((float)r->sc[1] * ky + 0.5f),
                                (int)((float)r->sc[2] * kx + 0.5f), (int)((float)r->sc[3] * ky + 0.5f) };
            res_transition(src->res, &src->state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            res_transition(dst->res, &dst->state, D3D12_RESOURCE_STATE_RENDER_TARGET);
            bar_flush();
            eng_fullscreen_pass_at(s_blit_pso, obj_rtv(r->resolve_dst), dst->w, dst->h, obj_srv(r->depth), vp, rc);
            break; }
        case ENG_REC_DEPTH_RESOLVE: {
            EngObj* src = eng_owner(r->depth); EngObj* dst = eng_owner(r->resolve_dst);
            ID3D12PipelineState* pso = r->resolve_packed ? s_depth_pack_pso : s_depth_pso;
            if (dst && dst->dsinit && dst->has_rtv) {   /* RSX_DS_HEAP: first use of a placed snapshot's range: aliasing barrier, then a zero clear (what a committed one holds) */
                dst->dsinit = 0;
                if (s_nbar == 32) bar_flush();
                D3D12_RESOURCE_BARRIER* ab = &s_bar[s_nbar++];
                memset(ab, 0, sizeof *ab);
                ab->Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
                ab->Aliasing.pResourceBefore = NULL; ab->Aliasing.pResourceAfter = dst->res;
                res_transition(dst->res, &dst->state, D3D12_RESOURCE_STATE_RENDER_TARGET);
                bar_flush();
                const float z4[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                CALL(s_list, ClearRenderTargetView, obj_rtv(r->resolve_dst), z4, 0, NULL);
            }
            if (!src || !dst || !dst->has_rtv || !pso) break;
            { const unsigned long long ga = (unsigned long long)dst->gw * dst->gh;   /* log v2 zmix */
              const int cl = ga <= 512ull * 512ull ? 0 : ga <= 1024ull * 1024ull ? 1 : 2;
              const int ex = dst->w == src->w && dst->h == src->h, tw = src->ms != NULL, pk = r->resolve_packed != 0;
              InterlockedIncrement(&s_zm[cl]); InterlockedIncrement(&s_zm[pk ? 6 : 5]);
              if (ex) InterlockedIncrement(&s_zm[3]);
              if (tw) InterlockedIncrement(&s_zm[4]);
              if (!pk && ex && !tw) InterlockedIncrement(&s_zm[7]);
              if (!pk && ex && !(src->gw >= 960 && src->gh >= 540)) InterlockedIncrement(&s_zm[8]); }
            /* An MSAA depth twin holds the depth: its first sample. */
            ID3D12PipelineState* pms = r->resolve_packed ? s_depth_pack_ms_pso : s_depth_ms_pso;
            /* The packed copy may be the clip (960x704) of a bigger zeta (1280x720):
             * the shader Loads by pixel, so a smaller destination is its top-left
             * crop. Requiring equal sizes sent ToD's soft smoke (depth as colour)
             * to src->res, which under MSAA never gets the 3D depth: the smoke
             * read ~0 and faded out entirely. */
            if (src->ms && pms && dst->w <= src->w && dst->h <= src->h) {
                { static int n = 0; if (n < 4 && ++n) fprintf(stderr, "[rsx engine/d3d12] depth resolve from MSAA twin: %s %ux%u from %ux%u\n",
                                                             r->resolve_packed ? "packed" : "depth", dst->w, dst->h, src->w, src->h); }
                res_transition(src->ms, &src->ms_state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
                res_transition(dst->res, &dst->state, D3D12_RESOURCE_STATE_RENDER_TARGET);
                bar_flush();
                eng_fullscreen_pass(pms, obj_rtv(r->resolve_dst), dst->w, dst->h, obj_srv_ms((u32)(src - s_obj) + 1));
                break;
            }
            res_transition(src->res, &src->state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            res_transition(dst->res, &dst->state, D3D12_RESOURCE_STATE_RENDER_TARGET);
            bar_flush();
            eng_fullscreen_pass(pso, obj_rtv(r->resolve_dst), dst->w, dst->h, obj_srv(r->depth));
            break; }
        }
    }
    /* Every twin's draws into its target before the list ends: the present,
     * readbacks and the next list read the targets. */
    eng_ms_resolve_all();
    s_mt_base += rec_n; s_mt_scanned = 0;
    /* Close the last pass. */
    if (s_pheap && s_pass_n[s_slot])
        CALL(s_list, EndQuery, s_pheap, D3D12_QUERY_TYPE_TIMESTAMP, s_slot * (ENG_PASS_MAX + 1u) + s_pass_n[s_slot]);
}

/* ---- submit ---------------------------------------------------------------- */

/* Close and execute the open list; signal its fence. Rings and queries of
 * the list are the caller's to account for. */
static void eng_submit_list(u64* out_fence)
{
    bar_flush();
    if (s_q_used && s_qread)
        CALL(s_list, ResolveQueryData, s_qheap, D3D12_QUERY_TYPE_OCCLUSION, s_slot * ENG_QUERIES_PER_SLOT, s_q_used,
             s_qread, (UINT64)s_slot * ENG_QUERIES_PER_SLOT * 8u);
    if (s_pheap && s_pass_n[s_slot])
        CALL(s_list, ResolveQueryData, s_pheap, D3D12_QUERY_TYPE_TIMESTAMP, s_slot * (ENG_PASS_MAX + 1u), s_pass_n[s_slot] + 1u,
             s_pread, (UINT64)s_slot * (ENG_PASS_MAX + 1u) * 8u);
    if (s_tsheap) {
        CALL(s_list, EndQuery, s_tsheap, D3D12_QUERY_TYPE_TIMESTAMP, s_slot * 2u + 1u);
        CALL(s_list, ResolveQueryData, s_tsheap, D3D12_QUERY_TYPE_TIMESTAMP, s_slot * 2u, 2, s_tsread, (UINT64)s_slot * 16u);
    }
    HRESULT hr = CALL0(s_list, Close);
    if (SUCCEEDED(hr) && s_list_bad_copy && s_copy_strict_test) hr = E_INVALIDARG;   /* as a runtime without the relaxation fails it */
    s_list_bad_copy = 0;
    s_list_open = 0;
    if (FAILED(hr)) {
        fprintf(stderr, "[rsx engine/d3d12] command list Close failed: 0x%08lX (removed 0x%08lX)\n",
                (long)hr, (long)CALL0(s_dev, GetDeviceRemovedReason)); eng_dred_dump("present");
        s_ready = 0; *out_fence = 0; return;
    }
    ID3D12CommandList* lists[1] = { (ID3D12CommandList*)s_list };
    if (s_tsheap) QueryPerformanceCounter(&s_ts_slot[s_slot].cpu_exec);
    CALL(s_queue, ExecuteCommandLists, 1, lists);
    *out_fence = fence_signal();
    s_alloc_fence[s_slot] = *out_fence;
    for (u32 i = 0; i < s_smp_spare_n; i++)
        if (s_smp_spare_inlist[i]) { s_smp_spare_inlist[i] = 0; s_smp_spare_fence[i] = *out_fence; }
    if (s_tsheap) { s_ts_slot[s_slot].fence = *out_fence; s_ts_slot[s_slot].kind = s_submit_kind; }
}

/* A completed submit: its query counts and the reports waiting on them. */
static void eng_poll_submits(void)
{
    for (;;) {
        /* Oldest first: a report rides on the submit open when the walker met
         * it, and the queries it counts may be in an earlier one, whose counts
         * have to be in before it reads them. In table order a later submit in
         * a lower slot was delivered first. */
        EngSubmit* s = NULL;
        for (int i = 0; i < ENG_MAX_SUBMITS; i++)
            if (s_sub[i].live && (!s || s_sub[i].fence < s->fence)) s = &s_sub[i];
        if (!s || !fence_done(s->fence)) break;
        if (s->q_count && s_qread) {
            D3D12_RANGE rg = { (SIZE_T)s->q_slot * ENG_QUERIES_PER_SLOT * 8u, ((SIZE_T)s->q_slot * ENG_QUERIES_PER_SLOT + s->q_count) * 8u };
            u64* p = NULL;
            if (SUCCEEDED(CALL(s_qread, Map, 0, &rg, (void**)&p))) {
                for (u32 k = 0; k < s->q_count; k++) s_vis_count[s->q_vis[k] & ENG_Q_INDEX] += eng_q_count(s->q_vis[k], p[s->q_slot * ENG_QUERIES_PER_SLOT + k]);
                D3D12_RANGE wr = {0, 0}; CALL(s_qread, Unmap, 0, &wr);
            }
        }
        for (u32 k = 0; k < s->n_reports; k++) {
            rsx_draw_engine_query_result(s->reports[k].index, s_vis_count[s->reports[k].slot]);
        }
        free(s->q_vis); free(s->reports);
        memset(s, 0, sizeof *s);
    }
}

static void eng_dump_frame(u32 surface);

/* RSX_FT_DUMP=<file>: one CSV line per present, written when the Present call returns: QPC time (us), the Present's return
 * code, DXGI's frame statistics (present count, the refresh it was shown at, the last vblank's count and QPC time) and the
 * compositor's timing (vblank QPC, refresh period). Off unless set; the cost is a call or two per present. */
static void eng_ft_dump(void)
{
    static int on = -1; static FILE* fp; static unsigned n;
    if (on < 0) {
        const char* e = getenv("RSX_FT_DUMP");
        on = (e && *e) ? 1 : 0;
        if (on) { fp = fopen(e, "w"); if (!fp) on = 0; else fprintf(fp, "n,qpc_us,hr,last_present,pc,present_refresh,sync_refresh,sync_qpc_us,dwm_vblank_us,dwm_period_us\n"); }
    }
    if (on <= 0) return;
    LARGE_INTEGER q, f; QueryPerformanceCounter(&q); QueryPerformanceFrequency(&f);
    const double us = 1e6 / (double)f.QuadPart;
    DXGI_FRAME_STATISTICS st; memset(&st, 0, sizeof st);
    const HRESULT hr = CALL(s_swap, GetFrameStatistics, &st);
    UINT lp = 0; CALL(s_swap, GetLastPresentCount, &lp);
    double per = 0.0; long long vb = 0; unsigned long long cr = 0;
    if (!rsx_vblank_snapshot(&per, &vb, &cr)) { per = 0.0; vb = 0; }
    fprintf(fp, "%u,%.1f,%08lX,%u,%u,%u,%u,%.1f,%.1f,%.2f\n", n++, (double)q.QuadPart * us, (unsigned long)hr, lp, (unsigned)st.PresentCount,
            (unsigned)st.PresentRefreshCount, (unsigned)st.SyncRefreshCount, (double)st.SyncQPCTime.QuadPart * us, (double)vb * us, per * 1e6);
    if ((n & 127) == 0) fflush(fp);
}

/* A paced present (rsx_present_pace.h) on a thread of its own. The pacing
 * holds a frame back for a few milliseconds; on the walker that is a few
 * milliseconds before the next frame's commands are looked at, and a title
 * that waits for its flip, or for a query the walker has not reached, loses
 * them from every frame (Drakengard 3 with a cap that follows its frame
 * times: 95 fps became 69). So the walker hands the present over and goes
 * on; it needs it done only where it needs the swap chain itself -- before
 * the next frame's back buffer, and before the swap chain is changed or let
 * go -- and there it asks for the frame at once (eng_present_join).
 * Not in exclusive full screen, where DXGI calls back into the window's
 * thread from Present and that thread must not be waiting for this one. */
static HANDLE s_pres_thread, s_pres_go, s_pres_done;
static volatile LONG s_pres_quit;
static UINT s_pres_interval, s_pres_flags;

static DWORD WINAPI eng_present_thread(LPVOID arg)
{
    { extern void ps3_ccx_role(int); ps3_ccx_role(3); }
    (void)arg;
    SetThreadDescription(GetCurrentThread(), L"rsx walker+present");
    ps3_beacon_thread(BEACON_PRESENTER);
    for (;;) {
        WaitForSingleObject(s_pres_go, INFINITE);   /* idle: beacon depth 0 */
        if (s_pres_quit) break;
        BEACON_WAIT("pres_pace");
        rsx_present_pace();
        BEACON_POP();
        BEACON_WAIT("pres_Present");
        const HRESULT hr = CALL(s_swap, Present, s_pres_interval, s_pres_flags);
        eng_ft_dump();
        BEACON_POP();
        if (FAILED(hr)) {
            static int n = 0;
            if (n < 8 && ++n) fprintf(stderr, "[rsx engine/d3d12] Present failed: 0x%08lX (removed 0x%08lX)\n",
                                 (long)hr, (long)CALL0(s_dev, GetDeviceRemovedReason)); eng_dred_dump("present");
        }
        SetEvent(s_pres_done);
    }
    return 0;
}

/* The present handed over last has been made. If it is still being held for
 * its turn, it is shown now: the walker waits for a present, never for the
 * pacing of one. */
static void eng_present_join(void)
{
    if (!s_pres_thread || WaitForSingleObject(s_pres_done, 0) == WAIT_OBJECT_0) return;
    rsx_present_pace_hurry(1);
    BEACON_WAIT("present_join");
    eng_wait_serv(s_pres_done);
    BEACON_POP();
    rsx_present_pace_hurry(0);
}

/* 0 when there is no thread to give it to: the caller presents itself. */
static int eng_present_async(UINT interval, UINT flags)
{
    if (!s_pres_thread) {
        if (!s_pres_go) s_pres_go = CreateEventA(NULL, FALSE, FALSE, NULL);
        if (!s_pres_done) s_pres_done = CreateEventA(NULL, TRUE, TRUE, NULL);
        if (!s_pres_go || !s_pres_done) return 0;
        s_pres_quit = 0;
        s_pres_thread = CreateThread(NULL, 0, eng_present_thread, NULL, 0, NULL);
        if (!s_pres_thread) return 0;
    }
    s_pres_interval = interval; s_pres_flags = flags;
    ResetEvent(s_pres_done);
    SetEvent(s_pres_go);
    return 1;
}

static void eng_present_thread_stop(void)
{
    if (!s_pres_thread) return;
    eng_present_join();
    s_pres_quit = 1;
    SetEvent(s_pres_go);
    WaitForSingleObject(s_pres_thread, 2000);
    CloseHandle(s_pres_thread);
    s_pres_thread = NULL;
}

static void eng_swapchain_release(void)
{
    if (!s_swap) return;
    eng_present_join();
    if (s_display == DISP_FULLSCREEN) CALL(s_swap, SetFullscreenState, FALSE, NULL);
    for (u32 i = 0; i < 3; i++) RELEASE(s_backbuf[i]);
    RELEASE(s_swap);
}

/* RSX_DISPLAY / RSX_FULLSCREEN / RSX_WINDOW -> s_display and the window
 * size: a windowed window of RSX_WINDOW (else the base size), the others
 * over the display at its current mode. */
static void eng_read_display(void)
{
    const char* dm = getenv("RSX_DISPLAY");
    const char* fs = getenv("RSX_FULLSCREEN");
    s_display = DISP_WINDOWED;
    if (dm && *dm) {
        if (!_stricmp(dm, "borderless")) s_display = DISP_BORDERLESS;
        else if (!_stricmp(dm, "fullscreen") || !_stricmp(dm, "exclusive")) s_display = DISP_FULLSCREEN;
        else if (_stricmp(dm, "windowed") && _stricmp(dm, "window"))
            fprintf(stderr, "[RSX d3d12] RSX_DISPLAY=%s ignored (windowed, borderless or fullscreen)\n", dm);
    } else if (fs && *fs && *fs != '0') s_display = DISP_BORDERLESS;
    const char* ws = getenv("RSX_WINDOW");
    unsigned ww = 0, wh = 0;
    const int have_ws = ws && *ws && sscanf(ws, "%ux%u", &ww, &wh) == 2 && ww >= 320 && wh >= 240 && ww <= 16384 && wh <= 16384;
    if (ws && *ws && !have_ws) fprintf(stderr, "[RSX d3d12] RSX_WINDOW=%s ignored (want <w>x<h>)\n", ws);
    s_win_w = s_width; s_win_h = s_height;
    if (s_display == DISP_WINDOWED) { if (have_ws) { s_win_w = ww; s_win_h = wh; } }
    else { s_win_w = (u32)GetSystemMetrics(SM_CXSCREEN); s_win_h = (u32)GetSystemMetrics(SM_CYSCREEN); }
}

static int eng_read_vsync(void)
{
    const char* v = getenv("RSX_VSYNC");
    return (v && *v) ? (atoi(v) ? 1 : 0) : 1;
}

/* g_rsx_display_reload (rsx_draw_engine.c): the settings page changed
 * RSX_VSYNC / RSX_DISPLAY / RSX_WINDOW / RSX_SCALE. Picked up by the message pump, on
 * the thread that owns the window and presents. V-sync alone is the next
 * present's interval; a new mode or size drains the GPU, restyles the
 * window and makes a new swap chain (a frame or two of black). */
static void eng_rescale(void);   /* below */
static void eng_display_reload(void)
{
    ENG_JOIN("display_reload");
    g_rsx_display_reload = 0;
    const EngDisplayMode od = s_display;
    const u32 ow = s_win_w, oh = s_win_h;
    const int ov = s_vsync;
    const float os = s_scale;
    const int osk = s_shadow_k, obk = s_bloom_k;
    s_vsync = eng_read_vsync(); rsx_vblank_set_vsync(s_vsync);
    eng_read_display();
    eng_read_scale();
    if (s_scale != os || s_shadow_k != osk || s_bloom_k != obk) eng_rescale();
    if ((u32)eng_ms_want() != s_ms_n) eng_ms_apply(eng_ms_want());
    if (s_vsync != ov) fprintf(stderr, "[RSX d3d12] vsync %d\n", s_vsync);
    if (s_display == od && s_win_w == ow && s_win_h == oh) return;
    fence_wait(fence_signal());
    const EngDisplayMode nd = s_display;
    s_display = od;
    eng_swapchain_release();
    s_display = nd;
    if (s_display == DISP_WINDOWED) {
        RECT wr = {0, 0, (LONG)s_win_w, (LONG)s_win_h};
        AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);
        const int w = wr.right - wr.left, h = wr.bottom - wr.top;
        const int x = (GetSystemMetrics(SM_CXSCREEN) - w) / 2, y = (GetSystemMetrics(SM_CYSCREEN) - h) / 2;
        SetWindowLongPtrA(s_hwnd, GWL_STYLE, WS_OVERLAPPEDWINDOW | WS_VISIBLE);
        SetWindowPos(s_hwnd, HWND_NOTOPMOST, x > 0 ? x : 0, y > 0 ? y : 0, w, h, SWP_FRAMECHANGED | SWP_SHOWWINDOW);
    } else {
        SetWindowLongPtrA(s_hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);
        SetWindowPos(s_hwnd, HWND_TOP, 0, 0, (int)s_win_w, (int)s_win_h, SWP_FRAMECHANGED | SWP_SHOWWINDOW);
    }
    IDXGIFactory4* factory = NULL;
    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory4, (void**)&factory)) || eng_swapchain_create(factory) != 0)
        fprintf(stderr, "[RSX d3d12] could not make a swap chain for the new display mode\n");
    RELEASE(factory);
    fprintf(stderr, "[RSX d3d12] display: %s %ux%u, vsync %d\n",
            s_display == DISP_FULLSCREEN ? "full screen" : s_display == DISP_BORDERLESS ? "borderless" : "windowed",
            s_win_w, s_win_h, s_vsync);
}

/* RSX_SCALE changed while running (the reload below). With the GPU idle:
 * every colour and depth target at the internal resolution is made again at
 * the new one under the same handle (the views cut from it pointed at the
 * new resource), the pipelines whose fragment program reads WPOS get that
 * program rebuilt for the new divisor, and the pools of host-sized
 * scratch targets are emptied. The targets come back empty: the title draws
 * nearly all of them every frame, so this is a frame or two of black. */
static void eng_rescale(void)
{
    s_zown_gen++;
    fence_wait(fence_signal());
    eng_lock(&s_pipe_lock, "pipe_lock");
    u32 np = 0;
    for (u32 i = 0; i < s_pipe_count; i++) {
        EngPipeline* p = &s_pipe[i];
        if (!p->ps_plain || !p->vs) continue;
        ID3DBlob* ps = eng_fp_blob(p->ps_plain);
        if (!ps) continue;
        CALL0(ps, AddRef);
        RELEASE(p->ps);
        p->ps = ps;
        p->desc.PS.pShaderBytecode = CALL0(ps, GetBufferPointer);
        p->desc.PS.BytecodeLength = CALL0(ps, GetBufferSize);
        for (int c = 0; c < 3; c++) { RELEASE(p->pso[c]); p->failed[c] = 0; }
        /* The MSAA builds copy p->desc at creation: with the old PS (old WPOS divisor) they would keep drawing
         * into the MSAA twins (striped fur/dither after a scale change from F1). */
        for (int c = 0; c < 3; c++) { RELEASE(p->psoms[c]); p->failedms[c] = 0; }
        np++;
    }
    ReleaseSRWLockExclusive(&s_pipe_lock);
    for (u32 i = 0; i < s_fallback_n; i++) eng_obj_release(NULL, s_fallback_depth[i]);
    s_fallback_n = 0;
    eng_dpool_flush_all();   /* depthpool: the parked sizes are stale after a rescale (GPU idle) */
    for (u32 i = 0; i < s_snap_pool_n; i++) eng_snap_pool_free_entry(i);   /* GPU idle (fence_wait above); a placed entry's range goes back */
    s_snap_pool_n = 0;
    u32 nt = 0;
    for (u32 i = 0; i < s_obj_count; i++) {
        EngObj* o = &s_obj[i];
        const u32 h = i + 1;
        if (o->retired || !o->res || !o->gw || !o->gh) continue;
        if (o->kind != OBJ_COLOR && o->kind != OBJ_DEPTH) continue;
        const int was = o->w != o->gw || o->h != o->gh;
        if (o->xk > 0.0f && o->kind == OBJ_COLOR) {
            /* RSX_SHADOW_SCALE / RSX_BLOOM_SCALE / RSX_SCALE changed: the extra multiplier again, with the same
             * clamp as eng_shadow_k (canvas size, 2048 host pixels). A target made without xk stays without. */
            int k = o->xbloom ? s_bloom_k : s_shadow_k;
            if (!o->xbloom)
                while (k > 1 && ((u64)o->gw * k > s_width || (u64)o->gh * k > s_height ||
                                 (float)o->gw * s_scale * (float)k > 2048.0f || (float)o->gh * s_scale * (float)k > 2048.0f)) k >>= 1;
            o->xk = k > 1 ? (float)k : 0.0f;
        }
        const int now = o->xk > 0.0f || eng_scales(o->gw, o->gh);
        if (!was && !now) continue;
        const float nf = o->xk > 0.0f ? eng_xk_f(o) : s_scale;
        if (o->xx > 0.0f && o->kind == OBJ_COLOR) o->xx = eng_hstretch_on() ? ENG_HSTRETCH_K : 0.0f;
        const u32 nw = now ? scf_dim(o->gw, o->xx > 0.0f && o->kind == OBJ_COLOR ? nf * o->xx : nf) : o->gw, nh = now ? scf_dim(o->gh, nf) : o->gh;
        if (nw == o->w && nh == o->h) continue;
        RELEASE(o->res);
        if (o->heap) { theap_done(o->heap, o->heap_off, o->heap_size); o->heap = 0; o->heap_off = o->heap_size = 0; }   /* RSX_RT_HEAP: GPU idle (fence_wait above); the new one is committed, zero */
        eng_ms_drop(o);   /* made again at its next MSAA pass */
        o->ms_dirty = o->ms_stale = 0;
        D3D12_CLEAR_VALUE cv = {0};
        if (o->kind == OBJ_COLOR) {
            cv.Format = o->fmt;
            o->res = make_texture(o->fmt, nw, nh, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                  D3D12_RESOURCE_STATE_RENDER_TARGET, &cv);
            o->state = D3D12_RESOURCE_STATE_RENDER_TARGET;
            if (!o->res) continue;
            eng_write_srv(h, o->res, o->fmt, 1, 1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
            CALL(s_dev, CreateRenderTargetView, o->res, NULL, obj_rtv(h));
        } else {
            cv.Format = ENG_DEPTH_FMT; cv.DepthStencil.Depth = 1.0f;
            o->res = make_texture(ENG_DEPTH_RES_FMT, nw, nh, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                                  D3D12_RESOURCE_STATE_DEPTH_WRITE, &cv);
            o->state = D3D12_RESOURCE_STATE_DEPTH_WRITE;
            if (!o->res) continue;
            D3D12_DEPTH_STENCIL_VIEW_DESC dd = {0};
            dd.Format = ENG_DEPTH_FMT; dd.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
            CALL(s_dev, CreateDepthStencilView, o->res, &dd, obj_dsv(h));
            eng_write_srv(h, o->res, ENG_DEPTH_SRV_FMT, 1, 1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
        }
        o->w = nw; o->h = nh;
        for (u32 v = 0; v < s_view_count; v++)
            if (s_view[v].surface == h) {
                eng_write_srv(s_view[v].view, o->res, o->fmt, 1, 1, eng_mapping(s_view[v].remap, s_view[v].format));
                EngObj* vo = eng_obj(s_view[v].view);
                if (vo) { vo->w = nw; vo->h = nh; }
            }
        nt++;
    }
    fprintf(stderr, "[RSX d3d12] internal resolution x%.2f (%ux%u): %u targets and %u programs rebuilt\n",
            s_scale, sc_dim(s_width), sc_dim(s_height), nt, np);
}

/* Back in exclusive full screen after a focus loss. The GPU is drained first
 * (the back buffers are released and recreated), so this costs a frame. */
static void eng_backbuf_rebuild(const char* why)
{
    for (u32 i = 0; i < 3; i++) RELEASE(s_backbuf[i]);
    /* explicit size: with 0x0 DXGI took the window client area, which after leaving full screen is the window's first
     * windowed rect (640x480, 07/11 b54): a tiny, cut picture on return */
    HRESULT hr = CALL(s_swap, ResizeBuffers, 0, s_win_w, s_win_h, DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH);
    if (FAILED(hr)) { fprintf(stderr, "[rsx engine/d3d12] ResizeBuffers (%s) failed: 0x%08lX\n", why, (long)hr); return; }
    for (u32 i = 0; i < 3; i++) {
        if (FAILED(CALL(s_swap, GetBuffer, i, &IID_ID3D12Resource, (void**)&s_backbuf[i]))) return;
        CALL(s_dev, CreateRenderTargetView, s_backbuf[i], NULL, cpu_handle(s_backbuf_rtv_heap, s_rtv_step, i));
    }
    fprintf(stderr, "[RSX d3d12] back buffers rebuilt (%s)\n", why);
}
/* After the window thread left exclusive full screen on a focus loss (07/11: every Present then failed with INVALID_CALL). */
static void eng_fullscreen_left(void)
{
    InterlockedExchange(&s_fs_left, 0);
    if (!s_swap || s_headless) return;
    eng_present_join();
    fence_wait(fence_signal());
    eng_backbuf_rebuild("left full screen");
}
static void eng_fullscreen_restore(void)
{
    if (!s_swap || s_headless) { s_fs_restore = 0; return; }
    if (IsIconic(s_hwnd)) return;   /* the focus comes back before the window is restored: re-entering then gave a black screen (07/11) */
    s_fs_restore = 0;
    eng_present_join();
    BOOL fs = FALSE;
    if (FAILED(CALL(s_swap, GetFullscreenState, &fs, NULL)) || fs) return;
    fence_wait(fence_signal());
    /* the window back to the whole screen first (asynchronous: this is not the window thread, which may be waiting for us) */
    SetWindowPos(s_hwnd, HWND_TOP, 0, 0, (int)s_win_w, (int)s_win_h, SWP_NOACTIVATE | SWP_ASYNCWINDOWPOS);
    if (FAILED(CALL(s_swap, SetFullscreenState, TRUE, NULL))) {
        fprintf(stderr, "[rsx engine/d3d12] could not re-enter full screen; staying windowed\n");
        eng_backbuf_rebuild("stayed windowed");
        return;
    }
    fprintf(stderr, "[RSX d3d12] focus back: re-entered exclusive full screen\n");
    eng_backbuf_rebuild("full screen");
}

/* The frame after FXAA, as large as the frame: an object with no guest
 * size, so no pass treats it as scaled and eng_rescale leaves it alone. */
static u32 s_aa_obj;
static u32 eng_aa_target(u32 w, u32 h)
{
    EngObj* o = eng_obj(s_aa_obj);
    if (o && o->w == w && o->h == h) return s_aa_obj;
    if (o) eng_obj_release(NULL, s_aa_obj);
    s_aa_obj = 0;
    D3D12_CLEAR_VALUE cv = {0}; cv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    ID3D12Resource* t = make_texture(DXGI_FORMAT_R8G8B8A8_UNORM, w, h, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                     D3D12_RESOURCE_STATE_RENDER_TARGET, &cv);
    const u32 handle = eng_obj_add(t, OBJ_COLOR, DXGI_FORMAT_R8G8B8A8_UNORM, w, h, 1, 1, D3D12_RESOURCE_STATE_RENDER_TARGET);
    if (!handle) return 0;
    s_obj[handle - 1].gw = s_obj[handle - 1].gh = 0;
    eng_write_srv(handle, t, DXGI_FORMAT_R8G8B8A8_UNORM, 1, 1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
    CALL(s_dev, CreateRenderTargetView, t, NULL, obj_rtv(handle));
    s_obj[handle - 1].has_rtv = 1;
    s_aa_obj = handle;
    return handle;
}

/* ---- RSX_ENCODE_THREAD=1: the frame's command list is recorded on a thread of its own -------------------
 *
 * The walker produces the records of frame N (s_rec) while the encode thread records frame N-1 from the
 * other array, submits it and presents. Why that is race-free:
 *  - Ownership. A job owns, from the hand-over until the walker has seen s_enc_done, everything the encode
 *    reads or writes: its record array, a frozen copy of its stage chain, the command list/allocator slot,
 *    the descriptor rings, EngObj state (resource states, ms flags), s_sub[sub_idx], s_fence_value, the
 *    present thread. The walker, meanwhile, may only call the entries that touch its own side: LB->pending,
 *    the other record array, the stage chunks it takes for the NEXT chain, s_samp (append only), s_vs_cb_*,
 *    s_vis_cur, the pipeline table (under its lock). Every other entry of the backend starts with ENG_JOIN
 *    (waits for s_enc_done), after which the state is exactly what the one-thread engine would see.
 *  - The stage. Chunks of the chain handed over are held (in_use 1, fence ~0, never done) from the
 *    hand-over; the encode thread publishes the real fence (eng_stage_set_fence) after its last
 *    ExecuteCommandLists, so a chunk is not reused while any list of the frame can still read it. The
 *    walker reads those fences with interlocked reads.
 *  - Order. s_submit_seq (stage age, slot choice, reuse_vs_constants window) moves on at the hand-over;
 *    the job carries its own. Fence values are signalled by the encode thread in job order, and a job is
 *    handed over only after the previous one was joined.
 *  - Callbacks into the draw engine (query results) and retirement run on the walker thread only, at the
 *    hand-over and in the synchronous path, never on the encode thread.
 * Where the engine has to be synchronous (mid-frame flush, readback, wait=1, headless, RT AO) the work is
 * done as before on the walker thread, after a join. Off (default), none of this code runs. */
typedef struct {
    u32 present_surface, seq, n_rec, chain_n, sub_idx, n_reports;
    EngRecord* rec;
    int chain[ENG_STAGE_POOL];
    EngVisReport* reports;       /* the frame's pending reports (malloc), moved into the submit record */
} EngJob;
static EngJob s_job;
static HANDLE s_enc_thread, s_enc_go, s_enc_done;
static volatile LONG s_enc_quit;
static int s_enc_mode = -1;      /* RSX_ENCODE_THREAD */
static struct { const char* why; unsigned n; double ms; } s_enc_jw[32]; static unsigned s_enc_njw;
static unsigned s_enc_n_async, s_enc_n_sync, s_enc_n_idle; static double s_enc_wait_prev_ms;

static void eng_exec_frame(u32 present_surface, int wait, u32 seq, const EngRecord* rec, u32 nrec, EngSubmit* sub,
                           const EngVisReport* reps, u32 nreps, const int* chain, const u32* chain_n, int async);

static void eng_enc_stats(void)
{
    static LARGE_INTEGER last; LARGE_INTEGER now, f; QueryPerformanceCounter(&now); QueryPerformanceFrequency(&f);
    if (!last.QuadPart) { last = now; return; }
    if ((s_enc_n_async + s_enc_n_sync) < 240) return;
    fprintf(stderr, "[enc-thread] %u async + %u sync presents in %.1f s; the walker waited %.2f ms/present for the previous frame's encode (%u hand-overs found it done); joins forced by the walker's own calls:",
            s_enc_n_async, s_enc_n_sync, (double)(now.QuadPart - last.QuadPart) / (double)f.QuadPart,
            s_enc_n_async ? s_enc_wait_prev_ms / s_enc_n_async : 0.0, s_enc_n_idle);
    for (unsigned i = 0; i < s_enc_njw; i++) fprintf(stderr, " %s x%u (%.2f ms)", s_enc_jw[i].why, s_enc_jw[i].n, s_enc_jw[i].ms);
    fprintf(stderr, "\n");
    if (ep_on() && s_ep_n) {
        fprintf(stderr, "[encprof] per encoded frame (%u): job wall %.2f ms, thread CPU %.2f ms, Present+pacing %.2f ms, fence waits %.2f ms (GPU/in-flight, not CPU)\n",
                s_ep_n, s_ep_wall / s_ep_n, s_ep_cpu / s_ep_n, s_ep_present / s_ep_n, s_ep_fence / s_ep_n);
        s_ep_wall = s_ep_cpu = s_ep_present = s_ep_fence = 0; s_ep_n = 0;
    }
    if (rel_defer_on()) { fprintf(stderr, "[rel-defer] RSX_REL_DEFER=1: %llu releases deferred, %llu flushed, %u pending now\n", s_rel_n_defer, s_rel_n_flush, s_rel_pn); }
    s_enc_n_async = s_enc_n_sync = s_enc_n_idle = 0; s_enc_wait_prev_ms = 0; s_enc_njw = 0; last = now;
}

/* Wait for the job out (walker thread). `why` is a string literal, counted for the log. */
static void eng_enc_join_why(const char* why, int is_handover)
{
    if (!s_enc_pending) return;
    LARGE_INTEGER a, b, f; QueryPerformanceCounter(&a);
    const int idle = WaitForSingleObject(s_enc_done, 0) == WAIT_OBJECT_0;
    RSX_B2P_ADD(B2C_ENCJOIN, 1);
    if (!idle) { RSX_B2P_ENTER(B2P_WENC); RSX_B2P_ADD(B2C_ENCBLK, 1); BEACON_WAIT("enc_join"); eng_wait_serv(s_enc_done); BEACON_POP(); RSX_B2P_LEAVE(); }
    QueryPerformanceCounter(&b); QueryPerformanceFrequency(&f);
    const double ms = (double)(b.QuadPart - a.QuadPart) * 1000.0 / (double)f.QuadPart;
    g_rsx_frame_join_ms += ms;   /* surf54: for the [hitch] line */
    s_enc_pending = 0;
    if (s_rel_pn && !t_rel_hold && !LB->spec) eng_rel_flush();   /* lane111 RSX_REL_DEFER */
    if (is_handover) { s_enc_wait_prev_ms += ms; if (idle) s_enc_n_idle++; }
    else {
        unsigned i = 0;
        while (i < s_enc_njw && s_enc_jw[i].why != why) i++;
        if (i == s_enc_njw && i < 32) { s_enc_jw[i].why = why; s_enc_jw[i].n = 0; s_enc_jw[i].ms = 0; s_enc_njw++; }
        if (i < 32) { s_enc_jw[i].n++; s_enc_jw[i].ms += ms; }
    }
}

static DWORD WINAPI eng_encode_thread(LPVOID arg)
{
    { extern void ps3_ccx_role(int); ps3_ccx_role(3); }
    (void)arg;
    SetThreadDescription(GetCurrentThread(), L"rsx encode");
    t_on_enc = 1;
    ps3_beacon_thread(BEACON_PRESENTER);
    for (;;) {
        WaitForSingleObject(s_enc_go, INFINITE);   /* idle: beacon depth 0 */
        if (s_enc_quit) break;
        EngJob* j = &s_job;
        EngSubmit* sub = &s_sub[j->sub_idx];
        BEACON_PUSH("eng_encode_job");
        const int epo = ep_on(); const double ew0 = epo ? ep_now() : 0.0, ec0 = epo ? ep_cpu() : 0.0;
        eng_exec_frame(j->present_surface, 0, j->seq, j->rec, j->n_rec, sub, j->reports, j->n_reports, j->chain, &j->chain_n, 1);
        BEACON_POP();
        if (epo) { s_ep_wall += ep_now() - ew0; s_ep_cpu += ep_cpu() - ec0; s_ep_n++; }
        free(j->reports); j->reports = NULL;
        SetEvent(s_enc_done);
        if (reap_on()) eng_reap_drain(reap_budget_ms());   /* rissa128: the real Release of what the walker retired, while it walks the next frame */
    }
    return 0;
}

static int eng_enc_start(void)
{
    if (s_enc_thread) return 1;
    if (!s_rec_store1) s_rec_store1 = (EngRecord*)calloc(ENG_MAX_RECORDS, sizeof(EngRecord));
    if (!s_rec_store1) return 0;
    if (!s_enc_go) s_enc_go = CreateEventA(NULL, FALSE, FALSE, NULL);
    if (!s_enc_done) s_enc_done = CreateEventA(NULL, FALSE, FALSE, NULL);
    if (!s_enc_go || !s_enc_done) return 0;
    s_enc_quit = 0;
    s_enc_thread = CreateThread(NULL, 0, eng_encode_thread, NULL, 0, NULL);
    return s_enc_thread != NULL;
}

static void eng_enc_stop(void)
{
    if (!s_enc_thread) return;
    eng_enc_join_why("stop", 1);
    s_enc_quit = 1;
    SetEvent(s_enc_go);
    WaitForSingleObject(s_enc_thread, 5000);
    CloseHandle(s_enc_thread);
    s_enc_thread = NULL;
}

static int eng_enc_wanted(void)
{
    /* Default ON since 1.0 (02/11, quiet-window A/B: Metropolis 24.5 -> 26.4 fps, heavy frames 35.7 -> 33.5 ms,
     * Imperial Fight Fest 49 -> 54, Pirate Base 55 -> 56; 5.5 min stability run clean); RSX_ENCODE_THREAD=0 turns it off. */
    if (s_enc_mode < 0) { const char* e = getenv("RSX_ENCODE_THREAD"); s_enc_mode = (e && e[0] == '0') ? 0 : 1; }
    return s_enc_mode;
}

/* A present with the encode on its own thread. Walker thread. 0: not possible, the caller goes the
 * synchronous way (nothing has been touched). */
static int eng_submit_async(u32 present_surface)
{
    if (!eng_enc_start()) return 0;
    eng_enc_join_why("handover", 1);                 /* the previous frame is encoded and submitted */
    eng_poll_submits();
    /* A free submit record for the job; waiting for the oldest if all are out (what the one-thread path does
     * after encoding). The poll that frees it calls into the draw engine, so it is made here. */
    int si = -1;
    for (;;) {
        for (int i = 0; i < ENG_MAX_SUBMITS && si < 0; i++) if (!s_sub[i].live) si = i;
        if (si >= 0) break;
        int oldest = 0;
        for (int i = 1; i < ENG_MAX_SUBMITS; i++) if (s_sub[i].fence < s_sub[oldest].fence) oldest = i;
        fence_wait(s_sub[oldest].fence);
        eng_poll_submits();
    }
    EngJob* j = &s_job;
    j->present_surface = present_surface;
    j->seq = s_submit_seq;
    j->rec = s_rec; j->n_rec = s_rec_count;
    j->chain_n = s_chain_n; memcpy(j->chain, s_chain, s_chain_n * sizeof s_chain[0]);
    j->sub_idx = (u32)si;
    j->reports = NULL; j->n_reports = 0;
    if (s_vis_npending) {
        j->reports = (EngVisReport*)malloc(s_vis_npending * sizeof(EngVisReport));
        if (j->reports) { memcpy(j->reports, s_vis_pending, s_vis_npending * sizeof(EngVisReport)); j->n_reports = s_vis_npending; }
        s_vis_npending = 0;
    }
    memset(&s_sub[si], 0, sizeof s_sub[si]);          /* not live: invisible to polls until the job fills it */
    eng_stage_submitted_x(0, 1);                      /* the chain is the job's from here; empties s_chain_n */
    s_rec = (s_rec == s_rec_store0) ? s_rec_store1 : s_rec_store0;
    eng_rec_reset(); s_rt_mark = ~0u;
    s_submit_seq++;
    if (s_dropped) {
        fprintf(stderr, "[rsx engine/d3d12] dropped %u record(s) (cap %d)\n", s_dropped, ENG_MAX_RECORDS);
        s_dropped = 0;
    }
    /* What earlier frames retired (before the job runs: its mid-frame flush edits those fences). */
    eng_collect_retired();
    eng_collect_retired_buffers();
    s_vis_job_lo = s_vis_frame_lo; s_vis_frame_lo = s_vis_next;   /* the job's begin numbers: [lo, s_vis_next) */
    s_enc_pending = 1;
    s_enc_n_async++;
    SetEvent(s_enc_go);
    eng_enc_stats();
    return 1;
}

/* s_walker_tid (above eng_pso): the thread that encodes -- PSO time there is walker time. */
/* RSX_WPROF=1: where a present's time goes (ms, summed; printed as [wprof4] by rsx_draw_engine.c). 0 join+poll before the frame,
 * 1 wait for the allocator slot, 2 encode, 3 post (blit/overlay), 4 submit, 5 present hand-over/pace/Present, 6 in-flight fence wait,
 * 7 retire after the frame, 8 whole eng_submit. */
double g_pw_ms[12]; int g_pw_on = -1;
static double pw_t(void)
{
    static double inv; LARGE_INTEGER q;
    if (!inv) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); inv = 1000.0 / (double)f.QuadPart; }
    QueryPerformanceCounter(&q); return (double)q.QuadPart * inv;
}
static int pw_on(void)
{
    if (g_pw_on < 0) { const char* e = getenv("RSX_WPROF"); g_pw_on = (e && e[0] && e[0] != '0') ? 1 : 0; }
    return g_pw_on;
}
#define PW_LAP(i, t) do { if (pw_on()) { const double n_ = pw_t(); g_pw_ms[i] += n_ - (t); (t) = n_; } } while (0)
static u32 s_last_present_surface;     /* svpick: the surface of the last present, for rsx_d3d12_engine_repaint */
static ULONGLONG s_last_present_tick;
static void eng_submit_impl(u32 present_surface, int wait)
{
    if (present_surface) { s_last_present_surface = present_surface; s_last_present_tick = GetTickCount64(); }
    if (!s_walker_tid) s_walker_tid = GetCurrentThreadId();
    if (!s_ready) { ENG_JOIN("not_ready"); eng_rec_reset(); s_rt_mark = ~0u; eng_stage_submitted(0); s_vis_npending = 0; return; }
    if (s_fs_left && present_surface) { ENG_JOIN("fs_left"); eng_fullscreen_left(); }
    if (s_fs_restore && present_surface) { ENG_JOIN("fs_restore"); eng_fullscreen_restore(); }
    eng_lanes_flatten();   /* RSX_LANES: the block lists become s_rec */
    if (!s_rec_count && !present_surface) return;
    double pt = pw_on() ? pw_t() : 0.0;
    /* RSX_ENCODE_THREAD: a present that nobody waits for, in a window, without RT AO. */
    if (present_surface && !wait && !s_headless && s_swap && s_blit_pso && eng_enc_wanted() && !g_rsx_rt_on && !s_frame_has_rt
        && eng_owner(present_surface) && eng_submit_async(present_surface)) {
        PW_LAP(0, pt);
        return;
    }
    ENG_JOIN(wait ? "submit_wait" : "submit_sync");
    if (eng_enc_wanted()) s_enc_n_sync++;
    eng_poll_submits();
    PW_LAP(0, pt);

    eng_exec_frame(present_surface, wait, s_submit_seq, s_rec, s_rec_count, NULL, s_vis_pending, s_vis_npending,
                   s_chain, &s_chain_n, 0);
    pt = pw_on() ? pw_t() : 0.0;
    s_vis_npending = 0;
    s_frame_has_rt = 0;
    eng_rec_reset(); s_rt_mark = ~0u;
    s_submit_seq++;
    if (s_dropped) {
        fprintf(stderr, "[rsx engine/d3d12] dropped %u record(s) (cap %d)\n", s_dropped, ENG_MAX_RECORDS);
        s_dropped = 0;
    }
    eng_poll_submits();
    eng_collect_retired();
    eng_collect_retired_buffers();
    if (reap_on()) eng_reap_drain(reap_budget_ms());   /* rissa128: no encode thread on this path, the walker drains a little per frame */
    PW_LAP(7, pt);
}

/* Encode `rec`, make the present, close, execute and signal. The synchronous path calls it on the walker
 * thread with sub NULL (it picks the record, as it always did) and the live s_chain; the encode thread
 * calls it with the job's record, frozen chain and a record reserved for it (async). */
static void eng_exec_frame(u32 present_surface, int wait, u32 seq, const EngRecord* rec, u32 nrec, EngSubmit* sub,
                           const EngVisReport* reps, u32 nreps, const int* chain, const u32* chain_n, int async)
{
    double pt = (!async && pw_on()) ? pw_t() : 0.0; const int pwk = (!async && pw_on());
    s_slot = seq % ENG_FRAMES;
    s_enc_seq = seq;
    fence_wait(s_alloc_fence[s_slot]);
    eng_list_begin();
    s_view_chain = chain; s_view_chain_n = chain_n;
    if (pwk) PW_LAP(1, pt);

    { extern unsigned long long g_wp_enc_cyc; static int wp = -1;   /* RSX_WPROF (rsx_draw_engine.c) */
      if (wp < 0) { const char* e = getenv("RSX_WPROF"); wp = (e && e[0] && e[0] != '0') ? 1 : 0; }
      s_mt_final = present_surface != 0;
      if (wp) { const unsigned long long a = __rdtsc(); eng_encode_records(rec, nrec, seq); g_wp_enc_cyc += __rdtsc() - a; }
      else eng_encode_records(rec, nrec, seq); }
    if (present_surface) eng_mt_frame_end();
    if (pwk) PW_LAP(2, pt);

    u32 bbi = 0; int windowed = 0;
    if (present_surface) {
        EngObj* src = eng_owner(present_surface);
        if (src && s_blit_pso) {
            ID3D12Resource* dst; D3D12_CPU_DESCRIPTOR_HANDLE rtv;
            D3D12_RESOURCE_STATES before;
            if (s_headless) { dst = s_offscreen; rtv = cpu_handle(s_backbuf_rtv_heap, s_rtv_step, 0); before = s_offscreen_state; }
            else {
                eng_present_join();   /* the last frame's present, if it is paced on its thread */
                bbi = CALL0(s_swap, GetCurrentBackBufferIndex);
                dst = s_backbuf[bbi]; rtv = cpu_handle(s_backbuf_rtv_heap, s_rtv_step, bbi);
                before = D3D12_RESOURCE_STATE_PRESENT; windowed = 1;
            }
            /* RSX_AA=fxaa: the frame through FXAA first, at its own size. */
            u32 shown = present_surface;
            if (rsx_aa_mode() == 1 && s_fxaa_pso && !s_headless) {
                const u32 aa = eng_aa_target(src->w, src->h);
                EngObj* ao = eng_obj(aa);
                if (ao && ao->res) {
                    res_transition(ao->res, &ao->state, D3D12_RESOURCE_STATE_RENDER_TARGET);
                    res_transition(src->res, &src->state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
                    bar_flush();
                    eng_fullscreen_pass(s_fxaa_pso, obj_rtv(aa), src->w, src->h, obj_srv(present_surface));
                    res_transition(ao->res, &ao->state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
                    src = ao; shown = aa;
                }
            }
            D3D12_RESOURCE_STATES st = before;
            res_transition(dst, &st, D3D12_RESOURCE_STATE_RENDER_TARGET);
            res_transition(src->res, &src->state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            bar_flush();
            /* The area-filtered blit for the window, placed by
             * rsx_present_layout; the headless target keeps the plain one
             * over the whole of it, which the replay references were made with. */
            float lvp[4]; int lrect[4];
            if (!s_headless && rsx_present_layout(src->w, src->h, s_win_w, s_win_h, lvp, lrect)) {
                const float black[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
                CALL(s_list, ClearRenderTargetView, rtv, black, 0, NULL);
            }
            eng_fullscreen_pass_at(s_present_pso && !s_headless ? s_present_pso : s_blit_pso, rtv, s_win_w, s_win_h,
                                   obj_srv(shown), s_headless ? NULL : lvp, s_headless ? NULL : lrect);
            if (windowed && s_overlay.render)   /* the port's overlay, over the finished frame */
                s_overlay.render(s_dev, s_queue, s_list, (uint64_t)rtv.ptr, s_win_w, s_win_h, ENG_FRAMES);
            if (windowed) res_transition(dst, &st, D3D12_RESOURCE_STATE_PRESENT);
            else s_offscreen_state = st;
            bar_flush();
        }
    }

    if (pwk) PW_LAP(3, pt);
    /* The submit record: what completion has to deliver. */
    if (!sub) {
        for (int i = 0; i < ENG_MAX_SUBMITS && !sub; i++) if (!s_sub[i].live) sub = &s_sub[i];
        if (!sub) {   /* every slot in flight: wait for the oldest */
            int oldest = 0;
            for (int i = 1; i < ENG_MAX_SUBMITS; i++) if (s_sub[i].fence < s_sub[oldest].fence) oldest = i;
            fence_wait(s_sub[oldest].fence);
            eng_poll_submits();
            sub = &s_sub[oldest];
        }
    }
    memset(sub, 0, sizeof *sub);
    sub->q_slot = s_slot; sub->q_count = s_q_used;
    if (s_q_used) { sub->q_vis = (u32*)malloc(s_q_used * sizeof(u32)); if (sub->q_vis) memcpy(sub->q_vis, s_q_vis, s_q_used * sizeof(u32)); else sub->q_count = 0; }
    if (nreps) {
        sub->reports = (EngVisReport*)malloc(nreps * sizeof(EngVisReport));
        if (sub->reports) { memcpy(sub->reports, reps, nreps * sizeof(EngVisReport)); sub->n_reports = nreps; }
    }
    u64 f = 0;
    s_submit_kind = wait;
    eng_submit_list(&f);
    s_submit_kind = 0;
    sub->fence = f;
    MemoryBarrier();
    sub->live = 1;
    if (async) eng_stage_set_fence(chain, *chain_n, f);
    else eng_stage_submitted(f);
    eng_pc_idle_save();
    if (pwk) PW_LAP(4, pt);

    if (windowed) {
        /* RSX_HITCH_LOG=<ms>: a Present or a pacing wait that took more than
         * a third of that, with its time -- whether a long frame was spent
         * here, in the display path, or before the walker had the frame. */
        static double hl = -1.0;
        if (hl < 0.0) { const char* e = getenv("RSX_HITCH_LOG"); hl = e ? atof(e) : 0.0; }
        LARGE_INTEGER pq0, pq1, pq2, pqf;
        const UINT pflags = (!s_vsync && s_tearing && s_display != DISP_FULLSCREEN) ? DXGI_PRESENT_ALLOW_TEARING : 0;
        if (hl > 0.0) { QueryPerformanceFrequency(&pqf); QueryPerformanceCounter(&pq0); }
        /* The frame's work is with the GPU; it is shown when its turn comes
         * (rsx_present_pace.h). Paced, the wait and the present are the
         * presenting thread's (above); in exclusive full screen, and when
         * that thread cannot be made, they are done here. */
        const double ep0 = (t_on_enc && ep_on()) ? ep_now() : 0.0;
        if (!(rsx_present_pace_on() && s_display != DISP_FULLSCREEN && eng_present_async((UINT)s_vsync, pflags))) {
            eng_present_join();
            BEACON_WAIT("pace_sync");
            rsx_present_pace();
            BEACON_POP();
            BEACON_WAIT("Present_sync");
            HRESULT hr = CALL(s_swap, Present, (UINT)s_vsync, pflags);
            eng_ft_dump();
            BEACON_POP();
            { /* menu110: how long a run of failed presents lasts (the first ones after a switch to exclusive full screen) */
              static int run; static ULONGLONG t0;
              if (FAILED(hr)) { if (!run++) t0 = GetTickCount64(); }
              else if (run) { fprintf(stderr, "[RSX d3d12] Present recovered after %d failed presents (%llu ms)\n", run, (unsigned long long)(GetTickCount64() - t0)); run = 0; } }
            if (FAILED(hr)) {
                static int n = 0;
                if (n < 8 && ++n) fprintf(stderr, "[rsx engine/d3d12] Present failed: 0x%08lX (removed 0x%08lX)\n",
                                     (long)hr, (long)CALL0(s_dev, GetDeviceRemovedReason)); eng_dred_dump("present");
            }
        }
        if (ep0 > 0.0) s_ep_present += ep_now() - ep0;
        if (hl > 0.0) QueryPerformanceCounter(&pq1);
        if (pwk) PW_LAP(5, pt);
        /* Pace: at most ENG_INFLIGHT presents queued. */
        if (f > ENG_INFLIGHT) fence_wait(f - ENG_INFLIGHT);
        if (pwk) PW_LAP(6, pt);
        if (hl > 0.0) {
            QueryPerformanceCounter(&pq2);
            const double pres = (double)(pq1.QuadPart - pq0.QuadPart) * 1000.0 / (double)pqf.QuadPart;
            const double pace = (double)(pq2.QuadPart - pq1.QuadPart) * 1000.0 / (double)pqf.QuadPart;
            if (pres > hl / 3.0 || pace > hl / 3.0)
                fprintf(stderr, "[hitch-present] submit %llu: Present %.1f ms, pacing wait %.1f ms\n",
                        (unsigned long long)f, pres, pace);
        }
    }
    if (wait || (present_surface && s_headless)) fence_wait(f);
    s_view_chain = s_chain; s_view_chain_n = &s_chain_n;
}

static void eng_submit(u32 present_surface, int wait)
{
    BEACON_PUSH("eng_submit");
    eng_submit_impl(present_surface, wait);
    BEACON_POP();
}

/* svpick: present the last frame again (with the overlay on top). The save-slot picker blocks the guest thread, so no
 * new flip comes while it is open; its wait loop calls this ~30x/s. Does nothing while the game is still presenting
 * (last present < 120 ms ago) or when there is no window. */
void rsx_d3d12_engine_repaint(void)
{
    if (!s_active || !s_ready || s_headless || !s_last_present_surface) return;
    if (GetTickCount64() - s_last_present_tick < 120) return;
    eng_submit(s_last_present_surface, 1);
    s_last_present_tick = GetTickCount64() - 100;
}

static void eng_submit_and_wait(void* user, u32 reason)
{
    (void)user; (void)reason;
    RSX_B2P_ENTER(B2P_WSUBMIT); RSX_B2P_ADD(B2C_SUBMIT, 1);
    eng_submit(0, 1);
    RSX_B2P_LEAVE();
}

/* ---- readback and present -------------------------------------------------- */

static ID3D12CommandAllocator* s_aux_alloc;
static ID3D12Resource* s_readback; static u32 s_readback_cap;

/* Diagnostics only (present log, trace triggers, surface dumps, frame grabs):
 * wait for the GPU so the bytes are the frame just submitted. */
static int s_rb_host;   /* PS3RECOMP_FRAME_HOST=1: eng_dump_frame reads the host-size surface (no guest resample) */
static void eng_readback(void* user, u32 surface, u32 x, u32 y, u32 w, u32 h, void* out, u32 out_pitch)
{
    ENG_JOIN("readback");
    (void)user;
    EngObj* o = eng_owner(surface);
    if (!o || !out || !out_pitch || !w || !h || !s_ready) return;
    /* The caller's rectangle is in guest pixels; the texture may be larger.
     * Read the matching host rectangle and resample it (nearest) on the way
     * out, so the caller sees the surface at the size it asked for. */
    const u32 gw = w, gh = h;
    const int scaled = !s_rb_host && o->gw && o->gh && (o->gw != o->w || o->gh != o->h);
    if (scaled) {
        if (x + w > o->gw || y + h > o->gh) return;
        const u32 x0 = (u32)((u64)x * o->w / o->gw), x1 = (u32)((u64)(x + w) * o->w / o->gw);
        const u32 y0 = (u32)((u64)y * o->h / o->gh), y1 = (u32)((u64)(y + h) * o->h / o->gh);
        x = x0; y = y0; w = x1 > x0 ? x1 - x0 : 1u; h = y1 > y0 ? y1 - y0 : 1u;
    }
    if (x + w > o->w || y + h > o->h) return;
    const u32 bpp = eng_bpp(o->fmt);
    const u32 pitch = (w * bpp + 255u) & ~255u;
    const u32 need = pitch * h;
    if (need > s_readback_cap) {
        RELEASE(s_readback);
        s_readback = make_buffer(D3D12_HEAP_TYPE_READBACK, need, D3D12_RESOURCE_STATE_COPY_DEST);
        s_readback_cap = s_readback ? need : 0;
        if (!s_readback) return;
    }
    if (!s_aux_alloc && FAILED(CALL(s_dev, CreateCommandAllocator, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator, (void**)&s_aux_alloc)))
        return;
    /* Everything submitted so far has to have run, including whatever list
     * last used the aux allocator. */
    fence_wait(s_fence_value);
    CALL0(s_aux_alloc, Reset);
    CALL(s_list, Reset, s_aux_alloc, NULL);
    res_transition(o->res, &o->state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    bar_flush();
    D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
    src.pResource = o->res; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; src.SubresourceIndex = 0;
    dst.pResource = s_readback; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint.Format = o->fmt;
    dst.PlacedFootprint.Footprint.Width = w; dst.PlacedFootprint.Footprint.Height = h;
    dst.PlacedFootprint.Footprint.Depth = 1; dst.PlacedFootprint.Footprint.RowPitch = pitch;
    D3D12_BOX box = { x, y, 0, x + w, y + h, 1 };
    eng_check_copy(dst.PlacedFootprint.Offset, pitch, "readback");
    CALL(s_list, CopyTextureRegion, &dst, 0, 0, 0, &src, &box);
    CALL0(s_list, Close);
    ID3D12CommandList* lists[1] = { (ID3D12CommandList*)s_list };
    CALL(s_queue, ExecuteCommandLists, 1, lists);
    fence_wait(fence_signal());
    u8* p = NULL; D3D12_RANGE rg = { 0, need };
    if (SUCCEEDED(CALL(s_readback, Map, 0, &rg, (void**)&p))) {
        if (!scaled) {
            for (u32 r = 0; r < h; r++) memcpy((u8*)out + (size_t)r * out_pitch, p + (size_t)r * pitch, (size_t)w * bpp);
        } else {
            for (u32 r = 0; r < gh; r++) {
                const u8* srow = p + (size_t)(((u64)r * h + h / 2) / gh) * pitch;
                u8* drow = (u8*)out + (size_t)r * out_pitch;
                for (u32 c = 0; c < gw; c++)
                    memcpy(drow + (size_t)c * bpp, srow + (size_t)(((u64)c * w + w / 2) / gw) * bpp, bpp);
            }
        }
        D3D12_RANGE wr = {0, 0}; CALL(s_readback, Unmap, 0, &wr);
    }
}

/* Opt-in readback of the presented surface to a PPM, as the Metal backend:
 * RSX_REPLAY_DUMP_PATH (tools/rsx_replay names each file), PS3RECOMP_FRAME_GRAB
 * (<prefix>.req asks for one <prefix>.ppm; tools/autoplay.sh's HUD check) and
 * PS3RECOMP_FRAME_DUMP every PS3RECOMP_FRAME_EVERY presents. The METAL-named
 * variables are honoured too, so the Mac tools run unchanged. */
static const char* env2(const char* a, const char* b)
{
    const char* v = getenv(a);
    if (!v || !*v) v = getenv(b);
    return (v && *v) ? v : NULL;
}
static void eng_dump_frame(u32 surface)
{
    EngObj* o = eng_owner(surface);
    if (!o || o->fmt != DXGI_FORMAT_R8G8B8A8_UNORM) return;
    const char* path = getenv("RSX_REPLAY_DUMP_PATH");
    if (path && !*path) path = NULL;
    char seqpath[1024], grab_req[1024], grab_tmp[1024], grab_out[1024];
    int grabbing = 0;
    static const char* grab = (const char*)1;
    if (grab == (const char*)1) grab = env2("PS3RECOMP_FRAME_GRAB", "PS3RECOMP_METAL_FRAME_GRAB");
    if (!path && grab) {
        snprintf(grab_req, sizeof grab_req, "%s.req", grab);
        if (_access(grab_req, 0) == 0) {
            snprintf(grab_tmp, sizeof grab_tmp, "%s.ppm.tmp", grab);
            snprintf(grab_out, sizeof grab_out, "%s.ppm", grab);
            path = grab_tmp; grabbing = 1;
        }
    }
    if (!path) {
        static const char* dump = (const char*)1;
        if (dump == (const char*)1) dump = env2("PS3RECOMP_FRAME_DUMP", "PS3RECOMP_METAL_FRAME_DUMP");
        if (!dump) return;
        extern unsigned g_rsx_ssr_dump_tag, g_rsx_ssr_dump_frame;
        if (g_rsx_ssr_dump_tag) {   /* RSX_SSR_VIEWS: the last frame of an off / on / debug window */
            static const char* const nm[4] = { "", "off", "on", "dbg" };
            snprintf(seqpath, sizeof seqpath, "%s.%s.%06u.ppm", dump, nm[g_rsx_ssr_dump_tag & 3], g_rsx_ssr_dump_frame);
            path = seqpath; goto dump_now;
        }
        static unsigned frame, every; static int seq = -1;
        if (!every) { const char* e = env2("PS3RECOMP_FRAME_EVERY", "PS3RECOMP_METAL_FRAME_EVERY"); every = e ? (unsigned)atoi(e) : 120u; if (!every) every = 120u; }
        if (seq < 0) seq = env2("PS3RECOMP_FRAME_SEQ", "PS3RECOMP_METAL_FRAME_SEQ") ? 1 : 0;
        ++frame;
        { static unsigned ab = ~0u; if (ab == ~0u) { const char* e = getenv("RT_AO_AB"); ab = e ? (unsigned)atoi(e) : 0; }
          if (rsx_rt_view_of(s_pres_n) >= 0 || rsx_rt_view_dump(s_pres_n)) { if (!rsx_rt_view_dump(s_pres_n)) return; frame = s_pres_n; }
          else if (ab >= 2) { if ((s_pres_n % ab) > 1) return; frame = s_pres_n; }
          else if ((frame % every) != 0) return; }
        path = dump;
        if (seq) { snprintf(seqpath, sizeof seqpath, "%s.%06u.ppm", dump, frame); path = seqpath; }
    }
    dump_now:;
    static int host = -1; if (host < 0) host = getenv("PS3RECOMP_FRAME_HOST") ? 1 : 0;
    const u32 w = host ? o->w : (o->gw ? o->gw : o->w), h = host ? o->h : (o->gh ? o->gh : o->h);
    if (!w || !h || w > 16384 || h > 16384) return;
    u8* rgba = (u8*)malloc((size_t)w * h * 4);
    u8* rgb = (u8*)malloc((size_t)w * h * 3);
    if (!rgba || !rgb) { free(rgba); free(rgb); return; }
    s_rb_host = host; eng_readback(NULL, surface, 0, 0, w, h, rgba, w * 4); s_rb_host = 0;
    for (size_t p = 0; p < (size_t)w * h; p++) { rgb[p*3] = rgba[p*4]; rgb[p*3+1] = rgba[p*4+1]; rgb[p*3+2] = rgba[p*4+2]; }
    FILE* f = fopen(path, "wb");
    if (f) {
        fprintf(f, "P6\n%u %u\n255\n", w, h);
        fwrite(rgb, 1, (size_t)w * h * 3, f);
        fclose(f);
        if (grabbing) { remove(grab_out); rename(grab_tmp, grab_out); remove(grab_req); }
        else fprintf(stderr, "[rsx engine/d3d12] captured frame to %s\n", path);
    }
    free(rgba); free(rgb);
}

/* RSX_VRAM_LOG=<s>: every <s> seconds, the video memory Windows charges the
 * device against its budget, and the engine's own objects by kind (bytes
 * estimated from size, format and samples; placed textures as their 64 MB
 * heaps), so a run that runs out of VRAM shows what filled it. */
static u64 eng_px_bytes(DXGI_FORMAT f)
{
    switch (f) {
    case DXGI_FORMAT_BC1_UNORM: case DXGI_FORMAT_BC1_UNORM_SRGB: return 0;   /* half a byte: see the caller */
    case DXGI_FORMAT_BC2_UNORM: case DXGI_FORMAT_BC3_UNORM: case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_UNORM_SRGB: case DXGI_FORMAT_R8_UNORM: return 1;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R32G32_FLOAT: case ENG_DEPTH_FMT:
    case ENG_DEPTH_RES_FMT: return 8;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return 16;
    case DXGI_FORMAT_R16_UNORM: case DXGI_FORMAT_B5G6R5_UNORM: case DXGI_FORMAT_R8G8_UNORM: return 2;
    default: return 4;
    }
}
/* lane114: DXGI's budget swings 7250 <-> 6432 MB between two queries a few ms apart (the OS sweep), and callers read it
 * several times per frame. One real query at most every 4 ms (the answer is shared by every caller of the frame), and the
 * budget returned is the minimum of the last 2 s of real answers, so it stops flickering. RSX_VRAM_RAW=1 gives the old
 * behaviour (every call asks DXGI, raw budget). */
static int eng_vram_query(unsigned long long* used, unsigned long long* budget)
{
    if (!s_adapter3) return 0;
    static int raw = -1; if (raw < 0) { const char* e = getenv("RSX_VRAM_RAW"); raw = (e && e[0] == '1'); }
    DXGI_QUERY_VIDEO_MEMORY_INFO vi = {0};
    if (raw) {
        if (FAILED(CALL(s_adapter3, QueryVideoMemoryInfo, 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &vi))) return 0;
        *used = vi.CurrentUsage; *budget = vi.Budget;
        eng_mp_setup(); if (s_mp_budget_ovr) *budget = s_mp_budget_ovr;   /* hitch108 RSX_VRAM_BUDGET_MB */
        return 1;
    }
    enum { WN = 32 };
    static ULONGLONG t_last, bt[WN]; static unsigned long long bb[WN], u_last; static unsigned bn, bh;
    const ULONGLONG now = GetTickCount64();
    if (!t_last || now - t_last >= 4) {
        if (FAILED(CALL(s_adapter3, QueryVideoMemoryInfo, 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &vi))) return 0;
        t_last = now; u_last = vi.CurrentUsage;
        if (!bn || now - bt[(bh + WN - 1) % WN] >= 100) { bt[bh] = now; bb[bh] = vi.Budget; bh = (bh + 1) % WN; if (bn < WN) bn++; }
        else if (vi.Budget < bb[(bh + WN - 1) % WN]) bb[(bh + WN - 1) % WN] = vi.Budget;
    }
    unsigned long long m = ~0ull;
    for (unsigned i = 0; i < bn; i++) if (now - bt[i] <= 2000 && bb[i] < m) m = bb[i];
    if (m == ~0ull) { m = bb[(bh + WN - 1) % WN]; }
    *used = u_last; *budget = m;
    eng_mp_setup(); if (s_mp_budget_ovr) *budget = s_mp_budget_ovr;   /* hitch108 RSX_VRAM_BUDGET_MB */
    return 1;
}
static void eng_vram_log(void)
{
    static double every = -1.0; static ULONGLONG last;
    if (every < 0.0) { const char* e = getenv("RSX_VRAM_LOG"); every = e ? atof(e) : 0.0; }
    if (every <= 0.0) return;
    const ULONGLONG now = GetTickCount64();
    if (last && (double)(now - last) < every * 1000.0) return;
    last = now;
    enum { K_TEX, K_COL, K_DEP, K_SNAP, K_MSC, K_MSD, K_RET, K_N };
    static const char* const nm[K_N] = { "textures (committed)", "colour targets", "depth targets", "snapshots",
                                         "MSAA colour twins", "MSAA depth twins", "retired, not yet freed" };
    u64 by[K_N] = {0}; u32 cnt[K_N] = {0};
    struct { u64 b; u32 h, w, hh, ms; int kind; } top[8]; u32 ntop = 0;
    for (u32 i = 0; i < s_obj_count; i++) {
        const EngObj* o = &s_obj[i];
        if (!o->res && !o->ms) continue;
        if (o->kind == OBJ_VIEW || o->kind == OBJ_NONE) continue;
        const u64 px = (u64)o->w * o->h * (o->faces ? o->faces : 1);
        const u64 pb = eng_px_bytes(o->fmt);
        u64 b = pb ? px * pb : px / 2;
        if (o->mips > 1) b = b * 4 / 3;
        u64 mb = o->ms ? px * eng_px_bytes(o->kind == OBJ_DEPTH ? ENG_DEPTH_RES_FMT : o->fmt) * s_ms_n : 0;
        int k = o->kind == OBJ_TEXTURE ? K_TEX : o->kind == OBJ_COLOR ? K_COL : o->kind == OBJ_DEPTH ? K_DEP : K_SNAP;
        if (o->kind == OBJ_TEXTURE && o->heap) b = 0;   /* counted with its heap */
        if (o->retired) k = K_RET;
        if (o->res) { by[k] += b; cnt[k]++; }
        if (mb) { const int km = o->retired ? K_RET : o->kind == OBJ_DEPTH ? K_MSD : K_MSC; by[km] += mb; cnt[km]++; }
        const u64 tb = (o->res ? b : 0) + mb;
        if (o->kind != OBJ_TEXTURE) {
            u32 j = ntop < 8 ? ntop++ : 8;
            if (j == 8) { u32 m = 0; for (u32 q = 1; q < 8; q++) if (top[q].b < top[m].b) m = q; if (top[m].b >= tb) continue; j = m; }
            top[j].b = tb; top[j].h = i + 1; top[j].w = o->w; top[j].hh = o->h; top[j].ms = o->ms ? s_ms_n : 1; top[j].kind = o->kind;
        }
    }
    u32 heaps = 0; for (u32 i = 0; i < s_ntheap; i++) if (s_theap[i].heap) heaps++;
    u64 snap = 0; for (u32 i = 0; i < s_snap_pool_n; i++) snap += (u64)s_snap_pool[i].w * s_snap_pool[i].h * eng_px_bytes(s_snap_pool[i].fmt);
    DXGI_QUERY_VIDEO_MEMORY_INFO vi = {0};
    { unsigned long long uu = 0, bu = 0; if (eng_vram_query(&uu, &bu)) { vi.CurrentUsage = uu; vi.Budget = bu; } }
    eng_mp_setup(); if (s_mp_budget_ovr) vi.Budget = s_mp_budget_ovr;   /* hitch108 */
    u64 sum = (u64)heaps * TEX_HEAP_BYTES + snap; for (int k = 0; k < K_N; k++) sum += by[k];
    u32 rth = 0; u64 rtl = 0; const u64 rtt = rtheap_stats(&rth, &rtl);
    if (rtt > rtl) sum += rtt - rtl;   /* the placed targets are in 'colour targets' already: only the heaps' free part is added */
    sum += s_dp_bytes;   /* depthpool: parked depth targets (not in s_obj any more) */
    u32 dsh_n = 0; u64 dsh_l = 0; const u64 dsh_t = dsheap_stats(&dsh_n, &dsh_l);   /* RSX_DS_HEAP: same rule (depth targets and snapshots inside stay in their own rows) */
    if (dsh_t > dsh_l) sum += dsh_t - dsh_l;
    fprintf(stderr, "[vram] in use %llu MB of a %llu MB budget; engine objects ~%llu MB: texture heaps %u x 64 MB, snapshot pool %llu MB",
            (unsigned long long)(vi.CurrentUsage >> 20), (unsigned long long)(vi.Budget >> 20), (unsigned long long)(sum >> 20),
            heaps, (unsigned long long)(snap >> 20));
    for (int k = 0; k < K_N; k++)
        if (cnt[k]) fprintf(stderr, ", %s %u = %llu MB", nm[k], cnt[k], (unsigned long long)(by[k] >> 20));
    if (s_dp_n) fprintf(stderr, ", depth pool %u = %llu MB", s_dp_n, (unsigned long long)(s_dp_bytes >> 20));
    if (rth) fprintf(stderr, ", RT heaps %u x %llu MB (placed targets inside: %llu MB)", rth, (unsigned long long)(s_rth_bytes >> 20), (unsigned long long)(rtl >> 20));
    if (dsh_n) fprintf(stderr, ", DS heaps %u x %llu MB (placed depth targets and snapshots inside: %llu MB)", dsh_n, (unsigned long long)(s_dsh_bytes >> 20), (unsigned long long)(dsh_l >> 20));
    fputc('\n', stderr);
    if (s_mp_log || s_mp_on || s_mp_cede_pct || s_mp_depth_max)   /* hitch108 */
        fprintf(stderr, "[vram108] created (window total): colour %u (%u over 85%% of the budget), depth %u (%u over 85%%), twins %u (%u over 85%%); twins parked %u reused %u evicted %u aged %u now %u (%llu MB), ceded %u by memory %u by depth cap\n",
                s_mp_made[0], s_mp_over85[0], s_mp_made[1], s_mp_over85[1], s_mp_made[2], s_mp_over85[2], s_mps.parked, s_mps.reused, s_mps.evicted, s_mps.aged,
                s_mp_n, (unsigned long long)(s_mp_bytes >> 20), s_mps.ceded_pct, s_mps.ceded_depth);
    { static int listed;   /* the depth targets once, guest and host sizes: what holds the most */
      if (!listed && cnt[K_DEP] >= 24) {
          listed = 1;
          for (u32 i = 0; i < s_obj_count; i++)
              if (s_obj[i].res && s_obj[i].kind == OBJ_DEPTH)
                  fprintf(stderr, "[vram]   depth h%u guest %ux%u host %ux%u%s\n", i + 1, s_obj[i].gw, s_obj[i].gh,
                          s_obj[i].w, s_obj[i].h, s_obj[i].retired ? " retired" : "");
      } }
    for (u32 j = 0; j < ntop; j++)
        fprintf(stderr, "[vram]   h%u %s %ux%u x%u samples: %llu MB\n", top[j].h,
                top[j].kind == OBJ_DEPTH ? "depth" : top[j].kind == OBJ_COLOR ? "colour" : "snapshot",
                top[j].w, top[j].hh, top[j].ms, (unsigned long long)(top[j].b >> 20));
}

/* logv2: the VRAM breakdown for the structured log. full=0: one-second tick (compact: categories as [n,mb], counts, top 3 sites); full=1: the anomaly record
 * (top 30 live resources, top 24 sites by live size and by objects created in the last 5 s). Same estimates as eng_vram_log. */
typedef struct { u64 b; u32 h; } V2Top;
static const char* v2_fmt_name(DXGI_FORMAT f)
{
    switch (f) {
    case DXGI_FORMAT_R8G8B8A8_UNORM: return "rgba8"; case DXGI_FORMAT_B8G8R8A8_UNORM: return "bgra8"; case DXGI_FORMAT_R16G16B16A16_FLOAT: return "rgba16f";
    case DXGI_FORMAT_R32_FLOAT: return "r32f"; case DXGI_FORMAT_R32G32B32A32_FLOAT: return "rgba32f"; case DXGI_FORMAT_R16_UNORM: return "r16";
    case DXGI_FORMAT_BC1_UNORM: return "bc1"; case DXGI_FORMAT_BC2_UNORM: return "bc2"; case DXGI_FORMAT_BC3_UNORM: return "bc3"; case DXGI_FORMAT_R8_UNORM: return "r8";
    case DXGI_FORMAT_B5G6R5_UNORM: return "b5g6r5"; case DXGI_FORMAT_R8G8_UNORM: return "rg8"; case DXGI_FORMAT_R32G32_FLOAT: return "rg32f";
    default: if (f == ENG_DEPTH_FMT) return "d32s8"; return "other";
    }
}
static u32 v2_win(const u32* v, const u32* tag) { u32 t = 0; const u32 now = g_v2_up_s; for (int i = 0; i < 8; i++) if (tag[i] && now - tag[i] < 5u) t += v[i]; return t; }
static int eng_v2_vram(int full, char* out, size_t n, unsigned long long* used, unsigned long long* budget)
{
    unsigned long long uu = 0, bu = 0;
    if (!eng_vram_query(&uu, &bu)) return 0;
    eng_mp_setup(); if (s_mp_budget_ovr) bu = s_mp_budget_ovr;
    *used = uu; *budget = bu;
    enum { K_TEX, K_COL, K_DEP, K_SNAP, K_MSC, K_MSD, K_RET, K_N };
    static const char* const nm[K_N] = { "tex", "colour", "depth", "snapshot", "msaa_colour", "msaa_depth", "retired" };
    u64 by[K_N] = {0}; u32 cnt[K_N] = {0};
    static u64 sb[V2_MAXSITE]; static u32 sn[V2_MAXSITE];   /* live bytes / objects per site (this scan only runs on the present thread) */
    memset(sb, 0, sizeof sb); memset(sn, 0, sizeof sn);
    V2Top top[30]; u32 ntop = 0; u32 live_obj = 0;
    u32 tw[64]; u32 ntw = 0, ntw_all = 0, tw_used1s = 0;   /* MSAA twins: which ones were drawn into in the last ~60 encoded frames */
    for (u32 i = 0; i < s_obj_count; i++) {
        const EngObj* o = &s_obj[i];
        if (!o->res && !o->ms) continue;
        if (o->kind == OBJ_VIEW || o->kind == OBJ_NONE) continue;
        if (o->ms && !o->retired) { ntw_all++; if (ntw < 64) tw[ntw++] = i; if (s_enc_frame - o->mt_last < 60u) tw_used1s++; }
        const u64 px = (u64)o->w * o->h * (o->faces ? o->faces : 1);
        const u64 pb = eng_px_bytes(o->fmt);
        u64 b = pb ? px * pb : px / 2;
        if (o->mips > 1) b = b * 4 / 3;
        const u64 mb = o->ms ? px * eng_px_bytes(o->kind == OBJ_DEPTH ? ENG_DEPTH_RES_FMT : o->fmt) * s_ms_n : 0;
        int k = o->kind == OBJ_TEXTURE ? K_TEX : o->kind == OBJ_COLOR ? K_COL : o->kind == OBJ_DEPTH ? K_DEP : K_SNAP;
        if (o->kind == OBJ_TEXTURE && o->heap) b = 0;
        if (o->retired) k = K_RET;
        if (o->res) { by[k] += b; cnt[k]++; }
        if (mb) { const int km = o->retired ? K_RET : o->kind == OBJ_DEPTH ? K_MSD : K_MSC; by[km] += mb; cnt[km]++; }
        const u64 tb = (o->res ? b : 0) + mb;
        live_obj++;
        sb[o->v2_site] += tb; sn[o->v2_site]++;
        if (full && (o->kind != OBJ_TEXTURE || tb >= (4u << 20))) {
            u32 j = ntop < 30 ? ntop++ : 30;
            if (j == 30) { u32 m = 0; for (u32 q = 1; q < 30; q++) if (top[q].b < top[m].b) m = q; if (top[m].b >= tb) continue; j = m; }
            top[j].b = tb; top[j].h = i + 1;
        }
    }
    u32 heaps = 0; for (u32 i = 0; i < s_ntheap; i++) if (s_theap[i].heap) heaps++;
    u64 snap = 0; for (u32 i = 0; i < s_snap_pool_n; i++) snap += (u64)s_snap_pool[i].w * s_snap_pool[i].h * eng_px_bytes(s_snap_pool[i].fmt);
    size_t o2 = 0;
#define APP(...) do { if (o2 < n) { int r_ = snprintf(out + o2, n - o2, __VA_ARGS__); if (r_ > 0) o2 += (size_t)r_; } } while (0)
    APP("\"vram\":[%llu,%llu],\"cat\":{", uu >> 20, bu >> 20);   /* [used MB, budget MB]; cat: name -> [objects, MB] */
    for (int k = 0; k < K_N; k++) APP("%s\"%s\":[%u,%llu]", k ? "," : "", nm[k], cnt[k], by[k] >> 20);
    u32 rth = 0; u64 rtl = 0; const u64 rtt = rtheap_stats(&rth, &rtl);   /* RT heaps: [heaps, MB]; the targets inside stay in "colour" */
    (void)rtl;
    u32 dsh_n = 0; u64 dsh_l = 0; const u64 dsh_t = dsheap_stats(&dsh_n, &dsh_l); (void)dsh_l;   /* RSX_DS_HEAP: [heaps, MB]; depth targets and snapshots inside stay in "depth" / "snap" */
    APP(",\"tex_heaps\":[%u,%llu],\"rt_heaps\":[%u,%llu],\"ds_heaps\":[%u,%llu],\"snap_pool\":[%u,%llu],\"twin_pool\":[%u,%llu],\"depth_pool\":[%u,%llu]}", heaps, ((u64)heaps * TEX_HEAP_BYTES) >> 20, rth, rtt >> 20, dsh_n, dsh_t >> 20, s_snap_pool_n, snap >> 20,
        s_mp_n, s_mp_bytes >> 20, s_dp_n, s_dp_bytes >> 20);
    APP(",\"live\":[%u,%u,%u,%u]", live_obj, s_obj_count, s_obj_free_count, s_retired_count);   /* objects, slots used, free slots, retired waiting */
    /* MSAA twins: how many exist and how many were drawn into in the last ~1 s; every 5 s (and in anomalies) the full list:
     * [d=depth / c=colour, w, h, MB, encoded frames since the last draw into it, guest address]. A twin idle for many seconds is memory
     * held for nothing (reads use the resolved copy). */
    APP(",\"twins\":[%u,%u]", ntw_all, tw_used1s);
    if (full || (g_v2_up_s % 5u) == 0) {
        APP(",\"twin_list\":[");
        for (u32 a = 0; a < ntw; a++) {
            const EngObj* o = &s_obj[tw[a]];
            const u64 mb = (u64)o->w * o->h * eng_px_bytes(o->kind == OBJ_DEPTH ? ENG_DEPTH_RES_FMT : o->fmt) * s_ms_n;
            APP("%s[\"%c\",%u,%u,%llu,%u,\"0x%08X\"]", a ? "," : "", o->kind == OBJ_DEPTH ? 'd' : 'c', o->w, o->h, mb >> 20, s_enc_frame - o->mt_last, o->v2_gaddr);
        }
        APP("]");
    }
    u32 ord[V2_MAXSITE]; u32 ns = 0; for (u32 i = 0; i < s_v2s_n; i++) ord[ns++] = i;
    const int lim = full ? 24 : 3;
    for (int pass = 0; pass < (full ? 2 : 1); pass++) {
        for (u32 a = 0; a < ns; a++) for (u32 b2 = a + 1; b2 < ns; b2++) {
            const V2Site *A = &s_v2s[ord[a]], *B = &s_v2s[ord[b2]];
            const u64 ka = pass ? v2_win(A->cr, A->tag) : sb[ord[a]], kb = pass ? v2_win(B->cr, B->tag) : sb[ord[b2]];
            if (kb > ka) { u32 t = ord[a]; ord[a] = ord[b2]; ord[b2] = t; }
        }
        APP(",\"%s\":[", pass ? "sites_by_created5s" : "sites");   /* [site, caller, live n, live MB, created 5s, destroyed 5s, twin created 5s, twin destroyed 5s, created total, destroyed total] */
        int w = 0;
        for (u32 a = 0; a < ns && w < lim; a++) {
            const u32 si = ord[a]; const V2Site* S = &s_v2s[si];
            const u32 cr = v2_win(S->cr, S->tag), de = v2_win(S->de, S->tag), tcr = v2_win(S->tcr, S->tag), tde = v2_win(S->tde, S->tag);
            if (!sb[si] && !cr && !de && !tcr && !tde) continue;
            if (S->fn == NULL && si) continue;
            APP("%s[\"%s:%d\",\"%s:%d\",%u,%llu,%u,%u,%u,%u,%llu,%llu]",
                w ? "," : "", S->fn ? S->fn : "other", S->line, S->cfn ? S->cfn : "-", S->cline, sn[si], sb[si] >> 20, cr, de, tcr, tde, S->cr_tot, S->de_tot);
            w++;
        }
        APP("]");
    }
    if (full) {
        for (u32 a = 0; a < ntop; a++) for (u32 b2 = a + 1; b2 < ntop; b2++) if (top[b2].b > top[a].b) { V2Top t = top[a]; top[a] = top[b2]; top[b2] = t; }
        APP(",\"top_resources\":[");
        for (u32 a = 0; a < ntop; a++) {
            const EngObj* o = &s_obj[top[a].h - 1]; const V2Site* S = &s_v2s[o->v2_site];
            const char* kn = o->kind == OBJ_TEXTURE ? "texture" : o->kind == OBJ_COLOR ? "colour" : o->kind == OBJ_DEPTH ? "depth" : "snapshot";
            APP("%s{\"h\":%u,\"type\":\"%s\",\"fmt\":\"%s\",\"w\":%u,\"hh\":%u,\"guest\":\"%ux%u\",\"samples\":%u,\"mb\":%llu,\"age_frames\":%u,\"twin_idle_enc_frames\":%d,\"retired\":%d,"
                "\"site\":\"%s:%d\",\"caller\":\"%s:%d\",\"gaddr\":\"0x%08X\",\"ginfo\":%u}",
                a ? "," : "", top[a].h, kn, v2_fmt_name(o->fmt), o->w, o->h, o->gw, o->gh, o->ms ? s_ms_n : 1, top[a].b >> 20,
                o->v2_site ? g_v2_frame - o->v2_born : 0, o->ms ? (int)(s_enc_frame - o->mt_last) : -1, o->retired, S->fn ? S->fn : "?", S->line,
                S->cfn ? S->cfn : "-", S->cline, o->v2_gaddr, o->v2_ginfo);
        }
        APP("]");
    }
#undef APP
    if (o2 >= n) { out[n - 1] = 0; }
    return 1;
}

static void eng_present(void* user, u32 surface)
{
    (void)user;
    eng_debug_drain();
    eng_vram_log();
    { static int on = -1; static unsigned n = 0;
      if (on < 0) on = getenv("RSX_OBJ_STATS") ? 1 : 0;
      if (on && (++n % 600) == 0) {
          fprintf(stderr, "[rsx engine/d3d12] objects: %u slots used, %u free (present %u); per 600 presents: %u textures, %u targets, %u snapshots (+%u reused, pool %u), %u vertex buffers created\n",
                  s_obj_count, s_obj_free_count, n, s_stat_tex, s_stat_rt, s_stat_snap, s_stat_snap_reused, s_snap_pool_n, s_stat_buf);
          s_stat_tex = s_stat_rt = s_stat_snap = s_stat_snap_reused = s_stat_buf = 0; } }
    if (!eng_owner(surface)) { eng_submit(0, 0); return; }
    eng_submit(surface, 0);
    /* The frame grab shows what was presented: after FXAA when it is on. */
    eng_dump_frame(rsx_aa_mode() == 1 && s_aa_obj && !s_headless ? s_aa_obj : surface);
    s_pres_n++;
}

static const rsx_draw_backend s_engine_backend = {
    .user                 = NULL,
    .init                 = eng_init,
    .shutdown             = eng_shutdown,
    .submit_and_wait      = eng_submit_and_wait,
    .texture_create       = eng_texture_create,
    .texture_upload       = eng_texture_upload,
    .texture_release      = eng_obj_release,
    .color_target_create  = eng_color_target_create,
    .color_target_release = eng_obj_release,
    .surface_view         = eng_surface_view,
    .depth_target_create  = eng_depth_target_create,
    .depth_target_release = eng_obj_release,
    .depth_snapshot       = eng_depth_snapshot,
    .depth_snapshot_rgba8 = eng_depth_snapshot_rgba8,
    .color_snapshot       = eng_color_snapshot,
    .color_copy_rect      = eng_color_copy_rect,
    .query_begin          = eng_query_begin,
    .query_set            = eng_query_set,
    .query_report         = eng_query_report,
    .pipeline_create      = eng_pipeline_create,
    .pipeline_release     = eng_pipeline_release,
    .bind_targets         = eng_bind_targets,
    .bind_pipeline        = eng_bind_pipeline,
    .bind_vs_constants    = eng_bind_vs_constants,
    .reuse_vs_constants   = eng_reuse_vs_constants,
    .bind_ps_constants    = eng_bind_ps_constants,
    .bind_textures        = eng_bind_textures,
    .bind_vertex_textures = eng_bind_vertex_textures,
    .set_viewport         = eng_set_viewport,
    .set_scissor          = eng_set_scissor,
    .set_stencil_ref      = eng_set_stencil_ref,
    .set_blend_color      = eng_set_blend_color,
    .draw                 = eng_draw,
    .buffer_wrap          = eng_buffer_wrap,
    .buffer_release       = eng_buffer_release,
    .draw_buffer          = eng_draw_buffer,
    .rt_ao_inject         = eng_rt_ao_inject,
    .rt_ao_mark           = eng_rt_ao_mark,
    .rt_ao_defer          = eng_rt_ao_defer,
    .clear_color          = eng_clear_color,
    .clear_depth_stencil  = eng_clear_depth_stencil,
    .clear_depth_stencil_rect = eng_clear_depth_stencil_rect,
    .present              = eng_present,
    .readback             = eng_readback,
};

/* ---- public API -------------------------------------------------------------- */

static void eng_release_device(void)
{
    eng_enc_stop();
    eng_present_thread_stop();
    if (s_swap && s_display == DISP_FULLSCREEN) CALL(s_swap, SetFullscreenState, FALSE, NULL);
    RELEASE(s_zero_cb); RELEASE(s_readback); s_readback_cap = 0; RELEASE(s_aux_alloc);
    RELEASE(s_null_tex); RELEASE(s_blit_pso); RELEASE(s_depth_pso); RELEASE(s_depth_pack_pso);
    RELEASE(s_present_pso); RELEASE(s_fxaa_pso);
    RELEASE(s_depth_ms_pso); RELEASE(s_depth_pack_ms_pso); RELEASE(s_zcopy_ms_pso);
    RELEASE(s_helper_vs_blob); RELEASE(s_helper_blit_blob);
    RELEASE(s_rootsig); RELEASE(s_helper_rootsig);
    RELEASE(s_qheap); RELEASE(s_qread); RELEASE(s_tsheap); RELEASE(s_tsread); RELEASE(s_pheap); RELEASE(s_pread);
    for (u32 i = 0; i < ENG_FRAMES; i++) { RELEASE(s_smp_gpu[i]); RELEASE(s_alloc[i]); s_alloc_fence[i] = 0; }
    for (u32 i = 0; i < s_smp_spare_n; i++) { RELEASE(s_smp_spare[i]); s_smp_spare_fence[i] = 0; s_smp_spare_inlist[i] = 0; }
    s_smp_spare_n = 0; s_smp_cur = NULL;
    RELEASE(s_srv_gpu); RELEASE(s_srv_cpu); RELEASE(s_rtv_cpu); RELEASE(s_dsv_cpu); RELEASE(s_smp_cpu);
    RELEASE(s_rtv_ms); RELEASE(s_dsv_ms); RELEASE(s_srv_ms);
    RELEASE(s_list);
    if (s_fence_event) { CloseHandle(s_fence_event); s_fence_event = NULL; }
    RELEASE(s_fence); s_fence_value = 0;
    for (u32 i = 0; i < 3; i++) RELEASE(s_backbuf[i]);
    RELEASE(s_backbuf_rtv_heap); RELEASE(s_offscreen); RELEASE(s_swap);
    RELEASE(s_queue); RELEASE(s_dev);
    if (s_hwnd) { DestroyWindow(s_hwnd); s_hwnd = NULL; }
    s_ready = 0;
}

int rsx_d3d12_engine_init(u32 width, u32 height, const char* title)
{
    const char* hl = getenv("PS3RECOMP_D3D12_HEADLESS");
    s_headless = (hl && *hl && *hl != '0');
    s_width = width ? width : 1280; s_height = height ? height : 720;
    s_vsync = eng_read_vsync(); rsx_vblank_set_vsync(s_vsync);
    eng_read_scale();
    { const char* e = getenv("RSX_BUF_POOL"); if (e && *e == '0') g_eng_buf_pool = 0; }
    { const char* e = getenv("RSX_TEX_HEAPS"); if (e && *e == '0') g_eng_tex_heaps = 0; }
    { const char* e = getenv("RSX_VB_PAGES"); if (e && *e == '0') g_eng_vb_pages = 0; }
    /* Per-monitor DPI awareness, before any window exists: without it a
     * 4K display at 125% scaling reports 3072x1728, the borderless window
     * covers that and Windows stretches it, and a 1280x720 window is drawn
     * at 1600x900 through the desktop compositor's scaler. */
    { HMODULE u32 = GetModuleHandleA("user32.dll");
      typedef BOOL (WINAPI *SetCtxFn)(DPI_AWARENESS_CONTEXT);
      SetCtxFn set_ctx = u32 ? (SetCtxFn)(void*)GetProcAddress(u32, "SetProcessDpiAwarenessContext") : NULL;
      if (!set_ctx || !set_ctx(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) SetProcessDPIAware(); }
    eng_read_display();

    /* The engine is this backend's default; PS3RECOMP_RSX_ENGINE=vtable is
     * the way back to the rsx_state path. The query needs a registered
     * backend to answer, so register first and undo on a no. */
    rsx_draw_engine_set_backend(&s_engine_backend);
    { extern void (*g_rsx_lanes_block_hook)(unsigned, unsigned, unsigned); g_rsx_lanes_block_hook = rsx_d3d12_lanes_block; }   /* RSX_LANES */
    rsx_draw_engine_set_default(1);
    if (!rsx_draw_engine_enabled()) { rsx_draw_engine_set_backend(NULL); return -1; }

    if (!s_headless) {
        s_hwnd = eng_create_window(s_win_w, s_win_h, title);
        if (s_hwnd) rsx_host_mouse_attach(s_hwnd);
        if (s_hwnd) s_win_tid = GetWindowThreadProcessId(s_hwnd, NULL);
        if (!s_hwnd) { fprintf(stderr, "[RSX d3d12] window creation failed\n"); rsx_draw_engine_set_backend(NULL); return -1; }
    }
    if (eng_init_device(s_win_w, s_win_h) != 0) {
        fprintf(stderr, "[RSX d3d12] device initialisation failed; falling back to the vtable path\n");
        eng_release_device();
        rsx_draw_engine_set_backend(NULL);
        return -1;
    }
    if (eng_ms_want() > 1) eng_ms_apply(eng_ms_want());
    s_zero_cb = make_buffer(D3D12_HEAP_TYPE_UPLOAD, 16384, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (s_zero_cb) { void* m = NULL; D3D12_RANGE nr = {0, 0}; if (SUCCEEDED(CALL(s_zero_cb, Map, 0, &nr, &m))) { memset(m, 0, 16384); CALL(s_zero_cb, Unmap, 0, NULL); } }
    s_ready = 1;
    if (rsx_draw_engine_init(s_width, s_height) != 0) {
        fprintf(stderr, "[RSX d3d12] draw engine init failed\n");
        eng_release_device();
        rsx_draw_engine_set_backend(NULL);
        return -1;
    }
    s_active = 1;
    fprintf(stderr, "[RSX d3d12] %s (%ux%u), register-file draw engine, vsync %d, internal resolution x%.2f (%ux%u)\n",
            s_headless ? "headless" : s_display == DISP_FULLSCREEN ? "full screen" : s_display == DISP_BORDERLESS ? "borderless" : "windowed",
            s_win_w, s_win_h, s_vsync, s_scale, sc_dim(s_width), sc_dim(s_height));
    return 0;
}

void rsx_d3d12_engine_shutdown(void)
{
    if (!s_active) return;
    rsx_draw_engine_shutdown();       /* flushes through submit_and_wait */
    rsx_draw_engine_set_backend(NULL);
    eng_shutdown(NULL);
    eng_release_device();
    s_active = 0;
}

int rsx_d3d12_engine_active(void) { return s_active; }

int rsx_d3d12_engine_pump_messages(void)
{
    if (s_headless) return 0;
    { static int once; if (!once && s_win_tid && GetCurrentThreadId() != s_win_tid) { once = 1; fprintf(stderr, "[RSX d3d12] note: the message pump runs on thread %lu, the window belongs to %lu\n", (unsigned long)GetCurrentThreadId(), (unsigned long)s_win_tid); } }
    /* RSX_SCALE_FLIP=<seconds>:<scale>[:<aa>]: test hook (off by default), the F1 page's RSX_SCALE/RSX_AA change at a given time. */
    { static int fl = -1; static double at; static char nsb[16], aab[16]; static ULONGLONG t0;
      if (fl < 0) { const char* e = getenv("RSX_SCALE_FLIP"); fl = (e && sscanf(e, "%lf:%15[0-9.]:%15s", &at, nsb, aab) >= 2) ? 1 : 0; }
      if (fl == 1 && s_ready) {
          if (!t0) t0 = GetTickCount64();
          if (GetTickCount64() - t0 > (ULONGLONG)(at * 1000.0)) { fl = 2; _putenv_s("RSX_SCALE", nsb); if (aab[0]) _putenv_s("RSX_AA", aab); g_rsx_display_reload = 1;
              fprintf(stderr, "[RSX d3d12] RSX_SCALE_FLIP: RSX_SCALE=%s AA=%s%c", nsb, aab, 10); } } }
    if (g_rsx_display_reload && s_ready && s_swap) {
        /* 04/11 menu108: the pump runs on the thread that queues draws (frame_clock), while the B2 thread is the one that
         * encodes and owns the engine state (encode-thread hand-over s_enc_pending/s_enc_done, the MSAA twins, the fence).
         * The reload touched all of that with B2 mid-frame: B2 then waited for ever on s_enc_done (consumed by the join
         * done here) and the ring filled. Drain B2 first (nothing more is queued while this thread is here), then take the
         * new AA mode so the encoder never sees it change mid-frame. */
        rsx_draw_engine_b2_drain();
        { extern volatile int g_rsx_aa_pending; extern int g_rsx_aa;
          if (g_rsx_aa_pending >= 0) { g_rsx_aa = g_rsx_aa_pending; g_rsx_aa_pending = -1; } }
        eng_display_reload();
    }
    MSG msg;
    while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) return -1;
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    return s_window_closed ? -1 : 0;
}

u32 rsx_d3d12_engine_debug_color(void) { return s_clear_argb; }
u32 rsx_d3d12_engine_readback_center(void) { return rsx_draw_engine_readback_center(); }

#endif /* _WIN32 */
