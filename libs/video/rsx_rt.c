/*
 * rsx_rt - ray-traced ambient occlusion on the static level geometry. See rsx_rt.h.
 *
 * Layout of this file:
 *   1. configuration (tod.ini keys arrive as environment variables)
 *   2. vertex-program registry (which translated programs are "static world")
 *   3. instance table, build queue, level-change detection  (CPU, draw-engine thread)
 *   4. D3D12/DXR: vertex-program evaluation, BLAS/TLAS, AO passes (backend thread; both
 *      are the same thread in this engine: the FIFO walker)
 */
#define RSX_RT_D3D12
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rsx_rt.h"
#include "rsx_rt_ao_dxil.h"

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")

typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;

#define CALL(o, m, ...) ((o)->lpVtbl->m((o), __VA_ARGS__))
#define CALL0(o, m)     ((o)->lpVtbl->m((o)))
#define RELEASE(p)      do { if (p) { (p)->lpVtbl->Release(p); (p) = NULL; } } while (0)

/* ======================================================================== 1. config */

int g_rsx_rt_on = 0;
RsxRtConfig g_rsx_rt = { 0, 3.0f, 4, 1, 1.1f, 0, 1, 0, 64, 1.7f, 8, 1, 1, 0.2f };

static float env_f(const char* n, float d, float lo, float hi)
{
    const char* e = getenv(n);
    if (!e || !*e) return d;
    float v = (float)atof(e);
    return v < lo ? lo : v > hi ? hi : v;
}
static int s_views_start = -1, s_views_win = 20, s_views_cycles = 2, s_view = -1, s_views_mask = 127;   /* mask: which views are dumped */
static float s_boost[2][4] = { { 3.0f, 1.1f, 1.7f, 1.0f }, { 4.0f, 1.3f, 1.8f, 0.2f } };   /* radius, strength, power, character weight */
int rsx_rt_view_of(u32 n)
{
    if (s_views_start < 0 || (int)n < s_views_start) return -1;
    return (int)(((n - (u32)s_views_start) / (u32)s_views_win) % 7u);
}
int rsx_rt_view_dump(u32 n)
{
    if (s_views_start < 0 || (int)n < s_views_start) return 0;
    const u32 k = n - (u32)s_views_start;
    return (k % (u32)s_views_win) == (u32)s_views_win - 1u && k / ((u32)s_views_win * 7u) < (u32)s_views_cycles &&
           ((s_views_mask >> (int)((k / (u32)s_views_win) % 7u)) & 1);
}
void rsx_rt_set_view(int v) { s_view = v; }
/* The debug mode in force: the views cycle overrides RT_AO_DEBUG (views 0..2 are debug 1..3). */
static int eff_debug(void) { return s_view >= 0 ? (s_view <= 2 ? s_view + 1 : 0) : g_rsx_rt.debug; }
static float eff_radius(void)   { return s_view >= 5 ? s_boost[s_view - 5][0] : g_rsx_rt.radius; }
static float eff_strength(void) { return s_view >= 5 ? s_boost[s_view - 5][1] : g_rsx_rt.strength; }
static float eff_power(void)    { return s_view >= 5 ? s_boost[s_view - 5][2] : g_rsx_rt.power; }
static float eff_char(void)     { return s_view >= 5 ? s_boost[s_view - 5][3] : g_rsx_rt.charatt; }

void rsx_rt_config_load(void)
{
    static int done;
    if (done) return;
    done = 1;
    g_rsx_rt.ao        = (int)env_f("RT_AO", 0, 0, 1);
    g_rsx_rt.quality   = (int)env_f("RT_AO_QUALITY", 1, 0, 2);
    g_rsx_rt.radius    = env_f("RT_AO_RADIUS", 3.0f, 0.05f, 50.0f);
    g_rsx_rt.rays      = (int)env_f("RT_AO_RAYS", g_rsx_rt.quality == 0 ? 1.0f : g_rsx_rt.quality == 1 ? 2.0f : 4.0f, 1, 8);  /* low 1, medium 2, high 4 (the history adds up) */
    g_rsx_rt.halfres   = (int)env_f("RT_AO_HALFRES", 1, 0, 1);
    g_rsx_rt.strength  = env_f("RT_AO_STRENGTH", 1.1f, 0, 3);
    g_rsx_rt.power     = env_f("RT_AO_POWER", 1.7f, 0.5f, 6.0f);
    g_rsx_rt.temporal  = (int)env_f("RT_AO_TEMPORAL", 8, 0, 64);
    g_rsx_rt.dyn       = (int)env_f("RT_AO_DYN", 1, 0, 1);
    g_rsx_rt.charatt   = env_f("RT_AO_CHAR", 0.2f, 0, 1);
    g_rsx_rt.debug     = (int)env_f("RT_AO_DEBUG", 0, 0, 3);
    { const char* v = getenv("RT_AO_VIEWS");
      if (v && *v) { int a = -1, b = 20, c = 2; if (sscanf(v, "%d,%d,%d", &a, &b, &c) >= 1) { s_views_start = a; s_views_win = b > 0 ? b : 20; s_views_cycles = c > 0 ? c : 2; } }
      const char* vm = getenv("RT_AO_VIEWS_DUMP");      /* e.g. "3,4": only those views are dumped */
      if (vm && *vm) { int m = 0; for (const char* c = vm; *c; c++) if (*c >= '0' && *c <= '6') m |= 1 << (*c - '0'); if (m) s_views_mask = m; }
      const char* bo = getenv("RT_AO_BOOST");
      if (bo && *bo) { float v[8] = { 4, 1.3f, 1.8f, 1.0f, 4, 1.3f, 1.8f, 0.4f }; if (sscanf(bo, "%f,%f,%f,%f;%f,%f,%f,%f", v, v + 1, v + 2, v + 3, v + 4, v + 5, v + 6, v + 7) >= 4) for (int i = 0; i < 8; i++) s_boost[i / 4][i % 4] = v[i]; } }
    g_rsx_rt.anim      = g_rsx_rt.temporal > 1 ? 1 : (int)env_f("RT_AO_ANIM", 0, 0, 1);              /* the history needs a new pattern every frame */
    g_rsx_rt.min_draws = (int)env_f("RT_AO_MIN_DRAWS", 64, 1, 100000);
    g_rsx_rt.trace     = (int)env_f("RT_AO_TRACE", 0, 0, 1000000);
    g_rsx_rt_on = g_rsx_rt.ao;
    if (g_rsx_rt_on)
        fprintf(stderr, "[rt] RT_AO=1 quality %d radius %.2f rays %d halfres %d strength %.2f power %.2f temporal %d dyn %d debug %d anim %d\n",
                g_rsx_rt.quality, g_rsx_rt.radius, g_rsx_rt.rays, g_rsx_rt.halfres, g_rsx_rt.strength, g_rsx_rt.power,
                g_rsx_rt.temporal, g_rsx_rt.dyn, g_rsx_rt.debug, g_rsx_rt.anim);
}

/* ============================================================ 2. vertex-program registry */

#define RT_MAX_ATTR 16
typedef struct {
    u64 id;
    char* hlsl;
    int is_static, is_dyn;             /* dyn: indirect constant addressing (skinning, instancing): rebuilt every frame */
    u32 nfields;
    int field_attr[RT_MAX_ATTR];       /* attribute index of each VSInput field, in declaration order */
    u16 uc[64]; u32 nuc;               /* constants the program reads (literal indices, camera slots 7..15 left out): they identify an instance */
    ID3D12PipelineState* cs;           /* the evaluation kernel, built on first use */
    int cs_failed;
} RtVs;
static RtVs* s_vs; static u32 s_vs_n, s_vs_cap;
static RtVs* s_vs_cache[256];

static RtVs* vs_find(u64 id)
{
    for (u32 i = 0; i < s_vs_n; i++) if (s_vs[i].id == id) return &s_vs[i];
    return NULL;
}

int rsx_rt_register_vs(u64 id, const char* hlsl)
{
    if (!g_rsx_rt_on || !hlsl) return 0;
    RtVs* v = vs_find(id);
    if (v) return v->is_static | (v->is_dyn << 1);
    if (s_vs_n == s_vs_cap) {
        s_vs_cap = s_vs_cap ? s_vs_cap * 2 : 256;
        s_vs = (RtVs*)realloc(s_vs, s_vs_cap * sizeof *s_vs);
        memset(s_vs_cache, 0, sizeof s_vs_cache);
        if (!s_vs) { s_vs_n = s_vs_cap = 0; return 0; }
    }
    v = &s_vs[s_vs_n++];
    memset(v, 0, sizeof *v);
    v->id = id;
    /* Static world program: nothing indexes the constants, no vertex textures; the clip\n     * position is linear in the vertex. HYPOTHESIS (docs/RAYTRACING_PLAN.md 2.4, widened after\n     * the first trace: most level geometry goes object -> clip through c[32..35]) that the game\n     * keeps the frame's world-to-clip matrix in c[8..11] whatever program draws, so that\n     * world = inverse(c[8..11]) * clip; RT_AO_DEBUG=2 checks it against the depth buffer. */
    const int ind = strstr(hlsl, "511u") != NULL;
    const int ok = !strstr(hlsl, "Texture") &&
                   strstr(hlsl, "VSOutput main(VSInput input)") && strstr(hlsl, "struct VSInput");
    if (ok) {
        const char* s = strstr(hlsl, "struct VSInput");
        const char* e = strchr(s, '}');
        for (const char* p = s; p && p < e && v->nfields < RT_MAX_ATTR;) {
            p = strstr(p, "float4 a");
            if (!p || p >= e) break;
            p += 8;
            v->field_attr[v->nfields++] = atoi(p);
        }
        v->is_static = !ind && v->nfields > 0;
        v->is_dyn = ind && v->nfields > 0;
        for (const char* p = hlsl; !ind && (p = strstr(p, "vp_c[")) != NULL && v->nuc < 64;) {
            p += 5;
            if (*p < '0' || *p > '9') continue;
            const int n = atoi(p);
            if (n < 16 || n > 42 || (n >= 32 && n <= 35)) continue;   /* globals, camera (c8..11) and object-to-clip (c32..35, camera dependent) are not identity */
            u32 i = 0; for (; i < v->nuc; i++) if (v->uc[i] == n) break;
            if (i == v->nuc) v->uc[v->nuc++] = (u16)n;
        }
    }
    { const char* dd = getenv("RT_AO_DUMP");       /* diagnostic: every program the module sees, with its class */
      if (dd && *dd) {
          char path[512]; snprintf(path, sizeof path, "%s/%016llx_%s.hlsl", dd, (unsigned long long)id, v->is_static ? "static" : v->is_dyn ? "dyn" : "other");
          FILE* f = fopen(path, "wb"); if (f) { fputs(hlsl, f); fclose(f); } } }
    if (v->is_static || v->is_dyn) {
        v->hlsl = (char*)malloc(strlen(hlsl) + 1);
        if (v->hlsl) strcpy(v->hlsl, hlsl); else v->is_static = v->is_dyn = 0;
    }
    return v->is_static | (v->is_dyn << 1);
}

/* ======================================================= 3. instances, queue, scene state */

typedef enum { IS_EMPTY = 0, IS_CAND, IS_QUEUED, IS_READY, IS_REJECT, IS_DEAD } InstState;

typedef struct {
    u64 key;
    u32 first_frame, last_frame, seen;
    u8 state;
    u32 n_tris, page;            /* page: AS page index + 1 */
    u32 dyn_frame;               /* frame in which this pose was already queued as an animated draw */
    u8 dynclass;                 /* program with indirect constants: the instance is kept only while it is seen (it may start to move) */
    u64 blas_va, blas_bytes;
} RtInst;

#define RT_TABLE (1u << 16)
static RtInst* s_tab; static u32 s_tab_used, s_tab_dead, s_ready_n;

typedef struct {
    u64 key, vs_id;
    u32 n_verts, nattr, n_idx, frame;
    float minv[16], eref[4];
    u8* verts; u32* idx;
    float cb[514 * 4];
} RtJob;
#define RT_QMAX 256
static RtJob* s_q[RT_QMAX]; static u32 s_qn;
/* Dynamic (animated) draws: queued every frame, built into per-slot BLAS buffers, never kept in the table. */
#define RT_DQMAX 384
#define RT_DYN_MAX 1024
static RtJob* s_dq[RT_DQMAX]; static u32 s_dqn;
static u32 s_dyn_noted_frame = ~0u, s_dyn_frame_jobs, s_dyn_frame_tris;
static u32 s_static_budget = 48;   /* new static instances built per frame: adapts to the measured build time */
static u32 s_last_static_built;
static u32 s_dyn_budget = 40;      /* animated draws per frame: adapts so that the build time stays near RT_AO_BUILD_MS */ static size_t s_dyn_frame_bytes;
static u64 s_dyn_va[RT_DYN_MAX]; static u32 s_dyn_n, s_dyn_frame = ~0u;     /* BLAS addresses of the current frame's animated draws */
static int s_prev_ok;                /* the AO history (previous camera and textures) is usable */

static u32 s_frame = ~0u, s_frame_draws, s_frame_hits, s_frame_jobs, s_lowhit_frames;
static size_t s_frame_bytes;
static u32 s_scene_gen = 1;
static u32 s_color_h, s_depth_h, s_injected_frame = ~0u, s_sampled_frame = ~0u;
static float s_cam[12 * 4];          /* c[7..11] then the viewport scale/offset block */
static int s_cam_ok;
static u32 s_ao_jobs;
static struct { u64 notes, cands, queued, rejected, evicted, resets, dyn_noted, dyn_built, dyn_rej, dyn_cam_diff, dyn_tris; u32 static_before_inject, static_after_inject; } s_st;
static int s_dxr_ok;                 /* set by rsx_rt_d3d12_init */

static void as_free_inst(RtInst* e);
static void inst_kill_all(void);

/* Word-wise FNV-style mix: every length used here is a multiple of 8 bytes (this runs once per static draw). */
static u64 fnv(u64 h, const void* p, size_t n)
{
    const u64* w = (const u64*)p;
    for (size_t i = 0; i < n / 8; i++) { u64 x; memcpy(&x, w + i, 8); h = (h ^ x) * 1099511628211ull; h ^= h >> 31; }
    return h;
}

static RtInst* tab_find(u64 key, int insert)
{
    if (!s_tab) {
        s_tab = (RtInst*)calloc(RT_TABLE, sizeof *s_tab);
        if (!s_tab) return NULL;
    }
    u32 i = (u32)(key ^ (key >> 29)) & (RT_TABLE - 1u);
    RtInst* tomb = NULL;
    for (u32 n = 0; n < RT_TABLE; n++, i = (i + 1) & (RT_TABLE - 1u)) {
        RtInst* e = &s_tab[i];
        if (e->state == IS_EMPTY) {
            if (!insert) return NULL;
            if (s_tab_used + s_tab_dead > RT_TABLE * 3 / 4) return NULL;
            if (tomb) { s_tab_dead--; e = tomb; } else s_tab_used++;
            memset(e, 0, sizeof *e);
            e->key = key; e->state = IS_CAND;
            return e;
        }
        if (e->state == IS_DEAD) { if (!tomb) tomb = e; continue; }
        if (e->key == key) return e;
    }
    return NULL;
}

static void tab_rehash(void)
{
    RtInst* old = s_tab;
    s_tab = NULL; s_tab_used = s_tab_dead = 0;
    if (!old) return;
    RtInst* nt = (RtInst*)calloc(RT_TABLE, sizeof *nt);
    if (!nt) { s_tab = old; return; }
    s_tab = nt;
    for (u32 i = 0; i < RT_TABLE; i++) {
        if (old[i].state == IS_EMPTY || old[i].state == IS_DEAD) continue;
        RtInst* e = tab_find(old[i].key, 1);
        if (e) *e = old[i];
    }
    free(old);
}

static int inv4(const double* m, double* out)
{
    double a[4][8];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { a[i][j] = m[i * 4 + j]; a[i][4 + j] = i == j; }
    for (int c = 0; c < 4; c++) {
        int p = c;
        for (int r = c + 1; r < 4; r++) if (fabs(a[r][c]) > fabs(a[p][c])) p = r;
        if (fabs(a[p][c]) < 1e-30) return 0;
        if (p != c) for (int j = 0; j < 8; j++) { double t = a[c][j]; a[c][j] = a[p][j]; a[p][j] = t; }
        const double d = 1.0 / a[c][c];
        for (int j = 0; j < 8; j++) a[c][j] *= d;
        for (int r = 0; r < 4; r++) if (r != c) {
            const double f = a[r][c];
            if (f != 0.0) for (int j = 0; j < 8; j++) a[r][j] -= f * a[c][j];
        }
    }
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) out[i * 4 + j] = a[i][4 + j];
    return 1;
}

/* clip = M * (world, 1) with M's rows c[8..11]; made relative to E = c[7].xyz so the
 * inverse keeps its precision far from the origin. */
static int cam_inverse(const float* c7_11, float* minv, float* eref)
{
    const float* e = c7_11;               /* c[7] */
    double m[16];
    for (int i = 0; i < 4; i++) {
        const float* r = c7_11 + 4 * (1 + i);      /* c[8+i] */
        m[i * 4 + 0] = r[0]; m[i * 4 + 1] = r[1]; m[i * 4 + 2] = r[2];
        m[i * 4 + 3] = (double)r[0] * e[0] + (double)r[1] * e[1] + (double)r[2] * e[2] + r[3];
    }
    double iv[16];
    if (!inv4(m, iv)) return 0;
    for (int i = 0; i < 16; i++) minv[i] = (float)iv[i];
    eref[0] = e[0]; eref[1] = e[1]; eref[2] = e[2]; eref[3] = 0;
    return 1;
}

/* The forward matrix (clip = M * (P, 1), P relative to E) that cam_inverse inverts. */
static void cam_matrix(const float* c7_11, float* mf)
{
    const float* e = c7_11;
    for (int i = 0; i < 4; i++) {
        const float* r = c7_11 + 4 * (1 + i);
        mf[i * 4 + 0] = r[0]; mf[i * 4 + 1] = r[1]; mf[i * 4 + 2] = r[2];
        mf[i * 4 + 3] = (float)((double)r[0] * e[0] + (double)r[1] * e[1] + (double)r[2] * e[2] + r[3]);
    }
}

static void cov_log(u32 frame);
static void new_frame(u32 frame)
{
    /* Level change: the instances that were ready are no longer drawn. */
    if (s_frame != ~0u) {
        if (s_ready_n >= 100 && s_frame_draws >= 100 && s_frame_hits == 0) s_lowhit_frames++;
        else s_lowhit_frames = 0;
        if (s_lowhit_frames >= 3) {
            fprintf(stderr, "[rt] scene reset (level change?): %u ready, last frame %u static draws, %u known\n",
                    s_ready_n, s_frame_draws, s_frame_hits);
            inst_kill_all();
            s_st.resets++;
            s_lowhit_frames = 0;
        }
    }
    s_frame = frame;
    s_frame_draws = s_frame_hits = s_frame_jobs = 0; s_frame_bytes = 0;
    if ((frame % 30u) == 0 && s_tab) {      /* a still character that starts to move leaves a ghost pose: drop unseen ones quickly */
        for (u32 i = 0; i < RT_TABLE; i++) {
            RtInst* e = &s_tab[i];
            if (e->state == IS_READY && e->dynclass && frame - e->last_frame > 30u) {
                as_free_inst(e); s_ready_n--; s_scene_gen++; s_st.evicted++;
                e->state = IS_DEAD; s_tab_dead++;
            }
        }
    }
    if ((frame % 120u) == 0 && s_tab) {
        for (u32 i = 0; i < RT_TABLE; i++) {
            RtInst* e = &s_tab[i];
            if (e->state == IS_EMPTY || e->state == IS_DEAD) continue;
            const u32 age = frame - e->last_frame;
            if ((e->state == IS_READY && age > 1800u) || (e->state != IS_READY && age > 90u)) {
                if (e->state == IS_READY) { as_free_inst(e); s_ready_n--; s_scene_gen++; s_st.evicted++; }
                e->state = IS_DEAD; s_tab_dead++;
            }
        }
        if (s_tab_dead > RT_TABLE / 8) tab_rehash();
    }
    if ((frame % 300u) == 0) cov_log(frame);
    if ((frame % 600u) == 0)
        fprintf(stderr, "[rt] frame %u: %u ready instances, %u queued, %llu notes (%llu cand, %llu rejected, %llu evicted), inject: %u static before / %u after\n",
                frame, s_ready_n, s_qn, (unsigned long long)s_st.notes, (unsigned long long)s_st.cands,
                (unsigned long long)s_st.rejected, (unsigned long long)s_st.evicted, s_st.static_before_inject, s_st.static_after_inject);
}

/* RT_AO_CVAR=1: which vertex constants change from frame to frame for the same mesh (diagnostic). */
static void cvar_note(const RsxRtDraw* d)
{
    enum { N = 4096, NC = 64 };
    static struct { u64 key; u32 frame; float c[NC * 4]; } *ring;
    static u32 changed[NC], obs[NC], per_vs_obs, tot;
    if (!ring) ring = calloc(N, sizeof *ring);
    if (!ring) return;
    const u64 key = d->vc_key ^ (d->vc_content * 31u);
    const u32 i = (u32)(key ^ (key >> 20)) & (N - 1u);
    if (ring[i].key == key && ring[i].frame != d->frame) {
        tot++;
        for (u32 k = 0; k < NC; k++) {
            obs[k]++;
            if (memcmp(ring[i].c + k * 4, d->cb + k * 4, 16)) changed[k]++;
        }
    }
    ring[i].key = key; ring[i].frame = d->frame;
    memcpy(ring[i].c, d->cb, sizeof ring[i].c);
    (void)per_vs_obs;
    if (tot && (tot % 20000u) == 0) {
        fprintf(stderr, "[rt-cvar] after %u repeat sightings, constants that changed (of 0..63): ", tot);
        for (u32 k = 0; k < NC; k++) if (obs[k] && changed[k] * 100u / obs[k] >= 2u) fprintf(stderr, "c%u:%u%% ", k, changed[k] * 100u / obs[k]);
        fputc('\n', stderr);
        tot++;
    }
}

/* An opaque animated draw (skinned character, prop with indirect constants): copy it for this frame's
 * dynamic acceleration structure. Its camera is the draw's own constants c7..c11. */
static void dyn_note(const RsxRtDraw* d)
{
    if (d->frame != s_dyn_noted_frame) { s_dyn_noted_frame = d->frame; s_dyn_frame_jobs = 0; s_dyn_frame_bytes = 0; s_dyn_frame_tris = 0; }
    s_st.dyn_noted++;
    if (s_cam_ok && memcmp(s_cam, d->cb + 7 * 4, 5 * 4 * sizeof(float))) s_st.dyn_cam_diff++;
    const size_t nidx = d->indices ? d->n_draw - d->n_draw % 3u : 0;
    const size_t bytes = (size_t)d->n_verts * d->stride + nidx * 4u;
    const u32 nattr = d->stride / 16u;
    if (nattr * 16u != d->stride || nattr == 0 || nattr > RT_MAX_ATTR || (!d->indices && d->n_draw < 3) || bytes > (8u << 20)) { s_st.dyn_rej++; return; }
    if (s_dqn >= RT_DQMAX || s_dyn_frame_jobs >= s_dyn_budget || s_dyn_frame_tris + d->n_draw / 3u > s_dyn_budget * 800u || s_dyn_frame_bytes + bytes > (12u << 20)) { s_st.dyn_rej++; return; }
    RtJob* j = (RtJob*)malloc(sizeof *j);
    if (!j) return;
    memset(j, 0, offsetof(RtJob, cb));
    if (!cam_inverse(d->cb + 7 * 4, j->minv, j->eref)) { free(j); s_st.dyn_rej++; return; }
    j->verts = (u8*)malloc((size_t)d->n_verts * d->stride);
    j->idx = nidx ? (u32*)malloc(nidx * 4u) : NULL;
    if (!j->verts || (nidx && !j->idx)) { free(j->verts); free(j->idx); free(j); return; }
    memcpy(j->verts, d->verts, (size_t)d->n_verts * d->stride);
    if (nidx) memcpy(j->idx, d->indices, nidx * 4u);
    memcpy(j->cb, d->cb, sizeof j->cb);
    j->vs_id = d->vs_id; j->n_verts = d->n_verts; j->nattr = nattr; j->frame = d->frame;
    j->n_idx = nidx ? (u32)nidx : (d->n_draw - d->n_draw % 3u);
    s_dq[s_dqn++] = j;
    s_dyn_frame_jobs++; s_dyn_frame_bytes += bytes; s_dyn_frame_tris += d->n_draw / 3u;
}

uint32_t rsx_rt_scene_color(void) { return s_color_h; }

void rsx_rt_note_draw(const RsxRtDraw* d)
{
    { static int cv = -1; if (cv < 0) { const char* e = getenv("RT_AO_CVAR"); cv = e && *e == '1'; }
      if (cv && g_rsx_rt_on && d->cb && d->verts) cvar_note(d); }
    if (!g_rsx_rt_on || !s_dxr_ok || !d->verts || !d->n_verts || !d->stride || !d->cb) return;
    if (d->frame != s_frame) new_frame(d->frame);
    s_st.notes++;
    s_frame_draws++;
    if (s_injected_frame == d->frame) s_st.static_after_inject++;
    s_color_h = d->color_h; s_depth_h = d->depth_h;
    RtVs* vs = s_vs_cache[(u32)(d->vs_id ^ (d->vs_id >> 17)) & 255u];
    if (!vs || vs->id != d->vs_id) { vs = vs_find(d->vs_id); s_vs_cache[(u32)(d->vs_id ^ (d->vs_id >> 17)) & 255u] = vs; }
    if (!vs) return;
    if (!vs->is_dyn) {
        memcpy(s_cam, d->cb + 7 * 4, 5 * 4 * sizeof(float));
        memcpy(s_cam + 5 * 4, d->cb + 512 * 4, 2 * 4 * sizeof(float));
        s_cam_ok = 1;
    } else if (!s_cam_ok) return;
    u64 k = 1469598103934665603ull;
    k = fnv(k, &d->vc_key, 8); k = fnv(k, &d->vc_content, 8); k = fnv(k, &d->vs_id, 8);
    for (u32 i = 0; i < vs->nuc; i++) k = fnv(k, d->cb + (size_t)vs->uc[i] * 4, 16);   /* the constants it reads, but the camera's */
    int mesh_known = 0;
    if (vs->is_dyn) {
        /* Indirect constants (a palette of matrices from c31 on): the pose is the constants 31..126. A pose seen in two frames is a
         * still object (a prop batch, a character standing still): it becomes a normal static instance. A pose seen once, of a
         * mesh drawn in an earlier frame, is moving: it gets a throw-away acceleration structure this frame. */
        k = fnv(k, d->cb + 31 * 4, 96 * 16);
        static struct { u64 key; u32 last, known; } mesh[16384];
        const u64 mk = d->vc_key ^ (d->vc_content * 0x9E3779B97F4A7C15ull) ^ d->vs_id;
        const u32 mi = (u32)(mk ^ (mk >> 31)) & 16383u;
        if (mesh[mi].key != mk) { mesh[mi].key = mk; mesh[mi].last = 0; mesh[mi].known = 0; }
        if (mesh[mi].last != d->frame) { mesh[mi].known = mesh[mi].last && d->frame - mesh[mi].last <= 8u; mesh[mi].last = d->frame; }
        mesh_known = (int)mesh[mi].known;
    }
    RtInst* e = tab_find(k, 1);
    if (!e) return;
    if (e->state == IS_CAND && e->seen == 0) s_st.cands++;
    if (e->state == IS_READY || e->state == IS_QUEUED) s_frame_hits++;
    if (e->last_frame != d->frame || e->seen == 0) {
        if (e->seen == 0) e->first_frame = d->frame;
        e->seen++;
        e->last_frame = d->frame;
    }
    if (vs->is_dyn && e->state == IS_CAND && e->seen < 2) {
        if (g_rsx_rt.dyn && mesh_known && !d->alpha_test && e->dyn_frame != d->frame) { e->dyn_frame = d->frame; dyn_note(d); }
        return;
    }
    if (e->state != IS_CAND || e->seen < 2) return;
    e->dynclass = (u8)vs->is_dyn;
    /* Seen in two frames with the same constants: it is static. Copy it, within the
     * frame's budget (the vertex cache entry may be gone by the time the GPU gets to it). */
    const size_t nidx = d->indices ? d->n_draw - d->n_draw % 3u : 0;
    const size_t bytes = (size_t)d->n_verts * d->stride + nidx * 4u;
    const u32 nattr = d->stride / 16u;
    if (nattr * 16u != d->stride || nattr == 0 || nattr > RT_MAX_ATTR || (!d->indices && d->n_draw < 3) || bytes > (40u << 20)) {
        e->state = IS_REJECT; s_st.rejected++; return;
    }
    if (s_qn >= RT_QMAX || s_frame_jobs >= s_static_budget || s_frame_bytes + bytes > (24u << 20)) return;   /* next frame */
    RtJob* j = (RtJob*)malloc(sizeof *j);
    if (!j) return;
    memset(j, 0, offsetof(RtJob, cb));
    if (!cam_inverse(d->cb + 7 * 4, j->minv, j->eref)) { free(j); e->state = IS_REJECT; s_st.rejected++; return; }
    j->verts = (u8*)malloc((size_t)d->n_verts * d->stride);
    j->idx = nidx ? (u32*)malloc(nidx * 4u) : NULL;
    if (!j->verts || (nidx && !j->idx)) { free(j->verts); free(j->idx); free(j); return; }
    memcpy(j->verts, d->verts, (size_t)d->n_verts * d->stride);
    if (nidx) memcpy(j->idx, d->indices, nidx * 4u);
    memcpy(j->cb, d->cb, sizeof j->cb);
    j->key = k; j->vs_id = d->vs_id; j->n_verts = d->n_verts; j->nattr = nattr;
    j->n_idx = nidx ? (u32)nidx : (d->n_draw - d->n_draw % 3u);
    s_q[s_qn++] = j;
    e->state = IS_QUEUED;
    s_frame_jobs++; s_frame_bytes += bytes; s_st.queued++;
}

static struct { u64 draws, tris; } s_cov[9];
void rsx_rt_cov(int why, u32 tris)
{
    if (why < 0 || why > 8) return;
    s_cov[why].draws++; s_cov[why].tris += tris;
}
static void cov_log(u32 frame)
{
    static const char* const nm[9] = { "fed", "no-vcache", "other-program", "no-zwrite(colour)", "prepass(no colour,no zwrite)", "blended", "no-ztest", "not-triangles", "no-cpu-copy" };
    fprintf(stderr, "[rt-cov] frame %u, draws into the scene target since the last line (tris):", frame);
    for (int i = 0; i < 9; i++) { fprintf(stderr, " %s %llu(%.0fk)", nm[i], (unsigned long long)s_cov[i].draws, (double)s_cov[i].tris / 1000.0); s_cov[i].draws = s_cov[i].tris = 0; }
    fputc('\n', stderr);
}

void rsx_rt_note_sampled(u32 surface_h, u32 frame)
{
    if (surface_h && surface_h == s_color_h && frame == s_frame) s_sampled_frame = frame;
}

int rsx_rt_should_inject(u32 frame, u32 target_h)
{
    if (!g_rsx_rt_on || !s_dxr_ok || !s_cam_ok) return 0;
    if (s_injected_frame == frame || frame != s_frame || s_sampled_frame != frame) return 0;
    if (target_h == s_color_h || s_frame_draws < (u32)g_rsx_rt.min_draws) return 0;
    s_injected_frame = frame;
    s_st.static_before_inject = s_frame_draws;
    s_st.static_after_inject = 0;
    return 1;
}

uint32_t rsx_rt_ao_job(uint32_t* color_h, uint32_t* depth_h)
{
    *color_h = s_color_h; *depth_h = s_depth_h;
    return ++s_ao_jobs;          /* the camera of the moment is read by the backend through rsx_rt_d3d12_ao */
}

/* Camera frozen at the injection (the backend encodes later, possibly after more draws). */
static float s_job_cam[12 * 4]; static u32 s_job_id; static int s_job_cam_ok;
static u32 s_job_frame;
void rsx_rt_freeze_camera(u32 job)
{
    memcpy(s_job_cam, s_cam, sizeof s_cam);
    s_job_id = job; s_job_cam_ok = 1; s_job_frame = s_frame;
}

/* RT_AO_TRACE: S static (vertex-cache hit) / s static (no entry yet) / O other opaque / B blended / N no depth write,
 * as run lengths, plus per-program draw and triangle totals of that frame; '|' marks the AO injection. */
void rsx_rt_trace(uint32_t frame, char cat, uint64_t vs_id, uint32_t tris, int alpha_test)
{
    static u32 fr = ~0u, nlog, nrun; static char last; static u32 run; static char line[4096]; static size_t ll;
    static struct { u64 id; u32 draws, tris; char c; u8 at; } pv[64]; static u32 npv;
    if (!g_rsx_rt.trace) return;
    if (frame != fr) {
        if (fr != ~0u && ll) {
            if (run) ll += (size_t)snprintf(line + ll, sizeof line - ll, "%c%u ", last, run);
            fprintf(stderr, "[rt-trace] frame %u: %s\n", fr, line);
            for (u32 i = 0; i < npv; i++)
                fprintf(stderr, "[rt-trace]   vs %016llx %c%s: %u draws, %u tris\n", (unsigned long long)pv[i].id, pv[i].c, pv[i].at ? " alpha-test" : "", pv[i].draws, pv[i].tris);
        }
        ll = 0; run = 0; last = 0; npv = 0; line[0] = 0;
        fr = frame;
        nlog = (frame % (u32)g_rsx_rt.trace) == 0;
    }
    if (!nlog) return;
    if (s_injected_frame == frame && last != '|' && ll < sizeof line - 40) {
        if (run) ll += (size_t)snprintf(line + ll, sizeof line - ll, "%c%u ", last, run);
        ll += (size_t)snprintf(line + ll, sizeof line - ll, "| "); last = '|'; run = 0;
    }
    if (cat == last) run++;
    else {
        if (run && last != '|' && ll < sizeof line - 40) ll += (size_t)snprintf(line + ll, sizeof line - ll, "%c%u ", last, run);
        last = cat; run = 1;
    }
    u32 i = 0;
    for (; i < npv; i++) if (pv[i].id == vs_id && pv[i].c == cat) break;
    if (i == npv && npv < 64) { pv[i].id = vs_id; pv[i].c = cat; pv[i].draws = pv[i].tris = 0; pv[i].at = (u8)alpha_test; npv++; }
    if (i < npv) { pv[i].draws++; pv[i].tris += tris; }
    (void)nrun;
}

static void job_free(RtJob* j) { if (j) { free(j->verts); free(j->idx); free(j); } }

/* ============================================================================= 4. D3D12 */

typedef struct { ID3D12Resource* res; u64 retire_seq; } Grave;
#define RT_GRAVES 4096
static Grave s_grave[RT_GRAVES]; static u32 s_grave_n;

static ID3D12Device* s_dev; static ID3D12Device5* s_dev5;
static ID3D12CommandQueue* s_queue;
static u64 s_seq_now;

typedef struct { ID3D12Resource* res; u64 va, size, used; u32 live; int full; } AsPage;
#define RT_PAGES 1024
static AsPage s_page[RT_PAGES]; static u32 s_page_n, s_page_cur;

static ID3D12RootSignature* s_rs_ao; static ID3D12RootSignature* s_rs_vs; static ID3D12RootSignature* s_rs_comp;
static ID3D12PipelineState *s_pso_trace, *s_pso_temporal, *s_pso_blur, *s_pso_up;
static ID3D12DescriptorHeap* s_cpu_heap; static u32 s_cpu_step;
enum { D_TLAS0 = 0, D_DTLAS0 = 4, D_RAW_SRV = 8, D_RAW_UAV, D_TMP_SRV, D_TMP_UAV, D_OUT_SRV, D_OUT_UAV,
       D_H0_SRV, D_H0_UAV, D_H1_SRV, D_H1_UAV, D_NDESC };
static ID3D12Resource *s_raw, *s_tmp, *s_out, *s_hist[2];       /* hist: RGBA16F (ao, distance, count), ping-pong */
static D3D12_RESOURCE_STATES s_raw_st, s_tmp_st, s_out_st, s_hist_st[2];
static u32 s_hist_w;                                             /* the history texture the next AO writes */
static float s_prev_m[16], s_prev_e[4]; static u32 s_prev_frame, s_prev_w, s_prev_h;
static ID3D12Resource* s_dyn_blas[4]; static const u64 s_dyn_cap = 24ull << 20;
static u8 s_ts_begun[4];
static u32 s_ao_w, s_ao_h, s_full_w, s_full_h;

static ID3D12Resource* s_up_arena[4]; static u8* s_up_map[4]; static u64 s_up_cap = 64u << 20;
static ID3D12Resource *s_pos_arena, *s_scr_arena; static u64 s_pos_cap = 32u << 20, s_scr_cap = 96u << 20;
static D3D12_RESOURCE_STATES s_pos_st;

typedef struct { ID3D12Resource *inst, *res, *scratch; u8* inst_map; u64 inst_cap, res_cap, scratch_cap, res_va; u32 gen, n; } Tlas;
static Tlas s_tlas[4], s_dtlas[4];

static ID3D12QueryHeap* s_qheap; static ID3D12Resource* s_qread; static u64 s_ts_freq;
static struct { double ms; u32 n; u32 valid[4]; double bms; u32 bn; u32 bvalid[4]; double bmax; double pass[5]; } s_ts;   /* pass: trace, temporal, blur, upsample, composite */   /* AO passes; BLAS/vertex builds */

static void grave_add(ID3D12Resource* r)
{
    if (!r) return;
    if (s_grave_n >= RT_GRAVES) { CALL0(r, Release); return; }   /* cannot happen in practice */
    s_grave[s_grave_n].res = r; s_grave[s_grave_n].retire_seq = s_seq_now; s_grave_n++;
}
static void grave_sweep(u64 seq)
{
    for (u32 i = 0; i < s_grave_n;) {
        if (seq > s_grave[i].retire_seq + 8) { CALL0(s_grave[i].res, Release); s_grave[i] = s_grave[--s_grave_n]; }
        else i++;
    }
}

static ID3D12Resource* mk_buffer(D3D12_HEAP_TYPE type, u64 bytes, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES st)
{
    D3D12_HEAP_PROPERTIES hp; memset(&hp, 0, sizeof hp); hp.Type = type;
    D3D12_RESOURCE_DESC bd; memset(&bd, 0, sizeof bd);
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = bytes; bd.Height = 1;
    bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; bd.Flags = flags;
    ID3D12Resource* r = NULL;
    if (FAILED(CALL(s_dev, CreateCommittedResource, &hp, D3D12_HEAP_FLAG_NONE, &bd, st, NULL, &IID_ID3D12Resource, (void**)&r)))
        return NULL;
    return r;
}

static void uav_barrier(ID3D12GraphicsCommandList* l, ID3D12Resource* r)
{
    D3D12_RESOURCE_BARRIER b; memset(&b, 0, sizeof b);
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; b.UAV.pResource = r;
    CALL(l, ResourceBarrier, 1, &b);
}
static void trans(ID3D12GraphicsCommandList* l, ID3D12Resource* r, D3D12_RESOURCE_STATES* st, D3D12_RESOURCE_STATES to)
{
    if (*st == to) return;
    D3D12_RESOURCE_BARRIER b; memset(&b, 0, sizeof b);
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r; b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = *st; b.Transition.StateAfter = to;
    CALL(l, ResourceBarrier, 1, &b);
    *st = to;
}

static D3D12_CPU_DESCRIPTOR_HANDLE cpu_desc(u32 i)
{
    D3D12_CPU_DESCRIPTOR_HANDLE h;
    CALL(s_cpu_heap, GetCPUDescriptorHandleForHeapStart, &h);
    h.ptr += (SIZE_T)i * s_cpu_step;
    return h;
}

/* ---- BLAS pages --------------------------------------------------------------- */

static int as_alloc(u64 bytes, u32* page, u64* off, u64* va)
{
    bytes = (bytes + 255u) & ~255ull;
    if (s_page_n && !s_page[s_page_cur].full && s_page[s_page_cur].used + bytes <= s_page[s_page_cur].size) {
        AsPage* p = &s_page[s_page_cur];
        *page = s_page_cur + 1; *off = p->used; *va = p->va + p->used; p->used += bytes; p->live++;
        return 1;
    }
    /* a page that is empty and not the current one has been put in the graveyard already */
    u32 slot = 0; int found = 0;
    for (u32 i = 0; i < s_page_n; i++) if (!s_page[i].res) { slot = i; found = 1; break; }
    if (!found) { if (s_page_n >= RT_PAGES) return 0; slot = s_page_n++; }
    const u64 size = bytes > (48ull << 20) ? bytes : (48ull << 20);
    ID3D12Resource* r = mk_buffer(D3D12_HEAP_TYPE_DEFAULT, size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
    if (!r) return 0;
    AsPage* p = &s_page[slot];
    if (s_page_n > 1 && s_page[s_page_cur].res) s_page[s_page_cur].full = 1;
    memset(p, 0, sizeof *p);
    p->res = r; p->va = CALL0(r, GetGPUVirtualAddress); p->size = size; p->used = bytes; p->live = 1;
    s_page_cur = slot;
    *page = slot + 1; *off = 0; *va = p->va;
    return 1;
}
static void as_free_inst(RtInst* e)
{
    if (e->page && e->page <= s_page_n) {
        AsPage* p = &s_page[e->page - 1];
        if (p->res && p->live) p->live--;
        if (p->res && p->live == 0 && (e->page - 1) != s_page_cur) { grave_add(p->res); memset(p, 0, sizeof *p); }
        else if (p->res && p->live == 0) { p->used = 0; }
    }
    e->page = 0; e->blas_va = 0; e->blas_bytes = 0;
}
static void inst_kill_all(void)
{
    if (s_tab)
        for (u32 i = 0; i < RT_TABLE; i++) {
            RtInst* e = &s_tab[i];
            if (e->state == IS_EMPTY || e->state == IS_DEAD) continue;
            if (e->state == IS_READY) as_free_inst(e);
            e->state = IS_DEAD; s_tab_dead++;
        }
    s_ready_n = 0; s_scene_gen++;
    for (u32 i = 0; i < s_qn; i++) job_free(s_q[i]);
    s_qn = 0;
    s_prev_ok = 0;
}

/* ---- init --------------------------------------------------------------------- */

static ID3DBlob* compile(const char* src, size_t len, const char* entry, const char* target, const char* what)
{
    ID3DBlob *b = NULL, *err = NULL;
    HRESULT hr = D3DCompile(src, len, what, NULL, NULL, entry, target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &b, &err);
    if (FAILED(hr)) {
        static int n; if (n++ < 6)
            fprintf(stderr, "[rt] %s compile failed: %.600s\n", what, err ? (const char*)CALL0(err, GetBufferPointer) : "?");
        RELEASE(err); return NULL;
    }
    RELEASE(err);
    return b;
}

static ID3D12RootSignature* mk_rs(const D3D12_ROOT_PARAMETER* p, u32 n)
{
    D3D12_ROOT_SIGNATURE_DESC d; memset(&d, 0, sizeof d);
    d.NumParameters = n; d.pParameters = p;
    ID3DBlob *blob = NULL, *err = NULL;
    if (FAILED(D3D12SerializeRootSignature(&d, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, &err))) {
        fprintf(stderr, "[rt] root signature: %s\n", err ? (const char*)CALL0(err, GetBufferPointer) : "?");
        RELEASE(err); return NULL;
    }
    ID3D12RootSignature* rs = NULL;
    CALL(s_dev, CreateRootSignature, 0, CALL0(blob, GetBufferPointer), CALL0(blob, GetBufferSize), &IID_ID3D12RootSignature, (void**)&rs);
    RELEASE(blob); RELEASE(err);
    return rs;
}

static ID3D12PipelineState* mk_cs(ID3D12RootSignature* rs, const void* code, size_t n, const char* what)
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC d; memset(&d, 0, sizeof d);
    d.pRootSignature = rs; d.CS.pShaderBytecode = code; d.CS.BytecodeLength = n;
    ID3D12PipelineState* pso = NULL;
    HRESULT hr = CALL(s_dev, CreateComputePipelineState, &d, &IID_ID3D12PipelineState, (void**)&pso);
    if (FAILED(hr)) { fprintf(stderr, "[rt] %s pipeline failed: 0x%08lX\n", what, (long)hr); return NULL; }
    return pso;
}

int rsx_rt_d3d12_init(ID3D12Device* dev, ID3D12CommandQueue* queue)
{
    s_dev = dev; s_queue = queue;
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5; memset(&o5, 0, sizeof o5);
    const HRESULT h5 = CALL(dev, CheckFeatureSupport, D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof o5);
    D3D12_FEATURE_DATA_SHADER_MODEL sm; memset(&sm, 0, sizeof sm); sm.HighestShaderModel = D3D_SHADER_MODEL_6_5;
    const HRESULT hs = CALL(dev, CheckFeatureSupport, D3D12_FEATURE_SHADER_MODEL, &sm, sizeof sm);
    const int tier11 = SUCCEEDED(h5) && o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1;
    const int sm65 = SUCCEEDED(hs) && sm.HighestShaderModel >= D3D_SHADER_MODEL_6_5;
    ID3D12Device5* d5 = NULL;
    const int has5 = SUCCEEDED(CALL(dev, QueryInterface, &IID_ID3D12Device5, (void**)&d5)) && d5;
    fprintf(stderr, "[rt] DXR probe: raytracing tier %s (code %d), shader model 6.5 %s, ID3D12Device5 %s -> RT_AO %s\n",
            tier11 ? "1.1" : (SUCCEEDED(h5) && o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0 ? "1.0 only" : "not supported"),
            SUCCEEDED(h5) ? (int)o5.RaytracingTier : -1, sm65 ? "yes" : "no", has5 ? "yes" : "no",
            !g_rsx_rt_on ? "off (not requested)" : (tier11 && sm65 && has5) ? "available" : "UNAVAILABLE");
    if (!g_rsx_rt_on || !tier11 || !sm65 || !has5) { RELEASE(d5); g_rsx_rt_on = 0; return 0; }
    s_dev5 = d5;

    D3D12_DESCRIPTOR_RANGE r[2]; memset(r, 0, sizeof r);
    r[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; r[0].NumDescriptors = 5; r[0].OffsetInDescriptorsFromTableStart = 0;
    r[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; r[1].NumDescriptors = 2; r[1].OffsetInDescriptorsFromTableStart = 5;
    D3D12_ROOT_PARAMETER p[4]; memset(p, 0, sizeof p);
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; p[0].Constants.Num32BitValues = 60;
    p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[1].DescriptorTable.NumDescriptorRanges = 2; p[1].DescriptorTable.pDescriptorRanges = r;
    s_rs_ao = mk_rs(p, 2);
    /* vertex-program evaluation: b0 VS constants, b1 job constants, t0 source, u0 output */
    memset(p, 0, sizeof p);
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; p[1].Constants.ShaderRegister = 1; p[1].Constants.Num32BitValues = 24;
    p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    s_rs_vs = mk_rs(p, 4);
    /* composite: one pixel-shader SRV */
    D3D12_DESCRIPTOR_RANGE cr; memset(&cr, 0, sizeof cr); cr.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; cr.NumDescriptors = 1;
    memset(p, 0, sizeof p);
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[0].DescriptorTable.NumDescriptorRanges = 1;
    p[0].DescriptorTable.pDescriptorRanges = &cr; p[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    s_rs_comp = mk_rs(p, 1);
    s_pso_trace = s_rs_ao ? mk_cs(s_rs_ao, k_cs_trace_dxil, sizeof k_cs_trace_dxil, "trace") : NULL;
    s_pso_temporal = s_rs_ao ? mk_cs(s_rs_ao, k_cs_temporal_dxil, sizeof k_cs_temporal_dxil, "temporal") : NULL;
    s_pso_blur  = s_rs_ao ? mk_cs(s_rs_ao, k_cs_blur_h_dxil, sizeof k_cs_blur_h_dxil, "blur") : NULL;
    s_pso_up    = s_rs_ao ? mk_cs(s_rs_ao, k_cs_up_dxil, sizeof k_cs_up_dxil, "upsample") : NULL;
    D3D12_DESCRIPTOR_HEAP_DESC hd; memset(&hd, 0, sizeof hd);
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = D_NDESC;
    CALL(dev, CreateDescriptorHeap, &hd, &IID_ID3D12DescriptorHeap, (void**)&s_cpu_heap);
    s_cpu_step = CALL(dev, GetDescriptorHandleIncrementSize, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    if (!s_rs_ao || !s_rs_vs || !s_rs_comp || !s_pso_trace || !s_pso_temporal || !s_pso_blur || !s_pso_up || !s_cpu_heap) {
        fprintf(stderr, "[rt] pipeline setup failed: RT_AO disabled\n");
        g_rsx_rt_on = 0; return 0;
    }
    {   /* GPU time of the AO passes */
        D3D12_QUERY_HEAP_DESC qh; memset(&qh, 0, sizeof qh); qh.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; qh.Count = 48;
        if (SUCCEEDED(CALL(dev, CreateQueryHeap, &qh, &IID_ID3D12QueryHeap, (void**)&s_qheap)))
            s_qread = mk_buffer(D3D12_HEAP_TYPE_READBACK, 384, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!s_qread || !queue || FAILED(CALL(queue, GetTimestampFrequency, &s_ts_freq)) || !s_ts_freq) { RELEASE(s_qheap); RELEASE(s_qread); }
    }
    s_dxr_ok = 1;
    fprintf(stderr, "[rt] RT_AO ready (offline-compiled DXIL, no DXC at run time)\n");
    return 1;
}

void rsx_rt_d3d12_shutdown(void)
{
    /* The process is ending: the OS reclaims everything; just stop touching the device. */
    s_dxr_ok = 0;
}

/* ---- vertex-program evaluation kernels ------------------------------------------- */

static ID3D12PipelineState* vs_kernel(RtVs* v)
{
    if (v->cs) return v->cs;
    if (v->cs_failed || !v->hlsl) return NULL;
    v->cs_failed = 1;
    const char* needle = "VSOutput main(VSInput input)";
    const char* at = strstr(v->hlsl, needle);
    if (!at) return NULL;
    size_t cap = strlen(v->hlsl) + 4096;
    char* src = (char*)malloc(cap);
    if (!src) return NULL;
    size_t n = (size_t)(at - v->hlsl);
    memcpy(src, v->hlsl, n);
    n += (size_t)snprintf(src + n, cap - n, "VSOutput vsmain(VSInput input)");
    const char* rest = at + strlen(needle);
    size_t rl = strlen(rest);
    memcpy(src + n, rest, rl); n += rl;
    n += (size_t)snprintf(src + n, cap - n,
        "\ncbuffer Job : register(b1) { float4 jminv[4]; float4 jeref; uint4 jinfo; };\n"
        "StructuredBuffer<float4> srcv : register(t0);\n"
        "RWStructuredBuffer<float4> dstv : register(u0);\n"
        "[numthreads(64,1,1)] void cs_main(uint3 id : SV_DispatchThreadID) {\n"
        "  if (id.x >= jinfo.x) return;\n"
        "  VSInput vin;\n");
    for (u32 i = 0; i < v->nfields; i++)
        n += (size_t)snprintf(src + n, cap - n, "  vin.a%d = srcv[id.x * jinfo.y + %uu];\n", v->field_attr[i], i);
    n += (size_t)snprintf(src + n, cap - n,
        "  VSOutput o = vsmain(vin);\n"
        "  float4 c = o.pos;\n"
        "  float4 w = float4(dot(jminv[0], c), dot(jminv[1], c), dot(jminv[2], c), dot(jminv[3], c));\n"
        "  dstv[jinfo.z + id.x] = float4(jeref.xyz + w.xyz / w.w, 1.0);\n"
        "}\n");
    ID3DBlob* b = compile(src, n, "cs_main", "cs_5_0", "vs kernel");
    free(src);
    if (!b) return NULL;
    v->cs = mk_cs(s_rs_vs, CALL0(b, GetBufferPointer), CALL0(b, GetBufferSize), "vs kernel");
    RELEASE(b);
    if (v->cs) v->cs_failed = 0;
    return v->cs;
}

/* ---- build the queued instances ------------------------------------------------------- */

void rsx_rt_d3d12_process(ID3D12GraphicsCommandList* list, u32 slot, u64 seq)
{
    if (!s_dxr_ok) return;
    s_seq_now = seq;
    rsx_rt_d3d12_ao_collect(slot);
    grave_sweep(seq);
    if (!s_qn && !s_dqn) return;
    if (s_qheap) CALL(list, EndQuery, s_qheap, D3D12_QUERY_TYPE_TIMESTAMP, 8u + slot * 2u);
    ID3D12GraphicsCommandList4* l4 = NULL;
    if (FAILED(CALL(list, QueryInterface, &IID_ID3D12GraphicsCommandList4, (void**)&l4)) || !l4) return;
    if (!s_up_arena[slot]) {
        s_up_arena[slot] = mk_buffer(D3D12_HEAP_TYPE_UPLOAD, s_up_cap, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        if (s_up_arena[slot]) { D3D12_RANGE nr = {0, 0}; CALL(s_up_arena[slot], Map, 0, &nr, (void**)&s_up_map[slot]); }
    }
    if (!s_pos_arena) {
        s_pos_arena = mk_buffer(D3D12_HEAP_TYPE_DEFAULT, s_pos_cap, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        s_scr_arena = mk_buffer(D3D12_HEAP_TYPE_DEFAULT, s_scr_cap, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        s_pos_st = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    if (!s_up_arena[slot] || !s_up_map[slot] || !s_pos_arena || !s_scr_arena) {
        fprintf(stderr, "[rt] arena allocation failed: RT_AO disabled\n");
        s_dxr_ok = 0; g_rsx_rt_on = 0; RELEASE(l4); return;
    }
    const u64 up_va = CALL0(s_up_arena[slot], GetGPUVirtualAddress);
    const u64 pos_va = CALL0(s_pos_arena, GetGPUVirtualAddress);
    const u64 scr_va = CALL0(s_scr_arena, GetGPUVirtualAddress);
    u64 up_used = 0, pos_used = 0;
    u32 done = 0, built = 0;
    uav_barrier((ID3D12GraphicsCommandList*)l4, NULL);
    CALL(l4, SetComputeRootSignature, s_rs_vs);
    for (; done < s_qn; done++) {
        RtJob* j = s_q[done];
        RtInst* e = tab_find(j->key, 0);
        RtVs* v = vs_find(j->vs_id);
        if (!e || e->state != IS_QUEUED || !v) { if (e && e->state == IS_QUEUED) { e->state = IS_REJECT; s_st.rejected++; } job_free(j); continue; }
        ID3D12PipelineState* kern = vs_kernel(v);
        if (!kern || v->nfields != j->nattr) { e->state = IS_REJECT; s_st.rejected++; job_free(j); continue; }
        const u64 vb = (u64)j->n_verts * j->nattr * 16u;
        const u64 vb_a = (vb + 255u) & ~255ull, cb_a = 8704u, ib = (u64)j->n_idx * 4u, ib_a = (j->idx ? ib + 255u : 0) & ~255ull;
        const u64 pos_b = ((u64)j->n_verts * 16u + 255u) & ~255ull;
        if (up_used + vb_a + cb_a + ib_a > s_up_cap || pos_used + pos_b > s_pos_cap) {
            if (done == 0) { e->state = IS_REJECT; s_st.rejected++; job_free(j); continue; }   /* too big for any arena */
            break;                                                                             /* next frame */
        }
        /* Geometry inputs and the BLAS size, to know the scratch before committing to the job. */
        D3D12_RAYTRACING_GEOMETRY_DESC g; memset(&g, 0, sizeof g);
        g.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES; g.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        g.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        g.Triangles.VertexCount = j->n_verts;
        g.Triangles.VertexBuffer.StartAddress = pos_va + pos_used;
        g.Triangles.VertexBuffer.StrideInBytes = 16;
        if (j->idx) {
            g.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT; g.Triangles.IndexCount = j->n_idx;
            g.Triangles.IndexBuffer = up_va + up_used + vb_a + cb_a;
        }
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in; memset(&in, 0, sizeof in);
        in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        in.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY; in.NumDescs = 1; in.pGeometryDescs = &g;
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pi; memset(&pi, 0, sizeof pi);
        CALL(s_dev5, GetRaytracingAccelerationStructurePrebuildInfo, &in, &pi);
        const u64 scr_b = (pi.ScratchDataSizeInBytes + 255u) & ~255ull;
        if (!pi.ResultDataMaxSizeInBytes || scr_b > s_scr_cap) { e->state = IS_REJECT; s_st.rejected++; job_free(j); continue; }
        if (scr_b > s_scr_cap - 0) { break; }
        u32 page; u64 aoff, ava;
        if (!as_alloc(pi.ResultDataMaxSizeInBytes, &page, &aoff, &ava)) { e->state = IS_REJECT; s_st.rejected++; job_free(j); continue; }

        /* Stage: source vertices, the constants (viewport epilogue neutralised so that o[0] is the clip position), indices. */
        u8* up = s_up_map[slot] + up_used;
        memcpy(up, j->verts, (size_t)vb);
        float* cb = (float*)(up + vb_a);
        memcpy(cb, j->cb, 514 * 16);
        cb[512 * 4 + 0] = cb[512 * 4 + 1] = cb[512 * 4 + 2] = 1.0f; cb[512 * 4 + 3] = 0.0f;
        cb[513 * 4 + 0] = cb[513 * 4 + 1] = cb[513 * 4 + 2] = cb[513 * 4 + 3] = 0.0f;
        if (j->idx) memcpy(up + vb_a + cb_a, j->idx, (size_t)ib);
        struct { float minv[16]; float eref[4]; u32 info[4]; } jc;
        memcpy(jc.minv, j->minv, sizeof jc.minv); memcpy(jc.eref, j->eref, sizeof jc.eref);
        jc.info[0] = j->n_verts; jc.info[1] = j->nattr; jc.info[2] = (u32)(pos_used / 16u); jc.info[3] = 0;

        CALL(l4, SetPipelineState, kern);
        CALL(l4, SetComputeRootConstantBufferView, 0, up_va + up_used + vb_a);
        CALL(l4, SetComputeRoot32BitConstants, 1, 24, &jc, 0);
        CALL(l4, SetComputeRootShaderResourceView, 2, up_va + up_used);
        CALL(l4, SetComputeRootUnorderedAccessView, 3, pos_va);
        CALL(l4, Dispatch, (j->n_verts + 63u) / 64u, 1, 1);
        uav_barrier((ID3D12GraphicsCommandList*)l4, s_pos_arena);
        trans((ID3D12GraphicsCommandList*)l4, s_pos_arena, &s_pos_st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd; memset(&bd, 0, sizeof bd);
        bd.DestAccelerationStructureData = ava;
        bd.Inputs = in;
        bd.ScratchAccelerationStructureData = scr_va + 0;
        CALL(l4, BuildRaytracingAccelerationStructure, &bd, 0, NULL);
        uav_barrier((ID3D12GraphicsCommandList*)l4, NULL);
        trans((ID3D12GraphicsCommandList*)l4, s_pos_arena, &s_pos_st, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        /* The arenas are reused by the next job after the barriers above: the vertex output
         * is rewritten from the start, the scratch too. Uploads are not (distinct offsets). */
        up_used += vb_a + cb_a + ib_a;
        pos_used = 0;                    /* positions are consumed by the build: reuse */
        e->state = IS_READY; e->page = page; e->blas_va = ava; e->blas_bytes = pi.ResultDataMaxSizeInBytes;
        e->n_tris = j->n_idx / 3u;
        s_ready_n++; s_scene_gen++; built++;
        (void)aoff;
        job_free(j);
    }
    memmove(s_q, s_q + done, (s_qn - done) * sizeof s_q[0]);
    s_qn -= done;
    s_last_static_built = built;

    /* ---- animated draws of the frame: a throw-away BLAS each (fast build), listed for the dynamic TLAS.
     * Batched: every job has its own region of the position arena and of the scratch, so the vertex kernels
     * run together, one barrier, then all builds together, one barrier. ---- */
    if (s_dqn) {
        if (!s_dyn_blas[slot]) s_dyn_blas[slot] = mk_buffer(D3D12_HEAP_TYPE_DEFAULT, s_dyn_cap, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                                             D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
        const u64 dva = s_dyn_blas[slot] ? CALL0(s_dyn_blas[slot], GetGPUVirtualAddress) : 0;
        static D3D12_RAYTRACING_GEOMETRY_DESC dg[RT_DQMAX]; static u64 d_res[RT_DQMAX], d_scr[RT_DQMAX];
        u32 nb = 0;
        u64 dused = 0, dscr = 0, dpos = 0;
        up_used = (up_used + 255u) & ~255ull;
        for (u32 q = 0; q < s_dqn; q++) {
            RtJob* j = s_dq[q];
            RtVs* v = vs_find(j->vs_id);
            ID3D12PipelineState* kern = (v && dva) ? vs_kernel(v) : NULL;
            if (!kern || v->nfields != j->nattr) { s_st.dyn_rej++; job_free(j); continue; }
            const u64 vb = (u64)j->n_verts * j->nattr * 16u;
            const u64 vb_a = (vb + 255u) & ~255ull, cb_a = 8704u, ib = (u64)j->n_idx * 4u, ib_a = (j->idx ? ib + 255u : 0) & ~255ull;
            const u64 pos_b = ((u64)j->n_verts * 16u + 255u) & ~255ull;
            if (up_used + vb_a + cb_a + ib_a > s_up_cap || dpos + pos_b > s_pos_cap) { s_st.dyn_rej++; job_free(j); continue; }
            D3D12_RAYTRACING_GEOMETRY_DESC* g = &dg[nb]; memset(g, 0, sizeof *g);
            g->Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES; g->Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
            g->Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
            g->Triangles.VertexCount = j->n_verts;
            g->Triangles.VertexBuffer.StartAddress = pos_va + dpos;
            g->Triangles.VertexBuffer.StrideInBytes = 16;
            if (j->idx) {
                g->Triangles.IndexFormat = DXGI_FORMAT_R32_UINT; g->Triangles.IndexCount = j->n_idx;
                g->Triangles.IndexBuffer = up_va + up_used + vb_a + cb_a;
            }
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in; memset(&in, 0, sizeof in);
            in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
            in.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
            in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY; in.NumDescs = 1; in.pGeometryDescs = g;
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pi; memset(&pi, 0, sizeof pi);
            CALL(s_dev5, GetRaytracingAccelerationStructurePrebuildInfo, &in, &pi);
            const u64 res_b = (pi.ResultDataMaxSizeInBytes + 255u) & ~255ull, scr_b = (pi.ScratchDataSizeInBytes + 255u) & ~255ull;
            if (!res_b || dscr + scr_b > s_scr_cap || dused + res_b > s_dyn_cap || (s_dyn_n >= RT_DYN_MAX && j->frame == s_dyn_frame)) { s_st.dyn_rej++; job_free(j); continue; }
            u8* up = s_up_map[slot] + up_used;
            memcpy(up, j->verts, (size_t)vb);
            float* cb = (float*)(up + vb_a);
            memcpy(cb, j->cb, 514 * 16);
            cb[512 * 4 + 0] = cb[512 * 4 + 1] = cb[512 * 4 + 2] = 1.0f; cb[512 * 4 + 3] = 0.0f;
            cb[513 * 4 + 0] = cb[513 * 4 + 1] = cb[513 * 4 + 2] = cb[513 * 4 + 3] = 0.0f;
            if (j->idx) memcpy(up + vb_a + cb_a, j->idx, (size_t)ib);
            struct { float minv[16]; float eref[4]; u32 info[4]; } jc;
            memcpy(jc.minv, j->minv, sizeof jc.minv); memcpy(jc.eref, j->eref, sizeof jc.eref);
            jc.info[0] = j->n_verts; jc.info[1] = j->nattr; jc.info[2] = (u32)(dpos / 16u); jc.info[3] = 0;
            CALL(l4, SetPipelineState, kern);
            CALL(l4, SetComputeRootConstantBufferView, 0, up_va + up_used + vb_a);
            CALL(l4, SetComputeRoot32BitConstants, 1, 24, &jc, 0);
            CALL(l4, SetComputeRootShaderResourceView, 2, up_va + up_used);
            CALL(l4, SetComputeRootUnorderedAccessView, 3, pos_va);
            CALL(l4, Dispatch, (j->n_verts + 63u) / 64u, 1, 1);
            up_used += vb_a + cb_a + ib_a;
            d_res[nb] = dused; d_scr[nb] = dscr;
            if (j->frame != s_dyn_frame) { s_dyn_frame = j->frame; s_dyn_n = 0; }
            s_dyn_va[s_dyn_n++] = dva + dused;
            dpos += pos_b; dscr += scr_b; dused += res_b; nb++;
            s_st.dyn_built++; s_st.dyn_tris += j->n_idx / 3u;
            job_free(j);
        }
        if (nb) {
            uav_barrier((ID3D12GraphicsCommandList*)l4, s_pos_arena);
            trans((ID3D12GraphicsCommandList*)l4, s_pos_arena, &s_pos_st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            for (u32 i = 0; i < nb; i++) {
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd; memset(&bd, 0, sizeof bd);
                bd.DestAccelerationStructureData = dva + d_res[i];
                bd.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
                bd.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
                bd.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY; bd.Inputs.NumDescs = 1; bd.Inputs.pGeometryDescs = &dg[i];
                bd.ScratchAccelerationStructureData = scr_va + d_scr[i];
                CALL(l4, BuildRaytracingAccelerationStructure, &bd, 0, NULL);
            }
            uav_barrier((ID3D12GraphicsCommandList*)l4, NULL);
            trans((ID3D12GraphicsCommandList*)l4, s_pos_arena, &s_pos_st, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        s_dqn = 0;
    }
    if (s_qheap && s_qread) {
        CALL(list, EndQuery, s_qheap, D3D12_QUERY_TYPE_TIMESTAMP, 8u + slot * 2u + 1u);
        CALL(list, ResolveQueryData, s_qheap, D3D12_QUERY_TYPE_TIMESTAMP, 8u + slot * 2u, 2, s_qread, 64u + (UINT64)slot * 16u);
        s_ts.bvalid[slot] = 1;
    }
    RELEASE(l4);
    (void)built;
}

/* ---- TLAS ----------------------------------------------------------------------------- */

/* Build a top-level structure over the given bottom-level addresses into `t`, SRV at descriptor `desc`. */
static int tlas_build(Tlas* t, ID3D12GraphicsCommandList4* l4, const u64* vas, const u8* masks, u32 k, int fast_build, u32 desc)
{
    if (!k) { t->n = 0; return 0; }
    const u64 need_inst = (u64)k * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
    if (!t->inst || t->inst_cap < need_inst) {
        if (t->inst) { grave_add(t->inst); }
        t->inst_cap = need_inst * 3 / 2 + 4096;
        t->inst = mk_buffer(D3D12_HEAP_TYPE_UPLOAD, t->inst_cap, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        t->inst_map = NULL;
        if (t->inst) { D3D12_RANGE nr = {0, 0}; CALL(t->inst, Map, 0, &nr, (void**)&t->inst_map); }
        if (!t->inst_map) return 0;
    }
    D3D12_RAYTRACING_INSTANCE_DESC* d = (D3D12_RAYTRACING_INSTANCE_DESC*)t->inst_map;
    for (u32 i = 0; i < k; i++) {
        memset(&d[i], 0, sizeof d[i]);
        d[i].Transform[0][0] = d[i].Transform[1][1] = d[i].Transform[2][2] = 1.0f;
        d[i].InstanceMask = masks ? masks[i] : 0xFF;
        d[i].Flags = D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE | D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE;
        d[i].AccelerationStructure = vas[i];
    }
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in; memset(&in, 0, sizeof in);
    in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    in.Flags = fast_build ? D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD : D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY; in.NumDescs = k;
    in.InstanceDescs = CALL0(t->inst, GetGPUVirtualAddress);
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pi; memset(&pi, 0, sizeof pi);
    CALL(s_dev5, GetRaytracingAccelerationStructurePrebuildInfo, &in, &pi);
    if (!t->res || t->res_cap < pi.ResultDataMaxSizeInBytes) {
        grave_add(t->res);
        t->res_cap = pi.ResultDataMaxSizeInBytes * 3 / 2 + 65536;
        t->res = mk_buffer(D3D12_HEAP_TYPE_DEFAULT, t->res_cap, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
        if (!t->res) return 0;
        t->res_va = CALL0(t->res, GetGPUVirtualAddress);
    }
    if (!t->scratch || t->scratch_cap < pi.ScratchDataSizeInBytes) {
        grave_add(t->scratch);
        t->scratch_cap = pi.ScratchDataSizeInBytes * 3 / 2 + 65536;
        t->scratch = mk_buffer(D3D12_HEAP_TYPE_DEFAULT, t->scratch_cap, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (!t->scratch) return 0;
    }
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd; memset(&bd, 0, sizeof bd);
    bd.DestAccelerationStructureData = t->res_va; bd.Inputs = in;
    bd.ScratchAccelerationStructureData = CALL0(t->scratch, GetGPUVirtualAddress);
    uav_barrier((ID3D12GraphicsCommandList*)l4, NULL);          /* the BLASes built before */
    CALL(l4, BuildRaytracingAccelerationStructure, &bd, 0, NULL);
    uav_barrier((ID3D12GraphicsCommandList*)l4, NULL);
    t->n = k;
    D3D12_SHADER_RESOURCE_VIEW_DESC sv; memset(&sv, 0, sizeof sv);
    sv.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sv.RaytracingAccelerationStructure.Location = t->res_va;
    CALL(s_dev, CreateShaderResourceView, NULL, &sv, cpu_desc(desc));
    return 1;
}

static int tlas_update(ID3D12GraphicsCommandList4* l4, u32 slot)
{
    Tlas* t = &s_tlas[slot];
    if (t->res && t->gen == s_scene_gen) return t->n > 0;
    u32 n = 0;
    for (u32 i = 0; i < RT_TABLE && s_tab; i++) if (s_tab[i].state == IS_READY) n++;
    if (!n) { t->gen = s_scene_gen; t->n = 0; return 0; }
    u64* vas = (u64*)malloc((size_t)n * (sizeof *vas + 1));
    if (!vas) return 0;
    u8* masks = (u8*)(vas + n);
    u32 k = 0;
    for (u32 i = 0; i < RT_TABLE && k < n; i++) if (s_tab[i].state == IS_READY) { masks[k] = s_tab[i].dynclass ? 2 : 1; vas[k++] = s_tab[i].blas_va; }
    const int ok = tlas_build(t, l4, vas, masks, k, 0, D_TLAS0 + slot);
    free(vas);
    if (ok) t->gen = s_scene_gen;
    return ok;
}

/* ---- AO textures ------------------------------------------------------------------------ */

static ID3D12Resource* mk_tex(u32 w, u32 h, DXGI_FORMAT fmt)
{
    D3D12_HEAP_PROPERTIES hp; memset(&hp, 0, sizeof hp); hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td; memset(&td, 0, sizeof td);
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; td.Width = w; td.Height = h; td.DepthOrArraySize = 1; td.MipLevels = 1;
    td.Format = fmt; td.SampleDesc.Count = 1; td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource* r = NULL;
    CALL(s_dev, CreateCommittedResource, &hp, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, NULL, &IID_ID3D12Resource, (void**)&r);
    return r;
}
static void mk_views(ID3D12Resource* r, DXGI_FORMAT fmt, u32 srv_i, u32 uav_i)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC sv; memset(&sv, 0, sizeof sv);
    sv.Format = fmt; sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sv.Texture2D.MipLevels = 1;
    CALL(s_dev, CreateShaderResourceView, r, &sv, cpu_desc(srv_i));
    D3D12_UNORDERED_ACCESS_VIEW_DESC uv; memset(&uv, 0, sizeof uv);
    uv.Format = fmt; uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    CALL(s_dev, CreateUnorderedAccessView, r, NULL, &uv, cpu_desc(uav_i));
}
/* The traced resolution: about 1000 pixels wide whatever the render scale (3 at 4K, 2 at 1080p, 2 at native), so that the cost of the
 * rays does not grow with RSX_SCALE. RT_AO_HALFRES=0 traces at full resolution. */
static u32 ao_scale(u32 fw)
{
    if (!g_rsx_rt.halfres) return 1u;
    u32 sc = (fw + 999u) / 1000u;
    return sc < 2u ? 2u : sc > 4u ? 4u : sc;
}
static int ensure_tex(u32 fw, u32 fh)
{
    const u32 sc = ao_scale(fw);
    const u32 aw = (fw + sc - 1) / sc, ah = (fh + sc - 1) / sc;
    if (s_raw && s_ao_w == aw && s_ao_h == ah && s_full_w == fw && s_full_h == fh) return 1;
    grave_add(s_raw); grave_add(s_tmp); grave_add(s_out); grave_add(s_hist[0]); grave_add(s_hist[1]);
    s_raw = mk_tex(aw, ah, DXGI_FORMAT_R32_FLOAT); s_tmp = mk_tex(aw, ah, DXGI_FORMAT_R32_FLOAT); s_out = mk_tex(fw, fh, DXGI_FORMAT_R32_FLOAT);
    s_hist[0] = mk_tex(aw, ah, DXGI_FORMAT_R16G16B16A16_FLOAT); s_hist[1] = mk_tex(aw, ah, DXGI_FORMAT_R16G16B16A16_FLOAT);
    s_prev_ok = 0; s_hist_w = 0;
    if (!s_raw || !s_tmp || !s_out || !s_hist[0] || !s_hist[1]) return 0;
    s_ao_w = aw; s_ao_h = ah; s_full_w = fw; s_full_h = fh;
    s_raw_st = s_tmp_st = s_out_st = s_hist_st[0] = s_hist_st[1] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    mk_views(s_raw, DXGI_FORMAT_R32_FLOAT, D_RAW_SRV, D_RAW_UAV); mk_views(s_tmp, DXGI_FORMAT_R32_FLOAT, D_TMP_SRV, D_TMP_UAV);
    mk_views(s_out, DXGI_FORMAT_R32_FLOAT, D_OUT_SRV, D_OUT_UAV);
    mk_views(s_hist[0], DXGI_FORMAT_R16G16B16A16_FLOAT, D_H0_SRV, D_H0_UAV); mk_views(s_hist[1], DXGI_FORMAT_R16G16B16A16_FLOAT, D_H1_SRV, D_H1_UAV);
    fprintf(stderr, "[rt] AO targets: %ux%u traced, %ux%u output\n", aw, ah, fw, fh);
    return 1;
}

int rsx_rt_d3d12_ao(const RsxRtAoIn* in, ID3D12Resource** ao_tex, D3D12_CPU_DESCRIPTOR_HANDLE* ao_srv)
{
    if (!s_dxr_ok || !s_job_cam_ok) return 0;
    ID3D12GraphicsCommandList* list = in->list;
    ID3D12GraphicsCommandList4* l4 = NULL;
    if (FAILED(CALL(list, QueryInterface, &IID_ID3D12GraphicsCommandList4, (void**)&l4)) || !l4) return 0;
    s_seq_now = in->seq;
    int ok = 0;
    if (!ensure_tex(in->w, in->h)) goto done;
    if (!tlas_update(l4, in->slot)) goto done;
    {
        const int have_dyn = g_rsx_rt.dyn && s_dyn_n && s_dyn_frame == s_job_frame &&
                             tlas_build(&s_dtlas[in->slot], l4, s_dyn_va, NULL, s_dyn_n, 1, D_DTLAS0 + in->slot);
        struct { float minv[16]; float eye_radius[4]; float vpx[4]; float vpz[4]; u32 dims[4]; u32 cfg[4];
                 float pm[16]; float pdelta[4]; float ext[4]; } P;
        float eref[4];
        if (!cam_inverse(s_job_cam, P.minv, eref)) goto done;
        P.eye_radius[0] = eref[0]; P.eye_radius[1] = eref[1]; P.eye_radius[2] = eref[2]; P.eye_radius[3] = eff_radius();
        const float* xf = s_job_cam + 5 * 4;           /* scale xyz,_ ; offset xyz,_ */
        P.vpx[0] = xf[0]; P.vpx[1] = xf[1]; P.vpx[2] = xf[4]; P.vpx[3] = xf[5];
        P.vpz[0] = xf[2]; P.vpz[1] = xf[6]; P.vpz[2] = eff_strength(); P.vpz[3] = 0.02f;
        const u32 sc = ao_scale(s_full_w);
        P.dims[0] = s_ao_w; P.dims[1] = s_ao_h; P.dims[2] = s_full_w; P.dims[3] = s_full_h;
        P.cfg[0] = g_rsx_rt.anim ? (u32)(in->seq & 1023u) : 0u; P.cfg[1] = (u32)g_rsx_rt.rays; P.cfg[2] = sc; P.cfg[3] = (u32)eff_debug();
        float mf[16]; cam_matrix(s_job_cam, mf);
        const int use_hist = g_rsx_rt.temporal > 1 && s_prev_ok && s_prev_w == s_full_w && s_prev_h == s_full_h &&
                             s_job_frame > s_prev_frame && s_job_frame - s_prev_frame <= 3;
        if (use_hist) {
            memcpy(P.pm, s_prev_m, sizeof P.pm);
            P.pdelta[0] = eref[0] - s_prev_e[0]; P.pdelta[1] = eref[1] - s_prev_e[1]; P.pdelta[2] = eref[2] - s_prev_e[2]; P.pdelta[3] = 1.0f;
        } else { memset(P.pm, 0, sizeof P.pm); memset(P.pdelta, 0, sizeof P.pdelta); }
        P.ext[0] = eff_power(); P.ext[1] = g_rsx_rt.temporal > 1 ? 1.0f / (float)g_rsx_rt.temporal : 1.0f;
        P.ext[2] = have_dyn ? 1.0f : 0.0f; P.ext[3] = eff_char();

        const u32 wi = s_hist_w, ri = wi ^ 1u;
        const u32 hs[2] = { D_H0_SRV, D_H1_SRV }, hu[2] = { D_H0_UAV, D_H1_UAV };
        const D3D12_CPU_DESCRIPTOR_HANDLE dsrv = in->depth_srv, stlas = cpu_desc(D_TLAS0 + in->slot);
        const D3D12_CPU_DESCRIPTOR_HANDLE dtlas = have_dyn ? cpu_desc(D_DTLAS0 + in->slot) : stlas;
        /* Table layout: t0 depth, t1 static TLAS, t2 source, t3 dynamic TLAS, t4 history, u0 out, u1 history out. */
        D3D12_CPU_DESCRIPTOR_HANDLE s1[7] = { dsrv, stlas, cpu_desc(D_RAW_SRV), dtlas, cpu_desc(hs[ri]), cpu_desc(D_RAW_UAV), cpu_desc(hu[wi]) };
        D3D12_CPU_DESCRIPTOR_HANDLE s2[7] = { dsrv, dsrv, cpu_desc(D_RAW_SRV), dsrv, cpu_desc(hs[ri]), cpu_desc(D_TMP_UAV), cpu_desc(hu[wi]) };
        D3D12_CPU_DESCRIPTOR_HANDLE s3[7] = { dsrv, dsrv, cpu_desc(D_TMP_SRV), dsrv, cpu_desc(hs[ri]), cpu_desc(D_RAW_UAV), cpu_desc(hu[wi]) };
        D3D12_CPU_DESCRIPTOR_HANDLE s4[7] = { dsrv, dsrv, cpu_desc(D_RAW_SRV), dsrv, cpu_desc(hs[ri]), cpu_desc(D_OUT_UAV), cpu_desc(hu[wi]) };
        /* Tables first: a full descriptor ring flushes the list under us. */
        const D3D12_GPU_DESCRIPTOR_HANDLE t1 = in->srv_table(s1, 7);
        const D3D12_GPU_DESCRIPTOR_HANDLE t2 = in->srv_table(s2, 7);
        const D3D12_GPU_DESCRIPTOR_HANDLE t3 = in->srv_table(s3, 7);
        const D3D12_GPU_DESCRIPTOR_HANDLE t4 = in->srv_table(s4, 7);
        if (s_qheap) CALL(l4, EndQuery, s_qheap, D3D12_QUERY_TYPE_TIMESTAMP, in->slot * 2u);
        trans(list, s_raw, &s_raw_st, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        trans(list, s_tmp, &s_tmp_st, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        trans(list, s_out, &s_out_st, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        trans(list, s_hist[wi], &s_hist_st[wi], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        CALL(l4, SetComputeRootSignature, s_rs_ao);
        CALL(l4, SetComputeRoot32BitConstants, 0, 60, &P, 0);
        /* 1. trace -> raw */
        CALL(l4, SetPipelineState, s_pso_trace);
        CALL(l4, SetComputeRootDescriptorTable, 1, t1);
        CALL(l4, Dispatch, (s_ao_w + 7) / 8, (s_ao_h + 7) / 8, 1);
        uav_barrier(list, s_raw);
        if (s_qheap) CALL(l4, EndQuery, s_qheap, D3D12_QUERY_TYPE_TIMESTAMP, 16u + in->slot * 4u + 0u);
        trans(list, s_raw, &s_raw_st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        trans(list, s_hist[ri], &s_hist_st[ri], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        /* 2. temporal: raw + history -> tmp, history' */
        CALL(l4, SetPipelineState, s_pso_temporal);
        CALL(l4, SetComputeRootDescriptorTable, 1, t2);
        CALL(l4, Dispatch, (s_ao_w + 7) / 8, (s_ao_h + 7) / 8, 1);
        uav_barrier(list, s_tmp);
        if (s_qheap) CALL(l4, EndQuery, s_qheap, D3D12_QUERY_TYPE_TIMESTAMP, 16u + in->slot * 4u + 1u);
        trans(list, s_tmp, &s_tmp_st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        trans(list, s_raw, &s_raw_st, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        /* 3. depth-aware blur: tmp -> raw */
        CALL(l4, SetPipelineState, s_pso_blur);
        CALL(l4, SetComputeRootDescriptorTable, 1, t3);
        CALL(l4, Dispatch, (s_ao_w + 7) / 8, (s_ao_h + 7) / 8, 1);
        uav_barrier(list, s_raw);
        if (s_qheap) CALL(l4, EndQuery, s_qheap, D3D12_QUERY_TYPE_TIMESTAMP, 16u + in->slot * 4u + 2u);
        trans(list, s_raw, &s_raw_st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        /* 4. bilateral upsample: raw -> out */
        CALL(l4, SetPipelineState, s_pso_up);
        CALL(l4, SetComputeRootDescriptorTable, 1, t4);
        CALL(l4, Dispatch, (s_full_w + 7) / 8, (s_full_h + 7) / 8, 1);
        uav_barrier(list, s_out);
        if (s_qheap) CALL(l4, EndQuery, s_qheap, D3D12_QUERY_TYPE_TIMESTAMP, 16u + in->slot * 4u + 3u);
        trans(list, s_out, &s_out_st, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        *ao_tex = s_out; *ao_srv = cpu_desc(D_OUT_SRV);
        memcpy(s_prev_m, mf, sizeof mf); memcpy(s_prev_e, eref, sizeof eref);
        s_prev_frame = s_job_frame; s_prev_w = s_full_w; s_prev_h = s_full_h; s_prev_ok = 1; s_hist_w = ri;
        ok = 1;
    }
done:
    RELEASE(l4);
    return ok;
}

/* ---- composite --------------------------------------------------------------------------- */

static const char kCompHLSL[] =
    "Texture2D<float> ao : register(t0);\n"
    "float4 vs_main(uint vid : SV_VertexID) : SV_Position { float2 p = float2((vid << 1) & 2, vid & 2); return float4(p * 2.0 - 1.0, 0.0, 1.0); }\n"
    "float4 ps_main(float4 pos : SV_Position) : SV_Target { float a = ao.Load(int3(pos.xy, 0)); return float4(a, a, a, 1.0); }\n";
typedef struct { int fmt; u32 samples; int debug; ID3D12PipelineState* pso; } CompPso;
static CompPso s_comp[8]; static u32 s_comp_n;
static ID3DBlob *s_comp_vs, *s_comp_ps;

static ID3D12PipelineState* comp_pso(int fmt, u32 samples, int debug)
{
    for (u32 i = 0; i < s_comp_n; i++)
        if (s_comp[i].fmt == fmt && s_comp[i].samples == samples && s_comp[i].debug == debug) return s_comp[i].pso;
    if (s_comp_n >= 8) return NULL;
    if (!s_comp_vs) s_comp_vs = compile(kCompHLSL, sizeof kCompHLSL - 1, "vs_main", "vs_5_0", "rt composite vs");
    if (!s_comp_ps) s_comp_ps = compile(kCompHLSL, sizeof kCompHLSL - 1, "ps_main", "ps_5_0", "rt composite ps");
    if (!s_comp_vs || !s_comp_ps) return NULL;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd; memset(&pd, 0, sizeof pd);
    pd.pRootSignature = s_rs_comp;
    pd.VS.pShaderBytecode = CALL0(s_comp_vs, GetBufferPointer); pd.VS.BytecodeLength = CALL0(s_comp_vs, GetBufferSize);
    pd.PS.pShaderBytecode = CALL0(s_comp_ps, GetBufferPointer); pd.PS.BytecodeLength = CALL0(s_comp_ps, GetBufferSize);
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID; pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    D3D12_RENDER_TARGET_BLEND_DESC* b = &pd.BlendState.RenderTarget[0];
    if (!debug) {
        b->BlendEnable = TRUE;
        b->SrcBlend = D3D12_BLEND_ZERO; b->DestBlend = D3D12_BLEND_SRC_COLOR; b->BlendOp = D3D12_BLEND_OP_ADD;       /* dest * ao */
        b->SrcBlendAlpha = D3D12_BLEND_ZERO; b->DestBlendAlpha = D3D12_BLEND_ONE; b->BlendOpAlpha = D3D12_BLEND_OP_ADD;
        b->RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_RED | D3D12_COLOR_WRITE_ENABLE_GREEN | D3D12_COLOR_WRITE_ENABLE_BLUE;
    } else b->RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pd.SampleMask = UINT_MAX; pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1; pd.RTVFormats[0] = (DXGI_FORMAT)fmt; pd.SampleDesc.Count = samples ? samples : 1;
    ID3D12PipelineState* pso = NULL;
    HRESULT hr = CALL(s_dev, CreateGraphicsPipelineState, &pd, &IID_ID3D12PipelineState, (void**)&pso);
    if (FAILED(hr)) { fprintf(stderr, "[rt] composite pipeline failed 0x%08lX (fmt %d x%u)\n", (long)hr, fmt, samples); return NULL; }
    s_comp[s_comp_n].fmt = fmt; s_comp[s_comp_n].samples = samples; s_comp[s_comp_n].debug = debug; s_comp[s_comp_n].pso = pso; s_comp_n++;
    return pso;
}

int rsx_rt_d3d12_composite(const RsxRtCompIn* in)
{
    if (!s_dxr_ok) return 0;
    ID3D12PipelineState* pso = comp_pso(in->fmt, in->samples, eff_debug() != 0);
    if (!pso) return 0;
    ID3D12GraphicsCommandList* l = in->list;
    const D3D12_GPU_DESCRIPTOR_HANDLE tab = in->srv_table(&in->ao_srv, 1);
    CALL(l, SetGraphicsRootSignature, s_rs_comp);
    CALL(l, OMSetRenderTargets, 1, &in->rtv, FALSE, NULL);
    D3D12_VIEWPORT vp = { 0.0f, 0.0f, (float)in->w, (float)in->h, 0.0f, 1.0f };
    D3D12_RECT sc = { 0, 0, (LONG)in->w, (LONG)in->h };
    CALL(l, RSSetViewports, 1, &vp);
    CALL(l, RSSetScissorRects, 1, &sc);
    CALL(l, SetPipelineState, pso);
    CALL(l, IASetPrimitiveTopology, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    CALL(l, SetGraphicsRootDescriptorTable, 0, tab);
    CALL(l, DrawInstanced, 3, 1, 0, 0);
    return 1;
}

/* GPU time of one AO (trace + filter + composite): end timestamp, resolve; read the slot's previous value. */
void rsx_rt_d3d12_ao_end(ID3D12GraphicsCommandList* l, u32 slot)
{
    if (!s_qheap || !s_qread) return;
    CALL(l, EndQuery, s_qheap, D3D12_QUERY_TYPE_TIMESTAMP, slot * 2u + 1u);
    CALL(l, ResolveQueryData, s_qheap, D3D12_QUERY_TYPE_TIMESTAMP, slot * 2u, 2, s_qread, (UINT64)slot * 16u);
    CALL(l, ResolveQueryData, s_qheap, D3D12_QUERY_TYPE_TIMESTAMP, 16u + slot * 4u, 4, s_qread, 128u + (UINT64)slot * 32u);
    s_ts.valid[slot] = 1;
}
void rsx_rt_d3d12_ao_collect(u32 slot)
{
    if (!s_qread) return;
    if (s_ts.valid[slot] || s_ts.bvalid[slot]) {
        u64* p = NULL; D3D12_RANGE rg = { 0, 384 };
        if (SUCCEEDED(CALL(s_qread, Map, 0, &rg, (void**)&p))) {
            if (s_ts.valid[slot]) { const u64 t0 = p[slot * 2], t1 = p[slot * 2 + 1];
                if (t1 > t0) {
                    s_ts.ms += (double)(t1 - t0) * 1000.0 / (double)s_ts_freq; s_ts.n++;
                    const u64* q = p + 16 + slot * 4;      /* after trace, temporal, blur, upsample */
                    if (q[0] >= t0 && q[1] >= q[0] && q[2] >= q[1] && q[3] >= q[2] && t1 >= q[3]) {
                        const u64 e[6] = { t0, q[0], q[1], q[2], q[3], t1 };
                        for (int i = 0; i < 5; i++) s_ts.pass[i] += (double)(e[i + 1] - e[i]) * 1000.0 / (double)s_ts_freq;
                    }
                } }
            if (s_ts.bvalid[slot]) { const u64 t0 = p[8 + slot * 2], t1 = p[8 + slot * 2 + 1];
                if (t1 > t0) { const double m = (double)(t1 - t0) * 1000.0 / (double)s_ts_freq; s_ts.bms += m; s_ts.bn++; if (m > s_ts.bmax) s_ts.bmax = m;
                    /* keep the build cost near 0.6 ms: fewer animated draws when over, a few more when well under */
                    if (s_last_static_built) {      /* static builds in that submit: they set the pace */
                        if (m > 4.0) { s_static_budget = s_static_budget * 3 / 4; if (s_static_budget < 8) s_static_budget = 8; }
                        else if (m < 1.5 && s_static_budget < 128) s_static_budget += 4;
                    } else if (m > 0.8) { s_dyn_budget = s_dyn_budget * 3 / 4; if (s_dyn_budget < 6) s_dyn_budget = 6; }
                    else if (m < 0.4 && s_dyn_budget < 96) s_dyn_budget++; } }
            D3D12_RANGE wr = {0, 0}; CALL(s_qread, Unmap, 0, &wr);
        }
    }
    s_ts.valid[slot] = 0; s_ts.bvalid[slot] = 0;
    static LARGE_INTEGER last, f;
    LARGE_INTEGER now; QueryPerformanceCounter(&now);
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    if (!last.QuadPart) last = now;
    if ((double)(now.QuadPart - last.QuadPart) / (double)f.QuadPart >= 5.0 && s_ts.n) {
        u64 blas = 0, tris = 0;
        for (u32 i = 0; s_tab && i < RT_TABLE; i++) if (s_tab[i].state == IS_READY) { blas += s_tab[i].blas_bytes; tris += s_tab[i].n_tris; }
        if (s_ts.n) fprintf(stderr, "[rt-pass] mean ms per pass over %u frames: trace %.3f, temporal %.3f, blur %.3f, upsample %.3f, composite+barriers %.3f\n", s_ts.n,
                s_ts.pass[0] / s_ts.n, s_ts.pass[1] / s_ts.n, s_ts.pass[2] / s_ts.n, s_ts.pass[3] / s_ts.n, s_ts.pass[4] / s_ts.n);
        fprintf(stderr, "[rt-gpu] AO %.3f ms mean GPU (trace+temporal+filter+composite) over %u frames, builds (vertex eval + BLAS + animated) %.3f ms mean / %.3f max over %u submits; scene %u instances, %.2f M tris, BLAS %.1f MB (uncompacted), %u in queue; dyn: %llu noted, %llu built (%llu tris), %llu rejected, %llu with another camera, %u instances in the last frame, animated budget %u\n",
                s_ts.ms / s_ts.n, s_ts.n, s_ts.bn ? s_ts.bms / s_ts.bn : 0.0, s_ts.bmax, s_ts.bn, s_ready_n, (double)tris / 1e6, (double)blas / 1048576.0, s_qn,
                (unsigned long long)s_st.dyn_noted, (unsigned long long)s_st.dyn_built, (unsigned long long)s_st.dyn_tris,
                (unsigned long long)s_st.dyn_rej, (unsigned long long)s_st.dyn_cam_diff, s_dyn_n, s_dyn_budget);
        s_ts.pass[0] = s_ts.pass[1] = s_ts.pass[2] = s_ts.pass[3] = s_ts.pass[4] = 0; s_ts.ms = 0; s_ts.n = 0; s_ts.bms = 0; s_ts.bn = 0; s_ts.bmax = 0; last = now;
    }
}
void rsx_rt_d3d12_ao_begin_pad(void) {}
