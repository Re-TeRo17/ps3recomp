/* rsx_draw_snap.c - see rsx_draw_snap.h. 27/10 walkF2. */
#include "rsx_draw_snap.h"
#include "rsx_draw_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern u32 g_rsx_dsp_vtx_gen;

#define SNAP_RING      64
#define SNAP_VP_RING   512            /* >= job slots in flight (B2): every in-flight draw may hold its own VP version */
#define SNAP_CLOG_WORDS (1u << 20)      /* 4 MB append-only constants log chunk ring */
#define SNAP_POISON    0xDEADBEEFu

static int s_mode = -1;
static u32 s_wl_idx[SNAP_MAX_REGS];
static u32 s_wl_n;
static int s_wl_ready;

static DrawSnap s_ring[SNAP_RING];
/* walk37: delta snapshots. The draw side's view keeps its registers from the previous draw, so a snapshot only has to carry
 * the registers that changed since the one before it (D keeps s_prev = what the view will hold once the previous job ran).
 * Needs every snapshot applied, in order, to the same view: not with RSX_DRAW_WORKERS (private views, full copy there). */
static u32 s_prev[SNAP_MAX_REGS];
static int s_delta = -1;
static u32 s_seq;
static u32 s_vp_ring[SNAP_VP_RING][RSX_DSP_VP_WORDS];
static u32 s_vp_ver, s_vp_last_gen = 0xFFFFFFFFu;
static const u32* s_vp_cur;
static u32 s_view_vp_ver;
static u32* s_clog;
static u32 s_clog_pos;
static rsx_dispatch s_view;
static rsx_dispatch s_wl_dummy;

/* ---- statistics (check mode) ---- */
static u8  s_rep[RSX_DSP_NUM_REGS];
static unsigned long long s_n_draws, s_n_diff_draws, s_n_state_bad, s_n_regs_bad;
static u32 s_last_diffs;
static int s_dump_done[2];

int snap_mode(void)
{
    if (s_mode < 0) {
        const char* a = getenv("RSX_DRAW_SPLIT"); const char* c = getenv("RSX_SNAP_CHECK"); const char* l = getenv("RSX_DSP_READ_LOG");
        const int chk = c && c[0] && c[0] != '0', rl = l && l[0] && l[0] != '0';
        /* default: B2 draw executor thread (RSX_DRAW_SPLIT=0 = old single-thread walker). The live-vs-view
         * diagnostics cannot run behind B2, so when they are asked for and no split was set explicitly the
         * default drops to 1 instead of silently discarding them. */
        const int sv = (a && a[0]) ? atoi(a) : ((chk || rl) ? 1 : 2);
        const int split = sv >= 1;
        s_mode = ((split || chk) ? 1 : 0) | (chk ? 2 : 0) | (rl ? 4 : 0);
        if (sv >= 2) {   /* RSX_DRAW_SPLIT=2: draw executor thread B2 (rsx_draw_engine.c); the live-vs-view diagnostics cannot run behind it */
            if (chk || rl) fprintf(stderr, "[snap] RSX_DRAW_SPLIT=2 ignores RSX_SNAP_CHECK / RSX_DSP_READ_LOG (they compare against the live file)\n");
            s_mode = 1 | 8;
        }
    }
    return s_mode;
}

/* ---- whitelist: the union of the registers the accessors and the render-state decoder read, found by tracing
 * them on a dummy file (so the list follows the accessors by construction), plus the raw rsx_dsp_reg(...) reads the
 * engine does itself (listed below; RSX_SNAP_CHECK proves the list complete on real traffic). ---- */
static const u32 s_wl_extra[] = {
    0x0208,          /* SET_ANTI_ALIASING_CONTROL (surface aa): eng_current_target_set, sink_end_impl log          */
    0x08E4,          /* eng_fp_reads gate                                                                          */
    0x1EE0, 0x1EE8,  /* point size / point sprite control (RSX_DROP_STATS only)                                    */
    0x1FC0,          /* vertex frequency divider mask (rsx_vertex_compact.c)                                       */
    0x08C0, 0x08C4,  /* SCISSOR_HORIZONTAL / VERTICAL                                                              */
    0x0334,          /* STENCIL_FUNC_REF                                                                           */
    0x031C, 0x0300,  /* blend color / blend color 2 (sink_end_impl)                                                */
    0x1D8C,          /* ZSTENCIL_CLEAR (clear path, free to carry)                                                 */
    0x1D90,          /* CLEAR_COLOR_VALUE                                                                          */
    0x1DB0, 0x1DAC,  /* RESTART_INDEX / ENABLE                                                                     */
};

static void wl_build(void)
{
    rsx_dispatch* d = &s_wl_dummy;
    for (u32 i = 0; i < RSX_DSP_NUM_REGS; i++) d->regs[i] = 0xFFFFFFFFu;
    rsx_dsp_trace_clear();
    rsx_dsp_trace_set(d, NULL);
    {
        rsx_dsp_surface sf; rsx_dsp_get_surface(d, &sf);
        rsx_dsp_viewport vp; rsx_dsp_get_viewport(d, &vp);
        rsx_dsp_index_array ia; rsx_dsp_get_index_array(d, &ia);
        for (u32 a = 0; a < RSX_DSP_NUM_VERTEX_ATTR; a++) {
            rsx_dsp_vertex_attr va; rsx_dsp_get_vertex_attr(d, a, &va);
            float df[4]; rsx_dsp_vertex_default(d, a, df);
        }
        for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++) { rsx_dsp_texture t; rsx_dsp_get_texture(d, u, &t); }
        for (u32 u = 0; u < RSX_DSP_NUM_VERTEX_TEXTURES; u++) { rsx_dsp_vertex_texture t; rsx_dsp_get_vertex_texture(d, u, &t); }
        u32 loc; (void)rsx_dsp_vertex_data_base_offset(d); (void)rsx_dsp_vertex_data_base_index(d);
        (void)rsx_dsp_fragment_program(d, &loc); (void)rsx_dsp_shader_control(d); (void)rsx_dsp_vp_start(d);
        (void)rsx_dsp_restart_index_enabled(d, 0); (void)rsx_dsp_restart_index_enabled(d, 1); (void)rsx_dsp_restart_index(d);
        (void)rsx_dsp_clear_color(d); (void)rsx_dsp_clear_zstencil(d);
        rsx_be_render_state rs; rsx_draw_engine_decode_render_state(d, &rs);
    }
    rsx_dsp_trace_set(NULL, NULL);
    const u8* bm = rsx_dsp_trace_bitmap();
    static u8 want[RSX_DSP_NUM_REGS];
    for (u32 i = 0; i < RSX_DSP_NUM_REGS; i++) want[i] = bm[i] & 1;
    for (u32 k = 0; k < sizeof s_wl_extra / sizeof s_wl_extra[0]; k++) want[s_wl_extra[k] >> 2] = 1;
    s_wl_n = 0;
    for (u32 i = 0; i < RSX_DSP_NUM_REGS && s_wl_n < SNAP_MAX_REGS; i++)
        if (want[i]) s_wl_idx[s_wl_n++] = i;
    rsx_dsp_trace_clear();
    s_wl_ready = 1;
    fprintf(stderr, "[snap] whitelist: %u registers (draw path reads), DrawSnap %u bytes\n", s_wl_n, (unsigned)sizeof(DrawSnap));
}

const u32* snap_whitelist(u32* n) { if (!s_wl_ready) wl_build(); if (n) *n = s_wl_n; return s_wl_idx; }

void snap_init(void)
{
    if (!snap_mode()) return;
    if (!s_wl_ready) wl_build();
    for (u32 i = 0; i < RSX_DSP_NUM_REGS; i++) s_view.regs[i] = SNAP_POISON;
    s_view.const_lo = RSX_DSP_NUM_CONSTANTS; s_view.const_hi = 0;
    if (!s_clog) s_clog = (u32*)malloc((size_t)SNAP_CLOG_WORDS * 4u);
    s_view_vp_ver = 0xFFFFFFFFu;
    s_delta = -1;   /* re-armed with the poisoned view: the first snapshot carries everything */
    fprintf(stderr, "[snap] mode %d (1 split, 2 check, 4 read-log)\n", snap_mode());
}

rsx_dispatch* snap_view(void) { return &s_view; }

DrawSnap* snap_fill(rsx_dispatch* live) { return snap_fill_into(&s_ring[s_seq % SNAP_RING], live); }

DrawSnap* snap_fill_into(DrawSnap* s, rsx_dispatch* live)
{
    s->seq = s_seq++;
    if (s_delta < 0) {
        const char* e = getenv("RSX_SNAP_DELTA"); const char* w = getenv("RSX_DRAW_WORKERS");
        s_delta = !(e && e[0] == '0') && !(w && atoi(w) > 0);
        for (u32 i = 0; i < SNAP_MAX_REGS; i++) s_prev[i] = SNAP_POISON;
    }
    if (s_delta) {
        u32 n = 0;
        for (u32 i = 0; i < s_wl_n; i++) {
            const u32 r = s_wl_idx[i], v = live->regs[r];
            if (v != s_prev[i]) { s_prev[i] = v; s->dridx[n] = r; s->reg[n] = v; n++; }
        }
        s->nreg = n; s->ridx = s->dridx;
    } else {
        s->nreg = s_wl_n; s->ridx = s_wl_idx;
        for (u32 i = 0; i < s_wl_n; i++) s->reg[i] = live->regs[s_wl_idx[i]];
    }
    s->prim = live->current_primitive;
    s->vtx_gen = g_rsx_dsp_vtx_gen;
    s->const_gen = live->const_gen;
    s->canvas_mark_at = live->canvas_mark_at;
    s->vp_gen = live->vp_gen;
    if (live->vp_gen != s_vp_last_gen) {          /* VP program changed: new immutable version (v2: dirty-range copy) */
        s_vp_ver++;
        u32* dst = s_vp_ring[s_vp_ver % SNAP_VP_RING];
        memcpy(dst, live->vp, sizeof s_vp_ring[0]);
        s_vp_cur = dst; s_vp_last_gen = live->vp_gen;
    }
    s->vp = s_vp_cur; s->vp_ver = s_vp_ver;
    /* constants: the slots written since the last snapshot, as a log chunk. The dispatcher already tracks lo/hi. */
    s->c_lo = RSX_DSP_NUM_CONSTANTS; s->c_hi = 0; s->c_delta = NULL;
    if (live->const_lo <= live->const_hi && live->const_hi < RSX_DSP_NUM_CONSTANTS) {
        const u32 lo = live->const_lo, hi = live->const_hi, w = (hi - lo + 1u) * 4u;
        if (s_clog_pos + w > SNAP_CLOG_WORDS) s_clog_pos = 0;
        memcpy(s_clog + s_clog_pos, live->constants[lo], w * 4u);
        s->c_lo = lo; s->c_hi = hi; s->c_delta = s_clog + s_clog_pos;
        s_clog_pos += w;
    }
    live->const_lo = RSX_DSP_NUM_CONSTANTS; live->const_hi = 0;   /* consumed here; the draw side owns the view's range */
    return s;
}

void snap_apply(const DrawSnap* s, rsx_dispatch* v)
{
    { const u32* ri = s->ridx; for (u32 i = 0; i < s->nreg; i++) v->regs[ri[i]] = s->reg[i]; }
    v->current_primitive = s->prim;
    v->const_gen = s->const_gen;
    v->canvas_mark_at = s->canvas_mark_at;
    v->vp_gen = s->vp_gen;
    if (s_view_vp_ver != s->vp_ver) { memcpy(v->vp, s->vp, sizeof v->vp); s_view_vp_ver = s->vp_ver; }
    if (s->c_lo <= s->c_hi) {
        memcpy(v->constants[s->c_lo], s->c_delta, (s->c_hi - s->c_lo + 1u) * 16u);
        if (s->c_lo < v->const_lo) v->const_lo = s->c_lo;      /* range accumulates until the draw side consumes it */
        if (s->c_hi > v->const_hi) v->const_hi = s->c_hi;
    }
}

/* A private view for a prep worker (RSX_DRAW_WORKERS): registers, primitive, gens and the VP words only (no constants, the
 * prep never reads them). *vp_ver is the caller's cache of the VP version copied into `v`. */
rsx_dispatch* snap_view_new(void)
{
    rsx_dispatch* v = (rsx_dispatch*)calloc(1, sizeof *v);
    if (v) { v->const_lo = RSX_DSP_NUM_CONSTANTS; v->const_hi = 0; }
    return v;
}
void snap_apply_prep(const DrawSnap* s, rsx_dispatch* v, u32* vp_ver)
{
    { const u32* ri = s->ridx; for (u32 i = 0; i < s->nreg; i++) v->regs[ri[i]] = s->reg[i]; }
    v->current_primitive = s->prim;
    v->const_gen = s->const_gen;
    v->canvas_mark_at = s->canvas_mark_at;
    v->vp_gen = s->vp_gen;
    if (*vp_ver != s->vp_ver) { memcpy(v->vp, s->vp, sizeof v->vp); *vp_ver = s->vp_ver; }
}

/* ---- diagnostics ---- */
static void dump_reads(const u8* bm, int which)
{
    char line[256]; u32 n = 0, nmiss = 0; int o = 0;
    static u8 inwl[RSX_DSP_NUM_REGS];
    fprintf(stderr, "[dsp-read] ---- registers read by the draw path (dump %d) ----\n", which);
    memset(inwl, 0, sizeof inwl);
    for (u32 i = 0; i < s_wl_n; i++) inwl[s_wl_idx[i]] = 1;
    for (u32 i = 0; i < RSX_DSP_NUM_REGS; i++) if (bm[i] & 1) {
        o += snprintf(line + o, sizeof line - o, " %04X", i << 2); n++;
        if (n % 16 == 0) { fprintf(stderr, "[dsp-read]%s\n", line); o = 0; }
        if (!inwl[i]) nmiss++;
    }
    if (o) fprintf(stderr, "[dsp-read]%s\n", line);
    fprintf(stderr, "[dsp-read] %u distinct registers read, %u of them NOT in the whitelist (%u words):", n, nmiss, s_wl_n);
    for (u32 i = 0; i < RSX_DSP_NUM_REGS; i++) if ((bm[i] & 1) && !inwl[i]) fprintf(stderr, " %04X", i << 2);
    fprintf(stderr, "\n");
}

void snap_trace_begin(const DrawSnap* s, rsx_dispatch* view, const rsx_dispatch* live)
{
    const int m = snap_mode();
    if (!(m & 6)) return;
    if ((m & 2) && s) {
        s_n_draws++;
        int bad = 0;
        if (memcmp(view->constants, live->constants, sizeof view->constants)) {
            bad |= 1;
            for (u32 k = 0; k < RSX_DSP_NUM_CONSTANTS; k++)
                if (memcmp(view->constants[k], live->constants[k], 16)) {
                    if (s_n_state_bad < 16)
                        fprintf(stderr, "[snapcheck] seq %u constants[%u] differ (view %08X live %08X)\n", s->seq, k,
                                view->constants[k][0], live->constants[k][0]);
                    break;
                }
        }
        if (memcmp(view->vp, live->vp, sizeof view->vp)) {
            bad |= 2;
            if (s_n_state_bad < 16) fprintf(stderr, "[snapcheck] seq %u VP words differ (vp_gen %u)\n", s->seq, live->vp_gen);
        }
        if (view->const_gen != live->const_gen || view->vp_gen != live->vp_gen || view->current_primitive != live->current_primitive ||
            view->canvas_mark_at != live->canvas_mark_at) {
            bad |= 4;
            if (s_n_state_bad < 16)
                fprintf(stderr, "[snapcheck] seq %u scalars: const_gen %u/%u vp_gen %u/%u prim %u/%u canvas %u/%u\n", s->seq,
                        view->const_gen, live->const_gen, view->vp_gen, live->vp_gen, view->current_primitive, live->current_primitive,
                        view->canvas_mark_at, live->canvas_mark_at);
        }
        if (bad) s_n_state_bad++;
        s_last_diffs = rsx_dsp_trace_diffs();
        rsx_dsp_trace_set(view, live);
    } else
        rsx_dsp_trace_set(view, NULL);
}

void snap_trace_end(rsx_dispatch* view, const rsx_dispatch* live, u32 frame)
{
    const int m = snap_mode();
    if (!(m & 6)) return;
    const u8* bm = rsx_dsp_trace_bitmap();
    if (m & 2) {
        const u32 d = rsx_dsp_trace_diffs();
        if (d != s_last_diffs) {
            s_n_diff_draws++;
            for (u32 i = 0; i < RSX_DSP_NUM_REGS; i++) if ((bm[i] & 6) && !s_rep[i]) {
                s_rep[i] = 1; s_n_regs_bad++;
                if (s_n_regs_bad <= 64)
                    fprintf(stderr, "[snapcheck] f%u register 0x%04X: view %08X live %08X%s\n", frame, i << 2, view->regs[i], live->regs[i],
                            view->regs[i] == SNAP_POISON ? "  (NOT IN WHITELIST)" : "  (stale/diverged)");
            }
        }
    }
    rsx_dsp_trace_set(NULL, NULL);
    for (int k = 0; k < 2; k++) {
        const u32 at = k ? 900u : 150u;
        if (!s_dump_done[k] && frame >= at) {
            s_dump_done[k] = 1;
            if (m & 4) dump_reads(bm, k);
            if (m & 2)
                fprintf(stderr, "[snapcheck] f%u: %llu draws checked, %llu draws with a differing register read (%llu distinct registers), %llu with differing constants/VP/scalars\n",
                        frame, s_n_draws, s_n_diff_draws, s_n_regs_bad, s_n_state_bad);
        }
    }
}
