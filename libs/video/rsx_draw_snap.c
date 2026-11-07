/* rsx_draw_snap.c - see rsx_draw_snap.h. 27/10 walkF2. */
#include "rsx_draw_snap.h"
#include "rsx_draw_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern u32 g_rsx_dsp_vtx_gen;

#define SNAP_RING      64
/* rissa128 RSX_B2_SLOTS: the job ring may be up to 1024 slots, so the VP ring (one version per in-flight draw) is 2048 and the constants log scales. */
#define SNAP_VP_RING   2048           /* >= max B2 slots (1024) + a look-back margin: every in-flight draw may hold its own VP version */
#define SNAP_CLOG_MIN  (1u << 20)       /* 4 MB append-only constants log chunk ring; slots*2048 words when RSX_B2_SLOTS > 512 */
static u32 s_clog_words = SNAP_CLOG_MIN;
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
/* walk108: RSX_WALK_SNAP2=1: the delta scan only visits the 16-register groups the dispatcher marked written (rsx_dispatch.h reg_dirty)
 * instead of comparing the whole whitelist (~560 registers) for every draw; =2 also runs the full scan and compares (keeps the full result). */
static int s_snap2 = -1, s_snap2_first;
static u32 s_wl_gs[RSX_DSP_NUM_REGS / 16 + 2];       /* s_wl_gs[g] = first whitelist entry in group g (the list is ascending) */
static unsigned long long s_s2_n, s_s2_mism, s_s2_groups, s_s2_regs;
static u32 s_seq;
static u32 s_vp_ring[SNAP_VP_RING][RSX_DSP_VP_WORDS];
static u32 s_vp_ver, s_vp_last_gen = 0xFFFFFFFFu;
static const u32* s_vp_cur;
/* rissa128 RSX_WALK_VPDUP=1|2 (default 0): a VP program that equals one of the last SNAP_VPD_N versions is not copied again: the
 * snapshot points at the old version (mode 1), so the lanes' per-view caches skip their 8704-byte memcpy too. Mode 2 only counts. */
static u32 s_vp_curver;                  /* version id s_vp_cur belongs to (== s_vp_ver unless a version was reused) */
#define SNAP_VPD_N 16                    /* look-back: 16 + max in-flight 1024 < SNAP_VP_RING */
static int s_vpdup = -1;
static unsigned long long s_vpd_chg, s_vpd_eq;
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

/* rissa128 RSX_B2_SLOTS=<n> (default 256; 64..1024, rounded up to a power of two): slots of the B2 job ring. Read once, before any thread exists.
 * Why: with RSX_LANES=3 a slot frees only when a chunk (K=48 jobs) retires in order, so 256 slots are far fewer in practice and the walker waited on
 * the full ring 2 ms per present (6 ring-full stalls per frame at ~1800 draws, tw1 06/11). */
u32 snap_b2_slots(void)
{
    static u32 v;
    if (!v) {
        const char* e = getenv("RSX_B2_SLOTS");
        u32 n = (e && atoi(e) > 0) ? (u32)atoi(e) : 256u;
        if (n < 64u) n = 64u;
        if (n > 1024u) n = 1024u;
        u32 p = 64u; while (p < n) p <<= 1;
        v = p;
        if (v != 256u) fprintf(stderr, "[snap] RSX_B2_SLOTS=%u (requested %s): job ring %u slots, VP ring %u, constants log %u KB\n", v, e ? e : "-", v, (unsigned)SNAP_VP_RING, (unsigned)((v * 2048u > SNAP_CLOG_MIN ? v * 2048u : SNAP_CLOG_MIN) >> 8));
    }
    return v;
}

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
    { u32 i = 0; for (u32 g = 0; g <= RSX_DSP_NUM_REGS / 16 + 1; g++) { while (i < s_wl_n && (s_wl_idx[i] >> 4) < g) i++; s_wl_gs[g] = i; } }
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
    { const u32 w = snap_b2_slots() * 2048u; s_clog_words = w > SNAP_CLOG_MIN ? w : SNAP_CLOG_MIN; }
    if (!s_clog) s_clog = (u32*)malloc((size_t)s_clog_words * 4u);
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
        s_snap2_first = 1;   /* walk108: the first snapshot after (re)arming compares everything */
        for (u32 i = 0; i < SNAP_MAX_REGS; i++) s_prev[i] = SNAP_POISON;
    }
    if (s_delta) {
        u32 n = 0;
        if (s_snap2 < 0) {
            const char* e2 = getenv("RSX_WALK_SNAP2"); s_snap2 = (e2 && e2[0]) ? atoi(e2) : 0;
            s_snap2_first = 1;
            if (s_snap2) fprintf(stderr, "[snap] RSX_WALK_SNAP2=%d: dirty-group delta scan%s\n", s_snap2, s_snap2 == 2 ? " + full-scan check" : "");
        }
        if (s_snap2 > 0 && !s_snap2_first) {
            static u32 tmp_prev[SNAP_MAX_REGS], ar[SNAP_MAX_REGS], av[SNAP_MAX_REGS];
            u32* prv = s_prev; u32* dri = s->dridx; u32* rgv = s->reg;
            if (s_snap2 == 2) { memcpy(tmp_prev, s_prev, s_wl_n * 4u); prv = tmp_prev; dri = ar; rgv = av; }
            u32 ng = 0;
            for (u32 w = 0; w < RSX_DSP_NUM_REGS / 16 / 64; w++) {
                unsigned long long m = live->reg_dirty[w];
                if (!m) continue;
                live->reg_dirty[w] = 0;
                while (m) {
                    const u32 g = (w << 6) + (u32)__builtin_ctzll(m); m &= m - 1ull; ng++;
                    for (u32 i = s_wl_gs[g], ie = s_wl_gs[g + 1]; i < ie; i++) {
                        const u32 r = s_wl_idx[i], v = live->regs[r];
                        if (v != prv[i]) { prv[i] = v; dri[n] = r; rgv[n] = v; n++; }
                    }
                }
            }
            s_s2_n++; s_s2_groups += ng;
            if (s_snap2 == 2) {
                u32 n2 = 0;
                for (u32 i = 0; i < s_wl_n; i++) {
                    const u32 r = s_wl_idx[i], v = live->regs[r];
                    if (v != s_prev[i]) { s_prev[i] = v; s->dridx[n2] = r; s->reg[n2] = v; n2++; }
                }
                int bad = (n2 != n);
                for (u32 k = 0; !bad && k < n; k++) if (s->dridx[k] != ar[k] || s->reg[k] != av[k]) bad = 1;
                if (bad) { s_s2_mism++; if (s_s2_mism <= 16) fprintf(stderr, "[snap2] MISMATCH seq %u: dirty scan %u regs, full scan %u regs\n", s->seq, n, n2); }
                n = n2;
                if ((s_s2_n % 200000ull) == 0) fprintf(stderr, "[snap2] checked %llu snapshots, mism %llu, avg groups %.1f\n", s_s2_n, s_s2_mism, (double)s_s2_groups / (double)s_s2_n);
            }
        } else {
            for (u32 i = 0; i < s_wl_n; i++) {
                const u32 r = s_wl_idx[i], v = live->regs[r];
                if (v != s_prev[i]) { s_prev[i] = v; s->dridx[n] = r; s->reg[n] = v; n++; }
            }
            if (s_snap2 > 0) { memset(live->reg_dirty, 0, sizeof live->reg_dirty); s_snap2_first = 0; }   /* every register was compared */
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
        int hit = 0; u32 hv = 0;
        if (s_vpdup < 0) { const char* e = getenv("RSX_WALK_VPDUP"); s_vpdup = (e && e[0] == '1') ? 1 : (e && e[0] == '2') ? 2 : 0; }
        if (s_vpdup) {   /* equal to one of the last SNAP_VPD_N versions? (version 0 was never written) */
            const u32 nk = s_vp_ver < SNAP_VPD_N ? s_vp_ver : SNAP_VPD_N;
            for (u32 k = 0; k < nk; k++) {
                const u32 v = s_vp_ver - k;
                if (memcmp(s_vp_ring[v % SNAP_VP_RING], live->vp, sizeof s_vp_ring[0]) == 0) { hit = 1; hv = v; break; }
            }
            s_vpd_chg++; s_vpd_eq += (unsigned long long)hit;
            if ((s_vpd_chg & 131071ull) == 0)
                fprintf(stderr, "[snap] vpdup: %llu changes, %llu equal (%.1f%%) mode %d\n", s_vpd_chg, s_vpd_eq,
                        100.0 * (double)s_vpd_eq / (double)s_vpd_chg, s_vpdup);
        }
        if (hit && s_vpdup == 1) {                /* reuse: no new version, no copy */
            s_vp_cur = s_vp_ring[hv % SNAP_VP_RING]; s_vp_curver = hv;
        } else {
            s_vp_ver++;
            u32* dst = s_vp_ring[s_vp_ver % SNAP_VP_RING];
            memcpy(dst, live->vp, sizeof s_vp_ring[0]);
            s_vp_cur = dst; s_vp_curver = s_vp_ver;
        }
        s_vp_last_gen = live->vp_gen;
    }
    s->vp = s_vp_cur; s->vp_ver = s_vp_curver;
    /* constants: the slots written since the last snapshot, as a log chunk. The dispatcher already tracks lo/hi. */
    s->c_lo = RSX_DSP_NUM_CONSTANTS; s->c_hi = 0; s->c_delta = NULL;
    if (live->const_lo <= live->const_hi && live->const_hi < RSX_DSP_NUM_CONSTANTS) {
        const u32 lo = live->const_lo, hi = live->const_hi, w = (hi - lo + 1u) * 4u;
        if (s_clog_pos + w > s_clog_words) s_clog_pos = 0;
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

void snap_apply_v(const DrawSnap* s, rsx_dispatch* v, u32* vp_ver)
{
    { const u32* ri = s->ridx; for (u32 i = 0; i < s->nreg; i++) v->regs[ri[i]] = s->reg[i]; }
    v->current_primitive = s->prim;
    v->const_gen = s->const_gen;
    v->canvas_mark_at = s->canvas_mark_at;
    v->vp_gen = s->vp_gen;
    if (*vp_ver != s->vp_ver) { memcpy(v->vp, s->vp, sizeof v->vp); *vp_ver = s->vp_ver; }
    if (s->c_lo <= s->c_hi) {
        memcpy(v->constants[s->c_lo], s->c_delta, (s->c_hi - s->c_lo + 1u) * 16u);
        if (s->c_lo < v->const_lo) v->const_lo = s->c_lo;
        if (s->c_hi > v->const_hi) v->const_hi = s->c_hi;
    }
}

void snap_view_copy(rsx_dispatch* d, const rsx_dispatch* s)
{
    if (!s_wl_ready) wl_build();
    for (u32 i = 0; i < s_wl_n; i++) d->regs[s_wl_idx[i]] = s->regs[s_wl_idx[i]];
    d->current_primitive = s->current_primitive;
    d->const_gen = s->const_gen; d->canvas_mark_at = s->canvas_mark_at; d->vp_gen = s->vp_gen;
    d->const_lo = s->const_lo; d->const_hi = s->const_hi;
    memcpy(d->constants, s->constants, sizeof d->constants);
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
