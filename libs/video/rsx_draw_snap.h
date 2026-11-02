/* rsx_draw_snap.h - DrawSnap: the register/constant state of ONE draw, captured at END by the dispatch side so the
 * draw side (sink_end_impl and everything it calls) can read it without touching the live register file g.rsx.
 * 27/10 walkF2, phase 2 of WALKER_PARALLELO.md (RSX_DRAW_SPLIT=1). Session 1: both sides still run on the SAME
 * thread; the point is that the draw path demonstrably needs nothing else (RSX_SNAP_CHECK=1).
 *
 *   RSX_DRAW_SPLIT=1   sink_end fills a DrawSnap from g.rsx, applies it to a private rsx_dispatch ("view"), and the
 *                      draw path reads the view through RV (rsx_draw_engine.c). The view's register file is poisoned
 *                      (0xDEADBEEF) everywhere the whitelist does not reach, so a missed register is loud.
 *   RSX_SNAP_CHECK=1   as above, plus: every register READ through the view is compared with the live file
 *                      (rsx_dsp_trace_read, rsx_dispatch.c); constants, VP words, gens and primitive of the view are
 *                      compared with the live ones before every draw. Differences are logged as [snapcheck].
 *   RSX_DSP_READ_LOG=1 logs the registers the draw path reads (union over the run; at frames 150 and 900, and the ones
 *                      missing from the whitelist). Combines with the above or runs alone on the live file.
 */
#ifndef PS3RECOMP_RSX_DRAW_SNAP_H
#define PS3RECOMP_RSX_DRAW_SNAP_H

#include "rsx_dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SNAP_MAX_REGS 1024            /* whitelist capacity (words); the real list is ~560 */

typedef struct DrawSnap {
    u32 prim;                         /* rsx.current_primitive at END                                  */
    u32 vtx_gen;                      /* g_rsx_dsp_vtx_gen at END (vkey memo)                          */
    u32 vp_gen;                       /* rsx.vp_gen                                                    */
    u32 vp_ver;                       /* version of the immutable VP copy `vp` points at               */
    const u32* vp;                    /* VP program words (RSX_DSP_VP_WORDS), immutable until recycled */
    u32 const_gen;                    /* rsx.const_gen                                                 */
    u32 c_lo, c_hi;                   /* constant slots written since the previous snapshot; lo>hi=none */
    const u32* c_delta;               /* (c_hi-c_lo+1)*4 words, append-only constants log chunk         */
    u32 canvas_mark_at;
    u32 seq;
    u32 nreg;                         /* entries in reg[] (full: the whitelist size; delta: only the registers that changed) */
    const u32* ridx;                  /* register index of reg[i]: the whitelist table (full) or dridx (delta)                 */
    u32 reg[SNAP_MAX_REGS];           /* regs[ridx[i]] (walk37: RSX_SNAP_DELTA, default on: only what changed since the previous snapshot) */
    u32 dridx[SNAP_MAX_REGS];         /* delta mode: register indices of reg[0..nreg)                                           */
} DrawSnap;

/* 0 = off (plain path); else bit0 split, bit1 check, bit2 read-log, bit3 (8) draw executor thread B2. Env read once. */
int  snap_mode(void);
void snap_init(void);
rsx_dispatch* snap_view(void);
/* D side: capture the state of the draw ending now. Consumes live->const_lo/hi (the draw side owns them afterwards). */
DrawSnap* snap_fill(rsx_dispatch* live);
/* Same, into caller storage (the B2 job slot). Only the D side calls it. */
DrawSnap* snap_fill_into(DrawSnap* dst, rsx_dispatch* live);
/* Draw side: bring `view` up to the state of `s` (view regs not in the whitelist stay poisoned). */
void snap_apply(const DrawSnap* s, rsx_dispatch* view);
rsx_dispatch* snap_view_new(void);
void snap_apply_prep(const DrawSnap* s, rsx_dispatch* view, u32* vp_ver);
/* Diagnostics around one draw (no-ops unless the mode asks). `view` is what the draw reads. */
void snap_trace_begin(const DrawSnap* s, rsx_dispatch* view, const rsx_dispatch* live);
void snap_trace_end(rsx_dispatch* view, const rsx_dispatch* live, u32 frame);

const u32* snap_whitelist(u32* n);

/* rsx_dispatch.c (tracing back end) */
void rsx_dsp_trace_set(const rsx_dispatch* target, const rsx_dispatch* live);
const u8* rsx_dsp_trace_bitmap(void);
void rsx_dsp_trace_clear(void);
u32 rsx_dsp_trace_diffs(void);

#ifdef __cplusplus
}
#endif
#endif
