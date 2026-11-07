/*
 * Even presents.
 *
 * A title that paces itself -- a frame-rate cap in its own main loop --
 * starts its frames at even intervals, but they do not reach the present at
 * even intervals: between the start of a frame and its present lie the
 * game's work, the render thread's and the RSX walker's, and each varies by
 * a millisecond or two from frame to frame. On the console the flip waited
 * for the vertical blank and nobody saw it. Here a frame is shown when it is
 * made (v-sync off, or a display that follows the frames), and a camera
 * turning at a steady rate is seen to turn unevenly: at a 60 fps cap, with
 * every frame simulated for exactly 16.67 ms, the presents came 13 to 20 ms
 * apart, half of them more than 15% off the one before.
 *
 * rsx_present_pace() is called by a backend on the presenting thread, just
 * before it presents, with the frame's GPU work already submitted. It learns
 * the spacing of the frames from their arrivals (or is told it: a host that
 * caps the frame rate knows it exactly) and how much they vary, and holds
 * each one until its turn on an even schedule a little behind them -- by
 * three times the variation, a few milliseconds -- so that nearly every
 * frame is early for its slot and waits for it. A frame that is late is
 * shown at once and the schedule starts again from it.
 *
 * It needs slack to work with: on a title running flat out the wait would
 * come out of the frame rate. So it is off unless the host says so
 * (rsx_present_pace_enable, or RSX_PRESENT_PACE=1 in the environment), and a
 * host turns it on while a cap is in force. And it must not take from the
 * title what it gives the eye: when the frames come slower than the cap the
 * host named, they are shown as they come; and a backend that holds a frame
 * on a thread of its own (rsx_present_pace_on) asks for it the moment the
 * next one needs the swap chain (rsx_present_pace_hurry), so the thread that
 * walks the command buffer never waits out a pacing delay.
 *
 *   RSX_PRESENT_PACE=1       on from the start (0: off whatever the host says)
 *   RSX_PRESENT_PACE_LOG=1   every 2 s: the spacing, its variation before
 *                            and after, the mean wait, the late frames
 *   RSX_PRESENT_PACE_LOG=2   and a line a frame
 */
#ifndef PS3RECOMP_RSX_PRESENT_PACE_H
#define PS3RECOMP_RSX_PRESENT_PACE_H

#ifdef __cplusplus
extern "C" {
#endif

/* On or off; any thread, at any time. */
void rsx_present_pace_enable(int on);

/* The frames' spacing in seconds, from a host that sets it (its frame-rate
 * cap); 0 when it does not know, and the arrivals are measured instead. Any
 * thread, at any time. */
void rsx_present_pace_period(double seconds);

/* In force now (the host's word and RSX_PRESENT_PACE): a backend whose
 * present holds up the thread that walks the command buffer asks, and gives
 * a paced present to a thread of its own. The wait is a few milliseconds a
 * frame, and a title that waits for its flip before it starts the next
 * frame would otherwise be slowed by every one of them. */
int rsx_present_pace_on(void);

/* The presenting thread, just before the present: returns when the frame is
 * due. At once when pacing is off. */
void rsx_present_pace(void);

/* From another thread, while rsx_present_pace() may be holding a frame: on,
 * it returns at once (now and until this is turned off again). */
void rsx_present_pace_hurry(int on);

/* ---- RSX_PACE_VBLANK=1 (default off): the real vertical blank ----
 * The fixed 60.000 Hz timers of the port (the flip pace, GCM_FLIP_PACE_HZ) and a real display
 * (59.94, 60.00x, 119.88 ...) beat against each other: every 1/|f_display - 60| seconds a
 * frame is shown twice or never. With this on, the flip pace follows the display: its period
 * is the measured refresh period (DwmGetCompositionTimingInfo: exact rate and the time of the
 * last vblank) times the whole number of refreshes that comes nearest to the wanted period,
 * and each flip is released a fixed lead before a vblank of that grid. Used only when the wanted
 * period is a whole multiple of the refresh within 2%; otherwise the fixed timer stays.
 *   RSX_PACE_VBLANK_LEAD_PCT=<n>  lead before the vblank, % of the paced period (default 60)
 *   RSX_PACE_VBLANK_LOG=1         every 5 s: grid period, locked/unlocked flips, late flips */
int rsx_vblank_lock_on(void);
/* v-sync on (1) or off (0): the lock applies only with v-sync on. Called by the backend whenever it reads RSX_VSYNC. */
void rsx_vblank_set_vsync(int on);
/* The wanted flip period (seconds) -> 1 and the released time grid if the lock applies. now/out in QPC ticks.
 * *target is the QPC tick at which the next flip is to be released (never before 'now'); *period_ticks the paced period. */
int rsx_vblank_next_release(double want_period_s, long long now_qpc, long long* target, long long* period_ticks);
/* The compositor's timing now (DwmGetCompositionTimingInfo, read at most every 200 ms): refresh period in seconds,
 * the QPC tick of the last vblank, the refresh counter. 0 when it is not there. */
int rsx_vblank_snapshot(double* period_s, long long* vblank_qpc, unsigned long long* refresh_count);

#ifdef __cplusplus
}
#endif

#endif
