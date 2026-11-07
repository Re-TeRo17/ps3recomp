/*
 * Even presents: see rsx_present_pace.h.
 */
#include "rsx_present_pace.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

static volatile int s_host_on = 0;          /* what the host asked for */
static volatile float s_host_period = 0.0f; /* the spacing it named, seconds; 0 unknown */
static volatile int s_hurry = 0;            /* show the frame that is being held now */
static int s_env = -1;                      /* RSX_PRESENT_PACE: -1 unread, 0 off, 1 on, 2 unset */
static int s_log = 0;

void rsx_present_pace_enable(int on) { s_host_on = on ? 1 : 0; }

void rsx_present_pace_period(double seconds)
{
    s_host_period = seconds > 0.0005 && seconds < 0.2 ? (float)seconds : 0.0f;
}

void rsx_present_pace_hurry(int on) { s_hurry = on ? 1 : 0; }

static void read_env(void)
{
    const char* e = getenv("RSX_PRESENT_PACE");
    const char* l = getenv("RSX_PRESENT_PACE_LOG");
    s_log = l ? atoi(l) : 0;
    s_env = !e || !*e ? 2 : (e[0] == '0' ? 0 : 1);
}

int rsx_present_pace_on(void)
{
    if (s_env < 0) read_env();
    return s_env == 1 || (s_env == 2 && s_host_on);
}

static double now_s(void)
{
#ifdef _WIN32
    static double inv = 0.0;
    LARGE_INTEGER q;
    if (inv == 0.0) { QueryPerformanceFrequency(&q); inv = 1.0 / (double)q.QuadPart; }
    QueryPerformanceCounter(&q);
    return (double)q.QuadPart * inv;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* A millisecond at most, and not exactly: the caller looks at the clock. */
static void doze(double seconds)
{
#ifdef _WIN32
    static HANDLE timer = NULL;
    if (!timer) {
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
        timer = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        if (!timer) timer = CreateWaitableTimerW(NULL, TRUE, NULL);
    }
    if (timer) {
        LARGE_INTEGER due;
        due.QuadPart = -(LONGLONG)(seconds * 1e7);   /* 100 ns units, relative */
        if (SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE)) { WaitForSingleObject(timer, 20); return; }
    }
    Sleep(1);
#else
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = (long)(seconds * 1e9);
    nanosleep(&ts, NULL);
#endif
}

/* Until t, or until the frame is asked for: 1 when it was. Sleeps in slices
 * while there is more than a millisecond and a half to go and spins through
 * the rest: a timer alone comes back up to a millisecond late, which is the
 * unevenness this is here to remove. */
static int wait_until(double t)
{
    for (;;) {
        if (s_hurry) return 1;
        const double left = t - now_s();
        if (left <= 0.0) return 0;
        if (left > 0.0015) doze(left - 0.0008 < 0.001 ? left - 0.0008 : 0.001);
#ifdef _WIN32
        else YieldProcessor();
#endif
    }
}

void rsx_present_pace(void)
{
    enum { HIST = 8 };
    static double hist[HIST];          /* the last arrival intervals */
    static int hist_n, hist_at;
    static double last;                /* the last arrival */
    static double period, jitter;      /* the frames' spacing and its mean deviation */
    static double due;                 /* the schedule: when the last frame was due */
    static double wait_avg;
    static double shown;               /* when the last frame was let through */
    /* the report */
    static double rep_next, rep_wait, rep_in2, rep_out2, rep_in1, rep_out1, rep_max;
    static unsigned rep_n, rep_late, rep_cut, rep_free, rep_in_big, rep_out_big;
    static double prev_in, prev_out;

    if (!rsx_present_pace_on()) { last = 0.0; return; }

    const double a = now_s();
    const double d = a - last;
    const int fresh = last <= 0.0 || d <= 0.0 || d > 0.1;   /* the first frame, or the first after a gap */
    last = a;
    if (fresh) {
        hist_n = 0; period = 0.0; due = a; shown = a; wait_avg = 0.0; prev_in = prev_out = 0.0;
        return;
    }
    hist[hist_at] = d; hist_at = (hist_at + 1) % HIST;
    if (hist_n < HIST) hist_n++;
    if (hist_n < 4) { due = a; shown = a; return; }

    /* The spacing of the arrivals: the median of the last few, so that one
     * long frame does not move it and a change of frame rate does at once. */
    double m[HIST];
    memcpy(m, hist, sizeof(double) * (size_t)hist_n);
    for (int i = 1; i < hist_n; i++) {
        const double v = m[i];
        int j = i - 1;
        while (j >= 0 && m[j] > v) { m[j + 1] = m[j]; j--; }
        m[j + 1] = v;
    }
    const double med = (hist_n & 1) ? m[hist_n / 2] : 0.5 * (m[hist_n / 2 - 1] + m[hist_n / 2]);

    /* The spacing to schedule by. A host that caps the frame rate names it,
     * and it is taken while the frames bear it out. When they come slower
     * than that, the title is not making its frames in time and there is no
     * slack to pace them with: they are shown as they come. (When they come
     * faster, the name is out of date, and the arrivals are measured.) */
    const double told = (double)s_host_period;
    int free_run = 0;
    if (told > 0.0 && med >= 0.8 * told) {
        if (med > 1.2 * told) free_run = 1;
        else {
            if (period != told) { due = a; wait_avg = 0.0; if (jitter <= 0.0 || jitter > 0.3 * told) jitter = 0.1 * told; }
            period = told;
        }
    } else {
        if (period <= 0.0 || med < 0.6 * period || med > 1.6 * period) {   /* a different frame rate: from here */
            period = med; jitter = 0.1 * med; due = a; wait_avg = 0.0;
        }
        period += (med - period) * 0.2;
    }

    double w = 0.0, target = 0.0;
    int late = 0, cut = 0;
    if (free_run) {
        due = a; wait_avg = 0.0;
    } else {
        double dev = fabs(d - period);
        if (dev > 0.5 * period) dev = 0.5 * period;
        jitter += (dev - jitter) * 0.05;

        /* How far behind the frames the schedule runs: three times their
         * usual deviation, at most four tenths of a frame and 8 ms. */
        target = 3.0 * jitter;
        if (target < 0.0003) target = 0.0003;
        if (target > 0.4 * period) target = 0.4 * period;
        if (target > 0.008) target = 0.008;

        due += period;
        w = due - a;
        wait_avg += (w - wait_avg) * 0.1;
        due -= 0.2 * (wait_avg - target);   /* steer the usual wait to the target */
        w = due - a;
        if (w <= 0.0) { due = a; w = 0.0; late = 1; }
        else if (w > 2.0 * target + 0.001) { due = a + target; w = target; wait_avg = target; }   /* the schedule has fallen behind the frames */
        if (w > 0.0 && wait_until(due)) { cut = 1; due = now_s(); wait_avg = 0.0; }
    }
    const double out = w > 0.0 ? now_s() : a;

    if (s_log > 0) {
        const double in_iv = d, out_iv = out - shown;
        rep_n++; rep_late += (unsigned)late; rep_cut += (unsigned)cut; rep_free += (unsigned)free_run; rep_wait += out - a;
        if (out - a > rep_max) rep_max = out - a;
        rep_in1 += in_iv; rep_in2 += in_iv * in_iv; rep_out1 += out_iv; rep_out2 += out_iv * out_iv;
        if (prev_in > 0.0 && (in_iv > 1.15 * prev_in || in_iv < 0.87 * prev_in)) rep_in_big++;
        if (prev_out > 0.0 && (out_iv > 1.15 * prev_out || out_iv < 0.87 * prev_out)) rep_out_big++;
        prev_in = in_iv; prev_out = out_iv;
        if (s_log > 1)
            fprintf(stderr, "[pace] in %.3f out %.3f wait %.3f ms%s\n", in_iv * 1e3, out_iv * 1e3, (out - a) * 1e3,
                    late ? " late" : cut ? " cut" : free_run ? " free" : "");
        if (a >= rep_next && rep_n > 1) {
            const double n = (double)rep_n;
            const double mi = rep_in1 / n, mo = rep_out1 / n;
            const double si = sqrt(fabs(rep_in2 / n - mi * mi)), so = sqrt(fabs(rep_out2 / n - mo * mo));
            fprintf(stderr, "[pace] %u frames %.2f ms apart (scheduled by %.2f): arrivals vary %.2f ms (%u%% over 15%% off the one before), "
                            "presents %.2f ms (%u%%); wait %.2f ms mean, %.2f at most; %u late, %u cut short, %u unpaced\n",
                    rep_n, mo * 1e3, period * 1e3, si * 1e3, (unsigned)(100.0 * rep_in_big / n + 0.5), so * 1e3,
                    (unsigned)(100.0 * rep_out_big / n + 0.5), rep_wait / n * 1e3, rep_max * 1e3, rep_late, rep_cut, rep_free);
            rep_next = a + 2.0;
            rep_n = rep_late = rep_cut = rep_free = rep_in_big = rep_out_big = 0;
            rep_wait = rep_in1 = rep_in2 = rep_out1 = rep_out2 = rep_max = 0.0;
        }
    }
    shown = out;
}

/* ---------------------------------------------------------------------------
 * RSX_PACE_VBLANK: the flip pace on the display's real vertical blank.
 * -------------------------------------------------------------------------*/
#ifdef _WIN32
#include <dwmapi.h>

typedef HRESULT (WINAPI *PFN_DwmGetCompositionTimingInfo)(HWND, DWM_TIMING_INFO*);
static PFN_DwmGetCompositionTimingInfo s_dwm_fn;
static int s_dwm_tried;

static int vb_env_on(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("RSX_PACE_VBLANK"); on = (e && *e && *e != '0') ? 1 : 0; }
    return on;
}

int rsx_vblank_lock_on(void) { return vb_env_on(); }

static double vb_qpc_freq(void)
{
    static double f = 0.0;
    if (f == 0.0) { LARGE_INTEGER q; QueryPerformanceFrequency(&q); f = (double)q.QuadPart; }
    return f;
}

/* One read of the compositor's timing: 1 when it answered. */
static int vb_read(double* period_s, long long* vb, unsigned long long* cref)
{
    if (!s_dwm_tried) {
        s_dwm_tried = 1;
        HMODULE m = LoadLibraryW(L"dwmapi.dll");
        if (m) s_dwm_fn = (PFN_DwmGetCompositionTimingInfo)GetProcAddress(m, "DwmGetCompositionTimingInfo");
    }
    if (!s_dwm_fn) return 0;
    DWM_TIMING_INFO ti;
    memset(&ti, 0, sizeof ti);
    ti.cbSize = sizeof ti;
    if (FAILED(s_dwm_fn(NULL, &ti))) return 0;
    double p = 0.0;
    if (ti.rateRefresh.uiDenominator && ti.rateRefresh.uiNumerator)
        p = (double)ti.rateRefresh.uiDenominator / (double)ti.rateRefresh.uiNumerator;
    else if (ti.qpcRefreshPeriod)
        p = (double)ti.qpcRefreshPeriod / vb_qpc_freq();
    if (p < 0.002 || p > 0.1 || ti.qpcVBlank == 0) return 0;
    *period_s = p; *vb = (long long)ti.qpcVBlank; *cref = (unsigned long long)ti.cRefresh;
    return 1;
}

int rsx_vblank_snapshot(double* period_s, long long* vblank_qpc, unsigned long long* refresh_count)
{
    static double p; static long long vb; static unsigned long long cr; static long long last_read; static int ok;
    LARGE_INTEGER q; QueryPerformanceCounter(&q);
    static int every = -1; if (every < 0) every = getenv("RSX_FT_DUMP") ? 1 : 0;
    if (!last_read || every || q.QuadPart - last_read > (long long)(vb_qpc_freq() * 0.2)) {
        double p2; long long vb2; unsigned long long cr2;
        ok = vb_read(&p2, &vb2, &cr2);
        if (ok) { p = p2; vb = vb2; cr = cr2; }
        last_read = q.QuadPart;
    }
    if (!ok) return 0;
    *period_s = p; *vblank_qpc = vb; *refresh_count = cr;
    return 1;
}

/* The grid: vblank number j is at anchor + (j - j0) * period. Re-anchored on every fresh reading. */
static struct {
    int valid; double period; long long anchor; long long j0;     /* period in QPC ticks (double) */
    long long next_j;                                             /* the vblank the next flip is aimed at; 0 = none */
    long long last_read;
    unsigned locked, late, relock, unlocked_calls;
    long long rep0;
} s_vb;

static void vb_refresh_grid(long long now)
{
    const double f = vb_qpc_freq();
    if (s_vb.valid && now - s_vb.last_read < (long long)(f * 0.25)) return;
    s_vb.last_read = now;
    double p; long long vb; unsigned long long cr;
    if (!vb_read(&p, &vb, &cr)) { return; }
    static double force = -1.0;
    if (force < 0.0) { const char* e = getenv("RSX_PACE_FORCE_HZ"); force = e && *e ? atof(e) : 0.0; if (force > 0.0) fprintf(stderr, "[vblank-lock] RSX_PACE_FORCE_HZ=%.3f: refresh simulated (real %.3f Hz)%c", force, 1.0 / p, 10); }
    const double pt = force >= 10.0 && force <= 1000.0 ? f / force : p * f;
    if (s_vb.valid) {
        /* keep the numbering: the new anchor is vblank number j0 + round((vb - anchor) / period) */
        const long long dj = (long long)floor(((double)(vb - s_vb.anchor)) / s_vb.period + 0.5);
        s_vb.j0 += dj;
    } else { s_vb.j0 = 0; s_vb.next_j = 0; }
    s_vb.anchor = vb; s_vb.period = pt; s_vb.valid = 1;
}

static long long vb_time(long long j) { return s_vb.anchor + (long long)((double)(j - s_vb.j0) * s_vb.period); }

static volatile int s_vb_vsync = 1;
void rsx_vblank_set_vsync(int on) { s_vb_vsync = on ? 1 : 0; }
static int s_vb_case = -1;
static void vb_case(int c, const char* what, double hz, long long k)
{
    if (c == s_vb_case) return;
    s_vb_case = c;
    fprintf(stderr, "[vblank-lock] case %d: %s (display %.3f Hz, %lld refresh(s) per flip)%c", c, what, hz, k, 10);
}

int rsx_vblank_next_release(double want_period_s, long long now, long long* target, long long* period_ticks)
{
    if (!vb_env_on()) return 0;
    if (!s_vb_vsync) { vb_case(4, "vsync off: fixed timer (a VRR display follows the presents; no vblank to lock to)", 0.0, 0); s_vb.next_j = 0; return 0; }
    static double lead_pct = -1.0; static int log_on = -1;
    if (lead_pct < 0.0) {
        const char* e = getenv("RSX_PACE_VBLANK_LEAD_PCT"); lead_pct = e && *e ? atof(e) : 60.0;
        if (lead_pct < 0.0 || lead_pct > 95.0) lead_pct = 60.0;
        const char* l = getenv("RSX_PACE_VBLANK_LOG"); log_on = l && *l && *l != '0';
    }
    vb_refresh_grid(now);
    if (!s_vb.valid) { s_vb.unlocked_calls++; return 0; }
    const double f = vb_qpc_freq();
    const double d = s_vb.period / f;                       /* refresh period, seconds */
    const double ratio = want_period_s / d;
    long long k = (long long)floor(ratio + 0.5);
    if (k >= 1 && k <= 8 && fabs(ratio - (double)k) <= 0.02 * (double)k) {
        vb_case(1, "refresh is a whole multiple of the wanted rate: locked to the vblank", 1.0 / d, k);
    } else {
        static int nm = -1;
        if (nm < 0) { const char* e = getenv("RSX_PACE_VBLANK_NONMULT"); nm = (e && !_stricmp(e, "timer")) ? 0 : 1; }
        if (!nm) { vb_case(3, "refresh not a multiple, RSX_PACE_VBLANK_NONMULT=timer: fixed timer", 1.0 / d, 0); s_vb.unlocked_calls++; s_vb.next_j = 0; return 0; }
        /* Not a multiple (75, 100, 144, 165 ...): the frames are a whole number of refreshes apart anyway when v-sync is on, so
         * pick that number: the smallest k that keeps the rate within 25% of the cap (the game's own clocks run to 75 fps). A uniform
         * 72 fps (144 Hz) is smoother than 60 fps shown as 2,3,2,3 refreshes. NONMULT=timer keeps the fixed 60 Hz timer. */
        k = (long long)ceil(0.8 * ratio - 1e-9);
        if (k < 1) k = 1;
        if (k > 8) { vb_case(3, "refresh far from the wanted rate: fixed timer", 1.0 / d, 0); s_vb.unlocked_calls++; s_vb.next_j = 0; return 0; }
        vb_case(2, "refresh NOT a multiple of the wanted rate: uniform whole-refresh pacing at refresh/k", 1.0 / d, k);
    }
    const long long pk = (long long)(s_vb.period * (double)k);
    const long long lead = (long long)((double)pk * lead_pct / 100.0);
    /* vblank number nearest 'now' on the grid (rounded up: the first vblank not yet behind us) */
    long long jn = s_vb.j0 + (long long)ceil((double)(now - s_vb.anchor) / s_vb.period);
    long long t;
    if (s_vb.next_j == 0) {                         /* first call after a break: this frame goes now, the grid starts from its vblank */
        t = now; s_vb.next_j = jn + k;
    } else {
        t = vb_time(s_vb.next_j) - lead;
        if (t >= now) s_vb.next_j += k;             /* early: held until its slot */
        else if (vb_time(s_vb.next_j) > now) { t = now; s_vb.next_j += k; s_vb.late++; }   /* past the slot but the vblank is still ahead: may still make it */
        else { t = now; s_vb.next_j = jn + k; s_vb.late++; s_vb.relock++; }                /* the vblank has gone: no hold (a game slower than the grid is never slowed), the grid restarts here */
    }
    s_vb.locked++;
    *target = t; *period_ticks = pk;
    if (log_on > 0) {
        if (!s_vb.rep0) s_vb.rep0 = now;
        if (now - s_vb.rep0 > (long long)(f * 5.0)) {
            fprintf(stderr, "[vblank-lock] grid %.4f ms (%.3f Hz), %lld refresh(s) per flip, lead %.1f ms: %u locked, %u late, %u relocks, %u unlocked\n",
                    d * 1e3, 1.0 / d, k, (double)lead / f * 1e3, s_vb.locked, s_vb.late, s_vb.relock, s_vb.unlocked_calls);
            s_vb.locked = s_vb.late = s_vb.relock = s_vb.unlocked_calls = 0; s_vb.rep0 = now;
        }
    }
    return 1;
}
#else
int rsx_vblank_lock_on(void) { return 0; }
void rsx_vblank_set_vsync(int on) { (void)on; }
int rsx_vblank_snapshot(double* p, long long* v, unsigned long long* c) { (void)p; (void)v; (void)c; return 0; }
int rsx_vblank_next_release(double w, long long n, long long* t, long long* p) { (void)w; (void)n; (void)t; (void)p; return 0; }
#endif
