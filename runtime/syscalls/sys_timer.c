/*
 * ps3recomp - Timer and time syscalls (implementation)
 */

#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE /* tm_gmtoff on glibc */
#endif
#include "sys_timer.h"
#include <time.h>
#include "sys_event.h"
#include "../memory/vm.h"
#include <stdlib.h>   /* getenv -- an implicit decl returns int, truncating the pointer */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/* Environment switches are read where they are used, some of them per call:
 * answer from the pointer-keyed cache (ps3emu/env_cache.h). */
#include "ps3emu/env_cache.h"
#define getenv(name) ps3_env(name)

/* ---------------------------------------------------------------------------
 * Globals
 * -----------------------------------------------------------------------*/
sys_timer_info g_sys_timers[SYS_TIMER_MAX];

#ifdef _WIN32
static LARGE_INTEGER s_qpc_freq;
static int           s_qpc_init = 0;
static uint64_t      s_tb_mult;   /* PS3_TIMEBASE_FREQ * 2^32 / qpc freq: the guest timebase without a division per mftb */

static void ensure_qpc_init(void)
{
    if (!s_qpc_init) {
        QueryPerformanceFrequency(&s_qpc_freq);
        s_tb_mult = (uint64_t)(((unsigned __int128)PS3_TIMEBASE_FREQ << 32) / (uint64_t)s_qpc_freq.QuadPart);
        s_qpc_init = 1;
    }
}

int64_t lv2_usec_deadline(uint64_t usec)
{
    LARGE_INTEGER now;
    ensure_qpc_init();
    QueryPerformanceCounter(&now);
    return now.QuadPart +
        (int64_t)((usec * (uint64_t)s_qpc_freq.QuadPart) / 1000000ULL);
}

int lv2_deadline_passed(int64_t deadline)
{
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return now.QuadPart >= deadline;
}
#endif

/* ---------------------------------------------------------------------------
 * PPU timebase (mftb/mftbu) -- THE guest clock.
 *
 * One global monotonic counter scaled to the PS3 timebase (79.8 MHz),
 * anchored at first use. The old lifter emission was a per-call-site
 * static that advanced 16667 ticks PER READ -- not a clock at all: every
 * guest timing loop (media pacers, throttles, profilers) computed garbage
 * from it, and each call site had a PRIVATE counter that only moved when
 * polled. Overflow-safe split multiply (rem < qpf, so
 * rem * 79.8e6 < ~8e14 << 2^63).
 * -----------------------------------------------------------------------*/
/* TOD_DT_DIAG=1: per-thread count of timebase reads + sampled cost (TSC cycles per read, every 64th call);
 * dt_diag_report() (called from the host flip) prints deltas every 5 s. */
#if defined(__GNUC__)
#include <x86intrin.h>
#endif
#define DT_MAX_THR 48
struct dt_thr { uint64_t calls, cyc, cyc_n, last_calls; unsigned tid; };
static struct dt_thr* g_dt_thr[DT_MAX_THR]; static int g_dt_nthr;
static __thread struct dt_thr* tl_dt;
volatile long long g_dt_cnt[4];   /* [0] sys_time_get_system_time  [1] cellGcmGetVBlankCount  [2] cellGcmGetLastFlipTime  [3] cellGcmGetFlipStatus */
int g_dt_on = -1;
static uint64_t ppu_timebase_impl(void);
static struct dt_thr* dt_reg(void)
{
    struct dt_thr* t = (struct dt_thr*)calloc(1, sizeof *t);
    t->tid = (unsigned)GetCurrentThreadId();
    int i = __atomic_fetch_add(&g_dt_nthr, 1, __ATOMIC_RELAXED);
    if (i >= DT_MAX_THR) { __atomic_fetch_sub(&g_dt_nthr, 1, __ATOMIC_RELAXED); return t; }
    g_dt_thr[i] = t; tl_dt = t; return t;
}
/* TOD_DT_DIAG=2: one-shot memory micro-benchmark at the first timebase read (process start). */
static double dt_chase(size_t bytes, size_t steps)
{
    const size_t n = bytes / 64; uint32_t* nx = (uint32_t*)VirtualAlloc(NULL, n * 64, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!nx) return -1;
    uint32_t* perm = (uint32_t*)malloc(n * sizeof(uint32_t)); for (size_t i = 0; i < n; i++) perm[i] = (uint32_t)i;
    uint64_t r = 88172645463325252ull;
    for (size_t i = n - 1; i > 0; i--) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; size_t j = r % (i + 1); uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    for (size_t i = 0; i < n; i++) nx[(size_t)perm[i] * 16] = perm[(i + 1) % n];
    uint32_t c = perm[0]; LARGE_INTEGER f, a, b; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&a);
    for (size_t i = 0; i < steps; i++) c = nx[(size_t)c * 16];
    QueryPerformanceCounter(&b);
    volatile uint32_t sink = c; (void)sink; free(perm); VirtualFree(nx, 0, MEM_RELEASE);
    return (double)(b.QuadPart - a.QuadPart) * 1e9 / (double)f.QuadPart / (double)steps;
}
static void dt_membench(void)
{
    const double a = dt_chase(1u << 20, 4000000), b = dt_chase(16u << 20, 4000000), c = dt_chase(256u << 20, 4000000);
    const size_t mb = 64u << 20; char* s = (char*)VirtualAlloc(NULL, 2 * mb, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    double gbs = -1;
    if (s) { memset(s, 1, 2 * mb); LARGE_INTEGER f, t0, t1; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t0);
        for (int k = 0; k < 8; k++) memcpy(s + mb, s, mb);
        QueryPerformanceCounter(&t1); gbs = 8.0 * mb / ((double)(t1.QuadPart - t0.QuadPart) / f.QuadPart) / 1e9; VirtualFree(s, 0, MEM_RELEASE); }
    fprintf(stderr, "[dtmb] cpu=%u chase 1MB=%.2f ns 16MB=%.2f ns 256MB=%.2f ns memcpy=%.1f GB/s\n", (unsigned)GetCurrentProcessorNumber(), a, b, c, gbs);
}
/* TOD_DT_DIAG=3: also bucket the timebase reads by host return address (symbol via dbghelp in the 5 s report). */
static struct { uintptr_t ra; uint64_t n, last, busy; } g_dt_ra[512]; static int g_dt_mode3;
static struct { uintptr_t key, f0, f1; uint64_t n; } g_dt_c2[256];
static void dt_ra_note(uintptr_t ra)
{
    unsigned h = (unsigned)((ra >> 2) ^ (ra >> 11)) & 511;
    const uint64_t now = __rdtsc();
    for (int k = 0; k < 512; k++, h = (h + 1) & 511) {
        if (g_dt_ra[h].ra == ra) { g_dt_ra[h].n++; const uint64_t d = now - g_dt_ra[h].last; if (d < 20000) g_dt_ra[h].busy += d; g_dt_ra[h].last = now; return; }
        if (!g_dt_ra[h].ra) { g_dt_ra[h].ra = ra; g_dt_ra[h].n = 1; g_dt_ra[h].last = now; return; } }
}
static void dt_ra_report(unsigned nf)
{
    typedef struct { unsigned long SizeOfStruct, TypeIndex; unsigned long long Reserved[2]; unsigned long Index, Size; unsigned long long ModBase; unsigned long Flags; unsigned long long Value, Address; unsigned long Register, Scope, Tag, NameLen, MaxNameLen; char Name[256]; } SYMI;
    typedef int (__stdcall *fInit)(HANDLE, const char*, int); typedef int (__stdcall *fFrom)(HANDLE, unsigned long long, unsigned long long*, SYMI*);
    static HMODULE dh; static fInit pInit; static fFrom pFrom; static int ok = -1;
    if (ok < 0) { dh = LoadLibraryA("dbghelp.dll"); if (dh) { pInit = (fInit)GetProcAddress(dh, "SymInitialize"); pFrom = (fFrom)GetProcAddress(dh, "SymFromAddr"); }
                  ok = (pInit && pFrom && pInit(GetCurrentProcess(), NULL, 1)); }
    for (int k = 0; k < 6; k++) {
        int b = -1; for (int j = 0; j < 512; j++) if (g_dt_ra[j].n && (b < 0 || g_dt_ra[j].n > g_dt_ra[b].n)) b = j;
        if (b < 0) break;
        char nm[300] = "?"; unsigned long long disp = 0;
        if (ok) { SYMI s; memset(&s, 0, sizeof s); s.SizeOfStruct = 88; s.MaxNameLen = 255; if (pFrom(GetCurrentProcess(), g_dt_ra[b].ra, &disp, &s)) snprintf(nm, sizeof nm, "%s+0x%llx", s.Name, disp); }
        fprintf(stderr, "[dt-ra] %.0f/flip, loop-busy %.2f ms/flip (<5.5us gaps) %s\n", (double)g_dt_ra[b].n / nf, (double)g_dt_ra[b].busy / 3.59e6 / nf, nm);
        g_dt_ra[b].n = 0; g_dt_ra[b].busy = 0;
    }
    for (int j = 0; j < 512; j++) g_dt_ra[j].n = 0;
    for (int k = 0; k < 8; k++) {   /* (callee, caller) pairs, 1 sample in 16 reads */
        int b = -1; for (int j = 0; j < 256; j++) if (g_dt_c2[j].n && (b < 0 || g_dt_c2[j].n > g_dt_c2[b].n)) b = j;
        if (b < 0) break;
        char n0[200] = "?", n1[200] = "?"; unsigned long long d0 = 0, d1 = 0;
        if (ok) { SYMI s; memset(&s, 0, sizeof s); s.SizeOfStruct = 88; s.MaxNameLen = 255;
                  if (pFrom(GetCurrentProcess(), g_dt_c2[b].f0, &d0, &s)) snprintf(n0, sizeof n0, "%s", s.Name);
                  memset(&s, 0, sizeof s); s.SizeOfStruct = 88; s.MaxNameLen = 255;
                  if (pFrom(GetCurrentProcess(), g_dt_c2[b].f1, &d1, &s)) snprintf(n1, sizeof n1, "%s+0x%llx", s.Name, d1); }
        fprintf(stderr, "[dt-ra2] %.0f/flip %s <- %s\n", 16.0 * (double)g_dt_c2[b].n / nf, n0, n1);
        g_dt_c2[b].n = 0;
    }
    for (int j = 0; j < 256; j++) g_dt_c2[j].n = 0;
}
uint64_t ppu_timebase_now(void)
{
    if (g_dt_on < 0) { const char* e = getenv("TOD_DT_DIAG"); g_dt_on = (e && e[0] >= '1'); g_dt_mode3 = (e && e[0] == '3'); if (e && e[0] == '2') dt_membench(); }
    if (__builtin_expect(g_dt_on, 0)) {
        struct dt_thr* t = tl_dt ? tl_dt : dt_reg();
        t->calls++;
        if (g_dt_mode3) {
            dt_ra_note((uintptr_t)__builtin_return_address(0));
            if ((t->calls & 15) == 0) {   /* caller chain by unwind tables: bucket (frame0, frame1) */
                void* bt[3] = { 0, 0, 0 };
                if (RtlCaptureStackBackTrace(1, 3, bt, NULL) >= 2) {
                    const uintptr_t key = (uintptr_t)bt[1] ^ ((uintptr_t)bt[0] << 1);
                    unsigned h = (unsigned)((key >> 2) ^ (key >> 10)) & 255;
                    for (int k = 0; k < 256; k++, h = (h + 1) & 255) {
                        if (g_dt_c2[h].key == key) { g_dt_c2[h].n++; break; }
                        if (!g_dt_c2[h].key) { g_dt_c2[h].key = key; g_dt_c2[h].f0 = (uintptr_t)bt[0]; g_dt_c2[h].f1 = (uintptr_t)bt[1]; g_dt_c2[h].n = 1; break; }
                    }
                }
            }
        }
        if ((t->calls & 63) == 0) {
            const unsigned long long a = __rdtsc(); const uint64_t r = ppu_timebase_impl(); t->cyc += __rdtsc() - a; t->cyc_n++;
            return r;
        }
    }
    return ppu_timebase_impl();
}
void dt_diag_report(unsigned flips_now)
{
    static unsigned long long t_last, tsc_last; static unsigned flips_last; static long long c_last[4];
    if (g_dt_on < 0) { const char* e = getenv("TOD_DT_DIAG"); g_dt_on = (e && e[0] >= '1'); g_dt_mode3 = (e && e[0] == '3'); if (e && e[0] == '2') dt_membench(); }
    if (!g_dt_on) return;
    extern unsigned long long ps3_qpc_us(void);
    const unsigned long long now = ps3_qpc_us(), tsc = __rdtsc();
    if (!t_last) { t_last = now; tsc_last = tsc; flips_last = flips_now; return; }
    if (now - t_last < 5000000ull) return;
    const double us = (double)(now - t_last), ghz = (double)(tsc - tsc_last) / us / 1000.0;
    const unsigned nf = flips_now - flips_last ? flips_now - flips_last : 1;
    fprintf(stderr, "[dt] t=%.0fs flips=%u tsc=%.2fGHz sysT=%lld vbl=%lld lastflip=%lld flipstat=%lld |",
            now / 1e6, nf, ghz, g_dt_cnt[0] - c_last[0], g_dt_cnt[1] - c_last[1], g_dt_cnt[2] - c_last[2], g_dt_cnt[3] - c_last[3]);
    for (int k = 0; k < 4; k++) c_last[k] = g_dt_cnt[k];
    const int n = g_dt_nthr < DT_MAX_THR ? g_dt_nthr : DT_MAX_THR;
    for (int i = 0; i < n; i++) {
        struct dt_thr* t = g_dt_thr[i]; const uint64_t c = t->calls, d = c - t->last_calls; t->last_calls = c;
        if (d < 1000) continue;
        fprintf(stderr, " [tid%u %.0f tb/flip, %.0f cyc/call (%.1f ns)]", t->tid, (double)d / nf,
                t->cyc_n ? (double)t->cyc / t->cyc_n : 0.0, t->cyc_n ? (double)t->cyc / t->cyc_n / ghz : 0.0);
        t->cyc = 0; t->cyc_n = 0;
    }
    fprintf(stderr, "\n");
    if (g_dt_mode3) dt_ra_report(nf);
    t_last = now; tsc_last = tsc; flips_last = flips_now;
}
#if defined(_WIN32) && defined(__x86_64__)
#include <intrin.h>
/* ---------------------------------------------------------------------------
 * PS3_TB_TSC=1 (perf115, default OFF): the guest timebase off the TSC instead of a QueryPerformanceCounter call per read.
 * Idea (and the shape of the calibration) adapted from RACiT's runtime patch (MIT, (c) 2026 HighDrexler;
 * ps3recomp-0001-racacit-runtime-fixes P:31528-31620). The guest reads the timebase ~100k times a frame (mftb spin polls,
 * profiler scopes); a QPC call is ~25-40 ns, an rdtsc ~7 ns. With an invariant TSC (CPUID 0x80000007 EDX bit 8, what QPC runs
 * on) the QPC reading is ESTIMATED as qpc0 + (tsc - tsc0) * mul >> 48, with mul = QPC ticks per TSC tick measured over a
 * baseline that only grows (calibrated after the first 50 ms, then re-anchored on the real QPC once a second). The estimate
 * feeds the SAME path as before (ps3_guest_ticks: pause/hold, then the 32.32 multiply), so the guest clock keeps its epoch.
 * Never goes backwards: a re-anchor starts from max(estimate, QPC), the slope is positive, and a thread that read its TSC just
 * before another re-anchored (tsc < tsc0) sees a zero delta, not a wrapped one.
 * PS3_TB_CHECK=1: at every re-anchor "[tb] re-anchor: ..." (how far the TSC estimate had drifted from the real QPC, in QPC
 * ticks and ns; the TSC rate) and per-thread monotonicity (reads where the value went backwards, worst step) every 5 s.
 * PS3_TB_TSC=0 / unset = a QPC call per read, exactly as before. */
typedef struct { uint64_t tsc0, qpc0, mul; } tb_tsc_anchor;   /* qpc = qpc0 + ((tsc - tsc0) * mul >> 48) */
static tb_tsc_anchor s_tsa[2];
static volatile long s_tsa_cur = -1;       /* -1: not calibrated yet */
static volatile long s_tsa_busy;
static volatile uint64_t s_tsa_next;       /* TSC value of the next re-anchor */
static uint64_t s_tsa_cal_tsc, s_tsa_cal_qpc;
static int s_tb_tsc = -1;                  /* -1 undecided, 0 off, 1 on */
static int s_tb_chk = -1;
static volatile long s_tb_back_n; static volatile long long s_tb_back_worst;   /* PS3_TB_CHECK: per-thread backwards steps */

static inline int64_t tsa_at(const tb_tsc_anchor* a, uint64_t t)
{
    const int64_t dt = (int64_t)(t - a->tsc0);
    if (dt <= 0) return (int64_t)a->qpc0;
    return (int64_t)(a->qpc0 + (uint64_t)(((unsigned __int128)(uint64_t)dt * a->mul) >> 48));
}

static int64_t tsa_slow(void)
{
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    const uint64_t t = __rdtsc();
    const uint64_t qpc = (uint64_t)now.QuadPart;
    const long cur = s_tsa_cur;
    if (InterlockedCompareExchange(&s_tsa_busy, 1, 0) != 0)          /* another thread is re-anchoring */
        return cur >= 0 ? tsa_at(&s_tsa[cur], t) : (int64_t)qpc;
    uint64_t r = qpc;
    if (!s_tsa_cal_tsc) { s_tsa_cal_tsc = t; s_tsa_cal_qpc = qpc; }
    else {
        const uint64_t dq = qpc - s_tsa_cal_qpc, qf = (uint64_t)s_qpc_freq.QuadPart, dts = t - s_tsa_cal_tsc;
        if (dq >= qf / 20u && dts) {
            const uint64_t mul = (uint64_t)((double)dq / (double)dts * 281474976710656.0);   /* QPC ticks per TSC tick, 2^-48 (double: no __udivti3 in this link) */
            if (cur >= 0) {
                const uint64_t est = (uint64_t)tsa_at(&s_tsa[cur], t);
                if (s_tb_chk > 0)
                    fprintf(stderr, "[tb] re-anchor: TSC estimate %+lld QPC ticks (%+.0f ns) from the real QPC; TSC %.4f GHz; baseline %.1f s%c",
                            (long long)(est - qpc), ((double)(long long)(est - qpc)) * 1e9 / (double)qf,
                            (double)dts / ((double)dq / (double)qf) / 1e9, (double)dq / (double)qf, 10);
                if (est > r) r = est;
            }
            const long nxt = cur >= 0 ? 1 - cur : 0;
            s_tsa[nxt].tsc0 = t; s_tsa[nxt].qpc0 = r; s_tsa[nxt].mul = mul;
            s_tsa_next = t + (uint64_t)((double)dts * (double)qf / (double)dq);   /* one second on */
            InterlockedExchange(&s_tsa_cur, nxt);
        }
    }
    InterlockedExchange(&s_tsa_busy, 0);
    return (int64_t)r;
}

static int tb_tsc_decide(void)
{
    const char* e = getenv("PS3_TB_TSC");
    int on = e && e[0] && e[0] != '0';
    if (on) { int r[4] = { 0, 0, 0, 0 }; __cpuid(r, 0x80000007); on = (r[3] & (1 << 8)) != 0;
              if (!on) fprintf(stderr, "[tb] PS3_TB_TSC=1 ignored: no invariant TSC\n"); }
    { const char* c = getenv("PS3_TB_CHECK"); s_tb_chk = (c && c[0] && c[0] != '0') ? 1 : 0; }
    s_tb_tsc = on;
    if (on) fprintf(stderr, "[tb] timebase from the invariant TSC (PS3_TB_TSC=1), calibrated against QPC%s\n", s_tb_chk > 0 ? " (PS3_TB_CHECK=1)" : "");
    return on;
}

/* The host clock in QPC ticks. */
static inline int64_t tb_host_qpc(void)
{
    int on = s_tb_tsc;
    if (__builtin_expect(on < 0, 0)) on = tb_tsc_decide();
    if (on) {
        /* Per thread, never backwards: a re-anchor can change the slope by a few ppm while another thread still reads the old
         * anchor, which could step one reader back by a tick (100 ns); the clamp makes it exact. */
        static __thread int64_t last;
        int64_t q;
        const long cur = s_tsa_cur;
        const uint64_t t = cur >= 0 ? __rdtsc() : 0;
        q = (cur >= 0 && t < s_tsa_next) ? tsa_at(&s_tsa[cur], t) : tsa_slow();
        if (q < last) q = last; else last = q;
        return q;
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return now.QuadPart;
}
#elif defined(_WIN32)
static inline int64_t tb_host_qpc(void)
{
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return now.QuadPart;
}
#endif

static uint64_t ppu_timebase_impl(void)
{
#ifdef _WIN32
    static LONGLONG t0 = 0;
    ensure_qpc_init();
    const int64_t q = tb_host_qpc();
    if (!t0) t0 = q;   /* benign race: same anchor either way */
    const int64_t g = ps3_guest_ticks(q);
    uint64_t d = g > t0 ? (uint64_t)(g - t0) : 0;
    /* d * F / q as a 32.32 fixed-point multiply (two 64-bit divisions were most of this function's own cost;
     * the mftb spin-polls of the guest call it ~100k times per frame). Error < d * 2^-32 / 8 ticks: nothing. */
    const uint64_t tb = (uint64_t)(((unsigned __int128)d * s_tb_mult) >> 32);
#if defined(__x86_64__)
    if (__builtin_expect(s_tb_chk > 0, 0)) {   /* PS3_TB_CHECK: this thread's reads must not go backwards */
        static __thread uint64_t last;
        if (tb < last) { InterlockedIncrement(&s_tb_back_n); const long long w = (long long)(last - tb);
                         if (w > s_tb_back_worst) s_tb_back_worst = w; }
        last = tb;
        static volatile uint64_t next_rep;
        const uint64_t nowt = __rdtsc();
        if (nowt > next_rep && InterlockedExchange64((volatile LONG64*)&next_rep, (LONG64)(nowt + 18000000000ull)) < (LONG64)nowt) {
            const long n = s_tb_back_n;
            fprintf(stderr, "[tb] monotonic check: %ld backwards reads so far, worst step %lld ticks\n", n, s_tb_back_worst);
        }
    }
#endif
    return tb;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const uint64_t ns = (uint64_t)ps3_guest_ticks((int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec);
    return (ns / 1000000000ull) * PS3_TIMEBASE_FREQ +
           (ns % 1000000000ull) * PS3_TIMEBASE_FREQ / 1000000000ull;
#endif
}

/* Guest time = host time - the time spent paused, held at s_guest_hold while
 * paused. Readers load the hold before the paused total and the resume
 * stores the total before clearing the hold, so a reader racing the resume
 * still clamps to the hold: the clocks never step back. */
static int64_t s_guest_paused;               /* host ticks spent paused */
static int64_t s_guest_hold = INT64_MAX;     /* while paused: the guest ticks the clocks hold at */

int64_t ps3_guest_ticks(int64_t host)
{
    const int64_t hold = __atomic_load_n(&s_guest_hold, __ATOMIC_ACQUIRE);
    const int64_t t = host - __atomic_load_n(&s_guest_paused, __ATOMIC_ACQUIRE);
    return t < hold ? t : hold;
}

void ps3_guest_clock_pause(int paused)
{
#ifdef _WIN32
    LARGE_INTEGER c;
    ensure_qpc_init();
    QueryPerformanceCounter(&c);
    const int64_t now = c.QuadPart, ms = s_qpc_freq.QuadPart / 1000;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const int64_t now = (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec, ms = 1000000;
#endif
    const int64_t hold = __atomic_load_n(&s_guest_hold, __ATOMIC_ACQUIRE);
    if (paused && hold == INT64_MAX) {
        /* A millisecond on: a reader that took its host time just before this
         * and has not seen the hold yet cannot pass it, then step back. */
        __atomic_store_n(&s_guest_hold, now + ms - s_guest_paused, __ATOMIC_RELEASE);
    } else if (!paused && hold != INT64_MAX) {
        __atomic_store_n(&s_guest_paused, now - hold, __ATOMIC_RELEASE);   /* on from the hold */
        __atomic_store_n(&s_guest_hold, INT64_MAX, __ATOMIC_RELEASE);
    }
}

static void write_be32(uint32_t addr, uint32_t val)
{
    uint32_t* p = (uint32_t*)vm_to_host(addr);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ || defined(_WIN32)
    val = ((val >> 24) & 0xFF) | ((val >> 8) & 0xFF00) |
          ((val <<  8) & 0xFF0000) | ((val << 24) & 0xFF000000u);
#endif
    *p = val;
}

static void write_be64(uint32_t addr, uint64_t val)
{
    uint64_t* p = (uint64_t*)vm_to_host(addr);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ || defined(_WIN32)
    val = ((val >> 56) & 0xFFULL) |
          ((val >> 40) & 0xFF00ULL) |
          ((val >> 24) & 0xFF0000ULL) |
          ((val >>  8) & 0xFF000000ULL) |
          ((val <<  8) & 0xFF00000000ULL) |
          ((val << 24) & 0xFF0000000000ULL) |
          ((val << 40) & 0xFF000000000000ULL) |
          ((val << 56) & 0xFF00000000000000ULL);
#endif
    *p = val;
}

/* ---------------------------------------------------------------------------
 * sys_timer_usleep
 *
 * r3 = microseconds
 * -----------------------------------------------------------------------*/
/* Called on every guest usleep before it sleeps, when set, with the guest
 * return address and the requested time. A guest thread polling with usleep
 * is usually waiting on something another host thread must produce -- the
 * port's FIFO walker writing a fence label -- so the hook can wake that
 * thread, and for a poll it recognises it may do the waiting itself (until
 * whatever it waits on has had a chance to change, at most `usec`): it then
 * returns nonzero and the full sleep is skipped. */
int (*g_lv2_usleep_hook)(uint32_t lr, uint64_t usec) = 0;

/* Called first on every guest usleep, when set: the title may use the time
 * the guest asked to sleep (run deferred work on this thread, ctx as it is)
 * and shorten `*usec` by what that took. */
void (*g_lv2_usleep_pre)(ppu_context* ctx, uint64_t* usec) = 0;

/* Windows: sleeps from this many microseconds up to 1 ms wait on a timer
 * and spin only their end (DOD3_AB=usleepspin moves it past 1000). */
int g_usleep_spin_below = 500;
#ifndef PS3_PRECISE_DEFAULT
#define PS3_PRECISE_DEFAULT 0   /* perf108: PS3_PRECISE_USLEEP=1 turns on the main's precise short polls */
#endif

/* TOD_PIPE_DIAG=1: guest-side wait accounting for the per-flip [pipe] line (rsx_draw_engine.c).
 * [0] = RSX sync wait (lr in the title's func_005xxxxx fence code), [1] = frame limiter (func_00692D40). */
volatile long long g_pipe_usl_us[2]; volatile long g_pipe_usl_n[2];
static int64_t sys_timer_usleep_impl(ppu_context* ctx);
int64_t sys_timer_usleep(ppu_context* ctx)
{
    { extern volatile int g_tl_on; extern void tl_usleep(int, uint64_t, uint32_t, uint32_t);   /* TOD_TIMELINE */
      if (g_tl_on) { const uint64_t us = LV2_ARG_U64(ctx, 0);
          { /* fence38: which cell does the title poll? 48C0A8 = spin534EC0 (cell in r31), 5BD1C8 = spin614FF0 (byte 0x10690080) */
            extern void tl_f38(int, uint32_t, uint64_t, uint64_t);
            const uint32_t lr0 = (uint32_t)ctx->lr;
            if (lr0 == 0x48C0A8u && vm_base) { const uint32_t ea0 = (uint32_t)ctx->gpr[31]; uint64_t q; memcpy(&q, vm_base + ea0, 8);
                tl_f38(3, lr0, ea0, __builtin_bswap64(q)); }
            else if (lr0 == 0x5BD1C8u && vm_base) tl_f38(3, lr0, 0x10690080u, vm_base[0x10690080u]); }
          tl_usleep(0, us, (uint32_t)ctx->lr, (uint32_t)ctx->thread_id);
          const int64_t r = sys_timer_usleep_impl(ctx); tl_usleep(1, us, (uint32_t)ctx->lr, (uint32_t)ctx->thread_id); return r; } }
    static int diag = -1;
    if (diag < 0) { const char* e = getenv("TOD_PIPE_DIAG"); diag = (e && e[0] == '1'); }
    if (!diag) return sys_timer_usleep_impl(ctx);
    extern unsigned long long ps3_qpc_us(void);
    const uint32_t lr = (uint32_t)ctx->lr;
    const int b = (lr >= 0x500000u && lr < 0x510000u) ? 0 : (lr >= 0x692000u && lr < 0x693000u) ? 1 : -1;
    const unsigned long long t0 = ps3_qpc_us();
    const int64_t r = sys_timer_usleep_impl(ctx);
    if (b >= 0) { g_pipe_usl_us[b] += (long long)(ps3_qpc_us() - t0); g_pipe_usl_n[b]++; }
    return r;
}
/* PS3_USLEEP_ACT=1 (perf108): requested vs ACTUAL duration of every guest usleep, per (caller lr, thread), every 5 s on
 * stderr: calls, average asked/actual us, and a histogram of the actual time (<40,<80,<160,<320,<640,<1280,<2560,more us).
 * Unlocked counters (a diagnostic). Off: one load+branch. */
#ifdef _WIN32
static int64_t sys_timer_usleep_impl0(ppu_context* ctx);
static int64_t sys_timer_usleep_impl(ppu_context* ctx)
{
    static int act = -1;
    if (act < 0) { const char* e = getenv("PS3_USLEEP_ACT"); act = (e && e[0] >= '1') ? 1 : 0; }
    if (__builtin_expect(!act, 1)) return sys_timer_usleep_impl0(ctx);
    ensure_qpc_init();
    enum { AH = 64, NB = 8 };
    static uint32_t a_lr[AH], a_tid[AH]; static uint64_t a_n[AH], a_ask[AH], a_act[AH]; static uint64_t a_h[AH][NB];
    static volatile int a_used; static long long win;
    const uint64_t asked = LV2_ARG_U64(ctx, 0);
    const uint32_t lr = (uint32_t)ctx->lr, tid = (uint32_t)ctx->thread_id;
    LARGE_INTEGER q0, q1; QueryPerformanceCounter(&q0);
    const int64_t r = sys_timer_usleep_impl0(ctx);
    QueryPerformanceCounter(&q1);
    const double us = (double)(q1.QuadPart - q0.QuadPart) * 1e6 / (double)s_qpc_freq.QuadPart;
    int i = 0, used = a_used;
    for (; i < used; i++) if (a_lr[i] == lr && a_tid[i] == tid) break;
    if (i == used && used < AH) { a_lr[used] = lr; a_tid[used] = tid; a_used = used + 1; }
    if (i < AH) {
        int b = 0; { double lim = 40.0; while (b < NB - 1 && us >= lim) { b++; lim *= 2.0; } }
        a_n[i]++; a_ask[i] += asked; a_act[i] += (uint64_t)us; a_h[i][b]++;
    }
    if (q1.QuadPart - win > 5 * s_qpc_freq.QuadPart) {
        static volatile long busy;
        if (win && !InterlockedExchange(&busy, 1)) {
            fprintf(stderr, "[usleep-act] 5 s, top (lr, tid) by total actual us:\n");
            for (int k = 0; k < 10; k++) {
                int best = -1;
                for (int j = 0; j < AH && j < a_used; j++) if (a_n[j] && (best < 0 || a_act[j] > a_act[best])) best = j;
                if (best < 0) break;
                fprintf(stderr, "  lr=0x%08X tid=%u n=%llu asked_avg=%.0f actual_avg=%.0f total_ms=%.1f hist[<40,80,160,320,640,1280,2560,more]=",
                        a_lr[best], a_tid[best], (unsigned long long)a_n[best], (double)a_ask[best] / (double)a_n[best],
                        (double)a_act[best] / (double)a_n[best], (double)a_act[best] / 1000.0);
                for (int h = 0; h < NB; h++) fprintf(stderr, "%s%llu", h ? "," : "", (unsigned long long)a_h[best][h]);
                fprintf(stderr, "\n");
                a_n[best] = 0; a_ask[best] = 0; a_act[best] = 0; for (int h = 0; h < NB; h++) a_h[best][h] = 0;
            }
            for (int j = 0; j < AH; j++) { a_n[j] = 0; a_ask[j] = 0; a_act[j] = 0; for (int h = 0; h < NB; h++) a_h[j][h] = 0; }
            fflush(stderr);
            InterlockedExchange(&busy, 0);
        }
        win = q1.QuadPart;
    }
    return r;
}
#define sys_timer_usleep_impl sys_timer_usleep_impl0
#else
static int64_t sys_timer_usleep_impl(ppu_context* ctx);
#endif
static int64_t sys_timer_usleep_impl(ppu_context* ctx)
{
    uint64_t usec = LV2_ARG_U64(ctx, 0);
    { extern void ps3_cpu_note(void); ps3_cpu_note(); }
    if (g_lv2_usleep_pre) g_lv2_usleep_pre(ctx, &usec);
    if (g_lv2_usleep_hook && g_lv2_usleep_hook((uint32_t)ctx->lr, usec)) return CELL_OK;
    /* TOD_USLEEP_HIST=1: every 5 s, the guest usleeps by caller LR and requested time (log2 bucket of us), top 16.
     * Unlocked counters: a diagnostic, a lost increment under a race changes nothing. */
    { static int on = -1;
      if (on < 0) { const char* e = getenv("TOD_USLEEP_HIST"); on = (e && e[0] >= '1'); }
      if (__builtin_expect(on, 0)) {
        enum { H = 512 };
        static uint32_t k_lr[H]; static uint8_t k_b[H]; static uint32_t k_n[H]; static uint64_t k_us[H]; static uint32_t k_tid[H];
        static unsigned long long win;
        extern unsigned long long ps3_qpc_us(void);
        unsigned b = 0; { uint64_t u = usec; while (u > 1 && b < 31) { u >>= 1; b++; } }
        const uint32_t lr = (uint32_t)ctx->lr;
        unsigned h = (lr * 2654435761u + b * 40503u) >> 7 & (H - 1);
        for (int k = 0; k < H; k++, h = (h + 1) & (H - 1)) {
            if (k_n[h] && k_lr[h] == lr && k_b[h] == b) { k_n[h]++; k_us[h] += usec; break; }
            if (!k_n[h]) { k_lr[h] = lr; k_b[h] = (uint8_t)b; k_n[h] = 1; k_us[h] = usec; k_tid[h] = (uint32_t)ctx->thread_id; break; }
        }
        const unsigned long long now = ps3_qpc_us();
        if (!win) win = now;
        else if (now - win >= 5000000ull) {
            win = now;
            static volatile long busy;
            if (!InterlockedExchange(&busy, 1)) {
                fprintf(stderr, "[usleep-hist] 5 s, top callers (lr, asked us bucket [2^b,2^b+1), calls, avg asked us, tid):\n");
                for (int r = 0; r < 16; r++) {
                    int best = -1;
                    for (int j = 0; j < H; j++) if (k_n[j] && (best < 0 || k_n[j] > k_n[best])) best = j;
                    if (best < 0) break;
                    fprintf(stderr, "  lr=0x%08X us=[%u,%u) n=%u avg=%.0f tid=%u\n", k_lr[best], 1u << k_b[best], 2u << k_b[best],
                            k_n[best], (double)k_us[best] / k_n[best], k_tid[best]);
                    k_n[best] = 0;
                }
                for (int j = 0; j < H; j++) k_n[j] = 0;
                fflush(stderr);
                InterlockedExchange(&busy, 0);
            }
        }
      } }
    /* A single sleep longer than a second is almost always a computed delay
     * gone wrong (a negative interval, a timebase in the wrong units) -- and a
     * thread asleep for an hour looks exactly like a hang. Say so. */
    if (usec > 1000000ull) { static int n = 0; if (n < 32 && ++n)
        fprintf(stderr, "[timer] LONG usleep(%llu us = %.1f s) lr=0x%08llX tid=%u\n",
                (unsigned long long)usec, usec / 1e6, (unsigned long long)ctx->lr, (unsigned)ctx->thread_id); }
    /* PS3_USLEEP_LR=<hex lr>: for the loop that sleeps at that lr, every 2 s
     * the asked sleep and the work time between sleeps (avg/max, ms) -- a
     * "work; usleep(period)" producer runs late when work + sleep > period. */
    { static long long want = -1;
      if (want < 0) { const char* e = getenv("PS3_USLEEP_LR"); want = e ? (long long)strtoull(e, NULL, 16) : 0; }
      if (want && (uint32_t)ctx->lr == (uint32_t)want) {
          extern unsigned long long ps3_qpc_us(void);
          static unsigned long long last, t0; static double psum, pmax; static unsigned n2;
          unsigned long long now = ps3_qpc_us();
          if (last) { double pd = (now - last) / 1000.0; psum += pd; if (pd > pmax) pmax = pd; n2++; }
          last = now;
          if (!t0) t0 = now;
          if (now - t0 >= 2000000ull && n2) {
              fprintf(stderr, "[usleep-lr] lr=0x%08X asked %.2f ms, %u loops in %.1f s, period avg %.2f max %.2f ms (work+oversleep = period - asked)\n",
                      (uint32_t)want, usec / 1000.0, n2, (now - t0) / 1e6, psum / n2, pmax);
              t0 = now; psum = pmax = 0; n2 = 0;
          }
      } }
    { extern unsigned long long ps3_qpc_us(void);
      static int n=0; if (n < 60 && ++n)
        fprintf(stderr, "[WAIT] t=%lluus timer_usleep(%llu us) lr=0x%08llX cia=0x%08llX\n", ps3_qpc_us(),
        (unsigned long long)usec, (unsigned long long)ctx->lr, (unsigned long long)ctx->cia); }
    /* PS3_POLLTOP=<seconds>: name the usleep poll sites.
     *
     * A title that is "hung" is usually not blocked on anything -- every thread
     * is spinning on sys_timer_usleep waiting for a flag, and [BLOCKSUM] only
     * says "syscall 141, 12s". This histograms the callers over a window and
     * prints the top ones, so the answer is a guest address rather than a
     * count. Keyed on ctx->lr: a host-stack backtrace through lifted code has
     * no unwind tables and invents chains (see the Guitar Hero III writeup),
     * and lr is the only guest chain that survives.
     *
     * Deliberately unlocked: the counts are a diagnostic, and a lost increment
     * under a race cannot change which site is at the top of a 30k-per-second
     * poll. */
    { static long s_pt = -1;
      if (s_pt < 0) { const char* e = getenv("PS3_POLLTOP");
                      s_pt = e ? (long)strtoul(e, 0, 10) : 0;
                      if (s_pt < 0) s_pt = 0; }
      if (s_pt > 0) {
        enum { PT_MAX = 128 };
        static uint32_t pt_lr[PT_MAX]; static uint32_t pt_n[PT_MAX];
        static uint64_t pt_us[PT_MAX]; static int pt_used = 0;
        static uint32_t pt_tid[PT_MAX];
        static unsigned long long pt_win = 0;
        uint32_t lr = (uint32_t)ctx->lr;
        int i = 0;
        for (; i < pt_used; i++) if (pt_lr[i] == lr) break;
        if (i == pt_used && pt_used < PT_MAX) {
            pt_lr[pt_used] = lr;
            /* The thread matters as much as the site: "who is waiting" is the
             * half that says whether a poll is the engine idling or the main
             * thread stuck. */
            pt_tid[pt_used] = (uint32_t)ctx->thread_id;
            pt_used++;
        }
        if (i < PT_MAX) { pt_n[i]++; pt_us[i] += usec; }
        { extern unsigned long long ps3_qpc_us(void);
          unsigned long long now = ps3_qpc_us();
          if (!pt_win) pt_win = now;
          else if (now - pt_win >= (unsigned long long)s_pt * 1000000ull) {
            fprintf(stderr, "--- usleep poll sites (last %lds) ---\n", s_pt);
            for (int k = 0; k < 8; k++) {
              int best = -1;
              for (int j = 0; j < pt_used; j++)
                if (pt_n[j] && (best < 0 || pt_n[j] > pt_n[best])) best = j;
              if (best < 0) break;
              fprintf(stderr, "  %8u calls  %6llums slept  tid=%-3u lr=0x%08X\n",
                      pt_n[best], (unsigned long long)(pt_us[best] / 1000),
                      pt_tid[best], pt_lr[best]);
              pt_n[best] = 0;
            }
            fflush(stderr);
            for (int j = 0; j < pt_used; j++) { pt_n[j] = 0; pt_us[j] = 0; }
            pt_win = now;
          } } } }

    /* PS3_WAIT_OBJ=<lr-hex>: when a usleep spin is reached from this return
     * address, dump the registers and the object they point at. A poll loop
     * tells you WHERE it is spinning; this tells you WHAT it is spinning on,
     * which is the part you actually need to find who never releases it. */
    { static long s_wo = -1;
      if (s_wo < 0) { const char* e = getenv("PS3_WAIT_OBJ");
                      s_wo = e ? (long)strtoul(e, 0, 16) : 0; }
      if (s_wo && (uint32_t)ctx->lr == (uint32_t)s_wo) {
        static int _n = 0;
        if (_n < 8 && ++_n) {
          /* Dump the words at EVERY plausible object register, not just r29.
           * Which register holds the object is per-title -- r29 was right for
           * the title this was written for and is a spin COUNTER in Guitar
           * Hero III, where the object is in r30/r31. Printing one guess makes
           * the tool silently report the wrong memory as "the thing it waits
           * on", which is worse than printing nothing. */
          fprintf(stderr, "[wait-obj] lr=0x%08X sleep=%lluus", (uint32_t)ctx->lr,
                  (unsigned long long)usec);
          static const int regs[] = { 28, 29, 30, 31, 3 };
          for (unsigned ri = 0; ri < sizeof regs / sizeof regs[0]; ri++) {
            const int r = regs[ri];
            const uint32_t o = (uint32_t)ctx->gpr[r];
            fprintf(stderr, "  r%d=0x%08X", r, o);
            /* Only deref something that looks like a guest pointer: a small
             * integer is a count, and dumping "memory at 4" is noise. */
            if (vm_base && o >= 0x10000u) {
              fprintf(stderr, "[");
              for (int w = 0; w < 4; w++) {
                const uint8_t* q = vm_base + o + (unsigned)w * 4u;
                uint32_t v = ((uint32_t)q[0] << 24) | ((uint32_t)q[1] << 16) |
                             ((uint32_t)q[2] << 8)  | (uint32_t)q[3];
                fprintf(stderr, "%s%08X", w ? " " : "", v);
              }
              fprintf(stderr, "]");
            }
          }
          /* The label a GPU fence wait polls usually sits behind a pointer
           * in the object (DoD3's func_008B0C70: *(r30+0x50)). */
          { const uint32_t o = (uint32_t)ctx->gpr[30];
            if (vm_base && o >= 0x10000u) {
              const uint8_t* q = vm_base + o + 0x50u;
              const uint32_t pa = ((uint32_t)q[0] << 24) | ((uint32_t)q[1] << 16) | ((uint32_t)q[2] << 8) | q[3];
              uint32_t lab = 0;
              if (pa >= 0x10000u) { const uint8_t* l = vm_base + pa;
                  lab = ((uint32_t)l[0] << 24) | ((uint32_t)l[1] << 16) | ((uint32_t)l[2] << 8) | l[3]; }
              fprintf(stderr, "  *(r30+0x50)=0x%08X -> 0x%08X (r29 target=0x%08X)", pa, lab, (uint32_t)ctx->gpr[29]);
            } }
          fprintf(stderr, "\n"); fflush(stderr);
        } } }

    /* POLLSITE: resolve the host chain of the 1ms poller (LBP bringup) to
     * guest functions -- names the stage that is starving. */
    { static int _ps = -1; if (_ps < 0) _ps = getenv("POLLSITE") ? 12 : 0;
      if (_ps > 0 && usec == 1000) { _ps--;
        extern void ppu_log_host_chain(const char*);
        ppu_log_host_chain("usleep1ms"); } }

#ifdef _WIN32
    /* A high-resolution waitable timer, one per thread and kept: a plain
     * waitable timer rounds to the scheduler's clock interval (1 ms with
     * timeBeginPeriod(1), 15.6 ms otherwise, and since Windows 10 2004 the
     * 1 ms is per process and not always honoured), so a guest audio or
     * render loop's usleep(1000) could take 2-16 ms. The HIGH_RESOLUTION
     * flag (Windows 10 1803+) fires within ~0.5 ms regardless; where the
     * flag is unknown the plain timer is the fallback. Creating and closing
     * a kernel object per call was also measurable at this title's rate. */
    if (usec >= 1000) {
        static __declspec(thread) HANDLE timer = NULL;
        if (!timer) {
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
            timer = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
            if (!timer) timer = CreateWaitableTimerW(NULL, TRUE, NULL);
        }
        if (timer) {
            /* PS3_USLEEP_LOG=<ms>: a sleep that asked for, or took, more than
             * that, with its caller and the frame -- whether a long frame was
             * the guest asking to sleep or the sleep running over. */
            static long ul = -1;
            if (ul < 0) { const char* e = getenv("PS3_USLEEP_LOG"); ul = e ? atol(e) : 0; }
            LARGE_INTEGER uq0, uq1, uqf;
            if (ul > 0) { QueryPerformanceFrequency(&uqf); QueryPerformanceCounter(&uq0); }
            LARGE_INTEGER due;
            due.QuadPart = -((LONGLONG)usec * 10); /* 100ns units, negative = relative */
            SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE);
            WaitForSingleObject(timer, INFINITE);
            if (ul > 0) {
                QueryPerformanceCounter(&uq1);
                const double took = (double)(uq1.QuadPart - uq0.QuadPart) * 1000.0 / (double)uqf.QuadPart;
                if (usec > (uint64_t)ul * 1000u || took > (double)ul) {
                    extern uint32_t g_rsx_engine_frame;
                    fprintf(stderr, "[usleep] frame %u tid %u lr=0x%08X asked %.2f ms, slept %.2f ms\n",
                            g_rsx_engine_frame, (unsigned)ctx->thread_id, (uint32_t)ctx->lr, usec / 1000.0, took);
                }
            }
        } else {
            Sleep((DWORD)(usec / 1000));
        }
    } else if (usec > 0 && usec >= (uint64_t)g_usleep_spin_below) {
        /* 0.5-1 ms: on the thread's high-resolution timer for all but the
         * last 250 us, spinning only those. The audio system loop's 800 us
         * polls spun two thirds of a core for their whole length, a core a
         * 6-core machine's game, render and RSX threads need. Below
         * g_usleep_spin_below (DOD3_AB=usleepspin sets it past 1000, the old
         * all-spin behaviour) the spin below as before. */
        ensure_qpc_init();
        LARGE_INTEGER q0, qn;
        QueryPerformanceCounter(&q0);
        const int64_t deadline = q0.QuadPart + (int64_t)((usec * (uint64_t)s_qpc_freq.QuadPart) / 1000000ULL);
        static __declspec(thread) HANDLE hr = NULL;
        if (!hr) hr = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        if (hr) {
            LARGE_INTEGER due;
            due.QuadPart = -((LONGLONG)(usec - 250) * 10);
            SetWaitableTimer(hr, &due, 0, NULL, NULL, FALSE);
            WaitForSingleObject(hr, INFINITE);
        }
        do {
            SwitchToThread();
            QueryPerformanceCounter(&qn);
        } while (qn.QuadPart < deadline);
    } else if (usec > 0) {
        /* Sub-millisecond sleep. Sleep()/waitable timers round up to the
         * ~0.5-1 ms scheduler quantum, far too coarse for a guest that relies
         * on accurate microsecond pacing -- a short usleep() spin waiting on
         * another thread to make progress otherwise collapses to a single yield
         * and busy-spins at full speed (so the wait is effectively a no-op).
         * Busy-wait to a precise QPC deadline, yielding inside the spin
         * (SwitchToThread) so a sibling host thread the guest may be waiting on
         * still gets the core. Mirrors RPCS3's "Usleep Only" TSC busy-tail. */
        ensure_qpc_init();
        LARGE_INTEGER qpc_start, qpc_now;
        QueryPerformanceCounter(&qpc_start);
        const int64_t qpc_deadline = qpc_start.QuadPart +
            (int64_t)((usec * (uint64_t)s_qpc_freq.QuadPart) / 1000000ULL);
        /* Poll runs: a thread that keeps coming back with short sleeps (a mftb/usleep poll of a job group or of the
         * walker's fence: tens of thousands a second, 30-50 us each) has nothing to do but wait, and the spin took
         * a whole core -- one the parallel walker's workers need. The first usl_esc_us of such a run still spins
         * to the exact deadline, so a one-off short sleep keeps its precision; past that, each further poll waits
         * on the thread's high-resolution timer for usl_nap_us (it wakes within ~0.5 ms) and returns without the
         * spin. A gap of more than usl_gap_us between two sleeps (the guest did real work) starts a new run.
         * TOD_USLEEP_ESC=<esc us>[,<nap us>[,<gap us>]]; 0 = the old pure spin. */
        static __declspec(thread) int64_t tl_last_end, tl_run_start;
        static int64_t usl_esc_us = -1, usl_nap_us = 100, usl_gap_us = 100;
        if (usl_esc_us < 0) {
            const char* e = getenv("TOD_USLEEP_ESC");
            long long a1 = 100, a2 = 100, a3 = 100;
            if (e) sscanf(e, "%lld,%lld,%lld", &a1, &a2, &a3);
            usl_nap_us = a2; usl_gap_us = a3; usl_esc_us = a1;
        }
        /* 30/10: the nap applies to the MAIN PPU thread only (thread_id 1, the one that spends ~50% of a core
         * polling job groups and the walker). The 27/10 try applied it to every guest thread and the audio
         * synth's periodic threads woke late (one 42 ms gap at fastoon). TOD_USLEEP_ESC_ALL=1 = every thread. */
        static int usl_all = -1;
        if (usl_all < 0) { const char* e = getenv("TOD_USLEEP_ESC_ALL"); usl_all = (e && e[0] == '1') ? 1 : 0; }
        /* perf108 PS3_PRECISE_USLEEP: the main's short polls (30 us at 534EC0: 6/frame, 1.5 ms/frame; 5BD1C8) used to hit this
         * nap after 100 us of a run, and a 100 us high-resolution-timer nap takes ~0.5 ms (the timer's resolution), so the
         * guest saw the cell it polls 0.2-0.5 ms late. Precise: the run spins (_mm_pause, not SwitchToThread: that can hand the
         * core to a ready host thread for a whole quantum) up to PS3_PRECISE_RUN_US (default 1500) before the first nap, then as
         * before -- the CPU bound is that many us per wait run. PS3_PRECISE_USLEEP=0 = the old behaviour. */
        static int pr = -1; static int64_t pr_run_us = 1500; static int pr_yield = 0;
        if (pr < 0) {
            const char* e = getenv("PS3_PRECISE_USLEEP"); pr = e ? (e[0] != '0') : PS3_PRECISE_DEFAULT;
            const char* r = getenv("PS3_PRECISE_RUN_US"); if (r) pr_run_us = atoll(r);
            const char* y = getenv("PS3_PRECISE_YIELD"); pr_yield = (y && y[0] == '1');
        }
        const int64_t esc_eff = pr ? (pr_run_us > usl_esc_us ? pr_run_us : usl_esc_us) : usl_esc_us;
        if (usl_esc_us > 0 && (usl_all || ctx->thread_id == 1)) {
            const int64_t f = s_qpc_freq.QuadPart;
            if (qpc_start.QuadPart - tl_last_end > usl_gap_us * f / 1000000) tl_run_start = qpc_start.QuadPart;
            if (qpc_start.QuadPart - tl_run_start > esc_eff * f / 1000000) {
                static __declspec(thread) HANDLE nap = NULL;
                if (!nap) nap = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
                if (nap) {
                    LARGE_INTEGER due; due.QuadPart = -(LONGLONG)(usl_nap_us * 10);
                    SetWaitableTimer(nap, &due, 0, NULL, NULL, FALSE);
                    WaitForSingleObject(nap, INFINITE);
                    QueryPerformanceCounter(&qpc_now);
                    tl_last_end = qpc_now.QuadPart;
                    return CELL_OK;
                }
            }
        }
        if (pr && !pr_yield && (usl_all || ctx->thread_id == 1)) {
            do {
                _mm_pause(); _mm_pause(); _mm_pause(); _mm_pause(); _mm_pause(); _mm_pause(); _mm_pause(); _mm_pause();
                QueryPerformanceCounter(&qpc_now);
            } while (qpc_now.QuadPart < qpc_deadline);
        } else
        do {
            SwitchToThread();
            QueryPerformanceCounter(&qpc_now);
        } while (qpc_now.QuadPart < qpc_deadline);
        tl_last_end = qpc_now.QuadPart;
    }
#else
    if (usec > 0) {
        struct timespec ts;
        ts.tv_sec  = (time_t)(usec / 1000000);
        ts.tv_nsec = (long)((usec % 1000000) * 1000);
        nanosleep(&ts, NULL);
    }
#endif

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_sleep
 *
 * r3 = seconds
 * -----------------------------------------------------------------------*/
int64_t sys_timer_sleep(ppu_context* ctx)
{
    uint32_t sec = LV2_ARG_U32(ctx, 0);

#ifdef _WIN32
    Sleep(sec * 1000);
#else
    sleep(sec);
#endif

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_time_get_current_time
 *
 * r3 = pointer to receive seconds (u64*)
 * r4 = pointer to receive nanoseconds (u64*)
 *
 * The wall clock: seconds since 1970, which a title's C library turns into
 * the date it shows and stamps its saves with.
 *
 * Windows: the host's wall clock at the first reading, carried on by the
 * guest's ticks -- so it stands still while the title is paused, like the
 * guest's other clocks (sys_timer.h). It used to be the performance counter
 * alone, divided into seconds: the time since the PC was started, taken for
 * the time since 1970. A machine up for four days told the title it was the
 * 5th of January 1970.
 *
 * PS3_TIME_LOG=1: the first answer of this call and of sys_time_get_timezone,
 * once each.
 * -----------------------------------------------------------------------*/
#ifdef _WIN32
static void wall_clock_now(uint64_t* sec, uint64_t* nsec)
{
    static volatile LONG s_lock = 0;
    static volatile LONG s_have = 0;
    static int64_t  s_base_ticks;
    static uint64_t s_base_100ns;      /* 100 ns units since 1970 */
    LARGE_INTEGER now;
    ensure_qpc_init();
    QueryPerformanceCounter(&now);
    const int64_t g = ps3_guest_ticks(now.QuadPart);
    if (!s_have) {
        FILETIME ft;
        ULARGE_INTEGER u;
        GetSystemTimePreciseAsFileTime(&ft);
        u.LowPart = ft.dwLowDateTime; u.HighPart = ft.dwHighDateTime;
        while (InterlockedCompareExchange(&s_lock, 1, 0) != 0) YieldProcessor();
        if (!s_have) {
            s_base_100ns = u.QuadPart - 116444736000000000ULL;   /* 1601 -> 1970 */
            s_base_ticks = g;
            InterlockedExchange(&s_have, 1);
        }
        InterlockedExchange(&s_lock, 0);
    }
    const int64_t d = g - s_base_ticks;
    const uint64_t el = d > 0 ? (uint64_t)d : 0, q = (uint64_t)s_qpc_freq.QuadPart;
    const uint64_t t = s_base_100ns + (el / q) * 10000000ULL + ((el % q) * 10000000ULL) / q;
    *sec = t / 10000000ULL;
    *nsec = (t % 10000000ULL) * 100ULL;
}
#endif

/* LV2 144: r3/r4 point to signed BE32 minute offsets. The first is the
 * standard UTC offset; the second is the current seasonal adjustment.
 * Derive the adjustment instead of assuming every DST shift is 60 minutes. */
int64_t sys_time_get_timezone(ppu_context* ctx)
{
    uint32_t zone_addr = LV2_ARG_PTR(ctx, 0);
    uint32_t dst_addr = LV2_ARG_PTR(ctx, 1);
    if (!zone_addr || !dst_addr || zone_addr > UINT32_MAX - 3 || dst_addr > UINT32_MAX - 3)
        return (int32_t)CELL_EFAULT;

    time_t now = time(NULL);
    struct tm local;
    int32_t standard_minutes, daylight_minutes;
#ifdef _WIN32
    _tzset();
    long west_seconds, daylight_bias;
    if (localtime_s(&local, &now) || _get_timezone(&west_seconds) || _get_dstbias(&daylight_bias))
        return (int32_t)CELL_EAGAIN;
    standard_minutes = (int32_t)(-west_seconds / 60);
    daylight_minutes = local.tm_isdst > 0 ? (int32_t)(-daylight_bias / 60) : 0;
#else
    if (!localtime_r(&now, &local)) return (int32_t)CELL_EAGAIN;
    daylight_minutes = 0;
    if (local.tm_isdst > 0) {
        struct tm standard = local;
        standard.tm_isdst = 0;
        time_t standard_epoch = mktime(&standard);
        if (standard_epoch == (time_t)-1) return (int32_t)CELL_EAGAIN;
        daylight_minutes = (int32_t)(difftime(standard_epoch, now) / 60);
    }
    standard_minutes = (int32_t)(local.tm_gmtoff / 60) - daylight_minutes;
#endif
    write_be32(zone_addr, (uint32_t)standard_minutes);
    write_be32(dst_addr, (uint32_t)daylight_minutes);
    { static int said = 0;
      if (!said && (said = 1) && getenv("PS3_TIME_LOG"))
          fprintf(stderr, "[SYS] sys_time_get_timezone: %d minutes east of UTC, summer time %d minutes\n",
                  standard_minutes, daylight_minutes); }
    return CELL_OK;
}

int64_t sys_time_get_current_time(ppu_context* ctx)
{
    uint32_t sec_addr  = LV2_ARG_PTR(ctx, 0);
    uint32_t nsec_addr = LV2_ARG_PTR(ctx, 1);

    uint64_t sec, nsec;

#ifdef _WIN32
    wall_clock_now(&sec, &nsec);
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    sec  = (uint64_t)ts.tv_sec;
    nsec = (uint64_t)ts.tv_nsec;
#endif

    if (sec_addr != 0) {
        write_be64(sec_addr, sec);
    }
    if (nsec_addr != 0) {
        write_be64(nsec_addr, nsec);
    }
    { static int said = 0;
      if (!said && (said = 1) && getenv("PS3_TIME_LOG")) {
          const time_t t = (time_t)sec;
          struct tm u;
#ifdef _WIN32
          gmtime_s(&u, &t);
#else
          gmtime_r(&t, &u);
#endif
          fprintf(stderr, "[SYS] sys_time_get_current_time: %llu s since 1970 (%04d-%02d-%02d %02d:%02d:%02d UTC)\n",
                  (unsigned long long)sec, u.tm_year + 1900, u.tm_mon + 1, u.tm_mday, u.tm_hour, u.tm_min, u.tm_sec);
      } }

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_time_get_timebase_frequency
 *
 * Returns the PS3 timebase frequency in r3.
 * -----------------------------------------------------------------------*/
int64_t sys_time_get_timebase_frequency(ppu_context* ctx)
{
    (void)ctx;
    return (int64_t)PS3_TIMEBASE_FREQ;
}

/* ---------------------------------------------------------------------------
 * Periodic timer thread
 * -----------------------------------------------------------------------*/

/* Forward declaration */
static void timer_send_event(sys_timer_info* t);

#ifdef _WIN32
static DWORD WINAPI timer_thread_proc(LPVOID param)
{
    sys_timer_info* t = (sys_timer_info*)param;

    while (t->running) {
        DWORD ms = (DWORD)(t->period_usec / 1000);
        if (ms == 0) ms = 1;

        DWORD result = WaitForSingleObject(t->stop_event, ms);
        if (result == WAIT_OBJECT_0) {
            break; /* stop signaled */
        }

        if (t->running) {
            timer_send_event(t);
        }
    }

    return 0;
}
#else
static void* timer_thread_proc(void* param)
{
    sys_timer_info* t = (sys_timer_info*)param;

    pthread_mutex_lock(&t->mtx);
    while (!t->stop_flag) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec  += (time_t)(t->period_usec / 1000000);
        ts.tv_nsec += (long)((t->period_usec % 1000000) * 1000);
        if (ts.tv_nsec >= 1000000000L) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000L;
        }

        int rc = pthread_cond_timedwait(&t->cv, &t->mtx, &ts);
        if (t->stop_flag) break;

        if (rc == ETIMEDOUT) {
            pthread_mutex_unlock(&t->mtx);
            timer_send_event(t);
            pthread_mutex_lock(&t->mtx);
        }
    }
    pthread_mutex_unlock(&t->mtx);

    return NULL;
}
#endif

static void timer_send_event(sys_timer_info* t)
{
    if (t->event_queue_id <= 0 || t->event_queue_id > SYS_EVENT_QUEUE_MAX)
        return;

    sys_event_queue_info* q = &g_sys_event_queues[t->event_queue_id - 1];
    if (!q->active)
        return;

    sys_event_t evt;
    evt.source = t->source;
    evt.data1  = t->data1;
    evt.data2  = 0;
    evt.data3  = 0;

    /* Push event (ignore if queue full) */
#ifdef _WIN32
    EnterCriticalSection(&q->lock);
#else
    pthread_mutex_lock(&q->lock);
#endif

    if (q->count < q->capacity) {
        q->buffer[q->tail] = evt;
        q->tail = (q->tail + 1) % q->capacity;
        q->count++;
#ifdef _WIN32
        WakeConditionVariable(&q->not_empty);
#else
        pthread_cond_signal(&q->not_empty);
#endif
    }

#ifdef _WIN32
    LeaveCriticalSection(&q->lock);
#else
    pthread_mutex_unlock(&q->lock);
#endif
}

/* ---------------------------------------------------------------------------
 * sys_timer_create
 *
 * r3 = pointer to receive timer ID (u32*)
 * -----------------------------------------------------------------------*/
int64_t sys_timer_create(ppu_context* ctx)
{
    uint32_t id_out_addr = LV2_ARG_PTR(ctx, 0);

    int slot = -1;
    for (int i = 0; i < SYS_TIMER_MAX; i++) {
        if (!g_sys_timers[i].active) { slot = i; break; }
    }
    if (slot < 0)
        return (int64_t)(int32_t)CELL_EAGAIN;

    sys_timer_info* t = &g_sys_timers[slot];
    memset(t, 0, sizeof(*t));
    t->active  = 1;
    t->running = 0;
    t->event_queue_id = 0;

#ifdef _WIN32
    t->stop_event    = NULL;
    t->thread_handle = NULL;
    t->timer_handle  = NULL;
#else
    pthread_mutex_init(&t->mtx, NULL);
    pthread_cond_init(&t->cv, NULL);
    t->stop_flag = 0;
#endif

    uint32_t timer_id = (uint32_t)(slot + 1);
    if (id_out_addr != 0) {
        write_be32(id_out_addr, timer_id);
    }

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_destroy
 *
 * r3 = timer_id
 * -----------------------------------------------------------------------*/
int64_t sys_timer_destroy(ppu_context* ctx)
{
    uint32_t timer_id = LV2_ARG_U32(ctx, 0);

    if (timer_id == 0 || timer_id > SYS_TIMER_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_timer_info* t = &g_sys_timers[timer_id - 1];
    if (!t->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    /* Stop if running */
    if (t->running) {
        t->running = 0;
#ifdef _WIN32
        if (t->stop_event) SetEvent(t->stop_event);
        if (t->thread_handle) {
            WaitForSingleObject(t->thread_handle, 3000);
            CloseHandle(t->thread_handle);
        }
        if (t->stop_event) CloseHandle(t->stop_event);
#else
        pthread_mutex_lock(&t->mtx);
        t->stop_flag = 1;
        pthread_cond_signal(&t->cv);
        pthread_mutex_unlock(&t->mtx);
        pthread_join(t->thread, NULL);
        pthread_mutex_destroy(&t->mtx);
        pthread_cond_destroy(&t->cv);
#endif
    }

    t->active = 0;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_connect_event_queue
 *
 * r3 = timer_id
 * r4 = queue_id
 * r5 = source
 * r6 = data1
 * -----------------------------------------------------------------------*/
int64_t sys_timer_connect_event_queue(ppu_context* ctx)
{
    uint32_t timer_id = LV2_ARG_U32(ctx, 0);
    uint32_t queue_id = LV2_ARG_U32(ctx, 1);
    uint64_t source   = LV2_ARG_U64(ctx, 2);
    uint64_t data1    = LV2_ARG_U64(ctx, 3);

    if (timer_id == 0 || timer_id > SYS_TIMER_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_timer_info* t = &g_sys_timers[timer_id - 1];
    if (!t->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    if (queue_id == 0 || queue_id > SYS_EVENT_QUEUE_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    t->event_queue_id = (int32_t)queue_id;
    t->source         = source;
    t->data1          = data1;

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_disconnect_event_queue
 *
 * r3 = timer_id
 * -----------------------------------------------------------------------*/
int64_t sys_timer_disconnect_event_queue(ppu_context* ctx)
{
    uint32_t timer_id = LV2_ARG_U32(ctx, 0);

    if (timer_id == 0 || timer_id > SYS_TIMER_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_timer_info* t = &g_sys_timers[timer_id - 1];
    if (!t->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    t->event_queue_id = 0;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_start
 *
 * r3 = timer_id
 * r4 = base_time (absolute start time, 0 = now)
 * r5 = period_usec
 * -----------------------------------------------------------------------*/
int64_t sys_timer_start(ppu_context* ctx)
{
    uint32_t timer_id    = LV2_ARG_U32(ctx, 0);
    /* uint64_t base_time = LV2_ARG_U64(ctx, 1); -- ignored for now */
    uint64_t period      = LV2_ARG_U64(ctx, 2);

    if (timer_id == 0 || timer_id > SYS_TIMER_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_timer_info* t = &g_sys_timers[timer_id - 1];
    if (!t->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    if (t->running)
        return (int64_t)(int32_t)CELL_EBUSY;

    if (period == 0)
        return (int64_t)(int32_t)CELL_EINVAL;

    t->period_usec = period;
    t->running     = 1;

#ifdef _WIN32
    t->stop_event    = CreateEventA(NULL, TRUE, FALSE, NULL);
    t->thread_handle = CreateThread(NULL, 0, timer_thread_proc, t, 0, NULL);
#else
    t->stop_flag = 0;
    pthread_create(&t->thread, NULL, timer_thread_proc, t);
#endif

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_stop
 *
 * r3 = timer_id
 * -----------------------------------------------------------------------*/
int64_t sys_timer_stop(ppu_context* ctx)
{
    uint32_t timer_id = LV2_ARG_U32(ctx, 0);

    if (timer_id == 0 || timer_id > SYS_TIMER_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_timer_info* t = &g_sys_timers[timer_id - 1];
    if (!t->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    if (!t->running)
        return CELL_OK;

    t->running = 0;

#ifdef _WIN32
    if (t->stop_event) SetEvent(t->stop_event);
    if (t->thread_handle) {
        WaitForSingleObject(t->thread_handle, 3000);
        CloseHandle(t->thread_handle);
        t->thread_handle = NULL;
    }
    if (t->stop_event) {
        CloseHandle(t->stop_event);
        t->stop_event = NULL;
    }
#else
    pthread_mutex_lock(&t->mtx);
    t->stop_flag = 1;
    pthread_cond_signal(&t->cv);
    pthread_mutex_unlock(&t->mtx);
    pthread_join(t->thread, NULL);
#endif

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * Registration
 * -----------------------------------------------------------------------*/
void sys_timer_init(lv2_syscall_table* tbl)
{
    memset(g_sys_timers, 0, sizeof(g_sys_timers));

#ifdef _WIN32
    ensure_qpc_init();
#endif

    lv2_syscall_register(tbl, SYS_TIMER_CREATE,                   sys_timer_create);
    lv2_syscall_register(tbl, SYS_TIMER_DESTROY,                  sys_timer_destroy);
    lv2_syscall_register(tbl, SYS_TIMER_START,                    sys_timer_start);
    lv2_syscall_register(tbl, SYS_TIMER_STOP,                     sys_timer_stop);
    lv2_syscall_register(tbl, SYS_TIMER_CONNECT_EVENT_QUEUE,      sys_timer_connect_event_queue);
    lv2_syscall_register(tbl, SYS_TIMER_DISCONNECT_EVENT_QUEUE,   sys_timer_disconnect_event_queue);

    lv2_syscall_register(tbl, SYS_TIME_GET_TIMEBASE_FREQUENCY, sys_time_get_timebase_frequency);
    lv2_syscall_register(tbl, SYS_TIMER_USLEEP,            sys_timer_usleep);
    lv2_syscall_register(tbl, SYS_TIMER_SLEEP,             sys_timer_sleep);
    lv2_syscall_register(tbl, SYS_TIME_GET_TIMEZONE,       sys_time_get_timezone);
    lv2_syscall_register(tbl, SYS_TIME_GET_CURRENT_TIME,   sys_time_get_current_time);
}
