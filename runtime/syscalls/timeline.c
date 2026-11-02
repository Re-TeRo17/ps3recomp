/*
 * ps3recomp - per-frame timeline tracer (diagnostic, OFF unless TOD_TIMELINE is set).
 *
 *   TOD_TIMELINE=<n>        capture n presented frames
 *   TOD_TIMELINE_AT=<sec>   start that many seconds after the first present (default 40)
 *   TOD_TIMELINE_OUT=<path> output (Chrome Trace JSON; default log/timeline/tl_<pid>.json)
 *   TOD_TIMELINE_RING=<n>   events per thread ring (default 262144, 40 B each)
 *
 * Hooks are `if (g_tl_on) tl_xxx(...)`: one load + branch of a global when off. Events go to a
 * per-thread ring (no locks), timestamps are rdtsc; at the end of the window the rings are
 * written as Chrome Trace JSON (chrome://tracing, ui.perfetto.dev) and parsed by
 * tools/timeline_analyze.py. See RIPARTENZA.md (26/10 timeline).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#define TL_TLS __declspec(thread)
#else
#include <x86intrin.h>
#define TL_TLS __thread
#endif

volatile int g_tl_on;                    /* 1 while capturing */
static int s_tl_cfg = -1;                /* -1 = not read, 0 = disabled, 1 = enabled */
static int s_tl_frames, s_tl_cap;
static double s_tl_at = 40.0;
static char s_tl_out[512];
static unsigned s_tl_epoch;
static volatile unsigned g_tl_epoch_v;

enum { K_PPUFN = 1, K_POLLWAIT, K_USLEEP, K_RUN, K_LLWAIT, K_WRITE, K_PUT, K_FIFO, K_PRESENT, K_IDLE, K_PACE,
       K_LABEL, K_PPUFLIP, K_MISC };

typedef struct { uint64_t ts, a, b; uint32_t dur, c; uint16_t kind; uint8_t ph, x; uint32_t pad; } tl_ev_t;   /* 40 B */
typedef struct tl_ring {
    tl_ev_t* ev; uint32_t n, cap, dropped; uint32_t os_tid; int sort; char name[64];
} tl_ring;

static tl_ring* g_rings[160];
static volatile long g_nrings;
static TL_TLS tl_ring* t_ring;
static TL_TLS char t_pname[64];
static TL_TLS int  t_psort;
static TL_TLS int  t_wid = -1, t_lane = -1;
static TL_TLS unsigned t_epoch;

static uint64_t s_tsc0, s_tsc_end; static double s_tsc_hz; static unsigned long long s_qpc0, s_qpc_hz;

static inline uint64_t tl_now(void) { return __rdtsc(); }

void tl_name(const char* name, int sort)
{
    snprintf(t_pname, sizeof t_pname, "%s", name); t_psort = sort;
    if (t_ring) { snprintf(t_ring->name, sizeof t_ring->name, "%s", name); t_ring->sort = sort; }
}

static tl_ring* tl_get_ring(void)
{
    if (t_ring) return t_ring;
    tl_ring* r = (tl_ring*)calloc(1, sizeof *r);
    if (!r) return NULL;
    r->cap = (uint32_t)s_tl_cap; r->ev = (tl_ev_t*)calloc(r->cap, sizeof(tl_ev_t));
    if (!r->ev) { free(r); return NULL; }
#ifdef _WIN32
    r->os_tid = GetCurrentThreadId();
#endif
    if (t_pname[0]) { snprintf(r->name, sizeof r->name, "%s", t_pname); r->sort = t_psort; }
    else { snprintf(r->name, sizeof r->name, "thr%u", r->os_tid); r->sort = 900; }
#ifdef _WIN32
    { typedef HRESULT (WINAPI *gtd_t)(HANDLE, PWSTR*); static gtd_t gtd; static int tried;
      if (!tried) { tried = 1; HMODULE k = GetModuleHandleA("kernel32.dll"); gtd = k ? (gtd_t)GetProcAddress(k, "GetThreadDescription") : NULL; }
      PWSTR d = NULL; if (gtd && SUCCEEDED(gtd(GetCurrentThread(), &d)) && d) {
          char b[40]; int n = WideCharToMultiByte(CP_UTF8, 0, d, -1, b, sizeof b - 1, NULL, NULL); if (n > 1) { b[n < 39 ? n : 39] = 0;
              for (char* q = b; *q; q++) if (*q == '"' || *q == 0x5C) *q = '_';
              if (r->name[0] == 't' && r->name[1] == 'h') { char t[64]; snprintf(t, sizeof t, "%s (%.40s)", r->name, b); snprintf(r->name, sizeof r->name, "%s", t); } }
          LocalFree(d); } }
#endif
    long i = _InterlockedIncrement(&g_nrings) - 1;
    if (i >= (long)(sizeof g_rings / sizeof g_rings[0])) { free(r->ev); free(r); return NULL; }
    g_rings[i] = r; t_ring = r;
    return r;
}

static inline void tl_push(int kind, int ph, uint64_t ts, uint32_t dur, uint32_t c, uint64_t a, uint64_t b, int x)
{
    tl_ring* r = t_ring ? t_ring : tl_get_ring();
    if (!r) return;
    if (r->n >= r->cap) { r->dropped++; return; }
    tl_ev_t* e = &r->ev[r->n++];
    e->ts = ts; e->dur = dur; e->c = c; e->a = a; e->b = b; e->kind = (uint16_t)kind; e->ph = (uint8_t)ph; e->x = (uint8_t)x;
}

/* ---- generic API ---------------------------------------------------------------------------- */
void tl_ev(int kind, int ph, uint32_t c, uint64_t a, uint64_t b)
{
    if (!g_tl_on) return;
    tl_push(kind, ph, tl_now(), 0, c, a, b, 0);
}
/* kinds exported for the other TUs as small ints */
void tl_rsx(int what, int ph, uint64_t a, uint64_t b)       /* what: 0 fifo, 1 present, 2 idle, 3 pace, 4 label, 5 ppu flip, 6 misc */
{
    static const int k[] = { K_FIFO, K_PRESENT, K_IDLE, K_PACE, K_LABEL, K_PPUFLIP, K_MISC };
    if (!g_tl_on) return;
    tl_push(k[what], ph, tl_now(), 0, 0, a, b, 0);
}

/* ---- PPU ---------------------------------------------------------------------------------- */
static void tl_ppu_name(uint32_t tid)
{
    if (t_ring) {
        if (t_ring->name[0] == 't' && t_ring->name[1] == 'h') { snprintf(t_ring->name, sizeof t_ring->name, "PPU tid%u", tid); t_ring->sort = 10 + (int)tid; }
    } else if (!t_pname[0]) { char b[64]; snprintf(b, sizeof b, "PPU tid%u", tid); tl_name(b, 10 + (int)tid); }
}
/* ph: 0 = B, 1 = E, 2 = instant. fn = guest function address (id), a = r3, b = lr | r4<<32 */
void tl_ppu_ev(int ph, uint32_t fn, uint64_t a, uint64_t b, uint32_t tid)
{
    if (!g_tl_on) return;
    tl_ppu_name(tid);
    tl_push(K_PPUFN, ph == 0 ? 'B' : ph == 1 ? 'E' : 'i', tl_now(), 0, fn, a, b, 0);
}
void tl_usleep(int ph, uint64_t usec, uint32_t lr, uint32_t tid)
{
    if (!g_tl_on) return;
    tl_ppu_name(tid);
    tl_push(K_USLEEP, ph == 0 ? 'B' : 'E', tl_now(), 0, lr, usec, 0, 0);
}

/* Poll-loop coalescer: func_005AC298 (the backoff of `while (poll(x) < 0) backoff()`) is called ~100k
 * times per frame; consecutive calls closer than the gap form ONE wait span (X event) with the lr of
 * the first call (the loop site), the last poll argument seen (func_007E5828 r3 = job group) and the
 * call count. */
static TL_TLS struct { uint64_t start, last, arg0, arg1; uint32_t n, lr; unsigned epoch; int open; } t_bo;
static TL_TLS uint64_t t_poll_arg;
static uint64_t s_gap_cycles = 100000;
static void tl_bo_close(void)
{
    if (!t_bo.open) return;
    uint64_t d = t_bo.last - t_bo.start;
    tl_push(K_POLLWAIT, 'X', t_bo.start, d > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)d, t_bo.lr, t_bo.arg0, ((uint64_t)t_bo.n << 32) | (uint32_t)t_bo.arg1, 0);
    t_bo.open = 0;
}
void tl_ppu_poll(uint64_t r3) { if (g_tl_on) t_poll_arg = r3; }
void tl_ppu_backoff(uint64_t lr, uint32_t tid)
{
    if (!g_tl_on) return;
    uint64_t now = tl_now();
    if (t_bo.epoch != g_tl_epoch_v) { t_bo.open = 0; t_bo.epoch = g_tl_epoch_v; }
    if (t_bo.open && now - t_bo.last > s_gap_cycles) tl_bo_close();
    if (!t_bo.open) { tl_ppu_name(tid); t_bo.open = 1; t_bo.start = now; t_bo.n = 0; t_bo.lr = (uint32_t)lr; t_bo.arg0 = t_poll_arg; }
    t_bo.arg1 = t_poll_arg; t_bo.last = now; t_bo.n++;
}
/* a PPU function span starting (waiter/RSX functions) closes any pending poll span first */
void tl_ppu_flush(void) { if (g_tl_on) tl_bo_close(); }

/* ---- SPU / SPURS ------------------------------------------------------------------------- */
#define LLW 6
static TL_TLS struct { uint32_t ea; uint32_t n; uint64_t t_base, t_last; uint8_t prev[128]; } t_ll[LLW];
static TL_TLS int t_ll_next;
static TL_TLS unsigned t_ll_epoch;
static void tl_ll_flush(uint64_t now_unused)
{
    (void)now_unused;
    for (int i = 0; i < LLW; i++) {
        if (t_ll[i].ea && t_ll[i].n >= 3)
            tl_push(K_LLWAIT, 'X', t_ll[i].t_base, (uint32_t)(t_ll[i].t_last - t_ll[i].t_base), (uint32_t)t_wid, t_ll[i].ea, t_ll[i].n, 1);
        t_ll[i].ea = 0; t_ll[i].n = 0;
    }
}
/* a persistent SPU thread-group thread (collision SPUs etc.): name the track and let its GETLLAR waits / PUTs be logged */
void tl_spu_thread(unsigned tid)
{
    char b[64]; snprintf(b, sizeof b, "SPU thread tid%u", tid); tl_name(b, 200 + (int)(tid & 31));
    t_wid = 4000 + (int)(tid & 255); t_lane = -1;
}
void tl_spu_run(int ph, int wid, int lane, int image)
{
    if (!g_tl_on) return;
    if (ph == 0) {
        if (lane < 0 && !t_pname[0]) { char b[64]; snprintf(b, sizeof b, wid >= 3000 ? "jobchain thr%u" : "spu task thr%u", (unsigned)GetCurrentThreadId()); tl_name(b, 300); }
        t_wid = wid; t_lane = lane;
        memset(t_ll, 0, sizeof t_ll); t_ll_epoch = g_tl_epoch_v;
        tl_push(K_RUN, 'B', tl_now(), 0, (uint32_t)wid, (uint64_t)lane, (uint64_t)image, 0);
    } else {
        uint64_t now = tl_now();
        if (t_ll_epoch == g_tl_epoch_v) tl_ll_flush(now);
        tl_push(K_RUN, 'E', now, 0, (uint32_t)wid, (uint64_t)lane, (uint64_t)image, 0);
        t_wid = -1;
    }
}
/* ea = line, mem = guest memory of the line; called on every GETLLAR (before the reservation) */
void tl_getllar(uint32_t ea, const uint8_t* mem)
{
    if (!g_tl_on || t_wid < 0) return;
    if (t_ll_epoch != g_tl_epoch_v) { memset(t_ll, 0, sizeof t_ll); t_ll_epoch = g_tl_epoch_v; }
    uint64_t now = tl_now();
    int i; for (i = 0; i < LLW; i++) if (t_ll[i].ea == ea) break;
    if (i == LLW) { i = t_ll_next; t_ll_next = (t_ll_next + 1) % LLW; if (t_ll[i].ea && t_ll[i].n >= 3) {
            tl_push(K_LLWAIT, 'X', t_ll[i].t_base, (uint32_t)(t_ll[i].t_last - t_ll[i].t_base), (uint32_t)t_wid, t_ll[i].ea, t_ll[i].n, 1); }
        t_ll[i].ea = ea; t_ll[i].n = 0; t_ll[i].t_base = now; t_ll[i].t_last = now; memcpy(t_ll[i].prev, mem, 128); return; }
    if (memcmp(t_ll[i].prev, mem, 128) == 0) { t_ll[i].n++; t_ll[i].t_last = now; return; }
    if (t_ll[i].n >= 3)
        tl_push(K_LLWAIT, 'X', t_ll[i].t_base, (uint32_t)(now - t_ll[i].t_base), (uint32_t)t_wid, ea, t_ll[i].n, 0);   /* x=0: ended by a change */
    t_ll[i].n = 0; t_ll[i].t_base = now; t_ll[i].t_last = now; memcpy(t_ll[i].prev, mem, 128);
}
/* a reserved line was written (PPU store or SPU PUT/PUTLLC); x: 0 = PPU/notify, 1 = SPU peer */
void tl_write(uint32_t line, int x)
{
    if (!g_tl_on) return;
    tl_push(K_WRITE, 'i', tl_now(), 0, (uint32_t)t_wid, line, (uint64_t)(t_lane + 1), x);
}
/* fence38 (diagnostic): generic tagged instant. who: 1 = PPU store seen by PS3_WATCH_WRITE (a = ea | lr << 32, b = guest-fn << 32 | value),
 * 3 = usleep inside a polled fence (a = polled ea, b = value there, c = lr). Only while capturing. */
extern uint8_t* vm_base;
static volatile uint32_t g_f38_ea;          /* armed cell (spin534EC0), 0 = none */
static volatile long g_f38_thr;
#ifdef _WIN32
static DWORD WINAPI tl_f38_sampler(LPVOID unused)
{
    (void)unused;
    tl_name("f38 sampler", 950);
    while (g_tl_on) {
        uint32_t ea = g_f38_ea;
        if (!ea) { SwitchToThread(); continue; }
        uint64_t v = *(volatile uint64_t*)(vm_base + ea);
        if (v) { tl_push(K_MISC, 'i', tl_now(), 0, 0, ea, __builtin_bswap64(v), 5); if (g_f38_ea == ea) g_f38_ea = 0; }
        else _mm_pause();
    }
    return 0;
}
#endif
void tl_f38(int who, uint32_t c, uint64_t a, uint64_t b)
{
    if (!g_tl_on) return;
    tl_push(K_MISC, 'i', tl_now(), 0, c, a, b, who);
#ifdef _WIN32
    if (who == 3 && c == 0x48C0A8u) {
        g_f38_ea = (uint32_t)a;
        if (!g_f38_thr && _InterlockedExchange(&g_f38_thr, 1) == 0) CreateThread(NULL, 0, tl_f38_sampler, NULL, 0, NULL);
    }
#endif
}
void tl_put(uint32_t ea, uint32_t size)
{
    if (!g_tl_on || t_wid < 0) return;
    tl_push(K_PUT, 'i', tl_now(), 0, (uint32_t)t_wid, ea, size, 0);
}

/* ---- capture control / dump --------------------------------------------------------------- */
static void tl_cfg(void)
{
    const char* e = getenv("TOD_TIMELINE");
    s_tl_frames = e ? atoi(e) : 0; s_tl_cfg = s_tl_frames > 0;
    if (!s_tl_cfg) return;
    const char* a = getenv("TOD_TIMELINE_AT"); if (a) s_tl_at = atof(a);
    const char* r = getenv("TOD_TIMELINE_RING"); s_tl_cap = r ? atoi(r) : 262144; if (s_tl_cap < 1024) s_tl_cap = 1024;
    const char* o = getenv("TOD_TIMELINE_OUT");
    if (o && *o) snprintf(s_tl_out, sizeof s_tl_out, "%s", o);
    else snprintf(s_tl_out, sizeof s_tl_out, "log/timeline/tl_%u.json",
#ifdef _WIN32
                  (unsigned)GetCurrentProcessId()
#else
                  0u
#endif
                  );
    const char* g = getenv("TOD_TIMELINE_GAP"); if (g) s_gap_cycles = (uint64_t)(atof(g) * 3590.0);   /* us -> cycles at ~3.59 GHz (before calibration) */
    fprintf(stderr, "[timeline] enabled: %d frames at +%.0f s -> %s\n", s_tl_frames, s_tl_at, s_tl_out);
}

static const char* kname(int k)
{
    switch (k) { case K_PPUFN: return "ppufn"; case K_POLLWAIT: return "pollwait"; case K_USLEEP: return "usleep"; case K_RUN: return "run";
    case K_LLWAIT: return "llwait"; case K_WRITE: return "write"; case K_PUT: return "put"; case K_FIFO: return "fifo"; case K_PRESENT: return "present";
    case K_IDLE: return "idle"; case K_PACE: return "pace"; case K_LABEL: return "label"; case K_PPUFLIP: return "ppuflip"; default: return "misc"; }
}

static void tl_dump(void)
{
    /* mkdir -p of the output directory (one level at a time) */
    { char d[512]; snprintf(d, sizeof d, "%s", s_tl_out);
      for (char* p = d; *p; p++) if (*p == '/' || *p == '\\') { char c = *p; *p = 0;
#ifdef _WIN32
          if (p != d && d[1] != ':') CreateDirectoryA(d, NULL); else if (p - d > 2) CreateDirectoryA(d, NULL);
#endif
          *p = c; } }
    FILE* f = fopen(s_tl_out, "wb");
    if (!f) { fprintf(stderr, "[timeline] cannot open %s\n", s_tl_out); return; }
    double hz = s_tsc_hz;
    fprintf(f, "{\"tl_meta\":{\"tsc_hz\":%.0f,\"frames\":%d,\"at\":%.1f},\"traceEvents\":[\n", hz, s_tl_frames, s_tl_at);
    int first = 1; unsigned long total = 0, dropped = 0;
    long nr = g_nrings; if (nr > (long)(sizeof g_rings / sizeof g_rings[0])) nr = (long)(sizeof g_rings / sizeof g_rings[0]);
    for (long i = 0; i < nr; i++) {
        tl_ring* r = g_rings[i]; if (!r) continue;
        fprintf(f, "%s{\"ph\":\"M\",\"pid\":1,\"tid\":%ld,\"name\":\"thread_name\",\"args\":{\"name\":\"%s\"}}", first ? "" : ",\n", i + 1, r->name); first = 0;
        fprintf(f, ",\n{\"ph\":\"M\",\"pid\":1,\"tid\":%ld,\"name\":\"thread_sort_index\",\"args\":{\"sort_index\":%d}}", i + 1, r->sort);
        uint32_t n = r->n; total += n; dropped += r->dropped;
        for (uint32_t j = 0; j < n; j++) {
            const tl_ev_t* e = &r->ev[j];
            if (e->ts < s_tsc0 || e->ts > s_tsc_end) continue;
            double us = (double)(e->ts - s_tsc0) * 1e6 / hz;
            char nm[96]; const char* ph = "i";
            switch (e->kind) {
            case K_PPUFN: snprintf(nm, sizeof nm, "fn_%08X", e->c); break;
            case K_POLLWAIT: snprintf(nm, sizeof nm, "poll lr=%08X", e->c); break;
            case K_USLEEP: snprintf(nm, sizeof nm, "usleep lr=%08X", e->c); break;
            case K_RUN: snprintf(nm, sizeof nm, "wid%d", (int)e->c); break;
            case K_LLWAIT: snprintf(nm, sizeof nm, "llwait %08X", (uint32_t)e->a); break;
            case K_WRITE: snprintf(nm, sizeof nm, "write %08X", (uint32_t)e->a); break;
            case K_PUT: snprintf(nm, sizeof nm, "put %08X", (uint32_t)e->a); break;
            default: snprintf(nm, sizeof nm, "%s", kname(e->kind)); }
            ph = e->ph == 'B' ? "B" : e->ph == 'E' ? "E" : e->ph == 'X' ? "X" : "i";
            fprintf(f, ",\n{\"ph\":\"%s\",\"pid\":1,\"tid\":%ld,\"ts\":%.3f,", ph, i + 1, us);
            if (e->ph == 'X') fprintf(f, "\"dur\":%.3f,", (double)e->dur * 1e6 / hz);
            if (e->ph == 'i') fprintf(f, "\"s\":\"t\",");
            fprintf(f, "\"name\":\"%s\",\"cat\":\"%s\",\"args\":{\"k\":\"%s\",\"c\":%u,\"a\":%llu,\"b\":%llu,\"x\":%u}}", nm, kname(e->kind), kname(e->kind),
                    e->c, (unsigned long long)e->a, (unsigned long long)e->b, e->x);
        }
    }
    fprintf(f, "\n]}\n"); fclose(f);
    fprintf(stderr, "[timeline] wrote %s (%lu events in rings, %lu dropped)\n", s_tl_out, total, dropped);
}

/* Called by the present path once per presented frame. */
void tl_present_tick(void)
{
    static int started, done, frames; static unsigned long long t_first;
    if (done) return;
    if (s_tl_cfg < 0) tl_cfg();
    if (!s_tl_cfg) { done = 1; return; }
    unsigned long long now = GetTickCount64();
    if (!t_first) t_first = now;
    if (!started) {
        if ((double)(now - t_first) / 1000.0 < s_tl_at) return;
        /* calibrate TSC against QPC over the first call: use the QPC pair at start and end of capture */
        LARGE_INTEGER q, f; QueryPerformanceCounter(&q); QueryPerformanceFrequency(&f);
        s_qpc0 = (unsigned long long)q.QuadPart; s_qpc_hz = (unsigned long long)f.QuadPart;
        s_tsc0 = tl_now(); started = 1; frames = 0;
        s_gap_cycles = 100000;   /* default 100k cycles ~ 28 us; TOD_TIMELINE_GAP (us) overrides below */
        { const char* g = getenv("TOD_TIMELINE_GAP"); if (g) s_gap_cycles = (uint64_t)(atof(g) * 3590.0); }
        g_tl_epoch_v++; g_tl_on = 1;
        tl_push(K_PRESENT, 'i', s_tsc0, 0, 0, 0, 0, 0);   /* frame 0 boundary (also names the walker thread) */
        fprintf(stderr, "[timeline] capture started\n");
        return;
    }
    tl_push(K_PRESENT, 'i', tl_now(), 0, 0, (uint64_t)(++frames), 0, 0);
    if (frames >= s_tl_frames) {
        g_tl_on = 0;
        s_tsc_end = tl_now();
        LARGE_INTEGER q; QueryPerformanceCounter(&q);
        s_tsc_hz = (double)(s_tsc_end - s_tsc0) * (double)s_qpc_hz / (double)((unsigned long long)q.QuadPart - s_qpc0);
        Sleep(20);   /* writers that passed the g_tl_on test finish */
        tl_dump();
        done = 1;
    }
}
