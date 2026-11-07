/* logv2: the structured second log (see rsx_logv2.h). Record types ("ev"):
 *   start        once: pid, argv, log paths, the env that matters
 *   tick         every 1 s: wall, uptime, frame, LEVEL (the game's loader buffer), fps/frametime, cpu/gpu, hero pos, settings,
 *                VRAM used/budget, breakdown by category, live counts, top allocation sites
 *   level_load_start / level_load_end / settings / teleport / stall / hitch / oom / device / crash / focus
 *   anomaly      VRAM +1 GB in 5 s (or over budget): top 30 live resources + allocation sites with created/destroyed in 5 s
 * Written by whichever thread has something to say (a lock around each line); one fflush per line. */
#include "rsx_logv2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#endif

volatile int g_v2_on = -1;
volatile unsigned g_v2_frame;
volatile unsigned g_v2_up_s;
volatile unsigned g_v2_stat[8];
int (*g_v2_ext_hook)(char* out, size_t n);
int (*g_v2_settings_hook)(char* out, size_t n);
int (*g_v2_vram_hook)(int full, char* out, size_t n, unsigned long long* used, unsigned long long* budget);
int (*g_v2_threads_hook)(int full, char* out, size_t n);
int (*g_v2_engine_hook)(char* out, size_t n);
volatile double g_v2_target_fps;
extern int (*g_rsx_level_hook)(char* out, size_t n);   /* rsx_draw_engine.c: the level the loader names in guest memory */

#ifdef _WIN32
static CRITICAL_SECTION s_cs;
static FILE* s_f;
static LARGE_INTEGER s_q0, s_qf;
static char s_level[64];
static char s_settings[512];
static char s_path[MAX_PATH * 2];

static double up_s(void) { LARGE_INTEGER q; QueryPerformanceCounter(&q); return (double)(q.QuadPart - s_q0.QuadPart) / (double)s_qf.QuadPart; }

const char* v2_esc(char* dst, size_t n, const char* src)
{
    size_t o = 0;
    for (; src && *src && o + 7 < n; src++) {
        const unsigned char c = (unsigned char)*src;
        if (c == '"' || c == '\\') { dst[o++] = '\\'; dst[o++] = (char)c; }
        else if (c < 0x20) { o += (size_t)snprintf(dst + o, n - o, "\\u%04x", c); }
        else dst[o++] = (char)c;
    }
    dst[o] = 0;
    return dst;
}

static void stamp(char* out, size_t n)
{
    SYSTEMTIME t; GetLocalTime(&t);
    snprintf(out, n, "%04u-%02u-%02uT%02u:%02u:%02u.%03u", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond, t.wMilliseconds);
}

/* writes one record: common fields + the body (already JSON fields) */
static void put_line(const char* type, const char* body)
{
    if (!s_f) return;
    char ts[40]; stamp(ts, sizeof ts);
    EnterCriticalSection(&s_cs);
    fprintf(s_f, "{\"ts\":\"%s\",\"up\":%.3f,\"f\":%u,\"lvl\":\"%s\",\"ev\":\"%s\"%s%s}\n", ts, up_s(), g_v2_frame, s_level, type,
            body && body[0] ? "," : "", body ? body : "");
    fflush(s_f);
    LeaveCriticalSection(&s_cs);
}

void v2_event(const char* type, const char* fmt, ...)
{
    if (!s_f && !v2_enabled()) return;
    if (!s_f) return;
    char body[6144]; va_list ap; va_start(ap, fmt); vsnprintf(body, sizeof body, fmt, ap); va_end(ap);
    put_line(type, body);
}

static __declspec(thread) v2_ctx_t t_ctx;
void v2_ctx_set(const char* fn, int line, unsigned gaddr, unsigned ginfo) { t_ctx.fn = fn; t_ctx.line = line; t_ctx.gaddr = gaddr; t_ctx.ginfo = ginfo; }
int v2_ctx_take(v2_ctx_t* out) { if (!t_ctx.fn) { out->fn = NULL; return 0; } *out = t_ctx; t_ctx.fn = NULL; return 1; }

static int s_once_state;   /* 0 not tried, 1 open, 2 failed */
int v2_resolve(void)
{
    if (g_v2_on >= 0) return g_v2_on;
    static volatile LONG busy;
    if (InterlockedExchange(&busy, 1)) return 0;
    if (g_v2_on >= 0) { busy = 0; return g_v2_on; }
    const char* e = getenv("TOD_LOG_V2");
    if (e && e[0] == '0') { g_v2_on = 0; busy = 0; return 0; }
    InitializeCriticalSection(&s_cs);
    QueryPerformanceCounter(&s_q0); QueryPerformanceFrequency(&s_qf);
    char dir[MAX_PATH * 2] = {0}, errp[MAX_PATH * 2] = {0}, stamp_s[32] = {0};
    HANDLE he = GetStdHandle(STD_ERROR_HANDLE);
    if (he && he != INVALID_HANDLE_VALUE && GetFileType(he) == FILE_TYPE_DISK) {
        char p[MAX_PATH * 2];
        DWORD r = GetFinalPathNameByHandleA(he, p, (DWORD)sizeof p, FILE_NAME_NORMALIZED);
        if (r > 0 && r < sizeof p) {
            const char* q = p; if (!strncmp(q, "\\\\?\\", 4)) q += 4;
            strncpy(errp, q, sizeof errp - 1);
            char* sl = strrchr(errp, '\\');
            if (sl) { size_t dl = (size_t)(sl - errp); memcpy(dir, errp, dl); dir[dl] = 0;
                const char* st = sl + 1;   /* "YYYYMMDD_HHMMSS.err": the same stamp as the normal log */
                if (strlen(st) >= 15 && st[8] == '_') { memcpy(stamp_s, st, 15); stamp_s[15] = 0; } }
        }
    }
    if (!dir[0]) { const char* d = getenv("TOD_LOG_V2_DIR"); if (d && d[0]) strncpy(dir, d, sizeof dir - 1); else strcpy(dir, "log"); CreateDirectoryA(dir, NULL); }
    if (!stamp_s[0]) { SYSTEMTIME t; GetLocalTime(&t); snprintf(stamp_s, sizeof stamp_s, "%04u%02u%02u_%02u%02u%02u", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond); }
    snprintf(s_path, sizeof s_path, "%s\\tod_v2_%s.jsonl", dir, stamp_s);
    s_f = fopen(s_path, "wb");
    if (!s_f) { g_v2_on = 0; busy = 0; return 0; }
    g_v2_on = 1; busy = 0;
    char a[1024], b[1024], c[1024], d2[1024];
    static const char* const keys[] = { "RSX_SCALE", "RSX_AA", "RSX_LANES", "RSX_VSYNC", "TOD_FPS", "RSX_VRAM_GOV", "RSX_ZPRE_RECENT", "RSX_HITCH_LOG",
                                        "RSX_VRAM_LOG", "RSX_SURF_POOL", "RSX_PACE_VBLANK", "RT_AO", "RSX_WINDOW", "RSX_DISPLAY" };
    char envs[1536]; size_t eo = 0; envs[0] = 0;
    for (size_t i = 0; i < sizeof keys / sizeof keys[0]; i++) { const char* v = getenv(keys[i]); if (!v) continue;
        eo += (size_t)snprintf(envs + eo, sizeof envs - eo, "%s\"%s\":\"%s\"", eo ? "," : "", keys[i], v2_esc(a, sizeof a, v)); if (eo > sizeof envs - 128) break; }
    char body[3600];
    snprintf(body, sizeof body, "\"v\":2,\"pid\":%lu,\"argv\":\"%s\",\"err\":\"%s\",\"v2\":\"%s\",\"cwd_env\":{%s}", (unsigned long)GetCurrentProcessId(),
             v2_esc(b, sizeof b, getenv("PS3_ARGV") ? getenv("PS3_ARGV") : ""), v2_esc(c, sizeof c, errp), v2_esc(d2, sizeof d2, s_path), envs);
    put_line("start", body);
    return 1;
}

/* ---- per-frame bookkeeping ---- */
static struct {
    double t0, tick_next, poll_next; unsigned n, slow, big; double sum, mx; float dt[1024]; unsigned ndt;
    unsigned long long vr_t[8]; unsigned long long vr_u[8]; unsigned nvr;   /* the last used-VRAM readings (1/s) for the 5 s growth test */
    double last_anom; int pend_ready; double pend_t0; unsigned pend_good; unsigned tick_n;
    double anom_mb, anom_gap; int init;
    unsigned long long seq; LARGE_INTEGER lastq; int haveq;
    unsigned an_supp; double an_peak_grew; unsigned long long an_peak_used;   /* anomaly conditions seen while rate-limited, since the last record */
    double cap_bytes, keep_s, cap_next; unsigned long long supp_hitch_tot; unsigned hitch_win_n, hitch_win_t, hitch_supp_win;
} V;

static int cmpf(const void* a, const void* b) { const float x = *(const float*)a, y = *(const float*)b; return x < y ? -1 : x > y; }

static void poll_changes(void)
{
    char lv[64] = {0};
    if (g_rsx_level_hook) { if (!g_rsx_level_hook(lv, sizeof lv)) lv[0] = 0; }
    if (strcmp(lv, s_level)) {
        char a[96], b[96];
        v2_event("level_load_start", "\"from\":\"%s\",\"to\":\"%s\"", v2_esc(a, sizeof a, s_level), v2_esc(b, sizeof b, lv));
        strncpy(s_level, lv, sizeof s_level - 1); s_level[sizeof s_level - 1] = 0;
        V.pend_ready = lv[0] ? 1 : 0; V.pend_t0 = up_s(); V.pend_good = 0;
    }
    if (g_v2_settings_hook) {
        char st[512]; st[0] = 0; g_v2_settings_hook(st, sizeof st);
        if (strcmp(st, s_settings)) {
            if (s_settings[0]) v2_event("settings", "\"old\":{%s},\"new\":{%s}", s_settings, st);
            strncpy(s_settings, st, sizeof s_settings - 1); s_settings[sizeof s_settings - 1] = 0;
        }
    }
}

int v2_rl_allow(int kind, unsigned* before)
{
    (void)kind;
    const unsigned sec = (unsigned)up_s();
    if (V.hitch_win_t != sec) { V.hitch_win_t = sec; V.hitch_win_n = 0; }
    if (V.hitch_win_n >= 10) { V.hitch_supp_win++; V.supp_hitch_tot++; return 0; }
    V.hitch_win_n++;
    if (before) { *before = V.hitch_supp_win; } V.hitch_supp_win = 0;
    return 1;
}

/* file cap: when the file passes the cap, thin OLD ticks only (1 in 10 kept, except the first and the last keep_s seconds); events and anomaly records are never dropped. */
static void compact_file(double now)
{
    EnterCriticalSection(&s_cs);
    fclose(s_f);
    char tmp[sizeof s_path + 8]; snprintf(tmp, sizeof tmp, "%s.tmp", s_path);
    FILE* in = fopen(s_path, "rb"); FILE* out = in ? fopen(tmp, "wb") : NULL;
    unsigned long long kept = 0, dropped = 0;
    if (in && out) {
        static char line[200 * 1024];
        while (fgets(line, sizeof line, in)) {
            int keep = 1;
            if (strstr(line, "\"ev\":\"tick\"")) {
                const char* u = strstr(line, "\"up\":"); const char* q = strstr(line, "\"seq\":");
                const double up = u ? atof(u + 5) : 0.0; const unsigned long long sq = q ? strtoull(q + 6, NULL, 10) : 0;
                if (up >= V.keep_s && up < now - V.keep_s && (sq % 10) != 0) keep = 0;
            }
            if (keep) { fputs(line, out); kept++; } else dropped++;
        }
    }
    if (in) fclose(in);
    if (out) fclose(out);
    if (in && out) { remove(s_path); rename(tmp, s_path); }
    s_f = fopen(s_path, "ab");
    LeaveCriticalSection(&s_cs);
    if (s_f) v2_event("compacted", "\"kept_lines\":%llu,\"dropped_ticks\":%llu,\"rule\":\"old ticks thinned to 1 in 10 (seq %% 10 == 0), first/last %.0f s and all events kept\"", kept, dropped, V.keep_s);
}

void v2_frame(double ft_ms, unsigned frame, unsigned draws)
{
    if (!v2_enabled() || !s_f) return;
    g_v2_frame = frame;
    const double now = up_s();
    g_v2_up_s = (unsigned)now;
    if (!V.init) { V.init = 1; V.t0 = now; V.tick_next = now + 1.0; V.poll_next = now; V.last_anom = -1e9;
        const char* e = getenv("TOD_LOGV2_ANOM_MB"); V.anom_mb = e ? atof(e) : 1024.0;
        e = getenv("TOD_LOGV2_ANOM_GAP_S"); V.anom_gap = e ? atof(e) : 30.0;
        e = getenv("TOD_LOGV2_CAP_MB"); V.cap_bytes = (e ? atof(e) : 200.0) * 1048576.0; e = getenv("TOD_LOGV2_KEEP_S"); V.keep_s = e ? atof(e) : 3600.0; V.cap_next = now + 60.0; }
    { LARGE_INTEGER q; QueryPerformanceCounter(&q); if (V.haveq) { const double m = (double)(q.QuadPart - V.lastq.QuadPart) * 1000.0 / (double)s_qf.QuadPart; if (ft_ms <= 0.0) ft_ms = m; } V.lastq = q; V.haveq = 1; }
    if (ft_ms > 0.0) {
        V.n++; V.sum += ft_ms; if (ft_ms > V.mx) V.mx = ft_ms;
        if (ft_ms > 34.0) V.slow++;
        if (ft_ms > 100.0) V.big++;
        if (V.ndt < 1024) V.dt[V.ndt++] = (float)ft_ms;
    }
    if (V.pend_ready) {
        if (draws >= 300) { if (++V.pend_good >= 3) { V.pend_ready = 0; char a[96]; v2_event("level_load_end", "\"level\":\"%s\",\"secs\":%.2f,\"note\":\"first 3 frames with >=300 draws after the loader buffer changed\"", v2_esc(a, sizeof a, s_level), now - V.pend_t0); } }
        else V.pend_good = 0;
    }
    if (now >= V.poll_next) { V.poll_next = now + 0.25; poll_changes(); }
    if (now < V.tick_next) return;
    const double span = now - (V.tick_next - 1.0);
    V.tick_next = now + 1.0;
    V.tick_n++;
    char ext[1400]; ext[0] = 0; if (g_v2_ext_hook) g_v2_ext_hook(ext, sizeof ext);
    char vr[6000]; vr[0] = 0; unsigned long long used = 0, budget = 0; int have_vr = 0;
    if (g_v2_vram_hook) have_vr = g_v2_vram_hook(0, vr, sizeof vr, &used, &budget);
    float p99 = 0.f; if (V.ndt) { qsort(V.dt, V.ndt, sizeof(float), cmpf); p99 = V.dt[(unsigned)((V.ndt - 1) * 0.99)]; }
    char thr[700]; thr[0] = 0; if (g_v2_threads_hook) g_v2_threads_hook(0, thr, sizeof thr);
    char eg[1400]; eg[0] = 0; if (g_v2_engine_hook) g_v2_engine_hook(eg, sizeof eg);
    const double fps = span > 0 ? V.n / span : 0.0;
    char body[9800];
    snprintf(body, sizeof body, "\"seq\":%llu,\"supp\":%llu,\"fps\":%.1f,\"ft_mean\":%.2f,\"ft_max\":%.1f,\"ft_p99\":%.1f,\"slow34\":%u,\"big100\":%u,\"frames\":%u,\"draws\":%u,"
             "\"table\":{\"surf\":%u,\"surf_cap\":%u,\"zeta\":%u,\"zeta_cap\":%u}%s%s%s%s%s%s%s%s",
             V.seq++, V.supp_hitch_tot, fps, V.n ? V.sum / V.n : 0.0, V.mx, p99, V.slow, V.big, V.n, draws,
             g_v2_stat[0], g_v2_stat[1], g_v2_stat[2], g_v2_stat[3], ext[0] ? "," : "", ext, have_vr ? "," : "", vr, thr[0] ? "," : "", thr, eg[0] ? "," : "", eg);
    put_line("tick", body);
    /* the VRAM budget Windows gives us changed (another program took or gave back video memory: Discord capture, a browser, ...) */
    if (have_vr && budget) {
        static unsigned long long last_b;
        const long long d = (long long)(budget >> 20) - (long long)(last_b >> 20);
        if (last_b && (d > 256 || d < -256)) v2_event("vram_budget", "\"from_mb\":%llu,\"to_mb\":%llu,\"used_mb\":%llu", last_b >> 20, budget >> 20, used >> 20);
        last_b = budget;
    }
    /* fps_drop: this second is below 80% of the median of the last 10 seconds (and below the limit), or below 55% of the limit.
     * Records the per-thread CPU (who is the bottleneck); rate-limited like the VRAM anomaly, with the suppressed count. */
    {
        static float hist[10]; static unsigned nh; static double last_drop = -1e9; static unsigned dsupp; static float dworst = 1e9f;
        float srt[10]; unsigned m = nh < 10 ? nh : 10; for (unsigned i = 0; i < m; i++) srt[i] = hist[i];
        qsort(srt, m, sizeof(float), cmpf);
        const double med = m >= 5 ? srt[m / 2] : 0.0;
        double tgt = g_v2_target_fps > 0 ? g_v2_target_fps : 60.0; if (tgt > 1000) tgt = 1000;
        const int drop = fps > 0 && V.n > 0 && ((med > 0 && fps < 0.80 * med && fps < 0.95 * tgt) || fps < 0.55 * tgt) && s_level[0] && !V.pend_ready;   /* not while a level loads */
        if (drop && now - last_drop < 30.0) { dsupp++; if (fps < dworst) dworst = (float)fps; }
        else if (drop) {
            last_drop = now;
            char* tb = (char*)malloc(4096); if (tb) { tb[0] = 0; if (g_v2_threads_hook) g_v2_threads_hook(1, tb, 4096);
                v2_event("anomaly", "\"kind\":\"fps_drop\",\"fps\":%.1f,\"median10\":%.1f,\"target\":%.0f,\"ft_max\":%.1f,\"draws\":%u,\"suppressed_since_last\":%u,\"suppressed_worst_fps\":%.1f%s%s",
                         fps, med, tgt, V.mx > 0 ? V.mx : 0.0, draws, dsupp, dsupp ? dworst : 0.0f, tb[0] ? "," : "", tb);
                free(tb); }
            dsupp = 0; dworst = 1e9f;
        }
        if (fps > 0) { hist[nh % 10] = (float)fps; nh++; }
    }
    V.n = V.slow = V.big = 0; V.sum = V.mx = 0; V.ndt = 0;
    if (now >= V.cap_next) { V.cap_next = now + 60.0; if (s_f && (double)ftell(s_f) > V.cap_bytes) compact_file(now); }
    /* anomaly: used VRAM grew by more than anom_mb within 5 s, or went over the budget */
    if (have_vr && used) {
        unsigned long long mn = used;
        for (unsigned i = 0; i < V.nvr; i++) if (now - (double)V.vr_t[i] / 1000.0 <= 5.0 && V.vr_u[i] < mn) mn = V.vr_u[i];
        const double grew = (double)(used - mn) / 1048576.0;
        const int over = budget && used > budget;
        if (V.nvr < 8) { V.vr_t[V.nvr] = (unsigned long long)(now * 1000.0); V.vr_u[V.nvr] = used; V.nvr++; }
        else { memmove(V.vr_t, V.vr_t + 1, 7 * sizeof V.vr_t[0]); memmove(V.vr_u, V.vr_u + 1, 7 * sizeof V.vr_u[0]); V.vr_t[7] = (unsigned long long)(now * 1000.0); V.vr_u[7] = used; }
        if ((grew > V.anom_mb || over) && now - V.last_anom < V.anom_gap) { V.an_supp++; if (grew > V.an_peak_grew) V.an_peak_grew = grew; if (used > V.an_peak_used) V.an_peak_used = used; }
        if ((grew > V.anom_mb || over) && now - V.last_anom >= V.anom_gap && g_v2_vram_hook) {
            V.last_anom = now;
            const size_t cap = 160 * 1024; char* big = (char*)malloc(cap);
            if (big) {
                unsigned long long u2 = 0, b2 = 0; big[0] = 0;
                g_v2_vram_hook(1, big, cap, &u2, &b2);
                char* line = (char*)malloc(cap + 512);
                if (line) {
                    snprintf(line, cap + 512, "\"kind\":\"vram\",\"loading\":%d,\"reason\":\"%s\",\"grew_mb\":%.0f,\"used_mb\":%llu,\"budget_mb\":%llu,\"suppressed_since_last\":%u,\"suppressed_peak_grew_mb\":%.0f,\"suppressed_peak_used_mb\":%llu,%s",
                             V.pend_ready, over && grew <= V.anom_mb ? "over_budget" : "grew", grew,
                             used >> 20, budget >> 20, V.an_supp, V.an_peak_grew, V.an_peak_used >> 20, big);
                    V.an_supp = 0; V.an_peak_grew = 0; V.an_peak_used = 0;
                    put_line("anomaly", line); free(line);
                }
                free(big);
            }
        }
    }
}
#else
const char* v2_esc(char* dst, size_t n, const char* src) { if (n) { strncpy(dst, src ? src : "", n - 1); dst[n - 1] = 0; } return dst; }
void v2_event(const char* type, const char* fmt, ...) { (void)type; (void)fmt; }
void v2_frame(double a, unsigned b, unsigned c) { (void)a; (void)b; (void)c; }
void v2_ctx_set(const char* fn, int line, unsigned gaddr, unsigned ginfo) { (void)fn; (void)line; (void)gaddr; (void)ginfo; }
int v2_ctx_take(v2_ctx_t* out) { out->fn = NULL; return 0; }
int v2_resolve(void) { g_v2_on = 0; return 0; }
#endif
