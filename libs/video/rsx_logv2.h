/* logv2 (05/11): the structured second log. One JSON object per line in log\tod_v2_<date>_<time>.jsonl (next to the normal .err).
 * ON by default, TOD_LOG_V2=0 turns it off. Every number in it is meant to be traceable to code: allocations carry their site
 * (file function:line of the create call and of the draw-engine caller). See rsx_logv2.c for the record types. */
#ifndef RSX_LOGV2_H
#define RSX_LOGV2_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
extern volatile int g_v2_on;                 /* -1 not resolved yet, 0 off, 1 on */
extern volatile unsigned g_v2_frame;         /* last present frame number */
extern volatile unsigned g_v2_up_s;          /* seconds since the log started (for the 5 s windows) */
extern volatile unsigned g_v2_stat[8];       /* filled by the draw engine each present: surf used, surf cap, zeta used, zeta cap, draws */
int v2_resolve(void);
static inline int v2_enabled(void) { return g_v2_on > 0 || (g_v2_on < 0 && v2_resolve()); }
/* fmt = the JSON fields of the record (no braces), e.g. "\"a\":%d,\"s\":\"%s\"" ; strings must be escaped with v2_esc. */
void v2_event(const char* type, const char* fmt, ...);
const char* v2_esc(char* dst, size_t n, const char* src);   /* JSON-safe copy of src into dst, returns dst */
void v2_frame(double ft_ms, unsigned frame, unsigned draws);   /* ft_ms <= 0: measured here between two calls */
int v2_rl_allow(int kind, unsigned* suppressed_before);       /* kind 0 = hitch records: at most 10 per second; the rest is counted, never lost silently */
/* hooks set by the other modules */
extern int (*g_v2_ext_hook)(char* out, size_t n);           /* outer: game time, hero pos, cpu/gpu, settings (JSON fields, no braces) */
extern int (*g_v2_settings_hook)(char* out, size_t n);      /* outer: the settings only (compared every 250 ms for "settings" events) */
extern int (*g_v2_vram_hook)(int full, char* out, size_t n, unsigned long long* used, unsigned long long* budget);   /* engine */
extern int (*g_v2_threads_hook)(int full, char* out, size_t n);  /* outer: per-thread CPU % (full=1: all busy threads, for the fps_drop anomaly) */
extern int (*g_v2_engine_hook)(char* out, size_t n);        /* draw engine: per-second counters of target/zeta creation and destruction, governor, pool, lane aborts ("eng" field of the tick) */
extern volatile double g_v2_target_fps;      /* outer: the fps limit in force (0 = unknown -> 60) */
/* The caller's context of the next GPU object created on this thread (draw engine -> backend). */
typedef struct { const char* fn; int line; unsigned gaddr, ginfo; } v2_ctx_t;
void v2_ctx_set(const char* fn, int line, unsigned gaddr, unsigned ginfo);
int  v2_ctx_take(v2_ctx_t* out);             /* 1 and clears it, 0 when none */
#define V2CTX(gaddr, ginfo) v2_ctx_set(__func__, __LINE__, (unsigned)(gaddr), (unsigned)(ginfo))
#define V2EV(type, ...) do { if (v2_enabled()) v2_event((type), __VA_ARGS__); } while (0)
#ifdef __cplusplus
}
#endif
#endif
