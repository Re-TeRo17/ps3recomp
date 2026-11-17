/*
 * ps3recomp - POSIX stand-ins for the Win32 calls the runtime's diagnostics use
 *
 * The ToD runtime's opt-in diagnostics (TOD_DT_DIAG, SPURS_CPU_STATS, SPURS_LOCK_DIAG, ...) were
 * written against Win32: thread cycle counters, the current processor number, dbghelp loaded at run
 * time, stack capture. Off Windows these give the closest host value or report "unavailable", so the
 * diagnostics compile everywhere and degrade to fewer details. Nothing here is on a game path.
 * On Windows this header includes nothing and defines nothing.
 */
#ifndef PS3RECOMP_WIN32_DIAG_COMPAT_H
#define PS3RECOMP_WIN32_DIAG_COMPAT_H

#ifndef _WIN32
#include "win32_compat.h"
#include <time.h>
#include <pthread.h>
#include <execinfo.h>
#include <dlfcn.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SUCCEEDED
#define SUCCEEDED(hr) (((long)(hr)) >= 0)
#define FAILED(hr)    (((long)(hr)) < 0)
#endif

/* Structured exception handling: none off Windows. The runtime uses __try only around best-effort
 * stall dumps; there the body runs unguarded and the handler never runs. */
#ifndef __try
#define __try        if (1)
#define __except(x)  else if (0)
#endif

/* Thread "cycles": the thread's CPU time in nanoseconds (callers only compare or sum them). */
static inline BOOL QueryThreadCycleTime(HANDLE thread, ULONG64* cycles)
{
    (void)thread;   /* always the calling thread in the runtime's uses (GetCurrentThread()) */
    struct timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) { *cycles = 0; return FALSE; }
    *cycles = (ULONG64)ts.tv_sec * 1000000000ull + (ULONG64)ts.tv_nsec;
    return TRUE;
}

/* No portable "which core am I on" on macOS: 0 (the diagnostics print it, nothing branches on it). */
static inline DWORD GetCurrentProcessorNumber(void) { return 0; }

/* Modules: the dbghelp symbolisers stay off (LoadLibraryA fails); the main image base is the
 * address the dynamic loader reports for it. */
typedef void* HMODULE;
static inline HMODULE LoadLibraryA(const char* name) { (void)name; return NULL; }
static inline void*   GetProcAddress(HMODULE m, const char* name) { (void)m; (void)name; return NULL; }
static inline BOOL    FreeLibrary(HMODULE m) { (void)m; return TRUE; }
static inline HMODULE GetModuleHandleA(const char* name)
{
    if (name) return NULL;
    Dl_info di;
    return dladdr((const void*)&GetCurrentProcessorNumber, &di) ? (HMODULE)di.dli_fbase : NULL;
}

/* Stack capture of the calling thread: backtrace(3), skipping `skip` frames. */
static inline unsigned short RtlCaptureStackBackTrace(unsigned long skip, unsigned long n, void** out, unsigned long* hash)
{
    void* buf[128];
    int got = backtrace(buf, 128);
    int k = 0;
    for (int i = (int)skip + 1; i < got && k < (int)n; i++) out[k++] = buf[i];
    if (hash) *hash = 0;
    return (unsigned short)k;
}

#ifdef __cplusplus
}
#endif
#endif /* !_WIN32 */
#endif /* PS3RECOMP_WIN32_DIAG_COMPAT_H */
