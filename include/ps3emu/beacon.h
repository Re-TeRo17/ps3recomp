/*
 * ps3recomp - phase beacons for the RSX walker and the present thread.
 *
 * A thread that registered a slot (ps3_beacon_thread) keeps a small stack of
 * "what am I doing" names with their entry time. A stall monitor on another
 * thread reads it (racily, read-only) and prints where the walker / presenter
 * has been for how long. A push marked `timed` also logs "[beacon] <name> took
 * N ms" when it is popped after BEACON_LOG_MS (default 500, env BEACON_LOG_MS).
 *
 * Cost: a TLS read, a tick count and three stores per push; nothing when the
 * calling thread has no slot (guest threads, vm_watch faults from them).
 * BEACON=0 turns it off.
 */
#ifndef PS3EMU_BEACON_H
#define PS3EMU_BEACON_H

#ifdef __cplusplus
extern "C" {
#endif

enum { BEACON_WALKER = 0, BEACON_PRESENTER = 1, BEACON_SLOTS = 2, BEACON_DEPTH = 10 };

typedef struct {
    const char* volatile       name[BEACON_DEPTH];
    volatile unsigned long long t0[BEACON_DEPTH];   /* ms (monotonic) of entry */
    volatile unsigned char     timed[BEACON_DEPTH];
    volatile int               depth;               /* 0: idle / not instrumented */
    volatile unsigned long     tid;                 /* host thread id, 0 = none yet */
} ps3_beacon_t;

extern ps3_beacon_t g_ps3_beacon[BEACON_SLOTS];

extern volatile unsigned long long g_ps3_recycle_wait_since;  /* ms tick: a guest thread is waiting for the walker to drain the ring (0: none) */
void               ps3_beacon_thread(int slot);        /* the calling thread owns `slot` */
void               ps3_beacon_push(const char* name, int timed);
void               ps3_beacon_pop(void);
unsigned long long ps3_beacon_ms(void);

#define BEACON_PUSH(n)  ps3_beacon_push((n), 0)
#define BEACON_WAIT(n)  ps3_beacon_push((n), 1)   /* logged when it lasts >= BEACON_LOG_MS */
#define BEACON_POP()    ps3_beacon_pop()

#ifdef __cplusplus
}
#endif
#endif
