#define SPU_IMPL_TU 1
/*
 * ps3recomp - SPU channel + indirect-branch runtime glue
 *
 * Implements the externs the SPU lifter (tools/spu_lifter.py) emits:
 *   - spu_rdch / spu_rchcnt / spu_wrch : SPU channel access. MFC channels are
 *     routed to the DMA engine (spu_dma.h); mailboxes, signal notification,
 *     events and the decrementer use the spu_context channel fields.
 *   - spu_indirect_branch : resolves ctx->pc to a lifted spu_func_* via a
 *     registry that generated code populates by calling spu_recomp_register().
 *
 * The MFC engine state is kept per spu_context here (spu_context.h does not
 * embed one), in a small lazily-populated registry.
 */

#include "spu_dma.h"
#include "spu_coherency.h" /* lock-line lock + PPU-write coherence bitmap */
#include "spu_helpers.h"   /* spu_splat_u32 / spu_ls_read128 (SMC microstep) */
#include "spu_lockstep.h"
void spu_hangline_note(char kind, int img, uint32_t spu, uint32_t pc, uint32_t ea, const uint8_t* line);  /* SPU_HANG_LINE */
#include "spu_interp.h"    /* spu_lifted_fn / spu_lifted_lookup -- defined below */
int64_t ps3_guest_ticks(int64_t host_ticks);   /* sys_timer.c: the decrementer stands still while paused */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
/* Plain setjmp/longjmp save and restore the signal mask on macOS -- a system
 * call each, on every SPURS job. Nothing in the SPU runtime touches the mask. */
#ifndef _WIN32
#define SPU_SETJMP(env)     _setjmp(env)
#define SPU_LONGJMP(env, v) _longjmp(env, v)
#else
#define SPU_SETJMP(env)     setjmp(env)
#define SPU_LONGJMP(env, v) longjmp(env, v)
#endif
/* SPU_NOUNWIND(env), right after setjmp's first return: on Windows x64 a
 * longjmp unwinds every frame back to the setjmp through RtlUnwindEx unless
 * the buffer's Frame is 0 -- then it restores the registers and jumps
 * (measured: 637 ns -> 14 ns across 8 frames). Every SPU job that halts ends
 * with one, hundreds a frame on the render thread and the RSX walker, where
 * the unwind (_NLG_Return2, RtlVirtualUnwind2) was 6-8% of the walker. The
 * frames in between are lifted SPU code and this C runtime: no __try, no
 * destructors, nothing an unwind would run. */
#if defined(_WIN32) && defined(_M_X64)
#define SPU_NOUNWIND(env)   (((_JUMP_BUFFER*)(env))->Frame = 0)
#else
#define SPU_NOUNWIND(env)   ((void)0)
#endif
#ifndef _WIN32
#include <sched.h>   /* sched_yield */
#endif
#include <time.h>
#include "../platform/win32_compat.h"

/* Environment switches are read where they are used, some of them per call:
 * answer from the pointer-keyed cache (ps3emu/env_cache.h). */
#include "ps3emu/env_cache.h"
#define getenv(name) ps3_env(name)

#ifdef __cplusplus
extern "C" {
#endif

/* The SPU decrementer ticks at the PS3 timebase, 79.8 MHz -- the same clock
 * sys_time_get_timebase_frequency reports to the PPU. Titles calibrate real
 * delays against it, so the rate has to be right, not merely non-zero. */
#define SPU_DECREMENTER_HZ  79800000ull

static uint64_t spu_host_ns(void)
{
#ifdef _WIN32
    static LARGE_INTEGER s_freq;
    LARGE_INTEGER c;
    if (!s_freq.QuadPart) QueryPerformanceFrequency(&s_freq);
    QueryPerformanceCounter(&c);
    c.QuadPart = ps3_guest_ticks(c.QuadPart);
    /* Split the divide to keep the multiply from overflowing: QPC counters are
     * large enough that (count * 1e9) wraps a u64 within hours of uptime. */
    return (uint64_t)(c.QuadPart / s_freq.QuadPart) * 1000000000ull
         + (uint64_t)(c.QuadPart % s_freq.QuadPart) * 1000000000ull
           / (uint64_t)s_freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ps3_guest_ticks((int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec);
#endif
}

/* ---------------------------------------------------------------------------
 * Clean SPU job abort (longjmp). The `br .` halt idiom and other terminal
 * spins can't be escaped by setting a status flag (lifted code never checks
 * it), so spu_halt() longjmps back to spu_run_with_halt() in the dispatcher.
 * -----------------------------------------------------------------------*/
#if defined(_MSC_VER)
#  define SPU_TLS __declspec(thread)
#else
#  define SPU_TLS __thread
#endif
/* Called after an SPU atomic commits a 128-byte line, with the lock-line lock
 * released. cellSpurs sets it so a PPU blocked in cellSpursEventFlagWait wakes
 * when a task sets the flag, instead of on its next 2 ms poll. */
void (*g_spu_line_commit_hook)(uint32_t line) = 0;
/* A second listener, for the title: the first belongs to cellSpurs's event
 * flags, which install it whenever they like. */
void (*g_spu_line_commit_hook2)(uint32_t line) = 0;
/* spu_workload.c: wakes tasks parked in WAIT_SIGNAL on that line's taskset. */
void spu_taskset_line_committed(uint32_t line);
/* SPU_LS_WATCH=0x1BE80[,0x927D80...]: watch up to 4 16-byte LS lines (reads
 * and writes) in every SPU image. See spu_ls_watch_hit2 for the fast path. */
#define SPU_WATCH_MAX 4
int g_spu_ls_watch_n = -1;
static unsigned s_spu_ls_watch[SPU_WATCH_MAX];
unsigned* spu_ls_watch_list(int* out_n)
{
    if (g_spu_ls_watch_n < 0) {
        int n = 0; const char* e = getenv("SPU_LS_WATCH");
        while (e && *e && n < SPU_WATCH_MAX) {
            s_spu_ls_watch[n++] = (unsigned)strtoul(e, (char**)&e, 0) & ~0xFu;
            while (*e == ',' || *e == ' ') e++;
        }
        g_spu_ls_watch_n = n;
    }
    *out_n = g_spu_ls_watch_n;
    return s_spu_ls_watch;
}
void spu_ls_watch_slow(uint32_t lsa, int is_write, const uint8_t* p, uint32_t pc, uint32_t lr)
{
    int n; spu_ls_watch_list(&n);
    if (!n) return;
    uint32_t a = lsa & (SPU_LS_MASK & ~0xFu);
    for (int i = 0; i < g_spu_ls_watch_n; i++) {
        if (s_spu_ls_watch[i] == a) {
            fprintf(stderr, "[spu-watch %s 0x%05X pc=0x%05X lr=0x%05X] "
                "%02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X\n",
                is_write ? "WR" : "rd", a, pc, lr,
                p[0],p[1],p[2],p[3], p[4],p[5],p[6],p[7],
                p[8],p[9],p[10],p[11], p[12],p[13],p[14],p[15]);
            fflush(stderr);
            break;
        }
    }
}

static SPU_TLS jmp_buf s_spu_halt_env;
static SPU_TLS int     s_spu_halt_armed = 0;

/* A pooled SPU task thread (spu_workload.c) runs one task after another, so
 * state a task keeps per host thread is reset at the start of each one; code
 * that counts "per task" in a thread-local compares g_spu_task_serial. */
SPU_THREAD_LOCAL unsigned g_spu_task_serial;
void spu_task_begin(void)
{
    g_spu_task_serial++;
    g_spu_trampoline_fn = 0;
    s_spu_halt_armed = 0;
}

/* A non-local return replaces the old host call chain without changing
 * any architectural registers, local store, or pending guest work. */
void spu_restart_dispatch(spu_context* ctx)
{
    g_spu_trampoline_fn = 0;
    if (s_spu_halt_armed) SPU_LONGJMP(s_spu_halt_env, 2);
    fprintf(stderr, "[spu] non-local return outside execution driver at 0x%05X\n", ctx->pc);
    ctx->status = SPU_STATUS_STOPPED_BY_HALT;
}

/* SPU->PPU outbound-mailbox delivery hook. The SPU writing WrOutMbox /
 * WrOutIntrMbox must wake PPU code blocked on the SPURS event queue bound to
 * the SPU thread group (e.g. cellSpursInitialize). lv2_register.c installs a
 * handler that maps spu_group_id -> connected event queue and pushes an event.
 * NULL until installed (plain SPU jobs with no PPU listener stay a no-op). */
int (*g_spu_user_event_hook)(spu_context*, uint32_t) = NULL;
void (*g_spu_out_mbox_hook)(uint32_t group_id, uint32_t spu_id,
                            int is_intr, uint32_t value) = 0;

/* Every MFC PUT an SPU completes, if anyone is listening. Installed by the GCM
 * layer (libs/video/cellGcmSys.c), which needs it to find the RSX pushbuffer:
 * a title whose SPUs build the command stream never tells libgcm where that
 * stream is, so the only evidence is the DMA that writes it. NULL until
 * installed, so a title with no SPU-built FIFO pays nothing. */
void (*g_spu_put_hook)(uint32_t ea, uint32_t size) = 0;

void spu_halt(spu_context* ctx)
{
    /* Register/context dump at halt-asserts: the WWS job's parameter guard
     * (heqi 0x1650 in job_wws_7BD900) checks (r12 bit0) & (r9 16-aligned) &
     * (r15 != 0) -- job descriptor/ABI values our execute path must supply.
     * Dump the low registers + the job's parameter area so a firing assert
     * names exactly which input is wrong. */
    { static int _n = 0;
      if (_n < 6 && ctx->status == SPU_STATUS_STOPPED_BY_HALT) { _n++;
        fprintf(stderr, "[spu-halt] img=%d pc=0x%05X regs:", ctx->image_id,
                ctx->pc & SPU_LS_MASK);
        for (int r = 3; r <= 16; r++)
            fprintf(stderr, " r%d=%08X", r, ctx->gpr[r]._u32[0]);
        fprintf(stderr, " r80=%08X r85=%08X\n",
                ctx->gpr[80]._u32[0], ctx->gpr[85]._u32[0]);
        fflush(stderr);
      } }
    if (s_spu_halt_armed) { s_spu_halt_armed = 0; SPU_LONGJMP(s_spu_halt_env, 1); }
}

/* Diagnostic: dump the taskset-policy scheduler's working tables (LS 0x2700..)
 * at func_00000E60 entry, to reverse why it computes "no runnable task". Env
 * YDKJ_E60. Called from the lifted taskset policy. */
void spu_dbg_e60(spu_context* ctx)
{
    static int s_e = -1; if (s_e < 0) s_e = getenv("YDKJ_E60") ? 1 : 0;
    if (!s_e) return;
    static int _d = 0; if (_d++ >= 6) return;
    const uint8_t* L = ctx->ls;
    #define RD(o) (((uint32_t)L[(o)]<<24)|((uint32_t)L[(o)+1]<<16)|((uint32_t)L[(o)+2]<<8)|L[(o)+3])
    fprintf(stderr, "[E60] r20=%08X r24=%08X | run2700=%08X ready2710=%08X pend2720=%08X en2730=%08X sig2740=%08X wait2750=%08X x2770=%08X\n",
            ctx->gpr[20]._u32[0], ctx->gpr[24]._u32[0],
            RD(0x2700), RD(0x2710), RD(0x2720), RD(0x2730), RD(0x2740), RD(0x2750), RD(0x2770));
    #undef RD
    fflush(stderr);
}

/* SPU `stop` / stop-and-signal. A real SPU stop HALTS the core; the previous
 * lifted emission only did `status=...; return;`, which unwound ONE frame and
 * let the caller's service loop keep running -- so a SPURS policy doing
 * stop-and-signal in a loop spun millions of times instead of yielding. Halt the
 * host SPU thread (longjmp to spu_run_with_halt) so the job stops AT the stop,
 * with ctx->status + the outbound mailbox value available for the dispatcher to
 * service. Env YDKJ_STOP_NOHALT restores the old (looping) behavior for A/B. */
void spu_stop(spu_context* ctx)
{
    /* A lifted `stop` sets status and RETURNS to its caller -- for a SPURS
     * policy this is stop-and-signal: it returns up into the kernel/policy
     * service loop, which continues (i.e. the SPU "resumes" past the stop). That
     * resume-by-return behavior is correct; the previous spin was caused by the
     * lack of (a) a PPU listener for the outbound mailbox and (b) mailbox
     * backpressure -- both handled in the channel write path, not here. So by
     * default DO NOT halt. Env YDKJ_STOP_HALT forces a hard halt (longjmp) for
     * A/B experiments (it makes the kernel terminate, which the game restarts). */
    static int s_halt = -1;
    if (s_halt < 0) s_halt = getenv("YDKJ_STOP_HALT") ? 1 : 0;
    ctx->status = SPU_STATUS_STOPPED_BY_STOP;
    if (s_halt) spu_halt(ctx);
}

/* Run a lifted SPU entry with a halt landing pad. Returns 1 if the job halted
 * (via spu_halt), 0 if it returned normally. */
int spu_run_with_halt(void (*entry)(spu_context*), spu_context* ctx)
{
    int halted = 0;
    /* Job-args probe: image-1 (the WWS job binary) entries arrive at the crt
     * with r3/r4 zeroed in SOME runs while the jm2 runner sets them -- count
     * and stamp every image-1 run here to find the argless caller. */
    if (ctx->image_id == 1) {
        static int _n = 0;
        if (_n < 12) { _n++;
            fprintf(stderr, "[rwh] #%d img=1 ctx=%p r3=%08X r4=%08X entry=%p\n",
                    _n, (void*)ctx, ctx->gpr[3]._u32[0], ctx->gpr[4]._u32[0],
                    (void*)entry);
            fflush(stderr); }
    }
    /* Also discard any saved register file keyed to this context pointer:
     * contexts are stack locals, so the address recurs and a stale save from
     * an earlier run would be restored over this job's arguments. */
    { extern void spu_irq_regs_forget(spu_context*); spu_irq_regs_forget(ctx); }
    ctx->steps = 0;   /* fresh run: step 0 is the entry */
    /* The host stack is empty here by definition, so no lifted caller frame
     * is live. A persistent context (a SPURS policy module re-entered every
     * scheduling quantum) otherwise carries the depth its last quantum
     * halted at: two frames per quantum for an idle job manager, and the
     * recursion guard halted the SPU at 2000 after a few minutes of idling. */
    ctx->host_depth = 0;
    ctx->irq_frame = 0;
    ctx->drain_ret_pc = 0;
    /* Nest-safe: a job run synchronously from inside another SPU's execution
     * re-enters here on the same host thread. The halt target is per-thread,
     * so save the outer one and put it back on the way out -- otherwise the
     * outer run's next halt longjmps into this (by then dead) frame, and a
     * stack-local job context skips its coherency unregister (GH3: crashes in
     * spu_coh_notify_write on a freed stack). */
    jmp_buf outer_env; const int outer_armed = s_spu_halt_armed;
    if (outer_armed) memcpy(outer_env, s_spu_halt_env, sizeof(jmp_buf));
    s_spu_halt_armed = 1;
    g_spu_trampoline_fn = 0;                        /* no stale transfer pending */
    /* Lockstep gate (env SPU_LOCKSTEP, default off): join the round-robin
     * ring and BLOCK until this ctx holds the run token, so only one lifted SPU
     * executes at a time. No-op when unarmed. The thread-local halt env above
     * makes a token pause/resume mid-run safe. */
    yz_lockstep_register(ctx);
    switch (SPU_SETJMP(s_spu_halt_env)) {
    case 1:
        halted = 1;
        g_spu_trampoline_fn = 0;
        break;
    case 2:
        /* A one-way guest stack reset invalidates every lifted host caller.
         * Re-enter its target with the same guest state on the driver stack. */
        ctx->host_depth = 0;
        g_spu_trampoline_fn = 0;
        spu_indirect_branch(ctx);
        SPU_DRAIN(ctx);
        break;
    default:
        SPU_NOUNWIND(s_spu_halt_env);
        /* SPU_DRAIN trampoline model: the top-level entry runs until its first
         * cross-function tail transfer, which sets g_spu_trampoline_fn and
         * returns; the drain loop re-enters each queued target until the SPU
         * halts (stop -> longjmp) or the trampoline empties. Nested brsl/bisl
         * calls drain inside their own call brackets (see the lifter). */
        entry(ctx);
        { static int _e = 0;
          if (getenv("SPU_ARGWATCH") && _e < 8) { _e++;
              fprintf(stderr, "[argwatch] after entry(): img=%d pc=0x%05X "
                      "r2=0x%08X r3=0x%08X r4=0x%08X\n", ctx->image_id,
                      (uint32_t)ctx->pc & SPU_LS_MASK, ctx->gpr[2]._u32[0],
                      ctx->gpr[3]._u32[0], ctx->gpr[4]._u32[0]);
              fflush(stderr); } }
        SPU_DRAIN(ctx);
    }
    yz_lockstep_unregister(ctx);   /* leave the ring; hand the token onward */
    s_spu_halt_armed = outer_armed;
    if (outer_armed) memcpy(s_spu_halt_env, outer_env, sizeof(jmp_buf));
    return halted;
}

/* ===========================================================================
 * One private MFC engine per live virtual context. Persistent policy contexts
 * outnumber physical SPUs even when their scheduling quanta run sequentially.
 * Registry nodes have stable addresses and survive a quantum, including stalled
 * lists and tag state. The registry lock protects lookup/publication/removal;
 * the context's single execution owner protects its engine after lookup.
 * Release is permitted only after that owner has stopped using the context.
 * ===========================================================================*/
typedef struct spu_mfc_slot {
    spu_context* ctx;
    mfc_engine mfc;
    struct spu_mfc_slot* next;
} spu_mfc_slot;

static spu_mfc_slot* s_mfc_slots;
static SRWLOCK s_mfc_claim_lock = SRWLOCK_INIT;

/* The engine this thread looked up last. Every DMA of a run asks for the
 * engine of the one context that runs it, and the registry's read lock on
 * each ask is two interlocked operations on one line all SPU threads share.
 * The answer stays good while no node has been freed since it was taken:
 * s_mfc_gen counts the frees (under the writer lock), and a node lives
 * until its owner releases it, which only happens once the owner has
 * stopped using the context. A context at a freed one's address -- a
 * stack context in the next job -- finds the count moved and looks up
 * again. SPU_MFC_TLS=0 asks the registry every time. */
static volatile uint32_t s_mfc_gen;
static SPU_TLS spu_context* t_mfc_ctx;
static SPU_TLS mfc_engine*  t_mfc_engine;
static SPU_TLS uint32_t     t_mfc_gen;
static int s_mfc_tls = -1;

static mfc_engine* mfc_for(spu_context* ctx)
{
    if (__builtin_expect(s_mfc_tls < 0, 0)) {
        const char* e = getenv("SPU_MFC_TLS");
        s_mfc_tls = !(e && e[0] == '0');
        { const char* c = getenv("SPU_MFC_CTXC"); if (s_mfc_tls && c && c[0] == '0') s_mfc_tls = 2; }   /* 2 = TLS only */
    }
    /* Context-resident cache first: same validity rule as the thread-local one (owner == this
     * context, no node freed since), without the TLS access. SPU_MFC_CTXC=0 skips it. */
    if (__builtin_expect(ctx->mfc_c_owner == (void*)ctx, 1) && ctx->mfc_c_gen == __atomic_load_n(&s_mfc_gen, __ATOMIC_ACQUIRE) && s_mfc_tls == 1)
        return (mfc_engine*)ctx->mfc_c_eng;
    if (t_mfc_ctx == ctx && t_mfc_gen == __atomic_load_n(&s_mfc_gen, __ATOMIC_ACQUIRE) && s_mfc_tls) {
        if (s_mfc_tls == 1) { ctx->mfc_c_owner = ctx; ctx->mfc_c_eng = t_mfc_engine; ctx->mfc_c_gen = t_mfc_gen; }
        return t_mfc_engine;
    }
    AcquireSRWLockShared(&s_mfc_claim_lock);
    for (spu_mfc_slot* slot = s_mfc_slots; slot; slot = slot->next) {
        if (slot->ctx == ctx) {
            t_mfc_ctx = ctx; t_mfc_engine = &slot->mfc; t_mfc_gen = s_mfc_gen;
            ReleaseSRWLockShared(&s_mfc_claim_lock);
            return &slot->mfc;
        }
    }
    ReleaseSRWLockShared(&s_mfc_claim_lock);

    /* A concurrent miss may publish while the read lock is released. Recheck
     * under the writer lock before creating this context's private engine. */
    AcquireSRWLockExclusive(&s_mfc_claim_lock);
    for (spu_mfc_slot* slot = s_mfc_slots; slot; slot = slot->next) {
        if (slot->ctx == ctx) {
            ReleaseSRWLockExclusive(&s_mfc_claim_lock);
            return &slot->mfc;
        }
    }
    spu_mfc_slot* slot = (spu_mfc_slot*)malloc(sizeof(*slot));
    if (!slot) {
        fprintf(stderr, "[SPU] cannot allocate private MFC engine for context %p\n", (void*)ctx);
        fflush(stderr);
        abort(); /* Shared fallback state would silently corrupt DMA completion. */
    }
    slot->ctx = ctx;
    mfc_engine_init(&slot->mfc);
    slot->next = s_mfc_slots;
    s_mfc_slots = slot;
    t_mfc_ctx = ctx; t_mfc_engine = &slot->mfc; t_mfc_gen = s_mfc_gen;
    ReleaseSRWLockExclusive(&s_mfc_claim_lock);
    return &slot->mfc;
}

void spu_mfc_release(spu_context* ctx)
{
    if (!ctx) return;
    AcquireSRWLockExclusive(&s_mfc_claim_lock);
    for (spu_mfc_slot** link = &s_mfc_slots; *link; link = &(*link)->next) {
        if ((*link)->ctx != ctx) continue;
        spu_mfc_slot* slot = *link;
        *link = slot->next;
        __atomic_add_fetch(&s_mfc_gen, 1, __ATOMIC_RELEASE);   /* before the node goes */
        free(slot);
        break;
    }
    ReleaseSRWLockExclusive(&s_mfc_claim_lock);
}

/* Exit safety net for contexts unwound off a departing thread's stack.
 * Pointer comparisons suffice: the abandoned context need not be readable. */
void spu_mfc_forget_range(uintptr_t lo, uintptr_t hi)
{
    AcquireSRWLockExclusive(&s_mfc_claim_lock);
    for (spu_mfc_slot** link = &s_mfc_slots; *link;) {
        spu_mfc_slot* slot = *link;
        uintptr_t owner = (uintptr_t)slot->ctx;
        if (owner >= lo && owner < hi) {
            *link = slot->next;
            __atomic_add_fetch(&s_mfc_gen, 1, __ATOMIC_RELEASE);
            free(slot);
        } else {
            link = &slot->next;
        }
    }
    ReleaseSRWLockExclusive(&s_mfc_claim_lock);
}

/* ===========================================================================
 * Atomic reservation (GETLLAR / PUTLLC / PUTLLUC) -- real lock-line semantics
 *
 * Multiple SPU kernel threads (the SPURS workload runtime runs several SPUs on
 * one shared lock-free queue) issue GETLLAR/PUTLLC on the SAME 128-byte lines.
 * Without honoring the reservation, two SPUs both "claim" the same slot and the
 * queue corrupts (observed: the 2nd claim returns garbage [1,1,1,1] -> the SPU
 * traps). PUTLLC must FAIL when the line changed since GETLLAR. We implement the
 * compare-and-swap under one global lock across all SPU host threads.
 * ===========================================================================*/
extern uint8_t* vm_base;

/* The spinlock guarding all atomic line ops now lives in spu_coherency.c as
 * spu_lockline_lock/unlock. It used to be a file-static here, which serialized
 * SPU against SPU but left the PPU free to store into a line in the middle of
 * a PUTLLC's compare-and-commit -- the update the SPU was about to make would
 * be written over, and the SPU would never learn the line had changed. Same
 * lock, same critical sections, now shared with the PPU store paths. */

/* Total PUTLLC attempts (all SPUs). The PM flow trace (spurs_policy.c) reads
 * the delta across one policy run to find the run that performed a claim. */
volatile unsigned g_spu_putllc_count = 0;
/* PUTLLCs that hit the WATCHED sync line (g_barrier_sync_watch) -- isolates the
 * work-run on the stalled job queue from the dozen idle jobmanager instances. */
volatile unsigned g_spu_putllc_sync_hit = 0;

/* Lowest LS address treated as "not job code". Jobs load at 0 and
 * their stacks top out well below this; a branch at or above it with no
 * lifted code is a runaway, not a service call. */
#define SPU_JM2_KERNEL_BASE 0x15000u

/* Link-register value handed to a SPURS job so its final `bi $r0` lands
 * somewhere we recognise. The real job manager passes a return address into
 * itself; we have no manager, so we pass this and treat arriving here as job
 * completion. Chosen above the context/descriptor we park near the top of LS. */
#define SPU_JOB_RETURN_LS   0x3FF00u

/* Returns 1 if `cmd` is an atomic line op and was handled here, else 0. */
/* GETLLAR poll backoff by EVENT instead of by timer (SPU_GETLLAR_EVWAIT=1, opt-in:
 * no measurable gain in the bench; default = the old Sleep(0)/Sleep(1) ladder). The caller saw the same 128-byte line
 * `t_pn` times in a row: its writer is another processor. A short spin watches the
 * line (costs one cache-line read per pause), then the SPU reserves the line and
 * sleeps on its channel CV: any PPU store / peer PUT into a reserved line raises
 * SPU_EVENT_LR and wakes it (spu_coh_notify_write*), ~tens of us instead of the >=1 ms
 * a Sleep(1) costs (the graphics policy module polls its job queue this way). The
 * sleep is bounded by 1 ms, so a write that does not notify (a path that skips the
 * coherence test) costs what Sleep(1) did. The LR bit our reservation raised is
 * removed again unless it was already set: the guest never asked for that event.
 * Returns 1 when it waited/observed, 0 to use the old ladder. */
#ifdef _WIN32
static int spu_getllar_evwait_inner(spu_context* ctx, uint32_t ea, const uint8_t* prev);
/* SPU_GETLLAR_BURST_US=<us> (31/10 spurs34, default 25; 0 = off): the PPU feeds wid0's job list in bursts of ~100 stores
 * 1-2 us apart into the polled line; every store woke the lane from its channel CV, which re-read the line, found the
 * list still incomplete and went back to sleep (~325 entries into the wait per wid0 run, each a lock-line lock, a
 * reservation, a CV sleep, and a wake call on the PPU's store path). When the previous wait of this thread ended less
 * than <us> ago the writer is mid-burst: spin on the line for up to <us> before arming the reservation. */
static int spu_getllar_evwait(spu_context* ctx, uint32_t ea, const uint8_t* prev)
{
    static long long s_bq = -1; static SPU_THREAD_LOCAL long long t_bw_last;
    if (s_bq < 0) { const char* e = getenv("SPU_GETLLAR_BURST_US"); long long us = e ? atoll(e) : 25;   /* default ON 31/10: 4/4 alternating combat pairs >= off, evwait entries per wid0 run -80% */
                    LARGE_INTEGER f; QueryPerformanceFrequency(&f); s_bq = us > 0 ? us * f.QuadPart / 1000000 : 0; }
    if (!s_bq) return spu_getllar_evwait_inner(ctx, ea, prev);
    const unsigned long long r35t0 = g_r35_on ? __rdtsc() : 0;
    if (g_r35_on) g_r35c[R35_EVW_N]++;
    LARGE_INTEGER q; QueryPerformanceCounter(&q);
    if (t_bw_last && q.QuadPart - t_bw_last < s_bq) {
        const uint8_t* mem = vm_base + ea; const long long end = q.QuadPart + s_bq;
        do {
            _mm_pause();
            if (memcmp(mem, prev, MFC_ATOMIC_LINE) != 0) { QueryPerformanceCounter(&q); t_bw_last = q.QuadPart; if (g_r35_on) g_r35c[R35_EVW_TSC] += __rdtsc() - r35t0; return 1; }
            QueryPerformanceCounter(&q);
        } while (q.QuadPart < end);
    }
    int r = spu_getllar_evwait_inner(ctx, ea, prev);
    QueryPerformanceCounter(&q); t_bw_last = q.QuadPart;
    if (g_r35_on) g_r35c[R35_EVW_TSC] += __rdtsc() - r35t0;
    return r;
}
static int spu_getllar_evwait_inner(spu_context* ctx, uint32_t ea, const uint8_t* prev)
{
    static int s_on = -1; static int s_spin = 150;
    if (s_on < 0) {
        const char* e = getenv("SPU_GETLLAR_EVWAIT"); s_on = !(e && e[0] == '0');   /* default ON 26/10: wid0 lane wall -14%, same total CPU, no Sleep(1) ladder */
        const char* sp = getenv("SPU_GETLLAR_SPIN"); if (sp) s_spin = atoi(sp);
    }
    if (!s_on) return 0;
    const uint8_t* mem = vm_base + ea;
    for (int i = 0; i < s_spin; i++) {
        _mm_pause();
        if (memcmp(mem, prev, MFC_ATOMIC_LINE) != 0) return 1;
    }
    const uint32_t had = ctx->event_status & SPU_EVENT_LR;
    spu_lockline_lock();
    spu_coh_reserve(ctx, ea);
    memcpy(ctx->resv_line, mem, MFC_ATOMIC_LINE);
    ctx->resv_ea = ea; ctx->resv_valid = 1;
    spu_lockline_unlock();
    if (memcmp(mem, prev, MFC_ATOMIC_LINE) != 0) return 1;   /* written before we armed */
    __atomic_fetch_add(&ctx->ch_waiters, 1, __ATOMIC_SEQ_CST);
    AcquireSRWLockExclusive((SRWLOCK*)&ctx->ch_wait_lock);
    if (!(ctx->event_status & SPU_EVENT_LR) && memcmp(mem, prev, MFC_ATOMIC_LINE) == 0) {
        /* SPU_GETLLAR_EVSTATS=1: how often the wait ends by a write notification, by the 1 ms timeout, and the time each takes */
        static int s_st = -1; if (s_st < 0) { const char* e = getenv("SPU_GETLLAR_EVSTATS"); s_st = e && e[0] == '1'; }
        LARGE_INTEGER q0, q1; if (s_st) QueryPerformanceCounter(&q0);
        const unsigned long long r35s0 = g_r35_on ? __rdtsc() : 0;
        BOOL ok = SleepConditionVariableSRW((CONDITION_VARIABLE*)&ctx->ch_wait_cv, (SRWLOCK*)&ctx->ch_wait_lock, 1, 0);
        if (g_r35_on) g_r35c[R35_EVW_SLP] += __rdtsc() - r35s0;
        if (s_st) {
            static volatile long long n_ok, n_to, us_ok, us_to; static volatile long n_all; LARGE_INTEGER f; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&q1);
            long long us = (q1.QuadPart - q0.QuadPart) * 1000000 / f.QuadPart;
            if (ok) { __atomic_add_fetch(&n_ok, 1, __ATOMIC_RELAXED); __atomic_add_fetch(&us_ok, us, __ATOMIC_RELAXED); }
            else    { __atomic_add_fetch(&n_to, 1, __ATOMIC_RELAXED); __atomic_add_fetch(&us_to, us, __ATOMIC_RELAXED); }
            if ((__atomic_add_fetch(&n_all, 1, __ATOMIC_RELAXED) % 500) == 0)
                fprintf(stderr, "[evwait] waits=%ld woken=%lld (avg %lld us) timeout=%lld (avg %lld us)\n", (long)n_all, n_ok, n_ok ? us_ok / n_ok : 0, n_to, n_to ? us_to / n_to : 0);
        }
    }
    ReleaseSRWLockExclusive((SRWLOCK*)&ctx->ch_wait_lock);
    __atomic_fetch_sub(&ctx->ch_waiters, 1, __ATOMIC_SEQ_CST);
    if (!had) __atomic_fetch_and(&ctx->event_status, ~(uint32_t)SPU_EVENT_LR, __ATOMIC_RELAXED);
    return 1;
}
#endif

/* SPURS_CPU_STATS run accounting (read by cellSpurs.c spurs_pm_worker): GETLLARs, GETLLARs that
 * re-read an unchanged line, entries into the poll back-off (Sleep(0)/Sleep(1)/evwait) and the
 * time spent inside it. Counted per host thread; diagnostic only. */
SPU_THREAD_LOCAL unsigned long long g_ra_getllar, g_ra_unch, g_ra_back, g_ra_back_qpc, g_ra_s1, g_ra_s1_qpc;
/* Where the back-off time goes: (line ea, caller lr) -> entries / QPC ticks, per thread. */
static SPU_THREAD_LOCAL struct { uint32_t ea, lr, pc; unsigned long long n, q; } t_ra_site[24];
static void spu_ra_site(uint32_t ea, uint32_t lr, uint32_t pc, unsigned long long q)
{
    int f = -1;
    for (int i = 0; i < 24; i++) {
        if (t_ra_site[i].n && t_ra_site[i].ea == ea && t_ra_site[i].lr == lr) { t_ra_site[i].n++; t_ra_site[i].q += q; return; }
        if (!t_ra_site[i].n && f < 0) f = i;
    }
    if (f >= 0) { t_ra_site[f].ea = ea; t_ra_site[f].lr = lr; t_ra_site[f].pc = pc; t_ra_site[f].n = 1; t_ra_site[f].q = q; }
}
void spu_ra_dump_reset(void)
{
    LARGE_INTEGER f; QueryPerformanceFrequency(&f);
    for (int i = 0; i < 24; i++) if (t_ra_site[i].n) {
        fprintf(stderr, "[lsite] ea=%08X lr=%05X pc=%05X n=%llu us=%.0f\n", t_ra_site[i].ea, t_ra_site[i].lr, t_ra_site[i].pc, t_ra_site[i].n, t_ra_site[i].q * 1e6 / f.QuadPart);
        t_ra_site[i].n = 0;
    }
}
static int spu_mfc_atomic(spu_context* ctx, uint32_t cmd)
{
    /* Classify FIRST. Every MFC_Cmd write lands here, and only the switch at
     * the bottom used to filter -- so the uncommitted-EA guard below reported
     * ORDINARY DMA as `[spu-atomic]`. A plain put (0x20) to a garbage EA read
     * as an atomic spin, which is a genuinely misleading place to start
     * debugging from. Non-atomic commands belong to the DMA engine. */
    if (cmd != MFC_GETLLAR_CMD && cmd != MFC_PUTLLC_CMD &&
        cmd != MFC_PUTLLUC_CMD && cmd != MFC_PUTQLLUC_CMD)
        return 0;

    if (__builtin_expect(g_r35_on, 0)) g_r35c[cmd == MFC_GETLLAR_CMD ? R35_GETLLAR : cmd == MFC_PUTLLC_CMD ? R35_PUTLLC : R35_PUTOTH]++;
    uint32_t ea  = ctx->mfc_eal & ~(uint32_t)(MFC_ATOMIC_LINE - 1);
    uint32_t lsa = ctx->mfc_lsa & SPU_LS_MASK;
    uint8_t* ls  = &ctx->ls[lsa];
    uint8_t* mem = vm_base + ea;

    { static int s_t = -1; if (s_t < 0) s_t = getenv("SPU_POLLTRACE") ? 1 : 0;
      if (s_t) { static uint64_t s_n = 0; static uint32_t s_lastea = 0;
        if ((++s_n % 2000000) == 0 || ea != s_lastea) {
          if ((s_n % 2000000) == 0)
            fprintf(stderr, "[atomcnt] %llu atomic ops; last cmd=0x%X ea=0x%08X\n",
                    (unsigned long long)s_n, cmd, ea);
          s_lastea = ea; } } }
    /* SPU_ATOMTRACE=1 keeps the old 40-line cap; =<N> raises it. A busy image
     * (FMOD) exhausts 40 before a quieter one issues its first atomic, which
     * reads as "that image never does atomics" when it simply never got a line. */
    { static int s_at = -1;
      if (s_at < 0) { const char* e = getenv("SPU_ATOMTRACE");
                      int v = e ? atoi(e) : 0; s_at = e ? (v > 1 ? v : 40) : 0; }
      if (s_at) { static int _a=0; if (_a < s_at && ++_a)
        fprintf(stderr, "[atom] cmd=0x%02X ea=0x%08X (img=%d)\n", cmd, ea, ctx->image_id); } }
    /* cri task (img22) atomic on the taskset: dump the loaded bitset line so we can
     * see if the task reads MY taskset (0x4005E000) with my READY bit, or elsewhere. */
    { static int s_ct=-1; if(s_ct<0) s_ct=getenv("SPU_ATOMTRACE")?1:0;
      if(s_ct && ctx->image_id == spu_cri_image() && cmd==0xD0 && mfc_ea_range_committed(ea,16)) {
        static int _c=0; if (_c < 24 && ++_c){
          uint8_t* m=vm_base+ea;
          #define BW(o) (((uint32_t)m[o]<<24)|((uint32_t)m[o+1]<<16)|((uint32_t)m[o+2]<<8)|m[o+3])
          fprintf(stderr,"[cri-atom] GETLLAR ea=0x%08X line[0..0x30]: %08X %08X %08X %08X | %08X %08X %08X %08X | %08X %08X %08X %08X\n",
            ea, BW(0),BW(4),BW(8),BW(0xC), BW(0x10),BW(0x14),BW(0x18),BW(0x1C), BW(0x20),BW(0x24),BW(0x28),BW(0x2C));
          #undef BW
        } } }
    /* YDKJ_CRI_R4: dump the CellSpursTaskset bitsets when the policy atomically
     * touches my taskset (0x0F000000), to watch the task-activation state machine
     * (why task0 isn't selected+first-run). running@0 ready@0x10 pending@0x20
     * enabled@0x30 signalled@0x40 waiting@0x50 (each 16B; word0 = MSB, task0=bit127). */
    { static int s_td = -1; if (s_td < 0) s_td = (getenv("YDKJ_CRI_CHAIN") && getenv("SPU_ATOMTRACE")) ? 1 : 0;
      if (s_td && ea >= 0x0F000000u && ea < 0x0F001900u) {
        extern uint8_t* vm_base;
        static int _t=0; if (vm_base && (_t < 24 && ++_t)) {
            uint8_t* t = vm_base + 0x0F000000u;
            #define TW(o) (((uint32_t)t[o]<<24)|((uint32_t)t[o+1]<<16)|((uint32_t)t[o+2]<<8)|t[o+3])
            fprintf(stderr, "[tset] %s run=%08X rdy=%08X pnd=%08X ena=%08X sig=%08X wait=%08X | wid=%08X last=%02X\n",
                    cmd==0xD0?"GET":cmd==0xB4?"PUT":"?", TW(0x00), TW(0x10), TW(0x20), TW(0x30), TW(0x40), TW(0x50), TW(0x74), t[0x73]);
            #undef TW
        }
      } }

    /* Guard atomic line ops against an uncommitted/garbage EA (e.g. a SPURS
     * policy computing a lock-line address from an incomplete instance context).
     * Same rationale as the DMA EA guard: a bad guest atomic must not segfault
     * the host. GETLLAR returns a zeroed line (no reservation); PUTLLC fails. */
    if (!mfc_ea_range_committed(ea, MFC_ATOMIC_LINE)) {
        static int s_w = 0;
        if (s_w < 16 && ++s_w)
            fprintf(stderr, "[spu-atomic] cmd=0x%X ea=0x%08X pc=0x%05X img=%d uncommitted -- skipped\n",
                    cmd, ea, (uint32_t)ctx->pc & SPU_LS_MASK, ctx->image_id);
        if (cmd == MFC_GETLLAR_CMD) {
            memset(ls, 0, MFC_ATOMIC_LINE);
            ctx->resv_ea = ea; ctx->resv_valid = 0; ctx->atomic_stat = 4;
        } else {
            ctx->atomic_stat = 1;   /* PUTLLC failure (line "moved") */
        }
        return 1;
    }

    /* SPU_ATOM_EA=<hex>: log every lock-line atomic touching that EA`s 128B line.
     * SPU_ATOM_FULL=1 additionally dumps 32 bytes of the line before and after.
     *
     * Built into ONE buffer and written with a single fprintf: several SPU host
     * threads and the PPU all write stderr, so a dump split across many fprintf
     * calls comes out shredded and unreadable exactly when it matters. */
    { static int s_ae = -1; static uint32_t s_aea;
      if (s_ae < 0) { const char* e = getenv("SPU_ATOM_EA");
        s_aea = e ? (uint32_t)strtoul(e, 0, 16) & ~127u : 0; s_ae = s_aea ? 1 : 0;
        if (s_ae == 1) { extern uint32_t g_barrier_sync_watch;
            g_barrier_sync_watch = s_aea; } /* arm PUTLLC OK/FAIL verdict log */ }
      if (s_ae == 1 && ((uint32_t)ea & ~127u) == s_aea) {
          static int _n = 0, s_cap = -1;
          /* SPU_ATOM_EA_MAX=<n>: lines to print (default 48, 0 = unlimited). */
          if (s_cap < 0) { const char* m = getenv("SPU_ATOM_EA_MAX"); s_cap = m ? atoi(m) : 48; }
          if (!s_cap || (_n < s_cap && ++_n)) {
              extern uint8_t* vm_base;
              const uint8_t* r = vm_base + ((uint32_t)ea & ~127u);
              static int s_af = -1;
              if (s_af < 0) { const char* f = getenv("SPU_ATOM_FULL"); s_af = f ? (atoi(f) > 1 ? 2 : 1) : 0; }
              char buf[1024]; int p = 0;
              /* lr (gpr[0]) as well as pc: these atomics sit in generic helpers,
               * so pc names the HELPER and only lr names the real caller. */
              p += snprintf(buf + p, sizeof buf - p,
                            "[atom-ea] cmd=0x%X img=%d pc=0x%05X lr=0x%05X ea=0x%08X",
                            cmd, ctx->image_id, (uint32_t)ctx->pc & SPU_LS_MASK,
                            ctx->gpr[0]._u32[0] & SPU_LS_MASK, (uint32_t)ea);
              int nb = s_af == 2 ? 128 : s_af ? 64 : 8;   /* 64 covers +0x30 pendingRecv; =2 the whole line */
              p += snprintf(buf + p, sizeof buf - p, " RAM=");
              for (int i = 0; i < nb && p < (int)sizeof buf - 4; i++)
                  p += snprintf(buf + p, sizeof buf - p, "%02X%s", r[i],
                                (i % 4 == 3) ? " " : "");
              if (cmd == MFC_PUTLLC_CMD) {
                  p += snprintf(buf + p, sizeof buf - p, " STORE=");
                  for (int i = 0; i < nb && p < (int)sizeof buf - 4; i++)
                      p += snprintf(buf + p, sizeof buf - p, "%02X%s", ls[i],
                                    (i % 4 == 3) ? " " : "");
              }
              fprintf(stderr, "%s\n", buf); fflush(stderr);
          }
      } }
    /* Bink sync-line atomic trace (armed by the PPU-side producer probe). */
    { extern uint32_t g_barrier_sync_watch;
      uint32_t b = g_barrier_sync_watch;
      if (b && ea >= (b & ~127u) && ea < ((b + 0xC0 + 127) & ~127u)) {
          static int _n = 0;
          if (_n < 64 && ++_n)
              fprintf(stderr, "[sync-atomic] %s img=%d pc=0x%05X ea=0x%08X\n",
                      cmd == MFC_GETLLAR_CMD ? "GETLLAR" :
                      cmd == MFC_PUTLLC_CMD ? "PUTLLC" : "PUTLLUC",
                      ctx->image_id, (uint32_t)ctx->pc, ea);
      } }
    switch (cmd) {
    case MFC_GETLLAR_CMD:
        /* Lockstep tick: a guest GETLLAR..PUTLLC poll loop is intra-function
         * (all gotos), so it never crosses the SPU_DRAIN tick site -- without a
         * tick here a polling SPU would hold the run token forever while the
         * peer that must WRITE the line waits for it (canersaka ticks the
         * GETLLAR fast+slow paths for exactly this reason). */
        yz_lockstep_tick(ctx);
        /* Back off a poll loop. A task that re-reads one line whose content has
         * not changed is waiting for another processor to write it; on the SPU
         * that costs nothing, here it pins a host core (Drakengard 3's Edge
         * zlib task polled its work queue this way at 100% of a core, hundreds
         * of thousands of GETLLARs a second). After 64 unchanged re-reads
         * sleep 50 us per poll, after 1024, 200 us: a real producer is seen
         * within that, a spin stops costing. SPU_GETLLAR_BACKOFF=0 disables. */
        { static SPU_THREAD_LOCAL uint32_t t_pea; static SPU_THREAD_LOCAL unsigned t_pn;
          static SPU_THREAD_LOCAL long long t_pq0;   /* QPC when this unchanged-poll streak passed 64 */
          static SPU_THREAD_LOCAL unsigned long long t_ptsc;   /* TSC of the previous GETLLAR on this thread */
          static SPU_THREAD_LOCAL uint8_t t_prev[MFC_ATOMIC_LINE];
          static int s_bo = -1;
          if (s_bo < 0) { const char* e = getenv("SPU_GETLLAR_BACKOFF"); s_bo = (e && *e == '0') ? 0 : 1; }
          if (s_bo) {
              const uint8_t* now = vm_base + ea;
              g_ra_getllar++;
              /* SPU_GETLLAR_GAPUS=<us> (0 = off): an unchanged-poll streak only counts while the thread keeps
               * polling. Across a gap (the lane ran something else, or the worker slept between runs) the ladder
               * restarted nowhere: the next run's first polls of the same unchanged line went straight to the
               * Sleep(1) rung (~1.6 ms). */
              { static unsigned long long s_gap = ~0ull;
                if (s_gap == ~0ull) { const char* e = getenv("SPU_GETLLAR_GAPUS"); long long us = e ? atoll(e) : 0;
                    s_gap = 0;
                    if (us > 0) { LARGE_INTEGER f, a, b; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&a);
                        unsigned long long r0 = __rdtsc();
                        do { QueryPerformanceCounter(&b); } while ((b.QuadPart - a.QuadPart) * 1000000 < 500 * f.QuadPart);
                        unsigned long long r1 = __rdtsc();
                        s_gap = (unsigned long long)(us * ((double)(r1 - r0) / 500.0)); } }
                if (s_gap) { unsigned long long tsc = __rdtsc(); if (t_ptsc && tsc - t_ptsc > s_gap) t_pn = 0; t_ptsc = tsc; } }
              if (ea == t_pea && memcmp(now, t_prev, MFC_ATOMIC_LINE) == 0) {
                  g_ra_unch++;
                  static int s_nth = -1;   /* SPU_GETLLAR_NTH: unchanged re-reads before the back-off starts (default 64) */
                  if (s_nth < 0) { const char* e = getenv("SPU_GETLLAR_NTH"); s_nth = e ? atoi(e) : 64; if (s_nth < 1) s_nth = 1; }
                  if (++t_pn > (unsigned)s_nth) {
                      static int s_ra = -1; if (s_ra < 0) s_ra = getenv("SPURS_CPU_STATS") ? 1 : 0;
                      LARGE_INTEGER ra0, ra1; if (s_ra) QueryPerformanceCounter(&ra0);
                      /* SPU_GETLLAR_S1US=<us> (0 = off, the count-only ladder): Sleep(1) (a ~1.6 ms wake-up for the
                       * next queue entry, measured 0.25-0.5 per wid0 run) only once the streak has lasted that long;
                       * until then yield with Sleep(0). A streak longer than the threshold still sleeps. */
                      static long long s_s1q = -2;
                      if (s_s1q == -2) { const char* e = getenv("SPU_GETLLAR_S1US"); long long us = e ? atoll(e) : 0;
                                         LARGE_INTEGER f; QueryPerformanceFrequency(&f); s_s1q = us > 0 ? us * f.QuadPart / 1000000 : 0; }
                      int s1_ok = 1;
                      if (s_s1q) { LARGE_INTEGER q; QueryPerformanceCounter(&q);
                                   if (t_pn == 65) t_pq0 = q.QuadPart;
                                   s1_ok = (q.QuadPart - t_pq0) >= s_s1q; }
                      struct timespec d = { 0, t_pn > 1024 ? 200000 : 50000 };
#ifdef _WIN32
                      if (spu_getllar_evwait(ctx, ea, t_prev)) { /* the line changed, or a write to it woke us */ } else
                      { static int s_s1 = -1;   /* SPU_GETLLAR_SLEEP1=0: never Sleep(1) (a >1 ms wake-up for the next job), only yield */
                        if (s_s1 < 0) { const char* e = getenv("SPU_GETLLAR_SLEEP1"); s_s1 = !(e && e[0] == '0'); }
                        Sleep(t_pn > 1024 && s_s1 && s1_ok ? 1 : 0); }
#else
                      nanosleep(&d, NULL);
#endif
                      if (s_ra) { QueryPerformanceCounter(&ra1); g_ra_back++; g_ra_back_qpc += ra1.QuadPart - ra0.QuadPart;
                                  spu_ra_site((uint32_t)ea, ctx->gpr[0]._u32[0] & SPU_LS_MASK, (uint32_t)ctx->pc & SPU_LS_MASK, ra1.QuadPart - ra0.QuadPart);
                                  if (t_pn > 1024 && s1_ok) { g_ra_s1++; g_ra_s1_qpc += ra1.QuadPart - ra0.QuadPart; } }
                  }
              } else { t_pea = ea; t_pn = 0; }
              memcpy(t_prev, now, MFC_ATOMIC_LINE);
          } }
        { extern volatile int g_tl_on; if (g_tl_on) { extern void tl_getllar(uint32_t, const uint8_t*); tl_getllar((uint32_t)ea, vm_base + (uint32_t)ea); } }   /* TOD_TIMELINE */
        spu_lockline_lock();
        /* Tell the PPU store paths this line is live, so a store into it goes
         * through the lock and raises SPU_EVENT_LR here instead of landing
         * unannounced. Nothing else about this transaction changes. */
        spu_coh_reserve(ctx, ea);

        /* ONE read of guest memory, then the snapshot from that copy.
         *
         * This used to memcpy from `mem` twice -- once to the local store, once
         * to resv_line -- and a PPU store landing BETWEEN the two reads gave the
         * SPU a stale line and a fresh snapshot. That is a permanent deadlock,
         * and it is the one this port spent a session chasing: the SPU compares
         * its (stale) copy, finds produced == consumed, and sleeps on
         * MFC_LLR_LOST_EVENT; spu_resv_lost_poll then compares memory against a
         * snapshot that ALREADY has the new value, finds no difference, and
         * never raises the event. Measured at a freeze: mirror 0x9E, snapshot
         * 0x9F, memory 0x9F, and the SPU polling rchcnt 1.6 billion times.
         *
         * Hardware cannot produce that state: GETLLAR is a single atomic
         * 128-byte read, so the reserved data and the reservation come from the
         * same instant. Copying the snapshot from `ls` restores that property,
         * and a store that lands after the read now leaves BOTH stale -- which
         * is what makes the lost-reservation event fire. */
        /* Torn-read guard: a plain 128-byte memcpy reads the line low to high while the PPU may be storing a
         * command word (low) and then the producer count (high); the SPU could then see the NEW count with the OLD
         * command (a bogus handler index -> branch into garbage -> the collision SPU dies and the PPU waits forever).
         * Hardware GETLLAR is atomic. Re-read until two consecutive copies agree (bounded). SPU_GETLLAR_STABLE=0 = old. */
        { static int s_gs = -1; if (s_gs < 0) { const char* e = getenv("SPU_GETLLAR_STABLE"); s_gs = !(e && e[0] == '0'); }
          memcpy(ls, mem, MFC_ATOMIC_LINE);              /* line -> local store */
          if (s_gs) { uint8_t chk[MFC_ATOMIC_LINE];
              for (int k = 0; k < 8; k++) {
                  __atomic_signal_fence(__ATOMIC_SEQ_CST);
                  memcpy(chk, mem, MFC_ATOMIC_LINE);
                  if (memcmp(chk, ls, MFC_ATOMIC_LINE) == 0) break;
                  memcpy(ls, chk, MFC_ATOMIC_LINE); } } }
        /* Snapshot from `ls`, NOT a second read of `mem`. Hardware GETLLAR is a
         * single atomic 128-byte read, so the reserved data and the reservation
         * come from the same instant. Reading guest memory twice lets a PPU store
         * land between them, leaving the SPU a STALE line and a FRESH snapshot --
         * and then both halves of the wait fail: the SPU compares its stale copy,
         * sees produced == consumed and sleeps on MFC_LLR_LOST_EVENT, while the
         * reservation poll compares memory against a snapshot that already holds
         * the new value, finds no difference, and never raises the event. Nobody
         * wakes anybody. Snapshotting from `ls` makes a later store leave BOTH
         * stale, which is exactly what makes the lost-reservation event fire. */
        memcpy(ctx->resv_line, ls, MFC_ATOMIC_LINE);   /* snapshot for compare */
        /* MFC_RdAtomicStat after a GETLLAR reads 4 (GETLLAR complete), not 0:
         * Havok's SPU allocator (GH3 collide task, LS 0xA66C) re-issues the
         * GETLLAR until that bit is set, and spun forever on 0. PUTLLUC is 2. */
        ctx->resv_ea = ea; ctx->resv_valid = 1; ctx->atomic_stat = 4;
        ctx->dbg_getllar++;
        spu_hangline_note('G', ctx->image_id, ctx->spu_id, (uint32_t)ctx->pc & SPU_LS_MASK, ea, ls);
        /* A GETLLAR re-issued millions of times on one line is an SPU spin-wait
         * that nobody is releasing (a ticket lock, a counter barrier). Say so
         * once per line with the line's head, so the stuck word is visible
         * without a per-function probe. */
        { static SPU_THREAD_LOCAL uint32_t t_ea; static SPU_THREAD_LOCAL unsigned long t_n;
          if (ea != t_ea) { t_ea = ea; t_n = 0; }
          else if (++t_n == 4000000) {
              const uint8_t* m = vm_base + ea;
              fprintf(stderr, "[spu-spin] img=%d pc=0x%05X lr=0x%05X GETLLAR ea=0x%08X x4M; line:"
                              " %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X\n",
                      ctx->image_id, (unsigned)(ctx->pc & SPU_LS_MASK), ctx->gpr[0]._u32[0] & SPU_LS_MASK, ea,
                      m[0],m[1],m[2],m[3], m[4],m[5],m[6],m[7], m[8],m[9],m[10],m[11], m[12],m[13],m[14],m[15]);
          } }
        spu_lockline_unlock();
        /* SPU_LLARWATCH=<hex EA>: every GETLLAR of that line, with the LSA it
         * used and the first word as it lands in BOTH places.
         *
         * The deadlock this settles: at a freeze the reservation snapshot holds
         * the fresh produced value (0xC9) while the SPU's local-store mirror
         * still holds the stale one (0xC8) -- and the compare reads the mirror.
         * Both memcpys above copy from the same `mem`, so that can only happen
         * if `lsa` was not where the SPU asked. Reading ctx->mfc_lsa from a
         * later poll cannot prove it (by then the SPU has issued other DMA and
         * overwritten the field, which is exactly the compare-two-moments
         * mistake that has cost this port ten retractions). So log it HERE, at
         * the GETLLAR, with the word that actually landed. */
        { static int s_lw2 = -2; static uint32_t s_lwea;
          if (s_lw2 == -2) { const char* e6 = getenv("SPU_LLARWATCH");
                             s_lw2 = e6 ? 1 : 0;
                             s_lwea = e6 ? (uint32_t)strtoul(e6, 0, 16) & ~127u : 0u; }
          if (s_lw2 && (ea & ~127u) == s_lwea) {
              static unsigned long ln;
              /* A histogram of the LSAs this line's GETLLARs use, not a
               * sample of them. resv_line has exactly one writer -- this
               * GETLLAR -- and it copies to &ls[mfc_lsa] from the same source,
               * so a snapshot that is fresh while LS 0x10800 is stale can only
               * mean some GETLLAR ran with a DIFFERENT mfc_lsa. Sampling every
               * 256th hit would miss exactly those. Count them all and print
               * every LSA seen. */
              { static uint32_t seen[8]; static unsigned long cnt[8]; static int ns;
                int f = -1;
                for (int z = 0; z < ns; z++) if (seen[z] == lsa) { f = z; break; }
                if (f < 0 && ns < 8) { f = ns; seen[ns] = lsa; cnt[ns] = 0; ns++; }
                if (f >= 0) cnt[f]++;
                if ((ln % 4096) == 0) {
                    fprintf(stderr, "[llar-lsa] %lu GETLLARs on 0x%08X;", ln, ea);
                    for (int z = 0; z < ns; z++)
                        fprintf(stderr, " lsa=0x%05X x%lu", seen[z], cnt[z]);
                    fprintf(stderr, "\n"); fflush(stderr);
                } }
              if (++ln <= 8 || (ln % 256) == 0) {
                  const uint8_t* q = (const uint8_t*)ls;
                  const uint32_t landed = ((uint32_t)q[0] << 24) | ((uint32_t)q[1] << 16)
                                        | ((uint32_t)q[2] << 8) | q[3];
                  const uint8_t* m8 = (const uint8_t*)mem;
                  const uint32_t frommem = ((uint32_t)m8[0] << 24) | ((uint32_t)m8[1] << 16)
                                         | ((uint32_t)m8[2] << 8) | m8[3];
                  const uint8_t* mir = ctx->ls + 0x10800u;
                  const uint32_t mirw = ((uint32_t)mir[0] << 24) | ((uint32_t)mir[1] << 16)
                                      | ((uint32_t)mir[2] << 8) | mir[3];
                  fprintf(stderr, "[llar] n=%lu spu%u pc=0x%05X lsa=0x%05X"
                                  " ea=0x%08X mem=0x%08X landed=0x%08X"
                                  " mirror[0x10800]=0x%08X r90=0x%08X\n",
                          ln, (unsigned)(ctx->spu_id & 7u),
                          (uint32_t)ctx->pc & SPU_LS_MASK, lsa, ea,
                          frommem, landed, mirw, ctx->gpr[90]._u32[0]);
                  fflush(stderr);
              }
          } }
        return 1;

    case MFC_PUTLLC_CMD:
        /* g_spu_putllc_count is no longer counted: nothing reads it, and a shared
         * counter bumped by every PUTLLC bounced a cache line between SPU threads. */
        { extern uint32_t g_barrier_sync_watch;
          uint32_t b = g_barrier_sync_watch;
          if (b && (ea & ~127u) == (b & ~127u)) g_spu_putllc_sync_hit++; }
        spu_lockline_lock();
        if (ctx->resv_valid && ctx->resv_ea == ea &&
            memcmp(mem, ctx->resv_line, MFC_ATOMIC_LINE) == 0) {
            { extern void spu_lockguard_check(spu_context*, uint32_t, const uint8_t*, uint32_t, const char*);
              spu_lockguard_check(ctx, ea, ls, MFC_ATOMIC_LINE, "PUTLLC"); }
            memcpy(mem, ls, MFC_ATOMIC_LINE);          /* commit local store */
            /* The run-time range watch (spu_dma.h) sees plain DMA only; an
             * atomic commit into the watched line is reported here too. */
            { extern uint32_t g_spu_watchea_dyn, g_spu_watchea_dyn_len;
              extern uint32_t g_spu_dma_rng_lo, g_spu_dma_rng_hi; extern void spu_dma_range_init(void);
              static int s_init = 0; if (!s_init) { s_init = 1; spu_dma_range_init(); }
              if (g_spu_dma_rng_hi && ea < g_spu_dma_rng_hi && g_spu_dma_rng_lo < ea + MFC_ATOMIC_LINE) {
                  static long rn = 0; extern long g_spu_dma_rng_cap;
                  if (g_spu_dma_rng_cap == 0 || (rn < g_spu_dma_rng_cap && ++rn)) {
                      uint32_t off = (g_spu_dma_rng_lo > ea ? g_spu_dma_rng_lo - ea : 0) & ~15u;
                      fprintf(stderr, "[dma-range] img=%d PUTLLC line=0x%08X pc=0x%05X at+0x%02X:",
                              ctx->image_id, ea, (uint32_t)ctx->pc & SPU_LS_MASK, off);
                      for (uint32_t k = off; k < off + 32 && k < MFC_ATOMIC_LINE; k += 4)
                          fprintf(stderr, " %02X%02X%02X%02X", ls[k], ls[k+1], ls[k+2], ls[k+3]);
                      fputc('\n', stderr); } }
              if (g_spu_watchea_dyn && ea < g_spu_watchea_dyn + g_spu_watchea_dyn_len &&
                  g_spu_watchea_dyn < ea + MFC_ATOMIC_LINE) {
                  static long wn = 0, wcap = -1;
                  if (wcap < 0) { const char* e = getenv("SPU_WATCHEA_DYN_MAX"); wcap = e ? atol(e) : 64; }
                  if (wcap == 0 || (wn < wcap && ++wn)) {
                      uint32_t off = (g_spu_watchea_dyn - ea) & ~15u;
                      fprintf(stderr, "[watchea-dyn] spu%u img=%d PUTLLC line=0x%08X pc=0x%05X at+0x%02X:",
                              ctx->spu_id & 7u, ctx->image_id, ea, (uint32_t)ctx->pc & SPU_LS_MASK, off);
                      for (uint32_t k = off; k < off + 32 && k < MFC_ATOMIC_LINE; k += 4)
                          fprintf(stderr, " %02X%02X%02X%02X", ls[k], ls[k+1], ls[k+2], ls[k+3]);
                      fputc('\n', stderr); } } }
            /* A committing PUTLLC is a line write like any other, so every
             * PEER reservation on it is lost and its SPU takes SPU_EVENT_LR.
             * Silent, this is the same lost update the PPU half was added to
             * close, with an SPU on the writing side: a peer is never told to
             * re-read, so it goes on polling a line it believes it still owns,
             * and a peer parked on RdEventStat sleeps through the commit it
             * was waiting for.
             *
             * Our own reservation is dropped FIRST so the notify does not
             * raise a self-LR: hardware CONSUMES the reservation on a
             * successful PUTLLC, it does not report it lost. Already under the
             * lock-line lock, which is what spu_coh_notify_write expects. */
            ctx->resv_valid = 0;
            { static int s_w = -1;
              if (s_w < 0) s_w = getenv("SPU_PUTLLC_WHY") ? 1 : 0;
              if (s_w) { static unsigned long long n;
                  if ((++n % 100000) == 1)
                      /* ...and the first four words of the line being written.
                       * A succeeding PUTLLC that repeats millions of times is a
                       * livelock, and the only useful question is what the SPU
                       * keeps seeing there -- the count alone cannot answer it. */
                      { uint32_t w[8] = {0,0,0,0,0,0,0,0};
                        for (int _i = 0; _i < 8; _i++) {
                            uint32_t t = 0;
                            memcpy(&t, (const unsigned char*)ctx->ls + ((lsa + _i*4) & SPU_LS_MASK), 4);
                            w[_i] = __builtin_bswap32(t);
                        }
                        fprintf(stderr, "[putllc-ok] %llu: img=%d ctx=%p ea=0x%08X"
                                        " line=%08X %08X %08X %08X %08X %08X %08X %08X\n",
                                n, ctx->image_id, (void*)ctx, ea,
                                w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]); } } }
            spu_hangline_note('C', ctx->image_id, ctx->spu_id, (uint32_t)ctx->pc & SPU_LS_MASK, ea, ls);
            spu_coh_notify_write(ea);
            ctx->atomic_stat = 0;                      /* PUTLLC_SUCCESS */
        } else {
            ctx->atomic_stat = 1;                      /* PUTLLC_FAILURE -> retry */
            spu_hangline_note('F', ctx->image_id, ctx->spu_id, (uint32_t)ctx->pc & SPU_LS_MASK, ea, ls);
            /* SPU_PUTLLC_WHY=1: which of the three conditions failed.
             * A PUTLLC retry loop that never succeeds is a livelock, and
             * "failed" alone cannot distinguish a lost reservation from a line
             * another agent is rewriting -- opposite causes, opposite fixes. */
            { static int s_w = -1;
              if (s_w < 0) s_w = getenv("SPU_PUTLLC_WHY") ? 1 : 0;
              if (s_w) {
                  static unsigned long long n_noresv, n_ea, n_line;
                  if (!ctx->resv_valid)            n_noresv++;
                  else if (ctx->resv_ea != ea)     n_ea++;
                  else                             n_line++;
                  static unsigned long long total;
                  if ((++total % 500000) == 0) {
                      fprintf(stderr, "[putllc-why] %llu fails: no-reservation=%llu"
                                      " ea-mismatch=%llu line-changed=%llu"
                                      " (last ea=0x%08X img=%d resv_ea=0x%08X"
                                      " getllar=%llu)\n",
                              total, n_noresv, n_ea, n_line, ea, ctx->image_id,
                              ctx->resv_ea, (unsigned long long)ctx->dbg_getllar);
                      fflush(stderr);
                  }
              } }
        }
        { extern uint32_t g_barrier_sync_watch;
          uint32_t b = g_barrier_sync_watch;
          if (b && ea >= (b & ~127u) && ea < ((b + 0xC0 + 127) & ~127u)) {
              static int _n = 0;
              if ((_n < 96 && ++_n) && ctx->atomic_stat == 0) {
                  uint32_t sn = ((uint32_t)ctx->ls[0x1C8]<<24)|((uint32_t)ctx->ls[0x1C9]<<16)|
                                ((uint32_t)ctx->ls[0x1CA]<<8)|ctx->ls[0x1CB];
                  extern uint8_t* vm_base;
                  const uint8_t* r = vm_base + b + 0x70;   /* row 3 (sync+0x40+16*3) */
                  fprintf(stderr, "[sync-atomic] PUTLLC-OK spuNum=%u ea=+0x%X lanes@row3={%u %u %u %u %u %u %u}\n",
                          sn, ea - b,
                          (r[2]<<8)|r[3],(r[4]<<8)|r[5],(r[6]<<8)|r[7],(r[8]<<8)|r[9],
                          (r[10]<<8)|r[11],(r[12]<<8)|r[13],(r[14]<<8)|r[15]);
              }
          } }
        /* SPU_ATOM_EA: the [atom-ea] line above prints every ATTEMPT, before
         * the compare. Say which of them committed, or a retried failure reads
         * as a lost update. */
        { static int s_v = -1; static uint32_t s_vl;
          if (s_v < 0) { const char* e = getenv("SPU_ATOM_EA");
                         s_vl = e ? (uint32_t)strtoul(e, 0, 16) & ~127u : 0; s_v = s_vl ? 1 : 0; }
          if (s_v && (ea & ~127u) == s_vl)
              fprintf(stderr, "[atom-ea] PUTLLC %s lr=0x%05X ea=0x%08X\n",
                      ctx->atomic_stat ? "FAIL" : "OK",
                      ctx->gpr[0]._u32[0] & SPU_LS_MASK, ea); }
        ctx->resv_valid = 0;                           /* reservation consumed */
        spu_lockline_unlock();
        if (ctx->atomic_stat == 0) {
            spu_taskset_line_committed(ea & ~127u);
            if (g_spu_line_commit_hook)  g_spu_line_commit_hook(ea & ~127u);
            if (g_spu_line_commit_hook2) g_spu_line_commit_hook2(ea & ~127u);
        }
        return 1;

    case MFC_PUTLLUC_CMD:
    case MFC_PUTQLLUC_CMD:
        spu_lockline_lock();
        { extern void spu_lockguard_check(spu_context*, uint32_t, const uint8_t*, uint32_t, const char*);
          spu_lockguard_check(ctx, ea, ls, MFC_ATOMIC_LINE, "PUTLLUC"); }
        memcpy(mem, ls, MFC_ATOMIC_LINE);              /* unconditional store */
        spu_hangline_note('U', ctx->image_id, ctx->spu_id, (uint32_t)ctx->pc & SPU_LS_MASK, ea, ls);
        /* Unconditional, so it invalidates EVERY reservation on the line --
         * this SPU's included, which is why the notify runs before the
         * bookkeeping below rather than after it. A peer left holding a
         * reservation here would commit a PUTLLC against a snapshot this store
         * has already overwritten. */
        spu_coh_notify_write(ea);
        ctx->resv_valid = 0; ctx->atomic_stat = 2;   /* PUTLLUC complete */
        spu_lockline_unlock();
        spu_taskset_line_committed(ea & ~127u);
        if (g_spu_line_commit_hook)  g_spu_line_commit_hook(ea & ~127u);
        if (g_spu_line_commit_hook2) g_spu_line_commit_hook2(ea & ~127u);
        return 1;

    default:
        return 0;
    }
}

static int channel_is_mfc(uint32_t ch)
{
    switch (ch) {
    case MFC_WrMSSyncReq: case MFC_RdTagMask:  case MFC_LSA:
    case MFC_EAH:         case MFC_EAL:         case MFC_Size:
    case MFC_TagID:       case MFC_Cmd:         case MFC_WrTagMask:
    case MFC_WrTagUpdate: case MFC_RdTagStat:   case MFC_RdListStallStat:
    case MFC_WrListStallAck: case MFC_RdAtomicStat:
        return 1;
    default:
        return 0;
    }
}

/* ===========================================================================
 * Cooperative abort (raw SPU stop / restart, see spu_raw.c)
 *
 * A requester (the PPU thread storing SPU_RunCntl=0) cannot interrupt a host
 * thread that is running lifted code, so it publishes the target ctx here and
 * the target polls at every channel access -- the only points the lifted code
 * calls back into the runtime -- and inside its parked channel waits. Slots
 * are matched by ctx pointer, NOT spu_id: SPURS SPUs share the small id space
 * with raw SPUs. s_abort_any is only a fast-path hint (one volatile load per
 * channel op when nothing is pending); the wait loop scans the slots itself, and
 * the requester re-posts until the thread is gone, so a hint cleared in a race
 * costs at most one retry.
 * ===========================================================================*/
#define SPU_ABORT_SLOTS 8
static spu_context* volatile s_abort_ctx[SPU_ABORT_SLOTS];
static volatile int          s_abort_any;

void spu_abort_request(spu_context* ctx)
{
    if (!ctx) return;
    for (int i = 0; i < SPU_ABORT_SLOTS; i++)
        if (s_abort_ctx[i] == ctx) goto posted;
    for (int i = 0; i < SPU_ABORT_SLOTS; i++)
        if (!s_abort_ctx[i]) { s_abort_ctx[i] = ctx; goto posted; }
    fprintf(stderr, "[spu] abort table full, request for ctx %p dropped\n", (void*)ctx);
    return;
posted:
    s_abort_any = 1;
    spu_ch_wake(ctx);
}

void spu_abort_cancel(spu_context* ctx)
{
    for (int i = 0; i < SPU_ABORT_SLOTS; i++)
        if (s_abort_ctx[i] == ctx) s_abort_ctx[i] = NULL;
}

/* Consume a pending abort for `ctx` (returns 1 if there was one). */
static int spu_abort_take(spu_context* ctx)
{
    R35C(R35_ABORT_TAKE);
    int hit = 0, left = 0;
    for (int i = 0; i < SPU_ABORT_SLOTS; i++) {
        if (s_abort_ctx[i] == ctx) { s_abort_ctx[i] = NULL; hit = 1; }
        else if (s_abort_ctx[i])   left = 1;
    }
    if (!left) s_abort_any = 0;
    return hit;
}

/* Called on the SPU's own host thread; does not return if an abort is pending
 * and this thread runs under spu_run_with_halt (the longjmp target). */
static inline void spu_abort_poll(spu_context* ctx)
{
    if (s_abort_any && spu_abort_take(ctx)) {
        ctx->status = SPU_STATUS_STOPPED;
        spu_halt(ctx);
    }
}

/* ===========================================================================
 * Channel write
 * ===========================================================================*/
volatile unsigned g_lprof_wrch[64]; int g_lprof_wait_on;   /* SPURS_LANE_PROF_WAIT=1 (cellSpurs.c sampler) */
int g_r35_on; SPU_THREAD_LOCAL unsigned long long g_r35c[32];   /* SPU_R35_COUNT */
/* SPU_R35_FAST (31/10 rt35): spu_wrch fast paths for MFC_Cmd (non-atomic: straight to mfc_submit), MFC_WrTagUpdate and
 * (inline, spu_helpers.h) MFC_WrTagMask, bypassing the diagnostic checks, channel_is_mfc, spu_mfc_atomic and the second switch
 * of mfc_channel_write. Same stores as the generic path. -1 until the first spu_wrch decides; off with SPU_R35_FAST=0 or when a
 * diagnostic that hooks the generic path is armed (SPU_CHHIST, SPURS_LANE_PROF_WAIT, SPU_TAGPROBE). */
int g_r35_fast = -1;
static __attribute__((noinline)) void spu_r35_fast_init(void)
{
    const char* e = getenv("SPU_R35_FAST"); int on = !(e && e[0] == '0');   /* default decided after the A/B */
    if (getenv("SPU_CHHIST") || getenv("SPURS_LANE_PROF_WAIT") || getenv("SPU_TAGPROBE") || getenv("SPU_COLLTAB_WATCH")) on = 0;   /* integ44: COLLTAB_WATCH hooks the generic path too */
    g_r35_fast = on;
}
extern int g_ctw_on; static void spu_colltab_check(spu_context* ctx, const char* where); static void ctw_note(spu_context* ctx, uint32_t channel, uint32_t v);
void spu_wrch(spu_context* ctx, uint32_t channel, u128 value)
{
    R35C(R35_WRCH);
    if (__builtin_expect(g_r35_on, 0)) { if (channel == MFC_Cmd) g_r35c[R35_WRCH_CMD]++; else if (channel == MFC_WrTagMask) g_r35c[R35_WRCH_TAGM]++; else if (channel == MFC_WrTagUpdate) g_r35c[R35_WRCH_TAGU]++; else g_r35c[R35_WRCH_OTH]++; }
    spu_abort_poll(ctx);
    /* SPU_CHHIST=1: which channels a job actually touches, and how often. A
     * worker that never issues an MFC command is either not being handed its
     * work descriptor or is waiting on a channel we never satisfy; the channel
     * mix distinguishes those. */
    { static int s_c = -1; if (s_c < 0) { const char* e = getenv("SPU_CHHIST"); s_c = e ? (atoi(e) > 0 ? atoi(e) : 2000) : 0; }
      if (s_c) { static unsigned long long w[128]; static unsigned long long n;
          w[channel & 127]++;
          if ((++n % (unsigned long long)s_c) == 0) { fprintf(stderr, "[chw] %llu writes:%c", n, 10);
              for (int i = 0; i < 128; i++) if (w[i])
                  fprintf(stderr, "   wrch ch%-3d %llu%c", i, w[i], 10); } } }
    uint32_t v = value._u32[0];  /* channel writes use the preferred slot */
    if (__builtin_expect(g_r35_fast != 0, 1)) {
        if (__builtin_expect(g_r35_fast < 0, 0)) spu_r35_fast_init();
        if (g_r35_fast > 0) {
            if (channel == MFC_Cmd) {
                if (v != MFC_GETLLAR_CMD && v != MFC_PUTLLC_CMD && v != MFC_PUTLLUC_CMD && v != MFC_PUTQLLUC_CMD) {
                    mfc_submit(mfc_for(ctx), ctx, v);
                    return;
                }
            } else if (channel == MFC_WrTagUpdate) {
                mfc_engine* m = mfc_for(ctx);
                ctx->mfc_tag_status = m->tag_completed & ctx->mfc_tag_mask;
                if (v != 0 && (ctx->event_mask & 0x1u)) { ctx->event_status |= 0x1u; spu_ch_wake(ctx); }
                return;
            }
        }
    }
    { extern int g_lprof_wait_on; extern volatile unsigned g_lprof_wrch[64];
      if (__builtin_expect(g_lprof_wait_on, 0)) g_lprof_wrch[__readgsdword(0x48) & 63] = (channel << 24) | (v & 0xFFFFFF); }

    if (__builtin_expect(g_ctw_on != 0, 0) && ctx->image_id == 110) {   /* SPU_COLLTAB_WATCH */
        spu_colltab_check(ctx, "before a channel write");
        ctw_note(ctx, channel, v);
    }
    if (channel_is_mfc(channel)) {
        /* Atomic line ops (GETLLAR/PUTLLC/...) need real reservation semantics,
         * not the plain GET/PUT the DMA engine would do. */
        if (channel == MFC_Cmd) {
            if (__builtin_expect(g_c37_on != 0, 0)) {   /* SPU_C37 */
                c37_atomic_pre(ctx);
                const int h = spu_mfc_atomic(ctx, v);
                c37_atomic_post();
                c37_atomic(ctx, v, h);
                if (h) return;
            } else if (spu_mfc_atomic(ctx, v))
                return;
        }
        mfc_channel_write(mfc_for(ctx), ctx, channel, v);
        if (__builtin_expect(g_ctw_on != 0, 0) && ctx->image_id == 110 && channel == MFC_Cmd) spu_colltab_check(ctx, "right after an MFC command");
        /* MFC tag-status-update EVENT producer (channel-stall milestone). Our
         * DMA completes synchronously, so the instant the program arms tag
         * notification (MFC_WrTagUpdate with a non-zero mode = any/all), the
         * tags in MFC_WrTagMask are already complete -- pend the tag event
         * (bit 0, MFC_TAG_STATUS_UPDATE_EVENT) when the SPU has it enabled, and
         * wake any host thread blocked in rdch SPU_RdEventStat. This is the
         * long-missing producer for event_status (previously only ever cleared
         * via WrEventAck -> RdEventStat waits could never complete). */
        if (channel == MFC_WrTagUpdate && v != 0 && (ctx->event_mask & 0x1u)) {
            ctx->event_status |= 0x1u;
            spu_ch_wake(ctx);
        }
        return;
    }

    switch (channel) {
    case SPU_WrOutMbox:
        /* A raw SPU's outbound mailbox is one entry deep and wrch stalls while
         * it is full. Buffering instead let two replies queue up, and a PPU
         * that tests bit 0 of the count -- Tornado Outbreak's boot handshake,
         * func_000D5A10 -- read count 2 as empty and spun forever (an
         * intermittent boot hang). Raw SPUs only: their reader is the PPU
         * polling the MMIO window, whereas SPURS-thread mailboxes are drained
         * by paths that do not wake a stalled writer. A raw SPU's local store
         * is the guest window, not its own buffer. */
        if (ctx->ls != ctx->ls_store)
            while (ctx->ch_out_mbox.count >= SPU_MBOX_DEPTH && ctx->status != SPU_STATUS_STOPPED)
                Sleep(0);
        spu_channel_write(&ctx->ch_out_mbox, v);
        /* SPU_DBG_MBOX=1: depth after the write. The image-1 SPUs publish their
         * LS work buffer here (LS 0x11D4) and then poll it, and the guest sets
         * class-2 mask 0x3 -- which excludes the plain-mailbox bit 0x10 -- so
         * this write raises no interrupt and the PPU must POLL to see it. If the
         * depth climbs and stays, nobody is polling and the message is stranded.
         * That is the difference between "the PPU is slow" and "the PPU never
         * looks", which no other probe here distinguishes. */
        { static int _d = -1; if (_d < 0) _d = getenv("SPU_DBG_MBOX") ? 1 : 0;
          if (_d) { static unsigned long _n = 0; if (++_n <= 24)
            fprintf(stderr, "[spu-outmbox] spu=%X wrote 0x%08X depth=%u\n",
                    ctx->spu_id, v, (unsigned)ctx->ch_out_mbox.count); } }
        { static int s_t = -1; if (s_t < 0) s_t = getenv("SPU_MBOXTRACE") ? 1 : 0;
          if (s_t) fprintf(stderr, "[spu-mbox] OUT  grp=0x%X spu=0x%X val=0x%08X\n",
                           ctx->spu_group_id, ctx->spu_id, v); }
        /* Plain mailbox data is consumed by the following interrupt request. */
        break;
    case SPU_WrOutIntrMbox:
        { static int s_t = -1; if (s_t < 0) s_t = getenv("SPU_MBOXTRACE") ? 1 : 0;
          if (s_t) fprintf(stderr, "[spu-mbox] INTR grp=0x%X spu=0x%X val=0x%08X\n",
                           ctx->spu_group_id, ctx->spu_id, v); }
        if (g_spu_user_event_hook && g_spu_user_event_hook(ctx, v)) break;
        spu_channel_write(&ctx->ch_out_intr_mbox, v);
        if (g_spu_out_mbox_hook) g_spu_out_mbox_hook(ctx->spu_group_id, ctx->spu_id, 1, v);
        break;
    case SPU_WrDec:          ctx->decrementer = v;
                             ctx->dec_base_ns = spu_host_ns();              break;
    case SPU_WrEventMask:    ctx->event_mask = v;                           break; /* WrEventMask */
    case SPU_WrEventAck:
        /* Under the lock-line lock, because every producer of the LR bit sets
         * it under that lock: the PPU coherent store, and a peer SPU's PUTLLC,
         * PUTLLUC or plain PUT. A bare read-modify-write here can read
         * event_status, have a concurrent |= SPU_EVENT_LR land in between, and
         * write the stale value back. The edge is then gone, and an SPU that
         * has just acknowledged the events it read goes back to sleep believing
         * it still holds a reservation it has already lost, which is the parked
         * SPURS kernel this whole mechanism exists to wake. */
        spu_lockline_lock();
        ctx->event_status &= ~v;
        spu_lockline_unlock();
        break;
    case SPU_WrSRR0:         ctx->srr0 = v;                                 break;
    default:
        /* Unknown / unhandled channel write -- ignore (matches a no-op SPU). */
        break;
    }
}

/* ===========================================================================
 * Channel-stall contract (faithful-adopt, from canersaka's fork).
 *
 * A blocking spu_rdch on an empty read channel parks the SPU host thread on a
 * per-SPU CV until a producer (mailbox/signal write, event raise) calls
 * spu_ch_wake -- never fabricating a value. Under SPU_LOCKSTEP the wait
 * releases the run token first (block_begin) so a peer SPU that must post the
 * awaited data is never blocked on this ctx. A 10 ms re-poll is the missed-wake
 * safety net. Opt-in (env SPU_CH_BLOCK=1), default OFF while the producers
 * (mailbox/signal writes, MFC tag-status event raise) are being wired -- until
 * then blocking a read whose producer is missing would hang, so default stays
 * legacy non-blocking with zero regression, exactly like the lockstep gate.
 * ===========================================================================*/
/* Set by runtime/spu/spu_raw.c once a raw SPU exists. A raw SPU runs on its own
 * host thread against a PPU that pokes its mailboxes by MMIO, so an empty
 * RdInMbox must PARK, not fabricate a zero -- there is no scheduler above it to
 * retry. Off unless a raw SPU is actually created, so no other port changes. */
int g_spu_force_ch_block = 0;

static int yz_ch_block(void)
{
    static int v = -1;
    if (v < 0) v = getenv("SPU_CH_BLOCK") ? 1 : 0;
    return v || g_spu_force_ch_block;
}

/* Would rdch complete right now? Plain reads + the 10 ms re-poll cover cross-
 * thread visibility (x86 TSO + the wait syscall's barrier); the s43 atomics are
 * a later refinement. */
/* MFC_LLR_LOST_EVENT (0x400) -- the only producer for it.
 *
 * The SPU "wait until another processor touches this variable" idiom is: GETLLAR
 * the 128-byte line, set the event mask to Lr, then block in
 * `rdch SPU_RdEventStat` until the reservation is lost. Nothing ever raised that
 * bit, so an SPU using it waited forever -- ps1_netemu's GPU core (SPU 4) parks
 * at pc=0x0A5E8 with evmask=0x400 exactly here while the PPU spins waiting on the
 * SPU. Deadlock, and the reason the PS1 core never starts.
 *
 * Losing a reservation means the reserved line changed, and GETLLAR already
 * snapshots it, so compare against that snapshot rather than hooking every store.
 * ponytail: polled, not store-hooked -- one 128-byte memcmp per 10 ms poll on an
 * already-blocked SPU, and nothing at all on the PPU store path. The ceiling is
 * one poll of wake latency, and an A-B-A write that restores the same bytes
 * between polls is missed; hook the writers if either ever matters.
 */

/* SPU_HANG_LINE=<hex EA>: the last 256 events on that lock line -- GETLLAR,
 * PUTLLC ok/fail, PUTLLUC, PPU/DMA notifies -- with who, pc and the line's
 * first words, dumped the first time an SPU has waited 2 s on RdEventStat.
 * The intermittent freeze (an SPU parked on SPU_EVENT_LR while its line never
 * changes again) needs the sequence that led there, not the end state. */
typedef struct { unsigned long long t; uint32_t pc, ea, w[4]; int img; char kind; uint8_t spu; } spu_hl_ev;
static spu_hl_ev s_hl[256]; static volatile long s_hl_n; static uint32_t s_hl_line = 1;
uint32_t spu_hangline(void)
{
    if (s_hl_line == 1) { const char* e = getenv("SPU_HANG_LINE");
                          s_hl_line = e ? (uint32_t)strtoul(e, 0, 16) & ~127u : 0; }
    return s_hl_line;
}
void spu_hangline_note(char kind, int img, uint32_t spu, uint32_t pc, uint32_t ea, const uint8_t* line)
{
    if (!spu_hangline() || (ea & ~127u) != s_hl_line) return;
    long i = InterlockedIncrement(&s_hl_n) - 1;
    spu_hl_ev* e = &s_hl[i & 255];
    e->t = GetTickCount64(); e->kind = kind; e->img = img; e->spu = (uint8_t)spu; e->pc = pc; e->ea = ea;
    for (int k = 0; k < 4; k++) {
        const uint8_t* b = line ? line + 4 * k : NULL;
        e->w[k] = b ? ((uint32_t)b[0] << 24 | (uint32_t)b[1] << 16 | (uint32_t)b[2] << 8 | b[3]) : 0;
    }
}
static void spu_hangline_dump(const spu_context* ctx)
{
    static volatile long done;
    if (!spu_hangline() || InterlockedExchange(&done, 1)) return;
    const long n = s_hl_n;
    const uint8_t* m = vm_base + s_hl_line;
    fprintf(stderr, "[hang-line] spu=%X waits; line 0x%08X now:", ctx->spu_id, s_hl_line);
    for (int k = 0; k < 32; k++) fprintf(stderr, "%s%02X", (k & 3) ? "" : " ", m[k]);
    fprintf(stderr, "; snapshot:");
    for (int k = 0; k < 32; k++) fprintf(stderr, "%s%02X", (k & 3) ? "" : " ", ctx->resv_line[k]);
    fprintf(stderr, "; %ld events\n", n);
    for (long i = n > 256 ? n - 256 : 0; i < n; i++) {
        const spu_hl_ev* e = &s_hl[i & 255];
        fprintf(stderr, "[hang-line] #%ld t=%llu %c img=%d spu=%u pc=0x%05X ea=0x%08X %08X %08X %08X %08X\n",
                i, e->t, e->kind, e->img, e->spu, e->pc, e->ea, e->w[0], e->w[1], e->w[2], e->w[3]);
    }
    fflush(stderr);
}

static void spu_resv_lost_poll(spu_context* ctx)
{
    if (!(ctx->event_mask & 0x400u) || !ctx->resv_valid) return;
    if (!vm_base || ctx->resv_ea == 0) return;
    if (memcmp(vm_base + ctx->resv_ea, ctx->resv_line, 128) != 0) {
        ctx->resv_valid = 0;
        ctx->event_status |= 0x400u;
        spu_hangline_note('L', ctx->image_id, ctx->spu_id, (uint32_t)ctx->pc & SPU_LS_MASK, ctx->resv_ea, vm_base + ctx->resv_ea);
    }
}

static int spu_ch_ready(spu_context* ctx, uint32_t channel)
{
    if (channel == SPU_RdEventStat) spu_resv_lost_poll(ctx);
    switch (channel) {
    case SPU_RdInMbox:      return ctx->rcv_evt_n != 0 || ctx->ch_in_mbox.count != 0;
    case SPU_RdSigNotify1:  return ctx->ch_sig_notify[0].count != 0;
    case SPU_RdSigNotify2:  return ctx->ch_sig_notify[1].count != 0;
    case SPU_RdEventStat:   return (ctx->event_status & ctx->event_mask) != 0;
    default:                return 1;   /* non-blocking channels: always ready */
    }
}

/* Signal the per-SPU wait CV so a blocked spu_rdch re-checks its predicate.
 * MUST be called by every producer AFTER it makes a read predicate true.
 *
 * Lost-wakeup protocol (Dekker style): a waiter increments ch_waiters (seq_cst)
 * BEFORE re-testing its predicate under ch_wait_lock; the producer publishes the
 * predicate state, issues a seq_cst fence, and reads ch_waiters. If a waiter is
 * registered, the producer takes and releases ch_wait_lock before waking: the
 * waiter holds that lock from its predicate test until SleepConditionVariableSRW
 * atomically releases it, so the producer either runs before the test (waiter
 * sees the state) or after the waiter is asleep (the Wake reaches it). With no
 * waiter registered the hot path stays lock-free. */
void spu_ch_wake(spu_context* ctx)
{
    if (!ctx) return;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    if (__atomic_load_n(&ctx->ch_waiters, __ATOMIC_SEQ_CST) == 0) return;
    AcquireSRWLockExclusive((SRWLOCK*)&ctx->ch_wait_lock);
    ReleaseSRWLockExclusive((SRWLOCK*)&ctx->ch_wait_lock);
    WakeAllConditionVariable((CONDITION_VARIABLE*)&ctx->ch_wait_cv);
}

/* SPU_CHWAIT_TIMEOUT_US: safety-net timeout of the LR (evmask 0x400) wait, default 1000.
 * Other channels keep 10 ms. SPU_CHWAIT_STATS=1: per-spu_id timeout/wake counts,
 * duration histogram, [chwait] summary every 5 s. */
static unsigned ch_wait_timeout_ms(uint32_t channel, uint32_t evmask)
{
    static int s_us = -1;
    if (s_us < 0) { const char* e = getenv("SPU_CHWAIT_TIMEOUT_US"); s_us = e ? atoi(e) : 1000; if (s_us < 1) s_us = 1; }
    if (channel == SPU_RdEventStat && (evmask & 0x400u)) { unsigned ms = (unsigned)((s_us + 999) / 1000); return ms ? ms : 1; }
    return 10;
}

#define CHW_NB 6   /* <50us <500us <2ms <5ms <9ms >=9ms */
static struct { volatile long to, wk, missed; volatile long h[CHW_NB]; } s_chw[64];
static volatile long long s_chw_next;
static void chw_missed(spu_context* ctx)   /* woke by TIMEOUT but the predicate was true: a producer did not notify */
{
    static int s_on2 = -1;
    if (s_on2 < 0) { const char* e = getenv("SPU_CHWAIT_STATS"); s_on2 = (e && e[0] == '1'); }
    if (s_on2) __atomic_fetch_add(&s_chw[(unsigned)ctx->spu_id & 63].missed, 1, __ATOMIC_RELAXED);
}
static void chw_account(spu_context* ctx, int timed_out, double us)
{
    static int s_on = -1;
    if (s_on < 0) { const char* e = getenv("SPU_CHWAIT_STATS"); s_on = (e && e[0] == '1'); }
    if (!s_on) return;
    unsigned id = (unsigned)ctx->spu_id & 63;
    if (timed_out) __atomic_fetch_add(&s_chw[id].to, 1, __ATOMIC_RELAXED);
    else           __atomic_fetch_add(&s_chw[id].wk, 1, __ATOMIC_RELAXED);
    int b = us < 50 ? 0 : us < 500 ? 1 : us < 2000 ? 2 : us < 5000 ? 3 : us < 9000 ? 4 : 5;
    __atomic_fetch_add(&s_chw[id].h[b], 1, __ATOMIC_RELAXED);
    long long now = (long long)GetTickCount64(), nx = s_chw_next;
    if (nx == 0) { __atomic_compare_exchange_n(&s_chw_next, &nx, now + 5000, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED); return; }
    if (now >= nx && __atomic_compare_exchange_n(&s_chw_next, &nx, now + 5000, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
        for (unsigned i = 0; i < 64; i++) {
            long t = s_chw[i].to, w = s_chw[i].wk;
            if (!t && !w) continue;
            fprintf(stderr, "[chwait] spu=%u timeout=%ld wake=%ld missed=%ld  hist(<50us,<500us,<2ms,<5ms,<9ms,>=9ms)=%ld/%ld/%ld/%ld/%ld/%ld\n",
                    i, t, w, s_chw[i].missed, s_chw[i].h[0], s_chw[i].h[1], s_chw[i].h[2], s_chw[i].h[3], s_chw[i].h[4], s_chw[i].h[5]);
        }
        fflush(stderr);
    }
}

/* Block the calling SPU host thread until `channel` is readable. */
static void spu_ch_wait(spu_context* ctx, uint32_t channel, const char* op)
{
    if (!yz_ch_block() || spu_ch_ready(ctx, channel)) return;
    LARGE_INTEGER c37q0; c37q0.QuadPart = 0;
    if (__builtin_expect(g_c37_on != 0, 0)) { c37_wait(ctx, channel, 0, 0.0); QueryPerformanceCounter(&c37q0); }

    { static unsigned long bn = 0; unsigned long n = ++bn;
      static int s_cb = -1; if (s_cb < 0) { const char* e = getenv("SPU_CHBLOCK_LOG"); s_cb = e && e[0] == '1'; }   /* SPU_CHBLOCK_LOG=1: the old every-512th line */
      if (s_cb ? (n <= 50 || (n % 512) == 0) : n <= 8)
        fprintf(stderr, "[ch-block] spu=%X pc=0x%05X op=%s ch=%u evstat=0x%X evmask=0x%X\n",
                ctx->spu_id, ctx->pc & SPU_LS_MASK, op, channel,
                ctx->event_status, ctx->event_mask); }

    /* SPU_CHWAIT_SPIN_US=<us> (default 0 = off): poll the channel for up to <us> before parking on the
     * condition variable. The raw collision SPUs (igCollBroadSpu / igcollspu) hand work to each other and to the
     * PPU main ~1300 times per 15 ms; each hand-off was a CV sleep + wake (~10 us on Windows), so the chain is
     * latency-bound (60-70% of their samples sit in SleepConditionVariableSRW at ~45% CPU). */
    {
        static int s_spin_us = -1;
        if (s_spin_us < 0) { const char* e = getenv("SPU_CHWAIT_SPIN_US"); s_spin_us = e ? atoi(e) : 0; if (s_spin_us < 0) s_spin_us = 0; }
        if (s_spin_us > 0) {
            LARGE_INTEGER q0, q, fq; QueryPerformanceCounter(&q0); QueryPerformanceFrequency(&fq);
            const long long lim = (long long)s_spin_us * fq.QuadPart / 1000000;
            for (;;) {
                for (int i = 0; i < 16; i++) _mm_pause();
                if (spu_ch_ready(ctx, channel)) {
                    if (g_c37_on > 0) { LARGE_INTEGER c37q1, c37f; QueryPerformanceCounter(&c37q1); QueryPerformanceFrequency(&c37f);
                                        c37_wait(ctx, channel, 1, (double)(c37q1.QuadPart - c37q0.QuadPart) * 1e6 / (double)c37f.QuadPart); }
                    return;
                }
                QueryPerformanceCounter(&q);
                if (q.QuadPart - q0.QuadPart >= lim) break;
            }
        }
    }

    ctx->status = SPU_STATUS_WAITING_CHANNEL;
    yz_lockstep_block_begin(ctx);          /* release the run token before the OS wait */
    {
        unsigned long long start = GetTickCount64(), next_hb = 2000;
        int last_to = 0;
        while (!spu_ch_ready(ctx, channel)) {
            spu_abort_poll(ctx);           /* raw SPU stopped by the PPU while parked */
            __atomic_fetch_add(&ctx->ch_waiters, 1, __ATOMIC_SEQ_CST);   /* register BEFORE the predicate test */
            AcquireSRWLockExclusive((SRWLOCK*)&ctx->ch_wait_lock);
            if (!spu_ch_ready(ctx, channel)) {
                LARGE_INTEGER q0, q1, fq; QueryPerformanceCounter(&q0);
                BOOL ok = SleepConditionVariableSRW((CONDITION_VARIABLE*)&ctx->ch_wait_cv,
                                          (SRWLOCK*)&ctx->ch_wait_lock,
                                          ch_wait_timeout_ms(channel, ctx->event_mask), 0);
                QueryPerformanceCounter(&q1); QueryPerformanceFrequency(&fq);
                last_to = !ok;
                chw_account(ctx, !ok, (double)(q1.QuadPart - q0.QuadPart) * 1e6 / (double)fq.QuadPart);
            }
            ReleaseSRWLockExclusive((SRWLOCK*)&ctx->ch_wait_lock);
            __atomic_fetch_sub(&ctx->ch_waiters, 1, __ATOMIC_SEQ_CST);
            unsigned long long waited = GetTickCount64() - start;
            if (waited >= next_hb) {
                fprintf(stderr, "[ch-wait] spu=%X pc=0x%05X ch=%u waited=%llums evstat=0x%X evmask=0x%X resv[valid=%d ea=0x%08X]\n",
                        ctx->spu_id, ctx->pc & SPU_LS_MASK, channel, waited,
                        ctx->event_status, ctx->event_mask,
                        ctx->resv_valid, ctx->resv_ea);
                fflush(stderr);
                if (channel == SPU_RdEventStat) spu_hangline_dump(ctx);
                next_hb = ((waited / 2000) + 1) * 2000;
            }
        }
        if (last_to) chw_missed(ctx);
        if (g_c37_on > 0) { LARGE_INTEGER c37q1, c37f; QueryPerformanceCounter(&c37q1); QueryPerformanceFrequency(&c37f);
                            c37_wait(ctx, channel, last_to ? 2 : 1, (double)(c37q1.QuadPart - c37q0.QuadPart) * 1e6 / (double)c37f.QuadPart); }
    }
    yz_lockstep_block_end(ctx);            /* rejoin the rotation, reacquire the token */
    ctx->status = SPU_STATUS_RUNNING;
}

/* ===========================================================================
 * Channel read (returns value in the preferred word slot)
 * ===========================================================================*/
u128 spu_rdch(spu_context* ctx, uint32_t channel)
{
    R35C(R35_RDCH);
    spu_abort_poll(ctx);
    { static int s_c = -1; if (s_c < 0) { const char* e = getenv("SPU_CHHIST"); s_c = e ? (atoi(e) > 0 ? atoi(e) : 2000) : 0; }
      if (s_c) { static unsigned long long r[128]; static unsigned long long n;
          r[channel & 127]++;
          if ((++n % (unsigned long long)s_c) == 0) { fprintf(stderr, "[chr] %llu reads:%c", n, 10);
              for (int i = 0; i < 128; i++) if (r[i])
                  fprintf(stderr, "   rdch ch%-3d %llu%c", i, r[i], 10); } } }
    /* Block (never fabricate) on an empty producer-fed read channel (opt-in
     * SPU_CH_BLOCK). RdEventStat now has a producer (the MFC tag-status event
     * raise above), but only block it when the SPU has actually enabled events
     * (event_mask != 0) -- a masked-off read must return 0 immediately, not
     * park forever. RdInMbox/RdSigNotify park on their PPU producers. */
    if (channel == SPU_RdInMbox || channel == SPU_RdSigNotify1 || channel == SPU_RdSigNotify2)
        spu_ch_wait(ctx, channel, "rdch");
    else if (channel == SPU_RdEventStat && ctx->event_mask != 0)
        spu_ch_wait(ctx, channel, "rdch");

    uint32_t v = 0;

    { static int s_t = -1; if (s_t < 0) s_t = getenv("SPU_POLLTRACE") ? 1 : 0;
      if (s_t) { static uint64_t s_c[10] = {0}; static uint64_t s_tot = 0;
        int b = (channel==SPU_RdInMbox)?0:(channel==SPU_RdSigNotify1)?1:(channel==SPU_RdSigNotify2)?2:
                (channel==SPU_RdDec)?3:(channel==SPU_RdEventStat)?4:(channel==SPU_RdEventMask)?5:
                (channel==MFC_RdTagStat)?6:(channel==MFC_RdAtomicStat)?7:(channel==SPU_RdMachStat)?8:9;
        s_c[b]++;
        if ((++s_tot % 2000000) == 0)
          fprintf(stderr, "[rdch] InMbox=%llu Sig1=%llu Sig2=%llu Dec=%llu EvStat=%llu EvMask=%llu TagStat=%llu AtomStat=%llu MachStat=%llu other=%llu\n",
            (unsigned long long)s_c[0],(unsigned long long)s_c[1],(unsigned long long)s_c[2],(unsigned long long)s_c[3],
            (unsigned long long)s_c[4],(unsigned long long)s_c[5],(unsigned long long)s_c[6],(unsigned long long)s_c[7],
            (unsigned long long)s_c[8],(unsigned long long)s_c[9]); } }

    if (channel_is_mfc(channel)) {
        v = mfc_channel_read(mfc_for(ctx), ctx, channel);
        return spu_make_preferred_u32(v);
    }

    switch (channel) {
    case SPU_RdInMbox:
        if (ctx->rcv_evt_n > 0) { v = ctx->rcv_evt[ctx->rcv_evt_i++]; ctx->rcv_evt_n--; }
        else v = spu_channel_read(&ctx->ch_in_mbox);
        break;
    case SPU_RdSigNotify1:  v = spu_channel_read(&ctx->ch_sig_notify[0]); break;
    case SPU_RdSigNotify2:  v = spu_channel_read(&ctx->ch_sig_notify[1]); break;
    /* The decrementer must actually DECREMENT. Returning the latched WrDec
     * value verbatim makes every `rdch $ch8` deadline poll spin forever: the
     * elapsed-time term is always zero, so the deadline never passes. LBP's
     * wwsjob SPURS policy module hangs on exactly that at LS 0x2D68:
     *
     *     2d6c:  lqa   $2, 0x1530     ; deadline
     *     2d74:  rdch  $5, $ch8       ; now  <- always read 0
     *     2d78:  sf    $6, $2, $5     ; now - deadline
     *     2d80:  cgti  $5, $6, 0
     *     2d88:  binz  $2, $0         ; leave once it goes positive
     *
     * It never leaves, never reaches its first DMA, and blocks the workload.
     * Wraparound on subtract is deliberate -- it is what hardware does, and the
     * `cgti > 0` test above is written to tolerate it. */
    case SPU_RdDec: {
        /* Never armed by a WrDec: stamp the base on first read rather than
         * measuring from epoch 0, which would subtract the host's entire
         * uptime and hand back a wild value to code that has every right to
         * read the decrementer before writing it. */
        if (!ctx->dec_base_ns) ctx->dec_base_ns = spu_host_ns();
        uint64_t elapsed = spu_host_ns() - ctx->dec_base_ns;
        /* Split seconds from the remainder before scaling: elapsed_ns * 79.8M
         * overflows 64 bits 231 s after the base stamp, and a context that
         * never writes its decrementer (the SPURS kernel's) keeps the stamp
         * from boot, so every timed loop on that SPU saw the counter jump
         * forward by ~16 s of ticks at 231 s, 462 s, 693 s... */
        const uint64_t secs = elapsed / 1000000000ull, rem = elapsed % 1000000000ull;
        uint32_t ticks   = (uint32_t)(secs * SPU_DECREMENTER_HZ +
                                      (rem * SPU_DECREMENTER_HZ) / 1000000000ull);
        v = ctx->decrementer - ticks;
        break;
    }
    case SPU_RdEventMask:   v = ctx->event_mask;                        break;
    case SPU_RdEventStat:   spu_resv_lost_poll(ctx); v = ctx->event_status; break;
    /* CBEA 9.8: numeric bit 0 is interrupt-enable, independent of run status.
     * This runtime models nonisolated SPUs, so the isolation bit stays clear. */
    case SPU_RdMachStat:    v = ctx->int_enable & 1u; break;
    case SPU_RdSRR0:        v = ctx->srr0;                              break;
    default:
        v = 0;
        break;
    }
    return spu_make_preferred_u32(v);
}

/* ===========================================================================
 * Channel count (rchcnt) -- how many entries can be read/written right now
 * ===========================================================================*/
/* hitch116 02/11: SPU_RAW_STOP_FAST (default on, =0 off). A raw SPU that polls channel counts (rchcnt) in a loop never reached
 * a wrch/rdch, so a PPU RunCntl=0 abort request was not seen until the polling ended; rchcnt now honours the abort too, and
 * raw_spu_stop (spu_raw.c) posts the request BEFORE waiting instead of after a 20 ms timeout. */
int spu_raw_stop_fast(void)
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("SPU_RAW_STOP_FAST"); v = (e && e[0] == '0') ? 0 : 1; }
    return v;
}

uint32_t spu_rchcnt(spu_context* ctx, uint32_t channel)
{
    if (__builtin_expect(s_abort_any, 0) && spu_raw_stop_fast()) spu_abort_poll(ctx);
    /* SPU_WHOPOLLS=<n>: rchcnt/park accounting PER SPU, printed every n calls
     * with the pc each SPU is sitting at.
     *
     * SPU_CHHIST sums every SPU into one histogram, which is exactly the wrong
     * shape for a deadlock: it showed rchcnt ch0/ch4 climbing past 1.7 billion
     * while MFC_WrTagUpdate (ch23) stayed frozen, i.e. somebody polls forever
     * and somebody else stopped doing DMA -- without saying whether that is one
     * SPU or two. The freeze under investigation is the PPU-side spin in
     * func_000D2298 waiting for spu4 to advance a counter at 0x002DF008, so
     * which SPU is parked, and on what, is the whole question. */
    { static int s_w = -1;
      if (s_w < 0) { const char* e = getenv("SPU_WHOPOLLS");
        s_w = e ? (atoi(e) > 0 ? atoi(e) : 4000000) : 0; }
      if (s_w) { static unsigned long long c[8][40]; static unsigned long long n;
          static uint32_t lastpc[8], lastst[8], lastmask[8];
          static uint32_t lastsig[8][2], lastmb[8], lastrea[8];
          static int lastrv[8], lastrd[8];
          static uint32_t lastline[8][8], lastls[8][8];
          static uint32_t lastlsa[8], lastr90[8];
          const unsigned sp = (unsigned)(ctx->spu_id & 7u);
          if (channel < 40u) c[sp][channel]++;
          lastpc[sp] = (uint32_t)ctx->pc & SPU_LS_MASK;
          lastst[sp] = ctx->event_status; lastmask[sp] = ctx->event_mask;
          lastsig[sp][0] = ctx->ch_sig_notify[0].count;
          lastsig[sp][1] = ctx->ch_sig_notify[1].count;
          lastmb[sp] = ctx->ch_in_mbox.count;
          lastrv[sp] = ctx->resv_valid; lastrea[sp] = ctx->resv_ea;
          lastlsa[sp] = ctx->mfc_lsa & SPU_LS_MASK;
          lastr90[sp] = ctx->gpr[90]._u32[0];
          lastrd[sp] = (ctx->resv_valid && vm_base && ctx->resv_ea)
                     ? (memcmp(vm_base + ctx->resv_ea, ctx->resv_line, 128) != 0)
                     : -1;
          { static int s_ld2 = -2; static uint32_t s_lo2;
            if (s_ld2 == -2) { const char* e5 = getenv("SPU_LSDUMP");
                               s_ld2 = e5 ? 1 : 0;
                               s_lo2 = e5 ? (uint32_t)strtoul(e5,0,16) : 0u; }
            if (s_ld2 && ctx->ls)
                for (int z = 0; z < 8; z++) {
                    const uint8_t* l8 = ctx->ls + ((s_lo2 + z * 4) & SPU_LS_MASK);
                    lastls[sp][z] = ((uint32_t)l8[0] << 24) | ((uint32_t)l8[1] << 16)
                                  | ((uint32_t)l8[2] << 8) | l8[3];
                } }
          for (int z = 0; z < 8; z++) {
              const uint8_t* b8 = ctx->resv_line + z * 4;
              lastline[sp][z] = ((uint32_t)b8[0] << 24) | ((uint32_t)b8[1] << 16)
                              | ((uint32_t)b8[2] << 8) | b8[3];
          }
          if ((++n % (unsigned long long)s_w) == 0) {
              fprintf(stderr, "[whopolls] %llu rchcnt calls\n", n);
              for (unsigned q = 0; q < 8; q++) {
                  int any = 0;
                  for (unsigned k = 0; k < 40; k++) if (c[q][k]) any = 1;
                  if (!any) continue;
                  /* The channel state each poll is actually testing. Counts
                   * alone cannot say whether a poll returns 0 or 1, and that is
                   * the difference between "the SPU is not being told" and "the
                   * SPU is told and ignores it". */
                  fprintf(stderr, "   spu%u pc=0x%05X ev[st=%08X mask=%08X]"
                                  " sig[%u %u] inmbox=%u", q, lastpc[q],
                          lastst[q], lastmask[q], lastsig[q][0], lastsig[q][1],
                          lastmb[q]);
                  /* The reservation is the other half of an evmask=0x400 wait:
                   * spu_resv_lost_poll returns immediately unless resv_valid,
                   * so a cleared reservation is indistinguishable from "nobody
                   * wrote the line" in event_status alone -- and they are
                   * different bugs. diff says whether the line has actually
                   * changed under the snapshot right now. */
                  /* mfc_lsa and r90 alongside the reservation. GETLLAR
                   * copies the line to &ls[mfc_lsa] and to resv_line from the
                   * same source, so a snapshot that disagrees with the local
                   * store can only mean mfc_lsa was not what the SPU asked for
                   * -- and the SPU computes both that LSA and the address its
                   * compare reads from r90. Printing the two together is what
                   * decides it. */
                  fprintf(stderr, " resv[v=%d ea=0x%08X diff=%d]"
                                  " mfc_lsa=0x%05X r90=0x%08X",
                          lastrv[q], lastrea[q], lastrd[q],
                          lastlsa[q], lastr90[q]);
                  /* And the line itself. "The reservation is intact and the
                   * line has not changed" still does not say what the SPU is
                   * waiting FOR; the words do. Snapshot side, so it is exactly
                   * what the SPU last read. */
                  /* SPU_LSDUMP=<hex LS offset>: 8 words of this SPU's local
                   * store. The reserved line says what the SPU can SEE; its LS
                   * says what it BELIEVES. For the spu4 deadlock those are the
                   * two numbers to compare -- its consumed counter is staged at
                   * LS 0x10888 (the counter PUT's lsa 0x10880, +8). */
                  { static int s_ld = -2; static uint32_t s_lo;
                    if (s_ld == -2) { const char* e4 = getenv("SPU_LSDUMP");
                                      s_ld = e4 ? 1 : 0;
                                      s_lo = e4 ? (uint32_t)strtoul(e4,0,16) : 0u; }
                    if (s_ld) {
                        fprintf(stderr, " ls[0x%05X:", s_lo);
                        for (int z = 0; z < 8; z++)
                            fprintf(stderr, " %08X", lastls[q][z]);
                        fprintf(stderr, "]");
                    } }
                  if (lastrv[q]) {
                      fprintf(stderr, " line[");
                      for (int z = 0; z < 8; z++)
                          fprintf(stderr, "%s%08X", z ? " " : "", lastline[q][z]);
                      fprintf(stderr, "]");
                  }
                  for (unsigned k = 0; k < 40; k++) if (c[q][k])
                      fprintf(stderr, " ch%u=%llu", k, c[q][k]);
                  fprintf(stderr, "\n");
              }
              fflush(stderr);
          } } }
    /* SPU_CHHIST also covers rchcnt. It used to instrument only wrch/rdch,
     * which is the one place a parked persistent worker is guaranteed NOT to
     * appear: park_on_empty_inmbox halts from inside THIS function, so a
     * worker that polls rchcnt and parks produced a completely empty channel
     * histogram and read as 'never touches a channel'. */
    { static int s_c = -1;
      if (s_c < 0) { const char* e = getenv("SPU_CHHIST");
        s_c = e ? (atoi(e) > 0 ? atoi(e) : 2000) : 0; }
      if (s_c) { static unsigned long long c[128]; static unsigned long long n;
          c[channel & 127]++;
          if ((++n % (unsigned long long)s_c) == 0) {
              fprintf(stderr, "[chc] %llu rchcnt:%c", n, 10);
              for (int i = 0; i < 128; i++) if (c[i])
                  fprintf(stderr, "   rchcnt ch%-3d %llu%c", i, c[i], 10);
              fflush(stderr); } } }
    { static int s_t = -1; if (s_t < 0) s_t = getenv("SPU_POLLTRACE") ? 1 : 0;
      if (s_t) { static uint64_t s_cnt[8] = {0}; static uint64_t s_total = 0;
        int b = (channel==SPU_RdInMbox)?0:(channel==SPU_RdEventStat)?1:(channel==SPU_RdSigNotify1)?2:
                (channel==SPU_RdSigNotify2)?3:(channel==MFC_RdTagStat)?4:(channel==SPU_WrOutMbox)?5:
                (channel==SPU_WrOutIntrMbox)?6:7;
        s_cnt[b]++;
        if ((++s_total % 2000000) == 0)
          fprintf(stderr, "[pollcnt] InMbox=%llu EvStat=%llu Sig1=%llu Sig2=%llu TagStat=%llu OutMbox=%llu OutIntr=%llu other=%llu (ch last=%u)\n",
                  (unsigned long long)s_cnt[0],(unsigned long long)s_cnt[1],(unsigned long long)s_cnt[2],
                  (unsigned long long)s_cnt[3],(unsigned long long)s_cnt[4],(unsigned long long)s_cnt[5],
                  (unsigned long long)s_cnt[6],(unsigned long long)s_cnt[7], channel); } }
    switch (channel) {
    case SPU_RdInMbox:
        /* Synchronous persistent-worker park: after its handshake the SPU polls
         * the inbound mailbox for PPU commands; if empty and parking is armed,
         * halt (longjmp out of spu_run_with_halt) instead of spinning forever. */
        /* A pending sys_spu_thread_receive_event reply counts as readable
         * mailbox words: the worker checks rchcnt to decide whether its reply
         * has arrived, and parking on an "empty" inbox that actually holds the
         * reply strands it. */
        if (ctx->park_on_empty_inmbox && ctx->rcv_evt_n == 0 &&
            ctx->ch_in_mbox.count == 0) {
            extern void spu_halt(spu_context*);
            spu_halt(ctx);
        }
        return (uint32_t)ctx->rcv_evt_n + ctx->ch_in_mbox.count;           /* readable */
    /* Free slots, CLAMPED. The queue behind each channel is SPU_CHANNEL_CAP
     * deep -- deeper than the hardware depth reported here -- so `count` can
     * exceed the depth and a bare subtraction underflows, telling the SPU it
     * has four billion free slots. That silently converts a bounded mailbox
     * into an unbounded one for any title that polls before writing, which is
     * every SPU-heavy one. */
    case SPU_WrOutMbox:
        return ctx->ch_out_mbox.count >= SPU_MBOX_DEPTH
             ? 0u : (uint32_t)(SPU_MBOX_DEPTH - ctx->ch_out_mbox.count);
    case SPU_WrOutIntrMbox:
        return ctx->ch_out_intr_mbox.count >= SPU_INTR_MBOX_DEPTH
             ? 0u : (uint32_t)(SPU_INTR_MBOX_DEPTH - ctx->ch_out_intr_mbox.count);
    case SPU_RdSigNotify1:   return ctx->ch_sig_notify[0].count;
    case SPU_RdSigNotify2:   return ctx->ch_sig_notify[1].count;
    case MFC_Cmd:            return MFC_QUEUE_DEPTH - mfc_for(ctx)->queue_count;
    /* An enabled event pending, or 0 -- NOT the `default: 1` this used to fall
     * through to. The SPU idiom is `rchcnt SPU_RdEventStat; brnz -> rdch`: it
     * asks whether an event is pending and only commits to the BLOCKING read if
     * the answer is yes. Answering 1 unconditionally lured it into a read that
     * then parked forever, because rdch correctly blocks while
     * (event_status & event_mask) == 0 -- rchcnt and rdch disagreed.
     *
     * ps1_netemu's audio SPU does exactly this at LS 0x8934 and parked at
     * 0x0A5E8 waiting for MFC_LLR_LOST_EVENT on a line nothing was ever going
     * to touch; on hardware it would simply have fallen through to 0x893C and
     * kept working. Same condition as spu_ch_ready's case, including the
     * lost-reservation poll, so the two now agree by construction. */
    case SPU_RdEventStat:
        spu_resv_lost_poll(ctx);
        return (ctx->event_status & ctx->event_mask) != 0;
    case MFC_RdTagStat:      return 1;  /* synchronous: status always ready */
    /* Stall-and-notify status is a single-value channel: count is 1 while a
     * newly-stalled tag is pending to be read, else 0. The wwsjob interrupt
     * handler gates on this (rchcnt $ch25; brz IhcExit) -- returning the old
     * default of 1 made it enter on spurious interrupts with an empty stat,
     * running IhcLoop on a zero mask (clz(0) -> bogus tagId/ack). */
    case MFC_RdListStallStat: return ctx->list_stall_stat ? 1 : 0;
    default:                 return 1;  /* default: channel ready */
    }
}

/* ===========================================================================
 * Indirect-branch dispatch + function registry
 * ===========================================================================*/
typedef void (*spu_fn)(spu_context*);

typedef struct {
    uint32_t addr;
    spu_fn   fn;
    int      image_id;   /* which recompiled image this function belongs to */
} spu_reg_entry;

/* One entry per lifted SPU function across EVERY registered image, and a game
 * that lifts its whole SPU workload set has a lot of them: Yakuza: Dead Souls
 * registers ~170k (cri_audio alone is ~35k, gs_task ~9k, the job binaries ~11k).
 * At 65536 the registry silently truncated every image registered past the cap
 * -- and the SPURS job-chain policy, the job binaries and the Edge geometry
 * task (gs_task) register LAST, so their functions were dropped wholesale and
 * every indirect branch into them fell through to a branch-to-0. Size it for
 * the real workload; the overflow is now loud (see spu_register_function). */
#define SPU_FN_REGISTRY_MAX 262144
static spu_reg_entry s_registry[SPU_FN_REGISTRY_MAX];
static uint32_t s_registry_count = 0;

/* Accessor for the sampling profiler in ppu_loader.cpp. Its address->function
 * map is built from the PPU table alone, which leaves the SPU-lifted bodies as
 * unowned gaps -- and an [entry, next_entry) extent then charges every SPU
 * sample to whichever PPU function happens to precede it. That produced a
 * confident and completely wrong "99% of guest time in one PPU function".
 * Handing the SPU host pointers over lets the map cover both, so the gaps are
 * real entries and the extents mean something. */
uint32_t spu_registry_size(void) { return s_registry_count; }

int spu_registry_entry(uint32_t i, void** host, uint32_t* ls_addr)
{
    if (i >= s_registry_count) return 0;
    if (host)    *host    = (void*)s_registry[i].fn;
    if (ls_addr) *ls_addr = s_registry[i].addr;
    return 1;
}

/* Hash index over the registry. spu_lookup runs on EVERY guest indirect
 * branch -- with the Bink decoder live that is millions of dispatches per
 * second, and the old linear scan (~950 binkspu entries walked per branch)
 * dominated movie playback. Buckets chain in REGISTRATION ORDER (tail
 * append) so the first-registered-match-wins semantics of the linear scan
 * are preserved exactly; the image-id wildcard rules stay in the bucket
 * walk, which is 1-3 entries (same LS addr across overlapping images).
 * Registration is startup-single-threaded; lookups treat the index as
 * read-only. Chain links store index+1 so zero-init means "empty". */
#define SPU_FN_HASH_SIZE 131072   /* power of two, ~= MAX/2 -> load factor ~2 */
static uint32_t s_hash_head[SPU_FN_HASH_SIZE];
static uint32_t s_hash_tail[SPU_FN_HASH_SIZE];
static uint32_t s_hash_next[SPU_FN_REGISTRY_MAX];

static inline uint32_t spu_fn_hash(uint32_t addr)
{
    return ((addr >> 2) * 2654435761u) & (SPU_FN_HASH_SIZE - 1);
}

/* Image currently being registered. SPURS images (kernel/policy/job) overlap in
 * LS, so each registers under a distinct id via spu_begin_image() before calling
 * its (prefixed) spu_recomp_register(). Single-image callers leave it 0. */
static int s_reg_image = 0;
void spu_begin_image(int image_id) { s_reg_image = image_id; }

void spu_register_function(uint32_t addr, spu_fn fn)
{
    if (s_registry_count >= SPU_FN_REGISTRY_MAX) {
        /* Silent truncation here dropped whole late-registered SPU images and
         * cost a multi-round hunt (the branch-to-0 looked like a lifter/overlay
         * bug). Never again: say so, once, loudly, with the number to raise the
         * cap to. */
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr,
                    "[spu] FATAL: SPU function registry full at %u entries "
                    "(SPU_FN_REGISTRY_MAX=%u) -- image %d function 0x%05X and all "
                    "later registrations DROPPED. Raise SPU_FN_REGISTRY_MAX.\n",
                    s_registry_count, (unsigned)SPU_FN_REGISTRY_MAX,
                    s_reg_image, addr);
            fflush(stderr);
        }
        return;
    }
    uint32_t i = s_registry_count;
    s_registry[i].addr = addr;
    s_registry[i].fn = fn;
    s_registry[i].image_id = s_reg_image;
    s_registry_count = i + 1;
    uint32_t h = spu_fn_hash(addr);
    s_hash_next[i] = 0;
    if (s_hash_head[h] == 0)
        s_hash_head[h] = i + 1;
    else
        s_hash_next[s_hash_tail[h] - 1] = i + 1;
    s_hash_tail[h] = i + 1;
}

/* Report a CROSS-IMAGE match: the registry served a function some other image
 * registered at this address, because one side's image_id was the 0 wildcard.
 *
 * This is a real hazard, reported with evidence by @canersaka (#59): SPURS job
 * chains load job binaries into descriptor-assigned LS slots, so the same binary
 * can sit at different addresses on different rounds. If the lift was fixed-base
 * and the exact match misses, the wildcard can serve ANOTHER image's function at
 * that address -- the job then runs the wrong program end to end, returns
 * cleanly, and the only symptom is something three layers downstream that never
 * happens. Their words: one log line would have saved a day.
 *
 * Logging, not refusing. Refusing the wildcard outright was measured and does
 * NOT survive contact: a dormant task image can register spans overlapping the
 * resident kernel's low LS range, so a legitimate service-to-kernel yield gets
 * falsely rejected. Telling resident from dormant needs residency tracking the
 * registry does not have. So this reports and keeps going.
 *
 * Deduped per (addr, from, to): spu_lookup runs millions of times a second, and
 * a substitution that repeats every dispatch would bury the log it is meant to
 * make readable. */
static void spu_report_cross_image(uint32_t addr, int want, int got)
{
    enum { SEEN_MAX = 64 };
    static struct { uint32_t addr; int want, got; } seen[SEEN_MAX];
    static unsigned n_seen = 0;
    for (unsigned i = 0; i < n_seen; i++)
        if (seen[i].addr == addr && seen[i].want == want && seen[i].got == got)
            return;
    if (n_seen < SEEN_MAX) {
        seen[n_seen].addr = addr; seen[n_seen].want = want; seen[n_seen].got = got;
        n_seen++;
    }
    fprintf(stderr,
            "[spu] CROSS-IMAGE dispatch: LS 0x%05X requested by image %d, served "
            "by image %d's function (id-0 wildcard). If this job was loaded at a "
            "runtime-chosen base, it may be running the WRONG program -- see "
            "issue #59.\n", addr, want, got);
}

spu_fn spu_lookup(uint32_t addr, int image_id)   /* exported: clang-built fast-path dispatch (spu_dispatch_mt.c) needs it */
{
    /* Per-thread cache of hits: dozens of job modules and overlays register
     * at the same LS addresses (0x4000...), so a bucket walk there is long.
     * The registry only grows; an entry is valid while its size is unchanged.
     * SPU_LOOKUP_CACHE=0: off. */
    static int s_lc = -1;
    if (s_lc < 0) { const char* e = getenv("SPU_LOOKUP_CACHE"); s_lc = !(e && e[0] == '0'); }
    static SPU_TLS struct { uint32_t addr, gen; int img; spu_fn fn; } t_lk[1024];
    const uint32_t lki = ((addr >> 2) ^ ((uint32_t)image_id * 0x9E3779B1u >> 22)) & 1023u;
    if (s_lc && t_lk[lki].fn && t_lk[lki].addr == addr && t_lk[lki].img == image_id &&
        t_lk[lki].gen == s_registry_count)
        return t_lk[lki].fn == (spu_fn)(uintptr_t)1 ? NULL : t_lk[lki].fn;   /* 1: cached miss */
    /* Match the context's active image; image_id 0 (context or entry) matches
     * any, for back-compat with single-image contexts. */
    for (uint32_t n = s_hash_head[spu_fn_hash(addr)]; n; n = s_hash_next[n - 1]) {
        const spu_reg_entry* e = &s_registry[n - 1];
        if (e->addr == addr &&
            (image_id == 0 || e->image_id == 0 || e->image_id == image_id)) {
            /* One compare on the return path; the report itself is rare. */
            if (e->image_id != image_id)
                spu_report_cross_image(addr, image_id, e->image_id);
            t_lk[lki].addr = addr; t_lk[lki].img = image_id;
            t_lk[lki].gen = s_registry_count; t_lk[lki].fn = e->fn;
            return e->fn;
        }
    }
    if (s_lc) { t_lk[lki].addr = addr; t_lk[lki].img = image_id;
                t_lk[lki].gen = s_registry_count; t_lk[lki].fn = (spu_fn)(uintptr_t)1; }
    return NULL;
}

/* Which image registered a function at this LS address, or -1 if none did.
 *
 * Deliberately NOT spu_lookup(addr, 0): that call means "dispatch here in a
 * context with no image", so it reports a cross-image substitution for every
 * hit in a real image. This is a question about the registry, not a dispatch,
 * and it runs once per SPU thread start. First match wins, in registration
 * order, so it agrees with which function spu_lookup would actually serve. */
int spu_image_of_function(uint32_t addr)
{
    for (uint32_t n = s_hash_head[spu_fn_hash(addr)]; n; n = s_hash_next[n - 1]) {
        const spu_reg_entry* e = &s_registry[n - 1];
        if (e->addr == addr) return e->image_id;
    }
    return -1;
}

/* Does a lifted function exist at this LS address (any image)? Lets the lv2
 * layer decide between real SPU execution and the PPU-fallback paths. */
int spu_have_function(uint32_t addr)
{
    return spu_image_of_function(addr) >= 0;
}

/* ---- Dynamic overlays: code that only exists at run time --------------------
 * Job bodies / overlays that a title streams into local store from data the
 * build never sees have no lift; the SPU microstepper hands them to the
 * (50-100x slower) interpreter. Two halves close that gap:
 *   - SPU_DUMP_UNLIFTED=<dir>: every distinct LS code region that goes to the
 *     interpreter is saved once (content hash + LS range + entry points seen),
 *     as ovl_<hash>_<lsa>_<size>.bin plus an .ent file of entry pcs. The region
 *     is the DMA GET (merged with LS+EA-contiguous neighbours: DMA-list
 *     elements) that last wrote the entry pc.
 *   - an offline lifter turns the dumps into lifted images keyed by that hash
 *     (title tools: lift_dynovl.py -> a registry calling spu_dynovl_register).
 *     At dispatch a miss inside a registered LS range is resolved by comparing
 *     the live LS bytes (16-byte head at the first entry, then the full hash, cached until a GET
 *     overlaps the range) -- a mismatch simply falls back to the interpreter.
 * SPU_DYNOVL=0 ignores the registered overlays (A/B against the interpreter). */
typedef struct { uint64_t hash; uint32_t lsa, size, head_off; int image_id; uint8_t head[16];
                 uint32_t nranges; const uint32_t* ranges;
                 uint32_t cmin, cmax;   /* bounding box of head + code ranges (absolute LS) */
                 uint8_t* ref; } spu_dynovl;   /* ref: the region as first verified (code ranges valid) */
#define SPU_DYNOVL_MAX 1024
static spu_dynovl s_dyn[SPU_DYNOVL_MAX];
static int s_dyn_n;
unsigned char g_spu_dyn_page[SPU_LS_SIZE >> 8];   /* 256-byte pages that hold a registered overlay */
static SPU_TLS const spu_context* t_dyn_ctx;
static SPU_TLS uint8_t t_dyn_ok[SPU_DYNOVL_MAX];
/* t_dyn_pgint[page]: how many overlays currently in a state that a GET into that page can change
 * (state 3: the 16 head bytes only; states 1, 2: the overlay's span; conservative for 2). A GET whose pages all
 * read 0 cannot invalidate anything: spu_dynovl_note_get skips the walk over the (dozens of) overlays that share
 * LS 0x4000.. (it was ~10% of wid0 lane time). Every state change goes through dyn_set(). */
static SPU_TLS uint16_t t_dyn_pgint[SPU_LS_SIZE >> 8];
static inline void dyn_interest(int i, int st, int delta)
{
    if (st == 0) return;
    const spu_dynovl* d = &s_dyn[i];
    uint32_t lo, hi;
    if (st == 3) { lo = d->lsa + d->head_off; hi = lo + 15; }
    else         { lo = d->lsa; hi = d->lsa + (d->size ? d->size - 1 : 0); }
    for (uint32_t pg = lo >> 8; pg <= (hi >> 8) && pg < (SPU_LS_SIZE >> 8); pg++) t_dyn_pgint[pg] = (uint16_t)(t_dyn_pgint[pg] + delta);
}
static inline void dyn_set(int i, uint8_t v)
{
    const uint8_t o = t_dyn_ok[i];
    if (o == v) return;
    dyn_interest(i, o, -1); t_dyn_ok[i] = v; dyn_interest(i, v, +1);
}
static SPU_TLS int t_dyn_last = -1;   /* last verified overlay (spu_dynovl_find fast path) */
/* t_dyn_ok: 0 unknown, 1 verified, 2 head ok but hash differs (until a GET into the
 * range), 3 head differs (until a GET over the 16 head bytes). A page whose every
 * overlay is in state 3 is negative while t_dyn_neg[page] == t_dyn_hgen: the
 * graphics jobs' own code at LS 0x4000.. is no overlay, and every branch there
 * walked all overlays registered at 0x4000 comparing heads. Only a GET over some
 * head bumps the generation, so data GETs into the range keep the cache. */
static SPU_TLS uint32_t t_dyn_neg[SPU_LS_SIZE >> 8];
static SPU_TLS uint32_t t_dyn_hgen = 1;
/* Indirect-branch cache generation (SPU_IBCACHE, spu_indirect_branch): bumped whenever a
 * dynovl verification decision (t_dyn_ok / the page-negative cache) is dropped or reset,
 * i.e. exactly when the answer of spu_dynovl_find for some pc can change. */
static SPU_TLS uint32_t t_ib_gen = 1;
/* Per-page head generation (SPU_DYNOVL_PGEN, default on): a page is negative while
 * t_dyn_neg[pg] == t_dyn_hgen + t_dyn_pgen[pg]. A GET over the head of an overlay in
 * state 3 bumps only the pages that overlay touches (not t_dyn_hgen, which every page
 * sees), so the job's data GETs / code streaming elsewhere no longer flush the cache. */
static SPU_TLS uint32_t t_dyn_pgen[SPU_LS_SIZE >> 8];
static long long s_dyn_verified, s_dyn_mismatch;
/* SPU_DYNOVL_NEG2 (default on): state 2 (head matches, code hash differs) joins state 3
 * in the page-negative cache. The graphics job modules share one 14.5 KB header at LS
 * 0x4000 (a dozen overlays with the same head): a module with no registered lift made
 * every branch re-walk all of them (head compare each, never negative). A GET into the
 * overlay's code/head resets it (note_get) and bumps its pages' generation. */
static int spu_dyn_neg2(void)
{
    static int s = -1;
    if (s < 0) { const char* e = getenv("SPU_DYNOVL_NEG2"); s = !(e && e[0] == '0'); }
    return s;
}

static uint64_t spu_dynovl_mix(uint64_t h, const uint8_t* p, uint32_t n)   /* n: multiple of 8 */
{
    for (uint32_t i = 0; i + 8 <= n; i += 8) {
        uint64_t w; memcpy(&w, p + i, 8);
        h = (h ^ w) * 0xFF51AFD7ED558CCDull;
        h ^= h >> 29;
    }
    return h;
}
static uint64_t spu_dynovl_fin(uint64_t h, uint32_t n)
{
    h ^= n; h *= 0xC4CEB9FE1A85EC53ull; h ^= h >> 32;
    return h;
}
uint64_t spu_dynovl_hash(const uint8_t* p, uint32_t n)   /* whole region (capture file names) */
{
    return spu_dynovl_fin(spu_dynovl_mix(0x9E3779B97F4A7C15ull, p, n), n);
}
/* Hash of only the bytes the lift was made from (code ranges: off,len pairs; the
 * region's data/bss part changes at run time and must not take part). */
static uint64_t spu_dynovl_hash_ranges(const uint8_t* base, const uint32_t* r, uint32_t nr)
{
    uint64_t h = 0x9E3779B97F4A7C15ull; uint32_t tot = 0;
    for (uint32_t i = 0; i < nr; i++) { h = spu_dynovl_mix(h, base + r[2*i], r[2*i+1]); tot += r[2*i+1]; }
    return spu_dynovl_fin(h, tot);
}

/* Are the overlay's code bytes live at `ls`? The first match is decided by the
 * hash, and the bytes are then kept: every later check is a memcmp of the code
 * ranges against them. The synth overlay (60 KB of code at LS 0x4000) is
 * re-verified each time a job streams its code back in, a few times per run;
 * the serial hash took ~10 us there, 7% of the audio SPU's time in combat.
 * SPU_DYNOVL_REF=0: hash every time. */
static int spu_dynovl_same(spu_dynovl* d, const uint8_t* ls)
{
    static int s_ref = -1;
    if (s_ref < 0) { const char* e = getenv("SPU_DYNOVL_REF"); s_ref = !(e && e[0] == '0'); }
    const uint8_t* ref = __atomic_load_n(&d->ref, __ATOMIC_ACQUIRE);
    if (ref) {
        for (uint32_t i = 0; i < d->nranges; i++)
            if (memcmp(ls + d->ranges[2*i], ref + d->ranges[2*i], d->ranges[2*i+1]) != 0) return 0;
        return 1;
    }
    if (spu_dynovl_hash_ranges(ls, d->ranges, d->nranges) != d->hash) return 0;
    if (s_ref && d->nranges) {
        uint8_t* copy = (uint8_t*)malloc(d->size);
        if (copy) {
            memcpy(copy, ls, d->size);
            uint8_t* none = NULL;
            if (!__atomic_compare_exchange_n(&d->ref, &none, copy, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) free(copy);
        }
    }
    return 1;
}

void spu_dynovl_register(uint64_t hash, uint32_t lsa, uint32_t size, const uint8_t head[16], uint32_t head_off, int image_id,
                         uint32_t nranges, const uint32_t* ranges)
{
    static int s_off = -1;
    if (s_off < 0) { const char* e = getenv("SPU_DYNOVL"); s_off = (e && e[0] == '0'); }
    if (s_off || s_dyn_n >= SPU_DYNOVL_MAX || !size || lsa + size > SPU_LS_SIZE) return;
    spu_dynovl* d = &s_dyn[s_dyn_n++];
    d->hash = hash; d->lsa = lsa; d->size = size; d->image_id = image_id;
    d->head_off = head_off < size - 15 ? head_off : 0;
    d->nranges = nranges; d->ranges = ranges; d->ref = NULL;
    memcpy(d->head, head, 16);
    d->cmin = lsa + d->head_off; d->cmax = lsa + d->head_off + 16;
    for (uint32_t r = 0; r < nranges; r++) {
        if (lsa + ranges[2*r] < d->cmin) d->cmin = lsa + ranges[2*r];
        if (lsa + ranges[2*r] + ranges[2*r+1] > d->cmax) d->cmax = lsa + ranges[2*r] + ranges[2*r+1];
    }
    for (uint32_t a = lsa >> 8; a <= (lsa + size - 1) >> 8; a++) g_spu_dyn_page[a] = 1;
}

/* Per 256-byte LS page: the registered overlays that touch it, in registration
 * order (built once, lazily, after registration): spu_dynovl_find walked all
 * ~s_dyn_n overlays on every branch whose page-negative cache was invalidated
 * (5% of the graphics lane-0 time, SPURS_LANE_PROF). SPU_DYNOVL_PAGELIST=0: old walk. */
static uint16_t* s_pg_list[SPU_LS_SIZE >> 8];
static uint16_t  s_pg_n[SPU_LS_SIZE >> 8];
static volatile LONG s_pg_state;   /* 0 = not built, 1 = building, 2 = built (for s_pg_built_n overlays) */
static int s_pg_built_n;
static int spu_dynovl_pagelist(void)
{
    if (s_pg_state == 2 && s_pg_built_n == s_dyn_n) return 1;   /* built (only when enabled) */
    static int s_on = -1;
    if (s_on < 0) { const char* e = getenv("SPU_DYNOVL_PAGELIST"); s_on = !(e && e[0] == '0'); }
    if (!s_on) return 0;
    if (s_pg_state == 2 && s_pg_built_n == s_dyn_n) return 1;
    if (InterlockedCompareExchange(&s_pg_state, 1, 0) != 0) {
        while (s_pg_state == 1) YieldProcessor();
        return s_pg_state == 2 && s_pg_built_n == s_dyn_n;
    }
    for (uint32_t pg = 0; pg < (SPU_LS_SIZE >> 8); pg++) {
        uint32_t n = 0;
        for (int i = 0; i < s_dyn_n; i++)
            if ((pg << 8) < s_dyn[i].lsa + s_dyn[i].size && (pg << 8) + 256 > s_dyn[i].lsa) n++;
        s_pg_list[pg] = n ? (uint16_t*)malloc(n * sizeof(uint16_t)) : NULL;
        s_pg_n[pg] = 0;
        for (int i = 0; i < s_dyn_n && s_pg_list[pg]; i++)
            if ((pg << 8) < s_dyn[i].lsa + s_dyn[i].size && (pg << 8) + 256 > s_dyn[i].lsa) s_pg_list[pg][s_pg_n[pg]++] = (uint16_t)i;
    }
    s_pg_built_n = s_dyn_n;
    MemoryBarrier();
    s_pg_state = 2;
    return 1;
}

/* Image id of the registered overlay whose bytes are live in LS at `pc`, else 0. */
int spu_dynovl_find(const spu_context* ctx, uint32_t pc)
{
    if (!s_dyn_n) return 0;
    if (t_dyn_ctx != ctx) { t_ib_gen++; memset(t_dyn_ok, 0, sizeof t_dyn_ok); memset(t_dyn_pgint, 0, sizeof t_dyn_pgint); t_dyn_ctx = ctx; t_dyn_last = -1; t_dyn_hgen++; memset(t_dyn_pgen, 0, sizeof t_dyn_pgen); memset(t_dyn_neg, 0, sizeof t_dyn_neg); }
    /* Fast path: the overlay this thread last verified still covers pc and no
     * GET has touched its range since (note_get clears t_dyn_ok): its bytes are
     * the verified ones. With dozens of overlays registered at LS 0x4000 the
     * walk below compared 16 bytes per overlay on every branch there. */
    if (t_dyn_last >= 0 && t_dyn_ok[t_dyn_last] == 1 &&
        pc - s_dyn[t_dyn_last].lsa < s_dyn[t_dyn_last].size)
        return s_dyn[t_dyn_last].image_id;
    const uint32_t pg = pc >> 8;
    static int s_neg = -1;   /* SPU_DYNOVL_NEG=0: no page-negative cache (A/B) */
    if (s_neg < 0) { const char* e = getenv("SPU_DYNOVL_NEG"); s_neg = !(e && e[0] == '0'); }
    if (s_neg && t_dyn_neg[pg] == t_dyn_hgen + t_dyn_pgen[pg]) return 0;
    const int use_pl = spu_dynovl_pagelist();
    const int s_neg2 = spu_dyn_neg2();
    const uint16_t* pl = use_pl ? s_pg_list[pg] : NULL; const int pln = use_pl ? s_pg_n[pg] : s_dyn_n;
    for (int ii = 0; ii < pln; ii++) {
        const int i = use_pl ? pl[ii] : ii;
        const spu_dynovl* d = &s_dyn[i];
        if (pc - d->lsa >= d->size || (s_neg && (t_dyn_ok[i] == 3 || (t_dyn_ok[i] == 2 && s_neg2)))) continue;
        if (memcmp(ctx->ls + d->lsa + d->head_off, d->head, 16) != 0) { dyn_set(i, 3); continue; }
        if (t_dyn_ok[i] == 1) { t_dyn_last = i; return d->image_id; }
        if (t_dyn_ok[i] == 2) continue;          /* known mismatch until the next GET into the range */
        if (!spu_dynovl_same(&s_dyn[i], ctx->ls + d->lsa)) {
            dyn_set(i, 2);
            if (__atomic_add_fetch(&s_dyn_mismatch, 1, __ATOMIC_RELAXED) <= 8)
                fprintf(stderr, "[spu-dyn] head matches but hash differs: lsa=0x%05X size=%u img=%d (falls back)\n",
                        d->lsa, d->size, d->image_id);
            continue;
        }
        dyn_set(i, 1);
        if (__atomic_add_fetch(&s_dyn_verified, 1, __ATOMIC_RELAXED) <= 16)
            fprintf(stderr, "[spu-dyn] overlay img=%d live at LS 0x%05X (%u B), hash verified\n",
                    d->image_id, d->lsa, d->size);
        t_dyn_last = i;
        return d->image_id;
    }
    /* Miss: the whole page is negative if every overlay touching it has a foreign head. */
    if (!s_neg) return 0;
    for (int ii = 0; ii < pln; ii++) {
        const int i = use_pl ? pl[ii] : ii;
        const spu_dynovl* d = &s_dyn[i];
        if ((pg << 8) >= d->lsa + d->size || (pg << 8) + 256 <= d->lsa || t_dyn_ok[i] == 3 || (t_dyn_ok[i] == 2 && s_neg2)) continue;
        if (memcmp(ctx->ls + d->lsa + d->head_off, d->head, 16) != 0) { dyn_set(i, 3); continue; }
        return 0;
    }
    t_dyn_neg[pg] = t_dyn_hgen + t_dyn_pgen[pg];
    return 0;
}

/* Ring of recent DMA GETs into LS (capture only). */
typedef struct { uint32_t lsa, size, ea; uint64_t seq; } spu_get_rec;
static SPU_TLS spu_get_rec t_get_ring[256];
static SPU_TLS uint64_t t_get_seq;
static int spu_dyn_capture_on(void)
{
    static int s = -1;
    if (s < 0) { const char* e = getenv("SPU_DUMP_UNLIFTED"); s = (e && *e) ? 1 : 0; }
    return s;
}

/* Does a GET over [lsa, lsa+size) touch the bytes that decide whether overlay d is
 * live (its code ranges, or the 16 head bytes)? A GET into the rest of its LS span
 * (job data / bss buffers that share the 71 KB region) leaves the verification valid;
 * flushing it on every such GET sent each branch back through the full re-verify (head
 * compare + 70 KB memcmp). SPU_DYNOVL_CODEHIT=0: any GET into the span flushes (old). */
static inline int spu_dyn_code_hit(const spu_dynovl* d, uint32_t lsa, uint32_t size)
{
    if (!d->nranges) return 1;
    if (!(lsa < d->cmax && lsa + size > d->cmin)) return 0;
    const uint32_t h = d->lsa + d->head_off;
    if (lsa < h + 16 && lsa + size > h) return 1;
    for (uint32_t r = 0; r < d->nranges; r++) {
        const uint32_t a = d->lsa + d->ranges[2*r], b = a + d->ranges[2*r+1];
        if (lsa < b && lsa + size > a) return 1;
    }
    return 0;
}

/* Apply a GET over [lsa, lsa+size) to overlay i's verification state. Returns 1 when a
 * negative state (2/3) was dropped: the pages it touches must be re-evaluated. */
static inline int spu_dyn_inval_one(int i, uint32_t lsa, uint32_t size, int neg2, int codehit)
{
    const int st = t_dyn_ok[i];
    if (st == 0) return 0;
    const spu_dynovl* d = &s_dyn[i];
    if (!(lsa < d->lsa + d->size && lsa + size > d->lsa)) return 0;
    if (st == 3) {
        const uint32_t h = d->lsa + d->head_off;
        if (lsa < h + 16 && lsa + size > h) { dyn_set(i, 0); t_ib_gen++; return 1; }
        return 0;
    }
    if (st == 2 && neg2) {
        if (spu_dyn_code_hit(d, lsa, size)) { dyn_set(i, 0); t_ib_gen++; return 1; }
        return 0;
    }
    if (!codehit || spu_dyn_code_hit(d, lsa, size)) { dyn_set(i, 0); t_ib_gen++; }
    return 0;
}

static void spu_dynovl_note_get(const spu_context* ctx, uint32_t ea, uint32_t lsa, uint32_t size)
{
    /* No registered overlay touches any page of [lsa, lsa+size): nothing to invalidate. */
    if (t_dyn_ctx == ctx && size) {
        static int s_pgi = -1; if (s_pgi < 0) { const char* e = getenv("SPU_DYN_PGINT"); s_pgi = !(e && e[0] == '0'); }
        int any = 0;
        for (uint32_t a = lsa >> 8, e = (lsa + size - 1) >> 8; a <= e && a < (SPU_LS_SIZE >> 8); a++) if (s_pgi ? t_dyn_pgint[a] : g_spu_dyn_page[a]) { any = 1; break; }
        if (!any && size < SPU_LS_SIZE) goto note_done;
    }
    if (t_dyn_ctx == ctx) {
        static int s_pgen = -1, s_codehit = -1; const int neg2 = spu_dyn_neg2();
        if (s_pgen < 0) { const char* e = getenv("SPU_DYNOVL_PGEN"); s_pgen = !(e && e[0] == '0'); }
        if (s_codehit < 0) { const char* e = getenv("SPU_DYNOVL_CODEHIT"); s_codehit = !(e && e[0] == '0'); }
        const uint32_t pa = lsa >> 8, pe = size ? (lsa + size - 1) >> 8 : pa;
        /* Overlays touching the GET's pages (page lists), else the full walk. */
        if (s_pgen && size && size < SPU_LS_SIZE && pe - pa < 16 && spu_dynovl_pagelist()) {
            for (uint32_t pg = pa; pg <= pe && pg < (SPU_LS_SIZE >> 8); pg++) {
                const uint16_t* pl = s_pg_list[pg];
                for (int ii = 0, n = s_pg_n[pg]; ii < n; ii++) {
                    const int i = pl[ii];
                    if (spu_dyn_inval_one(i, lsa, size, neg2, s_codehit))
                        for (uint32_t q = s_dyn[i].lsa >> 8, qe = (s_dyn[i].lsa + s_dyn[i].size - 1) >> 8; q <= qe; q++) t_dyn_pgen[q]++;
                }
            }
        } else {
            for (int i = 0; i < s_dyn_n; i++)
                if (spu_dyn_inval_one(i, lsa, size, neg2, s_codehit)) t_dyn_hgen++;
        }
    }
note_done:
    if (size >= 128 && spu_dyn_capture_on()) {
        spu_get_rec* r = &t_get_ring[t_get_seq & 255];
        r->lsa = lsa; r->size = size; r->ea = ea; r->seq = ++t_get_seq;
    }
}

/* Called right before the interpreter takes over at `entry` (the pc the
 * dispatcher could not resolve): save the code region once. */
static void spu_dynovl_capture(spu_context* ctx, uint32_t entry)
{
    if (!spu_dyn_capture_on()) return;
    static const char* dir; if (!dir) dir = getenv("SPU_DUMP_UNLIFTED");
    int k = -1; uint64_t best = 0;
    for (int i = 0; i < 256; i++) {
        const spu_get_rec* r = &t_get_ring[i];
        if (r->seq && entry >= r->lsa && entry - r->lsa < r->size && r->seq > best) { best = r->seq; k = i; }
    }
    if (k < 0) {
        static int _n; if (_n < 8 && ++_n)
            fprintf(stderr, "[spu-dyn] capture: no recent GET covers pc=0x%05X img=%d\n", entry, ctx->image_id);
        return;
    }
    uint32_t lo = t_get_ring[k].lsa, hi = lo + t_get_ring[k].size;
    uint32_t lo_ea = t_get_ring[k].ea, hi_ea = lo_ea + t_get_ring[k].size;
    for (int pass = 0; pass < 64; pass++) {
        int grew = 0;
        for (int i = 0; i < 256; i++) {
            const spu_get_rec* r = &t_get_ring[i];
            if (!r->seq || (r->seq > best ? r->seq - best : best - r->seq) > 48) continue;
            if (r->lsa == hi && r->ea == hi_ea) { hi += r->size; hi_ea += r->size; grew = 1; }
            else if (r->lsa + r->size == lo && r->ea + r->size == lo_ea) { lo = r->lsa; lo_ea = r->ea; grew = 1; }
        }
        if (!grew) break;
    }
    lo &= ~15u; hi = (hi + 15u) & ~15u;
    if (hi > SPU_LS_SIZE || lo >= hi) return;
    const uint32_t size = hi - lo;
    const uint64_t h = spu_dynovl_hash(ctx->ls + lo, size);
    static uint64_t seen[2048]; static uint32_t seen_lsa[2048]; static int nseen; static volatile int lock;
    static uint32_t ents[2048][32]; static int nents[2048];
    while (__atomic_exchange_n(&lock, 1, __ATOMIC_ACQUIRE)) { }
    int idx = -1;
    for (int i = 0; i < nseen; i++) if (seen[i] == h && seen_lsa[i] == lo) { idx = i; break; }
    char path[512];
    if (idx < 0 && nseen < 2048) {
        idx = nseen++; seen[idx] = h; seen_lsa[idx] = lo; nents[idx] = 0;
        snprintf(path, sizeof path, "%s/ovl_%016llX_%05X_%X.bin", dir, (unsigned long long)h, lo, size);
        FILE* f = fopen(path, "wb");
        if (f) { fwrite(ctx->ls + lo, 1, size, f); fclose(f); }
        fprintf(stderr, "[spu-dyn] captured region LS 0x%05X..0x%05X (%u B) hash %016llX img=%d ea=0x%08X -> %s\n",
                lo, hi, size, (unsigned long long)h, ctx->image_id, lo_ea, path);
    }
    if (idx >= 0) {
        int dup = 0;
        for (int j = 0; j < nents[idx]; j++) if (ents[idx][j] == entry) { dup = 1; break; }
        if (!dup && nents[idx] < 32) {
            ents[idx][nents[idx]++] = entry;
            snprintf(path, sizeof path, "%s/ovl_%016llX_%05X_%X.ent", dir, (unsigned long long)h, lo, size);
            FILE* f = fopen(path, "ab");
            if (f) { fprintf(f, "0x%05X\n", entry); fclose(f); }
        }
    }
    __atomic_store_n(&lock, 0, __ATOMIC_RELEASE);
}

/* ---- Relocated position-independent blobs ------------------------------------
 * A job manager streams the same small blob (a numeric job body: no channel ops,
 * only relative branches, one `bi $r0` out) to a different LS address on every
 * call, so a lift at one address (spu_dynovl_find) never matches and the
 * interpreter ran it, ~24 us a call. lift_dynovl.py lifts ONE instance at its
 * canonical base with spu_lifter.py --pic-reloc (lqr/stqr = imm + g_spu_reloc) and
 * registers it here with the reference bytes. At an unresolved pc the blob
 * runs when the 16-byte head and then ALL `size` live bytes equal the reference
 * (covers the constants behind the code); g_spu_reloc = pc - canon for the
 * duration. SPU_DYNOVL_RELOC=0 (or SPU_DYNOVL=0) ignores the registered blobs. */
typedef struct { uint64_t hash; uint32_t canon, size; const uint8_t* ref; int image_id; spu_fn fn; } spu_dynovl_rel;
#define SPU_DYNOVL_REL_MAX 32
static spu_dynovl_rel s_rel[SPU_DYNOVL_REL_MAX];
static int s_rel_n;
SPU_THREAD_LOCAL uint32_t g_spu_reloc;

void spu_dynovl_register_reloc(uint64_t hash, uint32_t canon, uint32_t size, const uint8_t* ref, int image_id)
{
    static int s_off = -1;
    if (s_off < 0) {
        const char* e = getenv("SPU_DYNOVL_RELOC"); const char* e2 = getenv("SPU_DYNOVL");
        s_off = (e && e[0] == '0') || (e2 && e2[0] == '0');
        if (s_off) fprintf(stderr, "[dynovl-reloc] disabled by SPU_DYNOVL_RELOC=0 / SPU_DYNOVL=0\n");
    }
    if (s_off || s_rel_n >= SPU_DYNOVL_REL_MAX || size < 16 || (canon & 15) || canon + size > SPU_LS_SIZE) return;
    spu_dynovl_rel* r = &s_rel[s_rel_n++];
    r->hash = hash; r->canon = canon; r->size = size; r->ref = ref; r->image_id = image_id; r->fn = NULL;
    fprintf(stderr, "[dynovl-reloc] registered blob %016llX canon LS 0x%05X (%u B) img=%d\n",
            (unsigned long long)hash, canon, size, image_id);
}

/* Index of the blob whose bytes are live at `pc`, else -1. */
static int spu_dynovl_reloc_match(const spu_context* ctx, uint32_t pc)
{
    for (int i = 0; i < s_rel_n; i++) {
        const spu_dynovl_rel* r = &s_rel[i];
        if (((pc - r->canon) & 15) || pc + r->size > SPU_LS_SIZE) continue;
        if (memcmp(ctx->ls + pc, r->ref, 16) != 0) continue;
        if (memcmp(ctx->ls + pc, r->ref, r->size) != 0) {
            static volatile long n_diff; long k = __atomic_add_fetch(&n_diff, 1, __ATOMIC_RELAXED);
            if (k <= 8)
                fprintf(stderr, "[dynovl-reloc] head matches but body differs at LS 0x%05X (blob %016llX): falls back\n",
                        pc, (unsigned long long)r->hash);
            continue;
        }
        return i;
    }
    return -1;
}

/* Run the blob live at ctx->pc. Entered like any lifted function (the dispatcher's
 * `fn(ctx)`), but the chunks inside are linked by trampolines that the enclosing
 * SPU_DRAIN would run after we returned, with g_spu_reloc already cleared: so the
 * hops are drained here, up to the point the blob leaves (`bi $r0`: a host return
 * at host_depth>0, else spu_indirect_branch pending at the link). */
static int spu_dynovl_reloc_enter(spu_context* ctx)
{
    const uint32_t pc = ctx->pc & SPU_LS_MASK;
    const int i = spu_dynovl_reloc_match(ctx, pc);
    if (i < 0) return 0;
    spu_dynovl_rel* r = &s_rel[i];
    spu_fn fn = r->fn;
    if (!fn) fn = r->fn = spu_lookup(r->canon, r->image_id);
    if (!fn) {
        static int _n = 0; if (_n < 4 && ++_n)
            fprintf(stderr, "[dynovl-reloc] no lifted entry at canon 0x%05X img=%d\n", r->canon, r->image_id);
        return 0;
    }
    static volatile long n_enter; long k = __atomic_add_fetch(&n_enter, 1, __ATOMIC_RELAXED);
    if (k <= 8 || (k & (k - 1)) == 0)
        fprintf(stderr, "[dynovl-reloc] enter #%ld: blob %016llX live at LS 0x%05X (canon 0x%05X, delta %d) img=%d\n",
                k, (unsigned long long)r->hash, pc, r->canon, (int)(pc - r->canon), ctx->image_id);
    const uint32_t saved = g_spu_reloc;
    g_spu_reloc = pc - r->canon;
    ctx->pc = r->canon;
    spu_check_stack_reset(ctx, fn);
    fn(ctx);
    while (g_spu_trampoline_fn && g_spu_trampoline_fn != spu_indirect_branch) {
        void (*tf)(spu_context*) = g_spu_trampoline_fn;
        g_spu_trampoline_fn = 0;
        spu_hop(ctx, tf);
        tf(ctx);
    }
    g_spu_reloc = saved;
    return 1;
}

/* spu_lifted_lookup's answer for a relocated blob: lets the interpreter rejoin at
 * its base; the call re-matches at ctx->pc (the caller set it to the lookup pc). */
static void spu_dynovl_reloc_thunk(spu_context* ctx)
{
    if (!spu_dynovl_reloc_enter(ctx)) g_spu_trampoline_fn = spu_indirect_branch;
}

/* The pure interpreter's fast-path "is this LSA already lifted?" probe
 * (spu_interp.c, rejoin path). It IS the registry lookup, so overlay eviction and
 * self-modifying-code invalidation are honored automatically. Lived in
 * spu_fn_registry.c before the registry was consolidated into this TU. */
SPU_THREAD_LOCAL int g_spu_force_interp;   /* SPU_INTERP_WID (spurs_policy.c) */
SPU_THREAD_LOCAL int g_spu_fast_fp;        /* host float ops in SPU_FAST_FP_TLS images (spu_helpers.h) */
SPU_THREAD_LOCAL int g_spu_rz;             /* this thread runs MXCSR=RZ|DAZ|FTZ; SPU_RZ_FP_TLS images use the host float ops (spu_helpers.h) */
/* SPU_RZ_FP exact-path redos (an exponent-255 operand or an overflow); a line at 1, 2, 4, 8, ... so a hot spot shows up */
void spu_rz_slow_note(void)
{
    static volatile long n;
    long v = __atomic_add_fetch(&n, 1, __ATOMIC_RELAXED);
    if ((v & (v - 1)) == 0) { fprintf(stderr, "[spu-rz] %ld op(s) fell back to the exact FP path\n", v); fflush(stderr); }
}
spu_lifted_fn spu_lifted_lookup(const spu_context* ctx, uint32_t lsa)
{
    if (g_spu_force_interp) return NULL;
    /* The resident overlay first, as spu_drain_call does: a WWS job module's
     * lifted entries are registered under the module's own image, not the
     * policy module's (image 2) that ctx->image_id names while it runs. With
     * the primary image only, the interpreter never saw the module's code and
     * ran on through it -- ~290k steps a hand-off, ~109 s of 210 s of host
     * time in metropolis. SPU_LOOKUP_OVL=0: primary image only, as before. */
    if (s_dyn_n && ctx && g_spu_dyn_page[(lsa & SPU_LS_MASK) >> 8]) {
        int di = spu_dynovl_find(ctx, lsa & SPU_LS_MASK);
        if (di) { spu_lifted_fn f = (spu_lifted_fn)spu_lookup(lsa, di); if (f) return f; }
    }
    if (s_rel_n && ctx && spu_dynovl_reloc_match(ctx, lsa & SPU_LS_MASK) >= 0) return spu_dynovl_reloc_thunk;
    static int s_ovl = -1;
    if (s_ovl < 0) { const char* e = getenv("SPU_LOOKUP_OVL"); s_ovl = !(e && e[0] == '0'); }
    if (s_ovl && ctx && ctx->resident_ovl) {
        spu_lifted_fn f = (spu_lifted_fn)spu_lookup(lsa, ctx->resident_ovl);
        if (f) return f;
    }
    return (spu_lifted_fn)spu_lookup(lsa, ctx ? ctx->image_id : 0);
}

/* ---- Swappable code overlays (see spu_context.resident_ovl) --------------
 * A title registers each runtime-streamed overlay's SOURCE content EA with
 * the image id its lifted functions were registered under. The MFC GET path
 * marks that overlay resident in the streaming context; dispatch retries a
 * primary-image miss against the resident overlay's registry. */
typedef struct { uint32_t src_ea; int image_id; uint8_t sig[16]; int has_sig; uint32_t span; } spu_ovl_src;
/* 6 FMOD codec/DSP overlays + up to 89 WWS job-code modules (all stream into
 * the same job code buffer at LS 0x4000, dispatched by content signature). */
#define SPU_OVL_SRC_MAX 128
static spu_ovl_src s_ovl_src[SPU_OVL_SRC_MAX];
static int s_ovl_src_count = 0;

/* SPU_MFC_FAST2 pre-filter state (read inline by spu_dma.h, see mfc_get_note_needed).
 * g_ovl_note_always stays 1 until the first spu_overlay_note_get computes it, so a
 * GET before that takes the full path. g_ovl_sig_n counts the content-signature
 * entries (they can only hit a GET of >= 512 bytes); g_ovl_ea_bloom is a bitmap
 * over a hash of every EA-keyed entry's source EA. */
volatile uint32_t g_ovl_note_always = 1;
volatile uint32_t g_ovl_sig_n = 0;
uint8_t g_ovl_ea_bloom[512];
/* bloom over the first 4 bytes of every content signature: a GET of >= 512 B whose first word is not in it
 * cannot match any signature (no false negatives), so spu_overlay_note_get can skip the whole source scan. */
uint8_t g_ovl_sig_bloom[512];
int g_ovl_bloom_on = -1;   /* SPU_OVL_BLOOM=0 turns the signature bloom off (A/B) */
static inline int spu_ovl_bloom_on(void) { if (g_ovl_bloom_on < 0) { const char* e = getenv("SPU_OVL_BLOOM"); g_ovl_bloom_on = !(e && e[0] == '0'); } return g_ovl_bloom_on; }
static inline uint32_t spu_ovl_sig_hash(const uint8_t* p) { uint32_t w; memcpy(&w, p, 4); return (w * 0x9E3779B1u) >> 20; }
static inline uint32_t spu_ovl_ea_hash(uint32_t ea) { return (ea * 0x9E3779B1u) >> 20; }

void spu_overlay_register_source(uint32_t content_ea, int image_id)
{
    if (s_ovl_src_count < SPU_OVL_SRC_MAX) {
        s_ovl_src[s_ovl_src_count].src_ea = content_ea;
        s_ovl_src[s_ovl_src_count].image_id = image_id;
        s_ovl_src[s_ovl_src_count].has_sig = 0;
        { uint32_t h = spu_ovl_ea_hash(content_ea);
          __atomic_fetch_or(&g_ovl_ea_bloom[h >> 3], (uint8_t)(1u << (h & 7)), __ATOMIC_RELEASE); }
        s_ovl_src_count++;
    }
}

/* Register a bounded code image that may coexist with other streamed images.
 * Its functions must be translated at the local-store addresses used by the title. */
void spu_overlay_register_region(uint32_t content_ea, uint32_t span, int image_id)
{
    if (!span || span > SPU_LS_SIZE || s_ovl_src_count >= SPU_OVL_SRC_MAX) return;
    spu_overlay_register_source(content_ea, image_id);
    s_ovl_src[s_ovl_src_count - 1].span = span;
}

/* Register one-way runtime entries that replace the guest call stack.
 * Matching the resolved function, not just its LS address, avoids affecting
 * unrelated overlays using the same address. Registration precedes execution. */
static struct { uint32_t entry; int image_id; } s_stack_reset[16];
static unsigned s_stack_reset_count;
unsigned g_spu_stack_reset_count;      /* spu_hop's inline test */
void spu_register_stack_reset_entry(uint32_t entry, int image_id)
{
    if (s_stack_reset_count < 16) {
        s_stack_reset[s_stack_reset_count].entry = entry & SPU_LS_MASK;
        s_stack_reset[s_stack_reset_count++].image_id = image_id;
        g_spu_stack_reset_count = s_stack_reset_count;
    }
}

/* Also called before direct trampoline transfers: generated calls can enter
 * a stack-switching routine without going through spu_indirect_branch. */
void spu_check_stack_reset(spu_context* ctx, void (*fn)(spu_context*))
{
    /* Nothing registered is the common case: answer it before the
     * thread-local halt flag, which costs a call on every hop on macOS. */
    if (!s_stack_reset_count || !ctx->host_depth || ctx->policy_mode || !s_spu_halt_armed) return;
    for (unsigned i = 0; i < s_stack_reset_count; ++i)
        if (ctx->pc == s_stack_reset[i].entry &&
            fn == spu_lookup(ctx->pc, s_stack_reset[i].image_id)) {
            g_spu_trampoline_fn = 0;
            SPU_LONGJMP(s_spu_halt_env, 2);
        }
}

/* SPURS taskset TASK entries (see spu_context.resident_task). A taskset can hold
 * several tasks whose lifts share the SAME LS base -- the co-resident task-code
 * region -- so no LS address identifies which task owns it. The title registers
 * each task's ELF ENTRY point with the image id its functions were registered
 * under; spu_indirect_branch adopts that image the moment the policy branches
 * into the entry from outside the region. This is what lets each such task be
 * registered under its own real image id instead of the id-0 wildcard, so a
 * dormant co-resident task can no longer shadow the one the policy launched. */
#define SPU_TASK_ENTRY_MAX 16
static struct { uint32_t entry; int image_id; } s_task_entry[SPU_TASK_ENTRY_MAX];
static int s_task_entry_count = 0;
void spu_taskset_register_task_entry(uint32_t entry, int image_id)
{
    if (s_task_entry_count < SPU_TASK_ENTRY_MAX) {
        s_task_entry[s_task_entry_count].entry = entry & SPU_LS_MASK;
        s_task_entry[s_task_entry_count].image_id = image_id;
        s_task_entry_count++;
    }
}
static int spu_taskset_task_image(uint32_t entry)
{
    for (int i = 0; i < s_task_entry_count; i++)
        if (s_task_entry[i].entry == entry) return s_task_entry[i].image_id;
    return 0;
}
/* Resume PCs and even fresh entry PCs overlap between task ELFs. The
 * taskset policy's TaskInfo identifies which ELF it has actually loaded. */
static struct { uint32_t elf_ea; int image_id, policy_image_id; }
    s_task_elf[SPU_TASK_ENTRY_MAX];
static int s_task_elf_count;
void spu_taskset_register_task_elf(uint32_t elf_ea, int image_id, int policy_image_id)
{
    if (s_task_elf_count < SPU_TASK_ENTRY_MAX) {
        s_task_elf[s_task_elf_count].elf_ea = elf_ea & ~7u;
        s_task_elf[s_task_elf_count].image_id = image_id;
        s_task_elf[s_task_elf_count++].policy_image_id = policy_image_id;
    }
}
static int spu_taskset_resident_image(const spu_context* ctx)
{
    const uint8_t* p = ctx->ls + 0x2794; /* SpursTasksetContext.taskInfo.elf */
    uint32_t elf = (((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                    ((uint32_t)p[2] << 8) | p[3]) & ~7u;
    for (int i = 0; i < s_task_elf_count; ++i)
        if (s_task_elf[i].elf_ea == elf &&
            s_task_elf[i].policy_image_id == ctx->resident_ovl)
            return s_task_elf[i].image_id;
    return 0;
}
/* Lowest LS address of the shared task-code region: the taskset tasks all lift
 * at LS 0x3000 (below it is the SPURS kernel/policy/context, 0x290..0x2FFF). */
#define SPU_TASKSET_TASK_LO 0x3000u

/* Content-signature variant: FMOD COPIES codec overlays to the heap before
 * streaming them into the swap slot, so the source EA is unknowable ahead of
 * time -- match the first 16 bytes of the streamed content instead. */
void spu_overlay_register_sig(const uint8_t sig[16], int image_id)
{
    if (s_ovl_src_count < SPU_OVL_SRC_MAX) {
        s_ovl_src[s_ovl_src_count].src_ea = 0;
        s_ovl_src[s_ovl_src_count].image_id = image_id;
        memcpy(s_ovl_src[s_ovl_src_count].sig, sig, 16);
        s_ovl_src[s_ovl_src_count].has_sig = 1;
        { uint32_t h = spu_ovl_sig_hash(sig); __atomic_fetch_or(&g_ovl_sig_bloom[h >> 3], (uint8_t)(1u << (h & 7)), __ATOMIC_RELEASE); }
        __atomic_add_fetch(&g_ovl_sig_n, 1, __ATOMIC_RELEASE);
        s_ovl_src_count++;
    }
}

/* Both of the above: a bounded image (translated at the LS addresses the
 * title uses, see spu_overlay_register_region) whose source is a heap copy, so
 * it is recognised by its first 16 bytes. MultiStream stages its DSP plugins
 * this way -- one heap block, one GET into the mixer's plugin area. */
void spu_overlay_register_sig_region(const uint8_t sig[16], uint32_t span, int image_id)
{
    if (!span || span > SPU_LS_SIZE || s_ovl_src_count >= SPU_OVL_SRC_MAX) return;
    spu_overlay_register_sig(sig, image_id);
    s_ovl_src[s_ovl_src_count - 1].span = span;
}

/* Called from the MFC GET path after the copy: ls points at the JUST-COPIED
 * bytes. EA match first (exact, cheap), then content signature for sizeable
 * chunks (overlay bodies are >= 0x500 bytes). */
/* SPU_DUMP_OVL=<dir>[,<lsa>]: write out every distinct code body streamed into
 * LS <lsa> (default 0x5000, the job code buffer), so job bodies the title
 * keeps in its own data can be lifted. A body larger than one DMA arrives as
 * consecutive GETs (contiguous EA and LS); those are appended to the same
 * file. One file per source EA: ovl_<ea>.bin. */
static void spu_ovl_dump(spu_context* ctx, uint32_t ea, uint32_t lsa, const uint8_t* ls, uint32_t size)
{
    static int s_on = -1; static char s_dir[400]; static uint32_t s_lsa = 0x5000;
    if (s_on < 0) { const char* e = getenv("SPU_DUMP_OVL"); s_on = e && *e;
        if (s_on) { snprintf(s_dir, sizeof s_dir, "%s", e);
                    char* c = strchr(s_dir, ','); if (c) { *c = 0; s_lsa = (uint32_t)strtoul(c + 1, 0, 0); } } }
    if (!s_on) return;
    static uint32_t s_seen[256]; static unsigned s_n;
    static SPU_TLS uint32_t t_src, t_next_ea, t_next_lsa;
    FILE* f = NULL; char path[512];
    if (lsa == s_lsa && size >= 256) {
        for (unsigned i = 0; i < s_n; i++) if (s_seen[i] == ea) { t_src = 0; return; }
        if (s_n < 256) s_seen[s_n++] = ea;
        t_src = ea;
        snprintf(path, sizeof path, "%s/ovl_%08X.bin", s_dir, ea);
        f = fopen(path, "wb");
        fprintf(stderr, "[spu-ovl-dump] img=%d body ea=0x%08X size=%u -> %s\n", ctx->image_id, ea, size, path);
    } else if (t_src && ea == t_next_ea && lsa == t_next_lsa) {
        snprintf(path, sizeof path, "%s/ovl_%08X.bin", s_dir, t_src);
        f = fopen(path, "ab");
    } else { t_src = 0; return; }
    if (f) { fwrite(ls, 1, size, f); fclose(f); }
    t_next_ea = ea + size; t_next_lsa = lsa + size;
}

/* SPU_LOCKGUARD=<hex ea>: the word at that EA is a ticket lock (serving,next
 * halfwords). Log any SPU write that moves either halfword by anything other
 * than 0 or +1 -- a stale whole-line store rolling the lock back. `src` is the
 * new bytes for [ea, ea+size). */
void spu_lockguard_check(spu_context* ctx, uint32_t ea, const uint8_t* src, uint32_t size, const char* what)
{
    static int64_t s_g = -2;
    if (s_g == -2) { const char* e = getenv("SPU_LOCKGUARD"); s_g = e ? (int64_t)strtoul(e, 0, 16) : -1; }
    if (s_g < 0 || !vm_base) return;
    uint32_t g = (uint32_t)s_g;
    if (g < ea || g + 4 > ea + size) return;
    const uint8_t* o = vm_base + g; const uint8_t* n = src + (g - ea);
    unsigned os = (o[0] << 8) | o[1], on = (o[2] << 8) | o[3];
    unsigned ns = (n[0] << 8) | n[1], nn = (n[2] << 8) | n[3];
    int bad = !(ns == os || ns == ((os + 1) & 0xFFFF)) || !(nn == on || nn == ((on + 1) & 0xFFFF));
    if (bad) { static int c; if (c < 32 && ++c)
        fprintf(stderr, "[lockguard] %s img=%d pc=0x%05X ea=0x%08X size=%u: %04X/%04X -> %04X/%04X\n",
                what, ctx ? ctx->image_id : -1, ctx ? (unsigned)(ctx->pc & SPU_LS_MASK) : 0u,
                ea, size, os, on, ns, nn); }
}

void spu_overlay_note_get(spu_context* ctx, uint32_t ea, const uint8_t* ls, uint32_t size)
{
    R35C(R35_NOTE_GET);
    if (g_ovl_note_always == 1u) {   /* first call: the env-driven diagnostics decide whether the inline filter may skip us */
        const char* d = getenv("SPU_DUMP_OVL");
        g_ovl_note_always = ((d && *d) || spu_dyn_capture_on()) ? 2u : 0u;   /* 2 = a diagnostic needs every GET */
    }
    uint32_t lsa = (uint32_t)(ls - ctx->ls);
    spu_ovl_dump(ctx, ea, lsa, ls, size);
    spu_dynovl_note_get(ctx, ea, lsa, size);
    for (unsigned slot = 0; slot < 4; ++slot) {
        if (!ctx->resident_code[slot].image_id) continue;
        uint32_t base = ctx->resident_code[slot].lsa;
        uint32_t end = base + ctx->resident_code[slot].size;
        if (lsa < end && lsa + size > base &&
            (lsa < base || ea != ctx->resident_code[slot].source_ea + (lsa - base)))
            ctx->resident_code[slot].image_id = 0;
    }
    if (spu_ovl_bloom_on()) { uint32_t he = spu_ovl_ea_hash(ea), hs = spu_ovl_sig_hash(ls);
      if (!((g_ovl_ea_bloom[he >> 3] >> (he & 7)) & 1u) && !(size >= 512 && ((g_ovl_sig_bloom[hs >> 3] >> (hs & 7)) & 1u))) return; }
    R35C(R35_NOTE_GET_SCAN);
    for (int i = 0; i < s_ovl_src_count; i++) {
        const spu_ovl_src* o = &s_ovl_src[i];
        int hit = o->has_sig ? (size >= 512 && memcmp(ls, o->sig, 16) == 0)
                             : (o->src_ea == ea);
        if (hit) {
            if (o->span) {
                if (lsa + o->span > SPU_LS_SIZE) return;
                for (unsigned slot = 0; slot < 4; ++slot) {
                    if (ctx->resident_code[slot].image_id && ctx->resident_code[slot].lsa != lsa)
                        continue;
                    ctx->resident_code[slot].lsa = lsa;
                    ctx->resident_code[slot].size = o->span;
                    ctx->resident_code[slot].source_ea = ea;
                    ctx->resident_code[slot].image_id = o->image_id;
                    return;
                }
                fprintf(stderr, "[spu-ovl] no free resident code span for image %d\n", o->image_id);
                return;
            }
            if (ctx->resident_ovl != o->image_id) {
                ctx->resident_ovl = o->image_id;
                { static int _n = 0; if (_n < 32 && ++_n)
                    fprintf(stderr, "[spu-ovl] img=%d streamed overlay src=0x%08X "
                            "-> LS 0x%05X size=%u -> resident ovl image %d%s\n",
                            ctx->image_id, ea, (uint32_t)(ls - ctx->ls), size,
                            ctx->resident_ovl, o->has_sig ? " (sig match)" : ""); }
            }
            return;
        }
    }
}

/* Match a 16-byte module header against the registered content signatures.
 * Returns the overlay image id, or 0 if none. Used by the WWS job-code dispatch
 * path: the job manager streams a module into the code buffer via a DMA LIST
 * (per-element GETs the single-transfer note_get hook doesn't see as one >=512
 * chunk), so resident_ovl is set at branch-in time by matching the buffer head. */
int spu_overlay_match_sig(const uint8_t hdr[16])
{
    for (int i = 0; i < s_ovl_src_count; i++) {
        const spu_ovl_src* o = &s_ovl_src[i];
        if (o->has_sig && memcmp(hdr, o->sig, 16) == 0)
            return o->image_id;
    }
    return 0;
}

/* HLE of the taskset Policy Module's task-syscall entry (LS 0xA70). A SPURS task
 * (e.g. the cri_mpv task, image 22) reads syscallAddr from its SpursTasksetContext
 * (LS 0x27C4) and branches to it to perform a task syscall (EXIT/YIELD/WAIT/POLL).
 * The real kernel has the PM code resident at 0xA70; we don't, so we plant 0xA70 as
 * syscallAddr (in the cri dispatch) and INTERCEPT a branch to it here to HLE the
 * syscall. num = r3&0xF (0x10 bit = the "2" variant), args in r4. Adopted from the
 * JonathanDC64/ps3recomp fork (aaea4158) which uses this to run SPURS tasks clean. */
/* Write the event bits a PPU-side Set owed this wait object into guest memory,
 * at object+0x00 (events) and every object+0x30 slot (pendingRecv).
 *
 * Called from the WAIT_SIGNAL handler immediately before control returns to the
 * guest. Doing it at Set time loses a race: the guest reads object+0x30 the
 * instant it resumes, and a task that took the latched-wake path was never
 * parked for the Set to target. Here the guest demonstrably has not read yet.
 *
 * ponytail: fills all 16 slots rather than the one the guest picked -- the slot
 * index is derived from state we do not model, and a wrong slot delivers
 * nothing. Narrow it if a title ever cares which slot fired. */
static void spu_ef_deliver_owed(spu_context* ctx, uint32_t wobj)
{
    (void)ctx;
    static int s_od = -1;
    if (s_od < 0) s_od = getenv("SPURS_EF_OBJ_DELIVER") ? 1 : 0;
    if (!s_od || !wobj || !vm_base) return;
    extern uint16_t spu_ef_bits_take(uint32_t);
    uint16_t bits = spu_ef_bits_take(wobj);
    if (!bits) return;
    uint8_t* m = vm_base + wobj;
    m[0] = (uint8_t)(bits >> 8); m[1] = (uint8_t)bits;          /* +0x00 events */
    for (uint32_t s = 0; s < 16; s++) {                          /* +0x30 slots */
        m[0x30 + 2*s]     = (uint8_t)(bits >> 8);
        m[0x30 + 2*s + 1] = (uint8_t)bits;
    }
    { static int _n = 0; if (_n < 8 && ++_n)
        fprintf(stderr, "[spu] delivered owed bits=0x%04X to object 0x%08X \n",
                (unsigned)bits, wobj); }
}

void spu_spurs_taskset_syscall(spu_context* ctx)   /* non-static: also called by the pure interpreter (spu_interp.c) */
{
    uint32_t raw = ctx->gpr[3]._u32[0];
    uint32_t num = raw & 0x0F;
    /* SPU_SYSCALL_IMG=<n>: every syscall of image n, uncapped (the 24-line
     * budget is spent by FMOD's WAIT_SIGNAL loop long before a later task). */
    static int s_si = -2;
    if (s_si == -2) { const char* e = getenv("SPU_SYSCALL_IMG"); s_si = e ? atoi(e) : -1; }
    { static int _n = 0; if ((_n < 24 && ++_n) || ctx->image_id == s_si)
        fprintf(stderr, "[spu] SPURS taskset syscall num=%u (raw=0x%X args=0x%08X) image=%d link/r0=0x%05X wobj@2FDC=0x%02X%02X%02X%02X\n",
                num, raw, ctx->gpr[4]._u32[0], ctx->image_id, ctx->gpr[0]._u32[0] & SPU_LS_MASK,
                ctx->ls[0x2FDC], ctx->ls[0x2FDD], ctx->ls[0x2FDE], ctx->ls[0x2FDF]); }
    /* The task-API argument is passed in LOCAL STORE at 0x2FD0, not in r4: the
     * caller does shufb(arg,...) -> stqd 0x2FD0 and only then sets r3 = number
     * (see func_00026DE0 / func_000272AC in image 22). Logging r4 shows caller
     * leftovers and hides what the task actually asked for. */
    { static int s_sa = -1; if (s_sa < 0) s_sa = getenv("YDKJ_SYSCALL_ARG") ? 1 : 0;
      if (s_sa) { static int _n = 0; if (_n < 40 && ++_n) {
        const uint8_t* a = &ctx->ls[0x2FD0];
        fprintf(stderr, "[spu] syscall num=%u LS[0x2FD0]=%02X%02X%02X%02X %02X%02X%02X%02X"
                        " %02X%02X%02X%02X %02X%02X%02X%02X img=%d\n",
                num, a[0],a[1],a[2],a[3], a[4],a[5],a[6],a[7],
                a[8],a[9],a[10],a[11], a[12],a[13],a[14],a[15], ctx->image_id);
        fflush(stderr); } } }
    /* NOTE (YDKJ cri_mpv): the cri task's BOOTSTRAP (func_00003040) calls the
     * task-API syscall and EXPECTS IT TO RETURN, then branches to the real task
     * entry (0x3050). Halting on num=0 here kills the task at bootstrap before it
     * runs. So for image 22 we DON'T halt on num=0 -- we return so the bootstrap
     * continues to the decode entry. (A genuine end-of-task EXIT would re-enter and
     * spin; if that happens, gate a real halt after the task has done work.)
     * For non-cri images keep the fork's EXIT=halt semantics. Env YDKJ_CRI_EXIT_HALT
     * forces the old halt behaviour for comparison. */
    /* The cri bootstrap calls EXIT once and EXPECTS IT TO RETURN, then branches to
     * the real task entry -- so image 22 cannot halt on the first one. But a task
     * that has since done work and calls EXIT again means it: returning there makes
     * it re-enter and spin (observed: 6x num=0 from link 0x26E18 once the task is
     * woken). Honour the first call, halt on the rest. Each task runs on its own
     * host thread, so a thread-local count is per task. */
    static _Thread_local int s_exit_seen = 0;
    static _Thread_local unsigned s_exit_serial;
    if (s_exit_serial != g_spu_task_serial) { s_exit_serial = g_spu_task_serial; s_exit_seen = 0; }
    if (num == 0 && ctx->image_id == spu_cri_image() && !getenv("YDKJ_CRI_EXIT_HALT")) {
        if (s_exit_seen++ == 0) { ctx->gpr[3]._u32[0] = 0; return; }   /* bootstrap */
        { static int _n = 0; if (_n < 8 && ++_n)
            fprintf(stderr, "[spu] cri task EXIT #%d -- halting (was spinning)\n",
                    s_exit_seen); }
        ctx->status = SPU_STATUS_STOPPED_BY_STOP;
        spu_halt(ctx);
        return;
    }
    if (num == 0 && (ctx->image_id != spu_cri_image() || getenv("YDKJ_CRI_EXIT_HALT"))) {
        ctx->status = SPU_STATUS_STOPPED_BY_STOP;
        spu_halt(ctx);          /* longjmp out to spu_run_with_halt; post-run writes exit code */
        return;
    }
    /* WAIT_SIGNAL(2): REAL semantics (RPCS3 spursTasksetProcessSyscall) --
     * consume the task's bit in the guest taskset's `signalled` bitset, or
     * SLEEP this task's host thread until _cellSpursSendSignal /
     * cellSpursEventFlagSet delivers one (the FMOD mixer's flag-A wait path:
     * the task registers its wait slot in the flag struct with atomics, then
     * syscalls WAIT_SIGNAL; the PPU-side Set satisfies the slot and signals).
     * Returning "success" here without waiting made the task spin on empty
     * work state forever -- the LBP boot deadlock. taskset/taskId come from
     * the SpursTasksetContext this runtime planted at LS 0x2700. */
    if (num == 2) {
        uint32_t ts  = ((uint32_t)ctx->ls[0x27BC] << 24) | ((uint32_t)ctx->ls[0x27BD] << 16) |
                       ((uint32_t)ctx->ls[0x27BE] << 8)  |  (uint32_t)ctx->ls[0x27BF];
        uint32_t tid = ((uint32_t)ctx->ls[0x27D4] << 24) | ((uint32_t)ctx->ls[0x27D5] << 16) |
                       ((uint32_t)ctx->ls[0x27D6] << 8)  |  (uint32_t)ctx->ls[0x27D7];
        /* The guest names its wait OBJECT in the task-API argument at LS 0x2FD0
         * (low nibble is flags). It reads its received event bits from
         * object+0x30, so this is both who to wake and where to deliver. */
        uint32_t wobj = (((uint32_t)ctx->ls[0x2FDC] << 24) |
                         ((uint32_t)ctx->ls[0x2FDD] << 16) |
                         ((uint32_t)ctx->ls[0x2FDE] << 8)  |
                          (uint32_t)ctx->ls[0x2FDF]) & ~0xFu;
        /* SPURS_EF_SPU_REPLY=1 -- DIAGNOSTIC BISECT, NOT A FIX.
         *
         * The guest task is supposed to set its SPU->PPU flag (object+0x100)
         * when a work cycle completes; it never does, so the PPU blocks on that
         * flag forever and can never enqueue the next unit of work. That is a
         * cycle: we cannot see what the PPU would do next without breaking it.
         *
         * On RE-ENTRY to WAIT_SIGNAL for the same object the previous cycle has
         * finished, so set the flag on the task`s behalf and watch where the PPU
         * goes. This FABRICATES guest state -- the repo`s own history says faked
         * success is the most expensive kind of bug (see the unresolved-NID note
         * in ppu_hle.cpp), so it stays opt-in and must never become the default.
         * If the PPU advances, the answer is what the task must do to earn it. */
        { static int s_rp = -1;
          if (s_rp < 0) { const char* e = getenv("SPURS_EF_SPU_REPLY");
                          s_rp = e ? atoi(e) : 0; }   /* 2 = reply on EVERY entry */
          if (s_rp && wobj) {
              static uint32_t seen[8]; static int seen_n = 0;
              int again = 0;
              for (int i = 0; i < seen_n; i++) if (seen[i] == wobj) { again = 1; break; }
              if (s_rp >= 2) again = 1;   /* force: answer "would the PPU move at all?" */
              if (!again) { if (seen_n < 8) seen[seen_n++] = wobj; }
              else {
                  extern void spurs_ef_set_from_spu(uint32_t, uint16_t);
                  spurs_ef_set_from_spu(wobj + 0x100u, 1);
                  static int _n = 0;
                  if (_n < 8 && ++_n)
                      fprintf(stderr, "[spu] SPU-REPLY: set flag 0x%08X for object "
                                      "0x%08X (cycle complete)\n", wobj + 0x100u, wobj);
              }
          } }

        if (ts) {
            extern int spu_taskset_wait_signal(uint32_t, uint32_t);
            /* Park = OS-level wait on this SPU's host thread. Under the lockstep
             * gate the thread HOLDS the global run token here; parking without
             * releasing it starves every other lifted SPU (observed: FMOD task 1
             * parks in WAIT_SIGNAL holding the token -> the mixer never runs
             * again -> the PPU audio pump blocks forever on flag 0x94F600). */
            /* Publish that this task is parked so a PPU-side event-flag Set can
             * reach it even though it registered no wait slot in the flag. */
            extern void spu_taskset_parked_add(uint32_t, uint32_t, uint32_t);
            extern void spu_taskset_parked_del(uint32_t, uint32_t);
            extern int spu_taskset_consume_wake(uint32_t);
            if (spu_taskset_consume_wake(ts)) {   /* a Set beat us to the park */
                spu_ef_deliver_owed(ctx, wobj);
                ctx->gpr[3]._u32[0] = 0;
                return;
            }
            yz_lockstep_block_begin(ctx);
            spu_taskset_parked_add(ts, tid, wobj);
            spu_taskset_wait_signal(ts, tid);
            spu_taskset_parked_del(ts, tid);
            yz_lockstep_block_end(ctx);
            spu_ef_deliver_owed(ctx, wobj);
        }
        ctx->gpr[3]._u32[0] = 0;
        return;
    }
    /* EXIT(0, cri bootstrap)/YIELD(1)/POLL(3)/RECV_WKL_FLAG(4):
     * report success and resume (return -> lifted caller continues at its link). */
    ctx->gpr[3]._u32[0] = 0;
}

/* ---------------------------------------------------------------------------
 * Micro-interpreter for RUNTIME-GENERATED stub code (SMC).
 *
 * The WWS jobmanager's interrupt handler GENERATES a register save/restore
 * stub above the static image (LS 0x3FEC0: a chain of stqd/lqd plus a `bi`
 * terminator, written at interrupt entry) and calls it. No lift can exist for
 * bytes that only come into being at runtime, so when the indirect dispatch
 * finds no registered function we interpret the LIVE LS bytes directly for
 * the small instruction set such stubs use, and exit back into lifted code at
 * the first branch (trampoline re-dispatch). Returns 1 if it executed to a
 * branch, 0 on an unknown opcode (caller falls through to BRANCH-TO-0). */
static int spu_smc_microstep(spu_context* ctx)
{
    const uint32_t entry_pc = ctx->pc & SPU_LS_MASK & ~3u;
    uint32_t pc = ctx->pc & SPU_LS_MASK & ~3u;
    for (int steps = 0; steps < 4096; steps++) {
        uint32_t w = ((uint32_t)ctx->ls[pc] << 24) | ((uint32_t)ctx->ls[pc+1] << 16) |
                     ((uint32_t)ctx->ls[pc+2] << 8) | ctx->ls[pc+3];
        uint32_t op11 = w >> 21, op9 = w >> 23, op8 = w >> 24, op7 = w >> 25;
        uint32_t rt = w & 0x7F, ra = (w >> 7) & 0x7F, rb = (w >> 14) & 0x7F;
        int32_t  i10 = (int32_t)(w << 8) >> 22;               /* bits 14-23 sext */
        int32_t  i16 = (int32_t)(int16_t)((w >> 7) & 0xFFFF);
        (void)rb;

        if (op11 == 0x1A8 || op11 == 0x1A9 || op11 == 0x1AA || op11 == 0x1AB) {
            /* bi / bisl / iret / bisled (+E/D interrupt bits) */
            if (op11 == 0x1AB &&                       /* bisled: only on event */
                (ctx->event_status & ctx->event_mask) == 0) { pc += 4; continue; }
            if (w & 0x40000) ctx->int_enable = 1;
            else if (w & 0x80000) ctx->int_enable = 0;
            uint32_t tgt = (op11 == 0x1AA) ? ctx->srr0
                                           : ctx->gpr[ra]._u32[0];
            if (op11 == 0x1A9 || op11 == 0x1AB)
                ctx->gpr[rt] = spu_splat_u32(pc + 4);
            ctx->pc = tgt & SPU_LS_MASK & ~3u;
            g_spu_trampoline_fn = spu_indirect_branch;
            return 1;
        }
        if (op11 >= 0x128 && op11 <= 0x12B) {
            /* biz / binz / bihz / bihnz rt,ra (+E/D bits, taken-only) */
            uint32_t cv = ctx->gpr[rt]._u32[0];
            int taken;
            switch (op11) {
            case 0x128: taken = (cv == 0); break;                     /* biz  */
            case 0x129: taken = (cv != 0); break;                     /* binz */
            case 0x12A: taken = ((cv & 0xFFFF) == 0); break;          /* bihz */
            default:    taken = ((cv & 0xFFFF) != 0); break;          /* bihnz*/
            }
            if (!taken) { pc += 4; continue; }
            if (w & 0x40000) ctx->int_enable = 1;
            else if (w & 0x80000) ctx->int_enable = 0;
            ctx->pc = ctx->gpr[ra]._u32[0] & SPU_LS_MASK & ~3u;
            g_spu_trampoline_fn = spu_indirect_branch;
            return 1;
        }
        if (op9 == 0x064 || op9 == 0x060 || op9 == 0x066 || op9 == 0x062) {
            /* br / bra / brsl / brasl */
            uint32_t tgt = (op9 == 0x060 || op9 == 0x062)
                         ? ((uint32_t)i16 << 2)
                         : (pc + ((uint32_t)i16 << 2));
            if (op9 == 0x066 || op9 == 0x062)
                ctx->gpr[rt] = spu_splat_u32(pc + 4);
            ctx->pc = tgt & SPU_LS_MASK & ~3u;
            g_spu_trampoline_fn = spu_indirect_branch;
            return 1;
        }
        if (op9 == 0x040 || op9 == 0x042 || op9 == 0x044 || op9 == 0x046) {
            /* brz / brnz / brhz / brhnz rt,label (relative) */
            uint32_t cv = ctx->gpr[rt]._u32[0];
            int taken;
            switch (op9) {
            case 0x040: taken = (cv == 0); break;
            case 0x042: taken = (cv != 0); break;
            case 0x044: taken = ((cv & 0xFFFF) == 0); break;
            default:    taken = ((cv & 0xFFFF) != 0); break;
            }
            if (!taken) { pc += 4; continue; }
            ctx->pc = (pc + ((uint32_t)i16 << 2)) & SPU_LS_MASK & ~3u;
            g_spu_trampoline_fn = spu_indirect_branch;
            return 1;
        }
        if (op8 == 0x24) {                                    /* stqd rt,i10(ra) */
            uint32_t a = (ctx->gpr[ra]._u32[0] + ((uint32_t)i10 << 4)) & SPU_LS_MASK & ~15u;
            spu_ls_write128(ctx, a, ctx->gpr[rt]);
            pc += 4; continue;
        }
        if (op8 == 0x34) {                                    /* lqd rt,i10(ra) */
            uint32_t a = (ctx->gpr[ra]._u32[0] + ((uint32_t)i10 << 4)) & SPU_LS_MASK & ~15u;
            ctx->gpr[rt] = spu_ls_read128(ctx, a);
            pc += 4; continue;
        }
        if (op9 == 0x041 || op9 == 0x061) {                   /* stqa / lqa (abs) */
            /* LSA = I16 << 2 (SPU ISA, RI16 form), as in spu_interp.c and
             * the lifter. */
            uint32_t a = (((w >> 7) & 0xFFFF) << 2) & SPU_LS_MASK & ~15u;
            if (op9 == 0x041) spu_ls_write128(ctx, a, ctx->gpr[rt]);
            else              ctx->gpr[rt] = spu_ls_read128(ctx, a);
            pc += 4; continue;
        }
        if (op9 == 0x047 || op9 == 0x067) {                   /* stqr / lqr (rel) */
            uint32_t a = (pc + ((uint32_t)i16 << 2)) & SPU_LS_MASK & ~15u;
            if (op9 == 0x047) spu_ls_write128(ctx, a, ctx->gpr[rt]);
            else              ctx->gpr[rt] = spu_ls_read128(ctx, a);
            pc += 4; continue;
        }
        if (op8 == 0x44) {                                    /* xori rt,ra,i10 */
            /* Runtime-generated job stubs toggle their own instruction words.
             * Use the signed RI10 immediate, just like statically lifted xori. */
            ctx->gpr[rt] = spu_xori(ctx->gpr[ra], i10);
            pc += 4; continue;
        }
        if (op8 == 0x1C) {                                    /* ai rt,ra,i10 */
            u128 r = ctx->gpr[ra];
            for (int k = 0; k < 4; k++) r._u32[k] += (uint32_t)i10;
            ctx->gpr[rt] = r; pc += 4; continue;
        }
        if (op11 == 0x040 || op11 == 0x0C0) {                 /* sf / a (word) */
            u128 x = ctx->gpr[ra], y = ctx->gpr[rb], r;
            for (int k = 0; k < 4; k++)
                r._u32[k] = (op11 == 0x040) ? y._u32[k] - x._u32[k]
                                            : x._u32[k] + y._u32[k];
            ctx->gpr[rt] = r; pc += 4; continue;
        }
        if (op9 == 0x081) { ctx->gpr[rt] = spu_splat_u32((uint32_t)i16); pc += 4; continue; }  /* il  */
        if (op9 == 0x082) { ctx->gpr[rt] = spu_splat_u32(((w >> 7) & 0xFFFF) << 16); pc += 4; continue; } /* ilhu */
        if (op9 == 0x0C1) { u128 r = ctx->gpr[rt];            /* iohl */
            for (int k = 0; k < 4; k++) r._u32[k] |= (w >> 7) & 0xFFFF;
            ctx->gpr[rt] = r; pc += 4; continue; }
        if (op7 == 0x21)  { ctx->gpr[rt] = spu_splat_u32((w >> 7) & 0x3FFFF); pc += 4; continue; } /* ila */
        if (op11 == 0x201 || op11 == 0x001) { pc += 4; continue; }  /* nop/lnop */
        if (op11 == 0x002 || op11 == 0x003) { pc += 4; continue; }  /* sync/dsync */
        /* Branch hints, all no-ops for execution: hbr is the 11-bit-opcode RR
         * form (0x1AC), hbra and hbrr are the 7-bit-opcode RI18 form (0x08,
         * 0x09) -- the same op7 group as ila (0x21) above, NOT op9. Testing
         * op9 here never matched either (hbrr 0x12033296 has op9 0x24), so a
         * runtime-generated stub carrying a branch hint decoded to UNKNOWN,
         * the microstep bailed, and the SPU fell into branch-to-0. */
        /* sagemono reached the same fix independently in #166, from a different
         * witness (hbrr 0x1200048C rather than 0x12033296). Same line, same
         * diagnosis, arrived at separately -- which is about as good as
         * corroboration gets for a decode bug. */
        if (op11 == 0x1AC || op7 == 0x08 || op7 == 0x09) { pc += 4; continue; } /* hbr/hbra/hbrr */

        /* Hand off to the real interpreter rather than giving up.
         *
         * This microstepper knows a dozen opcodes -- enough for the
         * runtime-generated stubs it was written for -- and spu_interp.c is a
         * full, selftested SPU interpreter that was already sitting here
         * unused on this path. Bailing on the first ordinary instruction is
         * strictly worse than running it.
         *
         * It matters for overlay code: a SPURS job policy module DMAs its job
         * bodies into local store and branches to them, so there is no lifted
         * body to find and the fallback IS the execution path. Guitar Hero
         * III`s job kernel dies here on an `ilh` at LS 0x6638 -- a perfectly
         * ordinary instruction the interpreter handles -- and with it the
         * decompression job it was running never completes. */
        { static int _n = 0;
          if (_n < 8 && ++_n)
              fprintf(stderr, "[spu-smc] microstep img=%d pc=0x%05X word 0x%08X not in the"
                      " microstep set -- handing to the interpreter (steps=%d)\n",
                      ctx->image_id, pc, w, steps); }
        ctx->pc = pc;
        /* Stop at the drain's return point too: it sits mid-function, so it is
         * no lifted entry, and a job's return would otherwise run on through
         * the caller's code (GH3's song-freeze; see drain_ret_pc). */
        const uint32_t return_pc = ctx->drain_ret_pc;
        /* How much runs interpreted: the interpreter is 50-100x slower
         * than lifted code, so a job that lands here often is a stall.
         * Reported every 64 hand-offs with the image and the time. */
        static long long s_n, s_steps; static double s_ms;
        extern uint64_t g_spu_interp_steps;
        /* SPU_SMC_DUMP=<dir>: the whole local store, once per (image, 4 KB
         * page of the entry pc), at a hand-off -- the job bodies that run
         * interpreted, so they can be identified and lifted. */
        { static const char* dd = (const char*)1;
          if (dd == (const char*)1) dd = getenv("SPU_SMC_DUMP");
          if (dd) {
              static uint32_t seen[256]; static int nseen;
              const uint32_t key = ((uint32_t)(ctx->image_id & 0xFFF) << 20) | (pc & 0x3F000u);
              int dup = 0;
              for (int k = 0; k < nseen; k++) if (seen[k] == key) { dup = 1; break; }
              if (!dup && nseen < 256) {
                  seen[nseen++] = key;
                  char path[512];
                  snprintf(path, sizeof path, "%s/ls_img%d_pc%05X.bin", dd, ctx->image_id, pc);
                  FILE* f = fopen(path, "wb");
                  if (f) { fwrite(ctx->ls, 1, SPU_LS_SIZE, f); fclose(f);
                           fprintf(stderr, "[spu-smc] dumped local store -> %s\n", path); }
              }
          } }
        spu_dynovl_capture(ctx, entry_pc);
        const uint64_t t0 = spu_host_ns();
        uint32_t result = spu_interp_run_until(ctx, pc, return_pc);
        const double ms = (double)(spu_host_ns() - t0) / 1e6;
        s_ms += ms; s_steps += (long long)g_spu_interp_steps;
        { static int s_v = -1; if (s_v < 0) { const char* e = getenv("SPU_SMC_LOG"); s_v = e && e[0] == '1'; }   /* SPU_SMC_LOG=1: the old every-64th-hand-off line (it was half of a normal run's log) */
          ++s_n; if (!(s_v ? ((s_n % 64) == 1 || ms > 2.0) : (s_n <= 4 || ms > 10.0))) goto smc_noprint; }
            fprintf(stderr, "[spu-smc] interpreter hand-offs: %lld, %.1f ms, %lld steps in all; this one img=%d from 0x%05X: %.2f ms, %llu steps\n",
                    s_n, s_ms, s_steps, ctx->image_id, pc, ms, (unsigned long long)g_spu_interp_steps);
        smc_noprint:;
        /* A live interpreter rejoin is a transfer, not a host return. Nested
         * drains can repair an empty trampoline, but the top-level driver
         * otherwise ends the run before executing the lifted target.
         * An already-owning drain must retain its own host continuation;
         * stopped states and diagnostic exits keep their existing behavior. */
        if (ctx->image_id >= 0 && (ctx->status & SPU_STATUS_RUNNING) &&
            (!return_pc || (ctx->pc & SPU_LS_MASK) != return_pc) &&
            spu_lifted_lookup(ctx, ctx->pc)) {
            g_spu_trampoline_fn = spu_indirect_branch;
            return 1;
        }
        return result ? 1 : 0;
    }
    { static int _n = 0; if (_n < 4 && ++_n)
        fprintf(stderr, "[spu-smc] microstep img=%d runaway (4096 steps from 0x%05X)\n",
                ctx->image_id, ctx->pc & SPU_LS_MASK); }
    return 0;
}

/* ---- SPU_IBCACHE (see spu_indirect_branch) ------------------------------------ */
typedef struct { uint32_t pc, img, ovl, task, pm, nreg, ndyn, ctx_lo, ctx_hi; uint32_t rc[4][3]; } spu_ib_key;
typedef struct { spu_ib_key key; spu_fn fn; uint32_t gen; int owner; uint32_t rcz; } spu_ib_ent;
static SPU_TLS spu_ib_ent t_ib[2048];
static int spu_ib_mode(void)
{
    static int s = -1;
    if (s < 0) {
        const char* e = getenv("SPU_IBCACHE");
        s = e ? atoi(e) : 1;
        if (getenv("SPU_JOBTRACE") || getenv("SPU_JOBEXEC") || getenv("LBP_HLE_JOBDONE")) s = 0;
    }
    return s;
}
static inline void spu_ib_make_key(const spu_context* ctx, spu_ib_key* k)
{
    k->pc = ctx->pc & SPU_LS_MASK; k->img = (uint32_t)ctx->image_id; k->ovl = (uint32_t)ctx->resident_ovl;
    k->task = (uint32_t)ctx->resident_task; k->pm = (uint32_t)ctx->policy_mode;
    k->nreg = s_registry_count; k->ndyn = (uint32_t)s_dyn_n;
    /* the dynovl verification state belongs to one context (find() resets it on a new one) */
    k->ctx_lo = (uint32_t)(uintptr_t)ctx; k->ctx_hi = (uint32_t)((uint64_t)(uintptr_t)ctx >> 32);
    for (unsigned i = 0; i < 4; i++) {
        if (ctx->resident_code[i].image_id) {
            k->rc[i][0] = (uint32_t)ctx->resident_code[i].image_id;
            k->rc[i][1] = ctx->resident_code[i].lsa; k->rc[i][2] = ctx->resident_code[i].size;
        } else { k->rc[i][0] = k->rc[i][1] = k->rc[i][2] = 0; }
    }
}

/* In-place hit test: the cached key against the context's current fields (same fields, same
 * values as spu_ib_make_key + spu_ib_key_eq, without building the 84-byte key). */
static inline __attribute__((always_inline)) int spu_ib_key_match(const spu_context* ctx, const spu_ib_key* k)
{
    uint32_t d = (k->pc ^ (uint32_t)(ctx->pc & SPU_LS_MASK)) | (k->img ^ (uint32_t)ctx->image_id) |
                 (k->ovl ^ (uint32_t)ctx->resident_ovl) | (k->task ^ (uint32_t)ctx->resident_task) |
                 (k->pm ^ (uint32_t)ctx->policy_mode) | (k->nreg ^ (uint32_t)s_registry_count) |
                 (k->ndyn ^ (uint32_t)s_dyn_n) | (k->ctx_lo ^ (uint32_t)(uintptr_t)ctx) |
                 (k->ctx_hi ^ (uint32_t)((uint64_t)(uintptr_t)ctx >> 32));
    if (d) return 0;
    for (unsigned i = 0; i < 4; i++) {
        const uint32_t im = (uint32_t)ctx->resident_code[i].image_id;
        if (im) {
            if ((k->rc[i][0] ^ im) | (k->rc[i][1] ^ ctx->resident_code[i].lsa) | (k->rc[i][2] ^ ctx->resident_code[i].size)) return 0;
        } else if (k->rc[i][0] | k->rc[i][1] | k->rc[i][2]) return 0;
    }
    return 1;
}

/* Key equality without the CRT memcmp call (84 bytes: it was a call into ucrt on every indirect branch,
 * ~4% of a wid0 run). SPU_IBKEY_FAST=0 restores the memcmp. */
static inline int spu_ib_key_eq(const spu_ib_key* a, const spu_ib_key* b)
{
    static int s_f = -1; if (s_f < 0) { const char* e = getenv("SPU_IBKEY_FAST"); s_f = !(e && e[0] == '0'); }
    if (!s_f) return memcmp(a, b, sizeof *a) == 0;
    const uint32_t* x = (const uint32_t*)a; const uint32_t* y = (const uint32_t*)b;
    uint32_t d = 0;
    for (unsigned i = 0; i < sizeof(spu_ib_key) / 4; i++) d |= x[i] ^ y[i];
    return d == 0;
}


/* ---- SPU_IB_FAST: frame-less front end of spu_indirect_branch ------------------------------
 * The full resolver is one huge function: every call paid its prologue (8 pushes, xmm6/7 spills,
 * 0x428-byte frame, /GS cookie) and the matching epilogue even on an IB-cache hit. This front end
 * does ONLY the cache hit (same table, same key, same generation) as a leaf (no calls, no frame)
 * with a musttail jump into the lifted function; everything else (misses, special pcs, resident
 * code slots, registered stack resets, debug envs) tail-calls the unchanged resolver, which also
 * fills the cache. A 2-way set (slot h and h^1) lowers the direct-mapped collision misses.
 * SPU_IB_FAST=1 enables it (default 0); SPU_IB_STATS=1 counts (hits per thread, flushed at misses). */
void spu_indirect_branch_full(spu_context* ctx);
#if defined(__clang__)
#  define SPU_IB_TAIL(call) do { __attribute__((musttail)) return call; } while (0)
#else
#  define SPU_IB_TAIL(call) do { call; return; } while (0)
#endif
static int s_ib_fast = -1;
static int s_ibc = 1;
static int s_ib_stats = 0;
static volatile long s_ibs_hit, s_ibs_cold, s_ibs_special;
static SPU_TLS unsigned long t_ibs_hit;
static void spu_ib_stats_print(const char* why)
{
    fprintf(stderr, "[ib-fast] %s: hits(flushed)=%ld cold(miss+special)=%ld special=%ld\n", why, s_ibs_hit, s_ibs_cold, s_ibs_special);
}
static void spu_ib_fast_init(void)
{
    const char* e = getenv("SPU_IB_FAST");
    int on = e ? atoi(e) : 1;   /* default ON 26/10 (confirmed 6+6 runs: wid7 -4%, wid0 -3%) */
    if (on != 1 && on != 2) on = 0;
    if (on && (spu_ib_mode() != 1 || getenv("SPU_IBCOV") || getenv("SPU_POLLTRACE") || getenv("YDKJ_POLTRACE"))) on = 0;
    { const char* c = getenv("SPU_IBC"); s_ibc = !(c && c[0] == '0'); }
    const char* st = getenv("SPU_IB_STATS");
    s_ib_stats = (on == 1 && st && atoi(st)) ? 1 : 0;
    s_ib_fast = on;
    if (on) fprintf(stderr, "[ib-fast] SPU_IB_FAST=%d (%s, 2-way cache)%s\n", on, on == 1 ? "frame-less IB hit path" : "VERIFY: full resolver + comparison with the fast probe", s_ib_stats ? " +stats" : "");
}
void spu_indirect_branch(spu_context* ctx);
/* collision net (spu_collnet_guard, below): the broad-phase dispatch `bisl` at 0x28A8..0x28DC of image 110 has just run
 * (link 0x28E0, r3 = the LS record 0x23000, r16 = the target it loaded from the table). Any other indirect branch
 * sees at least one of these differ, so it keeps the cached fast path. */
/* crash35: the two dispatches of img 110 are
 *   1st (bisl at 0x28A8..0x28DC, link 0x28E0): target r16 = LS[0x6D20 + 4*idx]
 *   2nd (bisl at 0x2D98.., link 0x2DC0):        target r82 = LS[0x6D30 + 4*idx]   (same idx, table shifted by 4 entries)
 * and the compare is on the MASKED register: a bisl masks the target to 256 KB, the lifted code keeps the raw word, so a wild table
 * word with high bits (r16 = 0x00xx136BE-like, read from LS beyond the 11-word table) used to make r16 != pc and the net was skipped. */
#define SPU_COLLNET_CALL(ctx, pc) \
    (__builtin_expect(((ctx)->gpr[0]._u32[0] == 0x28E0u && (ctx)->gpr[3]._u32[0] == 0x23000u && ((ctx)->gpr[16]._u32[0] & SPU_LS_MASK) == (pc)) || \
                      ((ctx)->gpr[0]._u32[0] == 0x2DC0u && (ctx)->gpr[3]._u32[0] == 0x23000u && ((ctx)->gpr[82]._u32[0] & SPU_LS_MASK) == (pc)), 0))
static int spu_collnet_guard(spu_context* ctx);
#define SPU_IB_SPECIAL(ctx, pc) \
    (__builtin_expect((ctx)->irq_saved | s_task_elf_count | s_task_entry_count | g_spu_stack_reset_count | \
                      (ctx)->resident_code[0].image_id | (ctx)->resident_code[1].image_id | \
                      (ctx)->resident_code[2].image_id | (ctx)->resident_code[3].image_id, 0) || \
     (pc) == SPURS_TASKSET_PM_SYSCALL_LS || (pc) == SPU_JOB_RETURN_LS || (ctx)->image_id == SPU_RESOLVER_ONLY_IMAGE_ID || \
     ((ctx)->policy_mode && ((pc) == SPURS_PM_EXIT_TO_KERNEL_LS || (pc) == SPURS_PM_SELECT_WORKLOAD_LS)))
static SPU_TLS int t_ib_vfy;          /* SPU_IB_FAST=2: the next full resolution is compared with t_ib_expect */
static SPU_TLS spu_fn t_ib_expect;
static spu_fn spu_ib_peek(spu_context* ctx);
__attribute__((noinline)) static void spu_ib_cold(spu_context* ctx)
{
    R35C(R35_IB_COLD);
    if (s_ib_fast < 0) spu_ib_fast_init();
    if (s_ib_fast == 2) { t_ib_expect = spu_ib_peek(ctx); t_ib_vfy = 1; }
    if (s_ib_fast == 1) {
        if (s_ib_stats) {
            uint32_t pc = ctx->pc & SPU_LS_MASK;
            if (SPU_IB_SPECIAL(ctx, pc))
                __atomic_add_fetch(&s_ibs_special, 1, __ATOMIC_RELAXED);
            __atomic_add_fetch(&s_ibs_hit, (long)t_ibs_hit, __ATOMIC_RELAXED); t_ibs_hit = 0;
            long k = __atomic_add_fetch(&s_ibs_cold, 1, __ATOMIC_RELAXED);
            if ((k & 0x3FFFF) == 0) spu_ib_stats_print("periodic");
        }
    }
    SPU_IB_TAIL(spu_indirect_branch_full(ctx));
}
/* xor with an optimisation barrier: keeps clang's SLP vectoriser from gathering the 9 scattered
 * context fields into vectors (vpinsrd chains cost more than the scalar xor/or). */
#define SPU_IBX(a, b) ({ uint32_t _t = (uint32_t)(a) ^ (uint32_t)(b); __asm__("" : "+r"(_t)); _t; })
static inline __attribute__((always_inline)) int spu_ib_ent_ok(const spu_context* ctx, uint32_t pc, const spu_ib_ent* e, uint32_t gen)
{
    if (!e->fn || e->gen != gen || !e->rcz) return 0;
    const spu_ib_key* k = &e->key;
    uint32_t d = SPU_IBX(k->pc, pc) | SPU_IBX(k->img, ctx->image_id) | SPU_IBX(k->ovl, ctx->resident_ovl) |
                 SPU_IBX(k->task, ctx->resident_task) | SPU_IBX(k->pm, ctx->policy_mode) |
                 SPU_IBX(k->nreg, s_registry_count) | SPU_IBX(k->ndyn, s_dyn_n) |
                 SPU_IBX(k->ctx_lo, (uint32_t)(uintptr_t)ctx) | SPU_IBX(k->ctx_hi, (uint32_t)((uint64_t)(uintptr_t)ctx >> 32));
    return d == 0;
}
/* ---- SPU_IBC: compact front cache (default on; SPU_IBC=0 off) ------------------------------------
 * The 112-byte entries of t_ib span two or three cache lines each and the 84-byte key is compared
 * field by field: the hit test (spu_ib_ent_ok, ~5% of wid9, 11-13% of wid0/wid7) is mostly L1 misses on
 * the entries. A hit here compares the nine state fields (image, overlay, task, policy mode,
 * registry size, dynovl count, context, invalidation generation) with ONE per-thread memo of the
 * state last seen (one hot line) and then only (pc, state id) of a 16-byte entry. Entries are filled
 * from verified t_ib hits, so a compact hit means "t_ib would have hit with the same function".
 * A state not in the 8-slot table, or a generation bump, gets a NEW id: old entries die. */
typedef struct { spu_fn fn; uint32_t pc, sid; } spu_ibc_ent;
typedef struct { uint32_t img, ovl, task, pm, nreg, ndyn, ctx_lo, ctx_hi, gen, sid; } spu_ibc_st;
static SPU_TLS spu_ibc_ent t_ibc[2048];
static SPU_TLS spu_ibc_st t_ibst;          /* the memo: the state last used */
static SPU_TLS spu_ibc_st t_ibst_tab[8];   /* recent states (sid 0 = empty) */
static SPU_TLS uint32_t t_ibst_n, t_ibst_rr;
static volatile long s_ibc_hit, s_ibc_bad;
static __attribute__((noinline)) void spu_ibc_refresh(const spu_context* ctx)
{
    spu_ibc_st c;
    c.img = (uint32_t)ctx->image_id; c.ovl = (uint32_t)ctx->resident_ovl; c.task = (uint32_t)ctx->resident_task;
    c.pm = (uint32_t)ctx->policy_mode; c.nreg = s_registry_count; c.ndyn = (uint32_t)s_dyn_n;
    c.ctx_lo = (uint32_t)(uintptr_t)ctx; c.ctx_hi = (uint32_t)((uint64_t)(uintptr_t)ctx >> 32); c.gen = t_ib_gen; c.sid = 0;
    for (unsigned i = 0; i < 8; i++) {
        const spu_ibc_st* t = &t_ibst_tab[i];
        if (t->sid && !((t->img ^ c.img) | (t->ovl ^ c.ovl) | (t->task ^ c.task) | (t->pm ^ c.pm) | (t->nreg ^ c.nreg) |
                        (t->ndyn ^ c.ndyn) | (t->ctx_lo ^ c.ctx_lo) | (t->ctx_hi ^ c.ctx_hi) | (t->gen ^ c.gen))) {
            t_ibst = *t; return;
        }
    }
    c.sid = ++t_ibst_n;
    t_ibst_tab[t_ibst_rr++ & 7] = c;
    t_ibst = c;
}
/* the context's current state matches the memo (and the memo's generation is current) */
static inline __attribute__((always_inline)) int spu_ibc_state_ok(const spu_context* ctx)
{
    const spu_ibc_st* st = &t_ibst;
    uint32_t d = SPU_IBX(st->img, ctx->image_id) | SPU_IBX(st->ovl, ctx->resident_ovl) | SPU_IBX(st->task, ctx->resident_task) |
                 SPU_IBX(st->pm, ctx->policy_mode) | SPU_IBX(st->nreg, s_registry_count) | SPU_IBX(st->ndyn, s_dyn_n) |
                 SPU_IBX(st->ctx_lo, (uint32_t)(uintptr_t)ctx) | SPU_IBX(st->ctx_hi, (uint32_t)((uint64_t)(uintptr_t)ctx >> 32)) |
                 SPU_IBX(st->gen, t_ib_gen);
    return d == 0 && st->sid != 0;
}
/* called after a verified t_ib hit for (ctx state, pc) -> fn */
static inline __attribute__((always_inline)) void spu_ibc_fill(const spu_context* ctx, uint32_t pc, spu_fn fn)
{
    if (!s_ibc) return;
    if (__builtin_expect(!spu_ibc_state_ok(ctx), 0)) spu_ibc_refresh(ctx);
    spu_ibc_ent* c = &t_ibc[(pc >> 2) * 2654435761u >> 21];
    c->fn = fn; c->pc = pc; c->sid = t_ibst.sid;
}
static inline __attribute__((always_inline)) spu_fn spu_ibc_probe(const spu_context* ctx, uint32_t pc)
{
    if (!spu_ibc_state_ok(ctx)) return NULL;
    const spu_ibc_ent* c = &t_ibc[(pc >> 2) * 2654435761u >> 21];
    return (c->pc == pc && c->sid == t_ibst.sid) ? c->fn : NULL;
}

/* second way of the set, a separate function so the first-way hit path keeps few live registers */
__attribute__((noinline)) static void spu_ib_way1(spu_context* ctx)
{
    const uint32_t pc = ctx->pc;     /* already masked by the caller */
    spu_ib_ent* e = &t_ib[((pc >> 2) * 2654435761u >> 21) ^ 1u];
    if (!spu_ib_ent_ok(ctx, pc, e, t_ib_gen)) { SPU_IB_TAIL(spu_ib_cold(ctx)); }
    if (__builtin_expect(s_ib_stats, 0)) ++t_ibs_hit;
    spu_ibc_fill(ctx, pc, e->fn);
    SPU_IB_TAIL(e->fn(ctx));
}
/* what the front end would dispatch to (NULL = it would fall back); SPU_IB_FAST=2 only */
__attribute__((noinline)) static spu_fn spu_ib_peek(spu_context* ctx)
{
    const uint32_t pc = ctx->pc & SPU_LS_MASK;
    if (SPU_IB_SPECIAL(ctx, pc)) return NULL;
    if (pc < SPU_TASKSET_TASK_LO) ctx->resident_task = 0;
    spu_ib_ent* e = &t_ib[(pc >> 2) * 2654435761u >> 21];
    spu_fn r = NULL;
    if (spu_ib_ent_ok(ctx, pc, e, t_ib_gen)) r = e->fn;
    else { e = &t_ib[((pc >> 2) * 2654435761u >> 21) ^ 1u];
           if (spu_ib_ent_ok(ctx, pc, e, t_ib_gen)) r = e->fn; }
    if (s_ibc) { spu_fn c = spu_ibc_probe(ctx, pc);   /* a compact hit must name the function a t_ib hit names (or t_ib missed since) */
        if (c) { if (r && c != r) { long b = __atomic_add_fetch(&s_ibc_bad, 1, __ATOMIC_RELAXED); if (b <= 16) fprintf(stderr, "[ibc] DISAGREE pc=0x%05X img=%d compact=%p t_ib=%p\n", pc, ctx->image_id, (void*)c, (void*)r); }
                 else __atomic_add_fetch(&s_ibc_hit, 1, __ATOMIC_RELAXED); } }
    return r;
}
static volatile long s_ibv_ok, s_ibv_bad, s_ibv_nopeek;
void spu_indirect_branch(spu_context* ctx)
{
    R35C(R35_IB);
    if (__builtin_expect(s_ib_fast != 1, 0)) {
        SPU_IB_TAIL(spu_ib_cold(ctx));
    }
    const uint32_t pc = ctx->pc & SPU_LS_MASK;
    if (SPU_IB_SPECIAL(ctx, pc) || SPU_COLLNET_CALL(ctx, pc)) {
        SPU_IB_TAIL(spu_ib_cold(ctx));
    }
    ctx->pc = pc;
    if (pc < SPU_TASKSET_TASK_LO) ctx->resident_task = 0;
    { spu_fn cf = spu_ibc_probe(ctx, pc);
      if (__builtin_expect(cf != NULL, 1)) {
          if (__builtin_expect(s_ib_stats, 0)) ++t_ibs_hit;
          SPU_IB_TAIL(cf(ctx));
      } }
    spu_ib_ent* e = &t_ib[(pc >> 2) * 2654435761u >> 21];
    if (__builtin_expect(!spu_ib_ent_ok(ctx, pc, e, t_ib_gen), 0)) {
        SPU_IB_TAIL(spu_ib_way1(ctx));
    }
    if (__builtin_expect(s_ib_stats, 0)) ++t_ibs_hit;
    spu_ibc_fill(ctx, pc, e->fn);
    SPU_IB_TAIL(e->fn(ctx));
}

/* ---- collision net: igCollBroadSpu (img 110) job-record dispatch ------------------------------------
 * 0x28A8 loads the 160-byte job record at LS 0x23000 (a plain GET at 0x2F90/0x2FA0), takes the u16 at +10 as an index
 * and does `bisl` to LS[0x6D20 + 4*idx] with NO range check. The table has 11 entries (idx 0..10: 1A90 2230 20C0 1F50
 * 1A60 1468 1438 1408 1498 1720 15B0). When the record is read while another thread is still writing it, idx is
 * garbage (e.g. 0x8000 = the "done" flag of the same word) and the branch goes to a table slot that holds 0 or any
 * data word: LS 0 (already survived below), or a wild address -> the SPU thread dies ("synthesised stop"), the other
 * collision SPU waits for its reply forever and the main thread waits for it: the combat freeze.
 *   1. idx > 10 and the memory copy of the record is valid now (the writer finished): take the settled record,
 *      redo the register computations 0x28A8 did and dispatch to the right handler -- the job runs, as on hardware.
 *   2. otherwise: behave like the empty handler (return to the link, r3 = 0x23000): the job is skipped and finishes.
 * SPU_COLLNET=0 off. SPU_COLLNET_TEST=<period>[:<mode>] injects a bad index every <period> dispatches (mode 0: the
 * re-read has to recover it, mode 1: straight to the empty handler) to exercise the path without waiting for the race. */
SPU_THREAD_LOCAL uint32_t g_collnet_rec_ea;
volatile long g_coll_get_retry, g_coll_get_calls;
static const uint32_t k_collnet_tab[11] = { 0x1A90, 0x2230, 0x20C0, 0x1F50, 0x1A60, 0x1468, 0x1438, 0x1408, 0x1498, 0x1720, 0x15B0 };
/* 0x28A8: rotqbyi r42,10 puts record bytes 10..13 in the preferred word and `and 0xFFFF` keeps the LOW halfword = bytes 12,13 (NOT 10,11:
 * the first version of this net looked at the wrong halfword, so it never saw a bad index) */
static inline uint32_t collnet_idx(const uint8_t* rec) { return ((uint32_t)rec[12] << 8) | rec[13]; }
static int spu_collnet_guard(spu_context* ctx)
{
    static int s_on = -1; static long s_per = 0, s_mode = 0;
    static volatile long s_calls, s_calls2, s_bad, s_fixed, s_nulled, s_forced;
    if (__builtin_expect(s_on < 0, 0)) {
        const char* e = getenv("SPU_COLLNET"); s_on = !(e && e[0] == '0');
        const char* t = getenv("SPU_COLLNET_TEST");
        if (t && t[0]) { s_per = atol(t); const char* c = strchr(t, ':'); s_mode = c ? atol(c + 1) : 0; }
    }
    if (!s_on) return 0;
    /* which of the two dispatches: 1st (link 0x28E0, r16, table 0x6D20, idx 0..10) or 2nd (link 0x2DC0, r82, table 0x6D30 = entries 4.., idx 0..6) */
    const int second = (ctx->gpr[0]._u32[0] & SPU_LS_MASK) == 0x2DC0u;
    const uint32_t off = second ? 4u : 0u, maxi = second ? 6u : 10u, link = second ? 0x2DC0u : 0x28E0u;
    uint8_t* rec = &ctx->ls[0x23000];
    if (g_ctw_on != 0) spu_colltab_check(ctx, "dispatch guard");
    uint32_t idx = collnet_idx(rec);
    uint32_t raw = second ? ctx->gpr[82]._u32[0] : ctx->gpr[16]._u32[0];
    long nc = __atomic_add_fetch(second ? &s_calls2 : &s_calls, 1, __ATOMIC_RELAXED);
    int forced = 0;
    if (s_per > 0 && idx <= maxi && (raw & SPU_LS_MASK) == k_collnet_tab[off + idx] && (nc % s_per) == 0) {
        /* test injection: mode 0/1 = bad idx in the LS record (memory stays valid), mode 2 = the register the bisl used is wild
         * while the LS record is fine, mode 3 = both (what the real event looked like); mode 1 = also forbid the re-read */
        forced = 1;
        if (s_mode != 2 && !second) rec[12] |= 0x80;   /* the 2nd dispatch comes after the 1st settled the record: only its register is hit */
        if (s_mode >= 2) { raw = 0x000136BEu | 0x40000000u; ctx->pc = raw & SPU_LS_MASK; if (second) ctx->gpr[82] = spu_ila(raw & SPU_LS_MASK); else ctx->gpr[16] = spu_ila(raw & SPU_LS_MASK); }
        idx = collnet_idx(rec);
        __atomic_add_fetch(&s_forced, 1, __ATOMIC_RELAXED);
    }
    /* the one test that matters: is the word the bisl branches to one of the handlers of the table for this record's index? */
    if (idx <= maxi && (raw & SPU_LS_MASK) == k_collnet_tab[off + idx]) return 0;
    if (second && idx > maxi && idx <= 10) return 0;   /* 2nd dispatch with a first-table-only index: hardware ill-defined, leave it */
    long nb = __atomic_add_fetch(&s_bad, 1, __ATOMIC_RELAXED);
    const uint32_t tgt_idx = idx, bad_target = ctx->pc;
    const int idx_bad = idx > 10;     /* kind B: the LS record has a wild index; kind A (!idx_bad): LS record fine, the register used by the bisl is not (LS changed after the load, or a wild table word) */
    uint8_t mem[160]; int have_mem = 0, rereads = 0;
    const uint32_t rea = g_collnet_rec_ea;
    long long c37mark = (idx_bad && !second && !forced) ? c37_stale(ctx, rea) : -1;   /* SPU_C37 */
    if (idx_bad && !second && rea && vm_base && !(forced && s_mode == 1)) {
        /* The LS record is the previous, already-finished job (done flag 0x100 in byte 12, index 0x0101): the producer of the
         * new record had not written it yet when the GET ran. Measured (f35 series): the memory copy settles after 4..1750 polls
         * (up to ~10 ms) and once not within 2000. Wait for it up to 200 ms (then Sleep(1) between polls) instead of skipping the job. */
        /* SPU_COLLNET_WAITMS=<ms> (default 20; 200 = the crash35 value): coll37 measured (c37_t5_2/_3, tracer SPU_C37) that the
         * record sometimes NEVER settles -- 2 of 2 events waited the whole 200 ms (12 lost frames) and still skipped the job -- while
         * the ones that settle do so within ~2 ms (<= 900 polls). Waiting longer than a few ms only adds a hitch. */
        static int s_wait_ms = -1;
        if (s_wait_ms < 0) { const char* e = getenv("SPU_COLLNET_WAITMS"); s_wait_ms = e ? atoi(e) : 20; if (s_wait_ms < 0) s_wait_ms = 0; }
        const ULONGLONG t0 = GetTickCount64();
        for (int i = 0; ; i++) {
            memcpy(mem, vm_base + rea, 160); have_mem = 1;
            if (collnet_idx(mem) <= 10) break;
            rereads++;
            if (i >= 1500) { if (GetTickCount64() - t0 > (ULONGLONG)s_wait_ms) break; Sleep(1); continue; }
            for (int k = 0; k < 32; k++) _mm_pause();
            if ((i & 7) == 7) SwitchToThread();
        }
    }
    int fixed = idx_bad ? (have_mem && collnet_idx(mem) <= 10) : 1;
    if (c37mark >= 0) c37_stale_done(c37mark, (uint32_t)rereads, have_mem ? mem : 0);
    if (second && idx_bad) fixed = 0;
    if (nb <= 1000) {
        fprintf(stderr, "[coll-net] img=%d %s dispatch idx=0x%04X target 0x%05X (reg raw 0x%08X) does not match the table (#%ld, %s, %s) rec_ea=0x%08X rereads=%d -> %s\n",
                ctx->image_id, second ? "2nd(0x2DC0)" : "1st(0x28E0)", tgt_idx, bad_target, raw, nb, forced ? "INJECTED" : "real",
                idx_bad ? "LS index wild" : "LS index ok, register stale/wild", rea, rereads,
                fixed ? "dispatching the right handler" : "empty handler (return to link)");
        if (nb <= 8) {
            fprintf(stderr, "[coll-net]   LS  rec:"); for (int b = 0; b < 160; b++) fprintf(stderr, "%s%02X", (b & 15) ? "" : " ", rec[b]);
            if (have_mem) { fprintf(stderr, "\n[coll-net]   MEM rec:"); for (int b = 0; b < 160; b++) fprintf(stderr, "%s%02X", (b & 15) ? "" : " ", mem[b]); }
            fprintf(stderr, "\n[coll-net]   r0=%08X r1=%08X r3=%08X r16=%08X r82=%08X r17(idx used)=%08X r41=%08X r42(record the lifted code loaded)=%08X.%08X.%08X.%08X recent pcs:",
                    ctx->gpr[0]._u32[0], ctx->gpr[1]._u32[0], ctx->gpr[3]._u32[0], ctx->gpr[16]._u32[0], ctx->gpr[82]._u32[0], ctx->gpr[17]._u32[0], ctx->gpr[41]._u32[0],
                    ctx->gpr[42]._u32[0], ctx->gpr[42]._u32[1], ctx->gpr[42]._u32[2], ctx->gpr[42]._u32[3]);
            extern unsigned spu_recent_pcs(const spu_context*, uint32_t* out, unsigned max);
            uint32_t pcs[12]; unsigned n = spu_recent_pcs(ctx, pcs, 12);
            for (unsigned i = 0; i < n; i++) fprintf(stderr, " %05X", pcs[i]);
            fprintf(stderr, "\n");
        }
    }
    if (fixed) {
        if (idx_bad) { memcpy(rec, mem, 160); idx = collnet_idx(rec); }
        if (!second) {
            /* the register work of 0x28A8 .. 0x28DC, redone with the settled record */
            ctx->gpr[83] = spu_ila(0xFFFF);
            ctx->gpr[42] = spu_ls_read128(ctx, 0x23000);
            ctx->gpr[14] = spu_ila(0x6D20);
            ctx->gpr[41] = spu_rotqbyi(ctx->gpr[42], 10);
            ctx->gpr[17] = spu_and(ctx->gpr[41], ctx->gpr[83]);
            ctx->gpr[12] = spu_shli(ctx->gpr[17], 2);
            ctx->gpr[10] = spu_a(ctx->gpr[12], ctx->gpr[14]);
        }
        /* the table is static program data (k_collnet_tab); use it directly, not whatever LS[0x6D20..] reads right now */
        const uint32_t t = k_collnet_tab[off + idx];
        if (second) ctx->gpr[82] = spu_ila(t); else ctx->gpr[16] = spu_ila(t);
        ctx->pc = t; __atomic_add_fetch(&s_fixed, 1, __ATOMIC_RELAXED); return 0;
    }
    __atomic_add_fetch(&s_nulled, 1, __ATOMIC_RELAXED);
    ctx->pc = link;
    g_spu_trampoline_fn = spu_indirect_branch;
    return 1;
}


/* ---- SPU_COLLTAB_WATCH=1 (diagnostic): who dirties the img-110 dispatch table LS 0x6D20..0x6D4B? -------------------
 * The real [coll-net] events show valid indices but a garbage target, i.e. the 11-word table (static ELF data,
 * k_collnet_tab) holds other bytes at the dispatch. Three tripwires, all out of the hot path:
 *   - every lifted store (spu_ls_write128 -> spu_ls_write_probe_smc, armed by this env) into 0x6D20..0x6D4F: logged;
 *   - every channel write of the thread (and right after each MFC command, with the command decoded, list
 *     elements included) compares the 48 bytes with the last good snapshot: a change names the last DMA/channel op;
 *   - the net events (guard / unlifted-pc path) compare too ("writer unknown": none of the above saw it). */
int g_ctw_on = -1;
#define CTW_LO 0x60u
#define CTW_N  0x6CF0u   /* program image 0x60..0x6D4F: code, jump tables, the 0x6D20 dispatch table (the parameter block 0x6D50.. is rewritten by the PPU) */
static SPU_THREAD_LOCAL uint8_t t_ctw_snap[CTW_N];
static SPU_THREAD_LOCAL int t_ctw_have, t_ctw_calls, t_ctw_bad;
static SPU_THREAD_LOCAL struct { uint32_t cmd, lsa, eah, eal, size, tag, pc; } t_ctw_dma;
static SPU_THREAD_LOCAL uint32_t t_ctw_ch, t_ctw_chv;
static volatile long s_ctw_logs;
static int ctw_table_good(const uint8_t* ls)
{
    for (int i = 0; i < 11; i++) {
        const uint8_t* q = ls + 0x6D20 + 4 * i;
        uint32_t w = ((uint32_t)q[0] << 24) | ((uint32_t)q[1] << 16) | ((uint32_t)q[2] << 8) | q[3];
        if (w != k_collnet_tab[i]) return 0;
    }
    return 1;
}
static void ctw_hex(const char* tag, const uint8_t* b)
{
    fprintf(stderr, "%s", tag);
    for (int i = 0; i < 0x30; i++) fprintf(stderr, "%s%02X", (i & 3) ? "" : " ", b[i]);
    fprintf(stderr, "\n");
}
static void ctw_pcs(const spu_context* ctx)
{
    extern unsigned spu_recent_pcs(const spu_context*, uint32_t* out, unsigned max);
    uint32_t pcs[16]; unsigned n = spu_recent_pcs(ctx, pcs, 16);
    fprintf(stderr, "[colltab]   pc=%05X r0=%08X r1=%08X recent pcs:", (unsigned)(ctx->pc & SPU_LS_MASK), ctx->gpr[0]._u32[0], ctx->gpr[1]._u32[0]);
    for (unsigned i = 0; i < n; i++) fprintf(stderr, " %05X", pcs[i]);
    fprintf(stderr, "\n");
}
void spu_colltab_wr(spu_context* ctx, uint32_t lsa, const uint8_t* p)
{
    if (g_ctw_on < 0) { const char* e = getenv("SPU_COLLTAB_WATCH"); g_ctw_on = (e && e[0] && e[0] != '0') ? 1 : 0; }
    if (!g_ctw_on) return;
    long k = __atomic_add_fetch(&s_ctw_logs, 1, __ATOMIC_RELAXED);
    if (k > 200) return;
    fprintf(stderr, "[colltab] LIFTED STORE LS 0x%05X = %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X\n", lsa,
            p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15]);
    ctw_pcs(ctx);
}
static void ctw_note(spu_context* ctx, uint32_t channel, uint32_t v)
{
    if (channel == MFC_Cmd) {
        t_ctw_dma.cmd = v; t_ctw_dma.lsa = ctx->mfc_lsa; t_ctw_dma.eah = ctx->mfc_eah; t_ctw_dma.eal = ctx->mfc_eal;
        t_ctw_dma.size = ctx->mfc_size; t_ctw_dma.tag = ctx->mfc_tag; t_ctw_dma.pc = (uint32_t)ctx->pc & SPU_LS_MASK;
    } else if (channel != MFC_LSA && channel != MFC_EAH && channel != MFC_EAL && channel != MFC_Size && channel != MFC_TagID) {
        t_ctw_ch = channel; t_ctw_chv = v;
    }
}
static void spu_colltab_check(spu_context* ctx, const char* where)
{
    if (g_ctw_on < 0) { const char* e = getenv("SPU_COLLTAB_WATCH"); g_ctw_on = (e && e[0] && e[0] != '0') ? 1 : 0; }
    if (!g_ctw_on) return;
    const uint8_t* t = &ctx->ls[0x6D20];
    if (!t_ctw_have) {
        if (ctw_table_good(ctx->ls)) {
            t_ctw_have = 1; memcpy(t_ctw_snap, &ctx->ls[CTW_LO], CTW_N);
            fprintf(stderr, "[colltab] good image snapshot taken (%s, call %d)\n", where, t_ctw_calls);
            ctw_hex("[colltab]   table:", t);
        } else if (++t_ctw_calls == 20000 || t_ctw_calls == 200) {
            fprintf(stderr, "[colltab] table is NOT the expected one after %d checks (%s)\n", t_ctw_calls, where);
            ctw_hex("[colltab]   table:", t);
        }
        return;
    }
    if (memcmp(t_ctw_snap, &ctx->ls[CTW_LO], CTW_N) == 0) return;
    int good = ctw_table_good(ctx->ls);
    long k = __atomic_add_fetch(&s_ctw_logs, 1, __ATOMIC_RELAXED);
    t_ctw_bad++;
    if (k <= 200) {
        fprintf(stderr, "[colltab] TABLE %s detected at %s (#%d)\n", good ? "RESTORED" : "CHANGED", where, t_ctw_bad);
        { int runs = 0;   /* changed byte runs of the image */
          for (uint32_t i = 0; i < CTW_N && runs < 6; ) {
              if (t_ctw_snap[i] == ctx->ls[CTW_LO + i]) { i++; continue; }
              uint32_t j = i; while (j < CTW_N && j - i < 48 && (t_ctw_snap[j] != ctx->ls[CTW_LO + j] || (j + 1 < CTW_N && t_ctw_snap[j + 1] != ctx->ls[CTW_LO + j + 1]))) j++;
              fprintf(stderr, "[colltab]   changed LS 0x%05X..0x%05X old:", CTW_LO + i, CTW_LO + j);
              for (uint32_t q = i; q < j; q++) fprintf(stderr, "%s%02X", ((q - i) & 3) ? "" : " ", t_ctw_snap[q]);
              fprintf(stderr, "  new:");
              for (uint32_t q = i; q < j; q++) fprintf(stderr, "%s%02X", ((q - i) & 3) ? "" : " ", ctx->ls[CTW_LO + q]);
              fprintf(stderr, "\n"); runs++; i = j + 1;
          } }
        ctw_hex("[colltab]   table 6D20 now:", t);
        fprintf(stderr, "[colltab]   last MFC cmd=0x%02X lsa=0x%05X ea=0x%08X%08X size=%u tag=%u (at pc %05X); last non-MFC channel ch%u=0x%08X\n",
                t_ctw_dma.cmd, t_ctw_dma.lsa, t_ctw_dma.eah, t_ctw_dma.eal, t_ctw_dma.size, t_ctw_dma.tag, t_ctw_dma.pc, t_ctw_ch, t_ctw_chv);
        if (t_ctw_dma.cmd & 0x04) {   /* list: dump its elements (stall/size, eal); for a list lsa = LS list address, eal = the EA high word source */
            uint32_t la = t_ctw_dma.eal & SPU_LS_MASK & ~7u; fprintf(stderr, "[colltab]   list @LS %05X (lsa=%05X):", la, t_ctw_dma.lsa);
            for (uint32_t e = 0; e < t_ctw_dma.size && e < 64; e += 8) {
                const uint8_t* q = &ctx->ls[(la + e) & SPU_LS_MASK];
                fprintf(stderr, " [%02X%02X sz=%u eal=%02X%02X%02X%02X]", q[0], q[1], (q[2] << 8) | q[3], q[4], q[5], q[6], q[7]);
            }
            fprintf(stderr, "\n");
        }
        ctw_pcs(ctx);
    }
    memcpy(t_ctw_snap, &ctx->ls[CTW_LO], CTW_N);
}

void spu_indirect_branch_full(spu_context* ctx)
{
    R35C(R35_IBFULL);
    const int ib_vfy = t_ib_vfy; if (ib_vfy) t_ib_vfy = 0;   /* SPU_IB_FAST=2 */
    /* Real SPU bi/bisl mask the target to the 256 KB local store; the high bits
     * of a computed pointer (e.g. a packed handle like 0x7a028803) are ignored.
     * Without this, any indirect branch through such a value fails the lookup
     * and falls into branch-to-0. All lifted funcs live below SPU_LS_SIZE, so
     * masking is a no-op for already-valid targets. */
    ctx->pc &= SPU_LS_MASK;
    if (SPU_COLLNET_CALL(ctx, ctx->pc) && ctx->image_id == 110 && spu_collnet_guard(ctx)) return;
    /* Interrupt-return register restore (see spu_drain.c spu_irq_regs_save):
     * the WWS handler's save/restore shim lives at unlifted top-of-LS, so we
     * enforce the preserve-all-registers hardware contract here -- the first
     * interrupts-enabled dispatch at the saved srr0 is the irete. */
    { extern int spu_irq_regs_maybe_restore(spu_context*);
      if (__builtin_expect(ctx->irq_saved != 0, 0) && spu_irq_regs_maybe_restore(ctx)) {
          static int s_it = -1; if (s_it < 0) s_it = getenv("SPU_IRQTRACE") ? 1 : 0;
          static int _r = 0;
          if (s_it && (_r < 200 && ++_r))
              fprintf(stderr, "[irq] IRET restore at srr0=0x%05X\n", ctx->pc & SPU_LS_MASK);
      } }
    /* SPURS kernel services (policy-module runs only): the HLE kernel plants
     * these two reserved addresses as exitToKernelAddr / selectWorkloadAddr in
     * the SpursKernelContext (spurs_policy.c). */
    if (ctx->policy_mode) {
        extern SPU_THREAD_LOCAL volatile unsigned g_spurs_pm_polls, g_spurs_pm_exited;
        if (ctx->pc == SPURS_PM_EXIT_TO_KERNEL_LS) {
            /* Module exit: the workload returned to the kernel (drained/yield).
             * Print gated: fires once per policy run = thousands/sec. */
            g_spurs_pm_exited = 1;
            { static SPU_THREAD_LOCAL int s_t = -1; if (s_t < 0) s_t = getenv("SPURS_PM_TRACE") ? 1 : 0;
              if (s_t) { static SPU_THREAD_LOCAL unsigned long _n = 0; unsigned long n = ++_n;
                if (n <= 64 || (n & 0xFFF) == 0)
                    fprintf(stderr, "[spurs-pm] exit-to-kernel#%lu (r3=0x%08X polls=%u)\n",
                            n, ctx->gpr[3]._u32[0], g_spurs_pm_polls); } }
            ctx->status = SPU_STATUS_STOPPED_BY_STOP;
            spu_halt(ctx);
            return;
        }
        if (ctx->pc == SPURS_PM_SELECT_WORKLOAD_LS) {
            /* cellSpursModulePoll: report "no contention — keep running".
             * (The host scheduler admits bounded workers; this service does not preempt.) */
            unsigned n = ++g_spurs_pm_polls;
            /* Contention, as the real kernel's select does it: another workload
             * eligible on THIS lane (priority != 0, room under maxContention)
             * that has work (readyCount or signal) and a priority at least ours
             * makes the poll answer "yield". Without it a WWS PM spinning on a
             * dependency (ToD: geomcullclip waits for data the PPU publishes
             * only after a SharedJobList job on the same lane ran) holds the
             * lane forever. SPURS_PM_NO_CONTEND=1 restores "always continue". */
            extern uint8_t* vm_base;
            #define PMR8(a)  ((uint32_t)vm_base[(uint32_t)(a)])
            #define PMR16(a) ((PMR8(a) << 8) | PMR8((a) + 1))
            #define PMR32(a) ((PMR16(a) << 16) | PMR16((a) + 2))
            uint32_t yield = 0;
            { static int s_nc = -1; if (s_nc < 0) s_nc = getenv("SPURS_PM_NO_CONTEND") ? 1 : 0;
              if (!s_nc) {
                const uint8_t* L = ctx->ls;
                #define LSW(o) (((uint32_t)L[(o)]<<24)|((uint32_t)L[(o)+1]<<16)|((uint32_t)L[(o)+2]<<8)|L[(o)+3])
                uint32_t sea = LSW(0x1C4), lane = LSW(0x1C8), cur = LSW(0x1DC);
                #undef LSW
                if (sea && lane < 8 && cur < 16) {
                    uint32_t en = PMR32(sea + 0xB0);
                    uint32_t sig = PMR16(sea + 0x70);
                    uint32_t mypr = PMR8(sea + 0xB00 + cur * 0x20 + 0x18 + lane);
                    for (uint32_t w = 0; w < 16 && !yield; w++) {
                        if (w == cur || !(en & (0x80000000u >> w)) || PMR8(sea + 0x80 + w) != 2) continue;
                        uint32_t pr = PMR8(sea + 0xB00 + w * 0x20 + 0x18 + lane);
                        if (!pr || pr < mypr) continue;
                        uint32_t maxc = PMR8(sea + 0x50 + w);
                        if (!maxc || PMR8(sea + 0x20 + w) >= maxc) continue;
                        if (PMR8(sea + 0x00 + w) || (sig & (0x8000u >> w))) yield = 1;
                    }
                    if (yield) { static volatile long s_y;
                        long k = __atomic_add_fetch(&s_y, 1, __ATOMIC_RELAXED);
                        if (k <= 8 || (k % 4096) == 0)
                            fprintf(stderr, "[spurs-pm] poll lane %u wid %u -> yield (contended, #%ld)\n", lane, cur, k); }
                } } }
            static volatile long s_pl; /* n restarts each PM run: ~1500 lines/s at metropolis */
            if (!yield && (n % 4096) == 0 || (!yield && n <= 4 && __atomic_add_fetch(&s_pl, 1, __ATOMIC_RELAXED) <= 16))
                fprintf(stderr, "[spurs-pm] poll #%u (r3=0x%08X) -> continue\n",
                        n, ctx->gpr[3]._u32[0]);
            ctx->gpr[3] = spu_make_preferred_u32(yield);
            ctx->pc = ctx->gpr[0]._u32[0] & SPU_LS_MASK;
            return;
        }
    }
    /* Synthetic HLE tasks have no resident taskset policy at 0xA70. Real
     * tasksets use the SAME syscall address, so the API table alone is not
     * evidence that this is an HLE context. LLE tasks must enter Sony's policy
     * to yield/select workloads instead of parking an entire SPU host thread.
     * Image 22 is the legacy standalone HLE CRI task runner. */
    if (ctx->pc == SPURS_TASKSET_PM_SYSCALL_LS) {
        uint32_t sc = ((uint32_t)ctx->ls[0x27C4] << 24) | ((uint32_t)ctx->ls[0x27C5] << 16)
                    | ((uint32_t)ctx->ls[0x27C6] << 8)  | ctx->ls[0x27C7];
        /* A task dispatched standalone (no policy module in this local store)
         * runs on the SpursTasksetContext spu_workload.c plants, whose
         * syscallAddr is this address; nothing is lifted at 0xA70 for it, so
         * the only alternative to the HLE is the synthesised-stop exit below
         * -- which is how LBP's audio tasks (images 6/7) ended on their first
         * syscall and left the loading thread waiting on their event flag. */
        int lifted = spu_lookup(SPURS_TASKSET_PM_SYSCALL_LS, ctx->image_id) ||
            (ctx->resident_task && spu_lookup(SPURS_TASKSET_PM_SYSCALL_LS, ctx->resident_task)) ||
            (ctx->resident_ovl  && spu_lookup(SPURS_TASKSET_PM_SYSCALL_LS, ctx->resident_ovl));
        int standalone = !ctx->policy_mode && sc == SPURS_TASKSET_PM_SYSCALL_LS && !lifted;
        if (ctx->image_id == spu_cri_image() || standalone ||
            (ctx->policy_mode && sc == SPURS_TASKSET_PM_SYSCALL_LS)) {
            spu_spurs_taskset_syscall(ctx);
            ctx->pc = ctx->gpr[0]._u32[0] & SPU_LS_MASK;
            return;
        }
    }
    /* YDKJ_CRI_R4: the taskset policy entry (LS 0xA00, image 23) writes r4 into
     * SpursTasksetContext.taskset @LS 0x27B8 (per RPCS3 cellSpursSpu.cpp). Our
     * kernel->policy handoff doesn't convey the taskset EA, so the policy DMAs
     * the taskset from garbage -> waiting!=0 -> wrong resume path -> savedContextLr=0.
     * Inject r4 = taskset EA (0x0F000000) at the policy entry dispatch (this is the
     * exact point before the entry reads r4, after the kernel's arg setup). */
    if (ctx->image_id == SPU_RESOLVER_ONLY_IMAGE_ID && !getenv("YDKJ_NO_CRI_R4")) {
        static int s_r4 = -1; if (s_r4 < 0) s_r4 = getenv("YDKJ_CRI_CHAIN") ? 1 : 0;
        if (s_r4) {
            /* Force ctxt->taskset @LS 0x27B8 = the REAL game taskset EA on every
             * image-23 branch, so the policy's atomic reads + context-EA computation
             * use the actual taskset (the r4 handoff sets it to garbage 0x0000FFFF via
             * a path we can't intercept). Was hardcoded 0x0F000000, which mismatched
             * the game's real taskset (0x45F1B000) -> policy DMA'd garbage. */
            extern uint32_t g_ydkj_real_taskset_ea;
            uint32_t ts = g_ydkj_real_taskset_ea ? g_ydkj_real_taskset_ea : 0x0F000000u;
            ctx->ls[0x27B8]=0x00; ctx->ls[0x27B9]=0x00; ctx->ls[0x27BA]=0x00; ctx->ls[0x27BB]=0x00;
            ctx->ls[0x27BC]=(uint8_t)(ts>>24); ctx->ls[0x27BD]=(uint8_t)(ts>>16); ctx->ls[0x27BE]=(uint8_t)(ts>>8); ctx->ls[0x27BF]=(uint8_t)ts;
            if (ctx->pc == 0xA00u) { static int _n=0; if (_n < 4 && ++_n)
                fprintf(stderr, "[cri-r4] policy entry pc=0xA00: forced ctxt->taskset LS[0x27B8]=0x0F000000\n"); }
        }
    }
    /* SPU_IBCOV: image-3 (Bink SPU) PC-page coverage. Track which 0x1000-byte LS
     * pages the task's indirect branches land in; dump the set periodically. If
     * coverage stays in the kernel/wait region (~0x13xxx) the decode routine never
     * runs; if it spans a wide high range, decode executes but doesn't output. */
    { static int s_cov = -1; if (s_cov < 0) s_cov = getenv("SPU_IBCOV") ? 1 : 0;
      if (s_cov && ctx->image_id == 3) {
        static uint8_t pages[64] = {0};   /* 64 pages * 0x1000 = 256KB LS */
        static uint64_t hits = 0;
        uint32_t pg = (ctx->pc & 0x3FFFF) >> 12;
        int newp = 0;
        if (pg < 64 && !pages[pg]) { pages[pg] = 1; newp = 1; }
        ++hits;
        if (newp || (hits % 200000) == 0) {
            char line[400]; int p = snprintf(line, sizeof line, "[ibcov3] hits=%llu tgt=0x%05X pages:", (unsigned long long)hits, ctx->pc & 0x3FFFF);
            for (int i = 0; i < 64; i++) if (pages[i]) p += snprintf(line+p, sizeof(line)-p, " 0x%X", i<<12);
            fprintf(stderr, "%s\n", line); }
      } }
    { static int s_t = -1; if (s_t < 0) s_t = getenv("SPU_POLLTRACE") ? 1 : 0;
      if (s_t) { static uint64_t s_n = 0; static uint32_t s_last = 0; static uint64_t s_run = 0;
        if (ctx->pc == s_last) s_run++; else { s_last = ctx->pc; s_run = 1; }
        if ((++s_n % 2000000) == 0)
          fprintf(stderr, "[ibranch] %llu indirect branches; current target=0x%05X run=%llu\n",
                  (unsigned long long)s_n, ctx->pc, (unsigned long long)s_run); } }
    /* Policy-entry trace: the SPURS policy at LS 0xA00 branches on
     * r8 = word at LS[r6] (must be 32 for the path that sets the dispatch ptr
     * LS[0x780]). Log r3..r6 + the word the kernel handed it, to see why the
     * wrong branch is taken. Env YDKJ_POLTRACE. */
    if (ctx->pc == 0xA00u) {
        static int64_t pt=-2; if (pt==-2){ const char* e=getenv("YDKJ_POLTRACE"); pt=e?1:0; }
        if (pt) { static int _p=0; if (_p < 8 && ++_p) {
            /* re-lifted policy entry uses r80 (kernel-set context base): r44=LS[r80+0xC0] */
            uint32_t r80=ctx->gpr[80]._u32[0] & SPU_LS_MASK;
            const uint8_t* q = ctx->ls + ((r80+0xC0)&SPU_LS_MASK);
            uint32_t ctxw = ((uint32_t)q[0]<<24)|((uint32_t)q[1]<<16)|((uint32_t)q[2]<<8)|q[3];
            fprintf(stderr, "[POLTRACE] policy@0xA00 r3=%08X r4=%08X r80=%08X  LS[r80+0xC0]=%08X\n",
                ctx->gpr[3]._u32[0], ctx->gpr[4]._u32[0], r80, ctxw);
            fflush(stderr);
        } }
    }
    /* SPU_IBCACHE (1 = on [default], 0 = off, 2 = check): per-thread cache pc -> lifted
     * function for the resolution below. The answer depends on the context's code
     * mapping (image, resident overlay/task/code slots, policy mode -- compared BY
     * VALUE in the key), on the function registry (its size is in the key) and on the
     * dynovl verification state (t_ib_gen, bumped by every drop/reset of that state).
     * Not used with tasksets registered (their resolution reads LS) nor with the job
     * trace diagnostics. Mode 2 resolves normally on every hit and counts disagreements. */
    spu_ib_ent* ib_ent = NULL; spu_ib_key ib_key; spu_fn ib_cached = NULL;
    {
        const int m = ib_vfy ? 2 : spu_ib_mode();
        if (m && !s_task_elf_count && !s_task_entry_count && ctx->pc != SPU_JOB_RETURN_LS) {
            if (ctx->pc < SPU_TASKSET_TASK_LO) ctx->resident_task = 0;   /* as the block below */
            ib_ent = &t_ib[((ctx->pc >> 2) * 2654435761u) >> 21];
            /* SPU_IBKEY_INPLACE (default 1; 0 = build the key first and compare it): the hit test
             * compares the cached key with the context fields in place, so a hit builds no key. */
            static int s_inplace = -1;
            if (s_inplace < 0) { const char* e = getenv("SPU_IBKEY_INPLACE"); s_inplace = !(e && e[0] == '0'); }
            int ib_hit;
            if (s_inplace) {
                ib_hit = ib_ent->fn && ib_ent->gen == t_ib_gen && spu_ib_key_match(ctx, &ib_ent->key);
                if (ib_hit && m == 1) {
                    spu_fn cf = ib_ent->fn;
                    spu_check_stack_reset(ctx, cf);
#if defined(__clang__)
                    __attribute__((musttail)) return cf(ctx);
#else
                    cf(ctx);
                    return;
#endif
                }
                spu_ib_make_key(ctx, &ib_key);
            } else {
                spu_ib_make_key(ctx, &ib_key);
                ib_hit = ib_ent->fn && ib_ent->gen == t_ib_gen && spu_ib_key_eq(&ib_ent->key, &ib_key);
                if (ib_hit && m == 1) {
                    spu_fn cf = ib_ent->fn;
                    spu_check_stack_reset(ctx, cf);
#if defined(__clang__)
                    __attribute__((musttail)) return cf(ctx);
#else
                    cf(ctx);
                    return;
#endif
                }
            }
            if (ib_hit) ib_cached = ib_ent->fn;
        }
    }
    /* Resolve both fresh launches and mid-function resumes from TaskInfo.
     * A scheduler call temporarily leaves the task region, but its return PC
     * is usually not an ELF entry. Prefer the policy's selected ELF so tasks
     * sharing an entry address cannot shadow one another. Keep the legacy
     * entry-only mapping for runners that have not registered ELF metadata. */
    if (ctx->pc < SPU_TASKSET_TASK_LO) {
        ctx->resident_task = 0;
    } else {
        int ti = spu_taskset_resident_image(ctx);
        if (ti && !ctx->resident_task && ctx->host_depth &&
            s_spu_halt_armed && !ctx->policy_mode) {
            /* The policy restored a task's guest registers and branches to
             * its saved PC. Its host frames still belong to the scheduler or
             * an earlier task invocation; none can satisfy this task's return.
             * Resume at depth zero so SPU_RET follows the restored guest link. */
            ctx->resident_task = ti;
            g_spu_trampoline_fn = 0;
            SPU_LONGJMP(s_spu_halt_env, 2);
        }
        if (!ti && !ctx->resident_task) ti = spu_taskset_task_image(ctx->pc);
        if (ti) ctx->resident_task = ti;
    }
    /* The launched task owns the shared task-code region: resolve it there FIRST
     * so a co-resident task at the same LS base cannot shadow it. (resident_task
     * is 0 outside the region, so this only ever fires for genuine task code.) */
    spu_fn fn = NULL;
    int code_owner = 0;
    for (unsigned slot = 0; slot < 4; ++slot) {
        if (ctx->resident_code[slot].image_id && ctx->pc >= ctx->resident_code[slot].lsa &&
            ctx->pc - ctx->resident_code[slot].lsa < ctx->resident_code[slot].size) {
            code_owner = ctx->resident_code[slot].image_id;
            fn = spu_lookup(ctx->pc, code_owner);
            break;
        }
    }
    /* Run-time captured overlays (hash-verified lifts, see spu_dynovl_find) own
     * their LS range outright: stale lifts of other images there must lose. */
    if (s_dyn_n && g_spu_dyn_page[(ctx->pc & SPU_LS_MASK) >> 8]) {
        int di = spu_dynovl_find(ctx, ctx->pc & SPU_LS_MASK);
        if (di) { fn = spu_lookup(ctx->pc, di); if (fn) code_owner = di; }
    }
    if (!code_owner && ctx->resident_task)
        fn = spu_lookup(ctx->pc, ctx->resident_task);
    /* Resident overlay next: streamed code overwrote that LS range, so its lift
     * is the truth there -- the base image's stale bytes at the same addresses
     * may also be registered (historical junk lifts) and must lose. */
    if (!code_owner && !fn && ctx->resident_ovl) fn = spu_lookup(ctx->pc, ctx->resident_ovl);
    if (!code_owner && !fn) fn = spu_lookup(ctx->pc, ctx->image_id);
    /* The job returned through the link register we planted: it is finished.
     * Its outermost frame ends in `bi $r0`, and r0 was 0 -- so without this the
     * return landed on LS 0, which is the job's OWN entry, and it ran a second
     * lap with dead registers: re-reading its parameters through a now-zero
     * context pointer and spinning on a DMA to what were really its own
     * opcodes. The work is already done by the time it returns. */
    if (!ctx->policy_mode && ctx->pc == SPU_JOB_RETURN_LS) {
        static int _n = 0;
        if (_n < 4 && ++_n)
            fprintf(stderr, "[spurs-job] img=%d returned to the job manager \n",
                    ctx->image_id);
        spu_halt(ctx);
        return;
    }
    /* Branch into high LS with no lifted code there.
     *
     * This was added believing the jobs called a resident SPURS job-manager
     * kernel at 0x16100 / 0x18160 / 0x1A100. That was WRONG. Those "absolute
     * branches" were the disassembler decoding ASCII as instructions: the
     * bytes behind one of them are 30 2C 20 43, i.e. "0, C" from the string
     * "ch0, F:0:400, 100, Channel 0 level, %". Every job embeds the same
     * parameter-description text, which is why the same phantom target
     * appeared in 9 of 12 unrelated images and looked like a shared ABI.
     *
     * The guard is kept because branching into unlifted high local store is
     * still an error worth stopping at rather than executing whatever is
     * there -- but it is a backstop, not a kernel interface. */
    /* SPU exit: the CRT does not branch to a `stop` in the image -- it BUILDS one
     * in memory and jumps to it. ps1_netemu's GPU core ends with
     *
     *     ori $r80,$r3,0 / andi $r80,$r80,255 / iohl $r80,0x2000
     *     stqd $r80,0x10($r1) / sync / ai $r3,$r1,16 / bi $r3
     *
     * i.e. `stop (0x2000 | status)` assembled onto the stack and executed there.
     * That target is a stack address with no lifted code, so the unlifted-branch
     * guard below called a clean exit "branched into unlifted LS 0x3FFB0 -- ending
     * the job" and reported stop_code 0, which reads as a runaway and sent a long
     * chase after phantom memory corruption. Decode the target word instead: a
     * `stop` is opcode 0 in the top 11 bits, so this is unambiguous. */
    if (!fn) {
        uint32_t p0 = ctx->pc & SPU_LS_MASK;
        /* collision net, last line of defence: a dispatch from the broad-phase record table (link 0x28E0, see
         * spu_collnet_guard) that still ends on a pc with no lifted code (a wild table word, not only LS 0) is
         * an empty handler too. SPU_COLLNET=0 keeps the stop. */
        const uint32_t cn_link = ctx->gpr[0]._u32[0] & SPU_LS_MASK;   /* 0x28E0 = 1st dispatch, 0x2DC0 = 2nd */
        if (ctx->image_id == 110 && (cn_link == 0x28E0u || cn_link == 0x2DC0u) && p0 != cn_link) {
            static int s_cn = -1;
            if (s_cn < 0) { const char* e = getenv("SPU_COLLNET"); s_cn = !(e && e[0] == '0'); }
            if (s_cn) {
                static volatile long n_cn = 0; long k = ++n_cn;
                if (g_ctw_on != 0) {
                    spu_colltab_check(ctx, "unlifted pc under a 0x28E0 dispatch");
                    ctw_hex("[colltab]   table at this event:", &ctx->ls[0x6D20]);
                    fprintf(stderr, "[colltab]   target pc=0x%05X idx word=0x%04X\n", p0, ((uint32_t)ctx->ls[0x2300C] << 8) | ctx->ls[0x2300D]);
                    fprintf(stderr, "[colltab]   registers whose preferred word == target:");
                    for (int rr = 0; rr < 128; rr++) if ((ctx->gpr[rr]._u32[0] & SPU_LS_MASK) == p0 || ctx->gpr[rr]._u32[0] == p0) fprintf(stderr, " r%d=%08X", rr, ctx->gpr[rr]._u32[0]);
                    fprintf(stderr, "\n[colltab]   r0=%08X r77=%08X r78=%08X r3=%08X r4=%08X r2=%08X sp-frame:", ctx->gpr[0]._u32[0], ctx->gpr[77]._u32[0], ctx->gpr[78]._u32[0], ctx->gpr[3]._u32[0], ctx->gpr[4]._u32[0], ctx->gpr[2]._u32[0]);
                    for (int o = 0; o < 6; o++) { const uint8_t* q = &ctx->ls[(ctx->gpr[1]._u32[0] + 0x10 * o) & SPU_LS_MASK & ~0xFu]; fprintf(stderr, " [%02X%02X%02X%02X %02X%02X%02X%02X]", q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7]); }
                    fprintf(stderr, "\n[colltab]   host_depth=%d image=%d\n", (int)ctx->host_depth, ctx->image_id);
                    fprintf(stderr, "[colltab]   dispatch regs: r10=%08X r11=%08X.%08X.%08X.%08X r12=%08X r14=%08X r16=%08X r17=%08X r41=%08X r42=%08X.%08X.%08X.%08X\n",
                            ctx->gpr[10]._u32[0], ctx->gpr[11]._u32[0], ctx->gpr[11]._u32[1], ctx->gpr[11]._u32[2], ctx->gpr[11]._u32[3], ctx->gpr[12]._u32[0], ctx->gpr[14]._u32[0], ctx->gpr[16]._u32[0], ctx->gpr[17]._u32[0], ctx->gpr[41]._u32[0],
                            ctx->gpr[42]._u32[0], ctx->gpr[42]._u32[1], ctx->gpr[42]._u32[2], ctx->gpr[42]._u32[3]);
                }
                if (k <= 64)
                    fprintf(stderr, "[coll-net] img=110 unlifted pc 0x%05X below a 0x%X dispatch (idx word 0x%04X): empty handler, returning to the link (#%ld)\n",
                            p0, cn_link, ((uint32_t)ctx->ls[0x2300C] << 8) | ctx->ls[0x2300D], k);
                if (k <= 8) {
                    extern unsigned spu_recent_pcs(const spu_context*, uint32_t* out, unsigned max);
                    uint32_t pcs[24]; unsigned n = spu_recent_pcs(ctx, pcs, 24);
                    fprintf(stderr, "[coll-net]   link=0x%05X r16=%08X r82=%08X r17=%08X r42=%08X.%08X.%08X.%08X LS rec[0..31]:", cn_link, ctx->gpr[16]._u32[0], ctx->gpr[82]._u32[0], ctx->gpr[17]._u32[0],
                            ctx->gpr[42]._u32[0], ctx->gpr[42]._u32[1], ctx->gpr[42]._u32[2], ctx->gpr[42]._u32[3]);
                    for (int b = 0; b < 32; b++) fprintf(stderr, "%s%02X", (b & 15) ? "" : " ", ctx->ls[0x23000 + b]);
                    fprintf(stderr, "\n");
                    fprintf(stderr, "[coll-net]   host_depth=%d r1=%08X r2=%08X r3=%08X r4=%08X r5=%08X r6=%08X recent pcs:", (int)ctx->host_depth,
                            ctx->gpr[1]._u32[0], ctx->gpr[2]._u32[0], ctx->gpr[3]._u32[0], ctx->gpr[4]._u32[0], ctx->gpr[5]._u32[0], ctx->gpr[6]._u32[0]);
                    for (unsigned i = 0; i < n; i++) fprintf(stderr, " %05X", pcs[i]);
                    fprintf(stderr, "\n");
                }
                ctx->pc = cn_link;
                g_spu_trampoline_fn = spu_indirect_branch;
                return;
            }
        }
        if (p0 + 3 < SPU_LS_SIZE) {
            uint32_t w = ((uint32_t)ctx->ls[p0] << 24) | ((uint32_t)ctx->ls[p0+1] << 16) |
                         ((uint32_t)ctx->ls[p0+2] << 8) | ctx->ls[p0+3];
            /* A `bisl` through a handler table whose slot is 0 lands here (LS 0
             * holds `stop 0`). ToD's collision SPUs index their tables with a
             * record field (+0x18, and +12 in the broadphase) that a late PUT of
             * the previous job can leave at 0x8000 -- a race the host's thread
             * scheduling opens wide and the hardware's DMA timing keeps shut.
             * Stopping the SPU there leaves the PPU spinning on the job's done
             * bit forever (the intermittent freeze, Cobalia loads). Treat it as
             * an empty handler instead: return to the link. SPU_NULL_CALL_RET=0
             * keeps the old stop. */
            if (p0 == 0 && w == 0u) {
                static int s_ncr = -1;
                if (s_ncr < 0) { const char* e = getenv("SPU_NULL_CALL_RET"); s_ncr = !(e && e[0] == '0'); }
                const uint32_t lr = ctx->gpr[0]._u32[0] & SPU_LS_MASK;
                uint32_t wl = 0;
                if (lr >= 4 && lr + 3 < SPU_LS_SIZE)
                    wl = ((uint32_t)ctx->ls[lr-4] << 24) | ((uint32_t)ctx->ls[lr-3] << 16) |
                         ((uint32_t)ctx->ls[lr-2] << 8) | ctx->ls[lr-1];
                if (s_ncr && lr && (wl >> 21) == 0x1A9u) {   /* the word before the link is a bisl */
                    static volatile long n_ncr = 0; long k = ++n_ncr;
                    if (k <= 16 || (k & (k - 1)) == 0)
                        fprintf(stderr, "[spu] img=%d bisl to LS 0 (null handler) returning to 0x%05X instead of stopping (#%ld)\n",
                                ctx->image_id, lr, k);
                    ctx->pc = lr;
                    g_spu_trampoline_fn = spu_indirect_branch;
                    return;
                }
            }
            if ((w >> 21) == 0u) {                 /* stop / stopd */
                ctx->stop_code = w & 0x3FFFu;
                static int _n = 0;
                if (_n < 8 && ++_n) {
                    fprintf(stderr, "[spu] img=%d exit: synthesised stop 0x%04X at LS 0x%05X\n",
                            ctx->image_id, ctx->stop_code, p0);
                    /* A zero word is also what unloaded local store reads as, so
                     * say where the branch came from: the last host-function
                     * entries on this thread, newest last. */
                    extern unsigned spu_recent_pcs(const spu_context*, uint32_t* out, unsigned max);
                    uint32_t pcs[32]; unsigned n = spu_recent_pcs(ctx, pcs, 32);
                    fprintf(stderr, "[spu]   recent pcs:");
                    for (unsigned i = 0; i < n; i++) fprintf(stderr, " %05X", pcs[i]);
                    fprintf(stderr, " | lr=0x%05X r3=0x%08X r4=0x%08X\n",
                            ctx->gpr[0]._u32[0] & SPU_LS_MASK, ctx->gpr[3]._u32[0], ctx->gpr[4]._u32[0]);
                    static volatile LONG s_dump_lock;
                    if (g_spu_get_ring_on > 0)
                        while (_InterlockedExchange(&s_dump_lock, 1)) Sleep(1);
                    if (g_spu_get_ring_on > 0) {
                        fprintf(stderr, "[stop-dump] img=%d gpr preferred words:", ctx->image_id);
                        for (int g = 0; g < 128; g++)
                            fprintf(stderr, "%s r%d=%08X", (g % 8) ? "" : "\n[stop-dump] ", g, ctx->gpr[g]._u32[0]);
                        fprintf(stderr, "\n");
                        /* Last GETs, newest last: what landed in LS, and what the
                         * same EA holds now. Equal = the command was already bad
                         * in memory; different = it was read while being written. */
                        spu_get_ring_t* r = spu_get_ring_for(ctx);
                        unsigned cnt = r->n < SPU_GET_RING ? r->n : SPU_GET_RING;
                        for (unsigned k = 0; k < cnt; k++) {
                            unsigned i = (r->n - cnt + k) & (SPU_GET_RING - 1);
                            uint32_t lsa = r->lsa[i] & SPU_LS_MASK, ea = r->ea[i];
                            uint32_t sz = r->size[i] > 192 ? 192 : r->size[i];
                            fprintf(stderr, "[stop-dump] get #%u pc=0x%05X lsa=0x%05X ea=0x%08X size=%u\n",
                                    r->n - cnt + k, r->pc[i], lsa, ea, r->size[i]);
                            if (k + 8 < cnt) continue;   /* bytes only for the newest 8 */
                            for (uint32_t o = 0; o < sz; o += 16) {
                                fprintf(stderr, "[stop-dump]   +%03X ls ", o);
                                for (int b = 0; b < 16; b++) fprintf(stderr, "%02X", ctx->ls[(lsa + o + b) & SPU_LS_MASK]);
                                if (vm_base && ea < 0xE0000000u) {
                                    fprintf(stderr, " mem ");
                                    for (int b = 0; b < 16; b++) fprintf(stderr, "%02X", vm_base[ea + o + b]);
                                }
                                fprintf(stderr, "\n");
                            }
                        }
                        /* Top of local store, where the stack lives: what overwrote it? */
                        for (uint32_t o = 0x3FB00; o < SPU_LS_SIZE; o += 16) {
                            fprintf(stderr, "[stop-dump] stack 0x%05X ", o);
                            for (int b = 0; b < 16; b++) fprintf(stderr, "%02X", ctx->ls[o + b]);
                            fprintf(stderr, "\n");
                        }
                        fflush(stderr);
                        _InterlockedExchange(&s_dump_lock, 0);
                    }
                }
                spu_halt(ctx);
                return;
            }
        }
    }

    if (!fn && !ctx->policy_mode && ctx->pc >= SPU_JM2_KERNEL_BASE) {
        static uint32_t seen[16]; static int n_seen = 0;
        int known = 0;
        for (int i = 0; i < n_seen; i++) if (seen[i] == ctx->pc) { known = 1; break; }
        if (!known && n_seen < 16) {
            seen[n_seen++] = ctx->pc;
            fprintf(stderr, "[spu] img=%d branched into unlifted LS 0x%05X "
                    "(lr=0x%05X) -- ending the job\n",
                    ctx->image_id, ctx->pc, ctx->gpr[0]._u32[0] & SPU_LS_MASK);
            { fprintf(stderr, "      last dispatched PCs (oldest first):");
              for (unsigned q = 0; q < 8; q++) {
                  unsigned idx = (ctx->dbg_pch_n + q) & 7u;
                  if (ctx->dbg_pch_n > q || ctx->dbg_pch[idx]) fprintf(stderr, " 0x%05X", ctx->dbg_pch[idx]);
              }
              fprintf(stderr, "%c", 10); }
            /* Is there real code at the target, or is the pc garbage? Eight
             * words at the target and at the return address separate "the lift
             * missed a function" from "this branch should never have happened". */
            { uint32_t a[2]; a[0] = ctx->pc & SPU_LS_MASK;
              a[1] = ctx->gpr[0]._u32[0] & SPU_LS_MASK;
              for (int k = 0; k < 2; k++) {
                  fprintf(stderr, "      LS[0x%05X]:", a[k]);
                  for (uint32_t o = 0; o < 32 && a[k] + o + 3 < SPU_LS_SIZE; o += 4)
                      fprintf(stderr, " %02X%02X%02X%02X",
                              ctx->ls[a[k]+o], ctx->ls[a[k]+o+1],
                              ctx->ls[a[k]+o+2], ctx->ls[a[k]+o+3]);
                  fprintf(stderr, "\n");
              } }
            fflush(stderr);
        }
        /* SPU_INTERP_UNLIFTED=1: if real code is present at the target, run it
         * through the interpreter instead of ending the job. No lift can exist
         * for a module the title loads at runtime -- You Don't Know Jack's FMOD
         * mixer relocates a DSP plugin into local store and calls it -- and the
         * interpreter rejoins the compiled path as soon as it reaches a lifted
         * address, which is what the plugin's return does. Opt-in so titles
         * that reach here on a genuinely bad pc keep the loud stop. */
        { uint32_t p = ctx->pc & SPU_LS_MASK;
          uint32_t w0 = ((uint32_t)ctx->ls[p] << 24) | ((uint32_t)ctx->ls[p+1] << 16) |
                        ((uint32_t)ctx->ls[p+2] << 8) | ctx->ls[p+3];
          static int s_iu = -1;
          if (s_iu < 0) { const char* e = getenv("SPU_INTERP_UNLIFTED"); s_iu = e ? 1 : 0; }
          if (s_iu && w0) {
              static int _n = 0;
              if (_n < 8 && ++_n)
                  fprintf(stderr, "[spu] img=%d interpreting unlifted LS 0x%05X "
                          "(runtime-loaded code)\n", ctx->image_id, p);
              spu_interp_run(ctx, p);
              return;
          } }
        spu_halt(ctx);
        return;
    }
    /* WWS job-code overlay match: the PM (image 2) streams a job module into the
     * code buffer (base = LS[0x1320]) and branches to base+entryOffset. Its 16-byte
     * ila header identifies which of the 89 lifted job modules it is; match it and
     * mark that overlay resident so the module's lifted functions dispatch. The
     * module arrives via a DMA LIST, so the single-GET note_get hook never fired. */
    if (!fn && ctx->image_id == 2 && ctx->pc >= 0x4000 && ctx->pc < 0x3E000) {
        uint32_t codeBase = ((uint32_t)ctx->ls[0x1320] << 24) | ((uint32_t)ctx->ls[0x1321] << 16) |
                            ((uint32_t)ctx->ls[0x1322] << 8) | ctx->ls[0x1323];
        if (codeBase + 16 <= SPU_LS_SIZE) {
            extern int spu_overlay_match_sig(const uint8_t hdr[16]);
            int img = spu_overlay_match_sig(&ctx->ls[codeBase]);
            if (img) {
                if (ctx->resident_ovl != img) {
                    ctx->resident_ovl = img;
                    static int _n = 0; if (_n < 32 && ++_n)
                        fprintf(stderr, "[spu-ovl] WWS job module resident: codeBase=0x%05X "
                                "pc=0x%05X -> overlay image %d\n", codeBase, ctx->pc & SPU_LS_MASK, img);
                }
                fn = spu_lookup(ctx->pc, img);
            }
        }
    }
    /* WWS job-code fallback: the jobmanager PM (image 2) stages each job's
     * code into an LS buffer and branches into it. The staged blobs are
     * lifted as their own images at their LS residence addresses (image 1 =
     * job_wws_7BD900 at the 0x14400-region buffer); the PM's own table can
     * never contain them. On a policy-module miss inside the job-buffer
     * range, retry against the job image before falling to the interpreter. */
    /* WWS job-code fallback -- but ONLY when real code is staged at the target.
     * A PM job whose code buffer is empty (the {0xA400,0} null-code placeholder)
     * must NOT resolve to image-1's unrelated function that happens to share the
     * file offset: that ran garbage, the job did no work, and its ring slot
     * never advanced -> the loader's job ring deadlocked. An empty buffer falls
     * through to the null-job handling below. */
    if (!fn && ctx->image_id == 2 && ctx->policy_mode &&
        ctx->pc >= 0x4000 && ctx->pc < 0x3E000) {
        uint32_t w0 = ((uint32_t)ctx->ls[ctx->pc] << 24) | ((uint32_t)ctx->ls[ctx->pc+1] << 16) |
                      ((uint32_t)ctx->ls[ctx->pc+2] << 8) | ctx->ls[ctx->pc+3];
        if (w0 != 0)
            fn = spu_lookup(ctx->pc, 1);
    }
    /* Null-code PM job (empty buffer): return control to the dispatcher's own
     * host-call bracket so its post-job continuation runs the completion
     * bookkeeping -- exactly as a real job that did nothing and returned.
     * (Falling to the SMC microstep would log a spurious BRANCH-TO-0 on the
     * 0x00000000 word; this is the same outcome, quiet.) */
    if (!fn && ctx->image_id == 2 && ctx->policy_mode &&
        ctx->pc >= 0x4000 && ctx->pc < 0x3E000 &&
        ((uint32_t)ctx->ls[ctx->pc] | ctx->ls[ctx->pc+1] |
         ctx->ls[ctx->pc+2] | ctx->ls[ctx->pc+3]) == 0) {
        static int _n = 0;
        if (_n < 8 && ++_n) {
            /* Ground-truth dump of the WWS code-buffer resolution at dispatch
             * (wwsjob-rosetta offsets). jobEntry = lsaJobCodeBuffer + [buf+0x10];
             * lsaJobCodeBuffer = Gbt_bufferPageNum<<10 from the RunJob command.
             * If the code buffer really is 0x4000 but empty -> the code LOAD is
             * the bug; if lsaJobCodeBuffer should be 0x14400 -> the RunJob
             * buffer-set resolution is. Dump both + the bufferSetArray rows. */
            uint32_t lsaCode = (uint32_t)ctx->ls[0x1330]<<24 | ctx->ls[0x1331]<<16 |
                               ctx->ls[0x1332]<<8 | ctx->ls[0x1333];
            uint32_t runJob  = (uint32_t)ctx->ls[0x12E0]<<24 | ctx->ls[0x12E1]<<16 |
                               ctx->ls[0x12E2]<<8 | ctx->ls[0x12E3];
            const uint8_t* bs = ctx->ls + 0xDF0;   /* g_WwsJob_bufferSetArray */
            fprintf(stderr, "[pm-jobres] pc=0x%05X lsaJobCodeBuffer(0x1330)=0x%08X "
                    "runJobNum(0x12E0)=0x%08X code@14400:%02X%02X%02X%02X\n"
                    "           bufferSetArray[0xDF0]: %02X%02X%02X%02X %02X%02X%02X%02X "
                    "%02X%02X%02X%02X %02X%02X%02X%02X\n",
                    ctx->pc & SPU_LS_MASK, lsaCode, runJob,
                    ctx->ls[0x14400],ctx->ls[0x14401],ctx->ls[0x14402],ctx->ls[0x14403],
                    bs[0],bs[1],bs[2],bs[3], bs[4],bs[5],bs[6],bs[7],
                    bs[8],bs[9],bs[10],bs[11], bs[12],bs[13],bs[14],bs[15]);
        }
        return;   /* enclosing SPU_DRAIN bracket resumes the dispatcher */
    }
    /* JOB-BODY EXECUTION probe: the WWS job manager enters a staged job via
     * `bie pJobCode` (changeloadtorunjob.spu), an INDIRECT branch to
     * lsaJobCodeBuffer + entryOffset. A successful dispatch with the pc inside
     * the job code buffer and a jobmod overlay resident (image ids 200+) is the
     * proof the lifted job module actually runs. Env SPU_JOBEXEC=1. */
    /* SPU_JOBTRACE=1: log EVERY dispatcher entry while a job module (overlay
     * id >= 200) is resident -- the job's cross-function control-flow trail.
     * The pm-flow drain hook can't see a job that performs no channel ops
     * (LBP's frozen job runs as a complete no-op: entry, no DMAs, return);
     * this names its early-out path instead. */
    { static int s_jt = -1;
      if (s_jt < 0) { const char* e = getenv("SPU_JOBTRACE"); s_jt = e ? 1 : 0; }
      if (s_jt && ctx->resident_ovl >= 200) { static int _n = 0;
        uint32_t p = ctx->pc & SPU_LS_MASK;
        /* Full trail, but only start logging once the job body proper begins
         * (pc in the job code buffer >= 0x4000) so the cap isn't spent on the
         * PM code the job calls back into -- and keep logging thereafter. */
        static int _armed = 0;
        if (p >= 0x4000) _armed = 1;
        if (_armed && (_n < 1500 && ++_n)) {
            fprintf(stderr, "[jobtrace] pc=0x%05X lr=0x%05X%s",
                    p, ctx->gpr[0]._u32[0] & SPU_LS_MASK,
                    fn ? "" : " (no lift)");
            /* JobApi entries: args r3-r5. Post-job return 0x3308: result r3 +
             * the job's saved GetBufferTag result r80. */
            if (p == 0x2F30 || p == 0x1700 || p == 0x1770 || p == 0x17C8)
                fprintf(stderr, " args r3=%08X.%08X r4=%08X r5=%08X",
                        ctx->gpr[3]._u32[0], ctx->gpr[3]._u32[1],
                        ctx->gpr[4]._u32[0], ctx->gpr[5]._u32[0]);
            if (p == 0x3308 || p == 0x3258)
                fprintf(stderr, " r3=%08X.%08X.%08X.%08X r80=%08X.%08X",
                        ctx->gpr[3]._u32[0], ctx->gpr[3]._u32[1],
                        ctx->gpr[3]._u32[2], ctx->gpr[3]._u32[3],
                        ctx->gpr[80]._u32[0], ctx->gpr[80]._u32[1]);
            /* Mid-GetBufferTag: r8 = the set/buf id bytes computed at 0x2F30
             * entry; if wrong here, something clobbered it ACROSS the
             * GetLogicalBuffer call (interrupt delivery?). */
            if (p == 0x2F6C)
                fprintf(stderr, " r8=%08X r10=%08X r13=%08X r15=%08X int=%d",
                        ctx->gpr[8]._u32[0], ctx->gpr[10]._u32[0],
                        ctx->gpr[13]._u32[0], ctx->gpr[15]._u32[0],
                        ctx->int_enable);
            if (p == 0x3030)   /* final BufferTag in r7 (result) */
                fprintf(stderr, " BufferTag r7=%08X.%08X",
                        ctx->gpr[7]._u32[0], ctx->gpr[7]._u32[1]);
            fprintf(stderr, "\n");
        }
      } }
    /* LBP_HLE_JOBDONE=1: HLE the WWS job COMPLETION. Our lifted SPURS PM never
     * reaches the store/DecrementDependency phase (the multi-SPU barrier never
     * converges), so a job's completion word -- passed by the PPU as param[4],
     * an EA in the 0x470Axxxx pool, embedded in the job's command list at LS
     * ~0xC00 -- is never written, and the PPU JobManagerWorker (guest-fn
     * 0x521BD4) spins on it forever (loading freeze). When a job body actually
     * dispatches (overlay >=200, entry 0x4A40), scan its staged command list
     * for the completion EA and write it nonzero, unblocking the worker. This
     * fakes ONLY the done-signal (not the job's data DMA), so it tests how far
     * the boot gets once loading stops deadlocking. */
    if (ctx->pc == 0x4A40 && ctx->resident_ovl >= 200) {
        static int s_hd = -1;
        if (s_hd < 0) { const char* e = getenv("LBP_HLE_JOBDONE"); s_hd = e ? 1 : 0; }
        if (s_hd) {
            extern uint8_t* vm_base;
            int wrote = 0;
            for (uint32_t o = 0xC00; o + 4 <= 0xE80 && wrote < 4; o += 4) {
                uint32_t w = (ctx->ls[o]<<24)|(ctx->ls[o+1]<<16)|(ctx->ls[o+2]<<8)|ctx->ls[o+3];
                if ((w & 0xFFFF0000u) == 0x470A0000u && vm_base) {
                    /* write nonzero to the completion word in main RAM (BE) */
                    if (vm_base[w]==0 && vm_base[w+1]==0 && vm_base[w+2]==0 && vm_base[w+3]==0) {
                        vm_base[w+3] = 1;
                        static int _n = 0; if (_n < 24 && ++_n)
                            fprintf(stderr, "[hle-jobdone] wrote completion 0x%08X=1 (job overlay %d)\n",
                                    w, ctx->resident_ovl);
                        wrote++;
                    }
                }
            }
        }
    }
    if (fn && ctx->pc >= 0x4000 && ctx->resident_ovl >= 200) {
        static int s_je = -1;
        if (s_je < 0) { const char* e = getenv("SPU_JOBEXEC"); s_je = e ? 1 : 0; }
        if (s_je) { static int _n = 0; if (_n < 24 && ++_n)
            fprintf(stderr, "[jobexec] DISPATCH job code pc=0x%05X overlay=%d "
                    "runJobNum=0x%02X%02X%02X%02X\n",
                    ctx->pc & SPU_LS_MASK, ctx->resident_ovl,
                    ctx->ls[0x12E0], ctx->ls[0x12E1], ctx->ls[0x12E2], ctx->ls[0x12E3]);
          /* SPU_JOBEXEC_DUMP=<path>: write the full 256KB LS the first time a job
           * entry (pc==codeBuffer+entryOffset) dispatches, to diff vs the RPCS3
           * oracle SPU4 dump (why our job bails where the oracle's runs). */
          static int s_dumped = 0;
          if (!s_dumped && ctx->pc == 0x4A40) {
              const char* dp = getenv("SPU_JOBEXEC_DUMP");
              if (dp) { FILE* f = fopen(dp, "wb");
                  if (f) { fwrite(ctx->ls, 1, SPU_LS_SIZE, f); fclose(f);
                      fprintf(stderr, "[jobexec] dumped LS at job entry -> %s\n", dp); }
                  s_dumped = 1; } } }
    }
    if (ib_vfy) {
        spu_fn ex = t_ib_expect; t_ib_expect = NULL;
        if (!ex) __atomic_add_fetch(&s_ibv_nopeek, 1, __ATOMIC_RELAXED);
        else if (ex == fn) __atomic_add_fetch(&s_ibv_ok, 1, __ATOMIC_RELAXED);
        else { long b = __atomic_add_fetch(&s_ibv_bad, 1, __ATOMIC_RELAXED);
            if (b <= 24) fprintf(stderr, "[ib-fast] VERIFY MISMATCH #%ld pc=0x%05X img=%d ovl=%d fast=%p full=%p\n", b, ctx->pc & SPU_LS_MASK, ctx->image_id, ctx->resident_ovl, (void*)ex, (void*)fn); }
        long tot = s_ibv_ok + s_ibv_bad + s_ibv_nopeek;
        if ((tot & 0x3FFFFF) == 0) fprintf(stderr, "[ib-fast] VERIFY: %ld agree, %ld DISAGREE, %ld not-probed (miss/special)\n", s_ibv_ok, s_ibv_bad, s_ibv_nopeek);
    }
    if (fn && ib_ent) {
        spu_ib_key k2; spu_ib_make_key(ctx, &k2);
        if (memcmp(&k2, &ib_key, sizeof k2) == 0) {       /* the resolution did not change the mapping */
            if (ib_cached) {
                static volatile long s_n, s_bad;
                long n = __atomic_add_fetch(&s_n, 1, __ATOMIC_RELAXED);
                if (ib_cached != fn) { long b = __atomic_add_fetch(&s_bad, 1, __ATOMIC_RELAXED);
                    if (b <= 24) fprintf(stderr, "[ib-check] MISMATCH #%ld pc=0x%05X img=%d ovl=%d cached=%p(owner %d) resolved=%p(owner %d) gen=%u/%u dyn_last=%d dynnow=%d\n",
                                         b, ctx->pc & SPU_LS_MASK, ctx->image_id, ctx->resident_ovl, (void*)ib_cached, ib_ent->owner, (void*)fn, code_owner,
                                         ib_ent->gen, t_ib_gen, t_dyn_last, (s_dyn_n && g_spu_dyn_page[(ctx->pc & SPU_LS_MASK) >> 8]) ? spu_dynovl_find(ctx, ctx->pc & SPU_LS_MASK) : -1); }
                if ((n & 0x3FFFFF) == 0) fprintf(stderr, "[ib-check] %ld hits compared, %ld mismatches\n", n, s_bad);
            }
            if (s_ib_fast >= 1 && ib_ent->fn && ib_ent->gen == t_ib_gen)
                t_ib[(unsigned)(ib_ent - t_ib) ^ 1u] = *ib_ent;   /* 2-way: the live entry moves to the neighbour slot */
            ib_ent->key = ib_key; ib_ent->fn = fn; ib_ent->gen = t_ib_gen; ib_ent->owner = code_owner;
            { uint32_t z = 0; for (unsigned i = 0; i < 4; i++) z |= ib_key.rc[i][0] | ib_key.rc[i][1] | ib_key.rc[i][2]; ib_ent->rcz = (z == 0); }
        }
    }
    if (fn) {
        spu_check_stack_reset(ctx, fn);
        /* MUSTTAIL: a guest loop that iterates through an indirect branch (the
         * Bink decoder's per-command dispatch does) must not grow the host
         * stack -- a plain call here leaked a resolver+callee frame per
         * iteration and blew the thread stack ~4k iterations into the first
         * really-decoding movie frame (silent 0x80000001 death). */
#if defined(__clang__)
        __attribute__((musttail)) return fn(ctx);
#else
        fn(ctx);
        return;
#endif
    }
    /* wwsjob JOB-CODE entry probe: the PM stages each job's code into a
     * buffer above its static image (0x3700..) and branches into it. No lift
     * exists at those pcs (the code arrives at runtime), so the branch lands
     * here. Log the entry + leading bytes -- the bytes identify WHICH job
     * blob was staged (match against lifted job images for dispatch). */
    /* Staged-job "return to kernel": a jm2-protocol job binary ends by
     * jumping to LS 0 (the jm2 kernel's home on real hardware). Inside the
     * wwsjob PM's LS, address 0 holds the interrupt VECTOR instead -- letting
     * the jump proceed runs the interrupt handler as the job's continuation
     * and wedges the pipeline (lanes freeze mid-queue). The PM called the job
     * with a return link in r0 (0x31C8 sets 0x3258); route the kernel-return
     * there: the PM resumes its post-job flow, which is exactly what the jm2
     * kernel hand-back does on hardware. */
    if (ctx->image_id == 2 && ctx->policy_mode && ctx->pc == 0 &&
        (ctx->gpr[0]._u32[0] & SPU_LS_MASK) >= 0xA00 &&
        (ctx->gpr[0]._u32[0] & SPU_LS_MASK) < 0x3700) {
        static int _n = 0;
        if (_n < 8 && ++_n)
            fprintf(stderr, "[wws-jobret] staged job returned to kernel (LS 0); "
                    "resuming PM at link 0x%05X\n", ctx->gpr[0]._u32[0] & SPU_LS_MASK);
        ctx->pc = ctx->gpr[0]._u32[0] & SPU_LS_MASK;
        spu_indirect_branch(ctx);      /* re-dispatch at the rewritten pc */
        return;
    }

    /* A SPURS *job* that branches back to LS 0 has RETURNED TO THE JOB MANAGER.
     * That is how a jm2 job signals completion: its crt tail-jumps to the
     * resident manager, which on hardware loads the next job and sets up r3/r4
     * afresh. We load the job at LS 0 and have no manager, so the jump lands on
     * the job's OWN entry and it runs a second lap -- with dead registers.
     *
     * Tokyo Jungle showed exactly that: lap one is correct at every step and
     * does the real work, then the crt jumps to 0 and lap two re-reads its
     * parameters through a now-zero context pointer (hence the reads of LS 0x20
     * and the DMA addresses that were really its own opcodes) and spins forever.
     * The work was already finished before the jump; the second lap is noise.
     *
     * The initial entry is called directly, not through here, so any arrival at
     * LS 0 in this path is a re-entry. Treat it as job end. */
    if (!ctx->policy_mode && ctx->pc == 0 && ctx->image_id > 0) {
        static int _n = 0;
        if (_n < 8 && ++_n)
            fprintf(stderr, "[spurs-job] img=%d returned to the job manager "
                    "(branch to LS 0) -- job complete\n", ctx->image_id);
        spu_halt(ctx);
        return;
    }
    if (ctx->image_id == 2 &&
        ((ctx->pc >= 0x3700 && ctx->pc < 0x3FE80) ||
         (ctx->pc < 0xA00 && ctx->pc != SPURS_PM_EXIT_TO_KERNEL_LS &&
          ctx->pc != SPURS_PM_SELECT_WORKLOAD_LS))) {
        static int _n = 0;
        if (_n < 12) {
            _n++;
            const uint8_t* p = ctx->ls + (ctx->pc & SPU_LS_MASK);
            fprintf(stderr, "[wws-jobentry] pc=0x%05X lr=0x%05X bytes:"
                    " %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X\n",
                    ctx->pc & SPU_LS_MASK, ctx->gpr[0]._u32[0] & SPU_LS_MASK,
                    p[0],p[1],p[2],p[3], p[4],p[5],p[6],p[7],
                    p[8],p[9],p[10],p[11], p[12],p[13],p[14],p[15]);
            /* Companion state: the OTHER load target (the real 0x100-byte
             * loads land at 0x14400 -- possibly the actual job code, with the
             * entry computed from the wrong slot's base) + the job records
             * the entry math reads (0x1320 base block, 0x1440 batch record). */
            { const uint8_t* q = ctx->ls + 0x14400;
              const uint8_t* r1 = ctx->ls + 0x1320;
              const uint8_t* r2 = ctx->ls + 0x1440;
              fprintf(stderr, "[wws-jobentry]   LS14400: %02X%02X%02X%02X %02X%02X%02X%02X"
                      "  LS1320: %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X"
                      "  LS1440: %02X%02X%02X%02X %02X%02X%02X%02X\n",
                      q[0],q[1],q[2],q[3], q[4],q[5],q[6],q[7],
                      r1[0],r1[1],r1[2],r1[3], r1[4],r1[5],r1[6],r1[7],
                      r1[8],r1[9],r1[10],r1[11], r1[12],r1[13],r1[14],r1[15],
                      r2[0],r2[1],r2[2],r2[3], r2[4],r2[5],r2[6],r2[7]); }
            fflush(stderr);
        }
    }
    /* No lifted function at this PC: it may be RUNTIME-GENERATED code (the
     * WWS jobmanager writes a save/restore stub above its static
     * image and calls it). Interpret the live LS bytes; on success the next
     * branch re-enters lifted code via the trampoline. A relocated PIC blob
     * (spu_dynovl_register_reloc) streamed to a new LS address runs lifted first. */
    if (s_rel_n && spu_dynovl_reloc_enter(ctx))
        return;
    if (spu_smc_microstep(ctx))
        return;
    /* Cap the unresolved-branch log PER IMAGE: a global cap let one noisy
     * image (the FMOD mixer's overlay calls) exhaust it and silently hide
     * every other image's misses -- LBP's loading jobs skipped their command
     * handlers for a whole session without a single log line. */
    { enum { BT0_MAX_IMG = 64, BT0_PER_IMG = 12 };
      static int _bt0[BT0_MAX_IMG];
      unsigned img = (ctx->image_id >= 0 && ctx->image_id < BT0_MAX_IMG)
                     ? (unsigned)ctx->image_id : 0;
      if (_bt0[img]++ < BT0_PER_IMG)
        fprintf(stderr, "[SPU] BRANCH-TO-0 unresolved pc=0x%05X image=%d lr=0x%05X\n",
                ctx->pc, ctx->image_id, ctx->gpr[0]._u32[0] & SPU_LS_MASK); }
    /* One-shot: the FMOD null-handler DSP node carries a PPU descriptor EA at
     * node+0x14 (observed 0x93C3C0). Dump it to identify which plugin/unit
     * type never got its SPU code streamed (env SPU_DSPDESC=<hex ea>). */
    { static int _d = -1; static uint32_t _ea = 0;
      if (_d < 0) { const char* e = getenv("SPU_DSPDESC");
        _ea = e ? (uint32_t)strtoul(e, 0, 16) : 0; _d = _ea ? 1 : 0; }
      if (_d == 1) { _d = 2;
        extern uint8_t* vm_base;
        fprintf(stderr, "[dspdesc] RAM[0x%08X]:", _ea);
        for (int k = 0; k < 0x60; k += 4)
            fprintf(stderr, " %02X%02X%02X%02X", vm_base[_ea+k], vm_base[_ea+k+1],
                    vm_base[_ea+k+2], vm_base[_ea+k+3]);
        fprintf(stderr, "\n"); fflush(stderr); } }
    /* SPU_MISS_DUMP_IMG=<n>: reserve the deep-dump budget for image n's misses
     * (the global 2-shot budget was always consumed by an earlier image's
     * misses, hiding the one under investigation). Unset = old behavior. */
    { static int s_img = -2; static uint32_t s_pcmin = 0;
      if (s_img == -2) { const char* e = getenv("SPU_MISS_DUMP_IMG"); s_img = e ? atoi(e) : -1;
        const char* p = getenv("SPU_MISS_DUMP_PCMIN"); s_pcmin = p ? (uint32_t)strtoul(p,0,0) : 0; }
      static int _n=0;
      uint32_t misspc = ctx->pc & SPU_LS_MASK;
      if ((s_img < 0 || ctx->image_id == s_img) && misspc >= s_pcmin && (_n < 2 && ++_n)) {
        fprintf(stderr, "[SPU] branch-to-0 lr=0x%05X r1=0x%05X\n",
                ctx->gpr[0]._u32[0] & SPU_LS_MASK, ctx->gpr[1]._u32[0] & SPU_LS_MASK);
#ifdef _WIN32
        void* frames[24]; unsigned short fn = RtlCaptureStackBackTrace(0, 24, frames, NULL);
        char* base = (char*)GetModuleHandleA(NULL);
        fprintf(stderr, "[SPU] host bt RVAs:");
        for (unsigned short i = 0; i < fn; i++)
            fprintf(stderr, " 0x%zX", (size_t)((char*)frames[i] - base));
        fprintf(stderr, "\n");
#endif
        /* State-diff oracle: dump full LS + all GPRs at the branch-to-0 so it can
         * be compared byte-for-byte against the RPCS3 savestate LS of the same
         * (cri_mpv) task. Path from YDKJ_SPU_LSDUMP, else ./recomp_spu_ls.bin. */
        const char* dp = getenv("YDKJ_SPU_LSDUMP");
        if (!dp || !*dp) dp = "recomp_spu_ls.bin";
        FILE* lf = fopen(dp, "wb");
        if (lf) { fwrite(ctx->ls, 1, SPU_LS_SIZE, lf); fclose(lf);
                  fprintf(stderr, "[SPU] dumped 256KB LS -> %s\n", dp); }
        fprintf(stderr, "[SPU] image_id=%d  GPR dump (r0..r127, hi64:lo64 of each quadword, preferred slot = _u32[0]):\n", ctx->image_id);
        for (int g = 0; g < 128; g++) {
            fprintf(stderr, " r%-3d=%08X %08X %08X %08X", g,
                    ctx->gpr[g]._u32[0], ctx->gpr[g]._u32[1],
                    ctx->gpr[g]._u32[2], ctx->gpr[g]._u32[3]);
            if ((g & 1) == 1) fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");
        /* echo the dispatch chain values the way func_00026DE0 computes them */
        { uint32_t bec0 = ctx->ls[0xBEC0]<<24 | ctx->ls[0xBEC1]<<16 | ctx->ls[0xBEC2]<<8 | ctx->ls[0xBEC3];
          fprintf(stderr, "[SPU] LS[0xBEC0].w0=0x%08X  LS[0x2d4e0:16]=%02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X\n",
            bec0,
            ctx->ls[0x2d4e0],ctx->ls[0x2d4e1],ctx->ls[0x2d4e2],ctx->ls[0x2d4e3],
            ctx->ls[0x2d4e4],ctx->ls[0x2d4e5],ctx->ls[0x2d4e6],ctx->ls[0x2d4e7],
            ctx->ls[0x2d4e8],ctx->ls[0x2d4e9],ctx->ls[0x2d4ea],ctx->ls[0x2d4eb],
            ctx->ls[0x2d4ec],ctx->ls[0x2d4ed],ctx->ls[0x2d4ee],ctx->ls[0x2d4ef]); }
    } }
    ctx->status = SPU_STATUS_STOPPED_BY_HALT;
}

/* ===========================================================================
 * Execution trace (for §3 validation: diff vs RPCS3 SPU interpreter)
 *
 * When the lifter is invoked with --trace, every emitted instruction is
 * surrounded by spu_trace_pc(ctx, PC) before execution and spu_trace_rt(
 * ctx, RT) after, for instructions whose destination is the rt slot. The
 * output is one line per event:
 *
 *     <PC-5hex>                          - PC about to execute
 *       r<rt> <hi-64hex> <lo-64hex>      - register written, post-state
 *
 * Direct to stderr by default; call spu_trace_init(path) once at startup
 * to redirect to a file. The format is intentionally minimal and stable
 * so a small converter can line it up against an RPCS3.log SPU trace.
 * ===========================================================================*/
static FILE* s_trace_fp = NULL;

void spu_trace_init(const char* path)
{
    if (!path || !*path) { s_trace_fp = stderr; return; }
    s_trace_fp = fopen(path, "w");
    if (!s_trace_fp) s_trace_fp = stderr;
}

/* Bounded by design: a --trace lift emits one call PER INSTRUCTION, so an image
 * that spins (a persistent SPURS policy module polling its job queue never
 * returns) would otherwise write stderr until the disk fills. The cap keeps the
 * useful part -- the entry path plus enough revolutions to show what the loop
 * tests -- and the tail IS the loop. SPU_TRACE_MAX=0 restores unbounded.
 * SPU_TRACE_FILE redirects off stderr so the trace does not interleave with the
 * boot log. */
/* Trace output is OPT-IN at runtime: a --trace lift bakes the calls into the
 * generated C forever, so a trace-lifted image shipped in a title build (LBP's
 * job kernel, kept from the WWS verification) would otherwise spew the
 * per-instruction log into every boot. Enable with SPU_TRACE=1 (stderr) or
 * SPU_TRACE_FILE=<path>. */
static long long s_trace_left = -1;     /* -1 uninit, -2 unbounded, 0 off */
/* Set when spu_trace_pc suppressed its line (gates below): the paired
 * spu_trace_rt call for the same instruction must also stay silent. */
static SPU_THREAD_LOCAL int s_trace_suppress;

void spu_trace_pc(spu_context* ctx, uint32_t pc)
{
    (void)ctx;
    /* SPU_TRACE_AT_WALL=1: hold the trace until the PPU's job-barrier probe
     * arms g_barrier_sync_watch, so the budget covers the post-ticket claim/
     * stage pass instead of boot-time idle polling. */
    { static int s_aw = -1;
      if (s_aw < 0) s_aw = getenv("SPU_TRACE_AT_WALL") ? 1 : 0;
      if (s_aw) { extern uint32_t g_barrier_sync_watch;
                  if (!g_barrier_sync_watch) { s_trace_suppress = 1; return; } } }
    /* SPU_TRACE_FLOWCTX=1: trace ONLY the run the SPURS_PM_FLOW tracer armed
     * (the stalled queue's spu0 dispatch) -- post-wall idle dispatches of the
     * other 13 jobmanager wids otherwise exhaust the budget in under a second. */
    { static int s_fc = -1;
      if (s_fc < 0) s_fc = getenv("SPU_TRACE_FLOWCTX") ? 1 : 0;
      if (s_fc) { extern void* volatile g_pm_flow_ctx;
                  if (g_pm_flow_ctx != (void*)ctx) { s_trace_suppress = 1; return; } } }
    /* SPU_TRACE_IMG=<n>: trace only contexts running image n (e.g. 1 = the
     * WWS job binary, to catch the crt corrupting the job's r3/r4 args). */
    { static int64_t s_ti = -2;
      if (s_ti == -2) { const char* e = getenv("SPU_TRACE_IMG");
                        s_ti = e ? atoll(e) : -1; }
      if (s_ti >= 0 && ctx->image_id != (int)s_ti) { s_trace_suppress = 1; return; } }
    s_trace_suppress = 0;
    if (s_trace_left < 0) {
        if (s_trace_left == -1) {
            const char* p = getenv("SPU_TRACE_FILE");
            if ((!p || !*p) && !getenv("SPU_TRACE")) { s_trace_left = 0; return; }
            if (p && *p && !s_trace_fp) spu_trace_init(p);
            const char* m = getenv("SPU_TRACE_MAX");
            s_trace_left = (m && *m) ? atoll(m) : 200000;
            if (s_trace_left == 0) s_trace_left = -2;   /* -2 = unbounded */
        }
    }
    if (!s_trace_fp) s_trace_fp = stderr;
    if (s_trace_left == 0) return;
    if (s_trace_left > 0 && --s_trace_left == 0) {
        fprintf(s_trace_fp, "-- SPU_TRACE_MAX reached; trace stopped --\n");
        fflush(s_trace_fp);
        return;
    }
    fprintf(s_trace_fp, "%05X\n", pc & SPU_LS_MASK);
}

void spu_trace_rt(spu_context* ctx, uint32_t rt)
{
    if (s_trace_suppress) return;          /* paired _pc line was gated out */
    if (s_trace_left == 0) return;         /* tracing off/stopped (see _pc) */
    if (!s_trace_fp) s_trace_fp = stderr;
    u128 v = ctx->gpr[rt & 0x7F];
    fprintf(s_trace_fp, "  r%-3u %016llX %016llX\n",
            (unsigned)(rt & 0x7F),
            (unsigned long long)v._u64[0],
            (unsigned long long)v._u64[1]);
}

/* ---- anim47: appoggi per l'aggancio nativo di wid0 (anim_hle.c) ----
 * spu_hle_ea_ok: l'intervallo EA e' committato (stessa funzione del percorso veloce dei DMA).
 * spu_hle_get_note_free: un GET su [lsa_lo, lsa_hi) NON puo' cambiare lo stato di spu_overlay_note_get (stessi test di
 * mfc_get_note_needed, senza i due bloom per EA/firma: i dati di animazione non sono mai sorgenti di overlay di codice). */
int spu_hle_ea_ok(uint32_t ea, uint32_t size) { return mfc_ea_range_committed(ea, size); }
/* 0 = i GET su [lsa_lo, lsa_hi) non toccano nulla; 1 = una diagnostica vuole ogni GET (rifiutare); 2 = serve chiamare spu_overlay_note_get
 * per ogni GET (span di codice residente o pagine di overlay dinamico nell'intervallo). */
int spu_hle_get_note_mode(const spu_context* spu, uint32_t lsa_lo, uint32_t lsa_hi)
{
    extern volatile uint32_t g_ovl_note_always;
    extern unsigned char g_spu_dyn_page[SPU_LS_SIZE >> 8];
    if (g_ovl_note_always) return 1;
    if (spu->resident_code[0].image_id | spu->resident_code[1].image_id |
        spu->resident_code[2].image_id | spu->resident_code[3].image_id) return 2;
    for (uint32_t pg = lsa_lo >> 8, pe = (lsa_hi - 1) >> 8; pg <= pe; pg++)
        if (g_spu_dyn_page[pg & ((SPU_LS_SIZE >> 8) - 1)]) return 2;
    return 0;
}
void spu_hle_note_get(spu_context* spu, uint32_t ea, uint32_t lsa, uint32_t size) { spu_overlay_note_get(spu, ea, spu->ls + lsa, size); }

#ifdef __cplusplus
}
#endif