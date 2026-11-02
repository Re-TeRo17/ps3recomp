/*
 * ps3recomp - PPU <-> SPU lock-line coherence
 *
 * An SPU that issues GETLLAR holds a reservation on a 128-byte line and then
 * spins, waiting for someone else to change it. On real hardware a store by
 * ANY other processor to that line kills the reservation and raises
 * SPU_EVENT_LR on the reserving SPU, which is how a parked SPURS idle service
 * learns that the PPU has posted work. Emulated, the two halves live on
 * different host threads with nothing between them, so the SPU half alone gets
 * PUTLLC right (it compares the line before committing) while the PPU half
 * never tells anyone it wrote. The service sleeps through the signal.
 *
 * This is the missing half. Every line an SPU reserves is marked in a bitmap;
 * the PPU store paths (runtime/ppu/ppu_memory.h and runtime/ppu/ppu_loader.cpp)
 * check the bitmap, and on a hit commit the store under the same lock-line lock
 * the SPU's GETLLAR/PUTLLC transaction takes, then notify. That does two things
 * at once: the store can no longer land inside a PUTLLC's compare/commit window
 * (where it would be silently reverted), and the reserving SPU gets its event.
 *
 * The bitmap check is a shift and a load against a global that stays zero until
 * an SPU actually reserves something, so a title that never runs SPU code pays
 * one predictable branch per store.
 */

#ifndef SPU_COHERENCY_H
#define SPU_COHERENCY_H

#include "spu_context.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* SPU external event: lock-line reservation lost. */
#define SPU_EVENT_LR  0x400u

/* The lock-line lock. One process-wide spinlock serializing every lock-line
 * transaction, SPU and PPU alike: the SPU's GETLLAR snapshot and PUTLLC
 * compare-and-commit, and a PPU store or stwcx to a line some SPU has
 * reserved. The critical sections are two 128-byte memcpy and a memcmp. */
void spu_lockline_lock(void);
void spu_lockline_unlock(void);

/* Mark `ea`'s 128-byte line as SPU-reserved and remember `ctx` as a possible
 * reservation holder. Called from the SPU's GETLLAR with the lock held.
 *
 * Lines are marked and never unmarked. A line the SPURS kernel reserves is
 * reserved again microseconds later, so unmarking would buy nothing; a stale
 * mark costs one locked PPU store to a line the kernel owns anyway. */
void spu_coh_reserve(spu_context* ctx, uint32_t ea);

/* Is `addr`'s line marked? The PPU store fast path -- a range-free O(1) bitmap
 * lookup guarded by a global that is zero until the first GETLLAR. */
int spu_coh_is_reserved(uint32_t addr);

/* A store by the PPU landed on `ea`'s line. Raise SPU_EVENT_LR on every SPU
 * whose live reservation covers it, drop that reservation, and wake the host
 * thread in case it is parked in `rdch SPU_RdEventStat`.
 *
 * Expects the lock-line lock held, so that the store this is reporting and the
 * reservation state it clears are seen by an SPU's PUTLLC as one transaction. */
void spu_coh_notify_write(uint32_t ea);
/* Remove a context from the reserving set. MUST be called before a
 * transient (stack-allocated) spu_context goes out of scope. */
void spu_coh_unregister(spu_context* ctx);
void spu_coh_forget_range(uintptr_t lo, uintptr_t hi);  /* thread exit */
/* As above, but leaves `self`'s own reservation alone: an SPU's own MFC
 * write does not cost it its reservation on hardware. */
void spu_coh_notify_write_except(uint32_t ea, const void* self);

/* Count of LR events delivered. A parked SPURS service that never wakes with
 * this at zero is a coherence miss; nonzero moves the question downstream. */
extern unsigned long g_spu_lr_raise;

/* coll37 tracer (SPU_C37=1, spu_c37.inc): g_c37_on -1 = undecided, 0 off, 1 on. */
extern volatile int g_c37_on;
void c37_dma(const spu_context* spu, uint32_t lsa, uint64_t ea, uint32_t size, uint32_t cmd, int post);
void c37_atomic(const spu_context* ctx, uint32_t cmd, int handled);
void c37_atomic_pre(const spu_context* ctx);
void c37_atomic_post(void);
void c37_ppu_w(uint32_t a, uint64_t v, int width, uint32_t fn, uint32_t lr, unsigned tid);
void c37_wait(const spu_context* ctx, uint32_t channel, int end, double us);
void c37_notify(uint32_t ea, int writer_img, uint32_t wm, uintptr_t ra);
long long c37_stale(const spu_context* ctx, uint32_t rec_ea);
void c37_stale_done(long long mark, uint32_t rereads, const uint8_t* mem);

#ifdef __cplusplus
}
#endif

#endif /* SPU_COHERENCY_H */
