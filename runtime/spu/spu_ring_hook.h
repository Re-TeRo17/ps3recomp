/* spu_ring_hook.h (ring103) -- gancio SPU nel prologo di igCollBroadSpu func_00002640 per i ticket ad ANELLO della collisione.
 * La riga SPU_RING_HOOK e' inserita dal build (tools/coll_ring_patch.cmake) in una COPIA del lifted; l'originale non si tocca.
 * Spento (g_spu_ring_on == 0, SPU_COLL_RING non impostato) il costo e' un byte letto. spu_ring_enter (spu_coll_sync.c / spu_coll_ring.inc)
 * ritorna 1 = ticket consegnato al pool nativo (registri/pc/ritorno come il lifted), 0 = esegui il corpo lifted come sempre. */
#ifndef SPU_RING_HOOK_H
#define SPU_RING_HOOK_H
#include "spu_context.h"
#ifdef __cplusplus
extern "C" {
#endif
extern uint8_t g_spu_ring_on;
int spu_ring_enter(spu_context* ctx, void (*self)(spu_context*));
#define SPU_RING_HOOK(ctx, self) \
    do { if (__builtin_expect(g_spu_ring_on != 0, 0) && spu_ring_enter((ctx), (self))) return; } while (0)
#define SPU_RING_RET(ctx) do { \
        (ctx)->pc = (ctx)->gpr[0]._u32[0]; \
        if ((ctx)->host_depth == 0) g_spu_trampoline_fn = spu_indirect_branch; \
        return 1; } while (0)
#ifdef __cplusplus
}
#endif
#endif
