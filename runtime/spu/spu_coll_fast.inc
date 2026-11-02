/* spu_coll_fast.inc (coll101) -- operazioni "veloci" per il ticket di collisione NATIVO (spu_coll_sync.c). DERIVATO dal gioco solo
 * indirettamente (non contiene codice SPU): sono versioni piu' rapide, bit-identiche, delle operazioni di spu_helpers.h.
 * Da includere DOPO spu_helpers.h e PRIMA dei file generati spu_coll_l2_gen_*.inc.
 *
 * 1) fa/fs/fm/fma/fms/fnms: gli helper originali testano g_spu_rz (variabile THREAD-LOCAL: ~4 istruzioni e 2 load per OGNI operazione,
 *    e il compilatore non puo' sollevare il test perche' le store a R[] possono aliasare un int). Il ticket nativo gira SEMPRE con
 *    g_spu_rz=1 e MXCSR=RZ|DAZ|FTZ (impostati da spu_coll_sync_run), quindi il ramo "esatto" e' morto: qui si chiama direttamente
 *    spu__rz_fin (che resta identico: ricade nel percorso esatto sui risultati fuori intervallo).
 * 2) fcgt/fceq/fcmgt/fcmeq: gli originali sono cicli scalari a 4 corsie con chiavi a 64 bit; qui versione SSE2 su 4 corsie.
 *    chiave(x) = (esp(x)==0) ? 0 : (segno ? -m : m), m = x & 0x7FFFFFFF  (stesso valore dell'originale; sta in int32).
 * Toggle: SPU_COLL_FAST=0 (env) -> il file e' incluso con COLL_FAST_OPS=0 nella copia "lenta" del motore. */
#ifndef COLL_FAST_INC
#define COLL_FAST_INC
#include <immintrin.h>
static inline __attribute__((always_inline)) __m128i cf_ld(u128 a) { return _mm_loadu_si128((const __m128i*)a._u32); }
static inline __attribute__((always_inline)) u128 cf_st(__m128i v) { u128 r; _mm_storeu_si128((__m128i*)r._u32, v); return r; }
/* magnitudine con i denormali (esp==0) a zero */
static inline __attribute__((always_inline)) __m128i cf_mag(__m128i x)
{
    const __m128i m = _mm_and_si128(x, _mm_set1_epi32(0x7FFFFFFF));
    const __m128i ez = _mm_cmpeq_epi32(_mm_and_si128(x, _mm_set1_epi32(0x7F800000)), _mm_setzero_si128());
    return _mm_andnot_si128(ez, m);
}
static inline __attribute__((always_inline)) __m128i cf_key(__m128i x)
{
    const __m128i m = cf_mag(x), s = _mm_srai_epi32(x, 31);
    return _mm_sub_epi32(_mm_xor_si128(m, s), s);
}
static inline __attribute__((always_inline)) u128 cf_fcgt(u128 a, u128 b) { return cf_st(_mm_cmpgt_epi32(cf_key(cf_ld(a)), cf_key(cf_ld(b)))); }
static inline __attribute__((always_inline)) u128 cf_fceq(u128 a, u128 b) { return cf_st(_mm_cmpeq_epi32(cf_key(cf_ld(a)), cf_key(cf_ld(b)))); }
static inline __attribute__((always_inline)) u128 cf_fcmgt(u128 a, u128 b) { return cf_st(_mm_cmpgt_epi32(cf_mag(cf_ld(a)), cf_mag(cf_ld(b)))); }
static inline __attribute__((always_inline)) u128 cf_fcmeq(u128 a, u128 b) { return cf_st(_mm_cmpeq_epi32(cf_mag(cf_ld(a)), cf_mag(cf_ld(b)))); }

#undef spu_fcgt
#undef spu_fceq
#undef spu_fcmgt
#undef spu_fcmeq
#define spu_fcgt(a, b)  cf_fcgt((a), (b))
#define spu_fceq(a, b)  cf_fceq((a), (b))
#define spu_fcmgt(a, b) cf_fcmgt((a), (b))
#define spu_fcmeq(a, b) cf_fcmeq((a), (b))

#ifdef spu_fa   /* percorso RZ disponibile (SPU_RZ_FP_TLS + AVX2/FMA) */
#undef spu_fa
#undef spu_fs
#undef spu_fm
#undef spu_fma
#undef spu_fms
#undef spu_fnms
#define spu_fa(a, b)      spu__rz_fin(_mm_add_ps(spu__rz_ld(a), spu__rz_ld(b)), 0, (a), (b), (b))
#define spu_fs(a, b)      spu__rz_fin(_mm_sub_ps(spu__rz_ld(a), spu__rz_ld(b)), 1, (a), (b), (b))
#define spu_fm(a, b)      spu__rz_fin(_mm_mul_ps(spu__rz_ld(a), spu__rz_ld(b)), 2, (a), (b), (b))
#define spu_fma(a, b, c)  spu__rz_fin(_mm_fmadd_ps(spu__rz_ld(a), spu__rz_ld(b), spu__rz_ld(c)), 3, (a), (b), (c))
#define spu_fms(a, b, c)  spu__rz_fin(_mm_fmsub_ps(spu__rz_ld(a), spu__rz_ld(b), spu__rz_ld(c)), 4, (a), (b), (c))
#define spu_fnms(a, b, c) spu__rz_fin(_mm_fnmadd_ps(spu__rz_ld(a), spu__rz_ld(b), spu__rz_ld(c)), 5, (a), (b), (c))
#endif
/* 3) csflt/cuflt con scala costante: int -> float "troncato" = cvtepi32_ps in RZ (MXCSR del ticket) poi * 2^(i8-155), esatto per i8 in [29,120]
 *    (risultato >= 2^-126 e < 2^128). cuflt solo se nessuna corsia ha il bit 31 (altrimenti l'originale). Altrimenti l'originale. */
static inline __attribute__((always_inline)) u128 cf_csflt_orig(u128 a, int i8) { return spu_csflt(a, i8); }
static inline __attribute__((always_inline)) u128 cf_cuflt_orig(u128 a, int i8) { return spu_cuflt(a, i8); }
static inline __attribute__((always_inline)) u128 cf_csflt(u128 a, int i8)
{
    if (!(i8 >= 29 && i8 <= 120)) return cf_csflt_orig(a, i8);
    const __m128 k = _mm_castsi128_ps(_mm_set1_epi32((int)((uint32_t)(127 - 155 + i8) << 23)));
    return cf_st(_mm_castps_si128(_mm_mul_ps(_mm_cvtepi32_ps(cf_ld(a)), k)));
}
static inline __attribute__((always_inline)) u128 cf_cuflt(u128 a, int i8)
{
    if (!(i8 >= 29 && i8 <= 120)) return cf_cuflt_orig(a, i8);
    const __m128i v = cf_ld(a);
    if (__builtin_expect(_mm_movemask_ps(_mm_castsi128_ps(v)) != 0, 0)) return cf_cuflt_orig(a, i8);
    const __m128 k = _mm_castsi128_ps(_mm_set1_epi32((int)((uint32_t)(127 - 155 + i8) << 23)));
    return cf_st(_mm_castps_si128(_mm_mul_ps(_mm_cvtepi32_ps(v), k)));
}
#undef spu_csflt
#undef spu_cuflt
#define spu_csflt(a, i8) cf_csflt((a), (i8))
#define spu_cuflt(a, i8) cf_cuflt((a), (i8))
#endif
