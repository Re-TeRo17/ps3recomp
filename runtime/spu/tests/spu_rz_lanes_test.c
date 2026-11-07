/* spu_rz_lanes_test.c (rissa128) -- host RZ float ops + spu__rz_redo_lanes vs the exact SPU ops (spu_fa..spu_fnms, per-lane
 * reference, MXCSR default). Built WITHOUT SPU_RZ_FP_TLS so spu_fa etc. are the exact path.
 * build: clang-cl /O2 -march=x86-64-v3 -I<runtime/spu> -I<include> spu_rz_lanes_test.c */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include "spu_helpers.h"
int g_spu_fixstats = 0;
int g_spu_fext = 1;
SPU_THREAD_LOCAL int g_spu_rz;
void spu_fix_note(int kind, int reason) { (void)kind; (void)reason; }
void spu_rz_slow_note(void) {}
static uint64_t s = 0x243F6A8885A308D3ull;
static uint32_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)s; }
static uint32_t lane(void) {
    uint32_t w = rnd();
    switch (rnd() & 7) {
    case 0: w |= 0x7F800000u; break;                        /* exponent 255 (extended range) */
    case 1: w = (w & 0x807FFFFFu) | 0x7F000000u; break;     /* exponent 254: overflow on mul/add */
    case 2: w &= 0x807FFFFFu; break;                        /* exponent 0 */
    case 3: w = (w & 0x80FFFFFFu) | 0x3F000000u; break;     /* ~1 */
    case 4: w = (w & 0x80FFFFFFu) | 0x00800000u; break;     /* near Smin: underflow */
    default: break;
    }
    return w;
}
static u128 host(int op, u128 a, u128 b, u128 c) {
    const __m128 x = _mm_loadu_ps((const float*)a._u32), y = _mm_loadu_ps((const float*)b._u32), z = _mm_loadu_ps((const float*)c._u32);
    __m128 r;
    switch (op) { case 0: r = _mm_add_ps(x, y); break; case 1: r = _mm_sub_ps(x, y); break; case 2: r = _mm_mul_ps(x, y); break;
                  case 3: r = _mm_fmadd_ps(x, y, z); break; case 4: r = _mm_fmsub_ps(x, y, z); break; default: r = _mm_fnmadd_ps(x, y, z); break; }
    __m128i ri = _mm_castps_si128(r);
    const __m128i t = _mm_and_si128(ri, _mm_set1_epi32(0x7FFFFFFF));
    u128 h; _mm_storeu_si128((__m128i*)h._u32, ri);
    if (_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(t, _mm_set1_epi32(0x7F7FFFFE)))) != 0) return spu__rz_redo_lanes(op, a, b, c, h);
    ri = _mm_andnot_si128(_mm_cmpeq_epi32(t, _mm_setzero_si128()), ri);
    _mm_storeu_si128((__m128i*)h._u32, ri); return h;
}
static u128 exact(int op, u128 a, u128 b, u128 c) {
    switch (op) { case 0: return spu_fa(a, b); case 1: return spu_fs(a, b); case 2: return spu_fm(a, b);
                  case 3: return spu_fma(a, b, c); case 4: return spu_fms(a, b, c); default: return spu_fnms(a, b, c); }
}
int main(void)
{
    long bad = 0, n = 0, redo = 0;
    const unsigned dflt = _mm_getcsr();
    for (long it = 0; it < 30000000; it++) {
        u128 a, b, c;
        for (int i = 0; i < 4; i++) { a._u32[i] = lane(); b._u32[i] = lane(); c._u32[i] = lane(); }
        const int op = (int)(rnd() % 6);
        _mm_setcsr(dflt); const u128 e = exact(op, a, b, c);
        _mm_setcsr(0xFFC0u); const u128 h = host(op, a, b, c);
        _mm_setcsr(dflt);
        n++;
        for (int i = 0; i < 4; i++) if (((h._u32[i] & 0x7FFFFFFFu) > 0x7F7FFFFEu)) { redo++; break; }
        if (memcmp(&e, &h, 16)) { if (bad++ < 12) printf("op=%d a=%08X b=%08X c=%08X exact=%08X host=%08X (lane0)\n", op, a._u32[0], b._u32[0], c._u32[0], e._u32[0], h._u32[0]); }
    }
    printf("spu_rz_lanes_test: %ld cases (%ld with a redone lane), %ld mismatches\n", n, redo, bad);
    return bad != 0;
}
