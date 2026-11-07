/* spu_fc_vec_test.c (rissa128) -- vector fceq/fcgt/fcmeq/fcmgt and cuflt/csflt fast paths vs the scalar *_ref forms.
 * build: clang-cl /O2 -march=x86-64-v3 -DSPU_RZ_FP_TLS -I<runtime/spu> -I<include> spu_fc_vec_test.c
 * Inputs: random words, words biased to the special classes (exponent 0 / 255, +-0, sign), all 256 scales for the converts. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "spu_helpers.h"
int g_spu_fixstats = 0;
int g_spu_fext = 1;
SPU_THREAD_LOCAL int g_spu_rz;
void spu_fix_note(int kind, int reason) { (void)kind; (void)reason; }
void spu_rz_slow_note(void) {}
static uint64_t s = 88172645463325252ull;
static uint32_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)s; }
static uint32_t word(void) {
    uint32_t w = rnd();
    switch (rnd() & 7) {
    case 0: w &= 0x807FFFFFu; break;                       /* exponent 0 (denormal-looking / zero) */
    case 1: w |= 0x7F800000u; break;                       /* exponent 255 */
    case 2: w &= 0x80000000u; break;                       /* +-0 */
    case 3: w = (w & 0x80FFFFFFu) | 0x3F000000u; break;    /* near 1.0 */
    case 4: w &= 0x00FFFFFFu; break;                       /* small integers (< 2^24) */
    case 5: w = (w & 1) ? (w & 0xFFFFFFu) : (0u - (w & 0xFFFFFFu)); break;   /* small signed ints */
    default: break;
    }
    return w;
}
static int eq(u128 a, u128 b) { return memcmp(&a, &b, 16) == 0; }
int main(void)
{
    long bad = 0, n = 0;
    for (long it = 0; it < 20000000; it++) {
        u128 a, b;
        for (int i = 0; i < 4; i++) { a._u32[i] = word(); b._u32[i] = (rnd() & 3) == 0 ? a._u32[i] ^ (rnd() & 0x80000000u) : word(); }
        n++;
        if (!eq(spu_fceq(a, b),  spu_fceq_ref(a, b)))  { if (bad++ < 10) printf("fceq  %08X %08X\n", a._u32[0], b._u32[0]); }
        if (!eq(spu_fcgt(a, b),  spu_fcgt_ref(a, b)))  { if (bad++ < 10) printf("fcgt  %08X %08X\n", a._u32[0], b._u32[0]); }
        if (!eq(spu_fcmeq(a, b), spu_fcmeq_ref(a, b))) { if (bad++ < 10) printf("fcmeq %08X %08X\n", a._u32[0], b._u32[0]); }
        if (!eq(spu_fcmgt(a, b), spu_fcmgt_ref(a, b))) { if (bad++ < 10) printf("fcmgt %08X %08X\n", a._u32[0], b._u32[0]); }
        const int i8 = (int)(rnd() & 0xFF);
        if (!eq(spu_cuflt(a, i8), spu_cuflt_ref(a, i8))) { if (bad++ < 10) printf("cuflt %08X i8=%d\n", a._u32[0], i8); }
        if (!eq(spu_csflt(a, i8), spu_csflt_ref(a, i8))) { if (bad++ < 10) printf("csflt %08X i8=%d\n", a._u32[0], i8); }
    }
    /* every scale with the boundary integers */
    const uint32_t edge[] = { 0, 1, 2, 3, 0x7FFFFF, 0x800000, 0xFFFFFF, 0x1000000, 0x7FFFFFFF, 0x80000000u, 0xFFFFFFFFu, 0xFF000001u, 0xFF000000u, 0x80000001u };
    for (int i8 = 0; i8 < 256; i8++) for (unsigned e = 0; e < sizeof edge / 4; e++) {
        u128 a; for (int i = 0; i < 4; i++) a._u32[i] = edge[(e + i) % (sizeof edge / 4)];
        n++;
        if (!eq(spu_cuflt(a, i8), spu_cuflt_ref(a, i8))) { if (bad++ < 20) printf("cuflt edge %08X i8=%d\n", a._u32[0], i8); }
        if (!eq(spu_csflt(a, i8), spu_csflt_ref(a, i8))) { if (bad++ < 20) printf("csflt edge %08X i8=%d\n", a._u32[0], i8); }
    }
    printf("spu_fc_vec_test: %ld cases, %ld mismatches\n", n, bad);
    return bad != 0;
}
