/* spu_rz_fp_test.c -- SPU_RZ_FP_TLS (spu_helpers.h): the host float ops under MXCSR = RZ|DAZ|FTZ (plus the exact-path
 * redo for exponent-255 operands / overflow) must give the very same bits as the exact SPU ops, for every input.
 * Compares fa/fs/fm/fma/fms/fnms against (1) the scalar reference spu_*_ref and (2) the production exact x86 path.
 *   clang-cl /O2 -march=x86-64-v3 -DSPU_RZ_FP_TLS -I<runtime/spu> -I<include> spu_rz_fp_test.c
 * Exit code 0 = no mismatch. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#define SPU_RZ_FP_TLS 1
#include "spu_helpers.h"

int g_spu_fixstats = 0;
int g_spu_fext = 1;
SPU_THREAD_LOCAL int g_spu_rz;
void spu_fix_note(int kind, int reason) { (void)kind; (void)reason; }
static long s_slow;
void spu_rz_slow_note(void) { s_slow++; }

static uint64_t s_rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { s_rng ^= s_rng << 13; s_rng ^= s_rng >> 7; s_rng ^= s_rng << 17; return (uint32_t)(s_rng >> 16); }

/* one lane drawn from a class that stresses the edges */
static uint32_t lane(void)
{
    uint32_t r = rnd(), sign = (rnd() & 1) ? 0x80000000u : 0, frac = rnd() & 0x7FFFFF;
    static int benign = -1; if (benign < 0) benign = getenv("RZ_BENIGN") != NULL;
    if (benign) { static const int ok[] = { 1, 2, 3, 6, 7, 8, 9, 13, 5 }; r = ok[r % 9]; } else r %= 14;
    switch (r) {
    case 0:  return rnd();                                          /* anything */
    case 1:  return sign | ((100 + rnd() % 56) << 23) | frac;       /* ordinary, close exponents (cancellation) */
    case 2:  return sign | ((1 + rnd() % 4) << 23) | frac;          /* just above the flush threshold */
    case 3:  return sign | (rnd() % 2 ? 0 : 0) | frac;              /* biased exponent 0: denormal / zero */
    case 4:  return sign | (0xFFu << 23) | frac;                    /* extended range, exponent 255 */
    case 5:  return sign | ((250 + rnd() % 5) << 23) | frac;        /* near the top */
    case 6:  return sign;                                           /* +-0 */
    case 7:  return sign | ((120 + rnd() % 16) << 23);              /* powers of two */
    case 8:  return sign | ((120 + rnd() % 16) << 23) | 0x7FFFFF;   /* all-ones significand */
    case 9:  return sign | ((120 + rnd() % 16) << 23) | (frac & 0x7FF000); /* short significand: exact sums */
    case 10: return 0x7FFFFFFFu;                                    /* Smax */
    case 11: return sign | 0x00800000u;                             /* Smin */
    case 12: return sign | ((rnd() % 256) << 23) | frac;
    default: return sign | ((126 + rnd() % 4) << 23) | frac;        /* around 1.0 */
    }
}
static u128 vec(void) { u128 v; for (int i = 0; i < 4; i++) v._u32[i] = lane(); return v; }

static int cmp(const char* op, u128 a, u128 b, u128 c, u128 got, u128 want, const char* ref, int three)
{
    if (memcmp(&got, &want, 16) == 0) return 0;
    printf("MISMATCH %s vs %s:\n", op, ref);
    for (int i = 0; i < 4; i++) if (got._u32[i] != want._u32[i])
        printf("  lane %d a=%08X b=%08X%s%08X got=%08X want=%08X\n", i, a._u32[i], b._u32[i], three ? " c=" : "", three ? c._u32[i] : 0, got._u32[i], want._u32[i]);
    return 1;
}

static __declspec(noinline) void rz_all(u128 a, u128 b, u128 c, u128* g)
{
    g[0] = spu_fa(a, b); g[1] = spu_fs(a, b); g[2] = spu_fm(a, b);
    g[3] = spu_fma(a, b, c); g[4] = spu_fms(a, b, c); g[5] = spu_fnms(a, b, c);
}
static __declspec(noinline) void ex_all(u128 a, u128 b, u128 c, u128* x)
{
    for (int k = 0; k < 6; k++) x[k] = spu__rz_exact(k, a, b, c, 0);
}
int main(int argc, char** argv)
{
    long n = argc > 1 ? atol(argv[1]) : 20000000;
    long bad = 0, tot = 0;
    g_spu_rz = 1;
    for (long it = 0; it < n && bad < 20; it++) {
        u128 a = vec(), b = vec(), c = vec();
        u128 g[6], x[6], rf[6];
        _mm_setcsr(SPU_RZ_MXCSR);
        rz_all(a, b, c, g);
        _mm_setcsr(0x1F80u);
        ex_all(a, b, c, x);
        rf[0] = spu_fa_ref(a, b); rf[1] = spu_fs_ref(a, b); rf[2] = spu_fm_ref(a, b);
        rf[3] = spu_fma_ref(a, b, c); rf[4] = spu_fms_ref(a, b, c); rf[5] = spu_fnms_ref(a, b, c);
        static const char* nm[6] = { "fa", "fs", "fm", "fma", "fms", "fnms" };
        for (int k = 0; k < 6; k++) {
            bad += cmp(nm[k], a, b, c, g[k], x[k], "exact-x86", k >= 3);
            bad += cmp(nm[k], a, b, c, g[k], rf[k], "ref", k >= 3);
            tot++;
        }
    }
    printf("%ld op-vectors checked (%ld exact-path redos), %ld mismatches\n", tot, s_slow, bad);
    return bad ? 1 : 0;
}


