/* anim_ref.c -- riferimento C scalare bit-esatto del comando di animazione SPU 0x6248
 * (job wid0 = jobmod_00b730f78d78, SPU 0x4000..0x7480). Specifica: formato.md.
 *
 * Aritmetica: u128 + helper spu_* di runtime/spu/spu_helpers.h (stessa semantica float SPU del lifted:
 * fm/fma troncano, niente denormali, range esteso). Ogni funzione porta il nome della funzione SPU
 * (anim_fXXXX = funzione a LS 0xXXXX) e il numero di istruzioni nello stesso ordine del disassemblato
 * (log\hle40a\wid0.dis). I registri SPU compaiono come r<N> dove trascrivere l'aritmetica float e'
 * piu' sicuro che riscriverla.
 *
 * ASSUNZIONE (verificata dal banco di prova): gli indirizzi LS passati ai load/store di mezze parole
 * e parole sono allineati alla loro dimensione, come sempre nel job (blocchi di arena multipli di 16).
 * Le costanti float del job (0x7100..0x7360, pattern shufb) si leggono dall'immagine LS (dati del job,
 * non codice): `ls` deve contenere l'immagine a 0x4000. */
#include "anim_ref.h"
#include "spu_helpers.h"
#include <string.h>
#include <stdio.h>
#include <immintrin.h>

/* anim46: profilo opzionale (-DANIM_PROF): cicli TSC per fase */
#ifdef ANIM_PROF
#include <intrin.h>
uint64_t g_prof[32]; uint64_t g_prof_n[32];
static const char* const g_prof_nm[32] = { "get+prep", "tree_walk+getl", "type3 (4120/4060/4CE8/5160)", "  f5160 (dentro type3)", "type1 blend (4BD8/4AE8/5250)", "G_5B78 (keyframe) totale", "G_53B8 (catena) totale", "f4D70 totale", "put+tail", "", "  f4D70 flag loop", "  f4D70 loop principale (6E80+matmul)", "  f4D70 scala finale", "f4258", "f4300", "4030/4060/4120 copie", "f41E8", "4CE8", "4D18", "6F20 GET", "5160 (call)" };
#define PROF_BEGIN(id) uint64_t _pf_##id = __rdtsc()
#define PROF_END(id) do { g_prof[id] += __rdtsc() - _pf_##id; g_prof_n[id]++; } while (0)
void anim_prof_print(double ghz) { for (int i = 0; i < 32; i++) if (g_prof_n[i]) printf("  prof[%2d] %-38s %10.1f ns/cmd  (%llu volte)\n", i, g_prof_nm[i] ? g_prof_nm[i] : "?", g_prof[i] / ghz / (double)g_prof_n[0], (unsigned long long)g_prof_n[i]); }

void anim_prof_reset(void) { memset(g_prof, 0, sizeof g_prof); memset(g_prof_n, 0, sizeof g_prof_n); }
#else
#define PROF_BEGIN(id) (void)0
#define PROF_END(id) (void)0
#endif

extern int anim_fx_add(anim_fx* fx, uint8_t cmd, uint8_t tag, uint32_t lsa, uint32_t ea, uint32_t size,
                       const uint8_t* bytes, uint32_t nbytes);

/* ------------------------------------------------------------------ accessori LS (big endian) */
static inline u128 LQ(const uint8_t* ls, uint32_t a)
{
    a &= 0x3FFF0u;
    __m128i v = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(ls + a)), _mm_setr_epi8(3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12));
    u128 r; memcpy(&r, &v, 16); return r;
}
static inline void SQ(uint8_t* ls, uint32_t a, u128 v)
{
    a &= 0x3FFF0u;
    __m128i x; memcpy(&x, &v, 16);
    _mm_storeu_si128((__m128i*)(ls + a), _mm_shuffle_epi8(x, _mm_setr_epi8(3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12)));
}
static inline uint32_t LW(const uint8_t* ls, uint32_t a)  { a &= 0x3FFFFu; uint32_t v; memcpy(&v, ls + a, 4); return _byteswap_ulong(v); }
static inline void     SW(uint8_t* ls, uint32_t a, uint32_t v) { a &= 0x3FFFFu; v = _byteswap_ulong(v); memcpy(ls + a, &v, 4); }
static inline uint32_t LH(const uint8_t* ls, uint32_t a)  { a &= 0x3FFFFu; uint16_t v; memcpy(&v, ls + a, 2); return _byteswap_ushort(v); }
static inline void     SH(uint8_t* ls, uint32_t a, uint32_t v) { a &= 0x3FFFFu; uint16_t w = _byteswap_ushort((uint16_t)v); memcpy(ls + a, &w, 2); }

/* replica la parola preferita (word 0) su 4 lane: == spu_shufb(x, x, ila 0x10203), ma una sola shuffle (anim46) */
static inline u128 splat_w0(u128 x) { __m128i v; memcpy(&v, &x, 16); v = _mm_shuffle_epi32(v, 0); u128 r; memcpy(&r, &v, 16); return r; }
#define SPLAT(x)  splat_w0(x)
#define W0(v)     ((v)._u32[0])

/* ================================================================== piccoli helper LS (funzioni SPU 0x4030..0x4D18) */

/* 0x4030: riempie n quadword a stride `stride` con il quadword `val` (tutto il registro). */
void anim_f4030(uint8_t* ls, uint32_t dst, uint32_t n, u128 val, uint32_t stride)
{
    if (n == 0) return;
    for (uint32_t i = 0; i < n; i++) { SQ(ls, dst, val); dst += stride; }
}

/* 0x4060: n elementi da 3 mezze parole (6 B) a src -> mezze parole dispari (+2,+6,+10) del quadword di
 * dst, la quarta (+14) = 0; le mezze parole pari (+0,+4,+8,+12) del quadword restano quelle di prima. */
void anim_f4060(uint8_t* ls, uint32_t dst, uint32_t src, uint32_t n, uint32_t stride)
{
    if (n == 0) return;
    for (uint32_t i = 0; i < n; i++) {
        SH(ls, dst + 2,  LH(ls, src));
        SH(ls, dst + 6,  LH(ls, src + 2));
        SH(ls, dst + 10, LH(ls, src + 4));
        SH(ls, dst + 14, 0);
        src += 6; dst += stride;
    }
}

/* 0x4120: come 4060 ma elementi da 4 mezze parole (8 B), tutte e 4 copiate. */
void anim_f4120(uint8_t* ls, uint32_t dst, uint32_t src, uint32_t n, uint32_t stride)
{
    if (n == 0) return;
    for (uint32_t i = 0; i < n; i++) {
        SH(ls, dst + 2,  LH(ls, src));
        SH(ls, dst + 6,  LH(ls, src + 2));
        SH(ls, dst + 10, LH(ls, src + 4));
        SH(ls, dst + 14, LH(ls, src + 6));
        src += 8; dst += stride;
    }
}

/* 0x41E8 (dst, vals, offs, n): per i<n: mezza parola a dst+offs[i] = vals[i]  (offs[i] senza segno 16 bit, pari). */
void anim_f41E8(uint8_t* ls, uint32_t dst, uint32_t vals, uint32_t offs, uint32_t n)
{
    if (n == 0) return;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t off = LH(ls, offs + 2*i);
        SH(ls, dst + off, LH(ls, vals + 2*i));
    }
}

/* 0x4258 (dst, A, B, ix, n, scale): per i<n: a=int16 A[i], b=int16 B[i], off=u16 ix[i]:
 *   mezza parola a dst+off = (int16)( a + cflts((float)(b-a) * scale) )  [ah: somma a 16 bit] */
void anim_f4258(uint8_t* ls, uint32_t dst, uint32_t A, uint32_t B, uint32_t ix, uint32_t n, u128 scale)
{
    if (n == 0) return;
    for (uint32_t i = 0; i < n; i++) {
        u128 ra = spu_zero(), rb = spu_zero(), rd;
        ra._u32[0] = (uint32_t)(int32_t)(int16_t)LH(ls, A + 2*i);
        rb._u32[0] = (uint32_t)(int32_t)(int16_t)LH(ls, B + 2*i);
        rd = spu_sf(ra, rb);                              /* b - a */
        rd = spu_csflt(rd, 155);                          /* -> float */
        rd = spu_fm(rd, scale);
        rd = spu_cflts(rd, 173);                          /* trunc */
        uint32_t off = LH(ls, ix + 2*i);
        u128 hraw = spu_zero(); hraw._u16[1] = 0;        /* halfword slot = bytes 2-3 */
        /* ah: somma 16 bit del solo halfword-slot: (rd.lo16 + a.lo16) */
        uint16_t s = (uint16_t)((rd._u32[0] & 0xFFFF) + (uint16_t)LH(ls, A + 2*i));
        (void)hraw;
        SH(ls, dst + off, s);
    }
}

/* 0x4300 (dst, A, B, ix, ix2, n, scale): per i<n: a=int8 A[i]*4, b=int8 B[i]*4 (halfword), base=int16 ix[i]
 *   x = a + base; y = base + b; out = (int16)( x + cflts((float)(y - x) * scale) ), a dst + u16 ix2[i]. */
void anim_f4300(uint8_t* ls, uint32_t dst, uint32_t A, uint32_t B, uint32_t ix, uint32_t ix2, uint32_t n, u128 scale)
{
    if (n == 0) return;
    for (uint32_t i = 0; i < n; i++) {
        int16_t a = (int16_t)((int16_t)(int8_t)ls[(A + i) & 0x3FFFF] * 4);
        int16_t b = (int16_t)((int16_t)(int8_t)ls[(B + i) & 0x3FFFF] * 4);
        int16_t base = (int16_t)LH(ls, ix + 2*i);
        uint16_t x = (uint16_t)(a + base), y = (uint16_t)(base + b);
        u128 rx = spu_zero(), ry = spu_zero(), rd;
        rx._u32[0] = (uint32_t)(int32_t)(int16_t)x; ry._u32[0] = (uint32_t)(int32_t)(int16_t)y;
        rd = spu_sf(rx, ry);                              /* y - x */
        rd = spu_csflt(rd, 155);
        rd = spu_fm(rd, scale);
        rd = spu_cflts(rd, 173);
        uint16_t s = (uint16_t)((rd._u32[0] & 0xFFFF) + x);
        SH(ls, dst + LH(ls, ix2 + 2*i), s);
    }
}

/* 0x4CE8 (dst, n, stride): n quadword = {255,255,255,255} (il 255). */
void anim_f4CE8(uint8_t* ls, uint32_t dst, uint32_t n, uint32_t stride)
{
    u128 v = spu_il(255);
    if (n == 0) return;
    for (uint32_t i = 0; i < n; i++) { SQ(ls, dst, v); dst += stride; }
}

/* 0x4D18 (dst, src, n, stride): per i<n: quadword dst = splat( (uint32) byte src[i] ). */
void anim_f4D18(uint8_t* ls, uint32_t dst, uint32_t src, uint32_t n, uint32_t stride)
{
    if (n == 0) return;
    for (uint32_t i = 0; i < n; i++) {
        u128 v = spu_zero(); v._u32[0] = ls[(src + i) & 0x3FFFF];
        SQ(ls, dst, spu_shufb(v, v, spu_ila(0x10203)));
        dst += stride;
    }
}

/* ================================================================== kernel float */

/* perf99: helper __m128 per i percorsi veloci RZ (f5160, f4D70) */
static inline __m128 v_LQ(const uint8_t* ls, uint32_t a)
{
    a &= 0x3FFF0u;
    return _mm_castsi128_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(ls + a)), _mm_setr_epi8(3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12)));
}
static inline __m128i v_LQi(const uint8_t* ls, uint32_t a) { return _mm_castps_si128(v_LQ(ls, a)); }
static inline void v_SQ(uint8_t* ls, uint32_t a, __m128 v)
{
    a &= 0x3FFF0u;
    _mm_storeu_si128((__m128i*)(ls + a), _mm_shuffle_epi8(_mm_castps_si128(v), _mm_setr_epi8(3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12)));
}
static inline __m128 v_SQz(uint8_t* ls, uint32_t a, __m128 v)   /* -0 -> +0 e store */
{
    const __m128i x = _mm_castps_si128(v);
    const __m128i t = _mm_and_si128(x, _mm_set1_epi32(0x7FFFFFFF));
    const __m128i y = _mm_andnot_si128(_mm_cmpeq_epi32(t, _mm_setzero_si128()), x);
    v_SQ(ls, a, _mm_castsi128_ps(y));
    return _mm_castsi128_ps(y);
}
static inline __m128i v_selb(__m128i a, __m128i b, __m128i m) { return _mm_or_si128(_mm_and_si128(m, b), _mm_andnot_si128(m, a)); }
#define V_SPL(v, k) _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v), (k) * 0x55))
/* maschera pshufb equivalente a spu_shufb(x, x, pat) se i byte di pat sono tutti < 0x20; ritorna 0 se pat ha codici speciali */
static int v_shufb_mask(const uint8_t* ls, uint32_t a, __m128i* out)
{
    uint8_t m[16];
    for (int j = 0; j < 16; j++) {
        uint8_t p = ls[(a + j) & 0x3FFFF];
        if (p >= 0x20) return 0;
        m[4 * (j >> 2) + 3 - (j & 3)] = (uint8_t)(4 * ((p & 15) >> 2) + 3 - (p & 3));
    }
    *out = _mm_loadu_si128((const __m128i*)m); return 1;
}


int g_f5160_off = 0;   /* diagnostica: 1 = sempre il corpo originale */
static inline float v_f32(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
/* frsqest + fi su UNA lane (identiche a spu_frsqest/spu_fi, che in f5160 lavorano 4 volte sullo stesso valore); ritorna i bit di y, 0 se fuori dal caso comune */
static inline uint32_t v_rsq_y(uint32_t d)
{
    uint32_t exp = (d >> 23) & 0xFFu, fi = (d >> 18) & 0x3Fu;
    uint32_t rexp = (exp == 0u) ? 0xFFu : (190u - (exp + 1u) / 2u);
    uint32_t est = (rexp << 23) | (spu_frsqest_fraction_lut[fi] & 0x7FFFFFu);
    uint32_t e2 = (est >> 23) & 0xFFu;
    const uint64_t base = (est >> 10) & 0x1FFFu, step = est & 0x3FFu, y = d & 0x7FFFFu;
    if (e2 == 0) return 0;
    uint64_t m = ((((uint64_t)1 << 13) | base) << 19) - step * y;
    if (m < ((uint64_t)1 << 32)) { m <<= 1; if (--e2 == 0) return 0; }
    return ((uint32_t)e2 << 23) | (uint32_t)((m >> 9) & 0x7FFFFFu);
}
/* una riga di f5160 con intrinseci diretti (MXCSR = RZ|DAZ|FTZ). Ritorna 0 se d fuori dall'intervallo sicuro (la riga va al corpo originale). */
static inline int v_f5160_row(uint8_t* ls, uint32_t p, uint32_t p3, __m128i sh0, __m128i sh1, __m128i sh2)
{
    const __m128i x0 = v_LQi(ls, p), x1 = v_LQi(ls, p + 16), x2 = v_LQi(ls, p + 32), x3 = v_LQi(ls, p3);
    const __m128i m3f = _mm_set1_epi32(0x3F);
    __m128i a = _mm_sllv_epi32(_mm_srai_epi32(_mm_slli_epi32(x0, 16), 16), _mm_and_si128(sh0, m3f));
    __m128i b = _mm_sllv_epi32(_mm_srai_epi32(_mm_slli_epi32(x1, 16), 16), _mm_and_si128(sh1, m3f));
    __m128i c = _mm_sllv_epi32(_mm_srai_epi32(_mm_slli_epi32(x2, 16), 16), _mm_and_si128(sh2, m3f));
    const __m128 k15 = _mm_set1_ps(3.0517578125e-05f), k8 = _mm_set1_ps(0.00390625f);
    const __m128 f0 = _mm_mul_ps(_mm_cvtepi32_ps(a), k15), f1 = _mm_mul_ps(_mm_cvtepi32_ps(b), k15), f2 = _mm_mul_ps(_mm_cvtepi32_ps(c), k15);
    /* cuflt: v >= 2^31 -> 2 * cvt(v >> 1) (stessa troncatura ai 24 bit alti) */
    const __m128 cu_lo = _mm_cvtepi32_ps(x3);
    const __m128 cu_hi = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srli_epi32(x3, 1)), _mm_set1_ps(2.0f));
    const __m128 f3 = _mm_mul_ps(_mm_blendv_ps(cu_lo, cu_hi, _mm_castsi128_ps(x3)), k8);
    const __m128 sq = _mm_mul_ps(f0, f0);
    __m128 s = _mm_add_ss(sq, _mm_shuffle_ps(sq, sq, 0x55));
    s = _mm_add_ss(s, _mm_shuffle_ps(sq, sq, 0xAA));
    s = _mm_add_ss(s, _mm_shuffle_ps(sq, sq, 0xFF));
    uint32_t dbits; { float df = _mm_cvtss_f32(s); memcpy(&dbits, &df, 4); }
    if (dbits < 0x21800000u || dbits >= 0x50000000u) return 0;
    const uint32_t yb = v_rsq_y(dbits);
    if (yb == 0) return 0;
    const __m128 d = _mm_set1_ps(v_f32(dbits)), y = _mm_set1_ps(v_f32(yb));
    const __m128 y2 = _mm_mul_ps(y, y), yh = _mm_mul_ps(y, _mm_set1_ps(0.5f));
    const __m128 t = _mm_fnmadd_ps(d, y2, _mm_set1_ps(1.0f));
    const __m128 rs = _mm_fmadd_ps(t, yh, y);
    const __m128 o0 = _mm_mul_ps(f0, rs);
    /* un risultato tiny/denormale della fma o un rs fuori scala: lascia la riga al corpo originale (non scritto nulla finora) */
    { uint32_t rb; float rf = _mm_cvtss_f32(rs); memcpy(&rb, &rf, 4); if ((rb & 0x7FFFFFFFu) < 0x20000000u || (rb & 0x7FFFFFFFu) > 0x5F000000u) return 0; }
    v_SQ(ls, p + 48, f3);
    v_SQ(ls, p + 16, f1);
    v_SQ(ls, p + 32, f2);
    v_SQz(ls, p, o0);
    return 1;
}


/* 0x5160 (blk, n): n righe da 64 B. Usa 0x34B40/50/60 = shift per riga1/riga0/riga2 (word0 di ciascuna). */
void anim_f5160(uint8_t* ls, uint32_t blk, uint32_t n)
{
    u128 SPL = spu_ila(0x10203);
    if (n == 0) return;
    u128 q1 = LQ(ls, 0x34B40), q0 = LQ(ls, 0x34B50), q2 = LQ(ls, 0x34B60);
    u128 sh0 = SPLAT(q0), sh1 = SPLAT(q1), sh2 = SPLAT(q2);
    u128 one = spu_ilhu(0x3F80), half = spu_ilhu(0x3F00);
    const int fastrz = g_spu_rz && (_mm_getcsr() & 0xE040u) == 0xE040u && !g_f5160_off;
    const __m128i v_sh0 = _mm_loadu_si128((const __m128i*)sh0._u32), v_sh1 = _mm_loadu_si128((const __m128i*)sh1._u32), v_sh2 = _mm_loadu_si128((const __m128i*)sh2._u32);
    uint32_t p = blk, p3 = blk + 48;
    for (uint32_t i = 0; i < n; i++) {
        u128 r0 = LQ(ls, p), r1 = LQ(ls, p + 16), r2 = LQ(ls, p + 32), r3 = LQ(ls, p3);
        p3 += 64;
        if (fastrz && v_f5160_row(ls, p, p3 - 64, v_sh0, v_sh1, v_sh2)) { p += 64; continue; }
        u128 a = spu_shl(spu_xshw(r0), sh0);
        u128 b = spu_shl(spu_xshw(r1), sh1);
        u128 c = spu_shl(spu_xshw(r2), sh2);
        u128 f0 = spu_csflt(a, 140), f1 = spu_csflt(b, 140), f2 = spu_csflt(c, 140), f3 = spu_cuflt(r3, 147);
        u128 sq = spu_fm(f0, f0);
        u128 s = spu_fa(sq, spu_shlqbyi(sq, 4));
        s = spu_fa(s, spu_shlqbyi(sq, 8));
        s = spu_fa(s, spu_shlqbyi(sq, 12));
        u128 d = SPLAT(s);
        u128 e = spu_frsqest(d);
        u128 y = spu_fi(d, e);
        u128 y2 = spu_fm(y, y), yh = spu_fm(y, half);
        u128 t = spu_fnms(d, y2, one);
        u128 rs = spu_fma(t, yh, y);
        SQ(ls, p + 48, f3);
        SQ(ls, p + 16, f1);
        SQ(ls, p + 32, f2);
        SQ(ls, p, spu_fm(f0, rs));
        p += 64;
    }
}

/* normalizza un quaternione v (lane0..3): v * rsqrt(|v|^2) con un passo di Newton (blocco ripetuto in 4BD8/5250). */
static inline u128 qnorm(u128 v, u128 SPL, u128 one, u128 half)
{
    u128 sq = spu_fm(v, v);
    u128 s = spu_fa(sq, spu_shlqbyi(sq, 4));
    s = spu_fa(s, spu_shlqbyi(sq, 8));
    s = spu_fa(s, spu_shlqbyi(sq, 12));
    u128 d = SPLAT(s);
    u128 e = spu_frsqest(d);
    u128 y = spu_fi(d, e);
    u128 y2 = spu_fm(y, y), yh = spu_fm(y, half);
    u128 t = spu_fnms(d, y2, one);
    u128 rs = spu_fma(t, yh, y);
    return spu_fm(v, rs);
}
static inline u128 hsum4(u128 q)   /* lane0 = q0+q1+q2+q3 con l'ordine ((q0+q1)+q2)+q3 */
{
    u128 s = spu_fa(q, spu_shlqbyi(q, 4));
    s = spu_fa(s, spu_shlqbyi(q, 8));
    return spu_fa(s, spu_shlqbyi(q, 12));
}

/* 0x4BD8 (P, Q, n, w): nlerp con inversione di emisfero. Per riga i: rows1,2 = lerp(P,Q, wi); row0 = nlerp.
 * wi = splat(Q.row3.w) * w. Scrive su P. */
void anim_f4BD8(uint8_t* ls, uint32_t P, uint32_t Q, uint32_t n, u128 param)
{
    u128 SPL = spu_ila(0x10203), sgn = LQ(ls, 0x7280), zero = spu_zero();
    u128 one = spu_ilhu(0x3F80), half = spu_ilhu(0x3F00);
    if (n == 0) return;
    for (uint32_t i = 0; i < n; i++) {
        u128 p0 = LQ(ls, P), q0 = LQ(ls, Q), q3 = LQ(ls, Q + 0x30);
        u128 p1 = LQ(ls, P + 0x10), p2 = LQ(ls, P + 0x20), q1 = LQ(ls, Q + 0x10), q2 = LQ(ls, Q + 0x20);
        Q += 64;
        u128 dp = spu_fm(p0, q0);
        u128 nq0 = spu_xor(q0, sgn);
        u128 d1 = spu_fs(q1, p1), d2 = spu_fs(q2, p2);
        u128 w = spu_fm(SPLAT(spu_rotqbyi(q3, 12)), param);
        u128 s = hsum4(dp);
        u128 x2 = spu_fma(d2, w, p2);
        u128 x1 = spu_fma(d1, w, p1);
        SQ(ls, P + 0x20, x2);
        SQ(ls, P + 0x10, x1);
        u128 neg = spu_fcgt(zero, SPLAT(s));
        u128 qq = spu_selb(q0, nq0, neg);
        u128 v = spu_fma(spu_fs(qq, p0), w, p0);
        SQ(ls, P, qnorm(v, SPL, one, half));
        P += 64;
    }
}

/* 0x4AE8 (P, Q, n, w): P.q = P.q (x) Q.q (prodotto di Hamilton), rows1,2 = lerp con wi = splat(Q.row3.w)*w. */
void anim_f4AE8(uint8_t* ls, uint32_t P, uint32_t Q, uint32_t n, u128 param)
{
    u128 SPL = spu_ila(0x10203), c7220 = LQ(ls, 0x7220), c7230 = LQ(ls, 0x7230), m3 = spu_fsmbi(0xF);
    if (n == 0) return;
    for (uint32_t i = 0; i < n; i++) {
        u128 p0 = LQ(ls, P), q0 = LQ(ls, Q), q3 = LQ(ls, Q + 0x30), p1 = LQ(ls, P + 0x10), p2 = LQ(ls, P + 0x20);
        u128 q1 = LQ(ls, Q + 0x10), q2 = LQ(ls, Q + 0x20);
        Q += 64;
        u128 r46 = spu_rotqbyi(p0, 12), r40 = spu_fm(p0, q0), r44 = spu_rotqbyi(q0, 12), r38 = spu_rotqbyi(q3, 12);
        u128 r42 = SPLAT(r46);
        u128 r28 = spu_fs(q1, p1);
        u128 r35 = SPLAT(r44);
        u128 r31 = spu_shufb(p0, p0, c7220);
        u128 r43 = SPLAT(r40);
        u128 r9 = spu_fm(r42, q0);
        u128 r39 = spu_rotqbyi(r40, 4);
        u128 r29 = spu_fs(q2, p2);
        u128 r14 = spu_shufb(q0, q0, c7230);
        u128 r37 = spu_rotqbyi(r40, 8);
        u128 r33 = spu_fs(r40, r43);
        u128 r32 = SPLAT(r38);
        u128 r6 = SPLAT(r39);
        u128 r27 = spu_shufb(p0, p0, c7230);
        u128 r10 = spu_fma(r35, p0, r9);
        u128 r30 = SPLAT(r37);
        u128 r3 = spu_fm(r32, param);
        u128 r7 = spu_shufb(q0, q0, c7220);
        u128 r8 = spu_fs(r33, r6);
        u128 r13 = spu_fma(r31, r14, r10);
        u128 r11 = spu_fma(r29, r3, p2);
        u128 r12 = spu_fma(r28, r3, p1);
        u128 r4 = spu_fs(r8, r30);
        u128 r2 = spu_fnms(r27, r7, r13);
        SQ(ls, P + 0x20, r11);
        SQ(ls, P + 0x10, r12);
        SQ(ls, P, spu_selb(r2, r4, m3));
        P += 64;
    }
}

/* 0x5250 (P, Q, n, w): P.q = Q.q (x) P.q poi nlerp da P.q con peso wi; rows1,2 = Q.row + P.row*wi (additivo). */
void anim_f5250(uint8_t* ls, uint32_t P, uint32_t Q, uint32_t n, u128 param)
{
    u128 SPL = spu_ila(0x10203), c7220 = LQ(ls, 0x7220), c7230 = LQ(ls, 0x7230), sgn = LQ(ls, 0x7280);
    u128 zero = spu_zero(), one = spu_ilhu(0x3F80), half = spu_ilhu(0x3F00), m3 = spu_fsmbi(0xF);
    if (n == 0) return;
    for (uint32_t i = 0; i < n; i++) {
        u128 q0 = LQ(ls, Q), p0 = LQ(ls, P), q3 = LQ(ls, Q + 0x30), p1 = LQ(ls, P + 0x10), q1 = LQ(ls, Q + 0x10), p2 = LQ(ls, P + 0x20);
        u128 r77 = spu_rotqbyi(q0, 12);
        u128 r73 = spu_fm(q0, p0);
        u128 r76 = spu_rotqbyi(p0, 12);
        u128 r60 = spu_shufb(q0, q0, c7220);
        u128 r62 = spu_shufb(p0, p0, c7230);
        u128 r74 = SPLAT(r77);
        u128 r67 = SPLAT(r76);
        u128 r51 = spu_shufb(q0, q0, c7230);
        u128 r75 = SPLAT(r73);
        u128 r68 = spu_fm(r74, p0);
        u128 r72 = spu_rotqbyi(r73, 4), r71 = spu_rotqbyi(r73, 8);
        u128 r52 = spu_shufb(p0, p0, c7220);
        u128 r64 = spu_fs(r73, r75);
        u128 r66 = spu_rotqbyi(q3, 12);
        u128 r65 = SPLAT(r72);
        u128 r59 = SPLAT(r71);
        u128 r61 = spu_fma(r67, q0, r68);
        u128 q2 = LQ(ls, Q + 0x20);
        Q += 64;
        u128 r63 = SPLAT(r66);
        u128 r58 = spu_fs(r64, r65);
        u128 r34 = spu_fm(r63, param);
        u128 r53 = spu_fma(r60, r62, r61);
        u128 r48 = spu_fs(r58, r59);
        u128 r50 = spu_fma(p2, r34, q2);
        u128 r47 = spu_fnms(r51, r52, r53);
        u128 r49 = spu_fma(p1, r34, q1);
        SQ(ls, P + 0x20, r50);
        u128 r36 = spu_selb(r47, r48, m3);
        SQ(ls, P + 0x10, r49);
        u128 dp = spu_fm(p0, r36);
        u128 r38 = spu_xor(r36, sgn);
        u128 s = hsum4(dp);
        u128 neg = spu_fcgt(zero, SPLAT(s));
        u128 r35 = spu_selb(r36, r38, neg);
        u128 r32 = spu_fs(r35, p0);
        u128 r8 = spu_fma(r32, r34, p0);
        SQ(ls, P, qnorm(r8, SPL, one, half));
        P += 64;
    }
}

/* ================================================================== 0x6E80: quaternione -> matrice 4x4 (4 righe a out, +0,+0x10,+0x20,+0x30) */
static inline void qmat(const uint8_t* ls, u128 q, u128 o[4]);
void anim_f6E80(uint8_t* ls, uint32_t out, u128 q)
{
    u128 o[4]; qmat(ls, q, o);
    SQ(ls, out, o[0]); SQ(ls, out + 0x10, o[1]); SQ(ls, out + 0x20, o[2]); SQ(ls, out + 0x30, o[3]);
}
static inline void qmat(const uint8_t* ls, u128 q, u128 o[4])
{
    u128 SPL = spu_ila(0x10203);
    u128 c7220 = LQ(ls, 0x7220), c7230 = LQ(ls, 0x7230), one = spu_ilhu(0x3F80);
    u128 m7300 = LQ(ls, 0x7300), m7360 = LQ(ls, 0x7360), m7340 = LQ(ls, 0x7340), m7320 = LQ(ls, 0x7320);
    u128 zero = spu_zero(), m3 = spu_fsmbi(0xF);
    u128 r23 = spu_fa(q, q);
    u128 r31 = spu_rotqbyi(r23, 12);
    u128 r28 = spu_shufb(q, q, c7220);
    u128 r29 = spu_shufb(r23, r23, c7220);
    u128 r21 = spu_shufb(q, q, c7230);
    u128 r26 = SPLAT(r31);
    u128 r27 = spu_fm(r28, r23);
    u128 r24 = spu_shufb(r23, r23, c7230);
    u128 r25 = spu_fnms(r28, r29, one);
    u128 r22 = spu_fm(r28, r26);
    u128 r18 = spu_fnms(r21, r26, r27);
    u128 r9  = spu_fnms(r21, r24, r25);
    u128 r12 = spu_fma(r21, r23, r22);
    u128 r6  = spu_selb(r18, zero, m3);
    u128 r10 = spu_selb(r9, r6, m7340);
    u128 r14 = spu_selb(r12, r9, m7300);
    u128 r8  = spu_selb(r6, r12, m7300);
    u128 r13 = spu_selb(r14, r6, m7360);
    o[0] = r13;
    o[1] = spu_selb(r10, r12, m7320);
    o[2] = spu_selb(r8, r9, m7320);
    o[3] = LQ(ls, 0x7270);
}

/* ================================================================== perf99: 0x4D70 veloce (percorso RZ)
 * Con g_spu_rz attivo e MXCSR = RZ|DAZ|FTZ le operazioni float host SONO le operazioni SPU, salvo (a) overflow/exp255 (range esteso) e (b) lo zero
 * negativo. Il percorso lento per-operazione (spu__rz_fin: controllo del risultato + correzione di -0 a OGNI op) costa ~1/2 del tempo del loop.
 * Qui, per ogni giunto: se TUTTI gli ingressi float hanno |x| < 2^20 (quindi nessun overflow/exp255 possibile: prodotti di al piu' 4 fattori < 2^84) il giunto
 * e' calcolato con intrinseci diretti senza controlli; l'unica correzione e' -0 -> +0 sulle 4 righe memorizzate (un -0 intermedio non cambia nessun
 * risultato non nullo). Altrimenti il giunto (e tutto il resto) va al corpo originale. Stesse load/store dell'originale nello stesso ordine. */
/* ritorna il numero di giunti elaborati (== N se tutto veloce); il resto (da i) lo fa il corpo originale */
int g_f4d70_off = 0;   /* banco/diagnostica: 1 = sempre il corpo originale */
static uint32_t f4D70_fast_loop(uint8_t* ls, uint32_t MW, uint32_t blkA, uint32_t N, uint32_t A2, uint32_t arr1, uint32_t tab, __m128 r91v, uint32_t* pcnt)
{
    __m128i M20, M30;
    if (!v_shufb_mask(ls, 0x7220, &M20) || !v_shufb_mask(ls, 0x7230, &M30)) return 0;
    const __m128i m7300 = v_LQi(ls, 0x7300), m7360 = v_LQi(ls, 0x7360), m7340 = v_LQi(ls, 0x7340), m7320 = v_LQi(ls, 0x7320);
    const __m128i m3 = _mm_setr_epi32(0, 0, 0, -1), m012 = _mm_setr_epi32(-1, -1, -1, 0);
    const __m128 one = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));
    const __m128 row3c = v_LQ(ls, 0x7270);
    const __m128i lim = _mm_set1_epi32(0x49800000), absm = _mm_set1_epi32(0x7FFFFFFF);
    uint32_t cnt = *pcnt;
    for (uint32_t i = 0; i < N; i++) {
        __m128 q = v_LQ(ls, blkA + 64 * i);
        int32_t par = (int16_t)LH(ls, A2 + 8 * i + 2);
        uint32_t pofs = (uint32_t)par << 6;
        uint32_t pA = blkA + pofs, pM = MW + pofs;
        __m128 r35 = v_LQ(ls, blkA + 64 * i + 0x20);
        __m128i flag1 = v_LQi(ls, arr1 + 16 * i);
        __m128 r33 = v_LQ(ls, pA + 0x10);
        __m128 P0 = v_LQ(ls, pM), P1 = v_LQ(ls, pM + 0x10), P2 = v_LQ(ls, pM + 0x20), P3 = v_LQ(ls, pM + 0x30);
        /* controllo ingressi (max di |x| per lane) */
        {
            __m128i mx = _mm_and_si128(_mm_castps_si128(q), absm);
            mx = _mm_max_epu32(mx, _mm_and_si128(_mm_castps_si128(r35), absm));
            mx = _mm_max_epu32(mx, _mm_and_si128(_mm_castps_si128(r33), absm));
            mx = _mm_max_epu32(mx, _mm_and_si128(_mm_castps_si128(P0), absm));
            mx = _mm_max_epu32(mx, _mm_and_si128(_mm_castps_si128(P1), absm));
            mx = _mm_max_epu32(mx, _mm_and_si128(_mm_castps_si128(P2), absm));
            mx = _mm_max_epu32(mx, _mm_and_si128(_mm_castps_si128(P3), absm));
            if (_mm_movemask_epi8(_mm_cmpgt_epi32(mx, lim)) != 0) { *pcnt = cnt; return i; }
        }
        /* 6E80: quaternione -> matrice (righe t0..t3) */
        const __m128 r23 = _mm_add_ps(q, q);
        const __m128 r26 = V_SPL(r23, 3);
        const __m128 r28 = _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(q), M20));
        const __m128 r29 = _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(r23), M20));
        const __m128 r21 = _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(q), M30));
        const __m128 r24 = _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(r23), M30));
        const __m128 r27q = _mm_mul_ps(r28, r23);
        const __m128 r25 = _mm_fnmadd_ps(r28, r29, one);
        const __m128 r22 = _mm_mul_ps(r28, r26);
        const __m128 r18 = _mm_fnmadd_ps(r21, r26, r27q);
        const __m128 r9  = _mm_fnmadd_ps(r21, r24, r25);
        const __m128 r12 = _mm_fmadd_ps(r21, r23, r22);
        const __m128i r9i = _mm_castps_si128(r9), r12i = _mm_castps_si128(r12);
        const __m128i r6 = v_selb(_mm_castps_si128(r18), _mm_setzero_si128(), m3);
        const __m128i r10 = v_selb(r9i, r6, m7340);
        const __m128i r14 = v_selb(r12i, r9i, m7300);
        const __m128i r8  = v_selb(r6, r12i, m7300);
        const __m128i r13 = v_selb(r14, r6, m7360);
        __m128 t0 = _mm_castsi128_ps(r13), t1 = _mm_castsi128_ps(v_selb(r10, r12i, m7320)), t2 = _mm_castsi128_ps(v_selb(r8, r9i, m7320));
        __m128 t3 = row3c;
        /* scala */
        const __m128i r27 = v_selb(_mm_castps_si128(t3), _mm_castps_si128(r35), m012);   /* riga3: xyz dalla traslazione */
        const __m128 r27f = _mm_castsi128_ps(r27);
        const __m128 r30 = _mm_castsi128_ps(v_selb(_mm_castps_si128(r33), _mm_castps_si128(r91v), flag1));
        const __m128 r28s = _mm_mul_ps(r27f, r33);
        t2 = _mm_mul_ps(t2, r30); t0 = _mm_mul_ps(t0, r30); t1 = _mm_mul_ps(t1, r30);
        t3 = _mm_castsi128_ps(v_selb(r27, _mm_castps_si128(r28s), m012));
        SH(ls, tab + 2 * cnt, i);
        /* prodotto con la matrice del genitore (stesso ordine di load/store dell'originale) */
        uint32_t O = MW + 64 * i;
        __m128 o0 = _mm_add_ps(_mm_fmadd_ps(P2, V_SPL(t0, 2), _mm_mul_ps(P0, V_SPL(t0, 0))), _mm_fmadd_ps(P3, V_SPL(t0, 3), _mm_mul_ps(P1, V_SPL(t0, 1))));
        v_SQz(ls, O, o0);
        P0 = v_LQ(ls, pM);
        __m128 o1 = _mm_add_ps(_mm_fmadd_ps(P2, V_SPL(t1, 2), _mm_mul_ps(P0, V_SPL(t1, 0))), _mm_fmadd_ps(P3, V_SPL(t1, 3), _mm_mul_ps(P1, V_SPL(t1, 1))));
        v_SQz(ls, O + 0x10, o1);
        P1 = v_LQ(ls, pM + 0x10);
        __m128 o2 = _mm_add_ps(_mm_fmadd_ps(P2, V_SPL(t2, 2), _mm_mul_ps(P0, V_SPL(t2, 0))), _mm_fmadd_ps(P3, V_SPL(t2, 3), _mm_mul_ps(P1, V_SPL(t2, 1))));
        v_SQz(ls, O + 0x20, o2);
        P2 = v_LQ(ls, pM + 0x20);
        __m128 o3 = _mm_add_ps(_mm_fmadd_ps(P2, V_SPL(t3, 2), _mm_mul_ps(P0, V_SPL(t3, 0))), _mm_fmadd_ps(P3, V_SPL(t3, 3), _mm_mul_ps(P1, V_SPL(t3, 1))));
        v_SQz(ls, O + 0x30, o3);
        cnt += LW(ls, arr1 + 6400 + 16 * i) & 1;
    }
    *pcnt = cnt;
    return N;
}

/* ================================================================== 0x4D70: gerarchia -> matrici mondo
 * MW = buffer matrici (N*64 B, risultato), blkA = blocco radice (0x79A0[0]: quat/scala/trasl. float),
 * A2 = record per giunto da 8 B { u16 flag(bit0), s16 genitore, ... }. Usa 0x34B30/0x34B40 (scala). */
void anim_f4D70(uint8_t* ls, uint32_t MW, uint32_t blkA, uint32_t N, uint32_t A2)
{
    u128 SPL = spu_ila(0x10203);
    u128 sh = SPLAT(LQ(ls, 0x34B40));
    u128 sc = spu_shl(LQ(ls, 0x34B30), sh);
    u128 r91 = spu_cuflt(sc, 140);                    /* "1.0" (o il valore float di riferimento) */
    u128 pat = LQ(ls, 0x70E0), m012 = spu_fsmbi(0xFFF0);
    uint32_t arr1 = MW + 64 * N;                      /* flag1[i] (16 B)   = arr1 + 16 i */
    uint32_t arr2 = arr1 + 6400;                      /* flag2[i] (16 B)   = arr2 + 16 i */
    uint32_t tab  = arr2 + 6400;                      /* tabella di mezze parole (giunti con flag2) */
    PROF_BEGIN(10);
    if (N) {
        for (uint32_t i = 0; i < N; i++) {
            u128 row1 = LQ(ls, blkA + 64 * i + 0x10);
            u128 w = LQ(ls, A2 + 8 * i);
            uint32_t hw0 = LH(ls, A2 + 8 * i);
            (void)w;
            u128 eq = spu_fceq(row1, r91);
            u128 g = spu_gb(eq);
            u128 r20 = spu_zero(); r20._u32[0] = hw0 & 1;
            u128 r17 = spu_ceqi(g, 15);
            u128 r19 = spu_clgti(r20, 0);
            u128 r15 = spu_xori(r17, -1);
            SQ(ls, arr1 + 16 * i, spu_shufb(r19, r19, pat));
            SQ(ls, arr2 + 16 * i, spu_shufb(r15, r15, pat));
        }
    }
    PROF_END(10);
    /* matrice identita' "genitore" a MW-64, scala 1.0 a blkA-48, genitore del giunto 0 = -1 */
    SQ(ls, MW - 64, LQ(ls, 0x7100)); SQ(ls, MW - 64 + 0x10, LQ(ls, 0x7110));
    SQ(ls, MW - 64 + 0x20, LQ(ls, 0x7120)); SQ(ls, MW - 64 + 0x30, LQ(ls, 0x7130));
    SQ(ls, blkA - 0x30, r91);
    SH(ls, A2 + 2, 0xFFFF);
    if (N == 0) return;
    uint32_t cnt = 0, i0 = 0;
    if (g_spu_rz && (_mm_getcsr() & 0xE040u) == 0xE040u && !g_f4d70_off) {
        __m128 r91v = _mm_loadu_ps((const float*)r91._u32);
        const __m128i mxv = _mm_and_si128(_mm_castps_si128(r91v), _mm_set1_epi32(0x7FFFFFFF));
        const __m128i mx2 = _mm_and_si128(_mm_castps_si128(v_LQ(ls, 0x7270)), _mm_set1_epi32(0x7FFFFFFF));
        if (_mm_movemask_epi8(_mm_cmpgt_epi32(_mm_max_epu32(mxv, mx2), _mm_set1_epi32(0x49800000))) == 0)
            i0 = f4D70_fast_loop(ls, MW, blkA, N, A2, arr1, tab, r91v, &cnt);
    }
    PROF_BEGIN(11);
    for (uint32_t i = i0; i < N; i++) {
        u128 q = LQ(ls, blkA + 64 * i);
        int32_t par = (int16_t)LH(ls, A2 + 8 * i + 2);
        u128 t[4];
        qmat(ls, q, t);                                 /* 6E80: matrice dal quaternione, in registri (anim46: niente finestra LS) */
        uint32_t pofs = (uint32_t)par << 6;
        uint32_t pA = blkA + pofs, pM = MW + pofs;
        u128 r35 = LQ(ls, blkA + 64 * i + 0x20);
        u128 flag1 = LQ(ls, arr1 + 16 * i);
        u128 r27 = spu_selb(t[3], r35, m012);          /* riga3: xyz dalla traslazione */
        u128 r33 = LQ(ls, pA + 0x10);                  /* scala del genitore */
        u128 r30 = spu_selb(r33, r91, flag1);
        u128 r28 = spu_fm(r27, r33);
        u128 t2 = spu_fm(t[2], r30), t0 = spu_fm(t[0], r30), t1 = spu_fm(t[1], r30);
        u128 t3 = spu_selb(r27, r28, m012);
        /* tabella: mezza parola = i in tab + 2*cnt (sempre scritta, cnt avanza solo se flag2) */
        SH(ls, tab + 2 * cnt, i);
        u128 P0 = LQ(ls, pM), P1 = LQ(ls, pM + 0x10), P2 = LQ(ls, pM + 0x20), P3 = LQ(ls, pM + 0x30);
        u128 sx0 = SPLAT(t0), sy0 = SPLAT(spu_rotqbyi(t0, 4)), sz0 = SPLAT(spu_rotqbyi(t0, 8)), sw0 = SPLAT(spu_rotqbyi(t0, 12));
        u128 sx1 = SPLAT(t1), sy1 = SPLAT(spu_rotqbyi(t1, 4)), sz1 = SPLAT(spu_rotqbyi(t1, 8)), sw1 = SPLAT(spu_rotqbyi(t1, 12));
        u128 sx2 = SPLAT(t2), sy2 = SPLAT(spu_rotqbyi(t2, 4)), sz2 = SPLAT(spu_rotqbyi(t2, 8)), sw2 = SPLAT(spu_rotqbyi(t2, 12));
        u128 sx3 = SPLAT(t3), sy3 = SPLAT(spu_rotqbyi(t3, 4)), sz3 = SPLAT(spu_rotqbyi(t3, 8)), sw3 = SPLAT(spu_rotqbyi(t3, 12));
        uint32_t O = MW + 64 * i;
        /* riga0 = (P0*x0 + P2*z0) + (P1*y0 + P3*w0) */
        u128 o0 = spu_fa(spu_fma(P2, sz0, spu_fm(P0, sx0)), spu_fma(P3, sw0, spu_fm(P1, sy0)));
        SQ(ls, O, o0);
        P0 = LQ(ls, pM);                               /* ricaricata dopo lo store (come lqx r59) */
        u128 o1 = spu_fa(spu_fma(P2, sz1, spu_fm(P0, sx1)), spu_fma(P3, sw1, spu_fm(P1, sy1)));
        SQ(ls, O + 0x10, o1);
        P1 = LQ(ls, pM + 0x10);
        u128 o2 = spu_fa(spu_fma(P2, sz2, spu_fm(P0, sx2)), spu_fma(P3, sw2, spu_fm(P1, sy2)));
        SQ(ls, O + 0x20, o2);
        P2 = LQ(ls, pM + 0x20);
        u128 o3 = spu_fa(spu_fma(P2, sz3, spu_fm(P0, sx3)), spu_fma(P3, sw3, spu_fm(P1, sy3)));
        SQ(ls, O + 0x30, o3);
        cnt += LW(ls, arr2 + 16 * i) & 1;
    }
    PROF_END(11);
    PROF_BEGIN(12);
    for (uint32_t k = 0; k < cnt; k++) {
        uint32_t idx = LH(ls, tab + 2 * k);
        u128 s = LQ(ls, blkA + 64 * idx + 0x10);
        u128 sxv = SPLAT(s), syv = SPLAT(spu_rotqbyi(s, 4)), szv = SPLAT(spu_rotqbyi(s, 8));
        uint32_t M = MW + 64 * idx;
        u128 m0 = LQ(ls, M), m1 = LQ(ls, M + 0x10), m2 = LQ(ls, M + 0x20);
        SQ(ls, M, spu_fm(m0, sxv));
        SQ(ls, M + 0x20, spu_fm(m2, szv));
        SQ(ls, M + 0x10, spu_fm(m1, syv));
    }
    PROF_END(12);
}

/* ================================================================== strato tipo 3 (0x5988) e nodo di blend tipo 1 (0x57D8) */
static int16_t body_N(const uint8_t* ls) { return (int16_t)LH(ls, 0x7480); }

/* ================================================================== MFC */
/* limite di validita' degli EA (0 = nessuno): il banco lo imposta alla dimensione della sua memoria; un DMA fuori
 * range (o fuori dal LS) setta D->fault e l'esecuzione ritorna ANIM_NOT_COVERED (nessun effetto). */
uint32_t g_anim_ea_limit = 0;
/* anim47 (aggancio in gioco): g_anim_strict = 1 -> ogni DMA deve essere "ben formato" come lo vuole il percorso veloce del runtime
 * (dimensione 1/2/4/8 allineata oppure multipla di 16 <= 16 KB, LS ed EA con gli stessi 4 bit bassi, intervallo EA committato
 * secondo g_anim_ea_ok); altrimenti d->fault e l'esecuzione ritorna ANIM_NOT_COVERED (il chiamante ricade sul lifted). */
int g_anim_strict = 0;
int (*g_anim_ea_ok)(uint32_t ea, uint32_t size) = 0;
__declspec(thread) void (*g_anim_note_cb)(uint32_t ea, uint32_t lsa, uint32_t size) = 0;   /* dopo ogni GET: contabilita' degli overlay del runtime (spu_overlay_note_get) */
typedef struct { const uint8_t* vm; uint8_t* vmw; uint8_t* ls; anim_fx* fx; anim_commit* cm; uint32_t lsa, eal, size, tag, mask; int fault; } dctx;
/* anim46: fx == NULL -> percorso veloce (nessun log degli effetti); vmw != NULL -> i PUT/PUTLLC sono memcpy diretti in memoria */

static int d_ok(dctx* d, uint32_t lsa, uint32_t ea, uint32_t size)
{
    if ((uint64_t)lsa + size > 0x40000u) { d->fault = 1; return 0; }
    if (g_anim_ea_limit && (uint64_t)ea + size > g_anim_ea_limit) { d->fault = 1; return 0; }
    if (g_anim_strict) {
        if (size == 0 || size > 0x4000u || (size >= 16 && (size & 15)) || ((lsa ^ ea) & 15) ||
            (size < 16 && ((size & (size - 1)) || (lsa & (size - 1)))) ||
            (g_anim_ea_ok && !g_anim_ea_ok(ea, size))) { d->fault = 1; return 0; }
    }
    return 1;
}
static void d_get(dctx* d, uint32_t tag, uint32_t lsa, uint32_t ea, uint32_t size)
{
    if (!d_ok(d, lsa, ea, size)) return;
    memcpy(d->ls + lsa, d->vm + ea, size);
    if (g_anim_note_cb) g_anim_note_cb(ea, lsa, size);
    if (d->fx) anim_fx_add(d->fx, 64, (uint8_t)tag, lsa, ea, size, 0, 0);
}
static void d_getl(dctx* d, uint32_t tag, uint32_t lsa, uint32_t list, uint32_t size)
{
    if (d->fx) anim_fx_add(d->fx, 68, (uint8_t)tag, lsa, list, size, d->ls + list, size);
    uint32_t l = lsa;
    for (uint32_t o = 0; o + 8 <= size; o += 8) {
        uint32_t sz = LW(d->ls, list + o) & 0x7FFF, ea = LW(d->ls, list + o + 4);
        if (!d_ok(d, l, ea, sz)) return;
        memcpy(d->ls + l, d->vm + ea, sz);
        if (g_anim_note_cb) g_anim_note_cb(ea, l, sz);
        l += (sz + 15) & ~15u;
    }
}
static void d_put(dctx* d, uint8_t cmd, uint32_t tag, uint32_t lsa, uint32_t ea, uint32_t size)
{
    if (d->fx) anim_fx_add(d->fx, cmd, (uint8_t)tag, lsa, ea, size, d->ls + lsa, size);
    if (d->vmw) { if (!d_ok(d, lsa, ea, size)) return; memcpy(d->vmw + ea, d->ls + lsa, size); }
}
/* 0x6F20: DMA a blocchi da 16384 B (almeno un comando, anche per size 0) */
static void d_chunked_put(dctx* d, uint32_t lsa, uint32_t ea, uint32_t size, uint32_t tag, uint8_t cmd)
{
    do {
        uint32_t c = size > 16384 ? 16384 : size;
        d_put(d, cmd, tag, lsa, ea, c);
        size -= c; lsa += c; ea += c;
    } while (size);
}
/* 0x6F20 per i GET (usato dalle funzioni generate): stessi blocchi da 16 KB, un evento per blocco */
static void d_chunked_get(dctx* d, uint32_t lsa, uint32_t ea, uint32_t size, uint32_t tag)
{
    do {
        uint32_t c = size > 16384 ? 16384 : size;
        d_get(d, tag, lsa, ea, c);
        if (d->fault) return;
        size -= c; lsa += c; ea += c;
    } while (size);
}

/* canali MFC visti dalle funzioni generate (anim_gen.py): wrch registra i parametri, il comando 64 (GET) esegue. */
static void d_wrch(dctx* d, uint32_t ch, uint32_t v)
{
    switch (ch) {
    case MFC_LSA: d->lsa = v; break;
    case MFC_EAL: d->eal = v; break;
    case MFC_Size: d->size = v; break;
    case MFC_TagID: d->tag = v; break;
    case MFC_WrTagMask: d->mask = v; break;
    case MFC_Cmd:
        if (v == 64) d_get(d, d->tag, d->lsa, d->eal, d->size);
        else d->fault = 1;                       /* altri comandi non emessi dalle funzioni generate */
        break;
    default: break;                              /* WrTagUpdate: nessun effetto */
    }
}
static u128 d_rdch(dctx* d, uint32_t ch)
{
    u128 v = spu_zero();
    if (ch == MFC_RdTagStat) v._u32[0] = d->mask;
    return v;
}

/* ---- chiamate a funzioni SPU verificate dai test unit (ABI: argomenti r3.. ; word0 = scalare, quadword intero = r5/r8/r9 dove float) ---- */
#define ARG(n) (R[n]._u32[0])
int g_anim_trace = 0;   /* banco: stampa gli argomenti delle chiamate alle funzioni verificate */
#define TR(name) do { if (g_anim_trace) fprintf(stderr, "  call " name " r3=%X r4=%X r5=%X r6=%X r7=%X r8=%X r9=%X\n", ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9)); } while (0)
static void call_4030(dctx* D, u128* R) { TR("4030"); PROF_BEGIN(15); anim_f4030(D->ls, ARG(3), ARG(4), R[5], ARG(6)); PROF_END(15); }
static void call_4060(dctx* D, u128* R) { TR("4060"); PROF_BEGIN(15); anim_f4060(D->ls, ARG(3), ARG(4), ARG(5), ARG(6)); PROF_END(15); }
static void call_4120(dctx* D, u128* R) { TR("4120"); PROF_BEGIN(15); anim_f4120(D->ls, ARG(3), ARG(4), ARG(5), ARG(6)); PROF_END(15); }
static void call_41E8(dctx* D, u128* R) { TR("41E8"); PROF_BEGIN(16); anim_f41E8(D->ls, ARG(3), ARG(4), ARG(5), ARG(6)); PROF_END(16); }
static void call_4258(dctx* D, u128* R) { TR("4258"); PROF_BEGIN(13); anim_f4258(D->ls, ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), R[8]); PROF_END(13); }
static void call_4300(dctx* D, u128* R) { TR("4300"); PROF_BEGIN(14); anim_f4300(D->ls, ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), R[9]); PROF_END(14); }
static void call_4CE8(dctx* D, u128* R) { TR("4CE8"); PROF_BEGIN(17); anim_f4CE8(D->ls, ARG(3), ARG(4), ARG(5)); PROF_END(17); }
static void call_4D18(dctx* D, u128* R) { TR("4D18"); PROF_BEGIN(18); anim_f4D18(D->ls, ARG(3), ARG(4), ARG(5), ARG(6)); PROF_END(18); }
static void call_5160(dctx* D, u128* R) { TR("5160"); PROF_BEGIN(20); anim_f5160(D->ls, ARG(3), ARG(4)); PROF_END(20); }
static void call_6F20(dctx* D, u128* R) { TR("6F20"); PROF_BEGIN(19); d_chunked_get(D, ARG(3), ARG(4), ARG(5), ARG(6)); PROF_END(19); if (ARG(7) != 64) D->fault = 1; }

/* ---- funzioni generate meccanicamente dal disassemblato (anim_gen.py): 0x53B8 (catena) e 0x5B78 (keyframe) ---- */
static int G_53B8(dctx* D, u128 a3, u128 a4, u128 a5, u128 a6, u128 a7, u128 a8, uint32_t sp);
/* anim_gen_5B78.inc / anim_gen_53B8.inc are a 1:1 mechanical translation of two SPU functions of the game, so they are NOT in this
   repository (git-ignored). Without them the native animation path refuses every command (fault -> ANIM_NOT_COVERED) and the
   game's own lifted SPU job runs instead: same result, a little slower (about 2.4 ms per frame on Ardolis). */
#if defined(__has_include)
#  if __has_include("anim_gen_5B78.inc") && __has_include("anim_gen_53B8.inc")
#    define ANIM_HAVE_GEN 1
#  endif
#endif
#ifdef ANIM_HAVE_GEN
#include "anim_gen_5B78.inc"
#include "anim_gen_53B8.inc"
#else
static int G_5B78(dctx* D, u128 a3, u128 a4, u128 a5, u128 a6, u128 a7, u128 a8, uint32_t sp)
{ (void)a3; (void)a4; (void)a5; (void)a6; (void)a7; (void)a8; (void)sp; D->fault = 1; return 0; }
static int G_53B8(dctx* D, u128 a3, u128 a4, u128 a5, u128 a6, u128 a7, u128 a8, uint32_t sp)
{ (void)a3; (void)a4; (void)a5; (void)a6; (void)a7; (void)a8; (void)sp; D->fault = 1; return 0; }
#endif
static u128 U32(uint32_t v) { u128 r = spu_zero(); r._u32[0] = v; return r; }

/* auto-test di sensibilita' (solo banco): 2 = inverte un bit nel blocco di una foglia tipo 0, 3 = nel blocco dopo una catena */
int g_anim_sab = 0;
static void sab_flip(dctx* D, uint32_t rec) { uint32_t blk = LW(D->ls, 0x79A0 + (LW(D->ls, rec + 4) << 2)); D->ls[(blk + 6) & 0x3FFFF] ^= 0x40; }

/* SP iniziale delle funzioni generate: r1 di 0x6248 dopo il suo prologo (0x3F000 - 240). Lo stack non e' confrontato. */
#define ANIM_SP 0x3EF10u

/* catena di modificatori (0x53B8): blk = blocco del nodo, chain = EA LS dell'inizio catena, n = n_catena (u16) */
static int run_chain(dctx* D, uint32_t rec, uint32_t chain, uint32_t n, uint32_t A2)
{
    uint32_t blk = LW(D->ls, 0x79A0 + (LW(D->ls, rec + 4) << 2));
    PROF_BEGIN(6); int r = G_53B8(D, U32(blk), U32(chain), U32(n), U32(A2), spu_zero(), spu_zero(), ANIM_SP); PROF_END(6);
    if (g_anim_sab == 3) sab_flip(D, rec);
    return r;
}

/* 0x5988: posa di riferimento. rec = record strato (0x74A0+32*i), A1 = blocco GET#1. Catena a node+16. */
static int layer_type3(dctx* D, uint32_t rec, uint32_t A1, uint32_t A2)
{
    uint8_t* ls = D->ls;
    uint32_t N = (uint32_t)(int32_t)body_N(ls);
    uint32_t a = (6 * N + 15) & ~15u, b = (8 * N + 15) & ~15u;
    uint32_t blk = LW(ls, 0x79A0 + (LW(ls, rec + 4) << 2));
    uint32_t src8 = A1 + a;
    anim_f4120(ls, blk, src8, N, 64);
    anim_f4060(ls, blk + 16, src8 + b, N, 64);
    anim_f4060(ls, blk + 32, A1, N, 64);
    anim_f4CE8(ls, blk + 48, N, 64);
    { PROF_BEGIN(3); anim_f5160(ls, blk, N); PROF_END(3); }
    uint32_t node = LW(ls, rec + 12), nch = LH(ls, node + 2);
    if (nch) return run_chain(D, rec, node + 16, nch, A2);
    return 0;
}

/* 0x57D8: blend di due blocchi in table[rec.w1]; catena a node+32 se n_catena != 0. */
static int layer_type1(dctx* D, uint32_t rec, uint32_t A2)
{
    uint8_t* ls = D->ls;
    uint32_t N = (uint32_t)(int32_t)body_N(ls);
    uint32_t node = LW(ls, rec + 12);
    uint32_t mode = LW(ls, node + 12);
    u128 SPL = spu_ila(0x10203);
    u128 w = spu_zero(); w._u32[0] = LW(ls, node + 16);
    u128 param = SPLAT(w);
    uint32_t b1 = LW(ls, 0x79A0 + (LW(ls, rec + 4) << 2));
    uint32_t b2 = LW(ls, 0x79A0 + (LW(ls, rec + 8) << 2));
    switch (mode) {
    case 0: case 1: anim_f4BD8(ls, b1, b2, N, param); break;
    case 2:         anim_f4AE8(ls, b1, b2, N, param); break;
    case 3:         anim_f5250(ls, b1, b2, N, param); break;
    default: break;
    }
    uint32_t nch = LH(ls, node + 2);
    if (nch) return run_chain(D, rec, node + 32, nch, A2);
    return 0;
}

/* ================================================================== 0x46A0: visita ricorsiva dell'albero -> record strato */
typedef struct { uint32_t cnt1, cnt2, cnt3, cursor; int ovf; } walk_st;

static uint32_t tree_walk(uint8_t* ls, walk_st* S, uint32_t P, uint32_t off)
{
    uint32_t node = P + off;
    uint32_t i = S->cnt1++;
    uint32_t rec = 0x74A0 + 32 * i;
    uint32_t type = LH(ls, node);
    uint32_t c2 = S->cnt2;
    SW(ls, rec, type);
    SW(ls, rec + 12, node);
    if (type == 1) {
        uint32_t r1 = tree_walk(ls, S, P, LW(ls, node + 4));
        uint32_t r2 = tree_walk(ls, S, P, LW(ls, node + 8));
        SW(ls, rec + 4, r1);
        SW(ls, rec + 8, r2);
        SW(ls, rec + 16, 0xFFFFFFFFu);
        return c2;
    }
    if (type == 0 || type == 3) {
        uint32_t N = (uint32_t)(int32_t)body_N(ls);
        uint32_t cur = S->cursor;
        uint32_t start = (cur + 127) & ~127u;
        uint32_t end = (((N + 7) & ~7u) << 6) + start;
        S->cursor = end;
        if (end > 0x34B00u) { S->ovf = 1; return c2; }
        SW(ls, 0x79A0 + 4 * S->cnt2, start);
        SW(ls, rec + 4, S->cnt2);
        SW(ls, rec + 8, 0xFFFFFFFFu);
        if (type == 0) {
            uint32_t c3 = S->cnt3;
            SW(ls, 0x78A0 + 8 * c3, 64);
            SW(ls, 0x78A4 + 8 * c3, LW(ls, node + 8));
            SW(ls, rec + 16, c3);
            S->cnt3++;
        } else {
            SW(ls, rec + 16, 0xFFFFFFFFu);
        }
        S->cnt2++;
    }
    return c2;
}

/* pre-scansione in sola lettura dell'albero (dai byte di memoria) per decidere se il comando e' coperto.
 * Limiti = capacita' delle tabelle del job in LS: 32 record strato (0x74A0..0x78A0), 32 elementi GETL (0x78A0..0x79A0),
 * 24 blocchi (0x79A0..0x7A00); oltre il job sovrascriverebbe la tabella vicina (e 0x7A00 = linea coda, 0x7A80 = descrittore):
 * non coperto. Tutti i tipi di nodo (0,1,2,3) e le catene (n_catena != 0) sono coperti; il nodo deve stare nella copia
 * dell'albero e, se ha una catena, anche le n*64 B della catena. */
int g_anim_why = 0;   /* banco: riga del sorgente che ha deciso ANIM_NOT_COVERED */
typedef struct { int nodes, leaves, depth_max; } scan_st;
static int tree_covered(const uint8_t* tree, uint32_t size, uint32_t off, int depth, scan_st* S)
{
    if (depth > 40 || (off & 3) || off + 16 > size || ++S->nodes > 32) { g_anim_why = __LINE__; return 0; }
    if (depth > S->depth_max) S->depth_max = depth;
    const uint8_t* n = tree + off;
    uint32_t type = ((uint32_t)n[0] << 8) | n[1], chain = ((uint32_t)n[2] << 8) | n[3];
    if (type == 0 || type == 3) {
        if (++S->leaves > 24) { g_anim_why = __LINE__; return 0; }
        if (chain && off + 16 + 64 * chain > size) return 0;
        return 1;
    }
    if (type == 1) {
        if (off + 32 > size || (chain && off + 32 + 64 * chain > size)) return 0;
        uint32_t o1 = ((uint32_t)n[4] << 24) | ((uint32_t)n[5] << 16) | ((uint32_t)n[6] << 8) | n[7];
        uint32_t o2 = ((uint32_t)n[8] << 24) | ((uint32_t)n[9] << 16) | ((uint32_t)n[10] << 8) | n[11];
        return tree_covered(tree, size, o1, depth + 1, S) && tree_covered(tree, size, o2, depth + 1, S);
    }
    return 1;   /* tipo 2 e altri: nessun blocco di uscita e nessun effetto (il job lo salta); l'indice restituito al padre e' cnt2 */
}

/* ================================================================== 0x6248 */
static int anim_cmd_impl(const uint8_t* vm, uint8_t* vmw, uint32_t cmd_ea, uint8_t* ls, anim_fx* fx, anim_commit* cm)
{
    const uint32_t DESC = 0x7A80, L = 0x34B00;
    PROF_BEGIN(0);
    dctx D; memset(&D, 0, sizeof D); D.vm = vm; D.vmw = vmw; D.ls = ls; D.fx = fx; D.cm = cm;
    u128 SPL = spu_ila(0x10203);
    uint32_t ea_out = LW(ls, DESC), ea_body = LW(ls, DESC + 4), ea_tree = LW(ls, DESC + 8);
    uint32_t tree_sz = LH(ls, DESC + 12);

    /* ---- copertura: decisa PRIMA di qualunque effetto ---- */
    {
        const uint8_t* body = vm + ea_body;
        int16_t n = (int16_t)(((uint32_t)body[0] << 8) | body[1]);
        if (n < 1 || n > 4000) return (g_anim_why = __LINE__, ANIM_NOT_COVERED);
        if (tree_sz < 32) return (g_anim_why = __LINE__, ANIM_NOT_COVERED);
        const uint8_t* tree = vm + ea_tree;
        uint32_t rootoff = ((uint32_t)tree[12] << 24) | ((uint32_t)tree[13] << 16) | ((uint32_t)tree[14] << 8) | tree[15];
        scan_st sc = { 0, 0, 0 };
        if (rootoff + 16 > tree_sz - 16) return (g_anim_why = __LINE__, ANIM_NOT_COVERED);
        if (!tree_covered(tree + 16, tree_sz - 16, rootoff, 0, &sc)) { if (!g_anim_why) g_anim_why = __LINE__; return ANIM_NOT_COVERED; }
    }

    uint32_t T0 = LW(ls, 0x34BC0), T1 = LW(ls, 0x34BD0), T2 = LW(ls, 0x34BE0), T3 = LW(ls, 0x34BF0), T4 = LW(ls, 0x34C00);
    uint32_t TP = LW(ls, 0x34C60), line_ea = LW(ls, 0x34BA0);
    SQ(ls, 0x34C70, spu_zero());
    uint32_t cur = 0x7B00;

    d_get(&D, T0, 0x7480, ea_body, 32);
    uint32_t N = (uint32_t)(int32_t)body_N(ls);
    uint32_t a = (6 * N + 15) & ~15u, b = (8 * N + 15) & ~15u;
    uint32_t A1 = (cur + 127) & ~127u;
    cur = A1 + 2 * a + b;
    if (cur > L) return ANIM_ARENA_OVF;
    {   /* GET#1: pose di riferimento + variabili di scala per 5160/4D70 (quadword interi) */
        u128 q = LQ(ls, 0x7490);
        d_get(&D, T1, A1, LW(ls, 0x7490 + 4), 2 * a + b);
        u128 r50 = spu_andi(spu_rotqbyi(q, 13), 255), r45 = spu_andi(spu_rotqbyi(q, 14), 255), r43 = spu_andi(spu_rotqbyi(q, 15), 255);
        u128 r40 = spu_sfi(r50, 15);
        u128 r38 = spu_shl(spu_il(1), r40);
        SQ(ls, 0x34B40, spu_shufb(r50, r50, SPL));
        SQ(ls, 0x34B50, spu_shufb(r45, r45, SPL));
        SQ(ls, 0x34B60, spu_shufb(r43, r43, SPL));
        SQ(ls, 0x34B30, spu_shufb(r38, r38, LQ(ls, 0x70D0)));
    }
    uint32_t A2 = (cur + 127) & ~127u;
    cur = A2 + b;
    if (cur > L) return ANIM_ARENA_OVF;
    uint32_t A3 = (cur + 127) & ~127u;
    cur = A3 + tree_sz;
    if (cur > L) return ANIM_ARENA_OVF;
    d_get(&D, T2, A3, ea_tree, tree_sz);
    d_get(&D, T3, A2, LW(ls, 0x7480 + 4), b);
    if (D.fault) return (g_anim_why = __LINE__, ANIM_NOT_COVERED);

    /* ---- sincronizzazione con la PPU: +1 (mod 128) al contatore a linea+10, GETLLAR/PUTLLC ---- */
    if (!cm) {
        if (fx) anim_fx_add(fx, 208, 0, 0x7A00, line_ea, 128, 0, 0);
        memcpy(ls + 0x7A00, vm + line_ea, 128);
        SH(ls, 0x7A00 + 10, (LH(ls, 0x7A00 + 10) + 1) & 127);
        if (fx) anim_fx_add(fx, 180, 0, 0x7A00, line_ea, 128, ls + 0x7A00, 128);
        if (vmw) memcpy(vmw + line_ea, ls + 0x7A00, 128);
    }

    PROF_END(0);
    PROF_BEGIN(1);
    cur = ((cur + 127) & ~127u) + 64;
    if (cur > L) return ANIM_ARENA_OVF;
    walk_st S = { 0, 0, 0, cur, 0 };
    tree_walk(ls, &S, A3 + 16, LW(ls, A3 + 12));
    if (S.ovf) return ANIM_ARENA_OVF;
    {   /* 0x4A58 lascia i contatori nelle word0 di 0x34B70 (strati) e 0x34B80 (GETL) */
        u128 q70 = LQ(ls, 0x34B70), q80 = LQ(ls, 0x34B80);
        q70._u32[0] = S.cnt1; q80._u32[0] = S.cnt3;
        SQ(ls, 0x34B70, q70); SQ(ls, 0x34B80, q80);
    }
    cur = S.cursor;
    uint32_t A5 = (cur + 127) & ~127u;
    cur = A5 + 64 * S.cnt3;
    if (cur > L) return ANIM_ARENA_OVF;
    d_getl(&D, T4, A5, 0x78A0, S.cnt3 * 8);
    if (D.fault) return (g_anim_why = __LINE__, ANIM_NOT_COVERED);

    PROF_END(1);
    uint32_t A6 = (cur + 127) & ~127u;
    if (A6 + 64 > L) return ANIM_ARENA_OVF;
    uint32_t MW = A6 + 128;
    uint32_t msz = (88 * N + 127) & ~127u;
    uint32_t A7 = MW + msz;
    if (A7 > L) return ANIM_ARENA_OVF;
    uint32_t bsz = (20 * N + 127) & ~127u;
    cur = A7 + 2 * bsz;
    if (cur > L) return ANIM_ARENA_OVF;
    /* i vettori temporanei di 4D70 (flag1/flag2/tabella) stanno oltre MW: se sforassero l'arena il job
     * sovrascriverebbe i globali (tag, contatori) -> caso non coperto */
    if (MW + 64 * N + 12800 + 2 * N + 16 > L) return (g_anim_why = __LINE__, ANIM_NOT_COVERED);

    /* ---- strati dal fondo ---- */
    for (int32_t i = (int32_t)S.cnt1 - 1; i >= 0; i--) {
        uint32_t rec = 0x74A0 + 32 * (uint32_t)i;
        uint32_t type = LW(ls, rec);
        int hrc = 0;
        if (type == 1) { PROF_BEGIN(4); hrc = layer_type1(&D, rec, A2); PROF_END(4); }
        else if (type == 3) { PROF_BEGIN(2); hrc = layer_type3(&D, rec, A1, A2); PROF_END(2); }
        else if (type == 0) {
            uint32_t hdr = A5 + (LW(ls, rec + 16) << 6);
            PROF_BEGIN(5); hrc = G_5B78(&D, U32(rec), U32(hdr), U32(A1), U32(MW), U32(A7), U32(A2), ANIM_SP); PROF_END(5);
            if (g_anim_sab == 2) sab_flip(&D, rec);
        }
        /* altri tipi (2, ...): nessuna azione */
        if (D.fault) return (g_anim_why = __LINE__, ANIM_NOT_COVERED);
        if (hrc == 4) return ANIM_ARENA_OVF;          /* heqi: il job si ferma (divisione per zero) */
        if (hrc) return (g_anim_why = __LINE__, ANIM_NOT_COVERED);
    }

    { PROF_BEGIN(7); anim_f4D70(ls, MW, LW(ls, 0x79A0), N, A2); PROF_END(7); }
    PROF_BEGIN(8);

    /* ---- uscita: matrici (PUT a blocchi) + flag di completamento (PUTF 4 B) ---- */
    {
        u128 q = LQ(ls, 0x7A90); q._u32[0] |= 1; SQ(ls, 0x7A90, q);
        if (cm) {   /* anim47: PUT e linea coda li fa il chiamante; qui solo la validazione preventiva dei intervalli di uscita */
            if (g_anim_strict) {
                if (ea_out < 0x10000u || cmd_ea + 16 < 0x10000u || (N * 64 & 15) || (ea_out & 15) || ((cmd_ea + 16) & 15)) return (g_anim_why = __LINE__, ANIM_NOT_COVERED);
                if ((line_ea & 127) || line_ea < 0x10000u) return (g_anim_why = __LINE__, ANIM_NOT_COVERED);
                if (g_anim_ea_ok) {
                    for (uint32_t o = 0; o < N * 64; o += 16384) { uint32_t r = N * 64 - o; if (!g_anim_ea_ok(ea_out + o, r > 16384 ? 16384 : r)) return (g_anim_why = __LINE__, ANIM_NOT_COVERED); }
                    if (!g_anim_ea_ok(cmd_ea + 16, 4) || !g_anim_ea_ok(line_ea, 128)) return (g_anim_why = __LINE__, ANIM_NOT_COVERED);
                }
            }
            cm->cmd_ea = cmd_ea; cm->line_ea = line_ea; cm->ea_out = ea_out; cm->mw = MW; cm->n = N; cm->tp = TP;
        } else {
        d_chunked_put(&D, MW, ea_out, N * 64, TP, 32);
        if (D.fault) return (g_anim_why = __LINE__, ANIM_NOT_COVERED);
        d_put(&D, 34, TP, 0x7A90, cmd_ea + 16, 4);
        }
        u128 c = LQ(ls, 0x34B90);
        for (int k = 0; k < 4; k++) c._u32[k] += 1;
        SQ(ls, 0x34B90, c);
    }
    PROF_END(8);
    return ANIM_OK;
}

/* Wrapper: esegue su una copia di lavoro; ls/fx vengono aggiornati SOLO se il risultato e' ANIM_OK
 * (per ANIM_NOT_COVERED / ANIM_ARENA_OVF / ANIM_BAD_INPUT nessun effetto su ls ne' fx). */
int anim_cmd_ref(const uint8_t* vm, uint32_t cmd_ea, uint8_t* ls, anim_fx* fx)
{
    static uint8_t scratch[0x40000];
    anim_fx tmp; anim_fx_init(&tmp);
    memcpy(scratch, ls, sizeof scratch);
    int rc = anim_cmd_impl(vm, 0, cmd_ea, scratch, &tmp, 0);
    if (rc == ANIM_OK) {
        memcpy(ls, scratch, sizeof scratch);
        for (int i = 0; i < tmp.n; i++) {
            const anim_ev* e = &tmp.ev[i];
            int has = (e->cmd == 32 || e->cmd == 34 || e->cmd == 180 || e->cmd == 200 || e->cmd == 68);
            anim_fx_add(fx, e->cmd, e->tag, e->lsa, e->ea, e->size, has ? tmp.data + e->doff : 0, has ? (e->cmd == 68 ? e->size : e->size) : 0);
        }
    }
    anim_fx_free(&tmp);
    return rc;
}


/* Test unit delle funzioni generate: which = 0x5B78 | 0x53B8; argomenti r3..r8 come word0. Esegue DIRETTAMENTE su ls/fx
 * (nessuna copia di lavoro). Ritorna 0 ok, 4 = heqi (il job si ferma), 5 = DMA fuori range. */
int anim_gen_call(int which, const uint8_t* vm, uint8_t* ls, anim_fx* fx, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7, uint32_t a8, uint32_t sp)
{
    dctx D; memset(&D, 0, sizeof D); D.vm = vm; D.ls = ls; D.fx = fx;
    int rc;
    if (which == 0x5B78) rc = G_5B78(&D, U32(a3), U32(a4), U32(a5), U32(a6), U32(a7), U32(a8), sp);
    else rc = G_53B8(&D, U32(a3), U32(a4), U32(a5), U32(a6), U32(a7), U32(a8), sp);
    if (D.fault && !rc) rc = 5;
    return rc;
}


/* anim46: percorso VELOCE. In-place su `ls` (nessuna copia di lavoro da 256 KB), nessun log degli effetti, GET/GETL = memcpy da `vm`,
 * PUT / PUTF / PUTLLC = memcpy diretti in `vm` (scrivibile). ATTENZIONE: per rc != ANIM_OK `ls` (e la linea coda / la memoria gia' scritte)
 * possono essere sporche: un aggancio reale deve fare prima un controllo preventivo (arena, heqi) o lavorare su copia. Solo per misure/replay. */
int anim_cmd_fast(uint8_t* vm, uint32_t cmd_ea, uint8_t* ls)
{
    return anim_cmd_impl(vm, vm, cmd_ea, ls, 0, 0);
}

/* anim47: vedi anim_ref.h. vm e' solo letta. */
int anim_cmd_compute(const uint8_t* vm, uint32_t cmd_ea, uint8_t* ls, anim_commit* cm)
{
    return anim_cmd_impl(vm, 0, cmd_ea, ls, 0, cm);
}
