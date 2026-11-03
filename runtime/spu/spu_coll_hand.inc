/* spu_coll_hand.inc (coll102) -- versione SCRITTA A MANO (intrinseci SSE4.1/AVX2/FMA) di un ciclo caldo del ticket di collisione nativo.
 * Codice NOSTRO: reimplementa la logica osservata (nessun codice del gioco copiato). Da includere DOPO spu_helpers.h / spu_coll_fast.inc,
 * PRIMA dei file generati spu_coll_l2_gen_*.inc (che contengono la macro COLL_HAND_FB_HOOK() davanti al ciclo).
 *
 * F_FB, ciclo 01644..0170C (record da 48 B -> test di sovrapposizione di una scatola -> compattazione dell'uscita):
 *   per ogni record: 4 float (centro) a +0, 3 half-word a +16 (coordinate intere, scala 2^-8), una parola a +32 e due parole a +40.
 *   si calcolano due scatole intermedie (10 operazioni float, tutte sulle corsie 0..2: la corsia 3 non influenza nulla),
 *   si scrivono SEMPRE una parola (a 'dest1') e un quad (a 'dest2'), e i due puntatori avanzano (4 e 16 byte) solo se il test passa;
 *   se dest1 supera il limite (0x2417C) entrambi tornano ai valori-limite (0x2417C / 0x281F0).
 *
 * Esattezza: le operazioni float sono le STESSE istruzioni host che usa spu_coll_fast.inc (add/sub/mul/fma sotto MXCSR RZ|DAZ|FTZ).
 * Cancello: se una qualunque delle 10 operazioni (corsie 0..2) ha |risultato| > 0x7F7FFFFE (esponente 255 / overflow: l'originale ricadrebbe
 * nel percorso "esatto" spu__rz_exact) l'iterazione NON si esegue qui: nessun effetto, e la fa il corpo originale generato.
 * Il -0 intermedio non viene corretto in +0: il segno di uno zero non cambia mai nessun risultato non nullo ne' i confronti (fcgt: chiave 0).
 * L'ULTIMA iterazione del ciclo la esegue sempre il corpo originale: i registri R[] a fine ciclo (anche i temporanei) sono quindi identici.
 * Ingressi controllati a ogni chiamata: MXCSR con RZ|DAZ|FTZ, le tre costanti di permutazione R24/R23/R12 = quelle attese, lunghezza multipla di 48.
 * Interruttore: runtime TOD_COLL_HAND (1.1: default 1 = acceso, =0 spento); compile-time -DCOLL_HAND=0 toglie tutto. */
#ifndef COLL_HAND_INC
#define COLL_HAND_INC
#ifndef COLL_HAND
#define COLL_HAND 1
#endif
#if COLL_HAND
#include <immintrin.h>

static int s_coll_hand;                         /* 1 acceso (default dalla 1.1), 0 spento */
#ifdef COLL_HAND_STATS
static long long s_ch_pass, s_ch_over, s_ch_rec[2];
#define CH_STAT(x) (x)
#else
#define CH_STAT(x) ((void)0)
#endif
static long long s_ch_r[3]; static long long s_ch_calls, s_ch_iters, s_ch_bad, s_ch_skip;   /* statistiche (non atomiche: solo diagnostica) */

static inline __attribute__((always_inline)) uint32_t ch_lsa(uint32_t a, uint32_t mask) { return a & mask; }

static __attribute__((noinline)) void coll_hand_fb(u128* R, uint8_t* ls)
{
    s_ch_calls++;
    /* 1) MXCSR: RZ (RC=11) + FTZ + DAZ */
    if ((_mm_getcsr() & 0xE040u) != 0xE040u) { s_ch_skip++; s_ch_r[0]++; return; }
    /* 2) costanti di permutazione attese (R24, R23, R12 vengono dall'LS del gioco) */
    const __m128i c24 = _mm_setr_epi32((int)0x80800001, (int)0x80800203, (int)0x80800405, (int)0x80808080);
    const __m128i c23 = _mm_setr_epi32((int)0x00010203, (int)0x08090A0B, (int)0x80800C0D, (int)0x80808080);
    const __m128i c12 = _mm_setr_epi32((int)0x04050607, (int)0x08090A0B, (int)0x00010203, (int)0x00010203);
    #define CH_LD(i) _mm_loadu_si128((const __m128i*)R[i]._u32)
    #define CH_EQ(a, b) (_mm_movemask_epi8(_mm_cmpeq_epi32((a), (b))) == 0xFFFF)
    if (!(CH_EQ(CH_LD(24), c24) && CH_EQ(CH_LD(23), c23) && CH_EQ(CH_LD(12), c12))) { s_ch_skip++; s_ch_r[1]++; return; }
    /* 3) numero di iterazioni: il ciclo termina quando R9[0] == R21[0]; se la distanza non e' un multiplo di 48 (ciclo "infinito") o e' enorme: originale */
    uint32_t a9 = R[9]._u32[0];
    const uint32_t span = R[21]._u32[0] - a9;
    if (span < 96u || span % 48u || span > 48u * 8192u) { s_ch_skip++; s_ch_r[2]++; return; }
    const uint32_t n = span / 48u - 1u;          /* iterazioni a mano; l'ultima resta all'originale */

    const __m128 r22 = _mm_castsi128_ps(CH_LD(22)), r18 = _mm_castsi128_ps(CH_LD(18)), r19 = _mm_castsi128_ps(CH_LD(19));
    const __m128 r17 = _mm_castsi128_ps(CH_LD(17)), r16 = _mm_castsi128_ps(CH_LD(16)), r14 = _mm_castsi128_ps(CH_LD(14));
    const __m128 r13 = _mm_castsi128_ps(CH_LD(13));
    const __m128i lim1 = CH_LD(15), lim2 = CH_LD(20);
    __m128i v10 = CH_LD(10), v11 = CH_LD(11);
    uint32_t a10 = R[10]._u32[0], a11 = R[11]._u32[0];

    const __m128i sw   = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    const __m128i k61  = _mm_setr_epi8(1, 0, (char)0x80, (char)0x80, 3, 2, (char)0x80, (char)0x80, 5, 4, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80);
    const __m128i k59  = _mm_setr_epi8(0, 1, 2, 3, 8, 9, 10, 11, (char)0x80, (char)0x80, 12, 13, (char)0x80, (char)0x80, (char)0x80, (char)0x80);
    const __m128 sc    = _mm_castsi128_ps(_mm_set1_epi32(0x3B800000));          /* 2^-8 */
    const __m128i absm = _mm_set1_epi32(0x7FFFFFFF);
    const __m128i lim  = _mm_set1_epi32(0x7F7FFFFE);
    const __m128i sgn  = _mm_set1_epi32((int)0x80000000);
    const __m128i inc_p10 = _mm_set1_epi32(4), inc_f10 = _mm_setr_epi32(0, 4, 4, 4);
    const __m128i inc_p11 = _mm_set1_epi32(16), inc_f11 = _mm_setr_epi32(0, 16, 16, 16);
    const __m128i lim1x = _mm_xor_si128(lim1, sgn);

    uint32_t done = 0;
    for (; done < n; done++) {
        const __m128i raw0 = _mm_load_si128((const __m128i*)(ls + ch_lsa(a9, 0x3FFF0u)));
        const __m128i raw1 = _mm_load_si128((const __m128i*)(ls + ch_lsa(a9 + 16u, 0x3FFF0u)));
        const __m128i raw2 = _mm_load_si128((const __m128i*)(ls + ch_lsa(a9 + 32u, 0x3FFF0u)));
        const __m128 f57 = _mm_castsi128_ps(_mm_shuffle_epi8(raw0, sw));
        const __m128 f58 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_shuffle_epi8(raw1, k61)), sc);   /* int 0..65535 * 2^-8: exact */
        __m128i acc;
        const __m128 f54 = _mm_add_ps(f57, f58);                                   acc = _mm_and_si128(_mm_castps_si128(f54), absm);
        const __m128 f56 = _mm_add_ps(f57, f54);                                   acc = _mm_max_epu32(acc, _mm_and_si128(_mm_castps_si128(f56), absm));
        const __m128 f55 = _mm_mul_ps(f56, r22);                                   acc = _mm_max_epu32(acc, _mm_and_si128(_mm_castps_si128(f55), absm));
        const __m128 f53 = _mm_sub_ps(r18, f55);                                   acc = _mm_max_epu32(acc, _mm_and_si128(_mm_castps_si128(f53), absm));
        const __m128 f52 = _mm_sub_ps(f54, f55);                                   acc = _mm_max_epu32(acc, _mm_and_si128(_mm_castps_si128(f52), absm));
        const __m128 f51 = _mm_mul_ps(f53, r19);                                   acc = _mm_max_epu32(acc, _mm_and_si128(_mm_castps_si128(f51), absm));
        const __m128 f49 = _mm_mul_ps(f52, r17);                                   acc = _mm_max_epu32(acc, _mm_and_si128(_mm_castps_si128(f49), absm));
        const __m128 f46 = _mm_add_ps(f52, r16);                                   acc = _mm_max_epu32(acc, _mm_and_si128(_mm_castps_si128(f46), absm));
        const __m128 f50 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(f53), 0x09));   /* [1,2,0,0] */
        const __m128 f48 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(f52), 0x09));
        const __m128 f44 = _mm_fnmadd_ps(f50, r14, f51);                           acc = _mm_max_epu32(acc, _mm_and_si128(_mm_castps_si128(f44), absm));
        const __m128 f43 = _mm_fmadd_ps(f48, r13, f49);                            acc = _mm_max_epu32(acc, _mm_and_si128(_mm_castps_si128(f43), absm));
        /* cancello: corsie 0..2 oltre 0x7F7FFFFE -> l'iterazione la fa l'originale */
        if (__builtin_expect((_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(acc, lim))) & 7) != 0, 0)) { s_ch_bad++; break; }
        const __m128 a53 = _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(f53), absm));
        const __m128 f42 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(f44), 0x09));
        const __m128 f40 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(f43), 0x09));
        const __m128 a42 = _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(f42), absm));
        const __m128 m = _mm_or_ps(_mm_cmpgt_ps(a53, f46), _mm_cmpgt_ps(a42, f40));
        const int pass = (_mm_movemask_ps(m) & 7) == 0;
        /* scritture, nell'ordine dell'originale: parola a dest1, poi quad a dest2 */
        { uint32_t w; memcpy(&w, (const uint8_t*)&raw2, 4); memcpy(ls + ch_lsa(a10, 0x3FFFCu), &w, 4); }
        _mm_store_si128((__m128i*)(ls + ch_lsa(a11, 0x3FFF0u)), _mm_shuffle_epi8(raw2, k59));
        /* avanzamento dei puntatori e limite (confronto unsigned per corsia) */
        const __m128i n10 = _mm_add_epi32(v10, pass ? inc_p10 : inc_f10), n11 = _mm_add_epi32(v11, pass ? inc_p11 : inc_f11);
        const __m128i over = _mm_cmpgt_epi32(_mm_xor_si128(n10, sgn), lim1x);
        CH_STAT((s_ch_pass += pass, s_ch_over += (_mm_movemask_ps(_mm_castsi128_ps(over)) & 1)));
        v10 = _mm_blendv_epi8(n10, lim1, over); v11 = _mm_blendv_epi8(n11, lim2, over);
        a10 = (uint32_t)_mm_cvtsi128_si32(v10); a11 = (uint32_t)_mm_cvtsi128_si32(v11);
        a9 += 48u;
    }
    if (done) {
        s_ch_iters += done;
        const __m128i d = _mm_add_epi32(CH_LD(9), _mm_set1_epi32((int)(48u * done)));
        _mm_storeu_si128((__m128i*)R[9]._u32, d);
        _mm_storeu_si128((__m128i*)R[10]._u32, v10);
        _mm_storeu_si128((__m128i*)R[11]._u32, v11);
    }
    #undef CH_LD
    #undef CH_EQ
}

/* ---- F_B60, ciclo 01004..010D4 (coll102, parte A) -------------------------------------------------------------------------------------------
 * Lista di voci da 32 B (R9 .. R11): per ogni voce si legge un indice a 16 bit (+12) e dei flag a 16 bit (+14); si imposta il bit 'indice' di una
 * bitmap di quad da 128 bit (base R85) ricordando se era gia' impostato; si scrive nel buffer di uscita (R10, record da 48 B) la voce piu' un terzo
 * quad [R92 + indice*128, parole 1..3 della seconda meta' della voce]; l'uscita avanza di 48 B solo se (bit non gia' impostato) && (flag & 8) &&
 * ((flag & R90) != 0 || (flag & R91) == 0). Tutto intero: nessun cancello, nessuna ricaduta; le ultime iterazione la fa sempre l'originale.
 * Lavora sui byte grezzi dell'LS (ordine SPU) con le stesse maschere di indirizzo dell'originale, stesso ordine di load/store. */
static long long s_cb_calls, s_cb_iters, s_cb_skip;
static __attribute__((noinline)) void coll_hand_b60(u128* R, uint8_t* ls)
{
    s_cb_calls++;
    #define CB_LD(i) _mm_loadu_si128((const __m128i*)R[i]._u32)
    #define CB_EQ(a, b) (_mm_movemask_epi8(_mm_cmpeq_epi32((a), (b))) == 0xFFFF)
    if (!(CB_EQ(CB_LD(12), _mm_set1_epi32(0xFFFF)) && CB_EQ(CB_LD(13), _mm_set1_epi32(0x00010203)) && CB_EQ(CB_LD(14), _mm_set1_epi32(1)) &&
          CB_EQ(CB_LD(15), _mm_setr_epi32(-1, 0, 0, 0)) &&
          CB_EQ(CB_LD(16), _mm_setr_epi32(0x00010203, 0x14151617, 0x18191A1B, 0x1C1D1E1F)))) { s_cb_skip++; return; }
    uint32_t p = R[9]._u32[0];
    const uint32_t span = R[11]._u32[0] - p;
    if (span < 64u || span % 32u || span > 32u * 4096u) { s_cb_skip++; return; }
    const uint32_t n = span / 32u - 1u;
    const uint32_t b85 = R[85]._u32[0], b92 = R[92]._u32[0];
    const __m128i r90 = CB_LD(90), r91 = CB_LD(91), zero = _mm_setzero_si128(), ones = _mm_set1_epi32(-1);
    __m128i v10 = CB_LD(10);
    uint32_t o = R[10]._u32[0];
    uint32_t k = 0;
    for (; k < n; k++, p += 32u) {
        const uint8_t* q = ls + ((p + 12u) & 0x3FFF0u);
        const uint32_t rot = (p + 26u) & 15u;
        const uint32_t idx = ((uint32_t)q[(rot + 2u) & 15u] << 8) | q[(rot + 3u) & 15u];
        uint8_t* bq = ls + ((b85 + ((idx >> 7) << 4)) & 0x3FFF0u);
        const uint32_t j = (3u - ((idx & 127u) >> 3)) & 15u, bit = 1u << (idx & 7u);
        const uint32_t already = bq[j] & bit;
        bq[j] |= (uint8_t)bit;
        const uint8_t* q2 = ls + ((p + 14u) & 0x3FFF0u);
        const uint32_t rot2 = (p + 28u) & 15u;
        const uint32_t f16 = ((uint32_t)q2[(rot2 + 2u) & 15u] << 8) | q2[(rot2 + 3u) & 15u];
        const __m128i a1 = _mm_load_si128((const __m128i*)(ls + ((p + 16u) & 0x3FFF0u)));
        const __m128i a0 = _mm_load_si128((const __m128i*)(ls + ((p + 0u) & 0x3FFF0u)));
        _mm_store_si128((__m128i*)(ls + ((o + 16u) & 0x3FFF0u)), a1);
        _mm_store_si128((__m128i*)(ls + ((o + 0u) & 0x3FFF0u)), a0);
        const uint32_t ptr = b92 + (idx << 7);
        const __m128i a3 = _mm_blend_epi32(a1, _mm_set1_epi32((int)__builtin_bswap32(ptr)), 1);     /* parola 0 = ptr (big-endian), il resto dalla voce */
        _mm_store_si128((__m128i*)(ls + ((o + 32u) & 0x3FFF0u)), a3);
        const __m128i hwi = _mm_set1_epi32((int)f16);
        const __m128i r22 = _mm_or_si128(_mm_cmpeq_epi32(_mm_and_si128(hwi, r90), zero), _mm_xor_si128(_mm_cmpeq_epi32(_mm_and_si128(hwi, r91), zero), ones));
        const __m128i r19 = _mm_cmpeq_epi32(_mm_and_si128(hwi, _mm_set1_epi32(8)), _mm_set1_epi32(8));
        const __m128i r20 = _mm_or_si128(_mm_cvtsi32_si128((int)already), r22);
        const __m128i r8 = _mm_and_si128(_mm_and_si128(_mm_cmpeq_epi32(r20, zero), r19), _mm_set1_epi32(1));
        v10 = _mm_add_epi32(v10, _mm_slli_epi32(_mm_add_epi32(_mm_add_epi32(r8, r8), r8), 4));
        o = (uint32_t)_mm_cvtsi128_si32(v10);
    }
    if (k) {
        s_cb_iters += k;
        _mm_storeu_si128((__m128i*)R[9]._u32, _mm_add_epi32(CB_LD(9), _mm_set1_epi32((int)(32u * k))));
        _mm_storeu_si128((__m128i*)R[10]._u32, v10);
    }
    #undef CB_LD
    #undef CB_EQ
}
#define COLL_HAND_B60_HOOK() do { if (__builtin_expect(s_coll_hand > 0, 0)) coll_hand_b60(R, ls); } while (0)
#define COLL_HAND_FB_HOOK() do { if (__builtin_expect(s_coll_hand > 0, 0)) coll_hand_fb(R, ls); } while (0)
static void coll_hand_init(void)
{
    const char* e = getenv("TOD_COLL_HAND");
    s_coll_hand = (e && *e == '0') ? 0 : 1;   /* 1.1: default ON, =0 off */
    if (s_coll_hand) fprintf(stderr, "[coll-hand] TOD_COLL_HAND=1: F_FB a mano (con cancello e ricaduta sul corpo generato)\n");
}
static void coll_hand_report(void)
{
    if (s_coll_hand) fprintf(stderr, "[coll-hand] B60: chiamate %lld iterazioni a mano %lld saltate %lld\n", s_cb_calls, s_cb_iters, s_cb_skip);
    if (s_coll_hand) fprintf(stderr, "[coll-hand] chiamate %lld iterazioni a mano %lld ricadute(cancello) %lld saltate(ingressi) %lld\n", s_ch_calls, s_ch_iters, s_ch_bad, s_ch_skip);
}
#else
#define COLL_HAND_FB_HOOK() do { } while (0)
#define COLL_HAND_B60_HOOK() do { } while (0)
static void coll_hand_init(void) {}
static void coll_hand_report(void) {}
#endif
#endif
