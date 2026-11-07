/* hand_d647.inc (wid45) -- BOZZA di riscrittura nativa A MANO (SSE4.1/AVX2/FMA) dei due cicli caldi del job SPURS
 * jobmod_d6470883a345 (wid4/wid5):  spu_func_00005908_p (loop loc_00005908) e spu_func_00005BB8_p (loop loc_00005BB8).
 * Codice NOSTRO: reimplementa la logica osservata (nessun codice del gioco copiato). Da includere DOPO spu_helpers.h
 * (serve solo il tipo u128; le operazioni float NON usano gli helper: sono le istruzioni host sotto MXCSR RZ|DAZ|FTZ).
 * Stesso schema di spu_coll_hand.inc: controlli di ingresso -> N-1 iterazioni a mano -> ricaduta: la funzione ritorna quante
 * iterazioni ha fatto (k) e SCRIVE in R[] solo i registri che portano lo stato del ciclo; poi il corpo ORIGINALE del ciclo
 * (label loc_xxx) riparte dallo stato scritto ed esegue le iterazioni rimanenti (almeno l'ULTIMA, sempre fatta dall'originale:
 * temporanei e registri di uscita sono quindi identici all'originale). k=0 => nessun effetto.
 *
 * ===== 5908: trasformazione + outcode di vertici (pipeline a 3 stadi) =====
 *  per ogni vertice (stride R7 byte, 4 s16 big-endian xyzw a R5): V = (float)s16 (csflt 139: esatto), poi
 *  T = M0*Vx + M1*Vy + M2*Vz + Tr  con M0=R13 M1=R12 M2=R23 Tr=R29 (3 fma: x,y,z della pipeline di 2 vertici prima),
 *  outcode: lt0.xyz (r31) e gt1.xyz (r32) -> parola w = 0xFFFF0000 | 0x11 | bit (lt: 7,6,5 ; gt: 3,2,1)
 *  scritture per iterazione (LS, nell'ordine dell'originale): R9 (V di 3 iterazioni prima) a [R4+R28], poi (Tx,Ty,Tz,w) del
 *  vertice precedente a [R3+R28]; R28 += 16. Ciclo do-while: n = R6/R7 iterazioni (R6 -= R7 finche' != 0).
 * ===== 5BB8: backface/outcode culling di triangoli con compattazione della lista indici =====
 *  lista di indici u16 a stride 6 (R4), per ogni triangolo v0,v1,v2 = quad a [R5 + idx*16] (x,y,z float + w = 0xFFFF00cc);
 *  scarta se (v0.w & v1.w & v2.w) != 0xFFFF0011 (stesso piano di fuori) o se NON (a.x*b.y > a.y*b.x) con a=v1-v0 b=v2-v0 (area<=0);
 *  i 6 byte del triangolo (3 u16) vengono scritti a [R3] con un merge a quadword (tabelle di permutazione a LS 0x5700/0x5780), R3 += 6 se tenuto.
 *  Pipeline a 2 stadi (la decisione usa i dati del triangolo letto un'iterazione prima).
 *
 * ESATTEZZA float: RZ|DAZ|FTZ host = troncamento SPU con denormali azzerati. Cancello: se un risultato float (corsie rilevanti) ha
 *  |bit| > 0x7F7FFFFE (inf/NaN host = esponente 255 SPU, o overflow) l'iterazione NON si esegue qui (nessun effetto) e il ciclo
 *  prosegue nell'originale. Zeri -0 -> +0 dove il valore e' scritto/portato (5908). 5BB8: solo le corsie 0,1 contano
 *  (la corsia w contiene 0xFFFF00cc = esponente 255: il motivo per cui l'originale e' lento) e il confronto e' a chiavi intere (fcgt).
 * Interruttore: TOD_D647_HAND (default da decidere: spento finche' check mode + gioco non lo approvano).
 */
#ifndef HAND_D647_INC
#define HAND_D647_INC
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define D647_LDR(i)    _mm_loadu_si128((const __m128i*)R[i]._u32)
#define D647_STR(i, v) _mm_storeu_si128((__m128i*)R[i]._u32, (v))
#define D647_EQ(a, b)  (_mm_movemask_epi8(_mm_cmpeq_epi8((a), (b))) == 0xFFFF)
#define D647_SPLAT(w)  _mm_set1_epi32((int)(w))

static inline __attribute__((always_inline)) __m128i d647_bs(__m128i v) {          /* byte-swap di ogni parola: LS grezza <-> u128 host */
    return _mm_shuffle_epi8(v, _mm_setr_epi8(3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12));
}
static inline __attribute__((always_inline)) __m128i d647_ldq(const uint8_t* ls, uint32_t a) {   /* quad grezzo (ordine SPU) */
    return _mm_load_si128((const __m128i*)(ls + (a & 0x3FFF0u)));
}
static inline __attribute__((always_inline)) void d647_stq(uint8_t* ls, uint32_t a, __m128i raw) {
    _mm_store_si128((__m128i*)(ls + (a & 0x3FFF0u)), raw);
}
static inline __attribute__((always_inline)) __m128i d647_absm(__m128 f) { return _mm_and_si128(_mm_castps_si128(f), _mm_set1_epi32(0x7FFFFFFF)); }
static inline __attribute__((always_inline)) __m128i d647_fixz(__m128 f) {          /* -0 -> +0 (la SPU non produce -0) */
    const __m128i i = _mm_castps_si128(f);
    return _mm_andnot_si128(_mm_cmpeq_epi32(_mm_and_si128(i, _mm_set1_epi32(0x7FFFFFFF)), _mm_setzero_si128()), i);
}
/* shufb con byte di controllo 0..31 (nessun codice speciale 0x80+): sorgenti a (0..15) e b (16..31), in ordine SPU grezzo */
static inline __attribute__((always_inline)) __m128i d647_shuf2(__m128i a, __m128i b, __m128i c) {
    return _mm_blendv_epi8(_mm_shuffle_epi8(a, c), _mm_shuffle_epi8(b, c), _mm_cmpgt_epi8(c, _mm_set1_epi8(15)));
}
/* chiave fcgt SPU (spu__sf_key): esponente 0 -> 0, altrimenti +-|x| */
static inline int64_t d647_key(uint32_t x) {
    if (!(x & 0x7F800000u)) return 0;
    const int64_t m = x & 0x7FFFFFFFu;
    return (x & 0x80000000u) ? -m : m;
}
#define D647_MAXN (1u << 20)

/* ------------------------------------------------------------------------------------------------------------------------------ */
/* 5908. Ingressi letti da R[]: 3,4,5,6,7, 8,9,10,11, 12,13,23,29, 14..22 (costanti), 26,27,28,30,31,32,33,35.
 * Scrive (se k>0): 5,6,8,9,10,11,26,27,28,30,31,32,33,35. Ritorna k. */
static __attribute__((noinline)) int d647_hand_5908(u128* R, uint8_t* ls)
{
    /* costanti di permutazione/immediati attesi (le tabelle sono nell'immagine del modulo a LS 0x56B0..0x56F0) */
    if (!(D647_EQ(D647_LDR(14), _mm_setr_epi32((int)0x80808080, (int)0x80808080, (int)0x000408C0, (int)0x101418C0)) &&
          D647_EQ(D647_LDR(15), _mm_setr_epi32((int)0x00010203, (int)0x04050607, (int)0x08090A0B, (int)0xC0C01213)) &&
          D647_EQ(D647_LDR(16), _mm_setr_epi32((int)0x00018080, (int)0x02038080, (int)0x04058080, (int)0x06078080)) &&
          D647_EQ(D647_LDR(17), D647_SPLAT(0x03030303)) && D647_EQ(D647_LDR(18), D647_SPLAT(0x08090A0B)) &&
          D647_EQ(D647_LDR(19), D647_SPLAT(0x04050607)) && D647_EQ(D647_LDR(20), D647_SPLAT(0x00010203)) &&
          D647_EQ(D647_LDR(21), D647_SPLAT(0x3F800000)) && D647_EQ(D647_LDR(22), _mm_setzero_si128())))
        return 0;
    uint32_t A = R[5]._u32[0];
    /* coerenza della pipeline all'ingresso: r33 = splat del byte basso di r5 (l'originale usa r33, la bozza usa r5) */
    if (!D647_EQ(D647_LDR(33), D647_SPLAT((A & 0xFFu) * 0x01010101u))) return 0;
    const uint32_t r6 = R[6]._u32[0], r7 = R[7]._u32[0];
    if (!r7 || (r6 % r7)) return 0;
    const uint32_t n = r6 / r7;
    if (n < 2u || n > D647_MAXN) return 0;

    const unsigned csr_old = _mm_getcsr();
    _mm_setcsr((csr_old & ~0x6000u) | 0xE040u);                         /* RZ | FTZ | DAZ */

    const __m128 M0 = _mm_castsi128_ps(D647_LDR(13)), M1 = _mm_castsi128_ps(D647_LDR(12)), M2 = _mm_castsi128_ps(D647_LDR(23)),
                 Tr = _mm_castsi128_ps(D647_LDR(29));
    const __m128 one = _mm_set1_ps(1.0f), zero = _mm_setzero_ps();
    const __m128i lim = _mm_set1_epi32(0x7F7FFFFE);
    const __m128i sw16 = _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80);
    __m128i r8 = D647_LDR(8), r9 = D647_LDR(9), r10 = D647_LDR(10), r11 = D647_LDR(11), r30 = D647_LDR(30), r35 = D647_LDR(35),
            r31 = D647_LDR(31), r32 = D647_LDR(32);
    const uint32_t o3 = R[3]._u32[0], o4 = R[4]._u32[0];
    uint32_t o28 = R[28]._u32[0];
    /* i due quad del vertice corrente (r26/r27 dell'originale): si portano come registri, ORDINE SPU grezzo. L'originale li carica per
     * l'iterazione successiva DOPO la prima scrittura (R9) e PRIMA della seconda (quad di uscita): se le uscite si sovrappongono
     * all'ingresso conta, quindi la bozza rilegge nello stesso punto. */
    __m128i Q0 = d647_bs(D647_LDR(26)), Q1 = d647_bs(D647_LDR(27));
    const __m128i iota = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    uint32_t k = 0;
    for (; k + 1u < n; k++) {
        /* 3 fma (RZ): 4 corsie, cancello su tutte (r2 = r8 viene salvato in R2 dall'originale: la corsia w conta) */
        const __m128 sx = _mm_castsi128_ps(_mm_shuffle_epi32(r11, 0x00)), sy = _mm_castsi128_ps(_mm_shuffle_epi32(r11, 0x55));
        const __m128 t25 = _mm_fmadd_ps(M0, sx, Tr);
        const __m128 t35 = _mm_fmadd_ps(M1, sy, t25);
        const __m128 t8  = _mm_fmadd_ps(M2, _mm_castsi128_ps(r30), _mm_castsi128_ps(r35));
        const __m128i acc = _mm_max_epu32(_mm_max_epu32(d647_absm(t25), d647_absm(t35)), d647_absm(t8));
        if (__builtin_expect(_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(acc, lim))) != 0, 0)) break;
        /* outcode dalle maschere dell'iterazione precedente: bit 24 di ogni corsia (= LSB del primo byte, come gbb) */
        const int lt = _mm_movemask_ps(_mm_castsi128_ps(_mm_slli_epi32(r31, 7))) & 7;
        const int gt = _mm_movemask_ps(_mm_castsi128_ps(_mm_slli_epi32(r32, 7))) & 7;
        const uint32_t code = 0x11u | ((uint32_t)(lt & 1) << 7) | ((uint32_t)(lt & 2) << 5) | ((uint32_t)(lt & 4) << 3) |
                              ((uint32_t)(gt & 1) << 3) | ((uint32_t)(gt & 2) << 1) | ((uint32_t)(gt & 4) >> 1);
        /* decodifica del vertice (letta PRIMA delle scritture, come l'originale che carica a fine iterazione precedente) */
        const __m128i w8 = d647_shuf2(Q0, Q1, _mm_add_epi8(iota, _mm_set1_epi8((char)(A & 15u))));       /* byte N..N+7 dei 32 */
        const __m128i V = _mm_castps_si128(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_shuffle_epi8(w8, sw16))));
        /* scrittura 1: R9 a [R4+R28]; poi il caricamento dei quad del vertice successivo (a r5+r7); poi scrittura 2: (r8 vecchio xyz, 0xFFFF0000|code) a [R3+R28] */
        d647_stq(ls, o4 + o28, d647_bs(r9));
        A += r7;
        Q0 = d647_ldq(ls, A); Q1 = d647_ldq(ls, A + 0x10u);
        d647_stq(ls, o3 + o28, d647_bs(_mm_insert_epi32(r8, (int)(0xFFFF0000u | code), 3)));
        /* avanzamento della pipeline */
        r8 = d647_fixz(t8);
        r9 = r10; r10 = r11;
        r30 = _mm_shuffle_epi32(r11, 0xAA);                            /* splat z di r10 (= r11 vecchio) */
        r11 = V;
        r35 = d647_fixz(t35);
        {   const __m128 f8 = _mm_castsi128_ps(r8);
            r32 = _mm_castps_si128(_mm_cmpgt_ps(f8, one));              /* fcgt(r8, 1.0) */
            r31 = _mm_castps_si128(_mm_cmplt_ps(f8, zero)); }           /* fcgt(0, r8) (r8 gia' senza -0, nessun denormale) */
        (void)t25;                                                      /* r25 esce solo come temporaneo: lo rifa l'originale */
        o28 += 16u;
    }
    if (k) {
        const __m128i kk = _mm_set1_epi32((int)k), s7 = D647_LDR(7);
        const __m128i dk = _mm_mullo_epi32(s7, kk);
        D647_STR(5, _mm_add_epi32(D647_LDR(5), dk));
        D647_STR(6, _mm_sub_epi32(D647_LDR(6), dk));
        D647_STR(28, _mm_add_epi32(D647_LDR(28), _mm_slli_epi32(kk, 4)));
        D647_STR(8, r8); D647_STR(9, r9); D647_STR(10, r10); D647_STR(11, r11);
        D647_STR(30, r30); D647_STR(35, r35); D647_STR(31, r31); D647_STR(32, r32);
        D647_STR(33, D647_SPLAT((A & 0xFFu) * 0x01010101u));
        D647_STR(26, d647_bs(Q0));
        D647_STR(27, d647_bs(Q1));
    }
    _mm_setcsr(csr_old);
    return (int)k;
}

/* ------------------------------------------------------------------------------------------------------------------------------ */
/* 5BB8. Ingressi: 3,4,5,6, 9,10,11, 12,13,14..21 (costanti), 28,29,30,31,35,36,43,44.
 * Scrive (se k>0): 3,4,6,9,10,11,28,29,30,31,35,36,43,44. Ritorna k. */
static __attribute__((noinline)) int d647_hand_5BB8(u128* R, uint8_t* ls)
{
    if (!(D647_EQ(D647_LDR(12), D647_SPLAT(0x04050607)) && D647_EQ(D647_LDR(13), D647_SPLAT(0x00010203)) &&
          D647_EQ(D647_LDR(14), D647_SPLAT(0xFFFFFFFAu)) &&
          D647_EQ(D647_LDR(17), D647_SPLAT(0xFFFF0011u)) &&
          D647_EQ(D647_LDR(18), _mm_setr_epi32((int)0x04050607, (int)0x00010203, (int)0x0C0D0E0F, (int)0x08090A0B)) &&
          D647_EQ(D647_LDR(19), _mm_setr_epi32((int)0x80800001, (int)0x80800203, (int)0x80800405, (int)0x80800607)) &&
          D647_EQ(D647_LDR(20), D647_SPLAT(0x03030303)) && D647_EQ(D647_LDR(21), D647_SPLAT(0x0C0D0E0F))))
        return 0;
    const uint32_t r6 = R[6]._u32[0];
    if (r6 % 6u) return 0;
    const uint32_t n = r6 / 6u;
    if (n < 2u || n > D647_MAXN) return 0;
    uint32_t o4 = R[4]._u32[0], o3 = R[3]._u32[0];
    const uint32_t vb = R[5]._u32[0], tA = R[16]._u32[0], tB = R[15]._u32[0];
    /* coerenza della pipeline all'ingresso */
    {   const __m128i t10 = D647_LDR(10);
        if ((uint32_t)_mm_extract_epi32(D647_LDR(28), 0) != ((uint32_t)_mm_extract_epi32(t10, 0) << 4) ||
            R[44]._u32[0] != (R[10]._u32[1] << 4) || R[35]._u32[0] != (R[10]._u32[2] << 4)) return 0;
        if (!D647_EQ(D647_LDR(29), D647_SPLAT((o4 & 0xFFu) * 0x01010101u))) return 0;
        }
    /* tabelle di merge (8 quad dopo ogni base): solo byte 0..0x1F; e l'area scritta non le tocca (le scritture sono [o3 .. o3+6n+32)) */
    {   const uint32_t span = 6u * n + 48u;
        const uint32_t wlo = o3 & 0x3FFF0u, whi = wlo + span;
        if (whi > 0x40000u) return 0;
        const uint32_t bases[2] = { tA, tB };
        for (int t = 0; t < 2; t++) {
            const uint32_t b0 = bases[t] & 0x3FFF0u;
            const uint32_t ne = (((bases[t] & 15u) + 120u) >> 4) + 1u;               /* quad raggiungibili con (r3&15)*8 + base: 8 se base allineata */
            if (b0 + ne * 16u > 0x40000u) return 0;
            if (b0 < whi && wlo < b0 + ne * 16u) return 0;
            for (uint32_t e = 0; e < ne; e++) {
                const __m128i c = d647_ldq(ls, b0 + 16u * (uint32_t)e);
                if (_mm_movemask_epi8(_mm_cmpgt_epi8(c, _mm_set1_epi8(0x1F))) != 0 || _mm_movemask_epi8(c) != 0) return 0;   /* tra 0 e 0x1F (con segno) */
            }
        }
    }
    const unsigned csr_old = _mm_getcsr();
    _mm_setcsr((csr_old & ~0x6000u) | 0xE040u);

    const __m128i lim = _mm_set1_epi32(0x7F7FFFFE);
    __m128i r9 = d647_bs(D647_LDR(9)), r10 = d647_bs(D647_LDR(10)), r11 = d647_bs(D647_LDR(11));   /* in ordine SPU grezzo */
    uint32_t c43 = R[43]._u32[3], x36 = R[36]._u32[0], y36 = R[36]._u32[1];
    __m128i m36 = D647_LDR(36), and43 = _mm_setzero_si128();
    /* i due quad della lista indici (r31/r30 dell'originale), ordine SPU grezzo: letti per l'iterazione successiva PRIMA della scrittura
     * del quad compattato di questa iterazione (conta se la compattazione e' sul posto) */
    __m128i Q0 = d647_bs(D647_LDR(31)), Q1 = d647_bs(D647_LDR(30));
    const __m128i ctl19b = _mm_setr_epi8((char)0x80, (char)0x80, 0, 1, (char)0x80, (char)0x80, 2, 3, (char)0x80, (char)0x80, 4, 5, (char)0x80, (char)0x80, 6, 7);
    uint32_t keeps = 0, k = 0;
    for (; k + 1u < n; k++) {
        /* triangolo corrente (letto da r10: tre indici u16 come parole) */
        const __m128i T = d647_bs(r10);
        const uint32_t i0 = (uint32_t)_mm_extract_epi32(T, 0), i1 = (uint32_t)_mm_extract_epi32(T, 1), i2 = (uint32_t)_mm_extract_epi32(T, 2);
        const __m128i q0 = d647_ldq(ls, vb + (i0 << 4)), q1 = d647_ldq(ls, vb + (i1 << 4)), q2 = d647_ldq(ls, vb + (i2 << 4));
        const __m128 h0 = _mm_castsi128_ps(d647_bs(q0)), h1 = _mm_castsi128_ps(d647_bs(q1)), h2 = _mm_castsi128_ps(d647_bs(q2));
        const __m128 a = _mm_sub_ps(h1, h0), b = _mm_sub_ps(h2, h0);
        const __m128 m = _mm_mul_ps(a, _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(b), 0xB1)));   /* (a.x*b.y, a.y*b.x, ...) */
        const __m128i acc = _mm_max_epu32(_mm_max_epu32(d647_absm(a), d647_absm(b)), d647_absm(m));
        if (__builtin_expect((_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(acc, lim))) & 3) != 0, 0)) break;
        /* triangolo successivo: indici dai quad portati (r10 = shufb(r31, r30, r19 + splat(r4 & 15)) dell'originale) */
        const __m128i Tn = d647_shuf2(Q0, Q1, _mm_add_epi8(ctl19b, _mm_set1_epi8((char)(o4 & 15u))));
        o4 += 6u;
        const __m128i nQ0 = d647_ldq(ls, o4), nQ1 = d647_ldq(ls, o4 + 0x10u);          /* r31/r30 per l'iterazione dopo: PRIMA della scrittura */
        /* decisione dello stadio precedente */
        const int keep = (c43 == 0xFFFF0011u) && (d647_key(x36) > d647_key(y36));
        /* merge a quadword */
        const uint32_t N8 = (o3 & 15u) << 3;
        const __m128i cA = d647_ldq(ls, N8 + tA), cB = d647_ldq(ls, N8 + tB);
        const __m128i r33 = d647_shuf2(r9, r11, cA);
        const __m128i r45 = d647_shuf2(r9, r33, cB);
        d647_stq(ls, o3, r33);
        r11 = keep ? r45 : r33;
        o3 += keep ? 6u : 0u; keeps += keep ? 1u : 0u;
        r9 = r10; r10 = Tn; Q0 = nQ0; Q1 = nQ1;
        and43 = _mm_and_si128(_mm_and_si128(q0, q1), q2);
        c43 = (uint32_t)__builtin_bswap32((uint32_t)_mm_extract_epi32(and43, 3));
        m36 = d647_fixz(m);                                             /* -0 -> +0: r40/r42 dell'originale copiano i bit di r36 */
        x36 = (uint32_t)_mm_extract_epi32(m36, 0); y36 = (uint32_t)_mm_extract_epi32(m36, 1);
    }
    if (k) {
        const __m128i kk = _mm_set1_epi32((int)k);
        D647_STR(3, _mm_add_epi32(D647_LDR(3), _mm_set1_epi32((int)(6u * keeps))));
        D647_STR(4, _mm_add_epi32(D647_LDR(4), _mm_mullo_epi32(_mm_set1_epi32(6), kk)));
        D647_STR(6, _mm_sub_epi32(D647_LDR(6), _mm_mullo_epi32(_mm_set1_epi32(6), kk)));
        const __m128i T = d647_bs(r10), r28 = _mm_slli_epi32(T, 4);
        D647_STR(9, d647_bs(r9)); D647_STR(10, T); D647_STR(11, d647_bs(r11));
        D647_STR(28, r28);
        D647_STR(44, _mm_shuffle_epi32(r28, 0x39)); D647_STR(35, _mm_shuffle_epi32(r28, 0x4E));
        D647_STR(43, d647_bs(and43)); D647_STR(36, m36);                /* r36: corsie 2,3 non lette dall'originale */
        D647_STR(29, D647_SPLAT((o4 & 0xFFu) * 0x01010101u));
        D647_STR(31, d647_bs(Q0)); D647_STR(30, d647_bs(Q1));
    }
    _mm_setcsr(csr_old);
    return (int)k;
}

/* Hook da inserire nei due corpi _p, DOPO i caricamenti dai ctx->gpr e PRIMA dell'etichetta del ciclo (i registri sono variabili locali:
 * si copiano in un array, si chiama, si ricopiano i registri scritti). Uso (5908_p, riga ~6765):
 *     D647_HOOK_5908(ls, r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r26,r27,r28,r29,r30,r31,r32,r33,r35)
 * e per 5BB8_p (riga ~7143) D647_HOOK_5BB8(...). Qui sotto la forma testata (vedi hand_d647_test.c: stesso R[]). */
/* TOD_D647_HAND: 0 (default) spento, 1 acceso, 2 acceso + statistiche ogni 64k chiamate. */
static int s_d647 = -1; static volatile long s_d647_c[2], s_d647_k[2], s_d647_n[2], s_d647_z[2];
static __attribute__((noinline)) int d647_init(void) { const char* e = getenv("TOD_D647_HAND"); s_d647 = e ? atoi(e) : 0; return s_d647; }
static __attribute__((noinline)) void d647_note(int w, int k, uint32_t n) {
    const long c = __atomic_add_fetch(&s_d647_c[w], 1, __ATOMIC_RELAXED); __atomic_fetch_add(&s_d647_k[w], k, __ATOMIC_RELAXED); __atomic_fetch_add(&s_d647_n[w], (long)n, __ATOMIC_RELAXED); if (!k) __atomic_fetch_add(&s_d647_z[w], 1, __ATOMIC_RELAXED);
    if ((c & 0xFFFF) == 1) fprintf(stderr, "[d647] %s: chiamate %ld, iterazioni a mano %ld su %ld, chiamate senza mano %ld\n", w ? "5BB8" : "5908", c, s_d647_k[w], s_d647_n[w], s_d647_z[w]);
}
#endif
