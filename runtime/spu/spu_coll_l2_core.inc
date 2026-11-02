/* spu_coll_l2_core.inc (coll57) -- DRIVER NATIVO PER TICKET della collisione L2 (tipi 0, 1, 2).
 * Condiviso fra il gioco (runtime\spu\spu_coll_l2.c) e il banco offline (log\coll53\coll_test.c, modo `chain`).
 *
 * Esegue UN ticket (= una chiamata di func_00002640 di img 110) in modo nativo, SENZA toccare memoria guest:
 *   1. stadio 110: F_PRO (prologo 2640..28A8: GET record da 160 B, GET 128 B del mondo, CLAIM sulla linea 10612E80, attesa bit 0x20)
 *      + F_Q28A8 (28A8..2DC8: fasi 1/2, test volumi, anello, chiusura) su una LS PRIVATA di 256 KB (ls110) con una linea di claim PRIVATA;
 *   2. raccordo: come l'anello di img 111 (pc 11E70): per ogni voce fra +4 e +8 della linea: azzera i due buffer di 16480 B
 *      (12BC0/12BD4/12BE4 = memset), GETLLAR (copia della linea in LS 0x13A80), +4 += 16, word0 |= 0x20, PUTLLC;
 *   3. stadio 111: F_V122A0 con r3 = EA della voce (alias dell'LS di img 110 = ls110 privata), r4 = -1.
 * Ogni PUT va in un LOG (overlay: i GET successivi lo vedono); NIENTE e' scritto in memoria guest dal driver. Chi lo usa (gioco) applica il
 * log nell'ordine (modo 1) o lo confronta con la memoria dopo le raw SPU (modo doppio).
 * Ritorni di l2_run_ticket: 0 ok; 8/9 = tipo/kind/percorso NON coperto (nessun effetto: il chiamante ricade sulle raw SPU);
 * 10..12 = errori del generato; 100+k = errore del driver (k: 1 attesa senza fine, 2 troppi comandi, 3 canale, 4 EA/alias, 5 DMA anomalo,
 * 6 spin, 7 rdch, 8 linea bloccata, 9 log pieno, 10 LS111 mancante, 11 linea/costanti incoerenti, 12 voci != conteggio).
 * Richiede: spu_helpers.h (u128, spu_*), NC (spu_coll_l2_nc.h), F_PRO/F_Q28A8/F_V122A0, <setjmp.h>, <string.h>. */
#include <setjmp.h>
#define L2_LS_SIZE 0x40000u
#define L2_MAXW    512
#define L2_WDCAP   (96 * 1024)

typedef struct { uint32_t ea, n, off, lsa; } L2Wr;
typedef struct L2 {
    uint8_t *ls110, *ls111;          /* LS private (256 KB, allineate a 64) */
    int      ls111_ok;               /* ls111 inizializzata */
    uint32_t alias110;               /* EA con cui img 111 vede la LS di img 110 (E01xxxxx) */
    uint32_t line_ea;                /* EA della linea di claim (10612E80) */
    uint8_t  line[128];              /* linea di claim PRIVATA */
    L2Wr     w[L2_MAXW]; uint32_t nw;
    uint8_t  wd[L2_WDCAP]; uint32_t wdn;
    uint8_t* cur; uint32_t lsa, eal, size, tag, mask, stat, spin, ncmd; int err;
    jmp_buf  jb;
    uint32_t n_entries;              /* voci dell'anello consumate dallo stadio 111 */
    u128     regs_out[128];          /* registri di img 110 a fine ticket (= quelli che lascerebbe func_2640) */
    uint32_t rec_ea;
    int      k12;
    long long q_pro, q_111, q_sync; long long q_get; uint32_t n_get, b_get, n_ls_get;    /* q89: tempi (tick QPC) stadio 110 / stadio 111 */
} L2;
static __thread int l2_ign12;   /* ring103: per thread (il PPU lo imposta a 1 per il sync; i worker ad anello restano rigorosi) */
static int l2_qpc;    /* coll101: SPU_COLL_SYNC_QPC=1 -> tempo per GET (costa ~40 ns a GET) */

static void (*l2_mem_rd)(uint32_t ea, uint8_t* dst, uint32_t n);   /* lettura memoria principale (live o modello del banco) */

static inline uint32_t l2_be32(const uint8_t* p) { return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]; }
static inline void l2_wbe32(uint8_t* p, uint32_t v) { p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16); p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v; }

static void l2_fail(L2* l, int code) { l->err = code; longjmp(l->jb, 1); }

static void l2_get_mem(L2* l, uint8_t* dst, uint32_t ea, uint32_t n)
{
    const uint32_t a = ea - l->alias110;
    if (a < L2_LS_SIZE) {                                   /* alias della LS di img 110 */
        if (a + n > L2_LS_SIZE) l2_fail(l, 4);
        memcpy(dst, l->ls110 + a, n); l->n_ls_get++;
        return;
    }
    if (ea >= 0xE0000000u || ea < 0x10000u) l2_fail(l, 4);  /* altri alias SPU / pagina nulla */
    if (l2_qpc) { LARGE_INTEGER qa, qb; QueryPerformanceCounter(&qa); l2_mem_rd(ea, dst, n); QueryPerformanceCounter(&qb); l->q_get += qb.QuadPart - qa.QuadPart; l->n_get++; l->b_get += n; }
    else { l2_mem_rd(ea, dst, n); l->n_get++; l->b_get += n; }   /* coll101: niente QPC per GET se non SPU_COLL_SYNC_QPC=1 */
    for (uint32_t k = 0; k < l->nw; k++) {                  /* overlay delle PUT di questo ticket, in ordine */
        const L2Wr* w = &l->w[k];
        const uint32_t lo = ea > w->ea ? ea : w->ea, hi = (ea + n < w->ea + w->n) ? ea + n : w->ea + w->n;
        if (lo < hi) memcpy(dst + (lo - ea), l->wd + w->off + (lo - w->ea), hi - lo);
    }
}

static void l2_xfer(L2* l, uint32_t cmd, uint32_t lsa, uint32_t ea, uint32_t n)
{
    lsa &= 0x3FFFFu;
    if (n == 0) return;                                     /* DMA da 0 byte: nessun effetto (sync59) */
    if (n > 16384u || lsa + n > L2_LS_SIZE) { static int dbg; if (dbg++ < 8) fprintf(stderr, "[l2-dbg] xfer anomalo cmd=0x%X lsa=0x%X ea=0x%08X n=%u tag=%u\n", cmd, lsa, ea, n, l->tag); l2_fail(l, 5); }
    if (cmd & 0x40) { l2_get_mem(l, l->cur + lsa, ea, n); return; }
    if (l->nw >= L2_MAXW || l->wdn + n > L2_WDCAP) l2_fail(l, 9);
    if (ea < 0x10000u || ea >= 0xE0000000u || (ea - l->alias110) < L2_LS_SIZE) l2_fail(l, 4);
    L2Wr* w = &l->w[l->nw++]; w->ea = ea; w->n = n; w->off = l->wdn; w->lsa = lsa;
    memcpy(l->wd + l->wdn, l->cur + lsa, n); l->wdn += n;
}

static void l2_atomic_ea(L2* l, uint32_t ea)
{
    ea &= ~127u;
    if (!l->line_ea) l->line_ea = ea;
    else if (l->line_ea != ea) l2_fail(l, 11);              /* una seconda linea: non prevista */
}

static void l2_cmd(L2* l, uint32_t cmd)
{
    if (++l->ncmd > 60000u) l2_fail(l, 2);
    const uint32_t lsa = l->lsa & 0x3FFFFu;
    switch (cmd) {
    case 0xD0: l2_atomic_ea(l, l->eal); memcpy(l->cur + (lsa & ~127u), l->line, 128); l->stat = 4; return;                    /* GETLLAR */
    case 0xB4: l2_atomic_ea(l, l->eal); memcpy(l->line, l->cur + (lsa & ~127u), 128); l->stat = 0; return;                   /* PUTLLC riuscito */
    case 0xB0: case 0xB8: l2_atomic_ea(l, l->eal); memcpy(l->line, l->cur + (lsa & ~127u), 128); l->stat = 2; return;       /* PUTLLUC / PUTQLLUC */
    case 0xC0: case 0xC8: case 0xCC: return;                                                                                 /* barriere */
    }
    if (cmd & 4) {                                                                                                           /* lista */
        uint32_t el = l->eal & 0x3FFFFu, dest = lsa;
        for (uint32_t i = 0; i < l->size / 8; i++, el += 8) {
            const uint32_t sf = l2_be32(l->cur + (el & 0x3FFFFu)), eal = l2_be32(l->cur + ((el + 4) & 0x3FFFFu)), xs = sf & 0x7FFFu;
            if (xs) { l2_xfer(l, cmd & ~4u, (dest & ~15u) | (eal & 15u), eal, xs); dest = (dest & ~15u) + ((xs + 15u) & ~15u); }
            if (sf >> 31) l2_fail(l, 5);
        }
        return;
    }
    l2_xfer(l, cmd, lsa, l->eal, l->size);
}

static void l2_wrch(NC* n, uint32_t ch, u128 v)
{
    L2* l = (L2*)n->user; const uint32_t w = v._u32[0];
    switch (ch) {
    case MFC_LSA: l->lsa = w; break;       case MFC_EAH: break;
    case MFC_EAL: l->eal = w; break;       case MFC_Size: l->size = w; break;
    case MFC_TagID: l->tag = w; break;     case MFC_Cmd: l2_cmd(l, w & 0xFFu); break;
    case MFC_WrTagMask: l->mask = w; break; case MFC_WrTagUpdate: break;
    case SPU_WrEventMask: case SPU_WrEventAck: break;
    default: l2_fail(l, 3);
    }
}
static u128 l2_rdch(NC* n, uint32_t ch)
{
    L2* l = (L2*)n->user;
    switch (ch) {
    case MFC_RdTagStat: return spu_pref_u32(l->mask);
    case MFC_RdAtomicStat: return spu_pref_u32(l->stat);
    case SPU_RdEventStat: if (++l->spin > 20000u) l2_fail(l, 6); return spu_pref_u32(0x400u);
    default: l2_fail(l, 7); return spu_zero();
    }
}
static uint32_t l2_rchcnt(NC* n, uint32_t ch) { (void)n; if (ch == MFC_Cmd) return 16; if (ch == SPU_RdEventStat) return 1; return 1; }

/* costruisce/riallinea una LS privata: segmento ELF dell'immagine a 0x60, poi i quad NON nulli della LS viva del raw SPU (dati scritti
 * dall'init: header di griglia, tabelle di callback...). raw puo' essere NULL (banco). */
static void l2_sync_ls(uint8_t* priv, const uint8_t* bin, uint32_t binsz, const uint8_t* raw)
{
    memset(priv, 0, L2_LS_SIZE);
    memcpy(priv + 0x60, bin, binsz);
    if (!raw) return;
    for (uint32_t off = 0; off < L2_LS_SIZE; off += 16) {
        uint64_t a, b; memcpy(&a, raw + off, 8); memcpy(&b, raw + off + 8, 8);
        if (a | b) memcpy(priv + off, raw + off, 16);
    }
}

static void l2_zero_ls(L2* l, uint32_t dst, uint32_t n)
{
    dst &= 0x3FFFFu;
    if (dst + n > L2_LS_SIZE) l2_fail(l, 4);
    memset(l->ls111 + dst, 0, n);
}

/* un ticket. regs_in = i 128 registri di img 110 all'ingresso di func_2640 (r3 = EA del record, r4, r1 = sp, r0). */
static int l2_run_ticket(L2* l, const u128* regs_in)
{
    NC* N = (NC*)_aligned_malloc(sizeof(NC), 64);
    if (!N) return 100;
    int rc = 0;
    l->nw = 0; l->wdn = 0; l->err = 0; l->spin = 0; l->ncmd = 0; l->stat = 0; l->mask = 0xFFFFFFFFu; l->n_entries = 0; l->k12 = 0; l->q_get = 0; l->n_get = 0; l->b_get = 0; l->n_ls_get = 0;
    memset(l->line, 0, 128); l->line_ea = 0;
    l->alias110 = l2_be32(l->ls110 + 0x6DC4);
    l->rec_ea = regs_in[3]._u32[0];
    if (setjmp(l->jb)) { _aligned_free(N); return 100 + l->err; }
    /* ---- stadio 110 ---- */
    memset(N, 0, sizeof *N);
    memcpy(N->R, regs_in, 2048);
    N->ls = l->ls110; N->wrch = l2_wrch; N->rdch = l2_rdch; N->rchcnt = l2_rchcnt; N->user = l;
    l->cur = l->ls110;
    { LARGE_INTEGER qa; QueryPerformanceCounter(&qa); l->q_111 = 0; l->q_pro = qa.QuadPart; }
    rc = F_PRO(N);
    { LARGE_INTEGER qa; QueryPerformanceCounter(&qa); l->q_pro = qa.QuadPart - l->q_pro; l->q_sync = qa.QuadPart; }
    if (rc) { _aligned_free(N); return rc; }
    memcpy(l->regs_out, N->R, 2048);
    /* ---- raccordo + stadio 111 ---- */
    {
        const uint32_t cnt = l2_be32(l->ls110 + 0x23100);
        uint8_t* L = l->ls111;
        if (!l->ls111_ok) { _aligned_free(N); return 110; }
        if (cnt) {
            l->q_111 = 1;
            if (l2_be32(L + 0x13908) != 0x13A80u || (l->line_ea && l2_be32(L + 0x13900) != l->line_ea)) l2_fail(l, 11);
            l->cur = L;
            for (;;) {
                l2_zero_ls(l, l2_be32(L + 0x13910 + 12), 16480u);        /* 12BC0: memset(LS[0x13910].w3, 0, 16480) */
                l2_zero_ls(l, l2_be32(L + 0x13920), 16480u);             /* 12BDC: memset(LS[0x13920].w0, 0, 16480) */
                uint8_t* cp = L + 0x13A80;                                /* 11E70: GETLLAR della linea */
                memcpy(cp, l->line, 128);
                const uint32_t w0 = l2_be32(cp), p4 = l2_be32(cp + 4), p8 = l2_be32(cp + 8);
                if (w0 & 1u) l2_fail(l, 8);                               /* attesa di img 111: qui non ci sarebbe chi la scioglie */
                if (!(p8 > p4)) break;                                    /* nessuna voce: img 111 aspetterebbe */
                l2_wbe32(cp + 4, p4 + 16); l2_wbe32(cp, w0 | 0x20u);      /* 11F10..11F48 */
                memcpy(l->line, cp, 128);                                 /* PUTLLC */
                memset(N, 0, sizeof *N);
                N->R[0] = spu_link(0x12BF0u); N->R[1] = spu_splat_u32(0x3FFB0u);
                N->R[3] = spu_splat_u32(p4); N->R[4] = spu_splat_u32(0xFFFFFFFFu);
                N->ls = L; N->wrch = l2_wrch; N->rdch = l2_rdch; N->rchcnt = l2_rchcnt; N->user = l;
                l->spin = 0;
                rc = F_V122A0(N);
                if (rc) { _aligned_free(N); return rc; }
                l->n_entries++;
                if (l->n_entries > 256) l2_fail(l, 12);
            }
            if (l->n_entries != cnt) { static int dbg12; if (dbg12++ < 10) fprintf(stderr, "[l2-dbg] k12: cnt(LS 0x23100)=%u voci consumate=%u line w0=%08X p4=%08X p8=%08X rec=%08X ty=%u\n", cnt, l->n_entries, l2_be32(l->line), l2_be32(l->line + 4), l2_be32(l->line + 8), l->rec_ea, (unsigned)(l->ls110[0x23000 + 0xD])); if (l2_ign12) l->k12 = 1; else l2_fail(l, 12); }
        }
    }
    if (l->q_111) { LARGE_INTEGER qa; QueryPerformanceCounter(&qa); l->q_111 = qa.QuadPart - l->q_sync; }
    _aligned_free(N);
    return 0;
}
