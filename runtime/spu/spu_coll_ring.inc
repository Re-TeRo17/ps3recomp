/* spu_coll_ring.inc (ring103) -- POOL di worker nativi per i ticket ASINCRONI AD ANELLO (record 0x10400000-0x105FFFFF, tipi 0..2), incluso in
 * fondo a spu_coll_sync.c (stesso motore L2). Nato da coll69 (spu_coll_pool.inc, che si BLOCCAVA a frame ~545).
 *
 * CAUSA DEL BLOCCO DI coll69 (letta nel lifted): il pop dell'anello (img 110 func_888, rami 8D0/9E0) fa `flags(+0x30 della linea 10612E80) |= 0x44`
 * (0x40 = occupato) PRIMA di chiamare func_2640; la chiusura del ticket (func_2640 fra 2A70 e 2B48) fa `flags &= ~0x44` (o ~4) con GETLLAR/PUTLLC
 * sulla linea VERA. Saltando func_2640 il nativo non rilasciava mai il bit: il kick del PPU (func_004A3288: attende (flags & 0x40) == 0 prima di
 * `|= 2`) restava in spin per sempre mentre entrambe le raw SPU dormivano sull'evento 0x400 (esattamente lo stato visto a frame ~545).
 * FIX: i worker rilasciano `flags &= ~0x44` con CAS atomico + notifica di coerenza quando l'ULTIMO ticket in volo e' stato committato.
 *
 *   SPU_COLL_RING=0 (default, spento) / 1 (pool nativo)   SPU_COLL_RING_N=<1..8> (worker, default 2)
 *   SPU_COLL_RING_SPIN_US=<n> (spin del worker prima di dormire, default 60)
 * Il raw 110 consegna (registri + snapshot LS viva di 256 KB) e ritorna; il worker esegue il driver 110+111, applica le PUT con commit IN ORDINE.
 * I ticket non-ad-anello (slot diretto, tipi >2, record illeggibili) fanno prima ring_drain() e girano lifted come sempre. Rifiuti/errori del driver:
 * coda di ricaduta; al gancio successivo (ticket ad anello valido) si consegna QUELLO al pool e si fa girare lifted il vecchio scambiando r3/r4. */
extern void spu_coh_notify_write_except(uint32_t ea, const void* self);
uint8_t g_spu_ring_on;
#define RING_SLOTS 64
typedef struct RJob { u128 regs[128]; spu_context* spu; uint8_t* snap; uint32_t rec; long long t_sub; long seq; } RJob;
static int s_rn, s_rspin_us = 60;
static RJob* s_rj[RING_SLOTS]; static int s_rfree[RING_SLOTS]; static int s_rnfree;
static int s_rq[RING_SLOTS]; static volatile long s_rqh, s_rqt;
static CRITICAL_SECTION s_rcs; static HANDLE s_rsem; static volatile long s_rwaiters;
static CRITICAL_SECTION s_rfcs; static struct { u128 r3, r4; } s_rfail[256]; static volatile long s_rfail_n;
static volatile long c_rcalls, c_rsub, c_rdone, c_rfail, c_rfailrun, c_rback, c_rrel, c_rnonring, c_rswap, c_rflag_clr;
static double s_rns_run, s_rns_wait, s_rns_sub, s_rns_s111, s_rns_l2, s_rns_cw, s_rns_snap; static volatile long c_s111;   /* rissa127: fasi */
static double s_rq_pro, s_rq_111, s_rn_get, s_rn_spin, s_rn_ent; static volatile long s_rmaxq;
static long long s_rqfreq; static volatile long s_rcommit; static long s_rseq;
static volatile long s_rinfl; static volatile uint32_t s_rline_ea;
static volatile long s_rerrk[16];
static volatile long c_r4hist[8][2];
static int s_rmode;
static volatile long c_ceq, c_cdiff, c_cerr, c_cref, s_cdlog, c_cskip, c_clost, c_cerrk[16];

static inline long long rq_now(void) { LARGE_INTEGER t; QueryPerformanceCounter(&t); return t.QuadPart; }
static void ring_drain(void) { const long tgt = s_rseq; while (__atomic_load_n(&s_rcommit, __ATOMIC_ACQUIRE) < tgt) _mm_pause(); }
static inline int ring_is_ring(uint32_t rec) { return rec >= 0x10400000u && rec < 0x10600000u; }

/* PUT in memoria guest da un worker (stesse regole di mfc_do_transfer_fast), notifica di coerenza a tutte le SPU salvo il thread 110 */
static void ring_put_guest(uint32_t ea, const uint8_t* src, uint32_t n, spu_context* spu)
{
    uint8_t* p = vm_base + ea;
    const uint32_t a0 = ea & ~127u, a1 = (ea + n - 1u) & ~127u;
    int reserved = 0;
    for (uint32_t a = a0; ; a += 128u) { if (sync_coh_reserved(a)) { reserved = 1; break; } if (a == a1) break; }
    __atomic_thread_fence(__ATOMIC_RELEASE);
    if (reserved) spu_lockline_lock();
    if (n == 16u) _mm_storeu_si128((__m128i*)p, _mm_loadu_si128((const __m128i*)src));
    else memcpy(p, src, n);
    for (uint32_t a = a0; ; a += 128u) {
        if (sync_coh_reserved(a)) {
            if (!reserved) { spu_lockline_lock(); spu_coh_notify_write_except(a, spu); spu_lockline_unlock(); }
            else spu_coh_notify_write_except(a, spu);
        }
        if (a == a1) break;
    }
    if (reserved) spu_lockline_unlock();
    if (g_spu_put_hook) g_spu_put_hook(ea, n);
}

/* flags(+0x30 della linea di claim) &= ~mask, CAS atomico sulla parola guest (big endian) + notifica di coerenza (le raw SPU in attesa sulla linea) */
static void ring_clear_flags(uint32_t line_ea, uint32_t mask)
{
    const uint32_t ea = line_ea + 0x30u;
    uint32_t* p = (uint32_t*)(vm_base + ea);
    const uint32_t a = ea & ~127u;
    const int reserved = sync_coh_reserved(a);
    if (reserved) spu_lockline_lock();
    uint32_t cur = __atomic_load_n(p, __ATOMIC_RELAXED);
    for (;;) {
        const uint32_t host = __builtin_bswap32(cur), nh = host & ~mask;
        if (nh == host) break;
        if (__atomic_compare_exchange_n(p, &cur, __builtin_bswap32(nh), 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) { __atomic_fetch_add(&c_rflag_clr, 1, __ATOMIC_RELAXED); break; }
    }
    if (reserved) { spu_coh_notify_write(a); spu_lockline_unlock(); }
    __atomic_fetch_add(&c_rrel, 1, __ATOMIC_RELAXED);
}

static void ring_sync_111(L2* l, uint32_t alias110)
{
    static __thread const uint8_t* c_raw111;
    const uint8_t* raw;
    if (c_raw111 && l2_be32(c_raw111 + 0x13908) == 0x13A80u && l2_be32(c_raw111 + 0x13900) - 0x10000000u < 0x10000000u) raw = c_raw111;
    else { raw = sync_find_ls111(alias110); c_raw111 = raw; }
    if (!raw) { l->ls111_ok = 0; return; }
    if (l->ls111_ok && memcmp(l->ls111 + 0x13930, raw + 0x13930, 96) == 0) return;
    l2_sync_ls(l->ls111, k_l2_bin111, sizeof k_l2_bin111, raw);
    l->ls111_ok = 1;
}

static void ring_print_stats(const char* why)
{
    fprintf(stderr, "[coll-ring] stats (%s) N=%d: chiamate %ld, non-anello %ld, sottomessi %ld, fatti %ld, ricaduta %ld (eseguiti %ld, scambi %ld), backpressure %ld, coda max %ld, in volo %ld; "
            "run medio %.1f us, attesa in coda media %.1f us, consegna (raw) %.1f us; rilasci flags %ld (CAS riusciti %ld, linea 0x%08X)\n",
            why, s_rn, c_rcalls, c_rnonring, c_rsub, c_rdone, c_rfail, c_rfailrun, c_rswap, c_rback, s_rmaxq, s_rinfl,
            c_rdone ? s_rns_run / c_rdone / 1000.0 : 0.0, c_rdone ? s_rns_wait / c_rdone / 1000.0 : 0.0, c_rsub ? s_rns_sub / c_rsub / 1000.0 : 0.0, c_rrel, c_rflag_clr, s_rline_ea);
    if (c_rdone) fprintf(stderr, "[coll-ring]   fasi worker (us/ticket): sync111 %.1f | motore L2 %.1f | attesa turno commit %.1f | snapshot LS sul raw %.1f | stadio110 %.1f stadio111 %.1f | GET %.1f spin %.1f voci %.2f\n",
                         s_rns_s111 / c_rdone / 1000.0, s_rns_l2 / c_rdone / 1000.0, s_rns_cw / c_rdone / 1000.0, c_rsub ? s_rns_snap / c_rsub / 1000.0 : 0.0, s_rq_pro / c_rdone / 1000.0, s_rq_111 / c_rdone / 1000.0, s_rn_get / c_rdone, s_rn_spin / c_rdone, s_rn_ent / c_rdone);
    if (s_rmode == 2) { fprintf(stderr, "[coll-ring]   doppio: confronti uguali %ld diversi %ld, errori %ld, rifiutati %ld, saltati %ld, persi %ld; errori per codice:", c_ceq, c_cdiff, c_cerr, c_cref, c_cskip, c_clost);
        for (int k = 0; k < 16; k++) if (c_cerrk[k]) fprintf(stderr, " k%d=%ld", k, c_cerrk[k]);
        fprintf(stderr, "\n"); }
    fprintf(stderr, "[coll-ring]   r4 all'ingresso (non-anello/anello):"); for (int i = 0; i < 8; i++) if (c_r4hist[i][0] || c_r4hist[i][1]) fprintf(stderr, " r4=%d:%ld/%ld", i, c_r4hist[i][0], c_r4hist[i][1]); fprintf(stderr, "\n");
    if (c_rfail) { fprintf(stderr, "[coll-ring]   rifiuti per codice:"); for (int i = 0; i < 16; i++) if (s_rerrk[i]) fprintf(stderr, " k%d=%ld", i, s_rerrk[i]); fprintf(stderr, "\n"); }
    fflush(stderr);
}
static void ring_inline_line(void);
static void ring_atexit(void) { if (s_rmode == 3) ring_inline_line(); else ring_print_stats("exit"); }

static void ring_run_job(L2* l, RJob* j)
{
    const long long t0 = rq_now();
    l->ls110 = j->snap;
    { const int was = l->ls111_ok; const uint8_t* before = NULL; (void)before;
      ring_sync_111(l, l2_be32(l->ls110 + 0x6DC4)); (void)was; }
    const long long ts1 = rq_now();
    int rc = l->ls111_ok ? 0 : 1;
    if (!rc) { l2_mem_rd = sync_rd_live; t_s_cur = l; rc = l2_run_ticket(l, j->regs); }
    if (rc == 0 && l->nw == 0) rc = 2;
    const long long ts2 = rq_now();
    s_rq_pro += (double)l->q_pro * 1e9 / (double)s_rqfreq; if (l->q_111) s_rq_111 += (double)l->q_111 * 1e9 / (double)s_rqfreq; s_rn_get += l->n_get + l->n_ls_get; s_rn_spin += l->spin; s_rn_ent += l->n_entries;
    /* commit IN ORDINE di pubblicazione: le PUT di un ticket si applicano solo dopo quelle dei precedenti */
    while (__atomic_load_n(&s_rcommit, __ATOMIC_ACQUIRE) != j->seq) _mm_pause();
    const long long ts3 = rq_now();
    s_rns_s111 += (double)(ts1 - t0) * 1e9 / (double)s_rqfreq; s_rns_l2 += (double)(ts2 - ts1) * 1e9 / (double)s_rqfreq; s_rns_cw += (double)(ts3 - ts2) * 1e9 / (double)s_rqfreq;
    if (rc != 0) {
        __atomic_fetch_add(&s_rerrk[(rc >= 100 ? rc - 100 : rc) & 15], 1, __ATOMIC_RELAXED);
        EnterCriticalSection(&s_rfcs);
        long n = s_rfail_n; if (n < 256) { s_rfail[n].r3 = j->regs[3]; s_rfail[n].r4 = j->regs[4]; __atomic_store_n(&s_rfail_n, n + 1, __ATOMIC_RELEASE); }
        LeaveCriticalSection(&s_rfcs);
        __atomic_fetch_add(&c_rfail, 1, __ATOMIC_RELAXED);
    } else {
        for (uint32_t k = 0; k < l->nw; k++) ring_put_guest(l->w[k].ea, l->wd + l->w[k].off, l->w[k].n, j->spu);
        if (l->line_ea) s_rline_ea = l->line_ea;
    }
    /* chiusura del ticket come func_2640: rilascia flags &= ~0x44 quando non c'e' piu' nulla in volo (prima del commit: ring_drain() poi vede il bit pulito) */
    if (__atomic_sub_fetch(&s_rinfl, 1, __ATOMIC_ACQ_REL) == 0 && s_rline_ea) ring_clear_flags(s_rline_ea, 0x44u);
    __atomic_store_n(&s_rcommit, j->seq + 1, __ATOMIC_RELEASE);
    const long long t1 = rq_now();
    s_rns_run += (double)(t1 - t0) * 1e9 / (double)s_rqfreq; s_rns_wait += (double)(t0 - j->t_sub) * 1e9 / (double)s_rqfreq;
    long d = __atomic_add_fetch(&c_rdone, 1, __ATOMIC_RELAXED);
    if ((d & 0x7FFF) == 0) ring_print_stats("periodic");
}

static DWORD WINAPI ring_worker(LPVOID arg)
{
    (void)arg;
    _mm_setcsr(0xFFC0u); g_spu_rz = 1;                       /* stesso MXCSR dei raw 110/111 */
    { const char* e = getenv("SPU_COLL_RING_IGN12"); l2_ign12 = e && e[0] == '1'; }   /* rissa127: k12 tollerato anche nei worker (verificato in modo 2) */
    L2* l = (L2*)_aligned_malloc(sizeof(L2), 64); memset(l, 0, sizeof *l);
    l->ls110 = NULL; l->ls111 = (uint8_t*)_aligned_malloc(L2_LS_SIZE, 64);
    memset(l->ls111, 0, L2_LS_SIZE);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    for (;;) {
        int idx = -1;
        EnterCriticalSection(&s_rcs);
        if (s_rqh != s_rqt) { idx = s_rq[s_rqt % RING_SLOTS]; s_rqt++; }
        LeaveCriticalSection(&s_rcs);
        if (idx < 0) {
            const long long lim = rq_now() + (long long)s_rspin_us * s_rqfreq / 1000000;
            while (s_rqh == s_rqt && rq_now() < lim) _mm_pause();
            if (s_rqh != s_rqt) continue;
            InterlockedIncrement(&s_rwaiters);
            if (s_rqh != s_rqt) {                            /* ritira la registrazione */
                long w = s_rwaiters; int ok = 0;
                while (w > 0) { if (InterlockedCompareExchange(&s_rwaiters, w - 1, w) == w) { ok = 1; break; } w = s_rwaiters; }
                if (!ok) WaitForSingleObject(s_rsem, INFINITE);
                continue;
            }
            WaitForSingleObject(s_rsem, INFINITE);
            continue;
        }
        RJob* j = s_rj[idx];
        ring_run_job(l, j);
        EnterCriticalSection(&s_rcs); s_rfree[s_rnfree++] = idx; LeaveCriticalSection(&s_rcs);
    }
    return 0;
}

static void ring_submit(spu_context* ctx, uint32_t rec)
{
    const long long ta = rq_now();
    int idx = -1; int spins = 0;
    for (;;) {
        EnterCriticalSection(&s_rcs); if (s_rnfree) idx = s_rfree[--s_rnfree]; LeaveCriticalSection(&s_rcs);
        if (idx >= 0) break;
        if (!spins++) __atomic_fetch_add(&c_rback, 1, __ATOMIC_RELAXED);
        _mm_pause();
    }
    RJob* j = s_rj[idx];
    memcpy(j->regs, ctx->gpr, 2048); j->spu = ctx; j->rec = rec;
    { const long long tsn = rq_now(); memcpy(j->snap, ctx->ls, L2_LS_SIZE); s_rns_snap += (double)(rq_now() - tsn) * 1e9 / (double)s_rqfreq; }
    j->t_sub = rq_now(); j->seq = s_rseq++;
    __atomic_fetch_add(&s_rinfl, 1, __ATOMIC_ACQ_REL);
    EnterCriticalSection(&s_rcs);
    s_rq[s_rqh % RING_SLOTS] = idx; s_rqh++;
    { long q = s_rqh - s_rqt; if (q > s_rmaxq) s_rmaxq = q; }
    LeaveCriticalSection(&s_rcs);
    __atomic_fetch_add(&c_rsub, 1, __ATOMIC_RELAXED);
    long w = s_rwaiters;
    while (w > 0) { if (InterlockedCompareExchange(&s_rwaiters, w - 1, w) == w) { ReleaseSemaphore(s_rsem, 1, NULL); break; } w = s_rwaiters; }
    s_rns_sub += (double)(rq_now() - ta) * 1e9 / (double)s_rqfreq;
}

/* ---- modo DOPPIO (SPU_COLL_RING=2), rissa127 v2: su ogni ticket ad anello valido il driver nativo gira PRIMA, sul raw 110, su una LS privata
 * sincronizzata dalla LS viva, e produce un LOG di PUT che NON si applica; poi gira il lifted vero. Il confronto si fa nell'istante giusto: al PUT del
 * lifted che scrive il flag di fine (u16 rec+0x18), PRIMA che arrivi in memoria (g_spu_ring_preput): i quad coperti da quel PUT si confrontano con i
 * dati in arrivo, gli altri con la memoria. (ring103 confrontava al ticket successivo: il PPU aveva gia' letto e riusato i risultati.)
 * Contatori: uguali/diversi, errori del nativo per codice (k15 = LS 111 non trovata, k14 = nessuna PUT), saltati (log grande), persi (flag mai visto). ---- */
static __thread L2* t_r2;
typedef struct { uint32_t rec, nw, used, key; L2Wr w[32]; uint8_t d[2048]; u128 regs[128]; uint8_t recb[160]; uint8_t* snap; uint8_t hdr_seen, hdr_b[16]; } RPend;   /* key = EA del primo byte dell'ULTIMA PUT del log nativo */
#define RPN 32
static RPend s_rp[RPN]; static SRWLOCK s_rplock = SRWLOCK_INIT; static volatile long s_rpn;
/* rissa128: su ogni DIFF il nativo si riesegue nell'istante del lifted (stessi registri, stessa LS 110 e stesso record di allora, mondo di adesso).
 * rerun == lifted -> la differenza e' il TEMPO (il mondo e' cambiato fra le due esecuzioni), rerun == nativo -> il nativo calcola diverso. */
static volatile long c_rr_lift, c_rr_nat, c_rr_other, c_rr_err, s_rrlog;
/* rissa128 esperimento: il flag di fine (+0x18) lo scrive il 111, l'intestazione (rec+0, PUT 16 B) il 110. Ipotesi: nei DIFF l'intestazione del lifted NON e' ancora
 * arrivata al momento del flag. Si conta se era arrivata; per i DIFF si aspetta il PUT tardivo e lo si confronta col nativo. Confermata se: DIFF sempre "non vista",
 * tardive tutte uguali al nativo, mai "mai arrivata"; ticket uguali sempre "vista". */
static volatile long c_ds_eq, c_ds_ne, c_d_seen, c_d_notseen, c_e_seen, c_e_notseen, c_late_eq, c_late_ne, c_late_never, s_nlate;
static int s_ring_hdr;   /* SPU_COLL_RING_HDR=1: the header experiment above (answered: CHIUSO, RIPARTENZA 06/11); off, mode 2 only compares */
static struct { uint32_t rec, used; uint8_t nat[16]; long long t; } s_late[64];
static __thread uint8_t* t_r2snap;
static __thread L2* t_r3;
static int ring_rerun_quad(RPend* q, uint32_t qea, uint8_t out[16])
{
    if (!q->snap) return -1;
    if (!t_r3) {
        L2* l = (L2*)_aligned_malloc(sizeof(L2), 64); memset(l, 0, sizeof *l);
        l->ls110 = (uint8_t*)_aligned_malloc(L2_LS_SIZE, 64); l->ls111 = (uint8_t*)_aligned_malloc(L2_LS_SIZE, 64);
        memset(l->ls111, 0, L2_LS_SIZE); t_r3 = l;
    }
    L2* l = t_r3;
    memcpy(l->ls110, q->snap, L2_LS_SIZE);
    ring_sync_111(l, l2_be32(l->ls110 + 0x6DC4));
    if (!l->ls111_ok) return -1;
    const int ig = l2_ign12; l2_ign12 = 1;
    t_rr_ov = q->recb; t_rr_ea = q->rec; t_rr_n = 160u;
    l2_mem_rd = sync_rd_live; t_s_cur = l;
    const int rc = l2_run_ticket(l, q->regs);
    t_rr_ov = NULL; l2_ign12 = ig;
    if (rc != 0) return -1;
    int found = 0;
    for (uint32_t i = 0; i < 16; i++) {
        const uint32_t e = qea + i;
        for (int k = (int)l->nw - 1; k >= 0; k--) { const L2Wr* w = &l->w[k]; if (e >= w->ea && e < w->ea + w->n) { out[i] = l->wd[w->off + (e - w->ea)]; found++; break; } }
    }
    return found == 16 ? 0 : -1;
}
static void ring_hexq(char* o, const uint8_t* p, uint32_t n) { for (uint32_t i = 0; i < n; i++) sprintf(o + 2 * i, "%02X", p[i]); }
static inline uint8_t ring_byte(uint32_t e, uint32_t pea, const uint8_t* psrc, uint32_t pn) { return (e >= pea && e < pea + pn) ? psrc[e - pea] : vm_base[e]; }
static void ring_cmp_pend(RPend* q, uint32_t pea, const uint8_t* psrc, uint32_t pn)
{
    uint32_t cov[256]; uint32_t ncov = 0, nbad = 0, bad_ea = 0; int first_bad = -1;
    for (int k = (int)q->nw - 1; k >= 0; k--) {
        const L2Wr* w = &q->w[k];
        for (uint32_t o = 0; o < w->n; o += 16u) {
            const uint32_t qea = (w->ea + o) & ~15u, len = (w->n - o) < 16u ? (w->n - o) : 16u;
            int seen = 0; for (uint32_t c = 0; c < ncov; c++) if (cov[c] == qea) { seen = 1; break; }
            if (seen) continue;
            if (ncov < 256) cov[ncov++] = qea;
            uint8_t a[16], b[16];
            for (uint32_t i = 0; i < len; i++) a[i] = ring_byte(w->ea + o + i, pea, psrc, pn);
            memcpy(b, q->d + w->off + o, len);
            if (len == 16u && w->ea + o == q->rec + 0x10u) { a[8] &= 0x7F; b[8] &= 0x7F; }
            if (memcmp(a, b, len) != 0) { nbad++; if (first_bad < 0) { first_bad = k; bad_ea = w->ea + o; } }
        }
    }
    if (nbad == 0) { __atomic_fetch_add(q->hdr_seen ? &c_e_seen : &c_e_notseen, 1, __ATOMIC_RELAXED); __atomic_fetch_add(&c_ceq, 1, __ATOMIC_RELAXED); return; }
    __atomic_fetch_add(&c_cdiff, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(q->hdr_seen ? &c_d_seen : &c_d_notseen, 1, __ATOMIC_RELAXED);
    uint8_t a[16], r[16] = {0}; const char* verd;
    for (uint32_t i = 0; i < 16; i++) a[i] = ring_byte(bad_ea + i, pea, psrc, pn);
    const L2Wr* wb = &q->w[first_bad];
    uint8_t nat[16]; memset(nat, 0, 16);
    { const uint32_t avail = wb->ea + wb->n - bad_ea; memcpy(nat, q->d + wb->off + (bad_ea - wb->ea), avail < 16u ? avail : 16u); }
    /* intestazione "vista" alla sonda ma memoria ancora vecchia: la sonda scatta PRIMA che la PUT atterri, e s_rplock serializza 110 e 111. Contano i byte in arrivo. */
    if (bad_ea == q->rec && q->hdr_seen) { if (memcmp(q->hdr_b, nat, 16) == 0) c_ds_eq++; else { c_ds_ne++; char h[40], g[40]; ring_hexq(h, q->hdr_b, 16u); ring_hexq(g, nat, 16u); if (c_ds_ne <= 20) fprintf(stderr, "[coll-ring] VISTA DIVERSA rec=0x%08X lifted=%s nativo=%s\n", q->rec, h, g); } }
    if (s_ring_hdr && bad_ea == q->rec && !q->hdr_seen) {           /* aspetta il PUT tardivo dell'intestazione */
        for (int i = 0; i < 64; i++) if (!s_late[i].used) { s_late[i].used = 1; s_late[i].rec = q->rec; memcpy(s_late[i].nat, nat, 16); s_late[i].t = rq_now(); __atomic_fetch_add(&s_nlate, 1, __ATOMIC_RELEASE); break; }
    }
    const int rr = ((bad_ea & 15u) == 0 && bad_ea + 16u <= wb->ea + wb->n) ? ring_rerun_quad(q, bad_ea, r) : -1;
    if (bad_ea == q->rec + 0x10u) { a[8] &= 0x7F; nat[8] &= 0x7F; r[8] &= 0x7F; }    /* come il confronto sopra */
    if (rr != 0) { __atomic_fetch_add(&c_rr_err, 1, __ATOMIC_RELAXED); verd = "rerun fallito"; }
    else if (memcmp(r, a, 16) == 0) { __atomic_fetch_add(&c_rr_lift, 1, __ATOMIC_RELAXED); verd = "rerun = LIFTED (tempo)"; }
    else if (memcmp(r, nat, 16) == 0) { __atomic_fetch_add(&c_rr_nat, 1, __ATOMIC_RELAXED); verd = "rerun = NATIVO (calcolo diverso)"; }
    else { __atomic_fetch_add(&c_rr_other, 1, __ATOMIC_RELAXED); verd = "rerun = altro"; }
    const long nl = __atomic_add_fetch(&s_cdlog, 1, __ATOMIC_RELAXED);
    if (nl <= 40) {
        char x[40], y[40], z[40] = "-";
        ring_hexq(x, nat, 16u); ring_hexq(y, a, 16u); if (rr == 0) ring_hexq(z, r, 16u);
        const uint32_t al = q->snap ? l2_be32(q->snap + 0x6DC4) : 0;
        const uint32_t live_cnt = (al >= 0xE0000000u) ? l2_be32(vm_base + al + 0x23100) : 0xFFFFFFFFu;
        const uint32_t snap_cnt = q->snap ? l2_be32(q->snap + 0x23100) : 0xFFFFFFFFu;
        const uint32_t rr_cnt = (rr == 0 && t_r3) ? l2_be32(t_r3->ls110 + 0x23100) : 0xFFFFFFFFu;
        fprintf(stderr, "[coll-ring] DIFF #%ld rec=0x%08X puts=%u quad diversi=%u primo_ea=0x%08X -> %s | LS110 0x23100: ingresso %u, viva ora %u, rerun %u; voci rerun %u k12 %d\n[coll-ring]   nativo=%s\n[coll-ring]   lifted=%s\n[coll-ring]   rerun =%s\n",
                nl, q->rec, q->nw, nbad, bad_ea, verd, snap_cnt, live_cnt, rr_cnt, t_r3 ? t_r3->n_entries : 0u, t_r3 ? t_r3->k12 : -1, x, y, z);
    }
}
static void ring_dual_line(void)
{
    fprintf(stderr, "[coll-ring] doppio: uguali %ld diversi %ld errori %ld rifiutati %ld saltati %ld persi %ld | errori per codice:", c_ceq, c_cdiff, c_cerr, c_cref, c_cskip, c_clost);
    for (int k = 0; k < 16; k++) if (c_cerrk[k]) fprintf(stderr, " k%d=%ld", k, c_cerrk[k]);
    fprintf(stderr, " | rerun: =lifted %ld =nativo %ld altro %ld falliti %ld\n", c_rr_lift, c_rr_nat, c_rr_other, c_rr_err);
    fprintf(stderr, "[coll-ring] intestazione al flag: DIFF vista %ld non vista %ld | uguali vista %ld non vista %ld | tardive uguali al nativo %ld diverse %ld mai arrivate %ld | viste con byte uguali al nativo %ld diversi %ld\n", c_d_seen, c_d_notseen, c_e_seen, c_e_notseen, c_late_eq, c_late_ne, c_late_never, c_ds_eq, c_ds_ne);
}
static void ring_preput(uint32_t ea, const uint8_t* src, uint32_t n)
{
    if (!__atomic_load_n(&s_rpn, __ATOMIC_ACQUIRE) && !__atomic_load_n(&s_nlate, __ATOMIC_ACQUIRE)) return;
    AcquireSRWLockExclusive(&s_rplock);
    if (__atomic_load_n(&s_nlate, __ATOMIC_ACQUIRE)) {
        const long long now = rq_now();
        for (int i = 0; i < 64; i++) {
            if (!s_late[i].used) continue;
            if (n == 16u && ea == s_late[i].rec) {
                const int same = memcmp(src, s_late[i].nat, 16) == 0;
                if (same) c_late_eq++; else c_late_ne++;
                if (c_late_eq + c_late_ne <= 40) fprintf(stderr, "[coll-ring] LATE rec=0x%08X %.1f us dopo il flag, uguale al nativo: %d\n", ea, (double)(now - s_late[i].t) * 1e6 / (double)s_rqfreq, same);
                s_late[i].used = 0; __atomic_fetch_sub(&s_nlate, 1, __ATOMIC_RELEASE);
            } else if (now - s_late[i].t > s_rqfreq / 10) { c_late_never++; s_late[i].used = 0; __atomic_fetch_sub(&s_nlate, 1, __ATOMIC_RELEASE); }     /* 100 ms */
        }
    }
    for (int i = 0; i < RPN; i++) {
        RPend* q = &s_rp[i];
        if (s_ring_hdr && q->used && ea == q->rec && n == 16u) { q->hdr_seen = 1; memcpy(q->hdr_b, src, 16); }
        if (!q->used || q->key < ea || q->key >= ea + n) continue;
        ring_cmp_pend(q, ea, src, n);
        q->used = 0; __atomic_fetch_sub(&s_rpn, 1, __ATOMIC_RELEASE);
        if (((c_ceq + c_cdiff) & 0x3FFF) == 0) ring_dual_line();
    }
    ReleaseSRWLockExclusive(&s_rplock);
}
static void ring_dual(spu_context* ctx, uint32_t rec, int eligible)
{
    if (!eligible) return;
    if (!t_r2) {
        L2* l = (L2*)_aligned_malloc(sizeof(L2), 64); memset(l, 0, sizeof *l);
        l->ls110 = (uint8_t*)_aligned_malloc(L2_LS_SIZE, 64); l->ls111 = (uint8_t*)_aligned_malloc(L2_LS_SIZE, 64);
        memset(l->ls110, 0, L2_LS_SIZE); memset(l->ls111, 0, L2_LS_SIZE);
        t_r2 = l;
    }
    L2* l = t_r2;
    sync_full_fast(l->ls110, ctx->ls);
    ring_sync_111(l, l2_be32(l->ls110 + 0x6DC4));
    if (!l->ls111_ok) { __atomic_fetch_add(&c_cerr, 1, __ATOMIC_RELAXED); __atomic_fetch_add(&c_cerrk[15], 1, __ATOMIC_RELAXED); return; }
    { static int ig = -1; if (ig < 0) { const char* e = getenv("SPU_COLL_RING_IGN12"); ig = e && e[0] == '1'; } l2_ign12 = ig; }   /* rissa127 diagnosi: k12 tollerato */
    /* rissa128: stato d'ingresso salvato per la riesecuzione su DIFF (LS 110 prima del motore, record di adesso) */
    if (!t_r2snap) t_r2snap = (uint8_t*)_aligned_malloc(L2_LS_SIZE, 64);
    memcpy(t_r2snap, l->ls110, L2_LS_SIZE);
    uint8_t recb[160]; memcpy(recb, vm_base + rec, 160);
    l2_mem_rd = sync_rd_live; t_s_cur = l;
    const int rc = l2_run_ticket(l, ctx->gpr);
    if (rc == 0 && l->k12) __atomic_fetch_add(&c_cerrk[13], 1, __ATOMIC_RELAXED);   /* k13 = k12 tollerato */
    if (rc == 8 || rc == 9) { __atomic_fetch_add(&c_cref, 1, __ATOMIC_RELAXED); return; }
    if (rc != 0 || l->nw == 0) {
        __atomic_fetch_add(&c_cerr, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&c_cerrk[rc == 0 ? 14 : ((rc >= 100 ? rc - 100 : rc) % 14)], 1, __ATOMIC_RELAXED);
        return;
    }
    uint32_t tot = 0; for (uint32_t k = 0; k < l->nw; k++) tot += l->w[k].n;
    if (l->nw > 32 || tot > 2048) { __atomic_fetch_add(&c_cskip, 1, __ATOMIC_RELAXED); return; }
    AcquireSRWLockExclusive(&s_rplock);
    int slot = -1;
    for (int i = 0; i < RPN; i++) if (s_rp[i].used && (s_rp[i].rec == rec || s_rp[i].key == l->w[l->nw - 1].ea)) { slot = i; __atomic_fetch_add(&c_clost, 1, __ATOMIC_RELAXED); break; }   /* stesso record ancora in attesa: il suo flag non e' mai arrivato */
    if (slot < 0) for (int i = 0; i < RPN; i++) if (!s_rp[i].used) { slot = i; break; }
    if (slot < 0) { slot = (int)((rec >> 4) % RPN); __atomic_fetch_add(&c_clost, 1, __ATOMIC_RELAXED); }
    RPend* q = &s_rp[slot];
    if (!q->used) __atomic_fetch_add(&s_rpn, 1, __ATOMIC_RELEASE);
    q->rec = rec; q->nw = l->nw; q->key = l->w[l->nw - 1].ea; q->hdr_seen = 0;
    memcpy(q->regs, ctx->gpr, 2048); memcpy(q->recb, recb, 160);
    { uint8_t* t = q->snap; q->snap = t_r2snap; t_r2snap = t; }      /* scambio di puntatori: niente seconda copia da 256 KB */
    uint32_t off = 0;
    for (uint32_t k = 0; k < l->nw; k++) { q->w[k] = l->w[k]; q->w[k].off = off; memcpy(q->d + off, l->wd + l->w[k].off, l->w[k].n); off += l->w[k].n; }
    q->used = 1;
    ReleaseSRWLockExclusive(&s_rplock);
}

/* ---- modo IN LINEA (SPU_COLL_RING=3), rissa128: il ticket ad anello si esegue col motore nativo DIRETTAMENTE sul thread raw 110, nell'istante in cui
 * il lifted lo farebbe. Niente worker, niente coda, niente commit fuori ordine; il raw 111 non riceve voci per i ticket ad anello (lo stadio 111 lo fa
 * il motore) -> spariscono i passaggi di mano 110->111->110 (sonno+sveglia CV) che oggi fanno aspettare il main a 4A2920.
 * Il motore lavora su LS PRIVATE (110 copiata dalla viva, 111 sincronizzata): se rifiuta o sbaglia, nulla e' stato toccato e gira il lifted.
 * Riuscito: PUT applicate in ordine (stesse regole delle DMA), poi flags &= ~0x44 sulla linea di claim (chiusura di func_2640 con r4 == 2). ---- */
static int s_ign12_cfg;
static __thread L2* t_ri;
static volatile long c_iok, c_ifail, c_ifk[16];
static double s_ins_sync, s_ins_l2, s_ins_put;
static void ring_inline_line(void)
{
    const long n = c_iok ? c_iok : 1;
    fprintf(stderr, "[coll-ring] in linea: eseguiti %ld, ricaduti sul lifted %ld (", c_iok, c_ifail);
    for (int k = 0; k < 16; k++) if (c_ifk[k]) fprintf(stderr, " k%d=%ld", k, c_ifk[k]);
    fprintf(stderr, " ) | us/ticket: sync LS %.1f, motore %.1f, PUT+chiusura %.1f\n", s_ins_sync / n / 1000.0, s_ins_l2 / n / 1000.0, s_ins_put / n / 1000.0);
}
static int ring_inline(spu_context* ctx)
{
    const long long t0 = rq_now();
    if (!t_ri) {
        L2* l = (L2*)_aligned_malloc(sizeof(L2), 64); memset(l, 0, sizeof *l);
        l->ls110 = (uint8_t*)_aligned_malloc(L2_LS_SIZE, 64); l->ls111 = (uint8_t*)_aligned_malloc(L2_LS_SIZE, 64);
        memset(l->ls110, 0, L2_LS_SIZE); memset(l->ls111, 0, L2_LS_SIZE);
        t_ri = l;
    }
    L2* l = t_ri;
    sync_full_fast(l->ls110, ctx->ls);
    ring_sync_111(l, l2_be32(l->ls110 + 0x6DC4));
    if (!l->ls111_ok) { __atomic_fetch_add(&c_ifail, 1, __ATOMIC_RELAXED); __atomic_fetch_add(&c_ifk[15], 1, __ATOMIC_RELAXED); return 0; }
    l2_ign12 = s_ign12_cfg;
    const long long t1 = rq_now();
    l2_mem_rd = sync_rd_live; t_s_cur = l;
    int rc = l2_run_ticket(l, ctx->gpr);
    if (rc == 0 && l->nw == 0) rc = 114;
    if (rc == 0 && !l->line_ea) rc = 113;       /* senza linea non sapremmo rilasciare flags 0x44 (il kick del PPU girerebbe per sempre): gira il lifted */
    const long long t2 = rq_now();
    if (rc != 0) {
        __atomic_fetch_add(&c_ifail, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&c_ifk[(rc >= 100 ? rc - 100 : rc) & 15], 1, __ATOMIC_RELAXED);
        return 0;
    }
    for (uint32_t k = 0; k < l->nw; k++) ring_put_guest(l->w[k].ea, l->wd + l->w[k].off, l->w[k].n, ctx);
    ring_clear_flags(l->line_ea, 0x44u);
    const long long t3 = rq_now();
    s_ins_sync += (double)(t1 - t0) * 1e9 / (double)s_rqfreq; s_ins_l2 += (double)(t2 - t1) * 1e9 / (double)s_rqfreq; s_ins_put += (double)(t3 - t2) * 1e9 / (double)s_rqfreq;
    if ((__atomic_add_fetch(&c_iok, 1, __ATOMIC_RELAXED) & 0x3FFF) == 0) ring_inline_line();
    return 1;
}

/* ingresso dal prologo di igCollBroadSpu func_00002640 (thread raw 110). 1 = consegnato, 0 = gira il lifted. */
int spu_ring_enter(spu_context* ctx, void (*self)(spu_context*))
{
    (void)self;
    if ((__atomic_add_fetch(&c_rcalls, 1, __ATOMIC_RELAXED) & 0x3FFF) == 0 && s_rmode == 2) ring_dual_line();   /* rissa127: riepilogo ogni 16k ingressi */
    const uint32_t rec = ctx->gpr[3]._u32[0];
    int eligible = 0;
    { const uint32_t r4 = ctx->gpr[4]._u32[0]; __atomic_fetch_add(&c_r4hist[r4 < 7u ? r4 : 7u][ring_is_ring(rec) ? 1 : 0], 1, __ATOMIC_RELAXED); }
    /* func_2640 rilascia flags &= ~0x44 solo se r4 == 2 (selb(-5,-69, ceqi(r4,2)) a 2A70/2AEC): quel caso e' il pop dell'anello */
    if (ring_is_ring(rec) && ctx->gpr[4]._u32[0] == 2u && (rec & 15u) == 0 && sync_page_ok(rec, 160u)) {
        const uint32_t ty = ((uint32_t)vm_base[rec + 12] << 8) | vm_base[rec + 13];
        eligible = (ty <= 2u);
    }
    if (s_rmode == 2) { ring_dual(ctx, rec, eligible && ctx->ls == vm_base + l2_be32(ctx->ls + 0x6DC4)); return 0; }
    if (s_rmode == 3) { if (eligible && ctx->ls && ctx->ls == vm_base + l2_be32(ctx->ls + 0x6DC4) && ring_inline(ctx)) SPU_RING_RET(ctx); return 0; }
    if (!eligible) { __atomic_fetch_add(&c_rnonring, 1, __ATOMIC_RELAXED); ring_drain(); return 0; }
    if (!ctx->ls || ctx->ls != vm_base + l2_be32(ctx->ls + 0x6DC4)) { ring_drain(); return 0; }    /* LS non e' l'alias guest atteso */
    /* ricaduta pendente: consegna il ticket corrente al pool e fai girare lifted quello rifiutato (r3/r4 scambiati) */
    if (s_rfail_n > 0) {
        u128 r3, r4;
        EnterCriticalSection(&s_rfcs);
        long n = s_rfail_n;
        if (n > 0) {
            r3 = s_rfail[n - 1].r3; r4 = s_rfail[n - 1].r4; __atomic_store_n(&s_rfail_n, n - 1, __ATOMIC_RELEASE);
            LeaveCriticalSection(&s_rfcs);
            ring_submit(ctx, rec);
            ctx->gpr[3] = r3; ctx->gpr[4] = r4;
            __atomic_fetch_add(&c_rswap, 1, __ATOMIC_RELAXED); __atomic_fetch_add(&c_rfailrun, 1, __ATOMIC_RELAXED);
            return 0;
        }
        LeaveCriticalSection(&s_rfcs);
    }
    ring_submit(ctx, rec);
    SPU_RING_RET(ctx);
}

static void ring_start(int n)
{
    QueryPerformanceFrequency((LARGE_INTEGER*)&s_rqfreq);
    InitializeCriticalSection(&s_rcs); InitializeCriticalSection(&s_rfcs);
    s_rsem = CreateSemaphore(NULL, 0, 1 << 20, NULL);
    for (int i = 0; i < RING_SLOTS; i++) {
        s_rj[i] = (RJob*)_aligned_malloc(sizeof(RJob), 64); memset(s_rj[i], 0, sizeof(RJob));
        s_rj[i]->snap = (uint8_t*)_aligned_malloc(L2_LS_SIZE, 64);
        s_rfree[s_rnfree++] = i;
    }
    for (int i = 0; i < n; i++) CreateThread(NULL, 0, ring_worker, NULL, 0, NULL);
}

__attribute__((constructor)) static void spu_coll_ring_init(void)
{
    const char* e = getenv("SPU_COLL_RING");
    /* release 1.2: this constructor runs before main(), so neither tod.ini nor main.cpp's performance defaults reach it (b58 ran without
     * the ring: 34% of seconds >=59 instead of 60%). Unset = 3 unless TOD_PERF_DEFAULTS=0 (1.1 behaviour). */
    const char* pd = getenv("TOD_PERF_DEFAULTS");
    const int perf_def = !(pd && pd[0] == '0');
    int m = e ? atoi(e) : (perf_def ? 3 : 0);
    if (getenv("SPU_LS_WATCH") || getenv("SPU_LS_LOWREAD") || getenv("SPU_SMC_WATCH") || getenv("SPU_COLLTAB_WATCH") || getenv("SPU_JOBTRACE") || getenv("SPU_JOBEXEC")) m = 0;
    if (m <= 0) return;
    const char* nn = getenv("SPU_COLL_RING_N"); s_rn = nn ? atoi(nn) : 2; if (s_rn < 1) s_rn = 1; if (s_rn > 8) s_rn = 8;
    const char* ps = getenv("SPU_COLL_RING_SPIN_US"); if (ps && *ps) s_rspin_us = atoi(ps);
    s_rmode = (m == 2 || m == 3) ? m : 1;
    { const char* e = getenv("SPU_COLL_RING_IGN12"); s_ign12_cfg = e ? e[0] == '1' : perf_def; }
    { const char* e = getenv("SPU_COLL_RING_HDR"); s_ring_hdr = e && e[0] == '1'; }
    if (s_rmode >= 2) QueryPerformanceFrequency((LARGE_INTEGER*)&s_rqfreq);
    if (s_rmode == 1) ring_start(s_rn);
    if (s_rmode == 2) { extern void (*g_spu_ring_preput)(uint32_t, const uint8_t*, uint32_t); g_spu_ring_preput = ring_preput; }
    g_spu_ring_on = 1;
    atexit(ring_atexit);
    fprintf(stderr, "[coll-ring] SPU_COLL_RING=%d: %s\n", m, m == 3 ? "ticket ad anello col motore nativo IN LINEA sul raw 110" : m == 2 ? "modo doppio (confronto nativo/lifted)" : "pool nativo per i ticket ad anello");
}
