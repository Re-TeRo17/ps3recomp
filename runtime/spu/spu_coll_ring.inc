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
static double s_rns_run, s_rns_wait, s_rns_sub; static volatile long s_rmaxq;
static long long s_rqfreq; static volatile long s_rcommit; static long s_rseq;
static volatile long s_rinfl; static volatile uint32_t s_rline_ea;
static volatile long s_rerrk[16];
static volatile long c_r4hist[8][2];
static int s_rmode;
static volatile long c_ceq, c_cdiff, c_cerr, c_cref;

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
    if (s_rmode == 2) fprintf(stderr, "[coll-ring]   doppio: confronti uguali %ld diversi %ld, errori %ld, rifiutati %ld\n", c_ceq, c_cdiff, c_cerr, c_cref);
    fprintf(stderr, "[coll-ring]   r4 all'ingresso (non-anello/anello):"); for (int i = 0; i < 8; i++) if (c_r4hist[i][0] || c_r4hist[i][1]) fprintf(stderr, " r4=%d:%ld/%ld", i, c_r4hist[i][0], c_r4hist[i][1]); fprintf(stderr, "\n");
    if (c_rfail) { fprintf(stderr, "[coll-ring]   rifiuti per codice:"); for (int i = 0; i < 16; i++) if (s_rerrk[i]) fprintf(stderr, " k%d=%ld", i, s_rerrk[i]); fprintf(stderr, "\n"); }
    fflush(stderr);
}
static void ring_atexit(void) { ring_print_stats("exit"); }

static void ring_run_job(L2* l, RJob* j)
{
    const long long t0 = rq_now();
    l->ls110 = j->snap;
    ring_sync_111(l, l2_be32(l->ls110 + 0x6DC4));
    int rc = l->ls111_ok ? 0 : 1;
    if (!rc) { l2_mem_rd = sync_rd_live; t_s_cur = l; rc = l2_run_ticket(l, j->regs); }
    if (rc == 0 && l->nw == 0) rc = 2;
    /* commit IN ORDINE di pubblicazione: le PUT di un ticket si applicano solo dopo quelle dei precedenti */
    while (__atomic_load_n(&s_rcommit, __ATOMIC_ACQUIRE) != j->seq) _mm_pause();
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
    memcpy(j->snap, ctx->ls, L2_LS_SIZE);
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

/* ---- modo DOPPIO (SPU_COLL_RING=2): su ogni ticket ad anello valido il driver nativo gira PRIMA, sul raw 110, su una LS privata sincronizzata dalla LS
 * viva (sync completo veloce) e produce un LOG di PUT che NON si applica; il gancio ritorna 0 (gira il lifted vero). Al gancio successivo
 * (il lifted ha finito) il log si confronta byte per byte con la memoria guest (bit 15 di rec+0x18 mascherato: il PPU puo' azzerarlo). Non annidato. ---- */
static __thread L2* t_r2; static __thread int t_r2_pend;
static volatile long c_cty[3]; static volatile long s_cdlog;
static void ring_hexq(char* o, const uint8_t* p, uint32_t n) { for (uint32_t i = 0; i < n; i++) sprintf(o + 2 * i, "%02X", p[i]); }
static void ring_cmp(L2* l)
{
    uint32_t cov[1024]; uint32_t ncov = 0, nbad = 0; int first_bad = -1; uint32_t bad_ea = 0;
    for (int k = (int)l->nw - 1; k >= 0; k--) {
        const L2Wr* w = &l->w[k];
        for (uint32_t o = 0; o < w->n; o += 16u) {
            const uint32_t qea = (w->ea + o) & ~15u, len = (w->n - o) < 16u ? (w->n - o) : 16u;
            int seen = 0; for (uint32_t c = 0; c < ncov; c++) if (cov[c] == qea) { seen = 1; break; }
            if (seen) continue;
            if (ncov < 1024) cov[ncov++] = qea;
            uint8_t a[16], b[16]; memcpy(a, vm_base + w->ea + o, len); memcpy(b, l->wd + w->off + o, len);
            if (len == 16u && w->ea + o == l->rec_ea + 0x10u) { a[8] &= 0x7F; b[8] &= 0x7F; }
            if (memcmp(a, b, len) != 0) { nbad++; if (first_bad < 0 || k > first_bad) { first_bad = k; bad_ea = w->ea + o; } }
        }
    }
    const int ty = ((int)(vm_base[l->rec_ea + 12]) << 8) | vm_base[l->rec_ea + 13];
    (void)ty;
    if (nbad == 0) { __atomic_fetch_add(&c_ceq, 1, __ATOMIC_RELAXED); return; }
    __atomic_fetch_add(&c_cdiff, 1, __ATOMIC_RELAXED);
    long nl = __atomic_add_fetch(&s_cdlog, 1, __ATOMIC_RELAXED);
    if (nl <= 12) {
        char a[80], b[80];
        fprintf(stderr, "[coll-ring] DIFF #%ld rec=0x%08X entries=%u puts=%u quad diversi=%u primo_ea=0x%08X (put #%d)\n", nl, l->rec_ea, l->n_entries, l->nw, nbad, bad_ea, first_bad);
        if (first_bad >= 0) { const L2Wr* w = &l->w[first_bad]; ring_hexq(a, l->wd + w->off + (bad_ea - w->ea), 16u); ring_hexq(b, vm_base + bad_ea, 16u); fprintf(stderr, "[coll-ring]   nativo=%s\n[coll-ring]   raw    =%s\n", a, b); }
    }
}
static void ring_dual(spu_context* ctx, uint32_t rec, int eligible)
{
    if (t_r2 && t_r2_pend) { ring_cmp(t_r2); t_r2_pend = 0; if ((((c_ceq + c_cdiff) & 0x3FFF) == 0)) fprintf(stderr, "[coll-ring] doppio: uguali %ld diversi %ld errori %ld rifiutati %ld\n", c_ceq, c_cdiff, c_cerr, c_cref); }
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
    if (!l->ls111_ok) { __atomic_fetch_add(&c_cerr, 1, __ATOMIC_RELAXED); return; }
    l2_mem_rd = sync_rd_live; t_s_cur = l;
    const int rc = l2_run_ticket(l, ctx->gpr);
    if (rc == 0 && l->nw > 0) { t_r2_pend = 1; __atomic_fetch_add(&c_cty[0], 1, __ATOMIC_RELAXED); }
    else if (rc == 8 || rc == 9) __atomic_fetch_add(&c_cref, 1, __ATOMIC_RELAXED);
    else __atomic_fetch_add(&c_cerr, 1, __ATOMIC_RELAXED);
}

/* ingresso dal prologo di igCollBroadSpu func_00002640 (thread raw 110). 1 = consegnato, 0 = gira il lifted. */
int spu_ring_enter(spu_context* ctx, void (*self)(spu_context*))
{
    (void)self;
    __atomic_fetch_add(&c_rcalls, 1, __ATOMIC_RELAXED);
    const uint32_t rec = ctx->gpr[3]._u32[0];
    int eligible = 0;
    { const uint32_t r4 = ctx->gpr[4]._u32[0]; __atomic_fetch_add(&c_r4hist[r4 < 7u ? r4 : 7u][ring_is_ring(rec) ? 1 : 0], 1, __ATOMIC_RELAXED); }
    /* func_2640 rilascia flags &= ~0x44 solo se r4 == 2 (selb(-5,-69, ceqi(r4,2)) a 2A70/2AEC): quel caso e' il pop dell'anello */
    if (ring_is_ring(rec) && ctx->gpr[4]._u32[0] == 2u && (rec & 15u) == 0 && sync_page_ok(rec, 160u)) {
        const uint32_t ty = ((uint32_t)vm_base[rec + 12] << 8) | vm_base[rec + 13];
        eligible = (ty <= 2u);
    }
    if (s_rmode == 2) { ring_dual(ctx, rec, eligible && ctx->ls == vm_base + l2_be32(ctx->ls + 0x6DC4)); return 0; }
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
    int m = e ? atoi(e) : 0;
    if (getenv("SPU_LS_WATCH") || getenv("SPU_LS_LOWREAD") || getenv("SPU_SMC_WATCH") || getenv("SPU_COLLTAB_WATCH") || getenv("SPU_JOBTRACE") || getenv("SPU_JOBEXEC")) m = 0;
    if (m <= 0) return;
    const char* nn = getenv("SPU_COLL_RING_N"); s_rn = nn ? atoi(nn) : 2; if (s_rn < 1) s_rn = 1; if (s_rn > 8) s_rn = 8;
    const char* ps = getenv("SPU_COLL_RING_SPIN_US"); if (ps && *ps) s_rspin_us = atoi(ps);
    s_rmode = (m == 2) ? 2 : 1;
    if (s_rmode == 1) ring_start(s_rn);
    g_spu_ring_on = 1;
    atexit(ring_atexit);
    fprintf(stderr, "[coll-ring] SPU_COLL_RING=%d: pool nativo per i ticket ad anello, %d worker\n", m, s_rn);
}
