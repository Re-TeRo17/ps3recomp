/* spu_coll_sync.c (sync59) -- collisione SINCRONA nativa nel thread PPU principale.
 *
 * Usa il motore di coll53/coll57 (spu_coll_l2_*.inc: F_PRO + F_Q28A8 per img 110, F_V122A0 per img 111, driver per ticket
 * l2_run_ticket) per eseguire il ticket dello SLOT sincrono (record da 160 B a 10612D80, sottomesso da 4A3288) nel chiamante, senza
 * svegliare le raw SPU. Il lato PPU (src\stubs\coll_sync.cpp) sostituisce func_004A3288:
 *   SPU_COLL_SYNC=0 originale (q92: il default e' ora 1; SPU_COLL_SYNC_FULL=0 / SPU_COLL_SYNC_IGN12=0 per i sync vecchi).
 *   SPU_COLL_SYNC=1 solo nativo con fallback: record preparato dal lifted con il puntatore del "kick" deviato su una parola privata
 *                   (le raw SPU NON vedono nulla), ticket nativo, log delle PUT applicato in ordine; tipo/kind non coperti o errore ->
 *                   si fa il kick vero e girano le raw SPU (nessun effetto parziale: il log si applica solo a ticket finito).
 *   SPU_COLL_SYNC=2 DOPPIO: nativo su copia PRIMA del kick, poi kick vero; a ticket chiuso dalle raw SPU confronto byte per byte di
 *                   tutto cio' che il nativo avrebbe scritto con la memoria vera. Il nativo non gira mai in parallelo con le raw SPU.
 * Compilato come le immagini lifted (clang -O2 -march=x86-64-v3, SPU_RZ_FP_TLS). Sul thread PPU si imposta MXCSR=RZ|DAZ|FTZ e
 * g_spu_rz=1 per la durata del ticket (come il thread raw SPU), poi si ripristina. */
#include "spu_context.h"
#include "spu_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <emmintrin.h>
#ifdef _WIN32
#include <windows.h>
#endif

#ifndef COLL_FAST_OPS
#define COLL_FAST_OPS 1   /* coll101: operazioni FP/confronto veloci (spu_coll_fast.inc); -DCOLL_FAST_OPS=0 = codice originale */
#endif
#if COLL_FAST_OPS
#include "spu_coll_fast.inc"
#endif
/* coll101: i registri R[] vivono in N->R, l'LS in un buffer a parte: N e' "restrict" nelle funzioni generate, cosi' le store all'LS (uint8_t*)
 * non invalidano i registri gia' caricati. -DCOLL_RESTRICT=0 = codice originale. */
#ifndef COLL_RESTRICT
#define COLL_RESTRICT 1
#endif
#if COLL_RESTRICT
#define COLL_NR __restrict
#else
#define COLL_NR
#endif
#include "spu_coll_l2_nc.h"
#include "spu_coll_l2_gen_all.inc"
#include "spu_coll_l2_gen_pro.inc"
#include "spu_coll_l2_gen_111.inc"
#include "spu_coll_l2_img.inc"
#include "spu_coll_l2_core.inc"
#include "spu_coll_sync_regs.inc"
#include "spu_ring_hook.h"

extern uint8_t* vm_base;
extern uint8_t g_vm_page_bitmap[65536 / 8];
extern int g_spu_coh_armed;
extern unsigned char g_spu_coh_bitmap[];
extern void (*g_spu_put_hook)(uint32_t ea, uint32_t size);
extern void spu_coh_notify_write(uint32_t ea);
extern void spu_lockline_lock(void);
extern void spu_lockline_unlock(void);
extern SPU_THREAD_LOCAL int g_spu_rz;

static __thread L2* t_s;           /* driver del thread */
static __thread L2* t_s_cur;       /* per la lettura memoria */
static int s_mode = -1;
static unsigned s_waitms = 50;

static volatile long c_calls, c_elig, c_native_ok, c_ref8, c_ref9, c_err, c_err_k[16], c_equal, c_diff, c_timeout, c_applied, c_fallback, c_notready, c_with111, c_entries, c_resync110, c_resync111;
static volatile long c_ty[3], c_ty_diff[3];
static double s_ns_native, s_ns_orig; static volatile long s_ns_n;
static volatile long s_diff_logged; static volatile long c_rel, c_rel_ty[3];
static double s_t_pro, s_t_111, s_t_prep, s_t_apply; static volatile long s_n_prep, s_n_apply; static volatile long s_ent_hist[8]; static volatile long s_nw_sum; static double s_t_get; static volatile long s_n_get, s_b_get, s_n_lsget;
static int s_fullsync = -1;
static volatile long c_k12_n, c_k12_eq, c_k12_diff;
#define SIGN 24
static struct { uint32_t key; uint32_t n; uint32_t lsa; uint8_t nat[16], raw[16]; } s_sig[SIGN]; static int s_nsig;

static int sync_page_ok(uint32_t ea, uint32_t n)
{
    const uint32_t last = ea + n - 1u;
    if (last < ea || !vm_base) return 0;
    const uint32_t pg[2] = { ea >> 16, last >> 16 };
    int ok = 1;
    for (int i = 0; i < (pg[0] == pg[1] ? 1 : 2); i++)
        if (!((g_vm_page_bitmap[pg[i] >> 3] >> (pg[i] & 7)) & 1)) { ok = 0; break; }
    if (ok) return 1;
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(vm_base + ea, &mbi, sizeof mbi) == 0) return 0;
    if (mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD))) return 0;
    return ((uint8_t*)mbi.BaseAddress + mbi.RegionSize) >= (vm_base + ea + n);
#else
    return 0;
#endif
}

static void sync_rd_live(uint32_t ea, uint8_t* dst, uint32_t n)
{
    if (!sync_page_ok(ea, n)) l2_fail(t_s_cur, 4);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    memcpy(dst, vm_base + ea, n);
    if (n <= 512u) {
        int it = 0;
        while (it < 8 && memcmp(dst, vm_base + ea, n) != 0) { memcpy(dst, vm_base + ea, n); it++; }
    }
}

static inline int sync_coh_reserved(uint32_t a) { return g_spu_coh_armed && ((g_spu_coh_bitmap[a >> 10] >> ((a >> 7) & 7)) & 1u); }

/* PUT in memoria guest dal thread PPU: release, copia, notifica di coerenza per le linee riservate da una SPU, hook */
static void sync_put_guest(uint32_t ea, const uint8_t* src, uint32_t n)
{
    uint8_t* p = vm_base + ea;
    const uint32_t a0 = ea & ~127u, a1 = (ea + n - 1u) & ~127u;
    int reserved = 0;
    for (uint32_t a = a0; ; a += 128u) { if (sync_coh_reserved(a)) { reserved = 1; break; } if (a == a1) break; }
    __atomic_thread_fence(__ATOMIC_RELEASE);
    if (reserved) spu_lockline_lock();
    if (n == 16u) _mm_storeu_si128((__m128i*)p, _mm_loadu_si128((const __m128i*)src));
    else memcpy(p, src, n);
    if (reserved) {
        for (uint32_t a = a0; ; a += 128u) { if (sync_coh_reserved(a)) spu_coh_notify_write(a); if (a == a1) break; }
        spu_lockline_unlock();
    }
    if (g_spu_put_hook) g_spu_put_hook(ea, n);
}

/* LS vive dei raw SPU = alias guest E0xxxxxx. img 110: la parola a 0x6DC4 e' il proprio alias; img 111: tabella callback a 0x13900 */
static const uint8_t* sync_find_ls110(void)
{
    for (uint32_t k = 0; k < 8; k++) {
        const uint32_t base = 0xE0000000u + k * 0x100000u;
        if (!sync_page_ok(base, 0x40000u)) continue;
        if (l2_be32(vm_base + base + 0x6DC4) == base) return vm_base + base;
    }
    return NULL;
}
static const uint8_t* sync_find_ls111(uint32_t alias110)
{
    for (uint32_t k = 0; k < 8; k++) {
        const uint32_t base = 0xE0000000u + k * 0x100000u;
        if (base == alias110) continue;
        if (!sync_page_ok(base, 0x40000u)) continue;
        const uint8_t* c = vm_base + base;
        const uint32_t e = l2_be32(c + 0x13900);
        if (l2_be32(c + 0x13908) == 0x13A80u && e >= 0x10000000u && e < 0x20000000u) return c;
    }
    return NULL;
}

int spu_coll_sync_mode(void)
{
    if (s_mode < 0) {
        const char* e = getenv("SPU_COLL_SYNC");
        int m = e ? atoi(e) : 1;   /* q92: default 1 (nativo con fallback, sync completo veloce); SPU_COLL_SYNC=0 lo disattiva */
        if (m < 0 || m > 2) m = 0;
        if (getenv("SPU_LS_WATCH") || getenv("SPU_LS_LOWREAD") || getenv("SPU_SMC_WATCH") || getenv("SPU_COLLTAB_WATCH") || getenv("SPU_JOBTRACE") || getenv("SPU_JOBEXEC")) m = 0;
        const char* w = getenv("SPU_COLL_SYNC_WAITMS"); if (w && *w) s_waitms = (unsigned)atoi(w);
        s_mode = m;
        if (m) fprintf(stderr, "[coll-sync] SPU_COLL_SYNC=%d (%s)\n", m, m == 2 ? "DOPPIO: nativo su copia + raw SPU vere + confronto" : "NATIVO con fallback alle raw SPU");
    }
    return s_mode;
}

static void sync_print_stats(const char* why);
static void sync_atexit(void) { sync_print_stats("exit"); }


/* ---- q89: sync LS110 incrementale. Il sync completo (sync62 FULL) costa ~46 us/ticket (memset 256 KB + scansione a quad), piu' del ticket
 * nativo stesso. Qui: S = insieme (appreso) di blocchi da 64 B dove la LS privata diverge dal bersaglio (quad grezzo non nullo, altrimenti
 * il modello = immagine ELF a 0x60) fra due ticket (scritture delle raw SPU + scritture del motore nativo). Per ticket si ri-sincronizza solo S;
 * ogni K ticket sync completo con confronto: i blocchi nuovi entrano in S (c_late = scoperte tardive). */
static uint8_t* s_tmpl110;
static uint64_t s_S[(L2_LS_SIZE / 64) / 64];
static unsigned s_K = 8, s_learn = 1024, s_cnt_full;
static int s_fast = -1;
static volatile long c_fast_inc, c_fast_full, c_late, c_Sn;
static inline int s_inS(uint32_t b) { return (int)((s_S[b >> 6] >> (b & 63)) & 1u); }
static inline void sync_tblock(uint8_t* priv, const uint8_t* raw, uint32_t b)
{
    const uint32_t o = b * 64u;
    for (uint32_t q = 0; q < 64u; q += 16u) {
        uint64_t x, y; memcpy(&x, raw + o + q, 8); memcpy(&y, raw + o + q + 8, 8);
        memcpy(priv + o + q, (x | y) ? raw + o + q : s_tmpl110 + o + q, 16);
    }
}
static void sync_ls110_fast(L2* l, const uint8_t* raw, int full)
{
    uint8_t* priv = l->ls110;
    if (!s_tmpl110) {
        s_tmpl110 = (uint8_t*)_aligned_malloc(L2_LS_SIZE, 64);
        memset(s_tmpl110, 0, L2_LS_SIZE); memcpy(s_tmpl110 + 0x60, k_l2_bin110, sizeof k_l2_bin110);
    }
    if (!full) {
        for (uint32_t w = 0; w < sizeof s_S / sizeof s_S[0]; w++) {
            uint64_t m = s_S[w];
            while (m) { const uint32_t b = (w << 6) + (uint32_t)__builtin_ctzll(m); m &= m - 1; sync_tblock(priv, raw, b); }
        }
        __atomic_fetch_add(&c_fast_inc, 1, __ATOMIC_RELAXED);
        return;
    }
    uint8_t tgt[64];
    for (uint32_t b = 0; b < L2_LS_SIZE / 64u; b++) {
        const uint32_t o = b * 64u;
        for (uint32_t q = 0; q < 64u; q += 16u) {
            uint64_t x, y; memcpy(&x, raw + o + q, 8); memcpy(&y, raw + o + q + 8, 8);
            memcpy(tgt + q, (x | y) ? raw + o + q : s_tmpl110 + o + q, 16);
        }
        if (memcmp(tgt, priv + o, 64) != 0) {
            memcpy(priv + o, tgt, 64);
            if (!s_inS(b)) { s_S[b >> 6] |= 1ull << (b & 63); __atomic_fetch_add(&c_late, 1, __ATOMIC_RELAXED); c_Sn++; }
        }
    }
    __atomic_fetch_add(&c_fast_full, 1, __ATOMIC_RELAXED);
}


/* ---- q92: sync completo veloce + modo "nessun sync" (l'LS privata evolve da sola come l'LS vera) + diagnostica dei blocchi divergenti.
 * target(quad) = quad grezzo non nullo ? grezzo : modello (immagine ELF a 0x60, zero altrove). Fuori dalla regione immagine il modello e' zero,
 * quindi target = grezzo: memcpy puro; dentro la regione immagine si fa il blend per quad. */
static const uint8_t* g_raw110_cur; static uint8_t* s_V; static uint8_t* s_Lpost; static int s_m1view = -1; static volatile long c_m1_upd;
static int s_nosync = -1;      /* SPU_COLL_SYNC_NOSYNC=1: nessun sync per ticket (solo dopo ricaduta sulle raw SPU / ticket non visti) */
static unsigned s_chk = 0;     /* SPU_COLL_SYNC_CHK=N: ogni N ticket confronta LS privata/viva (diagnostica), poi riallinea i blocchi diversi */
static unsigned s_chk_cnt;
static volatile int s_need_full = 0;
static volatile long c_chk_n, c_chk_diffticks, c_chk_blocks, c_full_fast, c_full_forced;
static uint32_t s_blkhist[L2_LS_SIZE / 64];
static double s_t_fullcopy; static volatile long s_n_fullcopy;
static void sync_tmpl_init(void)
{
    if (s_tmpl110) return;
    s_tmpl110 = (uint8_t*)_aligned_malloc(L2_LS_SIZE, 64);
    memset(s_tmpl110, 0, L2_LS_SIZE); memcpy(s_tmpl110 + 0x60, k_l2_bin110, sizeof k_l2_bin110);
}
static void sync_full_fast(uint8_t* priv, const uint8_t* raw)
{
    sync_tmpl_init();
    const uint32_t r0 = 0x60u & ~15u, r1 = (0x60u + (uint32_t)sizeof k_l2_bin110 + 15u) & ~15u;
    memcpy(priv, raw, r0);
    for (uint32_t off = r0; off < r1; off += 16u) {
        uint64_t a, b; memcpy(&a, raw + off, 8); memcpy(&b, raw + off + 8, 8);
        memcpy(priv + off, (a | b) ? raw + off : s_tmpl110 + off, 16);
    }
    memcpy(priv + r1, raw + r1, L2_LS_SIZE - r1);
}
/* diagnostica: blocchi da 64 B in cui la LS privata diverge dal bersaglio; riallinea quei blocchi. ritorna il numero di blocchi diversi */
static unsigned sync_chk_blocks(uint8_t* priv, const uint8_t* raw, const char* tag)
{
    sync_tmpl_init();
    unsigned n = 0; uint32_t list[16];
    for (uint32_t b = 0; b < L2_LS_SIZE / 64u; b++) {
        const uint32_t o = b * 64u; uint8_t tgt[64];
        for (uint32_t q = 0; q < 64u; q += 16u) {
            uint64_t x, y; memcpy(&x, raw + o + q, 8); memcpy(&y, raw + o + q + 8, 8);
            memcpy(tgt + q, (x | y) ? raw + o + q : s_tmpl110 + o + q, 16);
        }
        if (memcmp(tgt, priv + o, 64) != 0) { if (n < 16) list[n] = b; n++; s_blkhist[b]++; memcpy(priv + o, tgt, 64); }
    }
    c_chk_n++;
    if (n) {
        c_chk_diffticks++; c_chk_blocks += n;
        static int logged; if (logged < 60) { logged++; fprintf(stderr, "[coll-sync] CHK(%s) #%ld: %u blocchi LS110 diversi:", tag, c_chk_n, n); for (unsigned i = 0; i < n && i < 16; i++) fprintf(stderr, " %05X", list[i] * 64u); fprintf(stderr, "\n"); }
    }
    return n;
}
static void sync_print_blkhist(void)
{
    if (!s_chk) return;
    { const char* fn = getenv("SPU_COLL_SYNC_CHKFILE"); if (fn && *fn) { FILE* f = fopen(fn, "w"); if (f) { fprintf(f, "controlli %ld con_diff %ld blocchi %ld\n", c_chk_n, c_chk_diffticks, c_chk_blocks);
        for (uint32_t b = 0; b < L2_LS_SIZE / 64u; b++) if (s_blkhist[b]) fprintf(f, "%05X %u\n", b * 64u, s_blkhist[b]); fclose(f); } return; } }
    fprintf(stderr, "[coll-sync]   CHK: controlli %ld, con differenze %ld, blocchi diversi totali %ld; istogramma per blocco (blocco=64B, conteggio):\n", c_chk_n, c_chk_diffticks, c_chk_blocks);
    int pr = 0; for (uint32_t b = 0; b < L2_LS_SIZE / 64u && pr < 80; b++) if (s_blkhist[b]) { fprintf(stderr, "[coll-sync]     %05X:%u\n", b * 64u, s_blkhist[b]); pr++; }
}

static int spu_coll_sync_prepare_i(void);
/* LS private allineate alle LS vive; 1 se pronto */
int spu_coll_sync_prepare(void)
{
    LARGE_INTEGER qp0, qp1, qpf; QueryPerformanceCounter(&qp0);
    int r = spu_coll_sync_prepare_i();
    QueryPerformanceCounter(&qp1); QueryPerformanceFrequency(&qpf);
    s_t_prep += (double)(qp1.QuadPart - qp0.QuadPart) * 1e9 / (double)qpf.QuadPart; __atomic_fetch_add(&s_n_prep, 1, __ATOMIC_RELAXED);
    return r;
}
static int spu_coll_sync_prepare_i(void)
{
    if (!vm_base) return 0;
    if (s_fullsync < 0) { { const char* q = getenv("SPU_COLL_SYNC_IGN12"); l2_ign12 = !(q && *q == '0'); } { const char* q = getenv("SPU_COLL_SYNC_QPC"); l2_qpc = (q && *q == '1'); } const char* e = getenv("SPU_COLL_SYNC_FULL"); s_fullsync = !(e && *e == '0'); }
    /* cache degli alias LS (sync_find_* fanno fino a 16 VirtualQuery: ~40 us); validazione con le parole marcatore */
    static const uint8_t* c_raw110; static const uint8_t* c_raw111;
    const uint8_t* raw110;
    if (c_raw110 && l2_be32(c_raw110 + 0x6DC4) == (uint32_t)(c_raw110 - vm_base)) raw110 = c_raw110;
    else { raw110 = sync_find_ls110(); c_raw110 = raw110; c_raw111 = NULL; }
    if (!raw110) { __atomic_fetch_add(&c_notready, 1, __ATOMIC_RELAXED); s_need_full = 1; return 0; }
    if (!t_s) {
        L2* l = (L2*)_aligned_malloc(sizeof(L2), 64);
        if (!l) return 0;
        memset(l, 0, sizeof *l);
        l->ls110 = (uint8_t*)_aligned_malloc(L2_LS_SIZE, 64);
        l->ls111 = (uint8_t*)_aligned_malloc(L2_LS_SIZE, 64);
        if (!l->ls110 || !l->ls111) return 0;
        l2_sync_ls(l->ls110, k_l2_bin110, sizeof k_l2_bin110, raw110);
        memset(l->ls111, 0, L2_LS_SIZE);
        t_s = l;
        atexit(sync_atexit);
    }
    L2* l = t_s;
    if (s_fast < 0) { const char* e = getenv("SPU_COLL_SYNC_FAST"); s_fast = (e && *e == '1'); const char* k = getenv("SPU_COLL_SYNC_K"); if (k && *k) s_K = (unsigned)atoi(k); { const char* ln = getenv("SPU_COLL_SYNC_LEARN"); if (ln && *ln) s_learn = (unsigned)atoi(ln); } if (s_K < 1) s_K = 1; }
    g_raw110_cur = raw110;
    if (s_m1view < 0) { const char* e = getenv("SPU_COLL_SYNC_M1VIEW"); s_m1view = (e && *e == '1'); }
    if (s_m1view) {
        /* q92: vista "modo 1": cio' che vedrebbe il nativo se le raw SPU non eseguissero i ticket: LS viva senza le scritture delle raw SPU durante i ticket */
        if (!s_V) { s_V = (uint8_t*)_aligned_malloc(L2_LS_SIZE, 64); s_Lpost = (uint8_t*)_aligned_malloc(L2_LS_SIZE, 64); memcpy(s_V, raw110, L2_LS_SIZE); memcpy(s_Lpost, raw110, L2_LS_SIZE); }
        else for (uint32_t b = 0; b < L2_LS_SIZE / 64u; b++) { const uint32_t o = b * 64u; if (memcmp(raw110 + o, s_Lpost + o, 64) != 0) { memcpy(s_V + o, raw110 + o, 64); memcpy(s_Lpost + o, raw110 + o, 64); c_m1_upd++; } }
        raw110 = s_V;
    }
    { static int s_frozen = -1; static uint8_t* s_snap;
      if (s_frozen < 0) { const char* e = getenv("SPU_COLL_SYNC_FROZEN"); s_frozen = (e && *e == '1'); }
      if (s_frozen) { /* q92: l'LS "viva" resta quella del primo ticket (come nel modo 1, dove le raw SPU non eseguono i ticket nativi) */
          if (!s_snap) { s_snap = (uint8_t*)_aligned_malloc(L2_LS_SIZE, 64); memcpy(s_snap, raw110, L2_LS_SIZE); }
          raw110 = s_snap; } }
    if (s_nosync < 0) { const char* e = getenv("SPU_COLL_SYNC_NOSYNC"); s_nosync = (e && *e == '1'); const char* c = getenv("SPU_COLL_SYNC_CHK"); if (c && *c) s_chk = (unsigned)atoi(c); }
    if (s_nosync) {
        /* l'LS privata evolve da sola (il motore nativo scrive esattamente cio' che scriverebbe la SPU). Sync completo solo se le raw SPU hanno
         * eseguito ticket che il nativo non ha visto (ricaduta, tipo 8/9, errore) */
        if (s_need_full) {
            LARGE_INTEGER a, b, f; QueryPerformanceCounter(&a); sync_full_fast(l->ls110, raw110); QueryPerformanceCounter(&b); QueryPerformanceFrequency(&f);
            s_t_fullcopy += (double)(b.QuadPart - a.QuadPart) * 1e9 / (double)f.QuadPart; s_n_fullcopy++; c_full_forced++; s_need_full = 0;
        }
        else if (s_chk && ++s_chk_cnt >= s_chk) { s_chk_cnt = 0; sync_chk_blocks(l->ls110, raw110, "nosync"); }
    }
    else if (s_fast) {
        /* dopo l'init l'LS privata == bersaglio (tranne cio' che il motore ha scritto); i primi 64 ticket sono tutti completi (apprendimento) */
        const int full = (__atomic_fetch_add(&c_fast_full, 0, __ATOMIC_RELAXED) + c_fast_inc) < s_learn || (++s_cnt_full >= s_K);
        if (full) s_cnt_full = 0;
        sync_ls110_fast(l, raw110, full);
    }
    else if (s_fullsync) { LARGE_INTEGER a, b, f; QueryPerformanceCounter(&a); sync_full_fast(l->ls110, raw110); QueryPerformanceCounter(&b); QueryPerformanceFrequency(&f); s_t_fullcopy += (double)(b.QuadPart - a.QuadPart) * 1e9 / (double)f.QuadPart; s_n_fullcopy++; c_full_fast++; }
    else if (memcmp(l->ls110 + 0xC610, raw110 + 0xC610, 32) != 0 || memcmp(l->ls110 + 0x6DC0, raw110 + 0x6DC0, 16) != 0) {
        __atomic_fetch_add(&c_resync110, 1, __ATOMIC_RELAXED);
        l2_sync_ls(l->ls110, k_l2_bin110, sizeof k_l2_bin110, raw110);
    }
    const uint8_t* raw111;
    if (c_raw111 && l2_be32(c_raw111 + 0x13908) == 0x13A80u && l2_be32(c_raw111 + 0x13900) - 0x10000000u < 0x10000000u) raw111 = c_raw111;
    else { raw111 = sync_find_ls111(l2_be32(l->ls110 + 0x6DC4)); c_raw111 = raw111; }
    if (!raw111) { l->ls111_ok = 0; __atomic_fetch_add(&c_notready, 1, __ATOMIC_RELAXED); return 0; }
    if (!(l->ls111_ok && memcmp(l->ls111 + 0x13930, raw111 + 0x13930, 96) == 0)) {
        if (l->ls111_ok) __atomic_fetch_add(&c_resync111, 1, __ATOMIC_RELAXED);
        l2_sync_ls(l->ls111, k_l2_bin111, sizeof k_l2_bin111, raw111);
        l->ls111_ok = 1;
    }
    return 1;
}

/* esegue il ticket dello slot `rec` (record gia' preparato in memoria guest). 0 = ok (log pronto), altrimenti codice (8/9 non coperto, >=100 errore). */
int spu_coll_sync_run(uint32_t rec)
{
    L2* l = t_s;
    if (!l) return 100;
    if (rec < 0x10000u || (rec & 15u) || !sync_page_ok(rec, 160u)) return 104;
    l2_mem_rd = sync_rd_live; t_s_cur = l;
    u128 regs[128];
    memcpy(regs, k_sync_regs, 2048);
    regs[3]._u32[0] = rec;
    /* modo FP del thread raw SPU: RZ|DAZ|FTZ + g_spu_rz */
    const unsigned csr = _mm_getcsr(); const int rz = g_spu_rz;
    _mm_setcsr(0xFFC0u); g_spu_rz = 1;
    LARGE_INTEGER q0, q1, fq; QueryPerformanceFrequency(&fq); QueryPerformanceCounter(&q0);
    const int rc = l2_run_ticket(l, regs);
    QueryPerformanceCounter(&q1);
    _mm_setcsr(csr); g_spu_rz = rz;
    if (rc == 0) { s_t_pro += (double)l->q_pro * 1e9 / (double)fq.QuadPart; s_t_111 += (double)l->q_111 * 1e9 / (double)fq.QuadPart; { uint32_t e = l->n_entries; __atomic_fetch_add(&s_ent_hist[e == 0 ? 0 : e <= 2 ? 1 : e <= 4 ? 2 : e <= 8 ? 3 : e <= 16 ? 4 : e <= 32 ? 5 : e <= 64 ? 6 : 7], 1, __ATOMIC_RELAXED); } s_nw_sum += l->nw; s_t_get += (double)l->q_get * 1e9 / (double)fq.QuadPart; s_n_get += l->n_get; s_b_get += l->b_get; s_n_lsget += l->n_ls_get; }
    if (rc == 0) { s_ns_native += (double)(q1.QuadPart - q0.QuadPart) * 1e9 / (double)fq.QuadPart; __atomic_fetch_add(&s_ns_n, 1, __ATOMIC_RELAXED); }
    if (rc == 0) {
        if (l->nw == 0) { __atomic_fetch_add(&c_err, 1, __ATOMIC_RELAXED); return 105; }
        __atomic_fetch_add(&c_native_ok, 1, __ATOMIC_RELAXED);
        if (l->n_entries) { __atomic_fetch_add(&c_with111, 1, __ATOMIC_RELAXED); __atomic_fetch_add(&c_entries, (long)l->n_entries, __ATOMIC_RELAXED); }
        return 0;
    }
    if (rc == 8) __atomic_fetch_add(&c_ref8, 1, __ATOMIC_RELAXED);
    else if (rc == 9) __atomic_fetch_add(&c_ref9, 1, __ATOMIC_RELAXED);
    else { __atomic_fetch_add(&c_err, 1, __ATOMIC_RELAXED); __atomic_fetch_add(&c_err_k[(rc >= 100 ? rc - 100 : rc) & 15], 1, __ATOMIC_RELAXED); }
    return rc;
}

/* modo 1: applica il log nell'ordine del lifted (il quad del record con u16 +0x18 |= 0x8000 e' l'ULTIMO) */
void spu_coll_sync_apply(void)
{
    L2* l = t_s;
    LARGE_INTEGER qa0, qa1, qaf; QueryPerformanceCounter(&qa0);
    for (uint32_t k = 0; k < l->nw; k++) sync_put_guest(l->w[k].ea, l->wd + l->w[k].off, l->w[k].n);
    QueryPerformanceCounter(&qa1); QueryPerformanceFrequency(&qaf);
    s_t_apply += (double)(qa1.QuadPart - qa0.QuadPart) * 1e9 / (double)qaf.QuadPart; __atomic_fetch_add(&s_n_apply, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&c_applied, 1, __ATOMIC_RELAXED);
}

static void hexq(char* o, const uint8_t* p, uint32_t n) { for (uint32_t i = 0; i < n; i++) sprintf(o + 2 * i, "%02X", p[i]); }

/* modo 2: le raw SPU hanno chiuso il ticket; confronta quello che il nativo avrebbe scritto con la memoria */
int spu_coll_sync_compare(const uint8_t* rec_snap)
{
    L2* l = t_s;
    uint32_t cov[1024]; uint32_t ncov = 0;
    if (s_m1view > 0 && s_Lpost && g_raw110_cur) memcpy(s_Lpost, g_raw110_cur, L2_LS_SIZE);
    int first_bad = -1; uint32_t bad_ea = 0; uint32_t nbad = 0; int rel_hit = 0;
    for (int k = (int)l->nw - 1; k >= 0; k--) {
        const L2Wr* w = &l->w[k];
        for (uint32_t o = 0; o < w->n; o += 16u) {
            const uint32_t qea = (w->ea + o) & ~15u; const uint32_t len = (w->n - o) < 16u ? (w->n - o) : 16u;
            int seen = 0; for (uint32_t c = 0; c < ncov; c++) if (cov[c] == qea) { seen = 1; break; }
            if (seen) continue;
            if (ncov < 1024) cov[ncov++] = qea;
            if (memcmp(vm_base + w->ea + o, l->wd + w->off + o, len) != 0) {
                /* q89: differenza RILEVANTE = byte che il PPU legge: record +0..7, +0xE/+0xF, +0x18/+0x19 (bit15 escluso) */
                const uint32_t qa = w->ea + o;
                if (qa == l->rec_ea || qa == l->rec_ea + 0x10u) {
                    for (uint32_t bb = 0; bb < len; bb++) {
                        const uint32_t off = (qa - l->rec_ea) + bb;
                        if (vm_base[qa + bb] == l->wd[w->off + o + bb]) continue;
                        if (off < 8u || off == 0xEu || off == 0xFu || off == 0x19u || (off == 0x18u && ((vm_base[qa + bb] ^ l->wd[w->off + o + bb]) & 0x7Fu))) { rel_hit = 1; break; }
                    }
                }
                nbad++; if (first_bad < 0 || k > first_bad) { first_bad = k; bad_ea = w->ea + o; } }
        }
    }
    const int ty = ((rec_snap[12] << 8) | rec_snap[13]);
    if (ty >= 0 && ty < 3) __atomic_fetch_add(&c_ty[ty], 1, __ATOMIC_RELAXED);
    if (l->k12) { __atomic_fetch_add(&c_k12_n, 1, __ATOMIC_RELAXED); __atomic_fetch_add(nbad ? &c_k12_diff : &c_k12_eq, 1, __ATOMIC_RELAXED); }
    if (rel_hit) { __atomic_fetch_add(&c_rel, 1, __ATOMIC_RELAXED); if (ty >= 0 && ty < 3) __atomic_fetch_add(&c_rel_ty[ty], 1, __ATOMIC_RELAXED); }
    if (nbad == 0) { __atomic_fetch_add(&c_equal, 1, __ATOMIC_RELAXED); return 1; }
    __atomic_fetch_add(&c_diff, 1, __ATOMIC_RELAXED);
    if (ty >= 0 && ty < 3) __atomic_fetch_add(&c_ty_diff[ty], 1, __ATOMIC_RELAXED);
    if (first_bad >= 0) {
        const L2Wr* w = &l->w[first_bad]; uint32_t mask = 0;
        for (int b = 0; b < 16; b++) if (b < (int)w->n && vm_base[bad_ea + b] != l->wd[w->off + (bad_ea - w->ea) + b]) mask |= 1u << b;
        const uint32_t key = ((uint32_t)(ty & 3) << 30) | ((l->n_entries ? 1u : 0u) << 29) | ((uint32_t)first_bad << 24) | (mask << 0) | ((uint32_t)(l->nw & 15) << 16);
        int i = 0; for (; i < s_nsig; i++) if (s_sig[i].key == key) break;
        if (i == s_nsig && s_nsig < SIGN) { s_nsig++; s_sig[i].key = key; s_sig[i].n = 0; s_sig[i].lsa = w->lsa; memcpy(s_sig[i].nat, l->wd + w->off + (bad_ea - w->ea), 16); memcpy(s_sig[i].raw, vm_base + bad_ea, 16); }
        if (i < SIGN) s_sig[i].n++;
    }
    long nl = __atomic_add_fetch(&s_diff_logged, 1, __ATOMIC_RELAXED);
    if (nl <= 12) {
        char a[200], b[200];
        fprintf(stderr, "[coll-sync] DIFF #%ld rec=0x%08X tipo=%d entries=%u puts=%u quad diversi=%u primo_ea=0x%08X (put #%d)\n", nl, l->rec_ea, ty, l->n_entries, l->nw, nbad, bad_ea, first_bad);
        if (first_bad >= 0) {
            const L2Wr* w = &l->w[first_bad]; hexq(a, l->wd + w->off + (bad_ea - w->ea), 16u); hexq(b, vm_base + bad_ea, 16u);
            fprintf(stderr, "[coll-sync]   nativo=%s\n[coll-sync]   raw    =%s\n", a, b);
        }
        { char r[400]; hexq(r, rec_snap, 160u); fprintf(stderr, "[coll-sync]   record(ingresso)=%s\n", r); }
        for (uint32_t i = 0; i < l->nw && i < 12; i++) fprintf(stderr, "[coll-sync]   put %u: ea=0x%08X n=%u\n", i, l->w[i].ea, l->w[i].n);
    }
    return 0;
}

unsigned spu_coll_sync_waitms(void) { return s_waitms; }
void spu_coll_sync_note(int what)
{
    switch (what) {
    case 0: { long c = __atomic_add_fetch(&c_calls, 1, __ATOMIC_RELAXED); if ((c & 0xFFF) == 0) sync_print_stats("periodic"); break; }
    case 1: __atomic_fetch_add(&c_elig, 1, __ATOMIC_RELAXED); break;
    case 2: __atomic_fetch_add(&c_fallback, 1, __ATOMIC_RELAXED); s_need_full = 1; break;
    case 3: __atomic_fetch_add(&c_timeout, 1, __ATOMIC_RELAXED); s_need_full = 1; break;
    }
}

static void sync_print_stats(const char* why)
{
    if (why[0] == 'e') for (int i = 0; i < s_nsig; i++) { char a[40], b[40]; hexq(a, s_sig[i].nat, 16); hexq(b, s_sig[i].raw, 16);
        fprintf(stderr, "[coll-sync]   firma %d: tipo %u entries>0=%u put#%u puts=%u mascia byte=%04X lsa=0x%05X n=%u es.nat=%s raw=%s\n", i, s_sig[i].key >> 30, (s_sig[i].key >> 29) & 1, (s_sig[i].key >> 24) & 31, (s_sig[i].key >> 16) & 15, s_sig[i].key & 0xFFFF, s_sig[i].lsa, s_sig[i].n, a, b); }
    long tot = c_equal + c_diff;
    fprintf(stderr, "[coll-sync]   k12 tollerati: %ld (uguali %ld diversi %ld)\n", c_k12_n, c_k12_eq, c_k12_diff);
    fprintf(stderr, "[coll-sync] stats (%s) mode=%d: chiamate 4A3288 %ld, idonee(tipo 0..2) %ld, non pronto %ld; nativo ok %ld (con 111: %ld, voci %ld), rifiutati 8/9: %ld/%ld, errori %ld, ricaduti sulle raw SPU %ld; "
            "applicati(modo1) %ld; confronti(modo2) %ld: UGUALI %ld DIVERSI %ld (t0 %ld/%ld t1 %ld/%ld t2 %ld/%ld) timeout %ld; resync110 %ld resync111 %ld; nativo medio %.1f us\n",
            why, s_mode, c_calls, c_elig, c_notready, c_native_ok, c_with111, c_entries, c_ref8, c_ref9, c_err, c_fallback, c_applied, tot, c_equal, c_diff,
            c_ty_diff[0], c_ty[0], c_ty_diff[1], c_ty[1], c_ty_diff[2], c_ty[2], c_timeout, c_resync110, c_resync111, s_ns_n ? s_ns_native / (double)s_ns_n / 1000.0 : 0.0);
    if (s_ns_n) fprintf(stderr, "[coll-sync]   fasi (us/ticket): prepare %.2f (n=%ld) | stadio110 %.2f | stadio111 %.2f | apply %.2f (n=%ld) | PUT/ticket %.1f | voci hist [0,1-2,3-4,5-8,9-16,17-32,33-64,65+] = %ld %ld %ld %ld %ld %ld %ld %ld\n", s_n_prep ? s_t_prep / s_n_prep / 1000.0 : 0, s_n_prep, s_t_pro / s_ns_n / 1000.0, s_t_111 / s_ns_n / 1000.0, s_n_apply ? s_t_apply / s_n_apply / 1000.0 : 0, s_n_apply, (double)s_nw_sum / s_ns_n, s_ent_hist[0], s_ent_hist[1], s_ent_hist[2], s_ent_hist[3], s_ent_hist[4], s_ent_hist[5], s_ent_hist[6], s_ent_hist[7]);
    if (s_ns_n) fprintf(stderr, "[coll-sync]   GET memoria/ticket %.1f (%.0f B) in %.2f us; GET da LS(alias)/ticket %.1f\n", (double)s_n_get / s_ns_n, (double)s_b_get / s_ns_n, s_t_get / s_ns_n / 1000.0, (double)s_n_lsget / s_ns_n);
    fprintf(stderr, "[coll-sync]   DIFF RILEVANTI (byte letti dal PPU) %ld su %ld confronti (t0 %ld t1 %ld t2 %ld)\n", c_rel, tot, c_rel_ty[0], c_rel_ty[1], c_rel_ty[2]);
    if (s_fast > 0) fprintf(stderr, "[coll-sync]   FAST: sync incrementali %ld, completi %ld (K=%u), blocchi in S %ld, scoperte tardive %ld\n", c_fast_inc, c_fast_full, s_K, c_Sn, c_late);
    if (s_m1view > 0) fprintf(stderr, "[coll-sync]   M1VIEW attivo: blocchi aggiornati da scritture esterne (PPU) %ld\n", c_m1_upd);
    if (s_n_fullcopy) fprintf(stderr, "[coll-sync]   FULL copy veloce: %ld copie (fast %ld, forzate %ld), %.2f us/copia; nosync=%d chk=%u\n", s_n_fullcopy, c_full_fast, c_full_forced, s_t_fullcopy / s_n_fullcopy / 1000.0, s_nosync, s_chk);
    if (why[0] == 'e' || s_chk) sync_print_blkhist();
    if (c_err) { fprintf(stderr, "[coll-sync]   errori per codice:"); for (int i = 0; i < 16; i++) if (c_err_k[i]) fprintf(stderr, " k%d=%ld", i, c_err_k[i]); fprintf(stderr, "\n"); }
    fflush(stderr);
}
void spu_coll_sync_stats(void) { sync_print_stats("manual"); }

/* ring103: pool nativo dei ticket ad anello (SPU_COLL_RING) */
#include "spu_coll_ring.inc"
