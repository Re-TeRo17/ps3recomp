/* anim_hle.c (anim47) -- aggancio nativo del comando di animazione wid0 (funzione SPU 0x6248 di jobmod_00b730f78d78).
 *
 * SPU_HLE_WID0 = 0 spento, 1 attivo (DEFAULT dalla 0.5, anim80), 2 doppia modalita' (il lifted fa il lavoro vero, il nativo gira su una COPIA dell'LS e si
 * confrontano le matrici in uscita e l'LS: conta le differenze e logga le prime).
 *
 * Entra da anim_cap_run (capture_patch.c), chiamata dal gancio SPU_ANIM_CAP_HOOK all'inizio di func_00006248 / _p. Ritorna 1 se il
 * comando e' stato eseguito qui (il gancio fa return), 0 se il lifted deve correre.
 *
 * Modalita' 1, in ordine:
 *   1. controllo preventivo (nessun effetto): r3 == 0x7A80, nessun evento/interrupt pendente (il lifted lo consegnerebbe al primo
 *      confine di finestra), nessuna diagnostica armata, nessun GET che possa toccare la contabilita' degli overlay;
 *   2. calcolo nativo (anim_cmd_compute) in place sull'LS reale con MXCSR = RZ|DAZ|FTZ (gli helper float di anim_ref.c sono le
 *      operazioni host); tutti i GET sono memcpy validati (DMA ben formato, intervallo EA committato). Se rifiuta (rc != OK) la
 *      memoria principale NON e' stata toccata e il lifted rifa' il comando da capo;
 *   3. commit con i canali MFC del RUNTIME, nello stesso ordine del lifted: GETLLAR/PUTLLC della linea coda (+1 mod 128 a linea+10,
 *      con ripetizione se la prenotazione e' persa), PUT delle matrici a blocchi da 16 KB, PUTF del flag di completamento, attesa
 *      del tag. Cosi' coerenza PPU (bitmap di prenotazione, notify), timeline, tag e eventi sono quelli di sempre.
 *   4. ritorno al chiamante come il lifted: r0/r1/r80-r91 invariati, pc = r0, int_enable = 1, r3 = stato del tag. */
#define _CRT_SECURE_NO_WARNINGS
#include "anim_ref.h"
#include "capture_patch.h"
#include "spu_helpers.h"
#include "spu_context.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <intrin.h>
#include <immintrin.h>
#include <windows.h>

extern uint8_t* vm_base;
extern int g_spu_pcheck;
extern uint32_t g_spu_watchea_dyn, g_barrier_sync_watch;
extern SPU_THREAD_LOCAL int g_spu_rz;
u128 spu_rdch(spu_context* ctx, uint32_t channel);
int spu_hle_ea_ok(uint32_t ea, uint32_t size);
int spu_hle_get_note_mode(const spu_context* spu, uint32_t lsa_lo, uint32_t lsa_hi);
void spu_hle_note_get(spu_context* spu, uint32_t ea, uint32_t lsa, uint32_t size);

int g_anim_hle_mode = 0;
static int s_inited;

/* ------------------------------------------------------------------ contatori */
enum { R_OK = 0, R_R3, R_PEND, R_DIAG, R_NOTE, R_CNC, R_COVF, R_OTHER, R_N };
static const char* const R_NAME[R_N] = { "ok", "r3!=7A80", "evento/irq pendente", "diagnostica armata", "overlay-note", "non coperto", "arena/heqi", "altro" };
static volatile long long s_cnt[R_N];
static volatile long long s_cyc_compute, s_cyc_commit, s_nnative;
static volatile long long s_dual_n, s_dual_declined, s_dual_putdiff, s_dual_lsdiff, s_dual_linediff, s_dual_ok, s_dual_ppuw, s_dual_ppuw_max;
static volatile LONG s_report_tick;
static double s_cyc_per_ns = 0.0;
/* doppia modalita': tempo del lifted e del nativo (RZ) sugli STESSI comandi, per classe di N (<=4, <=16, <=64, <=128, <=400, >400) */
static volatile long long s_dl_n[6], s_dl_lift[6], s_dl_nat[6];
static int nbucket(uint32_t n) { return n <= 4 ? 0 : n <= 16 ? 1 : n <= 64 ? 2 : n <= 128 ? 3 : n <= 400 ? 4 : 5; }

#define INC(v) __atomic_fetch_add(&(v), 1, __ATOMIC_RELAXED)
#define ADD(v, x) __atomic_fetch_add(&(v), (x), __ATOMIC_RELAXED)

static __declspec(thread) int t_in;
static __declspec(thread) spu_context* t_note_ctx;   /* mode 1: ctx per spu_overlay_note_get dopo ogni GET nativo (solo se serve) */
static void note_cb(uint32_t ea, uint32_t lsa, uint32_t size) { spu_hle_note_get(t_note_ctx, ea, lsa, size); }
static CRITICAL_SECTION s_cs;

/* ---- anim55: strumentazione (ANIM_HLE_DIAG=1; ANIM_HLE_WATCH=1 aggiunge un thread che stampa lo stato ogni secondo) ----
 * Serve a capire perche' (b39_a47q B1/B2) il gioco si impianta con SPU_HLE_WID0=1: stato corrente del commit, tentativi di
 * PUTLLC falliti per motivo, comandi ripetuti (stesso descrittore due volte di fila), ultimi comandi in un anello. */
static int s_diag;
static volatile long long s_st_state, s_st_t0;                 /* 0 fuori, 1 calcolo, 2 commit: linea coda, 3 commit: DMA/attesa tag */
static volatile long long s_cm_n, s_cm_retry, s_cm_retry_max, s_cm_fail_noresv, s_cm_fail_ea, s_cm_fail_line, s_cm_spin_max;
static volatile long long s_cm_cyc_max, s_cp_cyc_max, s_rep_same, s_rep_cnt;
static volatile uint32_t s_last_cmd_ea, s_last_line_ea, s_last_cnt;
static spu_context* volatile s_last_ctx;
typedef struct { uint32_t cmd_ea, cnt_ls, w1, w0, retries, flag_after, n; unsigned long long tsc; int rc; } ring_ent;
static ring_ent s_ring[64]; static volatile long s_ring_pos;
static uint8_t s_last_desc[32];
static __declspec(thread) uint32_t t_retries;

void anim_hle_init(void)
{
    if (s_inited) return;
    s_inited = 1;
    const char* e = getenv("SPU_HLE_WID0");
    int m = e ? atoi(e) : 1;   /* anim80: default ATTIVO (0.5); SPU_HLE_WID0=0 lo spegne */
    if (m < 0 || m > 2) m = 0;
    g_anim_hle_mode = m;
    { /* perf99: percorsi veloci RZ del calcolo (f4D70 / f5160); ANIM_FAST=0 li spegne (corpo originale) */
      extern int g_f4d70_off, g_f5160_off; const char* f = getenv("ANIM_FAST"); if (f && f[0] == '0') { g_f4d70_off = 1; g_f5160_off = 1; }
      fprintf(stderr, "[anim-hle] percorsi veloci f4D70/f5160: %s\n", (f && f[0] == '0') ? "OFF" : "ON"); }
    if (!m) return;
    InitializeCriticalSection(&s_cs);
    { const char* d = getenv("ANIM_HLE_DIAG"); const char* w = getenv("ANIM_HLE_WATCH"); s_diag = (d && d[0] == '1') || (w && w[0] == '1');
      if (w && w[0] == '1') { extern void anim_hle_watch_start(void); anim_hle_watch_start(); } }
    g_anim_strict = 1;
    g_anim_ea_ok = spu_hle_ea_ok;
    LARGE_INTEGER f; QueryPerformanceFrequency(&f);
    unsigned long long t0 = __rdtsc(); LARGE_INTEGER a, b; QueryPerformanceCounter(&a);
    do { QueryPerformanceCounter(&b); } while ((b.QuadPart - a.QuadPart) * 1000 < f.QuadPart * 20);   /* 20 ms */
    unsigned long long t1 = __rdtsc();
    s_cyc_per_ns = (double)(t1 - t0) / ((double)(b.QuadPart - a.QuadPart) * 1e9 / (double)f.QuadPart);
    fprintf(stderr, "[anim-hle] SPU_HLE_WID0=%d (%s) tsc %.3f GHz\n", m, m == 1 ? "attivo" : "doppia modalita'", s_cyc_per_ns);
}

static void hle_report(void)
{
    long long n = s_nnative;
    char b[900]; int o = 0;
    o += snprintf(b + o, sizeof b - o, "[anim-hle] mode %d: nativi %lld", g_anim_hle_mode, n);
    for (int i = 1; i < R_N; i++) if (s_cnt[i]) o += snprintf(b + o, sizeof b - o, ", rifiutati[%s] %lld", R_NAME[i], s_cnt[i]);
    if (n) o += snprintf(b + o, sizeof b - o, " | ns/cmd calcolo %.0f commit %.0f", s_cyc_compute / (double)n / s_cyc_per_ns, s_cyc_commit / (double)n / s_cyc_per_ns);
    if (s_dual_n) o += snprintf(b + o, sizeof b - o, " | DUAL: confrontati %lld uguali %lld rifiutati dal nativo %lld DIFF matrici %lld LS %lld linea(info) %lld riscritture-PPU(benigne) %lld (max matrici %lld)",
                                s_dual_n, s_dual_ok, s_dual_declined, s_dual_putdiff, s_dual_lsdiff, s_dual_linediff, s_dual_ppuw, s_dual_ppuw_max);
    fprintf(stderr, "%s\n", b);
    if (s_dual_n) {
        static const char* const bn[6] = { "N<=4", "<=16", "<=64", "<=128", "<=400", ">400" };
        char t[500]; int p = snprintf(t, sizeof t, "[anim-hle] dual ns/cmd lifted|nativoRZ per classe N:");
        long long tl = 0, tn = 0, tc = 0;
        for (int i = 0; i < 6; i++) if (s_dl_n[i]) { p += snprintf(t + p, sizeof t - p, " %s %.0f|%.0f (n=%lld)", bn[i], s_dl_lift[i] / (double)s_dl_n[i] / s_cyc_per_ns, s_dl_nat[i] / (double)s_dl_n[i] / s_cyc_per_ns, s_dl_n[i]); tl += s_dl_lift[i]; tn += s_dl_nat[i]; tc += s_dl_n[i]; }
        if (tc) p += snprintf(t + p, sizeof t - p, " | media %.0f|%.0f", tl / (double)tc / s_cyc_per_ns, tn / (double)tc / s_cyc_per_ns);
        fprintf(stderr, "%s\n", t);
    }
}

/* ------------------------------------------------------------------ utilita' */
static u128 V32(uint32_t v) { u128 r = spu_zero(); r._u32[0] = v; return r; }
static uint32_t LW_(const uint8_t* ls, uint32_t a) { return ((uint32_t)ls[a] << 24) | ((uint32_t)ls[a + 1] << 16) | ((uint32_t)ls[a + 2] << 8) | ls[a + 3]; }

/* Gate preventivo. Ritorna 0 = ok, altrimenti il motivo (R_*). Nessun effetto. */
static int hle_gate(spu_context* c)
{
    extern int g_spu_hop_slow; extern volatile int g_yz_lockstep_on; extern unsigned g_spu_stack_reset_count; extern void* volatile g_pm_flow_ctx;
    if (g_spu_pcheck || g_spu_watchea_dyn || g_barrier_sync_watch || g_spu_hop_slow != 0 || g_yz_lockstep_on || g_spu_stack_reset_count ||
        g_pm_flow_ctx == (void*)c) return R_DIAG;
    if ((c->event_status & c->event_mask) || c->irq_saved) return R_PEND;
    int nm = spu_hle_get_note_mode(c, 0x7480, 0x34B00);
    if (nm == 1) return R_NOTE;
    t_note_ctx = nm == 2 ? c : 0;
    return R_OK;
}

/* calcolo con MXCSR RZ|DAZ|FTZ. Ritorna il codice di anim_cmd_compute. */
static int hle_compute(const uint8_t* vm, uint32_t cmd_ea, uint8_t* ls, anim_commit* cm)
{
    const unsigned old = _mm_getcsr();
    const int old_rz = g_spu_rz;
    _mm_setcsr(0xFFC0u); g_spu_rz = 1;
    int rc = anim_cmd_compute(vm, cmd_ea, ls, cm);
    g_spu_rz = old_rz; _mm_setcsr(old);
    return rc;
}

static void wr_dma(spu_context* c, uint32_t lsa, uint32_t ea, uint32_t size, uint32_t tag, uint32_t cmd)
{
    spu_wrch(c, MFC_LSA, V32(lsa));
    spu_wrch(c, MFC_EAL, V32(ea));
    spu_wrch(c, MFC_Size, V32(size));
    spu_wrch(c, MFC_TagID, V32(tag));
    spu_wrch(c, MFC_Cmd, V32(cmd));
}

/* commit: come 0x6508..0x65F0 (linea coda), 0x6F20 (PUT a blocchi da 16 KB), 0x6890 (PUTF), 0x68D0 (attesa). Ritorna RdTagStat. */
static uint32_t hle_commit(spu_context* c, const anim_commit* cm)
{
    uint8_t* ls = c->ls;
    uint32_t retries = 0;
    if (s_diag) { s_st_state = 2; s_st_t0 = __rdtsc(); }
    for (;;) {
        wr_dma(c, 0x7A00, cm->line_ea, 128, 0, 208);                           /* GETLLAR */
        if (spu_rdch(c, MFC_RdAtomicStat)._u32[0] == 0) goto backoff;
        {   /* halfword a linea+10: (v + 1) & 127 */
            uint32_t v = (((uint32_t)ls[0x7A0A] << 8) | ls[0x7A0B]) + 1u;
            v &= 127u; ls[0x7A0A] = (uint8_t)(v >> 8); ls[0x7A0B] = (uint8_t)v;
        }
        wr_dma(c, 0x7A00, cm->line_ea, 128, 0, 180);                           /* PUTLLC */
        if (spu_rdch(c, MFC_RdAtomicStat)._u32[0] == 0) break;
        if (s_diag) {
            if (!c->resv_valid) INC(s_cm_fail_noresv); else if ((c->resv_ea & ~127u) != (cm->line_ea & ~127u)) INC(s_cm_fail_ea); else INC(s_cm_fail_line);
        }
    backoff:
        retries++;
        for (int i = 0; i < 50; i++) _mm_pause();                              /* il lifted fa un ciclo di attesa di 50 giri */
    }
    t_retries = retries;
    if (s_diag) { s_st_state = 3; if (retries) { ADD(s_cm_retry, retries); if ((long long)retries > s_cm_retry_max) s_cm_retry_max = retries; } }
    {   uint32_t size = cm->n * 64u, lsa = cm->mw, ea = cm->ea_out;
        do {
            uint32_t ch = size > 16384u ? 16384u : size;
            wr_dma(c, lsa, ea, ch, cm->tp, 32);
            size -= ch; lsa += ch; ea += ch;
        } while (size);
    }
    wr_dma(c, 0x7A90, cm->cmd_ea + 16, 4, cm->tp, 34);                         /* PUTF */
    spu_wrch(c, MFC_WrTagMask, V32(1u << (cm->tp & 31)));
    spu_wrch(c, MFC_WrTagUpdate, V32(2));
    uint32_t r = spu_rdch(c, MFC_RdTagStat)._u32[0];
    if (s_diag) s_st_state = 0;
    return r;
}

static void hle_return(spu_context* c, uint32_t tagstat)
{
    c->gpr[3] = V32(tagstat);
    c->int_enable = 1;
    c->pc = c->gpr[0]._u32[0];                    /* SPU_RET */
    if (c->host_depth == 0) g_spu_trampoline_fn = spu_indirect_branch;
}


/* thread osservatore (ANIM_HLE_WATCH=1): una riga al secondo con lo stato del commit e della coda; piu' righe di dettaglio se non
 * passano comandi nativi da un secondo (gioco fermo o wid0 in attesa del produttore). */
static DWORD WINAPI watch_thread(LPVOID unused)
{
    (void)unused;
    long long last_n = -1; int quiet_ticks = 0;
    for (;;) {
        Sleep(1000);
        long long n = s_nnative;
        spu_context* c = s_last_ctx;
        const int idle = (n == last_n); last_n = n;
        quiet_ticks = idle ? quiet_ticks + 1 : 0;
        if (!c) continue;
        const uint8_t* ls = c->ls;
        uint32_t line_ea = LW_(ls, 0x34BA0), base = LW_(ls, 0x34BB0), cnt = LW_(ls, 0x34B90);
        const uint8_t* ln = line_ea && vm_base ? vm_base + line_ea : 0;
        const unsigned long long now = __rdtsc();
        const double cps = s_cyc_per_ns > 0 ? s_cyc_per_ns : 3.0;
        fprintf(stderr, "[anim-watch] nat=%lld%s state=%lld (%.1f ms in state) retry=%lld max=%lld failNoResv=%lld failEa=%lld failLine=%lld sameDesc=%lld cmpMax=%.0fus cmtMax=%.0fus | LS cnt34B90=%u base=%08X line=%08X",
                n, idle ? " IDLE" : "", s_st_state, s_st_state ? (now - (unsigned long long)s_st_t0) / cps / 1e6 : 0.0, s_cm_retry, s_cm_retry_max,
                s_cm_fail_noresv, s_cm_fail_ea, s_cm_fail_line, s_rep_same, s_cp_cyc_max / cps / 1e3, s_cm_cyc_max / cps / 1e3, cnt, base, line_ea);
        if (ln) fprintf(stderr, " | VM line: w0=%08X w1=%08X w2=%08X w3=%08X +10=%02X%02X", LW_(ln, 0), LW_(ln, 4), LW_(ln, 8), LW_(ln, 12), ln[10], ln[11]);
        fputc(10, stderr);
        if (idle && quiet_ticks >= 2 && (quiet_ticks % 3) == 2) {
            /* anello degli ultimi comandi + flag attuali dei descrittori (PPU: attende il bit0 del flag di completamento) */
            long pos = s_ring_pos;
            fprintf(stderr, "[anim-watch] ultimi comandi (piu' recente per ultimo):%c", 10);
            for (int k = 12; k >= 1; k--) {
                const ring_ent* r = &s_ring[(pos - k) & 63];
                if (!r->cmd_ea) continue;
                uint32_t fl_now = vm_base ? LW_(vm_base, r->cmd_ea + 16) : 0;
                fprintf(stderr, "   cmd_ea=%08X cnt=%u line.w0=%08X line.w1=%08X N=%u retries=%u rc=%d flag(dopo)=%08X flag(ora)=%08X t=%.1fs%c", r->cmd_ea, r->cnt_ls, r->w0, r->w1, r->n, r->retries,
                        r->rc, r->flag_after, fl_now, (now - r->tsc) / cps / 1e9, 10);
            }
            if (ln && base && vm_base) {
                fprintf(stderr, "[anim-watch] descrittori intorno al contatore (EA base+32*k, flag +16):");
                for (int k = (int)cnt - 3; k <= (int)cnt + 3; k++) {
                    if (k < 0) continue;
                    uint32_t e = base + 32u * (uint32_t)k;
                    fprintf(stderr, " [%d]%08X:w0=%08X w4=%08X", k, e, LW_(vm_base, e), LW_(vm_base, e + 16));
                }
                fputc(10, stderr);
            }
        }
        fflush(stderr);
    }
    return 0;
}
void anim_hle_watch_start(void) { HANDLE h = CreateThread(0, 0, watch_thread, 0, 0, 0); if (h) CloseHandle(h); }

/* ------------------------------------------------------------------ doppia modalita' */
static uint8_t s_lsc[0x40000];
static uint8_t s_lsc2[0x40000], s_lsc3[0x40000];
static int cmp_range(const uint8_t* a, const uint8_t* b, uint32_t lo, uint32_t hi, uint32_t* first)
{
    int n = 0;
    for (uint32_t q = lo & ~15u; q < hi; q += 16) {
        if (memcmp(a + q, b + q, 16) == 0) continue;
        if (!n) *first = q;
        n++;
    }
    return n;
}

static int hle_dual(spu_context* c, void (*self)(spu_context*), uint32_t cmd_ea)
{
    EnterCriticalSection(&s_cs);
    memcpy(s_lsc, c->ls, 0x40000);
    memcpy(s_lsc2, c->ls, 0x40000);
    anim_commit cm;
    const uint32_t sp = c->gpr[1]._u32[0];
    const uint32_t lr = c->gpr[0]._u32[0];
    const unsigned long long tn0 = __rdtsc();
    int rc = hle_compute(vm_base, cmd_ea, s_lsc, &cm);
    const unsigned long long tn1 = __rdtsc();
    uint32_t line_ea = LW_(c->ls, 0x34BA0);
    const uint8_t pre_cnt_hi = rc == ANIM_OK ? vm_base[line_ea + 10] : 0, pre_cnt_lo = rc == ANIM_OK ? vm_base[line_ea + 11] : 0;
    /* anim68: in dual mode con ANIM_CAPTURE aperto, istantanea pre-comando (solo N>=200) da scrivere se il lifted differisce dal nativo */
    anim_cap_rec* cap68 = 0;
    if (getenv("ANIM_CAPTURE")) { uint32_t eb = ((uint32_t)c->ls[0x7A84] << 24) | ((uint32_t)c->ls[0x7A85] << 16) | ((uint32_t)c->ls[0x7A86] << 8) | c->ls[0x7A87];
        int n68 = (int16_t)(((uint32_t)vm_base[eb] << 8) | vm_base[eb + 1]); if (n68 >= 200) cap68 = anim_cap_begin(c->ls, vm_base, 0x7A80, cmd_ea); }
    const unsigned long long tl0 = __rdtsc();
    t_in = 1; self(c);
    if (g_spu_trampoline_fn) spu_drain_call(c, lr);
    t_in = 0;
    const unsigned long long tl1 = __rdtsc();
    static uint8_t s_snap68[4000 * 64 + 64]; uint32_t snap68_n = 0, snap68_ea = 0;
    if (rc == ANIM_OK && cm.n <= 4000) { snap68_n = cm.n; snap68_ea = cm.ea_out; memcpy(s_snap68, vm_base + cm.ea_out, (size_t)cm.n * 64); }  /* anim68: foto dell'uscita subito dopo il lifted */
    INC(s_dual_n);
    if (rc == ANIM_OK) { int b = nbucket(cm.n); INC(s_dl_n[b]); ADD(s_dl_lift[b], (long long)(tl1 - tl0)); ADD(s_dl_nat[b], (long long)(tn1 - tn0)); }
    if (rc != ANIM_OK) { INC(s_dual_declined); INC(s_cnt[rc == ANIM_ARENA_OVF ? R_COVF : R_CNC]); LeaveCriticalSection(&s_cs); return 1; }
    int bad = 0;
    /* anim80: la memoria differisce dal nativo ma la LS del lifted (da cui parte il PUT) == LS del nativo su tutte le matrici, e la matrice 0 e' uguale
       => il PPU ha riscritto in memoria (giunto 4 e discendenti) dopo il PUT/flag: benigno, il PPU lo fara' uguale con il percorso nativo. */
    int ppuw = 0;
    if (memcmp(vm_base + cm.ea_out, s_lsc + cm.mw, (size_t)cm.n * 64) != 0 && cm.n >= 2) {
        uint32_t n_ls = 0, n_mem = 0, first_mem = 0xFFFFFFFFu;
        for (uint32_t m = 0; m < cm.n; m++) {
            if (memcmp(c->ls + cm.mw + 64u * m, s_lsc + cm.mw + 64u * m, 64)) n_ls++;
            if (memcmp(vm_base + cm.ea_out + 64u * m, s_lsc + cm.mw + 64u * m, 64)) { if (first_mem == 0xFFFFFFFFu) first_mem = m; n_mem++; }
        }
        if (n_ls == 0 && first_mem > 0 && n_mem < cm.n) {
            ppuw = 1; INC(s_dual_ppuw); if ((long long)n_mem > s_dual_ppuw_max) s_dual_ppuw_max = n_mem;
            static int shownp; if (shownp < 12) { shownp++; fprintf(stderr, "[anim-hle] PPUW riscrittura PPU (benigna): %u/%u matrici in memoria != nativo, prima m%u, LS-lifted==LS-nativo, snapshot-dopo-lifted==LS-lifted: %s\n", n_mem, cm.n, first_mem, memcmp(s_snap68, c->ls + cm.mw, (size_t)snap68_n * 64) == 0 ? "SI" : "NO (riscritta gia' prima)"); }
        }
    }
    /* matrici in uscita */
    if (!ppuw && memcmp(vm_base + cm.ea_out, s_lsc + cm.mw, (size_t)cm.n * 64) != 0) {
        INC(s_dual_putdiff); bad = 1;
        static int shown;
        /* integ65: dettaglio (quante matrici, quali, e se la memoria cambia ancora = PPU che riscrive dopo il flag) */
        if (shown < 12) {
            uint32_t nm = 0, idx[8]; uint8_t snap[64];
            for (uint32_t m = 0; m < cm.n; m++) if (memcmp(vm_base + cm.ea_out + 64u * m, s_lsc + cm.mw + 64u * m, 64)) { if (nm < 8) idx[nm] = m; nm++; }
            memcpy(snap, vm_base + cm.ea_out + 64u * (nm ? idx[0] : 0), 64);
            for (volatile int w = 0; w < 200000; w++) _mm_pause();
            int still = memcmp(snap, vm_base + cm.ea_out + 64u * (nm ? idx[0] : 0), 64) != 0;
            if (nm) {
                const uint8_t* L = snap; const uint8_t* Nt = s_lsc + cm.mw + 64u * idx[0];
                fprintf(stderr, "[anim-hle] DIFF m%u lifted:", idx[0]);
                for (int w = 0; w < 16; w++) fprintf(stderr, " %08X", LW_(L, 4 * w));
                fprintf(stderr, "\n[anim-hle] DIFF m%u nativo:", idx[0]);
                for (int w = 0; w < 16; w++) fprintf(stderr, " %08X", LW_(Nt, 4 * w));
                fputc(10, stderr);
            }
            { anim_commit cm2; memcpy(s_lsc3, s_lsc2, 0x40000); int rc2 = hle_compute(vm_base, cmd_ea, s_lsc3, &cm2);
              int same2 = rc2 == ANIM_OK && cm2.n == cm.n && memcmp(vm_base + cm.ea_out, s_lsc3 + cm2.mw, (size_t)cm.n * 64) == 0;
              fprintf(stderr, "[anim-hle] DIFF ricalcolo nativo ora (input riletti): %s\n", same2 ? "UGUALE al lifted = CORSA sugli input" : "ANCORA DIVERSO = bug nativo"); }
            if (snap68_n) { uint32_t nl2 = 0, nn2 = 0, i0 = 0xFFFFFFFFu; for (uint32_t m = 0; m < snap68_n; m++) { int dl = memcmp(s_snap68 + 64u * m, c->ls + cm.mw + 64u * m, 64) != 0; int dn = memcmp(s_snap68 + 64u * m, s_lsc + cm.mw + 64u * m, 64) != 0; if (dl) { if (!nl2) i0 = m; nl2++; } nn2 += dn; }
              fprintf(stderr, "[anim-hle] DIFF68 foto subito dopo il lifted: vs LS-lifted %u matrici diverse (prima %d), vs LS-nativo %u; LS-lifted==LS-nativo (MW): %s", nl2, (int)i0, nn2, memcmp(c->ls + cm.mw, s_lsc + cm.mw, (size_t)snap68_n * 64) == 0 ? "SI" : "NO"); fputc(10, stderr);
              if (nl2) { fprintf(stderr, "[anim-hle] DIFF68 m%u memoria:", i0); for (int w = 0; w < 16; w++) fprintf(stderr, " %08X", LW_(s_snap68 + 64u * i0, 4 * w)); fprintf(stderr, "  LS-lifted:"); for (int w = 0; w < 16; w++) fprintf(stderr, " %08X", LW_(c->ls + cm.mw + 64u * i0, 4 * w)); fputc(10, stderr); } }
            fprintf(stderr, "[anim-hle] DIFF dettaglio: %u matrici su %u diverse, prime:", nm, cm.n);
            for (uint32_t k = 0; k < nm && k < 8; k++) fprintf(stderr, " %u", idx[k]);
            fprintf(stderr, " | la memoria della prima cambia ancora dopo il confronto: %s\n", still ? "SI (riscritta da altri)" : "no");
        }
        if (cap68) { anim_cap_end(cap68, c->ls, vm_base); fprintf(stderr, "[anim-hle] DIFF catturata (ANIM_CAPTURE)\n"); }
        if (shown < 6) { shown++; uint32_t o = 0; while (o < cm.n * 64 && vm_base[cm.ea_out + o] == s_lsc[cm.mw + o]) o++;
            fprintf(stderr, "[anim-hle] DIFF matrici cmd_ea=%08X ea_out=%08X N=%u: primo byte diverso +%u (lifted %02X nativo %02X)\n", cmd_ea, cm.ea_out, cm.n, o, vm_base[cm.ea_out + o], s_lsc[cm.mw + o]); }
    }
    /* LS: tutto tranne la linea coda (0x7A00-0x7A7F, corsa col PPU), lo stack del lifted, e le word di garbage dei contatori 0x34B70/80 */
    {
        uint32_t first = 0, nd = 0, nd1;
        uint32_t sp_lo = (sp - 0x1800u) & ~15u, sp_hi = (sp + 0x40u + 15u) & ~15u;   /* i frame dei callee del lifted stanno sotto sp */
        static const uint32_t ranges[][2] = { { 0x4000u, 0x7A00u }, { 0x7A80u, 0x34B00u }, { 0x34B00u, 0x34B70u }, { 0x34B90u, 0x40000u } };
        for (int r = 0; r < 4; r++) {
            uint32_t lo = ranges[r][0], hi = ranges[r][1];
            /* spezza intorno allo stack */
            if (sp_hi <= lo || sp_lo >= hi) { nd1 = cmp_range(c->ls, s_lsc, lo, hi, &first); }
            else { nd1 = 0; if (sp_lo > lo) nd1 += cmp_range(c->ls, s_lsc, lo, sp_lo, &first); if (sp_hi < hi) nd1 += cmp_range(c->ls, s_lsc, sp_hi, hi, &first); }
            if (nd1 && !nd) {
                static int shown; if (shown < 6) { shown++;
                    uint32_t a = first, k = 0; while (k < 16 && c->ls[a + k] == s_lsc[a + k]) k++;
                    fprintf(stderr, "[anim-hle] DIFF LS cmd_ea=%08X sp=%05X: primo quad diverso a LS %05X (+%u: lifted %02X nativo %02X), quad diversi in questa zona: %u; elenco:", cmd_ea, sp, a, k, c->ls[a + k], s_lsc[a + k], nd1);
                    { int cnt = 0; for (uint32_t q = ranges[r][0] & ~15u; q < ranges[r][1] && cnt < 40; q += 16) if (memcmp(c->ls + q, s_lsc + q, 16)) { fprintf(stderr, " %05X", q); cnt++; } }
                    fprintf(stderr, "\n"); }
            }
            nd += nd1;
        }
        /* contatori 0x34B70 / 0x34B80: solo word0 */
        int ncnt = 0;
        if (memcmp(c->ls + 0x34B70, s_lsc + 0x34B70, 4)) { nd++; ncnt |= 1; }
        if (memcmp(c->ls + 0x34B80, s_lsc + 0x34B80, 4)) { nd++; ncnt |= 2; }
        if (nd) { INC(s_dual_lsdiff); bad = 1;
            static int shown2; if (shown2 < 6) { shown2++;
                fprintf(stderr, "[anim-hle] DIFF LS (riepilogo) cmd_ea=%08X nd=%u contatori=%d: 34B70 lifted %08X nativo %08X; 34B80 lifted %08X nativo %08X; 34B90 lifted %08X nativo %08X\n", cmd_ea, nd, ncnt,
                        LW_(c->ls, 0x34B70), LW_(s_lsc, 0x34B70), LW_(c->ls, 0x34B80), LW_(s_lsc, 0x34B80), LW_(c->ls, 0x34B90), LW_(s_lsc, 0x34B90)); } }
    }
    /* contatore di linea: (pre + 1) & 127 atteso in LS 0x7A0A (informativo: il PPU puo' averlo cambiato nel frattempo) */
    {   uint32_t pre = ((uint32_t)pre_cnt_hi << 8) | pre_cnt_lo, got = ((uint32_t)c->ls[0x7A0A] << 8) | c->ls[0x7A0B];
        if (((pre + 1) & 127u) != got) INC(s_dual_linediff); }
    if (!bad) INC(s_dual_ok);
    LeaveCriticalSection(&s_cs);
    return 1;
}

/* ------------------------------------------------------------------ ingresso */
int anim_hle_run(struct spu_context* ctx, void (*self)(struct spu_context*))
{
    spu_context* c = (spu_context*)ctx;
    if (t_in || !vm_base) return 0;
    const uint32_t r3 = c->gpr[3]._u32[0], cmd_ea = c->gpr[4]._u32[0];
    int why;
    if (r3 != 0x7A80) { INC(s_cnt[R_R3]); return 0; }
    if ((why = hle_gate(c)) != R_OK) { INC(s_cnt[why]); goto report; }
    if (g_anim_hle_mode == 2) { int r = hle_dual(c, self, cmd_ea); if (!((++s_report_tick) & 8191)) hle_report(); return r; }
    {
        anim_commit cm;
        const unsigned long long t0 = __rdtsc();
        uint32_t d_cnt = 0, d_w0 = 0, d_w1 = 0;
        if (s_diag) {
            s_last_ctx = c; s_st_state = 1; s_st_t0 = t0;
            d_cnt = LW_(c->ls, 0x34B90); d_w0 = LW_(c->ls, 0x7A00); d_w1 = LW_(c->ls, 0x7A04);
            if (cmd_ea == s_last_cmd_ea) { INC(s_rep_cnt); if (memcmp(s_last_desc, c->ls + 0x7A80, 32) == 0) INC(s_rep_same); }
            s_last_cmd_ea = cmd_ea; memcpy(s_last_desc, c->ls + 0x7A80, 32);
        }
        g_anim_note_cb = t_note_ctx ? note_cb : 0;     /* NB: globale, ma scritta solo con il gate gia' passato; vedi sotto */
        int rc = hle_compute(vm_base, cmd_ea, c->ls, &cm);
        g_anim_note_cb = 0;
        if (rc != ANIM_OK) { if (s_diag) s_st_state = 0; INC(s_cnt[rc == ANIM_ARENA_OVF ? R_COVF : R_CNC]); goto report; }
        const unsigned long long t1 = __rdtsc();
        if (s_diag) { s_last_line_ea = cm.line_ea; if ((long long)(t1 - t0) > s_cp_cyc_max) s_cp_cyc_max = (long long)(t1 - t0); }
        uint32_t ts = hle_commit(c, &cm);
        const unsigned long long t2 = __rdtsc();
        if (s_diag) {
            if ((long long)(t2 - t1) > s_cm_cyc_max) s_cm_cyc_max = (long long)(t2 - t1);
            INC(s_cm_n);
            long p = __atomic_fetch_add(&s_ring_pos, 1, __ATOMIC_RELAXED);
            ring_ent* r = &s_ring[p & 63];
            r->cmd_ea = cmd_ea; r->cnt_ls = d_cnt; r->w0 = d_w0; r->w1 = d_w1; r->n = cm.n; r->rc = rc; r->tsc = t2;
            r->retries = t_retries; r->flag_after = LW_(vm_base, cmd_ea + 16);
        }
        hle_return(c, ts);
        INC(s_nnative); ADD(s_cyc_compute, (long long)(t1 - t0)); ADD(s_cyc_commit, (long long)(t2 - t1));
        if (!((++s_report_tick) & 8191)) hle_report();
        return 1;
    }
report:
    if (!((++s_report_tick) & 8191)) hle_report();
    return 0;
}
