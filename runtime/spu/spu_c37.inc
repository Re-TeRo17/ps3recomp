/* ---------------------------------------------------------------------------------------------------------------
 * coll37 tracer (SPU_C37=1, off by default): who writes the 160-byte job record of the raw collision SPU img 110
 * (EA 0x10612D80), in which ORDER relative to the claim line 0x10612E80, and who wakes the raw SPUs.
 * Included by spu_coherency.c. Everything is out of the hot path: the hooks test g_c37_on (-1 = not decided yet).
 *   ring of events (SPU DMA GET/PUT, atomics, PPU stores via PS3_WATCH_WRITE, memory changes seen by a poller thread,
 *   waits/wakes of img 110/111), time = QueryPerformanceCounter. Dumped on a stale record (spu_collnet_guard),
 *   at SPU_C37_AT seconds (default 75) and a wait/wake summary every 10 s.
 *   SPU_C37_WIN=lo:hi (hex, default 10612C00:10613000)  SPU_C37_REC=hex (poller, default 10612D80)  SPU_C37_CLAIM=hex (10612E80)
 *   SPU_C37_ALLDMA=1 also logs every DMA of img 110/111 (not only the ones in the window)
 *   SPU_C37_POLL=0 no poller thread
 * ------------------------------------------------------------------------------------------------------------- */
typedef struct { uint64_t t; uint32_t ea, pc, size, aux; int16_t img; uint8_t kind, cmd, spu, pad; uint8_t d[16]; } c37_ev_t;
volatile int g_c37_on = -1;
#define C37_N (1u << 17)
static c37_ev_t* s_c37;
static volatile long long s_c37_i;
static volatile long s_c37_state;
static uint32_t s_c37_lo = 0x10612C00u, s_c37_hi = 0x10613000u, s_c37_rec = 0x10612D80u, s_c37_claim = 0x10612E80u;
static int s_c37_all, s_c37_at = 75, s_c37_poll;
static uint64_t s_c37_f, s_c37_t0;
static __declspec(thread) int t_c37_wimg;   /* image id of the SPU that is committing an atomic (so a notify can name the writer) */

static uint64_t c37_now(void) { LARGE_INTEGER q; QueryPerformanceCounter(&q); return (uint64_t)q.QuadPart; }
static c37_ev_t* c37_new(char kind)
{
    long long i = _InterlockedIncrement64(&s_c37_i) - 1;
    c37_ev_t* e = &s_c37[i & (C37_N - 1)];
    memset(e, 0, sizeof *e);
    e->t = c37_now();
    e->kind = (uint8_t)kind;
    return e;
}
static int c37_in_win(uint32_t ea, uint32_t size) { return ea < s_c37_hi && ea + (size ? size : 1) > s_c37_lo; }

/* ---- request queue for dumps (executed by the poller thread, after the deadline) ---- */
typedef struct { volatile long used; long long from, to; uint64_t anchor, deadline; char tag[24]; } c37_req_t;
static c37_req_t s_c37_req[8];

/* ---- wait / wake statistics ---- */
typedef struct { int img; uint32_t pc, ch; unsigned long n, to; double us; } c37_wst_t;
static c37_wst_t s_c37_w[48];
typedef struct { uint32_t line; int writer; int woken_img; unsigned long n; } c37_kst_t;   /* writer: -1 PPU, else image of the SPU */
static c37_kst_t s_c37_k[48];
static volatile long s_c37_wl, s_c37_kl;

static void c37_fmt_ev(const c37_ev_t* e, uint64_t anchor)
{
    double us = ((double)(int64_t)(e->t - anchor)) * 1e6 / (double)s_c37_f;
    uint32_t w[4]; memcpy(w, e->d, 16);
#define BE(x) __builtin_bswap32(x)
    switch (e->kind) {
    case 'W': { uint64_t v; memcpy(&v, e->d, 8);
        fprintf(stderr, "[c37] %+10.1f W  PPU %s ea=%08X w%u val=0x%llX guest-fn=%08X lr=%08X tid=%u\n", us, (e->size & 0x80) ? "ATOMIC" : "store", e->ea, e->size & 0x7F, (unsigned long long)v, e->aux, e->pc, e->spu);
        if (e->size & 0x80) { uint32_t r0, r1; memcpy(&r0, e->d + 8, 4); memcpy(&r1, e->d + 12, 4); fprintf(stderr, "[c37]            record bytes 8..15 in memory at this moment: %08X %08X\n", BE(r0), BE(r1)); } break; }
    case 'M': fprintf(stderr, "[c37] %+10.1f M  mem changed ea=%08X -> %08X %08X %08X %08X\n", us, e->ea, BE(w[0]), BE(w[1]), BE(w[2]), BE(w[3])); break;
    case 'w': fprintf(stderr, "[c37] %+10.1f w  img=%d spu=%u pc=%05X WAIT begin ch=%u\n", us, e->img, e->spu, e->pc, e->aux); break;
    case 'x': fprintf(stderr, "[c37] %+10.1f x  img=%d spu=%u pc=%05X WAIT end   ch=%u %s waited=%uus\n", us, e->img, e->spu, e->pc, e->aux & 0xFFFF, (e->aux >> 16) ? "TIMEOUT" : "woken", e->size); break;
    case 'N': fprintf(stderr, "[c37] %+10.1f N  notify line=%08X by %s woke imgmask=0x%X (110=1 111=2 other=4) caller_rva=%08X\n", us, e->ea, e->img < 0 ? "PPU" : "SPU", e->aux, e->pc);
        if (e->img >= 0) fprintf(stderr, "[c37]            (writer img=%d)\n", e->img); break;
    case 'S': fprintf(stderr, "[c37] %+10.1f S  STALE record seen by img=%d spu=%u pc=%05X rec_ea=%08X LS[0..16]=%08X %08X %08X %08X\n", us, e->img, e->spu, e->pc, e->ea, BE(w[0]), BE(w[1]), BE(w[2]), BE(w[3])); break;
    case 'R': fprintf(stderr, "[c37] %+10.1f R  re-read finished: rereads=%u (waited %uus) MEM[0..16]=%08X %08X %08X %08X\n", us, e->aux, e->size, BE(w[0]), BE(w[1]), BE(w[2]), BE(w[3])); break;
    default:
        fprintf(stderr, "[c37] %+10.1f %c  img=%d spu=%u pc=%05X cmd=%02X ea=%08X sz=%u lsa=%05X | %08X %08X %08X %08X\n", us, e->kind, e->img, e->spu, e->pc, e->cmd, e->ea, e->size, e->aux, BE(w[0]), BE(w[1]), BE(w[2]), BE(w[3]));
    }
#undef BE
}
static void c37_dump(long long from, long long to, uint64_t anchor, const char* tag)
{
    if (to - from > (long long)C37_N - 16) from = to - (long long)C37_N + 16;
    if (from < 0) from = 0;
    fprintf(stderr, "[c37] ===== dump %s: events %lld..%lld =====\n", tag, from, to);
    for (long long i = from; i < to; i++) {
        const c37_ev_t* e = &s_c37[i & (C37_N - 1)];
        if (!e->kind) continue;
        c37_fmt_ev(e, anchor ? anchor : s_c37[from & (C37_N - 1)].t);
    }
    fprintf(stderr, "[c37] ===== end dump %s =====\n", tag);
    fflush(stderr);
}
static void c37_request(long long from, long long to, uint64_t anchor, double ahead_us, const char* tag)
{
    for (int k = 0; k < 8; k++)
        if (_InterlockedCompareExchange(&s_c37_req[k].used, 2, 0) == 0) {
            s_c37_req[k].from = from; s_c37_req[k].to = to; s_c37_req[k].anchor = anchor;
            s_c37_req[k].deadline = c37_now() + (uint64_t)(ahead_us * (double)s_c37_f / 1e6);
            strncpy(s_c37_req[k].tag, tag, 23); s_c37_req[k].tag[23] = 0;
            _InterlockedExchange(&s_c37_req[k].used, 1);
            return;
        }
}
static void c37_print_stats(void)
{
    for (int i = 0; i < 48 && i < s_c37_wl; i++)
        if (s_c37_w[i].n)
            fprintf(stderr, "[c37-wait] img=%d pc=%05X ch=%u waits=%lu timeouts=%lu total=%.1fms avg=%.1fus\n", s_c37_w[i].img, s_c37_w[i].pc, s_c37_w[i].ch,
                    s_c37_w[i].n, s_c37_w[i].to, s_c37_w[i].us / 1000.0, s_c37_w[i].us / (double)s_c37_w[i].n);
    for (int i = 0; i < 48 && i < s_c37_kl; i++)
        if (s_c37_k[i].n)
            fprintf(stderr, "[c37-wake] line=%08X writer=%s%d woke img=%d : %lu\n", s_c37_k[i].line, s_c37_k[i].writer < 0 ? "PPU" : "SPU img", s_c37_k[i].writer < 0 ? 0 : s_c37_k[i].writer, s_c37_k[i].woken_img, s_c37_k[i].n);
    fflush(stderr);
}

static DWORD WINAPI c37_poller(LPVOID arg)
{
    (void)arg;
    uint8_t last[18][16]; uint32_t ea_of[18]; int n = 0;
    for (int k = 0; k < 10; k++) ea_of[n++] = s_c37_rec + 16u * k;
    for (int k = 0; k < 8; k++) ea_of[n++] = s_c37_claim + 16u * k;
    for (;;) {   /* wait until the guest memory is committed */
        MEMORY_BASIC_INFORMATION mbi;
        if (vm_base && VirtualQuery(vm_base + s_c37_rec, &mbi, sizeof mbi) && mbi.State == MEM_COMMIT && VirtualQuery(vm_base + s_c37_claim, &mbi, sizeof mbi) && mbi.State == MEM_COMMIT) break;
        Sleep(100);
    }
    for (int k = 0; k < n; k++) memcpy(last[k], vm_base + ea_of[k], 16);
    const uint64_t at = s_c37_t0 + (uint64_t)s_c37_at * s_c37_f;
    int did_at = 0; uint64_t next_stat = c37_now() + 10 * s_c37_f;
    for (unsigned it = 0;; it++) {
        for (int k = 0; k < n; k++) {
            if (memcmp(last[k], vm_base + ea_of[k], 16) != 0) {
                uint8_t cur[16]; memcpy(cur, vm_base + ea_of[k], 16);
                c37_ev_t* e = c37_new('M'); e->ea = ea_of[k]; memcpy(e->d, cur, 16);
                memcpy(last[k], cur, 16);
            }
        }
        if ((it & 63) == 0) {
            uint64_t now = c37_now();
            for (int k = 0; k < 8; k++)
                if (s_c37_req[k].used == 1 && now >= s_c37_req[k].deadline) {
                    long long to = s_c37_req[k].to ? s_c37_req[k].to : s_c37_i;
                    c37_dump(s_c37_req[k].from, to, s_c37_req[k].anchor, s_c37_req[k].tag);
                    _InterlockedExchange(&s_c37_req[k].used, 0);
                }
            if (!did_at && now >= at) { did_at = 1; long long e1 = s_c37_i; c37_dump(e1 - 700, e1, 0, "periodic"); }
            if (now >= next_stat) { next_stat = now + 10 * s_c37_f; c37_print_stats(); }
        }
        _mm_pause();
    }
    return 0;
}

static void c37_init(void)
{
    if (_InterlockedCompareExchange(&s_c37_state, 1, 0) != 0) { while (s_c37_state != 2) Sleep(0); return; }
    const char* e = getenv("SPU_C37");
    int on = (e && e[0] && e[0] != '0');
    if (on) {
        LARGE_INTEGER f; QueryPerformanceFrequency(&f); s_c37_f = (uint64_t)f.QuadPart; s_c37_t0 = c37_now();
        s_c37 = (c37_ev_t*)calloc(C37_N, sizeof(c37_ev_t));
        if (!s_c37) on = 0;
    }
    if (on) {
        if ((e = getenv("SPU_C37_WIN"))) { char* c = 0; s_c37_lo = (uint32_t)strtoul(e, &c, 16); if (c && *c == ':') s_c37_hi = (uint32_t)strtoul(c + 1, 0, 16); }
        if ((e = getenv("SPU_C37_REC"))) s_c37_rec = (uint32_t)strtoul(e, 0, 16);
        if ((e = getenv("SPU_C37_CLAIM"))) s_c37_claim = (uint32_t)strtoul(e, 0, 16);
        if ((e = getenv("SPU_C37_ALLDMA"))) s_c37_all = e[0] == '1';
        if ((e = getenv("SPU_C37_AT"))) s_c37_at = atoi(e);
        fprintf(stderr, "[c37] ON window=%08X..%08X rec=%08X claim=%08X alldma=%d dump_at=%ds\n", s_c37_lo, s_c37_hi, s_c37_rec, s_c37_claim, s_c37_all, s_c37_at);
        if (!((e = getenv("SPU_C37_POLL")) && e[0] == '0')) { s_c37_poll = 1; CreateThread(0, 0, c37_poller, 0, 0, 0); }
    }
    g_c37_on = on;
    _InterlockedExchange(&s_c37_state, 2);
}

/* DMA of an SPU: post=0 before the transfer (PUT: the LS data about to be written), post=1 after (GET: the LS data that landed) */
void c37_dma(const spu_context* spu, uint32_t lsa, uint64_t ea, uint32_t size, uint32_t cmd, int post)
{
    if (g_c37_on < 0) c37_init();
    if (g_c37_on <= 0) return;
    int get = (cmd & 0x40) != 0;
    if (get != (post != 0)) return;                 /* GET: log after, PUT: log before */
    int img = spu->image_id;
    if (!(c37_in_win((uint32_t)ea, size) || (s_c37_all && (img == 110 || img == 111)))) return;
    c37_ev_t* e = c37_new(get ? 'G' : 'P');
    e->img = (int16_t)img; e->spu = (uint8_t)(spu->spu_id & 7u); e->pc = (uint32_t)spu->pc & SPU_LS_MASK; e->cmd = (uint8_t)cmd;
    e->ea = (uint32_t)ea; e->size = size; e->aux = lsa & SPU_LS_MASK;
    memcpy(e->d, spu->ls + (lsa & SPU_LS_MASK), 16);
}
/* atomic command just executed (GETLLAR 0xD0 / PUTLLC 0xB4 / PUTLLUC 0xB0, 0xB8); the LS holds the line */
void c37_atomic(const spu_context* ctx, uint32_t cmd, int handled)
{
    if (g_c37_on <= 0 || !handled) return;
    uint32_t ea = ctx->mfc_eal & ~127u;
    if (!c37_in_win(ea, 128) && !(s_c37_all && (ctx->image_id == 110 || ctx->image_id == 111))) return;
    char k = cmd == 0xD0 ? 'l' : cmd == 0xB4 ? (ctx->atomic_stat == 0 ? 'C' : 'F') : 'U';
    c37_ev_t* e = c37_new(k);
    e->img = (int16_t)ctx->image_id; e->spu = (uint8_t)(ctx->spu_id & 7u); e->pc = (uint32_t)ctx->pc & SPU_LS_MASK; e->cmd = (uint8_t)cmd;
    e->ea = ea; e->size = 128; e->aux = ctx->mfc_lsa & SPU_LS_MASK;
    { const uint8_t* ln = ctx->ls + (ctx->mfc_lsa & SPU_LS_MASK);   /* word0 (status) | flags word at +0x30 (EB0) | queue head/tail at +0x40 (EC0, EC4) */
      memcpy(e->d, ln, 4); memcpy(e->d + 4, ln + 0x30, 4); memcpy(e->d + 8, ln + 0x40, 8); }
}
void c37_atomic_pre(const spu_context* ctx) { if (g_c37_on > 0) t_c37_wimg = ctx->image_id; }
void c37_atomic_post(void) { t_c37_wimg = 0; }
/* a PPU store (PS3_WATCH_WRITE range) */
void c37_ppu_w(uint32_t a, uint64_t v, int width, uint32_t fn, uint32_t lr, unsigned tid)
{
    c37_ev_t* e = c37_new('W');
    e->ea = a; e->size = (uint32_t)width; memcpy(e->d, &v, 8); e->aux = fn; e->pc = lr; e->spu = (uint8_t)tid; e->img = -1;
    if (vm_base) memcpy(e->d + 8, vm_base + s_c37_rec + 8, 8);   /* record bytes 8..15 as they are in memory right now (idx at 12..13) */
}
/* wait of a raw collision SPU: end=0 begin, end=1 woken, end=2 timed out */
void c37_wait(const spu_context* ctx, uint32_t channel, int end, double us)
{
    if (g_c37_on <= 0 || (ctx->image_id != 110 && ctx->image_id != 111)) return;
    c37_ev_t* e = c37_new(end ? 'x' : 'w');
    e->img = (int16_t)ctx->image_id; e->spu = (uint8_t)(ctx->spu_id & 7u); e->pc = (uint32_t)ctx->pc & SPU_LS_MASK; e->aux = channel | (end == 2 ? 0x10000u : 0);
    e->size = (uint32_t)us;
    if (!end) return;
    long n = s_c37_wl; int f = -1;
    for (int i = 0; i < n && i < 48; i++) if (s_c37_w[i].img == ctx->image_id && s_c37_w[i].pc == e->pc && s_c37_w[i].ch == channel) { f = i; break; }
    if (f < 0) { long k = _InterlockedIncrement(&s_c37_wl) - 1; if (k >= 48) return; f = (int)k; s_c37_w[f].img = ctx->image_id; s_c37_w[f].pc = e->pc; s_c37_w[f].ch = channel; }
    s_c37_w[f].n++; if (end == 2) s_c37_w[f].to++; s_c37_w[f].us += us;
}
/* a coherence notification: `wm` = images of the contexts it woke (reservation lost on this line) */
void c37_notify(uint32_t ea, int writer_img, uint32_t wm, uintptr_t ra)
{
    if (g_c37_on <= 0) return;
    extern char __ImageBase;
    if (c37_in_win(ea & ~127u, 128)) {
        c37_ev_t* e = c37_new('N'); e->ea = ea & ~127u; e->img = (int16_t)writer_img; e->aux = wm; e->pc = (uint32_t)(ra ? (char*)ra - &__ImageBase : 0);
    }
    if (wm & 3u) {   /* statistics of who wakes the raw collision SPUs, over ALL lines */
        for (int wi = 1; wi <= 2; wi++) if (wm & (uint32_t)wi) {
            int img = 109 + wi; long n = s_c37_kl; int f = -1;
            for (int i = 0; i < n && i < 48; i++) if (s_c37_k[i].line == (ea & ~127u) && s_c37_k[i].writer == writer_img && s_c37_k[i].woken_img == img) { f = i; break; }
            if (f < 0) { long k = _InterlockedIncrement(&s_c37_kl) - 1; if (k >= 48) return; f = (int)k; s_c37_k[f].line = ea & ~127u; s_c37_k[f].writer = writer_img; s_c37_k[f].woken_img = img; }
            s_c37_k[f].n++;
        }
    }
}
/* the stale record: returns the event index to hand to c37_stale_done */
long long c37_stale(const spu_context* ctx, uint32_t rec_ea)
{
    if (g_c37_on <= 0) return -1;
    c37_ev_t* e = c37_new('S'); e->img = (int16_t)ctx->image_id; e->spu = (uint8_t)(ctx->spu_id & 7u); e->pc = (uint32_t)ctx->pc & SPU_LS_MASK; e->ea = rec_ea;
    memcpy(e->d, ctx->ls + 0x23000, 16);
    return s_c37_i - 1;
}
void c37_stale_done(long long mark, uint32_t rereads, const uint8_t* mem)
{
    if (g_c37_on <= 0 || mark < 0) return;
    c37_ev_t* e = c37_new('R'); e->aux = rereads; e->size = (uint32_t)(((double)(e->t - s_c37[mark & (C37_N - 1)].t)) * 1e6 / (double)s_c37_f);
    if (mem) memcpy(e->d, mem, 16);
    static volatile long s_n; long n = _InterlockedIncrement(&s_n);
    if (n <= 6) { char tag[24]; snprintf(tag, sizeof tag, "stale#%ld", n);
        if (s_c37_poll) c37_request(mark - 260, 0, s_c37[mark & (C37_N - 1)].t, 1500.0, tag);
        else c37_dump(mark - 260, s_c37_i, s_c37[mark & (C37_N - 1)].t, tag); }
}

/* a successful PPU stwcx./stdcx. (reported only inside the window) */
void c37_ppu_atomic_impl(uint32_t ea, uint64_t val, int width, uint32_t fn, uint32_t lr, unsigned tid)
{
    if (!c37_in_win(ea, (uint32_t)width)) return;
    c37_ppu_w(ea, val, width | 0x80, fn, lr, tid);
}
