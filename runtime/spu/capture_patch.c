/* capture_patch.c -- cattura in gioco dei comandi wid0 0x6248 (anim46). NON applicato a nessun worktree.
 *
 * ===================================================================================================
 * ISTRUZIONI PER INSERIRLO (da fare in una COPIA di lavoro, es. come coll_native_patch.cmake di coll43)
 * ===================================================================================================
 * 1) Sorgenti: copiare (o referenziare) in ps3recomp\runtime\spu\ i file
 *        capture_patch.c  capture_patch.h  anim_ref.c  anim_ref.h  anim_fx.c  anim_gen_5B78.inc  anim_gen_53B8.inc
 *    (tutti in log\anim46\) e aggiungere al target del gioco, SOLO per capture_patch.c, -DANIM_CAP_GAME:
 *        set_property(SOURCE .../capture_patch.c APPEND PROPERTY COMPILE_DEFINITIONS ANIM_CAP_GAME)
 *    anim_ref.c include "spu_helpers.h" e usa <immintrin.h>: serve -march=x86-64-v3 (come il resto del codice SPU lifted).
 *    NB: anim_ref.c definisce simboli non static (anim_f4030..., g_anim_*): nessun conflitto noto col runtime.
 * 2) Gancio nel codice lifted del job (jobmod_00b730f78d78_e2F88_at00A10200\spu_recomp.c, e NON nell'originale in
 *    tod-recomp\spu_p: lavorare su una copia in build\gen\anim\): lo script anim_cap_insert.py (stesso dir) inserisce
 *        SPU_ANIM_CAP_HOOK(ctx, <la funzione stessa>);
 *    come PRIMA istruzione (dopo l'eventuale riga spu_pcheck_enter) di
 *        ..._spu_func_00006248, ..._spu_func_00006248_p, ..._spu_func_00006248_pn   (quelle che esistono)
 *    ed un #include "capture_patch.h" in testa. Uso:  python anim_cap_insert.py <spu_recomp.c in> <spu_recomp.c out>
 *    Le ~25 copie dynovl del job (stessa testa 0x4000-0x7480, hle40a) hanno le STESSE funzioni con prefisso diverso:
 *    lo script accetta piu' file, e l'hash dell'immagine (0x4000-0x7480) registrato in ogni record dice quale copia ha girato.
 * 3) Il gancio e' a costo zero da spento: legge UN byte (g_anim_cap_on). Attivo solo se esiste l'env ANIM_CAPTURE.
 * 4) Esecuzione: ANIM_CAPTURE=C:\\...\\wid0.cap ANIM_CAPTURE_N=20000 tod.exe ...   (default N = 20000; a N raggiunto la cattura
 *    si ferma da sola e g_anim_cap_on torna 0). Il file e' flushato a ogni record: si puo' chiudere il gioco in qualunque momento.
 *    Dimensione tipica ~2-40 KB per comando (dipende da N e dalle foglie): 20000 comandi ~ qualche centinaio di MB al massimo;
 *    usare ANIM_CAPTURE_N=3000 per una prima prova (~ una decina di secondi di gioco).
 * 5) Offline: anim_test replay <file> [ripetizioni]   (build: build46.ps1 -Out replay -Flags "-O2 -DSPU_RZ_FP_TLS").
 *    Stampa: comandi catturati, coperti dal ref (OK / NOT_COVERED / ARENA_OVF), divergenze sui PUT (fatali), differenze di LS
 *    (informative), istogrammi di N / foglie / DMA, e ns/comando di anim_cmd_fast sui dati REALI.
 *
 * COSA REGISTRA (per comando):  hash+immagine del job (0x4000..0x7480) ; r3 ; r4 (= cmd_ea) ; LS [0x7480,0x7B00) e [0x34B00,0x34D00)
 *   a ingresso e a uscita ; tutti i GET che il comando fa (EA, size, BYTE letti: corpo, A1, albero, A2, linea coda, elementi GETL
 *   = intestazioni keyframe, flussi 1 e keyframe dei GET di 0x5B78) ; i byte scritti dai PUT (matrici N*64 B, flag di completamento
 *   4 B a cmd_ea+16, linea coda) letti dalla memoria principale subito dopo il ritorno della funzione lifted.
 * COME SI SA QUALI INTERVALLI: a ingresso il gancio esegue anim_cmd_ref su una COPIA dell'LS (non tocca ne' LS ne' memoria reali)
 *   con il registro degli effetti: i GET/GETL/GETLLAR/PUT che il ref emette danno gli intervalli. Se il ref risponde NOT_COVERED /
 *   ARENA_OVF il record contiene comunque corpo, albero e il PUT atteso [ea_out, 64*N) + flag (cosi' si contano anche i non coperti).
 *   Limite noto: un byte letto dal job reale ma NON dal ref non e' registrato; se ne vede l'effetto come divergenza sui PUT.
 * CORSA: il PPU puo' riscrivere la linea coda tra il PUTLLC del job e la lettura di uscita: la linea e' registrata ma nel replay si
 *   confrontano solo i 2 byte a +10 (contatore) e senza essere fatale. Le matrici in uscita sono lette subito dopo il ritorno
 *   (microsecondi dopo il PUT); il PPU le consuma dopo il flag, quindi la finestra di corsa e' trascurabile ma non nulla.
 * =================================================================================================== */
#define _CRT_SECURE_NO_WARNINGS
#include "capture_patch.h"
#include "anim_ref.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------------------------ utilita' */
#define FOURCC(a,b,c,d) ((uint32_t)(a) | ((uint32_t)(b) << 8) | ((uint32_t)(c) << 16) | ((uint32_t)(d) << 24))
#define TAG_IMG FOURCC('I','M','G',' ')
#define TAG_CMD FOURCC('C','M','D',' ')
#define IMG_LO  0x4000u
#define IMG_HI  0x7480u
#define LS_A_LO 0x7480u
#define LS_A_HI 0x7B00u
#define LS_B_LO 0x34B00u
#define LS_B_HI 0x34D00u

typedef struct { uint8_t* p; size_t n, cap; } bbuf;
static void bb_put(bbuf* b, const void* d, size_t n)
{
    if (b->n + n > b->cap) { size_t nc = b->cap ? b->cap : 4096; while (nc < b->n + n) nc *= 2; b->p = (uint8_t*)realloc(b->p, nc); b->cap = nc; }
    memcpy(b->p + b->n, d, n); b->n += n;
}
static void bb_u32(bbuf* b, uint32_t v) { bb_put(b, &v, 4); }
static void bb_u64(bbuf* b, uint64_t v) { bb_put(b, &v, 8); }

static uint32_t be32(const uint8_t* p) { return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]; }
static uint32_t be16(const uint8_t* p) { return ((uint32_t)p[0] << 8) | p[1]; }

static uint64_t img_hash(const uint8_t* ls)
{
    uint64_t h = 0x9E3779B97F4A7C15ull;
    for (uint32_t a = IMG_LO; a < IMG_HI; a += 8) { uint64_t w; memcpy(&w, ls + a, 8); h = (h ^ w) * 0x100000001B3ull; h ^= h >> 29; }
    return h;
}

/* ------------------------------------------------------------------------------------------ stato scrittura */
static FILE*    s_f;
static unsigned s_count, s_max = 20000;
static uint64_t s_seen[16]; static int s_nseen;

#define MAXR 512
struct anim_cap_rec {
    bbuf     pre;              /* payload parziale (fino ai GET inclusi) */
    uint32_t nexp, exp_ea[MAXR], exp_sz[MAXR], exp_kind[MAXR];
    uint32_t seq;
};
static struct anim_cap_rec s_rec;
static anim_fx s_fx;           /* grande: statico */
static uint8_t s_lscopy[0x40000];

int anim_cap_open(const char* path)
{
    s_f = fopen(path, "wb");
    if (!s_f) return -1;
    fwrite("ANIMCAP1", 1, 8, s_f);
    uint32_t v[2] = { 1, 0 }; fwrite(v, 4, 2, s_f);
    s_count = 0; s_nseen = 0;
    return 0;
}
void anim_cap_close(void) { if (s_f) { fclose(s_f); s_f = 0; } }
unsigned anim_cap_count(void) { return s_count; }

static void emit_chunk(uint32_t tag, const void* d, size_t n)
{
    uint32_t h[2] = { tag, (uint32_t)n };
    fwrite(h, 4, 2, s_f); fwrite(d, 1, n, s_f);
}

static void put_ls_regions(bbuf* b, const uint8_t* ls, uint32_t arena_hi)
{
    bb_u32(b, arena_hi > 0x7B00u ? 3 : 2);
    bb_u32(b, LS_A_LO); bb_u32(b, LS_A_HI - LS_A_LO); bb_put(b, ls + LS_A_LO, LS_A_HI - LS_A_LO);
    bb_u32(b, LS_B_LO); bb_u32(b, LS_B_HI - LS_B_LO); bb_put(b, ls + LS_B_LO, LS_B_HI - LS_B_LO);
    if (arena_hi > 0x7B00u) { bb_u32(b, 0x7B00u); bb_u32(b, arena_hi - 0x7B00u); bb_put(b, ls + 0x7B00u, arena_hi - 0x7B00u); }
}

static void exp_add(anim_cap_rec* r, uint32_t ea, uint32_t sz, uint32_t kind)
{
    if (r->nexp < MAXR) { r->exp_ea[r->nexp] = ea; r->exp_sz[r->nexp] = sz; r->exp_kind[r->nexp] = kind; r->nexp++; }
}

anim_cap_rec* anim_cap_begin(const uint8_t* ls, const uint8_t* vm, uint32_t r3, uint32_t cmd_ea)
{
    if (!s_f || s_count >= s_max) return 0;
    anim_cap_rec* r = &s_rec;
    free(r->pre.p); memset(r, 0, sizeof *r);
    r->seq = s_count;
    uint64_t hh = img_hash(ls);
    int known = 0; for (int i = 0; i < s_nseen; i++) if (s_seen[i] == hh) known = 1;
    if (!known && s_nseen < 16) {
        bbuf ib = { 0 }; bb_u64(&ib, hh); bb_u32(&ib, IMG_HI - IMG_LO); bb_put(&ib, ls + IMG_LO, IMG_HI - IMG_LO);
        emit_chunk(TAG_IMG, ib.p, ib.n); free(ib.p); s_seen[s_nseen++] = hh;
    }
    bbuf* b = &r->pre;
    bb_u32(b, r->seq); bb_u64(b, hh); bb_u32(b, r3); bb_u32(b, cmd_ea);

    /* pre-run del riferimento su una copia: serve a sapere quali intervalli il comando legge/scrive */
    memcpy(s_lscopy, ls, 0x40000);
    anim_fx_init(&s_fx);
    extern uint32_t g_anim_ea_limit; g_anim_ea_limit = 0;
    int rc = (r3 == 0x7A80) ? anim_cmd_ref(vm, cmd_ea, s_lscopy, &s_fx) : ANIM_BAD_INPUT;
    bb_u32(b, (uint32_t)rc);
    /* arena: opzionale (env ANIM_CAPTURE_ARENA=1): il contenuto 'stantio' dell'arena [0x7B00, estensione) a ingresso, perche' alcuni byte
     * dei blocchi non sovrascritti dal comando possono influire sul risultato (mezze parole pari non scritte da 0x4060/0x4120...) */
    uint32_t arena_hi = 0;
    if (rc == ANIM_OK && getenv("ANIM_CAPTURE_ARENA")) arena_hi = LS_B_LO;   /* tutta l'arena 0x7B00..0x34B00 (184 KB per comando: pesante) */
    if (arena_hi > LS_B_LO) arena_hi = LS_B_LO; arena_hi = (arena_hi + 15) & ~15u;
    put_ls_regions(b, ls, arena_hi);

    /* intervalli GET */
    struct { uint32_t ea, sz; } g[MAXR]; uint32_t ng = 0;
    uint32_t ea_out = be32(ls + 0x7A80), ea_body = be32(ls + 0x7A84), ea_tree = be32(ls + 0x7A88), tsz = be16(ls + 0x7A8C);
    if (rc == ANIM_OK) {
        for (int i = 0; i < s_fx.n && ng < MAXR; i++) {
            const anim_ev* e = &s_fx.ev[i];
            if (e->cmd == 64) { g[ng].ea = e->ea; g[ng].sz = e->size; ng++; }
            else if (e->cmd == 208) { g[ng].ea = e->ea; g[ng].sz = 128; ng++; }
            else if (e->cmd == 68) {                    /* GETL: lista di coppie {size, ea} nei byte registrati */
                const uint8_t* lst = s_fx.data + e->doff;
                for (uint32_t o = 0; o + 8 <= e->size && ng < MAXR; o += 8) { g[ng].sz = be32(lst + o) & 0x7FFF; g[ng].ea = be32(lst + o + 4); ng++; }
            }
        }
        for (int i = 0; i < s_fx.n; i++) {
            const anim_ev* e = &s_fx.ev[i];
            if (e->cmd == 32) exp_add(r, e->ea, e->size, 0);
            else if (e->cmd == 34) exp_add(r, e->ea, e->size, 1);
            else if (e->cmd == 180 || e->cmd == 200) exp_add(r, e->ea, e->size, 2);
        }
    } else if (r3 == 0x7A80) {                           /* non coperto: corpo, albero e PUT attesi come da formato */
        int16_t n = (int16_t)be16(vm + ea_body);
        g[ng].ea = ea_body; g[ng].sz = 32; ng++;
        if (tsz) { g[ng].ea = ea_tree; g[ng].sz = tsz; ng++; }
        if (n > 0 && n < 4000) exp_add(r, ea_out, (uint32_t)n * 64, 0);
        exp_add(r, cmd_ea + 16, 4, 1);
    }
    bb_u32(b, ng);
    for (uint32_t i = 0; i < ng; i++) { bb_u32(b, g[i].ea); bb_u32(b, g[i].sz); bb_put(b, vm + g[i].ea, g[i].sz); }
    anim_fx_free(&s_fx);
    return r;
}

void anim_cap_end(anim_cap_rec* r, const uint8_t* ls_after, const uint8_t* vm)
{
    if (!r || !s_f) return;
    bb_u32(&r->pre, r->nexp);
    for (uint32_t i = 0; i < r->nexp; i++) {
        bb_u32(&r->pre, r->exp_ea[i]); bb_u32(&r->pre, r->exp_sz[i]); bb_u32(&r->pre, r->exp_kind[i]);
        bb_put(&r->pre, vm + r->exp_ea[i], r->exp_sz[i]);
    }
    put_ls_regions(&r->pre, ls_after, 0);
    emit_chunk(TAG_CMD, r->pre.p, r->pre.n);
    fflush(s_f);
    s_count++;
}

/* ------------------------------------------------------------------------------------------ lettore */
static struct { uint64_t h; const uint8_t* p; uint32_t n; } s_imgs[16]; static int s_nimgs;

int anim_cap_load_file(const char* path, uint8_t** buf, uint32_t* size)
{
    FILE* f = fopen(path, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* p = (uint8_t*)malloc((size_t)n);
    if (fread(p, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(p); return -1; }
    fclose(f);
    if (n < 16 || memcmp(p, "ANIMCAP1", 8)) { free(p); return -1; }
    *buf = p; *size = (uint32_t)n; s_nimgs = 0;
    return 0;
}

static int rd_regions(const uint8_t* b, uint32_t size, uint32_t* o, anim_cap_span* out, uint32_t* n)
{
    if (*o + 4 > size) return -1;
    memcpy(n, b + *o, 4); *o += 4;
    if (*n > 4) return -1;
    for (uint32_t i = 0; i < *n; i++) {
        if (*o + 8 > size) return -1;
        memcpy(&out[i].addr, b + *o, 4); memcpy(&out[i].len, b + *o + 4, 4); *o += 8;
        if (*o + out[i].len > size) return -1;
        out[i].p = b + *o; *o += out[i].len;
    }
    return 0;
}

int anim_cap_next(uint8_t* buf, uint32_t size, uint32_t* pos, anim_cap_view* v, const uint8_t** img, uint32_t* imgsize)
{
    while (*pos + 8 <= size) {
        uint32_t tag, len; memcpy(&tag, buf + *pos, 4); memcpy(&len, buf + *pos + 4, 4);
        uint32_t o = *pos + 8, end = o + len;
        if (end > size) return -1;
        *pos = end;
        if (tag == TAG_IMG) {
            if (s_nimgs < 16) { memcpy(&s_imgs[s_nimgs].h, buf + o, 8); memcpy(&s_imgs[s_nimgs].n, buf + o + 8, 4); s_imgs[s_nimgs].p = buf + o + 12; s_nimgs++; }
            continue;
        }
        if (tag != TAG_CMD) continue;
        memset(v, 0, sizeof *v);
        if (o + 4 + 8 + 4 + 4 + 4 > end) return -1;
        memcpy(&v->seq, buf + o, 4); memcpy(&v->imghash, buf + o + 4, 8); memcpy(&v->r3, buf + o + 12, 4);
        memcpy(&v->cmd_ea, buf + o + 16, 4); memcpy(&v->rc_pre, buf + o + 20, 4); o += 24;
        if (rd_regions(buf, end, &o, v->ls_pre, &v->nls)) return -1;
        if (o + 4 > end) return -1;
        memcpy(&v->nget, buf + o, 4); o += 4;
        v->get = (anim_cap_span*)calloc(v->nget ? v->nget : 1, sizeof(anim_cap_span));
        for (uint32_t i = 0; i < v->nget; i++) {
            if (o + 8 > end) return -1;
            memcpy(&v->get[i].addr, buf + o, 4); memcpy(&v->get[i].len, buf + o + 4, 4); o += 8;
            if (o + v->get[i].len > end) return -1;
            v->get[i].p = buf + o; o += v->get[i].len;
        }
        if (o + 4 > end) return -1;
        memcpy(&v->nexp, buf + o, 4); o += 4;
        v->exp = (anim_cap_span*)calloc(v->nexp ? v->nexp : 1, sizeof(anim_cap_span));
        v->exp_kind = (uint32_t*)calloc(v->nexp ? v->nexp : 1, 4);
        for (uint32_t i = 0; i < v->nexp; i++) {
            if (o + 12 > end) return -1;
            memcpy(&v->exp[i].addr, buf + o, 4); memcpy(&v->exp[i].len, buf + o + 4, 4); memcpy(&v->exp_kind[i], buf + o + 8, 4); o += 12;
            if (o + v->exp[i].len > end) return -1;
            v->exp[i].p = buf + o; o += v->exp[i].len;
        }
        if (rd_regions(buf, end, &o, v->ls_post, &v->nlsa)) return -1;
        *img = 0; *imgsize = 0;
        for (int i = 0; i < s_nimgs; i++) if (s_imgs[i].h == v->imghash) { *img = s_imgs[i].p; *imgsize = s_imgs[i].n; }
        return 1;
    }
    return 0;
}
void anim_cap_view_free(anim_cap_view* v) { free(v->get); free(v->exp); free(v->exp_kind); memset(v, 0, sizeof *v); }

/* ------------------------------------------------------------------------------------------ strato B: gioco */
volatile uint8_t g_anim_cap_on = 0;

#ifdef ANIM_CAP_GAME
#include "spu_context.h"
#ifdef _WIN32
#include <windows.h>
static CRITICAL_SECTION s_cs; static int s_cs_init;
#define CAP_LOCK()   EnterCriticalSection(&s_cs)
#define CAP_UNLOCK() LeaveCriticalSection(&s_cs)
#endif
extern uint8_t* vm_base;
static __declspec(thread) int t_in;      /* guardia di rientro: il gancio richiama la funzione lifted stessa */

extern int g_anim_hle_mode;
void anim_hle_init(void);
int anim_hle_run(struct spu_context* ctx, void (*self)(struct spu_context*));
void anim_cap_init(void)
{
    static int done; if (done) return; done = 1;
    anim_hle_init();
    if (g_anim_hle_mode) g_anim_cap_on = 1;     /* anim47: il gancio passa da anim_cap_run anche per l'aggancio nativo */
    const char* path = getenv("ANIM_CAPTURE");
    if (!path || !*path) return;
    const char* n = getenv("ANIM_CAPTURE_N"); if (n && *n) s_max = (unsigned)atoi(n);
    InitializeCriticalSection(&s_cs); s_cs_init = 1;
    if (anim_cap_open(path)) { fprintf(stderr, "[anim-cap] impossibile aprire %s\n", path); return; }
    g_anim_cap_on = 1;
    fprintf(stderr, "[anim-cap] cattura wid0 0x6248 -> %s (max %u comandi)\n", path, s_max);
}
#if defined(__clang__) || defined(__GNUC__)
__attribute__((constructor)) static void anim_cap_ctor(void) { anim_cap_init(); }
#else
static int anim_cap_ctor2(void) { anim_cap_init(); return 0; }
#pragma section(".CRT$XCU", read)
__declspec(allocate(".CRT$XCU")) static int (*p_anim_cap_ctor)(void) = anim_cap_ctor2;
#endif

int anim_cap_run(struct spu_context* ctx, void (*self)(struct spu_context*))
{
    spu_context* c = (spu_context*)ctx;
    if (g_anim_hle_mode) return anim_hle_run(ctx, self);
    if (t_in || !vm_base) return 0;
    CAP_LOCK();
    if (!g_anim_cap_on || anim_cap_count() >= s_max) {
        if (g_anim_cap_on) { g_anim_cap_on = 0; anim_cap_close(); fprintf(stderr, "[anim-cap] fine: %u comandi\n", anim_cap_count()); }
        CAP_UNLOCK(); return 0;
    }
    uint32_t r3 = c->gpr[3]._u32[0], cmd_ea = c->gpr[4]._u32[0];
    anim_cap_rec* r = anim_cap_begin(c->ls, vm_base, r3, cmd_ea);
    t_in = 1; self(ctx);                          /* il comando lifted, invariato */
    if (g_spu_trampoline_fn) spu_drain_call(ctx, c->gpr[0]._u32[0]);   /* catena interrotta da un trampolino: completala qui (il PUT deve essere fatto prima di leggere) */
    t_in = 0;
    if (r) anim_cap_end(r, c->ls, vm_base);
    CAP_UNLOCK();
    return 1;
}
#endif /* ANIM_CAP_GAME */
