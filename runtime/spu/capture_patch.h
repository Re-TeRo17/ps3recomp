/* capture_patch.h -- cattura in gioco dei comandi di animazione wid0 (funzione SPU 0x6248) per rieseguirli offline
 * con anim_cmd_ref (anim46). NON applicato a nessun worktree: vedi capture_patch.md in testa a capture_patch.c.
 *
 * Due strati:
 *   A) anim_cap_*  : puro (dipende solo da anim_ref.h), usato sia dal gioco sia da `anim_test synthcap|replay`.
 *   B) anim_cap_run: colla col runtime (spu_context, vm_base), compilata solo con -DANIM_CAP_GAME.
 *
 * Costo da spento: UN byte letto (g_anim_cap_on) nel prologo delle funzioni func_00006248 / _p / _pn.
 * Attivazione: env ANIM_CAPTURE=<file> (e ANIM_CAPTURE_N=<max comandi>, default 20000). */
#ifndef ANIM_CAPTURE_PATCH_H
#define ANIM_CAPTURE_PATCH_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* ---- strato B (gioco) ---- */
extern volatile uint8_t g_anim_cap_on;      /* 0 = spento (default), 1 = attivo (settato da anim_cap_init se ANIM_CAPTURE e' definita) */
struct spu_context;
/* Esegue il comando 0x6248 con cattura: chiamata dal gancio con `self` = la funzione lifted stessa.
 * Ritorna 1 se ha eseguito (il gancio fa `return`), 0 se non ha catturato (il gancio lascia correre il corpo lifted). */
int anim_cap_run(struct spu_context* ctx, void (*self)(struct spu_context*));
void anim_cap_init(void);                    /* automatico (costruttore), chiamabile anche a mano */
#define SPU_ANIM_CAP_HOOK(ctx, self) \
    do { if (__builtin_expect(g_anim_cap_on, 0) && anim_cap_run((ctx), (self))) return; } while (0)

/* ---- strato A (puro) ---- */
typedef struct anim_cap_rec anim_cap_rec;
/* Apre il file (scrive l'intestazione). Ritorna 0 se ok. */
int  anim_cap_open(const char* path);
void anim_cap_close(void);
/* Prima del comando: `ls` = LS a ingresso (256 KB), `vm` = memoria principale (vm + EA), r3 = LS del descrittore, cmd_ea = r4.
 * Esegue anim_cmd_ref su una COPIA per sapere quali intervalli di memoria il comando legge/scrive, e salva lo stato.
 * Ritorna il record (da passare a anim_cap_end) o NULL se non catturabile. */
anim_cap_rec* anim_cap_begin(const uint8_t* ls, const uint8_t* vm, uint32_t r3, uint32_t cmd_ea);
/* Dopo il comando: `ls_after` = LS a uscita, `vm` = memoria: legge i byte PUT scritti dal job reale e scrive il record. */
void anim_cap_end(anim_cap_rec* r, const uint8_t* ls_after, const uint8_t* vm);
unsigned anim_cap_count(void);

/* ---- lettore (usato da anim_test replay) ---- */
typedef struct { uint32_t addr, len; const uint8_t* p; } anim_cap_span;
typedef struct {
    uint32_t seq, r3, cmd_ea, rc_pre;
    uint64_t imghash;
    uint32_t nls, nget, nexp, nlsa;
    anim_cap_span ls_pre[4], ls_post[4];
    anim_cap_span* get;                      /* nget */
    anim_cap_span* exp;                      /* nexp ; kind in exp_kind[] */
    uint32_t* exp_kind;                      /* 0 = PUT dati, 1 = PUTF completamento, 2 = linea coda (solo +10..+11 confrontato) */
} anim_cap_view;
/* Carica l'intero file in memoria. Ritorna 0 se ok. */
int  anim_cap_load_file(const char* path, uint8_t** buf, uint32_t* size);
/* Prossimo record a partire da *pos; riempie view (alloca get/exp: liberare con anim_cap_view_free); ritorna 1 = record, 0 = fine, -1 = errore.
 * `img`/`imgsize` = immagine del job (0x4000..) di quel record (chunk IMG con lo stesso hash), NULL se non trovata. */
int  anim_cap_next(uint8_t* buf, uint32_t size, uint32_t* pos, anim_cap_view* v, const uint8_t** img, uint32_t* imgsize);
void anim_cap_view_free(anim_cap_view* v);

#ifdef __cplusplus
}
#endif
#endif
