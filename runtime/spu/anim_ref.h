/* anim_ref.h -- riferimento C bit-esatto della funzione SPU 0x6248 del job wid0 (jobmod_00b730f78d78).
 * Vedi formato.md per la specifica. Codice nuovo (anim42), non fa parte di nessun worktree. */
#ifndef ANIM_REF_H
#define ANIM_REF_H
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Un comando MFC emesso dal job (nell'ordine di emissione). Per le PUT* e PUTLL* `data` punta
 * (via doff) ai byte trasferiti; per le GETL `data` e' la lista (size = byte della lista). */
typedef struct {
    uint8_t  cmd;       /* 64 GET, 68 GETL, 32 PUT, 34 PUTF, 208 GETLLAR, 180 PUTLLC, 200 PUTLLUC */
    uint8_t  tag;
    uint32_t lsa;       /* GETL: indirizzo LS di destinazione base; PUT: sorgente */
    uint32_t ea;        /* GETL: indirizzo LS della lista (EAL) */
    uint32_t size;
    uint32_t doff;      /* offset nei byte registrati (PUT, GETL) */
} anim_ev;

#define ANIM_MAX_EV 4096
typedef struct {
    anim_ev  ev[ANIM_MAX_EV];
    int      n;
    uint8_t* data;
    uint32_t ndata, cap;
    int      overflow;
} anim_fx;

void anim_fx_init(anim_fx* fx);
void anim_fx_free(anim_fx* fx);
/* Applica gli eventi PUT/PUTF/PUTLLC/PUTLLUC [from, fx->n) a una memoria scrivibile. */
void anim_fx_apply(const anim_fx* fx, int from, uint8_t* vm_rw);

/* Codici di ritorno di anim_cmd_ref */
enum {
    ANIM_OK           = 0,
    ANIM_NOT_COVERED  = 1,   /* caso non implementato (nessun effetto applicato su ls/fx e' garantito) */
    ANIM_ARENA_OVF    = 2,   /* il job farebbe `stopd` (arena oltre 0x34B00) */
    ANIM_BAD_INPUT    = 3
};

/* Elabora UN comando. `ls` = local store 256 KB con: immagine del job a 0x4000 (costanti 0x7100-0x7360),
 * descrittore da 32 B a 0x7A80, variabili globali del job (0x34B20.., tag, 0x34BA0 = EA linea coda...)
 * come lasciate dalla poll. `cmd_ea` = EA del descrittore (il job lo riceve in r4). Aggiorna `ls`
 * come farebbe il job e registra tutti i comandi MFC in `fx`. */
int anim_cmd_ref(const uint8_t* vm_base, uint32_t cmd_ea, uint8_t* ls, anim_fx* fx);
/* anim46: percorso veloce in-place, vedi anim_ref.c */
int anim_cmd_fast(uint8_t* vm, uint32_t cmd_ea, uint8_t* ls);

/* anim47: calcolo senza effetti visibili fuori dal LS. Esegue TUTTO il comando tranne la sincronizzazione della linea coda e i PUT (matrici + completamento),
 * che restano al chiamante (il gioco li fa con i canali MFC del runtime: coerenza PPU, tag, eventi, tempo). A ritorno ANIM_OK:
 * cm descrive cosa fare; l'LS e' gia' nello stato finale (0x7A90 con bit0, contatore 0x34B90 +1) meno la linea 0x7A00 (GETLLAR/PUTLLC).
 * Per rc != ANIM_OK la memoria principale non e' stata toccata (l'LS puo' essere sporca). */
typedef struct { uint32_t cmd_ea, line_ea, ea_out, mw, n, tp; } anim_commit;
int anim_cmd_compute(const uint8_t* vm, uint32_t cmd_ea, uint8_t* ls, anim_commit* cm);
/* anim47: modalita' "stretta" per il gioco (vedi d_ok in anim_ref.c) */
extern int g_anim_strict;
extern int (*g_anim_ea_ok)(uint32_t ea, uint32_t size);
extern __declspec(thread) void (*g_anim_note_cb)(uint32_t ea, uint32_t lsa, uint32_t size);

/* anim45: limite EA del banco (0 = nessuno) e chiamata diretta delle funzioni generate 0x5B78 / 0x53B8 (test unit). */
extern uint32_t g_anim_ea_limit;
int anim_gen_call(int which, const uint8_t* vm, uint8_t* ls, anim_fx* fx, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7, uint32_t a8, uint32_t sp);

#ifdef __cplusplus
}
#endif
#endif
