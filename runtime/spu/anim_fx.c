/* anim_fx.c -- registro degli effetti MFC (anim_fx) condiviso da anim_ref.c e anim_test.c. */
#include "anim_ref.h"
#include <stdlib.h>
#include <string.h>

void anim_fx_init(anim_fx* fx) { memset(fx, 0, sizeof *fx); }
void anim_fx_free(anim_fx* fx) { free(fx->data); memset(fx, 0, sizeof *fx); }

uint32_t anim_fx_addbytes(anim_fx* fx, const uint8_t* p, uint32_t n)
{
    if (fx->ndata + n > fx->cap) {
        uint32_t nc = fx->cap ? fx->cap * 2 : 65536;
        while (nc < fx->ndata + n) nc *= 2;
        fx->data = (uint8_t*)realloc(fx->data, nc);
        fx->cap = nc;
    }
    uint32_t off = fx->ndata;
    memcpy(fx->data + off, p, n);
    fx->ndata += n;
    return off;
}

int anim_fx_add(anim_fx* fx, uint8_t cmd, uint8_t tag, uint32_t lsa, uint32_t ea, uint32_t size,
                const uint8_t* bytes, uint32_t nbytes)
{
    if (fx->n >= ANIM_MAX_EV) { fx->overflow = 1; return -1; }
    anim_ev* e = &fx->ev[fx->n];
    e->cmd = cmd; e->tag = tag; e->lsa = lsa; e->ea = ea; e->size = size;
    e->doff = bytes ? anim_fx_addbytes(fx, bytes, nbytes) : 0;
    return fx->n++;
}

void anim_fx_apply(const anim_fx* fx, int from, uint8_t* vm)
{
    for (int i = from; i < fx->n; i++) {
        const anim_ev* e = &fx->ev[i];
        if (e->cmd == 32 || e->cmd == 34 || e->cmd == 180 || e->cmd == 200)
            memcpy(vm + e->ea, fx->data + e->doff, e->size);
    }
}
