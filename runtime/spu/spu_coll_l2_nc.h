/* spu_coll_l2_nc.h (coll57) -- il contesto "nativo" delle funzioni generate da coll_gen.py (coll_gen_*.inc).
 * Condiviso fra il gioco (ps3recomp\runtime\spu\spu_coll_l2.c) e il banco offline (log\coll53\coll_test.c). */
#ifndef SPU_COLL_L2_NC_H
#define SPU_COLL_L2_NC_H
typedef struct NC NC;
struct NC {
    u128 R[128];
    uint8_t* ls;
    void (*wrch)(NC*, uint32_t, u128);
    u128 (*rdch)(NC*, uint32_t);
    uint32_t (*rchcnt)(NC*, uint32_t);
    uint32_t unc, uncpc;   /* bersaglio e sito dell'ultimo bisl non coperto (rc 9) */
    void* user;            /* coll57: puntatore al driver (L2*) per i callback */
};
#endif
