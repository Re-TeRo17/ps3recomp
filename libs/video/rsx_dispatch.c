/*
 * ps3recomp - NV4097 method dispatcher (Track B / LAYER 2)
 *
 * See rsx_dispatch.h for the design and the clean-room source list.
 *
 * Register/method facts used here (envytools rnndb nv30-40_3d + Mesa nv30
 * + psdevwiki; RPCS3 consulted as a read-only semantics oracle):
 *
 *   0x0194 DMA_COLOR0, 0x018C DMA_COLOR1, 0x01B4/0x01B8 DMA_COLOR2/3,
 *   0x0198 DMA_ZETA                DMA handle 0xFEED0000 = RSX local memory,
 *                                  0xFEED0001 = main (IO-mapped) memory
 *   0x0200 RT_HORIZ  x[0:15] w[16:31]      0x0204 RT_VERT y/h
 *   0x0208 RT_FORMAT color[0:4] depth[5:7] type[8:11] aa[12:15]
 *   0x020C/021C/0280/0284 COLOR0-3_PITCH   0x0210/0218/0288/028C COLOR0-3_OFFSET
 *   0x0214 ZETA_OFFSET   0x022C ZETA_PITCH   0x0220 RT_ENABLE
 *   0x0A00 VIEWPORT_HORIZ x/w   0x0A04 VIEWPORT_VERT y/h
 *   0x0A20..0x0A2C VIEWPORT_TRANSLATE[4]  0x0A30..0x0A3C VIEWPORT_SCALE[4]
 *   0x0B80..0x0BFC VP_UPLOAD_INST window (load ptr auto-advances per vec4)
 *   0x1680+i*4 VTXBUF_OFFSET[i]  bit31 = location, [0:30] = offset
 *   0x1840+i*4 TEX_SIZE1[i] (control3) pitch[0:19] depth[20:31]
 *   0x1A00+i*0x20 fragment texture unit i (16 units, 8 words each):
 *     +0x00 TEX_OFFSET   +0x04 TEX_FORMAT dma[0:1] (1 = local, 2 = main)
 *                          cube[2] border[3] dims[4:7] fmt[8:15] mips[16:19]
 *     +0x08 TEX_WRAP     +0x0C TEX_ENABLE (control0, enable = bit 31)
 *     +0x10 TEX_SWIZZLE (component remap)  +0x14 TEX_FILTER
 *     +0x18 TEX_NPOT_SIZE height[0:15] width[16:31]  +0x1C TEX_BORDER_COLOR
 *   0x1738 VERTEX_DATA_BASE_OFFSET  0x173C VB_ELEMENT_BASE
 *   0x1740+i*4 VTXFMT[i] type[0:3] size[4:7] stride[8:15] frequency[16:31]
 *   0x1808 VERTEX_BEGIN_END  arg = primitive, 0 = end
 *   0x1814 VB_VERTEX_BATCH   first[0:23] (count-1)[24:31]
 *   0x1818 INLINE_ARRAY     vertex data pushed through the FIFO itself
 *   0x181C IDXBUF_OFFSET     0x1820 IDXBUF_FORMAT location[0:3] type[4:11]
 *   0x1824 VB_INDEX_BATCH    first[0:23] (count-1)[24:31]
 *   0x1D8C CLEAR_DEPTH_VALUE  0x1D90 CLEAR_COLOR_VALUE (A8R8G8B8)
 *   0x1D94 CLEAR_BUFFERS     Z=0x01 S=0x02 R=0x10 G=0x20 B=0x40 A=0x80
 *   0x1E9C VP_UPLOAD_FROM_ID (transform program load pointer)
 *   0x1EFC VP_UPLOAD_CONST_ID (transform constant load pointer; NOT
 *          auto-advancing — every burst re-reads it)
 *   0x1F00..0x1F7C VP_UPLOAD_CONST window: word i -> slot load + i/4, lane i%4
 *   0xE944 gcm driver-queue flip (established empirically from this title)
 */

#include "rsx_dispatch.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define M_DMA_COLOR1            0x018C
#define M_DMA_COLOR0            0x0194
#define M_DMA_ZETA              0x0198
#define M_DMA_COLOR2            0x01B4
#define M_DMA_COLOR3            0x01B8
#define M_RT_HORIZ              0x0200
#define M_RT_VERT               0x0204
#define M_RT_FORMAT             0x0208
#define M_COLOR0_PITCH          0x020C
#define M_COLOR0_OFFSET         0x0210
#define M_ZETA_OFFSET           0x0214
#define M_COLOR1_OFFSET         0x0218
#define M_COLOR1_PITCH          0x021C
#define M_RT_ENABLE             0x0220
#define M_ZETA_PITCH            0x022C
#define M_COLOR2_PITCH          0x0280
#define M_COLOR3_PITCH          0x0284
#define M_COLOR2_OFFSET         0x0288
#define M_COLOR3_OFFSET         0x028C
#define M_VIEWPORT_HORIZ        0x0A00
#define M_VIEWPORT_VERT         0x0A04
#define M_VIEWPORT_TRANSLATE    0x0A20
#define M_VIEWPORT_SCALE        0x0A30
#define M_VP_UPLOAD_INST        0x0B80
#define M_VTXBUF_OFFSET         0x1680
#define M_VERTEX_DATA_BASE      0x1738
#define M_VB_ELEMENT_BASE       0x173C
#define M_VTXFMT                0x1740
#define M_VERTEX_BEGIN_END      0x1808
#define M_VB_VERTEX_BATCH       0x1814
#define M_INLINE_ARRAY          0x1818
#define M_IDXBUF_OFFSET         0x181C
#define M_IDXBUF_FORMAT         0x1820
#define M_VB_INDEX_BATCH        0x1824
#define M_TEX_SIZE1             0x1840  /* + unit * 4  (control3)           */
#define M_TEX_OFFSET            0x1A00  /* + unit * 0x20 (8-word unit block) */
#define M_VERTEX_TEXTURE        0x0900  /* + unit * 0x20 (8-word unit block) */
#define M_FP_ACTIVE_PROGRAM     0x08E4  /* offset | dma-context [1:0]        */
#define M_FP_CONTROL            0x1D60
#define M_VP_START_FROM_ID      0x1EA0
#define M_VTX_ATTR_4F           0x1C00  /* + attr * 0x10, 4 float words      */
#define M_CLEAR_DEPTH_VALUE     0x1D8C
#define M_CLEAR_COLOR_VALUE     0x1D90
#define M_CLEAR_BUFFERS         0x1D94
#define M_VP_UPLOAD_FROM_ID     0x1E9C
#define M_VP_UPLOAD_CONST_ID    0x1EFC
#define M_VP_UPLOAD_CONST       0x1F00
#define M_GCM_FLIP_HEAD_0       0xE920
#define M_GCM_FLIP_HEAD_1       0xE924
#define M_GCM_DRIVER_FLIP       0xE944
/* Primitive restart: a sentinel index value that breaks a TRIANGLE_STRIP/
 * FAN mid-draw with no connecting triangle (RPCS3 oracle: gcm_enums.h
 * NV4097_SET_RESTART_INDEX_ENABLE/NV4097_SET_RESTART_INDEX; semantics in
 * rsx_methods.h restart_index_enabled()/restart_index() -- enabled only
 * when the enable bit is set AND, for 16-bit index buffers, the configured
 * value fits in 16 bits). */
#define M_RESTART_INDEX_ENABLE  0x1DAC
#define M_RESTART_INDEX         0x1DB0

/* Render state consumed directly by the replay harness's PSO builder
 * (rsx_dsp_reg reads straight out of the register file, so these already
 * took effect before being classified here; this fixes the coverage report
 * to match reality instead of showing them as TODO). Addresses from
 * tools/nv40_methods.py (envytools rnndb). */
#define M_ALPHA_FUNC_ENABLE     0x0304
#define M_ALPHA_FUNC_FUNC       0x0308
#define M_ALPHA_FUNC_REF        0x030C
#define M_BLEND_FUNC_ENABLE     0x0310
#define M_BLEND_FUNC_SRC        0x0314
#define M_BLEND_FUNC_DST        0x0318
#define M_COLOR_MASK            0x0324
#define M_MRT_COLOR_MASK        0x0370
#define M_DEPTH_FUNC            0x0A6C
#define M_DEPTH_WRITE_ENABLE    0x0A70
#define M_DEPTH_TEST_ENABLE     0x0A74
#define M_CULL_FACE             0x1830
#define M_FRONT_FACE            0x1834
#define M_CULL_FACE_ENABLE      0x183C
#define M_VTX_CACHE_INVALIDATE  0x1714
#define M_NV4097_NOP            0x0100
#define M_VP_ATTRIB_EN          0x1FF0
#define M_VP_RESULT_EN          0x1FF4

#define DMA_HANDLE_LOCAL        0xFEED0000u
#define DMA_HANDLE_MAIN         0xFEED0001u

u32 g_dsp_mcnt_on, g_dsp_hist_slow[0x800], g_dsp_hist_plain[0x800];   /* RSX_MCNT=1: per-method call counts */
static u32 dma_to_location(u32 handle)
{
    return handle == DMA_HANDLE_MAIN ? RSX_LOCATION_MAIN : RSX_LOCATION_LOCAL;
}

static void mark_class(rsx_dispatch* rsx, u32 method, u32 count, u8 klass)
{
    for (u32 i = 0; i < count; i++)
        rsx->klass[(method >> 2) + i] = klass;
}

/* RSX_VCKEY_MEMO: counts the changes of every register that eng_vc_key's plan and key read (the vertex array
 * offsets and formats, data base offset and element base, the constant attribute values, the index array,
 * restart index, divider mask). A write of the value a register holds does not count. Anything that writes
 * those registers behind rsx_dispatch_method (the immediate-mode fold and its VTXFMT swap, init) bumps it too. */
u32 g_rsx_dsp_vtx_gen;
static inline int dsp_vtx_tracked(u32 m)
{
    return (m >= 0x1680u && m < 0x16C0u) || (m >= 0x1738u && m < 0x1780u) || (m >= 0x1C00u && m < 0x1D00u) ||
           m == 0x181Cu || m == 0x1820u || m == 0x1DACu || m == 0x1DB0u || m == 0x1FC0u;
}

void rsx_dispatch_init(rsx_dispatch* rsx, const rsx_dispatch_sink* sink)
{
    g_rsx_dsp_vtx_gen++;
    memset(rsx, 0, sizeof(*rsx));
    if (sink)
        rsx->sink = *sink;

    /* s31 (scratch/s31_blue_emitter.md): the nv40 RESET DEFAULT for
     * COLOR_MASK is all-channels-on (RPCS3 rsx_methods.cpp:345,983 seeds
     * registers[NV4097_SET_COLOR_MASK]=0x1010101). Seeding it here makes a
     * raw register read trustworthy for every consumer: a game-written 0 is
     * a REAL "write no color channels" (the character shadow-mask
     * depth-prime pass), distinguishable from never-initialized. Without
     * the seed, consumers substituted mask=ALL for 0 and force-wrote the
     * depth-only pass's garbage FP output (the blue-character class). */
    rsx->regs[M_COLOR_MASK >> 2] = 0x01010101u;

    /* NV40 reset state is counter-clockwise front winding.  Leaving this
     * register zero made a live stream that relied on the hardware default
     * cull the exterior of orphanage meshes, while captured replay was correct
     * because its seeded register image already contained 0x901. */
    rsx->regs[M_FRONT_FACE >> 2] = 0x0901u;

    /* NV40 reset ZSTENCIL_CLEAR_VALUE (Z24S8: depth24<<8 | stencil8) is
     * 0xFFFFFF00 = depth 1.0, stencil 0 (RPCS3 rsx_methods.cpp reset image
     * as numbering oracle). Seeded so a raw register read is trustworthy for
     * clear consumers, mirroring COLOR_MASK/FRONT_FACE above. */
    rsx->regs[0x1D8C >> 2] = 0xFFFFFF00u;

    /* Execution methods */
    mark_class(rsx, M_VERTEX_BEGIN_END,  1, RSX_DSP_CLASS_EXEC);
    mark_class(rsx, M_VB_VERTEX_BATCH,   1, RSX_DSP_CLASS_EXEC);
    mark_class(rsx, M_INLINE_ARRAY,      1, RSX_DSP_CLASS_EXEC);
    mark_class(rsx, M_VB_INDEX_BATCH,    1, RSX_DSP_CLASS_EXEC);
    mark_class(rsx, M_CLEAR_BUFFERS,     1, RSX_DSP_CLASS_EXEC);
    mark_class(rsx, M_VP_UPLOAD_INST,   32, RSX_DSP_CLASS_EXEC);
    mark_class(rsx, M_VP_UPLOAD_CONST,  32, RSX_DSP_CLASS_EXEC);
    mark_class(rsx, M_GCM_FLIP_HEAD_0,   1, RSX_DSP_CLASS_EXEC);
    mark_class(rsx, M_GCM_FLIP_HEAD_1,   1, RSX_DSP_CLASS_EXEC);
    mark_class(rsx, M_GCM_DRIVER_FLIP,   1, RSX_DSP_CLASS_EXEC);

    /* State the getters decode */
    mark_class(rsx, M_DMA_COLOR1,        1, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_DMA_COLOR0,        2, RSX_DSP_CLASS_STATE); /* +ZETA */
    mark_class(rsx, M_DMA_COLOR2,        2, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_RT_HORIZ,         12, RSX_DSP_CLASS_STATE); /* 0x200..0x22C */
    mark_class(rsx, M_COLOR2_PITCH,      4, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_VIEWPORT_HORIZ,    2, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_VIEWPORT_TRANSLATE, 8, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_VTXBUF_OFFSET,    16, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_VERTEX_DATA_BASE,  2, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_VTXFMT,           16, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_IDXBUF_OFFSET,     2, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_TEX_SIZE1,        16, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_TEX_OFFSET, RSX_DSP_NUM_TEXTURES * 8, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_CLEAR_DEPTH_VALUE, 2, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_VP_UPLOAD_FROM_ID, 1, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_RESTART_INDEX_ENABLE, 2, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_VP_UPLOAD_CONST_ID, 1, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_FP_ACTIVE_PROGRAM, 1, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_FP_CONTROL,        1, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_VP_START_FROM_ID,  1, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_VTX_ATTR_4F,      64, RSX_DSP_CLASS_STATE);

    /* Alpha test / blend / color-mask / depth / cull: rsx_dsp_reg reads these
     * straight from the register file (replay_main.c's decode_render_state,
     * color-mask application), so they were already live; this only corrects
     * the coverage report. */
    mark_class(rsx, M_ALPHA_FUNC_ENABLE, 3, RSX_DSP_CLASS_STATE); /* ..FUNC_REF */
    mark_class(rsx, M_BLEND_FUNC_ENABLE, 5, RSX_DSP_CLASS_STATE); /* ..BLEND_EQUATION */
    mark_class(rsx, M_COLOR_MASK,        1, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_MRT_COLOR_MASK,    1, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_DEPTH_FUNC,        3, RSX_DSP_CLASS_STATE); /* ..TEST_ENABLE */
    mark_class(rsx, M_CULL_FACE,         1, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_FRONT_FACE,        1, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_CULL_FACE_ENABLE,  1, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_VP_ATTRIB_EN,      1, RSX_DSP_CLASS_STATE);
    mark_class(rsx, M_VP_RESULT_EN,      1, RSX_DSP_CLASS_STATE);

    /* Explicit no-ops: the vertex cache invalidate hint and the reserved
     * NOP method carry no state the replay harness needs (there is no CPU
     * vertex cache to invalidate; every draw re-fetches from guest memory
     * each time). Classifying them EXEC removes them from the TODO count
     * without pretending they do anything. */
    mark_class(rsx, M_VTX_CACHE_INVALIDATE, 1, RSX_DSP_CLASS_EXEC);
    mark_class(rsx, M_NV4097_NOP,           1, RSX_DSP_CLASS_EXEC);
}

void rsx_dispatch_seed_registers(rsx_dispatch* rsx, const u32* regs, u32 count)
{
    if (count > RSX_DSP_NUM_REGS)
        count = RSX_DSP_NUM_REGS;
    memcpy(rsx->regs, regs, count * sizeof(u32));
}

void rsx_dispatch_seed_transform_program(rsx_dispatch* rsx, const u32* words, u32 count)
{
    if (count > RSX_DSP_VP_WORDS)
        count = RSX_DSP_VP_WORDS;
    memcpy(rsx->vp, words, count * sizeof(u32));
    rsx->vp_gen++;
}

volatile u32 g_rsx_canvas_mark;

/* A constant word written: is it the mark, or the word that held it? */
static inline void dsp_canvas_mark(rsx_dispatch* rsx, u32 slot, u32 c, u32 v)
{
    const u32 at = slot * 4u + c + 1u;
    if (v == g_rsx_canvas_mark) rsx->canvas_mark_at = at;
    else if (rsx->canvas_mark_at == at) rsx->canvas_mark_at = 0;
}

void rsx_dispatch_seed_transform_constants(rsx_dispatch* rsx, const u32* words, u32 count)
{
    const u32 max_words = RSX_DSP_NUM_CONSTANTS * 4;
    if (count > max_words)
        count = max_words;
    memcpy(rsx->constants, words, count * sizeof(u32));
    rsx->const_gen++;
    rsx->const_lo = 0; rsx->const_hi = RSX_DSP_NUM_CONSTANTS - 1u;
    rsx->canvas_mark_at = 0;
    if (g_rsx_canvas_mark)
        for (u32 i = 0; i < count; i++) dsp_canvas_mark(rsx, i >> 2, i & 3u, words[i]);
}

/* Every NV4097_SET_VERTEX_DATA* variant writes the same constant ("current")
 * attribute register a disabled array feeds. Only the 4F form was decoded, so a
 * colour set with SetVertexData4ub read as the (0,0,0,1) default: Tornado
 * Outbreak's UI draws its vertex colour that way, alpha 0 failed the alpha test
 * on every glyph, and 58 draws a frame reached an all-black screen. Fold each
 * variant into the 4F slots rsx_dsp_vertex_default reads. Offsets and packing
 * from the SDK (gcm_implementation_sub.h; x is the LOW byte/half). */
static void vtx_const_set(rsx_dispatch* rsx, u32 attr, u32 c, float v)
{
    u32 w; memcpy(&w, &v, 4);
    rsx->regs[(M_VTX_ATTR_4F + attr * 0x10 + c * 4) >> 2] = w;
}

static void vtx_const_fold(rsx_dispatch* rsx, u32 m, u32 arg)
{
    float f;
    u32 attr, word;
    if (m >= 0x1E40 && m < 0x1E80) {                  /* DATA1F: (x,0,0,1) */
        attr = (m - 0x1E40) >> 2; memcpy(&f, &arg, 4);
        vtx_const_set(rsx, attr, 0, f); vtx_const_set(rsx, attr, 1, 0.0f);
        vtx_const_set(rsx, attr, 2, 0.0f); vtx_const_set(rsx, attr, 3, 1.0f);
    } else if (m >= 0x1880 && m < 0x1900) {           /* DATA2F: (x,y,0,1) */
        attr = (m - 0x1880) >> 3; word = ((m - 0x1880) >> 2) & 1; memcpy(&f, &arg, 4);
        vtx_const_set(rsx, attr, word, f);
        if (word == 0) { vtx_const_set(rsx, attr, 2, 0.0f); vtx_const_set(rsx, attr, 3, 1.0f); }
    } else if (m >= 0x1500 && m < 0x1600) {           /* DATA3F: (x,y,z,1) */
        attr = (m - 0x1500) >> 4; word = ((m - 0x1500) >> 2) & 3; memcpy(&f, &arg, 4);
        if (word < 3) vtx_const_set(rsx, attr, word, f);
        if (word == 0) vtx_const_set(rsx, attr, 3, 1.0f);
    } else if (m >= 0x1900 && m < 0x1940) {           /* DATA2S: (x,y,0,1), unnormalised */
        attr = (m - 0x1900) >> 2;
        vtx_const_set(rsx, attr, 0, (float)(int16_t)(arg & 0xFFFF));
        vtx_const_set(rsx, attr, 1, (float)(int16_t)(arg >> 16));
        vtx_const_set(rsx, attr, 2, 0.0f); vtx_const_set(rsx, attr, 3, 1.0f);
    } else if (m >= 0x1940 && m < 0x1980) {           /* DATA4UB: normalised bytes */
        attr = (m - 0x1940) >> 2;
        for (u32 c = 0; c < 4; c++)
            vtx_const_set(rsx, attr, c, (float)((arg >> (8 * c)) & 0xFF) / 255.0f);
    } else if ((m >= 0x1980 && m < 0x1A00) ||         /* DATA4S: unnormalised */
               (m >= 0x0A80 && m < 0x0B00)) {         /* DATA_SCALED4S: normalised */
        const u32 base = m >= 0x1980 ? 0x1980u : 0x0A80u;
        const float k = m >= 0x1980 ? 1.0f : 1.0f / 32767.0f;
        attr = (m - base) >> 3; word = ((m - base) >> 2) & 1;
        vtx_const_set(rsx, attr, word * 2 + 0, (float)(int16_t)(arg & 0xFFFF) * k);
        vtx_const_set(rsx, attr, word * 2 + 1, (float)(int16_t)(arg >> 16) * k);
    }
}

/* ---------------------------------------------------------------------------
 * Immediate-mode vertices.
 *
 * A title that never sets up a vertex array can draw straight out of the
 * "current attribute" registers: BEGIN_END(prim), then per vertex a few
 * SET_VERTEX_DATA{1F,2F,3F,4F,2S,4S,4UB} writes ending with the one that
 * completes attribute 0 (the position), which is what makes the hardware emit
 * a vertex -- a snapshot of every current attribute at that moment -- then
 * BEGIN_END(0). Ratchet & Clank: Tools of Destruction draws every Bink movie
 * frame this way (a full-screen QUADS with 2S position/texcoord and 4F colour
 * writes), so the dispatcher saw begin/end with nothing between them and the
 * movie never reached the screen.
 *
 * At END the snapshots are packed into the INLINE_ARRAY stream the sinks
 * already understand (float4 per attribute written during the primitive, in
 * ascending attribute order) and the VTXFMT registers are overridden to
 * describe exactly that stream until the sink has drawn it.
 *
 * The buffer is module-static: one dispatcher draws at a time.
 * -----------------------------------------------------------------------*/
#define IMM_MAX_VERTS 2048
static float s_imm_vtx[IMM_MAX_VERTS][RSX_DSP_NUM_VERTEX_ATTR][4];
static u32   s_imm_n;
static u32   s_imm_touched;       /* attributes written during this primitive */
static u32   s_imm_overflow;
static u32   s_imm_batches;       /* VB_VERTEX/INDEX_BATCH draws during this primitive */

static void imm_reset(void) { s_imm_n = 0; s_imm_touched = 0; s_imm_overflow = 0; s_imm_batches = 0; }

/* Which attribute a DATA* method writes, and whether this word completes it. */
static int imm_classify(u32 m, u32* attr, int* completes)
{
    if (m >= 0x1E40 && m < 0x1E80) { *attr = (m - 0x1E40) >> 2; *completes = 1; return 1; }                 /* 1F  */
    if (m >= 0x1880 && m < 0x1900) { *attr = (m - 0x1880) >> 3; *completes = (m & 4) != 0; return 1; }      /* 2F  */
    if (m >= 0x1500 && m < 0x1600) { *attr = (m - 0x1500) >> 4; *completes = ((m >> 2) & 3) == 2; return 1; } /* 3F  */
    if (m >= 0x1C00 && m < 0x1D00) { *attr = (m - 0x1C00) >> 4; *completes = ((m >> 2) & 3) == 3; return 1; } /* 4F  */
    if (m >= 0x1900 && m < 0x1940) { *attr = (m - 0x1900) >> 2; *completes = 1; return 1; }                 /* 2S  */
    if (m >= 0x1940 && m < 0x1980) { *attr = (m - 0x1940) >> 2; *completes = 1; return 1; }                 /* 4UB */
    if (m >= 0x1980 && m < 0x1A00) { *attr = (m - 0x1980) >> 3; *completes = (m & 4) != 0; return 1; }      /* 4S  */
    if (m >= 0x0A80 && m < 0x0B00) { *attr = (m - 0x0A80) >> 3; *completes = (m & 4) != 0; return 1; }      /* scaled 4S */
    return 0;
}

static void imm_note(rsx_dispatch* rsx, u32 m)
{
    u32 attr; int completes;
    if (!imm_classify(m, &attr, &completes) || attr >= RSX_DSP_NUM_VERTEX_ATTR) return;
    s_imm_touched |= 1u << attr;
    if (attr != 0 || !completes) return;
    if (s_imm_n >= IMM_MAX_VERTS) { s_imm_overflow = 1; return; }
    for (u32 a = 0; a < RSX_DSP_NUM_VERTEX_ATTR; a++) {
        u32 w[4];
        for (u32 c = 0; c < 4; c++) w[c] = rsx->regs[(M_VTX_ATTR_4F + a * 0x10 + c * 4) >> 2];
        if (!(w[0] | w[1] | w[2] | w[3])) { w[3] = 0x3F800000u; }   /* never written: (0,0,0,1) */
        memcpy(s_imm_vtx[s_imm_n][a], w, 16);
    }
    s_imm_n++;
}

/* Pack the snapshots into rsx->inline_data and override VTXFMT to match.
 * Returns 1 when the registers were overridden (caller restores `saved`). */
static int imm_flush(rsx_dispatch* rsx, u32 saved[RSX_DSP_NUM_VERTEX_ATTR])
{
    static int log = -1; if (log < 0) log = getenv("RSX_IMM_LOG") ? 1 : 0;
    if (!s_imm_n || rsx->inline_len) return 0;
    /* The primitive drew from its arrays: the DATA writes only set current
     * values. Whether attribute 0 has an array declared says nothing -- ToD's
     * Anark HUD never touches VTXFMT, so the 3D scene's declaration is still
     * there, and testing it dropped every HUD quad drawn after a 3D pass
     * (which panel showed depended on what drew before it). */
    if (s_imm_batches) return 0;
    u32 att[RSX_DSP_NUM_VERTEX_ATTR], nt = 0;
    for (u32 a = 0; a < RSX_DSP_NUM_VERTEX_ATTR; a++)
        if (s_imm_touched & (1u << a)) att[nt++] = a;
    if (!nt) return 0;
    if (s_imm_overflow || (u64)s_imm_n * nt * 16u > RSX_DSP_INLINE_MAX_BYTES) { rsx->inline_dropped++; return 0; }
    u32 off = 0;
    for (u32 k = 0; k < s_imm_n; k++)
        for (u32 t = 0; t < nt; t++) {
            u32 w[4]; memcpy(w, s_imm_vtx[k][att[t]], 16);
            for (u32 c = 0; c < 4; c++, off += 4) {
                rsx->inline_data[off + 0] = (u8)(w[c] >> 24); rsx->inline_data[off + 1] = (u8)(w[c] >> 16);
                rsx->inline_data[off + 2] = (u8)(w[c] >> 8);  rsx->inline_data[off + 3] = (u8)w[c];
            }
        }
    rsx->inline_len = off;
    for (u32 a = 0; a < RSX_DSP_NUM_VERTEX_ATTR; a++) {
        saved[a] = rsx->regs[(M_VTXFMT >> 2) + a];
        /* type 2 (float), size 4, stride 0 = tightly packed; 0x2 = disabled */
        rsx->regs[(M_VTXFMT >> 2) + a] = (s_imm_touched & (1u << a)) ? 0x42u : 0x2u;
    }
    g_rsx_dsp_vtx_gen++;
    if (log) { static int n = 0; if (n < 40 && ++n)
        fprintf(stderr, "[imm] prim=%u verts=%u attrs=0x%04X bytes=%u v0=(%g %g %g %g)\n", rsx->current_primitive,
                s_imm_n, s_imm_touched, off, s_imm_vtx[0][0][0], s_imm_vtx[0][0][1], s_imm_vtx[0][0][2], s_imm_vtx[0][0][3]); }
    return 1;
}

void rsx_dispatch_method(rsx_dispatch* rsx, u32 method, u32 arg)
{
    method &= 0xFFFFC;
    if (g_dsp_mcnt_on) g_dsp_hist_slow[(method >> 2) & 0x7FF]++;
    const u32 idx = method >> 2;
    /* regs[] covers methods 0..0xFFFC (subchannel bits included). A larger
     * value is not a method -- and indexing with it wrote past regs[] into
     * vp[] and constants[]: Tornado Outbreak's c[461] held two copies of a
     * "load constant 461" command header, i.e. FIFO words stored as shader
     * constants. Drop it, and say what it was. */
    if (idx >= RSX_DSP_NUM_REGS) {
        static int n = 0;
        if (n < 8 && ++n)
            fprintf(stderr, "[rsx-dispatch] method 0x%05X (arg 0x%08X) out of range -- dropped\n",
                    method, arg);
        return;
    }
    if (dsp_vtx_tracked(method) && rsx->regs[idx] != arg) g_rsx_dsp_vtx_gen++;
    rsx->regs[idx] = arg;
    if ((method >= 0x1500 && method < 0x1A00) || (method >= 0x1E40 && method < 0x1E80) ||
        (method >= 0x0A80 && method < 0x0B00)) {
        vtx_const_fold(rsx, method, arg);
        g_rsx_dsp_vtx_gen++;
    }
    if (rsx->in_begin_end) imm_note(rsx, method);

    /* Transform program upload window: word goes to instruction slot
     * VP_UPLOAD_FROM_ID; the load pointer advances after every completed
     * vec4 (hardware auto-increment, confirmed against the RPCS3 oracle). */
    if (method >= M_VP_UPLOAD_INST && method < M_VP_UPLOAD_INST + 32 * 4) {
        const u32 lane = ((method - M_VP_UPLOAD_INST) >> 2) & 3;
        u32* load = &rsx->regs[M_VP_UPLOAD_FROM_ID >> 2];
        const u32 pos = *load * 4 + lane;
        if (pos < RSX_DSP_VP_WORDS) {
            rsx->vp[pos] = arg;
            rsx->vp_gen++;
        }
        if (lane == 3)
            (*load)++;
        return;
    }

    /* Transform constant upload window: word i of the burst writes
     * constants[VP_UPLOAD_CONST_ID + i/4][i%4]; the load pointer does NOT
     * advance. */
    if (method >= M_VP_UPLOAD_CONST && method < M_VP_UPLOAD_CONST + 32 * 4) {
        const u32 word = (method - M_VP_UPLOAD_CONST) >> 2;
        const u32 slot = rsx->regs[M_VP_UPLOAD_CONST_ID >> 2] + (word >> 2);
        if (slot < RSX_DSP_NUM_CONSTANTS) {
            rsx->constants[slot][word & 3] = arg;
            rsx->const_gen++;
            if (slot < rsx->const_lo) rsx->const_lo = slot;
            if (slot > rsx->const_hi) rsx->const_hi = slot;
            if (g_rsx_canvas_mark) dsp_canvas_mark(rsx, slot, word & 3, arg);
        }
        /* RSX_CONST_LOG=<lo>,<hi>: every transform-constant upload into that
         * slot range (first 400) -- what a shader's constants really were. */
        /* RSX_CONST_LOG_FRAME=<n>: only while the draw engine's frame counter
         * is n or n+1 (the cap used to be spent on the title screen);
         * RSX_CONST_LOG_MAX=<lines> raises the 400-line cap. */
        { static int lo = -2, hi = 0; static long n = 0, cap = 400, from = -1;
          extern uint32_t g_rsx_engine_frame;
          if (lo == -2) { const char* e = getenv("RSX_CONST_LOG"); lo = -1;
                          if (e) sscanf(e, "%d,%d", &lo, &hi);
                          if ((e = getenv("RSX_CONST_LOG_MAX"))) cap = atol(e);
                          if ((e = getenv("RSX_CONST_LOG_FRAME"))) from = atol(e); }
          if (lo >= 0 && (int)slot >= lo && (int)slot <= hi && n < cap &&
              (from < 0 || ((long)g_rsx_engine_frame >= from && (long)g_rsx_engine_frame < from + 2))) { n++;
              float fv; memcpy(&fv, &arg, 4);
              fprintf(stderr, "[const-load] f%u c[%u].%c = 0x%08X (%.4g) (method 0x%04X id=%u)\n", g_rsx_engine_frame, slot,
                      "xyzw"[word & 3], arg, fv, method, rsx->regs[M_VP_UPLOAD_CONST_ID >> 2]); } }
        return;
    }

    switch (method) {
    case M_CLEAR_BUFFERS:
        if (rsx->sink.clear)
            rsx->sink.clear(rsx->sink.user, rsx, arg);
        break;

    case M_VERTEX_BEGIN_END:
        if (arg) {
            rsx->in_begin_end = 1;
            rsx->current_primitive = arg;
            rsx->inline_len = 0;
            imm_reset();
            if (rsx->sink.begin)
                rsx->sink.begin(rsx->sink.user, rsx, arg);
        } else {
            rsx->in_begin_end = 0;
            /* Immediate-mode vertices become an inline stream (see imm_flush). */
            u32 imm_saved[RSX_DSP_NUM_VERTEX_ATTR];
            const int imm = imm_flush(rsx, imm_saved);
            /* Hand an inline stream over BEFORE end: the sink reads end as
             * "draw what you have", so a batch delivered after it would be
             * drawn one primitive late, or not at all. */
            if (rsx->inline_len && rsx->sink.inline_array)
                rsx->sink.inline_array(rsx->sink.user, rsx,
                                       rsx->inline_data, rsx->inline_len);
            rsx->inline_len = 0;
            if (rsx->sink.end)
                rsx->sink.end(rsx->sink.user, rsx);
            if (imm)
                {
                    for (u32 a = 0; a < RSX_DSP_NUM_VERTEX_ATTR; a++)
                        rsx->regs[(M_VTXFMT >> 2) + a] = imm_saved[a];
                    g_rsx_dsp_vtx_gen++;
                }
            imm_reset();
        }
        break;

    /* INLINE_ARRAY: the vertex stream travels IN the pushbuffer rather than
     * being fetched from a vertex array, so there is no (first, count) --
     * every word between BEGIN_END(prim) and BEGIN_END(0) is vertex data.
     * The layout is still the ordinary VTXFMT declaration: enabled
     * attributes packed in ascending index order, one vertex every stride
     * bytes. Store the words back in guest byte order so the consumer
     * decodes them with the same big-endian readers a vertex array uses;
     * the dispatcher never interprets them.
     *
     * A title that builds geometry this way (Guitar Hero III draws its
     * entire 2D layer as inline QUADS) previously issued draws that arrived
     * carrying no vertices at all: the backend saw BEGIN/END with nothing
     * between them and counted an empty group. */
    case M_INLINE_ARRAY:
        if (rsx->inline_len + 4u <= RSX_DSP_INLINE_MAX_BYTES) {
            rsx->inline_data[rsx->inline_len + 0] = (u8)(arg >> 24);
            rsx->inline_data[rsx->inline_len + 1] = (u8)(arg >> 16);
            rsx->inline_data[rsx->inline_len + 2] = (u8)(arg >> 8);
            rsx->inline_data[rsx->inline_len + 3] = (u8)arg;
            rsx->inline_len += 4;
        } else if (rsx->inline_len) {
            /* Truncating leaves a torn final vertex; drop the whole stream
             * and count it instead of drawing something half-read. */
            rsx->inline_len = 0;
            rsx->inline_dropped++;
        }
        break;

    case M_VB_VERTEX_BATCH:
        s_imm_batches++;
        if (rsx->sink.draw_arrays)
            rsx->sink.draw_arrays(rsx->sink.user, rsx,
                                  arg & 0xFFFFFF, (arg >> 24) + 1);
        break;

    case M_VB_INDEX_BATCH:
        s_imm_batches++;
        if (rsx->sink.draw_index_array)
            rsx->sink.draw_index_array(rsx->sink.user, rsx,
                                       arg & 0xFFFFFF, (arg >> 24) + 1);
        break;

    case M_GCM_FLIP_HEAD_0:
    case M_GCM_FLIP_HEAD_1:
    case M_GCM_DRIVER_FLIP:
        if (rsx->sink.flip)
            rsx->sink.flip(rsx->sink.user, rsx, arg);
        break;

    default:
        break;
    }
}

/* A run of consecutive words into the transform constant or transform
 * program upload window, big-endian as the FIFO holds them: what
 * rsx_dispatch_method does with each, in one call (the walker's bulk path --
 * these windows are most of a frame's words). 0 when the run is not wholly
 * inside one window, or RSX_CONST_LOG wants to see each word. */
static inline u32 dsp_be32(const u8* p)
{
    return ((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | (u32)p[3];
}

int rsx_dispatch_upload_run(rsx_dispatch* rsx, u32 method, u32 count, const u8* be)
{
    static int log = -1;
    if (log < 0) log = getenv("RSX_CONST_LOG") ? 1 : 0;
    if (log || !count) return 0;
    const u32 end = method + count * 4u;
    if (method >= M_VP_UPLOAD_CONST && end <= M_VP_UPLOAD_CONST + 32u * 4u) {
        const u32 slot0 = rsx->regs[M_VP_UPLOAD_CONST_ID >> 2];
        const u32 mark = g_rsx_canvas_mark;
        u32 word = (method - M_VP_UPLOAD_CONST) >> 2, written = 0;
        for (u32 i = 0; i < count; i++, word++) {
            const u32 v = dsp_be32(be + i * 4u);
            rsx->regs[(method >> 2) + i] = v;
            const u32 slot = slot0 + (word >> 2);
            if (slot < RSX_DSP_NUM_CONSTANTS) {
                rsx->constants[slot][word & 3] = v;
                written++;
                if (slot < rsx->const_lo) rsx->const_lo = slot;
                if (slot > rsx->const_hi) rsx->const_hi = slot;
                if (mark) dsp_canvas_mark(rsx, slot, word & 3, v);
            }
        }
        rsx->const_gen += written;
        return 1;
    }
    if (method >= M_VP_UPLOAD_INST && end <= M_VP_UPLOAD_INST + 32u * 4u) {
        u32* load = &rsx->regs[M_VP_UPLOAD_FROM_ID >> 2];
        for (u32 i = 0; i < count; i++) {
            const u32 m = method + i * 4u, v = dsp_be32(be + i * 4u);
            rsx->regs[m >> 2] = v;
            const u32 lane = ((m - M_VP_UPLOAD_INST) >> 2) & 3;
            const u32 pos = *load * 4 + lane;
            if (pos < RSX_DSP_VP_WORDS) {
                rsx->vp[pos] = v;
                rsx->vp_gen++;
            }
            if (lane == 3)
                (*load)++;
        }
        return 1;
    }
    return 0;
}

/* RSX_FIFO_FAST=1: a run of consecutive words into registers that the dispatcher only stores -- the texture
 * units (0x1A00-0x1BFF: offset, format, address, control, filter, rect, border), the vertex array offsets
 * (0x1680-0x16BF) and formats (0x1740-0x177F). rsx_dispatch_method does nothing else for them: no fold
 * (none of the branches of vtx_const_fold matches these addresses), no upload window,
 * no case of the switch; imm_note only inside BEGIN/END, which is why a run in one is refused. What the engine
 * does with a texture or an array is at the draw, from regs[]. 0: not such a run, the caller does it word by word. */
int rsx_dispatch_plain_run(rsx_dispatch* rsx, u32 method, u32 count, const u8* be)
{
    if (!count || rsx->in_begin_end) return 0;
    const u32 end = method + count * 4u;
    if (!((method >= 0x1A00u && end <= 0x1C00u) || (method >= 0x1680u && end <= 0x16C0u) ||
          (method >= 0x1740u && end <= 0x1780u)))
        return 0;
    u32* r = &rsx->regs[method >> 2];
    for (u32 i = 0; i < count; i++) {
        const u32 v = dsp_be32(be + i * 4u);
        if (dsp_vtx_tracked(method + i * 4u) && r[i] != v) g_rsx_dsp_vtx_gen++;
        r[i] = v;
    }
    return 1;
}

/* RSX_FIFO_BLOCK / RSX_FIFO_RUN: methods that rsx_dispatch_method (and the engine's rsx_draw_engine_method in front of
 * it) does nothing for but store the word and count a change of a vertex-tracked register: not a constant fold,
 * not an upload window, not a case of the switch, not an engine-side special (queries, semaphore release,
 * snapshot stats, branch bits). Built once; a run of them outside BEGIN/END is stored in one loop. */
static unsigned char s_plain[0x800];
static int s_plain_ready;
static void dsp_plain_build(void)
{
    for (u32 i = 0; i < 0x800; i++) {
        const u32 m = i << 2;
        int ok = 1;
        if (m >= 0x1500u && m < 0x1600u) ok = 0;
        if (m >= 0x1880u && m < 0x1A00u) ok = 0;
        if (m >= 0x1E40u && m < 0x1E80u) ok = 0;
        if (m >= 0x0A80u && m < 0x0B00u) ok = 0;
        if (m >= 0x0B80u && m < 0x0C00u) ok = 0;       /* VP program upload */
        if (m >= 0x1F00u && m < 0x1F80u) ok = 0;       /* VP constant upload */
        switch (m) {
        case 0x1808: case 0x1814: case 0x1818: case 0x1824: case 0x1D94:      /* begin/end, batches, inline, clear */
        case 0x17C8: case 0x1800: case 0x1D70: case 0x1D84: case 0x1FD8: case 0x1FF8:   /* engine specials */
        case 0x1D6C: case 0x1D74: case 0x1A8:
            ok = 0; break;
        default: break;
        }
        if (m < 0x70u) ok = 0;                         /* NV406E (reference, semaphores) */
        s_plain[i] = (unsigned char)(ok ? (dsp_vtx_tracked(m) ? 2 : 1) : 0);
    }
    s_plain_ready = 1;
}

/* Stores the leading plain words of the run into regs[]; returns how many it took (0..count). A run inside
 * BEGIN/END is refused (imm_note wants to see each word). */
u32 rsx_dispatch_state_run(rsx_dispatch* rsx, u32 method, u32 count, int incr, const u8* be)
{
    if (rsx->in_begin_end) return 0;
    if (!s_plain_ready) dsp_plain_build();
    u32 i = 0;
    if (incr) {
        /* one pass over a consecutive run: class byte per register (1 plain, 2 plain + vertex-tracked) */
        if (method >= 0x2000u || method + count * 4u > 0x2000u) {
            for (; i < count; i++) {
                const u32 m = method + i * 4u;
                if (m >= 0x2000u || !s_plain[m >> 2]) break;
                const u32 v = dsp_be32(be + i * 4u);
                u32* r = &rsx->regs[m >> 2];
                if (dsp_vtx_tracked(m) && *r != v) g_rsx_dsp_vtx_gen++;
                *r = v;
            }
            return i;
        }
        const unsigned char* cl = &s_plain[method >> 2];
        u32* r = &rsx->regs[method >> 2];
        u32 chg = 0;
        for (; i < count; i++) {
            const unsigned c = cl[i];
            if (!c) break;
            u32 v; memcpy(&v, be + i * 4u, 4);
            v = __builtin_bswap32(v);
            chg += (c == 2u) & (r[i] != v);
        if (g_dsp_mcnt_on) g_dsp_hist_plain[((method >> 2) + i) & 0x7FF]++;
            r[i] = v;
        }
        if (chg) g_rsx_dsp_vtx_gen++;
        return i;
    }
    for (; i < count; i++) {
        const u32 m = method;
        if (m >= 0x2000u || !s_plain[m >> 2]) break;
        const u32 v = dsp_be32(be + i * 4u);
        u32* r = &rsx->regs[m >> 2];
        if (dsp_vtx_tracked(m) && *r != v) g_rsx_dsp_vtx_gen++;
        *r = v;
    }
    return i;
}

/* ---- getters ----------------------------------------------------------- */

u32 rsx_dsp_clear_color(const rsx_dispatch* rsx)
{
    return rsx_dsp_reg(rsx, M_CLEAR_COLOR_VALUE);
}

u32 rsx_dsp_clear_zstencil(const rsx_dispatch* rsx)
{
    return rsx_dsp_reg(rsx, M_CLEAR_DEPTH_VALUE);
}

void rsx_dsp_get_surface(const rsx_dispatch* rsx, rsx_dsp_surface* out)
{
    const u32 fmt = rsx_dsp_reg(rsx, M_RT_FORMAT);
    out->color_format = fmt & 0x1F;
    out->depth_format = (fmt >> 5) & 7;
    out->raster_type  = (fmt >> 8) & 0xF;

    const u32 h = rsx_dsp_reg(rsx, M_RT_HORIZ);
    const u32 v = rsx_dsp_reg(rsx, M_RT_VERT);
    out->clip_x = h & 0xFFFF;
    out->clip_w = h >> 16;
    out->clip_y = v & 0xFFFF;
    out->clip_h = v >> 16;

    out->color_offset[0] = rsx_dsp_reg(rsx, M_COLOR0_OFFSET);
    out->color_offset[1] = rsx_dsp_reg(rsx, M_COLOR1_OFFSET);
    out->color_offset[2] = rsx_dsp_reg(rsx, M_COLOR2_OFFSET);
    out->color_offset[3] = rsx_dsp_reg(rsx, M_COLOR3_OFFSET);
    out->color_pitch[0]  = rsx_dsp_reg(rsx, M_COLOR0_PITCH);
    out->color_pitch[1]  = rsx_dsp_reg(rsx, M_COLOR1_PITCH);
    out->color_pitch[2]  = rsx_dsp_reg(rsx, M_COLOR2_PITCH);
    out->color_pitch[3]  = rsx_dsp_reg(rsx, M_COLOR3_PITCH);
    out->color_location[0] = dma_to_location(rsx_dsp_reg(rsx, M_DMA_COLOR0));
    out->color_location[1] = dma_to_location(rsx_dsp_reg(rsx, M_DMA_COLOR1));
    out->color_location[2] = dma_to_location(rsx_dsp_reg(rsx, M_DMA_COLOR2));
    out->color_location[3] = dma_to_location(rsx_dsp_reg(rsx, M_DMA_COLOR3));
    out->color_target  = rsx_dsp_reg(rsx, M_RT_ENABLE);
    out->zeta_offset   = rsx_dsp_reg(rsx, M_ZETA_OFFSET);
    out->zeta_pitch    = rsx_dsp_reg(rsx, M_ZETA_PITCH);
    out->zeta_location = dma_to_location(rsx_dsp_reg(rsx, M_DMA_ZETA));
}

void rsx_dsp_get_viewport(const rsx_dispatch* rsx, rsx_dsp_viewport* out)
{
    const u32 h = rsx_dsp_reg(rsx, M_VIEWPORT_HORIZ);
    const u32 v = rsx_dsp_reg(rsx, M_VIEWPORT_VERT);
    out->x = h & 0xFFFF;
    out->w = h >> 16;
    out->y = v & 0xFFFF;
    out->h = v >> 16;
    for (u32 i = 0; i < 4; i++) {
        const u32 t = rsx_dsp_reg(rsx, M_VIEWPORT_TRANSLATE + i * 4);
        const u32 s = rsx_dsp_reg(rsx, M_VIEWPORT_SCALE + i * 4);
        memcpy(&out->translate[i], &t, 4);
        memcpy(&out->scale[i], &s, 4);
    }
}

void rsx_dsp_get_vertex_attr(const rsx_dispatch* rsx, u32 index, rsx_dsp_vertex_attr* out)
{
    memset(out, 0, sizeof(*out));
    if (index >= RSX_DSP_NUM_VERTEX_ATTR)
        return;
    const u32 fmt = rsx_dsp_reg(rsx, M_VTXFMT + index * 4);
    const u32 off = rsx_dsp_reg(rsx, M_VTXBUF_OFFSET + index * 4);
    out->type      = fmt & 7;
    out->size      = (fmt >> 4) & 0xF;
    out->stride    = (fmt >> 8) & 0xFF;
    out->frequency = fmt >> 16;
    out->offset    = off & 0x7FFFFFFF;
    out->location  = off >> 31;
}

void rsx_dsp_get_texture(const rsx_dispatch* rsx, u32 unit, rsx_dsp_texture* out)
{
    memset(out, 0, sizeof(*out));
    if (unit >= RSX_DSP_NUM_TEXTURES)
        return;
    const u32 base = M_TEX_OFFSET + unit * 0x20;
    const u32 fmt  = rsx_dsp_reg(rsx, base + 0x04);
    out->offset    = rsx_dsp_reg(rsx, base);
    out->location  = (fmt & 3) == 2 ? RSX_LOCATION_MAIN : RSX_LOCATION_LOCAL;
    out->cubemap   = (fmt >> 2) & 1;
    out->dimension = (fmt >> 4) & 0xF;
    out->format    = (fmt >> 8) & 0xFF;
    out->mipmaps   = fmt >> 16;
    out->wrap      = rsx_dsp_reg(rsx, base + 0x08);
    out->control0  = rsx_dsp_reg(rsx, base + 0x0C);
    out->enabled   = out->control0 >> 31;
    out->remap     = rsx_dsp_reg(rsx, base + 0x10);
    out->filter    = rsx_dsp_reg(rsx, base + 0x14);
    const u32 rect = rsx_dsp_reg(rsx, base + 0x18);
    out->width     = rect >> 16;
    out->height    = rect & 0xFFFF;
    out->border_color = rsx_dsp_reg(rsx, base + 0x1C);
    const u32 sz1  = rsx_dsp_reg(rsx, M_TEX_SIZE1 + unit * 4);
    out->pitch     = sz1 & 0xFFFFF;
    out->depth     = sz1 >> 20;
}

void rsx_dsp_get_vertex_texture(const rsx_dispatch* rsx, u32 unit,
                                rsx_dsp_vertex_texture* out)
{
    memset(out, 0, sizeof(*out));
    if (unit >= RSX_DSP_NUM_VERTEX_TEXTURES)
        return;
    const u32 base = M_VERTEX_TEXTURE + unit * 0x20;
    const u32 fmt  = rsx_dsp_reg(rsx, base + 0x04);
    out->offset    = rsx_dsp_reg(rsx, base + 0x00);
    out->location  = (fmt & 3) == 2 ? RSX_LOCATION_MAIN : RSX_LOCATION_LOCAL;
    out->cubemap   = (fmt >> 2) & 1;
    out->dimension = (fmt >> 4) & 0xF;
    out->format    = (fmt >> 8) & 0xFF;
    out->mipmaps   = fmt >> 16;
    out->wrap      = rsx_dsp_reg(rsx, base + 0x08);
    out->control0  = rsx_dsp_reg(rsx, base + 0x0C);
    out->enabled   = out->control0 >> 31;
    const u32 ctl3 = rsx_dsp_reg(rsx, base + 0x10);
    out->pitch     = ctl3 & 0xFFFFF;
    out->depth     = ctl3 >> 20;
    out->filter    = rsx_dsp_reg(rsx, base + 0x14);
    const u32 rect = rsx_dsp_reg(rsx, base + 0x18);
    out->width     = rect >> 16;
    out->height    = rect & 0xFFFF;
    out->border_color = rsx_dsp_reg(rsx, base + 0x1C);
}

u32 rsx_dsp_vertex_data_base_offset(const rsx_dispatch* rsx)
{
    return rsx_dsp_reg(rsx, M_VERTEX_DATA_BASE);
}

u32 rsx_dsp_vertex_data_base_index(const rsx_dispatch* rsx)
{
    return rsx_dsp_reg(rsx, M_VB_ELEMENT_BASE);
}

u32 rsx_dsp_fragment_program(const rsx_dispatch* rsx, u32* location)
{
    const u32 v = rsx_dsp_reg(rsx, M_FP_ACTIVE_PROGRAM);
    if (location)
        *location = (v & 3) == 2 ? RSX_LOCATION_MAIN : RSX_LOCATION_LOCAL;
    return v & ~3u;
}

u32 rsx_dsp_shader_control(const rsx_dispatch* rsx)
{
    return rsx_dsp_reg(rsx, M_FP_CONTROL);
}

u32 rsx_dsp_vp_start(const rsx_dispatch* rsx)
{
    return rsx_dsp_reg(rsx, M_VP_START_FROM_ID);
}

int rsx_dsp_restart_index_enabled(const rsx_dispatch* rsx, int index_is_u32)
{
    if (!rsx_dsp_reg(rsx, M_RESTART_INDEX_ENABLE))
        return 0;
    /* RPCS3 oracle (rsx_methods.h restart_index_enabled()): a 16-bit index
     * buffer disables restart entirely if the configured sentinel doesn't
     * fit in 16 bits, rather than truncating it. */
    if (!index_is_u32 && rsx_dsp_reg(rsx, M_RESTART_INDEX) > 0xFFFFu)
        return 0;
    return 1;
}

u32 rsx_dsp_restart_index(const rsx_dispatch* rsx)
{
    return rsx_dsp_reg(rsx, M_RESTART_INDEX);
}

void rsx_dsp_vertex_default(const rsx_dispatch* rsx, u32 index, float out[4])
{
    out[0] = out[1] = out[2] = 0.0f;
    out[3] = 1.0f;
    if (index >= RSX_DSP_NUM_VERTEX_ATTR)
        return;
    u32 w[4];
    u32 any = 0;
    for (u32 i = 0; i < 4; i++) {
        w[i] = rsx_dsp_reg(rsx, M_VTX_ATTR_4F + index * 0x10 + i * 4);
        any |= w[i];
    }
    if (!any)
        return;   /* never written: hardware default (0,0,0,1) */
    memcpy(out, w, 16);
}

void rsx_dsp_get_index_array(const rsx_dispatch* rsx, rsx_dsp_index_array* out)
{
    out->offset   = rsx_dsp_reg(rsx, M_IDXBUF_OFFSET);
    const u32 fmt = rsx_dsp_reg(rsx, M_IDXBUF_FORMAT);
    out->location = fmt & 0xF;               /* location index directly     */
    out->is_u32   = ((fmt >> 4) & 0xFF) == 0; /* type 0 = u32, 1 = u16      */
}


/* ---- 27/10 walkF2: register-read tracing (see rsx_dsp_reg in rsx_dispatch.h) ---------------------------------
 * rsx_dsp_trace_set(target, live): reads through `target` are recorded in a per-register bitmap (bit 0 = read,
 * bit 1 = the value differed from `live`, bit 2 = it was the snapshot poison). live may be NULL. */
volatile u32 g_rsx_dsp_trace;
static const rsx_dispatch* s_tr_target;
static const rsx_dispatch* s_tr_live;
static u8 s_tr_bm[RSX_DSP_NUM_REGS];
static u32 s_tr_diffs;

void rsx_dsp_trace_set(const rsx_dispatch* target, const rsx_dispatch* live)
{
    s_tr_target = target; s_tr_live = live; g_rsx_dsp_trace = target ? 1u : 0u;
}
void rsx_dsp_trace_read(const rsx_dispatch* rsx, u32 idx)
{
    if (rsx != s_tr_target || idx >= RSX_DSP_NUM_REGS) return;
    u8 f = 1;
    if (s_tr_live && rsx->regs[idx] != s_tr_live->regs[idx]) { f |= 2; s_tr_diffs++; }
    if (rsx->regs[idx] == 0xDEADBEEFu) { f |= 4; s_tr_diffs++; }
    s_tr_bm[idx] |= f;
}
const u8* rsx_dsp_trace_bitmap(void) { return s_tr_bm; }
void rsx_dsp_trace_clear(void) { memset(s_tr_bm, 0, sizeof s_tr_bm); s_tr_diffs = 0; }
u32 rsx_dsp_trace_diffs(void) { return s_tr_diffs; }
