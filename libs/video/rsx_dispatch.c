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
#ifndef _WIN32   /* off Windows: the Win32 names this file uses come from the shims (Windows has <windows.h>) */
#include "../../runtime/platform/win32_compat.h"
#include "../../runtime/platform/msvc_compat.h"
#endif

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

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

/* ---- RSX_WALK_PROF=1 (walk107): where the words of the VP constant window (0x1F00-0x1F7C) and of the other methods go. Walker thread only.
 * wc[]: 0 upload_run const words, 1 upload_run program words, 2 single-method const words (rsx_dispatch_method), 3 single 0x1EFC,
 * 4 single program words, 5 single other, 6 state_run words, 7 const_run (RSX_VPC_BLOCK) words, 8 const_run calls, 9 upload_run calls (const),
 * 10 single-method calls, 11 plain_run words, 12 const_run words refused (fell back). singles[]: histogram of the single-method words. */
int g_dsp_wc_on = -1;
static unsigned long long s_wc[16], s_wc_sing[0x800];
#define WC_ON() (g_dsp_wc_on > 0)
static void dsp_wc_env(void)
{
    const char* w = getenv("RSX_WALK_PROF"), *e3 = getenv("RSX_WPROF");
    g_dsp_wc_on = ((w && w[0] && w[0] != '0') || (e3 && e3[0] == '3')) ? 1 : 0;
}
/* RSX_VPC_BLOCK: 0 off (default), 1 on, 2 check (both paths, compare, keep the old result). */
static int s_vpc_mode = -1;
static int s_vpc_mism, s_vpc_chk;
static void dsp_vpc_env(void)
{
    const char* e = getenv("RSX_VPC_BLOCK");
    s_vpc_mode = (e && e[0]) ? atoi(e) : 0;
    if (s_vpc_mode < 0 || s_vpc_mode > 2) s_vpc_mode = 0;
    if (s_vpc_mode) fprintf(stderr, "[vpcblock] RSX_VPC_BLOCK=%d: runs into the VP constant window go over in blocks (SIMD byteswap)%s\n", s_vpc_mode, s_vpc_mode == 2 ? ", CHECK" : "");
}
void rsx_dsp_wc_pass_end(unsigned frame)
{
    if (g_dsp_wc_on <= 0) return;
    static unsigned f0; static unsigned long long l[16], ls[0x800]; static unsigned long long t0;
    const unsigned long long now = GetTickCount64();
    if (!t0) { t0 = now; f0 = frame; memcpy(l, s_wc, sizeof l); memcpy(ls, s_wc_sing, sizeof ls); return; }
    if (now - t0 < 2000) return;
    const double N = frame > f0 ? (double)(frame - f0) : 1.0;
    unsigned long long d[16]; for (int i = 0; i < 16; i++) d[i] = s_wc[i] - l[i];
    fprintf(stderr, "[walkprof4] words/frame: upload_run const %.0f prog %.0f | single const %.0f 0x1EFC %.0f prog %.0f other %.0f | state_run %.0f | const_run %.0f (calls %.0f, refused %.0f) | upload_run const calls %.0f single calls %.0f | vpc_block %d checked %d mism %d\n",
            d[0] / N, d[1] / N, d[2] / N, d[3] / N, d[4] / N, d[5] / N, d[6] / N, d[7] / N, d[8] / N, d[12] / N, d[9] / N, d[10] / N, s_vpc_mode, s_vpc_chk, s_vpc_mism);
    fprintf(stderr, "[walkprof4] singles top:");
    unsigned long long cp[0x800]; for (int i = 0; i < 0x800; i++) cp[i] = s_wc_sing[i] - ls[i];
    for (int k = 0; k < 14; k++) {
        unsigned long long best = 0; unsigned bi = 0;
        for (unsigned i = 0; i < 0x800; i++) if (cp[i] > best) { best = cp[i]; bi = i; }
        if (!best) break;
        fprintf(stderr, " %04X x%.0f", bi * 4u, (double)best / N); cp[bi] = 0;
    }
    fputc('\n', stderr);
    memcpy(l, s_wc, sizeof l); memcpy(ls, s_wc_sing, sizeof ls); t0 = now; f0 = frame;
}
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
    memset(rsx->reg_dirty, 0xFF, sizeof rsx->reg_dirty);   /* walk108: everything is new to the snapshot */
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
    memset(rsx->reg_dirty, 0xFF, sizeof rsx->reg_dirty);
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
    rsx_dsp_dirty_range(rsx, (M_VTX_ATTR_4F + attr * 0x10 + c * 4) >> 2, 1);
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
    rsx_dsp_dirty_range(rsx, M_VTXFMT >> 2, RSX_DSP_NUM_VERTEX_ATTR);
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
    if (g_dsp_wc_on < 0) dsp_wc_env();
    if (WC_ON()) {
        s_wc[10]++; s_wc_sing[(method >> 2) & 0x7FF]++;
        if (method >= 0x1F00u && method < 0x1F80u) s_wc[2]++;
        else if (method == 0x1EFCu) s_wc[3]++;
        else if (method >= 0x0B80u && method < 0x0C00u) s_wc[4]++;
        else s_wc[5]++;
    }
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
    rsx->reg_dirty[idx >> 10] |= 1ull << ((idx >> 4) & 63u);   /* walk108 */
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
        rsx_dsp_dirty_range(rsx, M_VP_UPLOAD_FROM_ID >> 2, 1);
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
                    rsx_dsp_dirty_range(rsx, M_VTXFMT >> 2, RSX_DSP_NUM_VERTEX_ATTR);
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

/* ---- RSX_VPC_BLOCK (walk107): a run of words in the VP constant window, in blocks. ----------------------------------------------------
 * The title writes its constants as ONE command that starts at TRANSFORM_CONSTANT_LOAD (0x1EFC) and goes on over 0x1F00..: the run never lies
 * wholly inside the window, so rsx_dispatch_upload_run refused it and every word went through rsx_draw_engine_method / rsx_dispatch_method
 * (21-24 ns each). Here the part of a run inside 0x1F00-0x1F7C is stored at once: byteswapped into regs[], the same words copied into
 * constants[] (constants[slot][lane] is flat: word w goes to flat index slot0*4 + w), const_lo/hi updated once, const_gen += words written.
 * Same result as the word-by-word path of rsx_dispatch_method / upload_run, which stays as the reference (RSX_VPC_BLOCK=2 runs both and compares). */
static void dsp_bswap_copy(u32* dst, const u8* be, u32 n)
{
    u32 i = 0;
#if defined(__SSSE3__) || (defined(__AVX2__))
    const __m128i sh = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    for (; i + 4 <= n; i += 4) {
        const __m128i v = _mm_loadu_si128((const __m128i*)(be + i * 4u));
        _mm_storeu_si128((__m128i*)(dst + i), _mm_shuffle_epi8(v, sh));
    }
#endif
    for (; i < n; i++) { u32 v; memcpy(&v, be + i * 4u, 4); dst[i] = __builtin_bswap32(v); }
}

/* reference: what rsx_dispatch_upload_run does for a run wholly inside the constant window */
static void dsp_const_ref(rsx_dispatch* rsx, u32 method, u32 count, const u8* be)
{
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
}

static void dsp_const_blk(rsx_dispatch* rsx, u32 method, u32 n, const u8* be)
{
    u32* r = &rsx->regs[method >> 2];
    dsp_bswap_copy(r, be, n);
    const u32 slot0 = rsx->regs[M_VP_UPLOAD_CONST_ID >> 2];
    const u32 w0 = (method - M_VP_UPLOAD_CONST) >> 2;
    const u64 fi = (u64)slot0 * 4u + w0, fmax = (u64)RSX_DSP_NUM_CONSTANTS * 4u;
    if (fi >= fmax) return;
    u32 valid = n; if (fi + valid > fmax) valid = (u32)(fmax - fi);
    memcpy(&rsx->constants[0][0] + fi, r, (size_t)valid * 4u);
    const u32 lo = slot0 + (w0 >> 2), hi = slot0 + ((w0 + valid - 1u) >> 2);
    if (lo < rsx->const_lo) rsx->const_lo = lo;
    if (hi > rsx->const_hi) rsx->const_hi = hi;
    rsx->const_gen += valid;
}

/* The part of the run [method, method+count*4) that lies in 0x1F00-0x1F7C, from `method` (which is inside it); returns the words taken. */
static u32 dsp_const_run(rsx_dispatch* rsx, u32 method, u32 count, const u8* be)
{
    static int log = -1;
    if (log < 0) log = getenv("RSX_CONST_LOG") ? 1 : 0;
    u32 n = (M_VP_UPLOAD_CONST + 32u * 4u - method) >> 2; if (n > count) n = count;
    if (log || g_rsx_canvas_mark) { if (WC_ON()) s_wc[12] += n; return 0; }
    if (s_vpc_mode == 2) {
        const u32 w0 = (method - M_VP_UPLOAD_CONST) >> 2, slot0 = rsx->regs[M_VP_UPLOAD_CONST_ID >> 2];
        u32 regs0[32], regs1[32];
        const u32 mi = method >> 2;
        memcpy(regs0, &rsx->regs[mi], n * 4u);
        const u32 lo0 = rsx->const_lo, hi0 = rsx->const_hi, gen0 = rsx->const_gen, cm0 = rsx->canvas_mark_at;
        const u32 s_lo = slot0 + (w0 >> 2), s_hi = slot0 + ((w0 + n - 1u) >> 2);
        const u32 ns = (s_lo < RSX_DSP_NUM_CONSTANTS) ? ((s_hi < RSX_DSP_NUM_CONSTANTS ? s_hi : RSX_DSP_NUM_CONSTANTS - 1u) - s_lo + 1u) : 0u;
        u32 c0[9][4], c1[9][4];   /* a run inside the window (<= 32 words) touches at most 8 slots */
        const u32 nsc = ns > 9 ? 9u : ns;
        if (nsc) memcpy(c0, rsx->constants[s_lo], nsc * 16u);
        dsp_const_ref(rsx, method, n, be);                                 /* the old result, left in place */
        memcpy(regs1, &rsx->regs[mi], n * 4u);
        const u32 lo1 = rsx->const_lo, hi1 = rsx->const_hi, gen1 = rsx->const_gen, cm1 = rsx->canvas_mark_at;
        if (nsc) memcpy(c1, rsx->constants[s_lo], nsc * 16u);
        memcpy(&rsx->regs[mi], regs0, n * 4u);                             /* back to the state before, then the block path */
        rsx->const_lo = lo0; rsx->const_hi = hi0; rsx->const_gen = gen0; rsx->canvas_mark_at = cm0;
        if (nsc) memcpy(rsx->constants[s_lo], c0, nsc * 16u);
        dsp_const_blk(rsx, method, n, be);
        int bad = memcmp(&rsx->regs[mi], regs1, n * 4u) != 0 || rsx->const_lo != lo1 || rsx->const_hi != hi1 || rsx->const_gen != gen1 || rsx->canvas_mark_at != cm1;
        if (nsc && memcmp(rsx->constants[s_lo], c1, nsc * 16u)) bad = 1;
        if (bad) {
            s_vpc_mism++;
            if (s_vpc_mism <= 8) fprintf(stderr, "[vpcblock] MISMATCH method 0x%04X n %u slot0 %u: lo %u/%u hi %u/%u gen %u/%u\n", method, n, slot0, rsx->const_lo, lo1, rsx->const_hi, hi1, rsx->const_gen, gen1);
            memcpy(&rsx->regs[mi], regs1, n * 4u);                         /* keep the old path's answer */
            rsx->const_lo = lo1; rsx->const_hi = hi1; rsx->const_gen = gen1; rsx->canvas_mark_at = cm1;
            if (nsc) memcpy(rsx->constants[s_lo], c1, nsc * 16u);
        }
        s_vpc_chk++;
    } else {
        dsp_const_blk(rsx, method, n, be);
    }
    if (WC_ON()) { s_wc[7] += n; s_wc[8]++; }
    return n;
}

/* program window (0x0B80-0x0BFC): the old per-word loop (reference) and the block version */
static void dsp_inst_ref(rsx_dispatch* rsx, u32 method, u32 count, const u8* be)
{
    u32* load = &rsx->regs[M_VP_UPLOAD_FROM_ID >> 2];
    for (u32 i = 0; i < count; i++) {
        const u32 m = method + i * 4u, v = dsp_be32(be + i * 4u);
        rsx->regs[m >> 2] = v;
        const u32 lane = ((m - M_VP_UPLOAD_INST) >> 2) & 3;
        const u32 pos = *load * 4 + lane;
        if (pos < RSX_DSP_VP_WORDS) { rsx->vp[pos] = v; rsx->vp_gen++; }
        if (lane == 3) (*load)++;
    }
}
static void dsp_inst_blk(rsx_dispatch* rsx, u32 method, u32 n, const u8* be)
{
    u32* load = &rsx->regs[M_VP_UPLOAD_FROM_ID >> 2];
    dsp_bswap_copy(&rsx->regs[method >> 2], be, n);
    const u32 lane0 = ((method - M_VP_UPLOAD_INST) >> 2) & 3u;
    const u32 l0 = *load;                                /* read after the register copy: the run can include FROM_ID itself? (no: window only) */
    const u64 pos0 = (u64)l0 * 4u + lane0;
    if (pos0 < RSX_DSP_VP_WORDS) {
        u32 nv = n; if (pos0 + nv > RSX_DSP_VP_WORDS) nv = (u32)(RSX_DSP_VP_WORDS - pos0);
        memcpy(&rsx->vp[pos0], &rsx->regs[method >> 2], (size_t)nv * 4u);
        rsx->vp_gen += nv;
    }
    *load = l0 + ((lane0 + n) >> 2);
}

int rsx_dispatch_upload_run(rsx_dispatch* rsx, u32 method, u32 count, const u8* be)
{
    static int log = -1;
    if (log < 0) log = getenv("RSX_CONST_LOG") ? 1 : 0;
    if (log || !count) return 0;
    const u32 end = method + count * 4u;
    rsx_dsp_dirty_range(rsx, method >> 2, count); rsx_dsp_dirty_range(rsx, M_VP_UPLOAD_FROM_ID >> 2, 1);   /* walk108 */
    if (g_dsp_wc_on < 0) dsp_wc_env();
    if (WC_ON()) { if (method >= M_VP_UPLOAD_CONST) { s_wc[0] += count; s_wc[9]++; } else s_wc[1] += count; }
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
        if (s_vpc_mode < 0) dsp_vpc_env();
        if (s_vpc_mode > 0 && count <= 32u) {
            /* RSX_VPC_BLOCK: the program words of the run in one block (positions are consecutive: pos0 + i, the load pointer advances once per 4 lanes) */
            if (s_vpc_mode == 2) {
                const u32 mi = method >> 2; u32 r0[32], r1[32], v0[32], v1[32];
                const u32 l0 = *load, g0 = rsx->vp_gen, pos0 = l0 * 4u + (((method - M_VP_UPLOAD_INST) >> 2) & 3u);
                u32 nv = pos0 < RSX_DSP_VP_WORDS ? (RSX_DSP_VP_WORDS - pos0 < count ? RSX_DSP_VP_WORDS - pos0 : count) : 0u;
                memcpy(r0, &rsx->regs[mi], count * 4u); if (nv) memcpy(v0, &rsx->vp[pos0], nv * 4u);
                dsp_inst_ref(rsx, method, count, be);
                memcpy(r1, &rsx->regs[mi], count * 4u); if (nv) memcpy(v1, &rsx->vp[pos0], nv * 4u);
                const u32 l1 = *load, g1 = rsx->vp_gen;
                memcpy(&rsx->regs[mi], r0, count * 4u); if (nv) memcpy(&rsx->vp[pos0], v0, nv * 4u); *load = l0; rsx->vp_gen = g0;   /* back, then the block */
                *load = l0; dsp_inst_blk(rsx, method, count, be);
                const int bad = memcmp(&rsx->regs[mi], r1, count * 4u) || (nv && memcmp(&rsx->vp[pos0], v1, nv * 4u)) || *load != l1 || rsx->vp_gen != g1;
                if (bad) { s_vpc_mism++; if (s_vpc_mism <= 8) fprintf(stderr, "[vpcblock] MISMATCH inst method 0x%04X n %u load %u/%u gen %u/%u\n", method, count, *load, l1, rsx->vp_gen, g1);
                           memcpy(&rsx->regs[mi], r1, count * 4u); if (nv) memcpy(&rsx->vp[pos0], v1, nv * 4u); *load = l1; rsx->vp_gen = g1; }
                s_vpc_chk++;
            } else dsp_inst_blk(rsx, method, count, be);
            if (WC_ON()) s_wc[13] += count;
            return 1;
        }
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
    rsx_dsp_dirty_range(rsx, method >> 2, count);   /* walk108 */
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
extern rsx_dispatch* g_rsx_dsp_main;
void rsx_dispatch_main_note(rsx_dispatch* rsx);
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
    if (!g_rsx_dsp_main) rsx_dispatch_main_note(rsx);
    if (rsx->in_begin_end) return 0;
    if (!s_plain_ready) dsp_plain_build();
    if (method < 0x10000u) rsx_dsp_dirty_range(rsx, method >> 2, incr ? count : 1u);   /* walk108: over-marks when the run stops early */
    u32 i = 0;
    if (incr) {
        if (s_vpc_mode < 0) dsp_vpc_env();
        if (s_vpc_mode > 0) {
            /* RSX_VPC_BLOCK: plain words, then the constant window in one block, alternating over the whole run */
            u32 tot = 0;
            while (tot < count) {
                const u32 m = method + tot * 4u;
                if (m >= M_VP_UPLOAD_CONST && m < M_VP_UPLOAD_CONST + 32u * 4u) {
                    const u32 k = dsp_const_run(rsx, m, count - tot, be + tot * 4u);
                    if (!k) break;
                    tot += k;
                    continue;
                }
                if (m >= 0x2000u) break;
                const unsigned char* cl = &s_plain[m >> 2];
                u32* r = &rsx->regs[m >> 2];
                u32 lim = count - tot;
                if (m + lim * 4u > 0x2000u) lim = (0x2000u - m) >> 2;
                if (m < M_VP_UPLOAD_CONST && m + lim * 4u > M_VP_UPLOAD_CONST) lim = (M_VP_UPLOAD_CONST - m) >> 2;   /* stop at the window */
                u32 j = 0, chg = 0;
                for (; j < lim; j++) {
                    const unsigned c = cl[j];
                    if (!c) break;
                    u32 v; memcpy(&v, be + (tot + j) * 4u, 4);
                    v = __builtin_bswap32(v);
                    chg += (c == 2u) & (r[j] != v);
                    if (g_dsp_mcnt_on) g_dsp_hist_plain[((m >> 2) + j) & 0x7FF]++;
                    r[j] = v;
                }
                if (chg) g_rsx_dsp_vtx_gen++;
                tot += j;
                if (WC_ON()) s_wc[6] += j;
                if (j < lim || !j) break;   /* a non-plain word (or the end): back to the caller */
            }
            return tot;
        }
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
        if (WC_ON()) s_wc[6] += i;
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

/* RSX_DISP_DIRECT=1 (walk109): the walker's block loop hands a command straight to the dispatcher instead of going through
 * rsx_draw_engine_method_run -> eng_method_run_impl -> rsx_draw_engine_method -> eng_method_impl for every word. Per-command
 * overhead measured at Apogee: ~30.000 one-word commands (0x1824 index batch x19.4K, 0x1808 begin/end x10.8K, 0x1814 x1K) and ~15.000
 * short runs per frame. The engine wrapper does nothing for these but store/forward (hist and trace are env diagnostics, the capture
 * flag is tested here too, queries/branch bits/invalidate stats are the excluded methods below), so the call order and effect are the
 * same as the wrapper's. The dispatcher pointer is the one the engine passed to rsx_dispatch_state_run the first time (the engine only
 * calls it once g.ready); until then, and whenever a diagnostic or a capture is on, 0 is returned and the caller uses the engine path. */
rsx_dispatch* g_rsx_dsp_main;
extern int g_rsx_capture_on;
extern int g_wkp_on;
extern void rsx_draw_engine_method(u32 method, u32 arg);
extern int rsx_wkp_enter(int b);
extern void rsx_wkp_leave(int o);
void rsx_dispatch_main_note(rsx_dispatch* rsx) { if (!g_rsx_dsp_main) g_rsx_dsp_main = rsx; }
int rsx_dispatch_direct_ok(void)
{
    static int ok = -1;
    if (ok < 0) {
        const char* e = getenv("RSX_DISP_DIRECT");
        ok = (e && e[0] && e[0] != '0' && !getenv("RSX_METHOD_HIST") && !getenv("RSX_MCNT") && !getenv("RSX_METHOD_TRACE_FRAME") && !getenv("RSX_FIFO_RUN_SLOW")) ? 1 : 0;
        if (e && e[0] && e[0] != '0') fprintf(stderr, "[walker] RSX_DISP_DIRECT=%s: %s\n", e, ok ? "block-loop commands go straight to the dispatcher" : "refused (a per-method diagnostic is on)");
    }
    return ok && g_rsx_dsp_main && !g_rsx_capture_on;
}
int rsx_dispatch_direct_run(u32 method, u32 count, int incr, const u8* be)
{
    rsx_dispatch* const rsx = g_rsx_dsp_main;
    const int wo = g_wkp_on > 0 ? rsx_wkp_enter(1 /* WKP_DISP */) : -1;
    if (count == 1) {
        const u32 m = method & 0x1FFCu;
        if (m == 0x1824u || m == 0x1808u || m == 0x1814u || m == 0x1818u) {
            rsx_dispatch_method(rsx, m, dsp_be32(be));
            if (wo >= 0) rsx_wkp_leave(wo);
            return 1;
        }
    }
    u32 i = 0;
    while (i < count) {
        i += rsx_dispatch_state_run(rsx, incr ? method + i * 4u : method, count - i, incr, be + i * 4u);
        if (i >= count) break;
        const u32 m = (incr ? method + i * 4u : method) & 0x1FFCu;
        const u32 arg = dsp_be32(be + i * 4u);
        if (m == 0x1824u || m == 0x1808u || m == 0x1814u || m == 0x1818u) rsx_dispatch_method(rsx, m, arg);
        else rsx_draw_engine_method(m, arg);
        i++;
    }
    if (wo >= 0) rsx_wkp_leave(wo);
    return 1;
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
