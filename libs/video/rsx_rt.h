/*
 * rsx_rt - optional ray-traced ambient occlusion (RTAO) on the static level geometry.
 *
 * Off by default (tod.ini RT_AO = 0): with it off the only cost anywhere is a test of
 * g_rsx_rt_on. On, the module
 *   - classifies vertex programs ("static" = world-to-clip through c[8..11], no
 *     indirect constant addressing, no vertex textures) when a pipeline is built,
 *   - is told about the opaque static draws that use them (the draw engine's vertex
 *     cache identifies the mesh) and, once an instance (mesh + per-object
 *     constants) is seen in two frames, copies its vertices,
 *   - on the GPU runs the translated vertex program over those vertices (compute),
 *     undoes the camera matrix (world = inverse(c[8..11]) * clip) and builds a
 *     bottom-level acceleration structure in world space,
 *   - keeps all built instances in a persistent top-level structure (the game culls
 *     on the CPU, so what is drawn this frame is not the whole neighbourhood),
 *   - after the last opaque draw of the scene pass, traces a few short
 *     cosine-weighted rays per pixel from the depth buffer with inline ray queries
 *     (DXR 1.1, SM 6.5 compute), filters the result depth-aware and multiplies the
 *     scene colour by it.
 * See docs/RAYTRACING_PLAN.md ("Status").
 */
#ifndef PS3RECOMP_RSX_RT_H
#define PS3RECOMP_RSX_RT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 1 when RT_AO=1 was asked for; read once by rsx_rt_config_load(). */
extern int g_rsx_rt_on;

typedef struct RsxRtConfig {
    int   ao;            /* RT_AO       0/1                                   */
    float radius;        /* RT_AO_RADIUS  world units (metres), default 1.0   */
    int   rays;          /* RT_AO_RAYS    1..8, default 3                    */
    int   halfres;       /* RT_AO_HALFRES 0/1, default 1                     */
    float strength;      /* RT_AO_STRENGTH 0..1, default 0.8                 */
    int   debug;         /* RT_AO_DEBUG   0 off, 1 raw AO, 2 primary-ray test */
    int   anim;          /* RT_AO_ANIM    1: noise changes every frame       */
    int   trace;         /* RT_AO_TRACE=<n>: every n frames log the draw-category timeline of one frame (diagnostic) */
    int   min_draws;     /* RT_AO_MIN_DRAWS static draws before a scene pass is believed */
    float power;         /* RT_AO_POWER   contrast curve: ao = vis^power, default 1.6 */
    int   temporal;      /* RT_AO_TEMPORAL history length in frames (0/1 = off), default 8 */
    int   dyn;           /* RT_AO_DYN     1: skinned/animated opaque draws are rebuilt every frame and cast AO */
    int   quality;       /* RT_AO_QUALITY 0 low, 1 medium (default), 2 high; sets rays/halfres unless given */
    float charatt;       /* RT_AO_CHAR    0..1: AO weight on pixels the static scene does not explain (characters), default 0.4 */
} RsxRtConfig;
extern RsxRtConfig g_rsx_rt;

void rsx_rt_config_load(void);

/* RT_AO_VIEWS=<start>,<window>[,<cycles>] (diagnostic): from present number <start> the frame cycles through six views, each
 * <window> presents long: 0 final AO term (grey), 1 geometry match (TLAS vs depth), 2 world-unit checker, 3 RT off,
 * 4 RT default, 5 RT boosted (RT_AO_BOOST=radius,strength,power). rsx_rt_view_of returns the view of a present, or -1. */
int  rsx_rt_view_of(uint32_t present_n);
int  rsx_rt_view_dump(uint32_t present_n);      /* 1 on the last present of a window, for the first <cycles> cycles */
void rsx_rt_set_view(int view);

/* ---- draw engine side (no D3D12 types) -------------------------------------- */

/* A vertex program's translated HLSL, identified by its hash. Returns 1 when it is
 * a "static world" program the module can evaluate (see rsx_rt.c). */
int rsx_rt_register_vs(uint64_t vs_id, const char* hlsl);   /* bit 0: static world program, bit 1: dynamic (indirect constants, e.g. skinning) */

typedef struct RsxRtDraw {
    uint64_t vc_key, vc_content, vs_id;
    const uint8_t* verts;      /* float-expanded, interleaved: stride bytes per vertex */
    uint32_t n_verts, stride;
    const uint32_t* indices;   /* NULL: sequential triangle list */
    uint32_t n_draw;           /* index count (or vertex count when not indexed) */
    const float* cb;           /* the draw's vertex constants: 512 float4, then viewport scale/offset */
    uint32_t frame;
    uint32_t color_h, depth_h; /* backend handles of the pass */
    int alpha_test;            /* the draw uses alpha test (foliage cards) */
} RsxRtDraw;

void rsx_rt_note_draw(const RsxRtDraw* d);
void rsx_rt_trace(uint32_t frame, char cat, uint64_t vs_id, uint32_t tris, int alpha_test);
/* Coverage counter: 0 fed, 1 no vertex-cache entry, 2 program not static/animated (or fixed function), 3 no depth write (colour on),
 * 4 no depth write and no colour, 5 blended, 6 no depth test, 7 not triangles, 8 no CPU copy of the vertices. */
void rsx_rt_cov(int why, uint32_t tris);

/* Called before every draw (target_h: its colour target). Returns 1 (once per frame) when the scene's
 * pass is finished, i.e. a draw into another target has sampled the scene colour (bloom/post): the caller then asks its
 * backend for an AO record with the job id from rsx_rt_ao_job(). */
uint32_t rsx_rt_scene_color(void);          /* backend handle of the scene colour target (0: none yet) */
void rsx_rt_note_sampled(uint32_t surface_h, uint32_t frame);   /* a draw samples this colour surface as a texture */
int  rsx_rt_should_inject(uint32_t frame, uint32_t target_h);
uint32_t rsx_rt_ao_job(uint32_t* color_h, uint32_t* depth_h);
void rsx_rt_freeze_camera(uint32_t job);   /* the camera of the last static draw, kept for that job */

/* ---- D3D12 side ----------------------------------------------------------------- */
#ifdef RSX_RT_D3D12
#include <d3d12.h>

/* Probe DXR on the device, log the result. Returns 1 when RT_AO can run
 * (tier 1.1, SM 6.5, ID3D12Device5). Never fails the caller. */
int  rsx_rt_d3d12_init(ID3D12Device* dev, ID3D12CommandQueue* queue);
void rsx_rt_d3d12_shutdown(void);
void rsx_rt_d3d12_ao_end(ID3D12GraphicsCommandList* l, uint32_t slot);
void rsx_rt_d3d12_ao_collect(uint32_t slot);

typedef D3D12_GPU_DESCRIPTOR_HANDLE (*RsxRtSrvTableFn)(const D3D12_CPU_DESCRIPTOR_HANDLE* src, uint32_t n);

/* Frame housekeeping at the start of each encode: build what the draw engine queued.
 * `seq` grows with every submit; `slot` is the allocator slot (0..3). */
void rsx_rt_d3d12_process(ID3D12GraphicsCommandList* list, uint32_t slot, uint64_t seq);

typedef struct RsxRtAoIn {
    ID3D12GraphicsCommandList* list;
    RsxRtSrvTableFn srv_table;
    ID3D12Resource* depth_res;
    D3D12_CPU_DESCRIPTOR_HANDLE depth_srv;   /* R32_FLOAT_X8X24 view of the single-sample depth */
    uint32_t w, h;                           /* colour target, host pixels */
    uint32_t job, slot;
    uint64_t seq;
} RsxRtAoIn;

/* Runs trace + filter; returns 1 and fills the full-resolution AO texture
 * (R32_FLOAT, state PIXEL_SHADER_RESOURCE) on success. */
int  rsx_rt_d3d12_ao(const RsxRtAoIn* in, ID3D12Resource** ao_tex, D3D12_CPU_DESCRIPTOR_HANDLE* ao_srv);

typedef struct RsxRtCompIn {
    ID3D12GraphicsCommandList* list;
    RsxRtSrvTableFn srv_table;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv;
    D3D12_CPU_DESCRIPTOR_HANDLE ao_srv;
    uint32_t w, h, samples;
    int      fmt;                            /* DXGI_FORMAT of the target */
} RsxRtCompIn;

/* Multiplies the target by the AO (or shows it, RT_AO_DEBUG). Leaves a graphics
 * root signature of its own bound: the caller restores its state. */
int  rsx_rt_d3d12_composite(const RsxRtCompIn* in);
#endif

#ifdef __cplusplus
}
#endif
#endif
