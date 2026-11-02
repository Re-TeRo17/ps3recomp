/*
 * ps3recomp - the register-file draw engine's D3D12 backend
 *
 * rsx_d3d12_backend.c owns the public rsx_d3d12_backend_* entry points and
 * selects between its own rsx_state vtable path and this engine backend at
 * init. These are the hooks it calls; nothing else needs them.
 */
#ifndef PS3RECOMP_RSX_D3D12_ENGINE_H
#define PS3RECOMP_RSX_D3D12_ENGINE_H

#include "ps3emu/ps3types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Create the window (unless PS3RECOMP_D3D12_HEADLESS=1), the device and the
 * engine's resources, register with rsx_draw_engine and init it. Returns 0
 * when the engine is the live path, -1 when the caller should fall back to
 * the vtable path. */
int  rsx_d3d12_engine_init(u32 width, u32 height, const char* title);
void rsx_d3d12_engine_shutdown(void);
int  rsx_d3d12_engine_active(void);
/* Pump the window's messages; -1 once it has been closed. */
int  rsx_d3d12_engine_pump_messages(void);

/* A port's overlay drawn over the presented frame (tod-recomp: the settings page).
 * message: every window message first, on the window's thread; non-zero = handled,
 *   *result is the answer. render: while the present records, on the thread that
 *   encodes, with the back buffer as the render target (RENDER_TARGET state) and
 *   its descriptor; the hook records into the command list and must leave the
 *   back buffer in that state. Both optional; render is called on every present,
 *   so it must return at once when the overlay is closed. dev/queue/list are the
 *   ID3D12Device / ID3D12CommandQueue / ID3D12GraphicsCommandList; rtv is the
 *   D3D12_CPU_DESCRIPTOR_HANDLE.ptr; frames = command allocator slots (reuse
 *   distance of per-frame GPU buffers). */
typedef struct RsxD3D12Overlay {
    int  (*message)(void* hwnd, unsigned msg, uintptr_t wparam, intptr_t lparam, intptr_t* result);
    void (*render)(void* dev, void* queue, void* list, uint64_t rtv, unsigned w, unsigned h, int frames);
} RsxD3D12Overlay;
void rsx_d3d12_set_overlay(const RsxD3D12Overlay* o);

/* Test hooks, as the Metal backend's. */
u32  rsx_d3d12_engine_debug_color(void);
u32  rsx_d3d12_engine_readback_center(void);

#ifdef __cplusplus
}
#endif
#endif
