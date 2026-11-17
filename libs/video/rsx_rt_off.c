/*
 * ps3recomp - ray-traced ambient occlusion, off (hosts without D3D12)
 *
 * rsx_rt.c implements RT_AO with DXR and is built on Windows only. The shared draw engine
 * reports draws and asks for injections through the functions below; here RT is never on,
 * so they record nothing and inject nothing. Same declarations: rsx_rt.h.
 */
#ifndef _WIN32
#include "rsx_rt.h"
#include <string.h>

int g_rsx_rt_on = 0;
RsxRtConfig g_rsx_rt;

void rsx_rt_config_load(void) { memset(&g_rsx_rt, 0, sizeof g_rsx_rt); g_rsx_rt_on = 0; }
int  rsx_rt_register_vs(uint64_t vs_id, const char* hlsl) { (void)vs_id; (void)hlsl; return 0; }
void rsx_rt_note_draw(const RsxRtDraw* d) { (void)d; }
void rsx_rt_trace(uint32_t frame, char cat, uint64_t vs_id, uint32_t tris, int alpha_test)
{ (void)frame; (void)cat; (void)vs_id; (void)tris; (void)alpha_test; }
void rsx_rt_cov(int why, uint32_t tris) { (void)why; (void)tris; }
uint32_t rsx_rt_scene_color(void) { return 0; }
void rsx_rt_note_sampled(uint32_t surface_h, uint32_t frame) { (void)surface_h; (void)frame; }
int  rsx_rt_should_inject(uint32_t frame, uint32_t target_h) { (void)frame; (void)target_h; return 0; }
uint32_t rsx_rt_ao_job(uint32_t* color_h, uint32_t* depth_h) { if (color_h) *color_h = 0; if (depth_h) *depth_h = 0; return 0; }
void rsx_rt_freeze_camera(uint32_t job) { (void)job; }
#endif
