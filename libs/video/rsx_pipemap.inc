/* rsx_pipemap.inc (pso110, 05/11) -- included by rsx_draw_engine.c, inside the Windows async-pipeline block, before eng_pipe_preload.
 *
 * THE PIPELINE MAP: a list of pipeline DESCRIPTORS, numbers only. No HLSL, no microcode, no translation of anything from the game:
 * for every pipeline the engine ever built, the identity of its vertex/fragment program (where it sits in the EBOOT image that
 * is in guest memory at start-up + hashes to verify it), every input the decompilers take besides the microcode (texture-derived
 * masks, shadow functions, unnormalised sizes, texel ops, SSR config, vertex layout mask, branch bits, alpha-test) and the render
 * state / layout / target formats that the pipeline key hashes. At start the engine rebuilds each pipeline from the player's own game
 * memory: re-translate (current decompilers), FXC, state object, and the dxbc + PSO library get filled in the LOCAL cache.
 *
 *   TOD_SHADER_MAP=1          record: every new pipeline appends a descriptor to <cache>\pipemap.bin (needs the program found in guest memory)
 *   TOD_SHADER_PREWARM        unset = auto: use the map if there is one (<exe>\shaders\pipemap.bin = the shipped one, plus <cache>\pipemap.bin);
 *                             the first start (cold library, or a map/codegen/GPU/driver not prewarmed yet) builds ALL of it BEFORE the game
 *                             runs, with a progress window, then writes <cache>\prewarm_*.ok. 0 = map ignored. 1 = force the blocking prewarm.
 *                             2 = never block, build the map in the background like pipes.bin.
 *   TOD_SHADER_PIPEMAP=<file> read this map instead of <exe>\shaders\pipemap.bin
 *   TOD_SHADER_PREWARM_THREADS=<n> workers of the blocking prewarm (default: every logical core)
 */
#define MAPF_MAGIC   0x50414D50u   /* 'PMAP' */
#define MAPF_VERSION 1u
typedef struct {
    u64 key, vp_fnv, fp_struct, vs_text, ps_text;     /* vs_text/ps_text: FNV of the HLSL when recorded (a check, not a key) */
    u32 vp_addr, vp_instrs, fp_addr, fp_size, fp_ctrl, cube_mask, vtex_mask, vp_bb;
    u32 shadow_mask, unnorm_mask, texop_mask, ssr_cfg, ssr_axis; float ssr_minup;
    u32 stride, rt_fmt, rt_count, pad;
    u8 shadow_funcs[16]; u32 unnorm_dim[16][2]; u32 texel_ops[16];
    rsx_be_render_state rs; rsx_vertex_layout_plan layout;
} map_rec;
typedef struct { u32 magic, version, sz_rec, sz_rs, sz_layout, keyrev; u64 pad; } map_hdr;

static pj_mutex_t s_map_mu = PJ_MUTEX_INIT;
static int s_map_rec_on = -1;
static FILE* s_map_fp; static int s_map_fp_tried;
static char s_map_local[MAX_PATH + 24];
static u64* s_map_keys; static u32 s_map_nk, s_map_cap;
static unsigned s_map_recorded, s_map_nofind;

static u64 map_fnv(const void* d, size_t n, u64 h)
{
    const unsigned char* p = (const unsigned char*)d;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
static int map_has(u64 k)
{
    if (!s_map_cap) return 0;
    for (u32 i = (u32)(k ^ (k >> 29)) & (s_map_cap - 1u);; i = (i + 1u) & (s_map_cap - 1u)) {
        if (!s_map_keys[i]) return 0;
        if (s_map_keys[i] == k) return 1;
    }
}
static void map_add_key(u64 k)
{
    if (!k || map_has(k)) return;
    if ((s_map_nk + 1u) * 2u > s_map_cap) {
        const u32 oc = s_map_cap; u64* ok = s_map_keys;
        s_map_cap = oc ? oc * 2u : 4096u;
        s_map_keys = (u64*)calloc(s_map_cap, sizeof(u64));
        if (!s_map_keys) { s_map_keys = ok; s_map_cap = oc; return; }
        for (u32 i = 0; i < oc; i++) if (ok[i]) {
            u32 q = (u32)(ok[i] ^ (ok[i] >> 29)) & (s_map_cap - 1u);
            while (s_map_keys[q]) q = (q + 1u) & (s_map_cap - 1u);
            s_map_keys[q] = ok[i];
        }
        free(ok);
    }
    u32 q = (u32)(k ^ (k >> 29)) & (s_map_cap - 1u);
    while (s_map_keys[q]) q = (q + 1u) & (s_map_cap - 1u);
    s_map_keys[q] = k; s_map_nk++;
}

/* Guest memory window that holds the EBOOT's read-only and data segments (PT_LOAD 0x10000000 +0x56b80, 0x10060000 +0x14ef00 file-backed). */
#define MAP_GUEST_LO 0x10000000u
#define MAP_GUEST_HI 0x101AF000u
static int map_guest_ok(void)
{
    return vm_base && (!ppu_vm_size || ppu_vm_size > MAP_GUEST_HI);
}
/* Where in the EBOOT image is the program that starts with these 16 bytes (raw guest order)? verify(addr) decides among candidates. */
static u32 map_guest_find(const u8* first16, int (*verify)(const u8* p, const void* ctx), const void* ctx)
{
    if (!map_guest_ok()) return 0;
    const u8* base = vm_base;
    for (u32 a = MAP_GUEST_LO; a + 16u <= MAP_GUEST_HI; a += 4u) {
        const u8* p = base + a;
        if (p[0] != first16[0] || p[1] != first16[1] || p[2] != first16[2] || p[3] != first16[3]) continue;
        if (memcmp(p, first16, 16) != 0) continue;
        if (verify(p, ctx)) return a;
    }
    return 0;
}
typedef struct { const u8* vp_host; u32 bytes; } map_vpctx;
/* the engine's vertex program is host-order words; the EBOOT's are big-endian bytes */
static void map_vp_to_guest(const u8* host, u32 bytes, u8* out)
{
    for (u32 i = 0; i + 4 <= bytes; i += 4) {
        u32 w; memcpy(&w, host + i, 4);
        out[i] = (u8)(w >> 24); out[i + 1] = (u8)(w >> 16); out[i + 2] = (u8)(w >> 8); out[i + 3] = (u8)w;
    }
}
static int map_vp_verify(const u8* p, const void* c)
{
    const map_vpctx* x = (const map_vpctx*)c;
    return memcmp(p, x->vp_host, x->bytes) == 0;   /* vp_host here = already converted to guest order */
}
typedef struct { u32 size; u64 id; } map_fpctx;
static int map_fp_verify(const u8* p, const void* c)
{
    const map_fpctx* x = (const map_fpctx*)c;
    return eng_fp_struct_id(p, x->size) == x->id;
}

static int map_record_enabled(void)
{
    if (s_map_rec_on >= 0) return s_map_rec_on;
    const char* e = getenv("TOD_SHADER_MAP");
    s_map_rec_on = (e && e[0] == '1') ? 1 : 0;
    return s_map_rec_on;
}

static const char* map_cache_dir(void)
{
    return g_rsx_cache_dir_hook ? g_rsx_cache_dir_hook() : NULL;
}

/* Called for every pipeline created at a draw (serial: the lane engine aborts to the serial path for a new pipeline). */
static void eng_map_record(u64 key, const u8* vp_uc, u32 vp_instrs, const u8* fp_uc, u32 fp_size, u32 fp_ctrl,
                           u32 cube_mask, u32 vtex_mask, u32 vp_bb, u32 shadow_mask, const u8* shadow_funcs,
                           u32 unnorm_mask, u32 unnorm_dim[16][2], u32 texop_mask, const u32* texel_ops, u32 ssr_cfg,
                           const rsx_be_render_state* rs, const rsx_vertex_layout_plan* layout, u32 stride,
                           rsx_be_format rt_fmt, u32 rt_count, u64 vs_text, u64 ps_text)
{
    if (!map_record_enabled() || !key || !vp_uc || !fp_uc || !vp_instrs || !fp_size) return;
    const char* dir = map_cache_dir();
    if (!dir) return;
    pj_lock(&s_map_mu);
    if (map_has(key)) { pj_unlock(&s_map_mu); return; }
    map_rec r; memset(&r, 0, sizeof r);
    r.key = key;
    const u32 vbytes = vp_instrs * 16u;
    r.vp_fnv = map_fnv(vp_uc, vbytes, 1469598103934665603ull);
    r.fp_struct = eng_fp_struct_id(fp_uc, fp_size);
    r.vs_text = vs_text; r.ps_text = ps_text;
    { u8* g16 = (u8*)malloc(vbytes);
      if (g16) { map_vp_to_guest(vp_uc, vbytes, g16); map_vpctx c = { g16, vbytes };
                 r.vp_addr = map_guest_find(g16, map_vp_verify, &c); free(g16); } }
    { map_fpctx c = { fp_size, r.fp_struct }; r.fp_addr = map_guest_find(fp_uc, map_fp_verify, &c); }
    if (!r.vp_addr || !r.fp_addr) {
        s_map_nofind++;
        static int nl; if (nl < 12 && ++nl)
            fprintf(stderr, "[pso-map] key %016llx: program not found in the EBOOT image (vp %s, fp %s): not recorded\n",
                    (unsigned long long)key, r.vp_addr ? "found" : "NOT found", r.fp_addr ? "found" : "NOT found");
        map_add_key(key);   /* do not search again */
        pj_unlock(&s_map_mu); return;
    }
    r.vp_instrs = vp_instrs; r.fp_size = fp_size; r.fp_ctrl = fp_ctrl; r.cube_mask = cube_mask; r.vtex_mask = vtex_mask; r.vp_bb = vp_bb;
    r.shadow_mask = shadow_mask; r.unnorm_mask = unnorm_mask; r.texop_mask = texop_mask; r.ssr_cfg = ssr_cfg;
    r.ssr_minup = s_ssr_minup; r.ssr_axis = (u32)s_ssr_axis;
    r.stride = stride; r.rt_fmt = (u32)rt_fmt; r.rt_count = rt_count;
    memcpy(r.shadow_funcs, shadow_funcs, 16); memcpy(r.unnorm_dim, unnorm_dim, sizeof r.unnorm_dim); memcpy(r.texel_ops, texel_ops, sizeof r.texel_ops);
    /* pso112: deterministic file. rsx_be_render_state is all u32 (no padding); the layout plan has 3 tail padding bytes and
     * attrs[]/slot_by_attr[] past count can hold stale stack bytes: copy field by field into the zeroed record. */
    r.rs = *rs;
    r.layout.mask = layout->mask; r.layout.stride = layout->stride; r.layout.count = layout->count;
    { const u32 n = layout->count < RSX_DSP_NUM_VERTEX_ATTR ? layout->count : RSX_DSP_NUM_VERTEX_ATTR;
      for (u32 i = 0; i < n; i++) r.layout.attrs[i] = layout->attrs[i];
      for (u32 i = 0; i < RSX_DSP_NUM_VERTEX_ATTR; i++) r.layout.slot_by_attr[i] = layout->slot_by_attr[i]; }
    if (!s_map_fp_tried) {
        s_map_fp_tried = 1;
        snprintf(s_map_local, sizeof s_map_local, "%s\\pipemap.bin", dir);
        FILE* t = fopen(s_map_local, "rb");
        int ok = 0;
        if (t) { map_hdr h; ok = fread(&h, sizeof h, 1, t) == 1 && h.magic == MAPF_MAGIC && h.version == MAPF_VERSION &&
                                 h.sz_rec == sizeof(map_rec) && h.sz_rs == sizeof(rsx_be_render_state) && h.sz_layout == sizeof(rsx_vertex_layout_plan);
                 fclose(t); }
        s_map_fp = fopen(s_map_local, ok ? "ab" : "wb");
        if (s_map_fp && !ok) {
            map_hdr h = { MAPF_MAGIC, MAPF_VERSION, (u32)sizeof(map_rec), (u32)sizeof(rsx_be_render_state), (u32)sizeof(rsx_vertex_layout_plan), 0, 0 };
            fwrite(&h, sizeof h, 1, s_map_fp);
        }
    }
    if (s_map_fp) {
        if (fwrite(&r, sizeof r, 1, s_map_fp) == 1) { fflush(s_map_fp); s_map_recorded++; }
        map_add_key(key);
    }
    pj_unlock(&s_map_mu);
}

/* ---- progress window (its own thread, plain GDI; Esc stops waiting, the builds go on in the background) ---- */
static volatile LONG s_pg_cur, s_pg_tot, s_pg_stop, s_pg_abort, s_pg_phase;
static HANDLE s_pg_thread; static ULONGLONG s_pg_t0;
static LRESULT CALLBACK pg_proc(HWND h, UINT m, WPARAM w, LPARAM l)
{
    switch (m) {
    case WM_TIMER:
        if (s_pg_stop) { DestroyWindow(h); return 0; }
        InvalidateRect(h, NULL, FALSE); return 0;
    case WM_KEYDOWN: if (w == VK_ESCAPE) s_pg_abort = 1; return 0;
    case WM_ERASEBKGND: return 1;
    case WM_PAINT: {
        PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps);
        RECT rc; GetClientRect(h, &rc);
        HBRUSH bg = CreateSolidBrush(RGB(16, 18, 26)); FillRect(dc, &rc, bg); DeleteObject(bg);
        const long cur = s_pg_cur, tot = s_pg_tot ? s_pg_tot : 1;
        const int bw = rc.right - 40, bx = 20, by = rc.bottom - 44, bh = 18;
        RECT fr = { bx - 1, by - 1, bx + bw + 1, by + bh + 1 };
        HBRUSH fb = CreateSolidBrush(RGB(90, 96, 120)); FrameRect(dc, &fr, fb); DeleteObject(fb);
        RECT pr = { bx, by, bx + (int)((long long)bw * (cur > tot ? tot : cur) / tot), by + bh };
        HBRUSH pb = CreateSolidBrush(RGB(255, 170, 40)); FillRect(dc, &pr, pb); DeleteObject(pb);
        SetBkMode(dc, TRANSPARENT); SetTextColor(dc, RGB(235, 235, 240));
        HFONT f1 = CreateFontA(24, 0, 0, 0, FW_SEMIBOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Segoe UI");
        HFONT f2 = CreateFontA(16, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Segoe UI");
        HGDIOBJ old = SelectObject(dc, f1);
        char t[160]; snprintf(t, sizeof t, "Preparing shaders  %ld / %ld", cur, s_pg_tot);
        TextOutA(dc, 20, 14, t, (int)strlen(t));
        SelectObject(dc, f2);
        const double el = (double)(GetTickCount64() - s_pg_t0) / 1000.0;
        snprintf(t, sizeof t, s_pg_phase ? "Compiling for this graphics card. This happens once; %.0f s" : "Translating shaders... %.0f s", el);
        TextOutA(dc, 20, 50, t, (int)strlen(t));
        TextOutA(dc, 20, 72, "Esc: continue without waiting", 29);
        SelectObject(dc, old); DeleteObject(f1); DeleteObject(f2);
        EndPaint(h, &ps); return 0; }
    }
    return DefWindowProcA(h, m, w, l);
}
static DWORD WINAPI pg_thread(LPVOID a)
{
    (void)a;
    WNDCLASSA wc; memset(&wc, 0, sizeof wc);
    wc.lpfnWndProc = pg_proc; wc.hInstance = GetModuleHandleA(NULL); wc.lpszClassName = "TodShaderPrewarm";
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    RegisterClassA(&wc);
    const int W = 560, H = 130;
    const int sx = GetSystemMetrics(SM_CXSCREEN), sy = GetSystemMetrics(SM_CYSCREEN);
    HWND h = CreateWindowExA(WS_EX_TOPMOST | WS_EX_TOOLWINDOW, wc.lpszClassName, "Ratchet and Clank: shaders", WS_POPUP | WS_BORDER | WS_VISIBLE,
                             (sx - W) / 2, (sy - H) / 2, W, H, NULL, NULL, wc.hInstance, NULL);
    if (!h) return 0;
    SetTimer(h, 1, 100, NULL);
    SetForegroundWindow(h);
    MSG m;
    while (GetMessageA(&m, NULL, 0, 0) > 0) { TranslateMessage(&m); DispatchMessageA(&m); }
    UnregisterClassA(wc.lpszClassName, wc.hInstance);
    return 0;
}
static void pg_open(void) { s_pg_stop = 0; s_pg_abort = 0; s_pg_t0 = GetTickCount64(); s_pg_thread = CreateThread(NULL, 0, pg_thread, NULL, 0, NULL); }
static void pg_close(void)
{
    s_pg_stop = 1;
    if (s_pg_thread) { WaitForSingleObject(s_pg_thread, 2000); CloseHandle(s_pg_thread); s_pg_thread = NULL; }
}

/* ---- load: descriptors -> HLSL -> jobs ---- */
static unsigned s_map_regen, s_map_regen_bad, s_map_text_same, s_map_text_diff;

static char* map_read_file(const char* path, size_t* n_out)
{
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* b = n > (long)sizeof(map_hdr) ? (char*)malloc((size_t)n) : NULL;
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); b = NULL; }
    fclose(f);
    if (b) *n_out = (size_t)n;
    return b;
}

/* The same translation sequence as the creation path in eng_pipeline_get, from a descriptor. Returns 1 and a malloc'ed "vs\0ps\0" block. */
static int map_regen(const map_rec* r, u8* vp_host, char* vsb, char* psb, size_t cap, char** out, u32* vs_len, u32* ps_len)
{
    const u8* g = vm_base + r->vp_addr;
    const u32 vbytes = r->vp_instrs * 16u;
    if (!r->vp_instrs || r->vp_instrs > 512u || !r->fp_size || r->fp_size > 0x4000u) return 0;
    if (r->vp_addr < MAP_GUEST_LO || r->vp_addr + vbytes > MAP_GUEST_HI || r->fp_addr < MAP_GUEST_LO || r->fp_addr + r->fp_size > MAP_GUEST_HI) return 0;
    for (u32 i = 0; i + 4 <= vbytes; i += 4) {   /* guest big-endian words -> host words, as the register file hands them over */
        const u32 w = ((u32)g[i] << 24) | ((u32)g[i + 1] << 16) | ((u32)g[i + 2] << 8) | g[i + 3];
        memcpy(vp_host + i, &w, 4);
    }
    if (map_fnv(vp_host, vbytes, 1469598103934665603ull) != r->vp_fnv) return 0;   /* another game build / not this program */
    const u8* fp = vm_base + r->fp_addr;
    if (eng_fp_struct_id(fp, r->fp_size) != r->fp_struct) return 0;
    rsx_vp_set_branch_bits(r->vp_bb);
    int vi = rsx_vp_decompile_compact_ex(vp_host, vbytes, r->vtex_mask, r->layout.mask, vsb, cap);
    rsx_vp_set_branch_bits(0);
    rsx_fp_set_shadow_units(r->shadow_mask, r->shadow_funcs);
    rsx_fp_set_unnorm_units(r->unnorm_mask, (const u32 (*)[2])r->unnorm_dim);
    rsx_fp_set_texel_ops(r->texop_mask ? r->texel_ops : NULL);
    rsx_fp_set_saturate_alpha(eng_fp_sat_alpha(fp, r->fp_size));
    { static int wf = -1; if (wf < 0) wf = getenv("RSX_WPOS_FLIP") ? atoi(getenv("RSX_WPOS_FLIP")) : 1; rsx_fp_set_wpos_flip(wf); }
    if (r->ssr_cfg) { rsx_fp_set_ssr_debug((r->ssr_cfg >> 17) & 1u); rsx_fp_set_ssr_params(r->ssr_minup, (int)r->ssr_axis); }
    if (!r->ssr_cfg && rsx_fp_ssr_off_clean()) rsx_fp_set_ssr(-1, -1); else rsx_fp_set_ssr((int)((r->ssr_cfg >> 8) & 0xFFu), (int)(r->ssr_cfg & 0xFFu));
    u32 nconst = 0;
    int fi = rsx_fp_decompile_buffered_ex(fp, r->fp_size, r->fp_ctrl, r->cube_mask, psb, cap, &nconst);
    rsx_fp_set_ssr(-1, -1);
    rsx_fp_set_shadow_units(0, NULL); rsx_fp_set_unnorm_units(0, NULL); rsx_fp_set_texel_ops(NULL);
    rsx_fp_set_saturate_alpha(0); rsx_fp_set_wpos_flip(0);
    if (fi > 0 && r->rs.alpha_test_enable && rsx_fp_apply_alpha_test_buffered(psb, cap, r->rs.alpha_func) < 0) fi = -1;
    if (vi <= 0 || fi <= 0) return 0;
    const u32 vl = (u32)strlen(vsb) + 1u, pl = (u32)strlen(psb) + 1u;
    char* blk = (char*)malloc((size_t)vl + pl);
    if (!blk) return 0;
    memcpy(blk, vsb, vl); memcpy(blk + vl, psb, pl);
    *out = blk; *vs_len = vl; *ps_len = pl;
    return 1;
}

/* Reads one map file; queues a job for every record not yet in the table. Returns the number queued. Jobs own their text (job->own). */
static unsigned map_load_file(const char* path, eng_pipe_job** head, eng_pipe_job** tail, u64* idhash, long maxn,
                              u8* vp_host, char* vsb, char* psb, size_t cap)
{
    size_t n = 0; char* b = map_read_file(path, &n);
    if (!b) return 0;
    map_hdr h; memcpy(&h, b, sizeof h);
    if (h.magic != MAPF_MAGIC || h.version != MAPF_VERSION || h.sz_rec != sizeof(map_rec) || h.sz_rs != sizeof(rsx_be_render_state) ||
        h.sz_layout != sizeof(rsx_vertex_layout_plan)) {
        fprintf(stderr, "[pso-map] '%s': another format (rec %u rs %u layout %u): ignored\n", path, h.sz_rec, h.sz_rs, h.sz_layout);
        free(b); return 0;
    }
    *idhash = map_fnv(b, n, *idhash);
    unsigned queued = 0, recs = 0;
    for (size_t off = sizeof h; off + sizeof(map_rec) <= n; off += sizeof(map_rec)) {
        map_rec r; memcpy(&r, b + off, sizeof r);
        recs++;
        map_add_key(r.key);
        if ((long)s_man_queued >= maxn || g.n_pipelines + 64u >= ENG_MAX_PIPELINES || eng_pipe_find(r.key)) continue;
        char* blk = NULL; u32 vl = 0, pl = 0;
        s_map_regen++;
        if (!map_regen(&r, vp_host, vsb, psb, cap, &blk, &vl, &pl)) { s_map_regen_bad++; continue; }
        { /* did it come out as the text recorded (same decompilers: yes) */
          const u64 vh = map_fnv(blk, vl - 1u, 1469598103934665603ull), ph = map_fnv(blk + vl, pl - 1u, 1469598103934665603ull);
          if (vh == r.vs_text && ph == r.ps_text) s_map_text_same++; else s_map_text_diff++; }
        eng_pipe_job* j = (eng_pipe_job*)calloc(1, sizeof *j);
        if (!j) { free(blk); break; }
        j->vs = blk; j->ps = blk + vl; j->own = blk; j->borrowed = 1; j->mapjob = 1; j->key = r.key;
        j->rs = r.rs; j->layout = r.layout; j->stride = r.stride; j->rt_fmt = (rsx_be_format)r.rt_fmt; j->rt_count = r.rt_count;
        j->frame = g.frames;
        eng_pipeline* p = &g.pipelines[g.n_pipelines];
        memset(p, 0, sizeof *p);
        p->key = r.key; p->pending = 1; p->job = j;
        p->vp_hlsl = map_fnv(j->vs, vl - 1u, 1469598103934665603ull);
        p->fp_hlsl = map_fnv(j->ps, pl - 1u, 1469598103934665603ull);
        eng_pipe_index_add(g.n_pipelines);
        g.n_pipelines++;
        s_man_queued++; queued++;
        if (*tail) (*tail)->next = j; else *head = j;
        *tail = j;
    }
    fprintf(stderr, "[pso-map] '%s': %u descriptors, %u queued\n", path, recs, queued);
    free(b);
    return queued;
}

static unsigned long long map_marker_id(u64 mapid)
{
    extern unsigned long long rsx_fp_codegen_salt(void);
    u64 h = 1469598103934665603ull;
#ifdef RSX_CODEGEN_ID
    const u64 cg = (u64)(RSX_CODEGEN_ID) ^ (u64)rsx_fp_codegen_salt();
#else
    const u64 cg = (u64)rsx_fp_codegen_salt();
#endif
    const u64 sig = g_rsx_cache_sig_hook ? g_rsx_cache_sig_hook() : 0;
    h = map_fnv(&cg, sizeof cg, h); h = map_fnv(&mapid, sizeof mapid, h); h = map_fnv(&sig, sizeof sig, h);
    return h;
}

/* Called from eng_pipe_preload after the eligibility checks. Returns 1 if the map produced jobs. */
static int eng_map_preload(void)
{
    const char* pe = getenv("TOD_SHADER_PREWARM");
    const int mode = pe && pe[0] ? atoi(pe) : -1;     /* -1 auto, 0 off, 1 force blocking, 2 background */
    if (mode == 0 || !map_guest_ok()) return 0;
    const char* dir = map_cache_dir();
    if (!dir) return 0;
    char shipped[MAX_PATH + 32], local[MAX_PATH + 32];
    { const char* ov = getenv("TOD_SHADER_PIPEMAP");
      if (ov && *ov) snprintf(shipped, sizeof shipped, "%s", ov);
      else { char exe[MAX_PATH]; DWORD n = GetModuleFileNameA(NULL, exe, sizeof exe);
             if (!n || n >= sizeof exe) return 0;
             char* sl = strrchr(exe, '\\'); if (sl) *sl = 0;
             snprintf(shipped, sizeof shipped, "%s\\shaders\\pipemap.bin", exe); } }
    snprintf(local, sizeof local, "%s\\pipemap.bin", dir);
    if (GetFileAttributesA(shipped) == INVALID_FILE_ATTRIBUTES && GetFileAttributesA(local) == INVALID_FILE_ATTRIBUTES) return 0;

    LARGE_INTEGER qf, q0, q1; QueryPerformanceFrequency(&qf); QueryPerformanceCounter(&q0);
    eng_ssr_params();
    const size_t cap = sizeof s_vs_hlsl;
    char* vsb = (char*)malloc(cap); char* psb = (char*)malloc(cap); u8* vph = (u8*)malloc(512u * 16u);
    if (!vsb || !psb || !vph) { free(vsb); free(psb); free(vph); return 0; }
    long maxn = 12000;
    { const char* e = getenv("RSX_PSO_PRECOMPILE_MAX"); if (e && *e) maxn = atol(e); }
    eng_pipe_job *head = NULL, *tail = NULL;
    u64 idh = 1469598103934665603ull;
    unsigned q = map_load_file(shipped, &head, &tail, &idh, maxn, vph, vsb, psb, cap);
    if (strcmp(shipped, local)) q += map_load_file(local, &head, &tail, &idh, maxn, vph, vsb, psb, cap);
    free(vsb); free(psb); free(vph);
    QueryPerformanceCounter(&q1);
    const double tr = (double)(q1.QuadPart - q0.QuadPart) / (double)qf.QuadPart;
    fprintf(stderr, "[pso-map] %u descriptors regenerated (%u failed to resolve in this game's memory), text identical to the recording %u, different %u; translation %.2f s\n",
            s_map_regen, s_map_regen_bad, s_map_text_same, s_map_text_diff, tr);
    if (!q) return 0;

    /* blocking? */
    char marker[MAX_PATH + 64];
    snprintf(marker, sizeof marker, "%s\\prewarm_%016llx.ok", dir, map_marker_id(idh));
    const int warm = g_rsx_cache_warm_hook ? g_rsx_cache_warm_hook() : 0;
    const int have_marker = GetFileAttributesA(marker) != INVALID_FILE_ATTRIBUTES;
    const int block = mode == 1 || (mode == -1 && !(warm && have_marker));
    s_map_total = (long)q; s_map_done = 0;
    s_pw_total = (long)s_man_queued; timespec_get(&s_pw_t0, TIME_UTC);
    if (block) {
        int nt = 0; { const char* e = getenv("TOD_SHADER_PREWARM_THREADS"); if (e && atoi(e) > 0) nt = atoi(e); }
        if (nt <= 0) { SYSTEM_INFO si; GetSystemInfo(&si); nt = (int)si.dwNumberOfProcessors; }
        if (nt > PJ_MAX_THREADS) nt = PJ_MAX_THREADS;
        s_pj_force_threads = nt; s_pw_cap = nt; s_pw_prio_lowest = 0; s_pw_block = 1;
    }
    pj_lock(&s_pj_mu);
    eng_pipe_start_workers_locked();
    if (s_pj_started > 0) {
        if (s_pj_tail) s_pj_tail->next = head; else s_pj_head = head;
        s_pj_tail = tail;
        pj_broadcast(&s_pj_work);
    }
    pj_unlock(&s_pj_mu);
    if (block && s_pj_started > 0) {
        fprintf(stderr, "[pso-prewarm] blocking prewarm: %u pipelines, %d threads (library %s, marker %s)\n", q, s_pj_nthreads,
                warm ? "warm" : "cold/off", have_marker ? "present" : "absent");
        s_pg_tot = (long)q; s_pg_cur = 0; s_pg_phase = 1;
        const int show = !getenv("TOD_SHADER_PREWARM_NOUI");
        if (show) pg_open();
        ULONGLONG t0 = GetTickCount64(), lastlog = t0;
        long lastdone = -1;
        pj_lock(&s_pj_mu);
        while (s_map_done < s_map_total && !s_pg_abort) {
            struct timespec dl; timespec_get(&dl, TIME_UTC);
            dl.tv_nsec += 100 * 1000000L; if (dl.tv_nsec >= 1000000000L) { dl.tv_sec++; dl.tv_nsec -= 1000000000L; }
            pj_timedwait(&s_pj_done, &s_pj_mu, &dl);
            s_pg_cur = s_map_done;
            if (s_map_done != lastdone && GetTickCount64() - lastlog > 5000) {
                lastlog = GetTickCount64(); lastdone = s_map_done;
                fprintf(stderr, "[pso-prewarm] %ld / %ld (%.0f s)\n", s_map_done, s_map_total, (double)(lastlog - t0) / 1000.0);
            }
            if (GetTickCount64() - t0 > 30ull * 60ull * 1000ull) { fprintf(stderr, "[pso-prewarm] gave up after 30 min\n"); break; }
        }
        pj_unlock(&s_pj_mu);
        const double el = (double)(GetTickCount64() - t0) / 1000.0;
        fprintf(stderr, "[pso-prewarm] %ld pipeline in %.1f s (%ld failed, %s)\n", s_map_done, el, s_pw_fail, s_pg_abort ? "stopped by the user" : "complete");
        if (show) pg_close();
        if (!s_pg_abort && s_map_done >= s_map_total) {
            if (g_rsx_cache_flush_hook) g_rsx_cache_flush_hook();
            FILE* mf = fopen(marker, "wb");
            if (mf) { fprintf(mf, "prewarm %ld pipelines in %.1f s\n", s_map_done, el); fclose(mf); }
        }
        s_pw_block = 0; s_pw_cap = 2; s_pw_prio_lowest = 1;
    } else {
        fprintf(stderr, "[pso-prewarm] map: %u pipelines queued in the background (library %s, marker %s)\n", q, warm ? "warm" : "cold/off", have_marker ? "present" : "absent");
    }
    return 1;
}

#define map_record_enabled_fwd() map_record_enabled()
