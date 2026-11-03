/*
 * ps3recomp - cellSaveData HLE implementation
 *
 * Game save data management with callback-driven flow.
 * Save data is stored under: {root}/gamedata/dev_hdd0/home/00000001/savedata/{dirName}/
 */

#include "ps3emu/vm_watch.h"
#include "cellSaveData.h"
#include "ps3_psf.h"
#include "ps3emu/guest_call.h"
#include "../../runtime/ppu/ppu_memory.h"
#include "../../runtime/memory/vm.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>

/* glibc's <sys/stat.h> exposes st_atime/st_mtime/st_ctime as macros (st_atim.tv_sec
 * etc.) that collide with the identically named members of CellSaveData*Stat; undef
 * them so the struct fields resolve, and read host times via the explicit timespec
 * fields in the POSIX branches below. */
#ifdef st_atime
#  undef st_atime
#endif
#ifdef st_mtime
#  undef st_mtime
#endif
#ifdef st_ctime
#  undef st_ctime
#endif

#ifdef _WIN32
#  include <windows.h>
#  include <io.h>
#  include <direct.h>
#  define HOST_MKDIR(p) _mkdir(p)
#  define HOST_STAT     _stat64
#  define HOST_STAT_T   struct __stat64
#else
#  include <unistd.h>
#  include <dirent.h>
#  include <sys/types.h>
#  define HOST_MKDIR(p) mkdir(p, 0755)
#  define HOST_STAT     stat
#  define HOST_STAT_T   struct stat
#endif

/* ---------------------------------------------------------------------------
 * Internal helpers
 * -----------------------------------------------------------------------*/

void cellSaveData_flush_async(void);   /* hitch116: wait for deferred (PS3_SAVEDATA_ASYNC) writes */
static char s_save_root[1024] = "./gamedata/dev_hdd0/home/00000001/savedata";
/* PS3_SAVEDATA_ROOT=<dir>: keep save data somewhere else -- a test harness
 * can then run from a snapshot without touching the player's own saves. */
static void save_root_init(void)
{
    static int done = 0;
    if (done) return;
    done = 1;
    const char* e = getenv("PS3_SAVEDATA_ROOT");
    if (e && *e) { strncpy(s_save_root, e, sizeof s_save_root - 1); s_save_root[sizeof s_save_root - 1] = 0; }
}

/* Ensure directory (and parents) exist */
static void ensure_dirs(const char* path)
{
    char tmp[1024];
    strncpy(tmp, path, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';

    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            char saved = *p;
            *p = '\0';
            HOST_MKDIR(tmp);
            *p = saved;
        }
    }
    HOST_MKDIR(tmp);
}

static void build_save_path(char* buf, size_t buf_size, const char* dirName)
{
    save_root_init();
    snprintf(buf, buf_size, "%s/%s", s_save_root, dirName);
#ifdef _WIN32
    for (char* p = buf; *p; p++) {
        if (*p == '/') *p = '\\';
    }
#endif
}

/* The generic HLE adapter passes PPC register values raw, so string args
 * arrive as GUEST addresses (< 4GB). Translate to a host pointer once at
 * the entry point; the internal helpers all expect host char*. */
static const char* savedata_host_str(const char* p)
{
    uintptr_t v = (uintptr_t)p;
    if (v && v < 0x100000000ull)
        return (const char*)(vm_base + (uint32_t)v);
    return p;
}

static int dir_exists(const char* path)
{
    HOST_STAT_T st;
    if (HOST_STAT(path, &st) != 0)
        return 0;
#ifdef _WIN32
    return (st.st_mode & _S_IFDIR) != 0;
#else
    return S_ISDIR(st.st_mode);
#endif
}

/* Does this directory hold an actual save, as opposed to merely existing?
 *
 * Firmware writes a PARAM.SFO into every savedata directory, so its presence is
 * what separates a real save from an empty or half-made one. dir_exists() alone
 * is not that test, and the difference is not academic: an empty
 * BLUS30020-SYSTEM directory made Virtua Fighter 5's stat callback come back
 * CELL_SAVEDATA_CBRESULT_ERR_BROKEN (-3), because we had told it isNewData=0 --
 * "your save is there" -- and then handed it a directory with nothing in it.
 * Reporting isNewData=1 for that case is both true and the answer that lets a
 * title create the save it was looking for. */
static int dir_has_save(const char* path)
{
    if (!dir_exists(path)) return 0;
    char sfo[1024];
    snprintf(sfo, sizeof(sfo), "%s/PARAM.SFO", path);
    HOST_STAT_T st;
    return HOST_STAT(sfo, &st) == 0;
}


/* ---------------------------------------------------------------------------
 * Guest-callback marshalling
 *
 * The funcStat / funcFile callbacks the game registers are GUEST function
 * pointers (OPD addresses), not host-callable. We have to:
 *   1. Allocate guest memory for the StatGet/StatSet/CBResult structures
 *   2. Write the structs in big-endian format
 *   3. Dispatch the OPD via g_ps3_guest_caller with the guest pointers
 *   4. Read the structures back (game may have updated CBResult.result)
 *
 * Layout (sizes in PS3 ABI, all big-endian, 32-bit pointers):
 *   CellSaveDataCBResult:    4 + 4 + 4 + 4 + 4                = 20 bytes
 *   CellSaveDataStatGet:     4 + 4 + 56 + 1552 + 4*5 + 4      = 1640 bytes
 *   CellSaveDataStatSet:     4 + 4 + 4                        = 12 bytes (pointers + reCreateMode)
 *
 * We use a fixed scratch region at 0x024E0000 (128 KB, just before the
 * cmdbuf region at 0x02500000). Only one savedata operation is in flight
 * at a time, so a bump allocator is sufficient.
 * -----------------------------------------------------------------------*/

#define SAVEDATA_SCRATCH_BASE  0x024E0000u
#define SAVEDATA_SCRATCH_SIZE  0x00020000u  /* 128 KB */

static uint32_t s_scratch_base = SAVEDATA_SCRATCH_BASE;
static uint32_t s_scratch_size = SAVEDATA_SCRATCH_SIZE;
static uint32_t s_scratch_next = SAVEDATA_SCRATCH_BASE;

/* Configure a committed, runtime-owned guest window before guest execution.
 * Ports with modules at the legacy default must supply nonoverlapping space. */
s32 cellSaveData_set_scratch_region(u32 base, u32 size)
{
    if (!base || (base & 15) || size < 0x20000u || (u64)base + size > UINT32_MAX)
        return CELL_SAVEDATA_ERROR_PARAM;
    s_scratch_base = base; s_scratch_size = size; s_scratch_next = base;
    return CELL_OK;
}

static void scratch_reset(void) { s_scratch_next = s_scratch_base; }

static uint32_t scratch_alloc(uint32_t size)
{
    /* 16-byte align */
    size = (size + 0xF) & ~0xFu;
    if ((u64)s_scratch_next + size > (u64)s_scratch_base + s_scratch_size)
        return 0;
    uint32_t addr = s_scratch_next;
    s_scratch_next += size;
    /* Zero-init */
    memset(vm_base + addr, 0, size);
    return addr;
}

/* Big-endian struct field writers. Offsets are in PS3 ABI layout.
 *
 * CellSaveDataCBResult (sysutil_savedata.h) is 20 bytes, and invalidMsg is a
 * POINTER, not an inline buffer:
 *   int   result;                +0
 *   unsigned int progressBarInc; +4
 *   int   errNeedSizeKB;         +8
 *   char *invalidMsg;            +12   (32-bit guest pointer, NULL when unused)
 *   void *userdata;              +16   (32-bit guest pointer)
 *
 * userdata is the pointer the title handed to cellSaveData*(..., userdata) and
 * it MUST be echoed back here: a callback typically recovers its own object
 * from it. LBP's funcStat is literally
 *     v = cbResult->userdata; r = (*(v+104))(cbResult, get, set); ...
 * so leaving it zero made it dispatch through guest address 104, read the VM's
 * zero-fill, and report CELL_SAVEDATA_CBRESULT_ERR_NODATA (-4) -- it never
 * reached its own handler at all. */
static void marshal_cbresult_init(uint32_t addr, s32 result, uint32_t userdata_ea)
{
    vm_write32(addr + 0,  (uint32_t)result);     /* result */
    vm_write32(addr + 4,  0);                    /* progressBarInc */
    vm_write32(addr + 8,  0);                    /* errNeedSizeKB */
    vm_write32(addr + 12, 0);                    /* invalidMsg = NULL */
    vm_write32(addr + 16, userdata_ea);          /* userdata (echoed back) */
}

static s32 marshal_cbresult_read_result(uint32_t addr)
{
    return (s32)vm_read32(addr + 0);
}

/* PS3 ABI layout of CellSaveDataStatGet (all big-endian on PPC, 32-bit pointers):
 *   s32 hddFreeSizeKB                     +0
 *   u32 isNewData                         +4
 *   CellSaveDataDirStat dir               +8   (s64*3 + char[32] = 56 bytes)
 *   CellSaveDataSystemFileParam getParam  +64  (128+128+1024+4+4+8+256 = 1552 bytes)
 *   u32 bind                              +1616
 *   s32 sizeKB                            +1620
 *   s32 sysSizeKB                         +1624
 *   u32 fileNum                           +1628
 *   u32 fileListNum                       +1632
 *   ptr fileList                          +1636 (32-bit guest ptr)
 *   reserved[64]                          +1640
 *   total                                 1704 bytes
 */
#define SAVEDATA_STATGET_SIZE   1704u
#define SAVEDATA_STATSET_SIZE   16u    /* setParam ptr + reCreateMode + indicator ptr */
#define SAVEDATA_CBRESULT_SIZE  20u    /* s32 + u32 + s32 + char* + void* */

/* CellSaveDataFileGet is 0x44 (excSize + reserved[64]); CellSaveDataFileSet is
 * 0x30 (fileOperation, reserved, fileType, secureFileId[16], fileName ptr,
 * fileOffset, fileSize, fileBufSize, fileBuf ptr). */
#define SAVEDATA_FILEGET_SIZE   0x44u
#define SAVEDATA_FILESET_SIZE   0x30u

/* Preserve callback-updated userdata and read the requested operation back
 * before scratch is reused. FileGet reports the previous operation's byte count. */
static s32 dispatch_func_file(uint32_t func_opd, uint32_t* userdata_ea,
                              u32 exc_size, CellSaveDataFileSet* out)
{
    if (!g_ps3_guest_caller) return CELL_SAVEDATA_CBRESULT_ERR_FAILURE;
    scratch_reset();
    u32 cb_ea = scratch_alloc(SAVEDATA_CBRESULT_SIZE);
    u32 get_ea = scratch_alloc(SAVEDATA_FILEGET_SIZE);
    u32 set_ea = scratch_alloc(SAVEDATA_FILESET_SIZE);
    if (!cb_ea || !get_ea || !set_ea) return CELL_SAVEDATA_CBRESULT_ERR_FAILURE;
    marshal_cbresult_init(cb_ea, CELL_SAVEDATA_CBRESULT_OK_NEXT, *userdata_ea);
    vm_write32(get_ea, exc_size);
    g_ps3_guest_caller(func_opd, cb_ea, get_ea, set_ea, 0, 0, 0, 0, 0);
    *userdata_ea = vm_read32(cb_ea + 16);
    memset(out, 0, sizeof(*out));
    out->fileOperation = vm_read32(set_ea);
    out->fileType = vm_read32(set_ea + 8);
    memcpy(out->secureFileId, vm_base + set_ea + 12, 16);
    u32 name = vm_read32(set_ea + 28), buffer = vm_read32(set_ea + 44);
    out->fileName = name ? (char*)vm_base + name : NULL;
    out->fileOffset = vm_read32(set_ea + 32);
    out->fileSize = vm_read32(set_ea + 36);
    out->fileBufSize = vm_read32(set_ea + 40);
    out->fileBuf = buffer ? vm_base + buffer : NULL;
    /* fread/fwrite on fileBuf go through the kernel, which cannot fault in a
     * demand-committed guest page (see sys_fs_read). Commit it first. */
    if (buffer && out->fileBufSize) { vm_commit(buffer, out->fileBufSize); vm_watch_touch(buffer, out->fileBufSize); }
    return marshal_cbresult_read_result(cb_ea);
}

static void marshal_statget_init(uint32_t addr, int is_new, const char* dirName,
                                  s32 sizeKB, u32 fileNum)
{
    vm_write32(addr + 0,    1024 * 1024);    /* hddFreeSizeKB = 1 GB */
    vm_write32(addr + 4,    is_new ? 1 : 0); /* isNewData */
    vm_write64(addr + 8,    0);              /* dir.st_atime */
    vm_write64(addr + 16,   0);              /* dir.st_mtime */
    vm_write64(addr + 24,   0);              /* dir.st_ctime */
    if (dirName) {
        size_t len = strnlen(dirName, CELL_SAVEDATA_DIRNAME_SIZE - 1);
        memcpy(vm_base + addr + 32, dirName, len);
        ((char*)vm_base)[addr + 32 + len] = 0;
    }
    /* getParam (+64..+1615) left zero — no PARAM.SFO data */
    vm_write32(addr + 1616, 0);              /* bind */
    vm_write32(addr + 1620, (uint32_t)sizeKB); /* sizeKB */
    vm_write32(addr + 1624, 0);              /* sysSizeKB */
    vm_write32(addr + 1628, fileNum);        /* fileNum */
    vm_write32(addr + 1632, 0);              /* fileListNum */
    vm_write32(addr + 1636, 0);              /* fileList ptr (NULL — no files) */
}

/* Dispatch funcStat callback via the guest-caller hook.
 * Returns the cbResult.result value the callback wrote, or
 * CELL_SAVEDATA_CBRESULT_ERR_FAILURE if no dispatcher is installed. */
static s32 dispatch_func_stat_full(uint32_t func_opd, int is_new, const char* dirName,
                                   uint32_t* userdata_ea, const CellSaveDataStatGet* input,
                                   CellSaveDataStatSet* output)
{
    if (!g_ps3_guest_caller) return CELL_SAVEDATA_CBRESULT_ERR_FAILURE;

    scratch_reset();
    uint32_t cb_ea  = scratch_alloc(SAVEDATA_CBRESULT_SIZE);
    uint32_t get_ea = scratch_alloc(SAVEDATA_STATGET_SIZE);
    uint32_t set_ea = scratch_alloc(SAVEDATA_STATSET_SIZE);
    if (!cb_ea || !get_ea || !set_ea) {
        printf("[cellSaveData] scratch alloc failed\n");
        return CELL_SAVEDATA_CBRESULT_ERR_FAILURE;
    }

    marshal_cbresult_init(cb_ea, CELL_SAVEDATA_CBRESULT_OK_NEXT, *userdata_ea);
    marshal_statget_init(get_ea, is_new, dirName, 0, 0);
    if (input) {
        vm_write32(get_ea, (u32)input->hddFreeSizeKB);
        vm_write64(get_ea + 8, (u64)input->dir.st_atime);
        vm_write64(get_ea + 16, (u64)input->dir.st_mtime);
        vm_write64(get_ea + 24, (u64)input->dir.st_ctime);
        memcpy(vm_base + get_ea + 64, &input->getParam, 1552);
        vm_write32(get_ea + 64 + 1280, input->getParam.attribute);
        vm_write32(get_ea + 1616, input->bind);
        vm_write32(get_ea + 1620, (u32)input->sizeKB);
        vm_write32(get_ea + 1624, (u32)input->sysSizeKB);
        vm_write32(get_ea + 1628, input->fileNum);
        vm_write32(get_ea + 1632, input->fileListNum);
        u32 files = input->fileListNum ? scratch_alloc(input->fileListNum * 56) : 0;
        if (input->fileListNum && !files) return CELL_SAVEDATA_CBRESULT_ERR_FAILURE;
        vm_write32(get_ea + 1636, files);
        for (u32 i = 0; i < input->fileListNum; i++) {
            const CellSaveDataFileStat* f = input->fileList + i;
            u32 dst = files + i * 56;
            vm_write32(dst, f->fileType); vm_write64(dst + 8, f->st_size);
            vm_write64(dst + 16, (u64)f->st_atime);
            vm_write64(dst + 24, (u64)f->st_mtime);
            vm_write64(dst + 32, (u64)f->st_ctime);
            memcpy(vm_base + dst + 40, f->fileName, CELL_SAVEDATA_FILENAME_SIZE);
        }
    }
    /* StatSet zero-init by scratch_alloc */

    printf("[cellSaveData] dispatching funcStat OPD=0x%08X (cb=0x%X get=0x%X set=0x%X, isNew=%d, userdata=0x%08X)\n",
           func_opd, cb_ea, get_ea, set_ea, is_new, *userdata_ea);
    g_ps3_guest_caller(func_opd, cb_ea, get_ea, set_ea, 0, 0, 0, 0, 0);

    *userdata_ea = vm_read32(cb_ea + 16);
    if (output) {
        u32 params = vm_read32(set_ea);
        output->setParam = params ? (CellSaveDataSystemFileParam*)(vm_base + params) : NULL;
        output->reCreateMode = vm_read32(set_ea + 4);
        output->indicator = NULL;
    }
    s32 result = marshal_cbresult_read_result(cb_ea);
    printf("[cellSaveData] funcStat returned cbResult.result=%d\n", result);
    return result;
}
/* gp101 03/11: newest-entry selection for the title's fixedList (see dispatch_func_select). */
static int savedata_pick_legacy(void)
{
    const char* e = getenv("PS3_SAVEDATA_PICK");
    return e && !strcmp(e, "legacy");
}

/* Modification time of a save directory: GAME.SAV when present, else the directory; 0 when it does not exist. */
static long long savedata_dir_mtime(const char* name)
{
    char path[1100];
    struct stat st;
    snprintf(path, sizeof path, "%s/%s/GAME.SAV", s_save_root, name);
    if (stat(path, &st) == 0) return (long long)st.st_mtime;
    snprintf(path, sizeof path, "%s/%s", s_save_root, name);
    if (stat(path, &st) == 0) return (long long)st.st_mtime;
    return 0;
}

/* 1 when the newData name is empty or already a directory on disk. */
static int savedata_new_name_taken(uint32_t name_ea)
{
    char nm[64];
    for (int i = 0; i < 63; i++) { nm[i] = (char)vm_read8(name_ea + (uint32_t)i); if (!nm[i]) break; }
    nm[63] = 0;
    if (!nm[0]) return 1;
    char path[1100];
    snprintf(path, sizeof path, "%s/%s", s_save_root, nm);
    const int taken = dir_exists(path);
    if (taken) printf("[cellSaveData] newData '%s' already exists on disk -> not created again%c", nm, 10);
    return taken;
}

/* EA of the fixedList entry (48-byte DirList records, dirName first) with the newest mtime; first entry on a tie. */
static uint32_t savedata_newest_fixed(uint32_t fixed_ea, uint32_t n)
{
    uint32_t best = fixed_ea;
    long long best_t = -1;
    if (n > 64) n = 64;
    for (uint32_t i = 0; i < n; i++) {
        char nm[33];
        for (int k = 0; k < 32; k++) nm[k] = (char)vm_read8(fixed_ea + i * 48 + (uint32_t)k);
        nm[32] = 0;
        if (!nm[0]) continue;
        const long long t = savedata_dir_mtime(nm);
        if (t > best_t) { best_t = t; best = fixed_ea + i * 48; }
    }
    return best;
}


/* svpick 03/11: the on-screen save-slot picker. The port's overlay (src/ui/tod_overlay.cpp) installs the hook; with no
 * hook, TOD_SAVE_PICKER=0, PS3_SAVEDATA_PICK=legacy, or a single choice, the automatic pick above stays as it was.
 * Hook: blocks the calling (guest) thread until the user chooses. Returns the index in names[] (0..n-1), -1 = cancel,
 * -2 = no UI available (fall back to the automatic pick). */
typedef int (*cellSaveData_pick_hook_t)(int is_save, int n, const char* const* names, const char* const* infos,
                                        int new_idx, int def_idx);
static cellSaveData_pick_hook_t s_pick_hook;
void cellSaveData_set_pick_hook(cellSaveData_pick_hook_t fn) { s_pick_hook = fn; }

#define CELL_SAVEDATA_RET_CANCEL_HLE 1   /* CELL_CANCEL: what the system returns when the user cancels the list dialog */
static void read_param_sfo(const char* save_path, CellSaveDataSystemFileParam* param);

static int savedata_picker_enabled(void)
{
    const char* e = getenv("TOD_SAVE_PICKER");
    if (e && e[0] == '0') return 0;
    return s_pick_hook != NULL && !savedata_pick_legacy();
}

/* Ask the user which of the fixedList entries (and the newData slot, new_ea != 0) to use. Returns the EA of the chosen
 * name (fixed[i] or new_ea), 0 + *cancel=1 on cancel, 0 + *cancel=0 when there is nothing to ask / no UI. */
static uint32_t savedata_ask_user(uint32_t fixed_ea, uint32_t fixed_n, uint32_t new_ea, int is_save, int* cancel)
{
    enum { MAXC = 65 };
    static char names[MAXC][40], infos[MAXC][160];
    const char* pn[MAXC]; const char* pi[MAXC];
    uint32_t eas[MAXC];
    int n = 0, new_idx = -1, def_idx = 0;
    long long best_t = -1;
    *cancel = 0;
    if (!savedata_picker_enabled()) return 0;
    if (fixed_n > 64) fixed_n = 64;
    for (uint32_t i = 0; i < fixed_n && fixed_ea; i++) {
        char nm[33];
        for (int k = 0; k < 32; k++) nm[k] = (char)vm_read8(fixed_ea + i * 48 + (uint32_t)k);
        nm[32] = 0;
        if (!nm[0]) continue;
        snprintf(names[n], sizeof names[n], "%s", nm);
        char path[1100]; CellSaveDataSystemFileParam prm;
        snprintf(path, sizeof path, "%s/%s", s_save_root, nm);
        read_param_sfo(path, &prm);
        prm.subTitle[sizeof prm.subTitle - 1] = 0;
        const long long t = savedata_dir_mtime(nm);
        if (t > best_t) { best_t = t; def_idx = n; }
        char when[40] = "";
        if (t > 0) { time_t tt = (time_t)t; struct tm* lt = localtime(&tt); if (lt) strftime(when, sizeof when, "%Y-%m-%d  %H:%M", lt); }
        snprintf(infos[n], sizeof infos[n], "%s%s%s", when, (when[0] && prm.subTitle[0]) ? "   -   " : "", prm.subTitle);
        pn[n] = names[n]; pi[n] = infos[n]; eas[n] = fixed_ea + i * 48; n++;
    }
    if (is_save && new_ea && n < MAXC) {
        snprintf(names[n], sizeof names[n], "(new save)");
        for (int k = 0; k < 32; k++) { char c = (char)vm_read8(new_ea + (uint32_t)k); if (!c) { infos[n][k] = 0; break; } infos[n][k] = c; infos[n][k + 1] = 0; }
        pn[n] = names[n]; pi[n] = infos[n]; eas[n] = new_ea; new_idx = n; def_idx = n; n++;
    }
    if (n < 2) return 0;
    printf("[savepick] %s: asking the user, %d choices (newData=%s)%c", is_save ? "save" : "load", n, new_idx >= 0 ? "yes" : "no", 10);
    const int r = s_pick_hook(is_save, n, pn, pi, new_idx, def_idx);
    if (r == -2) { printf("[savepick] no UI -> automatic pick%c", 10); return 0; }
    if (r < 0 || r >= n) { printf("[savepick] cancelled by the user%c", 10); *cancel = 1; return 0; }
    printf("[savepick] user chose #%d '%s'%c", r, pn[r], 10);
    return eas[r];
}

/* Dispatch a List/Fixed selection callback.
 *
 * The four List/Fixed entry points used to call the guest OPD as a HOST
 * function pointer with HOST struct pointers -- `funcList(&cbResult, &listGet,
 * &listSet)`. That is an immediate access violation the first time a title
 * actually reaches one, and none did until Virtua Fighter 5 called
 * cellSaveDataFixedLoad: rip == the guest OPD address, 0x00691F30.
 *
 * ponytail: this dispatches through the guest caller with ZEROED guest
 * structures rather than marshalling the directory list. With no save data on
 * the VFS the list is empty anyway, which is the honest answer, and a title
 * sees "nothing to select" and proceeds with defaults. Marshal CellSaveData-
 * ListGet/ListSet properly when a title needs to pick an existing save; the
 * layouts are the only missing piece, and dispatch_func_stat next door is the
 * shape to copy. */
/* List callback. The directory list is marshalled into the title's own
 * CellSaveDataSetBuf (SDK: "dirList uses CellSaveDataSetBuf *buf"), and the
 * directory the title picks is read back out of the GUEST CellSaveDataListSet:
 * the callers used to inspect a host listSet nothing ever filled, so every
 * List{Save,Load}2 with no existing saves returned NODATA -- Tornado
 * Outbreak's "New Game" then showed "Unable to save game data".
 *
 * Guest layouts (sysutil_savedata.h, 32-bit pointers):
 *   ListGet  { dirNum +0; dirListNum +4; dirList* +8; reserved[64] }
 *   ListSet  { focusPosition +0; focusDirName* +4; fixedListNum +8;
 *              fixedList* +12; newData* +16; reserved* +20 }
 *   NewData  { iconPosition +0; dirName* +4; icon* +8; reserved* +12 }
 *   DirList  { dirName[32]; listParam[8]; reserved[8] }  (48 bytes)
 *   SetBuf   { dirListMax +0; fileListMax +4; reserved[6]; bufSize +32; buf* +36 }
 *
 * `selected` receives the chosen directory ("" when none): the first listed
 * entry, else the new-save name, else the focus name. For a save with nothing
 * listed the new-save name wins -- headless, there is no user to pick. */
static s32 dispatch_func_select(uint32_t func_opd, const CellSaveDataDirList* dirList,
                                uint32_t dirCount, uint32_t setBuf_ea, int is_save,
                                uint32_t userdata_ea, const char* who, char selected[64], int* cancelled)
{
    if (cancelled) *cancelled = 0;
    selected[0] = '\0';
    if (!g_ps3_guest_caller) return CELL_SAVEDATA_CBRESULT_ERR_FAILURE;

    scratch_reset();
    uint32_t cb_ea  = scratch_alloc(SAVEDATA_CBRESULT_SIZE);
    uint32_t get_ea = scratch_alloc(SAVEDATA_STATGET_SIZE);
    uint32_t set_ea = scratch_alloc(SAVEDATA_STATSET_SIZE);
    if (!cb_ea || !get_ea || !set_ea) return CELL_SAVEDATA_CBRESULT_ERR_FAILURE;
    for (uint32_t i = 0; i < 24; i += 4) vm_write32(set_ea + i, 0);

    uint32_t buf_ea = setBuf_ea ? vm_read32(setBuf_ea + 36) : 0;
    uint32_t cap = setBuf_ea ? vm_read32(setBuf_ea + 32) / 48u : 0;
    if (setBuf_ea && vm_read32(setBuf_ea + 0) < cap) cap = vm_read32(setBuf_ea + 0);
    uint32_t listed = buf_ea ? (dirCount < cap ? dirCount : cap) : 0;
    for (uint32_t i = 0; i < listed; i++)
        for (uint32_t k = 0; k < 48; k++)
            vm_write8(buf_ea + i * 48 + k, (uint8_t)((const char*)&dirList[i])[k]);

    marshal_cbresult_init(cb_ea, CELL_SAVEDATA_CBRESULT_OK_NEXT, userdata_ea);
    vm_write32(get_ea + 0, dirCount);        /* dirNum     */
    vm_write32(get_ea + 4, listed);          /* dirListNum */
    vm_write32(get_ea + 8, listed ? buf_ea : 0);

    printf("[cellSaveData] dispatching %s OPD=0x%08X (dirNum=%u listed=%u)%c",
           who, func_opd, dirCount, listed, 10);
    g_ps3_guest_caller(func_opd, cb_ea, get_ea, set_ea, 0, 0, 0, 0, 0);

    s32 result = marshal_cbresult_read_result(cb_ea);
    const uint32_t fixed_n = vm_read32(set_ea + 8), fixed = vm_read32(set_ea + 12);
    const uint32_t newd = vm_read32(set_ea + 16), focus = vm_read32(set_ea + 4);
    const uint32_t new_name = newd ? vm_read32(newd + 4) : 0;
    uint32_t pick = 0;
    /* gp101 03/11: the title's list callbacks hand the system UI EVERY save in fixedList (load: funcList 0x6C74B8
     * sets fixedList=dirList, newData=0; save: funcList 0x695780 adds a newData entry while slots < max). No UI here,
     * so "fixedList[0]" was the alphabetically first dir: a load took the OLDER of two saves, and a save with newData
     * overwrote SAVE_1 instead of creating the new slot. Now a save with newData creates that slot (unless it exists
     * on disk), and a load / save over an existing slot takes the NEWEST fixedList entry.
     * PS3_SAVEDATA_PICK=legacy restores the old rule. */
    const int legacy_pick = savedata_pick_legacy();
    int ask_cancel = 0;
    const uint32_t asked = (result >= 0 && (fixed_n || new_name)) ? savedata_ask_user(fixed, fixed_n, new_name, is_save, &ask_cancel) : 0;
    if (ask_cancel) { if (cancelled) *cancelled = 1; return result; }
    if (asked) pick = asked;
    else if (!legacy_pick && is_save && new_name && !savedata_new_name_taken(new_name)) pick = new_name;
    else if (is_save && new_name && fixed_n == 0) pick = new_name;
    else if (fixed_n && fixed)               pick = legacy_pick ? fixed : savedata_newest_fixed(fixed, fixed_n);
    else if (new_name)                       pick = new_name;
    else if (focus)                          pick = focus;
    if (pick) {
        for (int i = 0; i < 63; i++) {
            selected[i] = (char)vm_read8(pick + (uint32_t)i);
            if (!selected[i]) break;
        }
        selected[63] = '\0';
    }
    printf("[cellSaveData] %s returned cbResult.result=%d fixed=%u newData=%s -> '%s'%c",
           who, result, fixed_n, new_name ? "yes" : "no", selected, 10);
    return result;
}

/* Enumerate save directories matching a prefix. Returns count, fills dirList up to max. */
static u32 enumerate_save_dirs(const char* prefix, CellSaveDataDirList* dirList, u32 max)
{
    u32 count = 0;

    cellSaveData_flush_async();
    save_root_init();
    ensure_dirs(s_save_root);

#ifdef _WIN32
    {
        char search[1024];
        snprintf(search, sizeof(search), "%s\\*", s_save_root);
        for (char* p = search; *p; p++) {
            if (*p == '/') *p = '\\';
        }

        WIN32_FIND_DATAA fd;
        HANDLE hFind = FindFirstFileA(search, &fd);
        if (hFind == INVALID_HANDLE_VALUE)
            return 0;

        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
                continue;
            if (fd.cFileName[0] == '.')
                continue;
            if (prefix && prefix[0] && strncmp(fd.cFileName, prefix, strlen(prefix)) != 0)
                continue;
            if (count < max && dirList) {
                memset(&dirList[count], 0, sizeof(CellSaveDataDirList));
                strncpy(dirList[count].dirName, fd.cFileName,
                        CELL_SAVEDATA_DIRNAME_SIZE - 1);
            }
            count++;
        } while (FindNextFileA(hFind, &fd));

        FindClose(hFind);
    }
#else
    {
        DIR* dp = opendir(s_save_root);
        if (!dp)
            return 0;

        struct dirent* de;
        while ((de = readdir(dp)) != NULL) {
            if (de->d_name[0] == '.')
                continue;
            /* Check it's a directory */
            char full[1024];
            snprintf(full, sizeof(full), "%s/%s", s_save_root, de->d_name);
            if (!dir_exists(full))
                continue;
            if (prefix && prefix[0] && strncmp(de->d_name, prefix, strlen(prefix)) != 0)
                continue;
            if (count < max && dirList) {
                memset(&dirList[count], 0, sizeof(CellSaveDataDirList));
                strncpy(dirList[count].dirName, de->d_name,
                        CELL_SAVEDATA_DIRNAME_SIZE - 1);
            }
            count++;
        }
        closedir(dp);
    }
#endif

    return count;
}

/* System metadata belongs to the service, not the guest file list. */
static int savedata_system_file(const char* name)
{
    return !strcmp(name, "PARAM.SFO") || !strcmp(name, "PARAM.PFD");
}

static u32 savedata_content_type(const char* name)
{
    if (!strcmp(name, "ICON0.PNG")) return CELL_SAVEDATA_FILETYPE_CONTENT_ICON0;
    if (!strcmp(name, "ICON1.PAM")) return CELL_SAVEDATA_FILETYPE_CONTENT_ICON1;
    if (!strcmp(name, "PIC1.PNG")) return CELL_SAVEDATA_FILETYPE_CONTENT_PIC1;
    if (!strcmp(name, "SND0.AT3")) return CELL_SAVEDATA_FILETYPE_CONTENT_SND0;
    /* Secure-file metadata is not persisted by this simplified backend yet. */
    return CELL_SAVEDATA_FILETYPE_NORMALFILE;
}

/* Which files were written as SECUREFILE. A real save records them in
 * PARAM.PFD; ours keeps one name per line there, and the file stays hidden
 * from the guest's list (savedata_system_file). Without it every file came
 * back as NORMALFILE, and a title that looks for its secure payload in the
 * stat list (The Simpsons Arcade Game) called its own fresh save corrupt. */
static void pfd_path(const char* save_path, char* out, size_t n)
{
    snprintf(out, n, "%s/PARAM.PFD", save_path);
#ifdef _WIN32
    for (char* p = out; *p; p++) if (*p == '/') *p = '\\';
#endif
}

static int pfd_has(const char* save_path, const char* name)
{
    char path[1024], line[64];
    pfd_path(save_path, path, sizeof(path));
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    int found = 0;
    while (!found && fgets(line, sizeof(line), fp)) {
        line[strcspn(line, "\r\n")] = 0;
        found = !strcmp(line, name);
    }
    fclose(fp);
    return found;
}

static void pfd_add(const char* save_path, const char* name)
{
    if (pfd_has(save_path, name)) return;
    char path[1024];
    pfd_path(save_path, path, sizeof(path));
    FILE* fp = fopen(path, "ab");
    if (fp) { fprintf(fp, "%s\n", name); fclose(fp); }
}

/* ---------------------------------------------------------------------------
 * hitch116 02/11: PS3_SAVEDATA_ASYNC=1 (default OFF) -- asynchronous file writes for cellSaveDataListAutoSave.
 *
 * The title's "Auto Save Thread" calls ListAutoSave during play and the whole game stalls ~310-480 ms around it
 * (log/hitch104: ListAutoSave 79 ms wall, ICON0 101 KB + GAME.SAV 591 KB + PARAM.SFO through fopen/fwrite/fclose).
 * With the toggle on, process_file_op / write_param_sfo COPY the buffer, report success to the guest at once and a
 * worker thread performs the writes in FIFO order. Safety rules (a save is never lost or reordered):
 *   - only WRITE / WRITE_NOTRUNC and PARAM.SFO are deferred; READ and DELETE flush first and run synchronously;
 *   - every other savedata entry (savedata_execute, enumerate_save_dirs, cellSaveDataDelete) flushes first, so a
 *     load or a listing right after a save always sees the finished files;
 *   - sys_process_exit and the window-closed exit path call cellSaveData_flush_async() (atexit too);
 *   - a failed deferred write is logged ([savedata-async] FAILED); a failed enqueue falls back to the sync write.
 * ---------------------------------------------------------------------------*/
#ifdef _WIN32
#  define SD_TLS __declspec(thread)
#else
#  define SD_TLS __thread
#endif
static SD_TLS int t_sd_async;        /* this thread is inside an async-enabled ListAutoSave */

static int savedata_async_enabled(void)
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("PS3_SAVEDATA_ASYNC"); v = (e && e[0] == '1') ? 1 : 0; }
    return v;
}

/* THE write of one save file; the synchronous path and the worker both use it. 0 = ok, -1 = failed. */
static int savedata_write_now(const char* save_path, const char* file_path, const char* name,
                              int from_start_trunc, uint32_t offset, const void* data, size_t size, int secure)
{
    ensure_dirs(save_path);
    FILE* fp = fopen(file_path, from_start_trunc ? "wb" : "r+b");
    if (!fp) fp = fopen(file_path, "wb");
    if (!fp) {
        printf("[cellSaveData] file write: cannot open '%s'\n", file_path);
        return -1;
    }
    if (offset > 0) {
#ifdef _MSC_VER
        _fseeki64(fp, (long long)offset, SEEK_SET);
#else
        fseeko(fp, (off_t)offset, SEEK_SET);
#endif
    }
    size_t wrote = fwrite(data, 1, size, fp);
    int failed = wrote != size;
    if (fclose(fp) != 0) failed = 1;
    if (!failed && secure) pfd_add(save_path, name);
    return failed ? -1 : 0;
}

#ifdef _WIN32
typedef struct sd_job {
    struct sd_job* next;
    char save_path[1024], file_path[1024], name[CELL_SAVEDATA_FILENAME_SIZE + 1];
    int trunc, secure; uint32_t offset; size_t size; uint8_t* data;
} sd_job;
static CRITICAL_SECTION s_sd_cs;
static CONDITION_VARIABLE s_sd_work_cv, s_sd_idle_cv;
static INIT_ONCE s_sd_once = INIT_ONCE_STATIC_INIT;
static sd_job *s_sd_head, *s_sd_tail;
static int s_sd_pending;                       /* queued + running */
static volatile LONG s_sd_started;

static DWORD WINAPI sd_worker(LPVOID arg)
{
    (void)arg;
    EnterCriticalSection(&s_sd_cs);
    for (;;) {
        while (!s_sd_head) SleepConditionVariableCS(&s_sd_work_cv, &s_sd_cs, INFINITE);
        sd_job* j = s_sd_head;
        s_sd_head = j->next; if (!s_sd_head) s_sd_tail = NULL;
        LeaveCriticalSection(&s_sd_cs);
        LARGE_INTEGER q0, q1, f; QueryPerformanceCounter(&q0); QueryPerformanceFrequency(&f);
        int rc = savedata_write_now(j->save_path, j->file_path, j->name, j->trunc, j->offset, j->data, j->size, j->secure);
        QueryPerformanceCounter(&q1);
        fprintf(stderr, "[savedata-async] %s '%s' %u bytes in %.1f ms\n", rc ? "FAILED writing" : "wrote", j->name,
                (unsigned)j->size, (double)(q1.QuadPart - q0.QuadPart) * 1000.0 / (double)f.QuadPart);
        fflush(stderr);
        free(j->data); free(j);
        EnterCriticalSection(&s_sd_cs);
        if (--s_sd_pending == 0) WakeAllConditionVariable(&s_sd_idle_cv);
    }
    return 0;
}

static BOOL CALLBACK sd_init_once(PINIT_ONCE o, PVOID p, PVOID* c)
{
    (void)o; (void)p; (void)c;
    InitializeCriticalSection(&s_sd_cs);
    InitializeConditionVariable(&s_sd_work_cv);
    InitializeConditionVariable(&s_sd_idle_cv);
    HANDLE h = CreateThread(NULL, 0, sd_worker, NULL, 0, NULL);
    if (!h) return FALSE;
    CloseHandle(h);
    InterlockedExchange(&s_sd_started, 1);
    return TRUE;
}

/* Wait until every deferred write has hit the disk. Cheap no-op when nothing was ever queued. */
void cellSaveData_flush_async(void)
{
    if (!s_sd_started) return;
    EnterCriticalSection(&s_sd_cs);
    int waits = 0;
    while (s_sd_pending > 0) {
        if (!SleepConditionVariableCS(&s_sd_idle_cv, &s_sd_cs, 5000) && ++waits >= 6) {
            fprintf(stderr, "[savedata-async] flush gave up after 30 s (%d jobs pending)\n", s_sd_pending);
            break;
        }
    }
    LeaveCriticalSection(&s_sd_cs);
}

/* Copy the data and hand the write to the worker. Returns 1 = queued, 0 = could not (caller writes synchronously). */
static int savedata_async_queue(const char* save_path, const char* file_path, const char* name,
                                int trunc, uint32_t offset, const void* data, size_t size, int secure)
{
    if (strlen(save_path) >= sizeof(((sd_job*)0)->save_path) || strlen(file_path) >= sizeof(((sd_job*)0)->file_path) ||
        strlen(name) >= sizeof(((sd_job*)0)->name))
        return 0;
    if (!InitOnceExecuteOnce(&s_sd_once, sd_init_once, NULL, NULL)) return 0;
    sd_job* j = (sd_job*)calloc(1, sizeof(*j));
    uint8_t* d = (uint8_t*)malloc(size ? size : 1);
    if (!j || !d) { free(j); free(d); return 0; }
    memcpy(d, data, size);
    strcpy(j->save_path, save_path); strcpy(j->file_path, file_path); strcpy(j->name, name);
    j->trunc = trunc; j->secure = secure; j->offset = offset; j->size = size; j->data = d;
    static LONG s_atexit_done;
    if (!InterlockedExchange(&s_atexit_done, 1)) atexit(cellSaveData_flush_async);
    EnterCriticalSection(&s_sd_cs);
    if (s_sd_tail) s_sd_tail->next = j; else s_sd_head = j;
    s_sd_tail = j; s_sd_pending++;
    LeaveCriticalSection(&s_sd_cs);
    WakeConditionVariable(&s_sd_work_cv);
    return 1;
}
#else
void cellSaveData_flush_async(void) {}
static int savedata_async_queue(const char* a, const char* b, const char* c, int d, uint32_t e, const void* f, size_t g, int h)
{ (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h; return 0; }
#endif

static u32 savedata_file_type(const char* save_path, const char* name)
{
    u32 t = savedata_content_type(name);
    return t == CELL_SAVEDATA_FILETYPE_NORMALFILE && pfd_has(save_path, name)
         ? CELL_SAVEDATA_FILETYPE_SECUREFILE : t;
}

/* Enumerate files in a save directory. Returns count, fills fileList up to max. */
static u32 enumerate_save_files(const char* save_path,
                                 CellSaveDataFileStat* fileList, u32 max)
{
    u32 count = 0;

#ifdef _WIN32
    {
        char search[1024];
        snprintf(search, sizeof(search), "%s\\*", save_path);
        WIN32_FIND_DATAA fd;
        HANDLE hFind = FindFirstFileA(search, &fd);
        if (hFind == INVALID_HANDLE_VALUE)
            return 0;

        do {
            if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || savedata_system_file(fd.cFileName))
                continue;
            if (count < max && fileList) {
                memset(&fileList[count], 0, sizeof(CellSaveDataFileStat));
                fileList[count].fileType = savedata_file_type(save_path, fd.cFileName);
                strncpy(fileList[count].fileName, fd.cFileName,
                        CELL_SAVEDATA_FILENAME_SIZE - 1);
                ULARGE_INTEGER sz;
                sz.HighPart = fd.nFileSizeHigh;
                sz.LowPart  = fd.nFileSizeLow;
                fileList[count].st_size = (u64)sz.QuadPart;
            }
            count++;
        } while (FindNextFileA(hFind, &fd));

        FindClose(hFind);
    }
#else
    {
        DIR* dp = opendir(save_path);
        if (!dp)
            return 0;

        struct dirent* de;
        while ((de = readdir(dp)) != NULL) {
            if (de->d_name[0] == '.' || savedata_system_file(de->d_name))
                continue;
            char full[1024];
            snprintf(full, sizeof(full), "%s/%s", save_path, de->d_name);
            HOST_STAT_T st;
            if (HOST_STAT(full, &st) != 0)
                continue;
#ifdef _WIN32
            if (st.st_mode & _S_IFDIR)
                continue;
#else
            if (S_ISDIR(st.st_mode))
                continue;
#endif
            if (count < max && fileList) {
                memset(&fileList[count], 0, sizeof(CellSaveDataFileStat));
                fileList[count].fileType = savedata_file_type(save_path, de->d_name);
                strncpy(fileList[count].fileName, de->d_name,
                        CELL_SAVEDATA_FILENAME_SIZE - 1);
                fileList[count].st_size = (u64)st.st_size;
#ifdef _WIN32
                fileList[count].st_atime = (s64)st.st_atime;
                fileList[count].st_mtime = (s64)st.st_mtime;
                fileList[count].st_ctime = (s64)st.st_ctime;
#elif defined(__APPLE__)
/* Darwin names the sub-second stat fields st_*timespec, not glibc's st_*tim. */
                fileList[count].st_atime = (s64)st.st_atimespec.tv_sec;
                fileList[count].st_mtime = (s64)st.st_mtimespec.tv_sec;
                fileList[count].st_ctime = (s64)st.st_ctimespec.tv_sec;
#else
                fileList[count].st_atime = (s64)st.st_atim.tv_sec;
                fileList[count].st_mtime = (s64)st.st_mtim.tv_sec;
                fileList[count].st_ctime = (s64)st.st_ctim.tv_sec;
#endif
            }
            count++;
        }
        closedir(dp);
    }
#endif

    return count;
}

/* Execute file callback: read/write/delete files in the save directory */
static s32 process_file_op(const char* save_path, CellSaveDataFileSet* set)
{
    if (!set) return CELL_SAVEDATA_ERROR_PARAM;
    const char* name = set->fileName;
    switch (set->fileType) {
    case CELL_SAVEDATA_FILETYPE_CONTENT_ICON0: name = "ICON0.PNG"; break;
    case CELL_SAVEDATA_FILETYPE_CONTENT_ICON1: name = "ICON1.PAM"; break;
    case CELL_SAVEDATA_FILETYPE_CONTENT_PIC1: name = "PIC1.PNG"; break;
    case CELL_SAVEDATA_FILETYPE_CONTENT_SND0: name = "SND0.AT3"; break;
    case CELL_SAVEDATA_FILETYPE_SECUREFILE:
    case CELL_SAVEDATA_FILETYPE_NORMALFILE: break;
    default: return CELL_SAVEDATA_ERROR_PARAM;
    }
    if (!name || !name[0] || strnlen(name, CELL_SAVEDATA_FILENAME_SIZE) == CELL_SAVEDATA_FILENAME_SIZE ||
        strchr(name, '/') || strchr(name, '\\') || !strcmp(name, ".") || !strcmp(name, "..") ||
        (set->fileOperation != CELL_SAVEDATA_FILEOP_DELETE && !set->fileBuf))
        return CELL_SAVEDATA_ERROR_PARAM;
    char file_path[1024];
    snprintf(file_path, sizeof(file_path), "%s/%s", save_path, name);
#ifdef _WIN32
    for (char* p = file_path; *p; p++) {
        if (*p == '/') *p = '\\';
    }
#endif

    switch (set->fileOperation) {
    case CELL_SAVEDATA_FILEOP_READ: {
        cellSaveData_flush_async();
        FILE* fp = fopen(file_path, "rb");
        if (!fp) {
            printf("[cellSaveData] file read: cannot open '%s'\n", file_path);
            return CELL_SAVEDATA_ERROR_ACCESS_ERROR;
        }
        if (set->fileOffset > 0) {
#ifdef _MSC_VER
            _fseeki64(fp, (long long)set->fileOffset, SEEK_SET);
#else
            fseeko(fp, (off_t)set->fileOffset, SEEK_SET);
#endif
        }
        size_t read_size = (size_t)set->fileSize;
        if (read_size > set->fileBufSize) read_size = set->fileBufSize;
        size_t got = fread(set->fileBuf, 1, read_size, fp);
        int failed = ferror(fp);
        if (fclose(fp) != 0) failed = 1;
        return failed ? CELL_SAVEDATA_ERROR_ACCESS_ERROR : (s32)got;
    }

    case CELL_SAVEDATA_FILEOP_WRITE:
    case CELL_SAVEDATA_FILEOP_WRITE_NOTRUNC: {
        const int from_start = set->fileOperation == CELL_SAVEDATA_FILEOP_WRITE && set->fileOffset == 0;
        size_t write_size = (size_t)set->fileSize;
        if (write_size > set->fileBufSize) write_size = set->fileBufSize;
        const int secure = set->fileType == CELL_SAVEDATA_FILETYPE_SECUREFILE;
        if (t_sd_async && savedata_async_queue(save_path, file_path, name, from_start, set->fileOffset,
                                                set->fileBuf, write_size, secure))
            return (s32)write_size;                  /* deferred: the worker writes the copy */
        cellSaveData_flush_async();                  /* keep FIFO order if an earlier file is still queued */
        if (savedata_write_now(save_path, file_path, name, from_start, set->fileOffset, set->fileBuf, write_size, secure))
            return CELL_SAVEDATA_ERROR_ACCESS_ERROR;
        return (s32)write_size;
    }

    case CELL_SAVEDATA_FILEOP_DELETE:
        cellSaveData_flush_async();
        remove(file_path);
        return 0;

    default:
        return 0;
    }
}

/* Write PARAM.SFO as a real binary PSF (layout in ps3_psf.h). The old simplified text file was not a
 * PSF, so RPCS3 and a PS3 rejected the save. */
static void write_param_sfo(const char* save_path, const CellSaveDataSystemFileParam* param)
{
    if (!param) return;

    char sfo_path[1024];
    snprintf(sfo_path, sizeof(sfo_path), "%s/PARAM.SFO", save_path);
#ifdef _WIN32
    for (char* p = sfo_path; *p; p++) {
        if (*p == '/') *p = '\\';
    }
#endif

    /* SAVEDATA_DIRECTORY is the last path component of the save directory */
    const char* dir = save_path;
    for (const char* q = save_path; *q; q++)
        if ((*q == '/' || *q == '\\') && q[1]) dir = q + 1;

    /* the guest strings are fixed-size arrays that may lack a terminator: copy with one */
    char title[CELL_SAVEDATA_SYSP_TITLE_SIZE], sub[CELL_SAVEDATA_SYSP_SUBTITLE_SIZE];
    char detail[CELL_SAVEDATA_SYSP_DETAIL_SIZE], lp[CELL_SAVEDATA_SYSP_LPARAM_SIZE];
    snprintf(title, sizeof title, "%.*s", (int)sizeof title - 1, param->title);
    snprintf(sub, sizeof sub, "%.*s", (int)sizeof sub - 1, param->subTitle);
    snprintf(detail, sizeof detail, "%.*s", (int)sizeof detail - 1, param->detail);
    snprintf(lp, sizeof lp, "%.*s", (int)sizeof lp - 1, param->listParam);

    uint8_t buf[4096];
    size_t n = ps3_psf_build(buf, sizeof buf, dir, title, sub, detail, param->attribute, lp);
    if (!n) return;

    if (t_sd_async && savedata_async_queue(save_path, sfo_path, "PARAM.SFO", 1, 0, buf, n, 0)) return;
    cellSaveData_flush_async();
    FILE* fp = fopen(sfo_path, "wb");
    if (!fp) return;
    fwrite(buf, 1, n, fp);
    fclose(fp);
}

/* Read PARAM.SFO: a binary PSF, or the old simplified text file that earlier builds wrote (existing saves keep
 * loading; the next save rewrites them as PSF). */
static void read_param_sfo(const char* save_path, CellSaveDataSystemFileParam* param)
{
    if (!param) return;
    memset(param, 0, sizeof(CellSaveDataSystemFileParam));

    char sfo_path[1024];
    snprintf(sfo_path, sizeof(sfo_path), "%s/PARAM.SFO", save_path);
#ifdef _WIN32
    for (char* p = sfo_path; *p; p++) {
        if (*p == '/') *p = '\\';
    }
#endif

    FILE* fp = fopen(sfo_path, "rb");
    if (!fp) return;

    uint8_t buf[8192];
    size_t got = fread(buf, 1, sizeof buf, fp);
    fclose(fp);
    if (got >= 4 && buf[0] == 0 && buf[1] == 'P' && buf[2] == 'S' && buf[3] == 'F') {
        uint32_t attr = 0;
        if (ps3_psf_parse(buf, got, param->title, sizeof param->title, param->subTitle, sizeof param->subTitle,
                          param->detail, sizeof param->detail, &attr, param->listParam, sizeof param->listParam))
            param->attribute = attr;
        return;
    }

    /* old text format */
    char text[sizeof buf + 1];
    memcpy(text, buf, got); text[got] = 0;
    for (char* line = text; *line; ) {
        char* nl = line; while (*nl && *nl != '\n') nl++;
        char saved = *nl; *nl = 0;
        size_t len = strlen(line);
        while (len > 0 && line[len-1] == '\r') line[--len] = '\0';

        if (strncmp(line, "TITLE=", 6) == 0)
            strncpy(param->title, line + 6, CELL_SAVEDATA_SYSP_TITLE_SIZE - 1);
        else if (strncmp(line, "SUB_TITLE=", 10) == 0)
            strncpy(param->subTitle, line + 10, CELL_SAVEDATA_SYSP_SUBTITLE_SIZE - 1);
        else if (strncmp(line, "DETAIL=", 7) == 0)
            strncpy(param->detail, line + 7, CELL_SAVEDATA_SYSP_DETAIL_SIZE - 1);
        else if (strncmp(line, "ATTRIBUTE=", 10) == 0)
            param->attribute = (u32)atoi(line + 10);
        else if (strncmp(line, "LIST_PARAM=", 11) == 0)
            strncpy(param->listParam, line + 11, CELL_SAVEDATA_SYSP_LPARAM_SIZE - 1);
        if (!saved) break;
        line = nl + 1;
    }
}

/* Core save/load implementation shared by List/Fixed/Auto variants */
static s32 savedata_execute(const char* dirName, int is_save,
                             CellSaveDataSetBuf* setBuf,
                             CellSaveDataStatCallback funcStat,
                             CellSaveDataFileCallback funcFile,
                             void* userdata)
{
    if (!dirName || !funcStat)
        return CELL_SAVEDATA_ERROR_PARAM;

    cellSaveData_flush_async();   /* hitch116: a previous deferred autosave must be on disk before this call looks at it */
    char save_path[1024];
    build_save_path(save_path, sizeof(save_path), dirName);

    int is_new = !dir_has_save(save_path);

    printf("[cellSaveData] %s dir='%s' (new=%d)\n",
           is_save ? "SAVE" : "LOAD", dirName, is_new);

    /* If loading and directory doesn't exist, it's not an error -
       the stat callback will see isNewData=1 and can handle it */

    /* Prepare stat get */
    /* setBuf is a GUEST address, not a host pointer. Dereferencing it faulted
     * the moment the title autosaved: cellSaveDataAutoSave2 -> savedata_execute
     * read setBuf->fileListMax off a guest stack address and took an access
     * violation (0xD00DCF84 = r1+0x90). Read it out of guest memory, the way
     * the dirListMax read below already does. fileListMax is at +4. */
    u32 file_list_max = setBuf ? vm_read32((u32)(uintptr_t)setBuf + 4) : 64;
    if (file_list_max > CELL_SAVEDATA_LISTITEM_MAX) return CELL_SAVEDATA_ERROR_PARAM;
    CellSaveDataFileStat* fileList = NULL;
    u32 fileNum = 0;

    if (file_list_max > 0) {
        fileList = (CellSaveDataFileStat*)calloc(file_list_max, sizeof(CellSaveDataFileStat));
        if (!fileList) return CELL_SAVEDATA_ERROR_INTERNAL;
        if (!is_new) {
            fileNum = enumerate_save_files(save_path, fileList, file_list_max);
        }
    }

    CellSaveDataStatGet statGet;
    memset(&statGet, 0, sizeof(statGet));
    statGet.hddFreeSizeKB = 1024 * 1024; /* report 1GB free */
    statGet.isNewData = is_new ? 1 : 0;
    strncpy(statGet.dir.dirName, dirName, CELL_SAVEDATA_DIRNAME_SIZE - 1);

    if (!is_new) {
        HOST_STAT_T hst;
        if (HOST_STAT(save_path, &hst) == 0) {
#ifdef _WIN32
            statGet.dir.st_atime = (s64)hst.st_atime;
            statGet.dir.st_mtime = (s64)hst.st_mtime;
            statGet.dir.st_ctime = (s64)hst.st_ctime;
#elif defined(__APPLE__)
/* Darwin names the sub-second stat fields st_*timespec, not glibc's st_*tim. */
            statGet.dir.st_atime = (s64)hst.st_atimespec.tv_sec;
            statGet.dir.st_mtime = (s64)hst.st_mtimespec.tv_sec;
            statGet.dir.st_ctime = (s64)hst.st_ctimespec.tv_sec;
#else
            statGet.dir.st_atime = (s64)hst.st_atim.tv_sec;
            statGet.dir.st_mtime = (s64)hst.st_mtim.tv_sec;
            statGet.dir.st_ctime = (s64)hst.st_ctim.tv_sec;
#endif
        }
        read_param_sfo(save_path, &statGet.getParam);
    }

    statGet.fileNum = fileNum;
    statGet.fileListNum = fileNum < file_list_max ? fileNum : file_list_max;
    statGet.fileList = fileList;
    statGet.bind = 0;
    statGet.sizeKB = 0;
    statGet.sysSizeKB = 0;

    /* Call stat callback */
    CellSaveDataCBResult cbResult;
    memset(&cbResult, 0, sizeof(cbResult));
    cbResult.result = CELL_SAVEDATA_CBRESULT_OK_NEXT;

    CellSaveDataStatSet statSet;
    memset(&statSet, 0, sizeof(statSet));

    /* Same guest-OPD-as-host-pointer bug as funcFile below: this jumped into
     * guest memory as host code -- ACCESS VIOLATION: execute at host 0x34C250
     * -- the moment Tokyo Jungle autosaved. Route it through the marshalling
     * dispatcher, which builds StatGet/StatSet in guest memory. */
    u32 callback_userdata = (u32)(uintptr_t)userdata;
    cbResult.result = dispatch_func_stat_full((uint32_t)(uintptr_t)funcStat, is_new,
                         dirName, &callback_userdata, &statGet, &statSet);

    if (cbResult.result < 0) {
        printf("[cellSaveData] stat callback returned error %d\n", cbResult.result);
        free(fileList);
        if (cbResult.result == CELL_SAVEDATA_CBRESULT_ERR_NODATA)
            return CELL_SAVEDATA_ERROR_NODATA;
        return CELL_SAVEDATA_ERROR_CBRESULT;
    }

    /* Write PARAM.SFO if stat set provided params and we're saving */
    if (is_save && statSet.setParam) {
        ensure_dirs(save_path);
        CellSaveDataSystemFileParam params;
        memcpy(&params, statSet.setParam, sizeof(params));
        params.attribute = ps3_bswap32(params.attribute);
        write_param_sfo(save_path, &params);
    }

    /* Each NEXT callback requests one operation. LAST ends the sequence;
     * its zero-initialized FileSet is not an implicit read request. */
    if (funcFile && cbResult.result == CELL_SAVEDATA_CBRESULT_OK_NEXT) {
        u32 exc_size = 0;
        while (1) {
            CellSaveDataFileSet fileSet;
            cbResult.result = dispatch_func_file((u32)(uintptr_t)funcFile,
                                                 &callback_userdata, exc_size, &fileSet);
            if (cbResult.result != CELL_SAVEDATA_CBRESULT_OK_NEXT) break;
            s32 exc = process_file_op(save_path, &fileSet);
            printf("[cellSaveData] file op=%u type=%u name='%s' off=%u size=%u buf=%u -> %d\n",
                   fileSet.fileOperation, fileSet.fileType,
                   fileSet.fileName ? fileSet.fileName : "", fileSet.fileOffset,
                   fileSet.fileSize, fileSet.fileBufSize, exc);
            if (exc < 0) { free(fileList); return exc; }
            exc_size = (u32)exc;
        }
    }

    free(fileList);

    if (cbResult.result < 0) {
        if (cbResult.result == CELL_SAVEDATA_CBRESULT_ERR_NODATA)
            return CELL_SAVEDATA_ERROR_NODATA;
        return CELL_SAVEDATA_ERROR_CBRESULT;
    }

    printf("[cellSaveData] %s complete for '%s'\n",
           is_save ? "SAVE" : "LOAD", dirName);
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * API implementations
 * -----------------------------------------------------------------------*/

s32 cellSaveDataListSave2(u32 version, CellSaveDataSetList* setList,
                           CellSaveDataSetBuf* setBuf,
                           CellSaveDataListCallback funcList,
                           CellSaveDataStatCallback funcStat,
                           CellSaveDataFileCallback funcFile,
                           u32 container, void* userdata)
{
    printf("[cellSaveData] ListSave2(version=%u)\n", version);

    if (!setList || !setBuf || !funcList || !funcStat)
        return CELL_SAVEDATA_ERROR_PARAM;

    /* Enumerate existing save dirs */
    /* CellSaveDataSetList is { u32 sortType; u32 sortOrder; char* dirNamePrefix; }
     * and CellSaveDataSetBuf starts with u32 dirListMax -- guest layout, so the
     * prefix EA is the word at +8 and needs translating before it is used as a
     * host string. The host structs put dirNamePrefix at +8 as an 8-byte
     * pointer, so a cast would have read half of it plus padding. */
    u32 dir_max = vm_read32((u32)(uintptr_t)setBuf + 0);
    if (dir_max == 0) dir_max = CELL_SAVEDATA_DIRLIST_MAX;
    CellSaveDataDirList* dirList = (CellSaveDataDirList*)calloc(dir_max, sizeof(CellSaveDataDirList));

    u32 dirCount = enumerate_save_dirs(
        savedata_host_str((const char*)(uintptr_t)vm_read32((u32)(uintptr_t)setList + 8)) ?
            savedata_host_str((const char*)(uintptr_t)vm_read32((u32)(uintptr_t)setList + 8)) : "",
        dirList, dir_max);

    /* Call list callback */
    CellSaveDataCBResult cbResult;
    memset(&cbResult, 0, sizeof(cbResult));
    cbResult.result = CELL_SAVEDATA_CBRESULT_OK_NEXT;

    CellSaveDataListGet listGet;
    memset(&listGet, 0, sizeof(listGet));
    listGet.dirListNum = dirCount < dir_max ? dirCount : dir_max;
    listGet.dirList = dirList;

    CellSaveDataListSet listSet;
    memset(&listSet, 0, sizeof(listSet));

    char selectedDir[64];
    int user_cancel = 0;
    cbResult.result = dispatch_func_select((uint32_t)(uintptr_t)funcList, dirList,
                                          listGet.dirListNum, (uint32_t)(uintptr_t)setBuf, 1,
                                          (uint32_t)(uintptr_t)userdata, "funcList", selectedDir, &user_cancel);
    (void)listSet;

    if (user_cancel) {   /* svpick: the user backed out of the slot picker; the title's stat/file callbacks never run */
        free(dirList);
        return CELL_SAVEDATA_RET_CANCEL_HLE;
    }
    if (cbResult.result < 0) {
        free(dirList);
        return CELL_SAVEDATA_ERROR_CBRESULT;
    }
    if (!selectedDir[0] && dirCount > 0) {
        u32 bi = 0; long long bt = -1;    /* gp101: newest, not alphabetically first */
        for (u32 i = 0; i < dirCount && i < dir_max && !savedata_pick_legacy(); i++) {
            const long long t = savedata_dir_mtime(dirList[i].dirName);
            if (t > bt) { bt = t; bi = i; }
        }
        strncpy(selectedDir, dirList[bi].dirName, sizeof selectedDir - 1);
    }
    if (!selectedDir[0]) {
        free(dirList);
        return CELL_SAVEDATA_ERROR_NODATA;
    }

    s32 result = savedata_execute(selectedDir, 1, setBuf, funcStat, funcFile, userdata);
    free(dirList);
    return result;
}

s32 cellSaveDataListLoad2(u32 version, CellSaveDataSetList* setList,
                           CellSaveDataSetBuf* setBuf,
                           CellSaveDataListCallback funcList,
                           CellSaveDataStatCallback funcStat,
                           CellSaveDataFileCallback funcFile,
                           u32 container, void* userdata)
{
    printf("[cellSaveData] ListLoad2(version=%u)\n", version);

    if (!setList || !setBuf || !funcList || !funcStat)
        return CELL_SAVEDATA_ERROR_PARAM;

    /* CellSaveDataSetList is { u32 sortType; u32 sortOrder; char* dirNamePrefix; }
     * and CellSaveDataSetBuf starts with u32 dirListMax -- guest layout, so the
     * prefix EA is the word at +8 and needs translating before it is used as a
     * host string. The host structs put dirNamePrefix at +8 as an 8-byte
     * pointer, so a cast would have read half of it plus padding. */
    u32 dir_max = vm_read32((u32)(uintptr_t)setBuf + 0);
    if (dir_max == 0) dir_max = CELL_SAVEDATA_DIRLIST_MAX;
    CellSaveDataDirList* dirList = (CellSaveDataDirList*)calloc(dir_max, sizeof(CellSaveDataDirList));

    u32 dirCount = enumerate_save_dirs(
        savedata_host_str((const char*)(uintptr_t)vm_read32((u32)(uintptr_t)setList + 8)) ?
            savedata_host_str((const char*)(uintptr_t)vm_read32((u32)(uintptr_t)setList + 8)) : "",
        dirList, dir_max);

    CellSaveDataCBResult cbResult;
    memset(&cbResult, 0, sizeof(cbResult));
    cbResult.result = CELL_SAVEDATA_CBRESULT_OK_NEXT;

    CellSaveDataListGet listGet;
    memset(&listGet, 0, sizeof(listGet));
    listGet.dirListNum = dirCount < dir_max ? dirCount : dir_max;
    listGet.dirList = dirList;

    CellSaveDataListSet listSet;
    memset(&listSet, 0, sizeof(listSet));

    char selectedDir[64];
    int user_cancel = 0;
    cbResult.result = dispatch_func_select((uint32_t)(uintptr_t)funcList, dirList,
                                          listGet.dirListNum, (uint32_t)(uintptr_t)setBuf, 0,
                                          (uint32_t)(uintptr_t)userdata, "funcList", selectedDir, &user_cancel);
    (void)listSet;

    if (user_cancel) {   /* svpick: the user backed out of the slot picker; the title's stat/file callbacks never run */
        free(dirList);
        return CELL_SAVEDATA_RET_CANCEL_HLE;
    }
    if (cbResult.result < 0) {
        free(dirList);
        return CELL_SAVEDATA_ERROR_CBRESULT;
    }
    if (!selectedDir[0] && dirCount > 0) {
        u32 bi = 0; long long bt = -1;    /* gp101: newest, not alphabetically first */
        for (u32 i = 0; i < dirCount && i < dir_max && !savedata_pick_legacy(); i++) {
            const long long t = savedata_dir_mtime(dirList[i].dirName);
            if (t > bt) { bt = t; bi = i; }
        }
        strncpy(selectedDir, dirList[bi].dirName, sizeof selectedDir - 1);
    }
    if (!selectedDir[0]) {
        free(dirList);
        return CELL_SAVEDATA_ERROR_NODATA;
    }

    s32 result = savedata_execute(selectedDir, 0, setBuf, funcStat, funcFile, userdata);
    free(dirList);
    return result;
}

/* Fixed callbacks have CellSaveDataFixedSet, not CellSaveDataListSet.
 * Marshal the PS3 layouts (32-bit pointers, big endian) and enter the guest
 * through its OPD bridge. Layout reference: RPCS3 cellSaveData.h. */
/* savedata_fixed's early returns were all silent, so a title whose save-data
 * load never completes produced exactly one log line ("FixedLoad2") and no hint
 * which of a dozen validation branches rejected it. Virtua Fighter 5 sits in
 * cellSaveDataFixedLoad2 and repeatedly offers to cancel the load; naming the
 * branch is the difference between that and a day of bisecting. Diagnostic
 * only -- the value returned is unchanged. */
static void savedata_reject_log(const char* fn, const char* why, int rc)
{
    printf("[cellSaveData] %s: %s (rc=0x%08X)\n", fn, why, (unsigned)rc);
}
#define SAVEDATA_REJECT(rc, why) \
    do { savedata_reject_log(__func__, (why), (rc)); return (rc); } while (0)

static s32 savedata_fixed(int is_save, CellSaveDataSetList* setList,
                         CellSaveDataSetBuf* setBuf, CellSaveDataFixedCallback funcFixed,
                         CellSaveDataStatCallback funcStat, CellSaveDataFileCallback funcFile,
                         void* userdata)
{
    if (!setList || !setBuf || !funcFixed)
        SAVEDATA_REJECT(CELL_SAVEDATA_ERROR_PARAM, "setList/setBuf/funcFixed is NULL");
    if (!g_ps3_guest_caller)
        SAVEDATA_REJECT(CELL_SAVEDATA_ERROR_INTERNAL, "no guest caller installed");
    u32 dir_max = vm_read32((u32)(uintptr_t)setBuf);
    if (dir_max > CELL_SAVEDATA_DIRLIST_MAX)
        SAVEDATA_REJECT(CELL_SAVEDATA_ERROR_PARAM, "dirListMax over CELL_SAVEDATA_DIRLIST_MAX");
    CellSaveDataDirList* dirs = dir_max ? calloc(dir_max, sizeof(*dirs)) : NULL;
    if (dir_max && !dirs) return CELL_SAVEDATA_ERROR_INTERNAL;
    u32 prefix = vm_read32((u32)(uintptr_t)setList + 8);
    u32 count = enumerate_save_dirs(prefix ? (char*)vm_base + prefix : "", dirs, dir_max);
    u32 listed = count < dir_max ? count : dir_max;
    scratch_reset();
    u32 cb = scratch_alloc(SAVEDATA_CBRESULT_SIZE);
    u32 get = scratch_alloc(0x4C); /* dirNum, dirListNum, dirList, reserved[64] */
    u32 set = scratch_alloc(12);   /* dirName, newIcon, option */
    u32 list = listed ? scratch_alloc(listed * 48) : 0;
    if (!cb || !get || !set || (listed && !list)) {
        free(dirs); return CELL_SAVEDATA_ERROR_INTERNAL;
    }
    marshal_cbresult_init(cb, CELL_SAVEDATA_CBRESULT_OK_NEXT, (u32)(uintptr_t)userdata);
    vm_write32(get, count); vm_write32(get + 4, listed); vm_write32(get + 8, list);
    if (listed) memcpy(vm_base + list, dirs, listed * 48);
    free(dirs);
    g_ps3_guest_caller((u32)(uintptr_t)funcFixed, cb, get, set, 0, 0, 0, 0, 0);
    s32 result = marshal_cbresult_read_result(cb);
    if (result == CELL_SAVEDATA_CBRESULT_OK_LAST ||
        result == CELL_SAVEDATA_CBRESULT_OK_LAST_NOCONFIRM) return CELL_OK;
    if (result != CELL_SAVEDATA_CBRESULT_OK_NEXT) {
        /* Name the value: the title chose this, so it says WHY it refused --
         * ERR_NODATA on a load means the dirList we handed it was empty or
         * held nothing it recognised, which is a different bug from a broken
         * save or a marshalling mistake. */
        const char* w = "funcFixed returned an unknown cbresult";
        switch (result) {
        case CELL_SAVEDATA_CBRESULT_ERR_NOSPACE: w = "funcFixed: ERR_NOSPACE"; break;
        case CELL_SAVEDATA_CBRESULT_ERR_FAILURE: w = "funcFixed: ERR_FAILURE"; break;
        case CELL_SAVEDATA_CBRESULT_ERR_BROKEN:  w = "funcFixed: ERR_BROKEN";  break;
        case CELL_SAVEDATA_CBRESULT_ERR_NODATA:  w = "funcFixed: ERR_NODATA -- it found nothing it wanted in our dirList"; break;
        case CELL_SAVEDATA_CBRESULT_ERR_INVALID: w = "funcFixed: ERR_INVALID"; break;
        }
        printf("[cellSaveData] savedata_fixed: cbresult=%d listed=%u/%u prefix='%s' root='%s'\n",
               result, listed, count,
               prefix ? (const char*)(vm_base + prefix) : "", s_save_root);
        SAVEDATA_REJECT(CELL_SAVEDATA_ERROR_CBRESULT, w);
    }
    if (!funcStat)
        SAVEDATA_REJECT(CELL_SAVEDATA_ERROR_PARAM, "funcStat is NULL");
    u32 selected = vm_read32(set);
    if (!selected || vm_read32(set + 8) > 1)
        SAVEDATA_REJECT(CELL_SAVEDATA_ERROR_PARAM, "funcFixed set no dirName, or an out-of-range option");
    const char* name = (const char*)vm_base + selected;
    size_t len = strnlen(name, CELL_SAVEDATA_DIRNAME_SIZE);
    if (!len || len == CELL_SAVEDATA_DIRNAME_SIZE || strchr(name, '/') ||
        strchr(name, '\\') || strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
        SAVEDATA_REJECT(CELL_SAVEDATA_ERROR_PARAM, "funcFixed set an unusable dirName");
    /* The selected string may be inside callback scratch, reused by funcStat. */
    char directory[CELL_SAVEDATA_DIRNAME_SIZE];
    memcpy(directory, name, len + 1);
    printf("[cellSaveData] %s: proceeding with dir='%s' (%u listed)\n",
           __func__, directory, listed);
    return savedata_execute(directory, is_save, setBuf, funcStat, funcFile,
                            (void*)(uintptr_t)vm_read32(cb + 16));
}

s32 cellSaveDataFixedSave2(u32 version, CellSaveDataSetList* setList,
                            CellSaveDataSetBuf* setBuf,
                            CellSaveDataFixedCallback funcFixed,
                            CellSaveDataStatCallback funcStat,
                            CellSaveDataFileCallback funcFile,
                            u32 container, void* userdata)
{
    (void)container;
    printf("[cellSaveData] FixedSave2(version=%u)\n", version);
    return savedata_fixed(1, setList, setBuf, funcFixed, funcStat, funcFile, userdata);
}

s32 cellSaveDataFixedLoad2(u32 version, CellSaveDataSetList* setList,
                            CellSaveDataSetBuf* setBuf,
                            CellSaveDataFixedCallback funcFixed,
                            CellSaveDataStatCallback funcStat,
                            CellSaveDataFileCallback funcFile,
                            u32 container, void* userdata)
{
    (void)container;
    printf("[cellSaveData] FixedLoad2(version=%u)\n", version);
    return savedata_fixed(0, setList, setBuf, funcFixed, funcStat, funcFile, userdata);
}

s32 cellSaveDataListAutoLoad(u32 version, u32 errDialog,
                            CellSaveDataSetList* setList, CellSaveDataSetBuf* setBuf,
                            CellSaveDataFixedCallback funcFixed,
                            CellSaveDataStatCallback funcStat,
                            CellSaveDataFileCallback funcFile,
                            u32 container, void* userdata)
{
    (void)container;
    if (errDialog > 2) return CELL_SAVEDATA_ERROR_PARAM;
    printf("[cellSaveData] ListAutoLoad(version=%u)\n", version);
    return savedata_fixed(0, setList, setBuf, funcFixed, funcStat, funcFile, userdata);
}

/* The save half of ListAutoLoad: the title picks the directory from the
 * listing in funcFixed (an empty listing on a first save, when it names a new
 * one), then funcStat/funcFile write it. Ratchet & Clank: Tools of
 * Destruction saves through this; unimplemented, it answered CELL_OK without
 * calling back and the title reported its game data corrupted. */
s32 cellSaveDataListAutoSave(u32 version, u32 errDialog,
                             CellSaveDataSetList* setList, CellSaveDataSetBuf* setBuf,
                             CellSaveDataFixedCallback funcFixed,
                             CellSaveDataStatCallback funcStat,
                             CellSaveDataFileCallback funcFile,
                             u32 container, void* userdata)
{
    (void)container;
    if (errDialog > 2) return CELL_SAVEDATA_ERROR_PARAM;
    printf("[cellSaveData] ListAutoSave(version=%u)\n", version);
#ifdef _WIN32
    /* hitch104 02/11: the title Auto Save Thread runs this; each autosave of the user 36-min play log coincided with a 310-480 ms frame. Log the wall time. */
    LARGE_INTEGER hq0, hq1, hqf; QueryPerformanceCounter(&hq0); QueryPerformanceFrequency(&hqf);
    /* hitch116: PS3_SAVEDATA_ASYNC=1 defers the file writes of THIS call to a worker (see savedata_async_queue). */
    t_sd_async = savedata_async_enabled();
    const s32 hr = savedata_fixed(1, setList, setBuf, funcFixed, funcStat, funcFile, userdata);
    t_sd_async = 0;
    QueryPerformanceCounter(&hq1);
    fprintf(stderr, "[savedata-time] ListAutoSave took %.1f ms (rc=0x%X)\n",
            (double)(hq1.QuadPart - hq0.QuadPart) * 1000.0 / (double)hqf.QuadPart, (unsigned)hr);
    fflush(stderr);
    return hr;
#else
    return savedata_fixed(1, setList, setBuf, funcFixed, funcStat, funcFile, userdata);
#endif
}

s32 cellSaveDataAutoSave2(u32 version, const char* dirName,
                           u32 errDialog,
                           CellSaveDataSetBuf* setBuf,
                           CellSaveDataStatCallback funcStat,
                           CellSaveDataFileCallback funcFile,
                           u32 container, void* userdata)
{
    dirName = savedata_host_str(dirName);
    printf("[cellSaveData] AutoSave2(version=%u, dir='%s')\n",
           version, dirName ? dirName : "<null>");

    /* Delegate to the non-_2 entry, which marshals. This used to call
     * savedata_execute(), which invokes funcStat as a HOST function pointer
     * with HOST struct addresses -- but funcStat is a guest OPD and the structs
     * must live in guest memory, so the callback wrote nothing the caller could
     * read and the first dereference faulted. The Simpsons Arcade Game crashed
     * exactly here on its first boot, creating its profile right after
     * AutoLoad2 had correctly reported ERR_NODATA. The two entries take
     * identical arguments; _2 is just the newer SDK name.
     *
     * savedata_execute() is still behind the List* and Fixed* entries and
     * carries the same defect for each of them. Fixing those needs list/file
     * marshalling that dispatch_func_stat does not do yet, so they are left
     * alone until a title actually reaches one. */
    return cellSaveDataAutoSave(version, dirName, errDialog, setBuf,
                                funcStat, funcFile, container, userdata);
}

s32 cellSaveDataAutoLoad2(u32 version, const char* dirName,
                           u32 errDialog,
                           CellSaveDataSetBuf* setBuf,
                           CellSaveDataStatCallback funcStat,
                           CellSaveDataFileCallback funcFile,
                           u32 container, void* userdata)
{
    dirName = savedata_host_str(dirName);
    (void)version; (void)errDialog; (void)setBuf; (void)funcFile;
    (void)container;
    printf("[cellSaveData] AutoLoad2(version=%u, dir='%s')\n",
           version, dirName ? dirName : "<null>");

    if (!dirName || !setBuf || !funcStat)
        return CELL_SAVEDATA_ERROR_PARAM;

    /* funcStat is the GAME's guest-OPD address. Marshal StatGet/Set/CB
     * into guest BE memory and dispatch through g_ps3_guest_caller.
     * Even on first-run-no-data, the game observes funcStat fire with
     * isNewData=1 so its title state machine can advance. */
    char save_path[1024];
    build_save_path(save_path, sizeof(save_path), dirName);
    int is_new = !dir_has_save(save_path);

    (void)is_new;
    /* Full sequence: funcStat, then the funcFile loop that reads the files.
     * Stopping after funcStat (OK_NEXT) left a title with a save on disk
     * waiting for its data forever -- GH3's second boot sat on "Checking HDD".
     *
     * flОw first-boot: its funcStat returns ERR_NODATA on a new profile
     * (isNewData=1). The callback already ran and told the game "no save",
     * so report AutoLoad as CELL_OK -- an ERROR return leaves the title
     * parked in MODE_AUTO_LOAD (no app loop, no flips).
     * Per the SDK the correct return here is CELL_SAVEDATA_ERROR_NODATA,
     * and a real title handles it; that flОw does not is a bug somewhere in
     * its MODE_AUTO_LOAD state machine we have not tracked down. Keep the
     * compat return until that is understood -- it was dropped once already
     * in the fold merge and cost a boot regression. */
    s32 r = savedata_execute(dirName, 0, setBuf, funcStat, funcFile, userdata);
    return r == CELL_SAVEDATA_ERROR_NODATA ? CELL_OK : r;
}

s32 cellSaveDataDelete2(u32 container)
{
    printf("[cellSaveData] Delete2(container=%u)\n", container);

    /* Without a directory name we can't delete anything meaningful.
       This variant typically shows a UI for deletion - just succeed. */
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * Old / non-_2 variants — same wrapper pattern as the _2 versions, with
 * the same guest-callback caveat.
 *
 * Older PS3 SDK builds (e.g. our flOw NPUA80001 dump) link the original
 * cellSaveDataAutoSave / AutoLoad / Delete instead of the _2 variants
 * RPCS3's flOw build uses. Same semantics, different NID.
 * -----------------------------------------------------------------------*/
/* The pre-3.00 List/Fixed entry points. Identical to their "2" forms except
 * that v2 added a trailing `userdata` the callbacks echo back; RPCS3 forwards
 * v1 with userdata = NULL and so do we. Registering only the "2" forms left a
 * title on the older API getting the unresolved-NID default, so its save-data
 * callbacks never fired and whatever it was loading never completed. Virtua
 * Fighter 5 calls FixedLoad during its boot and stops there.
 *
 * Same shape as cellMsgDialogOpen vs Open2 and cellGameDataCheckCreate vs
 * Create2 -- three of the four v1/v2 pairs in this tree had the same gap. */
s32 cellSaveDataListSave(u32 version, CellSaveDataSetList* setList,
                         CellSaveDataSetBuf* setBuf,
                         CellSaveDataListCallback funcList,
                         CellSaveDataStatCallback funcStat,
                         CellSaveDataFileCallback funcFile, u32 container)
{
    return cellSaveDataListSave2(version, setList, setBuf, funcList, funcStat,
                                 funcFile, container, NULL);
}

s32 cellSaveDataListLoad(u32 version, CellSaveDataSetList* setList,
                         CellSaveDataSetBuf* setBuf,
                         CellSaveDataListCallback funcList,
                         CellSaveDataStatCallback funcStat,
                         CellSaveDataFileCallback funcFile, u32 container)
{
    return cellSaveDataListLoad2(version, setList, setBuf, funcList, funcStat,
                                 funcFile, container, NULL);
}

s32 cellSaveDataFixedSave(u32 version, CellSaveDataSetList* setList,
                          CellSaveDataSetBuf* setBuf,
                          CellSaveDataFixedCallback funcFixed,
                          CellSaveDataStatCallback funcStat,
                          CellSaveDataFileCallback funcFile, u32 container)
{
    return cellSaveDataFixedSave2(version, setList, setBuf, funcFixed, funcStat,
                                  funcFile, container, NULL);
}

s32 cellSaveDataFixedLoad(u32 version, CellSaveDataSetList* setList,
                          CellSaveDataSetBuf* setBuf,
                          CellSaveDataFixedCallback funcFixed,
                          CellSaveDataStatCallback funcStat,
                          CellSaveDataFileCallback funcFile, u32 container)
{
    return cellSaveDataFixedLoad2(version, setList, setBuf, funcFixed, funcStat,
                                  funcFile, container, NULL);
}

s32 cellSaveDataAutoSave(u32 version, const char* dirName,
                          u32 errDialog,
                          CellSaveDataSetBuf* setBuf,
                          CellSaveDataStatCallback funcStat,
                          CellSaveDataFileCallback funcFile,
                          u32 container, void* userdata)
{
    dirName = savedata_host_str(dirName);
    (void)errDialog; (void)funcFile; (void)container;
    printf("[cellSaveData] AutoSave(version=%u, dir='%s')\n",
           version, dirName ? dirName : "<null>");
    if (!dirName || !setBuf || !funcStat)
        return CELL_SAVEDATA_ERROR_PARAM;

    /* This used to return CELL_OK WITHOUT running funcStat -- the game believed
     * its profile saved while nothing happened (silent data loss). Run the
     * guest funcStat callback the same proven way cellSaveDataAutoLoad does
     * (dispatch_func_stat marshals through g_ps3_guest_caller). The callback
     * fires and returns a real result the game can act on. (Persisting the file
     * bytes needs the funcFile marshalling loop; the stat callback running is
     * the correct, honest first step and matches the load path's behaviour.) */
    char save_path[1024];
    build_save_path(save_path, sizeof(save_path), dirName);
    int is_new = !dir_has_save(save_path);

    (void)is_new;
    /* Run the whole sequence: funcStat, then the funcFile loop that writes the
     * files. Stopping after funcStat left the title waiting for file callbacks
     * that never came -- GH3 sat on "Saving..." forever after its first
     * autosave. savedata_execute marshals both callbacks through guest memory
     * (dispatch_func_stat_full / dispatch_func_file). */
    return savedata_execute(dirName, 1, setBuf, funcStat, funcFile, userdata);
}

s32 cellSaveDataAutoLoad(u32 version, const char* dirName,
                          u32 errDialog,
                          CellSaveDataSetBuf* setBuf,
                          CellSaveDataStatCallback funcStat,
                          CellSaveDataFileCallback funcFile,
                          u32 container, void* userdata)
{
    dirName = savedata_host_str(dirName);
    (void)version; (void)errDialog; (void)setBuf; (void)funcFile;
    (void)container;
    printf("[cellSaveData] AutoLoad(version=%u, dir='%s')\n",
           version, dirName ? dirName : "<null>");
    if (!dirName || !setBuf || !funcStat)
        return CELL_SAVEDATA_ERROR_PARAM;

    char save_path[1024];
    build_save_path(save_path, sizeof(save_path), dirName);
    int is_new = !dir_has_save(save_path);

    (void)is_new;
    /* Same as cellSaveDataAutoLoad2 above, including flОw's first-boot compat
     * return (it calls this old non-_2 variant). */
    s32 r = savedata_execute(dirName, 0, setBuf, funcStat, funcFile, userdata);
    return r == CELL_SAVEDATA_ERROR_NODATA ? CELL_OK : r;
}

s32 cellSaveDataDelete(u32 version, const char* dirName,
                        u32 container)
{
    (void)version;
    dirName = savedata_host_str(dirName);
    printf("[cellSaveData] Delete(dir='%s', container=%u)\n",
           dirName ? dirName : "<null>", container);
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * User variants (cellSaveDataUserAutoSave/Load): same as the non-User forms
 * with a userId inserted after version. The trailing userdata is the guest's
 * NINTH argument (on the guest stack, not in r3-r10) — the generic adapter
 * only passes 8 registers, so declare 8 params and pass NULL through.
 * LittleBigPlanet auto-loads its profile through UserAutoLoad at boot.
 * -----------------------------------------------------------------------*/
s32 cellSaveDataUserAutoSave(u32 version, u32 userId, const char* dirName,
                             u32 errDialog, CellSaveDataSetBuf* setBuf,
                             CellSaveDataStatCallback funcStat,
                             CellSaveDataFileCallback funcFile, u32 container)
{
    (void)userId;
    printf("[cellSaveData] UserAutoSave(user=%u)\n", userId);
    return cellSaveDataAutoSave(version, dirName, errDialog, setBuf,
                                funcStat, funcFile, container, NULL);
}

/* NOTE: the User* entry points take NINE arguments -- cellSaveDataUserAutoLoad
 * inserts `userId` ahead of dirName, which pushes `userdata` into the ninth
 * slot. The generic HLE adapter only forwards r3..r10 (eight), so the ninth
 * argument is unreachable here and MUST come from the caller's parameter save
 * area. That is what hle_cellSaveDataUserAutoLoad (below) is for; this plain-C
 * entry only sees eight and cannot know userdata. It stays for direct/internal
 * callers, and the ctx handler overrides it on the NID path. */
/* ---------------------------------------------------------------------------
 * Nine-argument entry points
 *
 * cellSaveDataUserAutoLoad/UserAutoSave take nine arguments; the generic HLE
 * adapter forwards only r3..r10, so the ninth (userdata) never arrives and used
 * to be hardcoded to NULL. That is fatal: a funcStat callback typically recovers
 * its own object from cbResult->userdata. LBP's is
 *     v = cbResult->userdata;  r = sub(v + 104, cbResult, get, set);
 * so a NULL made it work off guest address 104, read the VM's zero-fill, and
 * return CELL_SAVEDATA_CBRESULT_ERR_NODATA (-4) -- it never reached its handler.
 *
 * Take the full context and read the ninth argument from the caller's parameter
 * save area. SAVEDATA_ARG9_SP_OFF can be overridden at runtime
 * (PS3_SAVEDATA_ARG9_OFF) and the surrounding window is dumped once, so the
 * offset is confirmed from a real call rather than assumed.
 * -----------------------------------------------------------------------*/
#include "../../runtime/ppu/ppu_context.h"

extern void ps3_hle_register_ctx(uint32_t nid, const char* name, void (*fn)(ppu_context*));

/* PPC64 ELFv1: back chain, CR, LR, 2 reserved, TOC = 48 bytes, then the
 * parameter save area shadowing r3.. one doubleword each -> arg9 at SP+112. */
#define SAVEDATA_ARG9_SP_OFF 112u

static uint32_t savedata_arg9(ppu_context* ctx)
{
    uint32_t sp = (uint32_t)ctx->gpr[1];
    static int off = -1;
    if (off < 0) {
        const char* e = getenv("PS3_SAVEDATA_ARG9_OFF");
        off = e ? (int)strtol(e, NULL, 0) : (int)SAVEDATA_ARG9_SP_OFF;
    }
    { static int dumped = 0;
      if (!dumped) { dumped = 1;
        printf("[cellSaveData] caller SP=0x%08X param-save-area window:\n", sp);
        for (uint32_t o = 48; o <= 136; o += 8)
            printf("    SP+%3u = 0x%016llX%s\n", o,
                   (unsigned long long)vm_read64(sp + o),
                   o == (uint32_t)off ? "   <- taking as arg9 (userdata)" : "");
      } }
    return (uint32_t)vm_read64(sp + (uint32_t)off);
}

static void hle_cellSaveDataUserAutoLoad(ppu_context* ctx)
{
    u32   version   = (u32)ctx->gpr[3];
    u32   userId    = (u32)ctx->gpr[4];
    const char* dir = (const char*)(uintptr_t)(u32)ctx->gpr[5];
    u32   errDialog = (u32)ctx->gpr[6];
    void* setBuf    = (void*)(uintptr_t)(u32)ctx->gpr[7];
    void* funcStat  = (void*)(uintptr_t)(u32)ctx->gpr[8];
    void* funcFile  = (void*)(uintptr_t)(u32)ctx->gpr[9];
    u32   container = (u32)ctx->gpr[10];
    uint32_t userdata = savedata_arg9(ctx);

    printf("[cellSaveData] UserAutoLoad(user=%u, userdata=0x%08X)\n", userId, userdata);
    ctx->gpr[3] = (uint64_t)(int64_t)(s32)
        cellSaveDataAutoLoad2(version, dir, errDialog,
                              (CellSaveDataSetBuf*)setBuf,
                              (CellSaveDataStatCallback)funcStat,
                              (CellSaveDataFileCallback)funcFile,
                              container, (void*)(uintptr_t)userdata);
}

/* UserAutoSave takes the same nine arguments as UserAutoLoad. Through the
 * generic eight-argument binding the ninth (userdata) is lost, so the title's
 * stat and file callbacks see cbResult->userdata == 0. */
static void hle_cellSaveDataUserAutoSave(ppu_context* ctx)
{
    u32   version   = (u32)ctx->gpr[3];
    u32   userId    = (u32)ctx->gpr[4];
    const char* dir = (const char*)(uintptr_t)(u32)ctx->gpr[5];
    u32   errDialog = (u32)ctx->gpr[6];
    void* setBuf    = (void*)(uintptr_t)(u32)ctx->gpr[7];
    void* funcStat  = (void*)(uintptr_t)(u32)ctx->gpr[8];
    void* funcFile  = (void*)(uintptr_t)(u32)ctx->gpr[9];
    u32   container = (u32)ctx->gpr[10];
    uint32_t userdata = savedata_arg9(ctx);

    printf("[cellSaveData] UserAutoSave(user=%u, userdata=0x%08X)\n", userId, userdata);
    /* cellSaveDataAutoSave, not AutoSave2: AutoSave2 maps dirName to a host
     * pointer and passes it on to AutoSave, which maps it again; a host
     * address below 4 GiB would then be read as a guest one. */
    ctx->gpr[3] = (uint64_t)(int64_t)(s32)
        cellSaveDataAutoSave(version, dir, errDialog,
                             (CellSaveDataSetBuf*)setBuf,
                             (CellSaveDataStatCallback)funcStat,
                             (CellSaveDataFileCallback)funcFile,
                             container, (void*)(uintptr_t)userdata);
}

/* ListAutoLoad's ninth argument is in the guest parameter save area. */
void ps3_savedata_list_auto_load(ppu_context* ctx)
{
    ctx->gpr[3] = (uint64_t)(int64_t)cellSaveDataListAutoLoad(
        (u32)ctx->gpr[3], (u32)ctx->gpr[4],
        (CellSaveDataSetList*)(uintptr_t)(u32)ctx->gpr[5],
        (CellSaveDataSetBuf*)(uintptr_t)(u32)ctx->gpr[6],
        (CellSaveDataFixedCallback)(uintptr_t)(u32)ctx->gpr[7],
        (CellSaveDataStatCallback)(uintptr_t)(u32)ctx->gpr[8],
        (CellSaveDataFileCallback)(uintptr_t)(u32)ctx->gpr[9],
        (u32)ctx->gpr[10], (void*)(uintptr_t)savedata_arg9(ctx));
}

/* ListAutoSave: the same nine arguments as ListAutoLoad. */
static void hle_cellSaveDataListAutoSave(ppu_context* ctx)
{
    ctx->gpr[3] = (uint64_t)(int64_t)cellSaveDataListAutoSave(
        (u32)ctx->gpr[3], (u32)ctx->gpr[4],
        (CellSaveDataSetList*)(uintptr_t)(u32)ctx->gpr[5],
        (CellSaveDataSetBuf*)(uintptr_t)(u32)ctx->gpr[6],
        (CellSaveDataFixedCallback)(uintptr_t)(u32)ctx->gpr[7],
        (CellSaveDataStatCallback)(uintptr_t)(u32)ctx->gpr[8],
        (CellSaveDataFileCallback)(uintptr_t)(u32)ctx->gpr[9],
        (u32)ctx->gpr[10], (void*)(uintptr_t)savedata_arg9(ctx));
}

void cellSaveData_register_ctx_handlers(void)
{
    /* Overrides the generated 8-arg registration: ctx handlers are dispatched
     * before the generic table. */
    ps3_hle_register_ctx(0x4DD03A4Eu, "cellSaveDataListAutoSave",
                         hle_cellSaveDataListAutoSave);
    ps3_hle_register_ctx(0xCDC6AEFDu, "cellSaveDataUserAutoLoad",
                         hle_cellSaveDataUserAutoLoad);
    ps3_hle_register_ctx(0x52AAC4FAu, "cellSaveDataUserAutoSave",
                         hle_cellSaveDataUserAutoSave);
}

s32 cellSaveDataUserAutoLoad(u32 version, u32 userId, const char* dirName,
                             u32 errDialog, CellSaveDataSetBuf* setBuf,
                             CellSaveDataStatCallback funcStat,
                             CellSaveDataFileCallback funcFile, u32 container)
{
    (void)userId;
    printf("[cellSaveData] UserAutoLoad(user=%u) [8-arg path: userdata unavailable]\n", userId);
    return cellSaveDataAutoLoad(version, dirName, errDialog, setBuf,
                                funcStat, funcFile, container, NULL);
}
