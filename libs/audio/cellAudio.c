/*
 * ps3recomp - cellAudio HLE implementation
 *
 * Real audio mixing and output. A background mixing thread reads audio data
 * from each active port's buffer in guest memory, mixes them, and outputs
 * to the host audio device.
 *
 * Backend selection:
 *   - Windows default: WASAPI (via mmdeviceapi)
 *   - Everywhere else / if PS3RECOMP_AUDIO_USE_SDL2 defined: SDL2 audio
 *
 * Define PS3RECOMP_AUDIO_USE_SDL2 to force SDL2 backend on Windows.
 */

#include "../../runtime/memory/vm.h"   /* vm_commit */
#include "cellAudio.h"
#include "../../runtime/ppu/ppu_memory.h"   /* vm_base, vm_read64, vm_write32 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

/* Event-queue push/lookup (runtime/syscalls/sys_event.c). Forward-declared
 * rather than including sys_event.h to avoid pulling PPU context/syscall
 * table types into the HLE audio module. */
extern int      sys_event_queue_push_by_id(uint32_t queue_id,
                                            uint64_t source, uint64_t data1,
                                            uint64_t data2,  uint64_t data3);
extern uint32_t sys_event_find_queue_by_key(uint64_t key);
extern uint32_t sys_event_queue_create_direct(uint64_t key, int32_t size);

/* ---------------------------------------------------------------------------
 * Backend selection
 * -----------------------------------------------------------------------*/

#if defined(PS3RECOMP_AUDIO_USE_SDL2)
  #define AUDIO_BACKEND_SDL2    1
  #define AUDIO_BACKEND_WASAPI  0
#elif defined(_WIN32)
  #define AUDIO_BACKEND_SDL2    0
  #define AUDIO_BACKEND_WASAPI  1
#else
  #define AUDIO_BACKEND_SDL2    1
  #define AUDIO_BACKEND_WASAPI  0
#endif

#if AUDIO_BACKEND_WASAPI
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  #include <mmdeviceapi.h>
  #include <audioclient.h>
#include <avrt.h>
#pragma comment(lib, "avrt.lib")
#ifndef AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
#define AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM      0x80000000
#endif
#ifndef AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY
#define AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY 0x08000000
#endif
  #include <process.h>

  /* Define WASAPI COM GUIDs (avoids needing uuid.lib linkage for these) */
  #ifdef __cplusplus
    #define GUID_SECT
  #else
    #define GUID_SECT
  #endif
  #ifndef DEFINE_AUDIO_GUID
    #ifdef INITGUID
      #define DEFINE_AUDIO_GUID(name, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) \
              const GUID name = { l, w1, w2, { b1, b2, b3, b4, b5, b6, b7, b8 } }
    #else
      #define DEFINE_AUDIO_GUID(name, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) \
              const GUID name = { l, w1, w2, { b1, b2, b3, b4, b5, b6, b7, b8 } }
    #endif
  #endif
  DEFINE_AUDIO_GUID(ps3r_CLSID_MMDeviceEnumerator, 0xBCDE0395,0xE52F,0x467C,0x8E,0x3D,0xC4,0x57,0x92,0x91,0x69,0x2E);
  DEFINE_AUDIO_GUID(ps3r_IID_IMMDeviceEnumerator,  0xA95664D2,0x9614,0x4F35,0xA7,0x46,0xDE,0x8D,0xB6,0x36,0x17,0xE6);
  DEFINE_AUDIO_GUID(ps3r_IID_IAudioClient,         0x1CB9AD4C,0xDBFA,0x4c32,0xB1,0x78,0xC2,0xF5,0x68,0xA7,0x03,0xB2);
  DEFINE_AUDIO_GUID(ps3r_IID_IAudioRenderClient,   0xF294ACFC,0x3146,0x4483,0xA7,0xBF,0xAD,0xDC,0xA7,0xC2,0x60,0xE2);
  DEFINE_AUDIO_GUID(ps3r_IID_IAudioClient3,        0x7ED4EE07,0x8E67,0x4CD4,0x8C,0x1A,0x2B,0x7A,0x59,0x87,0xAD,0x42);
  DEFINE_AUDIO_GUID(ps3r_IID_IMMNotificationClient, 0x7991EEC9,0x7E89,0x4D85,0x83,0x90,0x6C,0x70,0x3C,0xEC,0x60,0xC0);
  DEFINE_AUDIO_GUID(ps3r_IID_IUnknown,              0x00000000,0x0000,0x0000,0xC0,0x00,0x00,0x00,0x00,0x00,0x00,0x46);

  /* Map to the names used in the code */
  #define IID_IMMDeviceEnumerator  ps3r_IID_IMMDeviceEnumerator
  #define CLSID_MMDeviceEnumerator ps3r_CLSID_MMDeviceEnumerator
  #define IID_IAudioClient         ps3r_IID_IAudioClient
  #define IID_IAudioRenderClient   ps3r_IID_IAudioRenderClient
  #define IID_IAudioClient3        ps3r_IID_IAudioClient3
#endif

#if AUDIO_BACKEND_SDL2
  #include <SDL2/SDL.h>
#endif

/* Portable threading */
#ifdef _WIN32
  #include <windows.h>
  typedef HANDLE thread_t;
  typedef CRITICAL_SECTION mutex_t;
  #define mutex_init(m)    InitializeCriticalSection(m)
  #define mutex_destroy(m) DeleteCriticalSection(m)
  #define mutex_lock(m)    EnterCriticalSection(m)
  #define mutex_unlock(m)  LeaveCriticalSection(m)
#else
  #include <pthread.h>
  #include <unistd.h>
  #include <time.h>
  typedef pthread_t thread_t;
  typedef pthread_mutex_t mutex_t;
  #define mutex_init(m)    pthread_mutex_init(m, NULL)
  #define mutex_destroy(m) pthread_mutex_destroy(m)
  #define mutex_lock(m)    pthread_mutex_lock(m)
  #define mutex_unlock(m)  pthread_mutex_unlock(m)
#endif

/* ---------------------------------------------------------------------------
 * Internal state
 * -----------------------------------------------------------------------*/

/* Per-port audio buffer allocated on the host side.
 * In a full emulator these would be in guest VM memory; here we allocate
 * host buffers and expose their addresses through portAddr/readIndexAddr. */
#define AUDIO_PORT_BUF_MAX  (CELL_AUDIO_BLOCK_32 * CELL_AUDIO_BLOCK_SAMPLES * CELL_AUDIO_PORT_8CH)

typedef struct {
    int                  in_use;
    int                  running;
    CellAudioPortParam   param;
    float*               buffer;        /* host audio buffer (float samples) */
    u32                  buf_size;       /* buffer size in bytes */
    u64                  read_index;     /* current read position (block index) */
    u64                  write_index;    /* where the game is writing */
    float                level;          /* cellAudioSetPortLevel gain, 0..1 */
    /* For address reporting to guest */
    u64                  port_addr;      /* guest-visible buffer address */
    u64                  read_idx_addr;  /* guest-visible read index address */
} AudioPortSlot;

static int            s_audio_initialized = 0;
static AudioPortSlot  s_ports[CELL_AUDIO_PORT_MAX];

/* Event queue notification */
typedef struct {
    int  in_use;
    u64  key;
} AudioNotifySlot;
static AudioNotifySlot s_notify_queues[CELL_AUDIO_MAX_NOTIFY_EVENT_QUEUES];

/* Timestamp anchor for cellAudioGetPortTimestamp (microseconds, set at init) */
static u64            s_audio_start_us = 0;

/* Mixing thread */
static unsigned long long g_audio_dropped;   /* frames the device had no room for */
static volatile int  s_mix_thread_running = 0;
static thread_t      s_mix_thread;
#ifndef _WIN32
/* Is s_mix_thread a thread that exists and has not been joined? The Win32
 * branch asks that of the HANDLE itself, which it nulls after closing;
 * pthread_t has no such value, so carry the flag. */
static int           s_mix_thread_live = 0;
#endif
static mutex_t       s_audio_mutex;

/* Output mix buffer (stereo, one block worth) */
static float s_mix_buffer[CELL_AUDIO_BLOCK_SAMPLES * 2];
static unsigned long long g_audio_gaps, g_audio_gap_blocks;   /* AUDIO_GAPS totals */

/* The host's hold on the output (dod3: DOD3_UNFOCUSED, the window out of
 * focus). 1: mixed as usual, played as silence. 2: paused -- the ports are
 * not read and the title is not notified, so its audio (music, a movie's
 * soundtrack) waits where it is; the device is fed silence. */
volatile int g_audio_hold = 0;

/* ---------------------------------------------------------------------------
 * Host audio output backend
 * -----------------------------------------------------------------------*/

#if AUDIO_BACKEND_SDL2

static SDL_AudioDeviceID s_sdl_audio_dev = 0;

static int audio_backend_init(void)
{
    if (SDL_WasInit(SDL_INIT_AUDIO) == 0) {
#if defined(SDL_HINT_NO_SIGNAL_HANDLERS) && !defined(PS3RECOMP_RSX_VULKAN)
        /* SDL would take SIGINT and SIGTERM and nothing would act on them:
         * see pad_sdl_start in libs/input/cellPad.c. Whichever of the two
         * starts SDL first has to say so. */
        SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
#endif
        if (SDL_InitSubSystem(SDL_INIT_AUDIO) < 0) {
            printf("[cellAudio] SDL audio init failed: %s\n", SDL_GetError());
            return -1;
        }
    }

    SDL_AudioSpec want, have;
    memset(&want, 0, sizeof(want));
    want.freq     = CELL_AUDIO_SAMPLE_RATE;
    want.format   = AUDIO_F32SYS;
    want.channels = 2;
    want.samples  = CELL_AUDIO_BLOCK_SAMPLES;
    want.callback = NULL; /* We'll use SDL_QueueAudio */

    s_sdl_audio_dev = SDL_OpenAudioDevice(NULL, 0, &want, &have, 0);
    if (s_sdl_audio_dev == 0) {
        printf("[cellAudio] SDL_OpenAudioDevice failed: %s\n", SDL_GetError());
        return -1;
    }

    SDL_PauseAudioDevice(s_sdl_audio_dev, 0); /* Start playback */
    return 0;
}

static void audio_backend_shutdown(void)
{
    if (s_sdl_audio_dev) {
        SDL_CloseAudioDevice(s_sdl_audio_dev);
        s_sdl_audio_dev = 0;
    }
}

static void audio_backend_submit(const float* stereo_samples, u32 num_samples)
{
    if (s_sdl_audio_dev) {
        SDL_QueueAudio(s_sdl_audio_dev, stereo_samples,
                       num_samples * 2 * sizeof(float));
    }
}

/* SDL queues without bound: keep about four blocks ahead of the device. */
static int audio_backend_room(void)
{
    if (!s_sdl_audio_dev) return -1;
    return SDL_GetQueuedAudioSize(s_sdl_audio_dev) / (2 * sizeof(float)) <
           CELL_AUDIO_BLOCK_SAMPLES * 4 ? CELL_AUDIO_BLOCK_SAMPLES : 0;
}

static void audio_backend_wait(void) { SDL_Delay(1); }

static u32 audio_backend_queued_samples(void)
{
    if (s_sdl_audio_dev) {
        return SDL_GetQueuedAudioSize(s_sdl_audio_dev) / (2 * sizeof(float));
    }
    return 0;
}

#endif /* AUDIO_BACKEND_SDL2 */

#if AUDIO_BACKEND_WASAPI

static IAudioClient*        s_wasapi_client = NULL;
static IAudioRenderClient*  s_wasapi_render = NULL;
static HANDLE               s_wasapi_event  = NULL;
static UINT32               s_wasapi_buf_frames = 0;
static int                  s_wasapi_lowlat = 0;        /* opened through IAudioClient3 (small engine period) */
static UINT32               s_wasapi_period_frames = 0; /* engine period of the low-latency stream, else 0 */
static long long            s_wasapi_stream_lat_us = 0; /* IAudioClient::GetStreamLatency, for the logs */

/* Low-latency path (IAudioClient3). Its device buffer is only a few engine
 * periods (~6 ms at the 2.67 ms minimum), far less than the reserve the mixer
 * keeps against its own scheduling delays, so the reserve lives in a software
 * ring here and a small render thread, woken by the device event every
 * period, moves ring -> device. The mixer sees the ring as the device queue
 * (padding = frames in the ring, room = free frames), so its pacing and
 * underrun logic are unchanged. Starving the ring plays silence and counts
 * in s_rt_starved (the mixer treats it like an underrun and grows its reserve). */
#define AUDIO_RING_FRAMES 4096u   /* power of two */
static float                s_ring[AUDIO_RING_FRAMES * 2];
static volatile LONG        s_ring_head = 0, s_ring_tail = 0;   /* frames written / read, monotonic */
static volatile LONG        s_rt_starved = 0;
static volatile LONG        s_rt_devunder = 0;   /* the DEVICE was empty when the render thread woke (a glitch no ring counter sees) */
static volatile LONG        s_rt_stop = 0;
static HANDLE               s_rt_thread = NULL;
static UINT32               s_wasapi_dev_frames = 0;

static volatile LONG s_wasapi_reopen;   /* tentative def, see below */
static unsigned __stdcall audio_render_thread_func(void* arg)
{
    (void)arg;
    DWORD ti = 0;
    const char* mme = getenv("AUDIO_MMCSS");   /* same default as the mixer: off unless AUDIO_MMCSS=1 (see audio_mix_thread_func) */
    HANDLE mm = (mme && mme[0] == '1') ? AvSetMmThreadCharacteristicsW(L"Pro Audio", &ti) : NULL;
    if (mm) AvSetMmThreadPriority(mm, AVRT_PRIORITY_HIGH); else SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    SetThreadDescription(GetCurrentThread(), L"cellAudio render");
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
    int warm = 0;
    while (!s_rt_stop) {
        WaitForSingleObject(s_wasapi_event, 20);
        if (s_rt_stop) break;
        UINT32 pad = 0;
        if (FAILED(s_wasapi_client->lpVtbl->GetCurrentPadding(s_wasapi_client, &pad))) {
            InterlockedExchange(&s_wasapi_reopen, 1);
            Sleep(5); continue;
        }
        UINT32 avail = s_wasapi_dev_frames - pad;
        if (!avail) continue;
        if (pad == 0 && warm) InterlockedIncrement(&s_rt_devunder);
        const LONG head = s_ring_head; MemoryBarrier();
        UINT32 fill = (UINT32)(head - s_ring_tail);
        UINT32 n = fill < avail ? fill : avail;
        if (fill < avail && warm) InterlockedIncrement(&s_rt_starved);
        /* 27/10: only the frames really available go to the device (GetBuffer/ReleaseBuffer(n)); the rest of the
         * device buffer is not filled with zeros, which put a hole of silence before the next samples (click). */
        if (!n) continue;
        BYTE* buf = NULL;
        if (FAILED(s_wasapi_render->lpVtbl->GetBuffer(s_wasapi_render, n, &buf)) || !buf) continue;
        float* out = (float*)buf;
        UINT32 t = (UINT32)s_ring_tail;
        for (UINT32 i = 0; i < n; i++) {
            const UINT32 k = (t + i) & (AUDIO_RING_FRAMES - 1u);
            out[2 * i] = s_ring[2 * k]; out[2 * i + 1] = s_ring[2 * k + 1];
        }
        warm = 1;
        MemoryBarrier();
        s_ring_tail = (LONG)(t + n);
        s_wasapi_render->lpVtbl->ReleaseBuffer(s_wasapi_render, n, 0);
    }
    return 0;
}

/* Following the Windows default output. The device used to be picked once,
 * at cellAudioInit: switching the default output in Windows (speakers to
 * headphones) left the game on the old one, and when the old one went away
 * (a USB or Bluetooth headset unplugged) GetCurrentPadding failed with
 * AUDCLNT_E_DEVICE_INVALIDATED and the game stayed silent for good. Now an
 * IMMNotificationClient raises s_wasapi_reopen when the default render
 * endpoint changes, audio_backend_room raises it when the device is gone, and
 * the mixing thread -- the only thread that touches the client once it runs --
 * closes the stream and opens the current default (audio_backend_reopen).
 * AUDIO_FOLLOW_DEFAULT=0 keeps the device chosen at start. */
static IMMDeviceEnumerator* s_wasapi_enum = NULL;
static volatile LONG        s_wasapi_reopen = 0;
static int                  s_wasapi_follow = 1;

static HRESULT STDMETHODCALLTYPE notify_QueryInterface(IMMNotificationClient* self, REFIID riid, void** out)
{
    if (IsEqualGUID(riid, &ps3r_IID_IUnknown) || IsEqualGUID(riid, &ps3r_IID_IMMNotificationClient)) {
        *out = self;
        return S_OK;
    }
    *out = NULL;
    return E_NOINTERFACE;
}
/* One static object that lives as long as the process: no counting needed. */
static ULONG STDMETHODCALLTYPE notify_AddRef(IMMNotificationClient* self) { (void)self; return 1; }
static ULONG STDMETHODCALLTYPE notify_Release(IMMNotificationClient* self) { (void)self; return 1; }
static HRESULT STDMETHODCALLTYPE notify_OnDeviceStateChanged(IMMNotificationClient* self, LPCWSTR id, DWORD state)
{ (void)self; (void)id; (void)state; return S_OK; }
static HRESULT STDMETHODCALLTYPE notify_OnDeviceAdded(IMMNotificationClient* self, LPCWSTR id)
{ (void)self; (void)id; return S_OK; }
static HRESULT STDMETHODCALLTYPE notify_OnDeviceRemoved(IMMNotificationClient* self, LPCWSTR id)
{ (void)self; (void)id; return S_OK; }
static HRESULT STDMETHODCALLTYPE notify_OnDefaultDeviceChanged(IMMNotificationClient* self, EDataFlow flow, ERole role,
                                                               LPCWSTR id)
{
    (void)self; (void)id;
    /* Called on a system thread: only raise the flag, the mixer does the work. */
    if (flow == eRender && role == eConsole) InterlockedExchange(&s_wasapi_reopen, 1);
    return S_OK;
}
static HRESULT STDMETHODCALLTYPE notify_OnPropertyValueChanged(IMMNotificationClient* self, LPCWSTR id,
                                                               const PROPERTYKEY key)
{ (void)self; (void)id; (void)key; return S_OK; }

static IMMNotificationClientVtbl s_notify_vtbl = {
    notify_QueryInterface, notify_AddRef, notify_Release,
    notify_OnDeviceStateChanged, notify_OnDeviceAdded, notify_OnDeviceRemoved,
    notify_OnDefaultDeviceChanged, notify_OnPropertyValueChanged,
};
static IMMNotificationClient s_notify = { &s_notify_vtbl };

/* Open the current default render endpoint into s_wasapi_client/_render. */
static int audio_wasapi_open(void)
{
    HRESULT hr;
    IMMDevice* device = NULL;
    hr = s_wasapi_enum->lpVtbl->GetDefaultAudioEndpoint(s_wasapi_enum, eRender,
                                                         eConsole, &device);
    static int no_endpoint;   /* reported once per stretch without a device */
    if (FAILED(hr)) {
        if (!no_endpoint) printf("[cellAudio] Failed to get default audio endpoint: 0x%08lX\n", hr);
        no_endpoint = 1;
        return -1;
    }
    no_endpoint = 0;

    hr = device->lpVtbl->Activate(device, &IID_IAudioClient, CLSCTX_ALL,
                                   NULL, (void**)&s_wasapi_client);
    device->lpVtbl->Release(device);
    if (FAILED(hr)) {
        printf("[cellAudio] Failed to activate audio client: 0x%08lX\n", hr);
        return -1;
    }

    WAVEFORMATEX wfx;
    wfx.wFormatTag      = WAVE_FORMAT_IEEE_FLOAT;
    wfx.nChannels       = 2;
    wfx.nSamplesPerSec  = CELL_AUDIO_SAMPLE_RATE;
    wfx.wBitsPerSample  = 32;
    wfx.nBlockAlign     = wfx.nChannels * (wfx.wBitsPerSample / 8);
    wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;
    wfx.cbSize          = 0;

    /* Low-latency shared stream (IAudioClient3, Windows 10+): the engine
     * period of a classic shared stream is ~10 ms and the padding the mixer
     * sees drops in steps of that size, so a cushion of N ms is really N
     * minus up to a period, and the engine adds one or two periods of its
     * own on the way out. Asking for the minimum period the driver offers
     * (typically 2.7-10 ms) takes that off the A/V delay. The format must be
     * the one the engine mixes in (no automatic conversion here): when it is
     * not, or the call fails, the classic path below runs unchanged.
     * AUDIO_LOWLAT=1 enables (default off: classic path); AUDIO_PERIOD_FRAMES=<n> asks for a period. */
    s_wasapi_lowlat = 0; s_wasapi_period_frames = 0;
    { const char* e = getenv("AUDIO_LOWLAT");
      if (e && e[0] == '1') {
          IAudioClient3* ac3 = NULL;
          if (SUCCEEDED(s_wasapi_client->lpVtbl->QueryInterface(s_wasapi_client, &IID_IAudioClient3, (void**)&ac3)) && ac3) {
              UINT32 defp = 0, fund = 0, minp = 0, maxp = 0;
              if (SUCCEEDED(ac3->lpVtbl->GetSharedModeEnginePeriod(ac3, &wfx, &defp, &fund, &minp, &maxp)) && minp) {
                  /* 27/10: not the driver minimum (2.67 ms: the render thread would have to run every 2.67 ms,
                   * and any 5 ms delay empties the device): 480 frames = 10 ms unless AUDIO_PERIOD_FRAMES says otherwise. */
                  UINT32 want = 480;
                  { const char* pe = getenv("AUDIO_PERIOD_FRAMES"); if (pe && atol(pe) > 0) want = (UINT32)atol(pe); }
                  if (want < minp) want = minp;
                  if (want > maxp) want = maxp;
                  if (fund > 0) want = ((want + fund - 1) / fund) * fund;
                  s_wasapi_event = CreateEventW(NULL, FALSE, FALSE, NULL);
                  hr = ac3->lpVtbl->InitializeSharedAudioStream(ac3, AUDCLNT_STREAMFLAGS_EVENTCALLBACK, want, &wfx, NULL);
                  if (SUCCEEDED(hr)) {
                      s_wasapi_lowlat = 1; s_wasapi_period_frames = want;
                      s_rt_devunder = 0;
                      s_wasapi_client->lpVtbl->SetEventHandle(s_wasapi_client, s_wasapi_event);
                      printf("[cellAudio] WASAPI low-latency shared stream: period %u frames (%.2f ms; min %u, default %u, max %u)\n",
                             (unsigned)want, 1000.0 * want / CELL_AUDIO_SAMPLE_RATE, (unsigned)minp, (unsigned)defp, (unsigned)maxp);
                  } else {
                      printf("[cellAudio] WASAPI low-latency init failed 0x%08lX -- classic shared stream\n", hr);
                      CloseHandle(s_wasapi_event); s_wasapi_event = NULL;
                  }
              }
              ac3->lpVtbl->Release(ac3);
              if (!s_wasapi_lowlat) {
                  /* A client whose Initialize failed is not reusable: start over. */
                  s_wasapi_client->lpVtbl->Release(s_wasapi_client); s_wasapi_client = NULL;
                  IMMDevice* dev2 = NULL;
                  if (FAILED(s_wasapi_enum->lpVtbl->GetDefaultAudioEndpoint(s_wasapi_enum, eRender, eConsole, &dev2)) ||
                      FAILED(dev2->lpVtbl->Activate(dev2, &IID_IAudioClient, CLSCTX_ALL, NULL, (void**)&s_wasapi_client))) {
                      if (dev2) dev2->lpVtbl->Release(dev2);
                      return -1;
                  }
                  dev2->lpVtbl->Release(dev2);
              }
          }
      } }

    /* Event-driven mode, with an AUDIO_BUF_MS buffer (default 64 ms). The
     * mixer below keeps it nearly full, so this is how long a scheduling
     * hiccup -- the mixer or the guest's audio thread held off the CPU --
     * can last before the device runs dry. 20 ms stuttered on Drakengard 3
     * under load; the PS3's own output path is several 5.33 ms blocks deep. */
    long buf_ms = 64;
    { const char* e = getenv("AUDIO_BUF_MS"); if (e && atol(e) > 0) buf_ms = atol(e); }
    REFERENCE_TIME buf_duration = (REFERENCE_TIME)buf_ms * 10000; /* 100 ns units */
    if (!s_wasapi_lowlat)
    s_wasapi_event = CreateEventW(NULL, FALSE, FALSE, NULL);

    /* AUTOCONVERTPCM | SRC_DEFAULT_QUALITY: let the shared-mode mixer convert
     * and resample. Without them Initialize answers
     * AUDCLNT_E_UNSUPPORTED_FORMAT (0x88890008) on an endpoint whose mix
     * format is not 48 kHz stereo float -- a 44.1 kHz or 7.1 device -- and
     * the title ran silent. */
    const DWORD convert = AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
    if (!s_wasapi_lowlat)
    hr = s_wasapi_client->lpVtbl->Initialize(
        s_wasapi_client,
        AUDCLNT_SHAREMODE_SHARED,
        AUDCLNT_STREAMFLAGS_EVENTCALLBACK | convert,
        buf_duration, 0, &wfx, NULL);

    if (!s_wasapi_lowlat && FAILED(hr)) {
        /* Fallback: try without event callback */
        hr = s_wasapi_client->lpVtbl->Initialize(
            s_wasapi_client,
            AUDCLNT_SHAREMODE_SHARED,
            convert, buf_duration, 0, &wfx, NULL);
        if (FAILED(hr)) {
            printf("[cellAudio] WASAPI Initialize failed: 0x%08lX\n", hr);
            s_wasapi_client->lpVtbl->Release(s_wasapi_client);
            s_wasapi_client = NULL;
            CloseHandle(s_wasapi_event);
            s_wasapi_event = NULL;
            return -1;
        }
        CloseHandle(s_wasapi_event);
        s_wasapi_event = NULL;
    } else if (!s_wasapi_lowlat) {
        s_wasapi_client->lpVtbl->SetEventHandle(s_wasapi_client, s_wasapi_event);
    }

    s_wasapi_client->lpVtbl->GetBufferSize(s_wasapi_client, &s_wasapi_buf_frames);
    s_wasapi_dev_frames = s_wasapi_buf_frames;
    if (s_wasapi_lowlat && s_wasapi_dev_frames < 3u * s_wasapi_period_frames)
        printf("[cellAudio] WARNING low-latency device buffer %u frames is under 3 periods (%u): little margin against render-thread delays\n",
               (unsigned)s_wasapi_dev_frames, (unsigned)(3u * s_wasapi_period_frames));
    if (s_wasapi_lowlat) s_wasapi_buf_frames = AUDIO_RING_FRAMES;   /* the mixer's view: the ring */
    { REFERENCE_TIME sl = 0;
      s_wasapi_stream_lat_us = SUCCEEDED(s_wasapi_client->lpVtbl->GetStreamLatency(s_wasapi_client, &sl)) ? (long long)(sl / 10) : 0;
      printf("[cellAudio] WASAPI shared, %s, buffer %u frames (%.1f ms), stream latency %.1f ms\n",
             s_wasapi_event ? "event-driven" : "polled", (unsigned)s_wasapi_dev_frames,
             1000.0 * s_wasapi_dev_frames / CELL_AUDIO_SAMPLE_RATE, s_wasapi_stream_lat_us / 1000.0); }

    hr = s_wasapi_client->lpVtbl->GetService(
        s_wasapi_client, &IID_IAudioRenderClient, (void**)&s_wasapi_render);
    if (FAILED(hr)) {
        printf("[cellAudio] Failed to get render client: 0x%08lX\n", hr);
        s_wasapi_client->lpVtbl->Release(s_wasapi_client);
        s_wasapi_client = NULL;
        return -1;
    }

    s_wasapi_client->lpVtbl->Start(s_wasapi_client);
    if (s_wasapi_lowlat) {
        s_ring_head = s_ring_tail = 0; s_rt_stop = 0;
        s_rt_thread = (HANDLE)_beginthreadex(NULL, 0, audio_render_thread_func, NULL, 0, NULL);
        if (!s_rt_thread) { printf("[cellAudio] render thread failed to start\n"); }
    }
    return 0;
}

static void audio_wasapi_close(void)
{
    if (s_rt_thread) {
        s_rt_stop = 1;
        if (s_wasapi_event) SetEvent(s_wasapi_event);
        WaitForSingleObject(s_rt_thread, INFINITE);   /* 27/10: no timeout: the caller releases client/render right after */
        CloseHandle(s_rt_thread); s_rt_thread = NULL;
    }
    if (s_wasapi_client) {
        s_wasapi_client->lpVtbl->Stop(s_wasapi_client);
    }
    if (s_wasapi_render) {
        s_wasapi_render->lpVtbl->Release(s_wasapi_render);
        s_wasapi_render = NULL;
    }
    if (s_wasapi_client) {
        s_wasapi_client->lpVtbl->Release(s_wasapi_client);
        s_wasapi_client = NULL;
    }
    if (s_wasapi_event) {
        CloseHandle(s_wasapi_event);
        s_wasapi_event = NULL;
    }
}

static int audio_backend_init(void)
{
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr) && hr != S_FALSE && hr != RPC_E_CHANGED_MODE) {
        printf("[cellAudio] CoInitializeEx failed: 0x%08lX\n", hr);
        return -1;
    }
    hr = CoCreateInstance(&CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL,
                          &IID_IMMDeviceEnumerator, (void**)&s_wasapi_enum);
    if (FAILED(hr)) {
        printf("[cellAudio] Failed to create device enumerator: 0x%08lX\n", hr);
        s_wasapi_enum = NULL;
        return -1;
    }
    { const char* e = getenv("AUDIO_FOLLOW_DEFAULT"); s_wasapi_follow = !(e && e[0] == '0'); }
    if (s_wasapi_follow) {
        hr = s_wasapi_enum->lpVtbl->RegisterEndpointNotificationCallback(s_wasapi_enum, &s_notify);
        if (FAILED(hr))
            printf("[cellAudio] RegisterEndpointNotificationCallback failed: 0x%08lX"
                   " -- the output will not follow the Windows default\n", hr);
    }
    /* No device now (none plugged in) is not fatal when following the
     * default: the mixer opens one when it appears. */
    if (audio_wasapi_open() < 0 && !s_wasapi_follow) return -1;
    return 0;
}

static void audio_backend_shutdown(void)
{
    audio_wasapi_close();
    if (s_wasapi_enum) {
        if (s_wasapi_follow)
            s_wasapi_enum->lpVtbl->UnregisterEndpointNotificationCallback(s_wasapi_enum, &s_notify);
        s_wasapi_enum->lpVtbl->Release(s_wasapi_enum);
        s_wasapi_enum = NULL;
    }
}

/* On the mixing thread: move to the current default output if it changed or
 * the device went away. A failed open (no device at all) is retried every
 * half second; in between the mixer paces on its clock as it does with no
 * device. */
static void audio_backend_reopen(void)
{
    static ULONGLONG retry_at = 0;
    if (!s_wasapi_follow || !s_wasapi_enum) return;
    if (!s_wasapi_reopen && (s_wasapi_client || GetTickCount64() < retry_at)) return;
    InterlockedExchange(&s_wasapi_reopen, 0);
    audio_wasapi_close();
    if (audio_wasapi_open() == 0) {
        printf("[cellAudio] output moved to the Windows default device\n");
        retry_at = 0;
    } else {
        retry_at = GetTickCount64() + 500;
    }
}

static void audio_backend_submit(const float* stereo_samples, u32 num_samples)
{
    if (!s_wasapi_render) return;

    if (s_wasapi_lowlat) {   /* into the ring; the render thread feeds the device */
        const UINT32 fill = (UINT32)(s_ring_head - s_ring_tail);
        UINT32 n = num_samples;
        if (n > AUDIO_RING_FRAMES - fill) { g_audio_dropped += n - (AUDIO_RING_FRAMES - fill); n = AUDIO_RING_FRAMES - fill; }
        const UINT32 h = (UINT32)s_ring_head;
        for (UINT32 i = 0; i < n; i++) {
            const UINT32 k = (h + i) & (AUDIO_RING_FRAMES - 1u);
            s_ring[2 * k] = stereo_samples[2 * i]; s_ring[2 * k + 1] = stereo_samples[2 * i + 1];
        }
        MemoryBarrier();
        s_ring_head = (LONG)(h + n);
        return;
    }

    UINT32 padding = 0;
    s_wasapi_client->lpVtbl->GetCurrentPadding(s_wasapi_client, &padding);

    UINT32 available = s_wasapi_buf_frames - padding;
    if (num_samples > available) {
        g_audio_dropped += num_samples - available;
        num_samples = available;
    }
    if (num_samples == 0) return;

    BYTE* buf = NULL;
    HRESULT hr = s_wasapi_render->lpVtbl->GetBuffer(s_wasapi_render, num_samples, &buf);
    if (SUCCEEDED(hr) && buf) {
        memcpy(buf, stereo_samples, num_samples * 2 * sizeof(float));
        s_wasapi_render->lpVtbl->ReleaseBuffer(s_wasapi_render, num_samples, 0);
    }
}

/* Frames the device can take right now; -1 with no device open. */
static int audio_backend_room(void)
{
    if (!s_wasapi_client) return -1;
    if (s_wasapi_lowlat) return (int)(AUDIO_RING_FRAMES - (UINT32)(s_ring_head - s_ring_tail));
    UINT32 padding = 0;
    const HRESULT hr = s_wasapi_client->lpVtbl->GetCurrentPadding(s_wasapi_client, &padding);
    if (FAILED(hr)) {
        /* Device gone (AUDCLNT_E_DEVICE_INVALIDATED on a default-device
         * change, sleep, ...): pace on the clock rather than spin, and have
         * the mixer open the current default (audio_backend_reopen). */
        static int _n = 0; if (_n < 4 && ++_n)
            fprintf(stderr, "[cellAudio] GetCurrentPadding failed 0x%08lX -- clock pacing%c", (unsigned long)hr, 10);
        InterlockedExchange(&s_wasapi_reopen, 1);
        return -1;
    }
    return (int)(s_wasapi_buf_frames - padding);
}

/* Frames queued in the device now; -1 with no device. */
static int audio_backend_padding(void)
{
    if (!s_wasapi_client) return -1;
    if (s_wasapi_lowlat) return (int)(UINT32)(s_ring_head - s_ring_tail);
    UINT32 padding = 0;
    if (FAILED(s_wasapi_client->lpVtbl->GetCurrentPadding(s_wasapi_client, &padding))) return -1;
    return (int)padding;
}

/* Put `frames` of silence in the device queue (the cushion, see the mixer). */
static void audio_backend_prefill(u32 frames)
{
    static float zeros[CELL_AUDIO_BLOCK_SAMPLES * 2];
    /* 27/10: on the low-latency ring, silence goes in only when the ring is empty: on top of queued samples it is an
     * audible hole of cushion ms and a permanent extra latency. */
    if (s_wasapi_lowlat && (UINT32)(s_ring_head - s_ring_tail) > 0) return;
    while (frames >= CELL_AUDIO_BLOCK_SAMPLES) {
        audio_backend_submit(zeros, CELL_AUDIO_BLOCK_SAMPLES);
        frames -= CELL_AUDIO_BLOCK_SAMPLES;
    }
}

/* Block until the device has consumed some of its buffer. */
static void audio_backend_wait(void)
{
    if (s_wasapi_event) WaitForSingleObject(s_wasapi_event, 10);
    else Sleep(1);
}


#endif /* AUDIO_BACKEND_WASAPI */

/* ---------------------------------------------------------------------------
 * Mixing
 * -----------------------------------------------------------------------*/

/* Guest PCM is BIG-ENDIAN float32; the port buffer is a raw vm_base view, so
 * every sample must be byte-swapped before use. Reading it as a host float
 * turned the whole mix into denormal noise -- silence after clipping (LBP
 * had a verified end-to-end pipeline with no audible output). */
static inline float ld_be_f32(const float* p)
{
    u32 v;
    memcpy(&v, p, 4);
    v = (v >> 24) | ((v >> 8) & 0xFF00u) | ((v << 8) & 0xFF0000u) | (v << 24);
    float f;
    memcpy(&f, &v, 4);
    return f;
}

/* Samples that were NaN/Inf in a port buffer since the last AUDIO_CLIP_LOG
 * report (they are replaced by silence, see ld_be_f32_safe). */
static u32 s_dropped_nonfinite;

/* AUDIO_UNDERRUN_LOG=1 counters (see the mixer loop). */
static u32 s_ur_zero_reads;   /* port blocks read still cleared, after sound was heard */

/* ld_be_f32 for mixing: a non-finite sample would poison the whole mix (one
 * NaN reaches the device as a burst of noise), and an absurdly large one
 * would only be flattened by the limiter into a full-scale bang. Both
 * become harmless: NaN/Inf -> 0, magnitude capped at 16.0. */
static inline float ld_be_f32_safe(const float* p)
{
    float f = ld_be_f32(p);
    if (!isfinite(f)) { s_dropped_nonfinite++; return 0.0f; }
    if (f >  16.0f) return  16.0f;
    if (f < -16.0f) return -16.0f;
    return f;
}

/* Soft limiter on the final mix: linear up to 0.8, then a tanh knee that
 * approaches (never reaches) +-1.0 with matching slope at the joint, instead
 * of the flat-topped hard clip that turned loud overlaps into a bang. */
static inline float audio_soft_clip(float x)
{
    const float knee = 0.8f;
    float a = fabsf(x);
    if (a <= knee) return x;
    float y = knee + (1.0f - knee) * tanhf((a - knee) / (1.0f - knee));
    return x < 0.0f ? -y : y;
}

/* Mix one block from all active ports into s_mix_buffer (stereo float) */
static void audio_mix_one_block(void)
{
    memset(s_mix_buffer, 0, sizeof(s_mix_buffer));

    mutex_lock(&s_audio_mutex);

    for (int p = 0; p < CELL_AUDIO_PORT_MAX; p++) {
        AudioPortSlot* port = &s_ports[p];
        /* Diagnostic: a port that HOLDS PCM but was never started is inaudible
         * by design -- flag it once so "no PortStart" vs "no data" is decidable
         * from the log (LBP's Bink movie audio port sat exactly there). */
        if (port->in_use && !port->running && port->buffer) {
            static u8 warned[CELL_AUDIO_PORT_MAX];
            if (!warned[p]) {
                u32 probe = (u32)(port->param.nBlock * CELL_AUDIO_BLOCK_SAMPLES *
                                  port->param.nChannel);
                int nz = 0;
                for (u32 s = 0; s < probe; s += 64) if (port->buffer[s] != 0.0f) { nz = 1; break; }
                if (nz) {
                    warned[p] = 1;
                    fprintf(stderr, "[cellAudio] port %d HAS DATA but never started"
                            " -- game withheld PortStart\n", p);
                }
            }
        }
        if (!port->in_use || !port->running || !port->buffer)
            continue;

        u32 nch    = (u32)port->param.nChannel;
        u32 nblock = (u32)port->param.nBlock;
        /* Per-port gain from cellAudioSetPortLevel (muting at 0 is real). */
        float level = port->level;
        if (!(level >= 0.0f)) level = 0.0f;
        if (level > 1.0f) level = 1.0f;

        /* AUDIO_WRITEPOS=1: where the game writes relative to the index we
         * publish. Each mix, the blocks that went from cleared to written since
         * the last mix are counted by (block - published) mod nblock; the
         * histogram is printed every 1000 writes. This is the game's write
         * policy, measured, where the gap maps only showed snapshots. */
        { static int wp = -1; if (wp < 0) wp = getenv("AUDIO_WRITEPOS") ? 1 : 0;
          if (wp) {
              static u8 was[CELL_AUDIO_PORT_MAX][32]; static u32 hist[CELL_AUDIO_PORT_MAX][32]; static u32 total[CELL_AUDIO_PORT_MAX];
              const u32 pub = (u32)(port->read_index % nblock);
              for (u32 b = 0; b < nblock && b < 32; b++) {
                  const float* bp = port->buffer + b * CELL_AUDIO_BLOCK_SAMPLES * nch;
                  int nz = 0;
                  for (u32 s2 = 0; s2 < CELL_AUDIO_BLOCK_SAMPLES * nch; s2 += 16) if (bp[s2] != 0.0f) { nz = 1; break; }
                  if (nz && !was[p][b]) { hist[p][(b + nblock - pub) % nblock]++; total[p]++; }
                  was[p][b] = (u8)nz;
              }
              if (total[p] && (total[p] % 1000) == 0) {
                  fprintf(stderr, "[audio-writepos] port %d (%u blocks): writes land at published+", p, nblock);
                  for (u32 k = 0; k < nblock && k < 32; k++) if (hist[p][k]) fprintf(stderr, " %u:%u", k, hist[p][k]);
                  fprintf(stderr, "  (of %u)%c", total[p], 10);
              }
          } }
        /* Read one block at the current read_index -- or AUDIO_LAG_BLOCKS
         * behind it (default 0). A lag gives a producer that writes just
         * AHEAD of the published index more time, but it is wrong for one
         * that fills the slots just BEHIND it: Drakengard 3's MultiStream
         * writes at published-1 and published-2 (AUDIO_WRITEPOS=1: 12,144 of
         * 13,000 writes), keeping six or seven blocks of lead, so a lag of 2
         * read the very block being written and 2-5% of blocks played as
         * silence; with no lag the same run had none. */
        static int lag = -1;
        if (lag < 0) { const char* e = getenv("AUDIO_LAG_BLOCKS");
                       lag = e ? atoi(e) : 0;
                       if (lag < 0) lag = 0; }
        const u32 lag_b = (u32)lag < nblock - 1u ? (u32)lag : nblock - 2u;
        u32 block_idx = (u32)((port->read_index + nblock - lag_b) % nblock);
        u32 block_offset = block_idx * CELL_AUDIO_BLOCK_SAMPLES * nch;
        float* src = port->buffer + block_offset;

        /* AUDIO_PEAK diag: which blocks of this port's WHOLE ring hold data,
         * vs which block the read cursor is on -- distinguishes "writer never
         * writes" from "writer and reader cycle out of phase". */
        { static int s_pk = -1; if (s_pk < 0) s_pk = getenv("AUDIO_PEAK") ? 1 : 0;
          if (s_pk) { static _Thread_local unsigned _c;
            if ((++_c % 400) == 0) {
                char map[36]; u32 b;
                for (b = 0; b < nblock && b < 32; b++) {
                    const float* bp = port->buffer + b * CELL_AUDIO_BLOCK_SAMPLES * nch;
                    int nz = 0;
                    for (u32 s2 = 0; s2 < CELL_AUDIO_BLOCK_SAMPLES * nch; s2 += 16)
                        if (bp[s2] != 0.0f) { nz = 1; break; }
                    map[b] = nz ? '#' : '.';
                }
                map[b] = 0;
                fprintf(stderr, "[audio-ring] port %d ridx=%llu blk=%u ring=[%s]\n",
                        p, (unsigned long long)port->read_index, block_idx, map);
            } } }

        /* AUDIO_GAPS=1: a block still cleared when it is read, in the middle
         * of sound, is one the game did not write in time -- heard as a gap.
         * Each run of them that ends within 40 blocks (213 ms) of sound is
         * logged with the RSX frame it ended in, to line up with
         * [stutter]/[frametime]; longer runs are taken to be real silence. */
        { static int s_gaps = -1; if (s_gaps < 0) s_gaps = getenv("AUDIO_GAPS") ? 1 : 0;
          if (s_gaps) {
              static u32 run[CELL_AUDIO_PORT_MAX]; static u8 heard[CELL_AUDIO_PORT_MAX];
              static unsigned logged;
              /* AUDIO_GAPS=<n>: how many gaps to log one by one (default 400);
               * the totals below keep counting past it. */
              static unsigned cap = 0; if (!cap) { const char* e = getenv("AUDIO_GAPS"); cap = (e && atoi(e) > 1) ? (unsigned)atoi(e) : 400u; }
              extern uint32_t g_rsx_engine_frame;
              const u32 n = CELL_AUDIO_BLOCK_SAMPLES * nch;
              const u32* w = (const u32*)src;
              u32 k = 0;
              while (k < n && !w[k]) k++;
              if (k == n) run[p]++;
              else {
                  if (run[p] && run[p] <= 40 && heard[p]) {
                      g_audio_gaps++; g_audio_gap_blocks += run[p];
                      if (logged++ < cap) {
                          /* The ring now: '#' a block with data, '.' cleared, 'R' the one just read. */
                          char map[40]; u32 b;
                          for (b = 0; b < nblock && b < 32; b++) {
                              const float* bp = port->buffer + b * CELL_AUDIO_BLOCK_SAMPLES * nch;
                              int nz = 0;
                              for (u32 s2 = 0; s2 < CELL_AUDIO_BLOCK_SAMPLES * nch; s2 += 16) if (bp[s2] != 0.0f) { nz = 1; break; }
                              map[b] = b == block_idx ? 'R' : (nz ? '#' : '.');
                          }
                          map[b] = 0;
                          fprintf(stderr, "[audio-gap] port %d: %u block(s) (%.1f ms) not written in time, "
                                  "ending at frame %u; ring [%s] published %u\n", p, run[p],
                                  run[p] * CELL_AUDIO_BLOCK_SAMPLES * 1000.0 / CELL_AUDIO_SAMPLE_RATE,
                                  g_rsx_engine_frame, map, (unsigned)(port->read_index % nblock));
                      }
                  }
                  run[p] = 0; heard[p] = 1;
              }
          } }

        /* AUDIO_CH_RMS=1: every ~5 s, each channel's RMS and peak on this port
         * (before the downmix): where the game puts its sound -- LFE is dropped. */
        { static int on = -1; if (on < 0) on = getenv("AUDIO_CH_RMS") ? 1 : 0;
          if (on && nch <= 8) {
              static double ss[CELL_AUDIO_PORT_MAX][8]; static float pk[CELL_AUDIO_PORT_MAX][8]; static u32 nb[CELL_AUDIO_PORT_MAX];
              for (u32 s = 0; s < CELL_AUDIO_BLOCK_SAMPLES; s++)
                  for (u32 c = 0; c < nch; c++) {
                      const float v = ld_be_f32(&src[s * nch + c]);
                      if (!isfinite(v)) continue;
                      ss[p][c] += (double)v * v;
                      if (fabsf(v) > pk[p][c]) pk[p][c] = fabsf(v);
                  }
              if (++nb[p] >= 938) {
                  char buf[512]; int o = snprintf(buf, sizeof buf, "[audio-ch] port %d (%u ch) rms/peak:", p, nch);
                  double rr[8] = {0};
                  for (u32 c = 0; c < nch; c++) {
                      rr[c] = sqrt(ss[p][c] / (nb[p] * CELL_AUDIO_BLOCK_SAMPLES));
                      o += snprintf(buf + o, sizeof buf - o, " %.4f/%.3f", rr[c], (double)pk[p][c]);
                      ss[p][c] = 0; pk[p][c] = 0;
                  }
                  fprintf(stderr, "%s level %.2f%c", buf, (double)level, 10);
                  /* Self-diagnosis of the "channels 2-5 stay at zero for the whole run" state
                   * (about 1 run in 18, see RIPARTENZA 26/10 audiorace): L/R carry sound
                   * (rms > 0.005 in this window) while C, LFE, Ls, Rs are all exactly zero.
                   * One line per window, so a run can be classified from its .err alone and
                   * stopped early: after 3 consecutive BAD windows (>= 15 s) it says so once. */
                  if (nch >= 6) {
                      static u32 badrun[CELL_AUDIO_PORT_MAX], goodw[CELL_AUDIO_PORT_MAX];
                      const int lr = (rr[0] > 0.005 || rr[1] > 0.005);
                      const int surr = (rr[2] > 1e-6 || rr[3] > 1e-6 || rr[4] > 1e-6 || rr[5] > 1e-6);
                      if (lr && !surr) badrun[p]++; else if (lr && surr) { badrun[p] = 0; goodw[p]++; }
                      fprintf(stderr, "[audio-ch-state] port %d: %s (bad windows in a row %u, good windows %u)%c", p,
                              !lr ? "QUIET" : surr ? "OK (ch2-5 live)" : "BAD (L/R live, ch2-5 zero)", badrun[p], goodw[p], 10);
                      if (badrun[p] == 3 && goodw[p] == 0)
                          fprintf(stderr, "[audio-ch-alert] port %d: channels 2-5 silent from the start of the run%c", p, 10);
                  }
                  nb[p] = 0;
              }
          } }

        /* AUDIO_UNDERRUN_LOG: count blocks the guest had not written when read. */
        { static int ul = -1; if (ul < 0) ul = getenv("AUDIO_UNDERRUN_LOG") ? 1 : 0;
          if (ul) {
              static u8 heard2[CELL_AUDIO_PORT_MAX];
              const u32 n2 = CELL_AUDIO_BLOCK_SAMPLES * nch;
              const u32* w2 = (const u32*)src; u32 k2 = 0;
              while (k2 < n2 && !w2[k2]) k2++;
              if (k2 == n2) { if (heard2[p]) s_ur_zero_reads++; }
              else heard2[p] = 1;
          } }

        /* AUDIO_PORT_RAW=<file>: every block of every port with 8 channels
         * except port 0, raw (host-endian f32, all channels), prefixed by a
         * u32 port index and u32 block index -- the movie audio's layout. */
        { static FILE* rf = (FILE*)-1;
          if (rf == (FILE*)-1) { const char* e = getenv("AUDIO_PORT_RAW"); rf = e ? fopen(e, "wb") : NULL; }
          if (rf && p != 0) {
              u32 hdr[2] = { (u32)p, block_idx };
              fwrite(hdr, 4, 2, rf);
              for (u32 k = 0; k < CELL_AUDIO_BLOCK_SAMPLES * nch; k++) {
                  float f = ld_be_f32(&src[k]); fwrite(&f, 4, 1, rf); }
          } }

        /* Declick (AUDIO_DECLICK=0 off): a block the game did not write in
         * time reads as zeros, and the step from sound to 0 and back is the
         * crackle. An unwritten block instead decays the port's last output
         * sample to 0 over the block, and the first written block after one
         * ramps in over 64 samples. */
        static int s_dc = -1;
        if (s_dc < 0) { const char* e = getenv("AUDIO_DECLICK"); s_dc = !(e && e[0] == '0'); }
        static float dc_last[CELL_AUDIO_PORT_MAX][2]; static u8 dc_hole[CELL_AUDIO_PORT_MAX];
        int dc_empty = 0;
        if (s_dc) {
            const u32 n3 = CELL_AUDIO_BLOCK_SAMPLES * nch;
            const u32* w3 = (const u32*)src; u32 k3 = 0;
            while (k3 < n3 && !w3[k3]) k3++;
            dc_empty = (k3 == n3);
        }
        if (dc_empty) {
            const float l0 = dc_last[p][0] * level, r0 = dc_last[p][1] * level;
            if (l0 != 0.0f || r0 != 0.0f) {
                for (u32 s = 0; s < CELL_AUDIO_BLOCK_SAMPLES; s++) {
                    const float g = 1.0f - (float)(s + 1) / CELL_AUDIO_BLOCK_SAMPLES;
                    s_mix_buffer[s * 2 + 0] += l0 * g;
                    s_mix_buffer[s * 2 + 1] += r0 * g;
                }
            }
            dc_last[p][0] = dc_last[p][1] = 0.0f;
            dc_hole[p] = 1;
        }
        const int dc_ramp = !dc_empty && dc_hole[p];
        if (!dc_empty) dc_hole[p] = 0;
        for (u32 s = 0; s < CELL_AUDIO_BLOCK_SAMPLES && !dc_empty; s++) {
            float left, right;
            if (nch >= 2) {
                left  = ld_be_f32_safe(&src[s * nch + 0]);
                right = ld_be_f32_safe(&src[s * nch + 1]);
            } else {
                /* Mono: duplicate to both channels */
                left = right = ld_be_f32_safe(&src[s]);
            }

            /* 5.1 / 7.1 -> stereo, ITU-style: centre and surrounds at -3 dB
             * or lower, LFE left out (the console's stereo downmix drops it,
             * and its energy is what pushed impacts over full scale). The
             * result is scaled by 1/(1 + 0.707), the gain of the front pair
             * plus centre, so a full-scale front/centre signal stays in
             * range; the soft limiter below catches the rest.
             * Channel order: FL FR C LFE RL RR [SL SR]. */
            if (nch == 8 || nch == 6) {
                const float norm = 1.0f / 1.707f;
                float center = ld_be_f32_safe(&src[s * nch + 2]) * 0.707f;
                float rl     = ld_be_f32_safe(&src[s * nch + 4]) * 0.5f;
                float rr     = ld_be_f32_safe(&src[s * nch + 5]) * 0.5f;
                float sl = 0.0f, sr = 0.0f;
                if (nch == 8) {
                    sl = ld_be_f32_safe(&src[s * nch + 6]) * 0.5f;
                    sr = ld_be_f32_safe(&src[s * nch + 7]) * 0.5f;
                }
                left  = (left  + center + rl + sl) * norm;
                right = (right + center + rr + sr) * norm;
            }

            if (s_dc) {
                if (s == CELL_AUDIO_BLOCK_SAMPLES - 1) { dc_last[p][0] = left; dc_last[p][1] = right; }
                if (dc_ramp && s < 64) { const float g = (float)(s + 1) / 64.0f; left *= g; right *= g; }
            }

            left  *= level;
            right *= level;

            s_mix_buffer[s * 2 + 0] += left;
            s_mix_buffer[s * 2 + 1] += right;
        }

        /* Firmware clears a ring-buffer block after consuming it. Besides
         * preventing stale PCM from replaying when a producer stalls, the
         * cleared block is an observable consumption signal for producers. */
        memset(src, 0, CELL_AUDIO_BLOCK_SAMPLES * nch * sizeof(float));

        /* Keep the monotonic counter for tags/timestamps, but publish the
         * ring slot to the guest. SurMixer multiplies this value by the block
         * size directly; an unbounded counter walks beyond the audio buffer. */
        port->read_index++;
        if (port->read_idx_addr)
            vm_write64((u32)port->read_idx_addr, port->read_index % nblock);
    }

    mutex_unlock(&s_audio_mutex);

    /* AUDIO_CLIP_LOG=1: per second, the mixed samples over full scale (hard
     * clipped below), the peak and the non-finite ones -- whether a "bang"
     * is clipping or garbage. */
    { static int on = -1; static u32 blocks, over, nonfin; static float peak;
      if (on < 0) on = getenv("AUDIO_CLIP_LOG") ? 1 : 0;
      if (on) {
          nonfin += s_dropped_nonfinite; s_dropped_nonfinite = 0;
          for (u32 i = 0; i < CELL_AUDIO_BLOCK_SAMPLES * 2; i++) {
              const float v = s_mix_buffer[i];
              if (!isfinite(v)) { nonfin++; continue; }
              const float a = fabsf(v);
              if (a > 1.0f) over++;
              if (a > peak) peak = a;
          }
          if (++blocks >= 188) {
              if (over || nonfin)
                  fprintf(stderr, "[audio-clip] %u of %u samples over 1.0, peak %.2f, non-finite %u (port0 nch %u)\n",
                          over, blocks * CELL_AUDIO_BLOCK_SAMPLES * 2, (double)peak, nonfin,
                          (u32)s_ports[0].param.nChannel);
              blocks = over = nonfin = 0; peak = 0.0f;
          }
      } }
    s_dropped_nonfinite = 0;
    /* Soft-limit to (-1.0, 1.0) (the over-1.0 count above is taken before this) */
    for (u32 i = 0; i < CELL_AUDIO_BLOCK_SAMPLES * 2; i++) {
        float v = s_mix_buffer[i];
        if (!isfinite(v)) v = 0.0f;
        s_mix_buffer[i] = audio_soft_clip(v);
    }
}

/* ---------------------------------------------------------------------------
 * Mixing thread
 * -----------------------------------------------------------------------*/

static void audio_notify_event_queues(void)
{
    /* Push the audio-period event (data1 = CELL_AUDIO_EVENT_MIX = 0) to each
     * registered notify queue, so the game's audio loop (blocked on
     * sys_event_queue_receive) wakes once per block and writes the next one.
     * Resolve the queue by its ipc_key (set via cellAudioSetNotifyEventQueue). */
    for (int i = 0; i < CELL_AUDIO_MAX_NOTIFY_EVENT_QUEUES; i++) {
        if (!s_notify_queues[i].in_use) continue;
        uint32_t qid = sys_event_find_queue_by_key(s_notify_queues[i].key);
        if (qid)
            sys_event_queue_push_by_id(qid, s_notify_queues[i].key,
                                       0 /*CELL_AUDIO_EVENT_MIX*/, 0, 0);
    }
}

/* Default depth of the output reserve in ms (see AUDIO_LATENCY_MS in the mixer). */
#ifndef AUDIO_LATENCY_DEFAULT_MS
#define AUDIO_LATENCY_DEFAULT_MS 32
#endif
#if AUDIO_BACKEND_WASAPI
#define AUDIO_STREAM_LAT_MS() (s_wasapi_stream_lat_us / 1000.0)
#else
#define AUDIO_STREAM_LAT_MS() 0.0
#endif

#ifdef _WIN32
static unsigned __stdcall audio_mix_thread_func(void* arg)
{
    (void)arg;
    /* The mixer is what keeps the device fed: MMCSS "Pro Audio" puts it in
     * the scheduler class Windows reserves for audio engines, ahead of every
     * normal and high-priority thread of the game; without avrt (or if it
     * refuses) the thread is TIME_CRITICAL. */
    /* 30/10: MMCSS is OFF by default (AUDIO_MMCSS=1 turns it on). Measured on the Ryzen 5 3600: with the
     * mixer registered in "Pro Audio", a dozen OTHER threads of the game (the spinning PPU main thread
     * among them, 100% of a core) showed current priority 25 -- the MMCSS real-time band -- and the SPU
     * workers and the D3D12 driver threads sat at 11; with AUDIO_MMCSS=0 every one of them was back at
     * 10-13. How the registration spreads is not understood; TIME_CRITICAL (15) keeps the mixer above
     * every game thread without that side effect. */
    { const char* e = getenv("AUDIO_MMCSS");
      DWORD task_index = 0;
      HANDLE mm = (e && e[0] == '1') ? AvSetMmThreadCharacteristicsW(L"Pro Audio", &task_index) : NULL;
      if (mm) AvSetMmThreadPriority(mm, AVRT_PRIORITY_HIGH);
      else SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
      SetThreadDescription(GetCurrentThread(), L"cellAudio mixer");
      printf("[cellAudio] Mixing thread started (%s)\n", mm ? "MMCSS Pro Audio" : "TIME_CRITICAL"); }
#if AUDIO_BACKEND_WASAPI
    /* audio_backend_reopen creates COM objects on this thread. */
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
#endif

    /* Pace to the device clock. The PS3 consumes one block every 5.33 ms;
     * the old Sleep(2)/Sleep(5) loop keyed on a queue depth the 1056-frame
     * WASAPI buffer can never reach, so it mixed ~225 blocks/s and the device
     * dropped the surplus: 20% of the audio thrown away, and a guest mixer
     * (GH3's FMOD) that could not keep ahead of the read index played gaps.
     * With no device, pace on the performance counter instead. */
    /* One block every 5.33 ms by the clock, as the hardware delivers them.
     *
     * Pacing on device room alone mixed in BURSTS: WASAPI drains its buffer
     * a 10 ms period at a time, so two blocks' worth of room opened at once
     * and two blocks went out back to back, each with a notify. The title's
     * notify queue is one event deep (it expects one interrupt per block),
     * the second event was lost, it wrote one block for the two consumed,
     * and one block in every seven or eight played as silence: 13% of
     * Drakengard 3's battle audio missing, a regular crackle, while the Mac
     * (SDL, one block per callback) was clean. The device buffer is now only
     * a safety valve: full means our clock runs ahead of the DAC's, so the
     * schedule slips a block; nearly empty means it runs behind, so the
     * schedule gains one. */
    LARGE_INTEGER qf, q0; QueryPerformanceFrequency(&qf); QueryPerformanceCounter(&q0);
    unsigned long long blocks = 0;
    const long long period = (long long)CELL_AUDIO_BLOCK_SAMPLES * qf.QuadPart / CELL_AUDIO_SAMPLE_RATE;
    HANDLE pace_timer = CreateWaitableTimerExW(NULL, NULL, 0x00000002 /* HIGH_RESOLUTION */, TIMER_ALL_ACCESS);

    /* Output cushion. The mixer submits exactly one block per block period,
     * so the device queue holds ~1 block whatever AUDIO_BUF_MS is: the buffer
     * was a size, not a reserve. Any delay of this thread over ~5 ms (a
     * heavy frame with the game's threads saturating the cores -- crate
     * breaks, explosions) ran the device dry and WASAPI played silence: a
     * click per hitch. Now the queue is seeded with AUDIO_CUSHION_MS of
     * silence (default 32 ms = 6 blocks, within the 64 ms buffer) and stays
     * that deep, because blocks still leave at the clock rate; a late mixer
     * eats cushion instead of the output. The guest's notify cadence is
     * unchanged (wall clock). After a real underrun the cushion is re-seeded. */
    /* AUDIO_LATENCY_MS (tod.ini) is the depth of that reserve = the delay the
     * output adds to what the guest published (lip-sync of movies, which the
     * game times on the read index). AUDIO_CUSHION_MS, if set, wins. The
     * reserve is capped by what the device buffer can hold, and grows by a
     * block on every real underrun up to AUDIO_LATENCY_MAX_MS, so a machine
     * that needs more than the default settles there instead of crackling. */
    u32 cushion = 0, cushion_max = 0;
    { long ms = AUDIO_LATENCY_DEFAULT_MS;
      { const char* e = getenv("AUDIO_LATENCY_MS"); if (e && atol(e) >= 0) ms = atol(e); }
      { const char* e = getenv("AUDIO_CUSHION_MS"); if (e && atol(e) >= 0) ms = atol(e); }
      long mx = ms > 64 ? ms : 64;
      { const char* e = getenv("AUDIO_LATENCY_MAX_MS"); if (e && atol(e) >= ms) mx = atol(e); }
      cushion = (u32)(ms * CELL_AUDIO_SAMPLE_RATE / 1000);
      cushion -= cushion % CELL_AUDIO_BLOCK_SAMPLES;
      cushion_max = (u32)(mx * CELL_AUDIO_SAMPLE_RATE / 1000);
      cushion_max -= cushion_max % CELL_AUDIO_BLOCK_SAMPLES;
#if AUDIO_BACKEND_WASAPI
      /* leave room for the block being submitted and one more */
      if (s_wasapi_buf_frames > 2 * CELL_AUDIO_BLOCK_SAMPLES) {
          const u32 cap = ((s_wasapi_buf_frames - 2 * CELL_AUDIO_BLOCK_SAMPLES) / CELL_AUDIO_BLOCK_SAMPLES) * CELL_AUDIO_BLOCK_SAMPLES;
          if (cushion > cap) cushion = cap;
          if (cushion_max > cap) cushion_max = cap;
      }
#endif
      if (cushion_max < cushion) cushion_max = cushion;
      printf("[cellAudio] output latency target %u ms (max %u ms), AUDIO_LATENCY_MS to change\n",
             (unsigned)(cushion * 1000u / CELL_AUDIO_SAMPLE_RATE), (unsigned)(cushion_max * 1000u / CELL_AUDIO_SAMPLE_RATE)); }
    int cushion_seeded = 0;
    LONG rt_seen = 0; unsigned ur_trims = 0;
    const int ur_log = getenv("AUDIO_UNDERRUN_LOG") ? 1 : 0;
    unsigned ur_underruns = 0, ur_late = 0, ur_blocks = 0, ur_min_pad = 1u << 30;
    long long ur_max_late_us = 0;
    unsigned long long ur_pad_sum = 0, ur_pad_n = 0;
    ULONGLONG ur_t0 = GetTickCount64();
    while (s_mix_thread_running) {
        LARGE_INTEGER now; QueryPerformanceCounter(&now);
        const long long due = q0.QuadPart + (long long)blocks * period;
        if (now.QuadPart < due) {
            const long long wait_us = (due - now.QuadPart) * 1000000LL / qf.QuadPart;
            if (wait_us > 1200 && pace_timer) {
                LARGE_INTEGER rel; rel.QuadPart = -(wait_us - 400) * 10;   /* leave the tail to the spin */
                SetWaitableTimer(pace_timer, &rel, 0, NULL, NULL, FALSE);
                WaitForSingleObject(pace_timer, 5);
            } else {
                SwitchToThread();
            }
            continue;
        }
        { const long long late_us = (now.QuadPart - due) * 1000000LL / qf.QuadPart;
          if (late_us > 2000) { ur_late++; }
          if (late_us > ur_max_late_us) ur_max_late_us = late_us; }
        /* More than 50 ms behind (a debugger stop, the device going away):
         * drop the backlog instead of mixing it in a burst. */
        if (now.QuadPart - due > qf.QuadPart / 20) q0.QuadPart = now.QuadPart - (long long)blocks * period;
#if AUDIO_BACKEND_WASAPI
        audio_backend_reopen();
#endif
        { const int pad = audio_backend_padding();
          if (pad >= 0) {
              if (!cushion_seeded) {
                  if (pad == 0 && cushion) audio_backend_prefill(cushion);
                  cushion_seeded = 1;
              } else if ((pad == 0 || s_rt_starved != rt_seen) && blocks > 8) {
                  rt_seen = s_rt_starved;
                  /* The device ran dry before this block: an underrun. */
                  ur_underruns++;
                  if (cushion < cushion_max) cushion += CELL_AUDIO_BLOCK_SAMPLES;
                  if (cushion) audio_backend_prefill(cushion);
              }
              if ((unsigned)pad < ur_min_pad) ur_min_pad = (unsigned)pad;
              ur_pad_sum += (unsigned long long)pad + CELL_AUDIO_BLOCK_SAMPLES; ur_pad_n++;   /* queue right after this block goes in */
          } else cushion_seeded = 0; }
        const int room = audio_backend_room();
        if (room >= 0 && room < CELL_AUDIO_BLOCK_SAMPLES) { q0.QuadPart += period; continue; }   /* DAC slower: slip */
        if (room >= 0 && (UINT32)room >= s_wasapi_buf_frames - CELL_AUDIO_BLOCK_SAMPLES && blocks > 8)
            q0.QuadPart -= period;                                                                /* DAC faster: gain */
        /* Low-latency ring: if something held the render thread up once (device
         * start, a long stall) the backlog would stay for good, because the
         * ring only drains at the device rate. Above the reserve + 2 blocks,
         * let the schedule slip a block per period until it is back. */
        if (s_wasapi_lowlat && (unsigned)audio_backend_padding() > cushion + 2 * CELL_AUDIO_BLOCK_SAMPLES && blocks > 8) {
            q0.QuadPart += period; ur_trims++; continue;
        }
        blocks++;
        if (g_audio_hold == 2) {
            memset(s_mix_buffer, 0, sizeof s_mix_buffer);
            audio_backend_submit(s_mix_buffer, CELL_AUDIO_BLOCK_SAMPLES);
            continue;
        }
        /* Mix and submit one block */
        audio_mix_one_block();
        if (g_audio_hold) memset(s_mix_buffer, 0, sizeof s_mix_buffer);
        /* AUDIO_PEAK=1: report the mixed block's peak amplitude periodically so
         * "is any port producing sound" is answerable (LBP Bink movie audio). */
        { static int _ap = -1; if (_ap < 0) _ap = getenv("AUDIO_PEAK") ? 1 : 0;
          if (_ap) { static unsigned _n = 0; float pk = 0.0f;
            for (u32 i = 0; i < CELL_AUDIO_BLOCK_SAMPLES * 2; i++) {
                float a = s_mix_buffer[i]; if (a < 0) a = -a; if (a > pk) pk = a; }
            if ((++_n % 200) == 0 || (pk > 0.001f && _n < 40))
                fprintf(stderr, "[audio-peak] block#%u peak=%.4f\n", _n, pk); } }
        /* AUDIO_WAV=<file>: append the final mix as raw f32le stereo 48 kHz
         * (ffmpeg -f f32le -ar 48000 -ac 2 -i <file>). */
        { static FILE* _wf = (FILE*)-1;
          if (_wf == (FILE*)-1) { const char* e = getenv("AUDIO_WAV"); _wf = e ? fopen(e, "wb") : NULL; }
          if (_wf) fwrite(s_mix_buffer, sizeof(float), CELL_AUDIO_BLOCK_SAMPLES * 2, _wf); }
        audio_backend_submit(s_mix_buffer, CELL_AUDIO_BLOCK_SAMPLES);
        /* AUDIO_RATE=1: blocks mixed vs real time, and frames dropped. */
        { static int on = -1; if (on < 0) on = getenv("AUDIO_RATE") ? 1 : 0;
          if (on) { static ULONGLONG t0; static unsigned n, z; static unsigned long long d0;
            ULONGLONG now = GetTickCount64(); if (!t0) t0 = now; n++;
            { int nz = 0; for (u32 i = 0; i < CELL_AUDIO_BLOCK_SAMPLES * 2; i++) if (s_mix_buffer[i] != 0.0f) { nz = 1; break; }
              if (!nz) z++; }
            if (now - t0 >= 5000) {
                static unsigned long long g0, gb0;
                fprintf(stderr, "[audio-rate] gaps %llu (%llu blocks, %.1f%% of the window); ",
                        g_audio_gaps - g0, g_audio_gap_blocks - gb0, n ? 100.0 * (g_audio_gap_blocks - gb0) / n : 0.0);
                g0 = g_audio_gaps; gb0 = g_audio_gap_blocks;
                fprintf(stderr, "%.1f blocks/s (real time 187.5), silent %.1f%%, dropped %.0f frames/s, device buf %u%c",
                        n * 1000.0 / (now - t0), n ? 100.0 * z / n : 0.0, (g_audio_dropped - d0) * 1000.0 / (now - t0),
                #if AUDIO_BACKEND_WASAPI
                        (unsigned)s_wasapi_buf_frames, 10);
#else
                        0u, 10);
#endif
                n = z = 0; d0 = g_audio_dropped; t0 = now; } } }

        /* Notify event queues */
        audio_notify_event_queues();

        /* AUDIO_UNDERRUN_LOG=1: once a second -- device underruns (the device
         * queue was empty when a block was due), mixer-late blocks (>2 ms
         * after the schedule) with the worst lateness, the lowest device queue
         * seen, and port blocks the guest had not written when read. */
        ur_blocks++;
        if (ur_log) {
            ULONGLONG t = GetTickCount64();
            if (t - ur_t0 >= 1000) {
                fprintf(stderr, "[audio-ur] %.1fs: underruns %u, late-mix blocks %u (max %.1f ms), "
                        "min device queue %u frames (cushion %u), guest-unwritten reads %u, blocks %u, "
                        "avg queue after submit %.1f ms + stream %.1f ms = ~%.1f ms behind the read index, trims %u, render-starved %ld, device-empty %ld%c",
                        (t - ur_t0) / 1000.0, ur_underruns, ur_late, ur_max_late_us / 1000.0,
                        ur_min_pad == (1u << 30) ? 0u : ur_min_pad, cushion, s_ur_zero_reads, ur_blocks,
                        ur_pad_n ? 1000.0 * ur_pad_sum / ur_pad_n / CELL_AUDIO_SAMPLE_RATE : 0.0,
                        AUDIO_STREAM_LAT_MS(),
                        (ur_pad_n ? 1000.0 * ur_pad_sum / ur_pad_n / CELL_AUDIO_SAMPLE_RATE : 0.0) + AUDIO_STREAM_LAT_MS(), ur_trims, (long)s_rt_starved, (long)s_rt_devunder, 10);
                ur_underruns = ur_late = ur_blocks = 0; ur_min_pad = 1u << 30; ur_max_late_us = 0; ur_pad_sum = ur_pad_n = 0; ur_trims = 0;
                s_ur_zero_reads = 0; ur_t0 = t;
            }
        }
    }

    printf("[cellAudio] Mixing thread stopped\n");
    return 0;
}
#else
static void* audio_mix_thread_func(void* arg)
{
    (void)arg;
    printf("[cellAudio] Mixing thread started\n");
#ifdef __APPLE__
    /* The mixer is what keeps the device fed: let the scheduler treat it so. */
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    /* Pace to the device, as the Windows loop does. This used to mix a block
     * and then sleep a flat 2 ms (5 ms once four blocks were queued): faster
     * than the 5.33 ms a block lasts, so the mixer ran ahead of real time and
     * read the guest's port ring before the guest had written the block --
     * stale or silent 5 ms slices in the output, heard as crackling -- while a
     * 2 ms sleep that overslept under load let the four-block queue run dry.
     * Now: mix one block whenever the device holds less than
     * AUDIO_QUEUE_BLOCKS (default 8, 43 ms) and otherwise wait 1 ms, so blocks
     * leave at the device's rate and the guest is notified at it too. With no
     * device, pace on the clock. AUDIO_RATE=1 reports blocks/s and underruns. */
    int target = 8;
    { const char* e = getenv("AUDIO_QUEUE_BLOCKS"); if (e && atoi(e) > 0) target = atoi(e); }
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    unsigned long long blocks = 0;
    int rate_on = getenv("AUDIO_RATE") ? 1 : 0;
    struct timespec r0 = t0; unsigned rn = 0, dry = 0;

    while (s_mix_thread_running) {
        if (s_sdl_audio_dev) {
            const u32 queued = audio_backend_queued_samples();
            if (queued >= (u32)target * CELL_AUDIO_BLOCK_SAMPLES) { usleep(1000); continue; }
            if (queued == 0 && blocks > (unsigned long long)target) dry++;
        } else {
            struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
            const double el = (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec) * 1e-9;
            const double due = (double)blocks * CELL_AUDIO_BLOCK_SAMPLES / CELL_AUDIO_SAMPLE_RATE;
            if (el < due) { usleep(1000); continue; }
            if (el - due > 0.05) t0 = now, blocks = 0;   /* far behind: drop the backlog */
        }
        blocks++;
        if (g_audio_hold == 2) {   /* paused: as the Windows loop */
            memset(s_mix_buffer, 0, sizeof s_mix_buffer);
            audio_backend_submit(s_mix_buffer, CELL_AUDIO_BLOCK_SAMPLES);
            continue;
        }
        audio_mix_one_block();
        if (g_audio_hold) memset(s_mix_buffer, 0, sizeof s_mix_buffer);
        /* AUDIO_WAV=<file>: the final mix as raw f32le stereo 48 kHz
         * (ffmpeg -f f32le -ar 48000 -ac 2 -i <file>). */
        { static FILE* wf = (FILE*)-1;
          if (wf == (FILE*)-1) { const char* e = getenv("AUDIO_WAV"); wf = e ? fopen(e, "wb") : NULL; }
          if (wf) fwrite(s_mix_buffer, sizeof(float), CELL_AUDIO_BLOCK_SAMPLES * 2, wf); }
        audio_backend_submit(s_mix_buffer, CELL_AUDIO_BLOCK_SAMPLES);
        audio_notify_event_queues();
        if (rate_on) {
            rn++;
            struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
            const double el = (now.tv_sec - r0.tv_sec) + (now.tv_nsec - r0.tv_nsec) * 1e-9;
            if (el >= 5.0) {
                fprintf(stderr, "[audio-rate] %.1f blocks/s (real time 187.5), queue ran dry %u times, queued %u samples\n",
                        rn / el, dry, audio_backend_queued_samples());
                rn = 0; dry = 0; r0 = now;
            }
        }
    }

    printf("[cellAudio] Mixing thread stopped\n");
    return NULL;
}
#endif

static int audio_start_mix_thread(void)
{
    s_mix_thread_running = 1;

#ifdef _WIN32
    s_mix_thread = (HANDLE)_beginthreadex(NULL, 0, audio_mix_thread_func,
                                           NULL, 0, NULL);
    if (!s_mix_thread) {
        s_mix_thread_running = 0;
        return -1;
    }
#else
    if (pthread_create(&s_mix_thread, NULL, audio_mix_thread_func, NULL) != 0) {
        s_mix_thread_running = 0;
        return -1;
    }
    s_mix_thread_live = 1;
#endif
    return 0;
}

static void audio_stop_mix_thread(void)
{
    s_mix_thread_running = 0;
#ifdef _WIN32
    if (s_mix_thread) {
        WaitForSingleObject(s_mix_thread, 2000);
        CloseHandle(s_mix_thread);
        s_mix_thread = NULL;
    }
#else
    /* Same guard. Init leaves s_audio_initialized set even when the mixing
     * thread failed to start ("continuing silently" -- a title without audio
     * still runs), so Quit reaches here with s_mix_thread never assigned.
     * pthread_t has no null value to test, and joining an unset one, or
     * joining the same one twice, is undefined rather than an error the way
     * WaitForSingleObject on a null HANDLE is. */
    if (s_mix_thread_live) {
        pthread_join(s_mix_thread, NULL);
        s_mix_thread_live = 0;
    }
#endif
}

/* ---------------------------------------------------------------------------
 * API implementations
 * -----------------------------------------------------------------------*/

s32 cellAudioInit(void)
{
    /* PS3_NO_AUDIO=1: report that the audio library is unavailable. Middleware
     * that drives its mixer through SPU jobs will otherwise block the whole boot
     * waiting on work we cannot yet complete; failing here lets a title take its
     * silent path and keep going. Diagnostic first, workaround second. */
    if (getenv("PS3_NO_AUDIO")) {
        printf("[cellAudio] PS3_NO_AUDIO -- reporting audio unavailable\n");
        return (s32)CELL_AUDIO_ERROR_NOT_INIT;
    }
    printf("[cellAudio] Init()\n");

    if (s_audio_initialized)
        return CELL_AUDIO_ERROR_ALREADY_INIT;

    memset(s_ports, 0, sizeof(s_ports));
    memset(s_notify_queues, 0, sizeof(s_notify_queues));
    mutex_init(&s_audio_mutex);

    if (audio_backend_init() < 0) {
        printf("[cellAudio] WARNING: Audio backend init failed, continuing silently\n");
        /* Don't fail -- games should still run without audio */
    }


#ifdef _WIN32
    { LARGE_INTEGER f, c;
      QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
      s_audio_start_us = (u64)(c.QuadPart / f.QuadPart) * 1000000ULL +
          (u64)(c.QuadPart % f.QuadPart) * 1000000ULL / (u64)f.QuadPart; }
#else
    { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
      s_audio_start_us = (u64)ts.tv_sec * 1000000ULL + (u64)ts.tv_nsec / 1000ULL; }
#endif
    if (audio_start_mix_thread() < 0) {
        printf("[cellAudio] WARNING: Could not start mixing thread\n");
    }

    s_audio_initialized = 1;
    return CELL_OK;
}

s32 cellAudioQuit(void)
{
    printf("[cellAudio] Quit()\n");

    if (!s_audio_initialized)
        return CELL_AUDIO_ERROR_NOT_INIT;

    audio_stop_mix_thread();
    audio_backend_shutdown();

    /* Port buffers live in the guest vm_base arena (bump-allocated, not host
     * malloc) -- just drop the references; never free() them. */
    for (int i = 0; i < CELL_AUDIO_PORT_MAX; i++) {
        s_ports[i].buffer = NULL;
        s_ports[i].in_use = 0;
        s_ports[i].running = 0;
    }

    mutex_destroy(&s_audio_mutex);
    s_audio_initialized = 0;
    return CELL_OK;
}

s32 cellAudioPortOpen(const CellAudioPortParam* param, u32* portNum)
{
    if (!s_audio_initialized)
        return CELL_AUDIO_ERROR_NOT_INIT;

    /* param / portNum are GUEST addresses; translate, and read the BE u64 param
     * fields via vm_read64 (a raw param->nChannel faults / is host-endian). */
    uint32_t param_ea   = (uint32_t)(uintptr_t)param;
    uint32_t portNum_ea = (uint32_t)(uintptr_t)portNum;
    if (!param_ea || !portNum_ea)
        return CELL_AUDIO_ERROR_PARAM;

    u64 nch  = vm_read64(param_ea + 0);   /* nChannel */
    u64 nblk = vm_read64(param_ea + 8);   /* nBlock   */
    printf("[cellAudio] PortOpen(nChannel=%llu, nBlock=%llu)\n",
           (unsigned long long)nch, (unsigned long long)nblk);

    if (nch != CELL_AUDIO_PORT_2CH && nch != CELL_AUDIO_PORT_8CH)
        return CELL_AUDIO_ERROR_PARAM;
    if (nblk != CELL_AUDIO_BLOCK_8 && nblk != CELL_AUDIO_BLOCK_16 && nblk != CELL_AUDIO_BLOCK_32)
        return CELL_AUDIO_ERROR_PARAM;

    mutex_lock(&s_audio_mutex);

    /* Find a free port slot */
    s32 found = -1;
    for (u32 i = 0; i < CELL_AUDIO_PORT_MAX; i++) {
        if (!s_ports[i].in_use) {
            found = (s32)i;
            break;
        }
    }

    if (found < 0) {
        mutex_unlock(&s_audio_mutex);
        return CELL_AUDIO_ERROR_PORT_FULL;
    }

    AudioPortSlot* port = &s_ports[found];
    port->in_use  = 1;
    port->running = 0;
    /* Store the decoded (host-endian) param values, not the raw BE guest bytes. */
    memset(&port->param, 0, sizeof(port->param));
    port->param.nChannel = nch;
    port->param.nBlock   = nblk;
    port->read_index  = 0;
    port->write_index = 0;
    port->level       = 1.0f;

    /* Allocate the audio buffer in GUEST memory so portAddr/readIndexAddr are
     * real guest addresses (the game's mixer -- FMOD here -- writes PCM there
     * and validates them as 1 MB-aligned guest pointers; a host pointer trips
     * its "invalid parameter" check). Bump from a free main-memory window; the
     * host-side `buffer` is just the vm_base view of the same bytes. */
    u32 buf_samples = (u32)(nblk * CELL_AUDIO_BLOCK_SAMPLES * nch);
    port->buf_size = buf_samples * (u32)sizeof(float);

    /* Keep audio outside sys_memory (0x40000000..0x50000000), the HLE heap
     * (0x50000000..0x58000000), and sys_vm (0x60000000..0x70000000).
     * Reserve two 1-MB pages per port so reopening reuses its own storage. */
    u32 guest_buf = 0x58000000u + (u32)found * 0x200000u;
    u32 guest_ridx = guest_buf + 0x100000u;
    if (vm_commit(guest_buf, 0x200000u) != CELL_OK) {
        port->in_use = 0;
        mutex_unlock(&s_audio_mutex);
        return CELL_AUDIO_ERROR_AUDIOSYSTEM;
    }

    port->buffer = (float*)(vm_base + guest_buf);     /* host view of guest buffer */
    memset(port->buffer, 0, port->buf_size);
    vm_write64(guest_ridx, 0);                        /* read index counter (u64, see below) */

    port->port_addr     = guest_buf;
    port->read_idx_addr = guest_ridx;

    vm_write32(portNum_ea, (u32)found);   /* guest out-param */

    mutex_unlock(&s_audio_mutex);
    return CELL_OK;
}

s32 cellAudioPortClose(u32 portNum)
{
    printf("[cellAudio] PortClose(port=%u)\n", portNum);

    if (!s_audio_initialized)
        return CELL_AUDIO_ERROR_NOT_INIT;

    if (portNum >= CELL_AUDIO_PORT_MAX)
        return CELL_AUDIO_ERROR_PORT_NOT_OPEN;

    mutex_lock(&s_audio_mutex);

    if (!s_ports[portNum].in_use) {
        mutex_unlock(&s_audio_mutex);
        return CELL_AUDIO_ERROR_PORT_NOT_OPEN;
    }

    /* buffer points INTO the guest vm_base arena (reserved per port in PortOpen),
     * not a host malloc -- do NOT free() it (that corrupts the host heap).
     * The guest window is reclaimed wholesale when vm_base is released. */
    s_ports[portNum].buffer = NULL;

    s_ports[portNum].in_use  = 0;
    s_ports[portNum].running = 0;

    mutex_unlock(&s_audio_mutex);
    return CELL_OK;
}

s32 cellAudioPortStart(u32 portNum)
{
    printf("[cellAudio] PortStart(port=%u)\n", portNum);

    if (!s_audio_initialized)
        return CELL_AUDIO_ERROR_NOT_INIT;

    if (portNum >= CELL_AUDIO_PORT_MAX || !s_ports[portNum].in_use)
        return CELL_AUDIO_ERROR_PORT_NOT_OPEN;

    if (s_ports[portNum].running)
        return CELL_AUDIO_ERROR_PORT_ALREADY_RUN;

    mutex_lock(&s_audio_mutex);
    s_ports[portNum].running = 1;
    mutex_unlock(&s_audio_mutex);

    return CELL_OK;
}

/* Per-port gain. The guest passes a float in f1; the mixer applies it. */
s32 cellAudioSetPortLevel(u32 portNum, float level)
{
    if (!s_audio_initialized)
        return CELL_AUDIO_ERROR_NOT_INIT;

    if (portNum >= CELL_AUDIO_PORT_MAX || !s_ports[portNum].in_use)
        return CELL_AUDIO_ERROR_PORT_NOT_OPEN;

    if (!(level >= 0.0f)) level = 0.0f;   /* also catches NaN */
    if (level > 1.0f) level = 1.0f;

    mutex_lock(&s_audio_mutex);
    s_ports[portNum].level = level;
    mutex_unlock(&s_audio_mutex);

    return CELL_OK;
}

s32 cellAudioPortStop(u32 portNum)
{
    printf("[cellAudio] PortStop(port=%u)\n", portNum);

    if (!s_audio_initialized)
        return CELL_AUDIO_ERROR_NOT_INIT;

    if (portNum >= CELL_AUDIO_PORT_MAX || !s_ports[portNum].in_use)
        return CELL_AUDIO_ERROR_PORT_NOT_OPEN;

    if (!s_ports[portNum].running)
        return CELL_AUDIO_ERROR_PORT_NOT_RUN;

    mutex_lock(&s_audio_mutex);
    s_ports[portNum].running = 0;
    mutex_unlock(&s_audio_mutex);

    return CELL_OK;
}

s32 cellAudioGetPortBlockTag(u32 portNum, u64 blockNo, u64* tag)
{
    uint32_t tag_ea = (uint32_t)(uintptr_t)tag;
    if (!s_audio_initialized)
        return CELL_AUDIO_ERROR_NOT_INIT;
    if (portNum >= CELL_AUDIO_PORT_MAX)
        return CELL_AUDIO_ERROR_PARAM;
    if (!s_ports[portNum].in_use)
        return CELL_AUDIO_ERROR_PORT_NOT_OPEN;
    if (!tag_ea)
        return CELL_AUDIO_ERROR_PARAM;

    mutex_lock(&s_audio_mutex);
    AudioPortSlot* port = &s_ports[portNum];
    u64 nblk = port->param.nBlock ? port->param.nBlock : 1;
    if (blockNo >= nblk) {
        mutex_unlock(&s_audio_mutex);
        return CELL_AUDIO_ERROR_PARAM;
    }
    u64 t = port->read_index + blockNo - (port->read_index % nblk);
    if (blockNo < port->read_index % nblk) t += nblk;
    mutex_unlock(&s_audio_mutex);

    vm_write64(tag_ea, t);
    return CELL_OK;
}

s32 cellAudioGetPortTimestamp(u32 portNum, u64 tag, u64* stamp)
{
    uint32_t stamp_ea = (uint32_t)(uintptr_t)stamp;
    if (!s_audio_initialized)
        return CELL_AUDIO_ERROR_NOT_INIT;
    if (portNum >= CELL_AUDIO_PORT_MAX)
        return CELL_AUDIO_ERROR_PARAM;
    if (!s_ports[portNum].in_use)
        return CELL_AUDIO_ERROR_PORT_NOT_OPEN;
    if (!stamp_ea)
        return CELL_AUDIO_ERROR_PARAM;

    mutex_lock(&s_audio_mutex);
    u64 global_counter = s_ports[portNum].read_index;
    mutex_unlock(&s_audio_mutex);
    if (tag > global_counter)
        return CELL_AUDIO_ERROR_TAG_NOT_FOUND;

    u64 t = s_audio_start_us + tag * 256000000ULL / 48000ULL;
    vm_write64(stamp_ea, t);
    return CELL_OK;
}

/* cellAudioCreateNotifyEventQueue(sys_event_queue_t* id, u64* key)
 *
 * Creates the queue the game will block on for audio-period events, and
 * registers it for notification in one step -- the game never calls
 * SetNotifyEventQueue for this one.
 *
 * This was missing entirely. YDKJ imports it, got the unimplemented-NID path,
 * and then blocked on an id nothing had created: sys_event_queue_receive
 * answered CELL_ESRCH every time and the title logged 1.7 MILLION
 * "ERROR - LIBAUDIO DROPOUT" lines in a 60 s run.
 *
 * Both params are GUEST addresses and both are big-endian out-values; the id
 * is 32-bit and the key 64-bit, so they cannot be written with a plain store. */
s32 cellAudioCreateNotifyEventQueue(u32* id, u64* key)
{
    uint32_t id_ea  = (uint32_t)(uintptr_t)id;
    uint32_t key_ea = (uint32_t)(uintptr_t)key;

    printf("[cellAudio] CreateNotifyEventQueue(id_ea=0x%08X key_ea=0x%08X)\n",
           id_ea, key_ea);

    if (!s_audio_initialized)
        return CELL_AUDIO_ERROR_NOT_INIT;
    if (!id_ea || !key_ea)
        return CELL_AUDIO_ERROR_PARAM;

    /* A distinct key per queue: the notify path resolves queues BY KEY, and the
     * game creates its other queues with key 0, so reusing 0 here would make
     * find_queue_by_key ambiguous and post audio events to whichever matched. */
    static uint64_t s_next_key = 0x8000000000000001ull;
    uint64_t k = s_next_key++;

    uint32_t qid = sys_event_queue_create_direct(k, 8);
    if (!qid)
        return CELL_AUDIO_ERROR_EVENT_QUEUE;

    mutex_lock(&s_audio_mutex);
    int slot = -1;
    for (int i = 0; i < CELL_AUDIO_MAX_NOTIFY_EVENT_QUEUES; i++) {
        if (!s_notify_queues[i].in_use) { slot = i; break; }
    }
    if (slot < 0) { mutex_unlock(&s_audio_mutex); return CELL_AUDIO_ERROR_PARAM; }
    s_notify_queues[slot].in_use = 1;
    s_notify_queues[slot].key    = k;
    mutex_unlock(&s_audio_mutex);

    vm_write32(id_ea, qid);
    vm_write64(key_ea, k);
    printf("[cellAudio] CreateNotifyEventQueue -> id=%u key=0x%llX\n",
           qid, (unsigned long long)k);
    return CELL_OK;
}

s32 cellAudioSetNotifyEventQueue(u64 key)
{
    printf("[cellAudio] SetNotifyEventQueue(key=0x%llX)\n",
           (unsigned long long)key);

    if (!s_audio_initialized)
        return CELL_AUDIO_ERROR_NOT_INIT;

    mutex_lock(&s_audio_mutex);

    for (int i = 0; i < CELL_AUDIO_MAX_NOTIFY_EVENT_QUEUES; i++) {
        if (!s_notify_queues[i].in_use) {
            s_notify_queues[i].in_use = 1;
            s_notify_queues[i].key = key;
            mutex_unlock(&s_audio_mutex);
            return CELL_OK;
        }
    }

    mutex_unlock(&s_audio_mutex);
    return CELL_AUDIO_ERROR_PARAM; /* no free slots */
}

s32 cellAudioRemoveNotifyEventQueue(u64 key)
{
    printf("[cellAudio] RemoveNotifyEventQueue(key=0x%llX)\n",
           (unsigned long long)key);

    if (!s_audio_initialized)
        return CELL_AUDIO_ERROR_NOT_INIT;

    mutex_lock(&s_audio_mutex);

    for (int i = 0; i < CELL_AUDIO_MAX_NOTIFY_EVENT_QUEUES; i++) {
        if (s_notify_queues[i].in_use && s_notify_queues[i].key == key) {
            s_notify_queues[i].in_use = 0;
            mutex_unlock(&s_audio_mutex);
            return CELL_OK;
        }
    }

    mutex_unlock(&s_audio_mutex);
    return CELL_AUDIO_ERROR_PARAM;
}

s32 cellAudioGetPortConfig(u32 portNum, CellAudioPortConfig* config)
{
    if (!s_audio_initialized)
        return CELL_AUDIO_ERROR_NOT_INIT;

    if (portNum >= CELL_AUDIO_PORT_MAX || !s_ports[portNum].in_use)
        return CELL_AUDIO_ERROR_PORT_NOT_OPEN;

    if (!config)
        return CELL_AUDIO_ERROR_PARAM;

    /* `config` is a GUEST address; write the BE struct field-by-field at its
     * guest offsets (8-byte aligned u64s). */
    uint32_t cfg = (uint32_t)(uintptr_t)config;
    if (!cfg)
        return CELL_AUDIO_ERROR_PARAM;

    mutex_lock(&s_audio_mutex);
    AudioPortSlot* port = &s_ports[portNum];

    /* CellAudioPortConfig (cell/audio.h). sys_addr_t is uintptr_t, which on the
     * 32-bit PPU ABI is FOUR bytes -- not eight. Writing the two sys_addr_t
     * fields as u64 shifted every field past it and left the game reading the
     * high (zero) half of readIndexAddr as its pointer:
     *   sys_addr_t readIndexAddr;  u32 @  0
     *   uint32_t   status;         u32 @  4
     *   uint64_t   nChannel;       u64 @  8
     *   uint64_t   nBlock;         u64 @ 16
     *   uint32_t   portSize;       u32 @ 24
     *   sys_addr_t portAddr;       u32 @ 28      -> 32 bytes */
    vm_write32(cfg +  0, (u32)port->read_idx_addr);                          /* readIndexAddr */
    vm_write32(cfg +  4, port->running ? CELL_AUDIO_STATUS_RUN
                                       : CELL_AUDIO_STATUS_READY);           /* status */
    vm_write64(cfg +  8, port->param.nChannel);                             /* nChannel */
    vm_write64(cfg + 16, port->param.nBlock);                               /* nBlock */
    vm_write32(cfg + 24, port->buf_size);                                   /* portSize */
    vm_write32(cfg + 28, (u32)port->port_addr);                             /* portAddr */

    { static int _n = 0; if (_n < 24 && ++_n)
        fprintf(stderr, "[cellAudio] GetPortConfig(port=%u) status=%s bufEA=0x%08X ridxEA=0x%08X\n",
                portNum, port->running ? "RUN" : "READY",
                (u32)port->port_addr, (u32)port->read_idx_addr); }

    mutex_unlock(&s_audio_mutex);
    return CELL_OK;
}

s32 cellAudioPortGetStatus(u32 portNum, u32* status)
{
    if (!s_audio_initialized)
        return CELL_AUDIO_ERROR_NOT_INIT;

    if (portNum >= CELL_AUDIO_PORT_MAX || !status)
        return CELL_AUDIO_ERROR_PARAM;

    if (!s_ports[portNum].in_use) {
        vm_write32((u32)(uintptr_t)status, CELL_AUDIO_STATUS_CLOSE);
    } else if (s_ports[portNum].running) {
        vm_write32((u32)(uintptr_t)status, CELL_AUDIO_STATUS_RUN);
    } else {
        vm_write32((u32)(uintptr_t)status, CELL_AUDIO_STATUS_READY);
    }

    return CELL_OK;
}

s32 cellAudioSetPersonalDevice(s32 iPersonalStream, s32 iDevice)
{
    (void)iPersonalStream;
    (void)iDevice;
    printf("[cellAudio] SetPersonalDevice(stream=%d, device=%d) - stub\n",
           iPersonalStream, iDevice);
    return CELL_OK;
}

s32 cellAudioUnsetPersonalDevice(s32 iPersonalStream)
{
    (void)iPersonalStream;
    printf("[cellAudio] UnsetPersonalDevice(stream=%d) - stub\n", iPersonalStream);
    return CELL_OK;
}
