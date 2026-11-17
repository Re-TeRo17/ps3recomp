/*
 * ps3recomp - CPU-specific primitives under one name on x86-64 and arm64
 *
 *   ps3_cpu_ticks()   a fast, monotonic, constant-rate counter (x86: TSC; arm64: CNTVCT_EL0).
 *                     Callers calibrate it against QueryPerformanceCounter or use it for
 *                     relative costs, so its frequency differs per host and is never assumed.
 *   ps3_cpu_pause()   spin-wait hint (x86: pause; arm64: yield).
 *   ps3_fpcsr_get/set the float control register (x86: MXCSR; arm64: FPCR) and the bits for
 *                     round-toward-zero and flush-to-zero in each, as RACiT's spu_channels.c does.
 *
 * Off x86 it also supplies the spellings the x86 sources use (__rdtsc, _mm_pause), so a file whose
 * only x86 dependency is those can include this header instead of <x86intrin.h>/<intrin.h>.
 * On x86 (MSVC, clang-cl, clang, GCC) it includes the usual intrinsic header and changes nothing.
 */
#ifndef PS3RECOMP_PS3_CPU_H
#define PS3RECOMP_PS3_CPU_H

#include <stdint.h>

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#  define PS3_CPU_X86 1
#  if defined(_MSC_VER)
#    include <intrin.h>
#  else
#    include <x86intrin.h>
#  endif
static inline uint64_t ps3_cpu_ticks(void) { return (uint64_t)__rdtsc(); }
static inline void     ps3_cpu_pause(void) { _mm_pause(); }
static inline uint64_t ps3_fpcsr_get(void) { return (uint64_t)_mm_getcsr(); }
static inline void     ps3_fpcsr_set(uint64_t v) { _mm_setcsr((unsigned)v); }
#  define PS3_FPCSR_RZ   0x6000ull     /* MXCSR RC = 11: round toward zero */
#  define PS3_FPCSR_FTZ  0x8040ull     /* MXCSR FTZ | DAZ */
#  define PS3_FPCSR_MASK 0xE040ull

#elif defined(__aarch64__) || defined(_M_ARM64)
#  define PS3_CPU_ARM64 1
static inline uint64_t ps3_cpu_ticks(void) { uint64_t v; __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(v)); return v; }
static inline void     ps3_cpu_pause(void) { __asm__ __volatile__("yield" ::: "memory"); }
static inline uint64_t ps3_fpcsr_get(void) { uint64_t v; __asm__ __volatile__("mrs %0, fpcr" : "=r"(v)); return v; }
static inline void     ps3_fpcsr_set(uint64_t v) { __asm__ __volatile__("msr fpcr, %0" :: "r"(v)); }
#  define PS3_FPCSR_RZ   (3ull << 22)  /* FPCR RMode = 11: round toward zero */
#  define PS3_FPCSR_FTZ  (1ull << 24)  /* FPCR FZ (inputs and outputs) */
#  define PS3_FPCSR_MASK ((3ull << 22) | (1ull << 24))
#  ifndef __rdtsc
#    define __rdtsc() ps3_cpu_ticks()
#  endif
#  ifndef _mm_pause
#    define _mm_pause() ps3_cpu_pause()
#  endif
#  ifndef _mm_prefetch
#    define _MM_HINT_T0 3
#    define _MM_HINT_T1 2
#    define _MM_HINT_T2 1
#    define _MM_HINT_NTA 0
#    define _mm_prefetch(p, hint) __builtin_prefetch((const void*)(p), 0, (hint))
#  endif

#else
#  error "ps3_cpu.h: unsupported CPU"
#endif

#endif /* PS3RECOMP_PS3_CPU_H */
