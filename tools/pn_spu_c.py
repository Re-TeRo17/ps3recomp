#!/usr/bin/env python3
"""pn_spu_c.py -- give the promoted twins (NAME_p) of a lifted SPU image the preserve_none convention.

Post-pass over spu/<image>/spu_recomp.c (after promote_spu_c.py and tail_spu_c.py). See spu_context.h SPU_PN:
  - prototypes and definitions of every promoted twin get SPU_PN;
  - SPU_TAIL / SPU_TAIL_IB inside the promoted region become SPU_TAILP / SPU_TAIL_IBP (a real jmp, or the wrapper);
  - every use of a twin as a VALUE (function table, spu_pcheck_enter, g_spu_trampoline_fn) names the wrapper NAME_p_w,
    a normal-convention function defined right after the prototypes.
Idempotent (a file that already has SPU_PN is left alone).  Usage: python -I pn_spu_c.py <spu_dir> [--only SUBSTR]
"""
import argparse, glob, os, re

NAME = r'[A-Za-z0-9_]+_spu_func_[0-9A-Fa-f]{8}_p'
RX_PROTO = re.compile(r'^void (%s)\(spu_context\* ctx\);$' % NAME, re.M)
RX_DEF = re.compile(r'^void (%s)\(spu_context\* ctx\) \{$' % NAME, re.M)


def convert(src):
    if 'SPU_PN void' in src:
        return None, 'already'
    protos = RX_PROTO.findall(src)
    if not protos:
        return None, 'no promoted twins'
    first = RX_PROTO.search(src).start()
    head, tail = src[:first], src[first:]
    # wrappers right after the last prototype
    lastp = list(RX_PROTO.finditer(tail))[-1]
    names = [m.group(1) for m in RX_PROTO.finditer(tail)]
    tail = RX_PROTO.sub(lambda m: 'SPU_PN void %s(spu_context* ctx);' % m.group(1), tail)
    tail = RX_DEF.sub(lambda m: 'SPU_PN void %s(spu_context* ctx) {' % m.group(1), tail)
    tail = re.sub(r'\bSPU_TAIL\(ctx, ', 'SPU_TAILP(ctx, ', tail)
    tail = re.sub(r'\bSPU_TAIL_IB\(', 'SPU_TAIL_IBP(', tail)
    # values: pcheck second argument, trampoline assignments, table entries
    tail = re.sub(r'(spu_pcheck_enter\(ctx, [A-Za-z0-9_]+, )(%s)\)' % NAME, lambda m: m.group(1) + m.group(2) + '_w)', tail)
    tail = re.sub(r'(g_spu_trampoline_fn = )(%s);' % NAME, lambda m: m.group(1) + m.group(2) + '_w;', tail)
    tail = re.sub(r'(\{ 0x[0-9A-Fa-f]+u, )(%s)(, ")' % NAME, lambda m: m.group(1) + m.group(2) + '_w' + m.group(3), tail)
    end = list(re.finditer(r'^SPU_PN void %s\(spu_context\* ctx\);$' % NAME, tail, re.M))[-1].end()
    wr = '\n/* normal-convention entry points of the preserve_none twins (registry / trampoline / pcheck) */\n'
    wr += ''.join('static __attribute__((noinline)) void %s_w(spu_context* ctx) { %s(ctx); }\n' % (n, n) for n in names)
    tail = tail[:end] + '\n' + wr + tail[end:]
    return head + tail, '%d twins' % len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spu_dir')
    ap.add_argument('--only', default='')
    ap.add_argument('--all', action='store_true', help='also the policy module and the raw-SPU images (untested)')
    a = ap.parse_args()
    for f in sorted(glob.glob(os.path.join(a.spu_dir, '*', 'spu_recomp.c'))):
        if a.only and a.only not in f:
            continue
        if not a.all and os.path.basename(os.path.dirname(f)) in ('spurs_pm_wwsjob', 'igCollBroadSpu', 'igcollspu'):
            continue
        s = open(f, newline='').read()
        out, why = convert(s)
        print(os.path.basename(os.path.dirname(f)), why)
        if out is not None:
            open(f, 'w', newline='').write(out)


if __name__ == '__main__':
    main()
