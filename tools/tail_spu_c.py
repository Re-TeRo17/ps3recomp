#!/usr/bin/env python3
"""tail_spu_c.py -- turn the lifted functions' static "transfer" exits into guarded tail calls.

Post-pass over spu/<image>/spu_recomp.c (after promote_spu_c.py, but it works on any lifted text):

    { ctx->pc = 0x4B80; g_spu_trampoline_fn = FN; return; }       -> { SPU_TAIL(ctx, 0x4B80, FN); }
    ctx->pc = ctx->gpr[N]._u32[0]; g_spu_trampoline_fn = spu_indirect_branch; return;
                                                                  -> SPU_TAIL_IB(ctx, ctx->gpr[N]._u32[0]);

SPU_TAIL / SPU_TAIL_IB (runtime/spu/spu_context.h) jump straight into the target when no hop hook
is armed (no diagnostic, no pending interrupt, no lockstep / stack reset); otherwise they keep the
old behaviour (set the trampoline and return to the nearest drain loop). SPU_TAIL=1 turns the jump on at run time
(default off: measured +8% CPU per wid7 run -- Win64 xmm6-15 are callee-saved, so a tail call out of a
promoted function pays the full epilogue and gains nothing).  Idempotent.  Usage: python -I tail_spu_c.py <spu_dir> [--only SUBSTR] [--all]
"""
import argparse, os, re, glob

RX_A = re.compile(r'\{ ctx->pc = (0x[0-9A-Fa-f]+u?); g_spu_trampoline_fn = ([A-Za-z0-9_]+); return; \}')
RX_A2 = re.compile(r'ctx->pc = (0x[0-9A-Fa-f]+u?); g_spu_trampoline_fn = ([A-Za-z0-9_]+); return;')
RX_B = re.compile(r'ctx->pc = (ctx->gpr\[\d+\]\._u32\[0\]); g_spu_trampoline_fn = spu_indirect_branch; return;')


def tail_text(src):
    n = [0, 0]

    def ra(m):
        if int(m.group(1).rstrip('uU'), 16) == 0:
            return m.group(0)
        n[0] += 1
        return '{ SPU_TAIL(ctx, %s, %s); }' % (m.group(1), m.group(2))

    def ra2(m):
        if int(m.group(1).rstrip('uU'), 16) == 0:
            return m.group(0)
        n[0] += 1
        return 'SPU_TAIL(ctx, %s, %s);' % (m.group(1), m.group(2))

    def rb(m):
        n[1] += 1
        return 'SPU_TAIL_IB(ctx, %s);' % m.group(1)

    out = RX_A.sub(ra, src)
    out = RX_A2.sub(ra2, out)
    out = RX_B.sub(rb, out)
    if n[0] + n[1] == 0:
        return None, 'nothing'
    return out, f'{n[0]} direct + {n[1]} indirect exits'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spu_dir')
    ap.add_argument('--only', default='')
    ap.add_argument('--all', action='store_true', help='also the policy module and the raw-SPU images (untested)')
    a = ap.parse_args()
    tot = 0
    for f in sorted(glob.glob(os.path.join(a.spu_dir, '*', 'spu_recomp.c'))):
        if a.only and a.only not in f:
            continue
        if not a.all and os.path.basename(os.path.dirname(f)) in ('spurs_pm_wwsjob', 'igCollBroadSpu', 'igcollspu'):
            continue
        src = open(f, encoding='utf-8', newline='').read()
        new, info = tail_text(src)
        if new is None:
            continue
        open(f, 'w', encoding='utf-8', newline='').write(new)
        tot += 1
        print(f'  {os.path.basename(os.path.dirname(f))}: {info}')
    print(f'tail-called {tot} image(s)')


if __name__ == '__main__':
    main()
