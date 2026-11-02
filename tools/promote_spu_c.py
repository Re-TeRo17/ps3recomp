#!/usr/bin/env python3
"""promote_spu_c.py -- add register-promoted twins to already lifted SPU images.

Post-pass over spu/<image>/spu_recomp.c (the output of spu_lifter.py): for every lifted function
`<prefix>spu_func_XXXXXXXX` it appends a twin `<prefix>spu_func_XXXXXXXX_p` in which the SPU
registers are C locals (see promote_body() in spu_lifter.py), a second function table, and makes
`<prefix>spu_recomp_register` pick the promoted table when spu_promote_enabled() (env SPU_PROMOTE=1).
The plain functions stay untouched, so the same binary runs both and can compare them
(SPU_PROMOTE_CHECK, runtime/spu/spu_drain.c).  Idempotent: files that already carry the _p table
are skipped.  It works on the C text, so it applies to every image whatever tool lifted it
(jobmods, dynovl, policy module) with no re-lift drift.

Usage:  python -I promote_spu_c.py <spu_dir> [--only SUBSTR] [--jobs N]
"""
import argparse, os, re, sys, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spu_lifter import promote_body  # noqa: E402

FN_RE = re.compile(r'^void (__attribute__\(\(weak\)\) )?([A-Za-z0-9_]*spu_func_([0-9A-F]{8}))\(spu_context\* ctx\) \{\r?$')
TAB_RE = re.compile(r'^\s*\{ 0x([0-9A-F]{8})u, ([A-Za-z0-9_]+), "[^"]*" \},\r?$')
REG_RE = re.compile(r'^void ([A-Za-z0-9_]*)spu_recomp_register\(void\) \{\r?$')


def promote_text(src):
    nl = '\r\n' if '\r\n' in src else '\n'
    lines = src.split('\n')
    lines = [l.rstrip('\r') for l in lines]
    if any('spu_function_table_p' in l for l in lines):
        return None, 'already'
    funcs = []        # (name, addr, weak, body_lines)
    i = 0
    while i < len(lines):
        m = FN_RE.match(lines[i])
        if m:
            j = i + 1
            while lines[j] != '}':
                j += 1
            funcs.append((m.group(2), int(m.group(3), 16), bool(m.group(1)), lines[i + 1:j]))
            i = j
        i += 1
    if not funcs:
        return None, 'nofuncs'
    # prefix = name without spu_func_XXXXXXXX
    prefix = funcs[0][0][:funcs[0][0].rindex('spu_func_')]
    pre = re.escape(prefix)
    rx = re.compile(r'(' + pre + r'spu_func_[0-9A-Fa-f]{8})\b')
    ren = lambda t: rx.sub(r'\1_p', t)

    # every function name referenced anywhere (calls / trampolines) needs a _p symbol
    referenced = set()
    for _, _, _, body in funcs:
        for l in body:
            referenced.update(rx.findall(l))
    defined_names = {f[0] for f in funcs}
    table = []
    for l in lines:
        m = TAB_RE.match(l)
        if m:
            table.append((int(m.group(1), 16), m.group(2)))
    tnames = {n for _, n in table}

    pfuncs = []       # (name_p, weak, body)
    promoted = set()
    for name, addr, weak, body in funcs:
        if not any('ctx->gpr[' in l for l in body):
            continue
        # a stub / pure-trampoline function: nothing to promote
        pb = promote_body(body, ren, name)
        if pb is None:
            continue
        pfuncs.append((name + '_p', weak, pb))
        promoted.add(name)

    out = []
    block = []
    block.append('/* ---- register-promoted twins (promote_spu_c.py): locals instead of ctx->gpr[] ---- */')
    for n in sorted(referenced | defined_names):
        block.append(f'void {n}_p(spu_context* ctx);')
    for name, weak, body in pfuncs:
        block.append(f'void {"__attribute__((weak)) " if weak else ""}{name}(spu_context* ctx) {{')
        block.extend(body)
        block.append('}')
        block.append('')
    for n in sorted((referenced | defined_names) - promoted):
        block.append(f'void {n}_p(spu_context* ctx) {{ {n}(ctx); }}')
    block.append('')

    tab = ['static const spu_func_entry spu_function_table_p[] = {']
    for addr, n in table:
        nn = n + '_p'
        tab.append(f'    {{ 0x{addr:08X}u, {nn}, "{nn}" }},')
    tab.append('    { 0, NULL, NULL }')
    tab.append('};')
    tab.append('int spu_promote_enabled(void);')
    tab.append('')

    state = 0
    k = 0
    while k < len(lines):
        l = lines[k]
        if state == 0 and l.startswith('/* Function table */'):
            out.extend(block)
            out.append(l)
            state = 1
        elif state == 1 and l.startswith('void ') and 'spu_recomp_register(void) {' in l:
            out.extend(tab)
            out.append(l)
            # original body: for-loop over spu_function_table
            out.append('    const spu_func_entry* tab = spu_promote_enabled() ? spu_function_table_p : spu_function_table;')
            k += 1
            while lines[k] != '}':
                out.append(lines[k].replace('e = spu_function_table;', 'e = tab;'))
                k += 1
            out.append('}')
            state = 2
        else:
            out.append(l)
        k += 1
    if state != 2:
        return None, 'noreg'
    return nl.join(out), f'{len(pfuncs)}/{len(funcs)} promoted'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spu_dir')
    ap.add_argument('--only', default='')
    ap.add_argument('--all', action='store_true',
                    help='also promote the policy module and the raw-SPU collision images (untested, off by default)')
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.spu_dir, '*', 'spu_recomp.c')))
    tot = 0
    for f in files:
        if a.only and a.only not in f:
            continue
        if not a.all and os.path.basename(os.path.dirname(f)) in ('spurs_pm_wwsjob', 'igCollBroadSpu', 'igcollspu'):
            continue
        src = open(f, encoding='utf-8', newline='').read()
        new, info = promote_text(src)
        if new is None:
            print(f'  skip {os.path.basename(os.path.dirname(f))}: {info}')
            continue
        open(f, 'w', encoding='utf-8', newline='').write(new)
        tot += 1
        print(f'  {os.path.basename(os.path.dirname(f))}: {info}')
    print(f'promoted {tot} image(s)')


if __name__ == '__main__':
    main()
