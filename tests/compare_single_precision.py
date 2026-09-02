#!/usr/bin/env python3
"""Compare the RE_xxx convergence lines of two ADDA stdout captures."""
import re, sys, math
if len(sys.argv) != 3:
    print("usage: compare_single_precision.py double.out single.out")
    raise SystemExit(2)
pat=re.compile(r"RE_(\d+)\s*=\s*([0-9Ee+\-.]+)")
def read(p):
    out=[]
    for line in open(p, errors='replace'):
        m=pat.search(line)
        if m: out.append((int(m.group(1)),float(m.group(2))))
    return out
a,b=read(sys.argv[1]),read(sys.argv[2])
print(f"double residuals: {len(a)}; single residuals: {len(b)}")
for (ia,va),(ib,vb) in zip(a,b):
    if ia!=ib: print(f"iteration mismatch {ia} vs {ib}"); break
    rel=abs(va-vb)/max(abs(va),abs(vb),1e-30)
    print(f"RE_{ia:03d}: double={va:.12e} single={vb:.12e} rel_diff={rel:.3e}")
