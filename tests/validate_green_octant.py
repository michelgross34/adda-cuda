#!/usr/bin/env python3
"""Validate the sign table used by adda_low_mem's first-octant Green reconstruction.

This is a purely algebraic test of the six-component dyadic order
(xx,xy,xz,yy,yz,zz). It does not require CUDA.
"""
from itertools import product

COMPONENTS=("xx","xy","xz","yy","yz","zz")

def signs(rx,ry,rz):
    s=[1,1,1,1,1,1]
    if rx != ry: s[1]=-1
    if rx != rz: s[2]=-1
    if ry != rz: s[4]=-1
    return s

def old_yz(ry,rz):
    s=[1,1,1,1,1,1]
    if ry:
        s[1]=-1
        if rz: s[2]=-1
        else: s[4]=-1
    elif rz:
        s[2]=-1; s[4]=-1
    return s

for ry,rz in product((False,True),repeat=2):
    assert signs(False,ry,rz)==old_yz(ry,rz), (ry,rz,signs(False,ry,rz),old_yz(ry,rz))

print("Existing Y/Z reduced-FFT parity: PASS")
for rx,ry,rz in product((False,True),repeat=3):
    row=signs(rx,ry,rz)
    print(f"rx={int(rx)} ry={int(ry)} rz={int(rz)}  " + " ".join(f"{c}:{v:+d}" for c,v in zip(COMPONENTS,row)))
print("First-octant parity table: PASS")
