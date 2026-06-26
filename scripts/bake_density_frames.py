#!/usr/bin/env python3
"""Bake the full-resolution master simulation into per-frame GRAYSCALE DENSITY
maps for the Opportunities-page cosmic-web animation.

Key idea: the browser bins particles into a per-pixel accumulation grid anyway, so
we precompute that binning OFFLINE using ALL ~2.1M particles. The shipped file size
then depends only on grid resolution x frames -- NOT particle count -- so we get the
full-resolution web. The component tone-maps (violet tint + curve + brightness) live.

Each frame: project all particles to a GRID x GRID grid with the same soft splat as
the live renderer, store the raw accumulated density quantized to uint8 (with a
global scale so values use the 0..255 range). Frames are warp-distributed (dense at
late cosmic times) to match the scroll.

  python3 scripts/bake_density_frames.py [GRID] [N_FRAMES]   (default 512 118)
Reads  offline/cosmic-web-master.bin  ->  public/sim/cosmic-web.bin
Format: 'CWDN' | u16 ver=1 | u16 GRID | u16 M | u8 scale_hint(unused) | M*GRID*GRID u8
"""
import numpy as np, struct, os, sys, gzip

GRID    = int(sys.argv[1]) if len(sys.argv) > 1 else 512
NFRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 118
PA      = 0.5            # per-particle alpha weight (matches the live look)
EXPO    = 3.2           # frame warp: dense at late times

master = os.environ.get("COSMIC_OUT", "offline/cosmic-web-master.bin")
raw = open(master, "rb").read()
assert raw[:4] == b"CWEB"
ver, N, M = struct.unpack("<HIH", raw[4:12]); off = 12
mm = np.frombuffer(raw, dtype="<u2", count=M*N*3, offset=off).reshape(M, N, 3)

# warp-distributed frame selection (late-dense), matching the scroll mapping
u = np.linspace(0, 1, min(NFRAMES, M))
fidx = np.unique((( 1 - (1-u)**EXPO ) * (M - 1)).round().astype(int))
Mk = fidx.size

# soft splat offsets/weights (center + 4-neighbour + 4-corner), as in the component
SPLAT = [(0,0,1.0),(0,1,.5),(0,-1,.5),(1,0,.5),(-1,0,.5),
         (1,1,.25),(1,-1,.25),(-1,1,.25),(-1,-1,.25)]

def density(fi):
    p = mm[fi].astype(np.float32) / 65535.0
    xi = np.clip((p[:,0]*(GRID-2)+1).astype(np.int32), 1, GRID-2)
    yi = np.clip(((1-p[:,1])*(GRID-2)+1).astype(np.int32), 1, GRID-2)
    acc = np.zeros((GRID, GRID), np.float32)
    for dy, dx, wt in SPLAT:
        np.add.at(acc, (yi+dy, xi+dx), PA*wt)
    return acc

# first pass: find a global scale so the densest frame uses the uint8 range well
# (a single global scale keeps relative brightness consistent across frames).
print(f"baking {Mk} frames at {GRID}x{GRID} from {N} particles ...")
accs = []
gmax = 0.0
for k, fi in enumerate(fidx):
    a = density(fi); accs.append(a)
    gmax = max(gmax, np.percentile(a, 99.97))      # robust max (ignore a few spikes)
    if k % 20 == 0: print(f"  {k}/{Mk}")
scale = 255.0 / (gmax + 1e-9)

out = "public/sim/cosmic-web.bin"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "wb") as fh:
    fh.write(b"CWDN"); fh.write(struct.pack("<HHHB", 1, GRID, Mk, 0))
    for a in accs:
        fh.write(np.clip(a*scale, 0, 255).astype(np.uint8).tobytes())

sz = os.path.getsize(out)
gz = len(gzip.compress(open(out,'rb').read(), 6))
print(f"\nwrote {out}  {sz/1048576:.1f} MB raw  |  {gz/1048576:.1f} MB gzipped")
print(f"  {GRID}x{GRID} x {Mk} frames, uint8 density (full {N}-particle binning)")
