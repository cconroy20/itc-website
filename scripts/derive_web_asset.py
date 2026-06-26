#!/usr/bin/env python3
"""Derive the shipped web asset from the full-resolution master simulation.
Subsamples particles AND (optionally) frames -- the master is too big to load in
a browser (>2GB ArrayBuffer limit) -- writing a compact uint16 binary that the
CosmicWeb.astro component loads.

  python3 scripts/derive_web_asset.py [N_KEEP] [N_FRAMES] [BITS]   (default 100000 90 16)
  BITS = 16 (uint16, v4) or 8 (uint8, v5, half the size, coarser positions)
Reads  offline/cosmic-web-master.bin   ->   public/sim/cosmic-web.bin
"""
import numpy as np, struct, os, sys

NKEEP   = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
NFRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 90
BITS    = int(sys.argv[3]) if len(sys.argv) > 3 else 16
master = os.environ.get("COSMIC_OUT", "offline/cosmic-web-master.bin")
raw = open(master, "rb").read()
assert raw[:4] == b"CWEB"
ver = struct.unpack("<H", raw[4:6])[0]
N, M = struct.unpack("<IH", raw[6:12]); off = 12
assert ver == 2, "expected uint16 master"
mm = np.frombuffer(raw, dtype="<u2", count=M*N*3, offset=off).reshape(M, N, 3)

rng = np.random.default_rng(0)
pidx = np.sort(rng.choice(N, size=min(NKEEP, N), replace=False))            # particles
# Select frames WARP-DISTRIBUTED: dense at late cosmic times, sparse early, so the
# scroll (which dwells on late times) scrubs through plenty of frames late and few
# early. The browser then plays these LINEARLY in scroll. Matches warp 1-(1-u)^EXPO
# used previously, but baked into the frame selection so playback needs no warp.
EXPO = 3.2                                                                  # higher = more late-dense
u = np.linspace(0, 1, min(NFRAMES, M))
warped = 1 - (1 - u) ** EXPO                                                # 0..1, late-dense
fidx = np.unique((warped * (M - 1)).round().astype(int))                    # frames
Nk, Mk = pidx.size, fidx.size
# The browser renders a flat (x,y) projection and no longer uses depth, so DROP z:
# store only x,y per particle. Format v4 = 2D uint16; v5 = 2D uint8 (half size).
xy16 = mm[np.ix_(fidx, pidx, [0, 1])]                                       # (Mk, Nk, 2) uint16
if BITS == 8:
    out_arr = (xy16.astype(np.uint32) * 255 // 65535).astype(np.uint8)      # requantize to 0..255
    ver = 5
else:
    out_arr = xy16.astype("<u2"); ver = 4

out = "public/sim/cosmic-web.bin"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "wb") as fh:
    fh.write(b"CWEB"); fh.write(struct.pack("<HIH", ver, Nk, Mk)); fh.write(out_arr.tobytes())
print(f"wrote {out}  {os.path.getsize(out)/1048576:.1f} MB  ({Nk} particles x {Mk} frames, 2D uint{BITS})")
