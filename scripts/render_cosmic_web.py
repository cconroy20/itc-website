#!/usr/bin/env python3
"""Render public/sim/cosmic-web.bin to an MP4 to preview the simulation.
Projects 3D -> 2D with a brightness+size depth cue, violet on near-black, with
small semi-transparent points and additive glow (dense filaments light up).
Interpolates between saved frames for smooth, slow playback without re-simulating.
Output: ~/Desktop/cosmic-web.mp4
"""
import numpy as np, struct, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

import os
PATH=os.environ.get("COSMIC_OUT", "offline/cosmic-web-master.bin")
raw = open(PATH,"rb").read()
assert raw[:4]==b"CWEB"
ver = struct.unpack("<H", raw[4:6])[0]
N,M = struct.unpack("<IH", raw[6:12]); off=12
dt = np.dtype("<f4") if ver>=3 else np.dtype("<u2")
mm = np.frombuffer(raw, dtype=dt, count=M*N*3, offset=off).reshape(M, N, 3)  # view, no copy
def frame(fi):
    f = mm[fi].astype(np.float32)
    return f if ver>=3 else f/65535.0
print(f"N={N} frames={M}")

viol = np.array([167,139,250])/255.0           # accent violet #a78bfa

# render tunables
PT_SIZE   = 0.5        # base point size (small)
PT_ALPHA  = 0.06       # base transparency (lower -> glow builds up in dense regions)
INTERP    = 1          # no interpolation (play real frames; avoids interp seams/teleport)
FPS       = 18         # playback fps (lower = slower movie)
DPI       = 300        # frame resolution: figsize(7) * DPI = pixels per side
BITRATE   = 24000      # high bitrate so the fine detail survives compression

fig = plt.figure(figsize=(7,7), dpi=DPI)
ax = fig.add_axes([0,0,1,1]); ax.set_facecolor("#05050b"); fig.patch.set_facecolor("#05050b")
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xticks([]); ax.set_yticks([])
# additive-ish blend: many faint points overlap -> bright filaments
scat = ax.scatter([], [], s=PT_SIZE, c="white", linewidths=0)

def draw(frame_xyz):
    x, y, z = frame_xyz[:,0], frame_xyz[:,1], frame_xyz[:,2]
    depth = z
    size = PT_SIZE * (0.5 + depth*1.8)          # nearer = a bit bigger
    alpha = PT_ALPHA * (0.45 + depth*0.9)       # nearer = a bit brighter
    colors = np.zeros((N,4)); colors[:,0:3]=viol; colors[:,3]=np.clip(alpha,0,1)
    order = np.argsort(z)                        # back-to-front
    scat.set_offsets(np.c_[x[order], y[order]])
    scat.set_sizes(size[order]); scat.set_color(colors[order])

writer = FFMpegWriter(fps=FPS, bitrate=BITRATE)
out = os.path.expanduser("~/Desktop/cosmic-web-hq.mp4")
with writer.saving(fig, out, dpi=DPI):
    if INTERP <= 1:
        for fi in range(M):
            draw(frame(fi)); writer.grab_frame()
    else:
        for fi in range(M-1):
            A, B = frame(fi), frame(fi+1)
            for s in range(INTERP):
                t = s / INTERP
                draw(A*(1-t) + B*t); writer.grab_frame()
        draw(frame(M-1)); writer.grab_frame()
print(f"wrote {out}  ({(M-1)*INTERP+1} rendered frames @ {FPS}fps)")
