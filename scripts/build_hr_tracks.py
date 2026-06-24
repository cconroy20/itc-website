#!/usr/bin/env python3
"""Build the reduced HR-diagram track data for the Frontiers-page animation.

Reads raw MIST EEP track files (one per stellar mass) and writes a single
compact JSON (public/tracks/mist.json) holding only (age, logTeff, logL) per
mass, on a faithful-but-small sampling. The HRDiagram.astro component loads that
JSON and animates a coeval population evolving across the log Teff – log L plane.

Pipeline per track:
  1. Read native rows, keep only star_age (col 1), log_L (col 7), log_Teff
     (col 12); stop after the first MAX_ROWS data rows.
  2. Prepend a young "birth" point at YOUNG yr (holding the first real logT/logL)
     so every track is present from the animation's clock start.
  3. Douglas–Peucker thin in the (logTeff, logL) plane to drop near-collinear
     points while preserving curve shape to EPS dex.
  4. Re-densify so no gap between kept points exceeds MAXGAP in log10(age),
     inserting points interpolated (in log-age) from the full native track.
     This restores a smooth time cadence for the slow low-mass tracks.

Mass selection: everything up to MAX_MASS; below 2 Msun, thinned to a 0.05-Msun
grid (the native grid is 0.01-dense around 1 Msun, which is visually too dense).

Usage:
    python3 scripts/build_hr_tracks.py            # uses SRC below
    python3 scripts/build_hr_tracks.py /path/to/eeps
    MAXGAP=0.03 python3 scripts/build_hr_tracks.py   # override via env

Re-run after changing any parameter, then rebuild the site (npm run build).
The JSON is committed to the repo; the raw EEP files are NOT (too large).
"""
import os
import sys
import json
import math
import glob
import re

# --- parameters (override via environment variables) ----------------------
SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/cconroy/mist/v2.5/feh_p000_afe_p0_vvcrit0.4/eeps"
OUT = os.path.join(os.path.dirname(__file__), "..", "public", "tracks", "mist.json")

MAX_MASS = float(os.environ.get("MAX_MASS", 50.0))   # drop masses above this (Msun)
MAX_ROWS = int(os.environ.get("MAX_ROWS", 808))      # keep only first N native rows
LOWMASS_STEP = 0.05   # below 2 Msun, keep masses on this grid
MAXGAP = float(os.environ.get("MAXGAP", 0.015))      # max log10(age) gap (dex)
EPS = float(os.environ.get("EPS", 0.0035))           # Douglas–Peucker tol (dex)
YOUNG = float(os.environ.get("YOUNG", 1e5))          # birth-point age (yr)

# native EEP column indices (0-based): star_age, log_L, log_Teff
C_AGE, C_LOGL, C_LOGT = 0, 6, 11


def mass_of(path):
    return int(re.search(r"(\d{5})M\.track\.eep$", path).group(1)) / 100.0


def keep_mass(m):
    if m > MAX_MASS:
        return False
    if m < 2.0:                                       # thin the dense low-mass grid
        return abs(round(m / LOWMASS_STEP) * LOWMASS_STEP - m) < 1e-6
    return True


def read_track(path):
    age, logL, logT = [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            try:
                a, l, t = float(p[C_AGE]), float(p[C_LOGL]), float(p[C_LOGT])
            except (ValueError, IndexError):
                continue
            age.append(a); logL.append(l); logT.append(t)
            if len(age) >= MAX_ROWS:
                break
    return age, logL, logT


def douglas_peucker(T, L, eps):
    """Indices to keep so the (T,L) polyline stays within eps of the original."""
    n = len(T)
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        x0, y0, x1, y1 = T[i], L[i], T[j], L[j]
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy) or 1e-12
        dmax, idx = 0.0, -1
        for k in range(i + 1, j):
            d = abs(dy * (T[k] - x0) - dx * (L[k] - y0)) / seg
            if d > dmax:
                dmax, idx = d, k
        if dmax > eps:
            keep[idx] = True
            stack.append((i, idx)); stack.append((idx, j))
    return [k for k in range(n) if keep[k]]


def interp_native(age, logT, logL, a):
    """Linear interpolation (in log10 age) of (T,L) from the full native track."""
    if a <= age[0]:
        return (age[0], logT[0], logL[0])
    if a >= age[-1]:
        return (age[-1], logT[-1], logL[-1])
    lo, hi = 0, len(age) - 1
    while hi - lo > 1:
        m = (lo + hi) // 2
        if age[m] <= a:
            lo = m
        else:
            hi = m
    la0, la1, la = math.log10(age[lo]), math.log10(age[hi]), math.log10(a)
    f = (la - la0) / (la1 - la0 or 1)
    return (a, logT[lo] + f * (logT[hi] - logT[lo]),
            logL[lo] + f * (logL[hi] - logL[lo]))


def main():
    files = sorted(glob.glob(os.path.join(SRC, "*.track.eep")))
    if not files:
        sys.exit(f"No *.track.eep files found in {SRC}")
    files = [f for f in files if keep_mass(mass_of(f))]

    out = []
    for path in files:
        age, logL, logT = read_track(path)
        if len(age) < 2:
            continue
        if age[0] > YOUNG:                            # ensure presence from clock start
            age = [YOUNG] + age
            logT = [logT[0]] + logT
            logL = [logL[0]] + logL

        idx = douglas_peucker(logT, logL, EPS)
        logage = [math.log10(a) if a > 0 else 0 for a in age]

        pts = []
        for n, i in enumerate(idx):
            pts.append((age[i], logT[i], logL[i]))
            if n + 1 < len(idx):
                j = idx[n + 1]
                gap = logage[j] - logage[i]
                if gap > MAXGAP:                      # densify large age gaps
                    steps = int(gap / MAXGAP)
                    for s in range(1, steps + 1):
                        la = logage[i] + (logage[j] - logage[i]) * s / (steps + 1)
                        pts.append(interp_native(age, logT, logL, 10 ** la))

        out.append({
            "m": mass_of(path),
            "age": [round(p[0]) for p in pts],
            "logT": [round(p[1], 4) for p in pts],
            "logL": [round(p[2], 4) for p in pts],
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    text = json.dumps({"tracks": out}, separators=(",", ":"))
    with open(OUT, "w") as fh:
        fh.write(text)

    total = sum(len(t["age"]) for t in out)
    print(f"wrote {os.path.relpath(OUT)}  {len(text) / 1024:.0f} KB  | "
          f"{len(out)} masses, {total} points  | "
          f"MAX_MASS={MAX_MASS} MAXGAP={MAXGAP} EPS={EPS}")


if __name__ == "__main__":
    main()
