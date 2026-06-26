# Offline simulation masters

Large, **local-only** precomputed simulations (git-ignored). The shipped web
assets in `public/sim/` are *derived* from these — never commit the masters.

## cosmic-web-master.bin  (~7 GB)
Full-resolution cosmological N-body "cosmic web" for the Opportunities page.
128³ = 2,097,152 particles, 600 frames (every integration step), z=49→0,
uint16 box-fraction positions. Format: `CWEB` magic | u16 ver=2 | u32 N | u16 M
| M·N·3 u16.

Regenerate (~30 min):  `python3 scripts/build_cosmic_web.py`
Derive shipped asset:  `python3 scripts/derive_web_asset.py [N_particles] [N_frames] [bits]`
  e.g. `python3 scripts/derive_web_asset.py 100000 150 16`  → public/sim/cosmic-web.bin
Preview a master:      `python3 scripts/render_cosmic_web.py`  → ~/Desktop/cosmic-web.mp4
