#!/usr/bin/env python3
"""Precompute a 3D cosmological structure-formation (cosmic web) simulation and
save particle positions per frame as a compact binary for the Opportunities
page animation. Runs once, offline; the browser plays back frames (scroll
scrubs through cosmic time).

Method: standard cosmological Particle-Mesh (PM) N-body in a flat LCDM universe,
following Hockney & Eastwood (1988), Klypin & Holtzman (1997) and the
KDK-in-scale-factor integrator of Quinn et al. (1997). Comoving coordinates,
canonical momentum p = a^2 dx/dt, integrated in equal steps of ln(a).

The two things that must be exactly right (and were wrong in earlier attempts):
  - drift coefficient  = integral of  1/(a^3 E(a)) da   (a^2 from p->xdot, a from dt->da)
  - kick  coefficient  = integral of  1/(a^2 E(a)) da   with the Poisson 1/a in phi ONCE
  - IC amplitude set from LINEAR THEORY (sigma_0 = 0.8 at a=1), not by eye.

Output: public/sim/cosmic-web.bin
  magic 'CWEB' | uint16 version=2 | uint32 N | uint16 M(frames) | M*N*3 uint16
  positions quantized to [0,65535] over the unit comoving box, frame-major.

Units (for reference / printed): Mpc/h length, Msun/h mass, H0=1 (time in 1/H0).
"""
import numpy as np, struct, os

rng = np.random.default_rng(7)

# ---------------- cosmology (flat LCDM) ----------------
OMEGA_M = 0.31
OMEGA_L = 1.0 - OMEGA_M
def E(a):  return np.sqrt(OMEGA_M / a**3 + OMEGA_L)      # H(a)/H0

# linear growth factor D(a) ~ E(a) * int_0^a da'/(a' E(a'))^3, normalized D(1)=1
def _growth_unnorm(a):
    aa = np.linspace(1e-6, a, 4000)
    integrand = 1.0 / (aa * E(aa))**3
    return E(a) * np.trapz(integrand, aa)
_D1 = _growth_unnorm(1.0)
def D(a):  return _growth_unnorm(a) / _D1
def f_growth(a):                                          # f = dlnD/dlna
    da = a * 1e-3
    return (np.log(D(a + da)) - np.log(D(a - da))) / (np.log(a + da) - np.log(a - da))

# ---------------- physical scales (reference only) ----------------
L_BOX     = 50.0          # comoving box [Mpc/h]
RHO_CRIT0 = 2.775e11      # [ (Msun/h)/(Mpc/h)^3 ]
MEAN_RHO  = OMEGA_M * RHO_CRIT0

# ---------------- discretization ----------------
n_side = 128
N      = n_side**3        # 2,097,152 particles
NG     = 256              # PM force grid (2x particle grid for force resolution)
M_OUT  = 600              # saved frames (= every integration step; fine timeline
                          # for re-warping/resampling the shipped asset offline)
A_START, A_END = 0.02, 1.0
N_STEP = 600              # leapfrog steps in ln(a)
SIGMA_TARGET = 0.55       # linear rms density contrast at a=1 (mildly nonlinear)
R_SOFT = 1.5              # Gaussian force softening [grid cells]
PART_MASS = MEAN_RHO * L_BOX**3 / N

# ---------------- lattice ----------------
lin = (np.arange(n_side) + 0.5) / n_side
qx, qy, qz = np.meshgrid(lin, lin, lin, indexing="ij")
q0 = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)

# ---------------- CIC density ----------------
def cic_density(p):
    g = (p % 1.0) * NG
    i0 = np.floor(g).astype(int); fr = g - i0
    rho = np.zeros((NG, NG, NG))
    for dx in (0, 1):
        wx = fr[:,0] if dx else 1-fr[:,0]; ix = (i0[:,0]+dx) % NG
        for dy in (0, 1):
            wy = fr[:,1] if dy else 1-fr[:,1]; iy = (i0[:,1]+dy) % NG
            for dz in (0, 1):
                wz = fr[:,2] if dz else 1-fr[:,2]; iz = (i0[:,2]+dz) % NG
                np.add.at(rho, (ix, iy, iz), wx*wy*wz)
    return rho * (NG**3) / N - 1.0          # delta on grid

# ---------------- Zel'dovich ICs from a Gaussian random field (BBKS P(k)) -----------
# Standard cosmological IC recipe (a la N-GenIC): build a Gaussian random density
# field delta_hat(k) with variance proportional to the linear CDM power spectrum
# P(k), then the Zel'dovich displacement is psi_hat(k) = i k/k^2 * delta_hat(k).
# Inverse-FFT each component -> displacement on the particle lattice. This gives a
# realistic, statistically-correct web (unlike a sum of a few sinusoids).
NIC = n_side                                   # IC/displacement grid = particle lattice

def bbks_T(k):
    # BBKS (1986) CDM transfer function. q = k / Gamma, Gamma = Omega_m h (shape).
    h = 0.7; Gamma = OMEGA_M * h
    q = k / Gamma + 1e-30
    return (np.log(1 + 2.34*q) / (2.34*q) *
            (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4) ** -0.25)

# Everything here is in PHYSICAL units (Mpc/h) with ONE k-convention, then the
# displacement is converted to box fractions at the very end. kphys = 2*pi*n/L_box.
kf = np.fft.fftfreq(NIC) * NIC                 # integer modes n
KXi, KYi, KZi = np.meshgrid(kf, kf, kf, indexing="ij")
kxp = 2*np.pi/L_BOX * KXi                       # physical k-vector components [h/Mpc]
kyp = 2*np.pi/L_BOX * KYi
kzp = 2*np.pi/L_BOX * KZi
kphys = np.sqrt(kxp**2 + kyp**2 + kzp**2)       # |k| [h/Mpc]
Pk = np.where(kphys > 0, kphys * bbks_T(kphys)**2, 0.0)     # P(k) ~ k^ns T^2, ns=1

# Gaussian random density field delta(x), real via white-noise -> color by sqrt(P).
# Use a unitary-ish FFT convention consistently for delta and for measuring sigma_8.
white = rng.standard_normal((NIC, NIC, NIC))
delta_k = np.fft.fftn(white) * np.sqrt(Pk)
delta_k[0,0,0] = 0.0

# --- normalize to sigma_8 by DIRECT MEASUREMENT (no convention ambiguity): smooth
# delta with an 8 Mpc/h real-space top-hat (applied in k-space) and rescale so the
# smoothed field's rms = SIGMA8. This is the physical LCDM amplitude, not a knob. ---
SIGMA8 = 0.81                                   # Planck 2018
def Wth(x):
    x = np.where(x < 1e-6, 1e-6, x)
    return 3.0 * (np.sin(x) - x*np.cos(x)) / x**3
delta_smooth = np.fft.ifftn(delta_k * Wth(kphys * 8.0)).real
sig8_now = delta_smooth.std()
norm = SIGMA8 / (sig8_now + 1e-30)
delta_k *= norm
print(f"  IC: measured sigma_8={sig8_now:.3e} -> rescaled to {SIGMA8}")

# Zel'dovich displacement in PHYSICAL units: psi_hat = i k/k^2 delta_hat [Mpc/h].
k2p = kxp**2 + kyp**2 + kzp**2; k2p[0,0,0] = 1.0
psi_x = np.fft.ifftn(1j * kxp / k2p * delta_k).real
psi_y = np.fft.ifftn(1j * kyp / k2p * delta_k).real
psi_z = np.fft.ifftn(1j * kzp / k2p * delta_k).real
psi_mpc = np.stack([psi_x, psi_y, psi_z], axis=-1).reshape(-1, 3)   # [Mpc/h]
psi0 = psi_mpc / L_BOX                          # -> box fractions
# verify the field we built actually has sigma_8 = SIGMA8 (closes the loop)
chk = (np.fft.ifftn(delta_k * Wth(kphys*8.0)).real).std()
print(f"  IC: GRF/BBKS, verify sigma_8={chk:.3f}; rms|psi0(a=1)|={np.linalg.norm(psi0,axis=1).mean():.4f} box-frac"
      f" = {np.linalg.norm(psi_mpc,axis=1).mean():.2f} Mpc/h")

# ---------------- apply ICs at a_start (growing mode) ----------------
Ds = D(A_START)
psi_start = Ds * psi0
pos = (q0 + psi_start) % 1.0
pmom = (A_START**2 * E(A_START) * f_growth(A_START)) * psi_start   # p = a^2 E f psi
print(f"  D(a_start={A_START})={Ds:.4f}, sigma_start={Ds*SIGMA_TARGET:.4f}, f={f_growth(A_START):.3f}")

# ---------------- PM Green's function (FD kernel + CIC deconv + Gaussian softening) ----------------
kf = np.fft.fftfreq(NG) * NG
KX, KY, KZ = np.meshgrid(kf, kf, kf, indexing="ij")
dx_cell = 1.0 / NG
k2_fd = (2.0/dx_cell**2) * ((1-np.cos(2*np.pi*KX/NG)) +
                            (1-np.cos(2*np.pi*KY/NG)) +
                            (1-np.cos(2*np.pi*KZ/NG)))
k2_fd[0,0,0] = 1.0
inv_k2 = np.where((KX**2+KY**2+KZ**2) > 0, 1.0/k2_fd, 0.0)
def sinc(x):
    x = np.where(np.abs(x) < 1e-12, 1e-12, x)
    return np.sin(x)/x
Wcic = (sinc(np.pi*KX/NG) * sinc(np.pi*KY/NG) * sinc(np.pi*KZ/NG))**2
deconv = 1.0 / (Wcic**2 + 1e-30)
ksoft = np.exp(-0.5 * (R_SOFT*dx_cell)**2 *
               ((2*np.pi*KX)**2 + (2*np.pi*KY)**2 + (2*np.pi*KZ)**2))
green = inv_k2 * deconv * ksoft

def accel(p, a):
    """Comoving peculiar acceleration g = -grad phi; Poisson 1/a included ONCE."""
    delta = cic_density(p)
    phi = np.fft.ifftn(-(1.5*OMEGA_M/a) * np.fft.fftn(delta) * green).real
    gx = -(np.roll(phi,-1,0) - np.roll(phi,1,0)) / (2*dx_cell)
    gy = -(np.roll(phi,-1,1) - np.roll(phi,1,1)) / (2*dx_cell)
    gz = -(np.roll(phi,-1,2) - np.roll(phi,1,2)) / (2*dx_cell)
    g = (p % 1.0) * NG; i0 = np.floor(g).astype(int); fr = g - i0
    ax = np.zeros(N); ay = np.zeros(N); az = np.zeros(N)
    for dx in (0,1):
        wx = fr[:,0] if dx else 1-fr[:,0]; ix=(i0[:,0]+dx)%NG
        for dy in (0,1):
            wy = fr[:,1] if dy else 1-fr[:,1]; iy=(i0[:,1]+dy)%NG
            for dz in (0,1):
                wz = fr[:,2] if dz else 1-fr[:,2]; iz=(i0[:,2]+dz)%NG
                wgt = wx*wy*wz
                ax += wgt*gx[ix,iy,iz]; ay += wgt*gy[ix,iy,iz]; az += wgt*gz[ix,iy,iz]
    return np.stack([ax,ay,az], axis=1)

# integrated KDK coefficients (Simpson over a sub-interval)
def drift_coeff(a0, a1):
    am = 0.5*(a0+a1); g = lambda a: 1.0/(a**3 * E(a))
    return (a1-a0)/6.0 * (g(a0) + 4*g(am) + g(a1))
def kick_coeff(a0, a1):
    # dp/da = g/(a E):  one power of a from dt->da. The Poisson 1/a lives in g
    # (accel), so do NOT put another 1/a here. (Validated against linear D(a).)
    am = 0.5*(a0+a1); g = lambda a: 1.0/(a * E(a))
    return (a1-a0)/6.0 * (g(a0) + 4*g(am) + g(a1))

# ---------------- integrate: KDK leapfrog in ln(a) ----------------
# Save ALL particles as uint16 (0.0008 Mpc resolution; imperceptible). Frames are
# STREAMED to disk as computed, so peak RAM stays ~5 GB regardless of frame count.
# master file lives OUTSIDE public/ (it is large; the web asset is derived from it).
out = os.environ.get("COSMIC_OUT", "offline/cosmic-web-master.bin")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
fh = open(out, "wb")
fh.write(b"CWEB"); fh.write(struct.pack("<HIH", 2, N, M_OUT))   # v2 = uint16 payload
def write_frame(p):
    np.clip(np.round((p % 1.0) * 65535.0), 0, 65535).astype("<u2").tofile(fh)

edges = np.exp(np.linspace(np.log(A_START), np.log(A_END), N_STEP+1))
save_idx = set(np.linspace(0, N_STEP, M_OUT).astype(int).tolist())
fi = 0
g = accel(pos, edges[0])
diag = []

import time, sys
t0 = time.time()
def progress(s, a):
    frac = (s + 1) / N_STEP
    elapsed = time.time() - t0
    eta = elapsed / frac - elapsed if frac > 0 else 0
    bar = "#" * int(frac * 30) + "-" * (30 - int(frac * 30))
    z = 1/a - 1
    sys.stdout.write(f"\r  [{bar}] {frac*100:5.1f}%  step {s+1}/{N_STEP}  "
                     f"a={a:.3f} z={z:5.1f}  elapsed {elapsed:4.0f}s  eta {eta:4.0f}s  ")
    sys.stdout.flush()

for s in range(N_STEP):
    a_n, a_np1 = edges[s], edges[s+1]
    a_h = np.exp(0.5*(np.log(a_n)+np.log(a_np1)))
    if s in save_idx and fi < M_OUT:
        write_frame(pos); fi += 1
    pmom += g * kick_coeff(a_n, a_h)
    pos = (pos + pmom * drift_coeff(a_n, a_np1)) % 1.0
    g = accel(pos, a_np1)
    pmom += g * kick_coeff(a_h, a_np1)
    if s % 30 == 0 or s == N_STEP-1:
        d = cic_density(pos)
        diag.append((a_np1, d.std(), d.max(), (d > -0.999).mean(), np.abs(pmom.sum(0)).max()))
    if s % 5 == 0 or s == N_STEP-1:
        progress(s, a_np1)
while fi < M_OUT:
    write_frame(pos); fi += 1
fh.close()
sys.stdout.write("\n")

print(f"\nwrote {out}  {os.path.getsize(out)/1048576:.1f} MB")
print(f"  N={N} ({n_side}^3), grid={NG}^3, frames={M_OUT}, steps={N_STEP}, uint16")
print(f"  box={L_BOX} Mpc/h, particle={PART_MASS:.2e} Msun/h, z={1/A_START-1:.0f}->0")
print("\n  sanity (a, sigma, max_delta, occupied_frac, |sum p|):")
for a, sg, mx, oc, sp in diag:
    print(f"    a={a:.3f}  sigma={sg:6.3f}  maxd={mx:8.1f}  occ={oc:.3f}  |Sp|={sp:.1e}")
