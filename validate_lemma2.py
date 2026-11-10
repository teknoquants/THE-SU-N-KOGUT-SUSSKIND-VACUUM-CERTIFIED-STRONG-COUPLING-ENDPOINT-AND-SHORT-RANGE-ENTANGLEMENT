# MIT License
#
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Validation of Lemma 2 (ground-state transform) and every step of its proof.

The lemma is a statement about a Schroedinger operator -(g^2/2) Laplacian + V on a compact Riemannian manifold
with a positive ground state. It is checked here on the circle with a spectral (Fourier) discretization, where
all identities hold to spectral accuracy, and on a periodic graph, where the discrete analogue is exact.

What is checked
  (a) Pointwise identity  Omega^-1 A (Omega f) = -(g^2/2)(f'' + 2 h' f'),  h = log Omega.
  (b) Form identity  <Omega f, A Omega f> = (g^2/2) int |f'|^2 dpi,  dpi = Omega^2 dx  (30 random f).
  (c) Unitarity: f -> Omega f maps L^2(pi) isometrically onto L^2(dx) and constants to Omega.
  (d) Gap equivalence: E_1 - E_0 = (g^2/2) / C_P, where C_P is the best Poincare constant of pi
      (generalized eigenproblem), for three couplings.
  (e) Discrete analogue on a periodic graph (exact): <Omega f, A Omega f> = (g^2/2) sum_edges Omega_i Omega_j (f_i - f_j)^2.
"""
import math
import sys

import numpy as np
from scipy.linalg import eigh

from common import Report, fourier_diff


def model(n, g):
    x = np.arange(n) * 2 * math.pi / n
    V = 1.3 * np.cos(x) + 0.4 * np.cos(2 * x) + 0.2 * np.sin(3 * x)
    D1, D2 = fourier_diff(n)
    H = -(g ** 2 / 2) * D2 + np.diag(V)
    H = 0.5 * (H + H.T)
    ev, vec = np.linalg.eigh(H)
    Om = vec[:, 0] * np.sign(vec[:, 0].sum())
    dx = 2 * math.pi / n
    Om /= math.sqrt(np.sum(Om ** 2) * dx)
    return x, V, D1, D2, H, ev, Om, dx


def trig(x, rng, deg=6):
    a, b = rng.normal(size=(2, deg)) / (1 + np.arange(deg)) ** 2
    return sum(a[k] * np.cos((k + 1) * x) + b[k] * np.sin((k + 1) * x) for k in range(deg)) + rng.normal()


def main():
    rep = Report("Lemma 2")
    rng = np.random.default_rng(22)
    g, n = 1.2, 127      # odd n: no Nyquist mode, so D1 has kernel = constants only
    x, V, D1, D2, H, ev, Om, dx = model(n, g)
    A = H - ev[0] * np.eye(n)
    h = np.log(Om)
    rep.check("ground state is strictly positive", np.all(Om > 0), f"min Omega = {Om.min():.3e}")

    err = 0
    for _ in range(30):
        f = trig(x, rng)
        lhs = (A @ (Om * f)) / Om
        rhs = -(g ** 2 / 2) * (D2 @ f + 2 * (D1 @ h) * (D1 @ f))
        err = max(err, np.abs(lhs - rhs).max() / max(1, np.abs(rhs).max()))
    rep.check("(a) Omega^-1 A(Omega f) = -(g^2/2) D_L f pointwise", err < 1e-8, f"max rel err {err:.1e}")

    err = 0
    for _ in range(30):
        f = trig(x, rng)
        lhs = np.sum(Om * f * (A @ (Om * f))) * dx
        rhs = g ** 2 / 2 * np.sum((D1 @ f) ** 2 * Om ** 2) * dx
        err = max(err, abs(lhs - rhs) / abs(rhs))
    rep.check("(b) <Omega f, A Omega f> = (g^2/2) int |f'|^2 dpi", err < 1e-8, f"max rel err {err:.1e}")

    f, k = trig(x, rng), trig(x, rng)
    rep.check("(c) f -> Omega f is unitary L^2(pi) -> L^2(dx) and maps 1 to Omega",
              abs(np.sum(f * k * Om ** 2) * dx - np.sum((Om * f) * (Om * k)) * dx) < 1e-12 and
              abs(np.sum(Om ** 2) * dx - 1) < 1e-12)

    ok, det = True, []
    for gg in (0.6, 1.2, 2.5):
        x, V, D1, D2, H, ev, Om, dx = model(255, gg)   # finer grid: Omega is sharply peaked at small g
        W = np.diag(Om ** 2 * dx)
        K = D1.T @ W @ D1
        lam = eigh(0.5 * (K + K.T), W, eigvals_only=True)
        lam1 = lam[1]                      # lam[0] = 0 on constants
        gap = ev[1] - ev[0]
        rel = abs(gap - gg ** 2 / 2 * lam1) / gap
        ok &= rel < 1e-6 and abs(lam[0]) < 1e-8
        det.append(f"g={gg}: gap {gap:.6f}, (g^2/2)/C_P {gg ** 2 / 2 * lam1:.6f}")
    rep.check("(d) E_1 - E_0 = (g^2/2)/C_P", ok, "; ".join(det))

    # (e) graph analogue: H = (g^2/2) graph Laplacian + V on a cycle with random weights
    m = 40
    w = rng.uniform(0.5, 2.0, m)
    Lg = np.zeros((m, m))
    for i in range(m):
        j = (i + 1) % m
        Lg[i, i] += w[i]; Lg[j, j] += w[i]; Lg[i, j] -= w[i]; Lg[j, i] -= w[i]
    Vg = rng.normal(size=m)
    Hg = g ** 2 / 2 * Lg + np.diag(Vg)
    e, v = np.linalg.eigh(Hg)
    O = np.abs(v[:, 0])
    Ag = Hg - e[0] * np.eye(m)
    err = 0
    for _ in range(30):
        f = rng.normal(size=m)
        lhs = (O * f) @ Ag @ (O * f)
        rhs = g ** 2 / 2 * sum(w[i] * O[i] * O[(i + 1) % m] * (f[i] - f[(i + 1) % m]) ** 2 for i in range(m))
        err = max(err, abs(lhs - rhs) / abs(rhs))
    rep.check("(e) discrete ground-state transform is exact on a weighted cycle", err < 1e-10, f"max rel err {err:.1e}")
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
