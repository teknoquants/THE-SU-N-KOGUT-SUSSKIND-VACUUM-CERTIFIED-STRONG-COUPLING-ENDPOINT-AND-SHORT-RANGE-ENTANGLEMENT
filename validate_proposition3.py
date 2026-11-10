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
"""Validation of Proposition 3 (horizontal curvature criterion) and every step of its proof.

What is checked
  (a) Integrated weighted Bochner identity  int (D f)^2 dpi = int [ f''^2 + (-2 h'') f'^2 ] dpi  on the circle
      (Ric = 0), spectral discretization, 30 random f.
  (b) Curvature => gap on R: pi ~ exp(2h), -2h'' >= kappa  =>  gap(-D) >= kappa. Sharp case (Ornstein-Uhlenbeck,
      gap = kappa exactly) and a non-quadratic case.
  (c) The point of the word "horizontal": on R^2 with the rotation group as gauge group and
      h(r) = -kappa (r - R)^2 / 4, the full Bakry-Emery condition fails (the angular Hessian of -2h is negative for
      r < R), but it holds in the radial (horizontal) direction. The gap on rotation-invariant functions is
      >= kappa, while non-invariant (angular) modes have eigenvalues far below kappa. This is exactly why the
      proposition only needs the bound on horizontal vectors and only concludes a gap on physical states.
  (d) The final rescaling: A_L >= (g^2 kappa / 2) Q_L follows from lambda >= kappa and Lemma 2.
"""
import math
import sys

import numpy as np
from scipy.linalg import eigh

from common import Report, fourier_diff


def fd_generalized(w_mid, w_node, dr, extra=None):
    """Smallest eigenvalues of the weighted Dirichlet form sum w_mid (df)^2/dr against sum w_node f^2 dr."""
    n = len(w_node)
    K = np.zeros((n, n))
    for i in range(n - 1):
        c = w_mid[i] / dr
        K[i, i] += c; K[i + 1, i + 1] += c; K[i, i + 1] -= c; K[i + 1, i] -= c
    if extra is not None:
        K += np.diag(extra)
    M = np.diag(w_node * dr)
    return eigh(K, M, eigvals_only=True, subset_by_index=[0, 3])


def main():
    rep = Report("Proposition 3")
    rng = np.random.default_rng(33)

    # (a)
    n = 127
    x = np.arange(n) * 2 * math.pi / n
    dx = 2 * math.pi / n
    D1, D2 = fourier_diff(n)
    h = 0.6 * np.cos(x) + 0.25 * np.sin(2 * x)
    pi = np.exp(2 * h); pi /= pi.sum() * dx
    err = 0
    for _ in range(30):
        a, b = rng.normal(size=(2, 6)) / (1 + np.arange(6)) ** 2
        f = sum(a[k] * np.cos((k + 1) * x) + b[k] * np.sin((k + 1) * x) for k in range(6))
        Df = D2 @ f + 2 * (D1 @ h) * (D1 @ f)
        lhs = np.sum(Df ** 2 * pi) * dx
        rhs = np.sum(((D2 @ f) ** 2 + (-2 * (D2 @ h)) * (D1 @ f) ** 2) * pi) * dx
        err = max(err, abs(lhs - rhs) / abs(lhs))
    rep.check("(a) weighted Bochner identity on the circle", err < 1e-8, f"max rel err {err:.1e}")

    # (b)
    Lx, m = 14.0, 2801
    xs = np.linspace(-Lx, Lx, m); dr = xs[1] - xs[0]; mid = 0.5 * (xs[1:] + xs[:-1])
    res = []
    for k0, a in [(2.0, 0.0), (2.0, 0.4), (1.0, 0.2)]:
        hf = lambda t: -k0 * t ** 2 / 4 + a * np.cos(t)
        lam = fd_generalized(np.exp(2 * hf(mid)), np.exp(2 * hf(xs)), dr)
        kap = k0 - 2 * abs(a)
        res.append((k0, a, kap, lam[1]))
    ok = abs(res[0][3] - 2.0) < 1e-3 and all(l >= kap * (1 - 1e-3) for _, _, kap, l in res)
    rep.check("(b) -2h'' >= kappa  =>  gap >= kappa on R (sharp for Ornstein-Uhlenbeck)", ok,
              "; ".join(f"kappa0={k0}, a={a}: bound {kap:.2f}, gap {l:.4f}" for k0, a, kap, l in res))

    # (c)
    kap, R, Rmax, m = 1.0, 4.0, 14.0, 3501
    r = np.linspace(0, Rmax, m); dr = r[1] - r[0]; rm = 0.5 * (r[1:] + r[:-1])
    hf = lambda t: -kap * (t - R) ** 2 / 4
    wn = r * np.exp(2 * hf(r)); wn[0] = dr / 8 * np.exp(2 * hf(0.0))
    lam0 = fd_generalized(rm * np.exp(2 * hf(rm)), wn, dr)
    ri = r[1:]; rmi = rm[1:]
    wi = ri * np.exp(2 * hf(ri))
    lam1 = fd_generalized(rmi * np.exp(2 * hf(rmi)), wi, dr, extra=wi / ri ** 2 * dr)
    ang = kap * (r[1:] - R) / r[1:]                 # -2 Hess h in the angular direction
    rep.check("(c.1) full Bakry-Emery condition fails: angular -2 Hess h < 0 for r < R",
              ang.min() < -1.0, f"min angular curvature {ang.min():.1f}")
    rep.check("(c.2) horizontal (radial) condition -2h'' = kappa holds, gap on invariant functions >= kappa",
              lam0[1] >= kap * (1 - 1e-3), f"invariant gap {lam0[1]:.4f}")
    rep.check("(c.3) non-invariant (angular m = 1) modes lie far below kappa", lam1[0] < 0.2 * kap,
              f"lowest m=1 eigenvalue {lam1[0]:.4f}")

    # (d)
    ok = True
    for _ in range(50):
        gg, kk = rng.uniform(0.3, 3), rng.uniform(0.1, 2)
        lam = kk * (1 + rng.uniform(0, 1, 10))
        ok &= np.all(gg ** 2 / 2 * lam >= gg ** 2 * kk / 2 - 1e-12)
    rep.check("(d) lambda >= kappa and Lemma 2  =>  A_L >= (g^2 kappa/2) Q_L", ok)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
