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
"""Validation of Theorem 4 (uniform strong-coupling gap) and every step of its proof.

What is checked
  Links as sites      iota(x,i) = 2x + e_i is injective (exhaustive on a box).
  Supports            iota(dp) lies in 2x + Lambda_0 for every plaquette (exhaustive).
  Interactions        sup |ReTr rho(U)| = N, and at most 3 plaquettes share a lower corner (boundary included),
                      so sup_z ||phi_z|| <= beta(g) = 6N/(g^4 c_N).
  Threshold           g >= g0(N)  <=>  beta(g) <= beta*;  beta* < c1;  1 - c2 beta* >= 1/2.
  Padding             the link sites iota(dp) lie in {0..2L}^3; the support boxes 2x + Lambda_0 reach 2L+2 and
                      lie in B_L = {0..2L+2}^3, so the padding is necessary.
  Applying [2]        illustration of Yarotsky's mechanism: gap >= 1 - c beta uniformly in size (three-level chain).
  Decoupling          gap(H (x) 1 + 1 (x) T) = min{gap H, gap T} for unique ground states.
  Normalization       H = (g^2 c_N/2) H-bar + const and gap(H-bar) >= 1/2  =>  H - E0 >= (g^2 c_N/4)(1 - P0).
  Gauge descent       positivity improving + symmetry => the unique positive ground state is invariant.
  Threshold values    6N^2/(N^2-1) is 8 at N=2, 27/4 at N=3, decreasing to 6 (exact arithmetic), so
                      g0(N)^4 beta* = 6N^2/(N^2-1) is bounded uniformly in N.

What is NOT checked: the numerical values of Yarotsky's constants c1, c2 (they enter only through beta*).
"""
import itertools
import math
import sys
from fractions import Fraction

import numpy as np
from scipy.linalg import expm

from common import Report, c_N, beta_star, g0, haar_sun

E = np.eye(3, dtype=int)


def main():
    rep = Report("Theorem 4")
    rng = np.random.default_rng(10)
    L = 4
    links = [(x, i) for x in itertools.product(range(L + 1), repeat=3) for i in range(3) if x[i] + 1 <= L]
    img = {tuple(2 * np.array(x) + E[i]) for x, i in links}
    rep.check("(i.1) iota is injective", len(img) == len(links), f"{len(links)} links")

    ok, corner = True, {}
    for x in itertools.product(range(L + 1), repeat=3):
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            if x[i] + 1 <= L and x[j] + 1 <= L:
                xa = np.array(x)
                pts = [2 * xa + E[i], 2 * (xa + E[i]) + E[j], 2 * (xa + E[j]) + E[i], 2 * xa + E[j]]
                ok &= all(np.all((p - 2 * xa >= 0) & (p - 2 * xa <= 2)) for p in pts)
                corner[x] = corner.get(x, 0) + 1
    rep.check("(i.2) iota(dp) is contained in 2x + Lambda_0", ok)
    rep.check("(i.4) at most 3 plaquettes per lower corner (boundary included)",
              max(corner.values()) == 3 and min(corner.values()) >= 1)

    ok, det = True, []
    for N in (2, 3, 4):
        U = haar_sun(50000, N, rng)
        mx = np.max(np.abs(np.real(np.trace(U, axis1=1, axis2=2))))
        ok &= mx <= N + 1e-9
        det.append(f"N={N}: max sample {mx:.3f}, at 1: {N}")
    rep.check("(i.3) sup |ReTr rho(U)| = N (so ||psi_p|| = 2N/c_N)", ok, "; ".join(det))

    ok = True
    for N in (2, 3, 4):
        for c1, c2 in [(0.1, 2.0), (0.02, 0.5), (0.5, 9.0)]:
            bs = beta_star(c1, c2); G0 = g0(N, c1, c2)
            ok &= bs < c1 and 1 - c2 * bs >= 0.5 - 1e-12
            for g in np.linspace(0.5 * G0, 2 * G0, 81):
                ok &= (6 * N / (g ** 4 * c_N(N)) <= bs * (1 + 1e-12)) == (g >= G0 * (1 - 1e-12))
    rep.check("(ii.1)-(ii.2) g >= g0 <=> 6N/(g^4 c_N) <= beta*;  beta* < c1;  1 - c2 beta* >= 1/2", ok)

    ok = True
    BL = 2 * L + 2
    for x in itertools.product(range(L + 1), repeat=3):
        if corner.get(x, 0):
            z = 2 * np.array(x)
            ok &= np.all(z >= 0) and np.all(z + 2 <= BL)
    # link sites iota(dp) lie in {0..2L}^3, but the support boxes 2x + Lambda_0 reach 2L + 2 in the
    # direction normal to the plaquette: this is why B_L is padded to {0..2L+2}^3.
    ok2, reach = True, 0
    for x in itertools.product(range(L + 1), repeat=3):
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            if x[i] + 1 <= L and x[j] + 1 <= L:
                xa = np.array(x)
                pts = [2 * xa + E[i], 2 * (xa + E[i]) + E[j], 2 * (xa + E[j]) + E[i], 2 * xa + E[j]]
                ok2 &= all(np.all((p >= 0) & (p <= 2 * L)) for p in pts)
                reach = max(reach, int((2 * xa + 2).max()))
    rep.check("(ii.3) padding: link sites in {0..2L}^3; supports 2x + Lambda_0 in B_L = {0..2L+2}^3 (and need it)",
              ok and ok2 and reach == 2 * L + 2, f"max support coordinate {reach} = 2L+2")

    # (ii.4) Yarotsky-type illustration
    h = np.diag([0.0, 1.0, 1.6])
    gaps = {}
    for beta in (0.02, 0.05, 0.1):
        gl = []
        for Ln in range(2, 7):
            d = 3 ** Ln
            H = sum(np.kron(np.kron(np.eye(3 ** k), h), np.eye(3 ** (Ln - k - 1))) for k in range(Ln))
            for k in range(Ln - 1):
                X = rng.normal(size=(9, 9)); X = X + X.T; X *= beta / np.linalg.norm(X, 2)
                H = H + np.kron(np.kron(np.eye(3 ** k), X), np.eye(3 ** (Ln - k - 2)))
            ev = np.linalg.eigvalsh(H)
            gl.append(ev[1] - ev[0])
        gaps[beta] = gl
    ok = all(min(v) >= 1 - 4 * b for b, v in gaps.items()) and all(max(v) - min(v) < 0.3 for v in gaps.values())
    rep.check("(ii.4) illustration: gap >= 1 - 4 beta uniformly in size",
              ok, "; ".join(f"beta={b}: min gap {min(v):.3f}" for b, v in gaps.items()))

    # (ii.5) decoupling
    ok = True
    for _ in range(20):
        A = rng.normal(size=(6, 6)); A = A + A.T
        T = np.diag(np.sort(rng.uniform(0, 3, 4)))
        ea, et = np.linalg.eigvalsh(A), np.linalg.eigvalsh(T)
        full = np.linalg.eigvalsh(np.kron(A, np.eye(4)) + np.kron(np.eye(6), T))
        ok &= abs((full[1] - full[0]) - min(ea[1] - ea[0], et[1] - et[0])) < 1e-9
    rep.check("(ii.5) gap of a decoupled sum = min of gaps", ok)

    # (ii.6) rescaling
    ok = True
    for _ in range(20):
        Hb = rng.normal(size=(8, 8)); Hb = Hb + Hb.T
        ev, V = np.linalg.eigh(Hb)
        ev = np.concatenate([[0.0], 0.5 + np.abs(ev[1:] - ev[0])])   # E0 = 0, gamma >= 1/2
        Hb = V @ np.diag(ev) @ V.T
        gg, cn = rng.uniform(1, 3), c_N(3)
        H = gg ** 2 * cn / 2 * Hb + 7.0 * np.eye(8)
        P0 = np.outer(V[:, 0], V[:, 0])
        M = H - (gg ** 2 * cn / 2 * ev[0] + 7.0) * np.eye(8) - gg ** 2 * cn / 4 * (np.eye(8) - P0)
        ok &= np.linalg.eigvalsh(M).min() > -1e-9
    rep.check("(ii.6) gap(H-bar) >= 1/2  =>  H_L - E_0 >= (g^2 c_N/4)(1 - P_0)", ok)

    # (ii.7) positivity improving + symmetry
    n = 9
    Adj = np.zeros((n, n))
    for k in range(n):
        Adj[k, (k + 1) % n] = Adj[(k + 1) % n, k] = 1
    Vpot = np.diag(np.cos(2 * np.pi * np.arange(n) / n * 3))     # invariant under rotation by 3 sites
    Hs = -Adj + Vpot
    K = expm(-Hs)
    ev, Vv = np.linalg.eigh(Hs)
    gs = Vv[:, 0] * np.sign(Vv[:, 0].sum())
    rot = np.roll(np.eye(n), 3, axis=0)
    rep.check("(ii.7) positivity improving => unique positive ground state, invariant under the symmetry",
              (K > 0).all() and (gs > 0).all() and ev[1] - ev[0] > 1e-9 and np.allclose(rot @ gs, gs))

    # (iii) exact arithmetic
    q = [Fraction(6 * N * N, N * N - 1) for N in range(2, 200)]
    ok = q[0] == 8 and q[1] == Fraction(27, 4) and all(q[k + 1] < q[k] for k in range(len(q) - 1)) \
        and abs(float(q[-1]) - 6) < 1e-3 and max(q) == q[0]
    rep.check("(iii) g0(N)^4 beta* = 6N^2/(N^2-1): 8 at N=2, 27/4 at N=3, decreasing to 6", ok)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
