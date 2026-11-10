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
"""Validation of Definitions 1-11.

What is checked
  D1   The su(N) basis is orthonormal for <X,Y> = -Tr(XY); Killing form B(X,Y) = Tr(ad_X ad_Y) = 2N Tr(XY);
       c_N = (N^2-1)/N.
  D2   Link reversal U_ebar = U_e^{-1}; the plaquette holonomy closes; W(C) is a class function with |W| <= 1.
  D3   Gauge transformations leave every W(C) and every ReTr U_p invariant (Haar samples, SU(2) and SU(3)).
  D5   H-bar normalization: each h_e has gap 1 (character eigenvalues of SU(2)).
  D7   iota is injective; plaquette supports lie in 2x + Lambda_0; B_L contains them.
  D8   beta* < c1 and 1 - c2 beta* >= 1/2; g >= g0(N)  <=>  6N/(g^4 c_N) <= beta*.
  D9   zeta 1 lies in the centre of SU(N); Stokes: eta(dSigma) = sum_p Sigma_p (delta eta)(p).
  D10  The set P(C) of plaquettes containing a link of C has at most 8(r+s) elements.
"""
import itertools
import math
import sys

import numpy as np

from common import (Report, c_N, beta_star, g0, su_basis, ip, ad_matrix, haar_sun, chi_su2,
                    box_geometry, plaq_boundary, rect_loop, holonomy, E3)


def main():
    rep = Report("Definitions")
    rng = np.random.default_rng(1)

    # D1
    ok, det = True, []
    for N in range(2, 7):
        B = su_basis(N)
        G = np.array([[ip(a, b) for b in B] for a in B])
        ok &= len(B) == N * N - 1 and np.allclose(G, np.eye(N * N - 1))
        for _ in range(5):
            x, y = rng.normal(size=(2, N * N - 1))
            X = sum(c * T for c, T in zip(x, B)); Y = sum(c * T for c, T in zip(y, B))
            kil = np.trace(ad_matrix(X, B) @ ad_matrix(Y, B))
            ok &= abs(kil - 2 * N * np.real(np.trace(X @ Y))) < 1e-9
        det.append(f"N={N}")
    rep.check("D1 orthonormal basis; Killing form B(X,Y) = 2N Tr(XY)", ok, ", ".join(det))

    # D2, D3
    links, plaqs = box_geometry(2, 2, 1)
    li = {l: k for k, l in enumerate(links)}
    verts = sorted({l[0] for l in links} | {tuple(a + b for a, b in zip(l[0], E3[l[1]])) for l in links})
    vi = {v: k for k, v in enumerate(verts)}
    loop = rect_loop((0, 0, 0), 2, 1)
    for N in (2, 3):
        S = 2000
        U = haar_sun(S * len(links), N, rng).reshape(S, len(links), N, N)
        W = np.trace(holonomy(loop, U, li), axis1=-2, axis2=-1) / N
        Pp = [np.real(np.trace(holonomy(plaq_boundary(p), U, li), axis1=-2, axis2=-1)) for p in plaqs]
        h = haar_sun(S * len(verts), N, rng).reshape(S, len(verts), N, N)
        Ug = np.empty_like(U)
        for (v, i), k in li.items():
            t = tuple(a + b for a, b in zip(v, E3[i]))
            Ug[:, k] = h[:, vi[v]] @ U[:, k] @ np.conj(np.swapaxes(h[:, vi[t]], -1, -2))
        Wg = np.trace(holonomy(loop, Ug, li), axis1=-2, axis2=-1) / N
        Pg = [np.real(np.trace(holonomy(plaq_boundary(p), Ug, li), axis1=-2, axis2=-1)) for p in plaqs]
        rep.check(f"D2/D3 SU({N}): |W(C)| <= 1; W(C) and ReTr U_p gauge invariant",
                  np.all(np.abs(W) <= 1 + 1e-9) and np.allclose(W, Wg) and all(np.allclose(a, b) for a, b in zip(Pp, Pg)),
                  f"{S} configurations")
    Uc = haar_sun(1, 3, rng)[0]
    rev = [((links[0]), +1), ((links[0]), -1)]
    U1 = np.array([Uc] + [np.eye(3)] * (len(links) - 1))
    rep.check("D2 link reversal: U_e U_ebar = 1", np.allclose(holonomy(rev, U1, li), np.eye(3)))

    # D5
    ev = [(n * n - 1) / 2 / c_N(2) for n in range(1, 8)]
    U = haar_sun(20000, 2, rng)
    ok = abs(np.mean(chi_su2(U, 2) ** 2) - 1) < 0.03 and abs(np.mean(chi_su2(U, 1)) - 1) < 1e-12
    rep.check("D5 h_e = -Laplacian/c_N has kernel {const} and gap exactly 1 (SU(2): (n^2-1)/3)",
              ev[0] == 0 and abs(ev[1] - 1) < 1e-12 and all(e >= 1 for e in ev[1:]) and ok,
              f"first levels {[round(e, 3) for e in ev[:4]]}")

    # D7
    L = 4
    Ev = np.eye(3, dtype=int)
    lk = [(x, i) for x in itertools.product(range(L + 1), repeat=3) for i in range(3) if x[i] + 1 <= L]
    img = {tuple(2 * np.array(x) + Ev[i]) for x, i in lk}
    ok = True
    for x in itertools.product(range(L), repeat=3):
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            xa = np.array(x)
            pts = [2 * xa + Ev[i], 2 * (xa + Ev[i]) + Ev[j], 2 * (xa + Ev[j]) + Ev[i], 2 * xa + Ev[j]]
            ok &= all(np.all((p - 2 * xa >= 0) & (p - 2 * xa <= 2)) for p in pts)
            ok &= all(np.all((p >= 0) & (p <= 2 * L + 2)) for p in pts)
    rep.check("D7 iota injective; iota(dp) in 2x + Lambda_0; supports inside B_L", len(img) == len(lk) and ok)

    # D8
    ok = True
    for N in (2, 3, 5):
        for c1, c2 in [(0.1, 2.0), (0.02, 0.5), (0.5, 9.0), (1.0, 0.1)]:
            bs = beta_star(c1, c2); G0 = g0(N, c1, c2)
            ok &= bs < c1 and 1 - c2 * bs >= 0.5 - 1e-12
            for g in np.linspace(0.5 * G0, 2 * G0, 101):
                ok &= (6 * N / (g ** 4 * c_N(N)) <= bs * (1 + 1e-12)) == (g >= G0 * (1 - 1e-12))
    rep.check("D8 beta* < c1, 1 - c2 beta* >= 1/2, and g >= g0(N) <=> 6N/(g^4 c_N) <= beta*", ok)

    # D9
    ok = True
    for N in (2, 3, 4, 5):
        z = np.exp(2j * math.pi / N)
        U = haar_sun(10, N, rng)
        ok &= abs(z ** N - 1) < 1e-12 and np.allclose(np.linalg.det(z * np.eye(N)), 1)
        ok &= np.allclose(z * U, U * z)
    links, plaqs = box_geometry(3, 3, 2)
    li = {l: k for k, l in enumerate(links)}
    for N in (2, 3, 5):
        for _ in range(30):
            eta = rng.integers(0, N, len(links))
            Sig = rng.integers(0, N, len(plaqs))
            dS = np.zeros(len(links), int)
            for c, p in zip(Sig, plaqs):
                for l, o in plaq_boundary(p):
                    dS[li[l]] += c * o
            de = [sum(o * eta[li[l]] for l, o in plaq_boundary(p)) for p in plaqs]
            ok &= (eta @ dS - np.dot(Sig, de)) % N == 0
    rep.check("D9 zeta 1 is central with det 1; Stokes eta(dSigma) = <Sigma, delta eta> mod N", ok)

    # D10
    nb_ok, det = True, []
    links, plaqs = box_geometry(8, 8, 4)
    by_link = {}
    for p in plaqs:
        for l, _ in plaq_boundary(p):
            by_link.setdefault(l, set()).add(p)
    for r in range(1, 5):
        for s in range(1, 5):
            P = set().union(*(by_link[l] for l, _ in rect_loop((2, 2, 2), r, s)))
            nb_ok &= len(P) <= 8 * (r + s)
            if r == s:
                det.append(f"{r}x{s}: {len(P)} <= {8 * (r + s)}")
    rep.check("D10 |P(C)| <= 8(r+s)", nb_ok, "; ".join(det))
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
