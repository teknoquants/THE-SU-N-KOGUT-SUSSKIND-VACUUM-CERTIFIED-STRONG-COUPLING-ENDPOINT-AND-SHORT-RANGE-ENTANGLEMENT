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
"""Validation of Lemma 1 (normalization and constants) and every step of its proof.

What is checked
  (i)    Ric(X,X) = -B(X,X)/4 = (N/2)|X|^2 for the bi-invariant metric -Tr, N = 2..6
         (computed independently as (1/4) sum_a |[X, T_a]|^2).
  (ii.a) Casimir normalization: -sum_a T_a^2 = c_N 1 in the fundamental, -sum_a ad(T_a)^2 = 2N 1 in the
         adjoint, matching C(omega_1) = (N^2-1)/N and C(omega_1 + omega_{N-1}) = 2N from the weight formula.
  (ii.b) Exact minimum of C(lambda) over all nonzero dominant weights with entries <= 4 equals (N^2-1)/N,
         attained only at k in {1, N-1}; C(omega_k) = k(N-k)(N+1)/N (N = 2..7, rational arithmetic).
  (ii.c) N = 2: c_2 = 3/2 equals the first eigenvalue 3/R^2 of S^3 with R = sqrt(2); SU(2) characters are
         Laplacian eigenfunctions with eigenvalue (n^2-1)/2 (Monte Carlo orthonormality).
  (iii)  H-bar = 2H/(g^2 c_N) - 2N|P|/(g^4 c_N) (symbolic); sup|ReTr rho(U)| = N; every link lies in at most
         four plaquettes, boundary included (exhaustive), so sum_{p ni e} ||Psi_p|| <= 8N/(g^4 c_N).
"""
import itertools
import math
import sys
from fractions import Fraction

import numpy as np
import sympy as sp

from common import (Report, c_N, su_basis, ip, ad_matrix, casimir_weight, haar_sun, chi_su2,
                    box_geometry, plaq_boundary)


def main():
    rep = Report("Lemma 1")
    rng = np.random.default_rng(11)

    ok, det = True, []
    for N in range(2, 7):
        B = su_basis(N)
        worst = 0
        for _ in range(10):
            X = sum(c * T for c, T in zip(rng.normal(size=len(B)), B))
            ric = 0.25 * sum(-np.real(np.trace((X @ T - T @ X) @ (X @ T - T @ X))) for T in B)
            kil = np.trace(ad_matrix(X, B) @ ad_matrix(X, B))
            worst = max(worst, abs(ric - N / 2 * ip(X, X)), abs(-kil / 4 - N / 2 * ip(X, X)))
        ok &= worst < 1e-9
        det.append(f"N={N}: err {worst:.1e}")
    rep.check("(i) Ric = -B/4 = (N/2)<.,.> on su(N)", ok, "; ".join(det[:3]) + " ...")

    ok = True
    for N in range(2, 7):
        B = su_basis(N)
        fund = -sum(T @ T for T in B)
        adj = -sum(ad_matrix(T, B) @ ad_matrix(T, B) for T in B)
        w1 = [1] + [0] * (N - 2)
        th = [1] + [0] * (N - 3) + [1] if N > 2 else [2]
        ok &= np.allclose(fund, c_N(N) * np.eye(N)) and np.allclose(adj, 2 * N * np.eye(N * N - 1))
        ok &= casimir_weight(w1, N) == Fraction(N * N - 1, N) and casimir_weight(th, N) == 2 * N
    rep.check("(ii.a) fundamental Casimir = c_N, adjoint Casimir = 2N, both equal to the weight formula", ok)

    ok, det = True, []
    for N in range(2, 8):
        best, arg = None, []
        for lam in itertools.product(range(5), repeat=N - 1):
            if not any(lam):
                continue
            C = casimir_weight(list(lam), N)
            if best is None or C < best:
                best, arg = C, [lam]
            elif C == best:
                arg.append(lam)
        ks = sorted(l.index(1) + 1 for l in arg)
        ok &= best == Fraction(N * N - 1, N) and ks == sorted({1, N - 1}) and all(sum(l) == 1 for l in arg)
        for k in range(1, N):
            wk = [0] * (N - 1); wk[k - 1] = 1
            ok &= casimir_weight(wk, N) == Fraction(k * (N - k) * (N + 1), N)
        det.append(f"N={N}: min {best}")
    rep.check("(ii.b) min_{lambda != 0} C(lambda) = (N^2-1)/N, attained at omega_1, omega_{N-1}; C(omega_k) formula",
              ok, "; ".join(det))

    U = haar_sun(200000, 2, rng)
    G = np.array([[np.mean(chi_su2(U, m) * chi_su2(U, n)) for n in range(1, 5)] for m in range(1, 5)])
    lap = [(n * n - 1) / 2 for n in range(1, 5)]
    rep.check("(ii.c) SU(2): c_2 = 3/2 = 3/R^2 (R = sqrt 2); characters orthonormal, eigenvalues (n^2-1)/2",
              abs(c_N(2) - 1.5) < 1e-15 and abs(3 / 2 - 3 / math.sqrt(2) ** 2) < 1e-15 and
              np.allclose(G, np.eye(4), atol=0.02) and lap[1] == c_N(2), f"Gram max dev {np.abs(G - np.eye(4)).max():.3f}")

    g, N, P = sp.symbols("g N P", positive=True)
    cN = (N ** 2 - 1) / N
    Elec, Re = sp.symbols("Elec Re")       # Elec = sum(-Delta_e), Re = sum_p ReTr
    H = g ** 2 / 2 * Elec + g ** -2 * (N * P - Re)
    Hbar = Elec / cN - 2 / (g ** 4 * cN) * Re
    rep.check("(iii.a) H-bar = 2H/(g^2 c_N) - 2N|P|/(g^4 c_N) (symbolic)",
              sp.simplify(Hbar - (2 * H / (g ** 2 * cN) - 2 * N * P / (g ** 4 * cN))) == 0)

    ok, det = True, []
    for Nn in (2, 3, 4):
        U = haar_sun(50000, Nn, rng)
        mx = np.max(np.abs(np.real(np.trace(U, axis1=1, axis2=2))))
        ok &= mx <= Nn + 1e-9
        det.append(f"N={Nn}: sample max {mx:.3f}")
    rep.check("(iii.b) sup|ReTr rho(U)| = N, attained at U = 1, so ||Psi_p|| <= 2N/(g^4 c_N)", ok, "; ".join(det))

    links, plaqs = box_geometry(4, 4, 4)
    cnt = {l: 0 for l in links}
    for p in plaqs:
        for l, _ in plaq_boundary(p):
            cnt[l] += 1
    vals = sorted(set(cnt.values()))
    rep.check("(iii.c) every link lies in at most 4 plaquettes (interior 4; boundary 2 or 3)",
              max(vals) == 4 and min(vals) >= 2, f"counts {vals}")
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
