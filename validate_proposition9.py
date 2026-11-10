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
"""Validation of Proposition 9 (Schur-test criterion) and every step of its proof.

What is checked
  (a) Schur test for symmetric block matrices: ||H|| <= sup_e sum_f ||H_ef|| (100 random block matrices with
      3x3 blocks, the dimension of su(2), on the link graph of a box).
  (b) The row sum r_L of Definition 11, computed from a polymer decomposition H = sum_X Hess phi_X, dominates
      the operator norm of the full Hessian.
  (c) A per-polymer bound alone does not suffice: with every ||Hess phi_X|| <= eps but m polymers on the same
      links, ||H|| grows linearly in m, while the row sum tracks it.
  (d) Ric = (N/2) 1 and ||Hess h|| <= b  =>  Ric - 2 Hess h >= (N/2 - 2b) 1, positive iff b < N/4
      (random symmetric Hessians, N = 2..6).
"""
import sys

import numpy as np

from common import Report, box_geometry, plaq_boundary


def sym_block(n_links, d, pairs, rng, scale=1.0):
    H = np.zeros((n_links * d, n_links * d))
    for a, b in pairs:
        B = rng.normal(size=(d, d)) * scale
        if a == b:
            B = B + B.T
        H[a * d:(a + 1) * d, b * d:(b + 1) * d] += B
        if a != b:
            H[b * d:(b + 1) * d, a * d:(a + 1) * d] += B.T
    return H


def row_sum(H, n, d):
    return max(sum(np.linalg.norm(H[e * d:(e + 1) * d, f * d:(f + 1) * d], 2) for f in range(n)) for e in range(n))


def main():
    rep = Report("Proposition 9")
    rng = np.random.default_rng(99)
    links, plaqs = box_geometry(2, 2, 1)
    li = {l: k for k, l in enumerate(links)}
    n, d = len(links), 3
    pairs = set()
    for p in plaqs:
        ls = [li[l] for l, _ in plaq_boundary(p)]
        for a in ls:
            for b in ls:
                if a <= b:
                    pairs.add((a, b))
    pairs = sorted(pairs)

    ok, worst = True, 0
    for _ in range(100):
        H = sym_block(n, d, pairs, rng)
        nrm, rs = np.linalg.norm(H, 2), row_sum(H, n, d)
        ok &= nrm <= rs * (1 + 1e-12)
        worst = max(worst, nrm / rs)
    rep.check("(a) Schur test ||H|| <= sup_e sum_f ||H_ef||", ok, f"max ratio {worst:.3f}")

    ok = True
    for _ in range(20):
        polys = [rng.choice(n, size=rng.integers(1, 5), replace=False) for _ in range(20)]
        H = np.zeros((n * d, n * d))
        rL = np.zeros(n)
        for X in polys:
            HX = sym_block(n, d, [(a, b) for a in X for b in X if a <= b], rng, 0.3)
            H += HX
            for e in range(n):
                rL[e] += sum(np.linalg.norm(HX[e * d:(e + 1) * d, f * d:(f + 1) * d], 2) for f in range(n))
        ok &= np.linalg.norm(H, 2) <= rL.max() * (1 + 1e-12)
    rep.check("(b) polymer row sum r_L dominates ||Hess h||", ok)

    eps, det, ok = 0.05, [], True
    J = np.zeros((n * d, n * d)); J[0:d, 0:d] = np.eye(d); J[d:2 * d, d:2 * d] = np.eye(d)
    J[0:d, d:2 * d] = np.eye(d); J[d:2 * d, 0:d] = np.eye(d)
    J *= eps / np.linalg.norm(J, 2)
    prev = 0
    for m in (1, 10, 100):
        H = m * J
        nrm = np.linalg.norm(H, 2)
        ok &= abs(nrm - m * eps) < 1e-12 and nrm <= row_sum(H, n, d) + 1e-12 and nrm > prev
        prev = nrm
        det.append(f"m={m}: ||H|| = {nrm:.2f}")
    rep.check("(c) per-polymer bound eps does not bound ||H|| (grows like m eps); row sum does", ok, "; ".join(det))

    ok = True
    for N in range(2, 7):
        for _ in range(100):
            b = rng.uniform(0, N / 2)
            Hh = rng.normal(size=(8, 8)); Hh = Hh + Hh.T
            Hh *= b / np.linalg.norm(Hh, 2) * rng.uniform(0.2, 1.0)
            m = np.linalg.eigvalsh(N / 2 * np.eye(8) - 2 * Hh).min()
            ok &= m >= N / 2 - 2 * b - 1e-12
    rep.check("(d) Ric - 2 Hess h >= N/2 - 2b; kappa > 0 iff b < N/4", ok)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
