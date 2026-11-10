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
"""Validation of Proposition 6 (area count) and Corollary 7 (expansion terms), and every step of the proofs.

What is checked
  (a) Cellular structure: d o d = 0 from 3-cells to links in the open cubic complex (exhaustive).
  (b) Projection pi onto the plane of C is a chain map: d pi_* = pi_* d on random Z_N 2-chains (N = 2, 3, 5).
  (c) A planar box has no nonzero 2-cycles: d is injective on its 2-chains mod p (p = 2, 3, 5), so a planar 2-chain
      is determined by its boundary; the chain with boundary C is the filled rectangle with coefficients 1.
  (d) For 3D chains Sigma = filled rectangle + d(random 3-chain), so that dSigma = C: the stacked coefficient sums
      equal 1 over every unit square, and |supp Sigma| >= rs.
  (e) Exact minimum support (integer programming over a padded 3D box) equals rs for Z_2 and Z_3,
      1 <= r, s <= 3.
  (f) Corollary 7, exactly, in the Z_N gauge model: the vacuum expectation <Omega_L, W(C) Omega_L> of the full
      ground state (vacuum normalization included) vanishes to order lambda^{rs - 1} and is of order lambda^{rs}
      for rs = 1, 2 (N = 2, 3) and rs = 4 (N = 2, 2x2 box): log-log slope at small coupling.
"""
import math
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

from common import (Report, box_geometry, plaq_boundary, rect_loop, rect_plaqs, add, E3,
                    rank_mod_p, min_chain_support, ZNModel)


def cube_boundary(v):
    """d[v; e0,e1,e2] = sum_k (-1)^k ([v + e_k; omit k] - [v; omit k])."""
    out = []
    for k in range(3):
        i, j = [m for m in range(3) if m != k]
        out.append(((add(v, E3[k]), i, j), (-1) ** k))
        out.append(((v, i, j), -(-1) ** k))
    return out


def d2(chain, li, nl):
    out = np.zeros(nl, dtype=np.int64)
    for p, c in chain.items():
        for l, o in plaq_boundary(p):
            out[li[l]] += c * o
    return out


def main():
    rep = Report("Proposition 6 / Corollary 7")
    rng = np.random.default_rng(66)
    nx, ny, nz = 4, 4, 3
    links, plaqs = box_geometry(nx, ny, nz)
    li = {l: k for k, l in enumerate(links)}
    cubes = [(x, y, z) for x in range(nx) for y in range(ny) for z in range(nz)]

    ok = True
    for v in cubes:
        ch = {}
        for p, c in cube_boundary(v):
            ch[p] = ch.get(p, 0) + c
        ok &= not d2(ch, li, len(links)).any()
    rep.check("(a) d o d = 0 on 3-cells (orientation conventions consistent)", ok, f"{len(cubes)} cubes")

    # (b)
    plinks, psq = box_geometry(nx, ny, 0)
    pli = {l: k for k, l in enumerate(plinks)}

    def proj_chain(ch):
        out = {}
        for (v, i, j), c in ch.items():
            if (i, j) == (0, 1):
                q = ((v[0], v[1], 0), 0, 1)
                out[q] = out.get(q, 0) + c
        return out

    def proj_links(vec):
        out = np.zeros(len(plinks), dtype=np.int64)
        for (v, i), k in li.items():
            if i < 2:
                out[pli[((v[0], v[1], 0), i)]] += vec[k]
        return out
    ok = True
    for N in (2, 3, 5):
        for _ in range(100):
            ch = {p: int(rng.integers(0, N)) for p in plaqs if rng.random() < 0.3}
            lhs = d2(proj_chain(ch), pli, len(plinks)) % N
            rhs = proj_links(d2(ch, li, len(links))) % N
            ok &= np.array_equal(lhs, rhs)
    rep.check("(b) projection is a chain map: d pi_* = pi_* d", ok)

    # (c)
    A = np.zeros((len(plinks), len(psq)), dtype=np.int64)
    for c, p in enumerate(psq):
        for l, o in plaq_boundary(p):
            A[pli[l], c] += o
    ok = all(rank_mod_p(A, p_) == len(psq) for p_ in (2, 3, 5))
    rep.check("(c) planar box: d injective on 2-chains mod p (no 2-cycles)", ok, f"{len(psq)} squares")

    # (d)
    ok, det = True, []
    for N in (2, 3):
        for (r, s) in [(1, 1), (2, 1), (2, 2), (3, 2)]:
            for _ in range(25):
                ch = {}
                for p in rect_plaqs((0, 0, 1), r, s):
                    ch[p] = 1
                for v in cubes:
                    c = int(rng.integers(0, N)) if rng.random() < 0.4 else 0
                    for p, o in cube_boundary(v):
                        ch[p] = (ch.get(p, 0) + c * o) % N
                ch = {p: c for p, c in ch.items() if c % N}
                bd = d2(ch, li, len(links)) % N
                Cv = np.zeros(len(links), dtype=np.int64)
                for l, o in rect_loop((0, 0, 1), r, s):
                    Cv[li[l]] += o
                ok &= np.array_equal(bd, Cv % N)
                pj = proj_chain(ch)
                for a in range(nx):
                    for b in range(ny):
                        inside = a < r and b < s
                        ok &= (pj.get(((a, b, 0), 0, 1), 0) % N) == (1 if inside else 0)
                ok &= len(ch) >= r * s
        det.append(f"N={N}")
    rep.check("(d) dSigma = C  =>  stacked sums are 1 on each square and |supp Sigma| >= rs", ok, ", ".join(det))

    # (e)
    ok, det = True, []
    for N in (2, 3):
        for r in (1, 2, 3):
            for s in (1, 2, 3):
                if s < r:
                    continue
                m, st = min_chain_support(r, s, N)
                ok &= st == 0 and m == r * s
                det.append(f"Z{N} {r}x{s}:{m}")
    rep.check("(e) exact minimum support of a Z_N 2-chain with boundary C equals rs (MILP)", ok, " ".join(det))

    # (f)
    ok, det = True, []
    for N, nxx, nyy, loops in [(2, 2, 1, [(1, 1), (2, 1)]), (3, 2, 1, [(1, 1), (2, 1)]), (2, 2, 2, [(1, 1), (2, 1), (2, 2)])]:
        M = ZNModel(N, nxx, nyy)
        Psi = sum(M.psi(p) for p in M.plaqs)
        lams = [2e-3, 4e-3]
        vals = {rs: [] for rs in loops}
        for lam in lams:
            H = M.Hel + lam * np.diag(Psi)
            e, v = eigsh(H, k=1, which="SA", tol=1e-14, v0=M.vac.copy())
            Om = v[:, 0]
            for (r, s) in loops:
                vals[(r, s)].append(np.real(np.vdot(Om, M.wilson(rect_loop((0, 0, 0), r, s)) * Om)))
        for (r, s) in loops:
            slope = math.log(abs(vals[(r, s)][1] / vals[(r, s)][0])) / math.log(2)
            ok &= abs(slope - r * s) < 0.05
            det.append(f"N={N} {r}x{s}: slope {slope:.3f}")
    rep.check("(f) exact Z_N vacuum: <W(C)> = O(lambda^{rs}) (leading order exactly rs)", ok, "; ".join(det))
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
