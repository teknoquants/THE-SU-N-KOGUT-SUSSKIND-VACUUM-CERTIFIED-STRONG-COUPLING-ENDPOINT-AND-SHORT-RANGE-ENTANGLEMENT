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
"""Validation of Proposition 8 (conditional area law) and every step of its proof.

What is checked
  (a) Every plaquette has at most 12 neighbours in the plaquette adjacency graph (exhaustive, boundary included).
  (b) The number of connected plaquette sets of size n containing a given plaquette is <= (12e)^n (exact
      enumeration, n <= 5).
  (c) At most 8(r+s) plaquettes contain a link of C (exhaustive, 1 <= r, s <= 5).
  (d) Inequality chain on a grid q in (0, 1/2], 1 <= r, s <= 40:
        8(r+s) sum_{n >= rs} q^n <= 16(r+s) q^{rs} <= M e^{-sigma rs},  sigma = ln(1/q)/2,  M = 32/(e sigma),
      and the calculus bound x q^{x/2} <= 2/(e ln q^-1) (attained at x = 2/ln q^-1).
  (e) End-to-end on a truncated expansion: all connected sets X (|X| <= 4) that contain a plaquette touching C,
      with adversarial weights |w_X| = (K g^-4)^{|X|} on the sets allowed by Corollary 7 (|X| >= rs): the sum
      obeys the bound of Proposition 8.
"""
import math
import sys

import numpy as np

from common import Report, box_geometry, plaq_boundary, adjacency, count_animals, rect_loop


def main():
    rep = Report("Proposition 8")
    links, plaqs = box_geometry(6, 6, 6)
    nb, by_link = adjacency(plaqs)
    deg = max(len(v) for v in nb.values())
    rep.check("(a) plaquette adjacency degree <= 12", deg == 12, f"max degree {deg}")

    p0 = ((3, 3, 3), 0, 1)
    counts = count_animals(nb, p0, 5)
    ok = all(c <= (12 * math.e) ** n for n, c in enumerate(counts, start=1))
    rep.check("(b) connected sets of size n through a plaquette <= (12e)^n", ok,
              ", ".join(f"n={n}: {c} <= {(12 * math.e) ** n:.0f}" for n, c in enumerate(counts, start=1)))

    ok = True
    for r in range(1, 6):
        for s in range(1, 6):
            P = set().union(*(set(by_link[l]) for l, _ in rect_loop((0, 0, 3), r, s)))
            ok &= len(P) <= 8 * (r + s)
    rep.check("(c) at most 8(r+s) plaquettes contain a link of C", ok)

    ok, worst = True, 0.0
    for q in np.linspace(1e-4, 0.5, 400):
        sig = 0.5 * math.log(1 / q)
        M = 32 / (math.e * sig)
        for r in range(1, 41):
            for s in range(1, 41):
                A = r * s
                t1 = 8 * (r + s) * q ** A / (1 - q)
                t2 = 16 * (r + s) * q ** A
                t3 = M * math.exp(-sig * A)
                ok &= t1 <= t2 * (1 + 1e-12) and t2 <= t3 * (1 + 1e-12)
                if t3 > 1e-300:
                    worst = max(worst, t2 / t3)
        xs = np.linspace(0, 400, 4001)
        ok &= np.max(xs * q ** (xs / 2)) <= 2 / (math.e * math.log(1 / q)) * (1 + 1e-9)
    rep.check("(d) 8(r+s) sum q^n <= 16(r+s) q^{rs} <= M e^{-sigma rs};  x q^{x/2} <= 2/(e ln q^-1)", ok,
              f"max ratio middle/right {worst:.3f}")

    # (e)
    ok, det = True, []
    links, plaqs = box_geometry(5, 5, 4)
    nb, by_link = adjacency(plaqs)
    for (r, s) in [(1, 1), (2, 1), (2, 2)]:
        C = rect_loop((1, 1, 2), r, s)
        touch = set().union(*(set(by_link[l]) for l, _ in C))
        level = {frozenset([p]) for p in touch}
        sets = set(level)
        for _ in range(3):
            nxt = set()
            for S in level:
                for q in set().union(*(nb[x] for x in S)) - S:
                    nxt.add(S | {q})
            level = nxt
            sets |= level
        for Kg in (1 / (24 * math.e), 1 / (60 * math.e)):
            q = 12 * math.e * Kg
            sig = 0.5 * math.log(1 / q)
            M = 32 / (math.e * sig)
            total = sum(Kg ** len(X) for X in sets if len(X) >= r * s)
            ok &= total <= M * math.exp(-sig * r * s)
            det.append(f"{r}x{s}, q={q:.2f}: {total:.2e} <= {M * math.exp(-sig * r * s):.2e}")
    rep.check("(e) truncated expansion with adversarial weights obeys the bound", ok, "; ".join(det))
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
