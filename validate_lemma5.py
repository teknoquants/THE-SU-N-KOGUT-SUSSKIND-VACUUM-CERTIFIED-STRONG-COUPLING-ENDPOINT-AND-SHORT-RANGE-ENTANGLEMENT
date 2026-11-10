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
"""Validation of Lemma 5 (center selection rule) and every step of its proof.

What is checked
  (a) T_eta preserves Laplacian eigenspaces: rho(zeta U) = zeta^k rho(U) for irreducible rho (Schur): SU(2)
      characters chi_n(-U) = (-1)^(n-1) chi_n(U); SU(3) fundamental (zeta), adjoint (1), Sym^2 (zeta^2).
      Hence T_eta commutes with every -Laplacian_e and with every function of the electric Hamiltonian.
  (b) T_eta Psi_p T_eta^-1 = Psi_p iff (delta eta)(p) = 0 mod N; T_eta W(C) = zeta^{eta(C)} W(C)
      (SU(2), SU(3), SU(4) lattice configurations).
  (c) Character step (exact linear algebra over Z_p, p = 2, 3, 5): for 600 random plaquette sets X and loops C,
      C is not in span{dp : p in X}  <=>  there is eta with delta eta = 0 on X and eta(C) != 0; the constructed
      eta indeed fixes every Psi_p, p in X, and rotates W(C) by zeta^{eta(C)} != 1.
  (d) The conclusion, exactly, in the Z_N gauge model (N = 2, 3) with resolvents of the electric Hamiltonian
      inside F and F': matrix elements vanish to machine precision when C does not bound in X, and are nonzero
      when it does.
  (e) The conclusion for SU(2) and SU(3) by Haar Monte Carlo (F, F' polynomial in Psi_p): vanishing when C does
      not bound in X; the filled r x s rectangle gives the exact value 2^(-rs) for SU(2) (real characters) and
      (2N)^(-rs) for SU(3) (only the conjugate half of each ReTr contributes; each character convolution
      divides by N).
"""
import math
import sys

import numpy as np

from common import (Report, haar_sun, chi_su2, box_geometry, plaq_boundary, rect_loop, rect_plaqs, holonomy,
                    rank_mod_p, nullspace_mod_p, ZNModel)


def main():
    rep = Report("Lemma 5")
    rng = np.random.default_rng(55)

    # (a)
    U = haar_sun(5000, 2, rng)
    ok = all(np.allclose(chi_su2(-U, n), (-1) ** (n - 1) * chi_su2(U, n)) for n in range(1, 7))
    U3 = haar_sun(200, 3, rng)
    z = np.exp(2j * math.pi / 3)
    from common import su_basis, ad_matrix
    B = su_basis(3)

    def adj(V):
        return np.array([[np.real(-np.trace(Ta @ V @ Tb @ np.conj(V.T))) for Tb in B] for Ta in B])

    def sym2(V):
        return np.kron(V, V)            # contains Sym^2 + Lambda^2, both of triality 2
    ok3 = all(np.allclose(adj(z * V), adj(V)) and np.allclose(sym2(z * V), z ** 2 * sym2(V)) for V in U3[:20])
    rep.check("(a) rho(zeta U) = zeta^k rho(U): SU(2) chi_n, SU(3) fundamental/adjoint/Sym^2", ok and ok3)

    # (b)
    links, plaqs = box_geometry(3, 3, 1)
    li = {l: k for k, l in enumerate(links)}
    loop = rect_loop((0, 0, 0), 2, 3)
    ok = True
    for N in (2, 3, 4):
        zN = np.exp(2j * math.pi / N)
        Ucf = haar_sun(len(links), N, rng)[None]
        for _ in range(20):
            eta = rng.integers(0, N, len(links))
            Ut = Ucf * (zN ** eta)[None, :, None, None]
            for p in plaqs:
                de = sum(o * eta[li[l]] for l, o in plaq_boundary(p)) % N
                a = np.real(np.trace(holonomy(plaq_boundary(p), Ucf, li)[0]))
                b = np.real(np.trace(holonomy(plaq_boundary(p), Ut, li)[0]))
                ok &= (abs(a - b) < 1e-10) == (de == 0)
            eC = sum(o * eta[li[l]] for l, o in loop)
            ok &= np.allclose(np.trace(holonomy(loop, Ut, li)[0]), zN ** eC * np.trace(holonomy(loop, Ucf, li)[0]))
    rep.check("(b) T_eta fixes Psi_p iff (delta eta)(p) = 0 mod N; T_eta W(C) = zeta^{eta(C)} W(C)", ok)

    # (c)
    links, plaqs = box_geometry(2, 2, 2)
    li = {l: k for k, l in enumerate(links)}
    loops = [rect_loop((0, 0, 0), 1, 1), rect_loop((0, 0, 1), 2, 1), rect_loop((0, 0, 0), 2, 2)]
    ok, n_in, n_out = True, 0, 0
    for p_ in (2, 3, 5):
        for _ in range(200):
            X = [plaqs[k] for k in rng.choice(len(plaqs), rng.integers(1, 14), replace=False)]
            A = np.zeros((len(links), len(X)), dtype=np.int64)
            for c, p in enumerate(X):
                for l, o in plaq_boundary(p):
                    A[li[l], c] += o
            loop = loops[rng.integers(3)]
            Cv = np.zeros(len(links), dtype=np.int64)
            for l, o in loop:
                Cv[li[l]] += o
            in_span = rank_mod_p(np.column_stack([A, Cv]), p_) == rank_mod_p(A, p_)
            cands = [e for e in nullspace_mod_p(A.T, p_) if (e @ Cv) % p_]
            ok &= in_span == (len(cands) == 0)
            if in_span:
                n_in += 1
            else:
                n_out += 1
                eta = cands[0]
                zN = np.exp(2j * math.pi / p_)
                Ucf = haar_sun(len(links), p_, rng)[None]
                Ut = Ucf * (zN ** eta)[None, :, None, None]
                for p in X:
                    ok &= abs(np.trace(holonomy(plaq_boundary(p), Ucf, li)[0]) -
                              np.trace(holonomy(plaq_boundary(p), Ut, li)[0])) < 1e-10
                ok &= abs(zN ** int(eta @ Cv) - 1) > 1e-6
    rep.check("(c) C not in span{dp : p in X} mod N  <=>  a separating eta exists (and it works)", ok,
              f"{n_in} bounding, {n_out} non-bounding cases")

    # (d)
    ok, det = True, []
    for N in (2, 3):
        M = ZNModel(N, 2, 1)
        R = np.linalg.inv(1.3 * np.eye(M.dim) + M.Hel)
        pl, pr = M.plaqs
        Cl = plaq_boundary(pl)
        Cb = rect_loop((0, 0, 0), 2, 1)
        D = lambda p: np.diag(M.psi(p))
        cases = [
            ("dp_l, X={p_r}", Cl, R @ D(pr) @ R @ D(pr), R @ D(pr), False),
            ("dp_l, X={p_l}", Cl, R @ D(pl), np.eye(M.dim), True),
            ("rect, X={p_l}", Cb, R @ D(pl) @ R @ D(pl) @ R @ D(pl), R @ D(pl), False),
            ("rect, X={p_r}", Cb, R @ D(pr), R @ D(pr) @ R, False),
            ("rect, X={p_l,p_r}", Cb, R @ D(pl) @ R @ D(pr), np.eye(M.dim), True),
        ]
        for name, C, F, Fp, expect in cases:
            val = np.vdot(F @ M.vac, M.wilson(C) * (Fp @ M.vac))
            ok &= (abs(val) > 1e-6) == expect and (expect or abs(val) < 1e-13)
            det.append(f"N={N} {name}: {abs(val):.1e}")
    rep.check("(d) exact Z_N model with resolvents: <F Omega0, W F' Omega0> = 0 unless C bounds in X", ok,
              "; ".join(det[:5]))

    # (e)
    links, plaqs = box_geometry(2, 2, 0)
    li = {l: k for k, l in enumerate(links)}
    ok, det = True, []
    for N in (2, 3):
        S, batches = 50000, 4
        tests = []
        for (r, s) in [(1, 1), (1, 2), (2, 2)]:
            loop = rect_loop((0, 0, 0), r, s)
            P = rect_plaqs((0, 0, 0), r, s)
            tests.append((f"{r}x{s} filled", loop, P, (2.0 if N == 2 else 2.0 * N) ** (-r * s)))
            tests.append((f"{r}x{s} minus one", loop, P[:-1], 0.0))
        acc = {t[0]: [] for t in tests}
        for _ in range(batches):
            U = haar_sun(S * len(links), N, rng).reshape(S, len(links), N, N)
            tr = {p: np.real(np.trace(holonomy(plaq_boundary(p), U, li), axis1=-2, axis2=-1)) for p in plaqs}
            for name, loop, P, _ in tests:
                Wv = np.trace(holonomy(loop, U, li), axis1=-2, axis2=-1) / N
                acc[name].append(Wv * np.prod([tr[p] for p in P], axis=0) if P else Wv)
        for name, loop, P, exact in tests:
            v = np.concatenate(acc[name])
            m, se = v.mean(), v.std() / math.sqrt(len(v))
            if exact is None:
                good = abs(m) > 8 * se
            else:
                good = abs(m - exact) < 5 * se + 1e-12
            ok &= good
            det.append(f"SU({N}) {name}: {m.real:+.4f}±{se:.4f}")
    rep.check("(e) Haar MC: zero unless C bounds in X; filled rectangle exact (2^-rs, 6^-rs)", ok, "; ".join(det))
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
