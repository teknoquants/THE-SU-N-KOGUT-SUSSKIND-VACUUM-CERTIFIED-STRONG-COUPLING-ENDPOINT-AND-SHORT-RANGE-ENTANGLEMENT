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
"""Shared constants, group tools, lattice geometry and a small check recorder.

Companion code to "A Uniform Strong-Coupling Spectral Gap and a Center Selection Rule for the SU(N)
Kogut-Susskind Lattice Hamiltonian" (R. Cooperman and A.E. Petrunina V.V., XISOSIS, Vienna, 10 November 2026).
"""
import math
from fractions import Fraction

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


# ---------------------------------------------------------------- constants (Definitions 1, 8)
def c_N(N):
    """Smallest positive eigenvalue of -Laplacian on SU(N) with metric -Tr(XY)."""
    return (N * N - 1) / N


def beta_star(c1, c2):
    return min(c1 / 2.0, 1.0 / (2.0 * c2))


def g0(N, c1, c2):
    """g0(N) = (6N / (c_N beta*))^(1/4)  (Definition 8)."""
    return (6 * N / (c_N(N) * beta_star(c1, c2))) ** 0.25


# ---------------------------------------------------------------- su(N) and SU(N)
def su_basis(N):
    """Orthonormal basis of su(N) for <X,Y> = -Tr(XY): T_a = i H_a, H_a Hermitian, Tr(H_a H_b) = delta_ab."""
    Hs = []
    for j in range(N):
        for k in range(j + 1, N):
            A = np.zeros((N, N), complex); A[j, k] = A[k, j] = 1 / math.sqrt(2); Hs.append(A)
            B = np.zeros((N, N), complex); B[j, k] = -1j / math.sqrt(2); B[k, j] = 1j / math.sqrt(2); Hs.append(B)
    for m in range(1, N):
        D = np.zeros((N, N), complex)
        D[:m, :m] = np.eye(m); D[m, m] = -m
        D /= math.sqrt(m * m + m)
        Hs.append(D)
    return [1j * H for H in Hs]


def ip(X, Y):
    return float(np.real(-np.trace(X @ Y)))


def coords(X, basis):
    return np.array([ip(X, T) for T in basis])


def ad_matrix(X, basis):
    return np.array([coords(X @ T - T @ X, basis) for T in basis]).T


def haar_su2(n, rng):
    """n Haar-random SU(2) matrices, from uniform unit quaternions."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    a, b, c, d = q.T
    U = np.empty((n, 2, 2), dtype=complex)
    U[:, 0, 0] = a + 1j * b
    U[:, 0, 1] = c + 1j * d
    U[:, 1, 0] = -c + 1j * d
    U[:, 1, 1] = a - 1j * b
    return U


def haar_sun(n, N, rng):
    """n Haar-random SU(N) matrices via QR of complex Ginibre matrices."""
    if N == 2:
        return haar_su2(n, rng)
    X = (rng.normal(size=(n, N, N)) + 1j * rng.normal(size=(n, N, N))) / np.sqrt(2)
    Q, R = np.linalg.qr(X)
    d = np.diagonal(R, axis1=1, axis2=2)
    Q = Q * (d / np.abs(d))[:, None, :]
    det = np.linalg.det(Q)
    Q[:, :, 0] /= det[:, None]
    return Q


def chi_su2(U, n):
    """Character of the n-dimensional irrep of SU(2) on a batch of matrices."""
    half_tr = np.real(np.trace(U, axis1=1, axis2=2)) / 2.0
    theta = np.arccos(np.clip(half_tr, -1.0, 1.0))
    s = np.sin(theta)
    small = np.abs(s) < 1e-9
    val = np.sin(n * theta) / np.where(small, 1.0, s)
    val = np.where(small & (half_tr > 0), n, val)
    val = np.where(small & (half_tr < 0), n * (-1) ** (n - 1), val)
    return val


def casimir_weight(lam, N):
    """C(lambda) = sum_ij lam_i Ainv_ij (lam_j + 2), Ainv_ij = min(i,j) - ij/N (exact)."""
    C = Fraction(0)
    for i in range(1, N):
        for j in range(1, N):
            C += lam[i - 1] * (Fraction(min(i, j)) - Fraction(i * j, N)) * (lam[j - 1] + 2)
    return C


# ---------------------------------------------------------------- cubic lattice (Definition 2)
E3 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


def add(x, v, s=1):
    return tuple(a + s * b for a, b in zip(x, v))


def box_geometry(nx, ny, nz):
    """Links (v, i) and plaquettes (v, i, j), i < j, of the open box {0..nx} x {0..ny} x {0..nz}."""
    verts = [(x, y, z) for x in range(nx + 1) for y in range(ny + 1) for z in range(nz + 1)]
    inside = lambda v: 0 <= v[0] <= nx and 0 <= v[1] <= ny and 0 <= v[2] <= nz
    links = [(v, i) for v in verts for i in range(3) if inside(add(v, E3[i]))]
    plaqs = [(v, i, j) for v in verts for i in range(3) for j in range(i + 1, 3)
             if inside(add(add(v, E3[i]), E3[j]))]
    return links, plaqs


def plaq_boundary(p):
    """Oriented boundary of plaquette (v,i,j): v -> v+e_i -> v+e_i+e_j -> v+e_j -> v."""
    v, i, j = p
    return [((v, i), +1), ((add(v, E3[i]), j), +1), ((add(v, E3[j]), i), -1), ((v, j), -1)]


def rect_loop(x0, r, s, i=0, j=1):
    """Oriented links of the r x s rectangle in plane (i,j) with lower corner x0, counter-clockwise."""
    out, v = [], x0
    for _ in range(r):
        out.append(((v, i), +1)); v = add(v, E3[i])
    for _ in range(s):
        out.append(((v, j), +1)); v = add(v, E3[j])
    for _ in range(r):
        v = add(v, E3[i], -1); out.append(((v, i), -1))
    for _ in range(s):
        v = add(v, E3[j], -1); out.append(((v, j), -1))
    return out


def rect_plaqs(x0, r, s, i=0, j=1):
    return [(add(add(x0, E3[i], a), E3[j], b), i, j) for a in range(r) for b in range(s)]


def holonomy(path, U, li):
    """Ordered product along an oriented link path. U: array (..., n_links, N, N)."""
    M = None
    for l, o in path:
        A = U[..., li[l], :, :]
        if o < 0:
            A = np.conj(np.swapaxes(A, -1, -2))
        M = A if M is None else M @ A
    return M


def adjacency(plaqs):
    by_link = {}
    for p in plaqs:
        for l, _ in plaq_boundary(p):
            by_link.setdefault(l, []).append(p)
    nb = {p: set() for p in plaqs}
    for ps in by_link.values():
        for a in ps:
            for b in ps:
                if a != b:
                    nb[a].add(b)
    return nb, by_link


def count_animals(nb, p0, nmax):
    level = {frozenset([p0])}
    counts = [1]
    for _ in range(2, nmax + 1):
        nxt = set()
        for S in level:
            frontier = set().union(*(nb[q] for q in S)) - S
            for q in frontier:
                nxt.add(S | {q})
        level = nxt
        counts.append(len(level))
    return counts


def min_chain_support(r, s, N, depth=2):
    """Minimum support of a Z_N 2-chain with boundary the r x s rectangle, inside a padded 3D box (exact MILP)."""
    nx, ny, nz = r + 2, s + 2, depth
    links, plaqs = box_geometry(nx, ny, nz)
    li = {l: k for k, l in enumerate(links)}
    target = np.zeros(len(links), dtype=int)
    for l, o in rect_loop((1, 1, 1), r, s):
        target[li[l]] += o
    target %= N
    P, Lk = len(plaqs), len(links)
    nv = 2 * P + Lk
    A_eq = np.zeros((Lk, nv))
    for pi, p in enumerate(plaqs):
        for l, o in plaq_boundary(p):
            A_eq[li[l], pi] += o
    for l in range(Lk):
        A_eq[l, 2 * P + l] = -N
    A_ub = np.zeros((P, nv))
    for pi in range(P):
        A_ub[pi, pi] = 1
        A_ub[pi, P + pi] = -(N - 1)
    c = np.concatenate([np.zeros(P), np.ones(P), np.zeros(Lk)])
    lb = np.concatenate([np.zeros(P), np.zeros(P), -8 * np.ones(Lk)])
    ub = np.concatenate([(N - 1) * np.ones(P), np.ones(P), 8 * np.ones(Lk)])
    res = milp(c, integrality=np.ones(nv),
               constraints=[LinearConstraint(A_eq, target, target), LinearConstraint(A_ub, -np.inf, 0)],
               bounds=Bounds(lb, ub))
    return int(round(res.fun)), res.status


# ---------------------------------------------------------------- linear algebra mod a prime
def rank_mod_p(A, p):
    A = np.array(A, dtype=np.int64) % p
    r = 0
    rows, cols = A.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c] % p), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        A[r] = (A[r] * pow(int(A[r, c]), -1, p)) % p
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] = (A[i] - A[i, c] * A[r]) % p
        r += 1
        if r == rows:
            break
    return r


def nullspace_mod_p(A, p):
    """Basis of {x : A x = 0 mod p}."""
    A = np.array(A, dtype=np.int64) % p
    rows, cols = A.shape
    pivcols, r = [], 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c] % p), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        A[r] = (A[r] * pow(int(A[r, c]), -1, p)) % p
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] = (A[i] - A[i, c] * A[r]) % p
        pivcols.append(c)
        r += 1
        if r == rows:
            break
    free = [c for c in range(cols) if c not in pivcols]
    basis = []
    for f in free:
        x = np.zeros(cols, dtype=np.int64)
        x[f] = 1
        for k, pc in enumerate(pivcols):
            x[pc] = (-A[k, f]) % p
        basis.append(x)
    return basis


# ---------------------------------------------------------------- spectral tools on the circle
def fourier_diff(n, L=2 * math.pi):
    """Spectral first- and second-derivative matrices on n equispaced points of a circle of length L."""
    k = np.fft.fftfreq(n, d=1.0 / n) * (2 * math.pi / L)
    F = np.fft.fft(np.eye(n), axis=0)
    Finv = np.fft.ifft(np.eye(n), axis=0)
    k1 = k.copy()
    if n % 2 == 0:
        k1[n // 2] = 0.0
    D1 = np.real(Finv @ np.diag(1j * k1) @ F)
    D2 = np.real(Finv @ np.diag(-(k ** 2)) @ F)
    return D1, D2


# ---------------------------------------------------------------- reporting
class Report:
    def __init__(self, name):
        self.name = name
        self.rows = []

    def check(self, label, ok, detail=""):
        self.rows.append((label, bool(ok), detail))
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
        return ok

    def summary(self):
        n = len(self.rows)
        p = sum(r[1] for r in self.rows)
        print(f"  -> {self.name}: {p}/{n} checks passed")
        return p == n


# ---------------------------------------------------------------- exact Z_N lattice gauge model
class ZNModel:
    """Exact Z_N gauge theory on a small open planar box: L^2(Z_N) per link, -Laplacian of the N-cycle,
    Psi_p = -Re zeta^{a(dp)}, W(C) = zeta^{a(C)}. Its centre is Z_N itself, so T_eta acts exactly as in
    Lemma 5; it is used to test the selection rule with resolvents and full ground states exactly."""

    def __init__(self, N, nx, ny):
        self.N = N
        self.links, self.plaqs = box_geometry(nx, ny, 0)
        self.li = {l: k for k, l in enumerate(self.links)}
        n = len(self.links)
        self.n = n
        self.dim = N ** n
        idx = np.indices([N] * n).reshape(n, -1).T          # basis configurations a in Z_N^n
        self.cfg = idx
        self.zeta = np.exp(2j * math.pi / N)
        S = np.roll(np.eye(N), 1, axis=0)
        lap = 2 * np.eye(N) - S - S.T
        lap = lap / (2 - 2 * math.cos(2 * math.pi / N))       # normalized: gap 1
        self.lap1 = lap
        Hel = np.zeros((self.dim, self.dim))
        for k in range(n):
            Hel += np.kron(np.kron(np.eye(N ** k), lap), np.eye(N ** (n - k - 1)))
        self.Hel = Hel
        self.vac = np.ones(self.dim) / math.sqrt(self.dim)    # electric vacuum (constant function)

    def charge(self, path):
        q = np.zeros(self.dim, dtype=np.int64)
        for l, o in path:
            q += o * self.cfg[:, self.li[l]]
        return q % self.N

    def psi(self, p):
        return -np.real(self.zeta ** self.charge(plaq_boundary(p)))

    def wilson(self, path):
        return self.zeta ** self.charge(path)
