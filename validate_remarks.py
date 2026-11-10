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
"""Validation of the checkable part of Remark 10 (static hypotheses).

What is checked
  (a) The full ground state determines the Hamiltonian: V = E_0 + (g^2/2) Laplacian(Omega)/Omega recovers the
      potential exactly (spectral discretization on the circle, three potentials), and with it the whole spectrum
      and the gap.
  (b) Partial static data do not: two potentials with the same value of <cos x> in their ground states (the analogue
      of a Wilson-loop average) but different spectral gaps.
Remarks 11 and 12 are not numerical statements; the entanglement results quoted in Remark 12 are validated in the
companion suite for the paper "Certified Strong-Coupling Endpoint and Short-Range Entanglement".
"""
import math
import sys

import numpy as np
from scipy.optimize import brentq

from common import Report, fourier_diff


def ground(V, g, D2):
    H = -(g ** 2 / 2) * D2 + np.diag(V)
    ev, vec = np.linalg.eigh(0.5 * (H + H.T))
    Om = vec[:, 0] * np.sign(vec[:, 0].sum())
    return ev, Om


def main():
    rep = Report("Remark 10")
    n = 127
    x = np.arange(n) * 2 * math.pi / n
    D1, D2 = fourier_diff(n)
    g = 1.1
    ok, det = True, []
    for V in (1.5 * np.cos(x), np.cos(x) + 0.7 * np.cos(2 * x), 0.8 * np.sin(x) - 0.3 * np.cos(3 * x)):
        ev, Om = ground(V, g, D2)
        Vrec = ev[0] + (g ** 2 / 2) * (D2 @ Om) / Om
        ev2, _ = ground(Vrec, g, D2)
        err = np.abs(Vrec - V).max()
        ok &= err < 1e-8 and abs((ev2[1] - ev2[0]) - (ev[1] - ev[0])) < 1e-8
        det.append(f"max |V_rec - V| = {err:.1e}")
    rep.check("(a) Omega determines V = E0 + (g^2/2) Omega''/Omega, hence the gap", ok, "; ".join(det))

    def obs_gap(V):
        ev, Om = ground(V, g, D2)
        w = Om ** 2 / np.sum(Om ** 2)
        return float(np.sum(w * np.cos(x))), ev[1] - ev[0]
    target, _ = obs_gap(-1.0 * np.cos(x))
    a2 = brentq(lambda a: obs_gap(-a * np.cos(x) - 1.5 * np.cos(2 * x))[0] - target, 0.01, 10)
    o1, g1 = obs_gap(-1.0 * np.cos(x))
    o2, g2 = obs_gap(-a2 * np.cos(x) - 1.5 * np.cos(2 * x))
    rep.check("(b) equal static average <cos x>, different gaps", abs(o1 - o2) < 1e-9 and abs(g1 - g2) > 0.1,
              f"<cos x> = {o1:.4f} vs {o2:.4f}; gaps {g1:.3f} vs {g2:.3f}")
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
