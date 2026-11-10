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
"""Run every validation script for the white paper
"A Uniform Strong-Coupling Spectral Gap and a Center Selection Rule for the SU(N) Kogut-Susskind Lattice Hamiltonian".

Usage:  python run_all.py      (exit code 0 if every check passes)
"""
import sys
import time

import validate_definitions
import validate_lemma1
import validate_lemma2
import validate_proposition3
import validate_theorem4
import validate_lemma5
import validate_prop6_cor7
import validate_proposition8
import validate_proposition9
import validate_remarks

SUITES = [
    ("Definitions 1-11", validate_definitions),
    ("Lemma 1 - normalization and constants", validate_lemma1),
    ("Lemma 2 - ground-state transform", validate_lemma2),
    ("Proposition 3 - horizontal curvature criterion", validate_proposition3),
    ("Theorem 4 - uniform strong-coupling gap", validate_theorem4),
    ("Lemma 5 - center selection rule", validate_lemma5),
    ("Proposition 6 / Corollary 7 - area count and expansion terms", validate_prop6_cor7),
    ("Proposition 8 - conditional area law", validate_proposition8),
    ("Proposition 9 - Schur-test criterion", validate_proposition9),
    ("Remark 10 - static hypotheses", validate_remarks),
]


def main():
    results = []
    for title, mod in SUITES:
        print(f"\n=== {title} ===")
        t = time.time()
        ok = mod.main()
        results.append((title, ok, time.time() - t))
    print("\n=== Summary ===")
    for title, ok, dt in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {title}  ({dt:.1f} s)")
    print("\nNot validated (open inputs of the paper): the numerical values of Yarotsky's constants c1, c2; the\n"
          "expansion hypothesis E(g, K) of Proposition 8; the polymer row-sum bound of Proposition 9 for the actual\n"
          "vacuum; Remarks 11-12. SU(N) statements on the infinite lattice are checked on finite boxes, exact Z_N\n"
          "gauge models and one-dimensional models with the same operator structure.")
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
