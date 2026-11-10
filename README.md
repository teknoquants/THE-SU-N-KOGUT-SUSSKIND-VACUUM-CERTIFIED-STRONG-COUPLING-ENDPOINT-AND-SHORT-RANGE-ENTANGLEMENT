# Validation scripts: A Uniform Strong-Coupling Spectral Gap and a Center Selection Rule

Companion code to the white paper by R. Cooperman and A.E. Petrunina V.V., "A Uniform Strong-Coupling Spectral Gap
and a Center Selection Rule for the SU(N) Kogut–Susskind Lattice Hamiltonian" (XISOSIS, Vienna, 10 November 2026).
Released under the MIT License (see `LICENSE`; the licence is also reproduced at the top of every script).

GitHub: [GITHUB REPOSITORY URL — placeholder] · DOI: [DOI — placeholder]

## Run

```
pip install numpy scipy sympy
python run_all.py
```

Python 3.10+, NumPy ≥ 2.0, SciPy ≥ 1.9 (for `scipy.optimize.milp`), SymPy. Runtime is about 45 seconds.
`run_log.txt` is the output of a full run: 61 checks, all passing.

## Coverage

| Script | Paper item | Checks |
|---|---|---|
| `validate_definitions.py` | Definitions 1–3, 5, 7–10 | Orthonormal su(N) basis and Killing form B = 2N Tr (N = 2–6); \|W(C)\| ≤ 1 and gauge invariance of W(C), ReTr U_p (SU(2), SU(3)); gap 1 of h_e; embedding ι and box B_L; β*, g₀(N) equivalence; centre ζ𝟙 and the Stokes identity η(∂Σ) = ⟨Σ, δη⟩; \|𝒫(C)\| ≤ 8(r+s) |
| `validate_lemma1.py` | Lemma 1, all proof steps | Ric = N/2 computed from brackets; fundamental and adjoint Casimirs; **exact** minimum of C(λ) = (N²−1)/N over dominant weights (N = 2–7) and C(ω_k) = k(N−k)(N+1)/N; SU(2) characters; the H̄ identity (symbolic); ‖Ψ_p‖ bound; ≤ 4 plaquettes per link |
| `validate_lemma2.py` | Lemma 2, all proof steps | Pointwise and quadratic-form ground-state transforms (spectral accuracy); unitarity of f ↦ Ωf; gap = (g²/2)/C_P at three couplings; exact discrete analogue on a weighted cycle |
| `validate_proposition3.py` | Proposition 3, all proof steps | Weighted Bochner identity; curvature ⇒ gap (sharp on Ornstein–Uhlenbeck); a gauge-symmetric example where the full Bakry–Émery condition fails but the horizontal one holds, showing the gap on invariant functions while non-invariant modes lie below κ |
| `validate_theorem4.py` | Theorem 4, all proof steps | Embedding, supports, ≤ 3 plaquettes per corner, sup\|ReTr\| = N; threshold equivalence; **padding** (link sites in {0..2L}³, supports reach 2L+2); Yarotsky-type gap illustration; decoupling; rescaling to g²c_N/4; gauge descent; g₀(N)⁴β* = 6N²/(N²−1) exactly |
| `validate_lemma5.py` | Lemma 5, all proof steps | T_η preserves Laplacian eigenspaces; T_η Ψ_p = Ψ_p ⇔ δη(p) ≡ 0; T_η W = ζ^{η(C)} W; the character step by exact linear algebra over ℤ₂, ℤ₃, ℤ₅ (600 cases); the conclusion **exactly** in a ℤ_N gauge model with resolvents in F, F′; SU(2)/SU(3) Haar Monte Carlo with exact values 2^{−rs}, 6^{−rs} |
| `validate_prop6_cor7.py` | Proposition 6, Corollary 7 | ∂∘∂ = 0; projection is a chain map; no planar 2-cycles; stacked sums over every square equal 1; **exact minimum support = rs** by integer programming (ℤ₂, ℤ₃, r, s ≤ 3); full ℤ_N vacuum has ⟨W(C)⟩ = O(λ^{rs}) with leading exponent exactly rs (rs = 1, 2, 4) |
| `validate_proposition8.py` | Proposition 8, all proof steps | Adjacency degree 12; lattice-animal counts ≤ (12e)ⁿ; ≤ 8(r+s) touching plaquettes; the inequality chain and the constants σ, M on a grid; a truncated expansion with adversarial weights |
| `validate_proposition9.py` | Proposition 9, all proof steps | Schur test for block matrices; polymer row sum dominates ‖Hess h‖; a per-polymer bound alone fails; κ = N/2 − 2b |
| `validate_remarks.py` | Remark 10 | The full ground state determines V and the gap; equal static averages with different gaps |

## What is not validated

The numerical values of Yarotsky's constants c₁, c₂ (Theorem 4 uses them only through β*); the expansion
hypothesis 𝔈(g, K) of Proposition 8; the polymer row-sum bound of Proposition 9 for the actual vacuum; Remarks 11 and
12 (the entanglement results quoted in Remark 12 are validated in the companion suite for "Certified Strong-Coupling
Endpoint and Short-Range Entanglement"). Statements about SU(N) lattice gauge theory on arbitrarily large boxes are
checked on finite boxes, on exact ℤ_N gauge models with the same centre structure, and on one-dimensional models with
the same operator structure. Passing these checks confirms the proved statements and the logic of each implication;
it does not prove a continuum mass gap.
