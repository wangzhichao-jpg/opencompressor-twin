# Changelog

Versions follow PEP 440.

## 0.1.0a1 — 2026-10-20 (research preview)

### Validation
- Fresh frozen external test on Copeland ZP21K5E-PFV (AREP #11), R410A and R32 at 65°F suction temperature; protocol and code hashes fixed before the data were downloaded. The four-parameter model meets a 5% MARE / 10% P95 target but is clearly worse than the AHRI-style ten-coefficient baseline; the RBF residual extension improves the base model on both responses and is roughly on par with the baseline. See `docs/validation.md`.
- Held-out test on Copeland ZS21KAE-PFV (AREP #21), R404A; development-only cross-validation on its training groups.

### Features
- Unified package and `octwin` CLI for rolling-piston and scroll models: validation, grouped calibration, convex-hull domain refusal, split-conformal intervals, baselines (affine, AHRI-style ten-coefficient), offline HTML reports.
- Explicit RBF log-residual extension (`calibrate --residual-rbf`).
- Test-point recommendation: coverage, local D-optimal, kernel IVR, with relative costs.
- Research tools: fixed-budget learning curves, select-reveal-recalibrate replay, development-only cross-validation.

### Engineering
- ruff lint/format; numerical regression gate over frozen evidence; dependency lower bounds verified (NumPy 1.26, SciPy 1.12, CoolProp 7.0); public-tree hygiene check; CI for Linux/Windows/macOS × Python 3.11–3.14.
- Fixed an f-string that was valid only on Python ≥ 3.12.

### Known limitations
- Rolling-piston model not yet validated against measured data.
- Speed-dependent losses not modelled; all scored data are single nominal speed.
