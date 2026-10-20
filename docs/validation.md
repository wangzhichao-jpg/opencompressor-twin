# Validation report

Status as of 0.1.0a1 (October 2026). [中文版](validation.zh-CN.md)

This report separates four kinds of evidence and never mixes them: (1) a fresh frozen external test, (2) an earlier held-out test whose final points have since been reused, (3) development-only cross-validation, (4) software self-tests on synthetic data. Training-fit errors are never reported as validation.

## 1. Fresh frozen external test — AREP #11, Copeland ZP21K5E-PFV

**Protocol.** The datasets, split rule, compared methods, metrics and decision criteria were written and hashed **before the report was downloaded** (`data_sources/round5_external_protocol.json`, `data_sources/round5_protocol_freeze.json`). The scoring script verifies those hashes and refuses to run twice.

**Data.** ORNL/AHRI Low-GWP AREP Test Report #11 (Shrestha, Mahderekal, Sharma, Abdelaziz, 2013), Appendix A measured results, <https://info.ornl.gov/sites/publications/files/Pub44928.pdf>. Displacement 1.24 in³/rev; nominal-equivalent speed 3499.9 rpm from the published rated displacement rate (150.69 ft³/h), not a per-point measurement. Two separate datasets, never pooled:

- Table A3, R410A, 65°F suction temperature: 66 conditions → 42 train / 12 residual calibration / 12 test;
- Table A6, R32, 65°F suction temperature: 55 conditions → 35 / 10 / 10.

All 121 cells were transcribed from the PDF text layer, checked against the rendered pages, and passed the printed-rounding consistency check between capacity, power and EER. Blank cells were not filled. Convex-hull vertices of the pressure inputs are always in training, so the test is **interior interpolation**.

**Results (test points; MARE / P95, %).**

| Dataset | Method | Mass flow | Electrical power |
|---|---|---|---|
| R410A | Four-parameter | 1.395 / 2.472 | 2.798 / 6.171 |
| | Four-parameter + RBF | 0.360 / 0.663 | 0.263 / 0.533 |
| | Affine | 1.313 / 2.516 | 0.624 / 1.342 |
| | AHRI-style 10-coefficient | **0.294 / 0.610** | **0.136 / 0.292** |
| R32 | Four-parameter | 1.242 / 2.346 | 2.223 / 4.501 |
| | Four-parameter + RBF | 0.244 / 0.482 | **0.465 / 1.027** |
| | Affine | 0.528 / 1.490 | 1.128 / 1.771 |
| | AHRI-style 10-coefficient | **0.225 / 0.486** | 0.482 / 1.050 |

All points were predicted; none refused.

**Predeclared criteria.**

| Criterion | R410A | R32 |
|---|---|---|
| A — MARE ≤ 5% and P95 ≤ 10% per response | met (both models) | met (both models) |
| B — RBF better than base on both responses | yes | yes |
| C — RBF vs. strongest baseline (larger of the two MAREs) | worse (0.360 vs 0.294) | not worse (0.465 vs 0.482) |

**90% split-conformal intervals (reported, not judged).** Base model: flow 12/12 and 9/10 covered with mean relative widths of 7.0% and 5.6%; power 12/12 and 10/10 with 15.8% and 11.0%. RBF extension: much narrower (1.0–2.1%) but flow covered only 9/12 for R410A. Ten to twelve points cannot estimate coverage reliably.

**Recommendation strategies (secondary).** Random, maximin, local D-optimal and kernel IVR orders were frozen before scoring (5 seeds; budgets up to the full training pool). Kernel IVR beat maximin in 3 of 6 comparable budget cells — no majority. No strategy showed a consistent advantage.

**Limitations.** Same laboratory and instrument family as AREP #21; nominal-equivalent speed; fixed suction temperature; interior points only; CoolProp versus REFPROP in the source report.

## 2. Earlier held-out test — AREP #21, Copeland ZS21KAE-PFV, R404A

ORNL/AHRI AREP Test Report #21 (Shrestha, Sharma, Abdelaziz, 2013), Table A1, <https://info.ornl.gov/sites/publications/files/Pub42278.pdf>. 64 conditions at 65°F suction temperature; split 40 / 12 / 12 frozen before the first scoring.

| Method | Mass flow MARE / P95 / max | Electrical power MARE / P95 / max |
|---|---|---|
| Four-parameter | 0.624 / 1.232 / 1.478 | 1.708 / 4.385 / 4.978 |
| Affine | 0.606 / 1.220 / 1.500 | 0.833 / 2.179 / 2.246 |
| AHRI-style 10-coefficient | **0.408 / 0.692 / 0.768** | **0.158 / 0.422 / 0.657** |

Nominal 90% intervals of the four-parameter model covered 11/12 (flow) and 12/12 (power). These 12 final points were later reused in secondary learning-curve and replay analyses, so they are **no longer an unseen test** and were not used to choose any later method.

## 3. Development-only cross-validation (AREP #21 training groups)

Only the 40 original training groups were exported; the 24 reserved calibration/test labels are absent from the file. Nine hull groups stay in training; the other 31 are validated once in five folds. Hyperparameters of the RBF extension (length scale 0.5, ridge 0.01) were fixed beforehand.

| Method | Mass flow MARE / P95 | Electrical power MARE / P95 |
|---|---|---|
| Four-parameter | 0.709 / 1.541 | 1.278 / 2.994 |
| Affine | 0.617 / 1.535 | 0.676 / 1.950 |
| AHRI-style 10-coefficient | 0.432 / 0.935 | **0.232 / 0.596** |
| Four-parameter + RBF | **0.403 / 0.984** | 0.287 / 0.863 |

Residuals of the base model correlated with absolute suction pressure after controlling for pressure ratio (partial correlation 0.68 for flow, 0.75 for power): structure the four-parameter correlation does not capture. This motivated the residual extension, which section 1 then tested on unseen data. With only nine training groups the extension was worse than the base model for flow, so it is not enabled by default.

## 4. Exploratory and software-only evidence

- **Sanden TGVE08 dry points** (Zenodo 14603617, CC BY 4.0; shaft power after belt losses, not electrical input): 10 training / 2 test groups; both test points lie outside the training hull and were **refused**. No accuracy is claimed.
- **Synthetic demos** for both structures exercise the software pipeline only.
- Automated tests, a numerical regression gate (`tools/regression_check.py`), dependency lower bounds (NumPy 1.26, SciPy 1.12, CoolProp 7.0) and a public-tree hygiene check.

## 5. Not yet validated

- **Rolling-piston compressors** — no paired measured flow/power table has been admitted yet.
- Speed-dependent behaviour and cross-speed prediction; all scored data are single nominal speed.
- Boundary extrapolation, superheat-controlled tables, other laboratories.
- Any claim of fewer test points, bench-time savings or accuracy for a specific manufacturer's products.
