# Data sources, licenses and attribution / 数据来源、许可与署名

Only data published openly on the internet are included, always with their source and attribution. Each dataset is a separate measurement context and must not be pooled with others. Original reports are **not** redistributed; download them from the URLs below.

仅收录互联网上公开发表的数据，并注明来源与署名；各数据集为独立测量情境，不得合并。原始报告不随仓库分发，请从下列链接下载。

| Dataset (path) | Source | Kind | Terms | Attribution |
|---|---|---|---|---|
| `data/public/arep21_r404a_65f`, `data/development/arep21_train40` | ORNL/AHRI Low-GWP AREP Test Report #21, Appendix A Table A1, <https://info.ornl.gov/sites/publications/files/Pub42278.pdf> | Measured calorimeter results, Copeland ZS21KAE-PFV, R404A | Publicly available technical report; no explicit dataset licence. Numerical facts transcribed with attribution; the report itself is not redistributed. | S. Shrestha, V. Sharma, O. Abdelaziz, Oak Ridge National Laboratory, 2013 |
| `data/public/arep11_r410a_65f`, `data/public/arep11_r32_65f` | ORNL/AHRI Low-GWP AREP Test Report #11, Appendix A Tables A3 and A6, <https://info.ornl.gov/sites/publications/files/Pub44928.pdf> | Measured calorimeter results, Copeland ZP21K5E-PFV, R410A and R32 | Same as above | S. Shrestha, I. Mahderekal, V. Sharma, O. Abdelaziz, Oak Ridge National Laboratory, 2013 |
| `data/public/sanden_tgve08_dry` | Zenodo record 14603617, <https://zenodo.org/records/14603617>; related paper DOI 10.1016/j.ijrefrig.2025.05.013 | Measured shaft power after belt losses (not electrical input); exploratory only | CC BY 4.0 | Applied Thermodynamics Laboratory, University of Liège; N. Leclercq, Bederna, V. Lemort. Changes (row selection, unit conversion) are documented in `tools/prepare_sanden.py`. |
| `data_templates/rolling_piston_observations.csv` | This project | Empty header-only template | MIT (project licence) | — |
| Synthetic demo data (`octwin demo`) | This project | Software self-test only, not compressor validation | MIT (project licence) | Generated at run time |

## Scope notes / 范围说明

- No proprietary, confidential or partner-supplied measurement data are included in this repository.
- Manufacturer selection-software data are not used as experimental validation.
- AREP speeds are nominal-equivalent values from the published rated displacement rate and displacement, not per-point measured shaft speeds.
- The AREP reports used REFPROP; this project uses CoolProp. Header pressures are printed to whole psia and used as printed.
- 本仓库不含任何专有、保密或合作方提供的测量数据；AREP 转速为由额定位移率换算的名义等效值。
