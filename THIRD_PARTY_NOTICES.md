# Third-party notices / 第三方软件告知

OpenCompressor-Twin does **not** vendor or redistribute third-party source code or binaries. The wheel contains only the `octwin` package. The following packages are installed separately by the user's package manager as dependencies. License identifiers below were read from the installed package metadata on 2026-10-20; consult each project for the authoritative text.

本项目不内置、不再分发第三方源码或二进制；以下依赖由用户的包管理器单独安装。许可证标识取自2026-10-20本机已安装包的元数据，以各项目官方文本为准。

## Runtime dependencies / 运行依赖

| Package | Version range declared | Version verified | License (from package metadata) | Project |
|---|---|---|---|---|
| NumPy | >=1.26 | 1.26.4, 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 (2.5.4 metadata) | https://numpy.org |
| SciPy | >=1.12 | 1.12.0, 1.18.1 | BSD-3-Clause (OSI Approved :: BSD License) | https://scipy.org |
| CoolProp | >=7,<9 | 7.0.0, 8.0.0 | MIT | http://www.coolprop.org |

Thermophysical properties are computed with CoolProp. Results can differ from REFPROP, which the source test reports used; this is recorded as a limitation in each dataset's metadata. REFPROP is not used or distributed.

物性由CoolProp计算；原试验报告使用REFPROP，两者差异已在各数据集元数据中列为局限。本项目不使用、不分发REFPROP。

## Development and test tools / 开发与测试工具（不随安装包分发）

| Package | License | Use |
|---|---|---|
| pytest | MIT | test runner |
| ruff | MIT | lint and formatting |

## Data / 数据

Datasets, their sources, licenses and attribution are listed separately in [DATA_LICENSES.md](DATA_LICENSES.md).
