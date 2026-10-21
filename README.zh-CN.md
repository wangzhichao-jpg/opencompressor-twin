# OpenCompressor-Twin

[ [English](README.md) | **简体中文** ]

[![CI](https://github.com/wangzhichao-jpg/opencompressor-twin/actions/workflows/ci.yml/badge.svg)](https://github.com/wangzhichao-jpg/opencompressor-twin/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.11–3.14](https://img.shields.io/badge/Python-3.11–3.14-brightgreen.svg)](pyproject.toml)
[![视频: B站](https://img.shields.io/badge/Video-Bilibili-00A1D6.svg)](https://www.bilibili.com/video/BV12Xnw6TEY2/)
[![视频: YouTube](https://img.shields.io/badge/Video-YouTube-FF0000.svg)](https://youtu.be/IrX2Fq-NGEU)

**统一、可复现的容积式制冷压缩机性能建模工具：用实测点标定模型，预测未测工况并给出适用域，与行业基线诚实对比，并推荐下一个测点。**

仓库：[GitHub](https://github.com/wangzhichao-jpg/opencompressor-twin) · [Gitee 镜像](https://gitee.com/bigtaurus/opencompressor-twin)

![滚动转子式与涡旋式压缩机四分之一剖视模型](docs/assets/showcase.png)

> **当前状态：研究预览（0.1.0a1）。** 涡旋流程已在两款压缩机的公开量热计实测数据上检验；滚动转子模型已实现并通过软件测试，**但尚未经过滚动转子实测数据验证**。使用任何数字前请先阅读[验证报告](docs/validation.zh-CN.md)。

## 能做什么

```text
实测点 ─▶ 单位/边界/来源检查 ─▶ 压缩机模型（滚动转子 | 涡旋）
      ─▶ 按工况组标定 ─▶ 预测 + 适用域检查（可选 conformal 区间）
      ─▶ 下一测点推荐 ─▶ 冻结协议下与基线对比 ─▶ 离线 HTML 报告
```

- **一个安装包、一个命令行**（`octwin`）同时支持两种结构，由元数据选择模型，不是两套程序。
- **SI 单位、绝对压力、明确的功率边界**：轴功率不会被当作电输入；转速是逐点实测还是名义等效值，均有记录。
- **按工况组处理数据**：同一名义工况的重复测试共用组号，不会跨训练/校准/测试集。
- **保守的适用域**：超出训练输入凸包的点直接拒绝，不静默外推。
- **基线与模型并列**：每次基准评测同时拟合仿射面和 AHRI 540 风格十系数三次多项式。
- **可选的显式残差修正**（`--residual-rbf`）：在物理模型上叠加径向基函数对数残差修正，作为独立层保存和校验。
- **测点推荐**：覆盖式（maximin）、局部 D 最优参数信息、全输入核方差降低，均可按相对成本排序。
- **冻结协议评测工具**：固定预算学习曲线、“选点—揭示—重标定”回放、看不到留出答案的纯开发交叉验证。

## 验证摘要

留出实测点的平均绝对相对误差（MARE），格式为“流量 / 电功率”。完整表格、P95、最大误差、区间与局限见[验证报告](docs/validation.zh-CN.md)。

| 数据集（公开实验） | 测试点 | 四参数模型 | 四参数 + RBF | AHRI 风格十系数 |
|---|---:|---|---|---|
| Copeland ZP21K5E-PFV，R410A，吸气 65°F（AREP #11）——**协议在下载数据前冻结** | 12 | 1.40% / 2.80% | 0.36% / 0.26% | **0.29% / 0.14%** |
| Copeland ZP21K5E-PFV，R32，吸气 65°F（AREP #11）——**协议在下载数据前冻结** | 10 | 1.24% / 2.22% | **0.24% / 0.47%** | 0.23% / 0.48% |
| Copeland ZS21KAE-PFV，R404A，吸气 65°F（AREP #21） | 12 | 0.62% / 1.71% | 未在该集评分 | **0.41% / 0.16%** |

结论：

- 四参数物理模型在这些数据上达到 MARE≤5%、P95≤10% 的目标，但**在固定转速、固定吸气温度的训练域内插值中，十系数多项式明显更好**。
- RBF 残差修正稳定改进物理模型，与十系数基线**大致持平**（R410A 稍差、R32 稍好）；它增加了系数，不是“仅四参数”的成绩。
- 各种推荐策略都没有一致优于简单的 maximin 覆盖布点，**本项目不宣称节省测点或台架时间**。
- 尚未检验：滚动转子实测、跨转速预测、边界外推、独立实验室数据。

## 快速开始

需要 Python 3.11–3.14，建议使用独立环境；国内可使用清华镜像。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    POSIX: source .venv/bin/activate
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[test]"
pytest -q
octwin demo --output runs/demo          # 两种结构的合成数据软件自检
```

用冻结划分评测公开数据集：

```bash
octwin benchmark data/public/arep11_r410a_65f/data.csv \
  --meta data/public/arep11_r410a_65f/metadata.json \
  --split data/public/arep11_r410a_65f/split.json \
  --output runs/arep11_r410a
```

用自己的数据标定、预测与推荐：

```bash
octwin validate data.csv --meta metadata.json
octwin calibrate data.csv --meta metadata.json --split split.json --output model.json
octwin predict --model model.json --conditions inputs.csv --output predictions.csv
octwin recommend --model model.json --candidates candidates.csv --strategy kernel_ivr --budget 3 --approved --output next.json
```

数据格式、元数据字段与全部命令见[使用指南](docs/user-guide.zh-CN.md)；空模板见 `data_templates/`。

## 模型简述

`m = ρ_s · V_d · N/60 · η_v`，`P = m · Δh_s / η_is`；容积效率与等熵效率采用有界四参数关联式，压比项按结构区分（滚动转子 `PR − 1`，涡旋 `ln PR`）。物性来自 CoolProp，吸气须为单相蒸气。目前转速只进入理论吞吐量，**未建模随转速变化的损失**，这是已知的主要物理局限。

## 可复现性

- `data/public/`：公开数据集、元数据、冻结划分和转录审计；`data_sources/`：逐单元转录；`tools/prepare_*.py`：转换脚本。原始报告不随仓库分发，请按元数据中的链接下载。
- `evidence/`：冻结结果文件；`tools/regression_check.py` 重新计算并要求逐一精确一致。
- `data_sources/round5_external_protocol.json` 与 `round5_protocol_freeze.json` 记录了在下载 AREP #11 数据之前固定的协议与代码哈希。

## 贡献数据

最有价值的贡献是带同工况流量与压缩机功率的滚动转子（及涡旋）实测数据。请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)；只要来源、单位、功率边界和许可清楚，任何品牌的数据都欢迎。

## 许可证

原创代码采用 [MIT](LICENSE) 许可，© 2026 建科环能科技有限公司。依赖许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)；数据集保留各自条款与署名，见 [DATA_LICENSES.md](DATA_LICENSES.md)。
