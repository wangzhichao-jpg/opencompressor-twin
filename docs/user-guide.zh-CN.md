# 使用指南

适用版本：0.1.0a1。所有命令在仓库根目录执行。

## 1. 安装

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    POSIX: source .venv/bin/activate
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[test]"
pytest -q
octwin --version
```

已验证组合：Python 3.12 + NumPy 1.26.4 / SciPy 1.12.0 / CoolProp 7.0.0（依赖下限），以及 Python 3.12/3.14 + NumPy 2.5.4 / SciPy 1.18.1 / CoolProp 8.0.0。其他平台以 CI 结果为准。

## 2. 观测数据 CSV

全部 SI 单位，压力为**绝对压力**：

```text
point_id,group_id,suction_pressure_pa,discharge_pressure_pa,suction_temperature_k,speed_rpm,mass_flow_kg_s,power_w
```

- 一个 CSV 只放一个型号、一种工质、一种功率边界；不要把不同型号拼成一张图谱。
- `point_id` 唯一；同一名义工况的重复测试必须共用 `group_id`。输入完全相同而组号不同会被拒绝，以防信息泄漏。
- 吸气状态必须是 CoolProp 判定的单相蒸气；两相或液体吸气会被拒绝。
- 推荐用的候选 CSV 只含 `point_id` 和四个输入列，可另加正的相对成本列；带有流量或功率列的候选文件会被拒绝。
- 空模板：`data_templates/rolling_piston_observations.csv`。

## 3. 元数据 JSON

必填字段：

| 字段 | 取值与含义 |
|---|---|
| `dataset_id` | 数据集标识 |
| `source_kind` | `experiment` / `digitized_figure` / `manufacturer_map` / `synthetic` |
| `validation_role` | `software_test` / `public_validation` / `partner_application`；合成数据只能是 `software_test` |
| `source`、`license` | 来源与许可说明 |
| `pressure_basis` | 必须为 `absolute` |
| `speed_definition` | `mechanical_rpm`（逐点机械转速）或 `nominal_mechanical_rpm` |
| `power_boundary` | 例如 `electrical_input`、`shaft_power`；预测结果遵循同一边界，不互相换算 |
| `compressor` | `compressor_type`（`rolling_piston` / `scroll`）、`manufacturer`、`model_id`、`refrigerant`、`displacement_m3_rev` |

使用 `nominal_mechanical_rpm` 时必须给出 `nominal_speed_provenance`（`source_location`、`volume_rate_m3_s`），全表转速须固定且等于 `60 × 额定位移率 / 每转排量`。程序**不会**由电源频率推算转速。完整示例见 `data/public/arep11_r410a_65f/metadata.json`。

## 4. 命令

```bash
# 格式与单相检查（不代表数据已被准入为验证数据）
octwin validate data.csv --meta metadata.json

# 标定：有划分时只用 train 组；--residual-rbf 显式启用残差修正层
octwin calibrate data.csv --meta metadata.json --split split.json --output model.json
octwin calibrate data.csv --meta metadata.json --split split.json --residual-rbf --output model_rbf.json

# 预测：超出训练凸包的点被拒绝
octwin predict --model model.json --conditions inputs.csv --output predictions.csv

# 推荐下一测点：候选须经工程师批准（--approved）
octwin recommend --model model.json --candidates candidates.csv --strategy coverage --budget 3 --approved --output next.json
octwin recommend --model model.json --candidates candidates.csv --strategy kernel_ivr --cost-column relative_cost --budget 3 --approved --output next.json

# 冻结划分下的基准评测与离线报告
octwin benchmark data.csv --meta metadata.json --split split.json --output runs/case
octwin report --run runs/case

# 研究工具
octwin learning-curve   data.csv --meta metadata.json --split split.json --output runs/curve
octwin adaptive-replay  data.csv --meta metadata.json --split split.json --output runs/replay
octwin development-study dev.csv --meta dev_metadata.json --split development_protocol.json --output runs/dev
octwin demo --output runs/demo
```

划分文件格式：`{"train": [...], "calibration": [...], "test": [...]}`，组号无重叠且覆盖全部组。学习曲线与回放还需 `mandatory_hull_training_groups`（初始凸包组）。已有冻结结果的输出目录不允许被不同协议覆盖，请换新目录。

## 5. 模型与扩展

- **基础模型**：`m = ρ_s · V_d · N/60 · η_v`，`P = m · Δh_s / η_is`；容积效率与等熵效率为有界四参数关联式，压比项按结构区分（滚动转子 `PR − 1`，涡旋 `ln PR`）。转速只进入理论吞吐量，未建模随转速变化的损失。
- **标定**：有界相对最小二乘，三个确定性初值；同组重复点等权；报告局部雅可比秩与条件数。局部满秩不等于几何参数真实可辨识。
- **区间**：可选的组最大 split-conformal 相对区间，90% 名义水平至少需要 9 个独立校准组；它不是测量不确定度评定。
- **RBF 残差修正**：按组拟合 `log(实测/基础预测)`，长度尺度 0.5、岭 0.01 固定。启用后基础效率保存在 `base_*` 字段，原区间失效，需要另行校准；训练组很少时可能变差，默认不启用。
- **推荐策略**：`coverage`（maximin 覆盖，默认）、`d_optimal`（基础模型局部参数信息）、`kernel_ivr`（全输入核潜在方差降低）。分数是设计准则，不保证误差下降；`--budget` 是点数上限，成本只用于排序。

## 6. 公开数据与复现

| 目录 | 内容 |
|---|---|
| `data/public/arep21_r404a_65f` | AREP #21 表A1，Copeland ZS21KAE-PFV，R404A |
| `data/public/arep11_r410a_65f`、`arep11_r32_65f` | AREP #11 表A3/A6，Copeland ZP21K5E-PFV |
| `data/public/sanden_tgve08_dry` | Zenodo 14603617（CC BY 4.0），轴功率，探索性 |
| `data/development/arep21_train40` | 只含 AREP #21 原训练组的开发导出 |
| `evidence/` | 冻结结果；`python tools/regression_check.py` 要求逐一精确复现 |

原始报告不随仓库分发。需要从原 PDF 重建数据时，按元数据中的链接下载，放到转换脚本注释所示位置，再运行 `tools/transcribe_arep11.py`、`tools/prepare_arep11.py` 或 `tools/prepare_arep21.py`；脚本会先核对原文件哈希。`tools/run_round5_external.py` 是一次性评分脚本，代码或协议哈希与冻结记录不一致时拒绝运行。

## 7. 当前局限

- 滚动转子尚无实测验证；涡旋只在单一名义转速、固定吸气温度下做过域内插值检验。
- 在已检验的数据上，AHRI 风格十系数多项式在域内插值精度上优于四参数模型，与 RBF 扩展大致持平。
- 凸包覆盖只是数据域检查，不是安全运行包络。
- 不宣称减少测点、节省台时、认证或特定厂商产品精度。详见[验证报告](validation.zh-CN.md)。
