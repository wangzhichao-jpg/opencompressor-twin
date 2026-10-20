# Contributing / 贡献指南

Contributions of code, documentation and — above all — **measured compressor data** are welcome.

欢迎贡献代码、文档，尤其是**压缩机实测数据**。

## Development / 开发

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ruff check src tests tools && ruff format --check src tests tools
pytest -q
python tools/regression_check.py                  # frozen results must reproduce exactly
```

- SI units inside the code; absolute pressure; never convert shaft power to electrical input implicitly.
- Every new correlation or model needs its source, validity range and tests (normal, boundary and invalid inputs).
- Do not edit frozen evidence (`evidence/`, `data/public/*/split.json`, `data_sources/round5_*`) or the data-preparation scripts listed in `pyproject.toml` `extend-exclude`; propose a new versioned dataset or protocol instead.
- 代码内部使用 SI 与绝对压力；新关联式须注明文献、适用范围并附测试；不得修改已冻结证据，如需变化请新建带版本号的数据集或协议。

## Contributing measured data / 贡献实测数据

Most valuable: rolling-piston or scroll compressors with **paired mass flow and compressor power at the same steady condition**. Any manufacturer is welcome. Please provide:

1. Suction and discharge absolute pressure, suction temperature, mechanical speed (or a documented definition if only frequency is known), refrigerant mass flow, compressor power.
2. Compressor structure, model or anonymised sample ID, refrigerant, displacement and its source.
3. Power boundary (motor input, inverter input or shaft) and whether auxiliaries are included.
4. Whether mass flow is measured directly or calculated (and how).
5. Repeat tests of one nominal condition marked with a shared group ID.
6. Licence or permission for publication. **Data you are not allowed to publish cannot be added to this repository**; local use with the same tools is still possible.

最有价值的是同一稳态工况下**流量与压缩机功率配对**的滚动转子或涡旋实测数据，品牌不限。请提供上述六项信息；无权公开的数据不能进入本仓库，但可以在本地用同一工具处理。

## Pull requests

Fork, create a feature branch, keep commits focused, describe the change and how it was tested. CI must pass.
