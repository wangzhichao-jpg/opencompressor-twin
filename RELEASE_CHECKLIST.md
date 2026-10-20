# 版本发布核对清单 (Release Checklist)

## 1. 发布前验证

- [ ] 本地全量测试：`pytest -q`
- [ ] 代码规范：`ruff check src tests tools`、`ruff format --check src tests tools`
- [ ] 数值回归：`python tools/regression_check.py` 全部 OK
- [ ] 公开树检查：`python tools/check_public_tree.py --strict`
- [ ] CI（Linux / Windows / macOS × Python 3.11–3.14、依赖下限、打包检查）全部通过
- [ ] 版本号：`src/octwin/__init__.py` 与 `pyproject.toml` 一致；`CHANGELOG.md` 已更新
- [ ] `LICENSE`、`THIRD_PARTY_NOTICES.md`、`DATA_LICENSES.md` 完整；新增数据均有来源、许可与署名
- [ ] README 与 `docs/validation*.md` 中的数字与 `evidence/` 一致，未验证范围已明确写出

## 2. 打包

- [ ] `python -m build` 生成 sdist 与 wheel；检查包内只有程序文件与许可文件
- [ ] 生成 `SHA256SUMS.txt`
- [ ] 在全新虚拟环境中安装 wheel 并运行 `octwin demo`

## 3. 发布与同步

- [ ] `main` 分支打标签 `vX.Y.Z`
- [ ] 同步推送 GitHub（wangzhichao-jpg/opencompressor-twin）与 Gitee（bigtaurus/opencompressor-twin）
- [ ] 两边 Release 上传相同附件；匿名下载并核对哈希
- [ ] 从公开镜像下载 wheel 独立安装冒烟测试
