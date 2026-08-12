# GitHub 与 PyPI 发布检查清单

本目录是为独立 GitHub 仓库准备的，不要在上一级论文总目录直接执行
`git init` 或整目录上传。

## 已完成的仓库准备

- [x] 已创建独立公开仓库，未在论文总目录初始化 Git。
- [x] 已选择并添加 MIT `LICENSE`。
- [x] 作者、单位、资助信息已在 README 与软件元数据中对齐。
- [x] 已添加经验公式来源、单位、简化假设和适用范围说明。
- [x] 已明确代码包不含论文数据库、权重和投稿文件。
- [x] 已配置 CI、测试、构建、README 渲染和覆盖率检查。
- [x] 已添加模型卡、安全政策、行为准则、Issue/PR 模板和 Dependabot。
- [x] 已添加 TestPyPI/PyPI Trusted Publishing 工作流。

## 作者仍需确认

- [ ] 与全部合作者或所在单位确认代码版权归属。
- [ ] 确认公开代码不会影响尚未申请的专利或投稿要求。
- [ ] 最终确认包名 `pea-pgnn`、仓库名和论文中的方法名一致。
- [ ] 最终确认不公开的数据、模型权重和论文文件均未进入 Git 跟踪列表。
- [ ] 如果要宣称“复现论文结果”，另行归档数据版本、分组 ID、数据划分、
      超参数、随机种子、依赖版本、检查点和生成图表的脚本。

## 本地检查

在本目录运行：

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
python -m build
python -m twine check dist/*
```

务必在 `git commit` 前阅读 `git status`；理想情况下只应看到本仓库中的
源码、测试、文档、示例和配置文件，不应看到 `.venv`、`dist`、CSV、DOCX、
PDF、PPTX、模型权重或论文上级目录中的任何文件。

## 后续发布 PyPI 前

- [x] 已检查发行名当前尚未在 PyPI 注册；导入名仍为 `pea_pgnn`。
- [x] 已创建 `.github/workflows/release.yml`，使用短期 OIDC 身份而非长期 Token。
- [ ] 在 PyPI 的 Trusted Publishing 页面登记：项目 `pea-pgnn`、所有者
      `hunter137`、仓库 `pea-pgnn`、工作流 `release.yml`、环境 `pypi`。
- [ ] 在 TestPyPI 登记同样的发布者，但环境填写 `testpypi`。
- [ ] 在 GitHub 创建 `pypi` 和 `testpypi` environments；`pypi` 应要求人工批准。
- [ ] 手动运行 `release` 工作流发布到 TestPyPI，并在干净环境验证安装。
- [ ] 确认版本号、`CHANGELOG.md` 和 `CITATION.cff` 一致。
- [ ] 为该版本创建 `vX.Y.Z` 标签和 GitHub Release；发布事件将触发 PyPI 上传。
