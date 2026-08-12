# GitHub 公开前检查清单

本目录是为独立 GitHub 仓库准备的，不要在上一级论文总目录直接执行
`git init` 或整目录上传。

## 必须确认

- [ ] 与全部合作者或所在单位确认代码版权归属。
- [ ] 确认公开代码不会影响尚未申请的专利或投稿要求。
- [x] 已选择并添加 MIT `LICENSE`。
- [ ] 确认包名 `pea-pgnn`、仓库名和论文中的方法名是否最终一致。
- [ ] 补充最终作者姓名、单位、联系邮箱和项目主页。
- [ ] 检查经验公式的引用、适用范围、单位和实现说明。
- [ ] 确认不公开的数据、模型权重和论文文件均未进入 Git 跟踪列表。
- [ ] 如果要宣称“复现论文结果”，另行归档数据版本、分组 ID、数据划分、
      超参数、随机种子、依赖版本、检查点和生成图表的脚本。

## 本地检查

在本目录运行：

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
python -m build
```

首次上传 GitHub 时，可在本目录运行：

```bash
git init -b main
git add .
git status
git commit -m "Initial public package structure"
git remote add origin https://github.com/hunter137/pea-pgnn.git
git push -u origin main
```

务必在 `git commit` 前阅读 `git status`；理想情况下只应看到本仓库中的
源码、测试、文档、示例和配置文件，不应看到 `.venv`、`dist`、CSV、DOCX、
PDF、PPTX、模型权重或论文上级目录中的任何文件。

## 后续发布 PyPI 前

- [ ] 在 PyPI 检查发行名是否可用；导入名仍为 `pea_pgnn`。
- [ ] 先发布到 TestPyPI 并在干净环境中验证安装。
- [ ] 创建独立的 `.github/workflows/release.yml`。
- [ ] 使用 PyPI Trusted Publishing，而不是把长期 API Token 写入仓库。
- [ ] 为版本创建 Git 标签和 GitHub Release，例如 `v0.1.0`。
- [ ] 论文信息稳定后添加 `CITATION.cff`，并考虑用 Zenodo 归档版本 DOI。
