# `__init__.py` 代码说明

## 文件作用
`src/__init__.py` 只做两件事：
- 声明 `src` 是一个 Python 包
- 提供包级别简介 `"IT5006 Phase 2 predictive policing pipeline."`

## 关键内容
- `__all__ = []`：当前不做包级 API 导出。

## 在项目中的意义
- 允许你使用模块方式运行，例如：`python3 -m src.run_all`
- 不承载任何业务逻辑，属于包结构基础文件。
