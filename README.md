# minitorch

从零实现一个迷你版 PyTorch，用于系统学习其底层实现原理与核心设计原则。

本项目采用**渐进双层架构**：

- **阶段一（纯 Python）**：讲透自动微分、计算图、nn.Module、优化器等核心机制。
- **阶段二（C++ 核心 + pybind11 绑定）**：用 C++ 重写核心计算层，Python 前端不变，展示真实 PyTorch 的工程分层。

## 教程文档

在线文档托管于 GitHub Pages（MkDocs Material）：

```
mkdocs serve   # 本地预览
```

## 快速开始

```bash
uv sync --extra dev      # 安装开发依赖
uv pip install -e .      # 可编辑安装
pytest                  # 运行测试
```

## 目录结构

```
minitorch/
├── docs/               # 教学文档（MkDocs Material，13 章）
├── src/minitorch/      # 框架实现（Python 前端）
│   ├── ops/            # 算子
│   ├── autograd/       # 自动微分引擎
│   ├── nn/             # 神经网络模块
│   ├── optim/          # 优化器
│   ├── data/           # 数据加载
│   ├── amp/            # 混合精度
│   ├── fx/             # 图与编译
│   ├── serialization.py
│   └── _C_ext.*.pyd    # 编译产物（C++ 扩展）
├── cpp/                # 阶段二：C++ 核心层（三层分离）
│   ├── c10/            # 核心抽象（Storage / TensorImpl / Allocator）
│   ├── aten/           # 算子（ops / dispatcher / native/{cpu,cuda}）
│   ├── autograd/       # 自动微分（Node / Engine / ops / checkpoint / profiler）
│   ├── binding/        # pybind11 绑定
│   └── CMakeLists.txt
├── tests/              # 行为测试（两阶段共用，294 passed）
└── examples/           # 端到端示例
```

## 章节路线

| 章 | 主题 | 阶段 |
|---|---|---|
| 1 | 张量与存储 | 一 |
| 2 | 算子与分发 | 一 |
| 3 | 自动微分引擎 | 一 |
| 4 | 计算图机制 | 一 |
| 5 | nn.Module 体系 | 一 |
| 6 | 优化器系统 | 一 |
| 7 | 损失与训练循环 | 一 |
| 8 | C++ 核心重写 | 二 |
| 9 | C++ 高级特性（多线程 / Allocator / Profiler / Hooks / Checkpointing） | 二 |
| 10 | CUDA 与 dispatcher | 二 |
| 11 | 数据加载与采样 | 一 |
| 12 | 持久化与混合精度 | 一 |
| 13 | 图与编译导论 | 一 |

## 许可证

MIT