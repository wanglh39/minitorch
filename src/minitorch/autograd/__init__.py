"""autograd：自动微分引擎（Ch3 / Ch4）。

子模块：
  - function:  Function 基类、Node、Edge、apply 机制。
  - engine:    反向传播调度引擎（拓扑排序 + 执行）。
  - variable:  Tensor.backward 入口。
  - grad_mode: no_grad/enable_grad 上下文。
"""