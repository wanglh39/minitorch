// GradMode：梯度计算开关（Ch8 C++ autograd）
//
// 对应阶段一的 src/minitorch/autograd/grad_mode.py。
// thread_local 标志控制前向是否建图。NoGradGuard 是 RAII 守卫。
// 对应真实 PyTorch 的 c10::AutoGradMode / at::AutoNonVariableGuard。

#pragma once

#include <atomic>

namespace minitorch {

inline thread_local bool grad_mode_enabled = true;

inline bool is_grad_enabled() { return grad_mode_enabled; }

inline void set_grad_enabled(bool v) { grad_mode_enabled = v; }

class NoGradGuard {
public:
    NoGradGuard() : prev_(grad_mode_enabled) { grad_mode_enabled = false; }
    ~NoGradGuard() { grad_mode_enabled = prev_; }
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool prev_;
};

class EnableGradGuard {
public:
    EnableGradGuard() : prev_(grad_mode_enabled) { grad_mode_enabled = true; }
    ~EnableGradGuard() { grad_mode_enabled = prev_; }
    EnableGradGuard(const EnableGradGuard&) = delete;
    EnableGradGuard& operator=(const EnableGradGuard&) = delete;

private:
    bool prev_;
};

// ── Anomaly Detection ────────────────────────────────
// 检测 NaN/Inf 梯度，对应 torch.autograd.detect_anomaly
inline thread_local bool anomaly_check_enabled = false;

inline bool is_anomaly_check_enabled() { return anomaly_check_enabled; }
inline void set_anomaly_check_enabled(bool v) { anomaly_check_enabled = v; }

class AnomalyGuard {
public:
    AnomalyGuard() : prev_(anomaly_check_enabled) { anomaly_check_enabled = true; }
    ~AnomalyGuard() { anomaly_check_enabled = prev_; }
    AnomalyGuard(const AnomalyGuard&) = delete;
    AnomalyGuard& operator=(const AnomalyGuard&) = delete;
private:
    bool prev_;
};

} // namespace minitorch