// AccumulateGrad 实现（Ch8 C++ autograd）

#include "autograd/node.h"
#include "autograd/ops.h"
#include "aten/ops.h"


namespace minitorch {

AccumulateGrad::AccumulateGrad(TensorImplPtr var)
    : variable(std::move(var)) {
    name = "AccumulateGrad";
}

std::vector<TensorImplPtr> AccumulateGrad::apply(TensorImplPtr grad) {
    // 调用 backward hook（如果注册了）
    if (variable->backward_hook()) {
        auto hooked = variable->backward_hook()(grad);
        if (hooked) grad = hooked;
    }

    if (!variable->grad()) {
        variable->set_grad(grad);
    } else if (grad && grad->requires_grad()) {
        variable->set_grad(autograd::add(variable->grad(), grad));
    } else {
        auto existing = variable->grad();
        ops::add_inplace(existing, grad);
        variable->set_grad(existing);
    }
    return {};
}

} // namespace minitorch
