import torch
def clone_module_params(module):
    """
    深拷贝模块参数（用于 epoch 前后对比）
    """
    return [p.detach().clone() for p in module.parameters()]


def compute_relative_param_change(module, params_old):
    """
    计算模块参数的相对变化率
    """
    diff_norm = 0.0
    base_norm = 0.0

    for p, p_old in zip(module.parameters(), params_old):
        diff_norm += torch.norm(p - p_old, p=2).item()
        base_norm += torch.norm(p_old, p=2).item()

    return diff_norm / (base_norm + 1e-8)
