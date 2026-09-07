"""PPO 的同步辅助函数：生产 NCCL，CPU 测试 Gloo，不负责启动进程组。

runner 应先设置本 rank 的 CUDA 设备，再初始化进程组。两 rank 各 2048 个环境，
各采样 24 步，且每轮均分成 4 个 minibatch；各 rank 的局部 loss 是等量样本的
均值，因此梯度取 rank 平均等价于对应合并 batch 的梯度。不要在本模块重复 DDP。
未初始化分布式时保留原单卡数学路径，不创建 CUDA 上下文或复制模型。
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def active() -> bool:
    """仅在已有多 rank 进程组时启用集体操作；单 rank 组沿用单卡路径。"""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def world_size() -> int:
    """未初始化时返回 1，不隐式初始化通信或 CUDA。"""
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def rank() -> int:
    """未初始化时返回 0；这里的 rank 是全局 rank，不是 CUDA 物理编号。"""
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _collective_device() -> torch.device:
    """NCCL 标量使用 runner 已选择的本 rank CUDA 设备；Gloo 标量放在 CPU。"""
    if dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


@torch.no_grad()
def global_normalize(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """用全局 sum、sumsq、count 归一化优势，方差使用 population 定义。

    通信统计量用 float64 减少方差相减误差，输出保持输入 dtype/device；
    分布式标准差加 eps 的位置与原 PPO 相同。局部样本数可不同，但全局须非空。
    单卡路径严格使用原来的 mean/std(correction=0)，不改为另一种方差算法。
    """
    if not active():
        return (advantages - advantages.mean()) / (advantages.std(correction=0) + eps)
    values = advantages.detach().to(device=_collective_device(), dtype=torch.float64)
    statistics = torch.stack(
        (values.sum(), values.square().sum(), values.new_tensor(values.numel()))
    )
    dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    count = statistics[2]
    if count.item() <= 0:
        raise ValueError("Global advantage batch must not be empty")
    mean = statistics[0] / count
    variance = (statistics[1] / count - mean.square()).clamp_min(0.0)
    mean = mean.to(device=advantages.device, dtype=advantages.dtype)
    std = variance.sqrt().to(device=advantages.device, dtype=advantages.dtype)
    return (advantages - mean) / (std + eps)


@torch.no_grad()
def all_ranks_finite(loss: torch.Tensor) -> bool:
    """所有 rank 同步确认 loss 有限；任一 rank 有 NaN/Inf 时均返回 False。

    调用方应在所有 rank 上据此抛出相同异常，再进行 backward；不能先由某个
    rank 单独抛错，否则其它 rank 可能阻塞在后续梯度同步中。
    """
    finite = torch.isfinite(loss.detach()).all()
    if not active():
        return bool(finite)
    flag = finite.to(device=_collective_device(), dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


@torch.no_grad()
def average_gradients(model: torch.nn.Module) -> None:
    """在 backward 后、clip_grad_norm_ 前平均梯度；只用于等量局部 minibatch。

    先同步梯度存在标记，防止某 rank 的 None 梯度造成集体调用顺序不一致。
    仅部分 rank 有梯度的参数按其余 rank 为零来平均；所有 rank 都未使用的
    参数保持 grad=None，避免意外产生 Adam 状态或触发 weight decay。
    """
    if not active():
        return
    parameters = list(model.parameters())
    if not parameters:
        return
    present = torch.tensor(
        [parameter.grad is not None for parameter in parameters],
        dtype=torch.int32,
        device=_collective_device(),
    )
    dist.all_reduce(present, op=dist.ReduceOp.SUM)
    # 一次读取全部标记，避免为每个参数额外触发一次 CUDA -> CPU 同步。
    present_counts = present.cpu().tolist()
    for parameter, number_present in zip(parameters, present_counts):
        if number_present == 0:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size())


@torch.no_grad()
def mean_scalar(value: float) -> float:
    """平均各 rank 的标量 KL，让所有优化器作出相同的自适应学习率决策。"""
    if not active():
        return float(value)
    scalar = torch.tensor(value, dtype=torch.float64, device=_collective_device())
    if scalar.numel() != 1:
        raise ValueError("mean_scalar expects a scalar")
    dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
    return float(scalar.item() / world_size())


@torch.no_grad()
def reduce_metrics_totals(
    totals: dict[str, float], samples_seen: int
) -> tuple[dict[str, float], int]:
    """返回全局指标加权总和与全局样本计数，调用方再除以返回的计数。

    totals 应是局部 mean × 局部样本数的累加；所有 rank 必须提供同一组键。
    键排序后通信，允许各 rank 的字典插入顺序不同。未初始化时原样返回对象。
    """
    if not active():
        return totals, samples_seen
    keys = sorted(totals)
    values = torch.tensor(
        [totals[key] for key in keys] + [samples_seen],
        dtype=torch.float64,
        device=_collective_device(),
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    reduced = values.cpu().tolist()
    return dict(zip(keys, reduced[:-1])), int(reduced[-1])


@torch.no_grad()
def broadcast_model(model: torch.nn.Module) -> None:
    """将 rank0 的全部参数及 buffer 广播给各 rank；不改变优化器状态。"""
    if not active():
        return
    for value in (*model.parameters(), *model.buffers()):
        dist.broadcast(value, src=0)
