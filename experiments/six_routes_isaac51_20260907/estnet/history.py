"""History state changes once per control step, never while evaluating rewards."""
import torch


class History:
    def __init__(self, num_envs, steps, obs_dim, device):
        self.data = torch.zeros(num_envs, steps, obs_dim, device=device)
        self.needs_fill = torch.ones(num_envs, dtype=torch.bool, device=device)

    def reset(self, ids):
        self.needs_fill[ids] = True

    def append(self, obs, exclude_current=False):
        if obs.shape != self.data[:, -1].shape:
            raise ValueError("History observation shape mismatch")
        # Key图2输入是o_{t-1:t-h}，先取历史快照再推进；出生/重置时用首帧填充缺失历史。
        # 默认False保留已有EstNet检查点的时序契约，冻结的续训源码不受此扩展影响。
        self.data[self.needs_fill] = obs[self.needs_fill, None, :]
        past = self.data.clone() if exclude_current else None
        self.data[:, :-1] = self.data[:, 1:].clone()
        self.data[:, -1] = obs
        self.data[self.needs_fill] = obs[self.needs_fill, None, :]
        self.needs_fill[:] = False
        return past if exclude_current else self.data.clone()
