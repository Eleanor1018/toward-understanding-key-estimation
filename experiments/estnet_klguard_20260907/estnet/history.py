"""History state changes once per control step, never while evaluating rewards."""
import torch


class History:
    def __init__(self, num_envs, steps, obs_dim, device):
        self.data = torch.zeros(num_envs, steps, obs_dim, device=device)
        self.needs_fill = torch.ones(num_envs, dtype=torch.bool, device=device)

    def reset(self, ids):
        self.needs_fill[ids] = True

    def append(self, obs):
        if obs.shape != self.data[:, -1].shape:
            raise ValueError("History observation shape mismatch")
        self.data[:, :-1] = self.data[:, 1:].clone()
        self.data[:, -1] = obs
        self.data[self.needs_fill] = obs[self.needs_fill, None, :]
        self.needs_fill[:] = False
        return self.data.clone()
