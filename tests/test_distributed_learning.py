"""用真实双进程 Gloo 核对同步数学；不初始化 CUDA 或 Isaac Sim。"""

import copy
import datetime
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from estnet import distributed as sync


class TinyModel(nn.Module):
    """可训练小网络；一条支路仅在部分 rank 使用，另有全局未用参数。"""

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 4)
        self.output = nn.Linear(4, 1)
        self.slope = nn.Parameter(torch.tensor([0.2]))
        self.unused = nn.Parameter(torch.tensor([0.7]))
        self.register_buffer("marker", torch.tensor([3], dtype=torch.int64))

    def forward(self, x, use_slope=False):
        output = self.output(torch.tanh(self.hidden(x)))
        return output + self.slope * x[:, :1] if use_slope else output


def adapt_rate(optimizer, kl):
    """只用于核对相同全局 KL 导致相同 LR，阈值沿用现有 PPO。"""
    rate = optimizer.param_groups[0]["lr"]
    if kl > 0.02:
        rate = max(1e-5, rate / 1.5)
    elif 0 < kl < 0.005:
        rate = min(1e-3, rate * 1.5)
    optimizer.param_groups[0]["lr"] = rate


def distributed_worker(local_rank, init_uri, result_dir):
    """每 rank 执行相同集体操作，并独立与合并全局 batch 的更新作对比。"""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_uri,
        rank=local_rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        assert sync.active() and sync.world_size() == 2 and sync.rank() == local_rank
        # 局部样本数不等，必须按全局 count 加权，不能平均两个局部均值/标准差。
        uneven = torch.tensor([1.0, 3.0, 5.0, 7.0, 11.0], dtype=torch.float64)
        bounds = slice(0, 3) if local_rank == 0 else slice(3, 5)
        actual = sync.global_normalize(uneven[bounds])
        expected = (uneven - uneven.mean()) / (uneven.std(correction=0) + 1e-8)
        torch.testing.assert_close(actual, expected[bounds], rtol=1e-13, atol=1e-13)

        # rank 初始参数与 buffer 刻意不同，广播后应完全一致。
        torch.manual_seed(100 + local_rank)
        model = TinyModel().double()
        model.marker.fill_(7 + local_rank)
        sync.broadcast_model(model)
        assert model.marker.item() == 7
        reference = copy.deepcopy(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=5e-4)
        x = torch.arange(24, dtype=torch.float64).reshape(8, 3) / 10.0 - 1.0
        target = torch.sin(x[:, :1])
        advantages = torch.tensor(
            [-2.0, 3.0, 0.0, 1.0, 8.0, 4.0, -1.0, 5.0], dtype=torch.float64
        )
        local = slice(4 * local_rank, 4 * (local_rank + 1))
        normalized = sync.global_normalize(advantages[local])
        global_normalized = (advantages - advantages.mean()) / (
            advantages.std(correction=0) + 1e-8
        )
        torch.testing.assert_close(
            normalized, global_normalized[local], rtol=1e-13, atol=1e-13
        )
        max_parameter_error = 0.0
        for iteration in range(2):
            # 等量局部 mean loss 的梯度平均，应等价于合并后八个样本的 mean loss。
            weights = 1.0 + normalized[:, None] * 0.1
            prediction = model(x[local], use_slope=local_rank == 0)
            loss = ((prediction - target[local]).square() * weights).mean()
            assert sync.all_ranks_finite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            sync.average_gradients(model)

            reference_optimizer.zero_grad(set_to_none=True)
            global_prediction = torch.cat(
                (reference(x[:4], True), reference(x[4:], False))
            )
            global_loss = (
                (global_prediction - target).square()
                * (1 + 0.1 * global_normalized[:, None])
            ).mean()
            global_loss.backward()
            for parameter, combined in zip(model.parameters(), reference.parameters()):
                if combined.grad is None:
                    assert parameter.grad is None
                else:
                    torch.testing.assert_close(
                        parameter.grad, combined.grad, rtol=1e-11, atol=1e-13
                    )
            # 先平均再裁剪，验证被裁剪的实际 Adam 更新也与全局 batch 一致。
            nn.utils.clip_grad_norm_(model.parameters(), 0.1, error_if_nonfinite=True)
            nn.utils.clip_grad_norm_(
                reference.parameters(), 0.1, error_if_nonfinite=True
            )
            optimizer.step()
            reference_optimizer.step()
            for parameter, combined in zip(model.parameters(), reference.parameters()):
                error = (parameter - combined).abs().max().item()
                max_parameter_error = max(max_parameter_error, error)
                torch.testing.assert_close(parameter, combined, rtol=1e-11, atol=1e-13)

            # 不同局部 KL 先求全局均值，使 LR 决策在两 rank 与参考更新一致。
            local_kl = (
                (0.01, 0.04)[local_rank]
                if iteration == 0
                else (0.001, 0.002)[local_rank]
            )
            global_kl = sync.mean_scalar(local_kl)
            expected_kl = 0.025 if iteration == 0 else 0.0015
            assert math.isclose(global_kl, expected_kl, abs_tol=1e-15)
            adapt_rate(optimizer, global_kl)
            adapt_rate(reference_optimizer, expected_kl)
            assert (
                optimizer.param_groups[0]["lr"]
                == reference_optimizer.param_groups[0]["lr"]
            )

        # 梯度已同步，优化后每一位参数都应在 rank 之间相同，而非仅接近参考。
        flat = torch.cat(
            [parameter.detach().flatten() for parameter in model.parameters()]
        )
        gathered = [torch.empty_like(flat) for _ in range(2)]
        dist.all_gather(gathered, flat)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
        assert model.unused not in optimizer.state

        # 汇总使用样本权重；两个 rank 故意采用不同键插入顺序和样本计数。
        totals = (
            {"loss": 8.0, "entropy": 4.0}
            if local_rank == 0
            else {"entropy": 18.0, "loss": 18.0}
        )
        reduced, count = sync.reduce_metrics_totals(totals, 4 if local_rank == 0 else 6)
        assert reduced == {"entropy": 22.0, "loss": 26.0} and count == 10

        # 只有 rank1 产生 NaN，所有 rank 都在 backward 前走同一个报错路径。
        bad_loss = torch.tensor(float("nan") if local_rank == 1 else 1.0)
        rejected = False
        try:
            if not sync.all_ranks_finite(bad_loss):
                raise FloatingPointError("Non-finite loss on at least one rank")
        except FloatingPointError:
            rejected = True
        assert rejected
        Path(result_dir, f"rank-{local_rank}.json").write_text(
            json.dumps(
                {
                    "rank": local_rank,
                    "max_parameter_error": max_parameter_error,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "nan_rejected": rejected,
                    "updates": 2,
                }
            ),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


class DistributedLearningTests(unittest.TestCase):
    def test_uninitialized_helpers_preserve_single_process_behavior(self):
        """单卡归一化逐位相同，原 totals 对象、参数与 None 梯度保持不变。"""
        self.assertFalse(sync.active())
        self.assertEqual(sync.world_size(), 1)
        self.assertEqual(sync.rank(), 0)
        torch.manual_seed(44)
        advantages = torch.randn(29, dtype=torch.float32)
        expected = (advantages - advantages.mean()) / (
            advantages.std(correction=0) + 1e-8
        )
        self.assertTrue(torch.equal(sync.global_normalize(advantages), expected))
        totals = {"loss": 3.7}
        reduced, count = sync.reduce_metrics_totals(totals, 11)
        self.assertIs(reduced, totals)
        self.assertEqual(count, 11)
        self.assertEqual(sync.mean_scalar(0.007), 0.007)
        self.assertTrue(sync.all_ranks_finite(torch.tensor(1.0)))
        self.assertFalse(sync.all_ranks_finite(torch.tensor(float("nan"))))
        model = TinyModel()
        model(torch.ones(2, 3)).sum().backward()
        before = copy.deepcopy(model.state_dict())
        gradients = [
            None if p.grad is None else p.grad.clone() for p in model.parameters()
        ]
        sync.average_gradients(model)
        sync.broadcast_model(model)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        for parameter, old in zip(model.parameters(), gradients):
            if old is None:
                self.assertIsNone(parameter.grad)
            else:
                self.assertTrue(torch.equal(parameter.grad, old))

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "CPU Gloo unavailable"
    )
    def test_two_gloo_ranks_match_combined_batch_and_reject_nan_together(self):
        """真实两进程通信和两次 Adam 更新，无 mock 集体通信。"""
        with tempfile.TemporaryDirectory(prefix="estnet-gloo-") as temporary:
            root = Path(temporary)
            init_uri = (root / "process-group").as_uri()
            mp.spawn(
                distributed_worker, args=(init_uri, temporary), nprocs=2, join=True
            )
            results = [
                json.loads((root / f"rank-{i}.json").read_text(encoding="utf-8"))
                for i in range(2)
            ]
        self.assertEqual([result["rank"] for result in results], [0, 1])
        self.assertTrue(
            all(result["nan_rejected"] and result["updates"] == 2 for result in results)
        )
        self.assertTrue(
            all(result["max_parameter_error"] < 1e-12 for result in results)
        )
        self.assertEqual(results[0]["learning_rate"], results[1]["learning_rate"])


if __name__ == "__main__":
    unittest.main()
