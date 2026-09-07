"""真实双进程调用 KeyPPO.update，核对恢复的 Adam、全局 KL 和同步模型。"""
import copy
import datetime
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from estnet.key_networks import KeyPolicy
from estnet.key_ppo import KeyPPO
from tests.test_key_learning import key_config, rollout


def check_tree(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            check_tree(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            check_tree(left, right)
    else:
        assert first == second


def worker(rank, init_uri, folder):
    torch.set_num_threads(1)
    # 模拟共同的单卡检查点：先做真实 Adam 更新，再恢复至两 rank。
    snapshots = {}
    for variant in ('key1', 'key2'):
        torch.manual_seed(123)
        cfg = key_config(variant)
        parent = KeyPolicy(cfg)
        parent_ppo = KeyPPO(parent, cfg)
        parent_ppo.update(rollout(parent))
        parent_ppo.updates = 500
        for group in parent_ppo.optimizer.param_groups:
            group['lr'] = 3.9e-5
        snapshots[variant] = (copy.deepcopy(parent.state_dict()),
                              copy.deepcopy(parent_ppo.state_dict()))
    dist.init_process_group('gloo', init_method=init_uri, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=60))
    try:
        results = {}
        for variant, (weights, adam) in snapshots.items():
            cfg = key_config(variant)
            model = KeyPolicy(cfg)
            model.load_state_dict(weights)
            ppo = KeyPPO(model, cfg)
            ppo.load_state_dict(adam)
            assert ppo.updates == 500 and ppo.learning_rate == 3.9e-5
            check_tree(ppo.state_dict(), adam)
            torch.manual_seed(1000 + rank)
            for expected_update in (501, 502):
                # 两 rank 采样不同数据；共同更新后模型和所有 Adam 状态应逐位一致。
                metrics = ppo.update(rollout(model))
                assert ppo.updates == expected_update
                current = {'model': model.state_dict(), 'adam': ppo.state_dict(),
                           'metrics': metrics}
                peers = [None, None]
                dist.all_gather_object(peers, current)
                check_tree(peers[0], peers[1])
            assert any(not torch.equal(weights[k], v) for k, v in model.state_dict().items())
            # 注入一个 rank 的监督 NaN，确认实际 PPO 路径同步拒绝更新。
            bad = rollout(model)
            if rank == 1:
                bad['velocity'][0, 0] = float('nan')
            try:
                ppo.update(bad)
            except FloatingPointError:
                pass
            else:
                raise AssertionError('PPO accepted a non-finite peer loss')
            results[variant] = {'updates': ppo.updates, 'lr': ppo.learning_rate,
                                'peer_model_adam_metrics_equal': True}
        Path(folder, f'ppo-rank-{rank}.json').write_text(json.dumps(results), encoding='utf-8')
    finally:
        dist.destroy_process_group()


class DistributedPPOTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo unavailable')
    def test_key_resume_actual_ppo_keeps_ranks_identical(self):
        with tempfile.TemporaryDirectory(prefix='key-ppo-gloo-') as temporary:
            uri = (Path(temporary) / 'init').as_uri()
            mp.spawn(worker, args=(uri, temporary), nprocs=2, join=True)
            results = [json.loads(Path(temporary, f'ppo-rank-{r}.json').read_text()) for r in range(2)]
        self.assertEqual(results[0], results[1])
        self.assertEqual(set(results[0]), {'key1', 'key2'})


if __name__ == '__main__':
    unittest.main()
