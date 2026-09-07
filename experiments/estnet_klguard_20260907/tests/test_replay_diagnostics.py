"""真实tiny EstNet+Adam验证分支相同初始状态、独立更新及历史动量保留；不启动GPU。"""
from dataclasses import replace
import copy
import unittest

import torch

from estnet.config import Config
from estnet.networks import EstNet
from estnet.replay_diagnostics import _losses, run_counterfactuals, tree_sha256


class ReplayTests(unittest.TestCase):
    def test_same_initial_state_and_branch_order_independence_with_adam_momentum(self):
        torch.manual_seed(381)
        cfg = replace(Config(), encoder_hidden=(8,), actor_hidden=(8,), critic_hidden=(8,), num_envs=4)
        model = EstNet(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        n = 8
        data = {'history': torch.randn(n,50,42), 'obs': torch.randn(n,42),
            'command': torch.randn(n,7), 'critic': torch.randn(n,61),
            'velocity': torch.randn(n,3), 'returns': torch.randn(n),
            'advantages': torch.arange(n, dtype=torch.float32)-3.5}
        data['advantages'] /= data['advantages'].std(correction=0)

        def old_policy():
            with torch.no_grad():
                dist = model.distribution(data['history'],data['obs'],data['command'])
                action = dist.sample()
                data.update(action=action,old_mean=dist.mean.clone(),old_std=dist.stddev.clone(),
                    old_log_prob=dist.log_prob(action).sum(-1),old_value=model.value(data['critic']))
        old_policy()
        sum(_losses(model,data,cfg).values()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.max_grad_norm)
        optimizer.step()
        model.clamp_std_()
        old_policy()
        snapshot = {'schema':'estnet-guard-first-minibatch-replay-v1','iteration':2,'epoch':1,'minibatch':1,
            'full_rollout_count':96,'advantages_normalized_over_full_rollout':True,'indices':torch.arange(n),
            'data':copy.deepcopy(data),'model':copy.deepcopy(model.state_dict()),
            'optimizer':copy.deepcopy(optimizer.state_dict()),'cfg':cfg.to_dict(),'total_accepted_steps':1}
        before = tree_sha256(snapshot)
        first = run_counterfactuals(snapshot, branches=('full','entropy','zero_current_gradient'), cpu_threads=2,chunk_size=8)
        second = run_counterfactuals(snapshot, branches=('zero_current_gradient','entropy','full'), cpu_threads=2,chunk_size=8)
        self.assertEqual(before,tree_sha256(snapshot))
        for name,row in first['branches'].items():
            self.assertEqual(row['initial_model_sha256'],first['initial_model_sha256'])
            self.assertEqual(row['initial_adam_sha256'],first['initial_adam_sha256'])
            self.assertEqual(row['model_sha256_after'],second['branches'][name]['model_sha256_after'])
            self.assertEqual(row['adam_sha256_after'],second['branches'][name]['adam_sha256_after'])
            self.assertTrue(row['kl_snapshot_model_to_candidate_on_snapshot']['finite'])
        zero = first['branches']['zero_current_gradient']
        self.assertEqual(zero['preclip_gradient_norm'],0.)
        self.assertGreater(zero['parameter_change_from_initial']['actor']['l2'],0.)
        self.assertEqual(zero['parameter_change_from_initial']['actor'],
            first['branches']['entropy']['parameter_change_from_initial']['actor'])


if __name__ == '__main__':
    unittest.main()
