"""CPU runner contracts with a real EstNet and scripted vector transitions.

These tests exercise bookkeeping and episode boundaries, never robot dynamics.
The fake environment models DirectRLEnv's reset-before-observation ordering and
uses the production History class; it cannot establish simulator API support or
successful walking.
"""

import unittest
from dataclasses import replace

import torch

from estnet.config import Config
from estnet.history import History
from estnet.networks import EstNet
from estnet.run import collect_rollout, evaluate_first_episodes


def tiny_model(num_envs=3, **overrides):
    cfg = replace(
        Config(),
        num_envs=num_envs,
        horizon=3,
        encoder_hidden=(8,),
        actor_hidden=(8,),
        critic_hidden=(),
        **overrides,
    )
    model = EstNet(cfg)
    # Keep the real critic implementation, with a value that can be checked by
    # hand: V(s) is exactly the marker carried in critic coordinate zero.
    with torch.no_grad():
        model.critic[0].weight.zero_()
        model.critic[0].weight[0, 0] = 1.0
        model.critic[0].bias.zero_()
    return cfg, model


class ScriptedVectorEnv:
    """Three streams: timeout, true termination, and ordinary transitions."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = 3
        self.device = torch.device("cpu")
        self.marker = torch.tensor([1.0, 11.0, 21.0])
        self.history = History(3, cfg.history_steps, cfg.obs_dim, self.device)
        self.steps = 0
        self.actions = []
        # Reuse these outputs as a simulator normally does. The runner must
        # retain each transition, not aliases to the most recent values.
        self.reward = torch.zeros(3)
        self.terminal = torch.zeros(3, dtype=torch.bool)
        self.timeout = torch.zeros_like(self.terminal)
        self.final_critic = torch.zeros(3, cfg.critic_dim)

    def observation(self):
        obs = self.marker[:, None].expand(-1, self.cfg.obs_dim).clone()
        critic = torch.zeros(3, self.cfg.critic_dim)
        critic[:, 0] = self.marker
        return {
            "obs": obs,
            "history": self.history.append(obs),
            "command": torch.zeros(3, self.cfg.command_dim),
            "critic": critic,
            "velocity": self.marker[:, None].expand(-1, 3).clone() / 10,
        }

    def step(self, action):
        self.actions.append(action.clone())
        self.steps += 1
        self.marker += 1
        self.final_critic.zero_()
        self.final_critic[:, 0] = self.marker
        self.reward.copy_(torch.tensor([1.0, 10.0, 100.0]) * self.steps)
        self.terminal.zero_()
        self.timeout.zero_()
        reset_id = self.steps - 1
        if self.steps == 2:
            self.terminal[reset_id] = True
        else:
            self.timeout[reset_id] = True
        self.marker[reset_id] = 100.0 * self.steps + 1
        self.history.reset(torch.tensor([reset_id]))
        return (
            self.observation(),
            self.reward,
            self.terminal,
            self.timeout,
            {
                "final_critic": self.final_critic,
                "metrics": {"marker": self.final_critic[:, 0]},
                "reward_terms": {"scripted": self.reward},
            },
        )


class EvaluationVectorEnv:
    """A first-episode failure, survivor, dual boundary, and missing boundary.

    Reset episodes deliberately report huge good metrics and repeated timeouts.
    They must neither repair first-episode statistics nor inflate survival.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = 4
        self.device = torch.device("cpu")
        self.max_episode_length = 8
        self.steps = 0
        self.actions = []
        self.pre_action_observations = []
        self.first_done = torch.zeros(4, dtype=torch.bool)

    def observation(self):
        marker = self.steps / 100
        return {
            "obs": torch.full((4, self.cfg.obs_dim), marker),
            "history": torch.full(
                (4, self.cfg.history_steps, self.cfg.obs_dim), marker
            ),
            "command": torch.full((4, self.cfg.command_dim), marker),
        }

    def step(self, action):
        self.pre_action_observations.append(self.observation())
        self.actions.append(action.clone())
        self.steps += 1
        terminal = torch.zeros(4, dtype=torch.bool)
        timeout = self.first_done.clone()
        terminal[0] = self.steps == 2
        timeout[1] = self.steps == 8
        terminal[2] = self.steps == 3
        timeout[2] |= self.steps == 3
        metrics = {
            "forward_velocity": torch.full((4,), 0.4),
            "command_vx": torch.full((4,), 0.4),
            "velocity_error": torch.zeros(4),
            "double_support": torch.zeros(4),
            "alternating_landing": torch.ones(4),
            "stance_slip": torch.full((4,), 0.01),
            "ankle_height_left": torch.full((4,), 0.04 + 0.05 * (self.steps % 2)),
            "ankle_height_right": torch.full((4,), 0.09 - 0.05 * (self.steps % 2)),
        }
        metrics["forward_velocity"][0] = 0.0
        metrics["velocity_error"][0] = 0.4
        metrics["double_support"][0] = 1.0
        metrics["alternating_landing"][0] = 0.0
        metrics["stance_slip"][0] = 0.2
        # These values would visibly alter counts/means if a reset first episode
        # were accidentally marked active again.
        for name in ("forward_velocity", "alternating_landing"):
            metrics[name][self.first_done] = 100.0
        self.first_done |= terminal | timeout
        return (
            self.observation(),
            torch.zeros(4),
            terminal,
            timeout,
            {"metrics": metrics},
        )


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(71)

    def rollout(self):
        cfg, model = tiny_model(gamma=1.0, gae_lambda=1.0)
        env = ScriptedVectorEnv(cfg)
        initial = env.observation()
        final, batch, summary = collect_rollout(env, model, initial, cfg)
        return cfg, model, env, initial, final, batch, summary

    def test_timeout_bootstrap_uses_actual_final_state_and_stops_gae_trace(self):
        _, _, _, _, _, batch, _ = self.rollout()
        # Columns are environment streams; neither a reset state's V=101/301
        # nor the subsequent episode's rewards may replace the timeout target.
        torch.testing.assert_close(
            batch["next_value"].reshape(3, 3),
            torch.tensor(
                [[2.0, 12.0, 22.0], [102.0, 13.0, 23.0], [103.0, 202.0, 24.0]]
            ),
        )
        torch.testing.assert_close(
            batch["returns"].reshape(3, 3),
            torch.tensor(
                [[3.0, 30.0, 624.0], [108.0, 20.0, 524.0], [106.0, 232.0, 324.0]]
            ),
        )
        torch.testing.assert_close(
            batch["advantages"], batch["returns"] - batch["old_value"]
        )

    def test_rollout_fields_actions_and_likelihoods_share_time_environment_order(self):
        _, model, env, initial, final, batch, summary = self.rollout()
        marker = torch.tensor([1.0, 11.0, 21.0, 101.0, 12.0, 22.0, 102.0, 201.0, 23.0])
        torch.testing.assert_close(batch["obs"][:, 0], marker)
        torch.testing.assert_close(batch["critic"][:, 0], marker)
        torch.testing.assert_close(batch["velocity"][:, 0], marker / 10)
        torch.testing.assert_close(batch["old_value"], marker)
        torch.testing.assert_close(
            batch["action"], torch.stack(env.actions).flatten(0, 1)
        )
        with torch.no_grad():
            distribution = model.distribution(
                batch["history"], batch["obs"], batch["command"]
            )
        torch.testing.assert_close(
            batch["old_log_prob"], distribution.log_prob(batch["action"]).sum(-1)
        )
        torch.testing.assert_close(batch["old_mean"], distribution.mean)
        torch.testing.assert_close(batch["old_std"], distribution.stddev)
        torch.testing.assert_close(
            batch["reward"].reshape(3, 3),
            torch.tensor([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]]),
        )
        self.assertAlmostEqual(summary["reward/total"], 74.0)
        self.assertAlmostEqual(summary["reward/scripted"], 74.0)
        self.assertEqual(
            batch["terminated"].tolist(), [False] * 4 + [True] + [False] * 4
        )
        self.assertEqual(batch["truncated"].tolist(), [True] + [False] * 7 + [True])
        self.assertTrue(all(not tensor.requires_grad for tensor in batch.values()))
        torch.testing.assert_close(
            final["obs"][:, 0], torch.tensor([103.0, 202.0, 301.0])
        )
        torch.testing.assert_close(
            initial["obs"][:, 0], torch.tensor([1.0, 11.0, 21.0])
        )

    def test_history_reset_fills_only_new_episode_and_preserves_old_snapshots(self):
        cfg, _, env, initial, final, batch, _ = self.rollout()
        histories = batch["history"].reshape(3, 3, cfg.history_steps, cfg.obs_dim)
        torch.testing.assert_close(
            histories[1, 0, :, 0], torch.full((cfg.history_steps,), 101.0)
        )
        torch.testing.assert_close(
            histories[2, 1, :, 0], torch.full((cfg.history_steps,), 201.0)
        )
        torch.testing.assert_close(
            histories[2, 2, -3:, 0], torch.tensor([21.0, 22.0, 23.0])
        )
        torch.testing.assert_close(
            final["history"][2, :, 0], torch.full((cfg.history_steps,), 301.0)
        )
        torch.testing.assert_close(
            initial["history"][0, :, 0], torch.ones(cfg.history_steps)
        )
        saved = batch["history"].clone()
        env.history.data.fill_(-999.0)
        torch.testing.assert_close(batch["history"], saved)

    def test_evaluation_cannot_revive_finished_first_episodes_or_count_dual_boundary(
        self,
    ):
        cfg, model = tiny_model(num_envs=4, sim_dt=0.05)
        env = EvaluationVectorEnv(cfg)
        result = evaluate_first_episodes(env, model, env.observation(), cfg)
        self.assertEqual(env.steps, 8)
        self.assertEqual(result["survival_fraction"], 0.25)
        self.assertEqual(result["walking_gate_fraction"], 0.25)
        self.assertAlmostEqual(result["mean_episode_seconds"], 2.625)
        self.assertAlmostEqual(result["mean_alternating_landings"], 4.75)
        self.assertAlmostEqual(result["mean_velocity_ratio"], 0.75)
        self.assertAlmostEqual(result["metrics"]["velocity_error"], 0.1)
        self.assertAlmostEqual(result["metrics"]["forward_velocity"], 0.3)
        self.assertEqual(result["gate_fractions"]["both_feet_excursion"], 0.5)
        # Active first episodes use the actor mean; finished ones receive zero
        # actions so invalid post-reset observations need not enter the actor.
        with torch.no_grad():
            for step, (observation, action) in enumerate(zip(env.pre_action_observations, env.actions)):
                expected = model.distribution(
                    observation["history"], observation["obs"], observation["command"]
                ).mean
                if step >= 2:
                    expected[0] = 0.0
                if step >= 3:
                    expected[2] = 0.0
                torch.testing.assert_close(action, expected)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_evaluation_exits_when_all_first_episodes_finish(self):
        cfg, model = tiny_model(num_envs=4, sim_dt=0.05)
        env = EvaluationVectorEnv(cfg)
        original_step = env.step

        def finish_everyone(action):
            obs, reward, terminal, timeout, info = original_step(action)
            terminal[:] = True
            timeout[:] = True
            return obs, reward, terminal, timeout, info

        env.step = finish_everyone
        result = evaluate_first_episodes(env, model, env.observation(), cfg)
        self.assertEqual(env.steps, 1)
        self.assertEqual(result["survival_fraction"], 0.0)
        self.assertEqual(result["walking_gate_fraction"], 0.0)
        self.assertEqual(result["gate_fractions"]["both_feet_excursion"], 0.0)
        self.assertAlmostEqual(result["mean_episode_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main()
