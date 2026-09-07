import copy
import math
import unittest
from types import SimpleNamespace

import torch

from estnet.networks import EstNet
from estnet.ppo import PPO, compute_gae, gaussian_kl


def model_config():
    return SimpleNamespace(
        obs_dim=42, command_dim=7, critic_dim=61, action_dim=12, history_steps=50,
        encoder_hidden=(16, 8), actor_hidden=(16, 8), critic_hidden=(16, 8), init_std=0.8,
    )


def ppo_config(**overrides):
    fields = dict(
        learning_rate=5e-4, epochs=2, minibatches=2, gamma=.996, gae_lambda=.95,
        clip=.2, value_coef=1., entropy_coef=.008, velocity_coef=1.,
        max_grad_norm=1., desired_kl=.01, min_learning_rate=1e-5, max_learning_rate=1e-3,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def make_observation(count=16):
    return {
        "history": torch.randn(count, 50, 42), "obs": torch.randn(count, 42),
        "command": torch.randn(count, 7), "critic": torch.randn(count, 61),
        "velocity": torch.randn(count, 3),
    }


def make_rollout(model, count=16):
    observation = make_observation(count)
    action, log_prob, value, mean, std = model.act(observation)
    return dict(observation, action=action, old_log_prob=log_prob, old_value=value,
                old_mean=mean, old_std=std, returns=value + torch.randn(count),
                advantages=torch.randn(count))


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(23)

    def test_estnet_has_only_explicit_estimator_actor_and_critic(self):
        model = EstNet(model_config())
        observation = make_observation()
        self.assertEqual(model.estimate(observation["history"]).shape, (16, 3))
        self.assertEqual(set(dict(model.named_children())), {"estimator", "actor", "critic"})
        self.assertFalse(any("latent" in key or "decoder" in key for key in model.state_dict()))
        action, log_prob, value, mean, std = model.act(observation)
        self.assertEqual(action.shape, (16, 12))
        self.assertEqual(log_prob.shape, (16,))
        self.assertEqual(value.shape, (16,))
        distribution = model.distribution(observation["history"], observation["obs"], observation["command"])
        torch.testing.assert_close(log_prob, distribution.log_prob(action).sum(-1))
        self.assertTrue((action.abs() > 1).any())  # raw Gaussian, not hidden tanh
        torch.testing.assert_close(std, torch.full_like(std, .8))

    def test_ppo_policy_signal_reaches_estimator(self):
        model = EstNet(model_config())
        observation = make_observation()
        distribution = model.distribution(observation["history"], observation["obs"], observation["command"])
        action = distribution.mean.detach() + 0.4
        (-distribution.log_prob(action).sum(-1).mean()).backward()
        gradient = sum(p.grad.abs().sum().item() for p in model.estimator.parameters())
        self.assertGreater(gradient, 0.)

    def test_velocity_mse_trains_estimator_without_actor_gradient(self):
        model = EstNet(model_config())
        observation = make_observation()
        loss = (model.estimate(observation["history"]) - observation["velocity"]).square().mean()
        loss.backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.estimator.parameters()), 0.)
        self.assertTrue(all(p.grad is None for p in model.actor.parameters()))

    def test_gae_timeout_bootstrap_and_trace_separation(self):
        # The huge third step must not leak backward across the second timeout.
        rewards = torch.tensor([[1.], [2.], [1000.]])
        values = torch.tensor([[10.], [20.], [3000.]])
        next_values = torch.tensor([[20.], [30.], [4000.]])
        terminal = torch.zeros(3, 1, dtype=torch.bool)
        truncated = torch.tensor([[False], [True], [False]])
        advantage, returns = compute_gae(rewards, values, next_values, terminal, truncated, 1., 1.)
        torch.testing.assert_close(advantage[:, 0], torch.tensor([23., 12., 2000.]))
        torch.testing.assert_close(returns[:, 0], torch.tensor([33., 32., 5000.]))
        terminal[1] = True
        next_values[1] = float("nan")  # termination must not evaluate a reset value
        _, returns = compute_gae(rewards, values, next_values, terminal, truncated, 1., 1.)
        torch.testing.assert_close(returns[:2, 0], torch.tensor([3., 2.]))

    def test_gae_regular_discount_and_lambda(self):
        rewards = torch.tensor([[1.], [2.]])
        values = torch.tensor([[2.], [3.]])
        next_values = torch.tensor([[3.], [4.]])
        boundary = torch.zeros(2, 1, dtype=torch.bool)
        advantages, returns = compute_gae(rewards, values, next_values, boundary, boundary, .9, .5)
        torch.testing.assert_close(advantages[:, 0], torch.tensor([2.87, 2.6]))
        torch.testing.assert_close(returns[:, 0], torch.tensor([4.87, 5.6]))

    def test_finite_joint_update_and_optimizer_roundtrip(self):
        model = EstNet(model_config())
        learner = PPO(model, ppo_config())
        batch = make_rollout(model)
        originals = {key: tensor.clone() for key, tensor in batch.items()}
        before = copy.deepcopy(model.state_dict())
        metrics = learner.update(batch)
        self.assertTrue(all(isinstance(value, float) and math.isfinite(value) for value in metrics.values()))
        self.assertEqual(metrics["gradient_steps"], 4.)
        for name in ("estimator", "actor", "critic"):
            self.assertTrue(any(not torch.equal(before[key], tensor) for key, tensor in model.state_dict().items() if key.startswith(name)))
        for key, tensor in batch.items():
            torch.testing.assert_close(tensor, originals[key])
        clone = EstNet(model_config())
        clone.load_state_dict(model.state_dict())
        resumed = PPO(clone, ppo_config())
        resumed.load_state_dict(copy.deepcopy(learner.state_dict()))
        self.assertEqual(resumed.updates, 1)
        self.assertEqual(resumed.learning_rate, learner.learning_rate)
        self.assertEqual(len(resumed.optimizer.state), len(learner.optimizer.state))
        # A resumed Adam step must reproduce an uninterrupted step exactly.
        next_batch = make_rollout(model)
        torch.manual_seed(41)
        learner.update(next_batch)
        torch.manual_seed(41)
        resumed.update(next_batch)
        for left, right in zip(model.parameters(), clone.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_actual_update_velocity_only_reduces_mse(self):
        model = EstNet(model_config())
        learner = PPO(model, ppo_config(epochs=1, minibatches=1, value_coef=0., entropy_coef=0., desired_kl=0.))
        batch = make_rollout(model)
        batch["advantages"].zero_()
        before = (model.estimate(batch["history"]) - batch["velocity"]).square().mean().item()
        learner.update(batch)
        after = (model.estimate(batch["history"]) - batch["velocity"]).square().mean().item()
        self.assertLess(after, before)

    def test_exact_gaussian_kl_and_bidirectional_lr(self):
        old_mean = torch.zeros(2, 12)
        std = torch.ones_like(old_mean)
        torch.testing.assert_close(gaussian_kl(old_mean, std, old_mean, std), torch.zeros(2))
        torch.testing.assert_close(gaussian_kl(old_mean, std, old_mean + .1, std), torch.full((2,), .06))
        learner = PPO(EstNet(model_config()), ppo_config())
        initial = learner.learning_rate
        learner._adapt_learning_rate(.1)
        self.assertLess(learner.learning_rate, initial)
        learner._adapt_learning_rate(.001)
        self.assertAlmostEqual(learner.learning_rate, initial)

    def test_nonfinite_kl_is_not_masked_by_roundoff_clamp(self):
        model = EstNet(model_config())
        learner = PPO(model, ppo_config(epochs=1, minibatches=1))
        batch = make_rollout(model)
        batch["old_mean"][0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "policy KL"):
            learner.update(batch)


if __name__ == "__main__":
    unittest.main()
