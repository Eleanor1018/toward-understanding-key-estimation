"""Unitree G1 29-DOF 的 PPO 训练入口。

固定环境：Isaac Sim 4.5、PyTorch 2.5.1、CUDA 12.1。

环境需要返回五组观测：
    history、obs、command、privileged、explicit_target
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from config import ModelConfig
from policy import Actor, Critic, Decoder, Encoder
from ppo import (
    DiagonalGaussian,
    clipped_policy_loss,
    generalized_advantage_estimation,
    normalize_advantages,
    total_ppo_loss,
    value_loss,
)


OBSERVATION_KEYS = (
    "history",
    "obs",
    "command",
    "privileged",
    "explicit_target",
)


def parse_args_and_launch_simulator() -> tuple[argparse.Namespace, Any]:
    """读取训练参数并启动 Isaac Sim。"""

    from isaaclab.app import AppLauncher

    config = ModelConfig()

    parser = argparse.ArgumentParser(
        description="训练 Unitree G1 29-DOF 策略。"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Unitree-G1-29dof-KeyEstimation-v0",
        help="Isaac Lab 任务名称。",
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=config.discount_gamma)
    parser.add_argument("--gae-lambda", type=float, default=config.gae_lambda)
    parser.add_argument(
        "--policy-clip",
        type=float,
        default=config.ppo_clip_epsilon,
    )
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-loss-coef", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--prediction-loss-coef", type=float, default=2.0)
    parser.add_argument("--estimation-loss-coef", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--initial-action-std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/g1_key_estimation"),
    )
    # 添加 --headless、--device 等 Isaac Lab 参数。
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    return args, app_launcher.app


def create_environment(args: argparse.Namespace) -> Any:
    """创建并行仿真环境。"""

    # 必须在 Isaac Sim 启动后导入。
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401 -- 注册 Isaac Lab 任务
    from isaaclab_tasks.utils import parse_env_cfg
    import g1_env  # noqa: F401 -- 注册自定义任务

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
    )
    env_cfg.seed = args.seed
    return gym.make(args.task, cfg=env_cfg)


def get_observation_batch(
    observation: Any,
) -> dict[str, torch.Tensor]:
    """取出网络需要的五组观测。"""

    return {key: observation[key] for key in OBSERVATION_KEYS}


def clone_for_rollout(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """复制当前观测，避免被下一步仿真覆盖。"""

    return {name: tensor.detach().clone() for name, tensor in batch.items()}


@torch.no_grad()
def collect_rollout(
    env: Any,
    observation: Any,
    encoder: Encoder,
    actor: Actor,
    critic: Critic,
    action_distribution: DiagonalGaussian,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
) -> tuple[dict[str, torch.Tensor], Any]:
    """让所有并行环境运行一段 rollout。"""

    storage: dict[str, list[torch.Tensor]] = {
        key: [] for key in OBSERVATION_KEYS
    }
    for key in (
        "next_obs",
        "prediction_mask",
        "actions",
        "old_log_prob",
        "old_values",
        "rewards",
        "dones",
    ):
        storage[key] = []

    for _ in range(rollout_steps):
        batch = get_observation_batch(observation)
        saved_batch = clone_for_rollout(batch)

        latent, explicit_estimate = encoder(saved_batch["history"])
        action_mean = actor(
            saved_batch["obs"],
            saved_batch["command"],
            latent,
            explicit_estimate,
        )
        actions, log_prob = action_distribution.sample(action_mean)
        values = critic(
            saved_batch["obs"],
            saved_batch["command"],
            saved_batch["privileged"],
        )

        next_observation, rewards, terminated, truncated, _ = env.step(actions)
        next_batch = get_observation_batch(next_observation)

        rewards = rewards.reshape(-1)
        terminated = terminated.reshape(-1)
        truncated = truncated.reshape(-1)
        dones = torch.logical_or(terminated, truncated)

        for key in OBSERVATION_KEYS:
            storage[key].append(saved_batch[key])
        storage["next_obs"].append(next_batch["obs"].detach().clone())
        # episode 结束后会自动 reset，新的 obs 不能作为预测目标。
        storage["prediction_mask"].append((~dones).to(torch.float32))
        storage["actions"].append(actions.detach().clone())
        storage["old_log_prob"].append(log_prob.detach().clone())
        storage["old_values"].append(values.detach().clone())
        storage["rewards"].append(rewards.detach().clone())
        storage["dones"].append(dones.detach().clone())

        observation = next_observation

    final_batch = get_observation_batch(observation)
    last_values = critic(
        final_batch["obs"],
        final_batch["command"],
        final_batch["privileged"],
    )

    rollout = {name: torch.stack(items) for name, items in storage.items()}
    advantages, returns = generalized_advantage_estimation(
        rewards=rollout["rewards"],
        values=rollout["old_values"],
        dones=rollout["dones"],
        last_value=last_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    rollout["advantages"] = advantages
    rollout["returns"] = returns
    return rollout, observation


def flatten_time_and_env(tensor: torch.Tensor) -> torch.Tensor:
    """[时间, 环境, ...] → [时间 × 环境, ...]。"""

    return tensor.flatten(start_dim=0, end_dim=1)


def update_models(
    rollout: dict[str, torch.Tensor],
    encoder: Encoder,
    decoder: Decoder,
    actor: Actor,
    critic: Critic,
    action_distribution: DiagonalGaussian,
    optimizer: torch.optim.Optimizer,
    trainable_parameters: list[nn.Parameter],
    args: argparse.Namespace,
) -> dict[str, float]:
    """打乱 rollout，并执行多轮 PPO 更新。"""

    flat = {name: flatten_time_and_env(value) for name, value in rollout.items()}
    flat["advantages"] = normalize_advantages(flat["advantages"])
    sample_count = flat["actions"].shape[0]

    metric_sums = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "prediction_loss": 0.0,
        "estimation_loss": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
    }
    update_count = 0

    for _ in range(args.update_epochs):
        # 同一批 rollout 重复使用，每轮重新打乱。
        shuffled_indices = torch.randperm(
            sample_count,
            device=flat["actions"].device,
        )

        for indices in torch.tensor_split(shuffled_indices, args.num_minibatches):
            # 重新前向计算；旧概率和旧价值保持不变，作为 PPO 的参照。
            latent, explicit_estimate = encoder(flat["history"][indices])
            predicted_obs = decoder(latent, explicit_estimate)
            action_mean = actor(
                flat["obs"][indices],
                flat["command"][indices],
                latent,
                explicit_estimate,
            )
            new_log_prob = action_distribution.log_prob(
                action_mean,
                flat["actions"][indices],
            )
            entropy = action_distribution.entropy(action_mean)
            new_values = critic(
                flat["obs"][indices],
                flat["command"][indices],
                flat["privileged"][indices],
            )

            actor_loss, probability_ratio = clipped_policy_loss(
                new_log_prob=new_log_prob,
                old_log_prob=flat["old_log_prob"][indices],
                advantages=flat["advantages"][indices],
                clip_epsilon=args.policy_clip,
            )
            critic_loss = value_loss(
                new_values=new_values,
                old_values=flat["old_values"][indices],
                returns=flat["returns"][indices],
                clip_epsilon=args.value_clip,
            )

            # Decoder 预测下一步 obs，并忽略 episode 结束处。
            prediction_error = F.mse_loss(
                predicted_obs,
                flat["next_obs"][indices],
                reduction="none",
            ).mean(dim=-1)
            prediction_mask = flat["prediction_mask"][indices]
            valid_prediction_count = prediction_mask.sum().clamp_min(1.0)
            prediction_loss = (
                prediction_error * prediction_mask
            ).sum() / valid_prediction_count

            estimation_loss = F.mse_loss(
                explicit_estimate,
                flat["explicit_target"][indices],
            )

            # PPO 学习动作，两个辅助损失训练 Encoder 和 Decoder。
            ppo_loss = total_ppo_loss(
                policy_loss=actor_loss,
                critic_loss=critic_loss,
                entropy=entropy,
                value_loss_coefficient=args.value_loss_coef,
                entropy_coefficient=args.entropy_coef,
            )
            loss = (
                ppo_loss
                + args.prediction_loss_coef * prediction_loss
                + args.estimation_loss_coef * estimation_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio = new_log_prob - flat["old_log_prob"][indices]
                approximate_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (probability_ratio - 1.0).abs() > args.policy_clip
                ).to(torch.float32).mean()

            batch_metrics = {
                "loss": loss,
                "policy_loss": actor_loss,
                "value_loss": critic_loss,
                "entropy": entropy.mean(),
                "prediction_loss": prediction_loss,
                "estimation_loss": estimation_loss,
                "approx_kl": approximate_kl,
                "clip_fraction": clip_fraction,
            }
            for name, value in batch_metrics.items():
                metric_sums[name] += value.detach().item()
            update_count += 1

    return {
        name: total / update_count
        for name, total in metric_sums.items()
    }


def save_checkpoint(
    path: Path,
    iteration: int,
    config: ModelConfig,
    encoder: Encoder,
    decoder: Decoder,
    actor: Actor,
    critic: Critic,
    action_distribution: DiagonalGaussian,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    """保存模型与优化器。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "model_config": asdict(config),
            "train_args": dict(vars(args)),
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "action_distribution": action_distribution.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def run_training(args: argparse.Namespace) -> None:
    """循环执行采样和网络更新。"""

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = ModelConfig()
    env = create_environment(args)
    observation, _ = env.reset(seed=args.seed)
    device = torch.device(args.device)

    encoder = Encoder(config).to(device)
    decoder = Decoder(config).to(device)
    actor = Actor(config).to(device)
    critic = Critic(config).to(device)
    action_distribution = DiagonalGaussian(
        config.action_dim,
        initial_std=args.initial_action_std,
    ).to(device)

    modules: tuple[nn.Module, ...] = (
        encoder,
        decoder,
        actor,
        critic,
        action_distribution,
    )
    trainable_parameters = [
        parameter
        for module in modules
        for parameter in module.parameters()
    ]
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=args.learning_rate,
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Training {args.task} on {device}: {args.num_envs} envs, "
        f"{args.rollout_steps} steps per rollout."
    )

    for iteration in range(1, args.iterations + 1):
        rollout, observation = collect_rollout(
            env=env,
            observation=observation,
            encoder=encoder,
            actor=actor,
            critic=critic,
            action_distribution=action_distribution,
            rollout_steps=args.rollout_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        metrics = update_models(
            rollout=rollout,
            encoder=encoder,
            decoder=decoder,
            actor=actor,
            critic=critic,
            action_distribution=action_distribution,
            optimizer=optimizer,
            trainable_parameters=trainable_parameters,
            args=args,
        )

        if iteration == 1 or iteration % 10 == 0:
            mean_reward = rollout["rewards"].mean().item()
            print(
                f"iteration={iteration:05d} "
                f"reward/step={mean_reward:+.4f} "
                f"loss={metrics['loss']:.4f} "
                f"policy={metrics['policy_loss']:.4f} "
                f"value={metrics['value_loss']:.4f} "
                f"prediction={metrics['prediction_loss']:.4f} "
                f"estimation={metrics['estimation_loss']:.4f} "
                f"kl={metrics['approx_kl']:.6f}"
            )

        if iteration % args.save_interval == 0 or iteration == args.iterations:
            save_checkpoint(
                path=args.log_dir / f"checkpoint_{iteration:05d}.pt",
                iteration=iteration,
                config=config,
                encoder=encoder,
                decoder=decoder,
                actor=actor,
                critic=critic,
                action_distribution=action_distribution,
                optimizer=optimizer,
                args=args,
            )
    env.close()


if __name__ == "__main__":
    cli_args, simulation_app = parse_args_and_launch_simulator()
    try:
        run_training(cli_args)
    finally:
        simulation_app.close()
