"""真实新快照上的一步损失分支反事实；默认CPU，绝不连接训练器或执行仿真。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .config import Config
from .networks import EstNet
from .ppo_diagnostics import full_rollout_kl

BRANCHES = ('full', 'policy', 'velocity', 'entropy', 'value', 'zero_current_gradient')
FIELDS = ('history', 'obs', 'command', 'critic', 'velocity', 'action', 'old_log_prob',
          'old_value', 'old_mean', 'old_std', 'returns', 'advantages')


def tree_sha256(value: Any) -> str:
    """哈希张量内容、形状、dtype及嵌套结构；不依赖torch.save容器的非确定元数据。"""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(json.dumps(['tensor', str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b'{')
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
            digest.update(b'}')
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b'[')
            for part in item:
                visit(part)
            digest.update(b']')
        else:
            digest.update(json.dumps([type(item).__name__, item], allow_nan=False).encode())
    visit(value)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def _losses(model, data, cfg):
    """和本版PPO相同的四项损失；advantages已经整rollout归一化，不能再次归一化。"""
    distribution, velocity = model(data['history'], data['obs'], data['command'])
    ratio = torch.exp(distribution.log_prob(data['action']).sum(-1) - data['old_log_prob'])
    policy = -torch.minimum(ratio * data['advantages'],
        ratio.clamp(1-cfg.clip, 1+cfg.clip) * data['advantages']).mean()
    value = model.value(data['critic'])
    clipped = data['old_value'] + (value-data['old_value']).clamp(-cfg.clip, cfg.clip)
    value_loss = torch.maximum((value-data['returns']).square(),
        (clipped-data['returns']).square()).mean()
    velocity_loss = F.mse_loss(velocity, data['velocity'])
    entropy = distribution.entropy().sum(-1).mean()
    return {'policy': policy, 'value': cfg.value_coef*value_loss,
        'velocity': cfg.velocity_coef*velocity_loss, 'entropy': -cfg.entropy_coef*entropy}


@torch.no_grad()
def _baseline_outputs(model, data, chunk_size):
    means, stds, velocities = [], [], []
    for begin in range(0, len(data['action']), chunk_size):
        sl = slice(begin, begin+chunk_size)
        dist, velocity = model(data['history'][sl], data['obs'][sl], data['command'][sl])
        means.append(dist.mean.detach().clone())
        stds.append(dist.stddev.detach().clone())
        velocities.append(velocity.detach().cpu().double())
    return torch.cat(means), torch.cat(stds), torch.cat(velocities)


@torch.no_grad()
def _estimator_change(model, data, initial_velocity, chunk_size):
    count = len(data['action'])
    squares = torch.zeros(3, dtype=torch.float64)
    signed = torch.zeros_like(squares)
    absolute = torch.zeros_like(squares)
    maximum = torch.zeros_like(squares)
    for begin in range(0, count, chunk_size):
        stop = min(begin+chunk_size, count)
        current = model.estimate(data['history'][begin:stop]).cpu().double()
        if not torch.isfinite(current).all():
            return {'finite': False, 'samples': count, 'reason': 'nonfinite_estimator_output'}
        delta = current-initial_velocity[begin:stop]
        squares += delta.square().sum(0)
        signed += delta.sum(0)
        absolute += delta.abs().sum(0)
        maximum = torch.maximum(maximum, delta.abs().amax(0))
    return {'finite': True, 'samples': count, 'mean_change_xyz_mps': (signed/count).tolist(),
        'rmse_change_xyz_mps': (squares/count).sqrt().tolist(),
        'mean_abs_change_xyz_mps': (absolute/count).tolist(),
        'max_abs_change_xyz_mps': maximum.tolist()}


def _parameter_change(model, initial):
    groups = {}
    for name, parameter in model.named_parameters():
        group = name.split('.')[0]
        delta = parameter.detach().cpu().double()-initial[name].cpu().double()
        row = groups.setdefault(group, {'sum_squares': 0., 'max_abs': 0., 'elements': 0, 'finite': True})
        if not torch.isfinite(delta).all():
            row['finite'] = False
            continue
        row['sum_squares'] += float(delta.square().sum())
        row['max_abs'] = max(row['max_abs'], float(delta.abs().max()))
        row['elements'] += delta.numel()
    for row in groups.values():
        square = row.pop('sum_squares')
        row['l2'] = math.sqrt(square) if row['finite'] else None
        if not row['finite']:
            row['max_abs'] = None
    return groups


def run_counterfactuals(snapshot, *, device='cpu', limit_samples=None, chunk_size=1024,
                       cpu_threads=8, branches=BRANCHES, output_dir=None):
    """从同一初始模型/Adam做各一次候选步；无KL回滚和LR调度。

    only表示当前loss梯度只来自一项；其余参数显式零梯度，仍保留Adam历史动量。
    limit_samples只减少用于反向传播的数据，KL仍遍历整个已保存snapshot观测集。
    分块均值按样本数加权，再统一global clip；不改变原adv，不把分块视为多次Adam。
    """
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 8:
        raise ValueError('cpu_threads必须为1..8')
    torch.set_num_threads(cpu_threads)
    if snapshot.get('schema') != 'estnet-fixed-lr-first-minibatch-replay-v1':
        raise ValueError('必须使用本次真实fixed-LR minibatch快照schema')
    if snapshot.get('advantages_normalized_over_full_rollout') is not True:
        raise ValueError('快照必须含已整rollout归一化的advantages')
    for name in ('iteration', 'epoch', 'minibatch', 'full_rollout_count'):
        if type(snapshot.get(name)) is not int or snapshot[name] < 1:
            raise ValueError(f'快照{name}必须为正整数')
    cfg = Config(**snapshot['cfg'])
    cfg.validate()
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('chunk_size必须为正整数')
    if not branches or len(set(branches)) != len(branches) or any(x not in BRANCHES for x in branches):
        raise ValueError('无效或重复的反事实分支')
    target = torch.device(device)
    if target.type not in ('cpu', 'cuda') or target.type == 'cuda' and target.index is None:
        raise ValueError('device须为cpu或明确cuda:N；工具不会自行选择空卡')
    if target.type == 'cuda':
        torch.cuda.set_device(target)
    initial_model = copy.deepcopy(snapshot['model'])
    initial_adam = copy.deepcopy(snapshot['optimizer'])
    if not isinstance(initial_adam, dict) or not initial_adam.get('param_groups'):
        raise ValueError('需要完整Adam原state_dict，不能用仅模型快照')
    rates = [group.get('lr') for group in initial_adam['param_groups']]
    if any(not isinstance(x, (float, int)) or isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in rates):
        raise ValueError('快照Adam LR必须为有限正数')
    data = {key: snapshot['data'][key].detach().to(target) for key in FIELDS}
    count = len(data['action'])
    if count < 1 or count > snapshot['full_rollout_count']:
        raise ValueError('快照样本数超出完整rollout范围')
    indices = snapshot['indices']
    if (not isinstance(indices, torch.Tensor) or indices.shape != (count,)
            or indices.dtype != torch.int64 or indices.unique().numel() != count
            or int(indices.min()) < 0 or int(indices.max()) >= snapshot['full_rollout_count']):
        raise ValueError('快照indices与真实minibatch不一致')
    shapes = {'history': (count, cfg.history_steps, cfg.obs_dim), 'obs': (count, cfg.obs_dim),
        'command': (count, cfg.command_dim), 'critic': (count, cfg.critic_dim),
        'velocity': (count, 3), 'action': (count, cfg.action_dim),
        'old_mean': (count, cfg.action_dim), 'old_std': (count, cfg.action_dim)}
    for key, tensor in data.items():
        expected = shapes.get(key, (count,))
        if tensor.shape != expected or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f'快照{key}形状/dtype/有限性无效')
    if not (data['old_std'] > 0).all():
        raise ValueError('old_std必须为正')
    if limit_samples is not None and (type(limit_samples) is not int or not 1 <= limit_samples <= count):
        raise ValueError('limit_samples须为1..快照样本数')
    used = count if limit_samples is None else limit_samples

    def fresh():
        # 构造网络产生的CPU随机初始化立即被覆盖，且不污染调用者CPU RNG。
        with torch.random.fork_rng(devices=[]):
            model = EstNet(cfg).to(target)
        model.load_state_dict(initial_model, strict=True)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        optimizer.load_state_dict(copy.deepcopy(initial_adam))
        return model, optimizer

    base, optimizer = fresh()
    initial_model_hash = tree_sha256(base.state_dict())
    initial_adam_hash = tree_sha256(optimizer.state_dict())
    mean, std, initial_velocity = _baseline_outputs(base, data, chunk_size)
    initial_kl = full_rollout_kl(base, data, chunk_size)
    if not initial_kl['finite']:
        raise ValueError('初始快照模型分布非有限')
    snapshot_reference = {**data, 'old_mean': mean, 'old_std': std}
    del base, optimizer
    report = {'schema': 'estnet-fixed-lr-one-step-counterfactual-v1', 'status': 'running',
        'branches_requested': list(branches),
        'snapshot_iteration': snapshot['iteration'], 'snapshot_epoch': snapshot['epoch'],
        'snapshot_minibatch': snapshot['minibatch'], 'snapshot_total_optimizer_steps': snapshot.get('total_optimizer_steps'),
        'full_training_rollout_count': snapshot['full_rollout_count'], 'snapshot_samples': count,
        'optimization_samples': used, 'kl_observation_samples': count, 'subset_conditioned_experiment': used != count,
        'sample_selection': '快照原顺序前limit_samples个；不重新shuffle或归一化adv',
        'indices_sha256': tree_sha256(indices), 'optimization_indices_sha256': tree_sha256(indices[:used]),
        'device': str(target), 'cpu_threads': cpu_threads, 'backward_chunk_size': chunk_size,
        'initial_model_sha256': initial_model_hash, 'initial_adam_sha256': initial_adam_hash,
        'initial_learning_rates': rates, 'initial_kl_on_snapshot': initial_kl,
        'scope': '全部已保存snapshot观测上的KL，不是完整训练rollout的KL护栏验收；没有环境推进或历史415重放',
        'adam_semantics': '所有分支保留同一Adam历史；未连接的当前loss梯度填0，不用None跳过参数。zero_current_gradient仍可因历史动量/weight_decay改变权重。',
        'update_semantics': '每分支单次Adam+原global gradient clip+log_std clamp；不回滚、不改变LR、不更新训练计数',
        'branches': {}}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(output_dir/'manifest.json', {key: value for key, value in report.items() if key != 'branches'})
    for branch in branches:
        model, optimizer = fresh()
        start_model_hash = tree_sha256(model.state_dict())
        start_adam_hash = tree_sha256(optimizer.state_dict())
        if start_model_hash != initial_model_hash or start_adam_hash != initial_adam_hash:
            raise RuntimeError('反事实初始状态不一致')
        optimizer.zero_grad(set_to_none=True)
        components = {key: 0. for key in ('policy', 'value', 'velocity', 'entropy')}
        for begin in range(0, used, chunk_size):
            stop = min(begin+chunk_size, used)
            pieces = _losses(model, {key: tensor[begin:stop] for key, tensor in data.items()}, cfg)
            if any(not torch.isfinite(term) for term in pieces.values()):
                raise FloatingPointError('初始loss分支存在非有限值')
            weight = (stop-begin)/used
            for key, term in pieces.items():
                components[key] += float(term.detach())*weight
            if branch != 'zero_current_gradient':
                objective = (pieces['policy']+pieces['value']+pieces['entropy']+pieces['velocity']) if branch == 'full' else pieces[branch]
                (objective*weight).backward()
        # 用显式0梯度留下其他模块的历史动量影响，才能与full的同一Adam步骤比较。
        filled = 0
        for parameter in model.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
                filled += 1
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        model.clamp_std_()
        kl = full_rollout_kl(model, data, chunk_size)
        from_snapshot = full_rollout_kl(model, snapshot_reference, chunk_size)
        result = {'initial_model_sha256': start_model_hash, 'initial_adam_sha256': start_adam_hash,
            'learning_rates': [g['lr'] for g in optimizer.param_groups], 'weighted_loss_components': components,
            'current_objective': sum(components.values()) if branch == 'full' else components.get(branch, 0.),
            'preclip_gradient_norm': float(norm), 'parameters_receiving_explicit_zero_gradient': filled,
            'kl_rollout_old_to_candidate_on_snapshot': kl,
            'kl_snapshot_model_to_candidate_on_snapshot': from_snapshot,
            'estimator_change': _estimator_change(model, data, initial_velocity, chunk_size),
            'model_sha256_after': tree_sha256(model.state_dict()),
            'adam_sha256_after': tree_sha256(optimizer.state_dict()),
            'parameter_change_from_initial': _parameter_change(model, initial_model)}
        report['branches'][branch] = result
        if output_dir is not None:
            _write_json(output_dir/f'branch-{branch}.json', result)
            _write_json(output_dir/'result.json', report)
        print(json.dumps({'event': 'counterfactual_branch_complete', 'branch': branch,
            'snapshot_kl': kl['total'], 'kl_samples': count, 'gradient_samples': used}, allow_nan=False), flush=True)
        del model, optimizer
    report['status'] = 'completed'
    report['all_branch_candidates_finite'] = all(
        row['kl_rollout_old_to_candidate_on_snapshot']['finite'] and row['estimator_change']['finite']
        for row in report['branches'].values())
    if output_dir is not None:
        _write_json(output_dir/'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True, help='真实新训练保存的replay_minibatch.pt')
    parser.add_argument('--output-dir', type=Path, required=True, help='新建输出目录，拒绝覆盖旧反事实')
    parser.add_argument('--device', default='cpu', help='默认cpu；cuda:N仅在明确获准使用该卡时给出')
    parser.add_argument('--cpu-threads', type=int, default=8, help='1..8，避免CPU线程超出当前任务预算')
    parser.add_argument('--chunk-size', type=int, default=1024, help='前后向分块；仍只有一次Adam，不是多次小batch更新')
    parser.add_argument('--limit-samples', type=int, help='仅梯度用前K条，KL仍用完整snapshot；显式标为子集实验')
    args = parser.parse_args()
    snapshot = torch.load(args.snapshot, map_location='cpu', weights_only=True)
    report = run_counterfactuals(snapshot, device=args.device, limit_samples=args.limit_samples,
        chunk_size=args.chunk_size, cpu_threads=args.cpu_threads, output_dir=args.output_dir)
    report['snapshot_file'] = str(args.snapshot.resolve())
    report['snapshot_file_sha256'] = hashlib.sha256(args.snapshot.read_bytes()).hexdigest()
    _write_json(args.output_dir/'result.json', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
