"""纯 Torch 高度图采样；网格规格和参考系都是本复现的工程选择。

论文 arXiv:2403.05868v1 只描述脚附近的小高度图和基座附近的大高度图，
没有给出点数、范围或坐标系。这里的 18 + 81 点不是从论文 P103 反推的。
脚图每脚 3×3，x=(-.1, 0, .1) m、y=(-.05, 0, .05) m；基座图
9×9，x/y 从 -.4 到 .4 m，间隔 .1 m。所有网格仅随基座 yaw 旋转。
输出为采样原点到地面的世界竖直距离，单位米，不截断、不归一化。
"""

from __future__ import annotations

import torch

FOOT_MAP_POINTS = 18
BASE_MAP_POINTS = 81
FOOT_X_OFFSETS = (-0.1, 0.0, 0.1)
FOOT_Y_OFFSETS = (-0.05, 0.0, 0.05)
BASE_XY_OFFSETS = tuple(i / 10.0 for i in range(-4, 5))


def _check_tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int | None, ...],
    reference: torch.Tensor | None = None,
) -> None:
    """在数值运算前拒绝错位、非浮点或非有限输入，避免静默广播。"""
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.ndim != len(shape) or any(
        size is not None and value.shape[i] != size for i, size in enumerate(shape)
    ):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if reference is not None and (
        value.device != reference.device or value.dtype != reference.dtype
    ):
        raise ValueError(f"{name} must share the reference tensor's dtype and device")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _grid_offsets(
    reference: torch.Tensor, x_values: tuple[float, ...], y_values: tuple[float, ...]
) -> torch.Tensor:
    """构造局部水平偏移；x 为外层、y 为内层，z 始终为零。"""
    x, y = torch.meshgrid(
        reference.new_tensor(x_values), reference.new_tensor(y_values), indexing="ij"
    )
    return torch.stack((x.flatten(), y.flatten(), torch.zeros_like(x).flatten()), -1)


def _yaw_rotate(offsets: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """将非零 wxyz 四元数归一化，取 yaw 后绕世界 z 轴旋转网格。"""
    norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    if not bool(torch.isfinite(norm).all()) or bool((norm <= 0).any()):
        raise ValueError("base_quat_w must have a finite nonzero norm")
    w, x, y, z = (quat / norm).unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    cosine, sine = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    return torch.stack(
        (
            cosine * offsets[:, 0] - sine * offsets[:, 1],
            sine * offsets[:, 0] + cosine * offsets[:, 1],
            offsets[:, 2].expand(quat.shape[0], -1),
        ),
        dim=-1,
    )


def make_heightmap_points(
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
    foot_pos_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回脚图 [N,18,3] 和基座图 [N,81,3] 的世界采样原点。

    输入依次为 [N,3]、wxyz [N,4] 和 [N,2,3]。脚顺序必须为左、右，
    输出先左脚九点，再右脚九点；各图按 x 外层、y 内层排列。脚图的原点
    z 是各脚踝 link 高度，基座图的原点 z 是基座高度。roll/pitch 不倾斜网格。
    """
    _check_tensor("base_pos_w", base_pos_w, (None, 3))
    n = base_pos_w.shape[0]
    _check_tensor("base_quat_w", base_quat_w, (n, 4), base_pos_w)
    _check_tensor("foot_pos_w", foot_pos_w, (n, 2, 3), base_pos_w)
    foot_offsets = _yaw_rotate(
        _grid_offsets(base_pos_w, FOOT_X_OFFSETS, FOOT_Y_OFFSETS), base_quat_w
    )
    base_offsets = _yaw_rotate(
        _grid_offsets(base_pos_w, BASE_XY_OFFSETS, BASE_XY_OFFSETS), base_quat_w
    )
    # 先为每只脚独立平移，再展平脚维度，保留明确的左/右顺序。
    foot_points = (foot_pos_w[:, :, None, :] + foot_offsets[:, None, :, :]).reshape(
        n, FOOT_MAP_POINTS, 3
    )
    base_points = base_pos_w[:, None, :] + base_offsets
    _check_tensor("foot_points_w", foot_points, (n, FOOT_MAP_POINTS, 3))
    _check_tensor("base_points_w", base_points, (n, BASE_MAP_POINTS, 3))
    return foot_points, base_points


def relative_heights(
    points_w: torch.Tensor, ground_heights: torch.Tensor
) -> torch.Tensor:
    """按点计算原点 z − 地面 z，返回 [N,P]，保留逐点地形变化。

    ground_heights 必须是对每个世界 x/y 位置实际查询得到的地面高度 [N,P]。
    该函数本身不查询地形；输入非平坦地面时不可用一个常数替代逐点查询。
    允许负距离，以便显露穿透或参考高度错误，而不是用截断隐藏它们。
    """
    _check_tensor("points_w", points_w, (None, None, 3))
    _check_tensor("ground_heights", ground_heights, tuple(points_w.shape[:2]), points_w)
    heights = points_w[..., 2] - ground_heights
    _check_tensor("relative_heights", heights, tuple(points_w.shape[:2]))
    return heights


def flat_heightmaps(
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
    foot_pos_w: torch.Tensor,
    ground_z_per_env: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """仅为水平平面做精确解析地面查询，返回脚图 [N,18]、基座图 [N,81]。

    每个环境有一个水平地面 z，输入严格为 [N]，支持环境间地面高度不同。
    这不是粗糙地形或坡面的近似查询：更换地形后必须在 make_heightmap_points
    的逐点世界位置查询真实地面，再调用 relative_heights。
    """
    foot_points, base_points = make_heightmap_points(
        base_pos_w, base_quat_w, foot_pos_w
    )
    _check_tensor(
        "ground_z_per_env", ground_z_per_env, (base_pos_w.shape[0],), base_pos_w
    )
    foot_ground = ground_z_per_env[:, None].expand(-1, FOOT_MAP_POINTS)
    base_ground = ground_z_per_env[:, None].expand(-1, BASE_MAP_POINTS)
    return relative_heights(foot_points, foot_ground), relative_heights(
        base_points, base_ground
    )
