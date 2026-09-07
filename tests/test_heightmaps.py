"""验证高度图坐标与监督真值；这些纯数值测试不能证明机器人会走路。"""

import math
import unittest

import torch

from estnet.heightmaps import (
    flat_heightmaps,
    make_heightmap_points,
    relative_heights,
)


def sample_inputs():
    """两个环境的世界地面高度不同，左右脚高度也不同。"""
    base = torch.tensor([[1.0, 2.0, 0.8], [-2.0, 1.0, 2.9]], dtype=torch.float64)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float64)
    feet = torch.tensor(
        [[[1.1, 2.1, 0.12], [0.9, 1.9, 0.20]], [[-2.0, 1.1, 2.2], [-2.0, 0.9, 2.3]]],
        dtype=torch.float64,
    )
    ground = torch.tensor([0.0, 2.0], dtype=torch.float64)
    return base, quat, feet, ground


class HeightmapTest(unittest.TestCase):
    def test_grid_extents_centers_spacing_and_left_right_order(self):
        base, quat, feet, _ = sample_inputs()
        foot_points, base_points = make_heightmap_points(base, quat, feet)
        self.assertEqual(foot_points.shape, (2, 18, 3))
        self.assertEqual(base_points.shape, (2, 81, 3))
        self.assertEqual(foot_points.dtype, base.dtype)
        self.assertEqual(base_points.device, base.device)
        for foot, start in ((0, 0), (1, 9)):
            torch.testing.assert_close(foot_points[:, start + 4], feet[:, foot])
            torch.testing.assert_close(
                foot_points[:, start] - feet[:, foot],
                torch.tensor([[-0.1, -0.05, 0.0]] * 2, dtype=base.dtype),
            )
            torch.testing.assert_close(
                foot_points[:, start + 8] - feet[:, foot],
                torch.tensor([[0.1, 0.05, 0.0]] * 2, dtype=base.dtype),
            )
            torch.testing.assert_close(
                foot_points[:, start + 1] - foot_points[:, start],
                torch.tensor([[0.0, 0.05, 0.0]] * 2, dtype=base.dtype),
            )
        torch.testing.assert_close(base_points[:, 40], base)
        torch.testing.assert_close(
            base_points[:, 0] - base,
            torch.tensor([[-0.4, -0.4, 0.0]] * 2, dtype=base.dtype),
        )
        torch.testing.assert_close(
            base_points[:, -1] - base,
            torch.tensor([[0.4, 0.4, 0.0]] * 2, dtype=base.dtype),
        )
        torch.testing.assert_close(
            base_points[:, 1] - base_points[:, 0],
            torch.tensor([[0.0, 0.1, 0.0]] * 2, dtype=base.dtype),
        )

    def test_heading_rotates_offsets_about_each_center_without_rotating_centers(self):
        base, quat, feet, _ = sample_inputs()
        # 第一个环境 yaw=+90°，第二个 yaw=-90°；非单位四元数也应正确归一化。
        quat[0] = torch.tensor([math.sqrt(0.5), 0, 0, math.sqrt(0.5)]) * 2
        quat[1] = torch.tensor([math.sqrt(0.5), 0, 0, -math.sqrt(0.5)])
        foot_points, base_points = make_heightmap_points(base, quat, feet)
        expected_first_foot = torch.tensor(
            [[0.05, -0.1, 0], [-0.05, 0.1, 0]], dtype=base.dtype
        )
        torch.testing.assert_close(foot_points[:, 0] - feet[:, 0], expected_first_foot)
        torch.testing.assert_close(foot_points[:, 9] - feet[:, 1], expected_first_foot)
        torch.testing.assert_close(foot_points[:, 4], feet[:, 0])
        torch.testing.assert_close(foot_points[:, 13], feet[:, 1])
        torch.testing.assert_close(
            base_points[:, 0] - base,
            torch.tensor([[0.4, -0.4, 0], [-0.4, 0.4, 0]], dtype=base.dtype),
        )

    def test_roll_does_not_tilt_horizontal_grid(self):
        base, quat, feet, _ = sample_inputs()
        expected = make_heightmap_points(base, quat, feet)
        quat[:, 0] = math.cos(0.3)
        quat[:, 1] = math.sin(0.3)
        actual = make_heightmap_points(base, quat, feet)
        for observed, target in zip(actual, expected):
            torch.testing.assert_close(observed, target)

    def test_flat_query_uses_each_foot_height_and_environment_ground_offset(self):
        base, quat, feet, ground = sample_inputs()
        footmap, basemap = flat_heightmaps(base, quat, feet, ground)
        torch.testing.assert_close(
            footmap,
            torch.tensor(
                [[0.12] * 9 + [0.20] * 9, [0.2] * 9 + [0.3] * 9], dtype=base.dtype
            ),
        )
        torch.testing.assert_close(
            basemap, torch.tensor([[0.8] * 81, [0.9] * 81], dtype=base.dtype)
        )
        # 同时抬升机器人和地面不会改变相对高度，不把 world z 当监督值。
        shifted_base, shifted_feet = base.clone(), feet.clone()
        shifted_base[:, 2] += 7.0
        shifted_feet[:, :, 2] += 7.0
        shifted = flat_heightmaps(shifted_base, quat, shifted_feet, ground + 7.0)
        torch.testing.assert_close(shifted[0], footmap)
        torch.testing.assert_close(shifted[1], basemap)

    def test_nonflat_ground_query_preserves_pointwise_height_variation_and_sign(self):
        base, quat, feet, _ = sample_inputs()
        foot_points, _ = make_heightmap_points(base, quat, feet)
        # 已知解析起伏地形，只在此测试中查询每个采样点；不调用平面查询冒充地形。
        ground = 0.3 * foot_points[..., 0] + 0.2 * foot_points[..., 1].square()
        heights = relative_heights(foot_points, ground)
        torch.testing.assert_close(heights, foot_points[..., 2] - ground)
        self.assertGreater(torch.unique(heights[0, :9]).numel(), 3)
        self.assertTrue(bool((heights[0] < 0).all()))

    def test_rejects_wrong_shapes_without_broadcasting(self):
        base, quat, feet, ground = sample_inputs()
        for args in (
            (base[:, :2], quat, feet),
            (base, quat[:1], feet),
            (base, quat, feet[:, 0]),
        ):
            with (
                self.subTest(shapes=[tuple(x.shape) for x in args]),
                self.assertRaises(ValueError),
            ):
                make_heightmap_points(*args)
        with self.assertRaises(ValueError):
            flat_heightmaps(base, quat, feet, ground[:, None])
        points, _ = make_heightmap_points(base, quat, feet)
        for heights in (ground, torch.zeros(2, 1), torch.zeros(1, 18)):
            with self.assertRaises(ValueError):
                relative_heights(points, heights)
        with self.assertRaises(ValueError):
            relative_heights(points[..., :2], torch.zeros(2, 18, dtype=base.dtype))

    def test_rejects_nonfinite_inputs_and_zero_quaternion(self):
        originals = sample_inputs()
        for index in range(4):
            for invalid in (math.nan, math.inf, -math.inf):
                args = [value.clone() for value in originals]
                args[index].view(-1)[0] = invalid
                with (
                    self.subTest(argument=index, invalid=invalid),
                    self.assertRaises(ValueError),
                ):
                    flat_heightmaps(*args)
        base, quat, feet, _ = originals
        with self.assertRaises(ValueError):
            make_heightmap_points(base, torch.zeros_like(quat), feet)
        points, _ = make_heightmap_points(base, quat, feet)
        ground = torch.zeros(2, 18, dtype=base.dtype)
        ground[0, 0] = math.nan
        with self.assertRaises(ValueError):
            relative_heights(points, ground)

    def test_rejects_integer_and_mixed_dtype_inputs(self):
        base, quat, feet, ground = sample_inputs()
        with self.assertRaises(ValueError):
            make_heightmap_points(base.long(), quat, feet)
        with self.assertRaises(ValueError):
            make_heightmap_points(base, quat.float(), feet)
        with self.assertRaises(ValueError):
            flat_heightmaps(base, quat, feet, ground.float())


if __name__ == "__main__":
    unittest.main()
