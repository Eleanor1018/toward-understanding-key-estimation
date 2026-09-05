import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch

from mujoco_evaluate import (
    ACTION_DIM,
    DEFAULT_SCENE_PATH,
    build_observation,
    clamp_joint_position_targets,
    contact_diagnostics,
    configure_pd_actuators,
    load_policy,
    soft_joint_position_limits,
    validate_mujoco_contract,
)
from normalization import DEFAULT_JOINT_POSITIONS, JOINT_EFFORT_LIMITS


class MujocoPolicyContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_path = DEFAULT_SCENE_PATH
        cls.model = mujoco.MjModel.from_xml_path(str(cls.model_path))
        cls.data = mujoco.MjData(cls.model)
        with np.load(
            Path(__file__).resolve().parents[1]
            / "assets"
            / "motions"
            / "g1_walk_mimickit.npz",
            allow_pickle=False,
        ) as archive:
            cls.joint_names = tuple(
                str(name) for name in archive["joint_names"].tolist()
            )

    def test_joint_actuator_and_observation_contract(self) -> None:
        qpos, qvel, actuators = validate_mujoco_contract(
            self.model,
            self.joint_names,
        )
        self.assertEqual(tuple(qpos.shape), (ACTION_DIM,))
        self.assertEqual(tuple(qvel.shape), (ACTION_DIM,))
        self.assertEqual(tuple(actuators.shape), (ACTION_DIM,))

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = (0.0, 0.0, 0.8)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[qpos] = np.asarray(DEFAULT_JOINT_POSITIONS)
        mujoco.mj_forward(self.model, self.data)
        pelvis = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "pelvis",
        )
        observation = build_observation(
            self.model,
            self.data,
            pelvis,
            qpos,
            qvel,
            np.zeros(ACTION_DIM, dtype=np.float32),
        )
        self.assertEqual(tuple(observation.shape), (93,))
        np.testing.assert_allclose(observation[:3], (0.0, 0.0, -1.0), atol=1e-6)

    def test_pd_effort_limits_match_repository_contract(self) -> None:
        _, _, actuators = validate_mujoco_contract(self.model, self.joint_names)
        configure_pd_actuators(self.model, actuators)
        np.testing.assert_allclose(
            self.model.actuator_forcerange[actuators, 1],
            JOINT_EFFORT_LIMITS,
        )
        np.testing.assert_allclose(
            self.model.actuator_forcerange[actuators, 0],
            -np.asarray(JOINT_EFFORT_LIMITS),
        )

    def test_soft_joint_limits_match_isaac_center_scaling(self) -> None:
        limits = soft_joint_position_limits(self.model, self.joint_names)
        self.assertEqual(limits.shape, (ACTION_DIM, 2))
        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.joint_names
            ]
        )
        hard_limits = self.model.jnt_range[joint_ids]
        hard_center = hard_limits.mean(axis=1)
        np.testing.assert_allclose(limits.mean(axis=1), hard_center)
        np.testing.assert_allclose(
            limits[:, 1] - limits[:, 0],
            0.90 * (hard_limits[:, 1] - hard_limits[:, 0]),
        )

        targets = limits.mean(axis=1)
        targets[0] = hard_limits[0, 1]
        clipped, count = clamp_joint_position_targets(targets, limits)
        self.assertEqual(count, 1)
        self.assertEqual(clipped[0], limits[0, 1])

    def test_force_contacts_use_robot_weight_threshold(self) -> None:
        qpos, _, actuators = validate_mujoco_contract(
            self.model,
            self.joint_names,
        )
        configure_pd_actuators(self.model, actuators)
        self.model.opt.timestep = 0.001
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = (0.0, 0.0, 0.8)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[qpos] = np.asarray(DEFAULT_JOINT_POSITIONS)
        self.data.ctrl[actuators] = np.asarray(DEFAULT_JOINT_POSITIONS)
        for _ in range(1_000):
            mujoco.mj_step(self.model, self.data)
        pelvis = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "pelvis",
        )
        feet = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        )
        forces, illegal = contact_diagnostics(
            self.model,
            self.data,
            feet,
            pelvis,
        )
        threshold = 0.05 * self.model.body_mass.sum() * 9.81
        self.assertTrue(np.all(forces > threshold))
        self.assertFalse(illegal)

    def test_legacy_v5_checkpoint_contract_is_supported(self) -> None:
        checkpoint = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "policy_archive"
            / "forward_shuffle_v5_iter3500"
            / "checkpoint_03500.pt"
        )
        if not checkpoint.is_file():
            self.skipTest("local archived V5 checkpoint is not installed")
        _, _, config, loaded = load_policy(checkpoint, torch.device("cpu"))
        self.assertEqual(config.command_dim, 3)
        self.assertEqual(config.future_reference_dim, 0)
        self.assertEqual(
            loaded["input_normalization_type"], "g1_fixed_physical_scales_v1"
        )


if __name__ == "__main__":
    unittest.main()
