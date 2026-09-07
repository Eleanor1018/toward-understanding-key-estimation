import ast
import unittest
from pathlib import Path


G1_ENV_PATH = Path(__file__).resolve().parents[1] / "g1_env.py"


class V8TaskPurityTest(unittest.TestCase):
    def test_task_height_and_vertical_safety_do_not_follow_reference(self) -> None:
        source_text = G1_ENV_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source_text)
        environment = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "G1KeyEstimationEnv"
        )
        rewards = next(
            node
            for node in environment.body
            if isinstance(node, ast.FunctionDef) and node.name == "_get_rewards"
        )
        source = ast.get_source_segment(source_text, rewards)
        assert source is not None

        self.assertIn("height_error = base_height - DESIRED_BASE_HEIGHT", source)
        self.assertIn(
            "vertical_velocity = root_linear_velocity_world[:, 2]",
            source,
        )
        self.assertNotIn(
            "height_error = base_height - full_reference.root_height",
            source,
        )
        self.assertNotIn(
            "root_linear_velocity[:, 2] - reference_root_velocity[:, 2]",
            source,
        )


if __name__ == "__main__":
    unittest.main()
