from __future__ import annotations

import ast
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _constant_list(tree: ast.AST, variable_name: str) -> set[str]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        return {
            element.value
            for element in node.value.elts
            if isinstance(element, ast.Constant)
            and isinstance(element.value, str)
        }
    raise AssertionError(f"did not find list {variable_name}")


def _dict_constant_keys(node: ast.Dict) -> set[str]:
    return {
        key.value
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


class MetricsSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_tree = _tree(
            PACKAGE_DIR / "train_pegasus_iris_fast_line_follow_ppo.py"
        )
        env_tree = _tree(PACKAGE_DIR / "fast_line_follow_env.py")
        cls.reward_terms = _constant_list(env_tree, "REWARD_TERM_KEYS")

    def test_update_metrics_fields_match_row(self) -> None:
        fields = _constant_list(self.train_tree, "update_fields")
        fields.update(f"mean_{key}" for key in self.reward_terms)
        row_keys: set[str] = set()
        for node in ast.walk(self.train_tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "update_row"
                for target in node.targets
            ):
                if isinstance(node.value, ast.Dict):
                    row_keys.update(_dict_constant_keys(node.value))
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "update_row"
                and node.func.attr == "update"
                and node.args
                and isinstance(node.args[0], ast.Dict)
            ):
                row_keys.update(_dict_constant_keys(node.args[0]))
        row_keys.update(f"mean_{key}" for key in self.reward_terms)
        self.assertSetEqual(fields, row_keys)

    def test_episode_metrics_fields_match_row(self) -> None:
        fields = _constant_list(self.train_tree, "episode_fields")
        fields.update(
            f"return_{key.removeprefix('reward_')}"
            for key in self.reward_terms
        )
        episode_row_keys: set[str] | None = None
        for node in ast.walk(self.train_tree):
            if not isinstance(node, ast.Assign) or not any(
                isinstance(target, ast.Name) and target.id == "row"
                for target in node.targets
            ):
                continue
            if isinstance(node.value, ast.Dict):
                keys = _dict_constant_keys(node.value)
                if "episode" in keys and "done_reason" in keys:
                    episode_row_keys = keys
                    break
        self.assertIsNotNone(episode_row_keys)
        episode_row_keys.update(
            f"return_{key.removeprefix('reward_')}"
            for key in self.reward_terms
        )
        self.assertSetEqual(fields, episode_row_keys)


if __name__ == "__main__":
    unittest.main()
