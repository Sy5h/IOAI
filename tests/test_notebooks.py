"""Packaging tests for single-file notebook execution contracts."""

import ast
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_notebooks", ROOT / "tools/build_notebooks.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class NotebookTests(unittest.TestCase):
    def test_source_guards_removed_and_no_execution_in_definition_cell(self):
        for slug in builder.TASKS:
            nb = builder.notebook_for(slug)
            source = "".join(next(c for c in nb["cells"] if c["id"] == "solution-source")["source"])
            tree = ast.parse(source)
            self.assertFalse(any(isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)
                                 for node in tree.body), slug)
            self.assertTrue(any(isinstance(node, ast.FunctionDef) and node.name == "main"
                                for node in tree.body), slug)

    def test_potato_jupyter_does_not_start_interactive_loop(self):
        source = "".join(builder.notebook_for("day1_potato")["cells"][-1]["source"])
        calls = []
        env = {"__name__": "__main__", "main": lambda: calls.append(1), "print": lambda *a: None}
        exec(source, env)
        self.assertEqual(calls, [])
        env["__file__"] = "solution.py"
        exec(source, env)
        self.assertEqual(calls, [1])

    def test_noninteractive_notebooks_start_exactly_once_without_kernel_args(self):
        for slug in builder.TASKS:
            if slug == "day1_potato":
                continue
            source = "".join(builder.notebook_for(slug)["cells"][-1]["source"])
            calls = []
            exec(source, {"main": lambda argv: calls.append(argv)})
            self.assertEqual(calls, [[]], slug)

    def test_committed_notebooks_are_current_valid_and_small(self):
        for slug in builder.TASKS:
            path = ROOT / "notebooks" / slug / "solution.ipynb"
            content = path.read_text(encoding="utf-8")
            actual = json.loads(content)
            self.assertEqual(actual, builder.notebook_for(slug), slug)
            self.assertLess(len(content.encode()), 1_000_000, slug)
            for c in actual["cells"]:
                if c["cell_type"] == "code":
                    compile("".join(c["source"]), slug, "exec")
                    self.assertEqual(c["outputs"], [])
                    self.assertIsNone(c["execution_count"])
        self.assertEqual((ROOT / "notebooks/day2_ioai_field/custom_model.py").read_text(encoding="utf-8"),
                         (ROOT / "solutions/field_custom_model.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

