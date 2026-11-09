"""Offline regression checks. Run with Python directly (no upstream pytest hooks).

Only APCompatibilityTests needs NumPy; it can also run in a separate NumPy 2.x
environment without importing the rest of Ultralytics.
"""

import ast
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/local-settings"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/matplotlib"))
os.environ["YOLO_AUTOINSTALL"] = "false"


class APCompatibilityTests(unittest.TestCase):
    def test_known_pr_curves(self):
        import numpy as np

        tree = ast.parse((ROOT / "ultralytics/utils/metrics.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_ap")
        scope = {"np": np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "compute_ap", "exec"), scope)
        # Independent reference areas, including the existing 101-point end sentinel convention.
        for recall, precision, expected in [([1.0], [1.0], 0.995),
                                             ([0.5], [1.0], 0.75),
                                             ([0.0, 1.0], [0.0, 0.0], 0.0)]:
            with self.subTest(numpy=np.__version__, recall=recall, precision=precision):
                actual, _, _ = scope["compute_ap"](np.array(recall), np.array(precision))
                self.assertAlmostEqual(float(actual), expected, places=12)


class ModuleCompatibilityTests(unittest.TestCase):
    def test_eama_expansion_ratios(self):
        import torch
        from ultralytics.nn.Extramodule import C2f_EAMA

        torch.set_num_threads(2)
        for expansion in (0.25, 0.5, 0.75):
            with self.subTest(expansion=expansion):
                model = C2f_EAMA(32, 64, n=2, e=expansion)
                output = model(torch.rand(2, 32, 8, 8))
                self.assertEqual(tuple(output.shape), (2, 64, 8, 8))
                output.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_parser_passes_repeats_inside_custom_c2f(self):
        import torch
        from ultralytics.nn.tasks import parse_model

        for module in ("C2f_EAMA", "C2f_ODConv"):
            with self.subTest(module=module):
                config = {"nc": 6, "depth_multiple": 1.0, "width_multiple": 1.0,
                          "backbone": [[-1, 3, module, [64, True]]], "head": []}
                model, _ = parse_model(config, ch=32, verbose=False)
                self.assertEqual(len(model[0].m), 3)
                output = model(torch.rand(2, 32, 8, 8))
                self.assertEqual(tuple(output.shape), (2, 64, 8, 8))
                output.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_eama_rejects_invalid_grouping(self):
        from ultralytics.nn.Extramodule.EAMAattention import EMAttention

        for channels, factor in ((16, 32), (70, 32), (64, 0), (64, -1)):
            with self.subTest(channels=channels, factor=factor):
                with self.assertRaises(ValueError):
                    EMAttention(channels, factor=factor)


class CheckpointCompatibilityTests(unittest.TestCase):
    def check_roundtrip(self, with_args):
        import torch
        from ultralytics.utils.torch_utils import strip_optimizer

        parent = ROOT / "runs/compatibility-tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            source, stripped = Path(directory) / "source.pt", Path(directory) / "stripped.pt"
            model = torch.nn.Linear(2, 2)
            ema = torch.nn.Linear(2, 2)
            with torch.no_grad():
                ema.weight.fill_(0.25)
            checkpoint = {"model": model, "ema": ema, "optimizer": {"marker": 1},
                          "epoch": 2, "updates": 3, "best_fitness": 0.5,
                          "train_metrics": {"fitness": 0.5}}
            if with_args:
                checkpoint["train_args"] = {"task": "detect", "imgsz": 128, "unknown_key": 1}
            torch.save(checkpoint, source)
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            strip_optimizer(str(source), str(stripped))
            restored = torch.load(stripped, map_location="cpu", weights_only=False)
            self.assertEqual(original_hash, hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(restored["epoch"], -1)
            for key in ("optimizer", "ema", "updates", "best_fitness"):
                self.assertIsNone(restored[key])
            self.assertEqual(restored["model"].weight.dtype, torch.float16)
            self.assertTrue(torch.all(restored["model"].weight == 0.25).item())
            self.assertTrue(all(not p.requires_grad for p in restored["model"].parameters()))
            self.assertEqual(restored["train_metrics"]["fitness"], 0.5)
            self.assertIsInstance(restored["train_args"], dict)
            self.assertNotIn("unknown_key", restored["train_args"])
            if with_args:
                self.assertEqual(restored["train_args"]["imgsz"], 128)

    def test_training_checkpoint(self):
        self.check_roundtrip(with_args=True)

    def test_legacy_checkpoint_without_train_args(self):
        self.check_roundtrip(with_args=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
