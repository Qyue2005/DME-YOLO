"""Offline P0-03 checks. No installs, downloads, training data, or checkpoint files.

Examples (run from repository root):
  python tools/audit_p0_03.py --output /path/to/audit.json
  python tools/audit_p0_03.py --runtime --device cuda --output /path/to/audit.json

PyYAML enables the graph audit; --runtime requires an existing PyTorch environment.
The isolated module probes are NOT a full YOLO training/validation run.
"""

import argparse
import ast
import datetime
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

sys.dont_write_bytecode = True


def command(args, cwd=None):
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}


def attempt(fn):
    try:
        return {"status": "ok", "result": fn()}
    except Exception as exc:
        return {"status": "error", "type": type(exc).__name__, "message": str(exc)[:1600]}


def graph_audit(path):
    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    scale = next(iter(config["scales"]))
    depth, width, cap = config["scales"][scale]
    rows, channels, strides = [], [], []
    for i, (source, repeats, module, args) in enumerate(config["backbone"] + config["head"]):
        refs = source if isinstance(source, list) else [source]
        parents = [i + ref if ref < 0 else ref for ref in refs]
        if any(parent >= i or parent < -1 for parent in parents):
            raise ValueError(f"Invalid source reference at layer {i}: {parents}")
        in_ch = [3 if parent == -1 else channels[parent] for parent in parents]
        in_stride = [1 if parent == -1 else strides[parent] for parent in parents]
        stride = in_stride[0]
        out_ch = in_ch[0]
        if module in {"Conv", "ODConv", "SCDown", "C2f", "C2fCIB", "SPPF", "PSA"}:
            out_ch = math.ceil(min(args[0], cap) * width / 8) * 8
            if module in {"Conv", "ODConv", "SCDown"}:
                stride *= args[2] if len(args) > 2 else 1
        elif module == "nn.Upsample":
            stride /= args[1]
        elif module == "Concat":
            if len(set(in_stride)) != 1:
                raise ValueError(f"Spatial stride mismatch at layer {i}: {in_stride}")
            out_ch = sum(in_ch)
        elif module not in {"EAMA", "v10Detect"}:
            raise ValueError(f"Unsupported module in static estimator: {module}")
        row = {"layer": i, "module": module, "parents": parents,
               "effective_repeats": max(round(repeats * depth), 1) if repeats > 1 else repeats,
               "input_channels": in_ch, "input_strides": in_stride}
        if module != "v10Detect":
            row.update(output_channels=out_ch, stride=stride)
        if module == "EAMA":
            row["default_factor32_divisible"] = out_ch % 32 == 0
        rows.append(row)
        channels.append(out_ch)
        strides.append(stride)
    reachable, pending = set(), [len(rows) - 1]
    while pending:
        current = pending.pop()
        if current < 0 or current in reachable:
            continue
        reachable.add(current)
        pending.extend(rows[current]["parents"])
    return {"nc": config["nc"], "scale_key": scale, "scales": config["scales"],
            "method": "Static channel/stride propagation for listed module types; no tensor execution",
            "not_reaching_final_output": sorted(set(range(len(rows))) - reachable), "layers": rows}


def syntax_audit(root):
    listing = command(["git", "ls-files", "--", "*.py"], root)
    if listing.get("returncode") != 0:
        raise RuntimeError(listing)
    files = listing["stdout"].splitlines()
    errors = []
    for name in files:
        try:
            compile((root / name).read_bytes(), name, "exec", dont_inherit=True)
        except (SyntaxError, UnicodeError) as exc:
            errors.append({"file": name, "error": str(exc)})
    return {"files_checked": len(files), "errors": errors, "scope": "syntax only; imports not executed"}


def runtime_audit(root, device):
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(0)
    result = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "device_requested": device,
              "compiled_arches": torch.cuda.get_arch_list() if torch.cuda.is_available() else []}
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        result["compute_capability"] = list(torch.cuda.get_device_capability(0))

    def load_module(name, relative):
        spec = importlib.util.spec_from_file_location(name, root / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    od = load_module("audit_odconv", "ultralytics/nn/Extramodule/ODConv.py")
    ea = load_module("audit_eama", "ultralytics/nn/Extramodule/EAMAattention.py")

    def step(factory, shape):
        model = factory().to(device).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        x = torch.randn(*shape, device=device, requires_grad=True)
        y = model(x)
        loss = y.float().square().mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        finite = bool(torch.isfinite(loss).item() and gradients and
                      all(torch.isfinite(g).all().item() for g in gradients) and
                      torch.isfinite(x.grad).all().item())
        no_grad = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        optimizer.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        if not finite:
            raise RuntimeError("Non-finite loss or gradient")
        return {"input_shape": shape, "output_shape": list(y.shape), "loss": loss.item(),
                "finite": finite, "optimizer_steps": 1, "parameters_without_gradient": no_grad}

    result["isolated_modules_fp32"] = {
        "ODConv_48_96": attempt(lambda: step(lambda: od.ODConv(48, 96, 3, 2), [2, 48, 32, 32])),
        "ODConv_96_192": attempt(lambda: step(lambda: od.ODConv(96, 192, 3, 2), [2, 96, 16, 16])),
        "EAMA_576": attempt(lambda: step(lambda: ea.EAMA(576), [2, 576, 20, 20])),
        "C2f_EAMA_default": attempt(lambda: step(lambda: ea.C2f_EAMA(64, 64), [2, 64, 8, 8])),
        "C2f_EAMA_nondefault_e025": attempt(lambda: step(lambda: ea.C2f_EAMA(64, 64, e=0.25), [2, 64, 8, 8])),
    }

    def checkpoint_probe():
        # Only a fresh in-memory model created by this process is deserialized.
        stream = io.BytesIO()
        torch.save({"model": torch.nn.Linear(2, 2)}, stream)
        stream.seek(0)
        return type(torch.load(stream, map_location="cpu")["model"]).__name__

    result["default_torch_load_self_created_full_model"] = attempt(checkpoint_probe)

    def ap_probe():
        import numpy as np

        # Run exactly this repository's standalone compute_ap function without package imports.
        tree = ast.parse((root / "ultralytics/utils/metrics.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_ap")
        scope = {"np": np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "metrics.py:compute_ap", "exec"), scope)
        return {"numpy": np.__version__, "ap": float(scope["compute_ap"](np.array([0.5, 1.0]), np.array([1.0, 0.5]))[0])}

    result["compute_ap_isolated"] = attempt(ap_probe)
    result["full_model_training_validation"] = "NOT RUN: isolated probes do not load the full package, datasets, or weights"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    packages = {}
    for name in ["torch", "torchvision", "numpy", "scipy", "PyYAML", "timm", "einops", "PyWavelets",
                 "opencv-python", "huggingface-hub", "pandas", "matplotlib", "thop"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source_commit": command(["git", "rev-parse", "HEAD"], root),
              "python": platform.python_version(), "os": platform.platform(),
              "cpu_threads": os.cpu_count(), "packages": packages,
              "gpu_inventory": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"]),
              "syntax": attempt(lambda: syntax_audit(root)), "model_graphs": {}}
    for name in ["yolov10n.yaml", "yolov10n_ODConv.yaml", "yolov10m.yaml"]:
        result["model_graphs"][name] = attempt(lambda name=name: graph_audit(root / "ultralytics/cfg/models/v10" / name))
    if args.runtime:
        result["runtime"] = attempt(lambda: runtime_audit(root, args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audit saved to {args.output}; inspect per-check status (process success is not P0-03 acceptance).")


if __name__ == "__main__":
    main()
