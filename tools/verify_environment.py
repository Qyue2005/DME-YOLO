"""Verify the pinned detector dependencies, local imports, and actual CPU/CUDA ops."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "runs" / "local-settings"))
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ.setdefault("MPLCONFIGDIR", str(root / "runs" / "matplotlib"))

    from packaging.requirements import Requirement

    if sys.version_info[:2] not in {(3, 10), (3, 11)}:
        raise RuntimeError("The pinned detector environment requires Python 3.10 or 3.11")
    versions = {}
    for line in (root / "requirements-detector.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        version = importlib.metadata.version(requirement.name)
        if version not in requirement.specifier:
            raise RuntimeError(f"{requirement.name}=={version} does not satisfy {requirement}")
        versions[requirement.name] = version

    import cv2
    import torch
    import torchvision
    import ultralytics
    from ultralytics.nn.Extramodule import EAMA, ODConv

    if Path(ultralytics.__file__).resolve().parent != root / "ultralytics":
        raise RuntimeError("Imported another ultralytics installation instead of this repository")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    matrix = torch.eye(8, device=args.device)
    assert torch.equal(matrix @ matrix, matrix)
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device=args.device)
    keep = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device=args.device), 0.5)
    assert keep.tolist() == [0]
    if args.device == "cuda":
        torch.cuda.synchronize()
    print(json.dumps({"status": "passed", "python": sys.version.split()[0], "device": args.device,
                      "torch_cuda_build": torch.version.cuda, "local_package": True,
                      "matmul": "passed", "torchvision_nms": "passed", "packages": versions}, indent=2))


if __name__ == "__main__":
    main()
