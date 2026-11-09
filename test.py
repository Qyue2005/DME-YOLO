"""Validate a trusted local checkpoint; default to the validation split."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Trusted local checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Local dataset YAML")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    for key in ("weights", "data"):
        value = getattr(args, key)
        if not value.is_file():
            parser.error(f"--{key} must be an existing local file: {value}")

    from ultralytics import YOLOv10

    model = YOLOv10(str(args.weights.resolve()), task="detect")
    model.val(data=str(args.data.resolve()), split=args.split, imgsz=args.imgsz,
              batch=args.batch, device=args.device, workers=args.workers)


if __name__ == "__main__":
    main()
