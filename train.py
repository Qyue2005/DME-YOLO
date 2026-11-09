"""Train a local YOLOv10 configuration with explicit data and initialization."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Local dataset YAML")
    parser.add_argument("--model", type=Path, default=ROOT / "ultralytics/cfg/models/v10/yolov10n_ODConv.yaml")
    init = parser.add_mutually_exclusive_group(required=True)
    init.add_argument("--weights", type=Path, help="Trusted local pretrained checkpoint")
    init.add_argument("--scratch", action="store_true", help="Explicitly train from random initialization")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--amp", action="store_true", help="Enable AMP; upstream AMP check may download YOLOv8n")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/train")
    parser.add_argument("--name", default="exp")
    args = parser.parse_args()
    for key in ("data", "model", "weights"):
        value = getattr(args, key)
        if value is not None and not value.is_file():
            parser.error(f"--{key} must be an existing local file: {value}")
    if args.model.suffix not in {".yaml", ".yml"}:
        parser.error("--model must be a model YAML; use --weights for initialization")

    from ultralytics import YOLOv10

    model = YOLOv10(str(args.model.resolve()), task="detect")
    if args.weights:
        model.load(str(args.weights.resolve()))
    model.train(data=str(args.data.resolve()), task="detect", cache=False,
                imgsz=args.imgsz, epochs=args.epochs, single_cls=False, batch=args.batch,
                close_mosaic=0, workers=args.workers, device=args.device,
                optimizer=args.optimizer, lr0=args.lr0, lrf=args.lrf, seed=args.seed,
                deterministic=True, amp=args.amp,
                pretrained=str(args.weights.resolve()) if args.weights else False,
                project=str(args.project), name=args.name)


if __name__ == "__main__":
    main()
