"""Run local YOLOv10 engineering checks with random inputs and optional toy images.

No PCB data or pretrained weights are used. Metrics on toy images have no research meaning.
Artifacts are saved under runs/p0-03-smoke; the JSON is suitable for the audit document.
"""

import argparse
from copy import deepcopy
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trainer-smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    work = root / "runs/p0-03-smoke" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    work.mkdir(parents=True)
    os.environ["YOLO_CONFIG_DIR"] = str(work / "settings")
    os.environ["MPLCONFIGDIR"] = str(work / "matplotlib")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_VERBOSE"] = "false"

    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLOv10
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import YOLOv10DetectionModel, attempt_load_one_weight
    from ultralytics.utils import SETTINGS
    from ultralytics.utils import checks
    from ultralytics.utils.torch_utils import strip_optimizer

    SETTINGS.update({key: False for key in ("sync", "hub", "wandb", "tensorboard", "comet", "mlflow")})
    checks.ONLINE = False  # skip package update queries in this offline engineering check
    # Use a bundled local font, so the toy validator does not download Arial.
    import matplotlib
    font = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    (work / "settings/Arial.ttf").write_bytes(font.read_bytes())

    torch.set_num_threads(2)
    torch.manual_seed(0)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    report = {"scope": "engineering checks only; random initialization and synthetic inputs",
              "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
              "cuda_build": torch.version.cuda, "device": args.device,
              "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "source_sha256": {}, "models": {}}
    for name in ["ultralytics/utils/metrics.py", "ultralytics/utils/torch_utils.py", "ultralytics/engine/tuner.py",
                 "ultralytics/nn/tasks.py", "ultralytics/nn/Extramodule/EAMAattention.py",
                 "requirements-detector.txt", "tools/smoke_p0_03.py"]:
        report["source_sha256"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()

    def save_report():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def train_step(model, optimizer, amp=False):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = {"img": torch.rand(2, 3, 128, 128, device=args.device),
                 "batch_idx": torch.tensor([0, 1], device=args.device),
                 "cls": torch.tensor([[0.], [1.]], device=args.device),
                 "bboxes": torch.tensor([[.5, .5, .3, .3], [.4, .4, .25, .25]], device=args.device)}
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=128.0)
        with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=amp):
            loss, components = model(batch)
        assert torch.isfinite(loss).all() and torch.isfinite(components).all()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        inactive = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        scaler.step(optimizer)
        scaler.update()
        if args.device == "cuda":
            torch.cuda.synchronize()
        return {"finite_loss_and_gradients": True, "loss": float(loss.detach()),
                "loss_components": components.detach().float().cpu().tolist(),
                "optimizer_steps": 1, "parameters_without_gradient": inactive}

    try:
        for name in ["yolov10n.yaml", "yolov10n_ODConv.yaml"]:
            print(f"Checking {name}", flush=True)
            cfg = root / "ultralytics/cfg/models/v10" / name
            entry = report["models"][name] = {"yaml_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest()}
            model = YOLOv10DetectionModel(str(cfg), nc=6, verbose=False).to(args.device)
            model.args = get_cfg(overrides={"task": "detect"})
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            entry["parameters"] = sum(p.numel() for p in model.parameters())
            entry["strides"] = model.stride.tolist()
            entry["fp32"] = train_step(model, optimizer)
            if args.device == "cuda":
                entry["amp"] = train_step(model, optimizer, amp=True)

            # Exercise a resumable local checkpoint and the same strip function used by final_eval.
            del model.criterion
            checkpoint = work / f"{cfg.stem}-resume.pt"
            torch.save({"model": deepcopy(model).cpu(), "optimizer": optimizer.state_dict(),
                        "epoch": 0, "train_args": dict(model.args)}, checkpoint)
            del optimizer, model
            restored, state = attempt_load_one_weight(str(checkpoint), device=args.device)
            restored.args = get_cfg(overrides={"task": "detect"})
            optimizer = torch.optim.SGD(restored.parameters(), lr=0.001, momentum=0.9)
            optimizer.load_state_dict(state["optimizer"])
            entry["restored_optimizer_step"] = train_step(restored, optimizer)
            del optimizer, restored, state
            stripped = work / f"{cfg.stem}-stripped.pt"
            strip_optimizer(str(checkpoint), str(stripped))
            loaded = YOLOv10(str(stripped), task="detect")
            prediction = loaded.predict(source=np.zeros((128, 128, 3), dtype=np.uint8), imgsz=128,
                                        device=0 if args.device == "cuda" else "cpu", verbose=False, save=False)
            assert len(prediction) == 1 and torch.isfinite(prediction[0].boxes.data).all()
            entry["stripped_reload_predict"] = "passed"
            entry["status"] = "passed_with_known_inactive_parameters" if entry["fp32"]["parameters_without_gradient"] else "passed"
            del loaded, prediction
            if args.device == "cuda":
                torch.cuda.empty_cache()
            save_report()

        if args.trainer_smoke:
            print("Checking trainer/validator with a generated toy dataset", flush=True)
            dataset = work / "toy-data"
            for split, count in [("train", 4), ("val", 2)]:
                images, labels = dataset / "images" / split, dataset / "labels" / split
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                for i in range(count):
                    image = np.zeros((128, 128, 3), dtype=np.uint8)
                    cv2.rectangle(image, (44, 44), (84, 84), (80 + 20 * i, 180, 230), -1)
                    assert cv2.imwrite(str(images / f"toy-{i}.png"), image)
                    (labels / f"toy-{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.3125 0.3125\n")
            import yaml
            data = dataset / "data.yaml"
            data.write_text(yaml.safe_dump({"path": dataset.as_posix(), "train": "images/train", "val": "images/val",
                                            "names": [f"toy_{i}" for i in range(6)]}), encoding="utf-8")
            model = YOLOv10(str(root / "ultralytics/cfg/models/v10/yolov10n.yaml"), task="detect")
            model.train(data=str(data), epochs=1, imgsz=128, batch=2, workers=0,
                        device=0 if args.device == "cuda" else "cpu", amp=False, pretrained=False,
                        optimizer="SGD", lr0=0.001, cache=False, plots=False, val=True,
                        mosaic=0.0, close_mosaic=0, project=str(work), name="toy-trainer", verbose=False)
            assert (work / "toy-trainer/weights/best.pt").is_file()
            assert (work / "toy-trainer/weights/last.pt").is_file()
            report["toy_trainer"] = {"status": "passed", "epochs": 1, "train_images": 4, "val_images": 2,
                                      "imgsz": 128, "batch": 2, "amp": False,
                                      "covers": ["dataloader", "trainer", "checkpoint save", "strip_optimizer", "final validator"],
                                      "research_metrics": "not reported; synthetic images are not PCB evaluation data"}
        report["status"] = "passed; research-model issues and server validation remain pending"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
        raise
    finally:
        save_report()
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
