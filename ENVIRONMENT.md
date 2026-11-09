> **2026-11-09 P0-03 检查更新：** 本地默认 Python 为 3.14，原安装脚本不会严格核验解释器版本；现有环境与模型还存在已确认的依赖和兼容问题。以下原安装说明尚未按本次检查修正，请先阅读 [中文环境与模型检查报告](docs/experiments/P0-03_ENVIRONMENT_MODEL_AUDIT_ZH.md)，按专用环境方案完成验证。服务器候选为 Python 3.11、torch 2.9.1 / torchvision 0.24.1 / cu128，仍待实机验收。

# Windows environment setup

This project is intended to run on Windows with Python 3.10 and a fresh virtual environment.

## Recommended setup

GPU mode:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1 -Mode gpu
```

CPU mode:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1 -Mode cpu
```

After setup:

```powershell
.\.venv\Scripts\Activate.ps1
.\run_app.ps1
```

## What the setup script does

1. Downloads and installs Python 3.10.11 if it is missing.
2. Detects and backs up a broken existing `.venv`.
3. Creates a new virtual environment.
4. Installs PyTorch, project dependencies, and the local `ultralytics` package.
5. Verifies that the key imports work.

## Notes

- `gpu` mode installs CUDA 12.8 PyTorch wheels.
- `cpu` mode installs CPU-only PyTorch wheels.
- The old `.venv` in this repository is broken because it points to a missing Python 3.10 installation.
