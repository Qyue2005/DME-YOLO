# PCB 检测环境与本地验证

本仓库的检测代码已在 Windows、Python 3.11.16、RTX 3070 Laptop 8GB 上验证。主依赖为 PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128、NumPy 1.26.4。服务器与 RTX 5090 仍待实机验证。

- [P0-03 本轮修复与验证结果](docs/experiments/P0-03_LOCAL_FIXES_ZH.md)
- [首次环境/模型审计及服务器材料清单](docs/experiments/P0-03_ENVIRONMENT_MODEL_AUDIT_ZH.md)
- [检测直接依赖](requirements-detector.txt)
- [本机全部运行依赖快照](requirements-lock-win-py311-cu128.txt)

## Windows 安装

从仓库根目录执行；使用独立环境，不需要修改已有 Anaconda 环境或系统 Python。脚本接受 Python 3.10/3.11，本轮实际验证的是 3.11。将下面的解释器路径替换为本机真实路径：

```powershell
.\setup_windows.ps1 -Mode gpu -PythonExe "C:\path\to\Python311\python.exe" -CheckOnly
.\setup_windows.ps1 -Mode gpu -PythonExe "C:\path\to\Python311\python.exe"
```

`-CheckOnly` 只检查解释器与目标路径。脚本默认建立 `.venv`，安装固定配对的 CUDA 12.8 wheels、检测依赖和本地包，再检查依赖、导入与实际 CUDA/torchvision 运算。任何外部命令失败都会停止；已有无效环境会被保留，可通过 `-VenvDir .venv-new` 选择新目录。脚本不会自动安装系统 Python，也不会自动删除或搬移旧环境。

已有有效 `.venv` 时，可直接执行 `./setup_windows.ps1 -Mode gpu`。CPU 环境可使用 `-Mode cpu -VenvDir .venv-cpu`；CPU 专用 wheel 的完整安装流程本轮未实测。

界面和导出依赖使用可选参数：

```powershell
.\setup_windows.ps1 -Mode gpu -IncludeAppDependencies
```

本轮仅验证检测功能，未安装该可选组。原 `requirements-windows-gpu.txt` / `requirements-windows-cpu.txt` 分别保留 GPU / CPU ONNX Runtime，避免默认同时安装两者。`requirements.txt` 现在指向检测依赖；扩散生成目录的旧依赖不可叠加安装到该环境。

如需复用本轮 Windows/Python 3.11 的完整运行依赖版本，先安装对应 torch wheels，再安装快照并安装本仓库：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-win-py311-cu128.txt
.\.venv\Scripts\python.exe -m pip install -e . -c requirements-lock-win-py311-cu128.txt
.\.venv\Scripts\python.exe -m pip check
```

快照记录 Windows GPU 环境，不是 Linux/CPU 环境的通用锁。项目 `.venv` 不继承 base 环境的第三方包。若受限终端的临时目录不可写，可把该次进程的 TEMP/TMP 指向自己有写权限的工作目录后重试；本轮在工作区临时目录完成了安装。

## 无真实数据的本地验证

```powershell
.\.venv\Scripts\python.exe tools/verify_environment.py --device cuda
.\.venv\Scripts\python.exe tools/test_p0_03_compat.py
.\.venv\Scripts\python.exe tools/smoke_p0_03.py --device cuda --trainer-smoke --output runs/p0-03-smoke-result.json
```

第一条检查固定依赖、本仓库导入、矩阵运算和 torchvision NMS。第二条检查已知 PR 曲线的 AP、可信 checkpoint 的 strip/恢复兼容，以及可选 C2f 模块的参数/重复数处理。第三条使用随机初始化的模型和自动生成的合成图片，检查 FP32/AMP、检测损失反向、优化器恢复、权重保存/strip/重载推理，以及基线的 1 轮训练和最终验证。

`smoke_p0_03.py` 不下载预训练权重；输出与合成样例写入 `runs/`。它会记录当前 ODConv 配置中未获得梯度的参数，模型层连接问题仍须另行解决。合成图指标不代表 PCB 精度；本轮也没有测量正式 640 训练的显存上限、吞吐或论文 FPS。

## 有真实材料后的训练与验证

训练入口要求显式选择随机初始化或可信的本地权重，并传入实际数据 YAML。默认模型仍是原 `yolov10n_ODConv.yaml`；它的尺度和层引用待确认，正式实验前须先冻结修正后的配置。

```powershell
# 小样本工程检查；模型与数据文件应替换为已经核对的真实材料
.\.venv\Scripts\python.exe train.py --data "D:\data\pcb.yaml" --model "path\to\confirmed-model.yaml" --scratch --epochs 3 --batch 2 --imgsz 128 --device 0

# 使用可信本地预训练权重时，以 --weights 替代 --scratch
.\.venv\Scripts\python.exe train.py --data "D:\data\pcb.yaml" --model "path\to\confirmed-model.yaml" --weights "D:\weights\trusted.pt" --batch 8 --imgsz 640 --device 0

# 默认用验证集；测试集只在协议冻结后显式选择 --split test
.\.venv\Scripts\python.exe test.py --weights "runs\train\exp\weights\best.pt" --data "D:\data\pcb.yaml" --split val --device 0
```

`--scratch` / `--weights` 二选一，避免默认加载来源和匹配情况不明的 n 权重。入口默认关闭 AMP，需显式添加 `--amp`；原训练器的 AMP 自检可能下载 YOLOv8n，应在正式环境与协议中单独确认。seed、lr0、lrf、batch、workers 可显式传入，实际训练参数由训练器保存到 `args.yaml`。真实数据目录、标签与划分检查仍属于 P0-01/P0-02。

## 服务器待办

服务器到位后确认 GPU 实际型号、显存、系统、驱动、RAM、持久化存储和使用方式。检测环境候选仍为 Python 3.11、torch 2.9.1 / torchvision 0.24.1 / cu128；安装 `requirements-detector.txt` 和本地包后，重跑上述运算与模型检查，生成服务器自己的依赖快照。

5090 兼容性、正式 batch/显存、Linux 数据加载、扩散生成环境、导出和速度实验均未在服务器验收。详细顺序与材料清单见 P0-03 报告。
