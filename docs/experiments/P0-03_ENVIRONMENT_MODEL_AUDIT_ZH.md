# P0-03：本地环境、模型配置与服务器准备检查

**首次检查日期：2026-11-09。最新状态：本地环境/兼容性修复及完整模型、合成训练工程验证已完成；模型设计确认、真实数据与服务器验收待办。详见 [本轮修复与验证结果](P0-03_LOCAL_FIXES_ZH.md)。P0-03 总状态仍为“进行中”。**

本机具有可用的 GPU，可以承担代码检查和小规模调试。依赖及通用兼容问题已在后续修复，模型连接与尺度仍需确认，再使用真实数据及服务器开展正式实验。服务器仅按用户提供的“可能有一张 RTX 5090”制定候选方案；实际 GPU、显存、驱动、操作系统和可用资源均待确认。

首次检查对象为 [`d39e2806049762770df9487f7da5273b8242965e`](https://github.com/Qyue2005/DME-YOLO/tree/d39e2806049762770df9487f7da5273b8242965e)，训练代码仍对应上游 `f542ae0`。第 1–4 节保留首次检查的环境、失败现象和结构分析；后续修复状态以新报告为准。首次检查未安装依赖；后续已建立本地独立环境。本轮仍未修改论文主模型的尺度和层引用，也未进行真实 PCB 数据训练。

- 总计划：[现有内容与补实验推进计划](../../EXPERIMENT_PLAN_ZH.md)。
- 可重复执行的检查脚本：[tools/audit_p0_03.py](../../tools/audit_p0_03.py)。
- 本次机器可读结果：[本地检查 JSON](evidence/p0-03-local-audit-2026-11-09.json)。

## 1 首次检查时的本地环境

| 项目 | 本次实际检查结果 | 对后续工作的含义 |
| --- | --- | --- |
| 系统 | Windows，系统构建号 26300；不同 Python 的平台字符串分别显示 Windows 10/11 | 后续锁定实际解释器与平台；平台字符串差异不视为另一台机器 |
| CPU | AMD Ryzen 7 5800H；16 个逻辑线程 | 可进行数据检查、配置分析和轻量调试 |
| 内存 | 31.86 GiB 可见物理内存，约 32GB | 暂不需要为环境检查增加本地内存 |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU | 已进行实际 CUDA 模块检查 |
| 显存与驱动 | 8192 MiB；驱动 621.01；计算能力 8.6 | 小模型调试具备基础；完整模型最大 batch 尚未测量 |
| 工作盘空间 | 检查时 D 盘剩余约 820 GiB | 足以开始环境与小样本工作；完整生成数据仍需容量清单 |
| 默认 Python | 3.14.7；未安装 torch、torchvision、NumPy 等检测依赖 | 直接执行默认 `python train.py` 不具备运行条件 |
| 已有可用 Python | Anaconda base 3.13.9；另有 3.11、3.12、3.13 环境 | 应新建 PCB 专用环境，避免改动其他项目环境 |
| 可用于本次独立检查的环境 | 已有 `nlp` 环境：Python 3.11.16、PyTorch 2.14.0+cu132、NumPy 2.4.6、PyYAML 6.0.3 | 本次仅借用解释器运行检查；这不是已冻结的 PCB 实验环境 |
| 项目专用环境 | 仓库下未发现 `.venv` | 仓库中的 `pyvenv.cfg` 指向原作者 Linux/Python 3.8 路径，不是本机可用环境 |

检查依据包括 `nvidia-smi`、Python 包元数据、Windows CPU 注册表项、`GlobalMemoryStatusEx` 及工作盘剩余空间。`conda.exe` 直接调用曾出现 DLL 重定位错误，但使用 base Python 执行 `python -m conda env list --json` 成功，已有环境的解释器也可运行；不能据此认定整个 Anaconda 已损坏。

`nlp` 环境缺少 torchvision、OpenCV、SciPy、timm、einops、PyWavelets、huggingface-hub、pandas、thop 等组件。实际执行仓库 `import ultralytics` 时，首个失败为 `ModuleNotFoundError: No module named 'cv2'`。其他已发现的环境也没有齐备的 PCB 训练依赖。

## 2 首次检查的验证记录

以下均为代码或随机张量检查，不产生 PCB 检测精度结论。GPU 检查使用 FP32、seed=0，每个成功的模块完成一次前向、反向和 SGD 更新；没有检测损失、真实图片和完整 YOLO 训练循环。

| 检查项 | 结果 | 验证边界 |
| --- | --- | --- |
| 仓库已跟踪 Python 文件语法检查 | 201 个文件通过，无语法错误 | 不证明变量定义、依赖导入、运行逻辑正确 |
| CUDA 可用性 | PyTorch 实际识别本机 GPU；CUDA build 为 13.2 | 不代表已经测试候选的 PyTorch 2.9.1/cu128 环境 |
| ODConv：48→96 通道 | 输入 `[2,48,32,32]`，输出 `[2,96,16,16]`；损失及已产生的梯度有限 | `attention.bn.weight/bias` 无梯度，见 R08 |
| ODConv：96→192 通道 | 输入 `[2,96,16,16]`，输出 `[2,192,8,8]`；检查通过 | 空间尺寸缩小，仅验证算子路径 |
| EAMA：576 通道 | 输入、输出均为 `[2,576,20,20]`；检查通过 | 对应当前主配置的通道数；未测试混合精度 |
| C2f_EAMA 默认 `e=0.5` | `[2,64,8,8]` 前向、反向和更新通过 | 这是可选模块，当前主模型直接使用 EAMA |
| C2f_EAMA 改为 `e=0.25` | 可重复出现通道不匹配错误 | 属于模块扩展使用问题，见 R09 |
| AP 计算 | 提取仓库原始 `compute_ap()` 独立运行；在 NumPy 2.4.6 下报 `np.trapz` 不存在 | 已复现评估函数的兼容性问题，见 R04 |
| 完整模型对象的默认反序列化 | 用本进程新建的内存 checkpoint，默认 `torch.load()` 报 `UnpicklingError` | 验证新版默认行为；没有加载外部权重，见 R05 |
| 3 个主模型 YAML 的静态连接检查 | n、m 基线无输出不可达层；ODConv 配置的第 23 层不进入最终输出 | 这是按 YAML 推导的计算图，尚非完整模型运行时梯度测量 |
| 完整模型导入、前向、训练、验证 | 导入被缺失的 OpenCV 阻止；后续未运行 | P0-03 不能标记为全部完成 |

本次检查脚本会分别保存每项成功或失败。脚本退出码为 0 仅表示报告已生成，**不表示所有检查通过**。它不会安装依赖、下载模型或训练数据，也不会代替完整训练验收。

## 3 首次检查发现的问题及修复依据

“已确认”表示有代码证据或已复现；“待实机验证”表示尚未在完整环境或服务器上执行。下表描述首次发现时的证据与处理建议；R01/R04/R05/R07/R09 现已完成本地修复和验证，R06 入口部分已完成，具体边界见本轮修复报告。

| 编号 | 优先级与结论 | 证据、影响 | 建议处理及验收 |
| --- | --- | --- | --- |
| R01 | 立即：项目环境不齐，已确认 | 默认 Python 无训练依赖；现有 torch 环境无法导入仓库 | 创建 PCB 专用 Python 3.11 环境；安装本地分支及额外模块依赖，完成 `pip check`、导入、CUDA/torchvision 算子检查 |
| R02 | 正式训练前：模型尺度不一致，已确认 | `yolov10n_ODConv.yaml` 使用 `[0.67,0.75,768]`，与 m 的缩放系数相同；n 基线为 `[0.33,0.25,1024]` | 获取论文使用的实际 YAML；冻结同尺度基线和改进版，统一 `nc=6`、初始化与参数统计口径；不能仅改文件名后算作可比 |
| R03 | 正式训练前：层引用需修正或解释，已确认 | 检测头输入为 3 个 Concat；第 23 层输出不进入检测头；两处旁路引用也未随插层后移 | 保留原配置，另建有版本记录的候选配置；按第 4 节检查输出、梯度与参数量，再确认实验设计 |
| R04 | 运行验证前：NumPy API 不兼容，已复现 | `ultralytics/utils/metrics.py:524` 使用 `np.trapz`；本次 NumPy 2.4.6 下失败 | 短期在专用环境固定 NumPy 1.26.4；后续兼容新版本时改用可用的积分 API，并用同一组 PR 曲线检查 AP 数值一致 |
| R05 | 首轮训练结束前：权重加载适配不完整，已确认调用点，默认行为已复现 | `nn/tasks.py` 已显式 `weights_only=False`，但 `utils/torch_utils.py:486` 的 `strip_optimizer()` 未指定；`trainer.final_eval()` 会调用它；`engine/tuner.py` 也有默认加载 | 统一处理可信的自有训练 checkpoint；对完整模型序列化明确加载策略，或迁移为 state_dict；覆盖保存、续训、strip、最终验证。不是简单把所有外部权重设为不受限加载 |
| R06 | 运行前：入口路径、初始化与参数未冻结，已确认 | `train.py` 是另一台 Windows 的绝对路径，`test.py` 是原 Linux 路径；数据和权重未到位；n 权重可能只部分匹配当前较宽模型 | 改为参数或相对配置路径；训练前检查文件；显式记录 `task=detect`、6 类、尺度、预训练来源及 `Transferred x/y`、batch、seed、lr 和 AMP |
| R07 | 配置环境前：安装脚本可能选错解释器，已确认 | `setup_windows.ps1` 会在找到 `python` 后直接返回，不验证 3.10；本机将选中 3.14。torch/torchvision 未成对固定；`pip install -e . --no-deps` 不补全 pyproject 基础依赖 | 验证解释器版本，固定 torch/torchvision 对，补齐基础依赖；检查外部命令退出码后再打印安装成功。当前不直接运行该安装脚本 |
| R08 | 模型审计：ODConv 存在未使用 BN，已确认且有梯度证据 | `Attention.__init__()` 定义 `self.bn`，但 `forward()` 未调用；两组模块测试均发现其 weight/bias 无梯度 | 核对原设计后选择移除闲置参数或启用 BN；启用会改变计算和 batch=1 行为，不应在复现前静默改动 |
| R09 | 使用可选模块前：C2f_EAMA 参数适配有缺口，部分已复现 | `cv2` 输出 c2，但 EAMA 按 `2*int(c2*e)` 初始化；e≠0.5 会不匹配。`C2f_EAMA`、`C2f_ODConv` 未进入 `parse_model` 的内部重复数处理分支 | EAMA 按实际输出通道初始化；补全重复数处理后测试不同通道、重复数、e。当前主 YAML 不使用这两个 C2f 变体，可排在主链路之后 |
| R10 | 生成实验前：检测与生成依赖相互冲突，已确认 | 根目录固定 torch 2.9.1；`inversion/requirements.txt` 固定 torch 2.2.0/torchvision 0.17.0；FixedPoint 清单又有 torch 2.0.1 和原机器 `file://` 路径 | 检测、扩散生成分环境；在支持 5090 的 torch 上适配并锁定 diffusers/transformers/accelerate/hub。生成旧清单不能直接覆盖检测环境 |

本次也排除了两项容易误判的问题：`YOLO(...)` 遇到文件名中的 `yolov10` 会切换到 `YOLOv10`，因此不能单凭 `train.py` 的类名认定它用了 YOLOv8 损失；当前 `nn/tasks.py` 的常规权重加载已经适配 `weights_only`，遗漏的是其他路径。`default.yaml` 的 task 为 segment，但 YAML 模型入口会推断并覆盖任务；应在参数快照中明确 detect，而非直接断言当前入口必然执行分割。

## 4 模型连接的具体检查结果

层号按 `backbone + head` 从 0 开始，以下通道数由仓库 `parse_model` 的缩放规则静态推导。

| 配置 | 缩放系数 | 检测头的来源层 | 检测头输入通道 | 输出步幅 |
| --- | --- | --- | --- | --- |
| `yolov10n.yaml` | `[0.33,0.25,1024]` | `[16,19,22]`，均为特征处理层 | `[64,128,256]` | `[8,16,32]` |
| `yolov10m.yaml` | `[0.67,0.75,768]` | `[16,19,22]`，均为特征处理层 | `[192,384,576]` | `[8,16,32]` |
| `yolov10n_ODConv.yaml` | `[0.67,0.75,768]` | `[16,19,22]`，实际均为 Concat | `[576,1152,960]` | `[8,16,32]` |

当前 ODConv 配置中的 EAMA 是第 10 层，PSA 是第 11 层。EAMA 输入 576 通道可被默认分组数 32 整除，本次独立运行已通过。三个检测分支的空间步幅仍然匹配，**因此仅靠拼接尺寸检查可能发现不了层引用问题**。

若设计意图是在原 YOLOv10 拓扑中插入 EAMA，并保持原来的特征处理位置，则候选改动如下。这是需要验证的拓扑修复方案，不是已经确认的论文原配置。

| 位置 | 当前引用 | 候选引用 | 目的 |
| --- | --- | --- | --- |
| 第 19 层 Concat | `[-1,13]` | `[-1,14]` | 拼接上一条分支经过 C2f 后的特征 |
| 第 22 层 Concat | `[-1,10]` | `[-1,11]` | 使用经过 PSA 的主干特征 |
| 第 24 层 v10Detect | `[16,19,22]` | `[17,20,23]` | 使用三个处理后的尺度特征 |

在保持当前缩放系数的条件下，候选检测头输入通道应为 `[192,384,576]`。当前第 23 层虽然按顺序执行，却没有通向最终检测输出的数据路径；其参数无法通过检测输出获得训练梯度，这是计算图推断，仍需在完整模型中用梯度检查确认。修正会改变参数形状和权重匹配关系，旧权重不能默认直接复用。

此外，“缩放系数与 m 相同”不代表该模型与标准 m 完全相同：模块类型和连接也不同。首次检查尚未测量完整模型参数量；后续已实测 nc=6 时 n 基线为 2,709,380、原 ODConv 配置为 31,794,846 个参数，见本轮报告。FLOPs、正式训练峰值显存和速度仍未测量，不能使用文件注释或标准模型数字替代。

## 5 本机与服务器怎样分工

| 工作 | 本地判断 | 服务器判断及前置条件 |
| --- | --- | --- |
| 语法、路径、配置图、依赖审计 | 本次已完成主要检查 | 到服务器后再核对系统差异 |
| ODConv/EAMA 独立 FP32 检查 | 本次 GPU 检查已完成 | 实机再次检查；随后增加 AMP 与算子兼容验证 |
| 完整模型随机输入前向、梯度检查 | 本地 FP32/AMP、优化器恢复及权重重载推理已完成；原配置结构问题已记录 | 服务器首次部署时重新验证 |
| 真实小样本训练、保存、恢复、验证 | 数据到位后可从 batch=1/2、workers=0、cache=False 开始；实际容量待测 | 按同一验收步骤运行，不直接启动 150 轮 |
| 640 输入、论文 batch、完整消融与多 seed | 本机 8GB 的容量、时长未实测；不认定绝对不能训练 | 优先安排服务器；按实测峰值显存和吞吐选择 batch |
| SDXL/GNRI 生成 | 本次未执行；缺模型缓存、完整生成链路及数据 | 使用独立环境，先单样本串行检查，再评估精度、显存与批量生成 |
| ONNX/TensorRT 导出和 FPS | 尚未执行，非本阶段第一项 | 完整 PyTorch 推理通过后，单独核对自定义模块导出支持 |

本地小 batch 或缩小分辨率只能用于工程检查。正式对照需要统一输入、有效 batch、初始化和训练协议，不能把低配调试结果混入论文实验表。单卡服务器上检测训练与扩散生成宜先串行运行，实际是否并发由显存测量决定。

## 6 RTX 5090 服务器的候选配置

### 6.1 目前能确定的配置原则

NVIDIA 将 RTX 5090 列为计算能力 12.0。PyTorch 2.7 开始提供 Blackwell 与 CUDA 12.8 支持；官方历史版本页面给出了 `torch 2.9.1 + torchvision 0.24.1 + cu128` 的安装组合。因此建议先以这组作为检测环境候选，与仓库根目录的 torch 版本接近，而不是沿用论文的 torch 2.0 或生成目录的 torch 2.2。[NVIDIA GPU 表](https://developer.nvidia.com/cuda/gpus)、[PyTorch 2.7 发布说明](https://pytorch.org/blog/pytorch-2-7/)、[PyTorch 版本配对](https://pytorch.org/get-started/previous-versions/)。

这是兼容性方案，**尚未在目标服务器安装、解析全部依赖或运行验证，也不宣称是最新版本**。首次独立检查使用现有 2.14.0/cu132；后续已在本机新建 2.9.1/cu128 检测环境并通过完整模型和合成训练检查，但这不替代目标服务器验证。

| 层次 | 建议候选 | 验证要求 |
| --- | --- | --- |
| 系统 | 优先使用服务器已支持的 Linux 环境；若可选择，可采用 Ubuntu 22.04/24.04 x86_64 | 实际系统、容器镜像、权限由服务器材料确认 |
| NVIDIA 驱动 | 由管理员/镜像提供支持该 GPU 和所选 CUDA wheel 的驱动 | `nvidia-smi` 正常且真实 CUDA 运算通过；不仅查看版本字符串 |
| Python | 独立 Python 3.11 环境 `pcb-det` | 不复用原作者路径，不在 base 环境混装 |
| 检测框架 | `torch==2.9.1`、`torchvision==0.24.1`，使用官方 cu128 wheel | CUDA、卷积、torchvision 算子及自定义模块检查通过 |
| 数值库 | 首轮可考虑 NumPy 1.26.4、SciPy 1.10.1 | 避开本次已复现的 `np.trapz` 问题；SciPy 1.10.1 要求 Python <3.12；与其余依赖一起解析确认 |
| 检测依赖 | `pyproject.toml` 基础依赖，加 timm、einops、PyWavelets、huggingface-hub 等自定义路径需要的包 | 完成 `pip check` 与全部实际导入，再保存完整版本锁定文件 |
| 生成环境 | 另建 `pcb-gen`；使用支持目标 GPU 的 torch，重新验证 diffusers/transformers/accelerate/hub | 旧 SDXL 管线调用了库内部接口，版本升级可能需要改代码；不能仅解决 pip 安装就算完成 |
| 数据与缓存 | 数据、权重、HF 缓存和 runs 放入持久化目录 | 明确容量、目录权限、断开连接后的保留规则与备份方式 |

版本依据：[NumPy 1.26.4 元数据](https://pypi.org/project/numpy/1.26.4/)、[SciPy 1.10.1 元数据](https://pypi.org/project/scipy/1.10.1/)。CUDA 12.8 GA 发布说明列出的对应驱动为 Linux ≥570.26、Windows ≥570.65；这是该 Toolkit 版本的对应驱动信息，**不能把 CUDA 12.x 的通用最低驱动表当作特定 5090 显卡的可用性保证**。实际采用支持该卡的供应商驱动并做运算测试。[NVIDIA CUDA 12.8 说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)。

本模型的 ODConv/EAMA 路径使用标准 PyTorch 算子，本次未发现必须自行编译的 CUDA 扩展。第一步是验证驱动与 PyTorch wheel，不需要为了这些模块先编译 CUDA；后续如启用 xformers、TensorRT 或其他扩展，再按其要求单独配置。

### 6.2 服务器到位后的配置步骤

以下是 Linux/Conda 场景的候选流程，**本次未执行**。先检查服务器，再决定采用现有容器还是新建环境；不要在不清楚资源分配的情况下更换系统驱动。

1. 收集第 7 节信息，确认 GPU 与持久化存储。
2. 创建检测专用环境并安装成对框架：

   ```bash
   conda create -n pcb-det python=3.11 pip
   conda activate pcb-det
   python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
   ```

3. 在本仓库根目录安装检测依赖。以下为待解析验证的起点，不是完整锁定文件；显式限制主要版本，避免额外依赖替换框架或引入已发现的 NumPy 问题：

   ```bash
   python -m pip install -e . "torch==2.9.1" "torchvision==0.24.1" "numpy==1.26.4" "scipy==1.10.1" "opencv-python==4.9.0.80" "huggingface-hub>=0.23.2,<1" timm einops PyWavelets
   python -m pip check
   python -c "import torch, torchvision, ultralytics; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available()); print(ultralytics.__file__)"
   ```

   `ultralytics.__file__` 应指向这个 Fork。记录实际解析版本后再冻结依赖。ONNX、ONNX Runtime、TensorRT、Gradio 和扩散生成依赖按需另配；检测训练不需要先把根目录及两个生成清单全部混装。

4. 执行检查脚本，再验证 torchvision CUDA 算子、完整模型构建和连接。修复 R02–R06 后完成随机输入 FP32/AMP 检查，检查必要参数梯度及有限损失。
5. 数据到位后用少量已检查标注的真实训练/验证样本跑 1–3 轮，验证 best/last 保存、恢复训练、`strip_optimizer` 及最终验证。确认通过后保存 `pip freeze`、参数快照、硬件信息、日志和 commit。
6. 再扩大到 640 输入和候选 batch，测量峰值显存与每轮用时。正式实验的 batch、有效 batch 与优化器设置要对所有对照保持可比。

仅安装到第 3 步不代表 R05 已修复。PyTorch 2.6 起改变了默认反序列化行为，候选 2.9.1 同样需要检查训练末尾的权重处理路径。[PyTorch 序列化说明](https://docs.pytorch.org/docs/2.9/notes/serialization.html)。

## 7 后续需要的服务器与实验材料

这些是下次接入时的材料清单，具体硬件与使用方式后续确认。

| 类别 | 后续需要确认的内容 | 用途 |
| --- | --- | --- |
| 资源分配 | GPU 实际型号、数量、总显存和当时空闲显存，是否独占、是否有使用时限 | 决定 batch、调度和运行计划 |
| 系统 | 操作系统、CPU 核数、内存、驱动、现有 Python/Conda/容器镜像 | 决定采用和验证哪套环境 |
| 存储 | 持久化目录、剩余空间、缓存目录、运行结果保存位置 | 防止重启或容器回收后丢失材料 |
| 联网 | PyTorch/PyPI/Hugging Face/GitHub 是否可访问，是否需要镜像或离线包 | 提前准备 wheel 与模型缓存 |
| 使用方式 | 平台、终端/SSH/容器使用方式、是否需要作业队列 | 安排连接、断点与日志；访问凭据走安全交接 |
| 最小实验包 | 真实数据及标注、划分清单、论文 YAML、权重来源/哈希、原 args.yaml/日志 | 冻结尺度、数据协议及初始化，推进 P0-01/P0-02 |

Linux 服务器可先收集：

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,compute_cap --format=csv
cat /etc/os-release
lscpu
free -h
df -h
python --version
python -m pip show torch torchvision
```

建议最初预留约 100GB 持久化工作空间用于环境、模型缓存、少量数据与输出，并根据实物统计调整；这不是完整数据集的已测容量。CPU、内存和磁盘吞吐也会影响生成和训练，不能只根据“5090”给出运行时长或 batch 保证。

## 8 后续执行顺序与 P0-03 验收

1. **环境与通用兼容修复**：R01、R04、R05、R07 已完成本地修复和验证；R09 可选 C2f 参数兼容也已修复。服务器后续单独验收。
2. **模型与入口确认**：R02、R03、R06，同时向作者索取原配置；保留原版，记录候选版改动。R08 跟随模型设计确认；R06 的入口参数化已完成，实验协议仍待材料。
3. **完整模型运行验收**：前向、检测损失反向、必要参数梯度、保存/恢复/strip/验证均通过；真实小样本训练结果和日志可追溯。
4. **服务器容量与性能预检**：实测显存、batch、AMP 和吞吐后，才进入 P0-05。生成环境 R10 与 P0-04 并行推进。

| P0-03 子项 | 本轮状态 | 剩余条件 |
| --- | --- | --- |
| 本地硬件、解释器及依赖盘点 | 已完成 | 服务器到位后补其快照 |
| 模型尺度、层连接审计 | 静态与运行时梯度检查已完成 | 确认作者原配置并验证候选修正 |
| ODConv/EAMA 及完整模型 GPU 运行检查 | 本地 FP32/AMP 已完成 | 结构问题、真实数据与服务器验收仍待完成 |
| 专用环境与通用兼容修复 | 本地已完成，依赖快照已保存 | 服务器环境单独验证，论文结构配置尚未冻结 |
| 完整模型前向及梯度检查 | 本地已完成；原配置问题已记录 | 完成结构修正后复核 |
| 真实小样本训练、恢复和验证 | 待真实材料 | 基线合成训练工程检查已完成，不能替代真实数据实验 |
| 5090 实机适配与资源测量 | 待服务器材料 | 实际硬件、系统与使用方式 |

P0-03 的总状态保持“进行中”。本轮成果可直接用于下一次修复与服务器部署，但不能据此宣称论文复现完成或已有新的 AP/FPS 结果。

## 9 复查方法和代码证据

以下为首次审计脚本；当前兼容性回归与完整模型验收命令见 [本轮修复报告](P0-03_LOCAL_FIXES_ZH.md)。默认 torch.load 的行为探针与项目修复后的实际加载调用不是同一项检查。

在已有相应依赖的环境中，从仓库根目录运行：

```bash
# 仅清点环境、检查 Python 语法及模型静态连接；YAML 检查需要 PyYAML
python tools/audit_p0_03.py --output /path/to/p0-03-static.json

# 额外运行独立模块和已知兼容问题检查；需要已有 PyTorch
python tools/audit_p0_03.py --runtime --device cuda --output /path/to/p0-03-runtime.json
```

本次运行使用已存在的 Python 3.11 环境，并把 `--repo` 指向本地 Fork；输出文件未包含账号凭据或服务器地址。脚本中两个兼容问题探针和非默认 e 探针当前报错，是本报告的发现；修复后应检查对应结果是否改变。

主要代码定位（行号对应检查快照）：

- [模型 YAML](../../ultralytics/cfg/models/v10/yolov10n_ODConv.yaml)、[n 基线](../../ultralytics/cfg/models/v10/yolov10n.yaml)、[m 基线](../../ultralytics/cfg/models/v10/yolov10m.yaml)。
- [parse_model 与权重加载](../../ultralytics/nn/tasks.py)：缩放/重复数解析约 870–1030 行，常规权重加载约 768、789 行。
- [ODConv](../../ultralytics/nn/Extramodule/ODConv.py)、[EAMA 与 C2f_EAMA](../../ultralytics/nn/Extramodule/EAMAattention.py)。
- [AP 计算](../../ultralytics/utils/metrics.py)：524 行；[strip_optimizer](../../ultralytics/utils/torch_utils.py)：486 行；[训练结束流程](../../ultralytics/engine/trainer.py)：621 行。
- [训练入口](../../train.py)、[验证入口](../../test.py)、[Windows 安装脚本](../../setup_windows.ps1)、[基础依赖](../../pyproject.toml)、[检测依赖清单](../../requirements.txt)、[生成依赖](../../inversion/requirements.txt)。
