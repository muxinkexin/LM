# LM：Lite-Mono 的 RA-Depth 训练机制迁移操作手册

适用：Ubuntu 22.04、单张 RTX 5090、KITTI Raw 根目录 `/home/zy/KITTI`。下文尺寸统一按 **height × width** 写。

本次修改以 LM commit `832be2653fdf6a75cc4f9ec2f1234c8c2eb5ad7d` 为基准，参考 RA-Depth commit `4f7ebb436c33b61ed31e49ad1f98d66dffc596b9`。不替换 Lite-Mono 编码器/解码器，不引入 HRNet、教师网络、额外预测头或新 Python 模块。仅修改下面 9 个已有文件。一键脚本和交付说明保存在仓库外，手册写入原本为空的 `README.md`。

**这是训练机制迁移实现，不是完整 RA-Depth 架构复现，也不是已经证实能提升 KITTI 指标的实验结果。** 当前已完成 CPU 合成数据的机制、梯度、训练/保存/评估集成检查；尚未在你的 KITTI 与 RTX 5090 上完成训练、显存或精度测量。下文提供建议的首轮训练配置，不能在未做消融前称为实测最优。

## 1. 实际改动

| 原有文件 | 修改内容 |
| --- | --- |
| `datasets/mono_dataset.py` | 从原生 RGB 生成高尺度裁剪和低尺度拼接；时序帧共享几何和颜色扰动；关闭 RA 时不生成辅助视图 |
| `layers.py` | resize/crop 内参变换、对应区域对齐、有效掩码、跨尺度一致性损失 |
| `trainer.py` | 辅助视图光度重建和一致性损失；位姿共享；开关隔离；合成自检；数据路径检查 |
| `options.py` | RA 参数、自检、数据检查、多分辨率评估和指标对比参数 |
| `networks/depth_encoder.py` | 未列出的训练尺寸使用明确的低分辨率 dilation 配置，避免 `self.dilation` 缺失 |
| `evaluate_depth.py` | 输入分辨率独立于 checkpoint；PNG/CPU 支持；按权重恢复深度头；指标 JSON 与配对对比 |
| `test_simple.py` | 单图可指定新分辨率；按训练元数据恢复模型与深度头 |
| `train.py` | seed、TF32、自检/数据检查入口 |
| `README.md` | 本操作手册 |

仓库原有外部 scheduler 导入在仓库内没有对应实现；本版本统一改用 PyTorch 原生 `CosineAnnealingLR`，在 epoch 结束后推进。还修复验证阶段 pose 网络未切到 eval、颜色扰动在不同帧重复随机采样等问题。因此这里的 baseline 是**这些公共修复下、RA 关闭的对照组**，不声称与原仓库每一步数值完全相同，也不把改进全部归因于 RA。两组必须使用同一份改后代码、初始化、训练轮数及评估协议。

训练和 minibatch 验证只加载 RGB，不读取在线 LiDAR GT；完整指标使用原仓库已有的 `splits/eigen/gt_depths.npz`。

## 2. 迁移机制及轻量化

每个样本构造三个固定画布 H×W 的视图：

- MiS：原有缩放到 H×W 的基础视图。
- HiS：原始图按 1.1～2.0 倍目标尺寸缩放，再随机裁剪 H×W；同一时序样本的帧共享裁剪参数。
- LoS：原始图缩至 0.7～0.9 倍目标尺寸，再用重叠片段拼满画布。左上原始缩小区域用于监督，其余拼接片段仅作网络上下文。

每个视图使用**同一套** Lite-Mono 深度网络；位姿由基础时序视图预测，并共享给 HiS/LoS。内参使用实际整数 resize 尺寸和 crop 偏移变换，含像素中心修正：

```text
fx' = sx * fx
fy' = sy * fy
cx' = (cx + 0.5) * sx - 0.5 - dx
cy' = (cy + 0.5) * sy - 0.5 - dy
```

辅助光度损失排除拼接区、投影越界、相机后方区域及无效 SSIM 邻域。一致性损失仅比较同一场景的对应区域，使用基础分支 automask 和边界掩码。两条预测分支都可接收梯度。模型输出是归一化逆深度编码，不是立体匹配的像素 disparity，**不会乘输入缩放倍数**。

总损失为：

```text
L = L_base
  + ra_photo_weight * mean(L_photo+smooth_hi, L_photo+smooth_lo)
  + ramp * ra_consistency_weight * mean(L_cons_hi, L_cons_lo)
L_cons = masked_mean(0.85 * SSIM_error + 0.15 * L1_error)
ramp = min(1, (epoch + 1) / ra_warmup_epochs)
```

`--no_ssim` 时一致性只用 L1。只选 high/low 时 mean 只覆盖选中的分支。权重、平均方式和 5 epoch warmup 是为 Lite-Mono 提供的可调起点，与官方 RA-Depth 的三分支求和、权重 1 设置不同，需用消融验证。

| 项目 | RA 关闭 | RA 开启、both |
| --- | --- | --- |
| 深度网络可学习参数 | 原 Lite-Mono | 同一套参数，增加 0 |
| 每个训练 batch 深度前向次数 | 1 | 3 |
| 评估/部署深度前向次数 | 1 | 1 |
| 训练时间与显存 | 对照值 | 会增加，具体以实测为准 |
| 同一输入尺寸下推理网络结构 | 原结构 | 相同结构 |

CPU 自检统计，`--scales 0` 时 encoder+decoder 的精确参数量：Lite-Mono-tiny 为 **2,146,793**，Lite-Mono 为 **3,073,809**。该数值不含训练专用 PoseNet。提高推理输入尺寸本身仍会增加计算量。

## 3. 一键安装

先解压交付包到仓库外，例如 `/home/zy/lm_ra_delivery`。脚本内已包含 9 个文件的完整修改内容，无需再下载补丁。

首次系统准备：

```bash
sudo apt-get update
sudo apt-get install -y git python3.10-venv
nvidia-smi
```

运行：

```bash
cd /home/zy/lm_ra_delivery
bash install_lm_ra.sh /home/zy/LM --env /home/zy/.venvs/lm-ra
source /home/zy/.venvs/lm-ra/bin/activate
cd /home/zy/LM
```

脚本执行以下操作：

1. 目标不存在时 clone 公开 LM 仓库并 checkout 上述固定 commit；目标已存在时检查 commit 和原文件 SHA-256。
2. 若任何待修改文件有未知改动，停止并列出冲突，避免覆盖你的工作。已经完整安装的文件会识别为幂等重跑。
3. 在仓库外 `$HOME/LM_RA_backups/` 创建时间戳备份和恢复清单，再修改原有 9 个文件。
4. 在仓库外创建 Python 3.10 venv；安装 PyTorch **2.7.1 / torchvision 0.22.1 / cu128** 和固定的主要依赖。
5. 运行 GPU 分配、卷积反向及合成机制自检。**不启动正式 KITTI 训练，不自动改驱动，不下载 KITTI。**

RTX 5090 需要支持 Blackwell 的 PyTorch CUDA 构建。这里选 cu128；不照搬旧 Lite-Mono/RA-Depth 的 torch 1.x 环境。驱动须支持该 GPU 和 CUDA 12.8，脚本以实际 CUDA 运算结果检查。官方依据见文末 PyTorch 2.7 说明。PyTorch wheel 自带 CUDA runtime，此项目不编译 CUDA 扩展，通常无需单独安装系统 CUDA Toolkit。

可选模式：

```bash
# 只安装代码；用于已经有兼容环境的情况，不安装依赖或运行 GPU 自检。
PYTHON_BIN=python3 bash install_lm_ra.sh /home/zy/LM --code-only

# CPU 调试环境；不是 5090 正式训练配置。
bash install_lm_ra.sh /home/zy/LM --env /home/zy/.venvs/lm-ra-cpu --cpu
```

安装脚本、venv、备份、数据、模型权重和日志均不需要放进源码仓库。正常 Python 运行可能生成解释器缓存；脚本设置 `PYTHONDONTWRITEBYTECODE=1` 避免自检阶段生成缓存文件。

## 4. KITTI 路径与启动检查

根目录下面应直接按 KITTI Raw 的日期/drive 布局展开。例如下面的实际文件应存在：

```text
/home/zy/KITTI/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000001.png
```

不是让 `data_path` 指到某个 `image_02/data` 子目录。先检查全部 train/val/test 列表引用的帧以及已带的测试 GT 数量：

```bash
python train.py --check_data --data_path /home/zy/KITTI --png --scales 0
python train.py --self_test --model lite-mono --scales 0
```

如果你的图像是 JPG，训练、检查和评估命令都去掉 `--png`。路径检查会列出缺失的 RGB 文件；它不校验图像内容，也不会下载数据。深度 GT 和 split 顺序必须相匹配，不要随意替换原仓库 GT。

## 5. 首轮推荐训练配置

建议先固定 **192×640、batch 12、30 epochs、AdamW、单尺度 0、ResNet18 pose、seed 42**。这是便于验证机制及控制训练成本的起点。5090 的实际显存峰值仍受驱动、PyTorch 和并行数据加载影响；如 OOM，两组一起降至 batch 8。先保持 FP32；`--allow_tf32` 可在两组都添加后比较速度和指标。本版本没有 AMP、DDP 或 torch.compile 专用路径。

### 5.1 推荐：标准 Lite-Mono + 对应 ImageNet 初始化

Lite-Mono 标准版仍是约 3.1M 参数，比 tiny 多一点容量。官方 README 的 standard `lite-mono` 对应 **ImageNet Pretrained: yes** 链接指向 `lite-mono-pretrain.pth`。通过文末官方链接下载到仓库外 `/home/zy/pretrained/lite-mono-pretrain.pth`。不要把 KITTI 已训练的 `encoder.pth` 当作此 ImageNet 文件，也不要用 standard 权重初始化 tiny。

下面先定义两组公用参数。若在新终端运行，需要重新执行此代码块：

```bash
source /home/zy/.venvs/lm-ra/bin/activate
cd /home/zy/LM
PT=/home/zy/pretrained/lite-mono-pretrain.pth
test -f "$PT" || { echo "请先下载匹配的 ImageNet 权重: $PT"; exit 1; }
COMMON=(
  --data_path /home/zy/KITTI --png
  --log_dir /home/zy/LM_runs
  --model lite-mono --mypretrain "$PT" --weights_init pretrained
  --split eigen_zhou --height 192 --width 640
  --scales 0 --frame_ids 0 -1 1
  --pose_model_type separate_resnet --pose_model_input pairs
  --batch_size 12 --num_workers 8 --num_epochs 30
  --lr 0.0001 5e-6 31 0.0001 1e-5 31
  --weight_decay 0.01 --drop_path 0.2
  --seed 42 --save_frequency 1 --log_frequency 250
)
```

Baseline 启动，默认 RA 关闭：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py "${COMMON[@]}" --model_name baseline_s42
```

RA 机制启动：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py "${COMMON[@]}" --model_name ra_s42 \
  --ra_depth --ra_views both \
  --ra_high_range 1.1 2.0 --ra_low_range 0.7 0.9 \
  --ra_photo_weight 1.0 --ra_consistency_weight 0.1 --ra_warmup_epochs 5
```

两条命令顺序运行，不要在单卡上同时启动。`--weights_init pretrained` **只控制 PoseNet 的 ImageNet 初始化**；Lite-Mono 深度编码器需要显式的 `--mypretrain`。首次 PoseNet 初始化会从 torchvision 的公开权重地址下载 ResNet18。

Epoch 从 0 开始，30 轮训练的最后权重位于：

```text
/home/zy/LM_runs/baseline_s42/models/weights_29
/home/zy/LM_runs/ra_s42/models/weights_29
```

### 5.2 不依赖预训练文件的完整可启动对照

如果暂时没有标准 Lite-Mono 的 ImageNet 权重，可以直接使用以下两条命令。它们使用 LM 默认的 tiny 架构，深度与位姿均从随机初始化开始，**不用于声称复现原论文预训练精度**：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_path /home/zy/KITTI --png --log_dir /home/zy/LM_runs \
  --model lite-mono-tiny --weights_init scratch --model_name tiny_base_s42 \
  --split eigen_zhou --height 192 --width 640 --scales 0 \
  --batch_size 12 --num_workers 8 --num_epochs 30 --seed 42 \
  --lr 0.0001 5e-6 31 0.0001 1e-5 31

CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_path /home/zy/KITTI --png --log_dir /home/zy/LM_runs \
  --model lite-mono-tiny --weights_init scratch --model_name tiny_ra_s42 \
  --split eigen_zhou --height 192 --width 640 --scales 0 \
  --batch_size 12 --num_workers 8 --num_epochs 30 --seed 42 \
  --lr 0.0001 5e-6 31 0.0001 1e-5 31 \
  --ra_depth --ra_views both --ra_consistency_weight 0.1 --ra_warmup_epochs 5
```

### 5.3 正式长训前：20-step 冒烟测试

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_path /home/zy/KITTI --png --log_dir /home/zy/LM_runs \
  --model lite-mono --weights_init scratch --model_name ra_smoke \
  --height 192 --width 640 --scales 0 --ra_depth \
  --batch_size 2 --num_workers 2 --num_epochs 1 --max_train_steps 20 \
  --log_frequency 5 --seed 42
```

检查没有 NaN/Inf、可读出 `weights_0`，且 TensorBoard 中存在 `ra/photo_high`、`ra/photo_low`、`ra/consistency_high`、`ra/consistency_low`、`ra/weight`、`loss_base`。这些说明路径生效，**不能单凭 loss 下降证明泛化改善**。验证日志只跑原有基础视图，不含 RA 项，这是预期行为。

```bash
tensorboard --logdir /home/zy/LM_runs --bind_all
```

不要用 smoke 的权重替代正式训练结果。

## 6. 验证“其他分辨率”是否实际改善

### 6.1 单次评估

必须把同一 checkpoint 的网络构造配置与本次输入分辨率分开。本版本根据 checkpoint 的训练 H/W 构造 dilation，使用 `--eval_height/--eval_width` 控制真正输入。只把 `--height/--width` 改成另一组数字不是本版本的评估入口。

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py \
  --data_path /home/zy/KITTI --png \
  --load_weights_folder /home/zy/LM_runs/ra_s42/models/weights_29 \
  --model lite-mono --eval_height 320 --eval_width 1024 \
  --eval_batch_size 4 --num_workers 8 \
  --metrics_json /home/zy/LM_results/ra_s42/320x1024.json
```

评估不加 `--ra_depth`，训练视图不参与评估，也无需 `--eval_mono`（LM 没有此选项）。默认是单目 median scaling、Eigen test split、原有 crop、最大深度 80m，不启用 flip post-process。新权重会记录 `model/scales/height/width/ra_depth`；旧权重没有 model 元数据时必须指定匹配的 `--model`。常见旧版 `[0]` 和 `[0,1,2]` head 会自动识别。

非 32 倍数输入也支持：右侧/底部 replicate-pad 至 32 的倍数，预测后裁掉 padding。例如 `375×1242` 实际网络画布为 `384×1248`；这不等同于严格零 padding 的原生尺寸网络。训练尺寸仍要求 ≥64 且为 32 的倍数。

### 6.2 两组、多分辨率统一评估

以下固定使用第 29 epoch 作为预先约定的 checkpoint，避免用测试集挑选最优 epoch。若用独立 validation 选模，两组必须用同一选模规则，之后再执行测试评估。

```bash
set -e
for RUN in baseline_s42 ra_s42; do
  for HW in "128 416" "192 640" "256 832" "320 1024" "384 1280" "375 1242"; do
    read -r H W <<< "$HW"
    CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py \
      --data_path /home/zy/KITTI --png \
      --load_weights_folder "/home/zy/LM_runs/$RUN/models/weights_29" \
      --model lite-mono --eval_height "$H" --eval_width "$W" \
      --eval_batch_size 4 --num_workers 8 \
      --metrics_json "/home/zy/LM_results/$RUN/${H}x${W}.json"
  done
done
python evaluate_depth.py --compare_metrics \
  /home/zy/LM_results/baseline_s42 /home/zy/LM_results/ra_s42
```

tiny 实验把 RUN 替换为 `tiny_base_s42 tiny_ra_s42`，命令中的 model 改为 `lite-mono-tiny`。输入宽高大致覆盖训练尺寸的 0.65～2 倍；高分辨率 KITTI 若本身超出传感器有效信息，不意味着会增加场景信息。

`--compare_metrics` 检查分辨率、split、图像数、列表 SHA-256、median scaling、深度缩放及 post-process 协议是否一致，输出 AbsRel、RA−baseline 的变化、a1 变化以及 5,000 次配对图像 bootstrap 的区间。**AbsRel 差值小于 0 为改善，a1 差值大于 0 为改善。** 相邻 KITTI 帧相关，该区间可能偏乐观，不替代 drive/scene 层面的统计，也不替代多训练 seed。

认定机制“有效”至少需要：

1. 相同训练尺寸和相同测试协议下，RA 在多个偏离 192×640 的尺寸上更稳定，而不仅是在训练尺寸上更好。
2. 报告训练尺寸的绝对精度与偏离尺寸的退化量，同时观察低/高分辨率两侧，防止只适应放大裁剪。
3. 使用至少 3 个训练 seed，例如 42、43、44；两组都改 `--seed` 和 run 名字。seed 固定不保证 CUDA 的 bitwise 完全确定性。
4. 保持初始化、batch、epoch、数据 split、训练样本数、优化器和推理后处理一致。RA 每 step 的训练计算更多，应另外报告耗时与峰值显存，不将其当作等算力实验。
5. 使用独立 validation 调整一致性权重（建议 0.05/0.1/0.2）和增强范围；不要根据 Eigen test 指标反复挑参数。

跨尺度一致性变小只能证明优化目标被学习，不能排除平滑/常数预测。完整 GT 指标与多分辨率退化对照才是判断依据。当前交付没有填写任何 KITTI 精度提升数字。

## 7. 开关、消融与加载已有权重

| 目的 | 参数 |
| --- | --- |
| baseline、完全关闭额外训练视图 | 不加 `--ra_depth` |
| 完整迁移 | `--ra_depth --ra_views both --scales 0` |
| 只保留高尺度分支 | `--ra_depth --ra_views high --scales 0` |
| 只保留低尺度分支 | `--ra_depth --ra_views low --scales 0` |
| 增强+辅助光度监督，不加一致性 | `--ra_depth --scales 0 --ra_consistency_weight 0` |
| 保留一致性，不加辅助光度项 | `--ra_depth --scales 0 --ra_photo_weight 0` |
| 不做一致性 warmup | `--ra_warmup_epochs 0` |
| 纯 L1、不用 SSIM | `--no_ssim` |
| 评估 CPU 调试 | `--no_cuda --eval_batch_size 1 --num_workers 0` |
| THOP 单次估计 | 评估时 `--profile`；估计不含所有算子，不是实测吞吐 |

仅把一致性权重置零仍有辅助视图光度损失；要完全关闭机制，应去掉 `--ra_depth`。分支选择和权重在每次启动时可改，权重文件结构不增加 RA 专属参数。

本次 RA 训练路径明确支持：KITTI 单目、`separate_resnet`、`pairs`、`--scales 0`。不支持 RA 与 stereo、predictive_mask 或 v1_multiscale 组合，参数检查会直接报错。baseline 可继续使用原多尺度 `[0,1,2]`，但与 RA 公平对照时应统一为 `[0]`。

从已经训练的 baseline 权重启用 RA 微调：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_path /home/zy/KITTI --png --log_dir /home/zy/LM_runs \
  --model lite-mono --weights_init scratch --model_name ra_finetune \
  --load_weights_folder /home/zy/LM_runs/baseline_s42/models/weights_29 \
  --models_to_load encoder depth pose_encoder pose \
  --height 192 --width 640 --scales 0 --ra_depth \
  --batch_size 12 --num_workers 8 --num_epochs 10 --seed 42 \
  --lr 0.00003 0.000001 11 0.00003 0.000001 11
```

这里是**仅加载模型权重的微调**：优化器、epoch 和 scheduler 重新开始，不是精确断点续训；不同时传 `--mypretrain`。编码器 dilation 没有记录为 tensor，因此微调的 `--height/--width` 应与来源 checkpoint 的训练尺寸一致。用于评估的 checkpoint 会自动读取其训练尺寸。

如果要比较微调收益，另设从同一 baseline checkpoint 开始、相同额外训练步数但关闭 RA 的微调对照，不能把多训练 10 epochs 的 RA 与只训练 30 epochs 的 baseline 直接归因比较。本版本保存 optimizer 权重以保留检查信息，但不提供 epoch/scheduler/RNG 完整恢复功能。

## 8. 单图多分辨率推理

```bash
CUDA_VISIBLE_DEVICES=0 python test_simple.py \
  --load_weights_folder /home/zy/LM_runs/ra_s42/models/weights_29 \
  --image_path /home/zy/test.png --model lite-mono \
  --eval_height 375 --eval_width 1242
```

结果写到输入图片所在目录。依旧只运行一次 Lite-Mono 深度网络；单目自监督深度存在尺度歧义，median scaling 的测试结果不等同于绝对米制深度预测。

## 9. 自检、回退与故障定位

`python train.py --self_test --scales 0 --model lite-mono` 不读取 KITTI、不下载预训练文件，检查：原图生成的高低尺度画布、关闭时无辅助张量、像素中心内参变换、对应区域严格对齐、拼接区不参与一致性、零 mask 的数值安全、双分支梯度、开关前向次数、pose 的 train/eval 切换，以及 128×416 / 192×640 / 320×1024 / 375×1242 的输出。

交付时已用 CPU PyTorch 2.7.1 / torchvision 0.22.1 / NumPy 1.26.4 / Pillow 11.2.1 / timm 1.0.15 / scikit-image 0.25.2 完成检查。两种架构均通过上述自检。另外使用临时合成 PNG、真实 Trainer 初始化和优化器进行 baseline/RA 各 2-step 训练、验证、模型保存，随后实际 evaluator 加载 checkpoint，在 64×128 和 75×135 上读 PNG、算测试指标、输出 JSON、执行配对比较。**临时合成测试不用于评价 KITTI 精度，CPU 也不验证 5090 的性能。**

| 症状 | 处理 |
| --- | --- |
| `sm_120` 不兼容、`no kernel image` | 确认进入脚本创建的环境；检查 `torch.version.cuda` 为 12.8、wheel 支持 Blackwell，驱动兼容 |
| 找不到 RGB | 运行 `--check_data`；检查 Raw 根目录、日期层级与 `--png` |
| ImageNet 权重不匹配 | standard/tiny/small 不混用；检查是否下载了网页而非 `.pth`；使用原有安全加载机制 |
| OOM | 两组同降 batch 到 8 或 4，先用 `--ra_views high` 定位分支开销，随后恢复完整对照 |
| 开 RA 后未记录一致性 | 训练要有 `--ra_depth --scales 0`，看 train 而非 val 项；完整 `opt.json` 保存在 run 的 models 目录 |
| 其他分辨率结果几乎没变 | 看终端 `Computing predictions with size WxH`，确认用了新的 eval 参数 |
| 权重头加载失败 | 检查模型名和 scales；本版本严格检查关键权重，避免随机头参与评估 |
| 安装提示未知改动 | 原始目标文件不匹配；先保留你的改动，再对固定 commit 的干净 checkout 安装；不要盲目强制覆盖 |

安装日志会显示备份目录。需要恢复时，把下面 `BACKUP` 换成实际目录；此操作只还原本次改动的文件，保留其它源码和训练输出：

```bash
BACKUP=/home/zy/LM_RA_backups/实际时间戳目录
python3 - "$BACKUP" <<'PY'
from pathlib import Path
import json, shutil, sys
backup = Path(sys.argv[1]).resolve()
manifest = json.loads((backup / 'manifest.json').read_text())
repo = Path(manifest['repo']).resolve()
for name in manifest['files']:
    shutil.copy2(backup / name, repo / name)
print('Restored:', repo)
PY
```

在恢复前如又手动修改过这 9 个文件，请先另行保存那些新改动。

## 10. 官方参考

- [LM 仓库](https://github.com/muxinkexin/LM)
- [RA-Depth 论文](https://arxiv.org/abs/2207.11984)
- [RA-Depth 官方代码](https://github.com/hmhemu/RA-Depth)
- [Lite-Mono 官方说明与预训练下载入口](https://github.com/noahzn/Lite-Mono)
- [standard Lite-Mono ImageNet 预训练文件的官方公开分享](https://surfdrive.surf.nl/index.php/s/InMMGd5ZP2fXuia)
- [PyTorch 2.7：Blackwell 和 CUDA 12.8 支持](https://pytorch.org/blog/pytorch-2-7/)

使用这些机制撰写论文或报告时，请引用 Lite-Mono 与 RA-Depth，并说明这里保留 Lite-Mono 架构、修改损失权重与掩码实现的迁移设置。
