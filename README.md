# EM-CLIP

基于论文 *Efficient Motion-Centric CLIP for Compressed Video Action Recognition*
的压缩域动作识别实现。输入为真实 CoViAR I-frame、Motion Vector 和累积 Residual。
本仓库不自动下载模型，不用 RGB 全解码伪造 MV/Residual。

## 当前默认协议

新训练使用 `paper` 实现：MGSE → top-k 选择对应 I/R → MELSC → 视频/文本分类。
GSPL/LMPL 逐层共享论文 Eq.16 的 Residual 属性特征；SAG 每层加入两个新 prompt。
L_MG 为同类 multi-positive 双向 KL，L_ME 为 cosine/tau 的分类 CE，总损失默认等权。

- ViT-B/16：12 层、768 width、12 heads；CLIP 图文空间通常为 512。
- T=16、K=8；K=16 时 full T=32。Diamond 不创建 MGSE，直接采样 T=K。
- 输入 256，GOP=12，motion/classification tau 均固定默认 0.01。
- 30 epochs，LR=8e-6，cosine；dropout/stochastic depth=0。
- 训练 1x1；训练过程快速验证 1x1；独立评估脚本默认 4 temporal views x 3 crops。
- 训练 MGSE 使用真实训练类别的 label-word features；正式验证用 class_bank，绝不读取真实标签选帧。
- SSV2 禁用水平翻转；原生 BGR Residual 转 RGB，以匹配 RGB CLIP stem。
- 短视频补齐 T 个槽位时 mask 掉重复 GOP，优先选择不同 GOP。

AdamW、每 GPU batch4、WD0.2、betas=(0.9,0.98)、eps=1e-6、warmup0、full 微调、
class_bank 推理、输入归一化等是工程假设，不是论文完整披露的设置。
详细公式、假设、修复记录见 [IMPLEMENTATION_NOTES_EMCLIP.md](IMPLEMENTATION_NOTES_EMCLIP.md)。
修复前的 SSV2 训练审计保留在 [EMCLIP_PAPER_AUDIT.md](EMCLIP_PAPER_AUDIT.md)。

## 环境与数据

沿用现有 Python/PyTorch 环境。CLIP checkpoint 必须是已有的本地文件，支持官方
JIT archive 和 state dict。默认路径 `/home/fuh/CLIP-models/ViT-B-16.pt`，可用
`CLIP_CHECKPOINT` 覆盖。无需网络下载。

CoViAR 优先使用仓库 `pytorch-coviar/data_loader`，也可设置
`COVIAR_DATA_LOADER_DIR=/home/fuh/m2clip/Coviar/data_loader`。如需构建已提供的原生扩展：

```bash
cd pytorch-coviar/data_loader
bash install.sh
```

数据集配置的唯一来源为 `configs/emclip_datasets.py`。SSV2/K400 类别 CSV 已在仓库，
HMDB51/UCF101 可按真实列表目录推断类别；类别数、语义文本和标签范围均会校验。
列表支持 path num_frames label、ID num_frames label、path label 和带空格类别名。
可以通过 `--train-root/--val-root/--train-list/--val-list/--compressed-video-root` 覆盖路径。
错误样本会抛出含 dataset/list/path/GOP 的异常，不会换成其他样本掩盖错误。

## 验证后再训练

单卡 synthetic + 四数据集真实 smoke（不训练完整 epoch，缺数据/权重/扩展会明确 SKIP）：

```bash
NPROC_PER_NODE=1 bash scripts/smoke_emclip.sh
```

仅 synthetic forward/backward，不需要模型文件或数据：

```bash
python main_emclip.py --synthetic-smoke --emclip-implementation paper
python main_emclip.py --synthetic-smoke --emclip-variant diamond
```

服务器真实 SSV2 decode + no-grad forward（不构建训练 loader）：

```bash
python main_emclip.py --dataset ssv2_mpeg4 \
  --clip-checkpoint /home/fuh/CLIP-models/ViT-B-16.pt \
  --coviar-data-loader-dir /home/fuh/m2clip/Coviar/data_loader \
  --eval --preflight-only --num-workers 0 --no-pin-memory
```

四个训练脚本默认 4 个进程、每卡 batch4，并尊重外部 CUDA_VISIBLE_DEVICES。
如只用两卡，同时设置 `CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2`。脚本不强制固定 GPU 编号。

```bash
bash scripts/train_emclip_ssv2.sh
bash scripts/train_emclip_hmdb51.sh
bash scripts/train_emclip_ucf101.sh
bash scripts/train_emclip_k400.sh
```

4 卡正式 SSV2 训练示例，修复后的结构从原始 CLIP 开始：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 BATCH_SIZE=4 \
CLIP_CHECKPOINT=/home/fuh/CLIP-models/ViT-B-16.pt \
IMPLEMENTATION=paper bash scripts/train_emclip_ssv2.sh
```

K=16 和 diamond：

```bash
K=16 bash scripts/train_emclip_ssv2.sh
VARIANT=diamond K=8 bash scripts/train_emclip_ssv2.sh
```

`T`、`K`、`BATCH_SIZE`、`NPROC_PER_NODE`、`MASTER_PORT`、`NUM_WORKERS`、`OUTPUT_ROOT`
均可覆盖。脚本的额外 CLI 参数追加进完整命令并保存，例如：

```bash
bash scripts/train_emclip_ssv2.sh --emclip-train-mode freeze_text
bash scripts/train_emclip_ssv2.sh --temporal-position-encoding sinusoidal
```

第二条是论文未指定的顺序编码消融；默认 none 保留论文公式的结构假设。
每次只改一个因素，以免无法解释精度变化。`MGSE_TRAIN_TEXT_MODE=class_bank` 可做原先
训练选帧协议的对照；`MGSE_EVAL_TEXT_MODE=predicted_class` 是无需标签的推理消融。

脚本默认无 K400 EM-CLIP transfer。显式 `INIT_CHECKPOINT=/path/to/source.pth` 或
`K400_CHECKPOINT=/path/to/source.pth` 才使用 model-only 初始化；optimizer/scheduler/
scaler/epoch 全部重新开始。`RESUME` 优先，用于同一训练的完整状态恢复。
AMP 使用 GradScaler/梯度裁剪，敏感相关性和损失始终 float32；DDP 默认 unused=False。
新增日志包含 selected_unique_gops、valid_candidates 和有效分类尺度。LR 默认不随 world size 放大。

## 正式评估与旧 checkpoint

评估必须明确指定目标数据集权重，不会默认拿 K400 直接评估 SSV2。
`RESUME` 恢复目标训练权重；无需再提供原始 CLIP 文件。

```bash
RESUME=/path/to/ssv2/model_best.pth bash scripts/eval_emclip_ssv2.sh
RESUME=/path/to/hmdb51/model_best.pth bash scripts/eval_emclip_hmdb51.sh
RESUME=/path/to/ucf101/model_best.pth bash scripts/eval_emclip_ucf101.sh
RESUME=/path/to/k400/model_best.pth bash scripts/eval_emclip_k400.sh
```

上述均默认 4x3。快速评估可设置 `TEMPORAL_VIEWS=1 SPATIAL_CROPS=1`。
同一视频的 view logits 平均后只计一次准确率。GOP 不足 T 时，不同时间 view 可能相同；
方形输入的三个空间 crop 也可能相同，不会声称这些视图创造了新信息。

`--emclip-implementation auto` 对新训练选择 paper，对旧 checkpoint 自动选择 legacy，
保留旧结构、分类尺度和输入处理，支持旧 SSV2 最佳权重补做 4x3：

```bash
NPROC_PER_NODE=1 RESUME=/path/to/old_ssv2/model_best.pth \
IMPLEMENTATION=auto bash scripts/eval_emclip_ssv2.sh
```

旧模型评估成绩不代表修复后模型的成绩。显式 paper + legacy checkpoint 会报错，
不进行无法等价的参数迁移；新结构应从原始 CLIP 重新训练。新 checkpoint 保存模型和
输入配置，并检查 resume 类别顺序、optimizer 参数顺序。Latest 和 best 继续保存。

## 测试和边界

```bash
python -B -m pytest tests -q -p no:cacheprovider
python -B scripts/audit_emclip_paper.py
python -B scripts/audit_emclip_paper.py --implementation legacy
python -B scripts/audit_emclip_paper.py --ddp-smoke
```

最后一条通过 CPU Gloo/FileStore 检查两个 optimizer step，不依赖 Windows libuv TCPStore。
该检查不能替代 CUDA/NCCL 或 GPU fp16 AMP。当前本机没有真实数据、CoViAR 原生扩展、
CLIP/EM-CLIP checkpoint 或 CUDA，不能验证真实输入、正式训练精度和 GPU AMP。
验证记录见 `output_dir/emclip_audit_20261003/fix_verification.json`。

论文没有完整披露优化器、batch size、参数冻结策略及 MGSE 正式推理类别来源。
这是基于论文公式的工程复现，不保证与作者私有实现逐行一致或达到论文精度。
