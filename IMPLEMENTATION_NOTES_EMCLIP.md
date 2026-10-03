# EM-CLIP Implementation Notes

The root-level `dataset_coviar.py` supplied with the project is the canonical
compressed-video data implementation. The rest of EM-CLIP uses its structured
I/MV/Residual output directly.

## Module Map

- MGSE and ACG: `models/emclip_mgse.py`
- MELSC, IFE, GSPL, LMPL, SAG: `models/emclip_melsc.py`
- End-to-end model and prompt text encoder: `models/emclip.py`
- Original CLIP checkpoint parsing and coverage audit: `models/clip_checkpoint.py`
- `L_MG`: `losses/emclip_loss.py`
- CoViAR dataset and synchronized transforms: `dataset_coviar.py`
- Legacy dataset import compatibility (no independent implementation): `datasets/compressed_video_dataset.py`
- DDP/AMP/checkpoints: `engine_emclip.py`, `main_emclip.py`
- Dataset configs: `configs/emclip_datasets.py`

## Formula Mapping

| Equations | Paper operation | Code |
| --- | --- | --- |
| 1 | MV patch/CLS/position embedding | `PatchTokenEncoder.embed_patches` |
| 2-4 | MV Transformer and CLS projection | `MGSE._encode_motion` |
| 5-6 | Non-affine standardization and cosine | `_prepare_features`, `_normalize_token_features` |
| 7 | Temporal softmax, valid category-word average | `_ground_truth_saliency`; class bank is an inference assumption |
| 8 | Top-k and I/R retrieval | `select_topk_indices`, `gather_temporal` |
| 9-10 | Independent I/R embedding | `MELSC._initial_tokens` |
| 11-14 | I query and Residual-guided GSPL | `MELSC._layer_prompts`, `gs_attn` |
| 15-18 | Shared Residual FC/LN, LMPL | `lm_r_proj`, `lm_ln`, `lm_attn` |
| 19-22 | Fresh two-token SAG; I/R Transformer blocks | `MELSC.forward` |
| 23 | Temporal MHSA/FFN/average pooling | `temporal_blocks`, final pooling/projection |
| 24 | Bidirectional, multi-positive KL | `motion_text_kl_loss` |
| 25 | Video/text cosine divided by fixed tau | `EMCLIP.forward`, `classification_temperature` |
| 26 | L_MG + L_ME | `EMCLIP.forward` |
| 27-28 | GSPL multi-head attention details | `gs_attn` |
| 29-30 | Batch-wise alignment distributions | `motion_text_kl_loss` |

The paper omits temporal aggregation implementation details; the existing
pre-norm residual Transformer and final CLIP projection are retained as recorded
engineering choices. No temporal position encoding is silently assumed.

## Key Tensor Shapes

- Dataset output: I `[T,3,H,W]`, MV `[T,2,H,W]`, R `[T,3,H,W]`
- Batch: I `[B,T,3,H,W]`, MV `[B,T,2,H,W]`, R `[B,T,3,H,W]`
- MGSE motion features: `[B,T,D_text]`
- MGSE saliency: `[B,T]`
- Selected indices: `[B,K]`
- MELSC tokens: `z_i`, `z_r` as `[B,K,N,D]`, `N=1+(H/16)*(W/16)`
- GSPL/LMPL: `[B,K,D]`
- Video features: `[B,D_text]`
- Logits: `[B,C]`

## CoViAR Calls

`CoviarDataSet` in `dataset_coviar.py` uses `coviar.get_num_frames`, optional
`coviar.get_num_gops`, and `coviar.load`. It calls:

- I-frame: `load(path, gop_idx, 0, 0, False)`
- MV: `load(path, gop_idx, last_p_pos, 1, False)`
- Residual: `load(path, gop_idx, last_p_pos, 2, True)`

CoViAR is loaded lazily after argparse so `--coviar-data-loader-dir` selects the
requested native extension. When available, `get_num_gops(path)` is used for
GOP bounds; the list-file frame count is not trusted for native decoder indices.
Run `--preflight-compressed-inputs` with one process to synchronously decode the
first sample before starting DDP.

MV uses the last valid P-frame without accumulation. Residual uses `accumulate=True`
to obtain the difference from the GOP I-frame after cumulative motion compensation.
The native C implementation follows accumulated reference coordinates: this is
neither a pixel-wise sum of P-frame residuals nor an unwarped P_last minus I.
The paper does not specify the MV accumulate boolean; `False` is a documented assumption. If a GOP has no valid P-frame, MV/R fall back to explicit zeros for that GOP only.

The choices are configurable as diagnostic ablations with `--mv-accumulate`
and `--no-residual-accumulate`, while paper reproduction defaults remain direct
last-P MV and I-frame-relative cumulative residual. A decoder returning `None`
for a valid frame is an error with dataset/list/path/GOP context; it is never
silently replaced by another sample or by zeros.

## Sampling

Videos are split into `T` non-overlapping GOP segments. Training randomly picks one GOP per segment. Evaluation picks deterministic per-view offsets. If GOP count is smaller than `T`, tensor slots repeat uniformly, but only the
first occurrence of each GOP has a true `valid_mask` in the paper implementation.
ACG softmax and top-k exclude the other occurrences. If there are fewer than K
distinct GOPs, all valid GOPs are kept before the best is repeated to fill K.
`--duplicate-gop-policy keep` restores the historical behavior.

If GOP<T, all distinct available GOPs already fit in each view; the temporal views
can therefore coincide. There is no fabricated extra information. Three crops
can also coincide on square inputs. Report unique GOP counts and recognize this
limit when interpreting 4x3 evaluation.

The named presets use `T=16,K=8` and `T=32,K=16` for full EM-CLIP. Because
Diamond has no MGSE candidate-selection stage, `emclip_diamond_b16_k8` and
`emclip_diamond_b16_k16` sample the final GOPs directly with `T=K=8` and
`T=K=16`, respectively.

## Transforms and Normalization

Resize, crop, and flip parameters are shared across I/MV/R. Horizontal flip negates MV x. SSV2 flips are disabled by default because direction
labels change under mirroring; `--horizontal-flip on` is rejected for SSV2 until a
validated label permutation is implemented. Resize scales MV x and y by width and height ratios. I uses CLIP mean/std. MV is clamped to `[-20,20] / 20`. Native BGR residuals are reordered to RGB before entering the RGB CLIP-initialized
stem (`--residual-channel-order rgb`). Legacy checkpoints use BGR automatically.
Residual is divided by 255 and clamped to `[-1,1]`; CLIP mean/std is not applied to residual because residual is not RGB appearance. These scales are centralized in `CoviarDataSet` and exposed as `--mv-clamp`, `--residual-scale`, and `--residual-clamp`.

## Position Embedding and Patch Init

All ViT branches interpolate absolute position embeddings with bicubic interpolation. MV patch weights can be initialized from RGB weights by channel mean, repeated to two channels, scaled by `3/2`; see `build_mv_patch_embed_weight`.

## Branch Independence

I, residual, and MV encoders are separate module instances. They do not share `nn.Module` objects.

MELSC executes the matching I-frame and Residual CLIP block at every visual
layer, including the last Residual block. Prompts at layer `l` use the incoming
I/R CLS states; the updated Residual state is therefore consumed by layer
`l+1`. The final updated Residual state is exposed in debug output but is not
fused into the paper-specified final I-CLS classifier. Its final block,
`ln_post`, and projection remain frozen because they have no loss path; this
keeps DDP compatible with `find_unused_parameters=False` without inventing an
extra Residual classification term.

## Pretrained CLIP Loading and Audit

`--clip-checkpoint` and `--resume` have deliberately different loaders.
`--clip-checkpoint` first calls `torch.jit.load(...).eval()` so an original OpenAI
CLIP JIT archive is converted to a tensor state dict before any branch load. If
and only if JIT loading fails, it falls back to `torch.load(...,
weights_only=False)` and accepts a direct state dict, `{"state_dict": ...}`,
`{"model": ...}`, or an `nn.Module`. Non-tensor values and nonexistent paths are
reported as errors. A leading `module.` prefix is removed.

OpenAI CLIP keys are mapped as follows:

- `visual.*` (with `transformer.resblocks.*` renamed to `blocks.*`) initializes
  both `melsc.i_encoder` and `melsc.r_encoder` through independent copies.
- The same visual tensors initialize `mgse.motion_encoder`, except `conv1.weight`
  is changed from three channels to two by RGB channel mean, repeat, and `3/2`
  scaling.
- `token_embedding`, text `transformer.resblocks`, `positional_embedding`,
  `ln_final`, and `text_projection` initialize `text_encoder`; `logit_scale` is
  copied to the top-level CLIP key for compatibility. In the paper implementation
  it is frozen and ignored in favor of the fixed classification tau=0.01. The
  legacy implementation still learns the clamped CLIP scale.
- `visual.proj` initializes the I, residual, and MV visual projections.

CLIP ViT-B/16 does not use the visual width for its text transformer: visual
width/heads are 768/12, while text width/heads are 512/8 and the shared embedding
dimension is 512. These are separate `EMCLIPConfig` fields; conflating both
widths is rejected by the checkpoint shape audit.

For a 224-source/256-target B/16 checkpoint, the 14x14 source visual positional
grid is bicubically interpolated to 16x16 during initialization. Forward still
performs dynamic interpolation for smoke-test or non-default resolutions.
Loading reports source/loaded tensor counts, loaded/total parameter numel,
coverage, and missing/unexpected/shape-mismatch keys. I, R, and Text require at
least 99% coverage; MV requires at least 99% after excluding the intentionally
converted two-channel `conv1`. Required stem, position, normalization, and
projection tensors are checked explicitly.

`--pretrained-audit-only` stops before dataset or DataLoader construction. It
prints the four branch reports and key parameter statistics, then runs a small
dataset-free no-grad forward through all layers and requires every tensor output
to be finite. Under `torchrun`, all ranks participate and the process group is
destroyed in `finally` even if the original error propagates.

`--init-checkpoint` strictly loads only `checkpoint["model"]` from an EM-CLIP
training checkpoint. It is an optional cross-dataset transfer path; all four training launchers now
default to original CLIP without a hard-coded K400 model. Set INIT_CHECKPOINT
explicitly for an ablation. Target optimizer, scheduler, scaler, epoch, and best
accuracy remain fresh. Evaluation requires a target RESUME checkpoint or an
explicit INIT_CHECKPOINT transfer ablation; it never silently evaluates K400 on SSV2.

`--resume` restores the complete model/optimizer/scheduler/scaler/epoch/best
state and is reserved for the same run. It is mutually exclusive with
`--init-checkpoint`. After resume, the restored scheduler step is checked
against the target run's total steps; an out-of-range cross-dataset scheduler
raises before training rather than clamping cosine LR to zero.

## Compute accounting

`--profile-compute` profiles one batch-size-1 inference forward after a warmup
that creates the evaluation class-text cache. It reports supported-operation
FLOPs and GFLOPs/video together with measured latency, videos/second, and CUDA
peak allocated memory. This is the repeated video path, not the one-time class
text encoding cost. PyTorch does not provide FLOP formulas for every operator,
so the reported FLOPs are explicitly logged as a supported-operator count and
must not be interpreted as an exact architecture-wide total.
Latency is measured in an ordinary warm forward outside the profiler context,
and the measurement follows the run's AMP setting.

Training and validation epoch dictionaries also report wall time and global
videos/second. Training includes global candidate GOPs/second (`videos * T`),
which measures compressed-domain candidate processing volume rather than FLOPs.
For distributed runs, sample counts use SUM reduction and wall time uses MAX
reduction, yielding aggregate throughput limited by the slowest rank.

## MGSE Label Leakage

- `ground_truth`: uses true labels and is blocked in eval unless `--allow-mgse-label-leakage-for-diagnostic` is set.
- `class_bank`: default for validation/test; it never reads labels for selected indices.
- `--mgse-train-text-mode ground_truth`: paper category-conditioned training,
  independent of validation mode. This is the new paper model's default;
  validation remains `--mgse-text-mode class_bank`. Legacy defaults use class
  bank for both stages. A shared model can therefore train and validate normally.
- `predicted_class`: predicts a class from motion features first, then uses that class text.

Evaluation normally passes no labels into model selection. For the explicitly
leaky diagnostic only, `engine_emclip.evaluate` passes labels through the separate
`mgse_labels` argument; classification loss remains disabled for individual
views. This makes the warning flag functional without conflating selection labels
with training targets.

## CLIP Text Tokenization and Attention

`models/clip_tokenizer.py` implements the OpenAI CLIP byte-level BPE tokenizer
against the vendored official `bpe_simple_vocab_16e6.txt.gz` asset (SHA-256
`924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`).
The tokenizer validates the 49,408-entry vocabulary and SOT/EOT ids 49406/49407.
Label-word masks are the exact BPE span of the `{}` placeholder, so padding,
SOT, and EOT are excluded. The text Transformer uses the same causal attention
mask as CLIP. Entering training clears evaluation text caches; a later evaluation
therefore cannot reuse features from an older text-encoder state.

## Losses and DDP

`L_MG` constructs multi-positive targets where samples with identical labels are positives. It uses `F.kl_div(..., reduction="batchmean")`. Feature all-gather preserves gradients when `torch.distributed.nn.functional.all_gather` is available and falls back safely otherwise. Label all-gather is no-grad.

The gradient-preserving fallback is a custom autograd all-gather with an
all-reduce in backward; it does not detach remote features. Training forbids
splitting a DataLoader batch while `L_MG` is enabled because a per-micro-batch
all-gather would change the contrastive denominator and same-class positive bank.
Distributed validation uses a non-padding sampler, so metric sums never include
the duplicate samples inserted by PyTorch's training `DistributedSampler`.

## Hyperparameters

Paper-specified defaults: epochs 30, LR `8e-6`, cosine schedule, input 256, tau `0.01`, ViT-B/16, GOP size 12, `T=16`, `K=8`.

Engineering assumptions: AdamW, betas `(0.9,0.98)`, eps `1e-6`, weight decay `0.2`, warmup 0, batch size per GPU 4, full-batch forward for `L_MG`, grad clip 1.0, seed 1024, AMP on scripts, workers 8, and pinned memory. LR is not scaled by world size unless `--scale-lr-by-global-batch` is passed. AMP starts with scale 1024 because `tau=0.01` amplifies alignment gradients; sensitive similarity and loss operations explicitly disable autocast and run in float32. A detected fp16 backward overflow uses the standard GradScaler skip/backoff path without advancing the scheduler. Persistent overflow raises after eight consecutive batches with parameter names and scale history; non-AMP non-finite gradients raise immediately.

The paper does not publish source code or fully specify optimizer, batch size,
freezing, and MGSE inference details. These choices are engineering assumptions;
this implementation does not claim line-by-line identity with the private code or
guaranteed reproduction of the reported accuracy.

## Known Differences

No runtime network download is attempted. The shell launchers default to
`CLIP_CHECKPOINT=/home/fuh/CLIP-models/ViT-B-16.pt`, accept an environment
override, and append the corresponding CLI argument; they fall back to a
repository-local file only when it exists. Real training without a CLIP or resume
checkpoint is rejected unless `--allow-random-init` explicitly marks an ablation.

HMDB51/UCF101 class names may be inferred from real list path parents and
validated by numeric label. K400 and SSV2 use the repository-owned, ID-indexed
semantic mappings `configs/kinetics_400_labels.csv` and
`configs/something_v2_labels.csv`, respectively. Command-line class-name files
still take precedence.
The `freeze_clip` mode leaves original text, visual projections, and logit scale
frozen while training MGSE projection, GSPL, LMPL, temporal aggregation, and
other new layers.

## Commands

Train:

```bash
bash scripts/train_emclip_ssv2.sh
bash scripts/train_emclip_hmdb51.sh
bash scripts/train_emclip_ucf101.sh
bash scripts/train_emclip_k400.sh
```

4x3 evaluation:

```bash
RESUME=/path/to/model_best.pth TEMPORAL_VIEWS=4 SPATIAL_CROPS=3 bash scripts/eval_emclip_k400.sh
```

Smoke:

```bash
bash scripts/smoke_emclip.sh
```

## 2026-10-03 修复

默认 `--emclip-implementation auto`：新训练选择 `paper`；resume/init 从 checkpoint
识别结构。修复前的 checkpoint 没有配置元数据，以 `melsc.gs_r_proj.*` 识别为
`legacy`，保留独立 R FC/LN、ACG 仿射 LN、可学习分类尺度和历史 BGR/重复候选
输入处理。Legacy 的 module 注册顺序也保持一致，避免旧 optimizer 状态错配。
SSV2 标签不安全的训练翻转在所有实现中禁用；旧验证本来就不翻转。

新结构与旧结构的参数不能逐项等价迁移。显式要求 paper 而传入 legacy 权重会
提前报错，不会丢弃一半 R FC 参数或平均参数后冒称严格恢复。复现修复后的模型
应从原始 CLIP 重新开始；旧 SSV2 权重仍可用 auto/legacy 补做 4x3 评估。
新 checkpoint 保存 `model_config`、`run_config` 和 optimizer 参数名；自动恢复
输入/结构设置，显式 CLI 覆盖可用于消融。恢复训练会检查类别文本顺序和 optimizer
参数顺序。模型权重在 profile/preflight 前加载，避免检查的是另一个模型。

### 实际修改文件

- `models/emclip_melsc.py`：逐层共用 Eq.16 R 特征；可选正弦时间位置编码。
- `models/emclip_mgse.py`：Eq.5 无仿射标准化；训练/验证文本策略分离。
- `models/emclip.py`：固定分类温度，冻结兼容 logit_scale；paper/legacy 配置。
- `dataset_coviar.py`：SSV2 禁止 flip；R 转 RGB；重复候选 mask。
- `main_emclip.py`：参数、checkpoint 结构识别、输入恢复、配置 JSON。
- `engine_emclip.py`：checkpoint 配置/optimizer 顺序检查；unique GOP、有效候选、分类尺度日志。
- `amp_compat.py`：CUDA 与 CPU autocast 中的敏感计算均明确关闭混合精度。
- `scripts/_gpu_env.sh`：尊重外部 CUDA_VISIBLE_DEVICES/NPROC_PER_NODE，默认 4 进程。
- `scripts/train_emclip_{ssv2,hmdb51,ucf101,k400}.sh`：每卡 batch 4、原始 CLIP 起点；可选 transfer；diamond T=K，full T=2K。
- `scripts/eval_emclip_{ssv2,hmdb51,ucf101,k400}.sh`：明确目标 checkpoint、默认 4x3。
- `scripts/smoke_emclip.sh`：真实 SSV2 smoke 使用仓库类别 CSV。
- `tests/test_emclip_paper_protocol.py`：新增公式、采样、标签、温度、自动恢复等回归测试。
- `tests/test_emclip_shapes.py`、`tests/test_emclip_scripts.py`：paper/legacy 梯度和真实 shell 命令验证。
- `scripts/audit_emclip_paper.py`：可对照 paper/legacy，并运行双进程 CPU DDP。
- `README.md`、本文件、`EMCLIP_PAPER_AUDIT.md`：同步运行协议；历史审计保留为修复前记录。

### 严格设置与补充假设

论文明确的 B/16、12 层/12 heads/768 width、256 输入、GOP=12、T16/K8
或 T32/K16、30 epochs、LR8e-6 cosine、tau0.01、零 dropout/stochastic depth
均保持。L_ME 也固定 tau0.01，L_MG 与 L_ME 默认等权相加。

GSPL 与 LMPL 共享的是输入 R FC/LN 特征，不是两个 attention 的内部 W_Q/K/V。
I/R/MV 编码器仍然独立。每层 SAG 生成两个新 prompt，层后丢弃 prompt 输出。

论文未完整说明 optimizer/batch/warmup/冻结、未知类别推理、short GOP 补齐、
预处理细节或时间聚合的完整 block。AdamW/batch4/WD0.2/full、class_bank 推理、
重复 mask、RGB R、signed R/255、pre-norm temporal block 仍是明确记录的工程
选择。训练 ground_truth 只用于已知训练标签的类别条件选帧；正式验证不能喂真实
类别。推理假设可能影响精度，不能声称整个协议等同于作者私有实现。

`--temporal-position-encoding none` 是默认；跨 GOP 排列不敏感性仍是论文细节
缺口。`sinusoidal` 是额外消融，在进入 MELSC 层前给 I/R token 加 [1,K,1,D]
位置值，使先后顺序可区分；它不引入可训练参数，不代表论文规定使用这种编码。

### 已执行验证

测试和结构/CPU DDP 结果记录在 `output_dir/emclip_audit_20261003/fix_verification.json`。
真实数据/CoViAR 扩展、CUDA/NCCL、GPU fp16 AMP 及正式训练精度因本机条件未验证；
CPU bfloat16 的混合精度测试不能替代这些检查。没有进行完整训练或下载模型。

### 运行顺序

先执行 synthetic smoke 和服务器真实 preflight，再从原始 CLIP 启动新训练。
训练中关注 `selected_unique_gops`、`valid_candidates`、`classification_scale=100`
以及 train/val 曲线。正式 4x3 评估应指定目标数据集 `model_best.pth`。参考 README
的四数据集命令、单卡 smoke、4 卡 SSV2 训练、旧模型 4x3 命令。

由于论文未公开源码，并且没有完整披露优化器、batch size、参数冻结策略、MGSE
测试阶段类别文本来源等细节，本实现属于基于论文公式和描述的工程复现，不能
保证与作者私有实现逐行一致，也不承诺修复后达到论文精度。
