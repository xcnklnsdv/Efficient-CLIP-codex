# EM-CLIP Implementation Notes

This is a standalone implementation because the active workspace only contained `AGENTS.md`.

## Module Map

- MGSE and ACG: `models/emclip_mgse.py`
- MELSC, IFE, GSPL, LMPL, SAG: `models/emclip_melsc.py`
- End-to-end model and prompt text encoder: `models/emclip.py`
- Original CLIP checkpoint parsing and coverage audit: `models/clip_checkpoint.py`
- `L_MG`: `losses/emclip_loss.py`
- CoViAR dataset and synchronized transforms: `datasets/compressed_video_dataset.py`
- DDP/AMP/checkpoints: `engine_emclip.py`, `main_emclip.py`
- Dataset configs: `configs/emclip_datasets.py`

## Formula Mapping

Formulas 1-5, motion encoding and temporal candidate features, map to `MotionGuidedSaliencyExtraction._encode_motion`.
Formulas 6-11, action correlation and temporal softmax, map to `_ground_truth_saliency`, `_class_bank_saliency`, `_predicted_class_saliency`.
Formulas 12-14, top-k and selected I/R gather, map to `select_topk_indices` and `gather_temporal`.
Formulas 15-18, bidirectional motion-text KL, map to `motion_text_kl_loss`.
Formulas 19-22, independent I/R embedding and per-layer prompts, map to `_initial_tokens` and `_layer_prompts`.
Formulas 23-27, GSPL/LMPL/SAG per-layer execution, map to `MotionEmbeddedLongTermSpatiotemporalCorrelation.forward`.
Formulas 28-30, temporal aggregation, video-text logits, and CE, map to MELSC final pooling and `EMCLIP.forward`.

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

`CompressedVideoDataset` uses `coviar.get_num_frames` and `coviar.load`. It calls:

- I-frame: `load(path, gop_idx, 0, 0, False)`
- MV: `load(path, gop_idx, last_p_pos, 1, False)`
- Residual: `load(path, gop_idx, last_p_pos, 2, True)`

CoViAR is loaded lazily after argparse so `--coviar-data-loader-dir` selects the
requested native extension. When available, `get_num_gops(path)` is used for
GOP bounds; the list-file frame count is not trusted for native decoder indices.
Run `--preflight-compressed-inputs` with one process to synchronously decode the
first sample before starting DDP.

MV uses the last valid P-frame without accumulation. Residual uses `accumulate=True` to obtain cumulative residual relative to the GOP I-frame. If a GOP has no valid P-frame, MV/R fall back to explicit zeros for that GOP only.

## Sampling

Videos are split into `T` non-overlapping GOP segments. Training randomly picks one GOP per segment. Evaluation picks deterministic per-view offsets. If GOP count is smaller than `T`, indices are repeated uniformly and `valid_mask` remains true because each repeated item maps to an actual GOP.

## Transforms and Normalization

Resize, crop, and flip parameters are shared across I/MV/R. Horizontal flip negates MV x. Resize scales MV x and y by width and height ratios. I uses CLIP mean/std. MV is clamped to `[-20,20] / 20`. Residual is divided by 255 and clamped to `[-1,1]`; CLIP mean/std is not applied to residual because residual is not RGB appearance.

## Position Embedding and Patch Init

All ViT branches interpolate absolute position embeddings with bicubic interpolation. MV patch weights can be initialized from RGB weights by channel mean, repeated to two channels, scaled by `3/2`; see `build_mv_patch_embed_weight`.

## Branch Independence

I, residual, and MV encoders are separate module instances. They do not share `nn.Module` objects.

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
  copied to the top-level CLIP logit-scale parameter.
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

`--resume` only reads this project's training format and strictly loads
`checkpoint["model"]` before restoring optimizer, scheduler, scaler, epoch, and
best accuracy.

## MGSE Label Leakage

- `ground_truth`: uses true labels and is blocked in eval unless `--allow-mgse-label-leakage-for-diagnostic` is set.
- `class_bank`: default for validation/test; it never reads labels for selected indices.
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

HMDB51/UCF101/K400 class names may be inferred from real list path parents and
validated by numeric label. SSV2 requires an explicit semantic class-name file.
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
