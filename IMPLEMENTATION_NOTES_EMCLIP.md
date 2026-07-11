# EM-CLIP Implementation Notes

This is a standalone implementation because the active workspace only contained `AGENTS.md`.

## Module Map

- MGSE and ACG: `models/emclip_mgse.py`
- MELSC, IFE, GSPL, LMPL, SAG: `models/emclip_melsc.py`
- End-to-end model and prompt text encoder: `models/emclip.py`
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

## MGSE Label Leakage

- `ground_truth`: uses true labels and is blocked in eval unless `--allow-mgse-label-leakage-for-diagnostic` is set.
- `class_bank`: default for validation/test; it never reads labels for selected indices.
- `predicted_class`: predicts a class from motion features first, then uses that class text.

## Losses and DDP

`L_MG` constructs multi-positive targets where samples with identical labels are positives. It uses `F.kl_div(..., reduction="batchmean")`. Feature all-gather preserves gradients when `torch.distributed.nn.functional.all_gather` is available and falls back safely otherwise. Label all-gather is no-grad.

## Hyperparameters

Paper-specified defaults: epochs 30, LR `8e-6`, cosine schedule, input 256, tau `0.01`, ViT-B/16, GOP size 12, `T=16`, `K=8`.

Engineering assumptions: AdamW, betas `(0.9,0.98)`, eps `1e-6`, weight decay `0.2`, warmup 0, batch size per GPU 4, `--micro-batch-size 1` by default, grad clip 1.0, seed 1024, AMP on scripts, workers 8. The effective batch is split into GPU micro-batches and gradients are accumulated. LR is not scaled by world size unless `--scale-lr-by-global-batch` is passed.

由于论文未公开源码，并且没有完整披露优化器、batch size、参数冻结策略、MGSE测试阶段类别文本来源等细节，本实现属于基于论文公式和描述的工程复现，不能保证与作者私有实现逐行一致。

## Known Differences

The text encoder uses a deterministic local tokenizer when the OpenAI CLIP tokenizer is not present. Local checkpoints can be loaded through `--clip-checkpoint`, but no network download is attempted.

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
