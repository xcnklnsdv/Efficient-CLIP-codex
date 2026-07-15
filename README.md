# EM-CLIP Standalone Reproduction

This directory contains a standalone engineering reproduction of **Efficient Motion-Centric CLIP for Compressed Video Action Recognition**. It was created here because this workspace only contained `AGENTS.md`; it does not modify `C:\Users\Frank\Downloads\m2clip-origin-mp4`.

The implementation provides:

- MGSE and ACG with `class_bank`, `ground_truth`, and `predicted_class` text modes.
- MELSC with independent I-frame and residual ViT branches, GSPL, LMPL, SAG, and temporal aggregation.
- `L_MG` multi-positive bidirectional KL and `L_ME` video-text classification CE.
- CoViAR-backed compressed dataset loading for I/MV/Residual.
- DDP, AMP, resume, latest/best checkpoint, 1-view and multi-view evaluation support.
- The official offline OpenAI CLIP BPE vocabulary and causal text attention.

## Quick Checks

```bash
python -m pytest tests -q
python main_emclip.py --synthetic-smoke --model emclip_b16 --emclip-variant emclip
python main_emclip.py --dataset hmdb51_mpeg4 --clip-checkpoint /home/fuh/CLIP-models/ViT-B-16.pt --pretrained-audit-only
bash scripts/smoke_emclip.sh
```

## Checkpoint Types

`--clip-checkpoint` initializes EM-CLIP from an original pretrained CLIP file.
OpenAI TorchScript/JIT archives and regular state-dict containers are supported.
The loader maps `visual.*` separately into the independent I-frame, residual,
and MV branches, and maps CLIP text tensors into the text encoder. The MV patch
embedding is converted from RGB `[D,3,16,16]` to `[D,2,16,16]` by channel mean,
two-channel repeat, and `3/2` scaling. Visual position embeddings are bicubically
interpolated from the checkpoint grid (normally 14x14) to the configured grid
(16x16 for 256 input). ViT-B/16 uses visual width/heads 768/12 and the separate
CLIP text width/heads 512/8. Every branch must reach at least 99% parameter
coverage.

The repository includes `models/bpe_simple_vocab_16e6.txt.gz` from the official
OpenAI CLIP repository. Prompts use the original 49,408-token BPE vocabulary,
SOT/EOT ids 49406/49407, causal text attention, and BPE-derived label-token
masks. No tokenizer or model file is downloaded at runtime. Use
`--clip-bpe-path` only for an equivalent local vocabulary asset.

`--resume` is intentionally separate. It only accepts an EM-CLIP training
checkpoint containing `model`, `optimizer`, `scheduler`, `scaler`, `epoch`, and
`best_acc1`; it is not a fallback for original CLIP files.

Real training refuses silent random initialization. Pass `--clip-checkpoint`, or
resume a complete EM-CLIP checkpoint with `--resume`. `--allow-random-init` is
reserved for an explicit initialization ablation.

Audit a real checkpoint without constructing dataset loaders or training:

```bash
CUDA_VISIBLE_DEVICES=0 python main_emclip.py \
  --dataset hmdb51_mpeg4 \
  --clip-checkpoint /home/fuh/CLIP-models/ViT-B-16.pt \
  --pretrained-audit-only

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  --master_port=29512 \
  main_emclip.py \
  --dataset hmdb51_mpeg4 \
  --clip-checkpoint /home/fuh/CLIP-models/ViT-B-16.pt \
  --pretrained-audit-only
```

## Training

Default full EM-CLIP, 4 GPUs:

```bash
bash scripts/train_emclip_ssv2.sh
bash scripts/train_emclip_hmdb51.sh
bash scripts/train_emclip_ucf101.sh
bash scripts/train_emclip_k400.sh
```

Override common settings:

```bash
NPROC_PER_NODE=1 BATCH_SIZE=2 MASTER_PORT=29601 bash scripts/train_emclip_hmdb51.sh
NPROC_PER_NODE=4 BATCH_SIZE=4 bash scripts/train_emclip_hmdb51.sh
RESUME=output_dir/emclip/run/latest.pth bash scripts/train_emclip_hmdb51.sh
GPU_IDS=0,1 bash scripts/train_emclip_hmdb51.sh
GPU_IDS=2 bash scripts/train_emclip_ucf101.sh
CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_emclip_k400.sh
NUM_WORKERS=8 PIN_MEMORY=1 bash scripts/train_emclip_hmdb51.sh
COVIAR_DATA_LOADER_DIR=/home/fuh/Efficient-CLIP-codex/pytorch-coviar/data_loader bash scripts/train_emclip_hmdb51.sh
CLIP_CHECKPOINT=/home/fuh/CLIP-models/ViT-B-16.pt bash scripts/train_emclip_hmdb51.sh
CLASS_NAMES=/path/to/ssv2_classes.txt bash scripts/train_emclip_ssv2.sh
python main_emclip.py --dataset hmdb51_mpeg4 --preflight-compressed-inputs --num-workers 0 --no-pin-memory
```

All train/eval scripts source `scripts/_gpu_env.sh`. Edit
`EMCLIP_DEFAULT_GPU_IDS="0,1,2,3"` in that file to centrally select the default
physical GPUs. `GPU_IDS`, `GPUS`, and `CUDA_VISIBLE_DEVICES` remain available as
one-off overrides, in that priority order. Unless explicitly overridden,
`NPROC_PER_NODE` is derived from the selected GPU count.
The engineering defaults are `NUM_WORKERS=8` and pinned memory enabled. Diagnose
worker-unsafe CoViAR builds with `NUM_WORKERS=0 PIN_MEMORY=0` and
`--preflight-only`.
`BATCH_SIZE` is the effective per-GPU batch. Training defaults
`MICRO_BATCH_SIZE=BATCH_SIZE`: splitting a batch while `L_MG` is enabled is
rejected because it changes the global contrastive positive/negative bank.
Evaluation may use `MICRO_BATCH_SIZE=1` because it does not compute `L_MG`.
AMP uses an initial GradScaler scale of 1024 because MGSE/L_MG uses the
paper-specified `tau=0.01`. Correlation, temperature, softmax/log-softmax and
loss calculations are forced to true float32. A recoverable fp16 backward
overflow skips that optimizer step and reduces the scale; eight consecutive
overflows still raise an explicit error. Override with `--amp-init-scale` and
`--max-consecutive-amp-overflows` when diagnosing another GPU architecture.
Training and evaluation scripts append `--clip-checkpoint "${CLIP_CHECKPOINT}"`; the default is `/home/fuh/CLIP-models/ViT-B-16.pt`, with an environment override and a repository-local fallback when available.
The scripts prefer a local `pytorch-coviar/data_loader` directory when present, then fall back to `/home/fuh/m2clip/Coviar/data_loader`. Build the extension with `cd pytorch-coviar/data_loader && bash install.sh` if `coviar*.so` is missing.

HMDB51, UCF101, and K400 semantic class text can be inferred from class-directory
names when every label is present in the real lists. SSV2 numeric IDs do not
contain class semantics, so pass `CLASS_NAMES`/`--class-names` or
`LABEL_CSV`/`--label-csv`. Numeric placeholders such as `class 0` are rejected.
Dataset locations can be overridden with `--train-root`, `--val-root`,
`--train-list`, `--val-list`, and `--compressed-video-root`.

EM-CLIP-diamond examples:

```bash
VARIANT=diamond T=8 K=8 bash scripts/train_emclip_hmdb51.sh
VARIANT=diamond T=16 K=16 bash scripts/train_emclip_ucf101.sh
```

The equivalent named presets are `emclip_diamond_b16_k8` (`T=K=8`) and
`emclip_diamond_b16_k16` (`T=K=16`). Full EM-CLIP keeps its MGSE candidate
bank: `emclip_b16_k8` uses `T=16,K=8`, and `emclip_b16_k16` uses
`T=32,K=16`.

## Evaluation

Fast 1x1 evaluation:

```bash
RESUME=/path/to/model_best.pth bash scripts/eval_emclip_hmdb51.sh
```

Paper-style 4 temporal views x 3 spatial crops:

```bash
RESUME=/path/to/model_best.pth TEMPORAL_VIEWS=4 SPATIAL_CROPS=3 bash scripts/eval_emclip_k400.sh
```

`bash scripts/smoke_emclip.sh` always runs synthetic full/diamond checks and then
attempts one real CoViAR decode plus no-grad forward for all four datasets.
Missing data, lists, SSV2 class text, checkpoint, or native CoViAR support are
reported as explicit `SKIP` reasons.

## Notes

The paper does not release code and omits some optimizer, batch-size, freezing, and MGSE inference details. Those choices are documented in [IMPLEMENTATION_NOTES_EMCLIP.md](IMPLEMENTATION_NOTES_EMCLIP.md). This code is a runnable reproduction, not a claim that it will match the paper's private implementation or reported accuracy.
