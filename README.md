# EM-CLIP Standalone Reproduction

This directory contains a standalone engineering reproduction of **Efficient Motion-Centric CLIP for Compressed Video Action Recognition**. It was created here because this workspace only contained `AGENTS.md`; it does not modify `C:\Users\Frank\Downloads\m2clip-origin-mp4`.

The implementation provides:

- MGSE and ACG with `class_bank`, `ground_truth`, and `predicted_class` text modes.
- MELSC with independent I-frame and residual ViT branches, GSPL, LMPL, SAG, and temporal aggregation.
- `L_MG` multi-positive bidirectional KL and `L_ME` video-text classification CE.
- CoViAR-backed compressed dataset loading for I/MV/Residual.
- DDP, AMP, resume, latest/best checkpoint, 1-view and multi-view evaluation support.

## Quick Checks

```bash
python -m pytest tests -q
python main_emclip.py --synthetic-smoke --model emclip_b16 --emclip-variant emclip
bash scripts/smoke_emclip.sh
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
NPROC_PER_NODE=4 BATCH_SIZE=16 MICRO_BATCH_SIZE=1 bash scripts/train_emclip_hmdb51.sh
RESUME=output_dir/emclip/run/latest.pth bash scripts/train_emclip_hmdb51.sh
GPU_IDS=0,1 bash scripts/train_emclip_hmdb51.sh
GPU_IDS=2 bash scripts/train_emclip_ucf101.sh
CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_emclip_k400.sh
NUM_WORKERS=8 PIN_MEMORY=1 bash scripts/train_emclip_hmdb51.sh
COVIAR_DATA_LOADER_DIR=/home/fuh/Efficient-CLIP-codex/pytorch-coviar/data_loader bash scripts/train_emclip_hmdb51.sh
CLIP_CHECKPOINT=/home/fuh/Efficient-CLIP-codex/clip_vit_b_16.pth bash scripts/train_emclip_hmdb51.sh
python main_emclip.py --dataset hmdb51_mpeg4 --preflight-compressed-inputs --num-workers 0 --no-pin-memory
```

`GPU_IDS`, `GPUS`, and `CUDA_VISIBLE_DEVICES` all work. If `NPROC_PER_NODE` is not set, the scripts derive it from the number of comma-separated GPU ids.
The scripts default to `NUM_WORKERS=0` and `PIN_MEMORY=0` because some CoViAR builds segfault inside PyTorch DataLoader worker subprocesses. Increase workers only after a single-process data smoke test is stable.
`BATCH_SIZE` is the effective per-GPU batch; `MICRO_BATCH_SIZE` bounds each GPU forward and defaults to 1, so larger batches use gradient accumulation instead of moving all frames to the GPU at once.
Training and evaluation scripts append `--clip-checkpoint "${CLIP_CHECKPOINT}"` when `CLIP_CHECKPOINT` is set. If unset, they automatically use `${REPO_ROOT}/clip_vit_b_16.pth` when that file exists.
The scripts prefer a local `pytorch-coviar/data_loader` directory when present, then fall back to `/home/fuh/m2clip/Coviar/data_loader`. Build the extension with `cd pytorch-coviar/data_loader && bash install.sh` if `coviar*.so` is missing.

EM-CLIP-diamond examples:

```bash
VARIANT=diamond T=8 K=8 bash scripts/train_emclip_hmdb51.sh
VARIANT=diamond T=16 K=16 bash scripts/train_emclip_ucf101.sh
```

## Evaluation

Fast 1x1 evaluation:

```bash
RESUME=/path/to/model_best.pth bash scripts/eval_emclip_hmdb51.sh
```

Paper-style 4 temporal views x 3 spatial crops:

```bash
RESUME=/path/to/model_best.pth TEMPORAL_VIEWS=4 SPATIAL_CROPS=3 bash scripts/eval_emclip_k400.sh
```

## Notes

The paper does not release code and omits some optimizer, batch-size, freezing, and MGSE inference details. Those choices are documented in [IMPLEMENTATION_NOTES_EMCLIP.md](IMPLEMENTATION_NOTES_EMCLIP.md). This code is a runnable reproduction, not a claim that it will match the paper's private implementation or reported accuracy.
