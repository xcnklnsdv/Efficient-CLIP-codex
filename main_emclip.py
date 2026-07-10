import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from configs import DATASETS
from datasets import CompressedVideoDataset
from engine_emclip import (
    dist_ready,
    evaluate,
    is_main_process,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)
from models import EMCLIP, EMCLIPConfig, build_emclip_config_from_args
from models.emclip import add_emclip_args


def parse_args():
    parser = argparse.ArgumentParser("EM-CLIP training and evaluation")
    add_emclip_args(parser)
    parser.add_argument("--dataset", default="ssv2_mpeg4", choices=DATASETS.keys())
    parser.add_argument("--label-csv", default=None)
    parser.add_argument("--class-names", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=0.2)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=8)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", "--save-dir", dest="output_dir", default="output_dir/emclip")
    parser.add_argument("--eval", "--eval-only", dest="eval", action="store_true")
    parser.add_argument("--test-num-temporal-views", type=int, default=1)
    parser.add_argument("--test-num-spatial-crops", type=int, default=1)
    parser.add_argument("--verify-compressed-inputs", action="store_true")
    parser.add_argument("--scale-lr-by-global-batch", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")


def apply_model_variant_defaults(args):
    model = args.model.lower()
    if "diamond" in model:
        args.emclip_variant = "diamond"
    if model.endswith("_k16"):
        args.candidate_frames = 32
        args.selected_frames = 16
    elif model.endswith("_k8"):
        args.candidate_frames = 16
        args.selected_frames = 8


def load_class_names(args, num_classes):
    if args.class_names:
        with open(args.class_names, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip()]
    elif args.label_csv:
        names = []
        with open(args.label_csv, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                if len(row) == 1:
                    names.append(row[0].strip())
                else:
                    names.append(row[-1].strip())
    else:
        names = ["class %d" % i for i in range(num_classes)]
    if len(names) != num_classes:
        raise ValueError("class text count %d does not match NUM_CLASSES=%d." % (len(names), num_classes))
    return names


def build_datasets(args):
    cfg = DATASETS[args.dataset]
    common = dict(
        dataset_name=args.dataset,
        num_classes=cfg["NUM_CLASSES"],
        candidate_frames=args.candidate_frames,
        input_size=args.input_size,
        gop_size=args.gop_size or cfg.get("GOP_SIZE", 12),
        compressed_video_root=cfg.get("COMPRESSED_VIDEO_ROOT", None),
        verify_paths=args.verify_compressed_inputs,
    )
    train_dataset = None
    if not args.eval:
        train_dataset = CompressedVideoDataset(
            list_path=cfg["TRAIN_LIST"],
            data_root=cfg["TRAIN_ROOT"],
            random_sample=True,
            **common,
        )
    val_dataset = CompressedVideoDataset(
        list_path=cfg["VAL_LIST"],
        data_root=cfg["VAL_ROOT"],
        random_sample=False,
        num_temporal_views=args.test_num_temporal_views,
        num_spatial_crops=args.test_num_spatial_crops,
        **common,
    )
    return train_dataset, val_dataset


def build_scheduler(optimizer, steps_per_epoch, args):
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(0, args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_grad_scaler(device, enabled):
    enabled = enabled and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def synthetic_smoke(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = EMCLIPConfig(
        num_classes=5,
        class_names=["class %d" % i for i in range(5)],
        candidate_frames=4,
        selected_frames=2,
        input_size=64,
        patch_size=16,
        width=32,
        layers=1,
        heads=4,
        embed_dim=16,
        text_layers=1,
        temporal_aggregator_layers=1,
        emclip_variant=args.emclip_variant,
        mgse_text_mode=args.mgse_text_mode,
    )
    model = EMCLIP(config).to(device)
    batch = {
        "i_frames": torch.randn(2, 4, 3, 64, 64, device=device),
        "motion_vectors": torch.randn(2, 4, 2, 64, 64, device=device),
        "residuals": torch.randn(2, 4, 3, 64, 64, device=device),
        "valid_mask": torch.ones(2, 4, dtype=torch.bool, device=device),
        "labels": torch.tensor([0, 3], device=device),
    }
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    out = model(
        i_frames=batch["i_frames"],
        motion_vectors=batch["motion_vectors"] if config.emclip_variant == "emclip" else None,
        residuals=batch["residuals"],
        labels=batch["labels"],
        valid_mask=batch["valid_mask"],
        training_mode=True,
    )
    out["loss"].backward()
    optimizer.step()
    print(json.dumps({
        "loss": float(out["loss"].detach().cpu()),
        "logits_shape": list(out["logits"].shape),
        "selected_indices": out["selected_indices"].detach().cpu().tolist(),
        "device": str(device),
    }, indent=2))


def main():
    args = parse_args()
    apply_model_variant_defaults(args)
    if args.synthetic_smoke:
        synthetic_smoke(args)
        return
    set_seed(args.seed)
    device = init_distributed()
    cfg = DATASETS[args.dataset]
    class_names = load_class_names(args, cfg["NUM_CLASSES"])
    model_config = build_emclip_config_from_args(args, class_names)
    model = EMCLIP(model_config).to(device)
    total, trainable = model.parameter_counts()
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        print(args)
        print("Total params: %d (%.2f M)" % (total, total / 1e6))
        print("Trainable params: %d (%.2f M)" % (trainable, trainable / 1e6))

    train_dataset, val_dataset = build_datasets(args)
    if train_dataset is not None:
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if dist_ready() else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=True,
        )
    else:
        train_sampler = None
        train_loader = None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist_ready() else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, args.batch_size // 2),
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    if dist_ready():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    optimizer = None
    scheduler = None
    scaler = make_grad_scaler(device, args.amp)
    start_epoch = 0
    best_acc1 = -math.inf
    if not args.eval:
        lr = args.lr
        if args.scale_lr_by_global_batch:
            world = dist.get_world_size() if dist_ready() else 1
            lr = lr * args.batch_size * world / 4.0
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=args.weight_decay,
        )
        scheduler = build_scheduler(optimizer, len(train_loader), args)
    if args.resume:
        start_epoch, best_acc1 = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, map_location=device)

    if args.eval:
        metrics = evaluate(model, val_loader, device, args)
        if is_main_process():
            print(metrics)
        return

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, args)
        val_stats = evaluate(model, val_loader, device, args)
        if is_main_process():
            print("epoch=%d train=%s val=%s" % (epoch, train_stats, val_stats))
            latest = os.path.join(args.output_dir, "latest.pth")
            save_checkpoint(latest, model, optimizer, scheduler, scaler, epoch, best_acc1)
            if val_stats["val_acc1"] > best_acc1:
                best_acc1 = val_stats["val_acc1"]
                save_checkpoint(os.path.join(args.output_dir, "model_best.pth"), model, optimizer, scheduler, scaler, epoch, best_acc1)
    if dist_ready():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
