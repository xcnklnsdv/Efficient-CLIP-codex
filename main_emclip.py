import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from configs import DATASETS
from dataset_coviar import CoviarDataSet
from engine_emclip import (
    _autocast,
    dist_ready,
    evaluate,
    is_main_process,
    load_checkpoint,
    load_model_checkpoint,
    _load_checkpoint_file,
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
    parser.add_argument("--train-root", default=None)
    parser.add_argument("--val-root", default=None)
    parser.add_argument("--train-list", default=None)
    parser.add_argument("--val-list", default=None)
    parser.add_argument("--compressed-video-root", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=0.2)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=0,
        help=(
            "Per-forward batch; 0 uses the full DataLoader batch. A smaller value is only "
            "valid when L_MG is disabled because contrastive negatives cannot be split exactly."
        ),
    )
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=8)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp-init-scale",
        type=float,
        default=1024.0,
        help="Initial GradScaler scale; 1024 is conservative for MGSE tau=0.01.",
    )
    parser.add_argument("--amp-growth-interval", type=int, default=2000)
    parser.add_argument("--max-consecutive-amp-overflows", type=int, default=8)
    parser.add_argument(
        "--init-checkpoint",
        "--finetune",
        dest="init_checkpoint",
        default=None,
        help=(
            "Initialize model weights from an EM-CLIP checkpoint without restoring "
            "optimizer, scheduler, scaler, epoch, or best accuracy. Use this for "
            "K400-to-SSV2/HMDB51/UCF101 transfer."
        ),
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", "--save-dir", dest="output_dir", default="output_dir/emclip")
    parser.add_argument("--eval", "--eval-only", dest="eval", action="store_true")
    parser.add_argument("--test-num-temporal-views", type=int, default=1)
    parser.add_argument("--test-num-spatial-crops", type=int, default=1)
    parser.add_argument("--verify-compressed-inputs", action="store_true")
    parser.add_argument(
        "--preflight-compressed-inputs",
        action="store_true",
        help="Synchronously decode dataset item 0 before DDP training; useful for isolating CoViAR crashes.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Decode and forward one real compressed sample, then exit without constructing loaders.",
    )
    parser.add_argument(
        "--coviar-data-loader-dir",
        default=os.environ.get("COVIAR_DATA_LOADER_DIR"),
        help="Directory containing coviar Python extension, e.g. /home/fuh/m2clip/Coviar/data_loader.",
    )
    parser.add_argument(
        "--mv-clamp",
        type=float,
        default=20.0,
        help="Clamp motion-vector x/y values to this magnitude before scaling to [-1,1].",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=255.0,
        help="Divide signed CoViAR residual values by this scale before clamping.",
    )
    parser.add_argument(
        "--residual-clamp",
        type=float,
        default=1.0,
        help="Clamp scaled residual values to this symmetric magnitude.",
    )
    parser.add_argument(
        "--mv-accumulate",
        action="store_true",
        help="Diagnostic ablation: request I-frame-relative accumulated MV instead of last-P direct MV.",
    )
    parser.add_argument(
        "--no-residual-accumulate",
        dest="residual_accumulate",
        action="store_false",
        default=True,
        help="Diagnostic ablation: disable the default I-frame-relative accumulated residual.",
    )
    parser.add_argument("--scale-lr-by-global-batch", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--horizontal-flip", default="auto", choices=["auto", "on", "off"],
                        help="auto disables direction-changing flips for SSV2; on requires label-safe datasets.")
    parser.add_argument("--residual-channel-order", default=None, choices=["rgb", "bgr"],
                        help="Default RGB for paper initialization, BGR for historical checkpoints.")
    parser.add_argument("--duplicate-gop-policy", default=None, choices=["mask", "keep"],
                        help="mask excludes repeated candidate GOPs; keep reproduces historical sampling.")
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument(
        "--profile-compute",
        action="store_true",
        help=(
            "Profile one batch-size-1 cached-text inference and log supported-op "
            "FLOPs/video, latency, throughput, and peak allocated memory."
        ),
    )
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument(
        "--validate-class-names-only",
        action="store_true",
        help="Resolve and validate semantic class texts under DDP, then exit before model or data loading.",
    )
    parser.add_argument("--find-unused-parameters", action="store_true", default=False)
    parser.add_argument("--debug-unused-parameters", action="store_true", default=False)
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
    is_diamond = "diamond" in model
    if is_diamond:
        args.emclip_variant = "diamond"
    if model.endswith("_k16"):
        # Full EM-CLIP first scores 2K candidate GOPs with MGSE. Diamond has
        # no candidate-selection stage, so its TSN-style sampler produces the
        # final K GOPs directly.
        args.candidate_frames = 16 if is_diamond else 32
        args.selected_frames = 16
    elif model.endswith("_k8"):
        args.candidate_frames = 8 if is_diamond else 16
        args.selected_frames = 8


def resolve_implementation(args, explicit_options=None):
    """Choose architecture before building the model/optimizer, without migration.

    Old checkpoints lack metadata: their independent gs_r_proj and affine
    feature_ln keys identify legacy. New checkpoints save model and input
    settings. Explicit CLI overrides win except incompatible architectures.
    """
    if explicit_options is None:
        explicit_options = {arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")}
    path = args.resume or args.init_checkpoint
    checkpoint = _load_checkpoint_file(path, "cpu") if path else None
    requested = args.emclip_implementation
    if checkpoint is not None:
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
            raise TypeError("EM-CLIP checkpoint must contain a 'model' state_dict: %s" % path)
        inferred = "legacy" if any(key.startswith("melsc.gs_r_proj.") for key in checkpoint["model"]) else "paper"
        stored = checkpoint.get("model_config") or {}
        implementation = stored.get("implementation", inferred)
        if implementation != inferred:
            raise ValueError("Checkpoint architecture metadata disagrees with state_dict: %s" % path)
        if requested != "auto" and requested != implementation:
            raise ValueError(
                "Checkpoint uses %s, requested %s. Use --emclip-implementation auto/legacy for historical evaluation; "
                "start the paper model from --clip-checkpoint, without --resume/--init-checkpoint. "
                "No silent parameter migration is performed." % (implementation, requested)
            )
        args.emclip_implementation = implementation
        fields = {
            "candidate_frames": "candidate_frames", "selected_frames": "selected_frames",
            "input_size": "input_size", "gop_size": "gop_size", "emclip_variant": "emclip_variant",
            "mgse_text_mode": "mgse_text_mode", "mgse_train_text_mode": "mgse_train_text_mode",
            "mgse_class_aggregation": "mgse_class_aggregation", "motion_pooling": "motion_pooling",
            "classification_temperature": "classification_temperature",
            "mgse_temperature": "mgse_temperature", "temporal_aggregator_layers": "temporal_aggregator_layers",
            "temporal_position_encoding": "temporal_position_encoding",
        }
        if args.resume and not args.eval:
            fields.update({"emclip_train_mode": "emclip_train_mode", "lambda_mg": "lambda_mg", "lambda_me": "lambda_me"})
        for field, source_field in fields.items():
            flag = "--" + field.replace("_", "-")
            if flag not in explicit_options and source_field in stored:
                setattr(args, field, stored[source_field])
        inputs = checkpoint.get("run_config") or {}
        for field in ("residual_channel_order", "duplicate_gop_policy", "horizontal_flip", "mv_clamp",
                      "residual_scale", "residual_clamp", "mv_accumulate", "residual_accumulate"):
            flag = "--no-residual-accumulate" if field == "residual_accumulate" else "--" + field.replace("_", "-")
            if flag not in explicit_options and field in inputs:
                setattr(args, field, inputs[field])
    else:
        args.emclip_implementation = "paper" if requested == "auto" else requested
    if args.residual_channel_order is None:
        args.residual_channel_order = "rgb" if args.emclip_implementation == "paper" else "bgr"
    if args.duplicate_gop_policy is None:
        args.duplicate_gop_policy = "mask" if args.emclip_implementation == "paper" else "keep"


def resolve_dataset_config(args):
    cfg = dict(DATASETS[args.dataset])
    overrides = {
        "TRAIN_ROOT": args.train_root,
        "VAL_ROOT": args.val_root,
        "TRAIN_LIST": args.train_list,
        "VAL_LIST": args.val_list,
        "COMPRESSED_VIDEO_ROOT": args.compressed_video_root,
    }
    for key, value in overrides.items():
        if value:
            cfg[key] = value
    if not args.compressed_video_root:
        if args.eval and args.val_root:
            cfg["COMPRESSED_VIDEO_ROOT"] = args.val_root
        elif args.train_root:
            cfg["COMPRESSED_VIDEO_ROOT"] = args.train_root
        elif args.val_root and not cfg.get("TRAIN_ROOT"):
            cfg["COMPRESSED_VIDEO_ROOT"] = args.val_root
    return cfg


def _validate_class_names(names, num_classes, source, normalize_underscores=False):
    if len(names) != num_classes:
        raise ValueError(
            "class text count %d from %s does not match NUM_CLASSES=%d."
            % (len(names), source, num_classes)
        )
    normalized = []
    for name in names:
        name = " ".join(str(name).strip().split())
        if normalize_underscores:
            name = " ".join(name.replace("_", " ").split())
        normalized.append(name)

    empty = [index for index, name in enumerate(normalized) if not name]
    if empty:
        raise ValueError("empty class text entries in %s at indices %s" % (source, empty[:20]))
    numeric = []
    for index, name in enumerate(normalized):
        words = name.casefold().split()
        if name.isdecimal() or (
            len(words) == 2
            and words[0] in {"class", "label", "category"}
            and words[1].isdecimal()
        ):
            numeric.append(index)
    if numeric:
        raise ValueError(
            "purely numeric class texts are forbidden in %s at indices %s"
            % (source, numeric[:20])
        )
    duplicate_indices = {}
    first_index = {}
    for index, name in enumerate(normalized):
        key = name.casefold()
        if key in first_index:
            duplicate_indices.setdefault(key, [first_index[key]]).append(index)
        else:
            first_index[key] = index
    if duplicate_indices:
        examples = [
            "%r at indices %s" % (normalized[indices[0]], indices)
            for indices in list(duplicate_indices.values())[:10]
        ]
        raise ValueError("duplicate class texts in %s: %s" % (source, "; ".join(examples)))
    return normalized


def _read_plain_class_names(path):
    indexed = {}
    sequential = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            fields = line.replace("\t", " ").split(maxsplit=1)
            if len(fields) == 2 and fields[0].isdigit():
                class_index = int(fields[0])
                if class_index in indexed:
                    raise ValueError("duplicate class index %d in %s" % (class_index, path))
                indexed[class_index] = fields[1].strip()
            else:
                sequential.append(line)
    if indexed and sequential:
        raise ValueError("class-name file mixes indexed and sequential rows: %s" % path)
    if indexed:
        expected = list(range(max(indexed) + 1))
        if sorted(indexed) != expected:
            raise ValueError("class-name indices are not contiguous from zero in %s" % path)
        return [indexed[index] for index in expected]
    return sequential


def _read_class_csv(path):
    indexed = {}
    sequential = []
    index_headers = {"id", "index", "label", "label_id", "class_id", "class_index"}
    name_headers = {"name", "class", "class_name", "label_name", "category", "category_name"}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            row = [field.strip() for field in row]
            if not any(row):
                continue
            lowered = {field.casefold() for field in row if field}
            if lowered & index_headers and lowered & name_headers:
                continue
            if len(row) == 1:
                sequential.append(row[0])
                continue

            first_is_index = row[0].isdigit()
            last_is_index = row[-1].isdigit()
            if first_is_index:
                class_index = int(row[0])
                class_name = ",".join(row[1:]).strip()
            elif last_is_index:
                class_index = int(row[-1])
                class_name = ",".join(row[:-1]).strip()
            else:
                if row_number == 1 and lowered & (index_headers | name_headers):
                    continue
                sequential.append(row[-1])
                continue
            if class_index in indexed:
                raise ValueError("duplicate class index %d in %s" % (class_index, path))
            indexed[class_index] = class_name
    if indexed and sequential:
        raise ValueError("label CSV mixes indexed and sequential rows: %s" % path)
    if indexed:
        expected = list(range(max(indexed) + 1))
        if sorted(indexed) != expected:
            raise ValueError("label CSV indices are not contiguous from zero in %s" % path)
        return [indexed[index] for index in expected]
    return sequential


def _load_class_name_file(path, num_classes, source, dataset_name, csv_format=None):
    expanded = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if not expanded.is_absolute():
        repo_relative = Path(__file__).resolve().parent / expanded
        cwd_relative = Path.cwd() / expanded
        expanded = repo_relative if repo_relative.is_file() else cwd_relative
    if not expanded.is_file():
        raise ValueError("class-name file does not exist for %s: %s" % (source, expanded))
    use_csv = expanded.suffix.casefold() == ".csv" if csv_format is None else csv_format
    names = _read_class_csv(expanded) if use_csv else _read_plain_class_names(expanded)
    names = _validate_class_names(
        names,
        num_classes,
        expanded,
        normalize_underscores=dataset_name.casefold() == "k400",
    )
    return names, str(expanded.resolve())


def _log_class_names(names, source):
    if not is_main_process():
        return
    preview_count = min(5, len(names))
    print("[emclip] class names source=%s" % source, flush=True)
    print("[emclip] class names loaded=%d" % len(names), flush=True)
    print("[emclip] class names first5=%s" % names[:preview_count], flush=True)
    print("[emclip] class names last5=%s" % names[-preview_count:], flush=True)


def _k400_project_class_name_candidates():
    repo_root = Path(__file__).resolve().parent
    names = (
        "kinetics_400_labels.csv",
        "kinetics400_labels.csv",
        "k400_class_names.txt",
        "kinetics_classnames.txt",
        "category.csv",
        "k400_mlm_labels.txt",
        "k400_mlm_lables.txt",
    )
    roots = (
        repo_root / "configs",
        repo_root / "lists" / "k400",
        repo_root / "data" / "k400",
        repo_root / "datasets" / "k400",
        repo_root,
        Path("/home/fuh/CMPT/lists/k400"),
        Path("/home/fuh/m2clip"),
    )
    return [root / name for root in roots for name in names]


def _class_name_candidates_near_lists(cfg):
    names = (
        "kinetics_400_labels.csv",
        "kinetics400_labels.csv",
        "k400_class_names.txt",
        "kinetics_classnames.txt",
        "category.csv",
        "k400_mlm_labels.txt",
        "k400_mlm_lables.txt",
    )
    roots = []
    for key in ("TRAIN_LIST", "VAL_LIST"):
        list_path = cfg.get(key)
        if list_path:
            parent = Path(os.path.expandvars(os.path.expanduser(str(list_path)))).parent
            roots.extend((parent, parent / "labels", parent.parent / "lists" / "k400"))
    for key in ("TRAIN_ROOT", "VAL_ROOT", "COMPRESSED_VIDEO_ROOT"):
        data_root = cfg.get(key)
        if data_root:
            root = Path(os.path.expandvars(os.path.expanduser(str(data_root))))
            roots.extend((root, root / "datalist", root / "lists" / "k400"))
    candidates = []
    seen = set()
    for root in roots:
        for name in names:
            candidate = root / name
            key = os.path.normcase(os.path.abspath(str(candidate)))
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    return candidates


def _infer_class_names_from_lists(cfg, num_classes, dataset_name):
    inferred = {}
    inspected = []
    for key in ("TRAIN_LIST", "VAL_LIST"):
        list_path = cfg.get(key)
        if not list_path or not os.path.isfile(list_path):
            continue
        inspected.append(list_path)
        with open(list_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                parts = raw_line.strip().split()
                try:
                    label = int(parts[-1])
                except (ValueError, IndexError):
                    continue
                if label < 0 or label >= num_classes:
                    raise ValueError(
                        "dataset %s label %d is outside [0, %d] in line: %s"
                        % (dataset_name, label, num_classes - 1, raw_line.rstrip())
                    )
                video_token = next((token for token in parts if Path(token).suffix), parts[0])
                parent = Path(video_token).parent.name
                if not parent or parent == ".":
                    continue
                name = parent.replace("_", " ").strip()
                previous = inferred.get(label)
                if previous is not None and previous != name:
                    raise ValueError(
                        "conflicting inferred class text for label %d: %r versus %r"
                        % (label, previous, name)
                    )
                inferred[label] = name
    if len(inferred) == num_classes:
        return [inferred[index] for index in range(num_classes)], inspected
    return None, inspected


def load_class_names(args, num_classes, cfg=None):
    cfg = cfg or resolve_dataset_config(args)
    dataset_name = args.dataset

    def finish(names, source):
        _log_class_names(names, source)
        return names

    if getattr(args, "class_names", None):
        names, source = _load_class_name_file(
            args.class_names,
            num_classes,
            "command line --class-names",
            dataset_name,
        )
        return finish(names, source)
    if getattr(args, "label_csv", None):
        names, source = _load_class_name_file(
            args.label_csv,
            num_classes,
            "command line --label-csv",
            dataset_name,
            csv_format=True,
        )
        return finish(names, source)

    config_fields = (
        ("CLASS_NAMES", None),
        ("CLASS_NAMES_FILE", None),
        ("CLASS_NAME_FILE", None),
        ("CLASS_NAMES_PATH", None),
        ("LABEL_CSV", True),
        ("LABELS_CSV", True),
        ("LABEL_FILE", None),
        ("LABELS_FILE", None),
    )
    missing_configured = []
    for key, csv_format in config_fields:
        configured = cfg.get(key)
        if not configured:
            continue
        if key == "CLASS_NAMES" and isinstance(configured, (list, tuple)):
            names = _validate_class_names(
                configured,
                num_classes,
                "dataset config CLASS_NAMES",
                normalize_underscores=dataset_name.casefold() == "k400",
            )
            return finish(names, "dataset config CLASS_NAMES")
        try:
            names, source = _load_class_name_file(
                configured,
                num_classes,
                "dataset config %s" % key,
                dataset_name,
                csv_format=csv_format,
            )
        except ValueError as error:
            if "does not exist" in str(error):
                missing_configured.append(str(configured))
                continue
            raise
        return finish(names, source)

    rejected = []
    searched = []
    if dataset_name.casefold() == "k400":
        candidates = _k400_project_class_name_candidates()
        candidates.extend(_class_name_candidates_near_lists(cfg))
        seen = set()
        for candidate in candidates:
            candidate_key = os.path.normcase(os.path.abspath(str(candidate)))
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            searched.append(str(candidate))
            if not candidate.is_file():
                continue
            try:
                names, source = _load_class_name_file(
                    candidate,
                    num_classes,
                    "automatic K400 class-name discovery",
                    dataset_name,
                )
            except ValueError as error:
                rejected.append("%s (%s)" % (candidate, error))
                continue
            return finish(names, source)

    inferred, inspected = _infer_class_names_from_lists(cfg, num_classes, args.dataset)
    if inferred is not None:
        names = _validate_class_names(
            inferred,
            num_classes,
            "dataset lists %s" % inspected,
            normalize_underscores=dataset_name.casefold() == "k400",
        )
        return finish(names, "inferred from dataset lists %s" % inspected)
    details = []
    if missing_configured:
        details.append("missing configured files: %s" % missing_configured)
    if rejected:
        details.append("rejected discovered files: %s" % rejected)
    if dataset_name.casefold() == "k400":
        details.append("searched K400 mapping paths: %s" % searched)
    raise ValueError(
        "cannot obtain %d semantic class texts for dataset %s from lists %s. "
        "Pass --class-names or --label-csv; numeric placeholders are forbidden for CLIP training.%s"
        % (
            num_classes,
            args.dataset,
            inspected or "<no existing list files>",
            (" " + " ".join(details)) if details else "",
        )
    )


def build_datasets(args, class_names=None, cfg=None):
    cfg = cfg or resolve_dataset_config(args)
    class_to_idx = None
    if class_names is not None:
        class_to_idx = {}
        for index, name in enumerate(class_names):
            class_to_idx[name] = index
            class_to_idx[name.replace(" ", "_")] = index
    common = dict(
        dataset_name=args.dataset,
        num_classes=cfg["NUM_CLASSES"],
        candidate_frames=args.candidate_frames,
        input_size=args.input_size,
        gop_size=args.gop_size or cfg.get("GOP_SIZE", 12),
        compressed_video_root=cfg.get("COMPRESSED_VIDEO_ROOT", None),
        coviar_data_loader_dir=args.coviar_data_loader_dir,
        verify_paths=args.verify_compressed_inputs,
        class_to_idx=class_to_idx,
        mv_clamp=args.mv_clamp,
        residual_scale=args.residual_scale,
        residual_clamp=args.residual_clamp,
        mv_accumulate=args.mv_accumulate,
        residual_accumulate=args.residual_accumulate,
        horizontal_flip=(None if args.horizontal_flip == "auto" else args.horizontal_flip == "on"),
        residual_channel_order=args.residual_channel_order,
        deduplicate_candidates=args.duplicate_gop_policy == "mask",
    )
    train_dataset = None
    if not args.eval:
        train_dataset = CoviarDataSet(
            list_path=cfg["TRAIN_LIST"],
            data_root=cfg["TRAIN_ROOT"],
            random_sample=True,
            **common,
        )
    val_dataset = CoviarDataSet(
        list_path=cfg["VAL_LIST"],
        data_root=cfg["VAL_ROOT"],
        random_sample=False,
        num_temporal_views=args.test_num_temporal_views,
        num_spatial_crops=args.test_num_spatial_crops,
        **common,
    )
    return train_dataset, val_dataset


class DistributedEvalSampler(Sampler):
    """Shard evaluation without the duplicate padding used by DistributedSampler."""

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        if num_replicas <= 0 or rank < 0 or rank >= num_replicas:
            raise ValueError("invalid distributed sampler rank/world: %d/%d" % (rank, num_replicas))
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = max(0, len(self.dataset) - self.rank)
        return (remaining + self.num_replicas - 1) // self.num_replicas


def build_scheduler(optimizer, steps_per_epoch, args):
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(0, args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def validate_resume_scheduler(scheduler, target_total_steps):
    if scheduler is None:
        return
    if target_total_steps <= 0:
        raise ValueError("target_total_steps must be positive")
    if scheduler.last_epoch >= target_total_steps:
        raise RuntimeError(
            "--resume restored scheduler last_epoch=%d, but this run has only "
            "%d total steps. This usually means a checkpoint from another "
            "dataset was passed to --resume. Use --init-checkpoint for K400 "
            "transfer so optimizer/scheduler/epoch are reset."
            % (scheduler.last_epoch, target_total_steps)
        )


def make_grad_scaler(device, enabled, init_scale=1024.0, growth_interval=2000):
    enabled = enabled and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(
                "cuda",
                enabled=enabled,
                init_scale=init_scale,
                growth_interval=growth_interval,
            )
        except TypeError:
            return torch.amp.GradScaler(
                enabled=enabled,
                init_scale=init_scale,
                growth_interval=growth_interval,
            )
    return torch.cuda.amp.GradScaler(
        enabled=enabled,
        init_scale=init_scale,
        growth_interval=growth_interval,
    )


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
        text_width=32,
        text_heads=4,
        text_layers=1,
        temporal_aggregator_layers=1,
        emclip_variant=args.emclip_variant,
        mgse_text_mode=args.mgse_text_mode,
        mgse_train_text_mode=args.mgse_train_text_mode,
        implementation=args.emclip_implementation,
        classification_temperature=args.classification_temperature,
        temporal_position_encoding=args.temporal_position_encoding,
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


def _tensor_stats(name, tensor):
    value = tensor.detach().float()
    return (
        "%s shape=%s mean=%.6g std=%.6g norm=%.6g finite=%s"
        % (
            name,
            tuple(value.shape),
            float(value.mean().cpu()),
            float(value.std(unbiased=False).cpu()),
            float(value.norm().cpu()),
            bool(torch.isfinite(value).all().cpu()),
        )
    )


def _all_tensor_outputs_finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_all_tensor_outputs_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_tensor_outputs_finite(item) for item in value)
    return True


@torch.no_grad()
def profile_model_compute(model, device, amp=False):
    """Profile one video forward with cached class text at batch size one."""
    try:
        from torch.profiler import ProfilerActivity, profile
    except ImportError as exc:
        raise RuntimeError(
            "--profile-compute requires torch.profiler support in the installed PyTorch"
        ) from exc

    raw_model = model.module if hasattr(model, "module") else model
    config = raw_model.config
    temporal = int(config.candidate_frames)
    spatial = int(config.input_size)
    use_amp = bool(amp and device.type == "cuda")
    precision = "amp_fp16" if use_amp else "fp32"
    was_training = raw_model.training
    raw_model.eval()

    i_frames = torch.randn(1, temporal, 3, spatial, spatial, device=device)
    residuals = torch.randn_like(i_frames)
    motion_vectors = None
    if raw_model.use_mgse:
        motion_vectors = torch.randn(1, temporal, 2, spatial, spatial, device=device)
    valid_mask = torch.ones(1, temporal, dtype=torch.bool, device=device)

    def forward_once():
        with _autocast(device, use_amp):
            return raw_model(
                i_frames=i_frames,
                motion_vectors=motion_vectors,
                residuals=residuals,
                labels=None,
                valid_mask=valid_mask,
                training_mode=False,
            )

    # Warmup also builds the safe eval-only class-text cache. The reported
    # per-video compute therefore excludes text encoding that is amortized
    # across the full validation/test set.
    warmup_output = forward_once()
    if not _all_tensor_outputs_finite(warmup_output):
        raise FloatingPointError("compute profiling warmup produced non-finite output")
    del warmup_output
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    # Measure an ordinary warm forward separately. Timing the profiler context
    # itself would mostly measure profiler instrumentation overhead.
    start_time = time.perf_counter()
    output = forward_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time
    if not _all_tensor_outputs_finite(output):
        raise FloatingPointError("compute timing forward produced non-finite output")
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024.0 ** 2
        if device.type == "cuda"
        else 0.0
    )
    del output

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
        profiled_output = forward_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    forward_flops = int(sum(int(event.flops or 0) for event in prof.key_averages()))
    if forward_flops <= 0:
        raise RuntimeError(
            "torch.profiler reported zero FLOPs; this PyTorch build does not provide "
            "the required operator FLOP formulas"
        )
    if not _all_tensor_outputs_finite(profiled_output):
        raise FloatingPointError("compute profiling forward produced non-finite output")
    result = {
        "batch_size": 1,
        "candidate_frames": temporal,
        "selected_frames": int(config.selected_frames),
        "input_size": spatial,
        "precision": precision,
        "forward_flops": forward_flops,
        "forward_gflops_per_video": forward_flops / 1e9,
        "forward_latency_ms": elapsed_seconds * 1000.0,
        "forward_videos_per_second": 1.0 / max(elapsed_seconds, 1e-12),
        "peak_allocated_memory_mb": peak_memory_mb,
    }
    print(
        "[emclip][compute] cached_text_forward batch=1 precision=%s T=%d K=%d input=%dx%d "
        "flops=%d gflops_per_video=%.3f latency_ms=%.3f videos_per_second=%.3f "
        "peak_allocated_memory_mb=%.1f"
        % (
            precision,
            result["candidate_frames"],
            result["selected_frames"],
            spatial,
            spatial,
            forward_flops,
            result["forward_gflops_per_video"],
            result["forward_latency_ms"],
            result["forward_videos_per_second"],
            peak_memory_mb,
        ),
        flush=True,
    )
    print(
        "[emclip][compute] FLOPs include only operators supported by "
        "torch.profiler formulas; class-text encoding is cached and excluded",
        flush=True,
    )

    if was_training:
        raw_model.train()
    del i_frames, residuals, motion_vectors, valid_mask, profiled_output, prof
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


@torch.no_grad()
def run_pretrained_audit(model, args, device):
    """Audit pretrained coverage and run a dataset-free finite forward pass."""
    if model.pretrained_audit is None:
        raise RuntimeError("--pretrained-audit-only requires --clip-checkpoint")
    model.eval()
    if is_main_process():
        key_tensors = [
            ("I.conv1", model.melsc.i_encoder.conv1.weight),
            ("I.positional_embedding", model.melsc.i_encoder.positional_embedding),
            ("I.proj", model.melsc.i_encoder.proj),
            ("R.conv1", model.melsc.r_encoder.conv1.weight),
            ("R.positional_embedding", model.melsc.r_encoder.positional_embedding),
            ("R.proj", model.melsc.r_encoder.proj),
            ("Text.token_embedding", model.text_encoder.token_embedding.weight),
            ("Text.positional_embedding", model.text_encoder.positional_embedding),
            ("Text.text_projection", model.text_encoder.text_projection),
        ]
        if model.mgse is not None:
            key_tensors.extend([
                ("MV.conv1", model.mgse.motion_encoder.conv1.weight),
                ("MV.positional_embedding", model.mgse.motion_encoder.positional_embedding),
                ("MV.proj", model.mgse.motion_encoder.proj),
            ])
        for name, tensor in key_tensors:
            print("[emclip][pretrained] " + _tensor_stats(name, tensor), flush=True)

    # Dynamic position interpolation lets this use a small spatial grid while
    # still exercising I/R/MV/Text and every transformer layer.
    spatial_size = max(model.config.patch_size, min(64, model.config.input_size))
    temporal_size = model.config.selected_frames
    i_frames = torch.randn(1, temporal_size, 3, spatial_size, spatial_size, device=device)
    residuals = torch.randn_like(i_frames)
    motion_vectors = None
    if model.use_mgse:
        motion_vectors = torch.randn(
            1, temporal_size, 2, spatial_size, spatial_size, device=device
        )
    valid_mask = torch.ones(1, temporal_size, dtype=torch.bool, device=device)
    output = model(
        i_frames=i_frames,
        motion_vectors=motion_vectors,
        residuals=residuals,
        labels=None,
        valid_mask=valid_mask,
        training_mode=False,
    )
    local_finite = _all_tensor_outputs_finite(output)
    finite_tensor = torch.tensor(int(local_finite), dtype=torch.int32, device=device)
    if dist_ready():
        dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
    all_finite = bool(finite_tensor.item())
    if not all_finite:
        raise FloatingPointError("pretrained audit synthetic forward produced non-finite output")
    if is_main_process():
        print(
            "[emclip][pretrained] synthetic_input=I%s MV%s R%s"
            % (
                tuple(i_frames.shape),
                tuple(motion_vectors.shape) if motion_vectors is not None else None,
                tuple(residuals.shape),
            ),
            flush=True,
        )
        print("[emclip][pretrained] all_outputs_finite=True", flush=True)


def main_worker():
    args = parse_args()
    apply_model_variant_defaults(args)
    if args.init_checkpoint and args.resume:
        raise ValueError(
            "--init-checkpoint and --resume are mutually exclusive: use "
            "--init-checkpoint for cross-dataset transfer and --resume only for "
            "continuing the same training run."
        )
    resolve_implementation(args)
    if args.micro_batch_size < 0:
        raise ValueError("--micro-batch-size must be >= 0.")
    if args.amp_init_scale <= 0:
        raise ValueError("--amp-init-scale must be > 0.")
    if args.amp_growth_interval < 1:
        raise ValueError("--amp-growth-interval must be >= 1.")
    if args.max_consecutive_amp_overflows < 1:
        raise ValueError("--max-consecutive-amp-overflows must be >= 1.")
    if args.test_num_temporal_views < 1:
        raise ValueError("--test-num-temporal-views must be >= 1.")
    if args.test_num_spatial_crops not in (1, 3):
        raise ValueError("--test-num-spatial-crops must be 1 or 3.")
    if args.preflight_only:
        args.preflight_compressed_inputs = True
    if args.synthetic_smoke:
        synthetic_smoke(args)
        return
    if (
        not args.validate_class_names_only
        and not args.clip_checkpoint
        and not args.init_checkpoint
        and not args.resume
        and not args.allow_random_init
    ):
        raise RuntimeError(
            "real EM-CLIP training/evaluation requires --clip-checkpoint or --resume. "
            "Use --allow-random-init only for an explicit initialization ablation."
        )
    set_seed(args.seed)
    device = init_distributed()
    if is_main_process():
        world = dist.get_world_size() if dist_ready() else 1
        print(
            "[emclip] initialized world_size=%d device=%s batch_size=%d micro_batch_size=%d"
            % (world, device, args.batch_size, args.micro_batch_size),
            flush=True,
        )
    cfg = resolve_dataset_config(args)
    try:
        class_names = load_class_names(args, cfg["NUM_CLASSES"], cfg=cfg)
    except ValueError:
        if not args.pretrained_audit_only or args.validate_class_names_only:
            raise
        class_names = ["audit class %d" % index for index in range(cfg["NUM_CLASSES"])]
        if is_main_process():
            print(
                "[emclip][pretrained] class texts unavailable; using audit-only placeholders",
                flush=True,
            )
    if args.validate_class_names_only:
        if is_main_process():
            print("[emclip] class-name validation completed; exiting before model/data loading", flush=True)
        if dist_ready():
            dist.barrier()
        return
    model_config = build_emclip_config_from_args(args, class_names)
    model = EMCLIP(model_config).to(device)
    if is_main_process():
        print("[emclip] model moved to device", flush=True)
    total, trainable = model.parameter_counts()
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        print(args)
        print("Total params: %d (%.2f M)" % (total, total / 1e6))
        print("Trainable params: %d (%.2f M)" % (trainable, trainable / 1e6))
        print(
            "[emclip] implementation=%s MGSE train=%s eval=%s classification=%s "
            "residual_channels=%s duplicate_gops=%s temporal_position=%s"
            % (model.config.implementation, model.config.mgse_train_text_mode, model.config.mgse_text_mode,
               "fixed_tau=%.8g" % model.config.classification_temperature if model.config.implementation == "paper" else "learned_CLIP_scale",
               args.residual_channel_order, args.duplicate_gop_policy, model.config.temporal_position_encoding), flush=True,
        )
        with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, ensure_ascii=False, indent=2)
        if (
            args.mgse_text_mode == "ground_truth"
            and args.allow_mgse_label_leakage_for_diagnostic
        ):
            print(
                "WARNING: ground-truth MGSE selection is enabled for diagnostic use; "
                "reported validation accuracy contains label leakage.",
                flush=True,
            )

    if args.pretrained_audit_only:
        run_pretrained_audit(model, args, device)
        return

    if args.init_checkpoint:
        init_info = load_model_checkpoint(args.init_checkpoint, model, map_location="cpu")
        if is_main_process():
            target_action = (
                "evaluating the target dataset with transferred weights"
                if args.eval
                else "starting target training with fresh optimizer/scheduler/scaler at epoch 0"
            )
            print(
                "[emclip][init] loaded model weights from %s "
                "(source_epoch=%d source_best_acc1=%.6g); %s"
                % (
                    args.init_checkpoint,
                    init_info["source_epoch"],
                    init_info["source_best_acc1"],
                    target_action,
                ),
                flush=True,
            )
    elif args.resume:
        # Profiling/preflight must use the requested trained model too. Full
        # optimizer state is restored later after its parameters are built.
        load_checkpoint(args.resume, model, map_location="cpu")

    if args.profile_compute:
        if is_main_process():
            profile_model_compute(model, device, amp=args.amp)
        if dist_ready():
            dist.barrier()

    train_dataset, val_dataset = build_datasets(args, class_names=class_names, cfg=cfg)
    if is_main_process():
        print(
            "[emclip] datasets ready train=%s val=%d"
            % (len(train_dataset) if train_dataset is not None else "none", len(val_dataset)),
            flush=True,
        )
    if args.preflight_compressed_inputs and is_main_process():
        preflight_dataset = train_dataset if train_dataset is not None else val_dataset
        if len(preflight_dataset) == 0:
            raise RuntimeError("cannot preflight an empty compressed-video dataset")
        sample = preflight_dataset.preflight(0)
        print(
            "[emclip] compressed preflight OK path=%s label=%d I=%s MV=%s R=%s gops=%s candidates=%s"
            % (
                sample["metadata"]["video_path"],
                int(sample["label"]),
                tuple(sample["i_frames"].shape),
                tuple(sample["motion_vectors"].shape),
                tuple(sample["residuals"].shape),
                sample["metadata"]["gop_count"],
                sample["candidate_gop_indices"].tolist(),
            ),
            flush=True,
        )
        for name in ("i_frames", "motion_vectors", "residuals"):
            value = sample[name].float()
            print(
                "[emclip] %s dtype=%s min=%.6g max=%.6g mean=%.6g"
                % (
                    name,
                    sample[name].dtype,
                    float(value.min()),
                    float(value.max()),
                    float(value.mean()),
                ),
                flush=True,
            )
        model.eval()
        with torch.no_grad():
            preflight_output = model(
                i_frames=sample["i_frames"].unsqueeze(0).to(device),
                motion_vectors=(
                    sample["motion_vectors"].unsqueeze(0).to(device)
                    if model.use_mgse
                    else None
                ),
                residuals=sample["residuals"].unsqueeze(0).to(device),
                labels=None,
                valid_mask=sample["valid_mask"].unsqueeze(0).to(device),
                training_mode=False,
            )
        if not torch.isfinite(preflight_output["logits"]).all():
            raise FloatingPointError("real compressed preflight forward produced non-finite logits")
        print(
            "[emclip] real no_grad forward OK logits=%s selected=%s"
            % (
                tuple(preflight_output["logits"].shape),
                preflight_output["selected_indices"].detach().cpu().tolist(),
            ),
            flush=True,
        )
    if args.preflight_compressed_inputs and dist_ready():
        dist.barrier()
    if args.preflight_only:
        return
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
    val_sampler = DistributedEvalSampler(val_dataset) if dist_ready() else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, args.batch_size // 2),
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    if dist_ready():
        ddp_kwargs = {"find_unused_parameters": args.find_unused_parameters}
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[device.index], output_device=device.index)
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        if is_main_process():
            print("[emclip] DDP wrapper ready", flush=True)

    optimizer = None
    scheduler = None
    scaler = make_grad_scaler(
        device,
        args.amp,
        init_scale=args.amp_init_scale,
        growth_interval=args.amp_growth_interval,
    )
    if is_main_process() and args.amp:
        print(
            "[emclip] AMP GradScaler init_scale=%.1f growth_interval=%d"
            % (float(scaler.get_scale()), args.amp_growth_interval),
            flush=True,
        )
    start_epoch = 0
    best_acc1 = -math.inf
    if not args.eval:
        lr = args.lr
        if args.scale_lr_by_global_batch:
            world = dist.get_world_size() if dist_ready() else 1
            lr = lr * args.batch_size * world / 4.0
        optimizer_params = [p for p in model.parameters() if p.requires_grad]
        optimizer_param_count = sum(p.numel() for p in optimizer_params)
        if is_main_process():
            raw_model = model.module if hasattr(model, "module") else model
            all_param_count = sum(p.numel() for p in raw_model.parameters())
            trainable_param_count = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
            print("All params before optimizer: %d (%.2f M)" % (all_param_count, all_param_count / 1e6))
            print(
                "Requires-grad params before optimizer: %d (%.2f M)"
                % (trainable_param_count, trainable_param_count / 1e6)
            )
            print(
                "Optimizer params: %d (%.2f M)"
                % (optimizer_param_count, optimizer_param_count / 1e6)
            )
        optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=args.weight_decay,
        )
        scheduler = build_scheduler(optimizer, len(train_loader), args)
        if is_main_process():
            print("[emclip] optimizer and scheduler ready", flush=True)
    if args.resume:
        start_epoch, best_acc1 = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, map_location=device)
        if scheduler is not None:
            target_total_steps = max(1, args.epochs * len(train_loader))
            validate_resume_scheduler(scheduler, target_total_steps)
        if is_main_process():
            print(
                "[emclip][resume] restored model and training state from %s; "
                "start_epoch=%d best_acc1=%.6g scheduler_step=%s"
                % (
                    args.resume,
                    start_epoch,
                    best_acc1,
                    scheduler.last_epoch if scheduler is not None else "none",
                ),
                flush=True,
            )

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
            if val_stats["val_acc1"] > best_acc1:
                best_acc1 = val_stats["val_acc1"]
                save_checkpoint(os.path.join(args.output_dir, "model_best.pth"), model, optimizer, scheduler, scaler, epoch, best_acc1, run_config=vars(args))
            latest = os.path.join(args.output_dir, "latest.pth")
            save_checkpoint(latest, model, optimizer, scheduler, scaler, epoch, best_acc1, run_config=vars(args))


@record
def main():
    try:
        return main_worker()
    finally:
        # Preserve and propagate the original exception while ensuring torchrun
        # does not warn about a leaked process group.
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
