import math
import os
import random
import sys
import importlib
import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])
coviar_get_num_frames = None
coviar_load = None
coviar_import_error = None
coviar_preload_errors = []


@dataclass
class VideoListItem:
    relative_path: str
    label: int
    num_frames: Optional[int]
    raw_line: str


def _try_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _has_video_suffix(value):
    return Path(value).suffix.lower() in {".mp4", ".avi", ".webm", ".mkv", ".mov"}


def _strip_trailing_class_metadata(path_tokens, class_to_idx=None):
    """Drop class-name metadata from numeric-label list rows.

    K400 lists can use `relative/path.mp4 class_name label`. Paths may also
    contain spaces, so the safest signal is the token where the video suffix
    appears; any following non-label tokens are metadata, not path text.
    """
    if len(path_tokens) < 2:
        return " ".join(path_tokens)

    for index, token in enumerate(path_tokens):
        if _has_video_suffix(token):
            return " ".join(path_tokens[: index + 1])

    candidate_path = " ".join(path_tokens[:-1])
    trailing_name = path_tokens[-1]
    parent_name = Path(candidate_path).parent.name
    if (class_to_idx and trailing_name in class_to_idx) or parent_name == trailing_name:
        return candidate_path
    return " ".join(path_tokens)


def parse_video_list_line(line, num_classes, dataset_name, class_to_idx: Optional[Dict[str, int]] = None):
    raw_line = line.rstrip("\n")
    parts = raw_line.strip().split()
    if not parts:
        raise ValueError("empty list line in dataset %s" % dataset_name)

    label = _try_int(parts[-1])
    num_frames = None
    if label is not None:
        if len(parts) >= 3 and _try_int(parts[-2]) is not None:
            num_frames = int(parts[-2])
            relative_path = " ".join(parts[:-2])
        else:
            relative_path = _strip_trailing_class_metadata(parts[:-1], class_to_idx)
    else:
        if not class_to_idx:
            raise ValueError(
                "dataset %s list line has non-integer label and no class_to_idx mapping: %s"
                % (dataset_name, raw_line)
            )
        matched = None
        for start in range(1, len(parts)):
            label_name = " ".join(parts[start:])
            if label_name in class_to_idx:
                matched = (start, label_name)
                break
        if matched is None:
            raise ValueError("could not parse label name in dataset %s line: %s" % (dataset_name, raw_line))
        start, label_name = matched
        relative_path = " ".join(parts[:start])
        label = int(class_to_idx[label_name])

    if label < 0 or label >= num_classes:
        raise ValueError(
            "dataset %s label %d is outside [0, %d] for line: %s"
            % (dataset_name, label, num_classes - 1, raw_line)
        )
    if not relative_path:
        raise ValueError("dataset %s parsed an empty path from line: %s" % (dataset_name, raw_line))
    return VideoListItem(relative_path=relative_path, label=label, num_frames=num_frames, raw_line=raw_line)


def horizontal_flip_modalities(i_frames, motion_vectors, residuals):
    i_frames = i_frames.flip(dims=(-1,))
    residuals = residuals.flip(dims=(-1,))
    motion_vectors = motion_vectors.flip(dims=(-1,)).clone()
    motion_vectors[:, 0] = -motion_vectors[:, 0]
    return i_frames, motion_vectors, residuals


def resize_modalities(i_frames, motion_vectors, residuals, size):
    old_h, old_w = i_frames.shape[-2:]
    new_h, new_w = size
    i_frames = F.interpolate(i_frames, size=(new_h, new_w), mode="bicubic", align_corners=False)
    motion_vectors = F.interpolate(motion_vectors, size=(new_h, new_w), mode="bilinear", align_corners=False)
    residuals = F.interpolate(residuals, size=(new_h, new_w), mode="bilinear", align_corners=False)
    motion_vectors[:, 0] *= float(new_w) / float(old_w)
    motion_vectors[:, 1] *= float(new_h) / float(old_h)
    return i_frames, motion_vectors, residuals


def crop_modalities(i_frames, motion_vectors, residuals, top, left, height, width):
    bottom = top + height
    right = left + width
    return (
        i_frames[:, :, top:bottom, left:right],
        motion_vectors[:, :, top:bottom, left:right],
        residuals[:, :, top:bottom, left:right],
    )


def _coviar_candidate_dirs(extra_dir=None):
    repo_root = Path(__file__).resolve().parents[1]
    candidates = []
    if extra_dir:
        candidates.append(Path(extra_dir))
    for env_name in ("COVIAR_DATA_LOADER_DIR", "COVIAR_PATH"):
        if os.environ.get(env_name):
            candidates.append(Path(os.environ[env_name]))
    candidates.extend([
        repo_root / "pytorch-coviar" / "data_loader",
        repo_root / "Coviar" / "data_loader",
        repo_root / "coviar" / "data_loader",
        repo_root / "coviar",
        Path("/home/fuh/m2clip/Coviar/data_loader"),
        Path("/home/fuh/Efficient-CLIP-codex/Coviar/data_loader"),
        Path("/home/fuh/Efficient-CLIP-codex/pytorch-coviar/data_loader"),
    ])
    seen = set()
    for path in candidates:
        path = Path(path).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        yield path


def _coviar_ffmpeg_candidate_dirs(extra_dir=None):
    repo_root = Path(__file__).resolve().parents[1]
    candidates = []
    if extra_dir:
        candidate = Path(extra_dir).expanduser()
        if candidate.name == "data_loader":
            candidates.append(candidate / "ffmpeg" / "lib")
        elif candidate.name == "lib":
            candidates.append(candidate)
    for env_name in ("COVIAR_FFMPEG_LIB", "COVIAR_FFMPEG_DIR"):
        if os.environ.get(env_name):
            value = Path(os.environ[env_name]).expanduser()
            candidates.append(value / "lib" if value.name != "lib" else value)
    candidates.extend([
        repo_root / "pytorch-coviar" / "data_loader" / "ffmpeg" / "lib",
        repo_root / "Coviar" / "data_loader" / "ffmpeg" / "lib",
        repo_root / "coviar" / "data_loader" / "ffmpeg" / "lib",
        Path("/home/fuh/ffmpeg_coviar/lib"),
        Path("/home/fuh/m2clip/Coviar/data_loader/ffmpeg/lib"),
        Path("/home/fuh/Efficient-CLIP-codex/pytorch-coviar/data_loader/ffmpeg/lib"),
    ])
    seen = set()
    for path in candidates:
        path = Path(path).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        yield path


def _prepend_ld_library_path(paths):
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = [part for part in existing.split(os.pathsep) if part]
    new_parts = []
    for path in paths:
        text = str(path)
        if text and text not in new_parts and text not in existing_parts:
            new_parts.append(text)
    if new_parts:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(new_parts + existing_parts)


def _preload_coviar_ffmpeg_libraries(extra_dir=None):
    """Best-effort preload for FFmpeg libs used by coviar*.so.

    LD_LIBRARY_PATH must normally be set before Python starts. This preloader
    helps when the extension is imported with dlopen and the FFmpeg directory is
    known, and it also gives a better error trail when a library is missing.
    """
    global coviar_preload_errors
    coviar_preload_errors = []
    dirs = [path for path in _coviar_ffmpeg_candidate_dirs(extra_dir) if path.exists()]
    _prepend_ld_library_path(dirs)
    if os.name == "nt":
        for path in dirs:
            try:
                os.add_dll_directory(str(path))
            except (AttributeError, OSError) as exc:
                coviar_preload_errors.append("%s: %s" % (path, exc))
        return

    sonames = [
        "libavutil.so.55",
        "libswresample.so.2",
        "libavcodec.so.57",
        "libavformat.so.57",
        "libswscale.so.4",
    ]
    for directory in dirs:
        for soname in sonames:
            library = directory / soname
            if not library.exists():
                continue
            try:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                coviar_preload_errors.append("%s: %s" % (library, exc))


def ensure_coviar_loader(extra_dir=None):
    """Import CoViAR from an explicit, environment, local, or common server path."""
    global coviar_get_num_frames, coviar_load, coviar_import_error
    if coviar_get_num_frames is not None and coviar_load is not None:
        return coviar_get_num_frames, coviar_load

    for candidate in _coviar_candidate_dirs(extra_dir):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    _preload_coviar_ffmpeg_libraries(extra_dir)
    if sys.modules.get("coviar") is None:
        sys.modules.pop("coviar", None)
    try:
        module = importlib.import_module("coviar")
        coviar_get_num_frames = module.get_num_frames
        coviar_load = module.load
        coviar_import_error = None
        return coviar_get_num_frames, coviar_load
    except Exception as exc:
        coviar_get_num_frames = None
        coviar_load = None
        coviar_import_error = exc
        raise ImportError(
            "CompressedVideoDataset requires coviar. Build pytorch-coviar/data_loader "
            "with `cd pytorch-coviar/data_loader && bash install.sh`, or pass "
            "--coviar-data-loader-dir to a directory containing coviar*.so. If the "
            "error mentions libavutil.so.55, set COVIAR_FFMPEG_LIB to the directory "
            "containing FFmpeg 3.x libs. Tried data_loader dirs: %s. Tried FFmpeg "
            "lib dirs: %s. Preload errors: %s. Original import error: %s"
            % (
                ", ".join(str(p) for p in _coviar_candidate_dirs(extra_dir)),
                ", ".join(str(p) for p in _coviar_ffmpeg_candidate_dirs(extra_dir)),
                "; ".join(coviar_preload_errors) if coviar_preload_errors else "none",
                exc,
            )
        )


def _import_coviar_loader():
    try:
        get_num_frames, load = ensure_coviar_loader()
        return get_num_frames, load, None
    except ImportError as exc:
        return None, None, exc


coviar_get_num_frames, coviar_load, coviar_import_error = _import_coviar_loader()


def resolve_video_path(root, relative_path, dataset_name, raw_line, default_suffix=".mp4"):
    rel = Path(relative_path)
    candidate = rel if rel.is_absolute() else Path(root) / rel
    if candidate.suffix:
        return str(candidate)
    for suffix in (".mp4", ".avi", ".webm", ".mkv", ".mov"):
        suffixed = Path(str(candidate) + suffix)
        if suffixed.exists():
            return str(suffixed)
    return str(Path(str(candidate) + default_suffix))


def sample_gop_indices(num_frames, candidate_frames, gop_size, random_sample, temporal_view=0, num_temporal_views=1):
    if num_frames <= 0:
        raise ValueError("num_frames must be positive, got %d." % num_frames)
    gop_count = max(1, int(math.ceil(float(num_frames) / float(gop_size))))
    if gop_count >= candidate_frames:
        boundaries = np.linspace(0, gop_count, candidate_frames + 1)
        indices = []
        for start_f, end_f in zip(boundaries[:-1], boundaries[1:]):
            start = int(math.floor(start_f))
            end = max(start + 1, int(math.ceil(end_f)))
            end = min(end, gop_count)
            if random_sample:
                indices.append(random.randint(start, end - 1))
            else:
                if num_temporal_views <= 1:
                    frac = 0.5
                else:
                    frac = (temporal_view + 0.5) / float(num_temporal_views)
                indices.append(min(end - 1, start + int(math.floor((end - start) * frac))))
        valid = [True] * candidate_frames
    else:
        indices = np.round(np.linspace(0, gop_count - 1, candidate_frames)).astype(np.int64).tolist()
        valid = [True] * candidate_frames
    return indices, torch.tensor(valid, dtype=torch.bool), gop_count


class CompressedVideoDataset(torch.utils.data.Dataset):
    """CoViAR-backed compressed video dataset for EM-CLIP.

    The dataset never uses OpenCV full decoding to fabricate MV or residuals.
    """

    def __init__(
        self,
        dataset_name,
        list_path,
        data_root,
        num_classes,
        candidate_frames=16,
        input_size=256,
        gop_size=12,
        random_sample=True,
        class_to_idx=None,
        compressed_video_root=None,
        coviar_data_loader_dir=None,
        num_temporal_views=1,
        num_spatial_crops=1,
        verify_paths=False,
    ):
        self.dataset_name = dataset_name
        self.list_path = list_path
        self.data_root = data_root
        self.compressed_video_root = compressed_video_root or data_root
        self.num_classes = num_classes
        self.candidate_frames = candidate_frames
        self.input_size = input_size
        self.gop_size = gop_size
        self.random_sample = random_sample
        self.num_temporal_views = num_temporal_views
        self.num_spatial_crops = num_spatial_crops
        if coviar_load is None:
            ensure_coviar_loader(coviar_data_loader_dir)
        with open(list_path, "r", encoding="utf-8") as handle:
            self.items = [
                parse_video_list_line(line, num_classes, dataset_name, class_to_idx)
                for line in handle
                if line.strip()
            ]
        if verify_paths:
            for item in self.items:
                path = resolve_video_path(self.compressed_video_root, item.relative_path, dataset_name, item.raw_line)
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        "dataset=%s raw_line=%r parsed_path=%s does not exist"
                        % (dataset_name, item.raw_line, path)
                    )

    def __len__(self):
        return len(self.items)

    def _prepare_iframe(self, arr):
        if arr is None:
            raise RuntimeError("CoViAR returned None for I-frame.")
        arr = np.asarray(arr)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise RuntimeError("I-frame must be [H, W, 3], got %s." % (arr.shape,))
        arr = arr[..., ::-1].copy()
        tensor = torch.as_tensor(arr).float().permute(2, 0, 1) / 255.0
        return tensor

    def _prepare_mv(self, arr, ref_hw):
        if arr is None:
            return torch.zeros(2, *ref_hw, dtype=torch.float32)
        arr = np.asarray(arr)
        if arr.ndim != 3 or arr.shape[2] != 2:
            raise RuntimeError("MV must be [H, W, 2], got %s." % (arr.shape,))
        tensor = torch.as_tensor(arr).float().permute(2, 0, 1)
        return tensor

    def _prepare_residual(self, arr, ref_hw):
        if arr is None:
            return torch.zeros(3, *ref_hw, dtype=torch.float32)
        arr = np.asarray(arr)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise RuntimeError("Residual must be [H, W, 3], got %s." % (arr.shape,))
        return torch.as_tensor(arr).float().permute(2, 0, 1) / 255.0

    def _load_gop(self, path, gop_idx, num_frames):
        iframe = self._prepare_iframe(coviar_load(path, gop_idx, 0, 0, False))
        ref_hw = tuple(iframe.shape[-2:])
        frames_left = max(0, num_frames - gop_idx * self.gop_size)
        p_pos = min(self.gop_size - 1, frames_left - 1)
        if p_pos <= 0:
            mv = torch.zeros(2, *ref_hw, dtype=torch.float32)
            residual = torch.zeros(3, *ref_hw, dtype=torch.float32)
        else:
            # MV uses the last valid P-frame without accumulation. Residual uses
            # CoViAR accumulation so it is relative to the GOP I-frame.
            mv = self._prepare_mv(coviar_load(path, gop_idx, p_pos, 1, False), ref_hw)
            residual = self._prepare_residual(coviar_load(path, gop_idx, p_pos, 2, True), ref_hw)
        return iframe, mv, residual

    def _augment(self, i_frames, motion_vectors, residuals, random_sample, crop_id=0):
        target = self.input_size
        h, w = i_frames.shape[-2:]
        short = min(h, w)
        scale = float(target) / float(short)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        i_frames, motion_vectors, residuals = resize_modalities(i_frames, motion_vectors, residuals, (new_h, new_w))
        if random_sample:
            top = random.randint(0, max(0, new_h - target))
            left = random.randint(0, max(0, new_w - target))
        else:
            offsets = [0.5] if self.num_spatial_crops == 1 else [0.0, 0.5, 1.0]
            frac = offsets[min(crop_id, len(offsets) - 1)]
            if new_h > new_w:
                top = int(round((new_h - target) * frac))
                left = (new_w - target) // 2
            else:
                top = (new_h - target) // 2
                left = int(round((new_w - target) * frac))
        i_frames, motion_vectors, residuals = crop_modalities(
            i_frames, motion_vectors, residuals, top, left, target, target
        )
        if random_sample and random.random() < 0.5:
            i_frames, motion_vectors, residuals = horizontal_flip_modalities(i_frames, motion_vectors, residuals)
        return i_frames, motion_vectors, residuals

    def _normalize(self, i_frames, motion_vectors, residuals):
        mean = CLIP_MEAN.view(1, 3, 1, 1)
        std = CLIP_STD.view(1, 3, 1, 1)
        i_frames = (i_frames - mean) / std
        motion_vectors = motion_vectors.clamp(-20.0, 20.0) / 20.0
        residuals = residuals.clamp(-1.0, 1.0)
        return i_frames, motion_vectors, residuals

    def _load_view(self, item, temporal_view=0, crop_id=0):
        path = resolve_video_path(self.compressed_video_root, item.relative_path, self.dataset_name, item.raw_line)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "dataset=%s raw_line=%r parsed_path=%s does not exist"
                % (self.dataset_name, item.raw_line, path)
            )
        num_frames = item.num_frames if item.num_frames is not None else coviar_get_num_frames(path)
        gop_indices, valid_mask, gop_count = sample_gop_indices(
            num_frames,
            self.candidate_frames,
            self.gop_size,
            self.random_sample,
            temporal_view=temporal_view,
            num_temporal_views=self.num_temporal_views,
        )
        i_list, mv_list, r_list = [], [], []
        for gop_idx in gop_indices:
            try:
                iframe, mv, residual = self._load_gop(path, int(gop_idx), num_frames)
            except Exception as exc:
                raise RuntimeError(
                    "CoViAR load failed for dataset=%s raw_line=%r path=%s gop=%s: %s"
                    % (self.dataset_name, item.raw_line, path, gop_idx, exc)
                )
            i_list.append(iframe)
            mv_list.append(mv)
            r_list.append(residual)
        i_frames = torch.stack(i_list)
        motion_vectors = torch.stack(mv_list)
        residuals = torch.stack(r_list)
        i_frames, motion_vectors, residuals = self._augment(
            i_frames,
            motion_vectors,
            residuals,
            random_sample=self.random_sample,
            crop_id=crop_id,
        )
        i_frames, motion_vectors, residuals = self._normalize(i_frames, motion_vectors, residuals)
        return {
            "i_frames": i_frames,
            "motion_vectors": motion_vectors,
            "residuals": residuals,
            "valid_mask": valid_mask,
            "candidate_gop_indices": torch.tensor(gop_indices, dtype=torch.long),
            "metadata": {
                "video_path": path,
                "gop_count": gop_count,
                "raw_line": item.raw_line,
            },
        }

    def __getitem__(self, index):
        item = self.items[index]
        if self.random_sample:
            sample = self._load_view(item)
        else:
            views = []
            for temporal_view in range(self.num_temporal_views):
                for crop_id in range(self.num_spatial_crops):
                    views.append(self._load_view(item, temporal_view=temporal_view, crop_id=crop_id))
            sample = {
                "i_frames": torch.stack([v["i_frames"] for v in views]),
                "motion_vectors": torch.stack([v["motion_vectors"] for v in views]),
                "residuals": torch.stack([v["residuals"] for v in views]),
                "valid_mask": torch.stack([v["valid_mask"] for v in views]),
                "candidate_gop_indices": torch.stack([v["candidate_gop_indices"] for v in views]),
                "metadata": views[0]["metadata"],
            }
        sample["label"] = torch.tensor(item.label, dtype=torch.long)
        return sample
