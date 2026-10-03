"""Canonical CoViAR dataset used by the complete EM-CLIP pipeline.

This module reads real MPEG-4 I-frames, motion vectors, and residuals through
``coviar``.  It deliberately does not use OpenCV video decoding and it applies
one synchronized spatial transform to all three compressed representations.
"""

import ctypes
import importlib
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


GOP_SIZE = 12
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

# CoViAR remains lazy so --coviar-data-loader-dir is honored before the native
# extension is imported.  The module-level names also make the decoder easy to
# replace with a deterministic fake in unit tests.
coviar_get_num_frames = None
coviar_get_num_gops = None
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
    """Remove optional class-name metadata after a video path.

    K400 rows commonly use ``path.mp4 class_name label``.  A suffix is the
    unambiguous boundary and still permits spaces inside the path.
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


def parse_video_list_line(
    line,
    num_classes,
    dataset_name,
    class_to_idx: Optional[Dict[str, int]] = None,
):
    """Parse supported compressed-video list formats without guessing a suffix."""
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
                "dataset %s list line has a non-integer label and no class mapping: %s"
                % (dataset_name, raw_line)
            )
        matched = None
        # Try every split so paths and label names can both contain spaces.
        for start in range(1, len(parts)):
            label_name = " ".join(parts[start:])
            if label_name in class_to_idx:
                matched = (start, label_name)
                break
        if matched is None:
            raise ValueError(
                "could not parse label name in dataset %s line: %s"
                % (dataset_name, raw_line)
            )
        start, label_name = matched
        relative_path = " ".join(parts[:start])
        label = int(class_to_idx[label_name])

    if label < 0 or label >= num_classes:
        raise ValueError(
            "dataset %s label %d is outside [0, %d] for line: %s"
            % (dataset_name, label, num_classes - 1, raw_line)
        )
    if not relative_path:
        raise ValueError(
            "dataset %s parsed an empty path from line: %s" % (dataset_name, raw_line)
        )
    if num_frames is not None and num_frames <= 0:
        raise ValueError(
            "dataset %s has non-positive frame count %d for line: %s"
            % (dataset_name, num_frames, raw_line)
        )
    return VideoListItem(relative_path, int(label), num_frames, raw_line)


def horizontal_flip_modalities(i_frames, motion_vectors, residuals):
    """Flip I/MV/R together and reverse only the MV x component."""
    i_frames = i_frames.flip(dims=(-1,))
    residuals = residuals.flip(dims=(-1,))
    motion_vectors = motion_vectors.flip(dims=(-1,)).clone()
    motion_vectors[:, 0] = -motion_vectors[:, 0]
    return i_frames, motion_vectors, residuals


def resize_modalities(i_frames, motion_vectors, residuals, size):
    """Resize I/MV/R together and scale MV values with the spatial geometry."""
    old_h, old_w = i_frames.shape[-2:]
    new_h, new_w = size
    if min(old_h, old_w, new_h, new_w) <= 0:
        raise ValueError(
            "resize dimensions must be positive, got old=%s new=%s"
            % ((old_h, old_w), (new_h, new_w))
        )
    i_frames = F.interpolate(
        i_frames, size=(new_h, new_w), mode="bicubic", align_corners=False
    )
    motion_vectors = F.interpolate(
        motion_vectors, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    residuals = F.interpolate(
        residuals, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    motion_vectors[:, 0] *= float(new_w) / float(old_w)
    motion_vectors[:, 1] *= float(new_h) / float(old_h)
    return i_frames, motion_vectors, residuals


def crop_modalities(i_frames, motion_vectors, residuals, top, left, height, width):
    """Apply exactly the same crop window to I/MV/R."""
    source_h, source_w = i_frames.shape[-2:]
    if top < 0 or left < 0 or height <= 0 or width <= 0:
        raise ValueError(
            "invalid crop top=%d left=%d height=%d width=%d"
            % (top, left, height, width)
        )
    bottom = top + height
    right = left + width
    if bottom > source_h or right > source_w:
        raise ValueError(
            "crop [%d:%d, %d:%d] exceeds source size %s"
            % (top, bottom, left, right, (source_h, source_w))
        )
    return (
        i_frames[:, :, top:bottom, left:right],
        motion_vectors[:, :, top:bottom, left:right],
        residuals[:, :, top:bottom, left:right],
    )


def _coviar_candidate_dirs(extra_dir=None):
    repo_root = Path(__file__).resolve().parent
    candidates = []
    if extra_dir:
        candidates.append(Path(extra_dir))
    for env_name in ("COVIAR_DATA_LOADER_DIR", "COVIAR_PATH"):
        if os.environ.get(env_name):
            candidates.append(Path(os.environ[env_name]))
    candidates.extend(
        [
            repo_root / "pytorch-coviar" / "data_loader",
            repo_root / "Coviar" / "data_loader",
            repo_root / "coviar" / "data_loader",
            repo_root / "coviar",
            Path("/home/fuh/m2clip/Coviar/data_loader"),
            Path("/home/fuh/Efficient-CLIP-codex/Coviar/data_loader"),
            Path("/home/fuh/Efficient-CLIP-codex/pytorch-coviar/data_loader"),
        ]
    )
    seen = set()
    for path in candidates:
        path = Path(path).expanduser()
        key = str(path)
        if key not in seen:
            seen.add(key)
            yield path


def _coviar_ffmpeg_candidate_dirs(extra_dir=None):
    repo_root = Path(__file__).resolve().parent
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
    candidates.extend(
        [
            repo_root / "pytorch-coviar" / "data_loader" / "ffmpeg" / "lib",
            repo_root / "Coviar" / "data_loader" / "ffmpeg" / "lib",
            repo_root / "coviar" / "data_loader" / "ffmpeg" / "lib",
            Path("/home/fuh/ffmpeg_coviar/lib"),
            Path("/home/fuh/m2clip/Coviar/data_loader/ffmpeg/lib"),
            Path("/home/fuh/Efficient-CLIP-codex/pytorch-coviar/data_loader/ffmpeg/lib"),
        ]
    )
    seen = set()
    for path in candidates:
        path = Path(path).expanduser()
        key = str(path)
        if key not in seen:
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
    """Best-effort preload of the FFmpeg 3.x libraries used by coviar*.so."""
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
    """Load CoViAR from an explicit, environment, local, or known server path."""
    global coviar_get_num_frames, coviar_get_num_gops, coviar_load, coviar_import_error
    if extra_dir:
        loaded = sys.modules.get("coviar")
        loaded_file = getattr(loaded, "__file__", None)
        requested_dir = Path(extra_dir).expanduser().resolve()
        loaded_matches = False
        if loaded_file:
            loaded_path = Path(loaded_file).expanduser().resolve()
            loaded_matches = (
                loaded_path.parent == requested_dir or requested_dir in loaded_path.parents
            )
        if not loaded_matches:
            coviar_get_num_frames = None
            coviar_get_num_gops = None
            coviar_load = None
            sys.modules.pop("coviar", None)
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
        coviar_get_num_gops = getattr(module, "get_num_gops", None)
        coviar_load = module.load
        coviar_import_error = None
        return coviar_get_num_frames, coviar_load
    except Exception as exc:
        coviar_get_num_frames = None
        coviar_get_num_gops = None
        coviar_load = None
        coviar_import_error = exc
        raise ImportError(
            "CoviarDataSet requires coviar. Build pytorch-coviar/data_loader with "
            "`cd pytorch-coviar/data_loader && bash install.sh`, or pass "
            "--coviar-data-loader-dir. If libavutil.so.55 is missing, set "
            "COVIAR_FFMPEG_LIB to the FFmpeg 3.x lib directory. Tried loaders: %s. "
            "Tried FFmpeg libs: %s. Preload errors: %s. Original error: %s"
            % (
                ", ".join(str(p) for p in _coviar_candidate_dirs(extra_dir)),
                ", ".join(str(p) for p in _coviar_ffmpeg_candidate_dirs(extra_dir)),
                "; ".join(coviar_preload_errors) if coviar_preload_errors else "none",
                exc,
            )
        )


def resolve_video_path(
    root, relative_path, dataset_name, raw_line, default_suffix=".mp4"
):
    """Resolve suffix-less IDs and avoid duplicated train/val path components."""
    root = Path(os.path.expandvars(os.path.expanduser(str(root))))
    rel = Path(os.path.expandvars(os.path.expanduser(str(relative_path))))
    if rel.is_absolute():
        base_candidates = [rel]
    else:
        base_candidates = [root / rel]
        # Some K400 roots already end in train/val while their list rows also
        # include that component.  Prefer the non-duplicated existing path.
        if rel.parts and root.name.casefold() == rel.parts[0].casefold():
            base_candidates.insert(0, root.parent / rel)

    suffixes = ("",) if rel.suffix else (".mp4", ".avi", ".webm", ".mkv", ".mov")
    for base in base_candidates:
        for suffix in suffixes:
            candidate = base if not suffix else Path(str(base) + suffix)
            if candidate.exists():
                return str(candidate)

    # Return a deterministic path for the caller's detailed error message.
    fallback = base_candidates[0]
    if not fallback.suffix:
        fallback = Path(str(fallback) + default_suffix)
    return str(fallback)


def sample_gop_indices(
    num_frames,
    candidate_frames,
    gop_size,
    random_sample,
    temporal_view=0,
    num_temporal_views=1,
    gop_count=None,
    deduplicate_candidates=True,
):
    """Sample one GOP from each of T non-overlapping temporal segments."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive, got %d" % num_frames)
    if candidate_frames <= 0:
        raise ValueError("candidate_frames must be positive, got %d" % candidate_frames)
    if gop_size <= 0:
        raise ValueError("gop_size must be positive, got %d" % gop_size)
    if num_temporal_views <= 0 or not 0 <= temporal_view < num_temporal_views:
        raise ValueError(
            "temporal_view=%d must be in [0, num_temporal_views=%d)"
            % (temporal_view, num_temporal_views)
        )
    if gop_count is None:
        gop_count = int(math.ceil(float(num_frames) / float(gop_size)))
    gop_count = max(1, int(gop_count))

    if gop_count >= candidate_frames:
        boundaries = [
            (segment * gop_count) // candidate_frames
            for segment in range(candidate_frames + 1)
        ]
        indices = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end <= start:
                raise RuntimeError(
                    "internal GOP segmentation error: segment [%d, %d) is empty"
                    % (start, end)
                )
            if random_sample:
                indices.append(random.randint(start, end - 1))
            else:
                fraction = (
                    0.5
                    if num_temporal_views == 1
                    else (temporal_view + 0.5) / float(num_temporal_views)
                )
                offset = min(end - start - 1, int(math.floor((end - start) * fraction)))
                indices.append(start + offset)
    else:
        # Preserve fixed [T,...] tensors, but exclude duplicate observations
        # from saliency normalization/top-k. Repetition is only needed when
        # the number of distinct GOPs is below K, not merely below T.
        indices = (
            np.round(np.linspace(0, gop_count - 1, candidate_frames))
            .astype(np.int64)
            .tolist()
        )
    valid_mask = torch.ones(candidate_frames, dtype=torch.bool)
    if deduplicate_candidates:
        seen = set()
        for slot, index in enumerate(indices):
            valid_mask[slot] = index not in seen
            seen.add(index)
    return indices, valid_mask, gop_count


class CoviarDataSet(torch.utils.data.Dataset):
    """Return synchronized compressed representations for EM-CLIP.

    Training items contain I ``[T,3,H,W]``, MV ``[T,2,H,W]`` and residual
    ``[T,3,H,W]``.  Evaluation adds a leading view dimension ``V`` so logits
    can be averaged per video instead of treating views as separate samples.
    """

    def __init__(
        self,
        dataset_name,
        list_path,
        data_root,
        num_classes,
        candidate_frames=16,
        input_size=256,
        gop_size=GOP_SIZE,
        random_sample=True,
        class_to_idx=None,
        compressed_video_root=None,
        coviar_data_loader_dir=None,
        num_temporal_views=1,
        num_spatial_crops=1,
        verify_paths=False,
        default_video_suffix=".mp4",
        mv_clamp=20.0,
        residual_scale=255.0,
        residual_clamp=1.0,
        mv_accumulate=False,
        residual_accumulate=True,
        horizontal_flip=None,
        residual_channel_order="rgb",
        deduplicate_candidates=True,
    ):
        if num_classes <= 0:
            raise ValueError("num_classes must be positive, got %d" % num_classes)
        if candidate_frames <= 0 or input_size <= 0 or gop_size <= 0:
            raise ValueError(
                "candidate_frames, input_size and gop_size must be positive; got %s"
                % ((candidate_frames, input_size, gop_size),)
            )
        if num_temporal_views <= 0:
            raise ValueError("num_temporal_views must be positive")
        if num_spatial_crops not in (1, 3):
            raise ValueError("num_spatial_crops must be 1 or 3")
        if mv_clamp <= 0 or residual_scale <= 0 or residual_clamp <= 0:
            raise ValueError("normalization scales/clamps must be positive")

        self.dataset_name = str(dataset_name)
        # SSV2 direction labels change under mirroring. Disable flipping by
        # default instead of silently keeping the original class supervision.
        self.horizontal_flip = (
            not self.dataset_name.lower().startswith("ssv2")
            if horizontal_flip is None else bool(horizontal_flip)
        )
        if self.dataset_name.lower().startswith("ssv2") and self.horizontal_flip:
            raise ValueError("SSV2 horizontal flipping requires a validated label permutation; use --horizontal-flip off")
        if residual_channel_order not in ("rgb", "bgr"):
            raise ValueError("residual_channel_order must be rgb or bgr")
        self.residual_channel_order = residual_channel_order
        self.deduplicate_candidates = bool(deduplicate_candidates)
        self.list_path = str(list_path)
        self.data_root = str(data_root)
        self.compressed_video_root = str(compressed_video_root or data_root)
        self.num_classes = int(num_classes)
        self.candidate_frames = int(candidate_frames)
        self.input_size = int(input_size)
        self.gop_size = int(gop_size)
        self.random_sample = bool(random_sample)
        self.num_temporal_views = int(num_temporal_views)
        self.num_spatial_crops = int(num_spatial_crops)
        self.default_video_suffix = str(default_video_suffix)
        self.mv_clamp = float(mv_clamp)
        self.residual_scale = float(residual_scale)
        self.residual_clamp = float(residual_clamp)
        self.mv_accumulate = bool(mv_accumulate)
        self.residual_accumulate = bool(residual_accumulate)

        if coviar_load is None:
            ensure_coviar_loader(coviar_data_loader_dir)
        try:
            with open(self.list_path, "r", encoding="utf-8-sig") as handle:
                self.items = [
                    parse_video_list_line(line, self.num_classes, self.dataset_name, class_to_idx)
                    for line in handle
                    if line.strip()
                ]
        except OSError as exc:
            raise OSError(
                "cannot read list for dataset=%s at %s: %s"
                % (self.dataset_name, self.list_path, exc)
            ) from exc
        if not self.items:
            raise ValueError(
                "dataset=%s list contains no samples: %s"
                % (self.dataset_name, self.list_path)
            )
        if verify_paths:
            for item in self.items:
                path = self._resolve_item_path(item)
                if not os.path.isfile(path):
                    raise FileNotFoundError(self._missing_path_message(item, path))

    @property
    def classes(self):
        """Class indices present in the list (kept for legacy callers)."""
        return sorted({item.label for item in self.items})

    def __len__(self):
        return len(self.items)

    def _resolve_item_path(self, item):
        return resolve_video_path(
            self.compressed_video_root,
            item.relative_path,
            self.dataset_name,
            item.raw_line,
            default_suffix=self.default_video_suffix,
        )

    def _missing_path_message(self, item, path):
        return "dataset=%s raw_line=%r parsed_path=%s does not exist" % (
            self.dataset_name,
            item.raw_line,
            path,
        )

    def preflight(self, index=0):
        """Synchronously decode one view for a safe single-process smoke test."""
        if index < 0 or index >= len(self.items):
            raise IndexError(
                "preflight index %d is outside dataset length %d"
                % (index, len(self.items))
            )
        item = self.items[index]
        sample = self._load_view(item, temporal_view=0, crop_id=0)
        sample["label"] = torch.tensor(item.label, dtype=torch.long)
        return sample

    @staticmethod
    def _prepare_iframe(arr):
        if arr is None:
            raise RuntimeError("CoViAR returned None for I-frame")
        arr = np.asarray(arr)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise RuntimeError("I-frame must be [H,W,3], got %s" % (arr.shape,))
        # CoViAR exposes decoded appearance in BGR; CLIP expects RGB.
        rgb = arr[..., ::-1].copy()
        return torch.as_tensor(rgb).float().permute(2, 0, 1) / 255.0

    @staticmethod
    def _prepare_mv(arr, ref_hw):
        if arr is None:
            raise RuntimeError("CoViAR returned None for a valid motion-vector frame")
        arr = np.asarray(arr)
        if arr.ndim != 3 or arr.shape[2] != 2:
            raise RuntimeError("MV must be [H,W,2], got %s" % (arr.shape,))
        tensor = torch.as_tensor(arr).float().permute(2, 0, 1)
        if tuple(tensor.shape[-2:]) != tuple(ref_hw):
            raise RuntimeError(
                "MV spatial size %s does not match I-frame %s"
                % (tuple(tensor.shape[-2:]), tuple(ref_hw))
            )
        return tensor

    def _prepare_residual(self, arr, ref_hw):
        if arr is None:
            raise RuntimeError("CoViAR returned None for a valid residual frame")
        arr = np.asarray(arr)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise RuntimeError("Residual must be [H,W,3], got %s" % (arr.shape,))
        # Native CoViAR produces BGR residuals; the pretrained residual stem
        # uses RGB CLIP weights. BGR is retained only for legacy checkpoints.
        if self.residual_channel_order == "rgb":
            arr = arr[..., ::-1].copy()
        tensor = torch.as_tensor(arr).float().permute(2, 0, 1)
        if tuple(tensor.shape[-2:]) != tuple(ref_hw):
            raise RuntimeError(
                "residual spatial size %s does not match I-frame %s"
                % (tuple(tensor.shape[-2:]), tuple(ref_hw))
            )
        return tensor / self.residual_scale

    def _last_p_position(self, gop_idx, num_frames):
        frames_left = max(0, int(num_frames) - int(gop_idx) * self.gop_size)
        return min(self.gop_size - 1, frames_left - 1)

    def _load_gop(self, path, gop_idx, num_frames, gop_count):
        if gop_idx < 0 or gop_idx >= gop_count:
            raise IndexError(
                "dataset=%s requested GOP %d but CoViAR reports %d GOPs for %s"
                % (self.dataset_name, gop_idx, gop_count, path)
            )
        iframe = self._prepare_iframe(coviar_load(path, gop_idx, 0, 0, False))
        ref_hw = tuple(iframe.shape[-2:])
        p_position = self._last_p_position(gop_idx, num_frames)
        if p_position <= 0:
            # A GOP containing only an I-frame has no legal MV/residual query.
            mv = torch.zeros(2, *ref_hw, dtype=torch.float32)
            residual = torch.zeros(3, *ref_hw, dtype=torch.float32)
        else:
            mv = self._prepare_mv(
                coviar_load(path, gop_idx, p_position, 1, self.mv_accumulate),
                ref_hw,
            )
            residual = self._prepare_residual(
                coviar_load(path, gop_idx, p_position, 2, self.residual_accumulate),
                ref_hw,
            )
        return iframe, mv, residual, p_position

    def _augment(self, i_frames, motion_vectors, residuals, crop_id=0):
        target = self.input_size
        height, width = i_frames.shape[-2:]
        short = min(height, width)
        scale = float(target) / float(short)
        new_h, new_w = int(round(height * scale)), int(round(width * scale))
        i_frames, motion_vectors, residuals = resize_modalities(
            i_frames, motion_vectors, residuals, (new_h, new_w)
        )
        if self.random_sample:
            top = random.randint(0, max(0, new_h - target))
            left = random.randint(0, max(0, new_w - target))
        else:
            offsets = [0.5] if self.num_spatial_crops == 1 else [0.0, 0.5, 1.0]
            fraction = offsets[crop_id]
            if new_h > new_w:
                top = int(round((new_h - target) * fraction))
                left = (new_w - target) // 2
            else:
                top = (new_h - target) // 2
                left = int(round((new_w - target) * fraction))
        i_frames, motion_vectors, residuals = crop_modalities(
            i_frames, motion_vectors, residuals, top, left, target, target
        )
        if self.random_sample and self.horizontal_flip and random.random() < 0.5:
            i_frames, motion_vectors, residuals = horizontal_flip_modalities(
                i_frames, motion_vectors, residuals
            )
        return i_frames, motion_vectors, residuals

    def _normalize(self, i_frames, motion_vectors, residuals):
        mean = CLIP_MEAN.view(1, 3, 1, 1)
        std = CLIP_STD.view(1, 3, 1, 1)
        i_frames = (i_frames - mean) / std
        motion_vectors = motion_vectors.clamp(-self.mv_clamp, self.mv_clamp)
        motion_vectors = motion_vectors / self.mv_clamp
        residuals = residuals.clamp(-self.residual_clamp, self.residual_clamp)
        return i_frames, motion_vectors, residuals

    def _load_view(self, item, temporal_view=0, crop_id=0):
        path = self._resolve_item_path(item)
        if not os.path.isfile(path):
            raise FileNotFoundError(self._missing_path_message(item, path))

        # Native metadata is authoritative; list frame counts can be stale.
        num_frames = int(coviar_get_num_frames(path))
        if num_frames <= 0:
            raise RuntimeError(
                "dataset=%s CoViAR reported no frames for raw_line=%r path=%s"
                % (self.dataset_name, item.raw_line, path)
            )
        gop_count = (
            int(coviar_get_num_gops(path))
            if coviar_get_num_gops is not None
            else int(math.ceil(float(num_frames) / float(self.gop_size)))
        )
        if gop_count <= 0:
            raise RuntimeError(
                "dataset=%s CoViAR reported no GOPs for raw_line=%r path=%s"
                % (self.dataset_name, item.raw_line, path)
            )
        gop_indices, valid_mask, gop_count = sample_gop_indices(
            num_frames,
            self.candidate_frames,
            self.gop_size,
            self.random_sample,
            temporal_view=temporal_view,
            num_temporal_views=self.num_temporal_views,
            gop_count=gop_count,
            deduplicate_candidates=self.deduplicate_candidates,
        )

        i_list, mv_list, residual_list, p_positions = [], [], [], []
        for gop_idx in gop_indices:
            try:
                iframe, mv, residual, p_position = self._load_gop(
                    path, int(gop_idx), num_frames, gop_count
                )
            except Exception as exc:
                raise RuntimeError(
                    "CoViAR load failed for dataset=%s raw_line=%r path=%s gop=%s: %s"
                    % (self.dataset_name, item.raw_line, path, gop_idx, exc)
                ) from exc
            i_list.append(iframe)
            mv_list.append(mv)
            residual_list.append(residual)
            p_positions.append(p_position)
        try:
            i_frames = torch.stack(i_list)
            motion_vectors = torch.stack(mv_list)
            residuals = torch.stack(residual_list)
        except RuntimeError as exc:
            raise RuntimeError(
                "inconsistent decoded shapes for dataset=%s raw_line=%r path=%s: %s"
                % (self.dataset_name, item.raw_line, path, exc)
            ) from exc
        assert i_frames.shape[:2] == (self.candidate_frames, 3)
        assert motion_vectors.shape[:2] == (self.candidate_frames, 2)
        assert residuals.shape[:2] == (self.candidate_frames, 3)

        i_frames, motion_vectors, residuals = self._augment(
            i_frames, motion_vectors, residuals, crop_id=crop_id
        )
        i_frames, motion_vectors, residuals = self._normalize(
            i_frames, motion_vectors, residuals
        )
        expected_i = (self.candidate_frames, 3, self.input_size, self.input_size)
        expected_mv = (self.candidate_frames, 2, self.input_size, self.input_size)
        assert tuple(i_frames.shape) == expected_i, (tuple(i_frames.shape), expected_i)
        assert tuple(motion_vectors.shape) == expected_mv, (
            tuple(motion_vectors.shape),
            expected_mv,
        )
        assert tuple(residuals.shape) == expected_i, (tuple(residuals.shape), expected_i)
        return {
            "i_frames": i_frames,
            "motion_vectors": motion_vectors,
            "residuals": residuals,
            "valid_mask": valid_mask,
            "candidate_gop_indices": torch.tensor(gop_indices, dtype=torch.long),
            "metadata": {
                "video_path": path,
                "gop_count": gop_count,
                "num_frames": num_frames,
                "last_p_positions": p_positions,
                "raw_line": item.raw_line,
                "unique_candidate_gops": len(set(gop_indices)),
                "residual_channel_order": self.residual_channel_order,
            },
        }

    def __getitem__(self, index):
        item = self.items[index]
        if self.random_sample:
            sample = self._load_view(item)
        else:
            views = [
                self._load_view(item, temporal_view=temporal_view, crop_id=crop_id)
                for temporal_view in range(self.num_temporal_views)
                for crop_id in range(self.num_spatial_crops)
            ]
            # [V,T,C,H,W], where V=temporal_views*spatial_crops.
            sample = {
                "i_frames": torch.stack([view["i_frames"] for view in views]),
                "motion_vectors": torch.stack(
                    [view["motion_vectors"] for view in views]
                ),
                "residuals": torch.stack([view["residuals"] for view in views]),
                "valid_mask": torch.stack([view["valid_mask"] for view in views]),
                "candidate_gop_indices": torch.stack(
                    [view["candidate_gop_indices"] for view in views]
                ),
                "metadata": views[0]["metadata"],
            }
        sample["label"] = torch.tensor(item.label, dtype=torch.long)
        return sample


# Compatibility name for old project imports.  It intentionally refers to the
# same class so there is only one implementation and one behavior to maintain.
CompressedVideoDataset = CoviarDataSet


__all__ = [
    "CLIP_MEAN",
    "CLIP_STD",
    "GOP_SIZE",
    "CompressedVideoDataset",
    "CoviarDataSet",
    "VideoListItem",
    "crop_modalities",
    "ensure_coviar_loader",
    "horizontal_flip_modalities",
    "parse_video_list_line",
    "resolve_video_path",
    "resize_modalities",
    "sample_gop_indices",
]
