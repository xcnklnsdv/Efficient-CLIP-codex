import numpy as np
import pytest
import torch
from types import SimpleNamespace

import dataset_coviar as cvd
import main_emclip
from main_emclip import load_class_names
from dataset_coviar import (
    CoviarDataSet,
    crop_modalities,
    ensure_coviar_loader,
    horizontal_flip_modalities,
    parse_video_list_line,
    resolve_video_path,
    resize_modalities,
    sample_gop_indices,
)


def test_parse_video_list_line_supports_path_num_frames_label():
    item = parse_video_list_line(
        "class_a/video.mp4 123 4",
        num_classes=10,
        dataset_name="unit",
    )

    assert item.relative_path == "class_a/video.mp4"
    assert item.num_frames == 123
    assert item.label == 4


def test_parse_video_list_line_supports_path_class_name_label():
    item = parse_video_list_line(
        "train_256/cleaning_floor/UB61Z0vxSPg_000030_000040.mp4 cleaning_floor 60",
        num_classes=400,
        dataset_name="k400",
    )

    assert item.relative_path == "train_256/cleaning_floor/UB61Z0vxSPg_000030_000040.mp4"
    assert item.num_frames is None
    assert item.label == 60


def test_parse_video_list_line_supports_space_separated_label_names():
    item = parse_video_list_line(
        "videos/my clip.mp4 playing guitar",
        num_classes=3,
        dataset_name="unit",
        class_to_idx={"playing guitar": 2},
    )

    assert item.relative_path == "videos/my clip.mp4"
    assert item.label == 2


def test_horizontal_flip_negates_only_mv_x_component():
    i_frames = torch.zeros(2, 3, 4, 4)
    residuals = torch.zeros(2, 3, 4, 4)
    motion_vectors = torch.zeros(2, 2, 4, 4)
    motion_vectors[:, 0] = 3.0
    motion_vectors[:, 1] = 5.0

    _, flipped_mv, _ = horizontal_flip_modalities(i_frames, motion_vectors, residuals)

    assert torch.all(flipped_mv[:, 0] == -3.0)
    assert torch.all(flipped_mv[:, 1] == 5.0)


def test_resize_scales_motion_vector_components():
    i_frames = torch.zeros(1, 3, 4, 8)
    residuals = torch.zeros(1, 3, 4, 8)
    motion_vectors = torch.ones(1, 2, 4, 8)

    _, resized_mv, _ = resize_modalities(i_frames, motion_vectors, residuals, (8, 4))

    assert resized_mv.shape[-2:] == (8, 4)
    assert torch.allclose(resized_mv[:, 0], torch.full((1, 8, 4), 0.5))
    assert torch.allclose(resized_mv[:, 1], torch.full((1, 8, 4), 2.0))


def test_crop_uses_identical_spatial_window_for_all_modalities():
    base = torch.arange(1 * 1 * 4 * 4, dtype=torch.float32).view(1, 1, 4, 4)
    i_frames = base.repeat(1, 3, 1, 1)
    motion_vectors = base.repeat(1, 2, 1, 1)
    residuals = base.repeat(1, 3, 1, 1)

    cropped_i, cropped_mv, cropped_r = crop_modalities(
        i_frames,
        motion_vectors,
        residuals,
        top=1,
        left=1,
        height=2,
        width=2,
    )

    expected = torch.tensor([[5.0, 6.0], [9.0, 10.0]])
    assert torch.equal(cropped_i[0, 0], expected)
    assert torch.equal(cropped_mv[0, 0], expected)
    assert torch.equal(cropped_r[0, 0], expected)


def test_gop_segments_are_non_overlapping_when_gops_cover_candidates():
    for seed in range(20):
        __import__("random").seed(seed)
        indices, valid_mask, gop_count = sample_gop_indices(
            num_frames=17 * 12,
            candidate_frames=16,
            gop_size=12,
            random_sample=True,
            gop_count=17,
        )
        assert len(indices) == len(set(indices)) == 16
        assert valid_mask.all()
        assert gop_count == 17


def test_temporal_views_are_deterministic_and_stay_inside_disjoint_segments():
    views = []
    for view in range(4):
        indices, _, _ = sample_gop_indices(
            num_frames=64 * 12,
            candidate_frames=4,
            gop_size=12,
            random_sample=False,
            temporal_view=view,
            num_temporal_views=4,
            gop_count=64,
        )
        views.append(indices)
    assert len({tuple(indices) for indices in views}) == 4
    for indices in views:
        assert all(segment * 16 <= index < (segment + 1) * 16 for segment, index in enumerate(indices))


def test_class_names_are_inferred_by_label_from_realistic_list_paths(tmp_path):
    train_list = tmp_path / "train.txt"
    train_list.write_text(
        "walk/video_a.mp4 120 0\nrun_fast/video_b.mp4 run_fast 1\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(class_names=None, label_csv=None, dataset="unit")
    cfg = {"TRAIN_LIST": str(train_list), "VAL_LIST": str(train_list)}

    assert load_class_names(args, 2, cfg=cfg) == ["walk", "run fast"]


def test_ensure_coviar_loader_accepts_user_supplied_data_loader_dir(tmp_path, monkeypatch):
    fake_loader_dir = tmp_path / "Coviar" / "data_loader"
    fake_loader_dir.mkdir(parents=True)
    (fake_loader_dir / "coviar.py").write_text(
        "def get_num_frames(path):\n"
        "    return 12\n"
        "def load(path, gop_idx, pos_in_gop, representation_idx, accumulate):\n"
        "    return None\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(__import__("sys").modules, "coviar", None)
    monkeypatch.setattr(cvd, "coviar_get_num_frames", None)
    monkeypatch.setattr(cvd, "coviar_load", None)

    get_num_frames, load = ensure_coviar_loader(str(fake_loader_dir))

    assert get_num_frames("video.mp4") == 12
    assert load("video.mp4", 0, 0, 0, False) is None


def test_coviar_candidate_dirs_include_local_pytorch_coviar_first():
    candidates = [
        str(path).replace("\\", "/")
        for path in cvd._coviar_candidate_dirs()
    ]

    assert any(path.endswith("pytorch-coviar/data_loader") for path in candidates)
    local_idx = next(i for i, path in enumerate(candidates) if path.endswith("pytorch-coviar/data_loader"))
    server_idx = next(i for i, path in enumerate(candidates) if path == "/home/fuh/m2clip/Coviar/data_loader")
    assert local_idx < server_idx


def test_coviar_ffmpeg_candidate_dirs_include_local_pytorch_coviar_first():
    candidates = [
        str(path).replace("\\", "/")
        for path in cvd._coviar_ffmpeg_candidate_dirs()
    ]

    assert any(path.endswith("pytorch-coviar/data_loader/ffmpeg/lib") for path in candidates)
    local_idx = next(i for i, path in enumerate(candidates) if path.endswith("pytorch-coviar/data_loader/ffmpeg/lib"))
    server_idx = next(i for i, path in enumerate(candidates) if path == "/home/fuh/ffmpeg_coviar/lib")
    assert local_idx < server_idx


def test_training_entry_uses_root_dataset_coviar_as_canonical_source():
    assert main_emclip.CoviarDataSet is CoviarDataSet
    assert main_emclip.CoviarDataSet.__module__ == "dataset_coviar"


def test_k400_path_does_not_duplicate_train_component(tmp_path):
    root = tmp_path / "train"
    video = root / "jump" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.touch()

    resolved = resolve_video_path(
        root,
        "train/jump/clip.mp4",
        dataset_name="k400",
        raw_line="train/jump/clip.mp4 0",
    )

    assert resolved == str(video)


def _install_fake_coviar(monkeypatch, num_frames, num_gops, return_none_for=None):
    calls = []

    def fake_load(path, gop_idx, position, representation, accumulate):
        calls.append((gop_idx, position, representation, accumulate))
        if representation == return_none_for:
            return None
        channels = 2 if representation == 1 else 3
        dtype = np.uint8 if representation == 0 else np.int32
        fill = 10 if representation == 1 else (255 if representation == 2 else 32)
        return np.full((4, 4, channels), fill, dtype=dtype)

    monkeypatch.setattr(cvd, "coviar_get_num_frames", lambda path: num_frames)
    monkeypatch.setattr(cvd, "coviar_get_num_gops", lambda path: num_gops)
    monkeypatch.setattr(cvd, "coviar_load", fake_load)
    return calls


def _make_fake_dataset(tmp_path, monkeypatch, num_frames=14, num_gops=2, return_none_for=None):
    video = tmp_path / "clip.mp4"
    video.touch()
    video_list = tmp_path / "list.txt"
    video_list.write_text("clip.mp4 %d 1\n" % num_frames, encoding="utf-8")
    calls = _install_fake_coviar(
        monkeypatch,
        num_frames=num_frames,
        num_gops=num_gops,
        return_none_for=return_none_for,
    )
    dataset = CoviarDataSet(
        dataset_name="unit",
        list_path=str(video_list),
        data_root=str(tmp_path),
        num_classes=2,
        candidate_frames=num_gops,
        input_size=4,
        gop_size=12,
        random_sample=False,
    )
    return dataset, calls


def test_coviar_dataset_reads_last_valid_p_and_accumulated_residual(tmp_path, monkeypatch):
    dataset, calls = _make_fake_dataset(tmp_path, monkeypatch)

    sample = dataset.preflight(0)

    assert sample["candidate_gop_indices"].tolist() == [0, 1]
    assert sample["metadata"]["last_p_positions"] == [11, 1]
    assert sample["i_frames"].shape == (2, 3, 4, 4)
    assert sample["motion_vectors"].shape == (2, 2, 4, 4)
    assert sample["residuals"].shape == (2, 3, 4, 4)
    assert torch.allclose(sample["motion_vectors"], torch.full((2, 2, 4, 4), 0.5))
    assert torch.allclose(sample["residuals"], torch.ones(2, 3, 4, 4))
    assert (0, 11, 1, False) in calls
    assert (0, 11, 2, True) in calls
    assert (1, 1, 1, False) in calls
    assert (1, 1, 2, True) in calls


def test_coviar_eval_stacks_all_views_under_one_video(tmp_path, monkeypatch):
    dataset, _ = _make_fake_dataset(
        tmp_path,
        monkeypatch,
        num_frames=48,
        num_gops=4,
    )
    dataset.candidate_frames = 2
    dataset.num_temporal_views = 2
    dataset.num_spatial_crops = 3

    sample = dataset[0]

    assert sample["i_frames"].shape == (6, 2, 3, 4, 4)
    assert sample["motion_vectors"].shape == (6, 2, 2, 4, 4)
    assert sample["residuals"].shape == (6, 2, 3, 4, 4)
    assert sample["valid_mask"].shape == (6, 2)
    assert sample["candidate_gop_indices"].shape == (6, 2)
    assert sample["label"].item() == 1


def test_coviar_decode_failure_is_contextual_and_never_silently_zero_filled(
    tmp_path, monkeypatch
):
    dataset, _ = _make_fake_dataset(
        tmp_path,
        monkeypatch,
        num_frames=12,
        num_gops=1,
        return_none_for=1,
    )

    with pytest.raises(RuntimeError, match=r"dataset=unit.*gop=0.*returned None"):
        dataset.preflight(0)
