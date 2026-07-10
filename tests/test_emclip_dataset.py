import torch

import datasets.compressed_video_dataset as cvd
from datasets.compressed_video_dataset import (
    crop_modalities,
    ensure_coviar_loader,
    horizontal_flip_modalities,
    parse_video_list_line,
    resize_modalities,
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
