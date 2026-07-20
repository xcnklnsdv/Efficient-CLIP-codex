import io
import os
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

import main_emclip
from configs import DATASETS


REPO_ROOT = Path(__file__).resolve().parents[1]
K400_LABEL_CSV = REPO_ROOT / "configs" / "kinetics_400_labels.csv"


def _args(class_names=None, label_csv=None, dataset="k400"):
    return SimpleNamespace(class_names=class_names, label_csv=label_csv, dataset=dataset)


def _distributed_class_log_worker(rank, world_size, init_method, output_queue):
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        output = io.StringIO()
        with redirect_stdout(output):
            names = main_emclip.load_class_names(_args(), 400, cfg=DATASETS["k400"])
        output_queue.put((rank, len(names), output.getvalue()))
    finally:
        dist.destroy_process_group()


def test_repository_k400_mapping_has_400_ordered_semantic_classes(capsys):
    names = main_emclip.load_class_names(_args(), 400, cfg=DATASETS["k400"])

    assert len(names) == 400
    assert names[:5] == [
        "abseiling",
        "air drumming",
        "answering questions",
        "applauding",
        "applying cream",
    ]
    assert names[-5:] == ["wrestling", "writing", "yawning", "yoga", "zumba"]
    assert len(set(names)) == 400
    assert not any(name.isdecimal() for name in names)
    output = capsys.readouterr().out
    assert str(K400_LABEL_CSV.resolve()) in output
    assert "class names loaded=400" in output
    assert "class names first5=" in output
    assert "class names last5=" in output


@pytest.mark.parametrize(
    "rows",
    (
        "id,name\n0,playing_guitar\n1,walking_the_dog\n",
        "name,id\nplaying_guitar,0\nwalking_the_dog,1\n",
    ),
)
def test_k400_csv_accepts_both_id_name_orders_and_normalizes_underscores(tmp_path, rows):
    mapping = tmp_path / "labels.csv"
    mapping.write_text(rows, encoding="utf-8")

    names = main_emclip.load_class_names(
        _args(label_csv=str(mapping)),
        2,
        cfg={"NUM_CLASSES": 2},
    )

    assert names == ["playing guitar", "walking the dog"]


@pytest.mark.parametrize(
    ("rows", "message"),
    (
        ("id,name\n0,100\n1,200\n", "purely numeric"),
        ("id,name\n0,playing guitar\n1,playing guitar\n", "duplicate class texts"),
        ("id,name\n0,playing guitar\n0,walking dog\n", "duplicate class index"),
    ),
)
def test_k400_mapping_rejects_numeric_names_and_duplicates(tmp_path, rows, message):
    mapping = tmp_path / "labels.csv"
    mapping.write_text(rows, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        main_emclip.load_class_names(
            _args(label_csv=str(mapping)),
            2,
            cfg={"NUM_CLASSES": 2},
        )


def test_config_class_names_file_is_used_before_list_inference(tmp_path):
    mapping = tmp_path / "classes.txt"
    mapping.write_text("configured first\nconfigured second\n", encoding="utf-8")
    video_list = tmp_path / "train.txt"
    video_list.write_text("inferred_a/a.mp4 1 0\ninferred_b/b.mp4 1 1\n", encoding="utf-8")
    cfg = {
        "CLASS_NAMES_FILE": str(mapping),
        "TRAIN_LIST": str(video_list),
        "VAL_LIST": str(video_list),
    }

    assert main_emclip.load_class_names(_args(dataset="unit"), 2, cfg=cfg) == [
        "configured first",
        "configured second",
    ]


def test_nonzero_rank_does_not_print_class_name_log(monkeypatch, capsys):
    monkeypatch.setattr(main_emclip, "is_main_process", lambda: False)

    names = main_emclip.load_class_names(_args(), 400, cfg=DATASETS["k400"])

    assert len(names) == 400
    assert capsys.readouterr().out == ""


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_two_distributed_ranks_only_log_class_names_on_rank_zero(tmp_path):
    rendezvous_path = (tmp_path / "class_names_rendezvous").resolve().as_posix()
    prefix = "file:///" if os.name == "nt" else "file://"
    context = mp.get_context("spawn")
    output_queue = context.SimpleQueue()

    mp.spawn(
        _distributed_class_log_worker,
        args=(2, prefix + rendezvous_path, output_queue),
        nprocs=2,
        join=True,
    )

    results = {rank: (count, output) for rank, count, output in (output_queue.get() for _ in range(2))}
    assert results[0][0] == results[1][0] == 400
    assert "class names loaded=400" in results[0][1]
    assert results[1][1] == ""
