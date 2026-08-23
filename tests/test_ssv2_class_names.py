from pathlib import Path
from types import SimpleNamespace

import main_emclip
from configs import DATASETS


REPO_ROOT = Path(__file__).resolve().parents[1]
SSV2_LABEL_CSV = REPO_ROOT / "configs" / "something_v2_labels.csv"


def _args(dataset):
    return SimpleNamespace(class_names=None, label_csv=None, dataset=dataset)


def test_repository_ssv2_mapping_has_174_ordered_semantic_classes(capsys):
    names = main_emclip.load_class_names(
        _args("ssv2_mpeg4"),
        174,
        cfg=DATASETS["ssv2_mpeg4"],
    )

    assert len(names) == 174
    assert names[:5] == [
        "Approaching something with your camera",
        "Attaching something to something",
        "Bending something so that it deforms",
        "Bending something until it breaks",
        "Burying something in something",
    ]
    assert names[67] == "Pretending to open something without actually opening it"
    assert names[140] == "Spinning something that quickly stops spinning"
    assert names[-5:] == [
        "Twisting (wringing) something wet until water comes out",
        "Twisting something",
        "Uncovering something",
        "Unfolding something",
        "Wiping something off of something",
    ]
    assert len(set(names)) == 174
    assert not any(name.isdecimal() for name in names)
    output = capsys.readouterr().out
    assert str(SSV2_LABEL_CSV.resolve()) in output
    assert "class names loaded=174" in output


def test_both_ssv2_configs_use_the_repository_mapping():
    expected = str(SSV2_LABEL_CSV)

    assert DATASETS["ssv2"]["LABEL_CSV"] == expected
    assert DATASETS["ssv2_mpeg4"]["LABEL_CSV"] == expected
