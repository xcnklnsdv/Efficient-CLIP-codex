from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_launch_scripts_use_repo_root_entrypoint():
    scripts = sorted((REPO_ROOT / "scripts").glob("*emclip*.sh"))
    entrypoint_scripts = [
        script for script in scripts if "main_emclip.py" in script.read_text(encoding="utf-8")
    ]

    assert entrypoint_scripts, "expected at least one EM-CLIP launch script"
    for script in entrypoint_scripts:
        text = script.read_text(encoding="utf-8")
        assert 'REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"' in text

        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            assert "main_emclip.py" not in line or '"${REPO_ROOT}/main_emclip.py"' in line, (
                f"{script}:{lineno} uses main_emclip.py without anchoring it to REPO_ROOT"
            )


def test_main_emclip_records_torchrun_child_tracebacks():
    import main_emclip

    assert hasattr(main_emclip.main, "__wrapped__")


def test_k400_launch_script_writes_torchrun_rank_logs():
    text = (REPO_ROOT / "scripts" / "train_emclip_k400.sh").read_text(encoding="utf-8")

    assert "--log-dir" in text
    assert "--redirects" in text
    assert "--tee" in text


def test_k400_launch_scripts_explicitly_pass_repository_label_csv():
    for filename in ("train_emclip_k400.sh", "eval_emclip_k400.sh"):
        text = (REPO_ROOT / "scripts" / filename).read_text(encoding="utf-8")
        assert 'LABEL_CSV="${LABEL_CSV:-${REPO_ROOT}/configs/kinetics_400_labels.csv}"' in text
        assert 'CMD+=(--label-csv "${LABEL_CSV}")' in text


def test_ssv2_launch_scripts_explicitly_pass_repository_label_csv():
    for filename in ("train_emclip_ssv2.sh", "eval_emclip_ssv2.sh"):
        text = (REPO_ROOT / "scripts" / filename).read_text(encoding="utf-8")
        assert 'LABEL_CSV="${LABEL_CSV:-${REPO_ROOT}/configs/something_v2_labels.csv}"' in text
        assert 'CMD+=(--label-csv "${LABEL_CSV}")' in text


def test_scripts_require_explicit_gpu_ids_and_disable_unused_detection():
    gpu_text = (REPO_ROOT / "scripts" / "_gpu_env.sh").read_text(encoding="utf-8")
    k400_text = (REPO_ROOT / "scripts" / "train_emclip_k400.sh").read_text(encoding="utf-8")

    assert 'if [[ -z "${EMCLIP_SCRIPT_GPU_IDS+x}" ]]' in gpu_text
    assert 'SELECTED_GPU_IDS="${EMCLIP_SCRIPT_GPU_IDS}"' in gpu_text
    assert 'export CUDA_VISIBLE_DEVICES="${SELECTED_GPU_IDS}"' in gpu_text
    assert 'NPROC_PER_NODE="${#_EMCLIP_GPU_ID_ARRAY[@]}"' in gpu_text
    assert 'EMCLIP_SCRIPT_GPU_IDS="0,1,2"' in k400_text
    assert "FIND_UNUSED_PARAMETERS=${FIND_UNUSED_PARAMETERS:-false}" in k400_text


def test_training_scripts_enable_optional_compute_profile():
    for filename in (
        "train_emclip_hmdb51.sh",
        "train_emclip_ucf101.sh",
        "train_emclip_k400.sh",
        "train_emclip_ssv2.sh",
    ):
        text = (REPO_ROOT / "scripts" / filename).read_text(encoding="utf-8")
        assert "PROFILE_COMPUTE=${PROFILE_COMPUTE:-1}" in text
        assert "CMD+=(--profile-compute)" in text


def test_target_training_scripts_use_k400_as_model_initialization_not_resume():
    for filename in (
        "train_emclip_hmdb51.sh",
        "train_emclip_ucf101.sh",
        "train_emclip_ssv2.sh",
    ):
        text = (REPO_ROOT / "scripts" / filename).read_text(encoding="utf-8")
        assert 'INIT_CHECKPOINT="${INIT_CHECKPOINT-${K400_CHECKPOINT}}"' in text
        assert 'RESUME="${RESUME:-}"' in text
        assert 'CMD+=(--init-checkpoint "${INIT_CHECKPOINT}")' in text
        assert 'RESUME="${RESUME:-${K400_CHECKPOINT}}"' not in text


def test_ssv2_eval_supports_k400_transfer_without_misusing_resume():
    text = (REPO_ROOT / "scripts" / "eval_emclip_ssv2.sh").read_text(encoding="utf-8")

    assert 'EMCLIP_SCRIPT_GPU_IDS="0,1,2"' in text
    assert 'INIT_CHECKPOINT="${INIT_CHECKPOINT-${K400_CHECKPOINT}}"' in text
    assert 'RESUME="${RESUME:-}"' in text
    assert 'CMD+=(--resume "${RESUME}")' in text
    assert 'CMD+=(--init-checkpoint "${INIT_CHECKPOINT}")' in text
    assert text.index('CMD+=(--resume "${RESUME}")') < text.index(
        'CMD+=(--init-checkpoint "${INIT_CHECKPOINT}")'
    )


def test_ssv2_training_declares_gpu_ids_before_loading_gpu_environment():
    text = (REPO_ROOT / "scripts" / "train_emclip_ssv2.sh").read_text(encoding="utf-8")

    gpu_ids = text.index('EMCLIP_SCRIPT_GPU_IDS="0,1,2"')
    source = text.index('source "${SCRIPT_DIR}/_gpu_env.sh"')
    assert gpu_ids < source
