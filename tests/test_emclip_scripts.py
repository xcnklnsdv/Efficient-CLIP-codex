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


def test_scripts_centralize_default_gpu_ids_and_disable_unused_detection():
    gpu_text = (REPO_ROOT / "scripts" / "_gpu_env.sh").read_text(encoding="utf-8")
    k400_text = (REPO_ROOT / "scripts" / "train_emclip_k400.sh").read_text(encoding="utf-8")

    assert 'EMCLIP_DEFAULT_GPU_IDS="0,1,2,3"' in gpu_text
    assert 'GPU_IDS="${EMCLIP_DEFAULT_GPU_IDS}"' in gpu_text
    assert 'export CUDA_VISIBLE_DEVICES="${GPU_IDS}"' in gpu_text
    assert 'NPROC_PER_NODE="${#_EMCLIP_GPU_ID_ARRAY[@]}"' in gpu_text
    assert "FIND_UNUSED_PARAMETERS=${FIND_UNUSED_PARAMETERS:-false}" in k400_text
