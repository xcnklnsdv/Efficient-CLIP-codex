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
