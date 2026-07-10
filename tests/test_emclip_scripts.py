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
