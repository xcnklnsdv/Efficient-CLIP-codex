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


def test_default_k400_disables_unused_parameter_detection():
    text = (REPO_ROOT / "scripts" / "train_emclip_k400.sh").read_text(encoding="utf-8")
    assert "FIND_UNUSED_PARAMETERS=${FIND_UNUSED_PARAMETERS:-false}" in text


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


# Validate the commands that the shell actually produces. This catches
# overwritten caller GPU settings and unintended source checkpoints.
import os
import shlex
import shutil
import subprocess

import pytest


def _run_launcher(tmp_path, script, **overrides):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")
    checkpoint = tmp_path / "weights with spaces.pth"
    checkpoint.touch()
    env = dict(os.environ)
    for name in ("CUDA_VISIBLE_DEVICES", "NPROC_PER_NODE", "RESUME", "INIT_CHECKPOINT", "K400_CHECKPOINT",
                 "VARIANT", "T", "K", "EMCLIP_SCRIPT_GPU_IDS", "MGSE_TRAIN_TEXT_MODE"):
        env.pop(name, None)
    env.update({
        "EMCLIP_PRINT_CMD_ONLY": "1", "CLIP_CHECKPOINT": checkpoint.as_posix(),
        "OUTPUT_ROOT": (tmp_path / "outputs").as_posix(), "NPROC_PER_NODE": "2",
        "CUDA_VISIBLE_DEVICES": "5,7", "COVIAR_DATA_LOADER_DIR": tmp_path.as_posix(),
    })
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    result = subprocess.run([bash, (REPO_ROOT / "scripts" / script).as_posix()],
                            cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    return result, checkpoint


def _value(command, flag):
    return command[command.index(flag) + 1]


@pytest.mark.parametrize("dataset", ["ssv2", "hmdb51", "ucf101", "k400"])
def test_training_command_preserves_gpu_environment_and_starts_from_clip(tmp_path, dataset):
    result, checkpoint = _run_launcher(tmp_path, f"train_emclip_{dataset}.sh")
    assert result.returncode == 0, result.stderr
    assert "CUDA_VISIBLE_DEVICES=5,7 processes=2" in result.stdout
    command = shlex.split(result.stdout.splitlines()[-1])
    assert "--nproc_per_node=2" in command
    assert _value(command, "--batch-size") == "4"
    assert _value(command, "--clip-checkpoint") == checkpoint.as_posix()
    assert _value(command, "--epochs") == "30"
    assert _value(command, "--lr") == "8e-6"
    assert _value(command, "--input-size") == "256"
    assert "--init-checkpoint" not in command
    assert "--resume" not in command


@pytest.mark.parametrize("dataset", ["ssv2", "hmdb51", "ucf101", "k400"])
def test_evaluation_requires_target_checkpoint_and_defaults_to_4x3(tmp_path, dataset):
    missing, checkpoint = _run_launcher(tmp_path, f"eval_emclip_{dataset}.sh")
    assert missing.returncode != 0
    assert "Evaluation requires RESUME" in missing.stderr
    result, _ = _run_launcher(tmp_path, f"eval_emclip_{dataset}.sh", RESUME=checkpoint.as_posix())
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    assert _value(command, "--resume") == checkpoint.as_posix()
    assert _value(command, "--test-num-temporal-views") == "4"
    assert _value(command, "--test-num-spatial-crops") == "3"
    assert "--clip-checkpoint" not in command
    assert "--init-checkpoint" not in command


@pytest.mark.parametrize("variant,k,t", [("diamond", "8", "8"), ("diamond", "16", "16"), ("emclip", "16", "32")])
def test_launch_sampling_matches_variant_and_selected_frames(tmp_path, variant, k, t):
    result, _ = _run_launcher(tmp_path, "train_emclip_ssv2.sh", VARIANT=variant, K=k)
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    assert _value(command, "--candidate-frames") == t
    assert _value(command, "--selected-frames") == k


def test_k400_transfer_remains_explicit_model_only_initialization(tmp_path):
    checkpoint = tmp_path / "explicit_k400.pth"
    checkpoint.touch()
    result, _ = _run_launcher(tmp_path, "train_emclip_ssv2.sh", INIT_CHECKPOINT=checkpoint.as_posix())
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    assert _value(command, "--init-checkpoint") == checkpoint.as_posix()
    assert "--resume" not in command


def test_gpu_helper_rejects_more_processes_than_visible_devices(tmp_path):
    result, _ = _run_launcher(tmp_path, "train_emclip_ssv2.sh", NPROC_PER_NODE="3")
    assert result.returncode != 0
    assert "exceeds CUDA_VISIBLE_DEVICES" in result.stderr


@pytest.mark.parametrize("dataset", ["ucf101", "k400"])
def test_launcher_executes_with_default_gpu_visibility_using_mock_torchrun(tmp_path, dataset):
    # Exercise the post-command-print shell path without launching a training job.
    bash_env = tmp_path / "mock_torchrun.sh"
    bash_env.write_text('torchrun() { printf "MOCK_TORCHRUN\\n"; }\n', encoding="utf-8")
    result, _ = _run_launcher(
        tmp_path, f"train_emclip_{dataset}.sh", EMCLIP_PRINT_CMD_ONLY="0",
        CUDA_VISIBLE_DEVICES=None, NPROC_PER_NODE="4", BASH_ENV=bash_env.as_posix(),
    )
    assert result.returncode == 0, result.stderr
    assert "MOCK_TORCHRUN" in result.stdout
    assert "<caller default>" in result.stdout
