"""The TOFU baseline scripts (scripts/unlearn/) must compose valid Hydra configs of src/train.py and must not override a
method's own defaults (an unneeded `trainer.method_args.gamma=1.0` silently changed SimNPO / SatImp / RMU before).
`--cfg job` composes the config without loading a model or data."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def compose(method: str, env: str = ""):
    cmd = (
        f'{env} source scripts/unlearn/_common.sh; python src/train.py --config-name=unlearn $(method_args {method}) '
        'model=${FULL_MODEL} "${SPLIT_ARGS[@]}" "${EXTRA_ARGS[@]}" task_name=$(task_name ' + method + ') --cfg job --resolve'
    )
    proc = subprocess.run(["bash", "-c", cmd], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-500:] + proc.stderr[-1500:]
    return proc.stdout


ENV_3B_FORGET05 = "export MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95;"


def test_scripts_are_valid_shell():
    for sh in (ROOT / "scripts" / "unlearn").glob("*.sh"):
        assert subprocess.run(["bash", "-n", str(sh)]).returncode == 0, sh.name


@pytest.mark.parametrize("method", ["SimNPO", "RMU"])
def test_method_defaults_are_not_overridden(method):
    out = compose(method, ENV_3B_FORGET05)
    assert "task_name: tofu_Llama-3.2-3B-Instruct_forget05/" + method in out
    assert "forget_split: forget05" in out and "retain_split: retain95" in out and "holdout_split: holdout05" in out
    assert "saves/eval/tofu_Llama-3.2-3B-Instruct_retain95/TOFU_EVAL.json" in out
    if method == "SimNPO":
        assert "gamma: 0.125" in out  # the authors' value, not the 1.0 of the former scripts
    else:
        assert "retain_loss_type: EMBED_DIFF" in out  # RMU keeps its embedding-difference retain loss


def test_idk_dpo_uses_the_idk_forget_data_and_the_dpo_trainer():
    out = compose("IdkDPO", ENV_3B_FORGET05)
    assert "handler: DPO" in out and "TOFU_balanced_unlearn_idk" in out and "eval_on_start: false" in out.lower()


def test_tuning_overrides_go_into_the_trainer_args_and_the_folder_name():
    out = compose("GradDiff", "export LR=5e-6 EPOCHS=5;")
    assert "learning_rate: 5.0e-06" in out and "num_train_epochs: 5" in out
    assert "_lr5e-6_ep5" in out
