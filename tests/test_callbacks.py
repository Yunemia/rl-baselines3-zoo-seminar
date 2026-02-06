import shlex
import subprocess

import pytest
import stable_baselines3 as sb3

from rl_zoo3.utils import get_callback_list
from pathlib import Path
import sys
train_script = Path(__file__).parent.parent / "train.py"
train_script = train_script.resolve()  # absoluter Pfad

def _assert_eq(left, right):
    assert left == right, f"{left} != {right}"


def test_raw_stat_callback(tmp_path):
    cmd = (
        f'"{sys.executable}" {train_script} -n 200 --algo ppo --env CartPole-v1 --log-folder {tmp_path} '
        f"--tensorboard-log {tmp_path} -params callback:\"'rl_zoo3.callbacks.RawStatisticsCallback'\""
    )
    return_code = subprocess.call(cmd, shell=True)
    _assert_eq(return_code, 0)


@pytest.mark.parametrize(
    "callback",
    [
        None,
        "rl_zoo3.callbacks.RawStatisticsCallback",
        [
            {"stable_baselines3.common.callbacks.StopTrainingOnMaxEpisodes": dict(max_episodes=3)},
            "rl_zoo3.callbacks.RawStatisticsCallback",
        ],
        [sb3.common.callbacks.StopTrainingOnMaxEpisodes(3)],
    ],
)
def test_get_callback(callback):
    hyperparams = {"callback": callback}
    callback_list = get_callback_list(hyperparams)
    if callback is None:
        assert len(callback_list) == 0
    elif isinstance(callback, str):
        assert len(callback_list) == 1
    else:
        assert len(callback_list) == len(callback)
