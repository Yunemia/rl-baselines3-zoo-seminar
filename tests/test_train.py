import os
import shlex
import subprocess
from importlib.metadata import version

import pytest
import sys
from pathlib import Path
from gym.wrappers.time_limit import TimeLimit


train_script = Path(__file__).parent.parent / "train.py"
train_script = train_script.resolve()  # absoluter Pfad

def _assert_eq(left, right):
    assert left == right, f"{left} != {right}"


N_STEPS = 100

ALGOS = ("ppo", "a2c", "dqn")
# 'BreakoutNoFrameskip-v4'
ENV_IDS = ("CartPole-v1",)

experiments = {f"{algo}-{env_id}": (algo, env_id) for algo in ALGOS for env_id in ENV_IDS}
# Test for vecnormalize and frame-stack
experiments["ppo-BipedalWalkerHardcore-v3"] = ("ppo", "BipedalWalkerHardcore-v3")
# Test for SAC
experiments["sac-Pendulum-v1"] = ("sac", "Pendulum-v1")
# for TD3
experiments["td3-Pendulum-v1"] = ("td3", "Pendulum-v1")
# for DDPG
experiments["ddpg-Pendulum-v1"] = ("ddpg", "Pendulum-v1")


@pytest.mark.parametrize("experiment", experiments.keys())
def test_train(tmp_path, experiment):
    algo, env_id = experiments[experiment]

    cmd = f'"{sys.executable}" {train_script} -n {N_STEPS} --algo {algo} --env {env_id} --log-folder {tmp_path}'
    return_code = subprocess.call(cmd, shell=True)
    _assert_eq(return_code, 0)


def test_continue_training(tmp_path):
    algo = "a2c"
    if version("gymnasium") > "0.29.1":
        # See https://github.com/DLR-RM/stable-baselines3/pull/1837#issuecomment-2457322341
        # obs bounds have changed...
        env_id = "CartPole-v1"
    else:
        env_id = "Pendulum-v1"

    cmd = (
        f'"{sys.executable}" {train_script} -n {N_STEPS} --algo {algo} --env {env_id} --log-folder {tmp_path} '
        f"-i rl-trained-agents/a2c/{env_id}_1/{env_id}.zip"
    )
    return_code = subprocess.call(cmd, shell=True)
    _assert_eq(return_code, 0)


def test_save_load_replay_buffer(tmp_path):
    algo, env_id = "sac", "Pendulum-v1"
    cmd = (
        f'"{sys.executable}" {train_script} -n {N_STEPS} --algo {algo} --env {env_id} --log-folder {tmp_path} '
        "--save-replay-buffer -params buffer_size:1000 --env-kwargs g:8.0 --eval-env-kwargs g:5.0 "
    )
    return_code = subprocess.call(cmd, shell=True)
    _assert_eq(return_code, 0)

    assert os.path.isfile(os.path.join(tmp_path, "sac/Pendulum-v1_1/replay_buffer.pkl"))

    saved_model = os.path.join(tmp_path, "sac/Pendulum-v1_1/Pendulum-v1.zip")
    cmd += f"-i {saved_model}"

    return_code = subprocess.call(cmd, shell=True)
    _assert_eq(return_code, 0)


def test_parallel_train(tmp_path):
    cmd = (
        f'"{sys.executable}" {train_script} -n 1000 --algo sac --env Pendulum-v1 --log-folder {tmp_path} '
        "-params monitor_kwargs:'dict(info_keywords=(\"TimeLimit.truncated\",))' "
        "callback:'rl_zoo3.callbacks.ParallelTrainCallback'"
    )
    return_code = subprocess.call(cmd, shell=True)
    _assert_eq(return_code, 0)
    # Fails but idk how to fix.


def test_custom_yaml(tmp_path):
    cmd = [
        sys.executable,
        train_script,
        "-n", "1000",
        "--algo", "sac",
        "--env", "Pendulum-v1",
        "--log-folder", str(tmp_path),
        "-params",
        "monitor_kwargs:dict(info_keywords=('TimeLimit.truncated',))",
        "callback:rl_zoo3.callbacks.ParallelTrainCallback"
    ]

    return_code = subprocess.call(cmd,
                                  shell=False)  # shell=False ist wichtig für Windows
    _assert_eq(return_code, 0)


@pytest.mark.parametrize("config_file", ["hyperparams.python.ppo_config_example", "hyperparams/python/ppo_config_example.py"])
def test_python_config_file(tmp_path, config_file):
    # Use the example python config file for training
    cmd = (
        f'"{sys.executable}" {train_script} -n {N_STEPS} --algo ppo --env MountainCarContinuous-v0 --log-folder {tmp_path} '
        f"-conf {config_file} "
    )
    return_code = subprocess.call(cmd, shell=True)
    _assert_eq(return_code, 0)


def test_gym_packages(tmp_path):
    # Update python path so the test_env package is found
    env_variables = os.environ.copy()
    python_path = env_variables.get("PYTHONPATH", "")
    # Windows benutzt ; als Trenner
    python_path = python_path + ";tests\\dummy_env" if python_path else "tests\\dummy_env"
    env_variables["PYTHONPATH"] = python_path

    # Test gym packages
    cmd = (
        f'"{sys.executable}" {train_script} -n {N_STEPS} --algo ppo --env TestEnv-v0 --log-folder {tmp_path} '
        f"--gym-packages test_env --conf-file test_env.config "
    )
    return_code = subprocess.call(cmd, shell=True, env=env_variables)
    _assert_eq(return_code, 0)

