import glob
import os
import subprocess
from pathlib import Path

import optuna
import pytest
from optuna.trial import TrialState
import sys

train_script = Path(__file__).parent.parent / "train.py"
train_script = train_script.resolve()  # absoluter Pfad
parse_study_script = Path(__file__).parent.parent / "scripts/parse_study.py"
parse_study_script = parse_study_script.resolve()  # absoluter Pfad


def _assert_eq(left, right):
    assert left == right, f"{left} != {right}"


N_STEPS = 100
N_TRIALS = 2
N_JOBS = 1

ALGOS = ("ppo", "a2c")
ENV_IDS = ("CartPole-v1",)

experiments = {}
for algo in ALGOS:
    for env_id in ENV_IDS:
        experiments[f"{algo}-{env_id}"] = (algo, env_id)

# Weitere Experimente
experiments.update({
    "sac-Pendulum-v1": ("sac", "Pendulum-v1"),
    "td3-Pendulum-v1": ("td3", "Pendulum-v1"),
    "tqc-parking-v0": ("tqc", "parking-v0"),
    "tqc-Pendulum-v1": ("tqc", "Pendulum-v1"),
    "ppo_lstm-CartPoleNoVel-v1": ("ppo_lstm", "CartPoleNoVel-v1"),
})


@pytest.mark.parametrize("sampler", ["random", "tpe"])
@pytest.mark.parametrize("pruner", ["none", "halving", "median"])
@pytest.mark.parametrize("experiment", experiments.keys())
def test_optimize(tmp_path, sampler, pruner, experiment):
    algo, env_id = experiments[experiment]

    # Skip slow tests
    if algo not in {"a2c", "ppo"} and not (sampler == "random" and pruner == "median"):
        pytest.skip("Skipping slow tests")

    # Windows-kompatible params
    params = ["-params", "policy_kwargs:dict(net_arch=[32])"]
    if algo == "ppo":
        params += ["-params", "n_steps:10"]

    cmd = [
        sys.executable,
        str(train_script),
        "-n", str(N_STEPS),
        "--algo", algo,
        "--env", env_id,
        "--log-folder", str(tmp_path),
        *params,
        "--no-optim-plots",
        "--seed", "14",
        "--n-trials", str(N_TRIALS),
        "--n-jobs", str(N_JOBS),
        "--sampler", sampler,
        "--pruner", pruner,
        "--n-evaluations", "2",
        "--n-startup-trials", "1",
        "-optimize",
        "--device", "cpu"
    ]

    return_code = subprocess.call(cmd)
    _assert_eq(return_code, 0)


def test_optimize_log_path(tmp_path):
    algo, env_id = "a2c", "CartPole-v1"
    sampler = "random"
    pruner = "median"
    optimization_log_path = tmp_path / "optim_logs"

    params = ["-params", "policy_kwargs:dict(net_arch=[32])"]

    cmd = [
        sys.executable,
        str(train_script),
        "-n", str(N_STEPS),
        "--algo", algo,
        "--env", env_id,
        "--log-folder", str(tmp_path),
        *params,
        "--study-name", "demo",
        "--storage", str(tmp_path / "demo.log"),
        "--no-optim-plots",
        "--seed", "14",
        "--n-trials", str(N_TRIALS),
        "--n-jobs", str(N_JOBS),
        "--sampler", sampler,
        "--pruner", pruner,
        "--n-evaluations", "2",
        "--n-startup-trials", "1",
        "--optimization-log-path", str(optimization_log_path),
        "-optimize",
        "--device", "cpu"
    ]
    return_code = subprocess.call(cmd)
    _assert_eq(return_code, 0)

    assert optimization_log_path.is_dir()
    assert (optimization_log_path / "trial_1").is_dir()
    assert (optimization_log_path / "trial_1" / "evaluations.npz").is_file()

    study_path = next(iter(glob.glob(str(tmp_path / algo / "report_*.pkl"))))
    cmd = [
        sys.executable,
        str(parse_study_script),
        "-i", study_path,
        "--print-n-best-trials", str(N_TRIALS),
        "--save-n-best-hyperparameters", str(N_TRIALS),
        "-f", str(tmp_path / "best_hyperparameters")
    ]
    return_code = subprocess.call(cmd)
    _assert_eq(return_code, 0)

    cmd = [
        sys.executable,
        str(train_script),
        "-n", str(N_STEPS),
        "--algo", algo,
        "--env", env_id,
        "--log-folder", str(tmp_path),
        *params,
        "--storage", str(tmp_path / "demo.log"),
        "--study-name", "demo",
        "--trial-id", "1"
    ]
    return_code = subprocess.call(cmd)
    _assert_eq(return_code, 0)


def test_multiple_workers(tmp_path):
    study_name = "test-study"
    storage = f"sqlite:///{tmp_path}/optuna.db"
    n_trials = 2
    max_trials = 3
    n_workers = 3

    cmd = [
        sys.executable,
        str(train_script),
        "-n", "100",
        "--algo", "a2c",
        "--env", "Pendulum-v1",
        "--log-folder", str(tmp_path),
        "-params", "n_envs:1",
        "--n-evaluations", "1",
        "--no-optim-plots",
        "--seed", "12",
        "--n-trials", str(n_trials),
        "--max-total-trials", str(max_trials),
        "--storage", storage,
        "--study-name", study_name,
        "-optimize",
        "--device", "cpu"
    ]

    workers = []
    for _ in range(n_workers):
        worker = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        worker.wait()
        workers.append(worker)

    study = optuna.load_study(study_name=study_name, storage=storage)
    assert len(study.get_trials(states=(TrialState.COMPLETE, TrialState.PRUNED))) == max_trials

    for worker in workers:
        out, err = worker.communicate()
        assert worker.returncode == 0, f"STDOUT:\n{out}\nSTDERR:\n{err}"
