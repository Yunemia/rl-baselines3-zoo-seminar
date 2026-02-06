import subprocess
import sys

python_executable = sys.executable

seeds = range(1, 11)

for seed in seeds:
    print(f"Running seed {seed} ...")
    subprocess.run([
        python_executable,
        "../seminar/utils/train_custom.py",
        "--env", "QbertNoFrameskip-v4",
        "--algo", "dqn",
        "--conf", "../seminar/experiments/def_dqn_qbert.yml",
        "--seed", str(seed),
        "--device", "cuda",
        "--tensorboard-log", "../logs/",
    ])
