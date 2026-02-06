import pandas as pd
import ale_py
from stable_baselines3 import DQN
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
from wrapper.custom_wrappers import *
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os

experiments = ["1e7CaCFokus", "1e7Cropped", "1e7ScratchBaseline", "1e7ScratchFramestack1", "1e7ScratchFramestack2","1e7ScratchFramestack6",
               "1e7ScratchNoClip", "1e7Standard", "1e7StepPenalty", "1e7FireRemove"]

for experiment in experiments:
    base_path = f"logs/{experiment}/dqn/QbertNoFrameskip-v4_"

    model_versions = range(1, 11)  # v4_1 bis v4_10
    start_seed = 42
    num_seeds = 30

    all_model_scores = []

    def make_env(seed: int):
        def _init():
            env = gym.make("QbertNoFrameskip-v4")
            env = AtariWrapper(env, clip_reward=False, terminal_on_life_loss=False)
            env = Monitor(env)
            env.reset(seed=seed)
            return env
        return _init

    def make_env_crop(seed: int):
        def _init():
            env = gym.make("QbertNoFrameskip-v4")
            env = CropLifeandScoreWrapper(env)
            env = AtariWrapper(env, clip_reward=False, terminal_on_life_loss=False)
            env = Monitor(env)
            env.reset(seed=seed)
            return env
        return _init

    def make_env_fire_remove(seed: int):
        def _init():
            env = gym.make("QbertNoFrameskip-v4")
            env = AtariWrapper(env, clip_reward=False,
                               terminal_on_life_loss=False)
            env = RemoveFireWrapper(env)
            env = Monitor(env)
            env.reset(seed=seed)
            return env

        return _init

    for version in model_versions:
        model_path = f"{base_path}{version}/QbertNoFrameskip-v4.zip"
        if not os.path.exists(model_path):
            print(f"Modell {model_path} nicht gefunden, überspringe...")
            continue

        model = DQN.load(model_path)
        print(f"Evaluating model from experiment {experiment}: v4_{version}")

        all_scores = []
        all_ep_lengths = []
        for i in range(num_seeds):
            seed = start_seed + i
            if experiment == "1e7Cropped":
                env = DummyVecEnv([make_env_crop(seed)])
            elif experiment == "1e7FireRemove":
                env = DummyVecEnv([make_env_fire_remove(seed)])
            else:
                env = DummyVecEnv([make_env(seed)])

            if experiment == "1e7ScratchFramestack6":
                env = VecFrameStack(env, n_stack=6)
            elif experiment == "1e7ScratchFramestack1":
                env = VecFrameStack(env, n_stack=1)
            elif experiment == "1e7ScratchFramestack2":
                env = VecFrameStack(env, n_stack=2)
            else:
                env = VecFrameStack(env, n_stack=4)

            obs = env.reset()
            done = False
            ep_score = 0
            ep_length = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, info = env.step(action)
                done = dones[0]
                ep_score += reward[0]
                ep_length += 1

            all_scores.append(ep_score)
            all_ep_lengths.append(ep_length*4) # multiply by 4 since we work with a frameskip of 4 (Atari Wrapper) to get the same scale as the plots.
            env.close()

        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        print(f"v4_{version}: Mean ALE score over {num_seeds} seeds: {mean_score:.2f} ± {std_score:.2f}")
        mean_ep_length = np.mean(all_ep_lengths)
        std_ep_length = np.std(all_ep_lengths)
        print(f"v4_{version}: Mean ALE ep_length over {num_seeds} seeds: {mean_ep_length:.2f} ± {std_ep_length:.2f}")

        all_model_scores.append((experiment, version, mean_score, std_score, mean_ep_length, std_ep_length))




    #---------------------------- Save Evaluations for every seed ----------------------------
    csv_path = "evaluation_results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["experiment", "version", "mean_score", "std_score", "mean_ep_length", "std_ep_length"])

    # Delete old data
    for exp, ver, _, _, _, _ in all_model_scores:
        df = df[~((df["experiment"] == exp) & (df["version"] == ver))]

    new_rows = pd.DataFrame(all_model_scores, columns=["experiment", "version", "mean_score", "std_score", "mean_ep_length", "std_ep_length"])
    df = pd.concat([df, new_rows], ignore_index=True)

    df.sort_values(by=["experiment", "version"], inplace=True)
    df.to_csv(csv_path, index=False)
    print(df)