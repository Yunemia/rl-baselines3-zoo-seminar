import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import glob


# Framestack comparison
# plot_name = "FramestackVergleich"
# experiments = [
#     ("logs/1e7ScratchFramestack1/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Framestack 1", "yellow"),
#     ("logs/1e7ScratchFramestack2/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Framestack 2", "cyan"),
#     ("logs/1e7ScratchBaseline/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Framestack 4", "red"),
#     ("logs/1e7ScratchFramestack6/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Framestack 6", "green"),
# ]

# Raw reward
# plot_name = "Clipping"
# experiments = [
#     ("logs/1e7ScratchBaseline/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Clipped Reward", "red"),
#     ("logs/1e7ScratchNoClip/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Raw Reward", "green")
# ]


# Cropped Image
plot_name = "Crop"
experiments = [
    ("logs/1e7ScratchBaseline/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Baseline", "red"),
    ("logs/1e7Cropped/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Cropped Image", "green")
]


# Step Penalty
# plot_name = "StepPenalty"
# experiments = [
#     ("logs/1e7ScratchBaseline/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Baseline", "red"),
#     ("logs/1e7StepPenalty/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Step Penalty", "green")
# ]

# Completion Fokus
# plot_name = "CaCReward"
# experiments = [
#     ("logs/1e7ScratchBaseline/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Baseline", "red"),
#     ("logs/1e7CaCFokus/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Completion Focus", "green")
# ]

# Reward Normalisierung
# plot_name = "RewardStandardisierung"
# experiments = [
#     ("logs/1e7ScratchBaseline/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Baseline", "red"),
#     ("logs/1e7Standard/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Reward Scaling", "green")
# ]


# NoFire
# plot_name = "NoFire"
# experiments = [
#     ("logs/1e7ScratchBaseline/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Baseline", "red"),
#     ("logs/1e7FireRemove/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Removed Fire", "green")
# ]

# Baseline
# plot_name ="Baseline"
# experiments = [
#     ("logs/1e7ScratchBaseline/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Baseline", "red"),
# ]

# All in one
# plot_name = "AllinOne"
# experiments = [
#     ("logs/1e7ScratchBaseline/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Baseline", "red"),
#     ("logs/1e7ScratchFramestack1/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Framestack 1", "yellow"),
#     ("logs/1e7ScratchFramestack2/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Framestack 2", "cyan"),
#     ("logs/1e7ScratchFramestack6/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Framestack 6", "green"),
#     ("logs/1e7Cropped/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Cropped Image", "#3498DB"),
#     ("logs/1e7ScratchNoClip/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Raw Reward", "#9B59B6"),
#     ("logs/1e7Standard/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Reward Scaling", "#8B4513"),
#     ("logs/1e7StepPenalty/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Step Penalty", "orange"),
#     ("logs/1e7CaCFokus/dqn/QbertNoFrameskip-v4_*/evaluations.npz", "Completion Focus", "#95A5A6"),
#
# ]

dark_mode = True
#metric = "ep_length"
metric = "score"


smoothing_window = 20

plt.figure(figsize=(10,5))

for path_pattern, label, color in experiments:
    paths = glob.glob(path_pattern)
    if len(paths) == 0:
        print(f"Keine Evaluationsdateien gefunden für {label}")
        continue

    merged_timesteps, merged_results = [], []

    for path in paths:
        data = np.load(path, allow_pickle=True)
        timesteps = np.array(data["timesteps"])
        if metric == "score":
            results = np.array(data["results"])
        elif metric == "ep_length":
            results = np.array(data["ep_lengths"])
        else:
          raise ValueError("keine gültige Metrik")

        merged_timesteps.append(timesteps)
        merged_results.append(results)

    min_len = min(len(r) for r in merged_results)
    merged_timesteps = np.array([t[:min_len] for t in merged_timesteps])
    merged_results = np.array([r[:min_len] for r in merged_results])

    n_trials = len(merged_results)
    n_eval = min_len
    n_eval_episodes = min(len(r[0]) for r in merged_results)

    evaluations = np.array([r[:, :n_eval_episodes] for r in merged_results])
    evaluations = np.swapaxes(evaluations, 0, 1)

    mean_per_eval = np.mean(evaluations, axis=-1)  # n_eval x n_trials

    mean_of_means = np.mean(mean_per_eval, axis=1)  # n_eval

    se = np.std(mean_per_eval, axis=1) / np.sqrt(n_trials)

    if smoothing_window > 1:
      kernel = np.ones(smoothing_window) / smoothing_window
      mean_of_means = np.convolve(mean_of_means, kernel, mode='same')
      se = np.convolve(se, kernel, mode='same')

      # Cut off the end since the last results are distorted
      cut_off = int(smoothing_window/2)
      mean_of_means[-cut_off:] = np.nan
      se[-cut_off:] = np.nan


    mean_timesteps = np.mean(merged_timesteps, axis=0)

    plt.plot(mean_timesteps, mean_of_means, color=color, linewidth=2, label=label)
    plt.fill_between(mean_timesteps, mean_of_means - se, mean_of_means + se, color=color, alpha=0.2)

plt.xlim(0, 1e7)
if metric == "score":
    plt.ylabel("Mean Episode Score")
elif metric == "ep_length":
    plt.ylabel("Mean Episode Length")
plt.xlabel("Steps")
if metric == "score":
    plt.title("Scores Across Seeds")
elif metric == "ep_length":
    plt.title("Episode Lengths Across Seeds")
plt.legend()
plt.grid(True)
fig = plt.gcf()
if dark_mode:
    color_plot = "dark"
    ax = plt.gca()

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines['bottom'].set_color("white")
    ax.spines['top'].set_color("white")
    ax.spines['left'].set_color("white")
    ax.spines['right'].set_color("white")

    if plot_name == "AllinOne":
        leg = plt.legend(fontsize=8)
    else:
        leg = plt.legend()
    for text in leg.get_texts():
        text.set_color("white")
    leg.get_frame().set_facecolor("grey")

    leg.get_frame().set_edgecolor("white")
    leg.get_frame().set_alpha(0.9)
else:
  color_plot = "white"


ax = plt.gca()
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
plt.savefig(
    f"plots/{plot_name}_{metric}_{color_plot}.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

plt.show()
