import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import glob

experiment = {
    "path": "logs/1e7ScratchFramestack6/dqn/QbertNoFrameskip-v4_*/evaluations.npz",
}

metric = "score"
smoothing_window = 20
dark_mode = True
plot_name = "All_Seeds_Framestack6"

def smooth(x, window):
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    y = np.convolve(x, kernel, mode="same")
    y[-window//2:] = np.nan
    return y


paths = sorted(glob.glob(experiment["path"]))
if len(paths) == 0:
    raise RuntimeError("Keine Evaluationsdateien gefunden")

n_seeds = len(paths)
assert n_seeds == 10, f"Erwartet 10 Seeds, gefunden: {n_seeds}"

cmap = plt.get_cmap("tab10")

plt.figure(figsize=(10, 5))

for i, path in enumerate(paths):
    data = np.load(path, allow_pickle=True)

    timesteps = np.array(data["timesteps"])

    if metric == "score":
        results = np.array(data["results"])
    elif metric == "ep_length":
        results = np.array(data["ep_lengths"])
    else:
        raise ValueError("Ungültige Metrik")

    mean_per_eval = np.mean(results, axis=1)

    mean_per_eval = smooth(mean_per_eval, smoothing_window)

    plt.plot(
        timesteps,
        mean_per_eval,
        color=cmap(i),
        alpha=0.6,
        linewidth=1.5,
        label=f"Seed {i+1}"
    )

plt.xlim(0, 1e7)
plt.xlabel("Steps")

if metric == "score":
    plt.ylabel("Score")
    plt.title(f"Scores")
else:
    plt.ylabel("Episode Length")
    plt.title(f"Episode Lengths")

plt.grid(True)

ax = plt.gca()
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))


fig = plt.gcf()
if dark_mode:
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    for spine in ax.spines.values():
        spine.set_color("white")

    leg = plt.legend(fontsize=8, ncol=2)
    for text in leg.get_texts():
        text.set_color("white")
    leg.get_frame().set_facecolor("grey")
    leg.get_frame().set_edgecolor("white")
else:
    plt.legend(fontsize=8, ncol=2)


color_plot = "dark" if dark_mode else "white"
plt.savefig(
    f"plots/{plot_name}_{metric}_{color_plot}.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

plt.show()
