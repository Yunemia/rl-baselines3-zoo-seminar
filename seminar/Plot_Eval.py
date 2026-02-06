import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dark_mode = True

df = pd.read_csv("evaluation_overall.csv")

name_map = {
    '1e7Cropped': "Cropped Image",
    '1e7ScratchBaseline': "Baseline",
    '1e7CaCFokus': "Completion Focus",
    '1e7ScratchFramestack1': "Framestack 1",
    '1e7ScratchFramestack2': "Framestack 2",
    '1e7ScratchFramestack6': "Framestack 6",
    '1e7ScratchNoClip': "Raw Reward",
    '1e7Standard': "Reward Scaling",
    '1e7StepPenalty': "Step Penalty",
    '1e7FireRemove': "Removed Fire"
}

df['experiment_label'] = df['experiment'].map(name_map)

order = [
    "Baseline",
    "Removed Fire",
    "Framestack 1",
    "Framestack 2",
    "Framestack 6",
    "Cropped Image",
    "Raw Reward",
    "Reward Scaling",
    "Step Penalty",
    "Completion Focus",
]

df['experiment_label'] = pd.Categorical(df['experiment_label'], categories=order, ordered=True)
df = df.sort_values('experiment_label')

experiments = df['experiment_label'].tolist()
mean_score = df['overall_mean_score'].values
sd_score = df['overall_std_score'].values
mean_ep_length = df['overall_mean_ep_length'].values
sd_ep_length = df['overall_std_ep_length'].values

x = np.arange(len(experiments))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12,6))

bars1 = ax1.bar(x - width/2, mean_score, width, yerr=sd_score, capsize=5, label='Mean Score', color='#9B59B6', error_kw=dict(ecolor='white'))

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, mean_ep_length, width, yerr=sd_ep_length, capsize=5, label='Mean Episode Length', color='#F5B041', error_kw=dict(ecolor='white'))

if dark_mode:
    fig.patch.set_facecolor("black")
    ax1.set_facecolor("black")
    ax2.set_facecolor("black")

    ax1.tick_params(colors="white")
    ax2.tick_params(colors="white")
    ax1.xaxis.label.set_color("white")
    ax1.yaxis.label.set_color("white")
    ax2.yaxis.label.set_color("white")
    ax1.title.set_color("white")
    for spine in ax1.spines.values():
        spine.set_color("white")
    for spine in ax2.spines.values():
        spine.set_color("white")

    leg1 = ax1.legend(loc='upper left')
    leg2 = ax2.legend(loc='upper right')
    for leg in [leg1, leg2]:
        for text in leg.get_texts():
            text.set_color("white")
        leg.get_frame().set_facecolor("grey")
        leg.get_frame().set_edgecolor("white")
        leg.get_frame().set_alpha(0.9)

ax1.set_xticks(x)
ax1.set_xticklabels(experiments, rotation=45, ha='right')
ax1.set_ylabel('Mean Score')
ax2.set_ylabel('Mean Episode Length')
ax1.set_title('Mean Performance of the Final Agents')

plt.tight_layout()
plt.savefig("plots/Eval_plot.png", dpi=300, bbox_inches='tight')
plt.show()
