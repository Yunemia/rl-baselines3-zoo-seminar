import numpy as np
import pandas as pd
import os

experiments = ["1e7CaCFokus", "1e7Cropped", "1e7ScratchBaseline",
               "1e7ScratchFramestack1", "1e7ScratchFramestack2",
               "1e7ScratchFramestack6", "1e7ScratchNoClip",
               "1e7Standard", "1e7StepPenalty", "1e7FireRemove"]

for experiment in experiments:
  df = pd.read_csv("evaluation_results.csv")

  overall_stats = df.groupby("experiment").agg(
      overall_mean_score=("mean_score", "mean"),
      overall_std_score=("mean_score", "std"),
      overall_mean_ep_length=("mean_ep_length", "mean"),
      overall_std_ep_length=("mean_ep_length", "std")
  ).reset_index()

  current_stats = overall_stats[overall_stats["experiment"] == experiment].iloc[0]

  print(f"\n{experiment}:")
  print(f"Mean score: {current_stats['overall_mean_score']:.2f}, std: {current_stats['overall_std_score']:.2f}")
  print(f"Mean ep_length: {current_stats['overall_mean_ep_length']:.2f}, std: {current_stats['overall_std_ep_length']:.2f}")

  new_overall = pd.DataFrame([{
    "experiment": experiment,
    "overall_mean_score": current_stats["overall_mean_score"],
    "overall_std_score": current_stats["overall_std_score"],
    "overall_mean_ep_length": current_stats["overall_mean_ep_length"],
    "overall_std_ep_length": current_stats["overall_std_ep_length"]
  }])

  overall_csv_path = "evaluation_overall.csv"

  if os.path.exists(overall_csv_path):
    overall_df = pd.read_csv(overall_csv_path)
    # Delete old data
    overall_df = overall_df[overall_df["experiment"] != experiment]
    overall_df = pd.concat([overall_df, new_overall], ignore_index=True)
  else:
    overall_df = new_overall

  overall_df.to_csv(overall_csv_path, index=False)
  print(f"Updated overall CSV saved to {overall_csv_path}")
  print(overall_df)