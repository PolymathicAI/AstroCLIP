import os
from math import pi

import matplotlib.pyplot as plt
import numpy as np


def plot_radar(outputs: dict, metric: str, file_path: str, fontsize: int = 18):
    """Functionality for plotting radar chart"""
    questions = {}
    for key in outputs.keys():
        questions[key] = [
            outputs[key][question][metric] for question in outputs[key].keys()
        ]
    labels = outputs[key].keys()

    # Add Zoobot scores
    questions["ZooBot Reported"] = [
        zoobot_scores[question][metric] for question in zoobot_scores.keys()
    ]

    # Create radar chart
    angles = np.linspace(0, 2 * pi, len(questions[key]), endpoint=False).tolist()
    angles += angles[:1]  # complete the loop

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    colors = ["red", "red", "black", "blue"]
    styles = ["solid", "dashed", "solid", "solid"]

    # Plot each array on the radar chart
    for key in questions.keys():
        stats = [questions[key][i] for i in range(len(questions[key]))]
        stats += stats[:1]
        ax.plot(
            angles,
            stats,
            label=key,
            linewidth=2,
            linestyle=styles.pop(0),
            color=colors.pop(0),
        )

    # capitalize labels
    labels = [label.capitalize() for label in labels]

    # Add labels with specific fontsize
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Change r label to fontsize
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.set_xticks(angles[:-1], labels, fontsize=fontsize, color="black")

    # make theta labels not overlap with plot
    ax.set_ylim(0, 1.0)

    # Add legend
    legend = plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    plt.setp(
        legend.get_texts(), fontsize=fontsize
    )  # Explicitly set fontsize for legend

    # Save fig
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    plt.savefig(file_path)
    plt.close()


# ZooBot scores taken from Walmsley et al. (2021), https://arxiv.org/pdf/2102.08414
zoobot_scores = {
    "smooth": {"Accuracy": 0.94, "F1 Score": 0.94},
    "disk-edge-on": {"Accuracy": 0.99, "F1 Score": 0.99},
    "spiral-arms": {"Accuracy": 0.93, "F1 Score": 0.94},
    "bar": {"Accuracy": 0.82, "F1 Score": 0.81},
    "bulge-size": {"Accuracy": 0.84, "F1 Score": 0.84},
    "how-rounded": {"Accuracy": 0.93, "F1 Score": 0.93},
    "edge-on-bulge": {"Accuracy": 0.91, "F1 Score": 0.90},
    "spiral-winding": {"Accuracy": 0.78, "F1 Score": 0.79},
    "spiral-arm-count": {"Accuracy": 0.77, "F1 Score": 0.76},
    "merging": {"Accuracy": 0.88, "F1 Score": 0.85},
}
