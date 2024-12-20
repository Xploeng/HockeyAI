import glob
import json
import os

from dataclasses import dataclass
from typing import Any
import matplotlib.pyplot as plt

from PIL import Image


@dataclass
class EpisodeStatistics:
    episode: int
    rewards: list[float]
    info: dict[str, Any]


def save_json(data: Any, file_path: str) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def save_gif(frames: list[Image.Image], file_path: str, duration: int = 50) -> None:
    frames[0].save(
        file_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


def plot_rewards(agent_out_dir: str, n_episodes: int = 1, show: bool = True) -> None:
    file_path = glob.glob(agent_out_dir + "/episode_statistics/*.json")[0]

    with open(file_path) as f:
        episode_stats = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 3))

    for episode, stats in list(episode_stats.items())[:n_episodes]:
        rewards = stats["rewards"]
        ax.plot(range(len(rewards)), rewards, label=f"Rewards_{episode}")

    ax.set_xlabel("Time")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.tight_layout()

    figure_path = os.path.join(agent_out_dir, "figures", f"rewards_0:{episode}.png")
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path, bbox_inches="tight")

    if show:
        plt.show()
