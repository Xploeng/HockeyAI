import glob
import json
import os

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from PIL import Image


@dataclass
class EpisodeStatistics:
    episode: int
    rewards: list[float]
    states: list[list[float]] # TODO: Better way to save this? Pickle to keep tensor?
    info: dict[str, Any]
    # ? Keep track of actions as well?


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


class PendulumDims(Enum):
    ANGLE = 0
    ANGULAR_VELOCITY = 1


class CartPoleDims(Enum):
    CART_POSITION = 0
    CART_VELOCITY = 1
    POLE_ANGLE = 2
    POLE_ANGULAR_VELOCITY = 3


class Hockey(Enum):
    # TODO: Add hockey dimensions
    pass


@dataclass
class EnvironmentConfig:
    name: str
    input_dims: int
    ranges: dict[int, np.ndarray]
    dimension_labels: dict[int, str]
    preprocess_fn: callable = None  # Optional preprocessing function



def preprocess_pendulum_states(dots: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Convert Pendulum angle to cos/sin and angular velocity."""
    angles = xx.ravel()
    dots[:, 0] = np.cos(angles)  # cos(theta)
    dots[:, 1] = np.sin(angles)  # sin(theta)
    dots[:, 2] = yy.ravel()  # angular velocity
    return dots


def pendulum_config():
    return EnvironmentConfig(
        name="Pendulum-v1",
        input_dims=3,
        ranges={
            PendulumDims.ANGLE.value: np.linspace(-np.pi / 2, np.pi / 2, 50),
            PendulumDims.ANGULAR_VELOCITY.value: np.linspace(-3, 3, 50),
        },
        dimension_labels={
            PendulumDims.ANGLE.value: "Angle",
            PendulumDims.ANGULAR_VELOCITY.value: "Angular Velocity",
        },
        preprocess_fn=preprocess_pendulum_states,
    )


def cartpole_config():
    return EnvironmentConfig(
        name="CartPole-v1",
        input_dims=4,
        ranges={
            CartPoleDims.CART_POSITION.value: np.linspace(-4.8, 4.8, 50),
            CartPoleDims.CART_VELOCITY.value: np.linspace(-1, 1, 50),
            CartPoleDims.POLE_ANGLE.value: np.linspace(-0.418, 0.418, 50),
            CartPoleDims.POLE_ANGULAR_VELOCITY.value: np.linspace(-1, 1, 50),
        },
        dimension_labels={
            CartPoleDims.CART_POSITION.value: "Cart Position",
            CartPoleDims.CART_VELOCITY.value: "Cart Velocity",
            CartPoleDims.POLE_ANGLE.value: "Pole Angle",
            CartPoleDims.POLE_ANGULAR_VELOCITY.value: "Pole Angular Velocity",
        },
    )


def get_env_config(env_name: str) -> EnvironmentConfig:
    match env_name:
        case "Pendulum-v1":
            return pendulum_config()
        case "CartPole-v1":
            return cartpole_config()
        case "Hockey":
            return None # TODO: Add hockey config
        case _:
            raise ValueError(f"Unknown environment name: {env_name}")


def plot_q_function(
    value_function,
    config: EnvironmentConfig,
    plot_dim1: int,
    plot_dim2: int,
):
    """
    Generic function to plot Q-values for any environment.

    Args:
    - value_function: The Q-function to evaluate.
    - config: The environment configuration.
    - plot_dim1: First dimension to plot (index).
    - plot_dim2: Second dimension to plot (index).
    """
    plt.rcParams.update({"font.size": 12})

    if plot_dim1 not in config.ranges or plot_dim2 not in config.ranges:
        raise ValueError("Invalid dimensions for the given environment.")

    label_dim1 = config.dimension_labels.get(plot_dim1, f"Dim {plot_dim1}")
    label_dim2 = config.dimension_labels.get(plot_dim2, f"Dim {plot_dim2}")

    xx, yy = np.meshgrid(config.ranges[plot_dim1], config.ranges[plot_dim2])

    # Prepare input states
    dots = np.zeros((xx.size, config.input_dims))

    if config.preprocess_fn:
        dots = config.preprocess_fn(dots, xx, yy)
    else:
        dots[:, plot_dim1] = xx.ravel()
        dots[:, plot_dim2] = yy.ravel()

    # Evaluate Q-values and reshape
    values = value_function.max_q(dots).reshape(xx.shape)

    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(xx, yy, values, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.view_init(elev=20, azim=45, roll=0)
    ax.set_xlabel(label_dim1)
    ax.set_ylabel(label_dim2)
    ax.set_zlabel("Q-Value")
    plt.colorbar(surf)
    plt.title(f"Q-Value Surface ({label_dim1} vs {label_dim2})")

    return fig


def plot_q_function_all_dims(agent: Any, env_name: str, out_dir: str) -> None:
    """
    Generate Q-value surface plots for all possible dimension combinations.

    Args:
        agent: The agent with a `policy_net` attribute that has a `max_q` method.
        config: The environment configuration.
    """
    config = get_env_config(env_name)
    value_function = agent.policy_net

    dims = list(config.ranges.keys())
    for plot_dim1, plot_dim2 in combinations(dims, 2):
        fig = plot_q_function(
            value_function=value_function,
            config=config,
            plot_dim1=plot_dim1,
            plot_dim2=plot_dim2,
        )

        # Save figure with appropriate naming
        label_dim1 = config.dimension_labels.get(plot_dim1, f"dim_{plot_dim1}")
        label_dim2 = config.dimension_labels.get(plot_dim2, f"dim_{plot_dim2}")
        file_name = f"q_value_surface_{label_dim1}_vs_{label_dim2}.png"
        fig.savefig(os.path.join(out_dir, file_name), bbox_inches="tight")
        plt.close(fig)

    print(f"Value function surface figures saved to {out_dir}")
