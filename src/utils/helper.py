import os
import torch


def write_checkpoint(
    agent,
    optimizer,
    episode: int,
    dst_path: str,
):
    """
    Writes a checkpoint including agent, optimizer state dictionaries along with current episode statistics
    and ReplayMemory to a file.

    :param agent: The agent and all its state dictionaries
    :param optimizer: The pytorch optimizer
    :param epsiode: Current training epsiode
    :param dst_path: Path where the checkpoint is written to
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save(
        obj={
            "agent_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": episode + 1,
        },
        f=dst_path,
    )
