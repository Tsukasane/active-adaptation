import torch
import hydra
import numpy as np
import einops
import itertools
from omegaconf import OmegaConf

from omni.isaac.lab.app import AppLauncher
# from omni_drones.utils.wandb import init_wandb
# from omni_drones.utils.torchrl import SyncDataCollector

from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictSequential
from active_adaptation.learning import ALGOS
from collections import OrderedDict

import wandb
import logging
from tqdm import tqdm

import os
import datetime

@hydra.main(config_path="../cfg", config_name="play")
@torch.inference_mode()
@set_exploration_type(ExplorationType.MODE)
def main(cfg):
    cfg.headless = True
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    need_envs = cfg.get("rollout_frame", int(64))
    
    app_launcher = AppLauncher(cfg.app)
    simulation_app = app_launcher.app

    from helpers import EpisodeStats, make_env_policy, ObsNorm, export_onnx
    env, policy, vecnorm = make_env_policy(cfg)

    policy = policy.get_rollout_policy("eval")

    state_keys = [
        "robot",
        "history",
        "long_history",
        "ref_motion_",
        "priv"
    ]
    action_keys = [
        "loc",
        "scale",
        "action"
    ]
    next_keys = ("next", "robot")

    rollout = []
    embedding = []

    td_ = env.reset()
    
    for i in tqdm(range(env.max_episode_length), miniters=10):
        rollout.append(td_.select(*state_keys, strict=False).cpu()) # record state
        td_ = policy(td_)
        rollout[-1].update(td_.select(*action_keys, strict=False).cpu())
        if "mu" in td_.keys():
            embedding.append(td_["mu"].cpu())
        td, td_ = env.step_and_maybe_reset(td_)
        rollout[-1].update(td.select(next_keys, strict=False).cpu())
    
    truncated = td["next"]["truncated"].squeeze(-1).cpu()
    assert truncated.sum() > need_envs, f"Rollout env is not enough: {truncated.sum()}"
    print(f"Truncated envs: {truncated.sum()}, Success rate: {truncated.sum() / truncated.size(0)}")
    rollout = torch.stack(rollout, dim=1)[truncated][:need_envs]
    print(f"Rollout shape: {rollout.shape}")
    print(rollout)
    path = os.path.join(os.path.dirname(__file__), f"rollout-{cfg.task.name}.pt")
    # torch.save(rollout, path)

    if len(embedding) > 0:
        embedding = torch.stack(embedding, dim=1)[truncated][:need_envs]
        print(f"Embedding shape: {embedding.shape}")
        path = os.path.join(os.path.dirname(__file__), f"embedding-{cfg.task.name}.pt")
        torch.save(embedding, path)
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

