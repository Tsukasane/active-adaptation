import torch
import hydra
import numpy as np
import einops
import time
import sys
from tqdm import tqdm
from omegaconf import OmegaConf

from omni.isaac.lab.app import AppLauncher
from torchrl.envs import TransformedEnv, ExplorationType, set_exploration_type

from active_adaptation.learning import ALGOS

import wandb
import logging
from tqdm import tqdm
from helpers import make_env_policy

import os
import datetime
import termcolor

@torch.inference_mode()
def evaluate(
    env,
    policy: torch.nn.Module,
    seed: int=0, 
    exploration_type: ExplorationType=ExplorationType.MODE,
    render=False,
    keys=[("next", "stats")],
):
    """
    Evaluate the policy on the environment, selecting `keys` from the trajectory.
    If `render` is True, record and save the video.
    """

    state_keys = [
        "robot",
        "history",
        "ref_motion_",
    ]
    action_keys = [
        "loc",
        "scale",
    ]

    env.eval()
    env.set_seed(seed)

    tensordict_ = env.reset()
    s_a_pair = []

    inference_time = []
    with set_exploration_type(exploration_type):
        for i in tqdm(range(env.max_episode_length), miniters=10):
            s = time.perf_counter()
            s_a_pair.append(tensordict_.select(*state_keys, strict=False).cpu())  # record state
            tensordict_ = policy(tensordict_)
            s_a_pair[-1].update(tensordict_.select(*action_keys, strict=False).cpu())  # record action
            e = time.perf_counter()
            inference_time.append(e - s)
            tensordict, tensordict_ = env.step_and_maybe_reset(tensordict_)
    inference_time = np.mean(inference_time[5:])
    print(f"Average inference time: {inference_time:.4f} s")

    s_a_pair = torch.stack(s_a_pair, dim=1)

    return s_a_pair

@hydra.main(config_path="../cfg", config_name="eval", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app

    env, agent, vecnorm = make_env_policy(cfg)
    
    policy_eval = agent.get_rollout_policy("eval")
    sa_pair = evaluate(env, policy_eval, render=cfg.eval_render, seed=cfg.seed)
    
    print(termcolor.colored(sa_pair, "light_yellow"))
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    path = os.path.join(os.path.dirname(__file__), f"replay_buffer/sa_pair-{cfg.task.name}.pt")
    torch.save(sa_pair, path)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

