import torch
import torchvision
import hydra
import numpy as np
import einops
import wandb
import logging
import os
import time
import datetime

from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle

from omni.isaac.lab.app import AppLauncher
# from omni_drones.utils.wandb import init_wandb
from active_adaptation.utils.torchrl import SyncDataCollector
from tensordict.tensordict import TensorDictBase, TensorDict

# local import
from helpers import make_env_policy, EpisodeStats, evaluate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")
BUFFER_PATH = os.path.join(FILE_PATH, "replay_buffer")

keys = [
    "robot",
    "history",
    "ref_motion_",
    "loc",
    "scale",
    ]

@hydra.main(config_path=CONFIG_PATH, config_name="distill", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app

    env, policy, vecnorm = make_env_policy(cfg)
        
    class Sampler:
        def __init__(self, tensordict, num_mini_batch, mini_batch_size, device="cuda"):
            self.data = tensordict
            self.num_mini_batch = num_mini_batch
            self.mini_batch_size = mini_batch_size
            self.device = device

            self.perm = torch.randperm(
                (tensordict.shape[0] // mini_batch_size) * mini_batch_size,
                device=tensordict.device,
            ).reshape(mini_batch_size, -1)

        def __len__(self):
            return self.perm.shape[1]
        
        def generator(self):
            for _ in range(self.num_mini_batch):
                indice = self.perm[:, torch.randint(self.perm.shape[1], (1,))].squeeze()
                yield self.data[indice].to(self.device)
        
    run_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    def save(policy, checkpoint_name: str, artifact: bool=False):
        ckpt_path = os.path.join(run_dir, f"{checkpoint_name}.pt")
        state_dict = OrderedDict()
        state_dict["policy"] = policy.state_dict()
        state_dict["cfg"] = cfg
        torch.save(state_dict, ckpt_path)
        logging.info(f"Saved checkpoint to {str(ckpt_path)}")

    trajs_dis = os.path.join(BUFFER_PATH, cfg.buffer_name)
    trajs = os.listdir(trajs_dis)
    replay_file = trajs.pop(-1)

    expert_buffer = None
    for traj in trajs:
        path = os.path.join(trajs_dis, traj)
        buffer: TensorDict = torch.load(path).reshape(-1)
        buffer = buffer.select(*keys, strict=False)
        if expert_buffer == None:
            expert_buffer = buffer
        else:
            expert_buffer = torch.cat((expert_buffer, buffer), dim=0)

    expert_buffer.rename_key_("loc", "gt_loc")
    expert_buffer.rename_key_("scale", "gt_scale")

    REPLAY_BUFFER_PATH = os.path.join(BUFFER_PATH, cfg.buffer_name, replay_file)
    replay_buffer: TensorDict = torch.load(REPLAY_BUFFER_PATH).reshape(-1).select(*keys, strict=False)
    replay_buffer.rename_key_("loc", "gt_loc")
    replay_buffer.rename_key_("scale", "gt_scale")

    print("Expert Buffer: ", expert_buffer)      # num_tasks * num_envs_per_task * num_steps_per_env
    print("Replay Buffer: ", replay_buffer)
    
    epoch = max(policy.epoch, cfg.epoch)
    batch_size = max(policy.batch_size, cfg.batch_size)
    num_mini_batch = max(len(expert_buffer) // batch_size, len(replay_buffer) // batch_size)
    print(f"Behavioral Cloning: Epochs: {epoch}, Batch Size: {batch_size}, Num Mini Batch: {num_mini_batch}")

    expert_sampler, replay_sampler = Sampler(expert_buffer, num_mini_batch, batch_size, policy.device), Sampler(replay_buffer, num_mini_batch, batch_size, policy.device)

    for i in tqdm(range(epoch)):
        pbar = tqdm(zip(expert_sampler.generator(), replay_sampler.generator()))
        for expert_batch, replay_batch in pbar:

            kl_a = policy.kl_loss_a(replay_batch)
            kl_b = policy.kl_loss_b(expert_batch)

            kl_loss = kl_a + kl_b
            policy.optimizer.zero_grad()
            kl_loss.backward()
            policy.optimizer.step()
            
            pbar.set_description(f"KL Loss A: {kl_a.item():.4f}, KL Loss B: {kl_b.item():.4f}, KL Loss: {kl_loss.item():.4f}")

        if i % 10 == 0:
            save(policy, f"checkpoint_{i}")

    save(policy, "checkpoint_final")
    
    simulation_app.close()
    exit(0)

if __name__ == "__main__":
    main()