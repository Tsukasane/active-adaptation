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
REPLAY_BUFFER_PATH = os.path.join(FILE_PATH, "replay_buffer", "trajs-Walk.pt")

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

    run = wandb.init(
        job_type=cfg.wandb.job_type,
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        tags=cfg.wandb.tags,
    )
    run.config.update(OmegaConf.to_container(cfg))
    
    default_run_name = f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    run_idx = run.name.split("-")[-1]
    run.name = f"{run_idx}-{default_run_name}"
    setproctitle(run.name)

    cfg_save_path = os.path.join(run.dir, "cfg.yaml")
    OmegaConf.save(cfg, cfg_save_path)
    run.save(cfg_save_path, policy="now")
    run.save(os.path.join(run.dir, "config.yaml"), policy="now")

    env, policy, vecnorm = make_env_policy(cfg)

    save_interval = cfg.get("save_interval", -1)

    class BatchSampler:
        def __init__(self, tensordict, batch_size):
            self.data = tensordict

            self.perm = torch.randperm(
                (tensordict.shape[0] // batch_size) * batch_size,
                device=tensordict.device,
            ).reshape(batch_size, -1)

            self.length = self.perm.shape[1]

        def __len__(self):
            return self.length

        def sample(self):
            indice = self.perm[:, torch.randint(self.length, (1,))].squeeze()
            return self.data[indice]
        
    def save(policy, checkpoint_name: str, artifact: bool=False):
        ckpt_path = os.path.join(run.dir, f"{checkpoint_name}.pt")
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": run.name, "id": run.id}
        state_dict["policy"] = policy.state_dict()
        state_dict["cfg"] = cfg
        torch.save(state_dict, ckpt_path)
        if artifact:
            artifact = wandb.Artifact(
                f"{type(env).__name__}-{type(policy).__name__}", 
                type="model"
            )
            artifact.add_file(ckpt_path)
            run.log_artifact(artifact)
        run.save(ckpt_path, policy="now", base_path=run.dir)
        logging.info(f"Saved checkpoint to {str(ckpt_path)}")

    trajs = os.listdir(BUFFER_PATH)
    trajs.pop(trajs.index("trajs-Walk.pt"))
    expert_buffer = None
    for traj in trajs:
        path = os.path.join(BUFFER_PATH, traj)
        buffer: TensorDict = torch.load(path).reshape(-1).select(*keys, strict=False)
        if expert_buffer == None:
            expert_buffer = buffer
        else:
            expert_buffer = torch.cat((expert_buffer, buffer), dim=0)

    expert_buffer.rename_key_("loc", "gt_loc")
    expert_buffer.rename_key_("scale", "gt_scale")

    # print(expert_buffer)      # num_tasks * num_envs_per_task * num_steps_per_env
    replay_buffer: TensorDict = torch.load(REPLAY_BUFFER_PATH).reshape(-1).select(*keys, strict=False)
    replay_buffer.rename_key_("loc", "gt_loc")
    replay_buffer.rename_key_("scale", "gt_scale")

    epoch = policy.epoch
    batch_size = policy.batch_size

    expert_sampler, replay_sampler = BatchSampler(expert_buffer, batch_size), BatchSampler(replay_buffer, batch_size)

    for i in tqdm(range(epoch)):
        info = {}
        kl_a_losses, kl_b_losses, kl_losses = [], [], []
        for j in range(2 * max(len(expert_sampler), len(replay_sampler))):
            
            replay_batch = replay_sampler.sample().to(policy.device)
            expert_batch = expert_sampler.sample().to(policy.device)

            kl_a = policy.kl_loss_a(replay_batch)
            kl_b = policy.kl_loss_b(expert_batch)

            kl_loss = kl_a + kl_b
            policy.optimizer.zero_grad()
            kl_loss.backward()
            policy.optimizer.step()

            kl_a_losses.append(kl_a.item())
            kl_b_losses.append(kl_b.item())
            kl_losses.append(kl_loss.item())

        info["kl_loss_a"] = sum(kl_a_losses) / len(kl_a_losses)
        info["kl_loss_b"] = sum(kl_b_losses) / len(kl_b_losses)
        info["kl_loss"] = sum(kl_losses) / len(kl_losses)

        if i % 10 == 0:
            print(f"Iteration: {i}, KL Loss: {kl_loss}")

        if save_interval != -1 and i % save_interval == 0:
            save(policy, f"checkpoint_{i}")

        run.log(info)


    wandb.finish()
    exit(0)
    
    base_env.close()
    simulation_app.close()
    exit(0)

if __name__ == "__main__":
    main()