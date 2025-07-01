import torch
import hydra
import numpy as np
import einops
import itertools
import os
import datetime
import re
from omegaconf import OmegaConf

from isaaclab.app import AppLauncher

from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictSequential

from active_adaptation.utils.export import export_onnx
from active_adaptation.utils.wandb import parse_checkpoint_path


@hydra.main(config_path="../cfg", config_name="play", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    app_launcher = AppLauncher(cfg.app)
    simulation_app = app_launcher.app

    from scripts.helpers import EpisodeStats, make_env_policy, ObsNorm
    env, policy, vecnorm = make_env_policy(cfg)
    
    if cfg.export_policy:
        import time
        import copy
        
        # Load checkpoint to get wandb info and checkpoint number
        checkpoint_path = parse_checkpoint_path(cfg.checkpoint_path)
        wandb_run_id = "unknown"
        checkpoint_num = "unknown"
        
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, weights_only=False)
            # Get wandb run ID from state_dict
            if "wandb" in state_dict and "id" in state_dict["wandb"]:
                wandb_run_id = state_dict["wandb"]["id"]
            
            # Extract checkpoint number from filename
            filename = os.path.basename(checkpoint_path)
            match = re.search(r'checkpoint_(\d+)', filename)
            if match:
                checkpoint_num = match.group(1)
            elif filename.endswith('_final.pt'):
                checkpoint_num = "final"
        
        fake_input = env.observation_spec[0].rand().cpu()
        fake_input["is_init"] = torch.tensor(1, dtype=bool)
        fake_input["context_adapt_hx"] = torch.zeros(128)
        fake_input = fake_input.unsqueeze(0)

        def test(m, x):
            start = time.perf_counter()
            for _ in range(1000):
                m(x)
            return (time.perf_counter() - start) / 1000
        
        FILE_PATH = os.path.dirname(__file__)
        
        deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy"))
        obs_norm = ObsNorm.from_vecnorm(vecnorm, deploy_policy.in_keys)
        _policy = TensorDictSequential(obs_norm, deploy_policy).cpu()
        
        print(f"Inference time of policy: {test(_policy, fake_input)}")

        # Use new filename format with wandb_run_id and checkpoint_num
        os.makedirs(os.path.join(FILE_PATH, "exports", cfg.task.name), exist_ok=True)
        path = os.path.join(FILE_PATH, "exports", cfg.task.name, f"policy-{wandb_run_id}-{checkpoint_num}.pt")
        torch.save(_policy, path)

        meta = {}
        export_onnx(_policy, fake_input, path.replace(".pt", ".onnx"), meta)

        # export policy config
        dict_cfg = OmegaConf.to_container(cfg, resolve=True)
        ## observation
        policy_config = dict()
        obs_cfg = dict()
        obs_keys = ["policy", "command"]
        for k in obs_keys:
            obs_cfg[k] = dict_cfg["task"]["observation"][k]
        policy_config["observation"] = obs_cfg
        
        ## action
        policy_config["action_scale"] = dict_cfg["task"]["action"]["action_scaling"]

        ## joint names and stiffness/damping
        from active_adaptation.assets import get_asset_meta
        asset_meta = get_asset_meta(env.scene["robot"])
        policy_config["isaac_joint_names"] = asset_meta["joint_names_isaac"]
        policy_config["joint_kp"] = asset_meta["actuators"]["base_legs"]["stiffness"]
        policy_config["joint_kd"] = asset_meta["actuators"]["base_legs"]["damping"]
        policy_config["default_joint_pos"] = asset_meta["init_state"]["joint_pos"]

        ## policy joint names
        from active_adaptation.envs.mdp.action import JointPosition
        action_manager: JointPosition = env.action_manager
        policy_config["policy_joint_names"] = action_manager.joint_names

        ## motion length
        from active_adaptation.envs.mdp.commands.motion_tracking import MotionTrackingCommand
        command: MotionTrackingCommand = env.command_manager
        policy_config["motion_duration_second"] = command.dataset.lengths[0].item() * env.step_dt

        import yaml
        with open(path.replace(".pt", ".yaml"), "w") as f:
            yaml.dump(policy_config, f, sort_keys=False)
        print(f"Policy config saved to {path.replace('.pt', '.yaml')}")

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    policy = policy.get_rollout_policy("eval")
    
    env.base_env.eval()
    td_ = env.reset()
    assert not env.base_env.training
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        torch.compiler.cudagraph_mark_step_begin()
        for i in itertools.count():
            td_ = policy(td_)
            td, td_ = env.step_and_maybe_reset(td_)
            # td_.update(td["next"])
            episode_stats.add(td)

            if len(episode_stats) >= env.num_envs:
                print("Step", i)
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    print(k, torch.mean(v).item())
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

