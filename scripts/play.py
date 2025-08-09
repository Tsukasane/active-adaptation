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
        from active_adaptation.envs.mdp.action import JointPosition, ResidualJointPosition
        action_manager: JointPosition = env.action_manager
        policy_config["policy_joint_names"] = action_manager.joint_names
        if isinstance(action_manager, ResidualJointPosition):
            if action_manager.progress_iters == -1:
                policy_config["use_residual_action"] = True
                obs_cfg["_ref_joint_pos"] = {}
                obs_cfg["_ref_joint_pos"]["ref_joint_pos_future"] = {}
                obs_cfg["_ref_joint_pos"]["ref_joint_pos_future"]["joint_names"] = action_manager.joint_names
                obs_cfg["_ref_joint_pos"]["ref_joint_pos_future"]["body_names"] = []
                obs_cfg["_ref_joint_pos"]["ref_joint_pos_future"]["future_steps"] = [0]
            elif action_manager.progress_iters >= 0:
                assert env.current_iter >= action_manager.progress_iters, \
                    "Current iter must be greater than progress_iters to anneal the residual action."

        ## command
        command = env.command_manager
        command_obs = policy_config["observation"]["command"]
        if cfg.task.command._target_ == "active_adaptation.envs.mdp.commands.motion_tracking.command.MotionTrackingCommand":
            from active_adaptation.envs.mdp.commands.motion_tracking.command import MotionTrackingCommand
            command: MotionTrackingCommand
            motion_duration_second = command.dataset.lengths[0].item() * env.step_dt
            future_steps = command.future_steps.tolist()
            tracking_keypoint_names = command.tracking_keypoint_names
            tracking_joint_names = command.tracking_joint_names

            for obs_key in command_obs:
                command_obs[obs_key]["motion_duration_second"] = motion_duration_second
                command_obs[obs_key]["future_steps"] = future_steps
                command_obs[obs_key]["body_names"] = tracking_keypoint_names
                command_obs[obs_key]["joint_names"] = tracking_joint_names
                command_obs[obs_key]["root_body_name"] = "pelvis"
        elif cfg.task.command._target_ == "active_adaptation.envs.mdp.commands.hdmi.command.RobotTracking":
            from active_adaptation.envs.mdp.commands.hdmi.command import RobotTracking
            command: RobotTracking
            tracking_keypoint_names = command.tracking_keypoint_names
            tracking_joint_names = command.tracking_joint_names
            motion_duration_second = command.dataset.lengths[0].item() * env.step_dt
            future_steps = command.future_steps.tolist()
            tracking_keypoint_names = command.tracking_keypoint_names
            tracking_joint_names = command.tracking_joint_names
            root_body_name = command.root_body_name

            for obs_key in command_obs:
                command_obs[obs_key]["motion_duration_second"] = motion_duration_second
                command_obs[obs_key]["future_steps"] = future_steps
                command_obs[obs_key]["body_names"] = tracking_keypoint_names
                command_obs[obs_key]["joint_names"] = tracking_joint_names
                command_obs[obs_key]["root_body_name"] = root_body_name
        elif cfg.task.command._target_ == "active_adaptation.envs.mdp.commands.hdmi.command.RobotObjectTracking":
            from active_adaptation.envs.mdp.commands.hdmi.command import RobotObjectTracking
            command: RobotObjectTracking
            tracking_keypoint_names = command.tracking_keypoint_names
            tracking_joint_names = command.tracking_joint_names
            motion_duration_second = command.dataset.lengths[0].item() * env.step_dt
            future_steps = command.future_steps.tolist()
            tracking_keypoint_names = command.tracking_keypoint_names
            tracking_joint_names = command.tracking_joint_names
            root_body_name = command.root_body_name

            for obs_key in command_obs:
                command_obs[obs_key]["motion_duration_second"] = motion_duration_second
                command_obs[obs_key]["future_steps"] = future_steps
                command_obs[obs_key]["body_names"] = tracking_keypoint_names
                command_obs[obs_key]["joint_names"] = tracking_joint_names
                command_obs[obs_key]["root_body_name"] = root_body_name
            
            object_name = cfg.task.command.object_asset_name
            contact_target_pos_offset = np.array(cfg.task.command.contact_target_pos_offset).tolist()
            # convert to list
            policy_obs = policy_config["observation"]["policy"]
            ref_contact_obs_cfg = policy_obs.get("ref_contact_pos_b", None)
            if ref_contact_obs_cfg is not None:
                ref_contact_obs_cfg["object_name"] = object_name
                ref_contact_obs_cfg["root_body_name"] = root_body_name
                ref_contact_obs_cfg["contact_target_pos_offset"] = contact_target_pos_offset
        elif cfg.task.command._target_ == "active_adaptation.envs.mdp.commands.box_transport.command.BoxTransport":
            from active_adaptation.envs.mdp.commands.box_transport.command import BoxTransport
            command: BoxTransport
            object_name = command.object_asset_name
            root_body_name = command.root_body_name
            
            policy_obs = policy_config["observation"]["policy"]
            object_obs = ["object_pos_b", "object_ori_b"]
            for obs_key in object_obs:
                obs_cfg = policy_obs.get(obs_key, None)
                if obs_cfg is not None:
                    obs_cfg["object_name"] = object_name
                    obs_cfg["root_body_name"] = root_body_name

            ref_contact_obs_cfg = policy_obs.get("ref_contact_pos_b", None)
            if ref_contact_obs_cfg is not None:
                ref_contact_obs_cfg["object_name"] = object_name
                ref_contact_obs_cfg["root_body_name"] = root_body_name
                contact_target_pos_offset = np.array(cfg.task.command.contact_target_pos_offset).tolist()
                policy_obs["ref_contact_pos_b"]["contact_target_pos_offset"] = contact_target_pos_offset
                    
            
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
    with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
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