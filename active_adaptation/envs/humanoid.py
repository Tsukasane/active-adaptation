from math import inf
import torch

from omni.isaac.lab.sensors import ContactSensor, RayCaster
from omni.isaac.lab.actuators import DCMotor
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import yaw_quat
from omni.isaac.lab.utils.warp import raycast_mesh
from active_adaptation.utils.helpers import batchify
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)
from tensordict.tensordict import TensorDictBase, TensorDict

from active_adaptation.envs.locomotion import Env, LocomotionEnv

import active_adaptation.envs.mdp as mdp

class Humanoid(LocomotionEnv):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.max_episode_length = torch.ones(self.num_envs, dtype=torch.long, device=self.device) * self.command_manager.num_frames

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.num_envs)
        else:
            env_mask = torch.ones(self.num_envs, dtype=bool, device=self.device)
        env_ids = env_mask.nonzero().squeeze(-1)
        if len(env_ids):
            self._reset_idx(env_ids)

        for callback in self._reset_callbacks:
            callback(env_ids)

        self.scene.update(self.step_dt)
        if tensordict is None:
            tensordict = TensorDict({}, self.num_envs, device=self.device)
            self._compute_observation(tensordict)
        else:
            tensordict.update(self.observation_spec.zero())
        if self.record_now and env_mask[self.lookat_env_i]:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                self.complete_video_frames.extend(self.video_frames)
            self.video_frames = []
        return tensordict
    
    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state, start_frames, end_frames = self.command_manager.sample_init(env_ids)
        if not self.robot.is_fixed_base:
            self.robot.write_root_state_to_sim(
                init_root_state, 
                env_ids=env_ids
            )
        self.stats[env_ids] = 0.

        self.scene.reset(env_ids)

        self.episode_length_buf[env_ids] = start_frames
        self.max_episode_length[env_ids] = end_frames

        # in `self._reset_callbacks`
        # self.command_manager.reset(env_ids=env_ids)
        # self.action_manager.reset(env_ids=env_ids)

    class root_deviation(mdp.Termination):
        def __init__(self, env, max_distance: float):
            super().__init__(env)
            self.device = self.env.device
            self.max_distance = torch.tensor(max_distance, device=self.env.device)
            self.robot: Articulation = self.env.scene["robot"]

        def __call__(self):
            timestep = (self.env.episode_length_buf - 1).cpu()
            ref_root_translation = self.env.command_manager.root_translations[timestep].to(self.device) + self.env.scene.env_origins
            root_pos_w = self.robot.data.root_pos_w
            deviation = (root_pos_w - ref_root_translation).norm(dim=1, keepdim=True)
            return deviation > self.max_distance
        
    class root_rot_deviation(mdp.Termination):
        def __init__(self, env, max_theta: float):
            super().__init__(env)
            self.devive = self.env.device
            self.max_theta = torch.tensor(max_theta * 3.14 / 180, device=self.env.device)
            self.robot: Articulation = self.env.scene["robot"]

        def __call__(self) -> torch.Tensor:
            timestep = (self.env.episode_length_buf - 1).cpu()
            ref_root_orientation = self.env.command_manager.root_orientation[timestep].to(self.device)

            root_quat_w = self.asset.data.root_quat_w
            dot_product = dot(root_quat_w, ref_root_orientation)
            deviation = 2 * torch.acos(dot_product.abs().clamp(min=-1.0, max=1.0))

            return deviation > self.max_theta
    # class root_orientation(mdp.Reward):
            
    #     env: "Humanoid"

    #     def __init__(self, env, weight: float, enabled: bool = True):
    #         super().__init__(env, weight, enabled)
    #         self.asset: Articulation = self.env.scene["robot"]

    #     def compute(self) -> torch.Tensor:
    #         z = self.asset.data.projected_gravity_b[:, 2].square().unsqueeze(1)
    #         y = self.asset.data.projected_gravity_b[:, 1].abs().unsqueeze(1)
    #         return z - y
    
    # class arm_velocity_cum_error(mdp.Termination):
    #     def __init__(self, env, thres: float=0.8):
    #         super().__init__(env)
    #         self.threshold = thres
    #         self.asset: Articulation = self.env.scene["robot"]
    #         self.action_manager: mdp.action.HumanoidWithArm = self.env.action_manager
    #         if not isinstance(self.action_manager, mdp.action.HumanoidWithArm):
    #             raise ValueError("`HumanoidWithArm` action manager required")
    #         self.cum_error = self.action_manager.cum_error

    #     def __call__(self):
    #         return (self.cum_error > self.threshold).any(1, True)
    
    # class command_arm_linvel(mdp.Observation):
    #     def __init__(self, env):
    #         super().__init__(env)
    #         self.asset: Articulation = self.env.scene["robot"]
    #         self.action_manager: mdp.action.HumanoidWithArm = self.env.action_manager
    #         if not isinstance(self.action_manager, mdp.action.HumanoidWithArm):
    #             raise ValueError("`HumanoidWithArm` action manager required")

    #     def compute(self) -> torch.Tensor:
    #         return self.action_manager.command_arm_linvel.reshape(self.num_envs, -1)

def dot(a: torch.Tensor, b: torch.Tensor):
    return (a * b).sum(-1, True)