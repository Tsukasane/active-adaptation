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

    # Observations of reference motion
    class ref_orientation(mdp.Observation):

        env: "Humanoid"

        def __init__(self, env, steps: int=1):
            super().__init__(env)
            self.robot: Articulation = self.env.scene["robot"]
            self.steps = steps
            self.ref_orientation = self.env.command_manager.root_orientation

        def compute(self) -> torch.Tensor:
            timestep = self.env.episode_length_buf.cpu()
            max_frame = self.env.max_episode_length.cpu()
            step_range = torch.arange(self.steps)
            timestep = timestep.unsqueeze(-1) + step_range  # (num_envs, steps)
            timestep = torch.min(timestep, max_frame[:, None]-1)
            ref_orientation = self.ref_orientation[timestep].to(self.device).float()
            return ref_orientation.reshape(self.num_envs, -1)

    class ref_height(mdp.Observation):
        def __init__(self, env, steps: int=1):
            super().__init__(env)
            self.robot: Articulation = self.env.scene["robot"]
            self.steps = steps
            self.ref_root_translation = self.env.command_manager.root_translations

        def compute(self) -> torch.Tensor:
            timestep = self.env.episode_length_buf.cpu()
            max_frame = self.env.max_episode_length.cpu()
            step_range = torch.arange(self.steps)
            timestep = timestep.unsqueeze(-1) + step_range
            timestep = torch.min(timestep, max_frame[:, None]-1)
            ref_root_translation = self.ref_root_translation[timestep].to(self.device).float()
            return ref_root_translation[:, :, 2].reshape(self.num_envs, -1)
        
    class ref_keypoints(mdp.Observation):
        def __init__(self, env, steps: int=1):
            super().__init__(env)
            self.robot: Articulation = self.env.scene["robot"]
            self.steps = steps
            self.ref_keypoints = self.env.command_manager.kp    # (num_frames, num_joints, 3)

        def compute(self) -> torch.Tensor:
            timestep = self.env.episode_length_buf.cpu()
            max_frame = self.env.max_episode_length.cpu()
            step_range = torch.arange(self.steps)
            timestep = timestep.unsqueeze(-1) + step_range
            timestep = torch.min(timestep, max_frame[:, None]-1)
            ref_keypoints = self.ref_keypoints[timestep].to(self.device).float()    # (num_envs, steps, num_joints, 3)
            return ref_keypoints.reshape(self.num_envs, -1)
        
    class ref_keypoints_gap(mdp.Observation):
        def __init__(self, env, body_names: str, steps: int=1):
            super().__init__(env)
            self.robot: Articulation = self.env.scene["robot"]
            self.steps = steps
            self.ref_keypoints = self.env.command_manager.kp    # (num_frames, num_joints, 3)
            self.body_indices, self.body_names = self.robot.find_bodies(body_names, preserve_order=True)
            self.body_pos_local = torch.zeros(self.num_envs, len(self.body_indices), 3, device=self.device)

        def update(self):
            quat = self.robot.data.root_quat_w.unsqueeze(1)
            body_pos = self.robot.data.body_pos_w[:, self.body_indices]
            body_pos -= self.robot.data.root_pos_w.unsqueeze(1)
            self.body_pos_local = quat_rotate_inverse(quat, body_pos)

        def compute(self):
            timestep = self.env.episode_length_buf.cpu()
            max_frame = self.env.max_episode_length.cpu()
            step_range = torch.arange(self.steps)
            timestep = timestep.unsqueeze(-1) + step_range
            timestep = torch.min(timestep, max_frame[:, None]-1)
            ref_keypoints = self.ref_keypoints[timestep].to(self.device).float()   # (num_envs, steps, num_joints, 3)
            body_pos_local = self.body_pos_local.unsqueeze(1).expand_as(ref_keypoints)
            ref_keypoints_gap = ref_keypoints - body_pos_local
            return ref_keypoints_gap.reshape(self.num_envs, -1)
    
    class ref_trans_gap(mdp.Observation):
        def __init__(self, env, steps: int=1):
            super().__init__(env)
            self.robot: Articulation = self.env.scene["robot"]
            self.steps = steps
            self.ref_root_translation = self.env.command_manager.root_translations

        def compute(self):
            timestep = self.env.episode_length_buf.cpu()
            max_frame = self.env.max_episode_length.cpu()
            step_range = torch.arange(self.steps)
            timestep = timestep.unsqueeze(-1) + step_range
            timestep = torch.min(timestep, max_frame[:, None]-1)
            ref_root_translation = self.ref_root_translation[timestep].to(self.device).float()  # (num_envs, steps, 3)
            ref_root_translation += self.env.scene.env_origins.unsqueeze(1)
            self.root_pos = self.robot.data.root_pos_w.unsqueeze(1)
            root_quat_w = self.robot.data.root_quat_w.unsqueeze(1)
            self.gap = ref_root_translation - self.root_pos
            ref_trans_gap = quat_rotate_inverse(root_quat_w, self.gap)
            return ref_trans_gap.reshape(self.num_envs, -1)
        
        def debug_draw(self):
            self.env.debug_draw.vector(
                self.root_pos[:, 0],
                self.gap[:, 0],
                color=(1., 0., 1., 1.),
                size=1.
            )
    
    # Motion Tracking Reward
    class tracking_root_trans(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True, sigma: float = 0.1):
            super().__init__(env, weight, enabled)
            self.robot: Articulation = self.env.scene["robot"]
            self.sigma = sigma

        def compute(self) -> torch.Tensor:
            timestep = self.env.episode_length_buf.cpu()
            ref_root_translation = self.env.command_manager.root_translations[timestep].to(self.device) + self.env.scene.env_origins
            root_pos_w = self.robot.data.root_pos_w
            error = (root_pos_w - ref_root_translation).square().sum(-1, True)
            reward = torch.exp(- error.sqrt() / self.sigma)
            return reward
        
    class tracking_root_rot(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True, sigma: float = 0.1):
            super().__init__(env, weight, enabled)
            self.robot: Articulation = self.env.scene["robot"]
            self.sigma = sigma

        def compute(self) -> torch.Tensor:
            timestep = self.env.episode_length_buf.cpu()
            ref_root_orientation = self.env.command_manager.root_orientation[timestep].to(self.device)
            root_quat_w = self.robot.data.root_quat_w
            dot_product = dot(root_quat_w, ref_root_orientation)
            error = 2 * torch.acos(dot_product.abs().clamp(min=-1.0, max=1.0))
            reward = torch.exp(- error / self.sigma)
            return reward
        
    class tracking_keypoints(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True, sigma: float = 0.1, body_names: str = ".*"):
            super().__init__(env, weight, enabled)
            self.robot: Articulation = self.env.scene["robot"]
            self.sigma = sigma
            self.body_indices, self.body_names = self.robot.find_bodies(body_names, preserve_order=True)

        def compute(self) -> torch.Tensor:
            timestep = self.env.episode_length_buf.cpu()
            ref_keypoints = self.env.command_manager.kp[timestep].to(self.device)
            body_pos = self.robot.data.body_pos_w[:, self.body_indices]
            body_pos -= self.robot.data.root_pos_w.unsqueeze(1)
            root_quat_w = self.robot.data.root_quat_w.unsqueeze(1)
            body_pos_local = quat_rotate_inverse(root_quat_w, body_pos)

            diff = (ref_keypoints - body_pos_local).norm(dim=-1)
            error = diff.square().sum(-1, True)
            reward = torch.exp(- error.sqrt() / self.sigma)
            return reward

    # Early Termination Conditions
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
            self.device = self.env.device
            self.max_theta = torch.tensor(max_theta * 3.14 / 180, device=self.env.device)
            self.robot: Articulation = self.env.scene["robot"]

        def __call__(self) -> torch.Tensor:
            timestep = (self.env.episode_length_buf - 1).cpu()
            ref_root_orientation = self.env.command_manager.root_orientation[timestep].to(self.device)

            root_quat_w = self.robot.data.root_quat_w
            dot_product = dot(root_quat_w, ref_root_orientation)
            deviation = 2 * torch.acos(dot_product.abs().clamp(min=-1.0, max=1.0))

            return deviation > self.max_theta

def dot(a: torch.Tensor, b: torch.Tensor):
    return (a * b).sum(-1, True)