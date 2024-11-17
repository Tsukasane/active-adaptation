from math import inf
import torch

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import yaw_quat
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from active_adaptation.utils.helpers import batchify

from .locomotion import Reward, normalize

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

def dot(a: torch.Tensor, b: torch.Tensor):
    return (a * b).sum(-1, True)

class feet_clearance(Reward):
    """
    Avoid self-tripping by penalize a distance too small between the feet.
    """
    def __init__(self, env, feet_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_id, feet_names = self.asset.find_bodies(feet_names)
        self.feet_id = torch.tensor(self.feet_id, device=self.device)
        self.thres = 0.16
    
    def compute(self) -> torch.Tensor:
        feet_pos_w = self.asset.data.body_pos_w[:, self.feet_id]
        distance_xy = (feet_pos_w[:, 0, :2] - feet_pos_w[:, 1, :2]).norm(dim=-1, keepdim=True)
        return (- self.thres + distance_xy).clamp_max(0.) / self.thres


class knee_distance(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, clip_range=...):
        super().__init__(env, weight, enabled, clip_range)


class feet_swing(Reward):
    def __init__(self, env, feet_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_id = self.asset.find_bodies(feet_names)[0]
        self.phase: torch.Tensor = self.asset.data.phase
        self.command_manager = self.env.command_manager

        self.feet_vel_buf = torch.zeros(self.num_envs, 2, 3, 4, device=self.device)
    
    def update(self):
        self.feet_vel_buf[..., 1:] = self.feet_vel_buf[..., :-1]
        self.feet_vel_buf[..., 0] = self.asset.data.body_lin_vel_w[:, self.feet_id]

    def reset(self, env_ids):
        self.feet_vel_buf[env_ids] = 0.

    def compute(self) -> torch.Tensor:
        feet_linvel = self.feet_vel_buf.mean(-1)
        swing_vel = torch.zeros_like(feet_linvel)
        swing_vel[:] = self.command_manager.command_linvel.unsqueeze(1)
        phase_sin = self.phase.sin()
        swing_vel[:, 0] *= (phase_sin > +0.15).float().unsqueeze(1)
        swing_vel[:, 1] *= (phase_sin < -0.15).float().unsqueeze(1)
        swing_vel = quat_rotate(yaw_quat(self.asset.data.root_quat_w).unsqueeze(1), swing_vel)
        # reward = torch.exp(- 2 * (feet_linvel - swing_vel).abs().sum(-1)).sum(1, True)

        max_speed = self.command_manager.command_speed.unsqueeze(1).clamp_max(0.6)
        reward = dot(normalize(swing_vel), feet_linvel).clamp_max(max_speed)
        reward = torch.where(reward>0, reward.sqrt(), reward).sum(1, True)
        return reward.reshape(self.num_envs, 1) * (~self.command_manager.is_standing_env)
    
    def debug_draw(self):
        feet_pos = self.asset.data.body_pos_w[:, self.feet_id]
        feet_linvel = self.feet_vel_buf.mean(-1)
        swing_vel = torch.zeros_like(feet_pos)
        swing_vel[:] = self.command_manager.command_linvel.unsqueeze(1)
        swing_vel[:, 0] *= (self.phase.sin() > +0.15).float().unsqueeze(1)
        swing_vel[:, 1] *= (self.phase.sin() < -0.15).float().unsqueeze(1)
        swing_vel = quat_rotate(yaw_quat(self.asset.data.root_quat_w).unsqueeze(1), swing_vel)
        self.env.debug_draw.vector(feet_pos.reshape(-1, 3), swing_vel.reshape(-1, 3))
        self.env.debug_draw.vector(feet_pos.reshape(-1, 3), feet_linvel.reshape(-1, 3), color=(1., 0., 0.2, 1.))


class feet_orientation(Reward):

    def __init__(self, env, feet_names: str, weight: float, enabled: bool = True, body_name: str=None):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_id = self.asset.find_bodies(feet_names)[0]
        self.heading_feet = torch.tensor([[[1., 0., 0.]]], device=self.device)
        self.heading_root = torch.tensor([[[1., 0., 0.]]], device=self.device)
        if body_name is not None:
            self.body_id = self.asset.find_bodies(body_name)[0][0]
        else:
            self.body_id = None
    
    def compute(self) -> torch.Tensor:
        quat_feet = yaw_quat(self.asset.data.body_quat_w[:, self.feet_id])
        if self.body_id is not None:
            quat_body = yaw_quat(self.asset.data.body_quat_w[:, self.body_id]).unsqueeze(1)
        else:
            quat_body = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
        reward = dot(
            quat_rotate(quat_feet, self.heading_feet), 
            quat_rotate(quat_body, self.heading_root)
        )
        return (reward.square() * reward.sign()).mean(1)

    # def debug_draw(self):
    #     feet_pos = self.asset.data.body_pos_w[:, self.feet_id]
    #     quat_feet = yaw_quat(self.asset.data.body_quat_w[:, self.feet_id])
    #     self.env.debug_draw.vector(
    #         feet_pos.reshape(-1, 3),
    #         quat_rotate(quat_feet, self.heading_feet).reshape(-1, 3),
    #         color=(1., 1., 0., 1.)
    #     )
    #     self.env.debug_draw.vector(
    #         self.asset.data.root_pos_w, 
    #         quat_rotate(self.asset.data.root_quat_w, self.heading_root),
    #         color=(1., 1., 0., 1.)
    #     )


class joint_pos_default(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, joint_names: str=".*"):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids = self.asset.find_joints(joint_names)[0]
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids].clone()
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
    
    def compute(self) -> torch.Tensor:
        dev = self.asset.data.joint_pos[:, self.joint_ids] - self.default_joint_pos
        return - dev.square().mean(1, True)


class feet_step(Reward):
    def __init__(self, env, feet_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_id, feet_names = self.asset.find_bodies(feet_names)
        self.phase: torch.Tensor = self.asset.data.phase
        self.heading_root = torch.tensor([[1., 0., 0.]], device=self.device)
        self.command_manager = self.env.command_manager
        print(f"Feet names: {feet_names}, be aware of the order!")
    
    def compute(self) -> torch.Tensor:
        quat_root = yaw_quat(self.asset.data.root_quat_w)
        feet_displacement = (
            - self.asset.data.body_pos_w[:, self.feet_id[0]]
            + self.asset.data.body_pos_w[:, self.feet_id[1]]
        )
        feet_displacement = dot(quat_rotate(quat_root, self.heading_root), feet_displacement)
        phase_cos = self.phase.cos()
        reward = (phase_cos.sign().unsqueeze(1) * feet_displacement).clamp(max=0.4)
        return reward.reshape(self.num_envs, 1) * (~self.command_manager.is_standing_env)

    def debug_draw(self):
        quat_root = yaw_quat(self.asset.data.root_quat_w)
        feet_displacement = (
            - self.asset.data.body_pos_w[:, self.feet_id[0]]
            + self.asset.data.body_pos_w[:, self.feet_id[1]]
        )
        feet_displacement_projected = dot(quat_rotate(quat_root, self.heading_root), feet_displacement)
        phase_cos = self.phase.cos()
        reward = phase_cos.sign() * feet_displacement_projected.squeeze(1)
        positive = reward > 0
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[positive][:, self.feet_id[0]],
            feet_displacement[positive],
            color=(0., 1., 0., 1.)
        )
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[~positive][:, self.feet_id[0]],
            feet_displacement[~positive],
            color=(1., 0., 0., 1.)
        )


class body_orientation(Reward):
    def __init__(self, env, body_name, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_id, body_name = self.asset.find_bodies(body_name)
        self.body_id = self.body_id[0]
        self.x_vec = torch.tensor([1., 0., 0.], device=self.device)
        self.y_vec = torch.tensor([0., -1., 0.], device=self.device)
        self.body_heading_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self.root_heading_vec = torch.zeros(self.num_envs, 3, device=self.device)
    
    def update(self):
        body_yaw_quat = yaw_quat(self.asset.data.body_quat_w[:, self.body_id])
        root_yaw_quat = yaw_quat(self.asset.data.root_quat_w)
        self.body_heading_vec[:] = quat_rotate(body_yaw_quat, self.y_vec)
        self.root_heading_vec[:] = quat_rotate(root_yaw_quat, self.x_vec)
    
    def compute(self) -> torch.Tensor:
        reward = dot(self.body_heading_vec, self.root_heading_vec)
        return (reward.square() * reward.sign()).reshape(self.num_envs, 1)

    # def debug_draw(self):
    #     self.env.debug_draw.vector(
    #         self.asset.data.body_pos_w[:, self.body_id],
    #         self.body_heading_vec
    #     )


class arm_swing(Reward):
    def __init__(self, env, arm_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, body_names = self.asset.find_bodies(arm_names)
        self.phase: torch.Tensor = self.asset.data.phase
        self.command_manager = self.env.command_manager

        self.arm_vel_buf = torch.zeros(self.num_envs, 2, 3, 4, device=self.device)
    
    def update(self):
        self.arm_vel_buf[..., 1:] = self.arm_vel_buf[..., :-1]
        self.arm_vel_buf[..., 0] = self.asset.data.body_lin_vel_w[:, self.body_ids]

    def reset(self, env_ids):
        self.arm_vel_buf[env_ids] = 0.
    
    def compute(self) -> torch.Tensor:
        arm_linvel = self.arm_vel_buf.mean(-1)
        phase_sin = self.phase.sin()
        swing_vel = torch.zeros_like(arm_linvel)
        swing_vel[:] = self.command_manager.command_linvel.unsqueeze(1)
        phase_sin = self.phase.sin()
        swing_vel[:, 0] *= (phase_sin < +0.15).float().unsqueeze(1)
        swing_vel[:, 1] *= (phase_sin > -0.15).float().unsqueeze(1)
        swing_vel = quat_rotate(yaw_quat(self.asset.data.root_quat_w).unsqueeze(1), swing_vel)
        
        reward = (normalize(swing_vel) * arm_linvel).sum(-1)
        reward = reward.clamp(max=self.command_manager.command_speed).sum(1, True)
        # reward = torch.where(reward>0, reward.log1p().clamp(max=1.0), reward).sum(1, True)
        return reward.reshape(self.num_envs, 1) * (~self.command_manager.is_standing_env)


class arm_velocity(Reward):
    def __init__(self, env, arm_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, body_names = self.asset.find_bodies(arm_names)
    
    def compute(self) -> torch.Tensor:
        arm_linvel = self.asset.data.body_lin_vel_w[:, self.body_ids]
        root_linvel = self.asset.data.root_lin_vel_w
        d = (arm_linvel - root_linvel.unsqueeze(1)).square().sum(-1)
        return - d.mean(1, True)

r'''
in base.py : env
self._update( episode_length_buf += 1 )
self._compute_reward()
'''
    
class tracking_root(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, sigma: float = 0.1):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        
        self.sigma = sigma
        self.decay = self.env.command_manager.decay
    
    def compute(self) -> torch.Tensor:
        
        timestep = self.env.episode_length_buf.unsqueeze(1) - 1

        try:
            assert (timestep < self.env.max_episode_length).all()
        except AssertionError as e:
            timestep_idx = torch.where(timestep >= self.env.max_episode_length)[0]
            print(f"timestep in tracking root is greater than {self.env.max_episode_length} at {timestep_idx} with value {timestep[timestep_idx]}")

        batch_indices = torch.arange(self.num_envs, device=self.device)
        ref_root_translation = self.env.command_manager.ref_root_translations[batch_indices, timestep.squeeze(1)]
        ref_root_rotation = self.env.command_manager.ref_root_orient[timestep].squeeze(1)

        root_pos_w = self.asset.data.root_pos_w
        pos_err = (root_pos_w - ref_root_translation[:, :3]).square().sum(-1, True)

        root_rot_w = self.asset.data.root_quat_w
        dot_product = dot(root_rot_w, ref_root_rotation)
        angular_err = 2 * torch.acos(dot_product.abs().clamp(max=1.0))
        
        err = pos_err + angular_err

        self.env.command_manager._cum_error_root.mul_(self.decay).add_(err * self.env.step_dt)

        reward = torch.exp(- err.sqrt() / self.sigma)
        return reward
    
class tracking_velocity(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, sigma: float = 0.1):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        
        self.sigma = sigma
        self.decay = self.env.command_manager.decay

    def compute(self) -> torch.Tensor:
        timestep = self.env.episode_length_buf.unsqueeze(1) - 1
        self.ref_root_linear = self.env.command_manager.ref_root_linear[timestep].squeeze(1)
        ref_root_angular = self.env.command_manager.ref_root_angular[timestep].squeeze(1)
        ref_root_velocity = torch.cat([self.ref_root_linear, ref_root_angular], dim=1)
        err = (self.asset.data.root_vel_w - ref_root_velocity).square().sum(-1, True)

        self.env.command_manager._cum_error_vel.mul_(self.decay).add_(err * self.env.step_dt)

        reward = torch.exp(- err.sqrt() / self.sigma)
        return reward
    
    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w[:, :3],
            self.ref_root_linear,
            color=(0., 1., 0., 1.),
            size=1.0
        )


class tracking_qpos(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, sigma: float = 0.1, joint_names: str=".*"):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids = self.asset.find_joints(joint_names, preserve_order=True)[0]
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)

        self.sigma = sigma
        self.decay = self.env.command_manager.decay
    
    def compute(self) -> torch.Tensor:
        timestep = self.env.episode_length_buf.unsqueeze(1) - 1
        ref_qpos = self.env.command_manager.ref_qpos[timestep].squeeze(1)
        err = (self.asset.data.joint_pos[:, self.joint_ids] - ref_qpos).square()    # torch.Size([num_envs])
        
        self.env.command_manager._cum_error_qpos.mul_(self.decay).add_(err.mean(-1, True) * self.env.step_dt)
        
        reward = torch.exp(- err.mean(-1, True) / self.sigma)
        return reward

class tracking_keypoints(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, sigma: float = 0.1, body_names: str=".*"):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids = self.asset.find_bodies(body_names, preserve_order=True)[0]
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

        self.sigma = sigma
        self.decay = self.env.command_manager.decay
    
    def compute(self) -> torch.Tensor:
        timestep = self.env.episode_length_buf.unsqueeze(1) - 1
        ref_keypoints = self.env.command_manager.ref_keypoints[timestep].squeeze(1)
        body_pos_w = self.asset.data.body_pos_w[:, self.body_ids]   # torch.Size([num_envs, num_bodies, 3])
        root_position = self.asset.data.root_pos_w.unsqueeze(1)
        body_pos_w = body_pos_w - root_position
        root_quat = self.asset.data.root_quat_w.unsqueeze(1)

        body_pos_b = quat_rotate_inverse(root_quat, body_pos_w).reshape(self.num_envs, -1)
        err = (body_pos_b - ref_keypoints).square().sum(-1, True)
        
        self.env.command_manager._cum_error_keypoint.mul_(self.decay).add_(err * self.env.step_dt)
        
        reward = torch.exp(- err.sqrt() / self.sigma)
        if_need_keypoint = timestep < self.env.command_manager.num_frames
        return reward * if_need_keypoint.float()
    

class task_done(Reward):
    def __init__(self, env, 
                 weight: float, 
                 enabled: bool = True,
                 thres: float = 1.0,
                 number: float = 100,):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.thres = thres
        self.number = number

        self.num_frames = self.env.command_manager.num_frames

    def update(self):
        self._cum_error = self.env.command_manager._cum_error_root + \
                            self.env.command_manager._cum_error_vel + \
                            self.env.command_manager._cum_error_qpos + \
                            self.env.command_manager._cum_error_keypoint

    def compute(self) -> torch.Tensor:
        reach_thres = self._cum_error < self.thres
        reach_max_frame = (self.env.episode_length_buf == self.num_frames).unsqueeze(1)
        reward = (reach_thres & reach_max_frame).float() * self.number
        return reward
