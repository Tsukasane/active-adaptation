import torch
import numpy as np

from typing import TYPE_CHECKING, List, Tuple, Dict
from omegaconf import DictConfig

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor
    from active_adaptation.assets.objects import DoorArticulation
    from isaaclab.assets.rigid_object import RigidObject

from active_adaptation.envs.mdp import Reward as BaseReward, Observation as BaseObservation, Randomization as BaseRandomization
from active_adaptation.utils.motion import MotionDataset, MotionData
from active_adaptation.utils.math import (
    quat_rotate_inverse,
    quat_mul,
    quat_conjugate,
    axis_angle_from_quat
)
from .base import Command
from isaaclab.utils.math import yaw_quat, matrix_from_quat, quat_from_angle_axis, wrap_to_pi, quat_apply, euler_xyz_from_quat, matrix_from_euler
from isaaclab.utils.string import resolve_matching_names_values
from active_adaptation.utils.helpers import batchify

quat_apply = batchify(quat_apply)

def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    qw = quat[:, 0]
    qx = quat[:, 1]
    qy = quat[:, 2]
    qz = quat[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return yaw

def rand_uniform(low: float, high: float, size: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.rand(size, device=device) * (high - low) + low

class MotionTrackingCommand(Command):
    def __init__(
        self, env, data_path: str,
        cum_keypoint_pos_scale: float = 1.0,
        cum_keypoint_ori_scale: float = 3.0,
        cum_joint_pos_scale: float = 1.0,
        lift_height: float = 0.02,
        init_root_pos_noise: float = 0.02,
        init_root_ori_noise: float = 0.1,
        init_root_lin_vel_noise: float = 0.05,
        init_root_ang_vel_noise: float = 0.05,
        init_joint_pos_noise: float = 0.2,
        init_joint_vel_noise: float = 0.5,
        future_steps: List[int] = [1, 2, 8, 16],
        call_update: bool = True,
    ):
        super().__init__(env)
        self.contact_forces: ContactSensor = self.env.scene["contact_forces"]

        self.dataset = MotionDataset.create_from_path(
            data_path,
            target_fps=int(1/self.env.step_dt)
        ).to(self.device)

        # Set tracking body and joint names for observation and termination
        tracking_keypoint_names = [
            ".*_hip_(pitch|yaw)_link", 
            ".*_knee_link", 
            ".*_ankle_roll_link", 
            "pelvis", 
            "torso_link", 
            ".*_shoulder_pitch_link", 
            ".*_elbow_link", 
            ".*_wrist_yaw_link"
        ]
        self.tracking_keypoint_names = self.asset.find_bodies(tracking_keypoint_names)[1]
        self.tracking_body_indices_motion = []
        self.tracking_body_indices_asset = []
        for body_name in self.tracking_keypoint_names:
            self.tracking_body_indices_motion.append(self.dataset.body_names.index(body_name))
            self.tracking_body_indices_asset.append(self.asset.body_names.index(body_name))

        tracking_joint_names = [
            "waist_.*_joint", 
            ".*_hip_.*_joint", 
            ".*_knee_joint", 
            ".*_ankle_.*_joint", 
            ".*_shoulder_.*_joint", 
            ".*_elbow_joint"
        ]
        self.tracking_joint_names = self.asset.find_joints(tracking_joint_names)[1]
        self.tracking_joint_indices_motion = []
        self.tracking_joint_indices_asset = []
        for joint_name in self.tracking_joint_names:
            self.tracking_joint_indices_motion.append(self.dataset.joint_names.index(joint_name))
            self.tracking_joint_indices_asset.append(self.asset.joint_names.index(joint_name))
        
        # Set feet body and joint indices for motion contact and no contact reward
        feet_names = ".*ankle_roll_link"
        self.feet_ids_motion = self.dataset.find_bodies(feet_names)[0]
        self.feet_ids_asset = self.asset.find_bodies(feet_names)[0]
        self.feet_ids_sensor = self.contact_forces.find_bodies(feet_names)[0]
        
        # get root body and joint indices in motion for reset
        root_body_name = "pelvis"
        self.root_body_idx_motion = self.dataset.body_names.index(root_body_name)
        
        asset_joint_names = self.asset.joint_names
        self.asset_joint_idx_motion = [self.dataset.joint_names.index(joint_name) for joint_name in asset_joint_names]

        with torch.device(self.device):
            self._cum_error = torch.zeros(self.num_envs, 3)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self.future_steps = torch.tensor(future_steps)

            self.motion_ids = torch.zeros(self.num_envs, dtype=int)
            self.motion_len = torch.zeros(self.num_envs, dtype=int)
            self.t = torch.zeros(self.num_envs, dtype=torch.long)

            self.eval_t = torch.randint(0, 100, (self.num_envs,))

        self._cum_keypoint_pos_scale = cum_keypoint_pos_scale
        self._cum_keypoint_ori_scale = cum_keypoint_ori_scale
        self._cum_joint_pos_scale = cum_joint_pos_scale

        self.lift_height = lift_height
        self.init_root_pos_noise = init_root_pos_noise
        self.init_root_ori_noise = init_root_ori_noise
        self.init_root_lin_vel_noise = init_root_lin_vel_noise
        self.init_root_ang_vel_noise = init_root_ang_vel_noise
        self.init_joint_pos_noise = init_joint_pos_noise
        self.init_joint_vel_noise = init_joint_vel_noise

        self.num_tracking_bodies = len(self.tracking_body_indices_asset)
        self.num_tracking_joints = len(self.tracking_joint_indices_asset)
        self.num_future_steps = len(self.future_steps)

        if call_update:
            self._init_debug_draw()
            self.update()

    def _sample_motions(self, env_ids: torch.Tensor) -> None:
        # sample motion id and start time for each env
        motion_ids = torch.randint(0, self.dataset.num_motions, size=(len(env_ids),), device=self.device)
        motion_len = self.dataset.lengths[motion_ids]
        max_len = motion_len - self.future_steps[-1]
        start_phase = torch.rand(len(env_ids), device=self.device)
        start_t = (start_phase * max_len).long()

        self.motion_ids[env_ids] = motion_ids
        self.motion_len[env_ids] = motion_len
        self.t[env_ids] = start_t

        if not self.env.training:
            # self.t[env_ids] = self.eval_t[env_ids]
            self.t[env_ids] = 0

    def sample_init(self, env_ids: torch.Tensor) -> None:
        self._sample_motions(env_ids)

        # reset root state and joint position/velocity from motion
        self._motion_reset = self.dataset.get_slice(self.motion_ids[env_ids], self.t[env_ids], 1).squeeze(1)
        # shape: [len(env_ids), num_bodies/num_joints, 3/4/...]

        motion: MotionData = self._motion_reset
        init_root_pos = motion.body_pos_w[:, self.root_body_idx_motion]
        init_root_pos += self.env.scene.env_origins[env_ids]
        # only lift up root pos to avoid penetration
        init_root_pos_noise = (1 + torch.randn_like(init_root_pos[:, 2]).clamp(-1, 1)) * self.init_root_pos_noise
        init_root_pos[:, 2] += self.lift_height + init_root_pos_noise
        
        init_root_quat = motion.body_quat_w[:, self.root_body_idx_motion]
        # generate random quat
        random_axis = torch.rand(len(env_ids), 3, device=self.device)
        random_angle = torch.rand(len(env_ids), device=self.device).clamp(-1, 1) * self.init_root_ori_noise
        random_quat = quat_from_angle_axis(random_angle, random_axis)
        # remove yaw rotation
        random_quat = quat_mul(quat_conjugate(yaw_quat(random_quat)), random_quat)
        init_root_quat = quat_mul(random_quat, init_root_quat)

        init_root_lin_vel = motion.body_lin_vel_w[:, self.root_body_idx_motion]
        init_root_ang_vel = motion.body_ang_vel_w[:, self.root_body_idx_motion]

        init_root_lin_vel_noise = torch.randn_like(init_root_lin_vel).clamp(-1, 1) * self.init_root_lin_vel_noise
        init_root_ang_vel_noise = torch.randn_like(init_root_ang_vel).clamp(-1, 1) * self.init_root_ang_vel_noise
        init_root_lin_vel += init_root_lin_vel_noise
        init_root_ang_vel += init_root_ang_vel_noise

        # write to asset
        init_root_state = torch.cat([init_root_pos, init_root_quat, init_root_lin_vel, init_root_ang_vel], dim=-1)
        self.asset.write_root_state_to_sim(init_root_state, env_ids=env_ids)
        
        init_joint_pos = motion.joint_pos[:, self.asset_joint_idx_motion]
        init_joint_vel = motion.joint_vel[:, self.asset_joint_idx_motion]

        joint_pos_noise = torch.randn_like(init_joint_pos).clamp(-1, 1) * self.init_joint_pos_noise
        joint_vel_noise = torch.randn_like(init_joint_vel).clamp(-1, 1) * self.init_joint_vel_noise

        init_joint_pos += joint_pos_noise
        init_joint_vel += joint_vel_noise

        self.asset.write_joint_state_to_sim(init_joint_pos, init_joint_vel, env_ids=env_ids)

    def reset(self, env_ids):
        self._cum_error[env_ids] = 0.0

    @property
    def success(self):
        return (self.t >= self.motion_len - 1).unsqueeze(1)
    
    @property
    def finished(self):
        return (self.t >= self.motion_len).unsqueeze(1)
    
    TrackObservation = BaseObservation["MotionTrackingCommand"]

    class ref_joint_pos_future(TrackObservation):
        def compute(self):
            return self.command_manager.future_ref_motion.joint_pos[:, :, self.command_manager.tracking_joint_indices_motion].view(self.num_envs, -1)

    class ref_joint_vel_future(TrackObservation):
        def compute(self):
            return self.command_manager.future_ref_motion.joint_vel[:, :, self.command_manager.tracking_joint_indices_motion].view(self.num_envs, -1)

    class diff_body_pos_future_b(TrackObservation):
        def compute(self):
            return self.command_manager._diff_body_pos_future_b.view(self.num_envs, -1)
    
    class diff_body_lin_vel_future_b(TrackObservation):
        def compute(self):
            return self.command_manager._diff_body_lin_vel_future_b.view(self.num_envs, -1)

    class diff_body_ori_future_b(TrackObservation):
        def compute(self):
            return self.command_manager._diff_body_ori_future_b[:, :, :, :2, :].reshape(self.num_envs, -1)

    class diff_body_ang_vel_future_b(TrackObservation):
        def compute(self):
            return self.command_manager._diff_body_ang_vel_future_b.view(self.num_envs, -1)

    class ref_motion_phase(TrackObservation):
        def compute(self):
            return (self.command_manager.t / self.command_manager.motion_len).unsqueeze(1)
        
    
    class motion_feet_contact(TrackObservation):
        def __init__(self, body_names: str | List[str], motion_vel_thres: float=0.1, motion_height_thres: float=0.05, **kwargs):
            super().__init__(**kwargs)
            self.motion_vel_thres = motion_vel_thres
            self.motion_height_thres = motion_height_thres
            body_names = self.command_manager.asset.find_bodies(body_names)[1]
            self.body_indices_motion = []
            self.body_indices_sensor = []
            for name in body_names:
                body_idx_motion = self.command_manager.dataset.body_names.index(name)
                body_idx_sensor = self.command_manager.contact_forces.body_names.index(name)

                self.body_indices_motion.append(body_idx_motion)
                self.body_indices_sensor.append(body_idx_sensor)

            shape = (self.env.num_envs, self.command_manager.num_future_steps, len(body_names))
            self.motion_contact = torch.zeros(shape, dtype=torch.bool, device=self.device)
            
        def update(self):
            feet_vel_motion = self.command_manager.future_ref_motion.body_lin_vel_w[:, :, self.body_indices_motion]
            feet_vel_motion_norm = feet_vel_motion.norm(dim=-1)
            feet_pos_motion = self.command_manager.future_ref_motion.body_pos_w[:, :, self.body_indices_motion]
            feet_pos_motion_z = feet_pos_motion[..., 2]
            self.motion_contact[:] = (feet_vel_motion_norm < self.motion_vel_thres) & (feet_pos_motion_z < self.motion_height_thres)

        def compute(self):
            return self.motion_contact.view(self.num_envs, -1)
    
    class motion_feet_no_contact(TrackObservation):
        def __init__(self, body_names: str | List[str], motion_vel_thres: float=0.1, **kwargs):
            super().__init__(**kwargs)
            self.motion_vel_thres = motion_vel_thres
            body_names = self.command_manager.asset.find_bodies(body_names)[1]
            self.body_indices_motion = []
            self.body_indices_sensor = []
            for name in body_names:
                body_idx_motion = self.command_manager.dataset.body_names.index(name)
                body_idx_sensor = self.command_manager.contact_forces.body_names.index(name)

                self.body_indices_motion.append(body_idx_motion)
                self.body_indices_sensor.append(body_idx_sensor)

            shape = (self.env.num_envs, self.command_manager.num_future_steps, len(body_names))
            self.motion_no_contact = torch.zeros(shape, dtype=torch.bool, device=self.device)

        def update(self):
            feet_vel_motion = self.command_manager.future_ref_motion.body_lin_vel_w[:, :, self.body_indices_motion]
            feet_vel_motion_norm = feet_vel_motion.norm(dim=-1)
            self.motion_no_contact[:] = feet_vel_motion_norm > self.motion_vel_thres

        def compute(self):
            return self.motion_no_contact.view(self.num_envs, -1)
    
    TrackReward = BaseReward["MotionTrackingCommand"]

    class tracking_keypoint(TrackReward):
        def __init__(self, body_names: List[str] | str | None = None, sigma: float = 0.03, tolerance: float | Dict[str, float] = 0.0, **kwargs):
            super().__init__(**kwargs)
            if body_names is None:
                body_names = self.command_manager.tracking_keypoint_names
            
            self.sigma = sigma
            body_indices_motion, matched_names_motion = self.command_manager.dataset.find_bodies(body_names)
            body_indices_asset, matched_names_asset = self.command_manager.asset.find_bodies(body_names)

            matched_names = set(matched_names_motion) & set(matched_names_asset)
            assert set(matched_names) == set(matched_names_motion) == set(matched_names_asset), "body names in motion dataset and robot not matched"
            assert set(matched_names) <= set(self.command_manager.tracking_keypoint_names), "Some body names in motion dataset not found in tracking body names"
            
            self.body_indices_motion = []
            self.body_indices_asset = []
            self.body_names = list(sorted(matched_names))
            self.num_bodies = len(self.body_names)
            for body_name in self.body_names:
                body_idx_motion = body_indices_motion[matched_names_motion.index(body_name)]
                body_idx_asset = body_indices_asset[matched_names_asset.index(body_name)]

                self.body_indices_motion.append(body_idx_motion)
                self.body_indices_asset.append(body_idx_asset)

            self.tolerance = torch.zeros(len(self.body_names), device=self.device)
            if isinstance(tolerance, float):
                self.tolerance[:] = tolerance
            elif isinstance(tolerance, DictConfig):
                tolerance = dict(tolerance)
                tolerance_indices, tolerance_names, tolerance_values = resolve_matching_names_values(tolerance, self.body_names)
                self.tolerance[tolerance_indices] = torch.tensor(tolerance_values, device=self.device)
            else:
                raise ValueError(f"Invalid tolerance type: {type(tolerance)}")

        def compute(self):
            raise NotImplementedError

    class keypoint_pos_tracking_product(tracking_keypoint):
        def compute(self):
            body_pos_asset = self.command_manager.asset.data.body_pos_w[:, self.body_indices_asset]
            body_pos_motion = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]
            diff = body_pos_motion - body_pos_asset
            # shape: [num_envs, num_tracking_bodies, 3]
            error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
            # shape: [num_envs, num_tracking_bodies]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

    class keypoint_pos_tracking_local_product(tracking_keypoint):
        def compute(self):
            body_pos_asset = self.command_manager.asset.data.body_pos_w[:, self.body_indices_asset]
            body_pos_motion = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]

            root_pos_asset = self.command_manager.asset.data.root_pos_w.clone()
            root_pos_motion = self.command_manager.ref_body_pos_w[:, self.command_manager.root_body_idx_motion].clone()
            root_quat_asset = self.command_manager.asset.data.root_quat_w
            root_quat_motion = self.command_manager.ref_body_quat_w[:, self.command_manager.root_body_idx_motion]
            
            root_pos_asset[..., 2] = 0.0
            root_pos_motion[..., 2] = 0.0
            root_quat_asset = yaw_quat(root_quat_asset)
            root_quat_motion = yaw_quat(root_quat_motion)
            
            root_pos_asset = root_pos_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
            root_pos_motion = root_pos_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)
            root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
            root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

            body_pos_asset_relative = quat_rotate_inverse(root_quat_asset, body_pos_asset - root_pos_asset)
            body_pos_motion_relative = quat_rotate_inverse(root_quat_motion, body_pos_motion - root_pos_motion)

            diff = body_pos_motion_relative - body_pos_asset_relative
            # shape: [num_envs, num_tracking_bodies, 3]
            error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
            # shape: [num_envs, num_tracking_bodies]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

    class keypoint_ori_tracking_product(tracking_keypoint):
        def compute(self):
            body_ori_asset = self.command_manager.asset.data.body_quat_w[:, self.body_indices_asset]
            body_ori_motion = self.command_manager.ref_body_quat_w[:, self.body_indices_motion]
            diff = quat_mul(quat_conjugate(body_ori_motion), body_ori_asset)
            # shape: [num_envs, num_tracking_bodies, 4]
            error = torch.norm(axis_angle_from_quat(diff), dim=-1)
            error = (error - self.tolerance).clamp_min(0.0)
            # shape: [num_envs, num_tracking_bodies]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
        
    class keypoint_ori_tracking_local_product(tracking_keypoint):
        def compute(self):
            body_ori_asset = self.command_manager.asset.data.body_quat_w[:, self.body_indices_asset]
            body_ori_motion = self.command_manager.ref_body_quat_w[:, self.body_indices_motion]

            root_quat_asset = self.command_manager.asset.data.root_quat_w
            root_quat_motion = self.command_manager.ref_body_quat_w[:, self.command_manager.root_body_idx_motion]

            root_quat_asset = yaw_quat(root_quat_asset)
            root_quat_motion = yaw_quat(root_quat_motion)

            root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
            root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

            body_ori_asset_relative = quat_mul(quat_conjugate(root_quat_asset), body_ori_asset)
            body_ori_motion_relative = quat_mul(quat_conjugate(root_quat_motion), body_ori_motion)

            diff = quat_mul(quat_conjugate(body_ori_motion_relative), body_ori_asset_relative)
            # shape: [num_envs, num_tracking_bodies, 4]
            error = torch.norm(axis_angle_from_quat(diff), dim=-1)
            error = (error - self.tolerance).clamp_min(0.0)
            # shape: [num_envs, num_tracking_bodies]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
        
    class keypoint_lin_vel_tracking_product(tracking_keypoint):
        def compute(self):
            body_lin_vel_asset = self.command_manager.asset.data.body_lin_vel_w[:, self.body_indices_asset]
            body_lin_vel_motion = self.command_manager.ref_body_lin_vel_w[:, self.body_indices_motion]
            diff = body_lin_vel_motion - body_lin_vel_asset
            # shape: [num_envs, num_tracking_bodies, 3]
            error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
            # shape: [num_envs, num_tracking_bodies]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

    class keypoint_lin_vel_tracking_local_product(tracking_keypoint):
        def compute(self):
            body_lin_vel_asset = self.command_manager.asset.data.body_lin_vel_w[:, self.body_indices_asset].clone()
            body_lin_vel_motion = self.command_manager.ref_body_lin_vel_w[:, self.body_indices_motion].clone()

            root_pos_asset = self.command_manager.asset.data.root_pos_w.clone()
            root_pos_motion = self.command_manager.ref_body_pos_w[:, self.command_manager.root_body_idx_motion].clone()
            root_quat_asset = self.command_manager.asset.data.root_quat_w
            root_quat_motion = self.command_manager.ref_body_quat_w[:, self.command_manager.root_body_idx_motion]

            root_pos_asset[..., 2] = 0.0
            root_pos_motion[..., 2] = 0.0
            root_quat_asset = yaw_quat(root_quat_asset)
            root_quat_motion = yaw_quat(root_quat_motion)
            
            root_pos_asset = root_pos_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
            root_pos_motion = root_pos_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)
            root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
            root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

            body_lin_vel_asset_relative = quat_rotate_inverse(root_quat_asset, body_lin_vel_asset - root_pos_asset)
            body_lin_vel_motion_relative = quat_rotate_inverse(root_quat_motion, body_lin_vel_motion - root_pos_motion)
            
            diff = body_lin_vel_motion_relative - body_lin_vel_asset_relative
            # shape: [num_envs, num_tracking_bodies, 3]
            error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
            # shape: [num_envs, num_tracking_bodies]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

    class keypoint_ang_vel_tracking_product(tracking_keypoint):
        def compute(self):
            body_ang_vel_asset = self.command_manager.asset.data.body_ang_vel_w[:, self.body_indices_asset]
            body_ang_vel_motion = self.command_manager.ref_body_ang_vel_w[:, self.body_indices_motion]
            diff = body_ang_vel_motion - body_ang_vel_asset
            # shape: [num_envs, num_tracking_bodies, 3]
            error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
            # shape: [num_envs, num_tracking_bodies]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

    class keypoint_ang_vel_tracking_local_product(tracking_keypoint):
        def compute(self):
            body_ang_vel_asset = self.command_manager.asset.data.body_ang_vel_w[:, self.body_indices_asset]
            body_ang_vel_motion = self.command_manager.ref_body_ang_vel_w[:, self.body_indices_motion]

            root_quat_asset = self.command_manager.asset.data.root_quat_w
            root_quat_motion = self.command_manager.ref_body_quat_w[:, self.command_manager.root_body_idx_motion]

            root_quat_asset = yaw_quat(root_quat_asset)
            root_quat_motion = yaw_quat(root_quat_motion)

            root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
            root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

            body_ang_vel_asset_relative = quat_rotate_inverse(root_quat_asset, body_ang_vel_asset)
            body_ang_vel_motion_relative = quat_rotate_inverse(root_quat_motion, body_ang_vel_motion)

            diff = body_ang_vel_motion_relative - body_ang_vel_asset_relative
            # shape: [num_envs, num_tracking_bodies, 3]
            error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
            # shape: [num_envs, num_tracking_bodies]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

    class tracking_joint(TrackReward):
        def __init__(self, joint_names: List[str] | str | None = None, sigma: float = 0.03, tolerance: float | Dict[str, float] = 0.0, **kwargs):
            super().__init__(**kwargs)
            if joint_names is None:
                joint_names = self.command_manager.tracking_joint_names
        
            self.sigma = sigma
            joint_indices_asset, matched_names_asset = self.command_manager.asset.find_joints(joint_names)
            joint_indices_motion, matched_names_motion = self.command_manager.dataset.find_joints(joint_names)

            matched_names = set(matched_names_motion) & set(matched_names_asset)
            assert set(matched_names) == set(matched_names_motion) == set(matched_names_asset), "joint names in motion dataset and robot not matched"
            assert set(matched_names) <= set(self.command_manager.tracking_joint_names), "Some joint names in motion dataset not found in tracking joint names"

            self.joint_indices_motion = []
            self.joint_indices_asset = []
            self.joint_names = list(sorted(matched_names))
            for joint_name in self.joint_names:
                joint_idx_motion = joint_indices_motion[matched_names_motion.index(joint_name)]
                joint_idx_asset = joint_indices_asset[matched_names_asset.index(joint_name)]

                self.joint_indices_motion.append(joint_idx_motion)
                self.joint_indices_asset.append(joint_idx_asset)

            self.tolerance = torch.zeros(len(self.joint_names), device=self.env.device)
            if isinstance(tolerance, float):
                self.tolerance[:] = tolerance
            elif isinstance(tolerance, DictConfig):
                tolerance = dict(tolerance)
                tolerance_indices, tolerance_names, tolerance_values = resolve_matching_names_values(tolerance, matched_names_motion)
                self.tolerance[tolerance_indices] = torch.tensor(tolerance_values, device=self.env.device)
            else:
                raise ValueError(f"Invalid tolerance type: {type(tolerance)}")

    class joint_pos_tracking_product(tracking_joint):
        def compute(self):
            joint_pos_asset = self.command_manager.asset.data.joint_pos[:, self.joint_indices_asset]
            joint_pos_motion = self.command_manager.ref_joint_pos[:, self.joint_indices_motion]
            diff = joint_pos_motion - joint_pos_asset
            error = (diff.abs() - self.tolerance).clamp_min(0.0)
            # shape: [num_envs, num_tracking_joints]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
        
    class joint_vel_tracking_product(tracking_joint):
        def compute(self):
            joint_vel_asset = self.command_manager.asset.data.joint_vel[:, self.joint_indices_asset]
            joint_vel_motion = self.command_manager.ref_joint_vel[:, self.joint_indices_motion]
            diff = joint_vel_motion - joint_vel_asset
            error = (diff.abs() - self.tolerance).clamp_min(0.0)
            # shape: [num_envs, num_tracking_joints]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

    class feet_no_contact_when_motion_vel(TrackReward):
        def __init__(self, body_names: str | List[str], motion_vel_thres: float=0.1, **kwargs):
            super().__init__(**kwargs)
            self.motion_vel_thres = motion_vel_thres
            body_names = self.command_manager.asset.find_bodies(body_names)[1]
            self.body_indices_motion = []
            self.body_indices_sensor = []
            for name in body_names:
                body_idx_motion = self.command_manager.dataset.body_names.index(name)
                body_idx_sensor = self.command_manager.contact_forces.body_names.index(name)

                self.body_indices_motion.append(body_idx_motion)
                self.body_indices_sensor.append(body_idx_sensor)

            self.motion_no_contact = torch.zeros((self.env.num_envs, len(body_names)), dtype=torch.bool, device=self.device)
        
            if self.env.backend != "isaac":
                return

            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            import isaaclab.sim as sim_utils
            vis_markers_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/MotionNoContact",
                markers={
                    "motion_no_contact": sim_utils.SphereCfg(
                        radius=0.06,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.5, 0.0, 1.0),
                            metallic=1.0,
                            roughness=0.1,
                        ),
                    ),
                },
            )

            self.vis_markers = VisualizationMarkers(vis_markers_cfg)
            self.vis_markers_pos_w = torch.zeros((self.env.num_envs, len(body_names), 3), device=self.device)
        
        def update(self):
            # when feet vel in motion is large, feet should not be in contact
            feet_vel_motion = self.command_manager.ref_body_lin_vel_w[:, self.body_indices_motion]
            feet_vel_motion_norm = feet_vel_motion[..., :2].norm(dim=-1)
            self.motion_no_contact[:] = feet_vel_motion_norm > self.motion_vel_thres

        def compute(self):
            # shape: [num_envs, num_feet]
            contact_forces = self.command_manager.contact_forces.data.net_forces_w[:, self.body_indices_sensor]
            in_contact = contact_forces.norm(dim=-1) > 0.1
            # shape: [num_envs, num_feet]
            penalty = in_contact & self.motion_no_contact
            return -penalty.float().mean(dim=1, keepdim=True)

        def debug_draw(self):
            if self.env.backend != "isaac":
                return
            return

            self.vis_markers_pos_w.fill_(-100.0)
            self.vis_markers_pos_w[self.motion_no_contact] = self.command_manager.ref_body_pos_w[:, self.body_indices_motion][self.motion_no_contact]
            self.vis_markers.visualize(
                translations=self.vis_markers_pos_w.view(-1, 3),
            )

    class feet_contact_when_motion_contact(TrackReward):
        def __init__(self, body_names: str | List[str], motion_vel_thres: float=0.1, motion_height_thres: float=0.1, **kwargs):
            super().__init__(**kwargs)
            self.motion_vel_thres = motion_vel_thres
            self.motion_height_thres = motion_height_thres
            body_names = self.command_manager.asset.find_bodies(body_names)[1]
            self.body_indices_motion = []
            self.body_indices_sensor = []
            for name in body_names:
                body_idx_motion = self.command_manager.dataset.body_names.index(name)
                body_idx_sensor = self.command_manager.contact_forces.body_names.index(name)

                self.body_indices_motion.append(body_idx_motion)
                self.body_indices_sensor.append(body_idx_sensor)
            
            self.motion_contact = torch.zeros((self.env.num_envs, len(body_names)), dtype=torch.bool, device=self.device)
            
            if self.env.backend != "isaac":
                return

            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            import isaaclab.sim as sim_utils
            vis_markers_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/MotionFeetContact",
                markers={
                    "motion_contact": sim_utils.SphereCfg(
                        radius=0.06,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 0.5, 1.0),
                        ),
                    ),
                },
            )

            self.vis_markers = VisualizationMarkers(vis_markers_cfg)
            self.vis_markers_pos_w = torch.zeros((self.env.num_envs, len(body_names), 3), device=self.device)
        
        def update(self):
            # when feet vel in motion is small
            feet_vel_motion = self.command_manager.ref_body_lin_vel_w[:, self.body_indices_motion]
            feet_vel_motion_norm = feet_vel_motion[..., :2].norm(dim=-1)
            # when feet height is small
            feet_pos_motion = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]
            feet_pos_motion_z = feet_pos_motion[..., 2]
            self.motion_contact[:] = (feet_vel_motion_norm < self.motion_vel_thres) & (feet_pos_motion_z < self.motion_height_thres)

        def compute(self):
            # shape: [num_envs, num_feet]
            contact_forces = self.command_manager.contact_forces.data.net_forces_w[:, self.body_indices_sensor]
            in_contact = contact_forces.norm(dim=-1) > 0.1

            # shape: [num_envs, num_feet]
            reward = in_contact & self.motion_contact
            return reward.float().mean(dim=1, keepdim=True)
        
        def debug_draw(self):
            if self.env.backend != "isaac":
                return
            return

            self.vis_markers_pos_w.fill_(-100.0)
            self.vis_markers_pos_w[self.motion_contact] = self.command_manager.ref_body_pos_w[:, self.body_indices_motion][self.motion_contact]
            self.vis_markers.visualize(
                translations=self.vis_markers_pos_w.view(-1, 3),
            )


    # class feet_tracking(Reward):
    #     def compute(self):
    #         in_contact = self.command_manager.contact_forces.data.current_contact_time[:, self.command_manager.feet_ids_sensor] > 0.01
    #         first_contact = self.command_manager.contact_forces.compute_first_contact(0.02)[:, self.command_manager.feet_ids_sensor]
    #         diff = self.command_manager.ref_body_pos_w[:, self.command_manager.feet_ids_motion] - self.command_manager.asset.data.body_pos_w[:, self.command_manager.feet_ids_asset]
    #         error = diff.square().sum(-1)
    #         return - (error * first_contact).sum(1, True)

    def update(self):
        # future ref motion for actor observation
        self.future_ref_motion = self.dataset.get_slice(self.motion_ids, self.t, steps=self.future_steps)
        # shape: [num_envs, len(future_steps), num_bodies/num_joints, 3/4/...]

        # Observations: future ref and diff to body frame
        self.root_quat_yaw_w = yaw_quat(self.asset.data.root_quat_w)

        root_quat_w = self.root_quat_yaw_w.unsqueeze(1).unsqueeze(1).repeat(1, self.num_future_steps, self.num_tracking_bodies, 1)
        root_pos_w = self.asset.data.root_pos_w.unsqueeze(1).unsqueeze(1).repeat(1, self.num_future_steps, self.num_tracking_bodies, 1)

        body_pos_w = self.asset.data.body_pos_w[:, self.tracking_body_indices_asset].unsqueeze(1).repeat(1, self.num_future_steps, 1, 1)
        body_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.tracking_body_indices_asset].unsqueeze(1).repeat(1, self.num_future_steps, 1, 1)
        ref_body_pos_future_w = self.future_ref_motion.body_pos_w[..., self.tracking_body_indices_motion, :] + self.env.scene.env_origins[:, None, None, :]
        ref_body_lin_vel_future_w = self.future_ref_motion.body_lin_vel_w[..., self.tracking_body_indices_motion, :]
        # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]

        self.ref_body_pos_future_b = quat_rotate_inverse(root_quat_w, ref_body_pos_future_w - root_pos_w)
        self.ref_body_lin_vel_future_b = quat_rotate_inverse(root_quat_w, ref_body_lin_vel_future_w)
        self._diff_body_pos_future_b = quat_rotate_inverse(root_quat_w, ref_body_pos_future_w - body_pos_w)
        self._diff_body_lin_vel_future_b = quat_rotate_inverse(root_quat_w, ref_body_lin_vel_future_w - body_lin_vel_w)

        body_quat_w = self.asset.data.body_quat_w[:, self.tracking_body_indices_asset].unsqueeze(1).repeat(1, self.num_future_steps, 1, 1)
        body_ang_vel_w = self.asset.data.body_ang_vel_w[:, self.tracking_body_indices_asset].unsqueeze(1).repeat(1, self.num_future_steps, 1, 1)
        ref_body_quat_future_w = self.future_ref_motion.body_quat_w[..., self.tracking_body_indices_motion, :]
        ref_body_ang_vel_future_w = self.future_ref_motion.body_ang_vel_w[..., self.tracking_body_indices_motion, :]
        # shape: [num_envs, num_future_steps, num_tracking_bodies, 4]
        
        ref_body_quat_future_b = quat_mul(quat_conjugate(root_quat_w), ref_body_quat_future_w)
        self.ref_body_ori_future_b = matrix_from_quat(ref_body_quat_future_b)
        self.ref_body_ang_vel_future_b = quat_rotate_inverse(root_quat_w, ref_body_ang_vel_future_w)
        diff_body_quat_future_b = quat_mul(quat_conjugate(body_quat_w), ref_body_quat_future_b)
        self._diff_body_ori_future_b = matrix_from_quat(diff_body_quat_future_b)
        self._diff_body_ang_vel_future_b = quat_rotate_inverse(root_quat_w, ref_body_ang_vel_future_w - body_ang_vel_w)

        # Reward: current ref motion for reward computation
        self.current_ref_motion: MotionData = self.future_ref_motion[:, 0]
        self.ref_body_pos_w = self.current_ref_motion.body_pos_w + self.env.scene.env_origins[:, None, :]
        self.ref_body_lin_vel_w = self.current_ref_motion.body_lin_vel_w
        self.ref_body_quat_w = self.current_ref_motion.body_quat_w
        self.ref_body_ang_vel_w = self.current_ref_motion.body_ang_vel_w
        self.ref_joint_pos = self.current_ref_motion.joint_pos
        self.ref_joint_vel = self.current_ref_motion.joint_vel
        # shape: [num_envs, num_future_steps, num_tracking_bodies, xxx]
        
        # Termination: tracking error
        cur_tracking_body_pos_w = self.asset.data.body_pos_w[:, self.tracking_body_indices_asset]
        ref_tracking_body_pos_w = self.ref_body_pos_w[:, self.tracking_body_indices_motion]
        cur_tracking_body_quat_w = self.asset.data.body_quat_w[:, self.tracking_body_indices_asset]
        ref_tracking_body_quat_w = self.ref_body_quat_w[:, self.tracking_body_indices_motion]
        cur_tracking_joint_pos = self.asset.data.joint_pos[:, self.tracking_joint_indices_asset]
        ref_tracking_joint_pos = self.ref_joint_pos[:, self.tracking_joint_indices_motion]

        if self.env.backend == "isaac":
            self.all_marker_pos_w[0] = cur_tracking_body_pos_w
            self.all_marker_pos_w[1] = ref_tracking_body_pos_w
            # self.all_marker_pos_w[0] = ref_body_pos_future_w[:, 0]
            # self.all_marker_pos_w[1] = ref_body_pos_future_w[:, -1]


        diff_body_pos_w = ref_tracking_body_pos_w - cur_tracking_body_pos_w
        diff_body_quat_w = quat_mul(quat_conjugate(ref_tracking_body_quat_w), cur_tracking_body_quat_w)
        diff_joint_pos = ref_tracking_joint_pos - cur_tracking_joint_pos

        error_body_pos = diff_body_pos_w.norm(dim=-1)
        error_body_ori = torch.norm(axis_angle_from_quat(diff_body_quat_w), dim=-1)
        error_joint_pos = diff_joint_pos.abs()

        self._cum_error[:, 0] = error_body_pos.mean(dim=1) / self._cum_keypoint_pos_scale
        self._cum_error[:, 1] = error_body_ori.mean(dim=1) / self._cum_keypoint_ori_scale
        self._cum_error[:, 2] = error_joint_pos.mean(dim=1) / self._cum_joint_pos_scale
                
        self.t += 1
    
    def _init_debug_draw(self):
        if self.env.backend != "isaac":
            return
        
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils
        vis_markers_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Keypoints",
            markers={
                "robot": sim_utils.SphereCfg(
                    radius=0.04,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0)
                    ),
                ),
                "reference": sim_utils.SphereCfg(
                    radius=0.04,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0)
                    ),
                ),
            },
        )
        self.vis_markers = VisualizationMarkers(vis_markers_cfg)
        num_ref_markers = self.num_envs * self.num_tracking_bodies
        self.marker_indices = [0] * num_ref_markers + [1] * num_ref_markers
        self.all_marker_pos_w = torch.zeros(2, self.num_envs, self.num_tracking_bodies, 3, device=self.device)

    def debug_draw(self):
        if self.env.backend != "isaac":
            return
        
        # shape: [2, num_envs, num_tracking_bodies, 3]
        self.vis_markers.visualize(
            translations=self.all_marker_pos_w.reshape(-1, 3),
            marker_indices=self.marker_indices,
        )

        robot_keypoints_w = self.all_marker_pos_w[0].reshape(-1, 3)
        target_keypoints_w = self.all_marker_pos_w[1].reshape(-1, 3)
        self.env.debug_draw.vector(
            robot_keypoints_w,
            target_keypoints_w - robot_keypoints_w,
            color=(0, 0, 1, 1)
        )

        # in_contact = self.contact_forces.data.current_contact_time[:, self.feet_ids_sensor] > 0.01
        # feet_pos_asset = self.asset.data.body_pos_w[:, self.feet_ids_asset]
        # feet_pos_motion = self.ref_body_pos_w[:, self.feet_ids_motion]
        # diff = feet_pos_motion - feet_pos_asset
        # self.env.debug_draw.vector(
        #     self.asset.data.body_pos_w[:, self.feet_ids_asset].reshape(-1, 3),
        #     (diff * in_contact.unsqueeze(-1)).reshape(-1, 3),
        #     color=(0, 1, 0, 1),
        #     size=5.
        # )
