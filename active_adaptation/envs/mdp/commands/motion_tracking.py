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
from isaaclab.utils.math import yaw_quat, matrix_from_quat, quat_from_angle_axis, wrap_to_pi, quat_apply, euler_xyz_from_quat, matrix_from_euler, sample_uniform, quat_from_euler_xyz
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
        pose_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2)},
        velocity_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (-0.05, 0.05),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1)},
        init_joint_pos_noise: float = 0.1,
        init_joint_vel_noise: float = 0.1,
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

            self.eval_t = torch.randint(0, self.dataset.lengths[0], (self.num_envs,), device=self.device)

        self._cum_keypoint_pos_scale = cum_keypoint_pos_scale
        self._cum_keypoint_ori_scale = cum_keypoint_ori_scale
        self._cum_joint_pos_scale = cum_joint_pos_scale

        self.lift_height = lift_height

        pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.pose_range = torch.tensor(pose_range_list, device=self.device)
        velocity_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.velocity_range = torch.tensor(velocity_range_list, device=self.device)

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

        self.t[env_ids] = 0
        if not self.env.training:
            # self.t[env_ids] = self.eval_t[env_ids]
            self.t[env_ids] = 0
            pass

    def sample_init(self, env_ids: torch.Tensor) -> None:
        self._sample_motions(env_ids)

        # reset root state and joint position/velocity from motion
        self._motion_reset = self.dataset.get_slice(self.motion_ids[env_ids], self.t[env_ids], 1).squeeze(1)
        # shape: [len(env_ids), num_bodies/num_joints, 3/4/...]

        motion: MotionData = self._motion_reset
        init_root_pos = motion.body_pos_w[:, self.root_body_idx_motion]
        init_root_quat = motion.body_quat_w[:, self.root_body_idx_motion]
        init_root_lin_vel = motion.body_lin_vel_w[:, self.root_body_idx_motion]
        init_root_ang_vel = motion.body_ang_vel_w[:, self.root_body_idx_motion]

        # poses
        rand_samples = sample_uniform(self.pose_range[:, 0], self.pose_range[:, 1], (len(env_ids), 6), device=self.device)
        positions = init_root_pos + self.env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        positions[..., 2] += self.lift_height
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = quat_mul(init_root_quat, orientations_delta)

        # velocities
        rand_samples = sample_uniform(self.velocity_range[:, 0], self.velocity_range[:, 1], (len(env_ids), 6), device=self.device)
        velocities = torch.cat([init_root_lin_vel, init_root_ang_vel], dim=-1) + rand_samples

        # set into the physics simulation
        self.asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        self.asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)
    
        init_joint_pos = motion.joint_pos[:, self.asset_joint_idx_motion]
        init_joint_vel = motion.joint_vel[:, self.asset_joint_idx_motion]

        joint_pos_noise = torch.randn_like(init_joint_pos).clamp(-1, 1) * self.init_joint_pos_noise
        joint_vel_noise = torch.randn_like(init_joint_vel).clamp(-1, 1) * self.init_joint_vel_noise

        init_joint_pos += joint_pos_noise
        init_joint_vel += joint_vel_noise

        joint_pos_limits = self.asset.data.soft_joint_pos_limits[env_ids]
        init_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        joint_vel_limits = self.asset.data.soft_joint_vel_limits[env_ids]
        init_joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

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
        
    class ref_motion_phase_noise(TrackObservation):
        def compute(self):
            return torch.randn(self.num_envs, 1, device=self.device)
    
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
                body_idx_motion = self.command_manager.dataset.body_names.index(body_name)
                body_idx_asset = self.command_manager.asset.body_names.index(body_name)

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
            body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
            body_pos_motion = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]
            diff = body_pos_motion - body_pos_asset
            # shape: [num_envs, num_tracking_bodies, 3]
            error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
            # shape: [num_envs, num_tracking_bodies]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

    class keypoint_pos_tracking_local_product(tracking_keypoint):
        def compute(self):
            body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
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
            body_lin_vel_asset = self.command_manager.asset.data.body_link_lin_vel_w[:, self.body_indices_asset]
            body_lin_vel_motion = self.command_manager.ref_body_lin_vel_w[:, self.body_indices_motion]
            diff = body_lin_vel_motion - body_lin_vel_asset
            # shape: [num_envs, num_tracking_bodies, 3]
            error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
            # shape: [num_envs, num_tracking_bodies]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

    class keypoint_lin_vel_tracking_local_product(tracking_keypoint):
        def compute(self):
            body_lin_vel_asset = self.command_manager.asset.data.body_link_lin_vel_w[:, self.body_indices_asset].clone()
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
            body_ang_vel_asset = self.command_manager.asset.data.body_link_ang_vel_w[:, self.body_indices_asset]
            body_ang_vel_motion = self.command_manager.ref_body_ang_vel_w[:, self.body_indices_motion]
            diff = body_ang_vel_motion - body_ang_vel_asset
            # shape: [num_envs, num_tracking_bodies, 3]
            error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
            # shape: [num_envs, num_tracking_bodies]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

    class keypoint_ang_vel_tracking_local_product(tracking_keypoint):
        def compute(self):
            body_ang_vel_asset = self.command_manager.asset.data.body_link_ang_vel_w[:, self.body_indices_asset]
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
                joint_idx_motion = self.command_manager.dataset.joint_names.index(joint_name)
                joint_idx_asset = self.command_manager.asset.joint_names.index(joint_name)

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
        def __init__(self, body_names: str | List[str], motion_vel_thres: float=0.1, soft_discount: float=1.0, **kwargs):
            super().__init__(**kwargs)
            self.motion_vel_thres = motion_vel_thres
            self.soft_discount = soft_discount
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
            self.env.discount[penalty.any(dim=1)] *= self.soft_discount
            return -penalty.float().mean(dim=1, keepdim=True)

        def debug_draw(self):
            if self.env.backend != "isaac":
                return
            # return

            self.vis_markers_pos_w.fill_(-100.0)
            self.vis_markers_pos_w[self.motion_no_contact] = self.command_manager.ref_body_pos_w[:, self.body_indices_motion][self.motion_no_contact]
            self.vis_markers.visualize(
                translations=self.vis_markers_pos_w.view(-1, 3),
            )

    class feet_contact_when_motion_contact(TrackReward):
        def __init__(
            self, 
            body_names: str | List[str],
            motion_vel_thres: float=0.1,
            motion_height_thres: float=0.1,
            feet_gravity_thres: float=0.9,
            feet_contact_thres: float=1.0,
            soft_discount: float=1.0,
            **kwargs
        ):
            # WARNING: this soft discount is not correct, do not use it
            super().__init__(**kwargs)
            self.motion_height_thres = motion_height_thres
            self.motion_vel_thres = motion_vel_thres

            self.feet_gravity_thres = feet_gravity_thres
            self.feet_contact_thres = feet_contact_thres

            self.soft_discount = soft_discount
            body_names = self.command_manager.asset.find_bodies(body_names)[1]
            self.body_indices_motion = []
            self.body_indices_sensor = []
            self.body_indices_asset = []
            for name in body_names:
                body_idx_motion = self.command_manager.dataset.body_names.index(name)
                body_idx_sensor = self.command_manager.contact_forces.body_names.index(name)
                body_idx_asset = self.command_manager.asset.body_names.index(name)

                self.body_indices_motion.append(body_idx_motion)
                self.body_indices_sensor.append(body_idx_sensor)
                self.body_indices_asset.append(body_idx_asset)
            
            self.motion_contact = torch.zeros((self.env.num_envs, len(body_names)), dtype=torch.bool, device=self.device)
            self.gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.env.num_envs, len(body_names), 3)
            
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
            contact_forces = self.command_manager.contact_forces.data.net_forces_w[:, self.body_indices_sensor]

            feet_quat = self.command_manager.asset.data.body_quat_w[:, self.body_indices_asset]
            feet_projected_gravity = quat_rotate_inverse(feet_quat, self.gravity_w)

            feet_height = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset][..., 2]
            feet_lin_vel = self.command_manager.asset.data.body_link_lin_vel_w[:, self.body_indices_asset].norm(dim=-1)

            in_contact = (contact_forces.norm(dim=-1) > self.feet_contact_thres) \
                & (feet_projected_gravity[..., 2] < -self.feet_gravity_thres) \
                & (feet_height < self.motion_height_thres) \
                & (feet_lin_vel < self.motion_vel_thres)
            # shape: [num_envs, num_feet]

            penalty = (~in_contact) & self.motion_contact
            self.env.discount[penalty.any(dim=1)] *= self.soft_discount
            reward = in_contact & self.motion_contact
            return reward.float().mean(dim=1, keepdim=True)
        
        def debug_draw(self):
            if self.env.backend != "isaac":
                return
            # return

            self.vis_markers_pos_w.fill_(-100.0)
            self.vis_markers_pos_w[self.motion_contact] = self.command_manager.ref_body_pos_w[:, self.body_indices_motion][self.motion_contact]
            self.vis_markers.visualize(
                translations=self.vis_markers_pos_w.view(-1, 3),
            )


    # class feet_tracking(Reward):
    #     def compute(self):
    #         in_contact = self.command_manager.contact_forces.data.current_contact_time[:, self.command_manager.feet_ids_sensor] > 0.01
    #         first_contact = self.command_manager.contact_forces.compute_first_contact(0.02)[:, self.command_manager.feet_ids_sensor]
    #         diff = self.command_manager.ref_body_pos_w[:, self.command_manager.feet_ids_motion] - self.command_manager.asset.data.body_link_pos_w[:, self.command_manager.feet_ids_asset]
    #         error = diff.square().sum(-1)
    #         return - (error * first_contact).sum(1, True)
    TrackRandomization = BaseRandomization["MotionTrackingCommand"]
    
    class keypoint_virtual_force(TrackRandomization):
        def __init__(
            self,
            stiffness_range: Tuple[float, float]=(20.0, 30.0),
            annealing_steps: int=500,
            pos_tolerance: float | Dict[str, float] = 0.0,
            vel_tolerance: float | Dict[str, float] = 0.0,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.tracking_body_indices_motion = self.command_manager.tracking_body_indices_motion
            self.tracking_body_indices_asset = self.command_manager.tracking_body_indices_asset

            self.stiffness_start = rand_uniform(*stiffness_range, (self.env.num_envs, 1, 1), self.device)
            self.stiffness = self.stiffness_start.clone()
            self.damping = self.stiffness.sqrt() * 2
            self.annealing_steps = annealing_steps

            tracking_body_names = self.command_manager.tracking_keypoint_names
            from isaaclab.utils.string import resolve_matching_names_values
            self.pos_tolerance = torch.zeros(len(tracking_body_names), device=self.device)
            self.vel_tolerance = torch.zeros(len(tracking_body_names), device=self.device)
            if isinstance(pos_tolerance, float):
                self.pos_tolerance.fill_(pos_tolerance)
            elif isinstance(pos_tolerance, DictConfig):
                indices, names, values = resolve_matching_names_values(dict(pos_tolerance), tracking_body_names)
                self.pos_tolerance[indices] = torch.tensor(values, device=self.device)
            else:
                raise ValueError(f"Invalid type for pos_tolerance: {type(pos_tolerance)}")
            if isinstance(vel_tolerance, float):
                self.vel_tolerance.fill_(vel_tolerance)
            elif isinstance(vel_tolerance, DictConfig):
                indices, names, values = resolve_matching_names_values(dict(vel_tolerance), tracking_body_names)
                self.vel_tolerance[indices] = torch.tensor(values, device=self.device)
            else:
                raise ValueError(f"Invalid type for vel_tolerance: {type(vel_tolerance)}")
        
        def update(self):
            self.stiffness = self.stiffness_start * max(1.0 - self.env.current_iter / self.annealing_steps, 0.0)
            self.damping = self.stiffness.sqrt() * 2

        def step(self, substep):
            ref_keypoint_pos_w = self.command_manager.ref_body_pos_w[:, self.tracking_body_indices_motion]
            ref_keypoint_lin_vel_w = self.command_manager.ref_body_lin_vel_w[:, self.tracking_body_indices_motion]
            robot_keypoint_pos_w = self.command_manager.asset.data.body_link_pos_w[:, self.tracking_body_indices_asset]
            robot_keypoint_lin_vel_w = self.command_manager.asset.data.body_link_lin_vel_w[:, self.tracking_body_indices_asset]

            # compute force in world frame
            diff_pos_w = ref_keypoint_pos_w - robot_keypoint_pos_w
            diff_lin_vel_w = ref_keypoint_lin_vel_w - robot_keypoint_lin_vel_w
            diff_pos_w = diff_pos_w * (diff_pos_w.abs() > self.pos_tolerance.unsqueeze(-1))
            diff_lin_vel_w = diff_lin_vel_w * (diff_lin_vel_w.abs() > self.vel_tolerance.unsqueeze(-1))
            self.forces_w = forces_w = self.stiffness * diff_pos_w + self.damping * diff_lin_vel_w
            body_quat_w = self.command_manager.asset.data.body_quat_w[:, self.tracking_body_indices_asset]
            forces_b = quat_rotate_inverse(body_quat_w, forces_w)

            # apply force to asset
            ext_forces_b = self.command_manager.asset._external_force_b
            ext_forces_b[:, self.tracking_body_indices_asset] += forces_b
            self.command_manager.asset.has_external_wrench = True
        
        def debug_draw(self):
            if self.env.backend != "isaac":
                return

            # draw force as vectors
            body_pos_w = self.command_manager.asset.data.body_link_pos_w[:, self.command_manager.tracking_body_indices_asset]
            self.env.debug_draw.vector(
                body_pos_w.reshape(-1, 3),
                (self.forces_w / self.stiffness).reshape(-1, 3) * 5.0,
                # orange
                color=(1.0, 0.5, 0.0, 1.0)
            )
            
    def update(self):
        # future ref motion for actor observation
        self.future_ref_motion = self.dataset.get_slice(self.motion_ids, self.t, steps=self.future_steps)
        # shape: [num_envs, len(future_steps), num_bodies/num_joints, 3/4/...]

        # Observations: future ref and diff to body frame
        self.root_quat_yaw_w = yaw_quat(self.asset.data.root_quat_w)

        root_quat_w = self.root_quat_yaw_w.unsqueeze(1).unsqueeze(1).repeat(1, self.num_future_steps, self.num_tracking_bodies, 1)
        root_pos_w = self.asset.data.root_pos_w.unsqueeze(1).unsqueeze(1).repeat(1, self.num_future_steps, self.num_tracking_bodies, 1)

        body_pos_w = self.asset.data.body_link_pos_w[:, self.tracking_body_indices_asset].unsqueeze(1).repeat(1, self.num_future_steps, 1, 1)
        body_lin_vel_w = self.asset.data.body_link_lin_vel_w[:, self.tracking_body_indices_asset].unsqueeze(1).repeat(1, self.num_future_steps, 1, 1)
        ref_body_pos_future_w = self.future_ref_motion.body_pos_w[..., self.tracking_body_indices_motion, :] + self.env.scene.env_origins[:, None, None, :]
        ref_body_lin_vel_future_w = self.future_ref_motion.body_lin_vel_w[..., self.tracking_body_indices_motion, :]
        # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]

        self.ref_body_pos_future_b = quat_rotate_inverse(root_quat_w, ref_body_pos_future_w - root_pos_w)
        self.ref_body_lin_vel_future_b = quat_rotate_inverse(root_quat_w, ref_body_lin_vel_future_w)
        self._diff_body_pos_future_b = quat_rotate_inverse(root_quat_w, ref_body_pos_future_w - body_pos_w)
        self._diff_body_lin_vel_future_b = quat_rotate_inverse(root_quat_w, ref_body_lin_vel_future_w - body_lin_vel_w)

        body_quat_w = self.asset.data.body_quat_w[:, self.tracking_body_indices_asset].unsqueeze(1).repeat(1, self.num_future_steps, 1, 1)
        body_ang_vel_w = self.asset.data.body_link_ang_vel_w[:, self.tracking_body_indices_asset].unsqueeze(1).repeat(1, self.num_future_steps, 1, 1)
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
        cur_tracking_body_pos_w = self.asset.data.body_link_pos_w[:, self.tracking_body_indices_asset]
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

        # robot_keypoints_w = self.all_marker_pos_w[0].reshape(-1, 3)
        # target_keypoints_w = self.all_marker_pos_w[1].reshape(-1, 3)
        # self.env.debug_draw.vector(
        #     robot_keypoints_w,
        #     target_keypoints_w - robot_keypoints_w,
        #     color=(0, 0, 1, 1)
        # )

        # in_contact = self.contact_forces.data.current_contact_time[:, self.feet_ids_sensor] > 0.01
        # feet_pos_asset = self.asset.data.body_link_pos_w[:, self.feet_ids_asset]
        # feet_pos_motion = self.ref_body_pos_w[:, self.feet_ids_motion]
        # diff = feet_pos_motion - feet_pos_asset
        # self.env.debug_draw.vector(
        #     self.asset.data.body_link_pos_w[:, self.feet_ids_asset].reshape(-1, 3),
        #     (diff * in_contact.unsqueeze(-1)).reshape(-1, 3),
        #     color=(0, 1, 0, 1),
        #     size=5.
        # )


class MotionTrackingDoor(MotionTrackingCommand):
    def __init__(
        self,
        cum_lost_contact_steps: int=1,
        contact_eef_name: str="right_wrist_yaw_link",
        contact_target_pos: Tuple[float, float, float]=(0.0, -0.6, 1.0),
        contact_eef_pos_offset: Tuple[float, float, float]=(0.06, 0.0, 0.0),
        contact_eef_pos_thres: float=0.3,
        contact_step_range: Tuple[int, int]=(220, 320),
        reset_range: Tuple[int, int]=(0, 200),
        **kwargs
    ):
        super().__init__(**kwargs, call_update=False)

        door_object_name = "door"
        door_body_name = "Door"
        wall_body_name = "Wall"
        door_joint_name = "door_joint"

        self.door: DoorArticulation = self.env.scene.articulations[door_object_name]
        self.wall_body_id_motion = self.dataset.body_names.index(wall_body_name)

        self.door_joint_id_motion = self.dataset.joint_names.index(door_joint_name)
        self.door_joint_id_asset = self.door.joint_names.index(door_joint_name)

        with torch.device(self.device):
            self._cum_error = torch.zeros(self.num_envs, 4)
            self.lost_contact_steps = torch.zeros(self.num_envs, dtype=torch.int32)

            self.contact_target_pos = torch.tensor(contact_target_pos).unsqueeze(0).expand(self.num_envs, -1)
            self.contact_eef_pos_offset = torch.tensor(contact_eef_pos_offset).unsqueeze(0).expand(self.num_envs, -1)
            self.contact_target_pos_w = torch.zeros(self.num_envs, 3)
            self.contact_eef_pos_w = torch.zeros(self.num_envs, 3)
            # shape: [num_envs, 3]

        self._cum_lost_contact_steps = cum_lost_contact_steps
        self.contact_eef_pos_thres = contact_eef_pos_thres
        self.contact_step_range = contact_step_range

        self.reset_range = reset_range

        self.contact_parent_body_id_asset = self.door.body_names.index(door_body_name)
        self.eef_idx_sensor = self.contact_forces.body_names.index(contact_eef_name)
        self.eef_idx_asset = self.asset.body_names.index(contact_eef_name)

        self._init_debug_draw()
        self.update()
    
    def _sample_motions(self, env_ids: torch.Tensor) -> None:
        super()._sample_motions(env_ids)
        start_t = torch.randint(*self.reset_range, (len(env_ids),), device=self.device)
        self.t[env_ids] = start_t
        if not self.env.training:
            self.t[env_ids] = 0

    def sample_init(self, env_ids: torch.Tensor) -> None:
        super().sample_init(env_ids)

        motion: MotionData = self._motion_reset
        init_door_pos = motion.body_pos_w[:, self.wall_body_id_motion]
        init_door_pos[:, 2].fill_(0.0)
        init_door_quat = motion.body_quat_w[:, self.wall_body_id_motion]


        init_door_root_state_w = self.door.data.default_root_state[env_ids]
        init_door_root_state_w[:, 0:3] = init_door_pos + self.env.scene.env_origins[env_ids]
        init_door_root_state_w[:, 3:7] = init_door_quat
        init_door_root_state_w[:, 7:] = 0.0

        self.door.write_root_link_pose_to_sim(init_door_root_state_w[:, :7], env_ids=env_ids)
        self.door.write_root_link_velocity_to_sim(init_door_root_state_w[:, 7:], env_ids=env_ids)
        
        init_door_joint_pos = motion.joint_pos[:, self.door_joint_id_motion].unsqueeze(-1)
        init_door_joint_vel = motion.joint_vel[:, self.door_joint_id_motion].unsqueeze(-1)
        self.door.write_joint_position_to_sim(init_door_joint_pos, env_ids=env_ids, joint_ids=[self.door_joint_id_asset])
        self.door.write_joint_velocity_to_sim(init_door_joint_vel, env_ids=env_ids, joint_ids=[self.door_joint_id_asset])

    def reset(self, env_ids: torch.Tensor) -> None:
        super().reset(env_ids)
        self.lost_contact_steps[env_ids] = 0

    TrackDoorObservation = BaseObservation["MotionTrackingDoor"]
    
    class door_pos_b(TrackDoorObservation):
        def __init__(self, noise_std: float=0.0, **kwargs):
            super().__init__(**kwargs)
            self.noise_std = max(0.0, noise_std)

        def compute(self):
            door_pos_w = self.command_manager.door.data.root_pos_w
            robot_pos_w = self.command_manager.asset.data.root_pos_w
            robot_quat_w = self.command_manager.asset.data.root_quat_w
            robot_quat_yaw_w = yaw_quat(robot_quat_w)
            door_pos_b = quat_rotate_inverse(robot_quat_yaw_w, door_pos_w - robot_pos_w)
            door_pos_b = door_pos_b[:, :2]
            if self.noise_std > 0.0:
                door_pos_b = door_pos_b + torch.randn_like(door_pos_b).clamp(-3., 3.) * self.noise_std
            return door_pos_b
        
    class root_yaw(TrackDoorObservation):
        def __init__(self, noise_std: float=0.0, **kwargs):
            super().__init__(**kwargs)
            self.noise_std = max(0.0, noise_std)

        def compute(self):
            robot_quat_w = self.command_manager.asset.data.root_quat_w
            door_quat_w = self.command_manager.door.data.root_quat_w
            robot_yaw_w = yaw_from_quat(robot_quat_w)
            door_yaw_w = yaw_from_quat(door_quat_w)
            root_yaw = wrap_to_pi(robot_yaw_w - door_yaw_w - torch.pi)
            if self.noise_std > 0.0:
                root_yaw = root_yaw + torch.randn_like(root_yaw).clamp(-3., 3.) * self.noise_std
            return root_yaw.unsqueeze(-1)
        
    class door_joint_pos(TrackDoorObservation):
        def compute(self):
            door_joint_pos = self.command_manager.door.data.joint_pos[:, self.command_manager.door_joint_id_asset]
            return - door_joint_pos.unsqueeze(-1)
    
    class door_joint_vel(TrackDoorObservation):
        def compute(self):
            door_joint_vel = self.command_manager.door.data.joint_vel[:, self.command_manager.door_joint_id_asset]
            return - door_joint_vel.unsqueeze(-1)
        
    class door_joint_torque(TrackDoorObservation):
        def compute(self):
            door_joint_torque = self.command_manager.door.data.applied_torque[:, self.command_manager.door_joint_id_asset]
            return - door_joint_torque.unsqueeze(-1)
    
    class ref_door_joint_pos_future(TrackDoorObservation):
        def compute(self):
            ref_door_joint_pos = self.command_manager.future_ref_motion.joint_pos[:, :, self.command_manager.door_joint_id_motion]
            return - ref_door_joint_pos.view(self.num_envs, -1)
    
    TrackDoorReward = BaseReward["MotionTrackingDoor"]
    
    class door_joint_pos_tracking(TrackDoorReward):
        def __init__(self, sigma=0.1, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma

        def compute(self):
            door_joint_pos = self.command_manager.asset.data.joint_pos[:, self.command_manager.door_joint_id_asset]
            ref_door_joint_pos = self.command_manager.ref_door_joint_pos
            error = (door_joint_pos - ref_door_joint_pos).square()
            return torch.exp(- error / self.sigma).unsqueeze(-1)
    
    class eef_door_contact_pos(TrackDoorReward):
        def __init__(self, sigma: float=0.1, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma
            self.contact_step_range = self.command_manager.contact_step_range
            self.t = self.command_manager.t
            
            self.in_range = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.eef_pos_error = torch.zeros(self.num_envs, device=self.device)
        
        def update(self):
            self.in_range[:] = (self.t >= self.contact_step_range[0]) & (self.t <= self.contact_step_range[1])

            eef_pos_diff = self.command_manager.contact_eef_pos_w - self.command_manager.contact_target_pos_w
            self.eef_pos_error[:] = eef_pos_diff.norm(dim=-1)

        def compute(self):
            rew = torch.exp(- self.eef_pos_error / self.sigma)
            rew *= self.in_range
            return rew.unsqueeze(-1)
        
    class eef_door_contact(TrackDoorReward):
        def __init__(self, pos_thres: float=0.3, force_thres: float=1.0, **kwargs):
            super().__init__(**kwargs)
            self.pos_thres = pos_thres
            self.force_thres = force_thres
            self.contact_step_range = self.command_manager.contact_step_range
            self.t = self.command_manager.t
            self.eef_idx_sensor = self.command_manager.eef_idx_sensor
            
            self.in_range = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.eef_pos_error = torch.zeros(self.num_envs, device=self.device)
            self.contact_force = torch.zeros(self.num_envs, device=self.device)
        
        def update(self):
            self.in_range[:] = (self.t >= self.contact_step_range[0]) & (self.t <= self.contact_step_range[1])

            eef_pos_diff = self.command_manager.contact_eef_pos_w - self.command_manager.contact_target_pos_w
            self.eef_pos_error[:] = eef_pos_diff.norm(dim=-1)

            contact_force = self.command_manager.contact_forces.data.net_forces_w[:, self.eef_idx_sensor]
            self.contact_force[:] = contact_force.norm(dim=-1)

        def compute(self):
            rew = (self.eef_pos_error < self.pos_thres) & (self.contact_force > self.force_thres)
            rew *= self.in_range
            return rew.float().unsqueeze(-1)
        
    
    class feet_contact_force_xy(TrackDoorReward):
        def __init__(self, thres: float=1.0, **kwargs):
            super().__init__(**kwargs)
            self.contact_forces: ContactSensor = self.env.scene["contact_forces"]
            self.feet_ids = self.contact_forces.find_bodies(".*_ankle_roll_link")[0]
            self.thres = thres
        
        def compute(self):
            contact_forces = self.contact_forces.data.net_forces_w[:, self.feet_ids]
            contact_forces = (contact_forces[:, :, :2].norm(dim=-1) - self.thres).clamp_min(0.0)
            return - contact_forces.mean(1, True)
    
    class feet_contact_force_xy_log(TrackDoorReward):
        def __init__(self, thres: float=1.0, **kwargs):
            super().__init__(**kwargs)
            self.contact_forces: ContactSensor = self.env.scene["contact_forces"]
            self.feet_ids = self.contact_forces.find_bodies(".*_ankle_roll_link")[0]
            self.thres = thres
        
        def compute(self):
            contact_forces = self.contact_forces.data.net_forces_w[:, self.feet_ids]
            contact_forces = (contact_forces[:, :, :2].norm(dim=-1) - self.thres).clamp_min(0.0)
            return - torch.log(contact_forces + 1.0).mean(1, True)
    
    TrackDoorRandomization = BaseRandomization["MotionTrackingDoor"]
    
    class door_joint_randomization(TrackDoorRandomization):
        def __init__(
            self,
            friction_range: Tuple[float, float]=(0.0, 0.1),
            damping_range: Tuple[float, float]=(1.0, 10.0),
            armature_range: Tuple[float, float]=(0.0, 0.02),
            **kwargs
        ):
            super().__init__(**kwargs)
            self.door = self.command_manager.door
            self.friction_range = friction_range
            self.damping_range = damping_range
            self.armature_range = armature_range

            self.door_joint_id_asset = self.command_manager.door_joint_id_asset
        
        def startup(self):
            door_armature = rand_uniform(*self.armature_range, (self.door.num_instances, 1), self.device)
            self.door.write_joint_armature_to_sim(door_armature, joint_ids=[self.door_joint_id_asset])

        def reset(self, env_ids: torch.Tensor):
            door_friction = rand_uniform(*self.friction_range, (len(env_ids),), self.device)
            door_damping = rand_uniform(*self.damping_range, (len(env_ids),), self.device)

            # if not self.env.training:
            #     first_half = env_ids < self.num_envs // 2
            #     second_half = env_ids >= self.num_envs // 2

            #     door_friction[first_half] = 1.0
            #     door_damping[first_half] = 5.0

            #     door_friction[second_half] = 10.0
            #     door_damping[second_half] = 20.0

            #     print(f"env_ids: {env_ids}")
            #     print(f"door_friction: {door_friction}")
            #     print(f"door_damping: {door_damping}")

            self.door.friction[env_ids] = door_friction
            self.door.damping[env_ids] = door_damping
    
    class door_body_randomization(TrackDoorRandomization):
        def __init__(
            self,
            static_friction_range: Tuple[float, float]=(0.6, 1.0),
            dynamic_friction_range: Tuple[float, float]=(0.6, 1.0),
            restitution_range: Tuple[float, float]=(0.0, 0.2),
            mass_range: Tuple[float, float]=(1.0, 10.0),
            body_name: str="Door",
            **kwargs
        ):
            super().__init__(**kwargs)
            self.door = self.command_manager.door

            self.body_ids, _ = self.command_manager.door.find_bodies(body_name)
            assert len(self.body_ids) == 1

            self.mass_range = mass_range

            self.all_indices_cpu = torch.arange(self.door.num_instances)

            num_shapes_per_body = []
            for link_path in self.door.root_physx_view.link_paths[0]:
                link_physx_view = self.door._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                num_shapes_per_body.append(link_physx_view.max_shapes)
            cumsum = np.cumsum([0,] + num_shapes_per_body)
            self.shape_ids = torch.cat([
                torch.arange(cumsum[i], cumsum[i+1]) 
                for i in self.body_ids
            ])

            self.num_buckets = 64
            self.static_friction_buckets = rand_uniform(*tuple(static_friction_range), (self.num_buckets,), "cpu")
            self.dynamic_friction_buckets = rand_uniform(*tuple(dynamic_friction_range), (self.num_buckets,), "cpu")
            self.restitution_buckets = rand_uniform(*tuple(restitution_range), (self.num_buckets,), "cpu")

        def startup(self):
            masses = self.door.data.default_mass.clone()
            inertias = self.door.data.default_inertia.clone()
            new_masses = rand_uniform(*self.mass_range, (self.door.num_instances, 1), "cpu")
            
            scale = new_masses / masses[:, self.body_ids]
            masses[:, self.body_ids] *= scale
            inertias[:, self.body_ids] *= scale.unsqueeze(-1)
            self.door.root_physx_view.set_masses(masses, self.all_indices_cpu)
            self.door.root_physx_view.set_inertias(inertias, self.all_indices_cpu)
            assert torch.allclose(self.door.root_physx_view.get_masses(), masses, atol=1e-4)
            assert torch.allclose(self.door.root_physx_view.get_inertias(), inertias, atol=1e-4)

            materials = self.door.root_physx_view.get_material_properties().clone()
            shape = (self.door.num_instances, len(self.shape_ids))
            materials[:, self.shape_ids, 0] = self.static_friction_buckets[torch.randint(0, self.num_buckets, shape)]
            materials[:, self.shape_ids, 1] = self.dynamic_friction_buckets[torch.randint(0, self.num_buckets, shape)]
            materials[:, self.shape_ids, 2] = self.restitution_buckets[torch.randint(0, self.num_buckets, shape)]
            self.door.root_physx_view.set_material_properties(materials.flatten(), self.all_indices_cpu)
            assert torch.allclose(self.door.root_physx_view.get_material_properties(), materials, atol=1e-4)
            
    def update(self):
        super().update()
        # Reward: reference door joint position
        self.ref_door_joint_pos = self.current_ref_motion.joint_pos[:, self.door_joint_id_motion]
        self.ref_door_joint_vel = self.current_ref_motion.joint_vel[:, self.door_joint_id_motion]
        # shape: [num_envs]

        # Termination: contact with door
        contact_parent_pos_w = self.door.data.body_link_pos_w[:, self.contact_parent_body_id_asset]
        contact_parent_quat_w = self.door.data.body_quat_w[:, self.contact_parent_body_id_asset]
        self.contact_target_pos_w[:] = contact_parent_pos_w + quat_apply(contact_parent_quat_w, self.contact_target_pos)
        eef_pos_w = self.asset.data.body_link_pos_w[:, self.eef_idx_asset]
        eef_quat_w = self.asset.data.body_quat_w[:, self.eef_idx_asset]
        self.contact_eef_pos_w[:] = eef_pos_w + quat_apply(eef_quat_w, self.contact_eef_pos_offset)
        pos_error = (self.contact_target_pos_w - self.contact_eef_pos_w).norm(dim=-1) # shape: [num_envs]
        
        contact_force = self.contact_forces.data.net_forces_w[:, self.eef_idx_sensor]
        in_contact = (contact_force.norm(dim=-1) > 1.0) & (pos_error < self.contact_eef_pos_thres)
        in_range = (self.t >= self.contact_step_range[0]) & (self.t <= self.contact_step_range[1])

        increment_mask = in_range & ~in_contact
        self.lost_contact_steps[increment_mask] += 1
        self.lost_contact_steps[~increment_mask] = 0

        self._cum_error[:, 3] = self.lost_contact_steps / self._cum_lost_contact_steps
    
    def _init_debug_draw(self):
        super()._init_debug_draw()
        
        if self.env.backend != "isaac":
            return
        
        from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
        import isaaclab.sim as sim_utils
        vis_markers_cfg = VisualizationMarkersCfg(
            prim_path=f"/World/EefContact",
            markers={
                "eef": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.5),
                        metallic=1.0
                    )
                ),
                "target": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.5, 1.0),
                        metallic=1.0
                    )
                ),
            }
        )
        self.eef_contact_markers = VisualizationMarkers(vis_markers_cfg)
        self.eef_contact_markers_indices = [0] * self.num_envs + [1] * self.num_envs
        self.eef_contact_markers_pos_w = torch.zeros(2, self.num_envs, 3)

    def debug_draw(self):
        super().debug_draw()
        
        if self.env.backend != "isaac":
            return
        
        self.eef_contact_markers_pos_w[0] = self.contact_eef_pos_w
        self.eef_contact_markers_pos_w[1] = self.contact_target_pos_w
        in_range = (self.t >= self.contact_step_range[0]) & (self.t <= self.contact_step_range[1])
        self.eef_contact_markers_pos_w[:, ~in_range] = -1000
        
        self.eef_contact_markers.visualize(
            translations=self.eef_contact_markers_pos_w.view(-1, 3),
            marker_indices=self.eef_contact_markers_indices,
        )


class MotionTrackingBox(MotionTrackingCommand):
    def __init__(
        self, 
        # termination
        cum_lost_contact_steps: int=1,
        contact_eef_name: str=".*_wrist_yaw_link",
        contact_eef_pos_offset: List[Tuple[float, float, float]]=[(0.06, 0.0, 0.0), (0.06, 0.0, 0.0)], 
        contact_eef_pos_thres: float=0.1,
        contact_eef_ori_thres: float=0.5,
        contact_eef_frc_thres: float=1.0,
        cum_box_pos_error_scale: float=0.2,
        cum_box_ori_error_scale: float=0.2,
        # reset
        reset_range: Tuple[int, int]=(0, 100),
        box_reset_offset: Tuple[float, float, float]=(0.1, 0.0, 0.0),
        **kwargs
    ):
        super().__init__(**kwargs, call_update=False)
        box_object_name = "box"
        box_body_name = "box"
        
        self.box: RigidObject = self.env.scene.rigid_objects[box_object_name]

        self.box_body_id_motion = self.dataset.body_names.index(box_body_name)
        self.box_body_id_asset = self.box.body_names.index(box_body_name)

        scale = getattr(self.box.cfg.spawn, "scale", torch.ones(self.num_envs, 3))
        box_scale: torch.Tensor = scale.to(self.device)
        box_size = torch.tensor(self.box.cfg.spawn.size, device=self.device)
        self.box_size = box_size * box_scale
        self.contact_target_pos_offset = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.contact_target_pos_offset[:] = self.box_size.unsqueeze(1) / 2
        self.contact_target_pos_offset[:, 0, 1] = -0.15
        self.contact_target_pos_offset[:, 1, 1] = 0.15

        _, contact_eef_names = self.asset.find_bodies(contact_eef_name)
        assert len(contact_eef_names) == 2 == len(contact_eef_pos_offset)

        with torch.device(self.device):
            self._cum_error = torch.zeros(self.num_envs, 6)
            self.lost_contact_steps = torch.zeros(self.num_envs, dtype=torch.int32)

            self.contact_eef_pos_offset = torch.tensor(contact_eef_pos_offset).unsqueeze(0).expand(self.num_envs, -1, -1)
            self.contact_target_pos_w = torch.zeros(self.num_envs, len(contact_eef_names), 3)
            self.contact_eef_pos_w = torch.zeros(self.num_envs, len(contact_eef_names), 3)
            self.eef_contact_force = torch.zeros(self.num_envs, len(contact_eef_names), 3)
            # shape: [num_envs, num_contact_eefs, 3]

            self.box_reset_offset = torch.tensor(box_reset_offset).unsqueeze(0)
        
        self._cum_lost_contact_steps = cum_lost_contact_steps
        self._cum_box_pos_error_scale = cum_box_pos_error_scale
        self._cum_box_ori_error_scale = cum_box_ori_error_scale
        
        self.reset_range = reset_range

        self.contact_eef_pos_thres = contact_eef_pos_thres
        self.contact_eef_ori_thres = contact_eef_ori_thres
        self.contact_eef_frc_thres = contact_eef_frc_thres
        
        self.eef_idx_asset = []
        self.eef_idx_sensor = []
        for eef_name in contact_eef_names:
            self.eef_idx_asset.append(self.asset.body_names.index(eef_name))
            self.eef_idx_sensor.append(self.contact_forces.body_names.index(eef_name))
        
        data_path = kwargs["data_path"]
        from pathlib import Path
        import active_adaptation
        active_adaptation_path = Path(active_adaptation.__file__).parent.parent
        data_path = active_adaptation_path / Path(data_path)
        motion_data = np.load(data_path / "motion.npz")
        box_contact = motion_data["box_contact"]
        self.box_contact = torch.from_numpy(box_contact).to(self.device).type(torch.bool).squeeze(-1)
        # shape: [n_steps]
        
        self._init_debug_draw()
        self.update()
    
    def _sample_motions(self, env_ids: torch.Tensor) -> None:
        super()._sample_motions(env_ids)
        start_t = torch.randint(self.reset_range[0], self.reset_range[1], (len(env_ids),), device=self.device)
        self.t[env_ids] = start_t

    def sample_init(self, env_ids: torch.Tensor) -> None:
        super().sample_init(env_ids)

        motion: MotionData = self._motion_reset
        init_box_pos = motion.body_pos_w[:, self.box_body_id_motion]
        init_box_quat = motion.body_quat_w[:, self.box_body_id_motion]
        init_box_pos[:, 2] = self.box_size[env_ids, 2] / 2

        init_box_state_w = self.box.data.default_root_state[env_ids]
        init_box_state_w[:, 0:3] = init_box_pos + self.env.scene.env_origins[env_ids] + self.box_reset_offset
        init_box_state_w[:, 3:7] = init_box_quat
        init_box_state_w[:, 7:] = 0.0
        
        self.box.write_root_link_pose_to_sim(init_box_state_w[:, :7], env_ids=env_ids)
        self.box.write_root_link_velocity_to_sim(init_box_state_w[:, 7:], env_ids=env_ids)

    def reset(self, env_ids: torch.Tensor) -> None:
        super().reset(env_ids)
        self.lost_contact_steps[env_ids] = 0

    TrackBoxObservation = BaseObservation["MotionTrackingBox"]

    class box_pos_b(TrackBoxObservation):
        def __init__(self, noise_std: float=0.0, **kwargs):
            super().__init__(**kwargs)
            self.noise_std = max(0.0, noise_std)

        def compute(self):
            box_pos_w = self.command_manager.box.data.root_pos_w
            robot_pos_w = self.command_manager.asset.data.root_pos_w
            robot_quat_w = self.command_manager.asset.data.root_quat_w
            robot_quat_yaw_w = yaw_quat(robot_quat_w)
            box_pos_b = quat_rotate_inverse(robot_quat_yaw_w, box_pos_w - robot_pos_w)[..., :2]
            if self.noise_std > 0.0:
                box_pos_b = box_pos_b + torch.randn_like(box_pos_b).clamp(-3., 3.) * self.noise_std
            return box_pos_b
        
    class box_yaw(TrackBoxObservation):
        def __init__(self, noise_std: float=0.0, **kwargs):
            super().__init__(**kwargs)
            self.noise_std = max(0.0, noise_std)

        def compute(self):
            robot_quat_w = self.command_manager.asset.data.root_quat_w
            box_quat_w = self.command_manager.box.data.root_quat_w
            robot_yaw_w = yaw_from_quat(robot_quat_w)
            box_yaw_w = yaw_from_quat(box_quat_w)
            box_yaw = wrap_to_pi(box_yaw_w - robot_yaw_w)
            if self.noise_std > 0.0:
                box_yaw = box_yaw + torch.randn_like(box_yaw).clamp(-3., 3.) * self.noise_std
            return box_yaw.unsqueeze(-1)
        
        
    class box_lin_vel_b(TrackBoxObservation):
        def compute(self):
            box_vel_w = self.command_manager.box.data.root_lin_vel_w
            robot_quat_w = self.command_manager.asset.data.root_quat_w
            box_vel_w_b = quat_rotate_inverse(robot_quat_w, box_vel_w)
            return box_vel_w_b
        
    class box_ang_vel_b(TrackBoxObservation):
        def compute(self):
            box_ang_vel_w = self.command_manager.box.data.root_ang_vel_w
            robot_quat_w = self.command_manager.asset.data.root_quat_w
            box_ang_vel_w_b = quat_rotate_inverse(robot_quat_w, box_ang_vel_w)
            return box_ang_vel_w_b
        
    class diff_box_pos_future(TrackBoxObservation):
        def compute(self):
            ref_box_pos_future_w = self.command_manager.future_ref_motion.body_pos_w[:, :, self.command_manager.box_body_id_motion]
            cur_box_pos_w = self.command_manager.box.data.root_pos_w.unsqueeze(1)
            diff_box_pos_future_w = ref_box_pos_future_w - cur_box_pos_w
            # shape: [num_envs, num_future_steps, 2]

            robot_quat_w = self.command_manager.asset.data.root_quat_w
            robot_quat_yaw_w = yaw_quat(robot_quat_w).unsqueeze(1)
            diff_box_pos_future_w_b = quat_rotate_inverse(robot_quat_yaw_w, diff_box_pos_future_w)
            return diff_box_pos_future_w_b[..., :2].reshape(self.num_envs, -1)
        
    class box_friction(TrackBoxObservation):
        def compute(self):
            raise NotImplementedError
        
        
    class box_contact_future(TrackBoxObservation):
        def compute(self):
            return self.command_manager.future_ref_box_contact.float()
    
    class eef_contact_diff_pos_b(TrackBoxObservation):
        def compute(self):
            eef_contact_diff_pos_w = self.command_manager.contact_target_pos_w - self.command_manager.contact_eef_pos_w
            robot_quat_w = self.command_manager.asset.data.root_quat_w.unsqueeze(1)
            eef_contact_diff_b = quat_rotate_inverse(robot_quat_w, eef_contact_diff_pos_w)
            return eef_contact_diff_b.view(self.num_envs, -1)
    
    class eef_contact_diff_ori_b(TrackBoxObservation):
        def compute(self):
            eef_contact_diff_euler = self.command_manager.eef_target_euler_xyz - self.command_manager.eef_euler_xyz
            eef_contact_diff_mat = matrix_from_euler(eef_contact_diff_euler, "XYZ")
            return eef_contact_diff_mat[:, :, :2, :].reshape(self.num_envs, -1)
        
    TrackBoxReward = BaseReward["MotionTrackingBox"]
    
    class box_pos_tracking(TrackBoxReward):
        def __init__(self, sigma: float=0.1, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma

        def compute(self):
            box_pos_w = self.command_manager.box.data.root_pos_w
            ref_box_pos_w = self.command_manager.ref_box_pos_w
            diff_box_pos_w_b = ref_box_pos_w - box_pos_w
            error = diff_box_pos_w_b.norm(dim=-1)
            return torch.exp(-error / self.sigma).unsqueeze(-1)
        
    
    class box_ori_tracking(TrackBoxReward):
        def __init__(self, sigma: float=0.1, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma

        def compute(self):
            box_quat_w = self.command_manager.box.data.root_quat_w
            ref_box_quat_w = self.command_manager.ref_box_quat_w
            diff_box_quat_w = quat_mul(quat_conjugate(box_quat_w), ref_box_quat_w)
            error = axis_angle_from_quat(diff_box_quat_w).norm(dim=-1)
            return torch.exp(-error / self.sigma).unsqueeze(-1)
        
    class eef_contact_pos(TrackBoxReward):
        def __init__(self, sigma: float=0.1, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma
            self.box_contact = self.command_manager.box_contact
            
            self.eef_pos_error = torch.zeros(self.num_envs, 2, device=self.device)
        
        def update(self):
            self.in_range = self.command_manager.ref_box_contact
            eef_pos_diff = self.command_manager.contact_eef_pos_w - self.command_manager.contact_target_pos_w
            self.eef_pos_error[:] = eef_pos_diff.norm(dim=-1)

        def compute(self):
            rew = torch.exp(-self.eef_pos_error / self.sigma).mean(dim=-1)
            return (rew * self.in_range.float()).unsqueeze(-1)
        
    class eef_contact_ori(TrackBoxReward):
        def __init__(self, sigma: float=0.1, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma
            self.box_contact = self.command_manager.box_contact
            
            self.eef_ori_error = torch.zeros(self.num_envs, 2, 3, device=self.device)
        
        def update(self):
            self.in_range = self.command_manager.ref_box_contact
            eef_ori_diff = wrap_to_pi(self.command_manager.eef_euler_xyz - self.command_manager.eef_target_euler_xyz)
            self.eef_ori_error[:] = eef_ori_diff.abs()

        def compute(self):
            rew = torch.exp(-self.eef_ori_error / self.sigma).mean(dim=(1, 2))
            return (rew * self.in_range.float()).unsqueeze(-1)
        
    class eef_contact_all(TrackBoxReward):
        def __init__(
            self,
            pos_thres: float=0.05,
            ori_thres: float=0.1,
            frc_thres: float=2.0,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.box_contact = self.command_manager.box_contact
            
            self.eef_pos_error = torch.zeros(self.num_envs, 2, device=self.device)
            self.eef_ori_error = torch.zeros(self.num_envs, 2, 3, device=self.device)
            self.eef_frc = torch.zeros(self.num_envs, 2, device=self.device)

            self.pos_thres = pos_thres
            self.ori_thres = ori_thres
            self.frc_thres = frc_thres
        
        def update(self):
            self.in_range = self.command_manager.ref_box_contact

            eef_pos_diff = self.command_manager.contact_eef_pos_w - self.command_manager.contact_target_pos_w
            eef_ori_diff = wrap_to_pi(self.command_manager.eef_euler_xyz - self.command_manager.eef_target_euler_xyz)
            eef_frc = self.command_manager.eef_contact_force

            self.eef_pos_error[:] = eef_pos_diff.norm(dim=-1)
            self.eef_ori_error[:] = eef_ori_diff.abs()
            self.eef_frc[:] = eef_frc.norm(dim=-1)

        def compute(self):
            contact_all = (
                (self.eef_pos_error < self.pos_thres)
                & (self.eef_ori_error.abs() < self.ori_thres).all(dim=-1)
                & (self.eef_frc > self.frc_thres)
            ).float().mean(dim=-1)
            # shape: [num_envs]
            return (contact_all * self.in_range.float()).unsqueeze(-1)
        
    TrackBoxRandomization = BaseRandomization["MotionTrackingBox"]
    
    # TODO: add randomization for box friction and mass

    def update(self):
        super().update()
        self.ref_box_pos_w = self.current_ref_motion.body_pos_w[:, self.box_body_id_motion] + self.env.scene.env_origins
        self.ref_box_quat_w = self.current_ref_motion.body_quat_w[:, self.box_body_id_motion]
        # shape: [num_envs, 3]
        idx = (self.dataset.starts[self.motion_ids] + self.t).unsqueeze(1) + self.future_steps.unsqueeze(0)
        idx.clamp_max_(self.dataset.ends.unsqueeze(1)[self.motion_ids] - 1)
        self.future_ref_box_contact = self.box_contact[idx]
        self.ref_box_contact = self.future_ref_box_contact[:, 0]
        
        # Termination: contact and box far
        # eef target pos
        box_pos_w = self.box.data.root_pos_w.unsqueeze(1).expand(-1, 2, -1)
        box_quat_w = self.box.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
        self.contact_target_pos_w[:] = box_pos_w + quat_apply(
            box_quat_w.reshape(-1, 4), self.contact_target_pos_offset.reshape(-1, 3)
        ).view(self.num_envs, 2, 3)

        eef_pos_w = self.asset.data.body_link_pos_w[:, self.eef_idx_asset]
        eef_quat_w = self.asset.data.body_quat_w[:, self.eef_idx_asset]
        self.contact_eef_pos_w[:] = eef_pos_w + quat_apply(eef_quat_w, self.contact_eef_pos_offset)

        # eef target ori
        target_quat_w = yaw_quat(self.box.data.root_quat_w)
        eef_target_euler_xyz = euler_xyz_from_quat(target_quat_w)
        self.eef_target_euler_xyz = torch.stack(eef_target_euler_xyz, dim=1).unsqueeze(1)
        self.eef_target_euler_xyz[:, :, 2] += torch.pi

        eef_quat_w = self.asset.data.body_quat_w[:, self.eef_idx_asset]
        eef_euler_xyz = euler_xyz_from_quat(eef_quat_w.view(-1, 4))
        self.eef_euler_xyz = torch.stack(eef_euler_xyz, dim=1).view(self.num_envs, 2, 3)

        # eef contact
        self.eef_contact_force[:] = self.contact_forces.data.net_forces_w[:, self.eef_idx_sensor]

        eef_pos_diff = self.contact_eef_pos_w - self.contact_target_pos_w
        eef_euler_diff = wrap_to_pi(self.eef_euler_xyz - self.eef_target_euler_xyz)
        eef_contact_frc_norm = self.eef_contact_force.norm(dim=-1)

        in_contact = (eef_pos_diff.norm(dim=-1) < self.contact_eef_pos_thres) \
            & (eef_euler_diff.abs() < self.contact_eef_ori_thres).all(dim=-1) \
            & (eef_contact_frc_norm > self.contact_eef_frc_thres)
        in_range = self.ref_box_contact

        # if not self.env.training and in_range[0]:
        #     print(eef_pos_diff[0].norm(dim=-1))
        #     print(eef_euler_diff[0].abs().max(dim=-1).values)
        #     print(self.lost_contact_steps[0])
        #     print(self._cum_error[0])

        increment_mask = (in_range.unsqueeze(-1) & ~in_contact).any(dim=-1)
        self.lost_contact_steps[increment_mask] += 1
        self.lost_contact_steps[~increment_mask] = 0
        self._cum_error[:, 3] = self.lost_contact_steps / self._cum_lost_contact_steps

        box_pos_error = (self.ref_box_pos_w - self.box.data.root_pos_w).norm(dim=-1)
        box_quat_diff = quat_mul(quat_conjugate(self.box.data.root_quat_w), self.ref_box_quat_w)
        box_ori_error = axis_angle_from_quat(box_quat_diff).norm(dim=-1)
        self._cum_error[:, 4] = box_pos_error / self._cum_box_pos_error_scale
        self._cum_error[:, 5] = box_ori_error / self._cum_box_ori_error_scale
    
    def _init_debug_draw(self):
        super()._init_debug_draw()

        if self.env.backend != "isaac":
            return
        
        from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
        import isaaclab.sim as sim_utils
        vis_markers_cfg = VisualizationMarkersCfg(
            prim_path=f"/World/EefContact",
            markers={
                "left_eef": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.5),
                        metallic=1.0,
                    )
                ),
                "right_eef": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.5, 1.0),
                        metallic=1.0,
                    )
                ),
            }
        )
        self.eef_contact_markers = VisualizationMarkers(vis_markers_cfg)
        self.eef_contact_markers_indices = [0, 1] * (self.num_envs * 2)
        self.eef_contact_markers_pos_w = torch.zeros(self.num_envs, 2, 2, 3)

        vis_markers_cfg = VisualizationMarkersCfg(
            prim_path=f"/World/BoxPos",
            markers={
                "box": sim_utils.SphereCfg(
                    radius=0.02,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.5),
                        metallic=1.0
                    )
                ),
                "target": sim_utils.SphereCfg(
                    radius=0.02,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.5, 1.0),
                        metallic=1.0
                    )
                ),
            }
        )
        self.box_pos_markers = VisualizationMarkers(vis_markers_cfg)
        self.box_pos_markers_indices = [0] * self.num_envs + [1] * self.num_envs
        self.box_pos_markers_pos_w = torch.zeros(2, self.num_envs, 3)

        self.box_vis_offset = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        
    def debug_draw(self):
        super().debug_draw()

        if self.env.backend != "isaac":
            return
        
        self.eef_contact_markers_pos_w[:, 0, :, :] = self.contact_eef_pos_w
        self.eef_contact_markers_pos_w[:, 1, :, :] = self.contact_target_pos_w
        
        self.box_pos_markers_pos_w[0] = self.ref_box_pos_w + self.box_vis_offset
        self.box_pos_markers_pos_w[1] = self.box.data.root_pos_w + self.box_vis_offset
        
        self.eef_contact_markers.visualize(
            translations=self.eef_contact_markers_pos_w.view(-1, 3),
            marker_indices=self.eef_contact_markers_indices,
        )
        self.box_pos_markers.visualize(
            translations=self.box_pos_markers_pos_w.view(-1, 3),
            marker_indices=self.box_pos_markers_indices,
        )
        
        