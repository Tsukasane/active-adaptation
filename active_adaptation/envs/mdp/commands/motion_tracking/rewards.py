from active_adaptation.envs.mdp.commands.motion_tracking.command import MotionTrackingCommand
from active_adaptation.envs.mdp.base import Reward as BaseReward

from typing import List, Dict
from omegaconf import DictConfig
from isaaclab.utils.string import resolve_matching_names, resolve_matching_names_values
from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_conjugate, axis_angle_from_quat, yaw_quat

import torch


TrackReward = BaseReward[MotionTrackingCommand]

class _tracking_keypoint(TrackReward):
    def __init__(self, body_names: List[str] | str | None = None, sigma: float = 0.03, tolerance: float | Dict[str, float] = 0.0, **kwargs):
        super().__init__(**kwargs)
        if body_names is None:
            body_names = self.command_manager.tracking_keypoint_names
        
        self.sigma = sigma
        body_indices_motion, matched_names_motion = resolve_matching_names(body_names, self.command_manager.tracking_keypoint_names)
        body_indices_asset, matched_names_asset = resolve_matching_names(body_names, self.command_manager.asset.body_names)

        matched_names = set(matched_names_motion) & set(matched_names_asset)
        assert set(matched_names) == set(matched_names_motion) == set(matched_names_asset), "body names in motion dataset and robot not matched"
        assert set(matched_names) <= set(self.command_manager.tracking_keypoint_names), "Some body names in motion dataset not found in tracking body names"
        
        self.body_indices_motion = []
        self.body_indices_asset = []
        self.body_names = list(sorted(matched_names))
        self.num_bodies = len(self.body_names)
        for body_name in self.body_names:
            body_idx_motion = self.command_manager.tracking_keypoint_names.index(body_name)
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

class keypoint_pos_tracking_l2(_tracking_keypoint):
    def compute(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        body_pos_motion = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]
        diff = body_pos_motion - body_pos_asset
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp(min=0.0, max=1.0)
        # shape: [num_envs, num_tracking_bodies]
        return -error.square().mean(dim=-1).unsqueeze(1)

class keypoint_pos_tracking_product(_tracking_keypoint):
    def compute(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        body_pos_motion = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]
        diff = body_pos_motion - body_pos_asset
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

class keypoint_pos_tracking_local_product(_tracking_keypoint):
    def compute(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        body_pos_motion = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]

        root_pos_asset = self.command_manager.robot_root_pos_w.clone()
        root_pos_motion = self.command_manager.ref_root_pos_w.clone()
        root_quat_asset = self.command_manager.robot_root_quat_w
        root_quat_motion = self.command_manager.ref_root_quat_w
        
        root_pos_asset[..., 2] = 0.0
        root_pos_motion[..., 2] = 0.0
        root_quat_asset = yaw_quat(root_quat_asset)
        root_quat_motion = yaw_quat(root_quat_motion)
        
        root_pos_asset = root_pos_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_pos_motion = root_pos_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

        body_pos_asset_relative = quat_apply_inverse(root_quat_asset, body_pos_asset - root_pos_asset)
        body_pos_motion_relative = quat_apply_inverse(root_quat_motion, body_pos_motion - root_pos_motion)

        diff = body_pos_motion_relative - body_pos_asset_relative
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

    def debug_draw(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        body_pos_motion = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]

        root_pos_asset = self.command_manager.robot_root_pos_w.clone()
        root_pos_motion = self.command_manager.ref_root_pos_w.clone()
        root_quat_asset = self.command_manager.robot_root_quat_w
        root_quat_motion = self.command_manager.ref_root_quat_w
        
        root_pos_asset[..., 2] = 0.0
        root_pos_motion[..., 2] = 0.0
        root_quat_asset = yaw_quat(root_quat_asset)
        root_quat_motion = yaw_quat(root_quat_motion)
        
        root_pos_asset = root_pos_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_pos_motion = root_pos_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

        body_pos_asset_relative = quat_apply_inverse(root_quat_asset, body_pos_asset - root_pos_asset)
        body_pos_motion_relative = quat_apply_inverse(root_quat_motion, body_pos_motion - root_pos_motion)
        # self.env._debug_draw.vector(
        #     root_pos_asset,
        #     body_pos_asset_relative,
        #     color=(0.0, 1.0, 0.0),
        #     size=4.0,
        # )
        # self.env._debug_draw.vector(
        #     root_pos_motion,
        #     body_pos_motion_relative,
        #     color=(1.0, 0.0, 0.0),
        #     size=4.0,
        # )
        self.env.debug_draw.point(
            body_pos_asset_relative.reshape(-1, 3),
            color=(0.0, 1.0, 0.0, 1.0),
            size=20,
        )
        self.env.debug_draw.point(
            body_pos_motion_relative.reshape(-1, 3),
            color=(1.0, 0.0, 0.0, 1.0),
            size=20,
        )

class keypoint_ori_tracking_product(_tracking_keypoint):
    def compute(self):
        body_ori_asset = self.command_manager.asset.data.body_quat_w[:, self.body_indices_asset]
        body_ori_motion = self.command_manager.ref_body_quat_w[:, self.body_indices_motion]
        diff = quat_mul(quat_conjugate(body_ori_motion), body_ori_asset)
        # shape: [num_envs, num_tracking_bodies, 4]
        error = torch.norm(axis_angle_from_quat(diff), dim=-1)
        error = (error - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
    
class keypoint_ori_tracking_l2(_tracking_keypoint):
    def compute(self):
        body_ori_asset = self.command_manager.asset.data.body_quat_w[:, self.body_indices_asset]
        body_ori_motion = self.command_manager.ref_body_quat_w[:, self.body_indices_motion]
        diff = quat_mul(quat_conjugate(body_ori_motion), body_ori_asset)
        # shape: [num_envs, num_tracking_bodies, 4]
        error = torch.norm(axis_angle_from_quat(diff), dim=-1)
        error = (error - self.tolerance).clamp(min=0.0, max=1.0)
        # shape: [num_envs, num_tracking_bodies]
        return -error.square().mean(dim=-1).unsqueeze(1)
    
class keypoint_ori_tracking_local_product(_tracking_keypoint):
    def compute(self):
        body_ori_asset = self.command_manager.asset.data.body_quat_w[:, self.body_indices_asset]
        body_ori_motion = self.command_manager.ref_body_quat_w[:, self.body_indices_motion]

        root_quat_asset = self.command_manager.robot_root_quat_w
        root_quat_motion = self.command_manager.ref_root_quat_w

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
    
class keypoint_lin_vel_tracking_product(_tracking_keypoint):
    def compute(self):
        body_lin_vel_asset = self.command_manager.asset.data.body_com_lin_vel_w[:, self.body_indices_asset]
        body_lin_vel_motion = self.command_manager.ref_body_lin_vel_w[:, self.body_indices_motion]
        diff = body_lin_vel_motion - body_lin_vel_asset
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

class keypoint_lin_vel_tracking_local_product(_tracking_keypoint):
    def compute(self):
        body_lin_vel_asset = self.command_manager.asset.data.body_com_lin_vel_w[:, self.body_indices_asset].clone()
        body_lin_vel_motion = self.command_manager.ref_body_lin_vel_w[:, self.body_indices_motion].clone()

        root_pos_asset = self.command_manager.robot_root_pos_w.clone()
        root_pos_motion = self.command_manager.ref_root_pos_w.clone()
        root_quat_asset = self.command_manager.robot_root_quat_w
        root_quat_motion = self.command_manager.ref_root_quat_w

        root_pos_asset[..., 2] = 0.0
        root_pos_motion[..., 2] = 0.0
        root_quat_asset = yaw_quat(root_quat_asset)
        root_quat_motion = yaw_quat(root_quat_motion)
        
        root_pos_asset = root_pos_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_pos_motion = root_pos_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

        body_lin_vel_asset_relative = quat_apply_inverse(root_quat_asset, body_lin_vel_asset - root_pos_asset)
        body_lin_vel_motion_relative = quat_apply_inverse(root_quat_motion, body_lin_vel_motion - root_pos_motion)
        
        diff = body_lin_vel_motion_relative - body_lin_vel_asset_relative
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

class keypoint_ang_vel_tracking_product(_tracking_keypoint):
    def compute(self):
        body_ang_vel_asset = self.command_manager.asset.data.body_com_ang_vel_w[:, self.body_indices_asset]
        body_ang_vel_motion = self.command_manager.ref_body_ang_vel_w[:, self.body_indices_motion]
        diff = body_ang_vel_motion - body_ang_vel_asset
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

class keypoint_ang_vel_tracking_local_product(_tracking_keypoint):
    def compute(self):
        body_ang_vel_asset = self.command_manager.asset.data.body_com_ang_vel_w[:, self.body_indices_asset]
        body_ang_vel_motion = self.command_manager.ref_body_ang_vel_w[:, self.body_indices_motion]

        root_quat_asset = self.command_manager.robot_root_quat_w
        root_quat_motion = self.command_manager.ref_root_quat_w

        root_quat_asset = yaw_quat(root_quat_asset)
        root_quat_motion = yaw_quat(root_quat_motion)

        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

        body_ang_vel_asset_relative = quat_apply_inverse(root_quat_asset, body_ang_vel_asset)
        body_ang_vel_motion_relative = quat_apply_inverse(root_quat_motion, body_ang_vel_motion)

        diff = body_ang_vel_motion_relative - body_ang_vel_asset_relative
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

class _tracking_joint(TrackReward):
    def __init__(self, joint_names: List[str] | str | None = None, sigma: float = 0.03, tolerance: float | Dict[str, float] = 0.0, **kwargs):
        super().__init__(**kwargs)
        if joint_names is None:
            joint_names = self.command_manager.tracking_joint_names
    
        self.sigma = sigma
        joint_indices_asset, matched_names_asset = resolve_matching_names(joint_names, self.command_manager.asset.joint_names)
        joint_indices_motion, matched_names_motion = resolve_matching_names(joint_names, self.command_manager.tracking_joint_names)

        matched_names = set(matched_names_motion) & set(matched_names_asset)
        assert set(matched_names) == set(matched_names_motion) == set(matched_names_asset), "joint names in motion dataset and robot not matched"
        assert set(matched_names) <= set(self.command_manager.tracking_joint_names), "Some joint names in motion dataset not found in tracking joint names"

        self.joint_indices_motion = []
        self.joint_indices_asset = []
        self.joint_names = list(sorted(matched_names))
        for joint_name in self.joint_names:
            joint_idx_motion = self.command_manager.tracking_joint_names.index(joint_name)
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

class joint_pos_tracking_product(_tracking_joint):
    def compute(self):
        joint_pos_asset = self.command_manager.asset.data.joint_pos[:, self.joint_indices_asset]
        joint_pos_motion = self.command_manager.ref_joint_pos[:, self.joint_indices_motion]
        diff = joint_pos_motion - joint_pos_asset
        error = (diff.abs() - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_joints]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
    
class joint_vel_tracking_product(_tracking_joint):
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
        body_names = resolve_matching_names(body_names, self.command_manager.tracking_keypoint_names)[1]
        self.body_indices_motion = []
        self.body_indices_sensor = []
        for name in body_names:
            body_idx_motion = self.command_manager.tracking_keypoint_names.index(name)
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

class feet_no_contact_force_when_motion_vel(TrackReward):
    def __init__(self, body_names: str | List[str], motion_vel_thres: float=0.1, soft_discount: float=1.0, **kwargs):
        super().__init__(**kwargs)
        self.motion_vel_thres = motion_vel_thres
        self.soft_discount = soft_discount
        body_names = resolve_matching_names(body_names, self.command_manager.tracking_keypoint_names)[1]
        self.body_indices_motion = []
        self.body_indices_sensor = []
        for name in body_names:
            body_idx_motion = self.command_manager.tracking_keypoint_names.index(name)
            body_idx_sensor = self.command_manager.contact_forces.body_names.index(name)

            self.body_indices_motion.append(body_idx_motion)
            self.body_indices_sensor.append(body_idx_sensor)

        self.motion_no_contact = torch.zeros((self.env.num_envs, len(body_names)), dtype=torch.bool, device=self.device)
        self.contact_force_norm = torch.zeros((self.env.num_envs, len(body_names)), device=self.device)
    
    def update(self):
        # when feet vel in motion is large, feet should not be in contact
        feet_vel_motion = self.command_manager.ref_body_lin_vel_w[:, self.body_indices_motion]
        feet_vel_motion_norm = feet_vel_motion[..., :2].norm(dim=-1)
        self.motion_no_contact[:] = feet_vel_motion_norm > self.motion_vel_thres

        contact_forces = self.command_manager.contact_forces.data.net_forces_w[:, self.body_indices_sensor]
        self.contact_force_norm[:] = contact_forces.norm(dim=-1)

    def compute(self):
        penalty = (self.contact_force_norm > 0.1) & self.motion_no_contact
        self.env.discount[penalty.any(dim=1)] *= self.soft_discount
        return -(penalty * self.contact_force_norm).mean(dim=1, keepdim=True)

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
        body_names = resolve_matching_names(body_names, self.command_manager.tracking_keypoint_names)[1]
        self.body_indices_motion = []
        self.body_indices_sensor = []
        self.body_indices_asset = []
        for name in body_names:
            body_idx_motion = self.command_manager.tracking_keypoint_names.index(name)
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
        feet_projected_gravity = quat_apply_inverse(feet_quat, self.gravity_w)

        feet_height = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset][..., 2]
        feet_lin_vel = self.command_manager.asset.data.body_com_lin_vel_w[:, self.body_indices_asset].norm(dim=-1)

        in_contact = (contact_forces.norm(dim=-1) > self.feet_contact_thres) \
            & (feet_projected_gravity[..., 2] < -self.feet_gravity_thres) \
            & (feet_height < self.motion_height_thres) \
            & (feet_lin_vel < self.motion_vel_thres)
        # shape: [num_envs, num_feet]

        penalty = (~in_contact) & self.motion_contact
        self.env.discount[penalty.any(dim=1)] *= self.soft_discount
        return -penalty.float().mean(dim=1, keepdim=True)
    
    def debug_draw(self):
        if self.env.backend != "isaac":
            return
        # return

        self.vis_markers_pos_w.fill_(-100.0)
        self.vis_markers_pos_w[self.motion_contact] = self.command_manager.ref_body_pos_w[:, self.body_indices_motion][self.motion_contact]
        self.vis_markers.visualize(
            translations=self.vis_markers_pos_w.view(-1, 3),
        )

from active_adaptation.envs.mdp.commands.motion_tracking.command import MotionTrackingDoor
from typing import Tuple
from isaaclab.utils.math import quat_apply

TrackDoorReward = BaseReward[MotionTrackingDoor]

class door_joint_pos_tracking(TrackDoorReward):
    def __init__(self, sigma=0.1, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma

    def compute(self):
        door_joint_pos = self.command_manager.asset.data.joint_pos[:, self.command_manager.door_joint_id_asset]
        ref_door_joint_pos = self.command_manager.ref_door_joint_pos
        error = (door_joint_pos - ref_door_joint_pos).abs()
        return torch.exp(- error / self.sigma).unsqueeze(-1)
    
class eef_door_contact_pos(TrackDoorReward):
    def __init__(self, pos_tolerance: float=0.2, pos_sigma: float=0.2, eef_target_pos_offset: Tuple[float, float, float]=(0.0, -0.6, 1.5), **kwargs):
        super().__init__(**kwargs)
        self.pos_tolerance = pos_tolerance
        self.pos_sigma = pos_sigma

        self.eef_idx_sensor = self.command_manager.eef_idx_sensor
        self.eef_idx_asset = self.command_manager.eef_idx_asset
        self.door_body_id_asset = self.command_manager.door_body_id_asset

        self.eef_target_pos_offset = torch.tensor(eef_target_pos_offset, device=self.device).unsqueeze(0)

        self.eef_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.eef_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.eef_in_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        if not self.env.backend == "isaac":
            return
        
        # init debug draw
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.sim import SphereCfg, PreviewSurfaceCfg
        vis_markers_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/EefDoorContactPos",
            markers={
                "eef_target": SphereCfg(radius=0.04, visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))),
                "eef_pos": SphereCfg(radius=0.04, visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))),
            },
        )
        self.vis_markers = VisualizationMarkers(vis_markers_cfg)
        self.marker_indices = [0] * self.num_envs + [1] * self.num_envs
        self.all_marker_pos_w = torch.zeros(2, self.num_envs, 3, device=self.device)
    
    def update(self):
        self.eef_pos_w[:] = self.command_manager.asset.data.body_link_pos_w[:, self.eef_idx_asset]
        door_pos_w = self.command_manager.door.data.body_link_pos_w[:, self.door_body_id_asset]
        door_quat_w = self.command_manager.door.data.body_link_quat_w[:, self.door_body_id_asset]
        self.eef_target_pos_w[:] = door_pos_w + quat_apply(door_quat_w, self.eef_target_pos_offset)

        contact_force = self.command_manager.contact_forces.data.net_forces_w[:, self.eef_idx_sensor]
        self.eef_in_contact[:] = (contact_force.norm(dim=-1) > 1.0)

    def compute(self):
        pos_error = (self.eef_pos_w - self.eef_target_pos_w).norm(dim=-1)
        pos_error = (pos_error - self.pos_tolerance).clamp_min(0.0)
        rew = torch.exp(- pos_error / self.pos_sigma)
        rew = (rew - 1.0) * self.eef_in_contact
        return rew.float().unsqueeze(-1)
    
    def debug_draw(self):
        if not self.env.backend == "isaac":
            return
        
        self.all_marker_pos_w[0] = self.eef_target_pos_w
        self.all_marker_pos_w[1] = self.eef_pos_w
        self.vis_markers.visualize(
            translations=self.all_marker_pos_w.reshape(-1, 3),
            marker_indices=self.marker_indices,
        )

class eef_door_contact_force(TrackDoorReward):
    def __init__(self, force_thres: float=20.0, force_sigma: float=10.0, **kwargs):
        super().__init__(**kwargs)
        self.force_thres = force_thres
        self.force_sigma = force_sigma
        self.eef_idx_sensor = self.command_manager.eef_idx_sensor
        self.contact_force = torch.zeros(self.num_envs, device=self.device)
    
    def update(self):
        contact_force = self.command_manager.contact_forces.data.net_forces_w[:, self.eef_idx_sensor]
        self.contact_force[:] = contact_force.norm(dim=-1)

    def compute(self):
        rew = torch.exp((self.contact_force - self.force_thres).clamp_max(0.0) / self.force_sigma)
        return rew.float().unsqueeze(-1)
    
class eef_door_contact_slippage(TrackDoorReward):
    """penalize slippage of the eef (velocity parallel to the door plane) when contact with the door"""
    def __init__(self, slippage_thres: float=0.01, slippage_sigma: float=0.1, **kwargs):
        super().__init__(**kwargs)
        self.slippage_thres = slippage_thres
        self.slippage_sigma = slippage_sigma

        self.eef_idx_sensor = self.command_manager.eef_idx_sensor
        self.eef_idx_asset = self.command_manager.eef_idx_asset
        self.door_body_id_asset = self.command_manager.door_body_id_asset

        self.eff_in_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.slippage = torch.zeros(self.num_envs, device=self.device)
    
    def update(self):
        # get contact force
        contact_force = self.command_manager.contact_forces.data.net_forces_w[:, self.eef_idx_sensor]
        self.eff_in_contact[:] = (contact_force.norm(dim=-1) > 1.0)

        # get eef velocity in door frame
        eef_lin_vel_w = self.command_manager.asset.data.body_com_lin_vel_w[:, self.eef_idx_asset]
        door_quat_w = self.command_manager.door.data.body_link_quat_w[:, self.door_body_id_asset]
        eef_lin_vel_door = quat_apply_inverse(door_quat_w, eef_lin_vel_w)
        # shape: [num_envs, 3]

        # get slippage
        self.slippage[:] = eef_lin_vel_door[:, 1:].norm(dim=-1)
    
    def compute(self):
        slippage = (self.slippage - self.slippage_thres).clamp_min(0.0)
        rew = (torch.exp(- slippage / self.slippage_sigma) - 1.0) * self.eff_in_contact
        return rew.float().unsqueeze(-1)
    
class root_vel_pass_door(TrackDoorReward):
    def __init__(self, max_vel: float=0.5, **kwargs):
        super().__init__(**kwargs)
        self.max_vel = max_vel
        self.root_vel_door = torch.zeros(self.num_envs, 3, device=self.device)
    
    def update(self):
        root_vel_w = self.command_manager.asset.data.root_com_lin_vel_w
        door_quat_w = self.command_manager.door.data.root_link_quat_w
        root_vel_door = quat_apply_inverse(door_quat_w, root_vel_w)
        self.root_vel_door[:] = root_vel_door
    
    def compute(self):
        rew = (-self.root_vel_door[:, 0] / self.max_vel).clamp_max(1.0)
        return rew.float().unsqueeze(-1)
    
class root_pos_pass_door(TrackDoorReward):
    def __init__(self, max_pos_x: float=0.5, pos_sigma: float=0.5, **kwargs):
        super().__init__(**kwargs)
        self.max_pos_x = max_pos_x
        self.pos_sigma = pos_sigma

        self.root_pos_door = torch.zeros(self.num_envs, 3, device=self.device)
    
    def update(self):
        root_pos_w = self.command_manager.asset.data.root_link_pos_w
        door_pos_w = self.command_manager.door.data.root_link_pos_w
        door_quat_w = self.command_manager.door.data.root_link_quat_w
        root_pos_door = quat_apply_inverse(door_quat_w, root_pos_w - door_pos_w)
        self.root_pos_door[:] = root_pos_door
    
    def compute(self):
        dist_x = (self.max_pos_x - (-self.root_pos_door[:, 0])).clamp_min(0.0)
        dist_y = (self.root_pos_door[:, 1]).abs()
        rew = torch.exp(- (dist_x + dist_y) / self.pos_sigma)
        return rew.float().unsqueeze(-1)
    
class root_pos_pass_door_l1(TrackDoorReward):
    def __init__(self, max_pos_x: float=0.5, **kwargs):
        super().__init__(**kwargs)
        self.max_pos_x = max_pos_x

        self.root_pos_door = torch.zeros(self.num_envs, 3, device=self.device)
    
    def update(self):
        root_pos_w = self.command_manager.asset.data.root_link_pos_w
        door_pos_w = self.command_manager.door.data.root_link_pos_w
        door_quat_w = self.command_manager.door.data.root_link_quat_w
        root_pos_door = quat_apply_inverse(door_quat_w, root_pos_w - door_pos_w)
        self.root_pos_door[:] = root_pos_door
    
    def compute(self):
        dist_x = (self.max_pos_x - (-self.root_pos_door[:, 0])).clamp_min(0.0)
        dist_y = (self.root_pos_door[:, 1]).abs()
        rew = - (dist_x + dist_y)
        return rew.float().unsqueeze(-1)
    
class door_joint_vel_l2(TrackDoorReward):
    def __init__(self, joint_names: str, tolerance: float=0.5, **kwargs):
        super().__init__(**kwargs)
        self.joint_names = joint_names
        self.joint_ids = self.command_manager.door.find_joints(joint_names)[0]
        self.tolerance = tolerance
    
    def compute(self):
        door_joint_vel = self.command_manager.door.data.joint_vel[:, self.joint_ids]
        door_joint_vel = (door_joint_vel.abs() - self.tolerance).clamp_min(0.0)
        return - door_joint_vel.square().mean(1, True)
    
    
from active_adaptation.envs.mdp.commands.motion_tracking.command import MotionTrackingBox
from isaaclab.utils.math import wrap_to_pi
TrackBoxReward = BaseReward["MotionTrackingBox"]

class box_pos_tracking(TrackBoxReward):
    def __init__(self, sigma: float=0.1, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma

    def compute(self):
        box_pos_w = self.command_manager.box.data.root_link_pos_w
        ref_box_pos_w = self.command_manager.ref_box_pos_w
        diff_box_pos_w_b = ref_box_pos_w - box_pos_w
        error = diff_box_pos_w_b.norm(dim=-1)
        return torch.exp(-error / self.sigma).unsqueeze(-1)
    

class box_ori_tracking(TrackBoxReward):
    def __init__(self, sigma: float=0.1, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma

    def compute(self):
        box_quat_w = self.command_manager.box.data.root_link_quat_w
        ref_box_quat_w = self.command_manager.ref_box_quat_w
        diff_box_quat_w = quat_mul(quat_conjugate(box_quat_w), ref_box_quat_w)
        error = axis_angle_from_quat(diff_box_quat_w).norm(dim=-1)
        return torch.exp(-error / self.sigma).unsqueeze(-1)
    
class eef_contact_in_range(TrackBoxReward):
    def update(self):
        self.in_range = self.command_manager.ref_box_contact

    def compute(self):
        rew = self.in_range.float()
        return rew.unsqueeze(-1)
    
class eef_contact_pos(TrackBoxReward):
    def __init__(self, sigma: float=0.1, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.eef_pos_error = torch.zeros(self.num_envs, 2, device=self.device)
    
    def update(self):
        self.in_range = self.command_manager.ref_box_contact
        eef_pos_diff = self.command_manager.contact_eef_pos_w - self.command_manager.contact_target_pos_w
        self.eef_pos_error[:] = eef_pos_diff.norm(dim=-1)

    def compute(self):
        rew = torch.exp(-self.eef_pos_error / self.sigma).mean(dim=-1)
        rew = (rew - 1.0) * self.in_range.float()
        return rew.unsqueeze(-1)
    
class eef_contact_ori(TrackBoxReward):
    def __init__(self, sigma: float=0.1, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.eef_ori_error = torch.zeros(self.num_envs, 2, 3, device=self.device)
    
    def update(self):
        self.in_range = self.command_manager.ref_box_contact
        eef_ori_diff = wrap_to_pi(self.command_manager.contact_eef_euler_xyz - self.command_manager.contact_target_euler_xyz)
        self.eef_ori_error[:] = eef_ori_diff.abs()

    def compute(self):
        rew = torch.exp(-self.eef_ori_error / self.sigma).mean(dim=(1, 2))
        rew = (rew - 1.0) * self.in_range.float()
        return rew.unsqueeze(-1)
    
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
        eef_ori_diff = wrap_to_pi(self.command_manager.contact_eef_euler_xyz - self.command_manager.contact_target_euler_xyz)
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
