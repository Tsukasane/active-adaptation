from active_adaptation.envs.mdp.commands.hdmi.command import RobotTracking, RobotObjectTracking
from active_adaptation.envs.mdp.base import Observation as BaseObservation

import torch
from isaaclab.utils.math import (
    quat_apply_inverse,
    quat_mul,
    quat_conjugate,
    matrix_from_quat,
    yaw_quat,
)
from active_adaptation.utils.math import batchify
quat_apply_inverse = batchify(quat_apply_inverse)

RobotTrackObservation = BaseObservation[RobotTracking]

class ref_joint_pos_future(RobotTrackObservation):
    def compute(self):
        return self.command_manager.ref_joint_pos_future_.view(self.num_envs, -1)

class ref_joint_vel_future(RobotTrackObservation):
    def compute(self):
        return self.command_manager.ref_joint_vel_future_.view(self.num_envs, -1)
    
class ref_root_pos_future_b(RobotTrackObservation):
    """
    Reference root position in robot root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_future_steps = self.command_manager.num_future_steps
        self.ref_root_pos_future_b = torch.zeros(self.num_envs, num_future_steps, 3, device=self.device)

    def update(self):
        ref_root_pos_future_w = self.command_manager.ref_root_pos_future_w # shape: [num_envs, num_future_steps, 3]
        robot_root_pos_w = self.command_manager.robot_root_pos_w[:, None, :] # shape: [num_envs, 1, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]
        
        ref_root_pos_future_b = quat_apply_inverse(robot_root_quat_w, ref_root_pos_future_w - robot_root_pos_w)
        self.ref_root_pos_future_b = ref_root_pos_future_b

    def compute(self):
        return self.ref_root_pos_future_b.view(self.num_envs, -1)
    
class ref_root_ori_future_b(RobotTrackObservation):
    """
    Reference root orientation in robot root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_future_steps = self.command_manager.num_future_steps
        self.ref_root_ori_future_b = torch.zeros(self.num_envs, num_future_steps, 2, 3, device=self.device)

    def update(self):
        ref_root_quat_future_w = self.command_manager.ref_root_quat_future_w # shape: [num_envs, num_future_steps, 4]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]
        
        ref_root_quat_future_b = quat_mul(
            quat_conjugate(robot_root_quat_w).expand_as(ref_root_quat_future_w),
            ref_root_quat_future_w
        )
        ref_root_ori_future_b = matrix_from_quat(ref_root_quat_future_b)
        self.ref_root_ori_future_b = ref_root_ori_future_b[:, :, :2, :]

    def compute(self):
        return self.ref_root_ori_future_b.reshape(self.num_envs, -1)

class ref_body_pos_future_local(RobotTrackObservation):
    """
    Reference body position in motion root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref_body_pos_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, device=self.device)
    
    def update(self):
        ref_body_pos_future_w = self.command_manager.ref_body_pos_future_w    # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        ref_root_pos_w = self.command_manager.ref_root_pos_w[:, None, None, :].clone() # shape: [num_envs, 1, 1, 3]
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]

        ref_root_pos_w[..., 2] = 0.0
        ref_root_quat_w = yaw_quat(ref_root_quat_w)

        ref_body_pos_future_local = quat_apply_inverse(ref_root_quat_w, ref_body_pos_future_w - ref_root_pos_w)
        self.ref_body_pos_future_local = ref_body_pos_future_local
    
    def compute(self):
        return self.ref_body_pos_future_local.view(self.num_envs, -1)

class ref_body_ori_future_local(RobotTrackObservation):
    """
    Reference body orientation in motion root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref_body_ori_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, 3, device=self.device)
    
    def update(self):
        ref_body_quat_future_w = self.command_manager.ref_body_quat_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 4]
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]

        ref_root_quat_w = yaw_quat(ref_root_quat_w)

        ref_body_quat_future_local = quat_mul(
            quat_conjugate(ref_root_quat_w).expand_as(ref_body_quat_future_w),
            ref_body_quat_future_w
        )
        self.ref_body_ori_future_local = matrix_from_quat(ref_body_quat_future_local)
    
    def compute(self):
        return self.ref_body_ori_future_local[:, :, :, :2, :].reshape(self.num_envs, -1)

class diff_body_pos_future_b(RobotTrackObservation):
    """
    Reference body position in each robot body frame.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_body_pos_future_b = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, device=self.device)

    def update(self):
        ref_body_pos_future_w = self.command_manager.ref_body_pos_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        robot_body_pos_w = self.command_manager.robot_body_pos_w[:, None, :, :] # shape: [num_envs, 1, num_tracking_bodies, 3]
        robot_body_quat_w = self.command_manager.robot_body_quat_w[:, None, :, :] # shape: [num_envs, 1, num_tracking_bodies, 4]

        diff_body_pos_future_b = quat_apply_inverse(robot_body_quat_w, ref_body_pos_future_w - robot_body_pos_w)
        self.diff_body_pos_future_b = diff_body_pos_future_b

    def compute(self):
        return self.diff_body_pos_future_b.view(self.num_envs, -1)
    
class diff_body_lin_vel_future_b(RobotTrackObservation):
    """
    Reference body linear velocity in each robot body frame.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_body_lin_vel_future_b = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, device=self.device)

    def update(self):
        ref_body_lin_vel_future_w = self.command_manager.ref_body_lin_vel_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        robot_body_lin_vel_w = self.command_manager.robot_body_lin_vel_w[:, None, :, :] # shape: [num_envs, 1, num_tracking_bodies, 3]
        robot_body_quat_w = self.command_manager.robot_body_quat_w[:, None, :, :] # shape: [num_envs, 1, num_tracking_bodies, 4]

        diff_body_lin_vel_future_b = quat_apply_inverse(robot_body_quat_w, ref_body_lin_vel_future_w - robot_body_lin_vel_w)
        self.diff_body_lin_vel_future_b = diff_body_lin_vel_future_b

    def compute(self):
        return self.diff_body_lin_vel_future_b.view(self.num_envs, -1)

class diff_body_ori_future_b(RobotTrackObservation):
    """
    Reference body orientation in each robot body frame.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_body_ori_future_b = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, 3, device=self.device)
    
    def update(self):
        ref_body_quat_future_w = self.command_manager.ref_body_quat_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 4]
        robot_body_quat_w = self.command_manager.robot_body_quat_w[:, None, :, :] # shape: [num_envs, 1, num_tracking_bodies, 4]

        diff_body_quat_future_b = quat_mul(
            quat_conjugate(robot_body_quat_w).expand_as(ref_body_quat_future_w),
            ref_body_quat_future_w
        )
        self.diff_body_ori_future_b = matrix_from_quat(diff_body_quat_future_b)
    
    def compute(self):
        return self.diff_body_ori_future_b[:, :, :, :2, :].reshape(self.num_envs, -1)

class diff_body_pos_future_local(RobotTrackObservation):
    """
    Reference body position in each motion root frame - Robot body position in robot root frame.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_body_pos_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, device=self.device)

    def update(self):
        ref_body_pos_future_w = self.command_manager.ref_body_pos_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        ref_root_pos_w = self.command_manager.ref_root_pos_w[:, None, None, :].clone() # shape: [num_envs, 1, 1, 3]
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]

        robot_body_pos_w = self.command_manager.robot_body_pos_w # shape: [num_envs, num_tracking_bodies, 3]
        robot_root_pos_w = self.command_manager.robot_root_pos_w[:, None, :].clone() # shape: [num_envs, 1, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]

        ref_root_pos_w[..., 2] = 0.0
        robot_root_pos_w[..., 2] = 0.0
        ref_root_quat_w = yaw_quat(ref_root_quat_w)
        robot_root_quat_w = yaw_quat(robot_root_quat_w)

        ref_body_pos_future_local = quat_apply_inverse(ref_root_quat_w, ref_body_pos_future_w - ref_root_pos_w)
        robot_body_pos_local = quat_apply_inverse(robot_root_quat_w, robot_body_pos_w - robot_root_pos_w)

        self.diff_body_pos_future_local = ref_body_pos_future_local - robot_body_pos_local.unsqueeze(1)

    def compute(self):
        return self.diff_body_pos_future_local.view(self.num_envs, -1)
    
class diff_body_lin_vel_future_local(RobotTrackObservation):
    """
    Reference body linear velocity in motion root frame - Robot body linear velocity in robot root frame.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_body_lin_vel_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, device=self.device)
    
    def update(self):
        ref_body_lin_vel_future_w = self.command_manager.ref_body_lin_vel_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]
        robot_body_lin_vel_w = self.command_manager.robot_body_lin_vel_w # shape: [num_envs, num_tracking_bodies, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]

        ref_root_quat_w = yaw_quat(ref_root_quat_w)
        robot_root_quat_w = yaw_quat(robot_root_quat_w)

        ref_body_lin_vel_future_local = quat_apply_inverse(ref_root_quat_w, ref_body_lin_vel_future_w)
        robot_body_lin_vel_local = quat_apply_inverse(robot_root_quat_w, robot_body_lin_vel_w)

        self.diff_body_lin_vel_future_local = ref_body_lin_vel_future_local - robot_body_lin_vel_local.unsqueeze(1)

    def compute(self):
        return self.diff_body_lin_vel_future_local.view(self.num_envs, -1)

    
class diff_body_ori_future_local(RobotTrackObservation):
    """
    Reference body orientation in motion root frame - Robot body orientation in robot root frame.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_body_ori_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, 3, device=self.device)

    def update(self):
        ref_body_quat_future_w = self.command_manager.ref_body_quat_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 4]
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]
        robot_body_quat_w = self.command_manager.robot_body_quat_w # shape: [num_envs, num_tracking_bodies, 4]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]

        ref_root_quat_w = yaw_quat(ref_root_quat_w)
        robot_root_quat_w = yaw_quat(robot_root_quat_w)

        ref_body_quat_future_local = quat_mul(
            quat_conjugate(ref_root_quat_w).expand_as(ref_body_quat_future_w),
            ref_body_quat_future_w
        )
        robot_body_quat_local = quat_mul(
            quat_conjugate(robot_root_quat_w).expand_as(robot_body_quat_w),
            robot_body_quat_w
        ).unsqueeze(1)
        diff_body_quat_future = quat_mul(
            quat_conjugate(robot_body_quat_local).expand_as(ref_body_quat_future_w),
            ref_body_quat_future_local
        )
        self.diff_body_ori_future_local = matrix_from_quat(diff_body_quat_future)

    def compute(self):
        return self.diff_body_ori_future_local[:, :, :, :2, :].reshape(self.num_envs, -1)

class ref_motion_phase(RobotTrackObservation):
    def compute(self):
        return (self.command_manager.t / self.command_manager.motion_len).unsqueeze(1)

RobotObjectTrackObservation = BaseObservation[RobotObjectTracking]

class ref_contact_pos_b(RobotObjectTrackObservation):
    """
    Reference end-effector target position in robot root frame
    """
    def __init__(self, noise_std: float=0.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = noise_std
        self.ref_contact_pos_b = torch.zeros_like(self.command_manager.contact_target_pos_w)

    def update(self):
        ref_contact_target_pos_w = self.command_manager.contact_target_pos_w # shape: [num_envs, n, 3]
        robot_root_pos_w = self.command_manager.robot_root_pos_w[:, None, :] # shape: [num_envs, 1, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]

        ref_contact_pos_b = quat_apply_inverse(robot_root_quat_w, ref_contact_target_pos_w - robot_root_pos_w)
        if self.noise_std > 0.0:
            noise = torch.randn_like(ref_contact_pos_b).clamp(-1, 1) * self.noise_std
            ref_contact_pos_b += noise
        self.ref_contact_pos_b = ref_contact_pos_b

    def compute(self):
        return self.ref_contact_pos_b.view(self.num_envs, -1)

class diff_contact_pos_b(RobotObjectTrackObservation):
    """
    Reference end-effector target position in robot root frame - Robot end-effector position in robot root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_contact_pos_b = torch.zeros_like(self.command_manager.contact_target_pos_w)

    def update(self):
        ref_contact_target_pos_w = self.command_manager.contact_target_pos_w # shape: [num_envs, n, 3]
        contact_eef_pos_w = self.command_manager.contact_eef_pos_w # shape: [num_envs, n, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]
        
        diff_contact_pos_w = ref_contact_target_pos_w - contact_eef_pos_w
        self.diff_contact_pos_b = quat_apply_inverse(robot_root_quat_w, diff_contact_pos_w)

    def compute(self):
        return self.diff_contact_pos_b.view(self.num_envs, -1)
    
class object_pos_b(RobotObjectTrackObservation):
    """
    Object position in robot root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.object_pos_b = torch.zeros(self.num_envs, 3, device=self.device)

    def update(self):
        object_pos_w = self.command_manager.object.data.root_link_pos_w # shape: [num_envs, 3]
        robot_root_pos_w = self.command_manager.robot_root_pos_w # shape: [num_envs, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w # shape: [num_envs, 4]

        self.object_pos_b = quat_apply_inverse(robot_root_quat_w, object_pos_w - robot_root_pos_w)

    def compute(self):
        return self.object_pos_b.view(self.num_envs, -1)

class object_ori_b(RobotObjectTrackObservation):
    """
    Object orientation in robot root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.object_ori_b = torch.zeros(self.num_envs, 3, 3, device=self.device)

    def update(self):
        object_quat_w = self.command_manager.object.data.root_link_quat_w # shape: [num_envs, 4]
        robot_root_quat_w = self.command_manager.robot_root_quat_w # shape: [num_envs, 4]

        object_quat_b = quat_mul(
            quat_conjugate(robot_root_quat_w).expand_as(object_quat_w),
            object_quat_w
        )
        self.object_ori_b = matrix_from_quat(object_quat_b)

    def compute(self):
        return self.object_ori_b.view(self.num_envs, -1)
    
class object_joint_pos(RobotObjectTrackObservation):
    """
    Object joint position
    """
    def compute(self):
        return self.command_manager.object_joint_pos.unsqueeze(1)

class diff_object_pos_future(RobotObjectTrackObservation):
    """
    Object position in robot root frame - Robot end-effector position in robot root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_object_pos_future_b = torch.zeros(self.num_envs, self.command_manager.num_future_steps, 3, device=self.device)

    def update(self):
        ref_object_pos_future_w = self.command_manager.ref_object_pos_future_w # shape: [num_envs, num_future_steps, 3]
        object_pos_w = self.command_manager.object.data.root_link_pos_w.unsqueeze(1)
        diff_object_pos_future_w = ref_object_pos_future_w - object_pos_w

        object_quat_w = self.command_manager.object.data.root_quat_w.unsqueeze(1) # shape: [num_envs, 1, 4]
        self.diff_object_pos_future_b = quat_apply_inverse(object_quat_w, diff_object_pos_future_w)
    
    def compute(self):
        return self.diff_object_pos_future_b.view(self.num_envs, -1)

class diff_object_ori_future(RobotObjectTrackObservation):
    """
    Object orientation in robot root frame - Robot end-effector orientation in robot root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_object_ori_future_b = torch.zeros(self.num_envs, self.command_manager.num_future_steps, 3, 3, device=self.device)

    def update(self):
        ref_object_quat_future_w = self.command_manager.ref_object_quat_future_w # shape: [num_envs, num_future_steps, 4]
        object_quat_w = self.command_manager.object.data.root_link_quat_w.unsqueeze(1) # shape: [num_envs, 1, 4]
        
        diff_object_quat_future = quat_mul(
            quat_conjugate(object_quat_w).expand_as(ref_object_quat_future_w),
            ref_object_quat_future_w
        )
        self.diff_object_ori_future_b = matrix_from_quat(diff_object_quat_future)

    def compute(self):
        return self.diff_object_ori_future_b.view(self.num_envs, -1)

class diff_object_joint_pos_future(RobotObjectTrackObservation):
    """
    Object joint position - Robot end-effector joint position
    """
    def compute(self):
        ref_object_joint_pos_future = self.command_manager.ref_object_joint_pos_future
        object_joint_pos = self.command_manager.object_joint_pos
        diff_object_joint_pos_future = ref_object_joint_pos_future - object_joint_pos.unsqueeze(1)
        return diff_object_joint_pos_future

class ref_object_contact_future(RobotObjectTrackObservation):
    def compute(self):
        return self.command_manager.ref_object_contact_future.view(self.num_envs, -1)
