from active_adaptation.envs.mdp.commands.motion_tracking.command import MotionTrackingCommand
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


TrackObservation = BaseObservation[MotionTrackingCommand]

class ref_joint_pos_future(TrackObservation):
    def compute(self):
        return self.command_manager.ref_joint_pos_future_.view(self.num_envs, -1)

class ref_joint_vel_future(TrackObservation):
    def compute(self):
        return self.command_manager.ref_joint_vel_future_.view(self.num_envs, -1)
    
class ref_root_pos_future_b(TrackObservation):
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
    
class ref_root_ori_future_b(TrackObservation):
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

class ref_body_pos_future_b(TrackObservation):
    """
    Reference body position in robot root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref_body_pos_future_b = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, device=self.device)

    def update(self):
        ref_body_pos_future_w = self.command_manager.ref_body_pos_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        robot_root_pos_w = self.command_manager.robot_root_pos_w[:, None, None, :] # shape: [num_envs, 1, 1, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]

        ref_body_pos_future_b = quat_apply_inverse(robot_root_quat_w, ref_body_pos_future_w - robot_root_pos_w)
        self.ref_body_pos_future_b = ref_body_pos_future_b

    def compute(self):
        return self.ref_body_pos_future_b.view(self.num_envs, -1)
    
class ref_body_pos_future_local(TrackObservation):
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

class diff_body_pos_future_local(TrackObservation):
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
    
class diff_body_pos_future_b(TrackObservation):
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
    
class ref_body_lin_vel_future_b(TrackObservation):
    """
    Reference body linear velocity in robot root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref_body_lin_vel_future_b = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, device=self.device)
    
    def update(self):
        ref_body_lin_vel_future_w = self.command_manager.ref_body_lin_vel_future_w      # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]

        ref_body_lin_vel_future_b = quat_apply_inverse(robot_root_quat_w, ref_body_lin_vel_future_w)
        self.ref_body_lin_vel_future_b = ref_body_lin_vel_future_b
    
    def compute(self):
        return self.ref_body_lin_vel_future_b.view(self.num_envs, -1)

class ref_body_lin_vel_future_local(TrackObservation):
    """
    Reference body linear velocity in motion root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref_body_lin_vel_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, device=self.device)
    
    def update(self):
        ref_body_lin_vel_future_w = self.command_manager.ref_body_lin_vel_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]

        ref_body_lin_vel_future_local = quat_apply_inverse(ref_root_quat_w, ref_body_lin_vel_future_w)
        self.ref_body_lin_vel_future_local = ref_body_lin_vel_future_local
    
    def compute(self):
        return self.ref_body_lin_vel_future_local.view(self.num_envs, -1)

class diff_body_lin_vel_future_local(TrackObservation):
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

class diff_body_lin_vel_future_b(TrackObservation):
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
    
class ref_body_ori_future_b(TrackObservation):
    """
    Reference body orientation in robot root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref_body_ori_future_b = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, 3, device=self.device)
    
    def update(self):
        ref_body_quat_future_w = self.command_manager.ref_body_quat_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 4]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]

        ref_body_quat_future_b = quat_mul(
            quat_conjugate(robot_root_quat_w).expand_as(ref_body_quat_future_w),
            ref_body_quat_future_w
        )
        self.ref_body_ori_future_b = matrix_from_quat(ref_body_quat_future_b)
    
    def compute(self):
        return self.ref_body_ori_future_b[:, :, :, :2, :].reshape(self.num_envs, -1)

class ref_body_ori_future_local(TrackObservation):
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

class diff_body_ori_future_b(TrackObservation):
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
    
class diff_body_ori_future_local(TrackObservation):
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

class ref_motion_phase(TrackObservation):
    def compute(self):
        return (self.command_manager.t / self.command_manager.motion_len).unsqueeze(1)
    
class ref_motion_phase_noise(TrackObservation):
    def compute(self):
        return torch.randn(self.num_envs, 1, device=self.device)

from active_adaptation.envs.mdp.commands.motion_tracking.command import MotionTrackingDoor
from isaaclab.utils.math import wrap_to_pi

def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = torch.unbind(quat, dim=-1)
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return yaw

TrackDoorObservation = BaseObservation[MotionTrackingDoor]

class door_pos_b(TrackDoorObservation):
    def __init__(self, noise_std: float=0.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = max(0.0, noise_std)

    def compute(self):
        door_pos_w = self.command_manager.door.data.root_link_pos_w
        robot_pos_w = self.command_manager.asset.data.root_link_pos_w
        robot_quat_w = self.command_manager.asset.data.root_link_quat_w
        robot_quat_yaw_w = (robot_quat_w)
        door_pos_b = quat_apply_inverse(robot_quat_yaw_w, door_pos_w - robot_pos_w)
        door_pos_b = door_pos_b[:, :2]
        if self.noise_std > 0.0:
            door_pos_b = door_pos_b + torch.randn_like(door_pos_b).clamp(-3., 3.) * self.noise_std
        return door_pos_b
    
class root_yaw(TrackDoorObservation):
    def __init__(self, noise_std: float=0.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = max(0.0, noise_std)

    def compute(self):
        robot_quat_w = self.command_manager.asset.data.root_link_quat_w
        door_quat_w = self.command_manager.door.data.root_link_quat_w
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


from active_adaptation.envs.mdp.commands.motion_tracking.command import MotionTrackingBox

TrackBoxObservation = BaseObservation["MotionTrackingBox"]

class box_pos_b(TrackBoxObservation):
    def __init__(self, noise_std: float=0.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = max(0.0, noise_std)

    def compute(self):
        box_pos_w = self.command_manager.box.data.root_link_pos_w
        robot_pos_w = self.command_manager.asset.data.root_link_pos_w
        robot_quat_w = self.command_manager.asset.data.root_link_quat_w
        robot_quat_yaw_w = yaw_quat(robot_quat_w)
        box_pos_b = quat_apply_inverse(robot_quat_yaw_w, box_pos_w - robot_pos_w)[..., :2]
        if self.noise_std > 0.0:
            box_pos_b = box_pos_b + torch.randn_like(box_pos_b).clamp(-3., 3.) * self.noise_std
        return box_pos_b
    
class box_yaw(TrackBoxObservation):
    def __init__(self, noise_std: float=0.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = max(0.0, noise_std)

    def compute(self):
        robot_quat_w = self.command_manager.asset.data.root_link_quat_w
        box_quat_w = self.command_manager.box.data.root_link_quat_w
        robot_yaw_w = yaw_from_quat(robot_quat_w)
        box_yaw_w = yaw_from_quat(box_quat_w)
        box_yaw = wrap_to_pi(box_yaw_w + torch.pi - robot_yaw_w)
        if self.noise_std > 0.0:
            box_yaw = box_yaw + torch.randn_like(box_yaw).clamp(-3., 3.) * self.noise_std
        return box_yaw.unsqueeze(-1)
    
    
class box_lin_vel_b(TrackBoxObservation):
    def compute(self):
        box_vel_w = self.command_manager.box.data.root_com_lin_vel_w
        robot_quat_w = self.command_manager.asset.data.root_link_quat_w
        box_vel_w_b = quat_apply_inverse(robot_quat_w, box_vel_w)
        return box_vel_w_b
    
class box_ang_vel_b(TrackBoxObservation):
    def compute(self):
        box_ang_vel_w = self.command_manager.box.data.root_com_ang_vel_w
        robot_quat_w = self.command_manager.asset.data.root_link_quat_w
        box_ang_vel_w_b = quat_apply_inverse(robot_quat_w, box_ang_vel_w)
        return box_ang_vel_w_b
    
class diff_box_pos_future(TrackBoxObservation):
    def compute(self):
        ref_box_pos_future_w = self.command_manager.future_ref_motion.body_pos_w[:, :, self.command_manager.box_body_id_motion]
        cur_box_pos_w = self.command_manager.box.data.root_link_pos_w.unsqueeze(1)
        diff_box_pos_future_w = ref_box_pos_future_w - cur_box_pos_w
        # shape: [num_envs, num_future_steps, 2]

        robot_quat_w = self.command_manager.asset.data.root_link_quat_w
        robot_quat_yaw_w = yaw_quat(robot_quat_w).unsqueeze(1)
        diff_box_pos_future_w_b = quat_apply_inverse(robot_quat_yaw_w, diff_box_pos_future_w)
        return diff_box_pos_future_w_b[..., :2].reshape(self.num_envs, -1)
    
class diff_box_yaw_future(TrackBoxObservation):
    def compute(self):
        ref_box_quat_future_w = self.command_manager.future_ref_motion.body_quat_w[:, :, self.command_manager.box_body_id_motion]
        cur_box_quat_w = self.command_manager.box.data.root_link_quat_w.unsqueeze(1)
        ref_box_yaw_future_w = yaw_from_quat(ref_box_quat_future_w)
        cur_box_yaw_w = yaw_from_quat(cur_box_quat_w)
        diff_box_yaw_future_w = ref_box_yaw_future_w - cur_box_yaw_w
        return diff_box_yaw_future_w.view(self.num_envs, -1)
    
class box_friction(TrackBoxObservation):
    def compute(self):
        raise NotImplementedError
    
    
class box_contact(TrackBoxObservation):
    def compute(self):
        return self.command_manager.future_ref_box_contact[:, 0:1].float()

class box_contact_future(TrackBoxObservation):
    def compute(self):
        return self.command_manager.future_ref_box_contact.float()

class eef_contact_pos_b(TrackBoxObservation):
    def __init__(self, noise_std: float=0.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = max(0.0, noise_std)

    def compute(self):
        robot_quat_w = self.command_manager.asset.data.root_link_quat_w.unsqueeze(1)
        robot_pos_w = self.command_manager.asset.data.root_link_pos_w.unsqueeze(1)
        eef_contact_pos_w = self.command_manager.contact_eef_pos_w
        eef_contact_pos_b = quat_apply_inverse(yaw_quat(robot_quat_w), eef_contact_pos_w - robot_pos_w)
        if self.noise_std > 0.0:
            eef_contact_pos_b = eef_contact_pos_b + torch.randn_like(eef_contact_pos_b).clamp(-3., 3.) * self.noise_std
        return eef_contact_pos_b.reshape(self.num_envs, -1)

class eef_target_pos_b(TrackBoxObservation):
    def __init__(self, noise_std: float=0.0, episodic_noise_std: float=0.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = max(0.0, noise_std)
        self.episodic_noise_std = max(0.0, episodic_noise_std)
        self.step_noise = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.episode_noise = torch.zeros(self.num_envs, 2, 3, device=self.device)
    
    def reset(self, env_ids):
        if self.episodic_noise_std > 0.0:
            self.episode_noise[env_ids] = torch.empty(len(env_ids), 2, 3, device=self.device).uniform_(-1., 1.) * self.episodic_noise_std
    
    def update(self):
        if self.noise_std > 0.0:
            self.step_noise[:] = torch.randn(self.num_envs, 2, 3, device=self.device).clamp(-3., 3.) * self.noise_std

    def compute(self):
        robot_quat_w = self.command_manager.asset.data.root_link_quat_w.unsqueeze(1)
        robot_pos_w = self.command_manager.asset.data.root_link_pos_w.unsqueeze(1)
        eef_target_pos_w = self.command_manager.contact_target_pos_w
        eef_target_pos_b = quat_apply_inverse(yaw_quat(robot_quat_w), eef_target_pos_w - robot_pos_w)
        eef_target_pos_b = eef_target_pos_b + self.step_noise + self.episode_noise
        return eef_target_pos_b.reshape(self.num_envs, -1)

class eef_contact_diff_pos_b(TrackBoxObservation):
    def compute(self):
        eef_contact_diff_pos_w = self.command_manager.contact_target_pos_w - self.command_manager.contact_eef_pos_w
        robot_quat_w = self.command_manager.asset.data.root_link_quat_w.unsqueeze(1)
        eef_contact_diff_b = quat_apply_inverse(robot_quat_w, eef_contact_diff_pos_w)
        return eef_contact_diff_b.view(self.num_envs, -1)

class eef_contact_diff_ori_b(TrackBoxObservation):
    def compute(self):
        eef_contact_diff_euler = self.command_manager.contact_target_euler_xyz - self.command_manager.contact_eef_euler_xyz
        eef_contact_diff_mat = matrix_from_euler(eef_contact_diff_euler, "XYZ")
        return eef_contact_diff_mat[:, :, :2, :].reshape(self.num_envs, -1)

class eef_contact_diff_euler_b(TrackBoxObservation):
    def compute(self):
        eef_contact_diff_euler = self.command_manager.contact_target_euler_xyz - self.command_manager.contact_eef_euler_xyz
        return eef_contact_diff_euler.view(self.num_envs, -1)
