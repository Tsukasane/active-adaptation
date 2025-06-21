import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor

from active_adaptation.envs.mdp import Reward as BaseReward, Observation as BaseObservation
from active_adaptation.utils.motion import MotionDataset, MotionData
from active_adaptation.utils.math import (
    quat_rotate_inverse,
    quat_mul,
    quat_conjugate,
    axis_angle_from_quat
)
from .base import Command
from isaaclab.utils.math import yaw_quat, matrix_from_quat


class MotionTrackingCommand(Command):
    def __init__(self, env, data_path: str):
        super().__init__(env)
        self.contact_forces: ContactSensor = self.env.scene["contact_forces"]

        self.dataset = MotionDataset.create_from_path(
            data_path,
            target_fps=int(1/self.env.step_dt)
        ).to(self.device)

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
        tracking_keypoint_names = self.asset.find_bodies(tracking_keypoint_names)[1]
        self.tracking_body_indices_motion = []
        self.tracking_body_indices_asset = []
        for body_name in tracking_keypoint_names:
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
        tracking_joint_names = self.asset.find_joints(tracking_joint_names)[1]
        self.tracking_joint_indices_motion = []
        self.tracking_joint_indices_asset = []
        for joint_name in tracking_joint_names:
            self.tracking_joint_indices_motion.append(self.dataset.joint_names.index(joint_name))
            self.tracking_joint_indices_asset.append(self.asset.joint_names.index(joint_name))
        
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
            self.future_steps = torch.tensor([1, 2, 8, 16])

            self.motion_ids = torch.zeros(self.num_envs, dtype=int)
            self.motion_len = torch.zeros(self.num_envs, dtype=int)
            self.t = torch.zeros(self.num_envs, dtype=torch.long)

        self._cum_keypoint_pos_scale = 1.0
        self._cum_keypoint_ori_scale = 3.0
        self._cum_joint_pos_scale = 1.0

        self.num_tracking_bodies = len(self.tracking_body_indices_asset)
        self.num_tracking_joints = len(self.tracking_joint_indices_asset)
        self.num_future_steps = len(self.future_steps)

        self.update()

    def sample_init(self, env_ids: torch.Tensor) -> None:
        # sample motion id and start time for each env
        motion_ids = torch.randint(0, self.dataset.num_motions, size=(len(env_ids),), device=self.device)
        motion_len = self.dataset.lengths[motion_ids]
        max_len = motion_len - self.future_steps[-1]
        start_phase = torch.rand(len(env_ids), device=self.device)
        if not self.env.training:
            start_phase.zero_()
        start_t = (start_phase * max_len).long()

        self.motion_ids[env_ids] = motion_ids
        self.motion_len[env_ids] = motion_len
        self.t[env_ids] = start_t
    
        # reset root state and joint position/velocity from motion
        motion: MotionData = self.dataset.get_slice(self.motion_ids[env_ids], self.t[env_ids], 1).squeeze(1)
        # shape: [len(env_ids), num_bodies/num_joints, 3/4/...]

        lift_height = 0.05
        init_root_state = self.init_root_state[env_ids]
        origins = self.env.scene.env_origins[env_ids]
        init_root_state[:, :3] = origins + motion.body_pos_w[:, self.root_body_idx_motion]
        init_root_state[:, 2] += lift_height
        init_root_state[:, 3:7] = motion.body_quat_w[:, self.root_body_idx_motion]
        init_root_state[:, 7:10] = motion.body_lin_vel_w[:, self.root_body_idx_motion]
        init_root_state[:, 10:13] = motion.body_ang_vel_w[:, self.root_body_idx_motion]

        self.asset.write_root_state_to_sim(init_root_state, env_ids=env_ids)
        
        init_joint_pos = motion.joint_pos[:, self.asset_joint_idx_motion]
        init_joint_vel = motion.joint_vel[:, self.asset_joint_idx_motion]

        joint_pos_noise = torch.randn_like(init_joint_pos).clamp(-1, 1) * 0.2
        joint_vel_noise = torch.randn_like(init_joint_vel).clamp(-1, 1) * 0.5
        init_joint_pos += joint_pos_noise
        init_joint_vel += joint_vel_noise

        self.asset.write_joint_state_to_sim(init_joint_pos, init_joint_vel, env_ids=env_ids)

    def reset(self, env_ids):
        pass

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
    
    TrackReward = BaseReward["MotionTrackingCommand"]

    class keypoint_tracking(TrackReward):
        def __init__(self, sigma=0.1, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma

        def compute(self):
            body_pos_asset = self.command_manager.asset.data.body_pos_w[:, self.command_manager.tracking_body_indices_asset]
            body_pos_motion = self.command_manager.ref_body_pos_w[:, self.command_manager.tracking_body_indices_motion]
            diff = body_pos_motion - body_pos_asset
            error = diff.square().sum(-1, keepdim=True)
            return torch.exp(- error / self.sigma).mean(dim=1)
    
    class keypoint_pos_tracking_product(TrackReward):
        def __init__(self, sigma=0.1, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma

        def compute(self):
            body_pos_asset = self.command_manager.asset.data.body_pos_w[:, self.command_manager.tracking_body_indices_asset]
            body_pos_motion = self.command_manager.ref_body_pos_w[:, self.command_manager.tracking_body_indices_motion]
            diff = body_pos_motion - body_pos_asset
            # shape: [num_envs, num_tracking_bodies, 3]
            error = diff.norm(dim=-1)
            # shape: [num_envs, num_tracking_bodies]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

    class keypoint_ori_tracking_product(TrackReward):
        def __init__(self, sigma=0.1, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma

        def compute(self):
            body_ori_asset = self.command_manager.asset.data.body_quat_w[:, self.command_manager.tracking_body_indices_asset]
            body_ori_motion = self.command_manager.ref_body_quat_w[:, self.command_manager.tracking_body_indices_motion]
            diff = quat_mul(quat_conjugate(body_ori_motion), body_ori_asset)
            # shape: [num_envs, num_tracking_bodies, 4]
            error = torch.norm(axis_angle_from_quat(diff), dim=-1)
            # shape: [num_envs, num_tracking_bodies]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
        
    class keypoint_lin_vel_tracking_product(TrackReward):
        def __init__(self, sigma=0.1, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma

        def compute(self):
            body_lin_vel_asset = self.command_manager.asset.data.body_lin_vel_w[:, self.command_manager.tracking_body_indices_asset]
            body_lin_vel_motion = self.command_manager.ref_body_lin_vel_w[:, self.command_manager.tracking_body_indices_motion]
            diff = body_lin_vel_motion - body_lin_vel_asset
            # shape: [num_envs, num_tracking_bodies, 3]
            error = diff.norm(dim=-1)
            # shape: [num_envs, num_tracking_bodies]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

    class keypoint_ang_vel_tracking_product(TrackReward):
        def __init__(self, sigma=0.1, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma

        def compute(self):
            body_ang_vel_asset = self.command_manager.asset.data.body_ang_vel_w[:, self.command_manager.tracking_body_indices_asset]
            body_ang_vel_motion = self.command_manager.ref_body_ang_vel_w[:, self.command_manager.tracking_body_indices_motion]
            diff = body_ang_vel_motion - body_ang_vel_asset
            # shape: [num_envs, num_tracking_bodies, 3]
            error = diff.norm(dim=-1)
            # shape: [num_envs, num_tracking_bodies]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

    class joint_pos_tracking_product(TrackReward):
        def __init__(self, sigma=0.1, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma

        def compute(self):
            joint_pos_asset = self.command_manager.asset.data.joint_pos[:, self.command_manager.tracking_joint_indices_asset]
            joint_pos_motion = self.command_manager.ref_joint_pos[:, self.command_manager.tracking_joint_indices_motion]
            diff = joint_pos_motion - joint_pos_asset
            error = diff.abs()
            # shape: [num_envs, num_tracking_joints]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
        
    class joint_vel_tracking_product(TrackReward):
        def __init__(self, sigma=0.1, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma

        def compute(self):
            joint_vel_asset = self.command_manager.asset.data.joint_vel[:, self.command_manager.tracking_joint_indices_asset]
            joint_vel_motion = self.command_manager.ref_joint_vel[:, self.command_manager.tracking_joint_indices_motion]
            diff = joint_vel_motion - joint_vel_asset
            error = diff.abs()
            # shape: [num_envs, num_tracking_joints]
            return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

    class root_pos_tracking(TrackReward):
        def __init__(self, sigma=0.25, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma

        def compute(self):
            root_pos_asset = self.command_manager.asset.data.root_pos_w
            root_pos_motion = self.command_manager.ref_body_pos_w[:, self.command_manager.root_body_idx_motion]
            diff = root_pos_motion - root_pos_asset
            error = diff.square().sum(-1, keepdim=True)
            return torch.exp(- error / self.sigma)

    class root_ori_tracking(TrackReward):
        def __init__(self, sigma=0.15, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma

        def compute(self):
            root_quat_asset = self.command_manager.asset.data.root_quat_w
            root_quat_motion = self.command_manager.ref_body_quat_w[:, self.command_manager.root_body_idx_motion]
            root_quat_relative = quat_mul(quat_conjugate(root_quat_motion), root_quat_asset)
            error = torch.norm(axis_angle_from_quat(root_quat_relative), dim=-1, keepdim=True)
            return torch.exp(- error / self.sigma)
    
    class root_vel_tracking(TrackReward):
        def __init__(self, sigma=0.25, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma

        def compute(self):
            root_vel_asset = self.command_manager.asset.data.root_lin_vel_w
            root_vel_motion = self.command_manager.ref_body_lin_vel_w[:, self.command_manager.root_body_idx_motion]
            diff = root_vel_motion - root_vel_asset
            error = diff.square().sum(-1, keepdim=True)
            return torch.exp(- error / self.sigma)
        
    class joint_pos_tracking(TrackReward):
        def __init__(self, sigma=0.5, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma

        def compute(self):
            joint_pos_asset = self.command_manager.asset.data.joint_pos[:, self.command_manager.tracking_joint_indices_asset]
            joint_pos_motion = self.command_manager.ref_joint_pos[:, self.command_manager.tracking_joint_indices_motion]
            error = (joint_pos_motion - joint_pos_asset).square()
            return torch.exp(- error / self.sigma).mean(1, True)
    
    class joint_vel_tracking(TrackReward):
        def __init__(self, sigma=0.5, **kwargs):
            super().__init__(**kwargs)
            self.sigma = sigma

        def compute(self):
            joint_vel_asset = self.command_manager.asset.data.joint_vel[:, self.command_manager.tracking_joint_indices_asset]
            joint_vel_motion = self.command_manager.ref_joint_vel[:, self.command_manager.tracking_joint_indices_motion]
            error = (joint_vel_motion - joint_vel_asset).square()
            return torch.exp(- error / self.sigma).mean(1, True)

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
        ref_body_pos_future_w = self.future_ref_motion.body_pos_w[..., self.tracking_body_indices_motion, :]
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

        # current ref motion for reward computation
        self.current_ref_motion: MotionData = self.future_ref_motion[:, 0]
        self.ref_body_pos_w = self.current_ref_motion.body_pos_w + self.env.scene.env_origins[:, None, :]
        self.ref_body_lin_vel_w = self.current_ref_motion.body_lin_vel_w
        self.ref_body_quat_w = self.current_ref_motion.body_quat_w
        self.ref_body_ang_vel_w = self.current_ref_motion.body_ang_vel_w
        self.ref_joint_pos = self.current_ref_motion.joint_pos
        self.ref_joint_vel = self.current_ref_motion.joint_vel
        
        # # shape: [num_envs, num_future_steps, num_tracking_bodies, xxx]
        cur_tracking_body_pos_w = self.asset.data.body_pos_w[:, self.tracking_body_indices_asset]
        ref_tracking_body_pos_w = self.ref_body_pos_w[:, self.tracking_body_indices_motion]
        cur_tracking_body_quat_w = self.asset.data.body_quat_w[:, self.tracking_body_indices_asset]
        ref_tracking_body_quat_w = self.ref_body_quat_w[:, self.tracking_body_indices_motion]
        cur_tracking_joint_pos = self.asset.data.joint_pos[:, self.tracking_joint_indices_asset]
        ref_tracking_joint_pos = self.ref_joint_pos[:, self.tracking_joint_indices_motion]

        diff_body_pos_w = ref_tracking_body_pos_w - cur_tracking_body_pos_w
        diff_body_quat_w = quat_mul(quat_conjugate(ref_tracking_body_quat_w), cur_tracking_body_quat_w)
        diff_joint_pos = ref_tracking_joint_pos - cur_tracking_joint_pos

        error_body_pos = diff_body_pos_w.norm(dim=-1)
        error_body_ori = torch.norm(axis_angle_from_quat(diff_body_quat_w), dim=-1)
        error_joint_pos = diff_joint_pos.abs()
        # print("pos error: ", error_body_pos.mean(dim=1))
        # print("ori error: ", error_body_ori.mean(dim=1))
        # print("joint pos error: ", error_joint_pos.mean(dim=1))

        self._cum_error[:, 0] = error_body_pos.mean(dim=1) / self._cum_keypoint_pos_scale
        self._cum_error[:, 1] = error_body_ori.mean(dim=1) / self._cum_keypoint_ori_scale
        self._cum_error[:, 2] = error_joint_pos.mean(dim=1) / self._cum_joint_pos_scale
                
        self.t += 1

    def debug_draw(self):
        if self.env.backend == "mujoco":
            return
        
        target_keypoints_w = self.ref_body_pos_w[:, self.tracking_body_indices_motion].cpu()
        self.env.debug_draw.point(target_keypoints_w.reshape(-1, 3), color=(1, 0, 0, 1))

        robot_keypoints_w = self.asset.data.body_pos_w[:, self.tracking_body_indices_asset].cpu()
        self.env.debug_draw.point(robot_keypoints_w.reshape(-1, 3), color=(0, 1, 0, 1))

        self.env.debug_draw.vector(
            robot_keypoints_w.reshape(-1, 3),
            target_keypoints_w.reshape(-1, 3) - robot_keypoints_w.reshape(-1, 3),
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
