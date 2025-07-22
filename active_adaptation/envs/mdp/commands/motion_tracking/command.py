import torch
import numpy as np

from typing import TYPE_CHECKING, List, Tuple, Dict

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor
    from active_adaptation.assets.objects import DoorArticulation
    from isaaclab.assets.rigid_object import RigidObject

from active_adaptation.utils.motion import MotionDataset, MotionData
from ..base import Command
from isaaclab.utils.math import yaw_quat, quat_apply, euler_xyz_from_quat, sample_uniform, quat_from_euler_xyz, quat_mul

def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = torch.unbind(quat, dim=-1)
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return yaw

class MotionTrackingCommand(Command):
    def __init__(
        self, env, data_path: List[str] | str,
        # reset parameters
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
        # observation parameters
        future_steps: List[int] = [1, 2, 8, 16],
        call_update: bool = True,
    ):
        from . import observations
        from . import rewards
        from . import randomizations
        from . import terminations
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
        self.tracking_body_indices_motion = [self.dataset.body_names.index(name) for name in self.tracking_keypoint_names]
        self.tracking_body_indices_asset = [self.asset.body_names.index(name) for name in self.tracking_keypoint_names]

        tracking_joint_names = [
            "waist_.*_joint", 
            ".*_hip_.*_joint", 
            ".*_knee_joint", 
            ".*_ankle_.*_joint", 
            ".*_shoulder_.*_joint", 
            ".*_elbow_joint"
        ]
        self.tracking_joint_names = self.asset.find_joints(tracking_joint_names)[1]
        self.tracking_joint_indices_motion = [self.dataset.joint_names.index(name) for name in self.tracking_joint_names]
        self.tracking_joint_indices_asset = [self.asset.joint_names.index(name) for name in self.tracking_joint_names]

        # Set feet body and joint indices for motion contact and no contact reward
        feet_names = self.asset.find_bodies(".*ankle_roll_link")[1]
        self.feet_ids_motion = [self.dataset.body_names.index(name) for name in feet_names]
        self.feet_ids_asset = [self.asset.body_names.index(name) for name in feet_names]
        self.feet_ids_sensor = [self.contact_forces.body_names.index(name) for name in feet_names]

        # get root body and joint indices in motion for reset
        root_body_name = "pelvis"
        self.root_body_idx_motion = self.dataset.body_names.index(root_body_name)
        
        asset_joint_names = self.asset.joint_names
        self.asset_joint_idx_motion = [self.dataset.joint_names.index(joint_name) for joint_name in asset_joint_names]

        with torch.device(self.device):
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self.future_steps = torch.tensor(future_steps)

            self.motion_ids = torch.zeros(self.num_envs, dtype=int)
            self.motion_len = torch.zeros(self.num_envs, dtype=int)
            self.t = torch.zeros(self.num_envs, dtype=torch.long)

            self.eval_t = torch.randint(0, self.dataset.lengths[0], (self.num_envs,), device=self.device)

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

        if not self.env.training:
            # self.t[env_ids] = self.eval_t[env_ids]
            self.t[env_ids] = 0
            pass

    def sample_init(self, env_ids: torch.Tensor) -> None:
        self._sample_motions(env_ids)

        # reset root state and joint position/velocity from motion
        self._motion_reset: MotionData = self.dataset.get_slice(self.motion_ids[env_ids], self.t[env_ids], 1).squeeze(1)
        # shape: [len(env_ids), num_bodies/num_joints, 3/4/...]
        
        motion = self._motion_reset
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

        self.asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        self.asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)

        init_joint_pos = motion.joint_pos[:, self.asset_joint_idx_motion]
        init_joint_vel = motion.joint_vel[:, self.asset_joint_idx_motion]

        joint_pos_noise = sample_uniform(-1, 1, (init_joint_pos.shape[0], init_joint_pos.shape[1]), device=self.device) * self.init_joint_pos_noise
        joint_vel_noise = sample_uniform(-1, 1, (init_joint_vel.shape[0], init_joint_vel.shape[1]), device=self.device) * self.init_joint_vel_noise

        init_joint_pos += joint_pos_noise
        init_joint_vel += joint_vel_noise

        joint_pos_limits = self.asset.data.soft_joint_pos_limits[env_ids]
        joint_vel_limits = self.asset.data.soft_joint_vel_limits[env_ids]
        init_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        init_joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

        self.asset.write_joint_state_to_sim(init_joint_pos, init_joint_vel, env_ids=env_ids)

    @property
    def success(self):
        return (self.t >= self.motion_len - 1).unsqueeze(1)
    
    @property
    def finished(self):
        # if not self.env.training:
        #     return torch.ones(self.num_envs, 1, dtype=bool, device=self.device)
        return (self.t >= self.motion_len).unsqueeze(1)

    def update(self):
        # future ref motion for actor observation
        self.future_ref_motion = self.dataset.get_slice(self.motion_ids, self.t, steps=self.future_steps)
        # shape: [num_envs, len(future_steps), num_bodies/num_joints, 3/4/...]

        # Observations: future ref and diff to body frame
        self.ref_body_pos_future_w = self.future_ref_motion.body_pos_w[..., self.tracking_body_indices_motion, :] + self.env.scene.env_origins[:, None, None, :]
        self.ref_body_lin_vel_future_w = self.future_ref_motion.body_lin_vel_w[..., self.tracking_body_indices_motion, :]
        self.ref_body_quat_future_w = self.future_ref_motion.body_quat_w[..., self.tracking_body_indices_motion, :]
        self.ref_body_ang_vel_future_w = self.future_ref_motion.body_ang_vel_w[..., self.tracking_body_indices_motion, :]
        self.ref_joint_pos_future_ = self.future_ref_motion.joint_pos[..., self.tracking_joint_indices_motion]
        self.ref_joint_vel_future_ = self.future_ref_motion.joint_vel[..., self.tracking_joint_indices_motion]
        self.ref_root_pos_future_w = self.future_ref_motion.body_pos_w[..., self.root_body_idx_motion, :] + self.env.scene.env_origins[:, None, :]
        self.ref_root_quat_future_w = self.future_ref_motion.body_quat_w[..., self.root_body_idx_motion, :]
        self.ref_root_lin_vel_future_w = self.future_ref_motion.body_lin_vel_w[..., self.root_body_idx_motion, :]
        self.ref_root_ang_vel_future_w = self.future_ref_motion.body_ang_vel_w[..., self.root_body_idx_motion, :]

        # Reward: current robot and ref motion for reward computation
        self.robot_body_pos_w = self.asset.data.body_link_pos_w[:, self.tracking_body_indices_asset]
        self.robot_body_lin_vel_w = self.asset.data.body_com_lin_vel_w[:, self.tracking_body_indices_asset]
        self.robot_body_quat_w = self.asset.data.body_link_quat_w[:, self.tracking_body_indices_asset]
        self.robot_body_ang_vel_w = self.asset.data.body_com_ang_vel_w[:, self.tracking_body_indices_asset]
        self.robot_joint_pos = self.asset.data.joint_pos[:, self.tracking_joint_indices_asset]
        self.robot_joint_vel = self.asset.data.joint_vel[:, self.tracking_joint_indices_asset]
        self.robot_root_pos_w = self.asset.data.root_link_pos_w
        self.robot_root_quat_w = self.asset.data.root_link_quat_w

        self.current_ref_motion: MotionData = self.future_ref_motion[:, 0]
        self.ref_body_pos_w = self.ref_body_pos_future_w[:, 0]
        self.ref_body_lin_vel_w = self.ref_body_lin_vel_future_w[:, 0]
        self.ref_body_quat_w = self.ref_body_quat_future_w[:, 0]
        self.ref_body_ang_vel_w = self.ref_body_ang_vel_future_w[:, 0]
        self.ref_joint_pos = self.ref_joint_pos_future_[:, 0]
        self.ref_joint_vel = self.ref_joint_vel_future_[:, 0]
        self.ref_root_pos_w = self.ref_root_pos_future_w[:, 0]
        self.ref_root_quat_w = self.ref_root_quat_future_w[:, 0]
        self.ref_root_lin_vel_w = self.ref_root_lin_vel_future_w[:, 0]
        self.ref_root_ang_vel_w = self.ref_root_ang_vel_future_w[:, 0]
        # shape: [num_envs, num_future_steps, num_tracking_bodies, xxx]

        if self.env.backend == "isaac":
            self.all_marker_pos_w[0] = self.robot_body_pos_w
            self.all_marker_pos_w[1] = self.ref_body_pos_w
            # self.all_marker_pos_w[0] = self.ref_body_pos_future_w[:, 0]
            # self.all_marker_pos_w[1] = self.ref_body_pos_future_w[:, -1]

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


class MotionTrackingDoor(MotionTrackingCommand):
    def __init__(
        self,
        contact_eef_name: str="right_wrist_yaw_link",
        reset_range: Tuple[int, int]=(0, 200),
        door_reset_offset: Tuple[float, float, float]=(0.0, 0.0, 0.0),
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
        self.door_body_id_asset = self.door.body_names.index(door_body_name)

        self.door_reset_offset = torch.tensor(door_reset_offset, device=self.device)
        self.reset_range = reset_range

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

        motion = self._motion_reset
        init_door_pos = motion.body_pos_w[:, self.wall_body_id_motion] + self.door_reset_offset
        init_door_pos[:, 2].fill_(0.0)
        init_door_quat = motion.body_quat_w[:, self.wall_body_id_motion]

        init_door_root_state_w = self.door.data.default_root_state[env_ids]
        init_door_root_state_w[:, 0:3] = init_door_pos + self.env.scene.env_origins[env_ids]
        init_door_root_state_w[:, 3:7] = init_door_quat

        self.door.write_root_link_pose_to_sim(init_door_root_state_w[:, :7], env_ids=env_ids)
        
        init_door_joint_pos = motion.joint_pos[:, self.door_joint_id_motion].unsqueeze(-1)
        init_door_joint_vel = motion.joint_vel[:, self.door_joint_id_motion].unsqueeze(-1)
        self.door.write_joint_position_to_sim(init_door_joint_pos, env_ids=env_ids, joint_ids=[self.door_joint_id_asset])
        self.door.write_joint_velocity_to_sim(init_door_joint_vel, env_ids=env_ids, joint_ids=[self.door_joint_id_asset])
           
    def update(self):
        super().update()
        # Reward: reference door joint position
        self.ref_door_joint_pos = self.current_ref_motion.joint_pos[:, self.door_joint_id_motion]
        self.ref_door_joint_vel = self.current_ref_motion.joint_vel[:, self.door_joint_id_motion]
        # shape: [num_envs]

class MotionTrackingBox(MotionTrackingCommand):
    def __init__(
        self, 
        contact_eef_name: str=".*_wrist_yaw_link",
        reset_range: Tuple[int, int]=(0, 100),
        box_pose_range: Dict[str, Tuple[float, float]]={"x": (-0.05, 0.05), "y": (-0.0, 0.0), "z": (0.0, 0.0), "roll": (-0.0, 0.0), "pitch": (-0.0, 0.0), "yaw": (-0.1, 0.1)},
        **kwargs
    ):
        super().__init__(**kwargs, call_update=False)
        box_object_name = "box"
        box_body_name = "box"
        
        self.box: RigidObject = self.env.scene.rigid_objects[box_object_name]

        self.box_body_id_motion = self.dataset.body_names.index(box_body_name)
        self.box_body_id_asset = self.box.body_names.index(box_body_name)

        scale = getattr(self.box.cfg.spawn, "scale", None)
        if not isinstance(scale, torch.Tensor):
            scale_tensor = torch.ones(self.num_envs, 3)
            if scale is None:
                pass
            elif isinstance(scale, float):
                scale_tensor[:] = scale
            elif isinstance(scale, tuple):
                scale_tensor[:] = torch.tensor(scale, device=self.device)
            else:
                raise ValueError(f"Invalid scale type: {type(scale)}")
            scale = scale_tensor
        box_scale: torch.Tensor = scale.to(self.device)
        # box_size = torch.tensor(self.box.cfg.spawn.size, device=self.device) # this is for cuboid
        box_size = torch.tensor([1.0, 0.8, 0.8], device=self.device)
        self.box_size = box_size * box_scale
        _, contact_eef_names = self.asset.find_bodies(contact_eef_name)
        assert len(contact_eef_names) == 2

        with torch.device(self.device):
            self.contact_target_pos_offset = torch.zeros(self.num_envs, 2, 3, device=self.device)
            self.contact_target_pos_offset[:] = self.box_size.unsqueeze(1)
            self.contact_target_pos_offset[:, :, 0] = 0.0
            self.contact_target_pos_offset[:, 0, 1] = -0.15
            self.contact_target_pos_offset[:, 1, 1] = 0.15

            self.contact_eef_pos_offset = torch.tensor([(0.1, 0.0, 0.0), (0.1, 0.0, 0.0)]).unsqueeze(0).expand(self.num_envs, -1, -1)

            self.contact_target_pos_w = torch.zeros(self.num_envs, len(contact_eef_names), 3)
            self.contact_eef_pos_w = torch.zeros(self.num_envs, len(contact_eef_names), 3)
            self.eef_contact_force = torch.zeros(self.num_envs, len(contact_eef_names), 3)
            # shape: [num_envs, num_contact_eefs, 3]
        
        self.eef_idx_asset = []
        self.eef_idx_sensor = []
        for eef_name in contact_eef_names:
            self.eef_idx_asset.append(self.asset.body_names.index(eef_name))
            self.eef_idx_sensor.append(self.contact_forces.body_names.index(eef_name))
        
        pose_range_list = [box_pose_range.get(k, (-0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.box_pose_range = torch.tensor(pose_range_list, device=self.device)
        
        # data_path = kwargs["data_path"]
        # from pathlib import Path
        # import active_adaptation
        # active_adaptation_path = Path(active_adaptation.__file__).parent.parent
        # data_path = active_adaptation_path / Path(data_path)
        # motion_data = np.load(data_path / "motion.npz")
        # box_contact = motion_data["box_contact"]
        # self.box_contact = torch.from_numpy(box_contact).to(self.device).type(torch.bool).squeeze(-1)
        # # shape: [n_steps]
        
        motion_paths = self.dataset.motion_paths
        box_contact = []
        for motion_path in motion_paths:
            motion_data = np.load(motion_path)
            box_contact.append(motion_data["box_contact"])
        box_contact = np.concatenate(box_contact, axis=0)
        self.box_contact = torch.from_numpy(box_contact).to(self.device).type(torch.bool).squeeze(-1)
        # shape: [n_steps]

        self.reset_range = reset_range

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
        init_box_pos = motion.body_pos_w[:, self.box_body_id_motion]
        init_box_quat = motion.body_quat_w[:, self.box_body_id_motion]
        init_box_pos[:, 2] = 0.0

        rand_samples = sample_uniform(self.box_pose_range[:, 0], self.box_pose_range[:, 1], (len(env_ids), 6), device=self.device)
        init_box_pos += rand_samples[:, 0:3]

        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        init_box_quat = quat_mul(init_box_quat, orientations_delta)

        init_box_state_w = self.box.data.default_root_state[env_ids]
        init_box_state_w[:, 0:3] = init_box_pos + self.env.scene.env_origins[env_ids]
        init_box_state_w[:, 3:7] = init_box_quat
        init_box_state_w[:, 7:] = 0.0
        
        self.box.write_root_link_pose_to_sim(init_box_state_w[:, :7], env_ids=env_ids)
        self.box.write_root_com_velocity_to_sim(init_box_state_w[:, 7:], env_ids=env_ids)
        
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
        box_pos_w = self.box.data.root_link_pos_w.unsqueeze(1).expand(-1, 2, -1)
        box_quat_w = self.box.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
        self.contact_target_pos_w[:] = box_pos_w + quat_apply(
            box_quat_w.reshape(-1, 4), self.contact_target_pos_offset.reshape(-1, 3)
        ).view(self.num_envs, 2, 3)

        eef_pos_w = self.asset.data.body_link_pos_w[:, self.eef_idx_asset]
        eef_quat_w = self.asset.data.body_quat_w[:, self.eef_idx_asset]
        self.contact_eef_pos_w[:] = eef_pos_w + quat_apply(eef_quat_w, self.contact_eef_pos_offset)

        # eef target ori
        target_quat_w = yaw_quat(self.box.data.root_link_quat_w)
        eef_target_euler_xyz = euler_xyz_from_quat(target_quat_w)
        self.contact_target_euler_xyz = torch.stack(eef_target_euler_xyz, dim=1).unsqueeze(1)
        self.contact_target_euler_xyz[:, :, 2] += torch.pi

        eef_quat_w = self.asset.data.body_quat_w[:, self.eef_idx_asset]
        eef_euler_xyz = euler_xyz_from_quat(eef_quat_w.view(-1, 4))
        self.contact_eef_euler_xyz = torch.stack(eef_euler_xyz, dim=1).view(self.num_envs, 2, 3)

        # eef contact
        self.eef_contact_force[:] = self.contact_forces.data.net_forces_w[:, self.eef_idx_sensor]
    
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
        self.box_pos_markers_pos_w[1] = self.box.data.root_link_pos_w + self.box_vis_offset
        
        self.eef_contact_markers.visualize(
            translations=self.eef_contact_markers_pos_w.view(-1, 3),
            marker_indices=self.eef_contact_markers_indices,
        )
        self.box_pos_markers.visualize(
            translations=self.box_pos_markers_pos_w.view(-1, 3),
            marker_indices=self.box_pos_markers_indices,
        )
        
        
# class MotionTrackingTransportBox(MotionTrackingCommand):
#     def __init__(
#         self, 
#         # contact_eef_name: str=".*_wrist_yaw_link",
#         reset_range: Tuple[int, int]=(0, 200),
#         **kwargs
#     ):
#         super().__init__(**kwargs, call_update=False)
#         box_object_name = "box_small"
#         box_body_name = "box_small"
        
#         self.box: RigidObject = self.env.scene.rigid_objects[box_object_name]

#         self.box_body_id_motion = self.dataset.body_names.index(box_body_name)
#         self.box_body_id_asset = self.box.body_names.index(box_body_name)
    
#         from active_adaptation.utils.motion import lerp
#         import json
#         box_contacts = []
#         with open(self.dataset.motion_paths[0].parent / "meta.json", "rb") as f:
#             meta = json.load(f)
#         src_fps = meta["fps"]
#         tgt_fps = 1 / self.env.step_dt
#         for motion_path in self.dataset.motion_paths:
#             motion_data = np.load(motion_path)
#             T = motion_data["joint_pos"].shape[0]
#             end_t = T / src_fps
#             ts_source = np.arange(0, end_t, 1 / src_fps)
#             ts_target = np.arange(0, end_t, 1 / tgt_fps)
#             if ts_target[-1] > ts_source[-1]:
#                 ts_target = ts_target[:-1]

#             box_contact_ = lerp(ts_target, ts_source, motion_data["box_contact"]) > 0.5
#             box_contacts.append(box_contact_)
#         box_contact = np.concatenate(box_contacts, axis=0)
#         self.box_contact = torch.from_numpy(box_contact).to(self.device).type(torch.bool).squeeze(-1)
#         # shape: [n_steps]

#         self.reset_range = reset_range

#         self._init_debug_draw()
#         self.update()
    
#     def _sample_motions(self, env_ids):
#         super()._sample_motions(env_ids)
#         start_t = torch.randint(*self.reset_range, (len(env_ids),), device=self.device)
#         self.t[env_ids] = start_t
#         if not self.env.training:
#             self.t[env_ids] = 0

#     def sample_init(self, env_ids):
#         super().sample_init(env_ids)
        
#         motion: MotionData = self._motion_reset
#         init_box_pos = motion.body_pos_w[:, self.box_body_id_motion]
#         init_box_quat = motion.body_quat_w[:, self.box_body_id_motion]
#         init_box_pos[:, 2] = 0.3
        
#         init_box_state_w = self.box.data.default_root_state[env_ids]
#         init_box_state_w[:, 0:3] = init_box_pos + self.env.scene.env_origins[env_ids]
#         init_box_state_w[:, 3:7] = init_box_quat
#         init_box_state_w[:, 7:] = 0.0
        
#         self.box.write_root_link_pose_to_sim(init_box_state_w[:, :7], env_ids=env_ids)
#         self.box.write_root_com_velocity_to_sim(init_box_state_w[:, 7:], env_ids=env_ids)
    
#     def update(self):
#         super().update()
#         self.ref_box_pos_w = self.current_ref_motion.body_pos_w[:, self.box_body_id_motion] + self.env.scene.env_origins
#         self.ref_box_quat_w = self.current_ref_motion.body_quat_w[:, self.box_body_id_motion]
#         # shape: [num_envs, 3]
#         current_idx = self.dataset.starts[self.motion_ids] + self.t
#         current_idx.clamp_max_(self.dataset.ends[self.motion_ids] - 1)
#         self.ref_box_contact = self.box_contact[current_idx]
    
#     def debug_draw(self):
#         super().debug_draw()
#         # draw a point on box when ref_box_contact is True
#         if self.env.backend != "isaac":
#             return

#         box_pos_w = self.box.data.root_link_pos_w + torch.tensor([0.0, 0.0, 0.4], device=self.device)
#         box_pos_w = box_pos_w[self.ref_box_contact]
#         self.env.debug_draw.point(
#             box_pos_w,
#             color=(0.0, 1.0, 0.5, 1.0),
#             size=20,
#         )
        
# TrackTransportBoxObservation = BaseObservation["MotionTrackingTransportBox"]

# class box_pos_b_track_transport(TrackTransportBoxObservation):
#     def compute(self):
#         box_pos_w = self.command_manager.box.data.root_link_pos_w
#         robot_pos_w = self.command_manager.asset.data.root_link_pos_w
#         robot_quat_w = self.command_manager.asset.data.root_link_quat_w

#         robot_quat_yaw_w = yaw_quat(robot_quat_w)
#         box_pos_b = quat_rotate_inverse(robot_quat_yaw_w, box_pos_w - robot_pos_w)
#         return box_pos_b

# # class box_target_pos_b_track_transport(TrackTransportBoxObservation):
# #     def compute(self):
# #         box_pos_w = self.command_manager.box.data.root_link_pos_w
# #         robot_pos_w = self.command_manager.asset.data.root_link_pos_w
# #         robot_quat_w = self.command_manager.asset.data.root_link_quat_w

# #         robot_quat_yaw_w = yaw_quat(robot_quat_w)
# #         box_target_pos_b = quat_rotate_inverse(robot_quat_yaw_w, box_pos_w - robot_pos_w)
# #         return box_target_pos_b
        

# TrackTransportBoxReward = BaseReward["MotionTrackingTransportBox"]

# class box_pos_tracking_track_transport(TrackTransportBoxReward):
#     def __init__(self, sigma: float=0.1, **kwargs):
#         super().__init__(**kwargs)
#         self.sigma = sigma

#     def compute(self):
#         box_pos_w = self.command_manager.box.data.root_link_pos_w
#         ref_box_pos_w = self.command_manager.ref_box_pos_w
#         diff_box_pos_w_b = ref_box_pos_w - box_pos_w
#         error = diff_box_pos_w_b.norm(dim=-1)
#         return torch.exp(-error / self.sigma).unsqueeze(-1)
    
# class box_ori_tracking_track_transport(TrackTransportBoxReward):
#     def __init__(self, sigma: float=0.1, **kwargs):
#         super().__init__(**kwargs)
#         self.sigma = sigma

#     def compute(self):
#         box_quat_w = self.command_manager.box.data.root_link_quat_w
#         ref_box_quat_w = self.command_manager.ref_box_quat_w
#         diff_box_quat_w = quat_mul(quat_conjugate(box_quat_w), ref_box_quat_w)
#         error = axis_angle_from_quat(diff_box_quat_w).norm(dim=-1)
#         return torch.exp(-error / self.sigma).unsqueeze(-1)

# class box_contact_track_transport(TrackTransportBoxReward):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.left_hand_contact_sensor: ContactSensor = self.env.scene.sensors["left_hand_box"]
#         self.right_hand_contact_sensor: ContactSensor = self.env.scene.sensors["right_hand_box"]
#         self.hand_contact_forces = torch.zeros(self.num_envs, 2, 3, device=self.device)
#         self.hand_in_contact = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.bool)

#     def update(self):
#         self.hand_contact_forces[:, 0] = self.left_hand_contact_sensor.data.net_forces_w[:, 0, :]
#         self.hand_contact_forces[:, 1] = self.right_hand_contact_sensor.data.net_forces_w[:, 0, :]
#         hand_contact_frc_norm = self.hand_contact_forces.norm(dim=-1)
#         self.hand_in_contact[:] = (hand_contact_frc_norm > 2.0)
    
#     def compute(self):
#         ref_box_contact = self.command_manager.ref_box_contact
#         rew = ref_box_contact.unsqueeze(1) & self.hand_in_contact
#         # shape: [num_envs, 2]
#         return rew.float().mean(dim=-1, keepdim=True)
        

# TrackTransportBoxTermination = BaseTermination["MotionTrackingTransportBox"]

# class cum_lost_contact_steps_track_transport(TrackTransportBoxTermination):
#     def __init__(self, min_steps: int = 25, **kwargs):
#         super().__init__(**kwargs)
#         self.left_hand_contact_sensor: ContactSensor = self.env.scene.sensors["left_hand_box"]
#         self.right_hand_contact_sensor: ContactSensor = self.env.scene.sensors["right_hand_box"]
#         self.hand_contact_forces = torch.zeros(self.num_envs, 2, 3, device=self.device)
#         self.hand_in_contact = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.bool)

#         self.lost_contact_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
#         self.min_steps = min_steps
    
#     def reset(self, env_ids: torch.Tensor) -> None:
#         self.lost_contact_steps[env_ids] = 0
    
#     def update(self):
#         self.hand_contact_forces[:, 0] = self.left_hand_contact_sensor.data.net_forces_w[:, 0, :]
#         self.hand_contact_forces[:, 1] = self.right_hand_contact_sensor.data.net_forces_w[:, 0, :]
#         hand_contact_frc_norm = self.hand_contact_forces.norm(dim=-1)
#         self.hand_in_contact[:] = (hand_contact_frc_norm > 2.0)
        
#         lost_contact = (~self.hand_in_contact).any(dim=-1) & self.command_manager.ref_box_contact
#         self.lost_contact_steps[lost_contact] += 1
#         self.lost_contact_steps[~lost_contact] = 0
    
#     def __call__(self):
#         return (self.lost_contact_steps >= self.min_steps).float().unsqueeze(-1)
    