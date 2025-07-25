from active_adaptation.envs.mdp.base import Command
from active_adaptation.utils.motion import MotionDataset, MotionData

from typing import List, Dict, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor

import torch
import numpy as np
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz, quat_mul, quat_apply
from active_adaptation.utils.math import batchify
quat_apply = batchify(quat_apply)

class RobotTracking(Command):
    def __init__(
        self, env, data_path: List[str] | str,
        # reset parameters
        lift_height: float = 0.00,
        pose_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0., 0.),
            "pitch": (-0., 0.),
            "yaw": (-0., 0.)},
        velocity_range: Dict[str, Tuple[float, float]] = {
            "x": (-0., 0.),
            "y": (-0., 0.),
            "z": (-0., 0.),
            "roll": (-0., 0.),
            "pitch": (-0., 0.),
            "yaw": (-0., 0.)},
        init_joint_pos_noise: float = 0.0,
        init_joint_vel_noise: float = 0.0,
        # observation parameters
        future_steps: List[int] = [1, 2, 8, 16],
        call_update: bool = True,
        sample_motion: bool = False,
        replay_motion: bool = False,
        **kwargs,
    ):
        if kwargs:
            print("Warning: Unused kwargs in RobotTracking:", kwargs)
        from . import observations
        from . import rewards
        # from . import randomizations
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

        self.num_tracking_bodies = len(self.tracking_body_indices_asset)
        self.num_tracking_joints = len(self.tracking_joint_indices_asset)
        self.num_future_steps = len(future_steps)

        # get root body and joint indices in motion for reset
        root_body_name = "pelvis"
        self.root_body_idx_motion = self.dataset.body_names.index(root_body_name)
        
        asset_joint_names = self.asset.joint_names
        self.asset_joint_idx_motion = [self.dataset.joint_names.index(joint_name) for joint_name in asset_joint_names]

        with torch.device(self.device):
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self.future_steps = torch.tensor(future_steps)

            self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_len = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_starts = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_ends = torch.zeros(self.num_envs, dtype=torch.long)
            self.t = torch.zeros(self.num_envs, dtype=torch.long)
            self.replay_motion_t = torch.zeros(self.num_envs, dtype=torch.long)

            self.eval_t = torch.randint(0, self.dataset.lengths[0], (self.num_envs,), device=self.device)

        self.lift_height = lift_height

        pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.pose_range = torch.tensor(pose_range_list, device=self.device)
        velocity_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.velocity_range = torch.tensor(velocity_range_list, device=self.device)

        self.init_joint_pos_noise = init_joint_pos_noise
        self.init_joint_vel_noise = init_joint_vel_noise

        self.first_sample_motion = True
        self.sample_motion = sample_motion
        self.replay_motion = replay_motion

        if call_update:
            self._init_debug_draw()
            self.update()
        
    def _sample_motions(self, env_ids: torch.Tensor) -> None:
        if self.sample_motion or self.first_sample_motion:
            # sample motion id and start time for each env
            motion_ids = torch.randint(0, self.dataset.num_motions, size=(len(env_ids),), device=self.device)
            self.motion_ids[env_ids] = motion_ids
            self.motion_len[env_ids] = motion_len = self.dataset.lengths[motion_ids]
            self.motion_starts[env_ids] = self.dataset.starts[motion_ids]
            self.motion_ends[env_ids] = self.dataset.ends[motion_ids]
            self.first_sample_motion = False
        else:
            motion_len = self.motion_len[env_ids]


        max_len = motion_len - self.future_steps[-1]
        start_phase = torch.rand(len(env_ids), device=self.device)
        start_t = (start_phase * max_len).long()
        
        if not self.env.training:
            start_t.fill_(0)

        if self.replay_motion:
            self.replay_motion_t[env_ids] = (self.replay_motion_t[env_ids] + 1) % motion_len
            start_t = self.replay_motion_t[env_ids]

        self.t[env_ids] = start_t


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
        if self.replay_motion:
            return torch.ones(self.num_envs, 1, dtype=bool, device=self.device)
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

class RobotObjectTracking(RobotTracking):
    def __init__(
        self,
        object_asset_name: str,
        object_body_name: str,
        # for reset
        reset_range: Tuple[float, float] | None = None,
        object_pose_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0., 0.),
            "pitch": (-0., 0.),
            "yaw": (-0., 0.)},
        # for contact rewards
        contact_eef_body_name: List[str] = ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        ## offset from object to contact target position
        contact_target_pos_offset: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        ## offset from end effector
        contact_eef_pos_offset: List[Tuple[float, float, float]] = [(0.1, 0.0, 0.0), (0.1, 0.0, 0.0)],
        **kwargs
    ):
        super().__init__(**kwargs, call_update=False)

        assert len(contact_eef_body_name) == len(contact_target_pos_offset) == len(contact_eef_pos_offset), \
            "contact_eef_body_name, contact_target_pos_offset, and contact_eef_pos_offset must have the same length"

        self.eef_object_contact_forces: List[ContactSensor] = [self.env.scene.sensors[f"{eef_name}_{object_asset_name}_contact_forces"] for eef_name in contact_eef_body_name]

        with torch.device(self.device):
            self.contact_target_pos_offset = torch.tensor(contact_target_pos_offset, device=self.device).expand(self.num_envs, -1, -1)
            self.contact_eef_pos_offset = torch.tensor(contact_eef_pos_offset, device=self.device).expand(self.num_envs, -1, -1)

            self.contact_eef_body_indices_asset = [self.asset.body_names.index(name) for name in contact_eef_body_name]
            self.contact_eef_body_indices_filtered_sensor = [eef_sensor.body_names.index(eef_name) for (eef_name, eef_sensor) in zip(contact_eef_body_name, self.eef_object_contact_forces)]
            self.contact_eef_body_indices_sensor = [self.contact_forces.body_names.index(eef_name) for eef_name in contact_eef_body_name]

            self.contact_target_pos_w = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)
            self.contact_eef_pos_w = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)
            self.eef_contact_forces = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)
        
            
        self.rigid_object = self.env.scene.rigid_objects[object_asset_name]
        
        self.object_body_id_motion = self.dataset.body_names.index(object_body_name)
        
        pose_range_list = [object_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.object_pose_range = torch.tensor(pose_range_list, device=self.device)
        
        motion_paths = self.dataset.motion_paths
        assert len(motion_paths) == 1, "Only one motion path is supported for RobotObjectTracking"
        motion_data = np.load(motion_paths[0], allow_pickle=True)
        object_contact = motion_data["object_contact"]
        self._object_contact = torch.tensor(object_contact, device=self.device, dtype=torch.bool).squeeze(1)
        # shape: [num_steps]

        self._init_debug_draw()
        self.update()
    
    def _sample_motions(self, env_ids):
        super()._sample_motions(env_ids)
    
    def sample_init(self, env_ids):
        super().sample_init(env_ids)
        
        init_object_pos = self._motion_reset.body_pos_w[:, self.object_body_id_motion]
        init_object_quat = self._motion_reset.body_quat_w[:, self.object_body_id_motion]
        
        rand_samples = sample_uniform(self.object_pose_range[:, 0], self.object_pose_range[:, 1], (len(env_ids), 6), device=self.device)

        init_object_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        init_object_quat = quat_mul(init_object_quat, orientations_delta)
        
        init_object_state_w = self.rigid_object.data.default_root_state[env_ids]
        init_object_state_w[:, 0:3] = init_object_pos + self.env.scene.env_origins[env_ids]
        init_object_state_w[:, 3:7] = init_object_quat
        init_object_state_w[:, 7:]  = 0.0  # zero velocity
        
        self.rigid_object.write_root_link_pose_to_sim(init_object_state_w[:, 0:7], env_ids=env_ids)
        self.rigid_object.write_root_com_velocity_to_sim(init_object_state_w[:, 7:], env_ids=env_ids)
    
    def update(self):
        super().update()
        self.ref_object_pos_future_w = self.future_ref_motion.body_pos_w[..., self.object_body_id_motion, :] + self.env.scene.env_origins[:, None, :]
        self.ref_object_quat_future_w = self.future_ref_motion.body_quat_w[..., self.object_body_id_motion, :]
        self.ref_object_pos_w = self.ref_object_pos_future_w[:, 0]
        self.ref_object_quat_w = self.ref_object_quat_future_w[:, 0]
        # shape: [num_envs, 3/4]
        idx = (self.motion_starts + self.t).unsqueeze(1) + self.future_steps.unsqueeze(0)
        idx.clamp_max_(self.motion_ends.unsqueeze(1) - 1)
        self.ref_object_contact_future = self._object_contact[idx]
        self.ref_object_contact = self.ref_object_contact_future[:, 0]
        
        # contact
        object_pos_w = self.rigid_object.data.root_link_pos_w
        object_quat_w = self.rigid_object.data.root_link_quat_w
        self.contact_target_pos_w[:] = object_pos_w.unsqueeze(1) + quat_apply(object_quat_w.unsqueeze(1), self.contact_target_pos_offset)
        
        eef_pos_w = self.asset.data.body_link_pos_w[:, self.contact_eef_body_indices_asset]
        eef_quat_w = self.asset.data.body_link_quat_w[:, self.contact_eef_body_indices_asset]
        self.contact_eef_pos_w[:] = eef_pos_w + quat_apply(eef_quat_w, self.contact_eef_pos_offset)
        
        # for eef_sensor, eef_idx in zip(self.eef_object_contact_forces, self.contact_eef_body_indices_filtered_sensor):
        #     self.eef_contact_forces[:, eef_idx] = eef_sensor.data.force_matrix_w[:, eef_idx, 0]
        self.eef_contact_forces[:] = self.contact_forces.data.net_forces_w[:, self.contact_eef_body_indices_sensor]
        
    def _init_debug_draw(self):
        super()._init_debug_draw()

        if self.env.backend != "isaac":
            return
        
        from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
        import isaaclab.sim as sim_utils
        vis_markers_cfg = VisualizationMarkersCfg(
            prim_path=f"/World/EefContact",
            markers={
                "left": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.5),
                        metallic=1.0,
                    )
                ),
                "right": sim_utils.SphereCfg(
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

    def debug_draw(self):
        super().debug_draw()

        if self.env.backend != "isaac":
            return
        
        self.eef_contact_markers_pos_w[:, 0, :, :] = self.contact_eef_pos_w
        self.eef_contact_markers_pos_w[:, 1, :, :] = self.contact_target_pos_w
        
        self.eef_contact_markers.visualize(
            translations=self.eef_contact_markers_pos_w.view(-1, 3),
            marker_indices=self.eef_contact_markers_indices,
        )