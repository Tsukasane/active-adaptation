from active_adaptation.envs.mdp.commands.hdmi.command import RobotTracking
from typing import Dict, Tuple, List, TYPE_CHECKING
import torch
import numpy as np
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz, quat_mul, quat_apply, quat_apply_inverse

from active_adaptation.utils.math import batchify
quat_apply = batchify(quat_apply)
quat_apply_inverse = batchify(quat_apply_inverse)

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor


class BoxTransport(RobotTracking):
    def __init__(
        self,
        object_asset_name: str, # for finding the object in the scene
        object_body_name: str, # for the body that defines the contact target position
        # for reset
        object_pose_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0., 0.),
            "pitch": (-0., 0.),
            "yaw": (-0., 0.)},
        # for contact rewards
        contact_eef_body_name: List[str] = ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        contact_frc_eef_body_name: List[str | List[str]] = ["left_wrist_(roll|pitch|yaw)_link", "right_wrist_(roll|pitch|yaw)_link"],
        ## offset from object to contact target position
        contact_target_pos_offset: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        ## offset from end effector
        contact_eef_pos_offset: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        **kwargs
    ):
        from . import observations
        super().__init__(**kwargs, call_update=False)

        self.object_asset_name = object_asset_name
        self.object = self.env.scene.rigid_objects[object_asset_name]
        
        self.object_body_id_asset = self.object.body_names.index(object_body_name)
        self.object_body_id_motion = self.dataset.body_names.index(object_asset_name)

        pose_range_list = [object_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.object_pose_range = torch.tensor(pose_range_list, device=self.device)
        if self.replay_motion:
            self.object_pose_range.fill_(0.0)

        # setup contact body indices
        assert len(contact_eef_body_name) == len(contact_target_pos_offset) == len(contact_eef_pos_offset), \
            "contact_eef_body_name, contact_target_pos_offset, and contact_eef_pos_offset must have the same length"
        self.contact_eef_body_indices_asset = [self.asset.body_names.index(name) for name in contact_eef_body_name]

        self.eef_filtered_sensor: List[List[ContactSensor]] = []
        # [self.env.scene.sensors[f"{eef_name}_{object_asset_name}_contact_forces"] for eef_name in contact_eef_body_name] for object_asset_name in self.asset.data.object_names]
        self.eef_filtered_sensor_indices: List[List[int]] = []
        # = [eef_sensor.body_names.index(eef_name) for (eef_name, eef_sensor) in zip(contact_eef_body_name, self.eef_object_contact_forces)]
        for eef_name in contact_eef_body_name:
            eef_names = self.asset.find_bodies(eef_name)[1]
            sensors_for_this_eef = []
            sensor_indices_for_this_eef = []
            for eef_name in eef_names:
                eef_sensor_name = f"{eef_name}_{object_asset_name}_contact_forces"
                eef_sensor_filtered = self.env.scene.sensors[eef_sensor_name]
                sensors_for_this_eef.append(eef_sensor_filtered)
                sensor_indices_for_this_eef.append(eef_sensor_filtered.body_names.index(eef_name))
            self.eef_filtered_sensor.append(sensors_for_this_eef)
            self.eef_filtered_sensor_indices.append(sensor_indices_for_this_eef)

        self.contact_eef_body_indices_sensor = []
        for eef_name in contact_frc_eef_body_name:
            indices, names = self.contact_forces.find_bodies(eef_name)
            self.contact_eef_body_indices_sensor.append(indices)

        with torch.device(self.device):
            self.contact_target_pos_offset = torch.tensor(contact_target_pos_offset, device=self.device).repeat(self.num_envs, 1, 1)
            self.contact_eef_pos_offset = torch.tensor(contact_eef_pos_offset, device=self.device).repeat(self.num_envs, 1, 1)

            self.contact_target_pos_w = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)
            self.contact_target_lin_vel_w = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)
            self.contact_eef_pos_w = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)
            self.contact_eef_lin_vel_w = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)

            self.eef_contact_forces_w = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)
            self.eef_contact_forces_b = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)

        scale = getattr(self.object.cfg.spawn, "scale", None)
        if not isinstance(scale, torch.Tensor):
            scale_tensor = torch.ones(self.num_envs, 3)
            if scale is None:
                pass
            elif isinstance(scale, float):
                scale_tensor[:] = scale
            elif isinstance(scale, tuple):
                scale_tensor[:] = torch.tensor(scale)
            else:
                raise ValueError(f"Invalid scale type: {type(scale)}")
            scale = scale_tensor
        self.contact_target_pos_offset *= scale.unsqueeze(1).to(self.device)

        # load object contact data
        motion_paths = self.dataset.motion_paths
        motion_datas = [np.load(motion_path, allow_pickle=True) for motion_path in motion_paths]
        object_contacts = [motion_data["object_contact"] for motion_data in motion_datas]
        object_contact = np.concatenate(object_contacts, axis=0)
        self._object_contact = torch.tensor(object_contact, device=self.device, dtype=torch.bool).squeeze(1)
        # shape: [num_steps]

        box_final_frames = [motion_data["contact_end_frame"] for motion_data in motion_datas]
        box_final_frame = np.concatenate(box_final_frames, axis=0)
        _box_final_frame = torch.tensor(box_final_frame, device=self.device, dtype=torch.long)
        self._box_final_frame = _box_final_frame + self.dataset.starts
        # shape: [num_motions,]

        with torch.device(self.device):
            self.box_final_frame = torch.zeros(self.num_envs, dtype=torch.long)
            self.box_final_pos_w = torch.zeros(self.num_envs, 3)
            self.box_final_quat_w = torch.zeros(self.num_envs, 4)

        self._init_debug_draw()
        self.update()
    
    def _sample_motions(self, env_ids):
        if self.sample_motion or self.first_sample_motion:
            # sample motion id and start time for each env
            motion_ids = torch.randint(0, self.dataset.num_motions, size=(len(env_ids),), device=self.device)
            self.motion_ids[env_ids] = motion_ids
            self.motion_len[env_ids] = motion_len = self.dataset.lengths[motion_ids]
            self.motion_starts[env_ids] = self.dataset.starts[motion_ids]
            self.motion_ends[env_ids] = self.dataset.ends[motion_ids]

            self.box_final_frame[env_ids] = box_final_frame = self._box_final_frame[motion_ids]

            box_pos_w = self.dataset.data.body_pos_w[:, self.object_body_id_motion]
            box_quat_w = self.dataset.data.body_quat_w[:, self.object_body_id_motion]
            self.box_final_pos_w[env_ids] = box_pos_w[box_final_frame, :] + self.env.scene.env_origins[env_ids]
            self.box_final_quat_w[env_ids] = box_quat_w[box_final_frame, :]
            
            self.first_sample_motion = False
        else:
            motion_len = self.motion_len[env_ids]

        if self.reset_range is None:
            max_len = motion_len - self.future_steps[-1]
            start_phase = torch.rand(len(env_ids), device=self.device)
            start_t = (start_phase * max_len).long()
        else:
            start_t = torch.randint(*self.reset_range, (len(env_ids),), device=self.device)
            
        if not self.env.training:
            start_t.fill_(0)

        if self.replay_motion:
            self.replay_motion_t[env_ids] = (self.replay_motion_t[env_ids] + 1) % motion_len
            start_t = self.replay_motion_t[env_ids]

        self.t[env_ids] = start_t


    def sample_init(self, env_ids):
        if not self.env.training:
            self.object_pose_range.fill_(0.0)
        super().sample_init(env_ids)
         
        init_object_pos = self._motion_reset.body_pos_w[:, self.object_body_id_motion]
        init_object_quat = self._motion_reset.body_quat_w[:, self.object_body_id_motion]

        rand_samples = sample_uniform(self.object_pose_range[:, 0], self.object_pose_range[:, 1], (len(env_ids), 6), device=self.device)

        init_object_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        init_object_quat = quat_mul(init_object_quat, orientations_delta)
        
        init_object_state_w = self.object.data.default_root_state[env_ids]
        init_object_state_w[:, 0:3] = init_object_pos + self.env.scene.env_origins[env_ids]
        init_object_state_w[:, 3:7] = init_object_quat
        init_object_state_w[:, 7:]  = 0.0  # zero velocity

        self.object.write_root_link_pose_to_sim(init_object_state_w[:, 0:7], env_ids=env_ids)
        self.object.write_root_com_velocity_to_sim(init_object_state_w[:, 7:], env_ids=env_ids)

        # robot_pos_w = self.asset.data.root_link_pos_w[env_ids]
        # robot_quat_w = self.asset.data.root_link_quat_w[env_ids]
        # object_pos_b = quat_apply_inverse(robot_quat_w, (init_object_pos + self.env.scene.env_origins[env_ids]) - robot_pos_w)
        # from isaaclab.utils.math import quat_conjugate
        # object_quat_b = quat_mul(quat_conjugate(robot_quat_w), init_object_quat)
        # print(f"Object initial position in robot frame: {object_pos_b}, orientation: {object_quat_b}")
    
    def update(self):
        super().update()
        self.ref_object_pos_future_w = self.future_ref_motion.body_pos_w[..., self.object_body_id_motion, :] + self.env.scene.env_origins[:, None, :]
        self.ref_object_quat_future_w = self.future_ref_motion.body_quat_w[..., self.object_body_id_motion, :]
        self.ref_object_pos_w = self.ref_object_pos_future_w[:, 0]
        self.ref_object_quat_w = self.ref_object_quat_future_w[:, 0]
        self.object_pos_w = self.object.data.root_link_pos_w
        self.object_quat_w = self.object.data.root_link_quat_w

        idx = (self.motion_starts + self.t).unsqueeze(1) + self.future_steps.unsqueeze(0)
        idx.clamp_max_(self.motion_ends.unsqueeze(1) - 1)
        self.ref_object_contact_future = self._object_contact[idx]
        self.ref_object_contact = self.ref_object_contact_future[:, 0]
        
        # contact target and eef pos
        object_pos_w = self.object.data.body_link_pos_w[:, self.object_body_id_asset]
        object_quat_w = self.object.data.body_link_quat_w[:, self.object_body_id_asset]
        self.contact_target_pos_w[:] = object_pos_w.unsqueeze(1) + quat_apply(object_quat_w.unsqueeze(1), self.contact_target_pos_offset)
        object_lin_vel_w = self.object.data.body_com_lin_vel_w[:, self.object_body_id_asset]
        object_ang_vel_w = self.object.data.body_com_ang_vel_w[:, self.object_body_id_asset]
        self.contact_target_lin_vel_w[:] = object_lin_vel_w.unsqueeze(1) + torch.cross(object_ang_vel_w.unsqueeze(1), self.contact_target_pos_w - object_pos_w.unsqueeze(1), dim=-1)
        
        eef_pos_w = self.asset.data.body_link_pos_w[:, self.contact_eef_body_indices_asset]
        eef_quat_w = self.asset.data.body_link_quat_w[:, self.contact_eef_body_indices_asset]
        self.contact_eef_pos_w[:] = eef_pos_w + quat_apply(eef_quat_w, self.contact_eef_pos_offset)
        eef_lin_vel_w = self.asset.data.body_com_lin_vel_w[:, self.contact_eef_body_indices_asset]
        eef_ang_vel_w = self.asset.data.body_com_ang_vel_w[:, self.contact_eef_body_indices_asset]
        self.contact_eef_lin_vel_w[:] = eef_lin_vel_w + torch.cross(eef_ang_vel_w, self.contact_eef_pos_w - eef_pos_w, dim=-1)
        
        # from unfiltered contact forces
        # self.eef_contact_forces[:] = self.contact_forces.data.net_forces_w[:, self.contact_eef_body_indices_sensor]
        # for eef_idx, eef_sensor_indices in enumerate(self.contact_eef_body_indices_sensor):
        #     self.eef_contact_forces[:, eef_idx] = self.contact_forces.data.net_forces_w[:, eef_sensor_indices].sum(dim=1)      

        self.eef_contact_forces_w.zero_()
        for eef_idx, (eef_sensors, eef_sensor_indices) in enumerate(zip(self.eef_filtered_sensor, self.eef_filtered_sensor_indices)):
            for eef_sensor, eef_sensor_id in zip(eef_sensors, eef_sensor_indices):
                self.eef_contact_forces_w[:, eef_idx] += eef_sensor.data.force_matrix_w[:, eef_sensor_id, 0]

        self.eef_contact_forces_b[:] = quat_apply_inverse(object_quat_w.unsqueeze(1), self.eef_contact_forces_w)

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
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.3),
                        metallic=1.0,
                    )
                ),
                "right": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.3, 1.0),
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
        in_range_mask = self.ref_object_contact # shape [num_envs,]
        self.eef_contact_markers_pos_w[~in_range_mask] = -1000.0
        
        self.eef_contact_markers.visualize(
            translations=self.eef_contact_markers_pos_w.view(-1, 3),
            marker_indices=self.eef_contact_markers_indices,
        )

        # visualize contact forces
        self.env.debug_draw.vector(
            self.contact_eef_pos_w.reshape(-1, 3),
            self.eef_contact_forces_w.reshape(-1, 3) / 20,
            color=(1.0, 1.0, 1.0, 1.0),
            size=4.0,
        )
        
        # visualize final box position, as a red vector from current box position to final position
        self.env.debug_draw.vector(
            self.object_pos_w,
            self.box_final_pos_w - self.object_pos_w,
            color=(1.0, 0.2, 0.0, 1.0),
            size=4.0,
        )
        
