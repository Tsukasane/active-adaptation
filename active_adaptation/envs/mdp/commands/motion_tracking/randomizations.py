from active_adaptation.envs.mdp.commands.motion_tracking.command import MotionTrackingCommand
from active_adaptation.envs.mdp.base import Randomization as BaseRandomization

import torch
from typing import Dict, Tuple, List
from omegaconf import DictConfig
from isaaclab.utils.math import quat_apply_inverse, sample_uniform


TrackRandomization = BaseRandomization["MotionTrackingCommand"]

class keypoint_virtual_force(TrackRandomization):
    def __init__(
        self,
        body_names: str | List[str] = ".*",
        stiffness_range: Tuple[float, float]=(20.0, 30.0),
        annealing_steps: int=500,
        pos_tolerance: float | Dict[str, float] = 0.0,
        vel_tolerance: float | Dict[str, float] = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.annealing_steps = annealing_steps

        from isaaclab.utils.string import resolve_matching_names_values, resolve_matching_names
        tracking_body_names = self.command_manager.tracking_keypoint_names
        self.apply_force_body_names = resolve_matching_names(body_names, tracking_body_names)[1]
        self.apply_force_body_indices_asset = []
        self.apply_force_body_indices_motion = []
        for name in self.apply_force_body_names:
            body_idx_asset = self.command_manager.asset.body_names.index(name)
            body_idx_motion = tracking_body_names.index(name)

            self.apply_force_body_indices_asset.append(body_idx_asset)
            self.apply_force_body_indices_motion.append(body_idx_motion)
        
        self.pos_tolerance = torch.zeros(len(self.apply_force_body_names), device=self.device)
        if isinstance(pos_tolerance, float):
            self.pos_tolerance.fill_(pos_tolerance)
        elif isinstance(pos_tolerance, DictConfig):
            indices, names, values = resolve_matching_names_values(dict(pos_tolerance), self.apply_force_body_names)
            self.pos_tolerance[indices] = torch.tensor(values, device=self.device)
        else:
            raise ValueError(f"Invalid type for pos_tolerance: {type(pos_tolerance)}")

        self.vel_tolerance = torch.zeros(len(self.apply_force_body_names), device=self.device)
        if isinstance(vel_tolerance, float):
            self.vel_tolerance.fill_(vel_tolerance)
        elif isinstance(vel_tolerance, DictConfig):
            indices, names, values = resolve_matching_names_values(dict(vel_tolerance), self.apply_force_body_names)
            self.vel_tolerance[indices] = torch.tensor(values, device=self.device)
        else:
            raise ValueError(f"Invalid type for vel_tolerance: {type(vel_tolerance)}")

        self.stiffness_start = sample_uniform(*stiffness_range, (self.env.num_envs, 1, 1), self.device)
        self.stiffness = self.stiffness_start.clone()
        self.damping = self.stiffness.sqrt() * 2
        
        self.ref_keypoint_pos_w = self.command_manager.ref_body_pos_w[:, self.apply_force_body_indices_motion]
        self.ref_keypoint_lin_vel_w = self.command_manager.ref_body_lin_vel_w[:, self.apply_force_body_indices_motion]
    
    def reset(self, env_ids: torch.Tensor):
        # do not apply force in the first {decimation} steps
        # because update is called after step after a reset
        self.stiffness[env_ids] = 0.0
        self.damping[env_ids] = 0.0
    
    def update(self):
        self.stiffness = self.stiffness_start * max(1.0 - self.env.current_iter / self.annealing_steps, 0.0)
        self.damping = self.stiffness.sqrt() * 2

        self.ref_keypoint_pos_w = self.command_manager.ref_body_pos_w[:, self.apply_force_body_indices_motion]
        self.ref_keypoint_lin_vel_w = self.command_manager.ref_body_lin_vel_w[:, self.apply_force_body_indices_motion]

    def step(self, substep):
        if self.env.current_iter >= self.annealing_steps:
            return
        
        robot_keypoint_pos_w = self.command_manager.asset.data.body_link_pos_w[:, self.apply_force_body_indices_asset]
        robot_keypoint_lin_vel_w = self.command_manager.asset.data.body_com_lin_vel_w[:, self.apply_force_body_indices_asset]

        # compute force in world frame
        diff_pos_w = self.ref_keypoint_pos_w - robot_keypoint_pos_w
        diff_lin_vel_w = self.ref_keypoint_lin_vel_w - robot_keypoint_lin_vel_w
        diff_pos_w = diff_pos_w * (diff_pos_w.abs() > self.pos_tolerance.unsqueeze(-1))
        diff_lin_vel_w = diff_lin_vel_w * (diff_lin_vel_w.abs() > self.vel_tolerance.unsqueeze(-1))
        self.forces_w = forces_w = self.stiffness * diff_pos_w + self.damping * diff_lin_vel_w
        body_quat_w = self.command_manager.asset.data.body_quat_w[:, self.apply_force_body_indices_asset]
        forces_b = quat_apply_inverse(body_quat_w, forces_w)

        # apply force to asset
        ext_forces_b = self.command_manager.asset._external_force_b
        ext_forces_b[:, self.apply_force_body_indices_asset] += forces_b
        self.command_manager.asset.has_external_wrench = True
    
    def debug_draw(self):
        if self.env.backend != "isaac":
            return

        if self.env.current_iter >= self.annealing_steps:
            return

        # draw force as vectors
        body_pos_w = self.command_manager.asset.data.body_link_pos_w[:, self.apply_force_body_indices_asset]
        self.env.debug_draw.vector(
            body_pos_w.reshape(-1, 3),
            (self.forces_w / self.stiffness).reshape(-1, 3) * 2.0,
            # orange
            color=(1.0, 0.5, 0.0, 1.0)
        )

from active_adaptation.envs.mdp.commands.motion_tracking.command import MotionTrackingDoor
import numpy as np

TrackDoorRandomization = BaseRandomization[MotionTrackingDoor]

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
        door_armature = sample_uniform(*self.armature_range, (self.door.num_instances, 1), self.device)
        self.door.write_joint_armature_to_sim(door_armature, joint_ids=[self.door_joint_id_asset])

    def reset(self, env_ids: torch.Tensor):
        door_friction = sample_uniform(*self.friction_range, (len(env_ids),), self.device)
        door_damping = sample_uniform(*self.damping_range, (len(env_ids),), self.device)

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

        self.door._custom_friction[env_ids] = door_friction
        self.door._custom_damping[env_ids] = door_damping

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
        self.static_friction_buckets = sample_uniform(*tuple(static_friction_range), (self.num_buckets,), "cpu")
        self.dynamic_friction_buckets = sample_uniform(*tuple(dynamic_friction_range), (self.num_buckets,), "cpu")
        self.restitution_buckets = sample_uniform(*tuple(restitution_range), (self.num_buckets,), "cpu")

    def startup(self):
        masses = self.door.data.default_mass.clone()
        inertias = self.door.data.default_inertia.clone()
        new_masses = sample_uniform(*self.mass_range, (self.door.num_instances, 1), "cpu")
        
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


from active_adaptation.envs.mdp.commands.motion_tracking.command import MotionTrackingBox
TrackBoxRandomization = BaseRandomization["MotionTrackingBox"]

class box_body_randomization(TrackBoxRandomization):
    def __init__(
        self,
        static_friction_range: Tuple[float, float]=(0.6, 1.0),
        dynamic_friction_range: Tuple[float, float]=(0.6, 1.0),
        restitution_range: Tuple[float, float]=(0.0, 0.2),
        mass_range: Tuple[float, float]=(1.0, 10.0),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.box = self.command_manager.box

        self.mass_range = mass_range

        self.all_indices_cpu = torch.arange(self.box.num_instances)

        max_shapes = self.box.root_physx_view.max_shapes
        self.shape_ids = torch.arange(0, max_shapes) 

        self.num_buckets = 64
        self.static_friction_buckets = sample_uniform(*tuple(static_friction_range), (self.num_buckets,), "cpu")
        self.dynamic_friction_buckets = sample_uniform(*tuple(dynamic_friction_range), (self.num_buckets,), "cpu")
        self.restitution_buckets = sample_uniform(*tuple(restitution_range), (self.num_buckets,), "cpu")

    def startup(self):
        masses = self.box.data.default_mass.clone()
        inertias = self.box.data.default_inertia.clone()
        new_masses = sample_uniform(*self.mass_range, (self.box.num_instances, 1), "cpu")

        scale = new_masses / masses
        masses[:] *= scale
        inertias[:] *= scale
        self.box.root_physx_view.set_masses(masses, self.all_indices_cpu)
        self.box.root_physx_view.set_inertias(inertias, self.all_indices_cpu)
        assert torch.allclose(self.box.root_physx_view.get_masses(), masses, atol=1e-4)
        assert torch.allclose(self.box.root_physx_view.get_inertias(), inertias, atol=1e-4)

        materials = self.box.root_physx_view.get_material_properties().clone()
        shape = (self.box.num_instances, len(self.shape_ids))
        materials[:, self.shape_ids, 0] = self.static_friction_buckets[torch.randint(0, self.num_buckets, shape)]
        materials[:, self.shape_ids, 1] = self.dynamic_friction_buckets[torch.randint(0, self.num_buckets, shape)]
        materials[:, self.shape_ids, 2] = self.restitution_buckets[torch.randint(0, self.num_buckets, shape)]
        self.box.root_physx_view.set_material_properties(materials.flatten(), self.all_indices_cpu)
        assert torch.allclose(self.box.root_physx_view.get_material_properties(), materials, atol=1e-4)