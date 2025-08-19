from active_adaptation.envs.mdp.commands.hdmi.command import RobotObjectTracking
from active_adaptation.envs.mdp.base import Randomization as BaseRandomization

import torch
from typing import Dict, Tuple, List, TYPE_CHECKING
from omegaconf import DictConfig
from isaaclab.utils.math import quat_apply_inverse, sample_uniform


RobotObjectTrackRandomization = BaseRandomization[RobotObjectTracking]

class object_body_randomization(RobotObjectTrackRandomization ):
    def __init__(
        self,
        dynamic_friction_range: Tuple[float, float]=(0.6, 1.0),
        restitution_range: Tuple[float, float]=(0.0, 0.2),
        mass_range: Tuple[float, float]=(1.0, 10.0),
        static_friction_range: Tuple[float, float] | None = None,
        static_dynamic_friction_ratio_range: Tuple[float, float] | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Validate that only one of static_friction_range or static_dynamic_friction_ratio_range is specified
        if static_friction_range is not None and static_dynamic_friction_ratio_range is not None:
            raise ValueError("Cannot specify both static_friction_range and static_dynamic_friction_ratio_range")
        if static_friction_range is None and static_dynamic_friction_ratio_range is None:
            raise ValueError("Must specify either static_friction_range or static_dynamic_friction_ratio_range")
        
        self.object = self.command_manager.object

        self.mass_range = mass_range

        self.all_indices_cpu = torch.arange(self.object.num_instances)

        # randomize all shapes of the object
        max_shapes = self.object.root_physx_view.max_shapes
        self.shape_ids = torch.arange(0, max_shapes) 

        self.num_buckets = 64
        
        # Sample dynamic friction and restitution buckets
        self.dynamic_friction_buckets = sample_uniform(*tuple(dynamic_friction_range), (self.num_buckets,), "cpu")
        self.restitution_buckets = sample_uniform(*tuple(restitution_range), (self.num_buckets,), "cpu")
        
        # Handle static friction based on which parameter is specified
        if static_friction_range is not None:
            self.static_friction_buckets = sample_uniform(*tuple(static_friction_range), (self.num_buckets,), "cpu")
        else:
            self.static_dynamic_friction_ratio_buckets = sample_uniform(*tuple(static_dynamic_friction_ratio_range), (self.num_buckets,), "cpu")

    def startup(self):
        masses = self.object.data.default_mass.clone()
        inertias = self.object.data.default_inertia.clone()
        new_masses = sample_uniform(*self.mass_range, masses.shape, "cpu")

        scale = new_masses / masses
        masses[:] *= scale
        if inertias.ndim == 2:
            inertias[:] *= scale
        elif inertias.ndim == 3:
            inertias[:] *= scale.unsqueeze(-1)
        else:
            raise ValueError(f"Invalid shape for inertias: {inertias.shape}")
        self.object.root_physx_view.set_masses(masses, self.all_indices_cpu)
        self.object.root_physx_view.set_inertias(inertias, self.all_indices_cpu)
        assert torch.allclose(self.object.root_physx_view.get_masses(), masses, atol=1e-4)
        assert torch.allclose(self.object.root_physx_view.get_inertias(), inertias, atol=1e-4)

        materials = self.object.root_physx_view.get_material_properties().clone()
        shape = (self.object.num_instances, 1)
        dynamic_friction = self.dynamic_friction_buckets[torch.randint(0, self.num_buckets, shape)]
        restitution = self.restitution_buckets[torch.randint(0, self.num_buckets, shape)]
        if hasattr(self, "static_friction_buckets"):
            static_friction = self.static_friction_buckets[torch.randint(0, self.num_buckets, shape)]
        else:
            static_friction_ratio = self.static_dynamic_friction_ratio_buckets[torch.randint(0, self.num_buckets, shape)]
            static_friction = dynamic_friction * static_friction_ratio
        materials[:, self.shape_ids, 0] = static_friction
        materials[:, self.shape_ids, 1] = dynamic_friction
        materials[:, self.shape_ids, 2] = restitution
        self.object.root_physx_view.set_material_properties(materials.flatten(), self.all_indices_cpu)
        assert torch.allclose(self.object.root_physx_view.get_material_properties(), materials, atol=1e-4)

class object_joint_randomization(RobotObjectTrackRandomization):
    def __init__(
        self,
        friction_range: Tuple[float, float]=(0.0, 0.1),
        damping_range: Tuple[float, float]=(1.0, 10.0),
        armature_range: Tuple[float, float]=(0.0, 0.02),
        **kwargs
    ):
        super().__init__(**kwargs)
        if TYPE_CHECKING:
            from active_adaptation.assets.objects import CustomArticulation
        self.object: CustomArticulation = self.command_manager.object
        self.friction_range = friction_range
        self.damping_range = damping_range
        self.armature_range = armature_range

        self.joint_id_asset = 0
    
    def startup(self):
        door_armature = sample_uniform(*self.armature_range, (self.object.num_instances, 1), self.device)
        self.object.write_joint_armature_to_sim(door_armature, joint_ids=[self.joint_id_asset])

    def reset(self, env_ids: torch.Tensor):
        joint_friction = sample_uniform(*self.friction_range, (len(env_ids),), self.device)
        joint_damping = sample_uniform(*self.damping_range, (len(env_ids),), self.device)

        self.object._custom_friction[env_ids] = joint_friction
        self.object._custom_damping[env_ids] = joint_damping

# class keypoint_virtual_force(RobotTrackRandomization):
#     def __init__(
#         self,
#         body_names: str | List[str] = ".*",
#         stiffness_range: Tuple[float, float]=(20.0, 30.0),
#         annealing_steps: int=500,
#         pos_tolerance: float | Dict[str, float] = 0.0,
#         vel_tolerance: float | Dict[str, float] = 0.0,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.annealing_steps = annealing_steps

#         from isaaclab.utils.string import resolve_matching_names_values, resolve_matching_names
#         tracking_body_names = self.command_manager.tracking_keypoint_names
#         self.apply_force_body_names = resolve_matching_names(body_names, tracking_body_names)[1]
#         self.apply_force_body_indices_asset = []
#         self.apply_force_body_indices_motion = []
#         for name in self.apply_force_body_names:
#             body_idx_asset = self.command_manager.asset.body_names.index(name)
#             body_idx_motion = tracking_body_names.index(name)

#             self.apply_force_body_indices_asset.append(body_idx_asset)
#             self.apply_force_body_indices_motion.append(body_idx_motion)
        
#         self.pos_tolerance = torch.zeros(len(self.apply_force_body_names), device=self.device)
#         if isinstance(pos_tolerance, float):
#             self.pos_tolerance.fill_(pos_tolerance)
#         elif isinstance(pos_tolerance, DictConfig):
#             indices, names, values = resolve_matching_names_values(dict(pos_tolerance), self.apply_force_body_names)
#             self.pos_tolerance[indices] = torch.tensor(values, device=self.device)
#         else:
#             raise ValueError(f"Invalid type for pos_tolerance: {type(pos_tolerance)}")

#         self.vel_tolerance = torch.zeros(len(self.apply_force_body_names), device=self.device)
#         if isinstance(vel_tolerance, float):
#             self.vel_tolerance.fill_(vel_tolerance)
#         elif isinstance(vel_tolerance, DictConfig):
#             indices, names, values = resolve_matching_names_values(dict(vel_tolerance), self.apply_force_body_names)
#             self.vel_tolerance[indices] = torch.tensor(values, device=self.device)
#         else:
#             raise ValueError(f"Invalid type for vel_tolerance: {type(vel_tolerance)}")

#         self.stiffness_start = sample_uniform(*stiffness_range, (self.env.num_envs, 1, 1), self.device)
#         self.stiffness = self.stiffness_start.clone()
#         self.damping = self.stiffness.sqrt() * 2
        
#         self.ref_keypoint_pos_w = self.command_manager.ref_body_pos_w[:, self.apply_force_body_indices_motion]
#         self.ref_keypoint_lin_vel_w = self.command_manager.ref_body_lin_vel_w[:, self.apply_force_body_indices_motion]
    
#     def reset(self, env_ids: torch.Tensor):
#         # do not apply force in the first {decimation} steps
#         # because update is called after step after a reset
#         self.stiffness[env_ids] = 0.0
#         self.damping[env_ids] = 0.0
    
#     def update(self):
#         self.stiffness = self.stiffness_start * max(1.0 - self.env.current_iter / self.annealing_steps, 0.0)
#         self.damping = self.stiffness.sqrt() * 2

#         self.ref_keypoint_pos_w = self.command_manager.ref_body_pos_w[:, self.apply_force_body_indices_motion]
#         self.ref_keypoint_lin_vel_w = self.command_manager.ref_body_lin_vel_w[:, self.apply_force_body_indices_motion]

#     def step(self, substep):
#         if self.env.current_iter >= self.annealing_steps:
#             return
        
#         robot_keypoint_pos_w = self.command_manager.asset.data.body_link_pos_w[:, self.apply_force_body_indices_asset]
#         robot_keypoint_lin_vel_w = self.command_manager.asset.data.body_com_lin_vel_w[:, self.apply_force_body_indices_asset]

#         # compute force in world frame
#         diff_pos_w = self.ref_keypoint_pos_w - robot_keypoint_pos_w
#         diff_lin_vel_w = self.ref_keypoint_lin_vel_w - robot_keypoint_lin_vel_w
#         diff_pos_w = diff_pos_w * (diff_pos_w.abs() > self.pos_tolerance.unsqueeze(-1))
#         diff_lin_vel_w = diff_lin_vel_w * (diff_lin_vel_w.abs() > self.vel_tolerance.unsqueeze(-1))
#         self.forces_w = forces_w = self.stiffness * diff_pos_w + self.damping * diff_lin_vel_w
#         body_quat_w = self.command_manager.asset.data.body_quat_w[:, self.apply_force_body_indices_asset]
#         forces_b = quat_apply_inverse(body_quat_w, forces_w)

#         # apply force to asset
#         ext_forces_b = self.command_manager.asset._external_force_b
#         ext_forces_b[:, self.apply_force_body_indices_asset] += forces_b
#         self.command_manager.asset.has_external_wrench = True
    
#     def debug_draw(self):
#         if self.env.backend != "isaac":
#             return

#         if self.env.current_iter >= self.annealing_steps:
#             return

#         # draw force as vectors
#         body_pos_w = self.command_manager.asset.data.body_link_pos_w[:, self.apply_force_body_indices_asset]
#         self.env.debug_draw.vector(
#             body_pos_w.reshape(-1, 3),
#             (self.forces_w / self.stiffness).reshape(-1, 3) * 2.0,
#             # orange
#             color=(1.0, 0.5, 0.0, 1.0)
#         )
