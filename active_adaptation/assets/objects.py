import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg, Articulation
from isaaclab.assets.rigid_object import RigidObjectCfg
import torch

ASSET_PATH = os.path.dirname(__file__)

class DoorArticulation(Articulation):
    def _create_buffers(self):
        super()._create_buffers()
        self.friction = torch.zeros((self.num_instances,), device=self.device)
        self.damping = torch.zeros((self.num_instances,), device=self.device)
        self.door_joint_id = self.joint_names.index("door_joint")

        self.door_torques = torch.zeros((self.num_instances,), device=self.device)
    
    def _initialize_impl(self):
        super()._initialize_impl()

        # set joint stiffness and damping to 0
        joint_attrs_zero = torch.zeros((self.num_instances, self.num_joints), device=self.device)
        self.write_joint_stiffness_to_sim(joint_attrs_zero)
        self.write_joint_damping_to_sim(joint_attrs_zero)
        self.write_joint_friction_to_sim(joint_attrs_zero)

        # set actuator stiffness and damping to 0
        for actuator in self.actuators.values():
            actuator.stiffness.fill_(0.0)
            actuator.damping.fill_(0.0)
    
    def write_data_to_sim(self):
        door_joint_vel = self.data.joint_vel[:, self.door_joint_id]
        door_joint_friction = -torch.sign(door_joint_vel) * (door_joint_vel.abs() > 0.01) * self.friction
        door_joint_damping = -door_joint_vel * self.damping
        self.door_torques[:] = door_joint_friction + door_joint_damping
        
        self.set_joint_effort_target(self.door_torques.unsqueeze(-1), joint_ids=[self.door_joint_id])
        super().write_data_to_sim()

DOOR_CFG = ArticulationCfg(
    class_type=DoorArticulation,
    prim_path="{ENV_REGEX_NS}/Door",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/Doors/door.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            enabled_self_collisions=False
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "door_joint": IdealPDActuatorCfg(
            joint_names_expr="door_joint",
            # will be randomized
            stiffness=0.0, 
            damping=0.0,
            friction=0.0,
            effort_limit=100.0,
            velocity_limit=20.0,
        )
    },
)

BOX_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Box",
    # spawn=sim_utils.UrdfFileCfg(
    #     asset_path=f"{ASSET_PATH}/box/box.urdf",
    spawn=sim_utils.UsdFileCfg(
        scale=(1.0, 1.0, 1.0),
        usd_path=f"{ASSET_PATH}/box/box.usd",
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.5, 0.5, 0.5),
            metallic=0.2,
            roughness=0.2,
        ),
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=5.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
)

# from active_adaptation.assets.spawn import clone
# spawn_func = BOX_CFG.spawn.func.__wrapped__
# BOX_CFG.spawn.func = clone(spawn_func)
# BOX_CFG.spawn.scale_range = (0.9, 1.1)

# size: [1.0 0.8 0.8] z: 0.6 - 0.8
# friction 0.5-1.0
# mass 4-8