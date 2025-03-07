import os
import copy
import omni.isaac.lab.sim as sim_utils
import torch
from omni.isaac.lab_assets import ArticulationCfg, H1_CFG
from omni.isaac.lab.actuators import DCMotorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation


class Humanoid(Articulation):
    def _create_buffers(self):
        super()._create_buffers()

        if hasattr(self.cfg, "hand_body_name"):
            self.hand_body_ids = self.find_bodies(self.cfg.hand_body_name)[0]
            self.hand_pos_w = self.data.body_pos_w[:, self.hand_body_ids]

        if hasattr(self.cfg, "foot_body_name"):
            self.foot_body_ids = self.find_bodies(self.cfg.foot_body_name)[0]
            self.foot_pos_w = self.data.body_pos_w[:, self.foot_body_ids]
    
    def update(self, dt: float):
        super().update(dt)
        if hasattr(self.cfg, "hand_body_name"):
            self.hand_pos_w[:] = self.data.body_pos_w[:, self.hand_body_ids]
        if hasattr(self.cfg, "foot_body_name"):
            self.foot_pos_w[:] = self.data.body_pos_w[:, self.foot_body_ids]


ASSET_PATH = os.path.dirname(__file__)

H1_CFG = copy.deepcopy(H1_CFG)
H1_CFG.spawn.usd_path = f"{ASSET_PATH}/H1/h1_minimal.usd"
H1_CFG.actuators = {
    "base_legs": DCMotorCfg(
        joint_names_expr=[".*"],
        effort_limit=300.0,
        saturation_effort=300.0,
        velocity_limit=30.0,
        stiffness={
            ".*hip.*": 200,
            ".*knee.*": 300,
            ".*ankle.*": 40,
            "torso": 300,
            ".*shoulder.*": 100,
            ".*elbow.*": 100
        },
        damping={
            ".*hip.*": 5,
            ".*knee.*": 6,
            ".*ankle.*": 2,
            "torso": 6,
            ".*shoulder.*": 2,
            ".*elbow.*": 2
        },
        friction=0.0,
    )
}
