import os
import copy
import torch

from .quadruped import *
from .humanoid import *
from .arm import *
from .sirius import *
from .objects import *


ASSET_PATH = os.path.dirname(__file__)

ROBOTS = {
    "go2": UNITREE_GO2_CFG,
    "aliengo": UNITREE_ALIENGO_CFG,
    "h1": H1_CFG,
    "a1-arm": A1_CFG,
    "sirius_wheel": SIRIUS_WHEEL_CFG,
    "b1z1": UNITREE_B1Z1_CFG,
    "g1_29dof": G1_29DOF_CFG,
    "g1_GMT": G1_GMT_29DOF_CFG,
    "g1_HV": G1_HV_29DOF_CFG,
}

OBJECTS = {
    "door": DOOR_CFG,
    "box": BOX_CFG,
}


def get_asset_meta(asset: Articulation):
    if not asset.is_initialized:
        raise RuntimeError("Articulation is not initialized. Please wait until `sim.reset` is called.")
    meta = {
        "init_state": asset.cfg.init_state.to_dict(),
        "body_names_isaac": asset.body_names,
        "joint_names_isaac": asset.joint_names,
        "actuators": {},
    }
    if asset.is_initialized: # parsed values
        meta["default_joint_pos"] = asset.data.default_joint_pos[0].tolist()
        meta["stiffness"] = asset.data.joint_stiffness[0].tolist()
        meta["damping"] = asset.data.joint_damping[0].tolist()

    for actuator_name, actuator in asset.actuators.items():
        meta["actuators"][actuator_name] = actuator.cfg.to_dict()
    return meta

