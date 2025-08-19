import os

from .humanoid import *
from .objects import *
from .g1_wbt import *


ASSET_PATH = os.path.dirname(__file__)

ROBOTS = {
    "g1_29dof": G1_29DOF_CFG,
    "g1_GMT": G1_GMT_29DOF_CFG,
    "g1_HV": G1_HV_29DOF_CFG,
    "g1_WBT": G1_CYLINDER_CFG,
}

OBJECTS = {
    "door": DOOR_CFG,
    "box": BOX_CFG,
    "box_small": BOX_SMALL_CFG,
    "suitcase": SUITCASE_CFG,
    "stool": STOOL_CFG,
    "stool_low": STOOL_LOW_CFG,
    "ball": BALL_CFG,
    "foldchair": FOLDCHAIR_CFG,
    "stool_support": STOOL_SUPPORT_CFG,
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

