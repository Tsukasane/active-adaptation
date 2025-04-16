import os
import copy
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim.spawners.from_files.from_files import _spawn_from_usd_file, spawn_from_usd
from omni.isaac.lab.actuators import ImplicitActuatorCfg, DCMotorCfg

from .spawn import clone
from .humanoid import *


ASSET_PATH = os.path.dirname(__file__)

@clone
def spawn_with_payload(
    prim_path, 
    cfg, 
    translation, 
    orientation
):
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    import omni.physx.scripts.utils as script_utils
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core import objects
    from pxr import UsdPhysics

    parent_prim = prim_utils.get_prim_at_path(prim_path + "/base")
    path = prim_path + "/payload"
    payload = objects.DynamicCylinder(
        path, radius=0.1, height=0.05, translation=(0., 0., 0.1), mass=1.)

    stage = prim_utils.get_current_stage()
    # joint = script_utils.createJoint(stage, "Prismatic", payload.prim, parent_prim)
    # joint.GetAttribute('physics:axis').Set("X")
    joint = script_utils.createJoint(stage, "Fixed", parent_prim, payload.prim)

    return prim

ROBOTS = {
    "h1": H1_CFG,
    "g1": G1_CFG,
    "g1_sanity": G1_SANITY_CFG,
    "g1_inspire": G1_INSPIRE_CFG
}

for robot in ROBOTS.values():
    robot.spawn.func = clone(spawn_from_usd.__wrapped__)
