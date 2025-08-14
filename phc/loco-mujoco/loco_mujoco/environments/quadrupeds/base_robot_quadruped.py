from loco_mujoco.environments import LocoEnv
from loco_mujoco.core.utils import info_property


class BaseRobotQuadruped(LocoEnv):
    """
    Base Class for the Quadrupeds.

    """

    @info_property
    def sites_for_mimic(self):
        return []

    @info_property
    def root_body_name(self):
        return "trunk"
