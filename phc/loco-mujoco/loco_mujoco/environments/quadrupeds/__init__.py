from .base_robot_quadruped import BaseRobotQuadruped
from .unitreeA1 import UnitreeA1
from .unitreeA1_mjx import MjxUnitreeA1
from .unitreeGo2 import UnitreeGo2
from .unitreeGo2_mjx import MjxUnitreeGo2
from .bd_spot import BDSpot
from .bd_spot_mjx import MjxBDSpot
from .anymal_c import AnymalC
from .anymal_c_mjx import MjxAnymalC


# register environment
UnitreeA1.register()
MjxUnitreeA1.register()
UnitreeGo2.register()
MjxUnitreeGo2.register()
BDSpot.register()
MjxBDSpot.register()
AnymalC.register()
MjxAnymalC.register()


