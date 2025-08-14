from .base import Terrain
from .static import StaticTerrain
from .dynamic import DynamicTerrain
from .rough import RoughTerrain

# register all terrains
StaticTerrain.register()
RoughTerrain.register()
