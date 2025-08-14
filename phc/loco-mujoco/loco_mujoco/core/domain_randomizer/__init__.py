from .base import DomainRandomizer
from .no_randomization import NoDomainRandomization
from .default import DefaultRandomizer

# register all domain randomizers
NoDomainRandomization.register()
DefaultRandomizer.register()
