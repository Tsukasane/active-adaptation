import mujoco
from mujoco import MjSpec
from .bd_spot import BDSpot


class MjxBDSpot(BDSpot):

    mjx_enabled = True

    def __init__(self, timestep=0.002, n_substeps=5, **kwargs):
        if "model_option_conf" not in kwargs.keys():
            model_option_conf = dict(iterations=8, ls_iterations=12,
                                     cone=mujoco.mjtCone.mjCONE_PYRAMIDAL,
                                     impratio=1,
                                     disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        else:
            model_option_conf = kwargs["model_option_conf"]
            del kwargs["model_option_conf"]
        super().__init__(timestep=timestep, n_substeps=n_substeps, model_option_conf=model_option_conf, **kwargs)

    def _modify_spec_for_mjx(self, spec: MjSpec):
        """
        Mjx is bad in handling many complex contacts. To speed-up simulation significantly we apply
        some changes to the XML:
            1. Replace the complex foot meshes with primitive shapes. Here, one foot mesh is replaced with
               two capsules.
            2. Disable all contacts except the ones between feet and the floor.

        Args:
            spec (MjSpec): Mujoco specification.

        Returns:
            Modified Mujoco specification.

        """

        # --- 1. Make all geoms have contype and conaffinity of 0 ---
        for g in spec.geoms:
            g.contype = 0
            g.conaffinity = 0

        # --- 2. Define specific contact pairs ---
        spec.add_pair(geomname1="floor", geomname2="FL")
        spec.add_pair(geomname1="floor", geomname2="HL")
        spec.add_pair(geomname1="floor", geomname2="FR")
        spec.add_pair(geomname1="floor", geomname2="HR")

        return spec
