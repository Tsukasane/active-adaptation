import mujoco

from loco_mujoco.environments.humanoids.atlas import Atlas


class MjxAtlas(Atlas):

    mjx_enabled = True

    def __init__(self, timestep=0.002, n_substeps=5, **kwargs):
        if "model_option_conf" not in kwargs.keys():
            model_option_conf = dict(iterations=2, ls_iterations=4, disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        else:
            model_option_conf = kwargs["model_option_conf"]
            del kwargs["model_option_conf"]
        super().__init__(timestep=timestep, n_substeps=n_substeps, model_option_conf=model_option_conf, **kwargs)

    def _modify_spec_for_mjx(self, spec):
        """
        Mjx is bad in handling many complex contacts. To speed-up simulation significantly we apply
        some changes to the XML:
            1. Disable all contacts except the ones between feet and the floor.

        Args:
            spec: Mujoco specification.

        Returns:
            modified Mujoco specification.

        """

        # --- disable all contacts in geoms ---
        for g in spec.geoms:
            g.contype = 0
            g.conaffinity = 0

        # --- enable contacts between feet and floor ---
        spec.add_pair(geomname1="floor", geomname2="left_foot_back")
        spec.add_pair(geomname1="floor", geomname2="left_foot_front")
        spec.add_pair(geomname1="floor", geomname2="right_foot_back")
        spec.add_pair(geomname1="floor", geomname2="right_foot_front")

        return spec
