from math import pi
import torch
import torch.distributions as D
import math
from typing import Sequence, TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
import omni.isaac.lab.utils.math as math_utils
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, MultiUniform
from active_adaptation.utils.helpers import batchify
from omni.isaac.lab.utils.math import quat_apply_yaw, yaw_quat
from tensordict import TensorDict
from .base import Command

import joblib
import os
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from active_adaptation.envs.base import Env

class MotionClip(Command):
    def __init__(
            self, 
            env,
            motion_clip: str,
            teleop: bool=False,
        ):
        super().__init__(env, teleop=teleop)
        self.robot: Articulation = env.scene["robot"]
        
        data = joblib.load(motion_clip)
        self.smpl_joints = data['joints']               # [N, 24, 3]
        self.root_translations = self.smpl_joints[:, 0] # [N, 3] translation vector
        self.ref_root_translations = torch.tensor(self.root_translations, dtype=torch.float32, device=self.device)
        self.ref_root_translations = self.ref_root_translations.unsqueeze(0).repeat(self.num_envs, 1, 1) # [num_envs, N, 3]
        origin = self.env.scene.env_origins             # [num_envs, 3]
        self.ref_root_translations += origin.unsqueeze(1)

        self.ref_root_orient = R.from_rotvec(data['root_orient']).as_quat() # [N, 4] (x, y, z, w) quaternion
        self.ref_root_orient = torch.tensor(self.ref_root_orient, dtype=torch.float32, device=self.device)
        self.ref_root_orient = self.ref_root_orient[:, [3, 0, 1, 2]]        # [N, 4] (w, x, y, z) quaternion

        self.ref_qpos = torch.tensor(data['qpos'], dtype=torch.float32, device=self.device)                # [N, 23] qpos                          # [N, 25] qpos
        self.ref_keypoints = torch.tensor(data['keypoints'], dtype=torch.float32, device=self.device)      # [N, 12 * 3] keypoints

        self.num_frames = self.smpl_joints.shape[0]
        self.frame = torch.zeros(self.num_envs, 1, dtype=torch.int, device=self.device)
        
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        init_root_state = self.init_root_state[env_ids]     # (num_envs, 3 + 4 + 6)
        init_root_state[:, :3] = self.ref_root_translations[env_ids, 0]
        init_root_state[:, 3:7] = self.ref_root_orient[0]
        return init_root_state
    
    def reset(self, env_ids: torch.Tensor):
        self.frame[env_ids] = 0

        qpos = torch.cat([  self.ref_qpos[:, :5], torch.zeros(self.num_frames, 1, device=self.device),
                            self.ref_qpos[:, 5:10], torch.zeros(self.num_frames, 1, device=self.device),
                            self.ref_qpos[:, 10:]
                          ], dim=1)
        qpos = qpos[:, idx]
        
        self.robot.write_joint_state_to_sim(
            qpos[0],
            self.robot.data.default_joint_vel[env_ids],
            env_ids=env_ids
        )
    
    def update(self):
        self.frame += 1

idx = [0, 6, 12, 1, 7, 13, 19, 2, 8, 14, 20, 3, 9, 15, 21, 4, 10, 16, 22, 5, 11, 17, 23, 18, 24]