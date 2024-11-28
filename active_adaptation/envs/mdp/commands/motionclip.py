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
            joint_names: Sequence[str],
            decay: float = 0.98,
            teleop: bool = False,
        ):
        super().__init__(env, teleop=teleop)
        self.robot: Articulation = env.scene["robot"]
        
        data = joblib.load(motion_clip)
        self.root_translations = data['root_trans'] # [T, 3] translation vector
        self.ref_root_translations = torch.tensor(self.root_translations, dtype=torch.float32, device=self.device)
        self.ref_root_translations = self.ref_root_translations.unsqueeze(0).repeat(self.num_envs, 1, 1) # [num_envs, T, 3]
        origin = self.env.scene.env_origins             # [num_envs, 3]
        self.ref_root_translations += origin.unsqueeze(1)

        self.ref_root_orient = R.from_rotvec(data['root_orient']).as_quat() # [T, 4] (x, y, z, w) quaternion
        self.ref_root_orient = torch.tensor(self.ref_root_orient, dtype=torch.float32, device=self.device)
        self.ref_root_orient = self.ref_root_orient[:, [3, 0, 1, 2]]        # [T, 4] (w, x, y, z) quaternion

        self.ref_root_linear = data['root_linear_velocity'] # [T, 3] linear velocity
        self.ref_root_linear = torch.tensor(self.ref_root_linear, dtype=torch.float32, device=self.device)

        self.ref_root_angular = data['root_angular_velocity'] # [T, 3] angular velocity
        self.ref_root_angular = torch.tensor(self.ref_root_angular, dtype=torch.float32, device=self.device)

        self.ref_qpos = torch.tensor(data['qpos'], dtype=torch.float32, device=self.device)                # [T, 23] qpos
        self.ref_keypoints = torch.tensor(data['keypoints'], dtype=torch.float32, device=self.device)      # [T, 12 * 3] keypoints                                       # [N, 12, 3] keypoints

        self.num_frames = self.root_translations.shape[0]
        self.max_episode_length = self.env.max_episode_length

        padding_frames = self.max_episode_length - self.num_frames
        self.num_frames = torch.tensor(self.num_frames, dtype=torch.int, device=self.device)
        print(f"tracking {self.num_frames} frames of motion clip, padding to {self.max_episode_length} frames")

        print(f"padding {padding_frames} frames for reference motion")
        self.padding_ref_motion(padding_frames, joint_names)

        self.decay = decay
        self._cum_error_root = torch.zeros(self.num_envs, 1, device=self.device)
        self._cum_error_root_rot = torch.zeros(self.num_envs, 1, device=self.device)
        self._cum_error_vel = torch.zeros(self.num_envs, 1, device=self.device)
        self._cum_error_qpos = torch.zeros(self.num_envs, 1, device=self.device)
        self._cum_error_keypoint = torch.zeros(self.num_envs, 1, device=self.device)
        
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        init_root_state = self.init_root_state[env_ids]     # (num_envs, 3 + 4 + 6)
        init_root_state[:, :3] = self.ref_root_translations[env_ids, 0]
        init_root_state[:, 3:7] = self.ref_root_orient[0]
        return init_root_state
    
    def reset(self, env_ids: torch.Tensor):

        qpos = torch.cat([  self.ref_qpos[:, :5], torch.zeros(self.max_episode_length, 1, device=self.device),
                            self.ref_qpos[:, 5:10], torch.zeros(self.max_episode_length, 1, device=self.device),
                            self.ref_qpos[:, 10:]
                          ], dim=1)
        qpos = qpos[:, idx]
        
        self.robot.write_joint_state_to_sim(
            qpos[0],
            self.robot.data.default_joint_vel[env_ids],
            env_ids=env_ids
        )

        self._cum_error_root[env_ids] = 0
        self._cum_error_vel[env_ids] = 0
        self._cum_error_qpos[env_ids] = 0
        self._cum_error_keypoint[env_ids] = 0
    
    # def update(self):
        # for sanity check
        # root_state = self.robot.data.root_state_w.clone()
        # root_state[:, :3] = self.ref_root_translations[torch.arange(self.num_envs), self.frame.squeeze()] + torch.tensor([0., 0., 0.8], device=self.device)
        # root_state[:, 3:7] = self.ref_root_orient[self.frame.squeeze()]
        # env_ids = torch.arange(self.num_envs, device=self.device)
        # self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        # qpos = torch.cat([  self.ref_qpos[self.frame.squeeze(), :5], torch.zeros(self.num_envs, 1, device=self.device),     # left leg joints
        #                     self.ref_qpos[self.frame.squeeze(), 5:10], torch.zeros(self.num_envs, 1, device=self.device),   # right leg joints
        #                     self.ref_qpos[self.frame.squeeze(), 10:]                    # waist yaw joint and arm joints
        #                   ], dim=1)
        # qpos = qpos[:, idx]
        # self.robot.write_joint_state_to_sim(
        #     qpos,
        #     self.robot.data.default_joint_vel,
        #     env_ids=env_ids
        # )
        # return

    def padding_ref_motion(self, pad_frames: int, joint_names: Sequence[str]):
        last_translation = self.ref_root_translations[:, -1:, :]
        pad_translations = last_translation.expand(self.num_envs, pad_frames, 3)
        self.ref_root_translations = torch.cat([self.ref_root_translations, pad_translations], dim=1)

        last_orient = self.ref_root_orient[-1:, :]
        pad_orient = last_orient.expand(pad_frames, 4)
        self.ref_root_orient = torch.cat([self.ref_root_orient, pad_orient], dim=0)

        pad_linear = torch.zeros(pad_frames, 3, device=self.device)
        self.ref_root_linear = torch.cat([self.ref_root_linear, pad_linear], dim=0)

        pad_angular = torch.zeros(pad_frames, 3, device=self.device)
        self.ref_root_angular = torch.cat([self.ref_root_angular, pad_angular], dim=0)

        # padding qpos by default joint values
        joint_id, joint_names = self.asset.find_joints(joint_names, preserve_order=True)
        default_qpos = self.robot.data.default_joint_pos[0][joint_id]

        qpos_interpolate_frames = 50
        last_qpos = self.ref_qpos[-1:, :]   # [1, 23]
        t = torch.linspace(0, 1, qpos_interpolate_frames, device=self.device).unsqueeze(1)
        pad_qpos = (1 - t) * self.ref_qpos[-1:, :] + t * default_qpos.unsqueeze(0)  # [10, 23]
        self.ref_qpos = torch.cat([self.ref_qpos, pad_qpos], dim=0)         

        pad_qpos = default_qpos.unsqueeze(0).expand(pad_frames - qpos_interpolate_frames, -1)
        self.ref_qpos = torch.cat([self.ref_qpos, pad_qpos], dim=0)

        # padding keypoints by zeros and return 0 reward after num_frames
        pad_keypoints = torch.zeros(pad_frames, 12 * 3, device=self.device)
        self.ref_keypoints = torch.cat([self.ref_keypoints, pad_keypoints], dim=0)



idx = [0, 6, 12, 1, 7, 13, 19, 2, 8, 14, 20, 3, 9, 15, 21, 4, 10, 16, 22, 5, 11, 17, 23, 18, 24]