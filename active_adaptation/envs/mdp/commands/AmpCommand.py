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
import glob
import os
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from active_adaptation.envs.base import Env

quat_rotate_inverse = batchify(quat_rotate_inverse)

class AmpCommand(Command):
    def __init__(
            self, 
            env,
            motion_clip_dir: str,
            joint_names: Sequence[str],
            body_names: Sequence[str],
            decay: float = 0.98,
            teleop: bool = False,
        ):
        super().__init__(env, teleop=teleop)
        self.robot: Articulation = env.scene["robot"]
        self.joint_names = joint_names
        self.body_names = body_names
        self.decay = decay

        self.motion_clips, self.motion_clips_len, self.max_traj_len = self._load_motions(motion_clip_dir)
        self.num_motions = len(self.motion_clips)
        print(f"Loaded {self.num_motions} motion clips with max_traj_len: {self.max_traj_len}")
        # self.max_traj_len = max(self.max_traj_len, self.env.max_episode_length)
        self._stack_all_motions()

        # self.motion_ids = torch.randint(0, len(self.motion_clips), (self.num_envs,), device=self.device)
        self.motion_ids = torch.arange(self.num_motions, device=self.device).repeat(self.num_envs // self.num_motions + 1)[:self.num_envs]
        self.ref_root_trans = self.ref_root_trans[self.motion_ids]     # [num_envs, max_traj_len, 3]

        # add terrain origin to reference root translations
        origin = self.env.scene.env_origins     # [num_envs, 3]
        self.ref_root_trans += origin.unsqueeze(1)

        self.ref_root_orient = self.ref_root_orient[self.motion_ids]   # [num_envs, max_traj_len, 4]
        self.ref_root_linear = self.ref_root_linear[self.motion_ids]   # [num_envs, max_traj_len, 3]
        self.ref_root_angular = self.ref_root_angular[self.motion_ids] # [num_envs, max_traj_len, 3]
        self.ref_qpos = self.ref_qpos[self.motion_ids]                 # [num_envs, max_traj_len, num_joints]
        self.ref_keypoints = self.ref_keypoints[self.motion_ids]       # [num_envs, max_traj_len, num_keypoints * 3]

        print(f"reference motion root translation shape: {self.ref_root_trans.shape}")
        print(f"reference motion root orientation shape: {self.ref_root_orient.shape}")
        print(f"reference motion qpos shape: {self.ref_qpos.shape}")
        print(f"reference motion keypoints shape: {self.ref_keypoints.shape}")

        self.motion_clips_len = self.motion_clips_len[self.motion_ids] # [num_envs]

    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        init_root_state = self.init_root_state[env_ids]         # [num_envs, 3 + 4 + 3 + 3]
        init_root_state[:, :3] = self.ref_root_trans[env_ids, 0]     # [num_envs, 3]
        init_root_state[:, 3:7] = self.ref_root_orient[env_ids, 0]
        return init_root_state
    
    def reset(self, env_ids: torch.Tensor):
        qpos = self.ref_qpos[env_ids, 0]     # [num_envs, num_joints]
        qpos = torch.cat([qpos[:, :5], torch.zeros(env_ids.shape[0], 1, device=self.device),
                          qpos[:, 5:10], torch.zeros(env_ids.shape[0], 1, device=self.device),
                          qpos[:, 10:]
                        ], dim=1)               # [num_envs, num_joints]
        qpos = qpos[:, idx]
        
        self.robot.write_joint_state_to_sim(
            qpos,
            self.robot.data.default_joint_vel[env_ids],
            env_ids=env_ids
        )

    # for sanity check
    # def update(self):
    #     root_state = self.robot.data.root_state_w.clone()
    #     frame = self.env.episode_length_buf     # [num_envs]
    #     root_state[:, :3] = self.ref_root_trans[self.motion_ids, frame] + torch.tensor([0., 0., 0.8], device=self.device)
    #     root_state[:, 3:7] = self.ref_root_orient[self.motion_ids, frame]
    #     env_ids = torch.arange(self.num_envs, device=self.device)
    #     self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    #     qpos = torch.cat([  self.ref_qpos[self.motion_ids, frame, :5], torch.zeros(self.num_envs, 1, device=self.device),     # left leg joints
    #                         self.ref_qpos[self.motion_ids, frame, 5:10], torch.zeros(self.num_envs, 1, device=self.device),   # right leg joints
    #                         self.ref_qpos[self.motion_ids, frame, 10:]                    # waist yaw joint and arm joints
    #                       ], dim=1)
    #     qpos = qpos[:, idx]
    #     self.robot.write_joint_state_to_sim(
    #         qpos,
    #         self.robot.data.default_joint_vel,
    #         env_ids=env_ids
    #     )
    #     return

    def _load_motions(self, motion_clip_dir):
        motion_clips = []
        motion_clips_len = []
        for file in glob.glob(f"{motion_clip_dir}/*/*.pkl"):
            trajectory = joblib.load(file)
            motion_clips.append(trajectory)
            motion_clips_len.append(trajectory["keypoints"].shape[0] + 200)
        return motion_clips, motion_clips_len, max(motion_clips_len)

    def _stack_all_motions(self):
        self.ref_root_trans = []
        self.ref_root_orient = []
        self.ref_root_linear = []
        self.ref_root_angular = []
        self.ref_qpos = []
        self.ref_keypoints = []
        for clip in self.motion_clips:
            root_translations = torch.tensor(clip["root_trans"], dtype=torch.float32, device=self.device)

            root_orient_quat = R.from_rotvec(clip["root_orient"]).as_quat()
            root_orient_quat = torch.tensor(root_orient_quat, dtype=torch.float32, device=self.device)
            root_orient_quat = root_orient_quat[:, [3, 0, 1, 2]]

            root_linear = torch.tensor(clip["root_linear_velocity"], dtype=torch.float32, device=self.device)
            root_angular = torch.tensor(clip["root_angular_velocity"], dtype=torch.float32, device=self.device)

            qpos = torch.tensor(clip["qpos"], dtype=torch.float32, device=self.device)
            keypoints = torch.tensor(clip["keypoints"], dtype=torch.float32, device=self.device)

            root_translations, root_orient, root_linear, root_angular, qpos, keypoints = self._padding_reference_motion(
                                                root_translations, root_orient_quat, root_linear, root_angular, qpos, keypoints
                                                )
            
            self.ref_root_trans.append(root_translations)
            self.ref_root_orient.append(root_orient)
            self.ref_root_linear.append(root_linear)
            self.ref_root_angular.append(root_angular)
            self.ref_qpos.append(qpos)
            self.ref_keypoints.append(keypoints)

        self.ref_root_trans = torch.stack(self.ref_root_trans, dim=0)           # (num_clips, max_traj_len, 3)
        self.ref_root_orient = torch.stack(self.ref_root_orient, dim=0)         # (num_clips, max_traj_len, 4)
        self.ref_root_linear = torch.stack(self.ref_root_linear, dim=0)         # (num_clips, max_traj_len, 3)
        self.ref_root_angular = torch.stack(self.ref_root_angular, dim=0)       # (num_clips, max_traj_len, 3)
        self.ref_qpos = torch.stack(self.ref_qpos, dim=0)                       # (num_clips, max_traj_len, num_joints)
        self.ref_keypoints = torch.stack(self.ref_keypoints, dim=0)             # (num_clips, max_traj_len, num_keypoints * 3)

        self.motion_clips_len = torch.tensor(self.motion_clips_len, device=self.device)   # (num_clips)

    def _padding_reference_motion(self, root_trans, root_orient, root_linear, root_angular, qpos, keypoints):
        r'''
        root_trans: (num_frames, 3)
        root_orient: (num_frames, 4)
        root_linear: (num_frames, 3)
        root_angular: (num_frames, 3)
        qpos: (num_frames, num_joints)
        keypoints: (num_frames, num_keypoints * 3)
        '''
        num_frames = root_trans.shape[0]
        pad_frames = self.max_traj_len - num_frames

        last_translation = root_trans[-1:, :].repeat(pad_frames, 1)         # (pad_frames, 3)
        root_trans = torch.cat([root_trans, last_translation], dim=0)       # (max_traj_len, 3)

        last_orient = root_orient[-1:, :].repeat(pad_frames, 1)             # (pad_frames, 4)
        root_orient = torch.cat([root_orient, last_orient], dim=0)          # (max_traj_len, 4)

        zero_linear = torch.zeros(pad_frames, 3, device=self.device)        # (pad_frames, 3)
        root_linear = torch.cat([root_linear, zero_linear], dim=0)          # (max_traj_len, 3)

        zero_angular = torch.zeros(pad_frames, 3, device=self.device)       # (pad_frames, 3)
        root_angular = torch.cat([root_angular, zero_angular], dim=0)       # (max_traj_len, 3)

        joint_id, joint_names = self.asset.find_joints(self.joint_names, preserve_order=True)
        default_qpos = self.robot.data.default_joint_pos[0][joint_id]

        interpolate_frame = 50              # 1s = 50 frames
        t = torch.linspace(0, 1, interpolate_frame, device=self.device).unsqueeze(1)    # (interpolate_frame, 1)

        last_qpos = qpos[-1:, :]
        pad_qpos = (1-t) * last_qpos + t * default_qpos.unsqueeze(0)        # (interpolate_frame, num_joints)
        qpos = torch.cat([qpos, pad_qpos], dim=0)                        

        pad_qpos = default_qpos.unsqueeze(0).repeat(pad_frames, 1)         # (pad_frames, num_joints)
        qpos = torch.cat([qpos, pad_qpos], dim=0)                          # (max_traj_len, num_joints)

        body_id, body_names = self.asset.find_bodies(self.body_names, preserve_order=True)
        root_position = self.robot.data.root_pos_w.unsqueeze(1)
        root_quat = self.robot.data.root_quat_w.unsqueeze(1)
        body_pos_b = self.robot.data.body_pos_w - root_position
        body_pos_b = quat_rotate_inverse(root_quat, body_pos_b)
        default_body_pos = body_pos_b[0][body_id].reshape(1, -1)

        last_keypoints = keypoints[-1:, :]     # [1, 12 * 3]
        pad_keypoints = (1 - t) * last_keypoints + t * default_body_pos
        keypoints = torch.cat([keypoints, pad_keypoints], dim=0)

        pad_keypoints = default_body_pos.repeat(pad_frames, 1)
        keypoints = torch.cat([keypoints, pad_keypoints], dim=0)

        return root_trans, root_orient, root_linear, root_angular, qpos, keypoints

idx = [0, 6, 12, 1, 7, 13, 19, 2, 8, 14, 20, 3, 9, 15, 21, 4, 10, 16, 22, 5, 11, 17, 23, 18, 24]