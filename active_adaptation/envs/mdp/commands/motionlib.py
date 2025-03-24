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
import importlib.util
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d

spec = importlib.util.find_spec("active_adaptation")
package_path = spec.origin

quat_rotate_inverse = batchify(quat_rotate_inverse)

CURRENT_MOTION = 0

class MotionLib(Command):
    source_fps: int = 30
    target_fps: int = 50
    def __init__(
            self, 
            env,
            motion_clip_dir: str,
            dataset: str,
            occlusion: str,
            mode: str = "train",
            eval_id: int = None,
            teleop: bool = False,
        ):
        super().__init__(env, teleop=teleop)
        self.robot: Articulation = env.scene["robot"]

        package_dir = os.path.dirname(package_path)

        occlusion_path = os.path.join(package_dir, "..", motion_clip_dir, occlusion)
        occlusion_keys = list(joblib.load(occlusion_path).keys())

        motion_clip = os.path.join(package_dir, "..", motion_clip_dir, dataset) + ".pkl"

        data = joblib.load(motion_clip)
        data = {k.replace("_stageii", "_poses"): v for k, v in data.items()}
        data = {k: v for k, v in data.items() if k not in occlusion_keys}

        if eval_id is not None:
            data_keys = list(data.keys())
            data = {data_keys[eval_id]: data[data_keys[eval_id]]}
        
        self.env_origin = self.env.scene.env_origins
        self.bodys = [j[0] for j in joint_matches]
        self.load_data(data)

        self.mode = mode
        if mode == "play":
            from pynput import keyboard
            def on_press(key):
                global CURRENT_MOTION
                try:
                    if key.char == "n":
                        CURRENT_MOTION += 1
                        CURRENT_MOTION %= self.num_motions
                        print(f"\nSwitching to motion {CURRENT_MOTION}")
                except AttributeError:
                    pass
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
    
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        motion_ids = torch.randint(0, self.num_motions, (env_ids.shape[0],))
        start_frames = self.start_frames[motion_ids]
        end_frames = self.end_frames[motion_ids]

        motion_length = self.motion_length[motion_ids]
        r = torch.rand(motion_length.shape)
        offsets = (r * motion_length.float()).floor().long()
        start_frames += offsets

        if self.mode == "play" or self.mode == "eval":
            motion_ids = torch.ones(env_ids.shape[0], dtype=torch.long) * CURRENT_MOTION
            start_frames = self.start_frames[motion_ids]
            end_frames = self.end_frames[motion_ids]

        init_root_state = self.init_root_state[env_ids]     # (num_envs, 3 + 4 + 6) root position, root orientation, root linear velocity and root angular velocity
        init_root_state[:, :3] = self.root_translations[start_frames].to(self.device) + self.env_origin[env_ids]
        init_root_state[:, :3] += torch.tensor([0, 0, 0.03], device=self.device)
        init_root_state[:, 3:7] = self.root_orientation[start_frames].to(self.device)

        qpos = self.qpos[start_frames].to(self.device)
        self.robot.write_joint_state_to_sim(
            qpos,
            self.robot.data.default_joint_vel[env_ids],
            env_ids=env_ids
        )
        
        return init_root_state, start_frames.to(self.device), end_frames.to(self.device)
    
    def reset(self, env_ids: torch.Tensor):
        pass

    def load_data(self, data):
        self.motion_length = []
        self.root_translations = []
        self.root_orientation = []
        self.root_linear = []
        self.qpos = []
        self.kp = []

        mujoco_to_isaac_idx = mujoco_to_isaac()

        pbar = tqdm(data.items())
        for k, motion in pbar:
            pbar.set_description(f"Loading {k}: ")
            interpolated_root_trans = self.interpolate(motion, "root_trans_offset", self.source_fps, self.target_fps)
            interpolated_root_rot = self.interpolate(motion, "root_rot", self.source_fps, self.target_fps)
            interpolated_qpos = self.interpolate(motion, "dof", self.source_fps, self.target_fps)
            interpolated_kp = self.interpolate(motion, "smpl_joints", self.source_fps, self.target_fps)
            interpolated_kp_local = convert2local(interpolated_kp, interpolated_root_rot)

            self.motion_length.append(interpolated_root_trans.shape[0])
            self.root_translations.append(interpolated_root_trans)
            self.root_linear.append(torch.diff(interpolated_root_trans,
                                               dim=0,
                                               append=torch.zeros(1, 3)) * self.target_fps)
            self.root_orientation.append(interpolated_root_rot[:, [3, 0, 1, 2]])
            self.qpos.append(interpolated_qpos[:, mujoco_to_isaac_idx])
            self.kp.append(interpolated_kp_local)

        self.motion_length = torch.tensor(self.motion_length)
        self.root_translations = torch.cat(self.root_translations, dim=0).float()
        self.root_orientation = torch.cat(self.root_orientation, dim=0).float()
        self.root_linear = torch.cat(self.root_linear, dim=0).float()
        self.qpos = torch.cat(self.qpos, dim=0).float()
        self.kp = torch.cat(self.kp, dim=0).float()

        self.num_motions = len(data)
        self.num_frames = self.root_translations.shape[0]

        self.start_frames = torch.cat([torch.zeros(1), self.motion_length.cumsum(dim=0)[:-1]]).long()
        self.end_frames = self.motion_length.cumsum(dim=0).long()

    def interpolate(self, motion, key, source_fps, target_fps):
        motion_data = motion[key]
        motion_length = motion_data.shape[0]

        source_time = np.linspace(0, motion_length / source_fps, motion_length)
        target_length = int(np.ceil(motion_length * target_fps / source_fps))
        target_time = np.linspace(0, motion_length / source_fps, target_length)

        if key == "root_rot":
            rotations = R.from_quat(motion_data)
            slerp = Slerp(source_time, rotations)
            interpolated_quats = slerp(target_time).as_quat()
            return torch.tensor(interpolated_quats)

        elif key in ["root_trans_offset", "dof", "smpl_joints"]:
            interpolator = interp1d(source_time, motion_data, axis=0, kind="linear", fill_value="extrapolate")
            interpolated = interpolator(target_time)
            return torch.tensor(interpolated)

        else:
            raise NotImplementedError(f"Interpolation for key '{key}' is not implemented.")

    # # for sanity check
    # def update(self):
    #     self.frames = self.env.episode_length_buf.cpu()
    #     env_ids = torch.arange(self.num_envs, device=self.device)
        
    #     root_state = self.robot.data.root_state_w.clone()
    #     root_state[:, :3] = self.root_translations[self.frames].to(self.device) + self.env_origin + torch.tensor([0, 0, 1.], device=self.device)
    #     root_state[:, 3:7] = self.root_orientation[self.frames].to(self.device)
    #     self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    #     qpos = self.qpos[self.frames].to(self.device)
    #     self.robot.write_joint_state_to_sim(
    #         qpos,
    #         self.robot.data.default_joint_vel,
    #         env_ids=env_ids
    #     )
    #     return

def convert2local(kp, root_orientation):
    r'''
    Args:
        kp: (N, 24, 3)  torch tensor in global coordinate system
        root_orientation: (N, 4)    in format of quaternion (x, y, z, w)
    Returns:
        smpl_kp: (N, 24, 3) in local coordinate system
    '''
    smpl_idx = [SMPL_BONE_ORDER_NAMES.index(j[1]) for j in joint_matches]
    smpl_root = kp[:, 0:1, :]
    smpl_kp = kp - smpl_root
    smpl_kp = smpl_kp[:, smpl_idx, :]
    root_orient_inv = torch.tensor(R.from_quat(root_orientation.cpu().numpy()).inv().as_matrix())
    smpl_kp = torch.einsum('nij, nkj->nki', root_orient_inv, smpl_kp)
    # animate_3d(smpl_kp, root_orientation)
    return smpl_kp

def mujoco_to_isaac():
    mujoco_to_isaac = []
    for joint in isaacsim_joints:
        mujoco_index = mujoco_joints.index(joint)
        mujoco_to_isaac.append(mujoco_index)
    return mujoco_to_isaac

from matplotlib import pyplot as plt
import matplotlib.animation as animation

def animate_3d(joints, orientation):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vis = 1

    def update(num, data, line):
        ax.clear()
        ax.scatter(data[num][:, 0], data[num][:, 1], data[num][:, 2], c='y', marker='o')
        ax.scatter(data[num][vis, 0], data[num][vis, 1], data[num][vis, 2], c='r', marker='*', s=50)

        # unit_vector = np.array([1, 0, 0])
        # unit_vector = R.apply(R.from_quat(orientation[num]), unit_vector)
        # ax.quiver(0, 0, 0, unit_vector[0], unit_vector[1], unit_vector[2], color='r', length=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_title(f"Frame {num}")

        ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=0.1)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=0.1)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=0.1)
        return line,

    ani = animation.FuncAnimation(fig, update, frames=joints.shape[0], fargs=(joints, None), interval=50)
    plt.show()
            
SMPL_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]

joint_matches = [
    ["left_hip_pitch_link", "L_Hip"],
    ["left_knee_link", "L_Knee"],
    ["left_ankle_roll_link", "L_Ankle"],
    ["right_hip_pitch_link", "R_Hip"],
    ["right_knee_link", "R_Knee"],
    ["right_ankle_roll_link", "R_Ankle"],
    ["left_shoulder_roll_link", "L_Shoulder"],
    ["left_elbow_pitch_link", "L_Elbow"],
    ["left_zero_link", "L_Hand"],
    ["right_shoulder_roll_link", "R_Shoulder"],
    ["right_elbow_pitch_link", "R_Elbow"],
    ["right_zero_link", "R_Hand"]
]

mujoco_joints = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
    "left_zero_joint",
    "left_one_joint",
    "left_two_joint",
    "left_three_joint",
    "left_four_joint",
    "left_five_joint",
    "left_six_joint",
    "right_zero_joint",
    "right_one_joint",
    "right_two_joint",
    "right_three_joint",
    "right_four_joint",
    "right_five_joint",
    "right_six_joint",
]

isaacsim_joints = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "torso_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_elbow_roll_joint", "right_elbow_roll_joint",
    "left_five_joint",
    "left_three_joint",
    "left_zero_joint",
    "right_five_joint",
    "right_three_joint",
    "right_zero_joint",
    "left_six_joint",
    "left_four_joint",
    "left_one_joint",
    "right_six_joint",
    "right_four_joint",
    "right_one_joint",
    "left_two_joint",
    "right_two_joint",
]