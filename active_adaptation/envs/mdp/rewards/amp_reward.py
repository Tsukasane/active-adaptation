from math import inf
import torch

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import yaw_quat
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from active_adaptation.utils.helpers import batchify

from .locomotion import Reward, normalize

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

def dot(a: torch.Tensor, b: torch.Tensor):
    return (a * b).sum(-1, True)

class amp_tracking_root_trans(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, sigma: float = 0.1):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        
        self.sigma = sigma
        self.decay = self.env.command_manager.decay
    
    def compute(self) -> torch.Tensor:
        
        timestep = self.env.episode_length_buf - 1

        try:
            assert (timestep < self.env.max_episode_length).all()
        except AssertionError as e:
            timestep_idx = torch.where(timestep >= self.env.max_episode_length)[0]
            print(f"timestep in tracking root is greater than {self.env.max_episode_length} at {timestep_idx} with value {timestep[timestep_idx]}")

        batch_indices = torch.arange(self.num_envs, device=self.device)
        ref_root_translation = self.env.command_manager.ref_root_trans[batch_indices, timestep]

        root_pos_w = self.asset.data.root_pos_w
        assert root_pos_w.shape == ref_root_translation.shape
        err = (root_pos_w - ref_root_translation[:, :3]).square().sum(-1, True)

        reward = torch.exp(- err.sqrt() / self.sigma)
        return reward
    
class amp_tracking_root_rot(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, sigma: float = 0.1):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        
        self.sigma = sigma
        self.decay = self.env.command_manager.decay
    
    def compute(self) -> torch.Tensor:
        
        timestep = self.env.episode_length_buf - 1
        batch_indices = torch.arange(self.num_envs, device=self.device)
        ref_root_rotation = self.env.command_manager.ref_root_orient[batch_indices, timestep]

        root_rot_w = self.asset.data.root_quat_w
        assert root_rot_w.shape == ref_root_rotation.shape
        dot_product = dot(root_rot_w, ref_root_rotation)
        err = 2 * torch.acos(dot_product.abs().clamp(max=1.0))

        reward = torch.exp(- err / self.sigma)
        return reward
    
class amp_tracking_qpos(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, sigma: float = 0.1, joint_names: str=".*"):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids = self.asset.find_joints(joint_names, preserve_order=True)[0]
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)

        self.sigma = sigma
        self.decay = self.env.command_manager.decay
    
    def compute(self) -> torch.Tensor:
        timestep = self.env.episode_length_buf - 1
        if not (timestep < self.env.max_episode_length).all():
            timestep_idx = torch.where(timestep >= self.env.max_episode_length)[0]
            print(f"timestep in tracking qpos is greater than {self.env.max_episode_length} at {timestep_idx} with value {timestep[timestep_idx]}")

        if not (timestep < self.env.command_manager.max_traj_len).all():
            timestep_idx = torch.where(timestep >= self.env.command_manager.max_traj_len)[0]
            print(f"timestep in tracking qpos is greater than {self.env.command_manager.max_traj_len} at {timestep_idx} with value {timestep[timestep_idx]}")

        batch_indices = torch.arange(self.num_envs, device=self.device)
        ref_qpos = self.env.command_manager.ref_qpos[batch_indices, timestep]       # torch.Size([num_envs, 23])
        # fix arm joint 5 and joint 6
        indice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20]
        ref_qpos = ref_qpos[:, indice]
        assert ref_qpos.shape == self.asset.data.joint_pos[:, self.joint_ids].shape
        err = (self.asset.data.joint_pos[:, self.joint_ids] - ref_qpos).square()    # torch.Size([num_envs, num_joints])
        
        reward = torch.exp(- err.mean(-1, True) / self.sigma)
        return reward
    
class amp_tracking_keypoints(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, sigma: float = 0.1, 
                 upper_body_names: str=".*", upper_body_err_weight: float=1.0,
                 lower_body_names: str=".*", lower_body_err_weight: float=1.5):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.upper_body_ids = self.asset.find_bodies(upper_body_names, preserve_order=True)[0]
        self.upper_body_ids = torch.tensor(self.upper_body_ids, device=self.device)
        self.lower_body_ids = self.asset.find_bodies(lower_body_names, preserve_order=True)[0]
        self.lower_body_ids = torch.tensor(self.lower_body_ids, device=self.device)

        self.upper_body_err_weight = upper_body_err_weight
        self.lower_body_err_weight = lower_body_err_weight

        self.sigma = sigma
        self.decay = self.env.command_manager.decay
    
    def compute(self) -> torch.Tensor:
        timestep = self.env.episode_length_buf - 1
        batch_indices = torch.arange(self.num_envs, device=self.device)
        ref_keypoints = self.env.command_manager.ref_keypoints[batch_indices, timestep].reshape(self.num_envs, -1, 3)
        up_body_pos_w = self.asset.data.body_pos_w[:, self.upper_body_ids]   # torch.Size([num_envs, num_bodies, 3])
        low_body_pos_w = self.asset.data.body_pos_w[:, self.lower_body_ids]   # torch.Size([num_envs, num_bodies, 3])
        root_position = self.asset.data.root_pos_w.unsqueeze(1)
        up_body_pos_w = up_body_pos_w - root_position
        low_body_pos_w = low_body_pos_w - root_position
        root_quat = self.asset.data.root_quat_w.unsqueeze(1)

        up_body_pos_b = quat_rotate_inverse(root_quat, up_body_pos_w)
        assert up_body_pos_b.shape == ref_keypoints[:, :4].shape
        diff_up = up_body_pos_b - ref_keypoints[:, :4]
        diff_per_kp_up = diff_up.norm(dim=-1)
        err_up = diff_per_kp_up.square().sum(-1, True)

        low_body_pos_b = quat_rotate_inverse(root_quat, low_body_pos_w)
        assert low_body_pos_b.shape == ref_keypoints[:, 6:10].shape
        diff_low = low_body_pos_b - ref_keypoints[:, 6:10]
        diff_per_kp_low = diff_low.norm(dim=-1)
        err_low = diff_per_kp_low.square().sum(-1, True)

        err = err_up * self.upper_body_err_weight + err_low * self.lower_body_err_weight
        
        reward = torch.exp(- err.sqrt() / self.sigma)
        return reward
    
    def debug_draw(self):
        timestep = self.env.episode_length_buf - 1
        batch_indices = torch.arange(self.num_envs, device=self.device)
        ref_keypoints = self.env.command_manager.ref_keypoints[batch_indices, timestep].reshape(self.num_envs, -1, 3)       # keypoint in root local frame

        root_position = self.asset.data.root_pos_w.unsqueeze(1)
        root_quat = self.asset.data.root_quat_w.unsqueeze(1)

        kp_global = quat_rotate(root_quat, ref_keypoints) + root_position
        for i in range(kp_global.shape[1]):
            self.env.debug_draw.point(kp_global[:, i], color=(0., 1., 1., 1.), size = 30)

class amp_tracking_end_effector(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, sigma: float = 0.1, 
                 upper_body_names: str=".*", upper_body_err_weight: float=1.0,
                 lower_body_names: str=".*", lower_body_err_weight: float=1.5):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.upper_body_ids = self.asset.find_bodies(upper_body_names, preserve_order=True)[0]
        self.upper_body_ids = torch.tensor(self.upper_body_ids, device=self.device)
        self.lower_body_ids = self.asset.find_bodies(lower_body_names, preserve_order=True)[0]
        self.lower_body_ids = torch.tensor(self.lower_body_ids, device=self.device)

        self.upper_body_err_weight = upper_body_err_weight
        self.lower_body_err_weight = lower_body_err_weight

        self.sigma = sigma
        self.decay = self.env.command_manager.decay
    
    def compute(self) -> torch.Tensor:
        timestep = self.env.episode_length_buf - 1
        batch_indices = torch.arange(self.num_envs, device=self.device)
        ref_keypoints = self.env.command_manager.ref_keypoints[batch_indices, timestep].squeeze(1).reshape(self.num_envs, -1, 3)    # [num_envs, num_keypoints, 3]

        # l_shoulder, r_shoulder,
        # l_elbow, r_elbow,
        # l_wrist, r_wrist,
        # l_hip, r_hip,
        # l_knee, r_knee,
        # l_ankle, r_ankle

        hand_eff_kp = ref_keypoints[:, 4:6]     # [num_envs, 2, 3]
        foot_eff_kp = ref_keypoints[:, 10:12]   # [num_envs, 2, 3]

        up_body_pos_w = self.asset.data.body_pos_w[:, self.upper_body_ids]   # torch.Size([num_envs, num_bodies, 3])
        low_body_pos_w = self.asset.data.body_pos_w[:, self.lower_body_ids]   # torch.Size([num_envs, num_bodies, 3])
        root_position = self.asset.data.root_pos_w.unsqueeze(1)
        up_body_pos_w = up_body_pos_w - root_position
        low_body_pos_w = low_body_pos_w - root_position
        root_quat = self.asset.data.root_quat_w.unsqueeze(1)

        up_body_pos_b = quat_rotate_inverse(root_quat, up_body_pos_w)
        diff_up = up_body_pos_b - hand_eff_kp
        diff_per_kp_up = diff_up.norm(dim=-1)
        err_up = diff_per_kp_up.square().sum(-1, True)

        low_body_pos_b = quat_rotate_inverse(root_quat, low_body_pos_w)
        diff_low = low_body_pos_b - foot_eff_kp
        diff_per_kp_low = diff_low.norm(dim=-1)
        err_low = diff_per_kp_low.square().sum(-1, True)

        err = err_up * self.upper_body_err_weight + err_low * self.lower_body_err_weight
        
        reward = torch.exp(- err.sqrt() / self.sigma)
        return reward