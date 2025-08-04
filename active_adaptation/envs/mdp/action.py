import torch
from typing import Dict, Literal, Tuple, Union, TYPE_CHECKING
from tensordict import TensorDictBase
from isaaclab.assets import Articulation
import isaaclab.utils.string as string_utils
from isaaclab.utils.math import euler_xyz_from_quat, quat_mul, quat_conjugate, axis_angle_from_quat, quat_inv, quat_rotate_inverse, quat_rotate, yaw_quat

if TYPE_CHECKING:
    from active_adaptation.envs.base import Env

class ActionManager:
    
    action_dim: int

    def __init__(self, env):
        self.env: Env = env
        self.asset: Articulation = self.env.scene["robot"]
    
    def reset(self, env_ids: torch.Tensor):
        pass
    
    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device
        
class JointPosition(ActionManager):
    def __init__(
        self, 
        env,
        action_scaling: Dict[str, float] = 0.5,
        max_delay: int = 4,
        alpha: Union[float, Tuple[float, float], Dict[str, float], Dict[str, Tuple[float, float]]] = (0.5, 1.0),
        custom_command: Dict[str, float] = None,
        clip_joint_targets: float = None
    ):
        super().__init__(env)
        self.joint_ids, self.joint_names, self.action_scaling = string_utils.resolve_matching_names_values(
            dict(action_scaling), self.asset.joint_names, preserve_order=True)

        self.action_scaling = torch.tensor(self.action_scaling, device=self.device)
        self.max_delay = max_delay
        
        self.action_dim = len(self.joint_ids)
        
        import omegaconf
        if isinstance(alpha, float):
            self.alpha_range = torch.tensor([[alpha, alpha]], device=self.device).expand(self.action_dim, -1)
        elif isinstance(alpha, omegaconf.listconfig.ListConfig):
            self.alpha_range = torch.tensor(list(alpha), device=self.device).expand(self.action_dim, -1)
        elif isinstance(alpha, omegaconf.dictconfig.DictConfig):
            _, _, alpha_list = string_utils.resolve_matching_names_values(dict(alpha), self.asset.joint_names)
            if isinstance(alpha_list[0], float):
                alpha_list = [[alpha_list[i], alpha_list[i]] for i in range(len(alpha_list))]
            self.alpha_range = torch.tensor(alpha_list, device=self.device)
        else:
            raise ValueError(f"Invalid alpha type: {type(alpha)}")

        if custom_command is not None:
            custom_command = dict(custom_command)
            self.custom_command_joint_ids, _, self.custom_command = string_utils.resolve_matching_names_values(custom_command, self.asset.joint_names)
            self.custom_command = torch.tensor(self.custom_command, device=self.device)
        if clip_joint_targets is not None:
            self.clip_joint_targets = clip_joint_targets
            
        self.default_joint_pos = self.asset.data.default_joint_pos.clone()
        self.offset = torch.zeros_like(self.default_joint_pos)
        self.joint_limits = self.asset.data.joint_limits.clone().unbind(-1)
        self.decimation = int(self.env.step_dt / self.env.physics_dt)

        with torch.device(self.device):
            action_buf_hist = max(max_delay + 1, 5) if max_delay is not None else 5
            self.action_buf = torch.zeros(self.num_envs, self.action_dim, action_buf_hist) # at least 3 for action_rate_2_l2 reward
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, self.action_dim)
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)

    def reset(self, env_ids: torch.Tensor):
        self.delay[env_ids] = torch.randint(0, self.max_delay + 1, (len(env_ids), 1), device=self.device)
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0

        self.default_joint_pos[env_ids] = self.asset.data.default_joint_pos[env_ids].clone()
        self.default_joint_pos[env_ids] += self.offset[env_ids]

        alpha = torch.empty(len(env_ids), self.action_dim, device=self.device).uniform_(0, 1)
        alpha = self.alpha_range[:, 0] + alpha * (self.alpha_range[:, 1] - self.alpha_range[:, 0])
        self.alpha[env_ids] = alpha.pow(1.0 / self.decimation)

    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            action = tensordict["action"].clamp(-10, 10)
            self.action_buf[:, :, 1:] = self.action_buf[:, :, :-1]
            self.action_buf[:, :, 0] = action
            self.delay[:] = torch.randint(0, self.max_delay + 1, (self.num_envs, 1), device=self.device)
        dim = (self.delay - substep + self.decimation) // self.decimation
        action = self.action_buf.take_along_dim(dim.unsqueeze(1), dim=-1)
        self.applied_action.lerp_(action.squeeze(-1), self.alpha)

        pos_target = self.default_joint_pos.clone()
        pos_target[:, self.joint_ids] += self.applied_action * self.action_scaling
        if hasattr(self, "custom_command"):
            pos_target[:, self.custom_command_joint_ids] = self.custom_command
        if hasattr(self, "clip_joint_targets"):
            pos_target = self.asset.data.joint_pos + (pos_target - self.asset.data.joint_pos).clamp(-self.clip_joint_targets, self.clip_joint_targets)
        self.asset.set_joint_position_target(pos_target.clamp(*self.joint_limits))
        self.asset.write_data_to_sim()


def clamp_norm(x: torch.Tensor, max_norm: float):
    norm = x.norm(dim=-1, keepdim=True)
    return x * (max_norm / norm.clamp(min=1e-6)).clamp(max=1.0)