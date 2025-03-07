import torch
import abc

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sensors import ContactSensor

class Termination:
    def __init__(self, env):
        self.env = env
    
    def update(self):
        pass

    def reset(self, env_ids):
        pass
    
    @abc.abstractmethod
    def __call__(self) -> torch.Tensor:
        raise NotImplementedError
    
    @property
    def num_envs(self) -> int:
        return self.env.num_envs


def termination_func(func):
    class TermFunc(Termination):
        def __call__(self):
            return func(self.env)
    return TermFunc


class crash(Termination):
    def __init__(
        self, 
        env, 
        body_names_expr: str,
        t_thres: float = 0.,
        min_time: float = 0.,
        **kwargs
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_indices, self.body_names = self.contact_sensor.find_bodies(body_names_expr)
        self.t_thres = t_thres
        self._decay = 0.98
        self._thres = (self.t_thres / self.env.physics_dt) * 0.9
        self.count = torch.zeros(self.num_envs, len(self.body_indices), device=self.env.device)
        self.min_steps = int(min_time / self.env.step_dt)
        print(f"Terminate upon contact on {self.body_names}")
    
    def reset(self, env_ids):
        self.count[env_ids] = 0.
    
    def update(self):
        in_contact = self.contact_sensor.data.net_forces_w[:, self.body_indices].norm(dim=-1) > 1.0
        self.count.add_(in_contact.float()).mul_(self._decay)
        
    def __call__(self):
        valid = (self.env.episode_length_buf > self.min_steps)
        undesired_contact = (self.count > self._thres).any(-1)
        return (undesired_contact & valid).reshape(self.num_envs, 1)


class fall_over(Termination):
    def __init__(
        self, 
        env, 
        xy_thres: float=0.8,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.xy_thres = xy_thres
    
    def __call__(self):
        gravity_xy: torch.Tensor = self.asset.data.projected_gravity_b[:, :2]
        fall_over = gravity_xy.norm(dim=1, keepdim=True) >= self.xy_thres
        return fall_over
    
class max_traj_length(Termination):
    def __init__(self, env):
        super().__init__(env)
        self.max_traj_length = self.env.command_manager.motion_clips_len    # [num_envs]
    
    def __call__(self) -> torch.Tensor:
        return (self.env.episode_length_buf >= self.max_traj_length).unsqueeze(1) 
    
class root_deviation_m(Termination):
    def __init__(self, env, max_distance: float):
        super().__init__(env)
        self.max_distance = torch.tensor(max_distance, device=self.env.device)
        self.asset: Articulation = self.env.scene["robot"]
    
    def __call__(self) -> torch.Tensor:
        timestep = self.env.episode_length_buf.unsqueeze(1) - 1
        batch_indices = torch.arange(self.num_envs, device=self.env.device)
        ref_root_translation = self.env.command_manager.ref_root_trans[batch_indices, timestep.squeeze(1)]

        root_pos_w = self.asset.data.root_pos_w
        deviation = (root_pos_w - ref_root_translation).norm(dim=1, keepdim=True)

        return deviation > self.max_distance

class root_deviation(Termination):
    def __init__(self, env, max_distance: float):
        super().__init__(env)
        self.max_distance = torch.tensor(max_distance, device=self.env.device)
        self.asset: Articulation = self.env.scene["robot"]
    
    def __call__(self) -> torch.Tensor:
        timestep = self.env.episode_length_buf.unsqueeze(1) - 1
        batch_indices = torch.arange(self.num_envs, device=self.env.device)
        ref_root_translation = self.env.command_manager.ref_root_translations[batch_indices, timestep.squeeze(1)]

        root_pos_w = self.asset.data.root_pos_w
        deviation = (root_pos_w - ref_root_translation).norm(dim=1, keepdim=True)

        return deviation > self.max_distance
    
def dot(a: torch.Tensor, b: torch.Tensor):
    return (a * b).sum(-1, True)

class root_rot_deviation_m(Termination):
    def __init__(self, env, max_theta: float):
        super().__init__(env)
        radian = max_theta * 3.14 / 180
        self.max_theta = torch.tensor(radian, device=self.env.device)
        self.asset: Articulation = self.env.scene["robot"]

    def __call__(self) -> torch.Tensor:
        timestep = self.env.episode_length_buf.unsqueeze(1) - 1
        batch_indices = torch.arange(self.num_envs, device=self.env.device)
        ref_root_orientation = self.env.command_manager.ref_root_orient[batch_indices, timestep.squeeze(1)]

        root_quat_w = self.asset.data.root_quat_w
        dot_product = dot(root_quat_w, ref_root_orientation)
        deviation = 2 * torch.acos(dot_product.abs().clamp(min=-1.0, max=1.0))

        return deviation > self.max_theta
class root_rot_deviation(Termination):
    def __init__(self, env, max_theta: float):
        super().__init__(env)
        radian = max_theta * 3.14 / 180
        self.max_theta = torch.tensor(radian, device=self.env.device)
        self.asset: Articulation = self.env.scene["robot"]

    def __call__(self) -> torch.Tensor:
        timestep = self.env.episode_length_buf.unsqueeze(1) - 1
        ref_root_orientation = self.env.command_manager.ref_root_orient[timestep.squeeze(1)]

        root_quat_w = self.asset.data.root_quat_w
        dot_product = dot(root_quat_w, ref_root_orientation)
        deviation = 2 * torch.acos(dot_product.abs().clamp(min=-1.0, max=1.0))

        return deviation > self.max_theta

class tracking_error(Termination):
    def __init__(self, env, tracking_error_threshold):
        super().__init__(env)
        self.tracking_error_threshold = tracking_error_threshold
        self.asset: Articulation = self.env.scene["robot"]
    
    def __call__(self) -> torch.Tensor:
        return self.asset.data._tracking_error > self.tracking_error_threshold

class joint_acc_exceeds(Termination):
    def __init__(self, env, thres: float):
        super().__init__(env)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]
    
    def __call__(self) -> torch.Tensor:
        valid = (self.env.episode_length_buf > 2).unsqueeze(-1)
        return (
            valid & 
            (self.asset.data.joint_acc.abs() > self.thres).any(1, True)
        )

class impact_exceeds(Termination):
    def __init__(self, env, body_names: str, thres: float):
        super().__init__(env)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.body_ids = self.contact_sensor.find_bodies(body_names)[0]
    
    def __call__(self) -> torch.Tensor:
        impact_force = self.contact_sensor.data.net_forces_w_history[:, :, self.body_ids]
        return (impact_force.norm(dim=-1).mean(1) > self.thres).any(1, True)