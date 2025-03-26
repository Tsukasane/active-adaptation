from math import inf
import torch
import abc

from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import yaw_quat, wrap_to_pi
import omni.isaac.lab.utils.string as string_utils
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from active_adaptation.utils.helpers import batchify
from ..commands import *

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

class Reward:
    def __init__(self, env, weight: float, enabled: bool=True, clip_range=(-torch.inf, +torch.inf)):
        self.env = env
        self.weight = weight
        self.enabled = enabled
        self.clip_range = clip_range
    
    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device
    
    def step(self, substep: int):
        pass

    def update(self):
        pass

    def reset(self, env_ids):
        pass
    
    def __call__(self) -> torch.Tensor:
        return (self.weight * self.compute()).clip(*self.clip_range)
    
    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError

    def debug_draw(self):
        pass


def reward_func(func):
    class RewFunc(Reward):
        def compute(self):
            return func(self.env)
    return RewFunc


@reward_func
def energy_l2(self):
    asset: Articulation = self.scene["robot"]
    energy = (
        (asset.data.joint_vel * asset.data.applied_torque)
        .square()
        .sum(dim=-1, keepdim=True)
    )
    return - energy


@reward_func
def joint_acc_l2(self):
    asset: Articulation = self.scene["robot"]
    r = asset.data.joint_acc.square().sum(dim=-1, keepdim=True)
    return -r


@reward_func
def survival(self):
    return torch.ones(self.num_envs, 1, device=self.device)

class angvel_xy_l2(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, body_name: str=None):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        if body_name is not None:
            self.body_id = self.asset.find_bodies(body_name)[0][0]
        else:
            self.body_id = None
        self.world_frame = False
            
    def update(self):
        if self.body_id is not None:
            angvel = self.asset.data.body_ang_vel_w[:, self.body_id]
            if not self.world_frame:
                angvel = quat_rotate_inverse(self.asset.data.root_quat_w, angvel)
        else:
            if self.world_frame:
                angvel = self.asset.data.root_ang_vel_w
            else:
                angvel = self.asset.data.root_ang_vel_b
        self.angvel = angvel
    
    def compute(self) -> torch.Tensor:
        r = self.angvel[:, :2].square().sum(-1, True)
        return - r

class energy_l1(Reward):
    
    decay: float = 0.99

    def __init__(self, env, weight: float, enabled: bool = True, a={".*": 1.0}):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _, self.a = string_utils.resolve_matching_names_values(dict(a), self.asset.joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.a = torch.tensor(self.a, device=self.device)
        
        self.power = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)
        self.energy = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)
        self.count = torch.zeros(self.num_envs, 1, device=self.device)
        self.asset.data.energy_ema = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)

    def reset(self, env_ids):
        self.energy[env_ids] = 0.
        self.count[env_ids] = 0.

    def update(self):
        torques = self.asset.data.applied_torque[:, self.joint_ids]
        joint_vel = self.asset.data.joint_vel[:, self.joint_ids]
        self.power[:] = (torques * joint_vel).abs()
        
        self.energy.add_(self.power).mul_(self.decay)
        self.count.add_(1.).mul_(self.decay)
        self.asset.data.energy_ema[:] = self.energy / self.count

    def compute(self) -> torch.Tensor:
        return - (self.power * self.a).sum(1, keepdim=True)

class joint_torques_l2(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, joint_names: str=".*"):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids =  self.asset.find_joints(joint_names)[0]
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
    
    def compute(self) -> torch.Tensor:
        return - self.asset.data.applied_torque[:, self.joint_ids].square().sum(1, keepdim=True)


class feet_air_time(Reward):
    def __init__(
        self, 
        env, 
        body_names: str, 
        thres: float, 
        weight: float, 
        enabled: bool=True,
        condition_on_linvel: bool=True,
    ):
        super().__init__(env, weight, enabled)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.condition_on_linvel = condition_on_linvel

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.reward = torch.zeros(self.num_envs, 1, device=self.env.device)

    def compute(self):
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[:, self.body_ids]
        last_air_time = self.contact_sensor.data.last_air_time[:, self.body_ids]
        self.reward = torch.sum((last_air_time - self.thres).clamp_max(0.) * first_contact, dim=1, keepdim=True)
        self.reward *= (~self.env.command_manager.is_standing_env)
        if self.condition_on_linvel and hasattr(self.asset.data, "linvel_exp"):
            self.reward *= self.asset.data.linvel_exp
        return self.reward

class max_feet_height(Reward):
    def __init__(
        self,
        env,
        body_names: str,
        target_height: float,
        weight: float,
        enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.target_height = target_height

        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

        self.asset_body_ids, self.asset_body_names = self.asset.find_bodies(body_names)

        self.in_contact = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool, device=self.device)
        self.impact = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool, device=self.device)
        self.detach = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool, device=self.device)
        self.has_impact = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool, device=self.device)
        self.max_height = torch.zeros(self.num_envs, len(self.body_ids), device=self.device)
        self.impact_point = torch.zeros(self.num_envs, len(self.body_ids), 3, device=self.device)
        self.detach_point = torch.zeros(self.num_envs, len(self.body_ids), 3, device=self.device)

    def reset(self, env_ids):
        self.has_impact[env_ids] = False
    
    def update(self):
        contact_force = self.contact_sensor.data.net_forces_w_history[:, :, self.body_ids]
        feet_pos_w = self.asset.data.body_pos_w[:, self.asset_body_ids]
        in_contact = (contact_force.norm(dim=-1) > 0.01).any(dim=1)
        self.impact = (~self.in_contact) & in_contact
        self.detach = self.in_contact & (~in_contact)
        self.in_contact = in_contact
        self.has_impact.logical_or_(self.impact)
        self.impact_point[self.impact] = feet_pos_w[self.impact]
        self.detach_point[self.detach] = feet_pos_w[self.detach]
        self.max_height = torch.where(
            self.detach,
            feet_pos_w[:, :, 2],
            torch.maximum(self.max_height, feet_pos_w[:, :, 2])
        )

    def compute(self) -> torch.Tensor:
        reference_height = torch.maximum(self.impact_point[:, :, 2], self.detach_point[:, :, 2])
        max_height = self.max_height - reference_height
        r = self.impact * (max_height / self.target_height).clamp_max(1.0) 
        return r.sum(dim=1, keepdim=True)

    def debug_draw(self):
        feet_pos_w = self.asset.data.body_pos_w[:, self.asset_body_ids]
        self.env.debug_draw.point(
            feet_pos_w[self.impact],
            color=(1.0, 0., 0., 1.),
            size=20,
        )


class feet_contact_count(Reward):
    def __init__(self, env: "LocomotionEnv", body_names: str, weight: float, enabled: bool=True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.first_contact = torch.zeros(self.num_envs, len(self.body_ids), device=self.env.device)

    def compute(self):
        self.first_contact[:] = self.contact_sensor.compute_first_contact(self.env.step_dt)[:, self.body_ids]
        return self.first_contact.sum(1, keepdim=True)


from ..observations import _initialize_warp_meshes, raycast_mesh

class step_lift(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.mesh = _initialize_warp_meshes("/World/ground", "cuda")
        self.command_manager: Command2 = self.env.command_manager
        self.feet_ids = self.asset.find_bodies(".*foot")[0]

    def compute(self) -> torch.Tensor:
        ray_dir = self.command_manager.command_linvel_w
        ray_start_w = self.asset.data.body_pos_w[:, self.feet_ids]
        _, distance, normal, _ = raycast_mesh(
            ray_start_w,
            ray_dir,
            max_dist=0.2,
            mesh=self.mesh,
            return_distance=True,
            return_normal=True
        )
        distance = distance.nan_to_num(nan=1.0, posinf=1.0)
        hit_stair = (distance < 0.05) & (normal[:, :, 2].abs() < 0.05)
        feet_vel_z = self.asset.data.body_lin_vel_w[:, self.feet_ids, 2]
        r = hit_stair * feet_vel_z.clamp_min(0.0)
        return r.max(1, True).values


class quadruped_stand_always(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, clip_range=(-torch.inf, +torch.inf)):
        super().__init__(env, weight, enabled, clip_range)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids = self.asset.actuators["base_legs"].joint_indices

    def compute(self):
        jpos_error = (
            self.asset.data.joint_pos[:, self.joint_ids] - 
            self.asset.data.default_joint_pos[:, self.joint_ids]
        ).abs().sum(dim=1, keepdim=True)

        front_symmetry = self.asset.data.feet_pos_b[:, [0, 1], 1].sum(dim=1, keepdim=True).abs()
        back_symmetry = self.asset.data.feet_pos_b[:, [2, 3], 1].sum(dim=1, keepdim=True).abs()
        cost = - (jpos_error + front_symmetry + back_symmetry)

        return cost


class quadruped_stand(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, clip_range=(-torch.inf, +torch.inf)):
        super().__init__(env, weight, enabled, clip_range)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids = self.asset.actuators["base_legs"].joint_indices

    def compute(self):
        jpos_error = (
            self.asset.data.joint_pos[:, self.joint_ids] - 
            self.asset.data.default_joint_pos[:, self.joint_ids]
        ).abs().sum(dim=1, keepdim=True)

        front_symmetry = self.asset.data.feet_pos_b[:, [0, 1], 1].sum(dim=1, keepdim=True).abs()
        back_symmetry = self.asset.data.feet_pos_b[:, [2, 3], 1].sum(dim=1, keepdim=True).abs()
        cost = - (jpos_error + front_symmetry + back_symmetry)

        is_standing = self.env.command_manager.is_standing_env.squeeze(1)
        cost[~is_standing] = 0
        cost[is_standing] -= cost[is_standing].mean()
        return cost


class quadruped_stand_feet_contact_force(Reward):
    # expecting the foot to contact the ground firmly but not with too much force

    def __init__(self, env, weight, body_names, enabled=True, force_range=(10., 80.)):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.force_range = force_range
    
        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]

        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
    
    def compute(self):
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.body_ids]
        lower_bound, upper_bound = self.force_range
        force_penalty = (contact_forces < lower_bound).float() + (contact_forces > upper_bound).float()
        # force_penalty = (contact_forces - contact_forces.clamp(lower_bound, upper_bound)).abs()
        total_penalty = torch.sum(force_penalty, dim=(1, 2)).reshape(self.num_envs, 1)
        is_standing = self.env.command_manager.is_standing_env.squeeze(1)
        total_penalty[~is_standing] = 0
        total_penalty[is_standing] -= total_penalty[is_standing].mean(0)
        return - total_penalty
    
    def debug_draw(self):
        # draw contact forces on each of the body (orange)
        contact_forces = self.contact_sensor.data.net_forces_w_history.mean(1)[:, self.body_ids]
        body_pos_w = self.asset.data.body_pos_w[:, self.articulation_body_ids]
        is_standing = self.env.command_manager.is_standing_env.squeeze(1)
        self.env.debug_draw.vector(
            body_pos_w[is_standing].view(-1, 3),
            contact_forces[is_standing].view(-1, 3),
            # orange
            color=(1., 0.1, 0.1, 1.),
            size=5.0
        )
    
class is_standing_env(Reward):
    def __init__(self, env, weight: float, enabled: bool = False, clip_range=(-torch.inf, +torch.inf)):
        super().__init__(env, weight, enabled, clip_range)
    
    def compute(self) -> torch.Tensor:
        return self.env.command_manager.is_standing_env.reshape(self.num_envs, 1)

class stance_width(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, clip_range=(-torch.inf, +torch.inf), target_width=0.15):
        """penalize stance width smaller than target_width"""
        super().__init__(env, weight, enabled, clip_range)
        self.asset: Articulation = self.env.scene["robot"]
        self.target_width = target_width
    
    def compute(self) -> torch.Tensor:
        front_width = self.asset.data.feet_pos_b[:, [0, 1], 0].diff(dim=1).norm(dim=1, keepdim=True)
        back_width = self.asset.data.feet_pos_b[:, [2, 3], 0].diff(dim=1).norm(dim=1, keepdim=True)
        width = torch.cat([front_width, back_width], dim=1)
        return -(self.target_width - width).clamp_min(0.).sum(1, keepdim=True)
    

class stand_up_height(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, clip_range=(-torch.inf, +torch.inf)):
        super().__init__(env, weight, enabled, clip_range)
        self.asset: Articulation = self.env.scene["robot"]
    
    def compute(self) -> torch.Tensor:
        return (
            self.asset.data.root_pos_w[:, 2].clamp(max=0.7).square()
        ).unsqueeze(-1)

class stand_up_orientation(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, clip_range=(-torch.inf, +torch.inf)):
        super().__init__(env, weight, enabled, clip_range)
        self.asset: Articulation = self.env.scene["robot"]
    
    def compute(self) -> torch.Tensor:
        return (
            -self.asset.data.projected_gravity_b[:, 0].clamp_min_(-0.8)
        ).unsqueeze(-1)


class joint_vel_l2(Reward):
    def __init__(self, env, joint_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _ = self.asset.find_joints(joint_names)
    
    def compute(self) -> torch.Tensor:
        return - self.asset.data.joint_vel[:, self.joint_ids].square().sum(1, True)

class feet_swing_height(Reward):
    def __init__(self, env, target_height: float, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.target_height = target_height
        self.feet_ids = self.asset.find_bodies(".*foot.*")[0]

    def update(self):
        self.feet_pos_b = quat_rotate_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.asset.data.body_pos_w[:, self.feet_ids] - self.asset.data.root_pos_w.unsqueeze(1)
        )
        self.feet_vel_b = quat_rotate_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.asset.data.body_lin_vel_w[:, self.feet_ids]
        )

    def compute(self) -> torch.Tensor:
        hight_error = (self.asset.data.feet_height - self.target_height).abs()
        lateral_speed = (
            self.feet_vel_b[:, :, :2].square().sum(-1)
            + self.asset.data.body_ang_vel_w[:, self.feet_ids, 2].square()
        )
        return - (hight_error * lateral_speed).sum(1, keepdim=True)


class head_clearance(Reward):
    def __init__(self, env, target_height: float, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.target_height = target_height
        self.asset: Articulation = self.env.scene["robot"]
        self.head_height: torch.Tensor = self.asset.data.head_height

    def compute(self) -> torch.Tensor:
        return (self.head_height - self.target_height).clamp_max(0.)


class com_support(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_ids = self.asset.find_bodies(".*foot")[0]

    def compute(self) -> torch.Tensor:
        feet_center = self.asset.data.body_pos_w[:, self.feet_ids].mean(1)
        error = (self.asset.data.root_pos_w[:, :2] - feet_center[:, :2]).square().sum(1, keepdim=True)
        return torch.exp(- error / 0.2)


class com_linvel_exp(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        with torch.device(self.device):
            self.com_pos_w = torch.zeros(self.num_envs, 3)
            self.com_pos_w_last = torch.zeros(self.num_envs, 3)
            self.com_vel_w = torch.zeros(self.num_envs, 3)
        self.feet_ids = self.asset.find_bodies(".*foot")[0]

    def update(self):
        self.com_pos_w_last[:] = self.com_pos_w
        self.com_pos_w[:] = self.asset.data.body_pos_w[:, self.feet_ids].mean(1)
        self.com_vel_w[:] = (self.com_pos_w - self.com_pos_w_last) / self.env.step_dt
        self.com_vel_w[:, 2] = 0.

    def compute(self) -> torch.Tensor:
        com_linvel_b = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.com_vel_w
        )
        error = (com_linvel_b[:, :2] - self.env.command_manager.command_linvel[:, :2]).square().sum(1, keepdim=True)
        return torch.exp(- error / 0.25)

class joint_limits(Reward):
    def __init__(self, env, joint_names: str, offset: float, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids = self.asset.find_joints(joint_names)[0]
        self.joint_limits = self.asset.data.joint_limits[:, self.joint_ids].clone()
        self.joint_limits_max = self.joint_limits[:, :, 1] - offset
        self.joint_limits_min = self.joint_limits[:, :, 0] + offset

    def compute(self) -> torch.Tensor:
        joint_pos = self.asset.data.joint_pos[:, self.joint_ids]
        violation_min = (joint_pos - self.joint_limits_min).clamp_max(0.)
        violation_max = (self.joint_limits_max - joint_pos).clamp_max(0.)
        return (violation_min + violation_max).sum(1, keepdim=True)


def normalize(x: torch.Tensor):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

def shaped_error(error: torch.Tensor):
    """
    Shaped error for reward shaping.
    """
    return torch.maximum(error.abs(), error.square())