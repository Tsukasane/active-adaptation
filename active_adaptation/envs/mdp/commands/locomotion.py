from math import pi
import torch
import torch.distributions as D
import math
import warp as wp
from typing import Sequence, TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
import omni.isaac.lab.utils.math as math_utils
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, MultiUniform
from active_adaptation.utils.helpers import batchify
from omni.isaac.lab.utils.math import quat_apply_yaw, yaw_quat
from tensordict import TensorDict
from .base import Command
from ..observations import _initialize_warp_meshes, raycast_mesh

if TYPE_CHECKING:
    from active_adaptation.envs.base import Env

class Command1(Command):
    """
    Generate commands of liner velocity in body frame, angular velocity, and base height.
    """
    command_dim: int=4 # linvel_xy, angvel_z, base_height

    def __init__(
        self,
        env,
        speed_range=(0.5, 2.0),
        angvel_range=(-1.0, 1.0),
        base_height_range=(0.2, 0.4),
        resample_interval: int = 300,
        resample_prob: float = 0.75,
        stand_prob=0.2,
    ):
        super().__init__(env)
        self.robot: Articulation = env.scene["robot"]
        self.height_scanner = env.scene.sensors.get("height_scanner", None)
        self.speed_range = speed_range
        self.base_height_range = base_height_range
        self.angvel_range = angvel_range

        self.resample_interval = resample_interval
        self.resample_prob = resample_prob
        self.stand_prob = stand_prob

        with torch.device(env.device):
            self.target_yaw = torch.zeros(env.num_envs)
            self._target_base_height = torch.zeros(env.num_envs, 1)
            self._integrated_yaw = torch.zeros(env.num_envs, 3)

            self._command_direction = torch.zeros(env.num_envs, 3)
            self.command_speed = torch.zeros(env.num_envs, 1)
            self.command_linvel = torch.zeros(env.num_envs, 3)

            self._command_stand = torch.zeros(env.num_envs, 1, dtype=bool)
            self.command_angvel_yaw = torch.zeros(env.num_envs)
            
            self.command = torch.zeros(env.num_envs, self.command_dim)
        self.is_standing_env = self._command_stand
        self._command_heading = self._integrated_yaw

    def reset(self, env_ids: torch.Tensor):
        self.sample_vel_command(env_ids)
        self.sample_yaw_command(env_ids)

    def update(self):
        interval_reached = (self.env.episode_length_buf + 1) % self.resample_interval == 0
        resample_vel = interval_reached & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        resample_yaw = interval_reached & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        self.sample_vel_command(resample_vel.nonzero().squeeze(-1))
        self.sample_yaw_command(resample_yaw.nonzero().squeeze(-1))
        
        yaw_diff = self.target_yaw - self.robot.data.heading_w
        self.command_angvel_yaw[:] = math_utils.wrap_to_pi(yaw_diff).clamp(*self.angvel_range)

        command_speed = self.command_speed
        if self.height_scanner is not None:
            height_scan_z: torch.Tensor = self.height_scanner.data.ray_hits_w[:, :, [2]]
            near_stairs = height_scan_z.max(1)[0] - height_scan_z.min(1)[0] > 0.2
            assert near_stairs.shape == command_speed.shape
            command_speed = torch.where(
                near_stairs,
                command_speed.clamp(max=1.0),
                command_speed
            )
        self.is_standing_env[:] = torch.logical_and(
            self.command_angvel_yaw.unsqueeze(1).abs() < 0.1,
            self.command_speed < 0.2
        )
        self.command_linvel[:, :2] = command_speed * self._command_direction[:, :2]
        
        self.command[:, :2] = self.command_linvel[:, :2]
        self.command[:, 2] = self.command_angvel_yaw

    def sample_vel_command(self, env_ids: torch.Tensor):
        a = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        stand = torch.rand(len(env_ids), device=self.device) < self.stand_prob
        speed = torch.zeros(len(env_ids), device=self.device).uniform_(*self.speed_range)
        speed = speed * (~stand).float()
        
        self.command_speed[env_ids] = speed.unsqueeze(1)
        self._command_direction[env_ids, 0] = a.cos()
        self._command_direction[env_ids, 1] = a.sin() * 0.6
    
    def sample_yaw_command(self, env_ids: torch.Tensor):
        yaw = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        self.target_yaw[env_ids] = yaw
        self._integrated_yaw[env_ids, 0] = yaw.cos()
        self._integrated_yaw[env_ids, 1] = yaw.sin()
        
        self._target_base_height[env_ids] = sample_uniform(
            env_ids.shape, *self.base_height_range, self.env.device
        ).unsqueeze(1)
        self.command[:, 3:4] = self._target_base_height


class Command2(Command):
    def __init__(
        self, 
        env, 
        linvel_x_range=(-1.0, 1.0),
        linvel_y_range=(-1.0, 1.0),
        angvel_range=(-1, 1),
        yaw_stiffness_range=(0.5, 0.6),
        use_stiffness_ratio: float = 0.5,
        aux_input_range=(0.2, 0.4), 
        resample_interval: int = 300, 
        resample_prob: float = 0.75, 
        stand_prob=0.2,
        target_yaw_range=(0, torch.pi * 2),
        adaptive: bool = False,
        body_name: str = None,
        teleop: bool = False,
    ):
        super().__init__(env, teleop=teleop)
        self.robot: Articulation = env.scene["robot"]
        self.linvel_x_range = linvel_x_range
        self.linvel_y_range = linvel_y_range
        self.angvel_range = angvel_range
        self.use_stiffness_ratio = use_stiffness_ratio
        self.yaw_stiffness_range = yaw_stiffness_range
        self.aux_input_range = aux_input_range
        self.resample_interval = resample_interval
        self.resample_prob = resample_prob
        self.stand_prob = stand_prob
        self.adaptive = adaptive

        if self.adaptive:
            self.ground_mesh = _initialize_warp_meshes("/World/ground", "cuda")

        if body_name is not None:
            self.body_id = self.asset.find_bodies(body_name)[0][0]
            self.FWDVEC = torch.tensor([1., 0., 0.], device=self.device).expand(self.num_envs, 3)
        else:
            self.body_id = None

        with torch.device(self.device):
            if all(isinstance(r, Sequence) for r in target_yaw_range):
                self.target_yaw_dist = MultiUniform(torch.tensor(target_yaw_range))
            else:
                self.target_yaw_dist = D.Uniform(*torch.tensor(target_yaw_range))

            self.command = torch.zeros(self.num_envs, 4)
            self.target_yaw = torch.zeros(self.num_envs)
            self.yaw_stiffness = torch.zeros(self.num_envs)
            self.use_stiffness = torch.zeros(self.num_envs, dtype=bool)
            self.fixed_yaw_speed = torch.zeros(self.num_envs)

            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)

            self.command_speed = torch.zeros(self.num_envs, 1)
            self._target_direction = torch.zeros(self.num_envs, 3)
            self._target_linvel = torch.zeros(self.num_envs, 3)
            self.command_linvel = torch.zeros(self.num_envs, 3)
            self.command_linvel_w = torch.zeros(self.num_envs, 3)
            self.command_angvel = torch.zeros(self.num_envs)

            self.aux_input = torch.zeros(self.num_envs, 1)

            self._cum_error = torch.zeros(self.num_envs, 2)
            self._cum_linvel_error = self._cum_error[:, 0].unsqueeze(1)
            self._cum_angvel_error = self._cum_error[:, 1].unsqueeze(1)
            
        self._decay = 0.999
        self._sum_error = torch.tensor(0.0, device=self.device)
        self._count = torch.tensor(0.0, device=self.device)
        self._avg_error = torch.tensor(0.0, device=self.device)

        if self.teleop:
            self.key_mappings_pos = {
                "W": torch.tensor([self.linvel_x_range[1], 0., 0.], device=self.device),
                "S": torch.tensor([self.linvel_x_range[0], 0., 0.], device=self.device),
                "A": torch.tensor([0., self.linvel_y_range[1], 0.], device=self.device),
                "D": torch.tensor([0., self.linvel_y_range[0], 0.], device=self.device),
            }
        
    def reset(self, env_ids, reward_stats = None):
        self.command[env_ids] = 0.
        if not self.teleop:
            self.sample_vel_command(env_ids)
            self.sample_yaw_command(env_ids)
        self._cum_linvel_error[env_ids] = 0.
        self._cum_angvel_error[env_ids] = 0.
        self.env.extra["stats/avg_error"] = self._avg_error.item()
    
    def update(self):
        if self.body_id is not None:
            self.body_quat_w = self.asset.data.body_quat_w[:, self.body_id]
            forward_w = quat_rotate(self.body_quat_w, self.FWDVEC)
            self.body_heading_w = torch.atan2(forward_w[:, 1], forward_w[:, 0])
        else:
            self.body_heading_w = self.asset.data.heading_w
        if self.teleop:
            command_linvel_target = torch.tensor([0., 0., 0.], device=self.device)
            for key, vec in self.key_mappings_pos.items():
                if self.key_pressed[key]:
                    command_linvel_target.add_(vec)
            if not self.key_pressed["LEFT_SHIFT"]:
                command_linvel_target *= 0.6
            self.command_linvel.lerp_(command_linvel_target, 0.5)
        else:
            interval_reached = (self.env.episode_length_buf + 1) % self.resample_interval == 0
            resample_vel = interval_reached & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
            resample_yaw = interval_reached & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
            self.sample_vel_command(resample_vel.nonzero().squeeze(-1))
            self.sample_yaw_command(resample_yaw.nonzero().squeeze(-1))

        self.target_yaw[~self.use_stiffness] = self.body_heading_w[~self.use_stiffness]
        yaw_diff = self.target_yaw - self.body_heading_w
        command_yaw_speed = torch.clamp(
            self.yaw_stiffness * math_utils.wrap_to_pi(yaw_diff), 
            min=self.angvel_range[0],
            max=self.angvel_range[1]
        )
        self.command_angvel[:] = torch.where(self.use_stiffness, command_yaw_speed, self.fixed_yaw_speed)

        # this is used for terminating episodes where the robot is inactive due to whatever reason
        linvel_error = (self.robot.data.root_lin_vel_w[:, :2] - self.command_linvel_w[:, :2]).norm(dim=-1, keepdim=True)
        angvel_error = (self.command_angvel - self.robot.data.root_ang_vel_w[:, 2]).abs().unsqueeze(1)

        self._sum_error.add_(linvel_error.sum()).mul_(self._decay)
        self._count.add_(self.num_envs).mul_(self._decay)
        self._avg_error.copy_(self._sum_error / self._count)

        if self.adaptive:
            self.ray_start_w = self.robot.data.root_pos_w + torch.tensor([0., 0., -0.2], device=self.device)
            ray_direction = quat_rotate(self.asset.data.root_quat_w, self._target_direction)
            ray_direction[:, 2] = 0.
            self.ray_hit_w = raycast_mesh(self.ray_start_w, ray_direction, max_dist=2, mesh=self.ground_mesh)[0]
            distance_to_obstacle = (self.ray_hit_w - self.ray_start_w).norm(dim=-1, keepdim=True).nan_to_num(2.0)
            self.close_to_obstacle = distance_to_obstacle < 0.75
            fast = self.command_speed > 1.4
            target_linvel = torch.where((self.close_to_obstacle & fast), self._target_linvel / 2, self._target_linvel)
        else:
            target_linvel = self._target_linvel

        self._cum_linvel_error.mul_(0.98).add_(linvel_error * self.env.step_dt)
        self._cum_angvel_error.mul_(0.98).add_(angvel_error * self.env.step_dt)
        self.command_linvel[:] = self.command_linvel + clamp_norm((target_linvel - self.command_linvel) * 0.1, max=0.1)

        self.command_linvel_w[:] = quat_apply_yaw(self.robot.data.root_quat_w, self.command_linvel)
        self.command[:, :2] = self.command_linvel[:, :2]
        self.command[:, 2] = self.command_angvel
        self.command[:, 3] = self.aux_input.squeeze(1)
        # self.command[:, :2] = torch.tensor([1.0, 0.], device=self.device)
        self.is_standing_env[:, 0] = (
            (self.command_linvel.norm(dim=-1) < 0.1)
            & (self.command_angvel < 0.1)
        )

    
    def sample_vel_command(self, env_ids: torch.Tensor):
        linvel = torch.zeros(len(env_ids), 2, device=self.device)
        linvel[:, 0].uniform_(*self.linvel_x_range)
        linvel[:, 0] = torch.where(
            torch.rand(len(env_ids), device=self.device) < 0.2, 
            linvel[:, 0].abs(), linvel[:, 0]
        )
        linvel[:, 1].uniform_(*self.linvel_y_range)
        speed = linvel.norm(dim=-1, keepdim=True)
        direction = linvel / speed.clamp(1e-6)
        stand = (speed < 0.3) | (torch.rand(len(env_ids), 1, device=self.device) < self.stand_prob)
        speed = speed * (~stand)

        self.command_speed[env_ids] = speed
        self._target_direction[env_ids, :2] = direction
        self._target_linvel[env_ids, :2] = direction * speed

        self.aux_input[env_ids] = sample_uniform(env_ids.shape, *self.aux_input_range, self.device).unsqueeze(1)

    def sample_yaw_command(self, env_ids: torch.Tensor):
        self.target_yaw[env_ids] = self.target_yaw_dist.sample(env_ids.shape)
        self.yaw_stiffness[env_ids] = sample_uniform(env_ids.shape, *self.yaw_stiffness_range, self.device)
        self.use_stiffness[env_ids] = torch.rand(len(env_ids), device=self.device) < self.use_stiffness_ratio
        self.fixed_yaw_speed[env_ids] = sample_uniform(env_ids.shape, *self.angvel_range, self.device) 

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.robot.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            self.command_linvel_w,
            color=(1., 1., 1., 1.)
        )
        self.env.debug_draw.vector(
            self.robot.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            torch.stack([self.target_yaw.cos(), self.target_yaw.sin(), torch.zeros_like(self.target_yaw)], 1),
            color=(.2, .2, 1., 1.)
        )
        zeros = torch.zeros(self.num_envs, 1, device=self.device)
        self.env.debug_draw.vector(
            self.robot.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            torch.stack([zeros, zeros, self._cum_linvel_error], 1),
            color=(.2, 1., .2, 1.)
        )
        self.env.debug_draw.vector(
            self.robot.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
            torch.stack([zeros, zeros, self._cum_angvel_error], 1),
            color=(1., .2, .2, 2.)
        )
        if self.adaptive:
            self.env.debug_draw.vector(
                self.ray_start_w[self.close_to_obstacle.squeeze(1)],
                (self.ray_hit_w - self.ray_start_w)[self.close_to_obstacle.squeeze(1)],
                # self._target_direction,
                color=(1., 1., 0., 1.)
            )
    
    def fliplr(self, command: torch.Tensor) -> torch.Tensor:
        # flip y and yaw velocity
        return command * torch.tensor([1., -1., -1., 1.], device=self.device)


def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low

def sample_quat_yaw(size, yaw_range = (0, torch.pi * 2), device: torch.device = "cpu"):
    yaw = torch.rand(size, device=device).uniform_(*yaw_range)
    quat = torch.cat([
        torch.cos(yaw / 2).unsqueeze(-1),
        torch.zeros_like(yaw).unsqueeze(-1),
        torch.zeros_like(yaw).unsqueeze(-1),
        torch.sin(yaw / 2).unsqueeze(-1),
    ], dim=-1)
    return quat


def quat_from_yaw(yaw: torch.Tensor):
    return torch.cat([
        torch.cos(yaw / 2).unsqueeze(-1),
        torch.zeros_like(yaw).unsqueeze(-1),
        torch.zeros_like(yaw).unsqueeze(-1),
        torch.sin(yaw / 2).unsqueeze(-1),
    ], dim=-1)

@batchify
def yaw_rotate(yaw: torch.Tensor, vec: torch.Tensor):
    yaw_cos = torch.cos(yaw).squeeze(-1)
    yaw_sin = torch.sin(yaw).squeeze(-1)
    return torch.stack([
        yaw_cos * vec[:, 0] - yaw_sin * vec[:, 1],
        yaw_sin * vec[:, 0] + yaw_cos * vec[:, 1],
        vec[:, 2]
    ], 1)


def clamp_norm(x: torch.Tensor, min: float=0., max: float=torch.inf):
    x_norm = x.norm(dim=-1, keepdim=True).clamp(1e-6)
    x = torch.where(x_norm < min, x / x_norm * min, x)
    x = torch.where(x_norm > max, x / x_norm * max, x)
    return x


def quat_to_yaw(quat: torch.Tensor):
    q_w, q_x, q_y, q_z = quat.unbind(-1)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)
    return yaw % (2 * torch.pi)


@wp.func
def sample_uniform_wp(rng: wp.uint32, range: wp.vec2) -> float:
    return wp.randf(rng) * (range[1] - range[0]) + range[0]

vec5f = wp.vec(length=5, dtype=wp.float32)

@wp.kernel
def maybe_sample_force(
    kernel_seed: int,
    const_force_scale: wp.vec3,
    const_force_duration_range: wp.vec2,
    impulse_force_scale: wp.vec3,
    impulse_force_duration_range: wp.vec2,
    force_offset_scale: wp.vec3,
    sample_force: wp.array(dtype=wp.bool),
    force_type: wp.array(dtype=wp.int32),
    const_force: wp.array(dtype=vec5f),
    impulse_force: wp.array(dtype=vec5f),
    force_offset: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    if sample_force[tid]:
        if force_type[tid] == 0:
            pass
        elif (force_type[tid] == 1) and (const_force[tid][4] > const_force[tid][3]):
            rng = wp.rand_init(kernel_seed, tid)
            xy = wp.cw_mul(wp.sample_unit_cube(rng), const_force_scale)
            duration = sample_uniform_wp(rng, const_force_duration_range)
            const_force[tid] = vec5f(xy[0], xy[1], 0., duration, 0.)
        elif (force_type[tid] == 2) and (impulse_force[tid][4] > impulse_force[tid][3]):
            rng = wp.rand_init(kernel_seed, tid)
            xy = wp.cw_mul(wp.sample_unit_cube(rng), impulse_force_scale)
            duration = sample_uniform_wp(rng, impulse_force_duration_range)
            xy = xy / duration
            impulse_force[tid] = vec5f(xy[0], xy[1], 0., duration, 0.)
        
        rng = wp.rand_init(kernel_seed, tid + 1)
        force_offset[tid] = wp.cw_mul(wp.sample_unit_cube(rng), force_offset_scale)

