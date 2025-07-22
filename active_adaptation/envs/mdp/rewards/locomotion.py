import torch
from typing import TYPE_CHECKING, List
from isaaclab.utils.math import yaw_quat, quat_apply, quat_apply_inverse
import isaaclab.utils.string as string_utils
# from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from active_adaptation.envs.mdp.base import Reward

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor
    from isaaclab.assets import Articulation


class energy_l1(Reward):

    decay: float = 0.99

    def __init__(self, env, weight: float, enabled: bool = True, a={".*": 1.0}):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _, self.a = string_utils.resolve_matching_names_values(
            dict(a), self.asset.joint_names
        )
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.a = torch.tensor(self.a, device=self.device)

        self.power = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)
        self.energy = torch.zeros(
            self.num_envs, len(self.joint_ids), device=self.device
        )
        self.count = torch.zeros(self.num_envs, 1, device=self.device)
        self.asset.data.energy_ema = torch.zeros(
            self.num_envs, len(self.joint_ids), device=self.device
        )

    def reset(self, env_ids):
        self.energy[env_ids] = 0.0
        self.count[env_ids] = 0.0

    def update(self):
        torques = self.asset.data.applied_torque[:, self.joint_ids]
        joint_vel = self.asset.data.joint_vel[:, self.joint_ids]
        self.power[:] = (torques * joint_vel).abs()

        self.energy.add_(self.power).mul_(self.decay)
        self.count.add_(1.0).mul_(self.decay)
        self.asset.data.energy_ema[:] = self.energy / self.count

    def compute(self) -> torch.Tensor:
        return -(self.power * self.a).sum(1, keepdim=True)


class energy_dist_lr(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.energy_ema: Articulation = self.asset.data.energy_ema
        self.left_joint_ids = self.asset.find_joints("[F,R]L_.*_joint")[0]
        self.right_joint_ids = self.asset.find_joints("[F,R]R_.*_joint")[0]

    def compute(self) -> torch.Tensor:
        energy_left = self.energy_ema[:, self.left_joint_ids]
        energy_right = self.energy_ema[:, self.right_joint_ids]
        return -(energy_left - energy_right).square().sum(1, keepdim=True)


class energy_dist_fb(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.energy_ema: Articulation = self.asset.data.energy_ema
        self.front_joint_ids = self.asset.find_joints("F[L,R]_.*_joint")[0]
        self.rear_joint_ids = self.asset.find_joints("R[L,R]_.*_joint")[0]

    def compute(self) -> torch.Tensor:
        energy_front = self.energy_ema[:, self.front_joint_ids]
        energy_rear = self.energy_ema[:, self.rear_joint_ids]
        return -(energy_front - energy_rear).square().sum(1, keepdim=True)


class joint_torques_l2(Reward):
    def __init__(
        self, env, weight: float, enabled: bool = True, joint_names: str = ".*"
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids = self.asset.find_joints(joint_names)[0]
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)

    def compute(self) -> torch.Tensor:
        return (
            -self.asset.data.applied_torque[:, self.joint_ids]
            .square()
            .sum(1, keepdim=True)
        )


class joint_torques_berhu(Reward):
    """
    Berhu loss:
    L(x) = |x|, if |x| < c
    L(x) = (x^2 + c^2) / (2 * c), if |x| >= c
    """
    def __init__(self, env, c: float,weight: float, enabled: bool = True, joint_names: str = ".*"):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids = self.asset.find_joints(joint_names)[0]
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.c = c
    
    def compute(self) -> torch.Tensor:
        applied_torques = self.asset.data.applied_torque[:, self.joint_ids]
        return - torch.where(
            applied_torques < self.c,
            applied_torques,
            (applied_torques.square() + self.c**2) / (2 * self.c)
        ).sum(1, keepdim=True)


class undesired_contact(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)

        self.undesired_contact_cum = self.asset.data.undesired_contact_cum = (
            torch.zeros(self.num_envs, 1, device=self.device)
        )

        print(f"Penalizing contacts on {self.body_names}.")

    def reset(self, env_ids):
        self.undesired_contact_cum[env_ids] = 0.0

    def update(self):
        contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.0
        self.undesired_contact = -contact.float().sum(1, keepdim=True)
        self.undesired_contact_cum.add_(self.undesired_contact)

    def compute(self) -> torch.Tensor:
        return self.undesired_contact

    # def debug_draw(self):
    #     self.env.debug_draw.point(
    #         # self.contact_sensor.data.pos_w[:, self.body_ids],
    #         self.asset.data.body_pos_w[:, self.articulation_body_ids],
    #         color=(1., .6, .4, 1.),
    #         size=20,
    #     )


class feet_swing_height(Reward):
    def __init__(self, env, target_height: float, body_names: str | List[str], weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.target_height = target_height
        self.feet_ids = self.asset.find_bodies(body_names)[0]
        self.contact_forces: ContactSensor = self.env.scene["contact_forces"]
        self.feet_contact_ids = self.contact_forces.find_bodies(body_names)[0]

    def update(self):
        self.feet_pos_b = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.asset.data.body_pos_w[:, self.feet_ids]
            - self.asset.data.root_pos_w.unsqueeze(1),
        )
        self.feet_vel_b = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.asset.data.body_lin_vel_w[:, self.feet_ids],
        )
        if not hasattr(self.asset, "feet_height"):
            self.feet_height = self.asset.data.body_pos_w[:, self.feet_ids, 2]
        else:
            self.feet_height = self.asset.data.feet_height

    def compute(self) -> torch.Tensor:
        hight_error = (self.feet_height - self.target_height).abs()
        lateral_speed = (
            self.feet_vel_b[:, :, :2].square().sum(-1)
            + self.asset.data.body_ang_vel_w[:, self.feet_ids, 2].square()
        )
        # shape: [num_envs, num_feet]
        in_contact = self.contact_forces.data.net_forces_w[:, self.feet_contact_ids].norm(dim=2) > 0.5
        # shape: [num_envs, num_feet_contact]
        return -(hight_error * lateral_speed * (~in_contact)).sum(1, keepdim=True)


class head_clearance(Reward):
    def __init__(self, env, target_height: float, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.target_height = target_height
        self.asset: Articulation = self.env.scene["robot"]
        self.head_height: torch.Tensor = self.asset.data.head_height

    def compute(self) -> torch.Tensor:
        return (self.head_height - self.target_height).clamp_max(0.0)


class com_support(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_ids = self.asset.find_bodies(".*foot")[0]

    def compute(self) -> torch.Tensor:
        feet_center = self.asset.data.body_pos_w[:, self.feet_ids].mean(1)
        error = (
            (self.asset.data.root_pos_w[:, :2] - feet_center[:, :2])
            .square()
            .sum(1, keepdim=True)
        )
        return torch.exp(-error / 0.2)


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
        self.com_vel_w[:, 2] = 0.0

    def compute(self) -> torch.Tensor:
        com_linvel_b = quat_apply_inverse(self.asset.data.root_quat_w, self.com_vel_w)
        error = (
            (com_linvel_b[:, :2] - self.env.command_manager.command_linvel[:, :2])
            .square()
            .sum(1, keepdim=True)
        )
        return torch.exp(-error / 0.25)


class joint_limits(Reward):
    def __init__(
        self, env, joint_names: str, offset: float, weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids = self.asset.find_joints(joint_names)[0]
        self.joint_limits = self.asset.data.joint_limits[:, self.joint_ids].clone()
        self.joint_limits_max = self.joint_limits[:, :, 1] - offset
        self.joint_limits_min = self.joint_limits[:, :, 0] + offset

    def compute(self) -> torch.Tensor:
        joint_pos = self.asset.data.joint_pos[:, self.joint_ids]
        violation_min = (joint_pos - self.joint_limits_min).clamp_max(0.0)
        violation_max = (self.joint_limits_max - joint_pos).clamp_max(0.0)
        return (violation_min + violation_max).sum(1, keepdim=True)


class step_vel(Reward):
    def __init__(self, env, weight, enabled=True):
        super().__init__(env, weight, enabled,)
        self.asset: Articulation = self.env.scene["robot"]
        self.prev_root_pos_w = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.last_impact_time = torch.zeros(self.num_envs, 4, 1, device=self.device)
        self.command_manager: Impedance = self.env.command_manager

    def reset(self, env_ids):
        self.prev_root_pos_w[env_ids] = self.asset.data.root_pos_w[env_ids].unsqueeze(1)
        self.last_impact_time[env_ids] = 0.0

    def update(self):
        root_pos_w = self.asset.data.root_pos_w.unsqueeze(1)
        t = self.env.episode_length_buf.reshape(-1, 1, 1) * self.env.step_dt
        self.step_vel = (root_pos_w - self.prev_root_pos_w) / (
            t - self.last_impact_time
        )
        self.step_vel = torch.nan_to_num(self.step_vel, nan=0.0, posinf=0.0, neginf=0.0)
        self.prev_root_pos_w = torch.where(
            self.asset.impact.unsqueeze(-1), root_pos_w, self.prev_root_pos_w
        )
        self.last_impact_time = torch.where(
            self.asset.impact.unsqueeze(-1), t, self.last_impact_time
        )

    def compute(self):
        error_l1 = (
            self.command_manager.command_linvel_w[:, :2].unsqueeze(1)
            - self.step_vel[:, :, :2]
        ).norm(dim=-1)
        r = torch.exp(-error_l1) * self.asset.impact
        return r.sum(1, keepdim=True)


class oscillator(Reward):
    def __init__(
        self,
        env,
        feet_names: str = ".*_foot",
        omega_range=(2., 2.),
        margin: float = 0.0,
        weight=1.0,
        enabled=True,
    ):
        super().__init__(env, weight, enabled)
        self.margin = margin
        self.target_swing_height = 0.08

        self.asset: Articulation = self.env.scene["robot"]
        self.art_feet_ids = self.asset.find_bodies(feet_names)[0]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.command_manager: Command2 = self.env.command_manager

        self.feet_ids, feet_names = self.contact_sensor.find_bodies(feet_names)
        self.mass = self.asset.data.default_mass[0].sum().to(self.device)
        self.gravity = self.mass * 9.81

        # if not hasattr(self.asset, "phi"):
        #     self.asset.phi = torch.zeros(self.num_envs, 4, device=self.device)
        #     self.asset.phi_dot = torch.zeros(self.num_envs, 4, device=self.device)
        # self.asset.phi[:, 0] = torch.pi
        # self.asset.phi[:, 3] = torch.pi
        self.grf_substep = torch.zeros(
            self.num_envs,
            self.env.decimation,
            len(self.feet_ids),
            device=self.device,
        )
        self.omega_range = omega_range
        self.omega = torch.zeros(self.num_envs, 1, device=self.device)
        self.omega.uniform_(*self.omega_range).mul_(torch.pi)

        self.rest_target = torch.pi * 3 / 2
        self.keep_steping = torch.zeros(
            self.num_envs, 1, dtype=bool, device=self.device
        )

    # def reset(self, env_ids):
    #     self.keep_steping[env_ids] = (torch.rand(len(env_ids), 1, device=self.device) < 0.)
    #     self.asset.phi_dot[env_ids] = self.omega[env_ids]

    def post_step(self, substep):
        grf = self.contact_sensor.data.net_forces_w[:, self.feet_ids].norm(dim=-1)
        grf += self.asset._external_force_b[:, self.art_feet_ids].norm(dim=-1)
        self.grf_substep[:, substep] = grf

    def update(self):
        self.grf = self.grf_substep.mean(1) / self.gravity
        # inp = (
        #     (~self.command_manager.is_standing_env)
        #     | self.keep_steping
        # )
        # correction = self.trot(self.asset.phi, self.asset.phi_dot)
        # phi_dot = torch.where(
        #     inp,
        #     self.omega + correction,
        #     self.stand(self.asset.phi, self.asset.phi_dot),
        # )
        
        # self.asset.phi_dot = phi_dot
        # self.asset.phi += self.asset.phi_dot * self.env.step_dt
        # self.asset.phi = torch.where((self.asset.phi > torch.pi * 2).all(1, True), self.asset.phi - torch.pi * 2, self.asset.phi)

    def compute(self):
        phi_sin = self.asset.phi.sin()
        feet_height = self.asset.data.feet_height.clamp_max(self.target_swing_height)
        r = (
            (feet_height - self.grf.clamp_max(0.4))
            * phi_sin
            * (phi_sin.abs() > self.margin)
        )
        return r.sum(1, True)

    def stand(self, phi: torch.Tensor, phi_dot: torch.Tensor,):
        two_pi = torch.pi * 2
        target = self.rest_target
        dt = self.env.step_dt
        a = ((phi % two_pi) < target - 1e-4) & (((phi + phi_dot * dt) % two_pi) > target + 1e-4)
        b = ((phi % two_pi) - target).abs() < 1e-4
        phi_dot = torch.where(a, (((target - phi) % two_pi) / dt), phi_dot)
        return phi_dot * (~b)

    def trot(self, phi: torch.Tensor, phi_dot: torch.Tensor):
        phi_dot = torch.zeros_like(phi)
        phi_dot[:, 0] = (phi[:, 3] - phi[:, 0]) + (phi[:, 1] + torch.pi - phi[:, 0]) 
        phi_dot[:, 1] = (phi[:, 2] - phi[:, 1]) + (phi[:, 0] - torch.pi - phi[:, 1]) 
        phi_dot[:, 2] = (phi[:, 1] - phi[:, 2]) + (phi[:, 0] - torch.pi - phi[:, 2])
        phi_dot[:, 3] = (phi[:, 0] - phi[:, 3]) + (phi[:, 1] + torch.pi - phi[:, 3])
        return phi_dot


class gait(Reward):
    def __init__(self, env, weight, enabled=True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager: Command2 = self.env.command_manager
        self.phi: torch.Tensor = self.asset.phi

    def compute(self):
        fast = self.command_manager.command_speed > 1.6
        r_gallop = (self.phi[:, 0] - self.phi[:, 1]).square() + (
            self.phi[:, 2] - self.phi[:, 3]
        ).square()
        r_trot = (self.phi[:, 0] - self.phi[:, 3]).square() + (
            self.phi[:, 1] - self.phi[:, 2]
        ).square()
        r = torch.where(fast, r_gallop.unsqueeze(1), r_trot.unsqueeze(1))
        return -r


class quad_leg_swing(Reward):
    def __init__(self, env, weight, feet_names: str = ".*_foot", enabled=True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.feet_ids = self.asset.find_bodies(feet_names)[0]
        self.feet_ids_ = self.contact_sensor.find_bodies(feet_names)[0]
        self.grf_substep = torch.zeros(
            self.num_envs, self.env.decimation, 4, device=self.device
        )
        self.command_manager: Command2 = self.env.command_manager

    def post_step(self, substep):
        grf = self.contact_sensor.data.net_forces_w[:, self.feet_ids_].norm(dim=-1)
        self.grf_substep[:, substep] = grf

    def update(self):
        feet_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.feet_ids]
        root_lin_vel_w = self.asset.data.root_lin_vel_w
        self.feet_height = self.asset.data.body_pos_w[:, self.feet_ids, 2]
        self.dot = (feet_lin_vel_w * normalize(root_lin_vel_w).unsqueeze(1)).sum(
            -1
        )  # [num_envs, 4]
        self.swinging = self.grf_substep.mean(1) < 0.1

    def compute(self):
        r = self.dot.clamp(0.05, 0.5) + self.feet_height.clamp_max(0.06)
        r = torch.where(
            self.command_manager.is_standing_env,
            -self.swinging.sum(1, True),
            (r * self.swinging).max(1, True).values,
        )
        return r

    def debug_draw(self):
        feet_pos_w = self.asset.data.body_pos_w[:, self.feet_ids]
        swing_feet_pos_w = feet_pos_w[self.swinging]
        self.env.debug_draw.point(
            swing_feet_pos_w, color=(1.0, 0.0, 0.0, 1.0), size=15.0
        )


class pos_tracking(Reward):
    def __init__(self, env, weight, enabled = True):
        super().__init__(env, weight, enabled,)
        self.command_manager: Command3 = self.env.command_manager
        self.asset: Articulation = self.env.scene["robot"]

    def compute(self):
        diff = self.command_manager.des_key_pos_w - self.command_manager.key_pos_w
        error_l2 = diff.square().sum(-1, True)
        return torch.exp(-error_l2 / 0.25).mean(1)


class yaw_tracking(Reward):
    def __init__(self, env, weight, enabled = True):
        super().__init__(env, weight, enabled,)
        self.command_manager: Command3 = self.env.command_manager
        self.asset: Articulation = self.env.scene["robot"]
    
    def compute(self):
        return torch.cos(self.asset.data.heading_w.unsqueeze(1) - self.command_manager.des_yaw_w)


class vel_tracking(Reward):
    def __init__(self, env, weight, enabled = True):
        super().__init__(env, weight, enabled,)
        self.command_manager: Command3 = self.env.command_manager
        self.asset: Articulation = self.env.scene["robot"]
    
    def compute(self):
        diff = self.command_manager.des_key_pos_w - self.command_manager.key_pos_w
        target_vel_z = 2.0 * diff[:, 0, 2]
        r = (self.command_manager.key_vel_w[:, 0, 2] - target_vel_z) * (target_vel_z > 0.)
        return r.unsqueeze(1)

class vel_xy_tracking(Reward):
    def __init__(self, env, weight, enabled = True):
        super().__init__(env, weight, enabled,)
        self.command_manager: Command3 = self.env.command_manager
        self.asset: Articulation = self.env.scene["robot"]
    
    def compute(self):
        diff = self.command_manager.des_vel_w - self.asset.data.root_lin_vel_w
        error_l2 = diff[:, :2].square().sum(-1, True)
        return torch.exp(- error_l2 / 0.25)


def is_expr(expr):
    if isinstance(expr, str):
        return True
    else:
        return all(isinstance(x, str) for x in expr)


class joint_deviation_l1(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, joint_names: str=".*"):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        if is_expr(joint_names):
            self.joint_ids = self.asset.find_joints(joint_names)[0]
            self.joint_weights = None
        else:
            self.joint_ids, _, self.joint_weights = string_utils.resolve_matching_names_values(joint_names, self.asset.joint_names)
            self.joint_weights = torch.tensor(self.joint_weights, device=self.device)
            assert torch.all(self.joint_weights > 0)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids].clone()
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
    
    def compute(self) -> torch.Tensor:
        dev = self.asset.data.joint_pos[:, self.joint_ids] - self.default_joint_pos
        if self.joint_weights is not None:
            dev = dev * self.joint_weights
        return - dev.abs().sum(1, True)


class joint_deviation_l2(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, joint_names: str=".*"):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        if is_expr(joint_names):
            self.joint_ids = self.asset.find_joints(joint_names)[0]
            self.joint_weights = None
        else:
            self.joint_ids, _, self.joint_weights = string_utils.resolve_matching_names_values(joint_names, self.asset.joint_names)
            self.joint_weights = torch.tensor(self.joint_weights, device=self.device)
            assert torch.all(self.joint_weights > 0)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids].clone()
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
    
    def compute(self) -> torch.Tensor:
        dev = self.asset.data.joint_pos[:, self.joint_ids] - self.default_joint_pos
        if self.joint_weights is not None:
            dev = dev * self.joint_weights
        return - dev.square().sum(1, True)


class pitch_exp(Reward):
    def compute(self):
        error = self.env.command_manager.pitch_error_l2
        return torch.exp( -error ) - error

class lin_vel_exp(Reward):
    def compute(self):
        error = self.env.command_manager.lin_vel_error_l2
        return torch.exp( -error / 0.25) - 0.5 * error

class ang_vel_x_exp(Reward):
    def compute(self):
        error = self.env.command_manager.ang_vel_x_error_l2
        return torch.exp( -error / 0.25) - 0.5 * error

class ang_vel_z_exp(Reward):
    def compute(self):
        error = self.env.command_manager.ang_vel_z_error_l2
        return torch.exp( -error / 0.25) - 0.5 * error


class oscillator_biped(Reward):
    def __init__(self, env, weight, enabled=True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.gravity = self.asset.data.default_mass[0].sum().item() * 9.81
        self.contact_forces: ContactSensor = self.env.scene["contact_forces"]
        self.feet_ids = self.contact_forces.find_bodies(".*_ankle_roll_link")[0]

    def compute(self):
        self.sin_phase = self.asset.phi.sin()
        grf = self.contact_forces.data.net_forces_w[:, self.feet_ids].norm(dim=-1)
        r = (-grf/self.gravity * self.sin_phase).clamp_max(0.8).sum(1, True)
        return r

class oscillator_biped_contact(Reward):
    def __init__(self, env, weight, enabled=True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.gravity = self.asset.data.default_mass[0].sum().item() * 9.81
        self.contact_forces: ContactSensor = self.env.scene["contact_forces"]
        self.feet_ids = self.contact_forces.find_bodies(".*_ankle_roll_link")[0]

    def compute(self):
        self.sin_phase = self.asset.phi.sin()
        grf = self.contact_forces.data.net_forces_w[:, self.feet_ids].norm(dim=-1)
        should_contact = self.sin_phase > 0.0 # shape: [N, 2]
        in_contact = grf > 1.0 # shape: [N, 2]
        r = -(should_contact ^ in_contact).float().mean(1, True)
        return r


class quadruped_stand(Reward):
    def __init__(self, env, feet_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_ids = self.asset.find_bodies(feet_names)[0]
        if not hasattr(self.env.command_manager, "is_standing_env"):
            raise ValueError("is_standing_env is not defined in command_manager")
        self.command_manager = self.env.command_manager

    def compute(self):
        jpos_errors = (self.asset.data.joint_pos - self.asset.data.default_joint_pos).abs()
        feet_pos_w = self.asset.data.body_pos_w[:, self.feet_ids]
        feet_pos_b = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            feet_pos_w - self.asset.data.root_pos_w.unsqueeze(1)
        )
        front_symmetry = feet_pos_b[:, [0, 1], 1].sum(dim=1, keepdim=True).abs()
        back_symmetry = feet_pos_b[:, [2, 3], 1].sum(dim=1, keepdim=True).abs()
        cost = - (jpos_errors.sum(dim=1, keepdim=True) + front_symmetry + back_symmetry)

        return cost * self.command_manager.is_standing_env.reshape(self.num_envs, 1)


class lateral_swing_height(Reward):
    def __init__(self, env, feet_names: str, weight: float, enabled=True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_ids = self.asset.find_bodies(feet_names)[0]
        self.target_height = 0.16
        
    def compute(self):
        feet_pos_w = self.asset.data.body_pos_w[:, self.feet_ids]
        feet_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.feet_ids]
        feet_lin_vel_b = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            feet_lin_vel_w
        )
        feet_height_w = feet_pos_w[:, :, 2] - self.env.get_ground_height_at(feet_pos_w) # [N, 4]
        rew = torch.where(
            feet_lin_vel_b[:, :, 1].abs() > 0.4,
            (feet_height_w - self.target_height).clamp_max(0.),
            0.
        )
        return rew.sum(1, True)


class support_polygon(Reward):
    def __init__(self, env, margin: float, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.margin = margin
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.feet_ids, self.feet_names = self.asset.find_bodies(".*_foot")
        self.feet_ids_contact = self.contact_sensor.find_bodies(".*_foot")[0]

        body_masses = self.asset.data.default_mass.to(self.device)
        mass_total = body_masses.sum(dim=1)
        self.body_mass_ratio = body_masses / mass_total.unsqueeze(-1)
    
    def update(self):
        # self.com_pos_w = (self.asset.data.body_com_pos_w * self.body_mass_ratio.unsqueeze(-1)).sum(dim=1)
        self.com_pos_w = self.asset.data.root_com_pos_w
        self.feet_pos_w = self.asset.data.body_pos_w[:, self.feet_ids]
    
    def compute(self):
        feet_pos_b = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.feet_pos_w - self.com_pos_w.unsqueeze(1)
        )        
        in_contact = self.contact_sensor.data.net_forces_w[:, self.feet_ids_contact].norm(dim=-1, keepdim=True) > 0.1
        rew = ((feet_pos_b.abs() - self.margin).clamp_max(0.) * in_contact).sum(-1)
        rew = rew.min(dim=1).values
        return rew.reshape(self.num_envs, 1)
    
    def debug_draw(self):
        if self.env.backend == "isaac":
            self.env.debug_draw.vector(
                self.com_pos_w,
                torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3),
                color=(1.0, 0.0, 0.0, 1.0),
            )

