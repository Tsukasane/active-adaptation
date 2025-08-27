from math import inf
import torch
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor, RayCaster, Imu
    from isaaclab.sensors import Camera, TiledCamera
# from isaaclab.sensors import ContactSensor, RayCaster
# from isaaclab.actuators import DCMotor
# from isaaclab.assets import Articulation
# from isaaclab.utils.math import yaw_quat
# from isaaclab.utils.warp import raycast_mesh
import active_adaptation
from active_adaptation.utils.helpers import batchify
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)
from tensordict.tensordict import TensorDictBase, TensorDict

from active_adaptation.envs.locomotion import SimpleEnv

import active_adaptation.envs.mdp as mdp

ADAPTIVE_SIGMA = {
    "sigma": {
        "tracking_root_trans": 0.16,
        "tracking_root_rot": 0.16,
        "tracking_qpos": 0.16,
        "tracking_keypoints": 0.36,
        "tracking_eff": 0.36,
        "tracking_keypoints_local": 0.36,  # 新增：local keypoints 追踪的 sigma
        "tracking_eff_local": 0.36         # 新增：local eff 追踪的 sigma
    },
    "params": {
        "alpha": 1e-3
    }
}

class Humanoid(SimpleEnv):

    def __init__(self, cfg):
        super().__init__(cfg)
        # self.max_episode_length = torch.ones(self.num_envs, dtype=torch.long, device=self.device) * self.command_manager.num_frames
        self.start_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.end_frames = torch.ones(self.num_envs, dtype=torch.long, device=self.device) * self.command_manager.num_frames
        self._init_adaptive_sigma()

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.num_envs)
            env_ids = env_mask.nonzero().squeeze(-1)
            self.episode_count += env_ids.numel()
        else:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids):
            self._reset_idx(env_ids)
            self.scene.reset(env_ids)
        for callback in self._reset_callbacks:
            callback(env_ids)
        tensordict = TensorDict({}, self.num_envs, device=self.device)
        tensordict.update(self.observation_spec.zero())
        # self._compute_observation(tensordict)
        return tensordict
    
    def _reset_idx(self, env_ids: torch.Tensor):
        self.command_manager._update_stats(env_ids)
        init_root_state, start_frames, end_frames = self.command_manager.sample_init(env_ids)
        if not self.robot.is_fixed_base:
            self.robot.write_root_state_to_sim(
                init_root_state, 
                env_ids=env_ids
            )
        self.stats[env_ids] = 0.

        self.scene.reset(env_ids)

        self.episode_length_buf[env_ids] = self.start_frames[env_ids] = start_frames
        self.max_episode_length[env_ids] = self.end_frames[env_ids] = end_frames

        # in `self._reset_callbacks`
        # self.command_manager.reset(env_ids=env_ids)
        # self.action_manager.reset(env_ids=env_ids)

    def _compute_reward(self) -> TensorDictBase:
        rew_dict = super()._compute_reward()
        self.stats["episode_len"][:] = (self.episode_length_buf - self.start_frames).unsqueeze(1)
        self.stats["success"][:] = (self.episode_length_buf >= self.end_frames).unsqueeze(1).float()
        self.stats["episode_len_ratio"][:] = ((self.episode_length_buf - self.start_frames).float() / (self.end_frames - self.start_frames).float()).unsqueeze(1)
        return rew_dict

    # Observations of reference motion
    class ref_orientation(mdp.Observation):

        env: "Humanoid"

        def __init__(self, env, steps: int=1):
            super().__init__(env)
            self.robot: Articulation = self.env.scene["robot"]
            self.steps = steps
            self.ref_orientation = self.env.command_manager.root_orientation

        def compute(self) -> torch.Tensor:
            timestep = self.env.episode_length_buf.cpu()
            max_frame = self.env.max_episode_length.cpu()
            step_range = torch.arange(self.steps)
            timestep = timestep.unsqueeze(-1) + step_range  # (num_envs, steps)
            timestep = torch.min(timestep, max_frame[:, None]-1)
            ref_orientation = self.ref_orientation[timestep].to(self.device)
            return ref_orientation.reshape(self.num_envs, -1)

    class ref_height(mdp.Observation):
        def __init__(self, env, steps: int=1):
            super().__init__(env)
            self.robot: Articulation = self.env.scene["robot"]
            self.steps = steps
            self.ref_root_translation = self.env.command_manager.root_translations

        def compute(self) -> torch.Tensor:
            timestep = self.env.episode_length_buf.cpu()
            max_frame = self.env.max_episode_length.cpu()
            step_range = torch.arange(self.steps)
            timestep = timestep.unsqueeze(-1) + step_range
            timestep = torch.min(timestep, max_frame[:, None]-1)
            ref_root_translation = self.ref_root_translation[timestep].to(self.device)
            return ref_root_translation[:, :, 2].reshape(self.num_envs, -1)
        
    class ref_qpos(mdp.Observation):
        def __init__(self, env, joint_names, steps: int=1):
            super().__init__(env)
            self.robot: Articulation = self.env.scene["robot"]
            self.joint_indices, self.joint_names = self.robot.find_joints(joint_names, preserve_order=True)
            self.steps = steps
            self.ref_qpos = self.env.command_manager.qpos[:, self.joint_indices]

        def compute(self) -> torch.Tensor:
            timestep = self.env.episode_length_buf.cpu()
            max_frame = self.env.max_episode_length.cpu()
            step_range = torch.arange(self.steps)
            timestep = timestep.unsqueeze(-1) + step_range
            timestep = torch.min(timestep, max_frame[:, None]-1)
            ref_qpos = self.ref_qpos[timestep].to(self.device)
            return ref_qpos.reshape(self.num_envs, -1)
        
    class ref_keypoints(mdp.Observation):
        def __init__(self, env, steps: int=1):
            super().__init__(env)
            self.robot: Articulation = self.env.scene["robot"]
            self.steps = steps
            self.ref_keypoints = self.env.command_manager.kp_global    # (num_frames, num_joints, 3)

        def compute(self) -> torch.Tensor:
            timestep = self.env.episode_length_buf.cpu()
            max_frame = self.env.max_episode_length.cpu()
            step_range = torch.arange(self.steps)
            timestep = timestep.unsqueeze(-1) + step_range
            timestep = torch.min(timestep, max_frame[:, None]-1)
            ref_keypoints = self.ref_keypoints[timestep].to(self.device)    # (num_envs, steps, num_joints, 3)
            return ref_keypoints.reshape(self.num_envs, -1)
        
    class ref_keypoints_gap(mdp.Observation):
        def __init__(self, env, body_names: str, steps: int=1):
            super().__init__(env)
            self.robot: Articulation = self.env.scene["robot"]
            self.steps = steps
            self.ref_keypoints = self.env.command_manager.kp_global    # (num_frames, num_joints, 3)
            self.body_indices, self.body_names = self.robot.find_bodies(body_names, preserve_order=True)
            self.idx = [self.env.command_manager.bodys.index(name) for name in self.body_names]

            self.root_quat_w = self.robot.data.root_quat_w[:, None, None]    # (num_envs, 1, 1, 4)
            self.body_pos_global = self.robot.data.body_pos_w[:, self.body_indices]

        def update(self):
            self.root_quat_w = self.robot.data.root_quat_w[:, None, None]    # (num_envs, 1, 1, 4)
            self.body_pos_global = self.robot.data.body_pos_w[:, self.body_indices]

        def compute(self):
            timestep = self.env.episode_length_buf.cpu()
            max_frame = self.env.max_episode_length.cpu()
            step_range = torch.arange(self.steps)
            timestep = timestep.unsqueeze(-1) + step_range
            timestep = torch.min(timestep, max_frame[:, None]-1)
            ref_keypoints = self.ref_keypoints[timestep].to(self.device)   # (num_envs, steps, num_joints, 3)
            ref_keypoints.add_(self.env.scene.env_origins[:, None, None])
            ref_keypoints = ref_keypoints[:, :, self.idx, :]

            body_pos_global = self.body_pos_global.unsqueeze(1).expand_as(ref_keypoints)
            ref_keypoints_gap = quat_rotate_inverse(self.root_quat_w, ref_keypoints - body_pos_global)
            return ref_keypoints_gap.reshape(self.num_envs, -1)

        def debug_draw(self):
            if active_adaptation._BACKEND == "isaac":
                timestep = self.env.episode_length_buf.cpu()
                ref_keypoints = self.ref_keypoints[timestep].to(self.device)    # (num_envs, num_joints, 3)
                ref_keypoints.add_(self.env.scene.env_origins[:, None])
                ref_keypoints = ref_keypoints[:, self.idx, :]
                for i in range(ref_keypoints.shape[1]):
                    self.env.debug_draw.point(ref_keypoints[:, i], color=(1., 0., 0., 1.), size = 20)
                    self.env.debug_draw.point(self.body_pos_global[:, i], color=(0., 1., 0., 1.), size = 20)

    # 新增：Local keypoints 观测类
    class ref_keypoints_local(mdp.Observation):
        def __init__(self, env, steps: int=1):
            super().__init__(env)
            self.robot: Articulation = self.env.scene["robot"]
            self.steps = steps
            self.ref_keypoints = self.env.command_manager.kp_local    # 使用 local keypoints

        def compute(self) -> torch.Tensor:
            timestep = self.env.episode_length_buf.cpu()
            max_frame = self.env.max_episode_length.cpu()
            step_range = torch.arange(self.steps)
            timestep = timestep.unsqueeze(-1) + step_range
            timestep = torch.min(timestep, max_frame[:, None]-1)
            ref_keypoints = self.ref_keypoints[timestep].to(self.device)    # (num_envs, steps, num_joints, 3)
            return ref_keypoints.reshape(self.num_envs, -1)
        
    class ref_keypoints_gap_local(mdp.Observation):
        def __init__(self, env, body_names: str, steps: int=1):
            super().__init__(env)
            self.robot: Articulation = self.env.scene["robot"]
            self.steps = steps
            self.ref_keypoints_draw = self.env.command_manager.kp_global 
            self.ref_keypoints = self.env.command_manager.kp_local    # 使用 local keypoints
            self.body_indices, self.body_names = self.robot.find_bodies(body_names, preserve_order=True)
            self.idx = [self.env.command_manager.bodys.index(name) for name in self.body_names]

            self.root_quat_w = self.robot.data.root_quat_w[:, None, None]    # (num_envs, 1, 1, 4)
            self.body_pos_local = self.robot.data.body_pos_w[:, self.body_indices]

        def update(self):
            self.root_quat_w = self.robot.data.root_quat_w[:, None, None]    # (num_envs, 1, 1, 4)
            self.body_pos_local = self.robot.data.body_pos_w[:, self.body_indices]

        def compute(self):
            timestep = self.env.episode_length_buf.cpu()
            max_frame = self.env.max_episode_length.cpu()
            step_range = torch.arange(self.steps)
            timestep = timestep.unsqueeze(-1) + step_range
            timestep = torch.min(timestep, max_frame[:, None]-1)
            ref_keypoints = self.ref_keypoints[timestep].to(self.device)   # (num_envs, steps, num_joints, 3)
            ref_keypoints = ref_keypoints[:, :, self.idx, :]

            # 将当前机器人身体位置转换为局部坐标系
            root_pos_w = self.robot.data.root_pos_w[:, None, None]    # (num_envs, 1, 1, 3)
            root_quat_w = self.robot.data.root_quat_w[:, None, None]  # (num_envs, 1, 1, 4)
            
            # 计算相对于根节点的位置
            body_pos_relative = self.body_pos_local.unsqueeze(1) - root_pos_w
            # 转换到局部坐标系
            body_pos_local_coord = quat_rotate_inverse(root_quat_w, body_pos_relative)

            ref_keypoints_gap = ref_keypoints - body_pos_local_coord
            return ref_keypoints_gap.reshape(self.num_envs, -1)

        def debug_draw(self):
            if active_adaptation._BACKEND == "isaac":
                timestep = self.env.episode_length_buf.cpu()
                ref_keypoints = self.ref_keypoints_draw[timestep].to(self.device)    # (num_envs, num_joints, 3)
                ref_keypoints.add_(self.env.scene.env_origins[:, None])
                ref_keypoints = ref_keypoints[:, self.idx, :]
                for i in range(ref_keypoints.shape[1]):
                    self.env.debug_draw.point(ref_keypoints[:, i], color=(1., 0., 0., 1.), size = 20)
                    self.env.debug_draw.point(self.body_pos_local[:, i], color=(0., 1., 0., 1.), size = 20)
    
    class ref_trans_gap(mdp.Observation):
        def __init__(self, env, steps: int=1):
            super().__init__(env)
            self.robot: Articulation = self.env.scene["robot"]
            self.steps = steps
            self.ref_root_translation = self.env.command_manager.root_translations

        def compute(self):
            timestep = self.env.episode_length_buf.cpu()
            max_frame = self.env.max_episode_length.cpu()
            step_range = torch.arange(self.steps)
            timestep = timestep.unsqueeze(-1) + step_range
            timestep = torch.min(timestep, max_frame[:, None]-1)
            ref_root_translation = self.ref_root_translation[timestep].to(self.device)  # env world coords, (num_envs, steps, 3)
            ref_root_translation.add_(self.env.scene.env_origins.unsqueeze(1))
            self.root_pos = self.robot.data.root_pos_w.unsqueeze(1) # obtained from sim, the large world coords
            root_quat_w = self.robot.data.root_quat_w.unsqueeze(1)
            self.gap = ref_root_translation - self.root_pos
            ref_trans_gap = quat_rotate_inverse(root_quat_w, self.gap)
            return ref_trans_gap.reshape(self.num_envs, -1)
        
        def debug_draw(self):
            if active_adaptation._BACKEND == "isaac":
                self.env.debug_draw.vector(
                    self.root_pos[:, 0],
                    self.gap[:, 0],
                    color=(1., 0., 1., 1.),
                    size=1.
                )

    def _init_adaptive_sigma(self):
        self._adaptive_sigma = {k: torch.tensor(v, device=self.device) for k, v in ADAPTIVE_SIGMA["sigma"].items()}
        self._error_ema = {k: torch.tensor(v, device=self.device) for k, v in self._adaptive_sigma.items()}
        self._alpha = ADAPTIVE_SIGMA["params"]["alpha"]

    def _update_adaptive_sigma(self, error, term):
        self._error_ema[term] = self._error_ema[term] * (1 - self._alpha) + error * self._alpha
        self._adaptive_sigma[term] = min(self._adaptive_sigma[term], self._error_ema[term])
    
    # Motion Tracking Reward
    class tracking_root_trans(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.robot: Articulation = self.env.scene["robot"]

        def compute(self) -> torch.Tensor:
            timestep = (self.env.episode_length_buf-1).cpu()
            ref_root_translation = self.env.command_manager.root_translations[timestep].to(self.device) + self.env.scene.env_origins
            root_pos_w = self.robot.data.root_pos_w
            error = (root_pos_w - ref_root_translation).square().sum(-1, True).sqrt()
            # reward = torch.exp(- error / self.sigma)
            reward = torch.exp(- error / self.env._adaptive_sigma["tracking_root_trans"])
            self.env._update_adaptive_sigma(error.mean(), "tracking_root_trans")
            return reward
        
    class tracking_root_rot(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.robot: Articulation = self.env.scene["robot"]

        def compute(self) -> torch.Tensor:
            timestep = (self.env.episode_length_buf-1).cpu()
            ref_root_orientation = self.env.command_manager.root_orientation[timestep].to(self.device)
            root_quat_w = self.robot.data.root_quat_w
            dot_product = dot(root_quat_w, ref_root_orientation)
            error = 2 * torch.acos(dot_product.abs().clamp(min=-1.0, max=1.0))
            # reward = torch.exp(- error / self.sigma)
            reward = torch.exp(- error / self.env._adaptive_sigma["tracking_root_rot"])
            self.env._update_adaptive_sigma(error.mean(), "tracking_root_rot")
            return reward
        
    class tracking_qpos(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True, joint_names: str = ".*"):
            super().__init__(env, weight, enabled)
            self.robot: Articulation = self.env.scene["robot"]
            self.joint_indices, self.joint_names = self.robot.find_joints(joint_names, preserve_order=True)

        def compute(self) -> torch.Tensor:
            timestep = (self.env.episode_length_buf-1).cpu()
            ref_qpos = self.env.command_manager.qpos[timestep].to(self.device)[:, self.joint_indices]
            qpos = self.robot.data.joint_pos[:, self.joint_indices]
            error = (qpos - ref_qpos).square().mean(-1, True)
            # reward = torch.exp(- error / self.sigma)
            reward = torch.exp(- error / self.env._adaptive_sigma["tracking_qpos"])
            self.env._update_adaptive_sigma(error.mean(), "tracking_qpos")
            return reward
        
    class tracking_keypoints(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True, body_names: str = ".*"):
            super().__init__(env, weight, enabled)
            self.robot: Articulation = self.env.scene["robot"]
            self.body_indices, self.body_names = self.robot.find_bodies(body_names, preserve_order=True)
            self.idx = [self.env.command_manager.bodys.index(name) for name in self.body_names]

        def compute(self) -> torch.Tensor:
            timestep = (self.env.episode_length_buf-1).cpu()
            ref_keypoints = self.env.command_manager.kp_global[timestep].to(self.device)[:, self.idx]
            ref_keypoints.add_(self.env.scene.env_origins[:, None])

            body_pos_global = self.robot.data.body_pos_w[:, self.body_indices]

            diff = (ref_keypoints - body_pos_global).norm(dim=-1)
            error = diff.square().sum(-1, True).sqrt()
            # reward = torch.exp(- error / self.sigma)
            reward = torch.exp(- error / self.env._adaptive_sigma["tracking_keypoints"])
            self.env._update_adaptive_sigma(error.mean(), "tracking_keypoints")
            return reward

    class tracking_eff(tracking_keypoints): # NOTE(yiwen) seems only different sigma to tracking_keypoints for now.
        def __init__(self, env, weight: float, enabled: bool = True, body_names: str = ".*"):
            super().__init__(env, weight, enabled, body_names)

        def compute(self):
            timestep = (self.env.episode_length_buf-1).cpu()
            ref_keypoints = self.env.command_manager.kp_global[timestep].to(self.device)[:, self.idx]
            ref_keypoints.add_(self.env.scene.env_origins[:, None])

            body_pos_global = self.robot.data.body_pos_w[:, self.body_indices]

            diff = (ref_keypoints - body_pos_global).norm(dim=-1)
            error = diff.square().sum(-1, True).sqrt()
            # reward = torch.exp(- error / self.sigma)
            reward = torch.exp(- error / self.env._adaptive_sigma["tracking_eff"])
            self.env._update_adaptive_sigma(error.mean(), "tracking_eff")
            return reward

    # 新增：Local keypoints 追踪奖励类
    class tracking_keypoints_local(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True, body_names: str = ".*"):
            super().__init__(env, weight, enabled)
            self.robot: Articulation = self.env.scene["robot"]
            self.body_indices, self.body_names = self.robot.find_bodies(body_names, preserve_order=True)
            self.idx = [self.env.command_manager.bodys.index(name) for name in self.body_names]

        def compute(self) -> torch.Tensor:
            timestep = (self.env.episode_length_buf-1).cpu()
            ref_keypoints = self.env.command_manager.kp_local[timestep].to(self.device)[:, self.idx]

            # 将当前机器人身体位置转换为局部坐标系
            root_pos_w = self.robot.data.root_pos_w[:, None]    # (num_envs, 1, 3)
            root_quat_w = self.robot.data.root_quat_w[:, None]  # (num_envs, 1, 4)
            
            body_pos_global = self.robot.data.body_pos_w[:, self.body_indices]
            # 计算相对于根节点的位置
            body_pos_relative = body_pos_global - root_pos_w
            # 转换到局部坐标系
            body_pos_local = quat_rotate_inverse(root_quat_w, body_pos_relative)

            diff = (ref_keypoints - body_pos_local).norm(dim=-1)
            error = diff.square().sum(-1, True).sqrt()
            reward = torch.exp(- error / self.env._adaptive_sigma["tracking_keypoints_local"])
            self.env._update_adaptive_sigma(error.mean(), "tracking_keypoints_local")
            return reward

    class tracking_eff_local(tracking_keypoints_local): # NOTE(yiwen) seems only different sigma to tracking_keypoints_local for now.
        def __init__(self, env, weight: float, enabled: bool = True, body_names: str = ".*"):
            super().__init__(env, weight, enabled, body_names)

        def compute(self):
            timestep = (self.env.episode_length_buf-1).cpu()
            ref_keypoints = self.env.command_manager.kp_local[timestep].to(self.device)[:, self.idx]

            # 将当前机器人身体位置转换为局部坐标系
            root_pos_w = self.robot.data.root_pos_w[:, None]    # (num_envs, 1, 3)
            root_quat_w = self.robot.data.root_quat_w[:, None]  # (num_envs, 1, 4)
            
            body_pos_global = self.robot.data.body_pos_w[:, self.body_indices]
            # 计算相对于根节点的位置
            body_pos_relative = body_pos_global - root_pos_w
            # 转换到局部坐标系
            body_pos_local = quat_rotate_inverse(root_quat_w, body_pos_relative)

            diff = (ref_keypoints - body_pos_local).norm(dim=-1)
            error = diff.square().sum(-1, True).sqrt()
            reward = torch.exp(- error / self.env._adaptive_sigma["tracking_eff_local"])
            self.env._update_adaptive_sigma(error.mean(), "tracking_eff_local")
            return reward
        
    class tracking_contact(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.robot: Articulation = self.env.scene["robot"] # NOTE(yiwen) current state obtained from scene
            self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

            self.body_ids, self.body_names = self.contact_sensor.find_bodies(".*ankle_roll_link")
            self.body_ids = torch.tensor(self.body_ids, device=self.device)

        def compute(self) -> torch.Tensor:
            timestep = (self.env.episode_length_buf-1).cpu()
            in_contact = (self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.02).float() # NOTE(yiwen) the robot ankles and toes?
            ref_contact = self.env.command_manager.contact[timestep].to(self.device) # NOTE(yiwen) reference motion stored in env previously

            error = (in_contact - ref_contact).abs().mean(-1, True)
            reward = 1 - error
            return reward
        
    # Joint Position Penalty
    class joint_pos_default(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True, joint_names: str=".*"):
            super().__init__(env, weight, enabled)
            self.asset: Articulation = self.env.scene["robot"]
            self.joint_ids = self.asset.find_joints(joint_names)[0]
            self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids].clone() # NOTE(yiwen) angular, not the postion?
            self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        
        def compute(self) -> torch.Tensor:
            dev = self.asset.data.joint_pos[:, self.joint_ids] - self.default_joint_pos
            return - dev.square().mean(1, True)

    # Early Termination Conditions
    class dummy(mdp.Termination):
        def __init__(self, env):
            super().__init__(env)
            self.device = self.env.device

        def compute(self, termination: torch.Tensor) -> torch.Tensor: # NOTE(yiwen) not terminating any of the envs, probably for debugging
            return torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)

    class root_deviation(mdp.Termination):
        def __init__(self, env, max_distance: float):
            super().__init__(env)
            self.device = self.env.device
            self.max_distance = torch.tensor(max_distance, device=self.env.device)
            self.robot: Articulation = self.env.scene["robot"]

        def compute(self, termination: torch.Tensor) -> torch.Tensor:
            timestep = (self.env.episode_length_buf - 1).cpu()
            ref_root_translation = self.env.command_manager.root_translations[timestep].to(self.device)
            ref_root_translation.add_(self.env.scene.env_origins)
            root_pos_w = self.robot.data.root_pos_w
            deviation = (root_pos_w - ref_root_translation).norm(dim=1, keepdim=True)
            return deviation > self.max_distance
        
    class root_rot_deviation(mdp.Termination):
        def __init__(self, env, max_theta: float):
            super().__init__(env)
            self.device = self.env.device
            self.max_theta = torch.tensor(max_theta * 3.14 / 180, device=self.env.device)
            self.robot: Articulation = self.env.scene["robot"]

        def compute(self, termination: torch.Tensor) -> torch.Tensor:
            timestep = (self.env.episode_length_buf - 1).cpu()
            ref_root_orientation = self.env.command_manager.root_orientation[timestep].to(self.device)

            root_quat_w = self.robot.data.root_quat_w
            dot_product = dot(root_quat_w, ref_root_orientation)
            deviation = 2 * torch.acos(dot_product.abs().clamp(min=-1.0, max=1.0))

            return deviation > self.max_theta
        
    class track_kp_error(mdp.Termination):
        def __init__(self, env, max_distance: float, body_names: str = ".*"):
            super().__init__(env)
            self.device = self.env.device
            self.max_distance = torch.tensor(max_distance, device=self.env.device)
            self.robot: Articulation = self.env.scene["robot"]
            self.body_indices, self.body_names = self.robot.find_bodies(body_names, preserve_order=True)
            self.idx = [self.env.command_manager.bodys.index(name) for name in self.body_names]

        def compute(self, termination: torch.Tensor) -> torch.Tensor:
            timestep = (self.env.episode_length_buf - 1).cpu()
            ref_keypoints = self.env.command_manager.kp_global[timestep].to(self.device)[:, self.idx]
            ref_keypoints.add_(self.env.scene.env_origins[:, None])

            body_pos_global = self.robot.data.body_pos_w[:, self.body_indices]

            diff = (ref_keypoints - body_pos_global).norm(dim=-1)    # (num_envs, num_bodies)
            mean_diff = diff.mean(-1, True)     # (num_envs, 1)
            return mean_diff > self.max_distance

    # 新增：Local keypoints 误差终止条件类
    class track_kp_error_local(mdp.Termination):
        def __init__(self, env, max_distance: float, body_names: str = ".*"):
            super().__init__(env)
            self.device = self.env.device
            self.max_distance = torch.tensor(max_distance, device=self.device)
            self.robot: Articulation = self.env.scene["robot"]
            self.body_indices, self.body_names = self.robot.find_bodies(body_names, preserve_order=True)
            self.idx = [self.env.command_manager.bodys.index(name) for name in self.body_names]

        def compute(self, termination: torch.Tensor) -> torch.Tensor:
            timestep = (self.env.episode_length_buf - 1).cpu()
            ref_keypoints = self.env.command_manager.kp_local[timestep].to(self.device)[:, self.idx]

            # 将当前机器人身体位置转换为局部坐标系
            root_pos_w = self.robot.data.root_pos_w[:, None]    # (num_envs, 1, 3)
            root_quat_w = self.robot.data.root_quat_w[:, None]  # (num_envs, 1, 4)
            
            body_pos_global = self.robot.data.body_pos_w[:, self.body_indices]
            # 计算相对于根节点的位置
            body_pos_relative = body_pos_global - root_pos_w
            # 转换到局部坐标系
            body_pos_local = quat_rotate_inverse(root_quat_w, body_pos_relative)

            diff = (ref_keypoints - body_pos_local).norm(dim=-1)    # (num_envs, num_bodies)
            mean_diff = diff.mean(-1, True)     # (num_envs, 1)
            return mean_diff > self.max_distance

def dot(a: torch.Tensor, b: torch.Tensor):
    return (a * b).sum(-1, True)