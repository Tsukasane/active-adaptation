import torch
import numpy as np
import einops
from typing import Tuple, TYPE_CHECKING

from isaaclab.utils.math import quat_apply_yaw, quat_mul, quat_inv
from isaaclab.utils.string import resolve_matching_names
import active_adaptation
from active_adaptation.envs.mdp.base import Observation
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, yaw_quat, EMA
import active_adaptation.utils.symmetry as sym_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor, RayCaster, Imu
    from isaaclab.sensors import Camera, TiledCamera

if active_adaptation.get_backend() == "isaac":
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.trimesh.utils import make_plane
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
    from pxr import UsdGeom, UsdPhysics


class root_quat_w(Observation):
    def __init__(self, env, noise_std: float=0.):
        super().__init__(env)
        self.asset = self.env.scene["robot"]
        self.noise_std = noise_std

    def compute(self):
        return random_noise(self.asset.data.root_quat_w, self.noise_std)

class JointObs(Observation):
    def __init__(
        self, 
        env,
        joint_names: str=".*", 
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names, preserve_order=True)

class joint_pos(JointObs):
    def __init__(
        self, 
        env, 
        joint_names: str=".*",
        noise_std: float=0.0,
    ):
        super().__init__(env, joint_names)
        self.noise_std = noise_std

    def compute(self) -> torch.Tensor:
        return random_noise(self.asset.data.joint_pos[:, self.joint_ids], self.noise_std)


class joint_vel(JointObs):
    def __init__(
        self,
        env,
        joint_names: str=".*",
        noise_std: float=0.0
    ):
        super().__init__(env, joint_names)
        self.noise_std = noise_std
    
    def compute(self) -> torch.Tensor:
        return random_noise(self.asset.data.joint_vel[:, self.joint_ids], self.noise_std)

class root_height(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.asset = self.env.scene["robot"]

    def compute(self):
        return self.asset.data.root_pos_w[:, 2].unsqueeze(1)

class body_pos(Observation):
    def __init__(self, env, body_names: str, yaw_only: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.yaw_only = yaw_only
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.update()
        if self.env.backend == "mujoco":
            self.feet_marker_0 = self.env.scene.create_sphere_marker(0.05, [1, 0, 0, 0.5])
            self.feet_marker_1 = self.env.scene.create_sphere_marker(0.05, [1, 0, 0, 0.5])

    def update(self):
        if self.yaw_only:
            self.root_quat_w = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
        else:
            self.root_quat_w = self.asset.data.root_quat_w.unsqueeze(1)
        self.root_pos_w = self.asset.data.root_pos_w.unsqueeze(1)
        self.body_pos_w = self.asset.data.body_pos_w[:, self.body_indices]
        
    def compute(self):
        body_pos_b = quat_rotate_inverse(self.root_quat_w, self.body_pos_w - self.root_pos_w)
        return body_pos_b.reshape(self.num_envs, -1)
    
    def symmetry_transforms(self):
        return sym_utils.cartesian_space_symmetry(self.asset, self.body_names)
    
    def debug_draw(self):
        if self.env.backend == "mujoco":
            self.feet_marker_0.geom.pos = self.asset.data.body_pos_w[0, self.body_indices[0]]
            self.feet_marker_1.geom.pos = self.asset.data.body_pos_w[0, self.body_indices[1]]


class body_vel(Observation):
    def __init__(self, env, body_names: str, yaw_only: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.yaw_only = yaw_only
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.update()
    
    def update(self):
        if self.yaw_only:
            self.root_quat_w = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
        else:
            self.root_quat_w = self.asset.data.root_quat_w.unsqueeze(1)
        self.body_vel_w = self.asset.data.body_vel_w[:, self.body_indices]
        
    def compute(self):
        body_lin_vel_b = quat_rotate_inverse(self.root_quat_w, self.body_vel_w[:, :, :3])
        body_ang_vel_b = quat_rotate_inverse(self.root_quat_w, self.body_vel_w[:, :, 3:])
        return body_lin_vel_b.reshape(self.num_envs, -1)
    
    def symmetry_transforms(self):
        return sym_utils.cartesian_space_symmetry(self.asset, self.body_names)


class body_acc(Observation):
    
    def __init__(self, env, body_names, yaw_only: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.yaw_only = yaw_only
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        print(f"Track body acc for {self.body_names}")
        self.body_acc_b = torch.zeros(self.env.num_envs, len(self.body_indices), 3, device=self.env.device)

    def update(self):
        if self.yaw_only:
            quat = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
        else:
            quat = self.asset.data.root_quat_w.unsqueeze(1)
        body_acc_w = self.asset.data.body_lin_acc_w[:, self.body_indices]
        self.body_acc_b[:] = quat_rotate_inverse(quat, body_acc_w)
        
    def compute(self):
        return self.body_acc_b.reshape(self.env.num_envs, -1)

class root_angvel_b(Observation):
    def __init__(self, env, noise_std: float=0., yaw_only: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.yaw_only = yaw_only
        self.update()
    
    def update(self):
        if self.yaw_only:
            self.quat = yaw_quat(self.asset.data.root_quat_w)
        else:
            self.quat = self.asset.data.root_quat_w
        self.root_angvel_w = self.asset.data.root_ang_vel_w.clone()

    def compute(self) -> torch.Tensor:
        ang_vel_w = random_noise(self.root_angvel_w, self.noise_std) 
        ang_vel_b = quat_rotate_inverse(self.quat, ang_vel_w)
        return ang_vel_b.reshape(self.num_envs, -1)
    
    def symmetry_transforms(self):
        # left-right symmetry: flip only roll and yaw
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[-1., 1., -1.])
        return transform

class projected_gravity_b(Observation):
    def __init__(self, env, noise_std: float=0.):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.init_quat = self.asset.data.root_quat_w.clone()
        self.noise_std = noise_std
    
    def compute(self):
        # projected_gravity_b = quat_rotate_inverse(self.init_quat, self.asset.data.projected_gravity_b)
        projected_gravity_b = self.asset.data.projected_gravity_b
        noise = torch.randn_like(projected_gravity_b).clip(-3., 3.) * self.noise_std
        projected_gravity_b += noise
        return projected_gravity_b / projected_gravity_b.norm(dim=-1, keepdim=True)

    def symmetry_transforms(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1])
        return transform
    
    def lerp(self, obs_tm1, obs_t, t):
        gravity = torch.lerp(obs_tm1, obs_t, t)
        gravity = gravity / gravity.norm(dim=-1, keepdim=True)
        return gravity


class root_linvel_b(Observation):
    def __init__(self, env, gammas=(0.,), yaw_only: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.yaw_only = yaw_only
        self.ema = EMA(self.asset.data.root_lin_vel_w, gammas=gammas)
        self.ema.update(self.asset.data.root_lin_vel_w)
        self.update()
    
    def reset(self, env_ids: torch.Tensor):
        self.ema.reset(env_ids)
    
    def post_step(self, substep):
        self.ema.update(self.asset.data.root_lin_vel_w)
    
    def update(self):
        if self.yaw_only:
            self.quat = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
        else:
            self.quat = self.asset.data.root_quat_w.unsqueeze(1)

    def compute(self) -> torch.Tensor:
        linvel = self.ema.ema
        linvel = quat_rotate_inverse(self.quat, linvel)
        return linvel.reshape(self.num_envs, -1)
    
    def symmetry_transforms(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1])
        return transform

    # def debug_draw(self):
    #     if self.env.sim.has_gui() and self.env.backend == "isaac":
    #         if self.body_ids is None:
    #             linvel = self.asset.data.root_lin_vel_w
    #         else:
    #             linvel = (self.asset.data.body_lin_vel_w[:, self.body_ids] * self.body_masses).mean(1)
    #         self.env.debug_draw.vector(
    #             self.asset.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
    #             linvel,
    #             color=(0.8, 0.1, 0.1, 1.)
    #         )


class applied_torque(Observation):
    def __init__(self, env, joint_names: str=".*"):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
    
    def compute(self) -> torch.Tensor:
        applied_efforts = self.asset.data.applied_torque
        return applied_efforts[:, self.joint_ids]
    
    def symmetry_transforms(self):
        transform = sym_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform

class prev_actions(Observation):
    def __init__(self, env, steps: int=1, flatten: bool=True, permute: bool=False):
        super().__init__(env)
        self.steps = steps
        self.flatten = flatten
        self.permute = permute
        self.action_manager = self.env.action_manager
    
    def compute(self):
        action_buf = self.action_manager.action_buf[:, :, :self.steps].clone()
        if self.permute:
            action_buf = action_buf.permute(0, 2, 1)
        if self.flatten:
            return action_buf.reshape(self.num_envs, -1)
        else:
            return action_buf

    def symmetry_transforms(self):
        assert self.permute
        transform = self.action_manager.symmetry_transforms()
        return transform.repeat(self.steps)



def symlog(x: torch.Tensor, a: float=1.):
    return x.sign() * torch.log(x.abs() * a + 1.) / a

def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3., 3.) * std

meshes = {}

def _initialize_warp_meshes(mesh_prim_path, device):
    if mesh_prim_path in meshes:
        return meshes[mesh_prim_path]

    # check if the prim is a plane - handle PhysX plane as a special case
    # if a plane exists then we need to create an infinite mesh that is a plane
    mesh_prim = sim_utils.get_first_matching_child_prim(
        mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
    )
    # if we did not find a plane then we need to read the mesh
    if mesh_prim is None:
        # obtain the mesh prim
        mesh_prim = sim_utils.get_first_matching_child_prim(
            mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
        )
        # check if valid
        if mesh_prim is None or not mesh_prim.IsValid():
            raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
        # cast into UsdGeomMesh
        mesh_prim = UsdGeom.Mesh(mesh_prim)
        # read the vertices and faces
        points = np.asarray(mesh_prim.GetPointsAttr().Get())
        indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
        wp_mesh = convert_to_warp_mesh(points, indices, device=device)
    else:
        mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
        wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)
    # add the warp mesh to the list
    meshes[mesh_prim_path] = wp_mesh
    return wp_mesh