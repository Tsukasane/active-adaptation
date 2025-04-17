import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
_missing = {
    "bool":    np.bool_,
    "int":     np.int_,
    "float":   np.float_,
    "complex": np.complex_,
    "object":  np.object_,
    "unicode": np.unicode_,
    "str":     np.str_,
}
for name, new in _missing.items():
    if not hasattr(np, name):
        setattr(np, name, new)

from scipy.spatial.transform import Rotation as sRot
import joblib
import torch
from torch.autograd import Variable
from tqdm import tqdm
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES
from phc.utils.torch_humanoid_batch import Humanoid_Batch
from smpl_sim.utils.smoothing_utils import gaussian_filter_1d_batch
import hydra
from omegaconf import DictConfig

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

def animate_3d(joints, orientation=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(num, data, line):
        ax.clear()
        ax.scatter(data[num][:, 0], data[num][:, 1], data[num][:, 2], c='y', marker='o')

        if orientation is not None:
            unit_vector = np.array([1, 0, 0])
            direction = R.apply(R.from_rotvec(orientation[num]), unit_vector)
            ax.quiver(data[num][0, 0], data[num][0, 1], data[num][0, 2], direction[0], direction[1], direction[2], color='blue', length=1.0)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0, 2)
        ax.set_title(f"Frame {num}")

        ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=0.1)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=0.1)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=0.1)
        return line,

    ani = animation.FuncAnimation(fig, update, frames=joints.shape[0], fargs=(joints, None), interval=50)
    plt.show()
    
def process_motion(all_pkls, key_names, cfg):
    device = torch.device("cpu")
    
    humanoid_fk = Humanoid_Batch(cfg) # load forward kinematics model
    num_augment_joint = len(cfg.extend_config)

    #### Define corresonpdances between h1 and smpl joints
    robot_joint_names_augment = humanoid_fk.body_names_augment 
    robot_joint_pick = [i[0] for i in cfg.joint_matches]
    smpl_joint_pick = [i[1] for i in cfg.joint_matches]
    robot_joint_pick_idx = [robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]
    
    
    all_data = {}
    pbar = tqdm(key_names, position=0, leave=True)
    for data_key in pbar:
        data = all_pkls[data_key]
        joints = torch.from_numpy(data["joints"]).float()       # [T, J, 3]
        root_orientation = torch.from_numpy(data["root_rot"]).float()    # [T, 3] as rotation vector
        N = joints.shape[0]
        print('Processing:', data_key, " Motion length: ", N)

        animate_3d(joints, orientation=root_orientation)
        
        root_trans_offset = joints[:, 0].clone()

        dof_pos = torch.zeros((1, N, humanoid_fk.num_dof, 1))

        dof_pos_new = Variable(dof_pos.clone(), requires_grad=True)
        root_rot_new = Variable(root_orientation.clone(), requires_grad=True)
        optimizer = torch.optim.Adam([dof_pos_new, root_rot_new],lr=0.02)


        kernel_size = 5  # Size of the Gaussian kernel
        sigma = 0.75  # Standard deviation of the Gaussian kernel
        B, T, J, D = dof_pos_new.shape    

        
        for iteration in range(cfg.get("fitting_iterations", 500)):
            pose_aa_h1_new = torch.cat([root_rot_new[None, :, None], humanoid_fk.dof_axis * dof_pos_new, torch.zeros((1, N, num_augment_joint, 3)).to(device)], axis = 2)
            fk_return = humanoid_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ])
            
            
            if num_augment_joint > 0:
                diff = fk_return.global_translation_extend[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            else:
                diff = fk_return.global_translation[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
                
            loss_g = diff.norm(dim = -1).mean() + 0.01 * torch.mean(torch.square(dof_pos_new))
            loss = loss_g
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])

            pbar.set_description_str(f"{data_key}-Iter: {iteration} \t {loss.item() * 1000:.3f}")
            dof_pos_new.data = gaussian_filter_1d_batch(dof_pos_new.squeeze().transpose(1, 0)[None, ], kernel_size, sigma).transpose(2, 1)[..., None]
            
        dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])
        pose_aa_h1_new = torch.cat([root_rot_new[None, :, None], humanoid_fk.dof_axis * dof_pos_new, torch.zeros((1, N, num_augment_joint, 3)).to(device)], axis = 2)

        root_trans_offset_dump = root_trans_offset.clone()
        
        joints_dump = joints.numpy().copy()
        
        data_dump = {
                    "root_trans_offset": root_trans_offset_dump.squeeze().detach().numpy(),
                    "pose_aa": pose_aa_h1_new.squeeze().detach().numpy(),   
                    "dof": dof_pos_new.squeeze().detach().numpy(), 
                    "root_rot": sRot.from_rotvec(root_rot_new.detach().numpy()).as_quat(),
                    "smpl_joints": joints_dump, 
                    "fps": 30
                    }
        all_data[data_key] = data_dump
    return all_data
        

@hydra.main(version_base=None, config_path="../cfg", config_name="unitree_g1_fitting")
def main(cfg : DictConfig) -> None:
    
    motion_file = cfg.motion_file
    all_pkls = joblib.load(motion_file)
    print(all_pkls)
    key_names = list(all_pkls.keys())
    
    all_data = process_motion(all_pkls, key_names, cfg)


if __name__ == "__main__":
    main()
