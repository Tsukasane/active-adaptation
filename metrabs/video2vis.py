import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import requests
import time
from scipy.spatial.transform import Rotation as sRot
from matplotlib.animation import FuncAnimation, FFMpegWriter

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

CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),  # Pelvis to left hip, right hip, and spine
    (1, 4), (4, 7), (7, 10),  # Left leg
    (2, 5), (5, 8), (8, 11),  # Right leg
    (3, 6), (6, 9), (9, 12), (12, 15),  # Spine to head
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),  # Left arm
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),  # Right arm
]

def calculate_root_orientation(root, left_hip, right_hip, spine):
    # Compute right direction
    right_dir = right_hip - left_hip
    right_dir = right_dir / np.linalg.norm(right_dir)
    
    # Compute initial up direction
    up_dir = spine - root
    up_dir = up_dir / np.linalg.norm(up_dir)
    
    # Compute forward direction
    # forward_dir = np.cross(right_dir, up_dir)
    forward_dir = np.cross(up_dir, right_dir)
    forward_dir = forward_dir / np.linalg.norm(forward_dir)
    
    # Recompute up_dir to ensure orthogonality
    up_dir = np.cross(forward_dir, right_dir)
    up_dir = up_dir / np.linalg.norm(up_dir)
    
    # Construct rotation matrix
    rotation_matrix = np.column_stack((forward_dir, 
                                       right_dir, 
                                       up_dir))
    
    return rotation_matrix

class PoseVisualizer:
    def __init__(self, server_url="http://0.0.0.0:8080"):
        self.server_url = server_url
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fps = 20
        
        self.transform = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()

    def get_pose_data(self):
        try:
            response = requests.get(f"{self.server_url}/get_pose")
            if response.status_code == 200:
                data = response.json()
                if "j3d" in data:
                    return data["j3d"][0], data["dt"]
                else:
                    return None, None
            return None, None
        except Exception as e:
            print(f"Error getting pose data: {e}")
            return None, None
        
    def transform_pose(self, pose_data):
        joints = np.einsum('ij,kj->ki', self.transform, np.array(pose_data))

        root, left_hip, right_hip, spine = SMPL_BONE_ORDER_NAMES.index("Pelvis"), SMPL_BONE_ORDER_NAMES.index("L_Hip"), SMPL_BONE_ORDER_NAMES.index("R_Hip"), SMPL_BONE_ORDER_NAMES.index("Spine")
        
        orientation = calculate_root_orientation(joints[root], joints[left_hip], joints[right_hip], joints[spine])
        return joints, orientation
            
    def update(self, frame):
        pose_data, dt = self.get_pose_data()
        if dt is not None:
            self.fps = 1 / dt
        
        if pose_data is None:
            return []
            
        self.ax.clear()
        
        joints, orientation = self.transform_pose(pose_data)
        self.ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                        c='y', marker='o', s=50)
        
        unit_vector = np.array([1, 0, 0])
        direction = sRot.apply(sRot.from_matrix(orientation), unit_vector)
        self.ax.quiver(joints[0, 0], joints[0, 1], joints[0, 2], direction[0], direction[1], direction[2], color='g', length=1.0)

        for joint1_idx, joint2_idx in CONNECTIONS:
            joint1 = joints[joint1_idx]
            joint2 = joints[joint2_idx]
            self.ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], [joint1[2], joint2[2]], 'b-')
        
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([0, 2])
        
        self.ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=0.1)
        self.ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=0.1)
        self.ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=0.1)
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        self.ax.set_title(f'FPS: {self.fps:.1f}')
        
        return []
        
    def animate(self):
        ani = FuncAnimation(self.fig, 
                            self.update, 
                            interval=1000/self.fps,
                            blit=False,
                            cache_frame_data=False)
        # plt.show()
        ani.save('animation.mp4', writer=FFMpegWriter(fps=self.fps))

def main():
    visualizer = PoseVisualizer()
    visualizer.animate()

if __name__ == "__main__":
    main()
