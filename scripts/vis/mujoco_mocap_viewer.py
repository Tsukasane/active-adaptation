import numpy as np
import time
import threading

import mujoco
import mujoco.viewer

from active_adaptation.utils.motion import unitree_joint_names
from common import ZMQSubscriber, PORTS
from typing import List

scene = "active_adaptation/assets_mjcf/g1_29dof_nohand/g1_29dof_nohand.xml"
scene = "active_adaptation/assets_mjcf/g1_29dof_nohand/g1_29dof_nohand-suitcase.xml"
scene = "active_adaptation/assets_mjcf/g1_29dof_nohand/g1_29dof_nohand-plasticbox.xml"

class MuJoCoMocapViewer:
    def __init__(
        self,
        frequency: int = 50,
        mujoco_model_path: str = scene
    ):
        print("Initializing MuJoCo Mocap Viewer...")
        
        self.freq = frequency
        
        # Initialize MuJoCo model and viewer
        self.model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)

        
        # Get joint IDs and addresses
        mujoco_joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        shared_joint_names = list(sorted(set(mujoco_joint_names) & set(unitree_joint_names)))
        unitree_joint_indices = [unitree_joint_names.index(name) for name in shared_joint_names]
        mujoco_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in shared_joint_names]
        mujoco_qpos_adrs = [self.model.jnt_qposadr[joint_id] for joint_id in mujoco_joint_ids]
        self.joint_ids_unitree = np.array(unitree_joint_indices)
        self.joint_qpos_adrs = np.array(mujoco_qpos_adrs)

        free_joint_ids = [i for i in range(self.model.njnt) if self.model.joint(i).type == mujoco.mjtJoint.mjJNT_FREE]
        self.free_joint_names = [self.model.joint(i).name.replace('_root', '_pose') for i in free_joint_ids]
        self.free_joint_qpos_adrs = self.model.jnt_qposadr[free_joint_ids]
        self.free_joint_subscribers: List[ZMQSubscriber] = []

        # Initialize ZMQ subscribers
        self.joint_subscriber = ZMQSubscriber(PORTS['joint_pos'])
        for joint_name in self.free_joint_names:
            subscriber = ZMQSubscriber(PORTS[joint_name])
            self.free_joint_subscribers.append(subscriber)

        self.running = True
        self.comm_thread = threading.Thread(target=self.zmq_communication_loop)
        self.comm_thread.daemon = True
        self.comm_thread.start()

        print("MuJoCo Mocap Viewer initialized, waiting for data...")

    def zmq_communication_loop(self):
        """Handle ZMQ communication in a separate thread"""
        while self.running:
            for qpos_adr, subscriber in zip(self.free_joint_qpos_adrs, self.free_joint_subscribers):
                pose_msg = subscriber.receive_pose()
                if pose_msg:
                    pose = np.concatenate([pose_msg.position, pose_msg.quaternion])
                    self.data.qpos[qpos_adr:qpos_adr + 7] = pose
            
            joint_msg = self.joint_subscriber.receive_joint_state()
            if joint_msg:
                self.data.qpos[self.joint_qpos_adrs] = joint_msg.positions[self.joint_ids_unitree]
            
            time.sleep(0.005)

    def mujoco_update(self):
        """Update MuJoCo simulation at 50 Hz"""
        while self.running:
            if self.viewer is None:
                time.sleep(0.5)
                print("Waiting for MuJoCo model to be loaded...")
                continue
            
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()
            time.sleep(1.0 / self.freq)

    def run(self):
        """Main loop"""
        try:
            self.mujoco_update()
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if hasattr(self, 'comm_thread'):
            self.comm_thread.join()
        
        self.pelvis_subscriber.close()
        self.joint_subscriber.close()
        
        if self.viewer:
            self.viewer.close()

if __name__ == "__main__":
    viewer = MuJoCoMocapViewer()
    viewer.run()