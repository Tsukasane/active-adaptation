#!/usr/bin/env python3
"""
This script loads robot trajectory from npz files and publishes
the poses over ZMQ topics. It publishes:
  • pelvis pose — pelvis position and rotation from the robot trajectory file.
  • box pose — box position and rotation
  • joint positions — joint positions
  
"""

import argparse
import numpy as np
import time
import os
import shutil
import tempfile
from pathlib import Path

from active_adaptation.utils.motion import MotionDataset, MotionData, unitree_joint_names
from common import ZMQPublisher, PORTS

class SMPLPublisher:
    def __init__(
        self,
        data_file,
        rate=50,
    ):
        self.tmp_dir = tempfile.mkdtemp()
        data_path = Path(data_file)
        tmp_data_path = Path(self.tmp_dir) / data_path.name
        shutil.copy2(data_path, tmp_data_path)
        meta_path = data_path.parent / "meta.json"
        if meta_path.exists():
            shutil.copy2(meta_path, Path(self.tmp_dir) / "meta.json")
        else:
            print(f"Warning: meta.json not found at {meta_path}")

        dataset = MotionDataset.create_from_path(str(self.tmp_dir), target_fps=rate).to("cpu")
        motion_data: MotionData = dataset.data
        
        root_body_indices, root_body_names = dataset.find_bodies("pelvis")
        assert len(root_body_indices) == 1
        root_body_index = root_body_indices[0]

        self.root_pos_w = motion_data.body_pos_w[:, root_body_index].numpy()
        self.root_quat_w = motion_data.body_quat_w[:, root_body_index].numpy()

        matched_joint_names = list(sorted(set(unitree_joint_names) & set(dataset.joint_names)))
        joint_indices_dataset = [dataset.joint_names.index(name) for name in matched_joint_names]
        joint_indices_unitree = [unitree_joint_names.index(name) for name in matched_joint_names]
        self.joint_pos = np.zeros((motion_data.joint_pos.shape[0], len(unitree_joint_names)))
        self.joint_pos[:, joint_indices_unitree] = motion_data.joint_pos[:, joint_indices_dataset]

        box_body_index = dataset.body_names.index("box")
        self.box_pos_w = motion_data.body_pos_w[:, box_body_index].numpy()
        self.box_quat_w = motion_data.body_quat_w[:, box_body_index].numpy()

        # Create ZMQ publishers
        self.pelvis_publisher = ZMQPublisher(PORTS['pelvis_pose'])
        self.box_publisher = ZMQPublisher(PORTS['box_pose'])
        self.joint_publisher = ZMQPublisher(PORTS['joint_pos'])

        self.publish_rate = rate
        self.index = 0
        self.n_steps = dataset.num_steps

    def publish_once(self):
        print(f"Publishing at index {self.index}")
        if self.index >= self.n_steps:
            self.index = 0

        pelvis_pos = self.root_pos_w[self.index]
        pelvis_quat = self.root_quat_w[self.index]  # Expected order: [w, x, y, z]

        box_pos = self.box_pos_w[self.index]
        box_quat = self.box_quat_w[self.index]

        # Publish the poses
        self.pelvis_publisher.publish_pose(pelvis_pos, pelvis_quat)
        self.box_publisher.publish_pose(box_pos, box_quat)

        # Publish joint state
        joint_qpos = self.joint_pos[self.index]
        self.joint_publisher.publish_joint_state(joint_qpos)

        self.index += 1

    def run(self):
        """Run the publisher in a loop"""
        try:
            while True:
                start_time = time.time()
                self.publish_once()
                
                # Sleep to maintain the desired rate
                elapsed = time.time() - start_time
                sleep_time = max(0, 1.0 / self.publish_rate - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("Shutting down publisher...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.pelvis_publisher.close()
        self.box_publisher.close()
        self.joint_publisher.close()
        
        # Cleanup temporary directory
        if hasattr(self, 'tmp_dir') and os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
            print(f"Cleaned up temporary directory {self.tmp_dir}")

    def __del__(self):
        self.cleanup()

def main():
    parser = argparse.ArgumentParser(
        description="Publish pelvis, box, and joint poses using ZMQ."
    )
    parser.add_argument(
        "data",
        type=str,
        help="Path to the motion data directory",
    )
    parser.add_argument("--rate", type=float, default=50, help="Publishing rate [Hz].")
    args = parser.parse_args()

    publisher = SMPLPublisher(args.data, rate=args.rate)
    publisher.run()

if __name__ == "__main__":
    main()
