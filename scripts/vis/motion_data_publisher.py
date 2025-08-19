#!/usr/bin/env python3
"""
This script loads robot trajectory from npz files and publishes
the poses over ZMQ topics. It publishes:
  • pelvis pose — pelvis position and rotation from the robot trajectory file.
  • joint positions — joint positions
  
Keyboard controls:
  • Space: Pause/Resume playback
  • Left Arrow: Previous frame (when paused)
  • Right Arrow: Next frame (when paused)
  • Esc: Exit
"""

import argparse
import numpy as np
import time
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import List

from sshkeyboard import listen_keyboard, stop_listening
from active_adaptation.utils.motion import MotionDataset, MotionData
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

        print(motion_data.shape)
        print(motion_data.shape [0]/ rate)
        # breakpoint()

        # get joint names and state
        self.joint_names = dataset.joint_names
        self.joint_pos = motion_data.joint_pos.numpy()

        self.joint_names_publisher = ZMQPublisher(PORTS['joint_names'])
        self.joint_publisher = ZMQPublisher(PORTS['joint_pos'])

        # get body names and poses
        self.body_names = []
        self.body_publishers: List[ZMQPublisher] = []
        for body_name in dataset.body_names:
            try:
                publisher = ZMQPublisher(PORTS[f"{body_name}_pose"])
                self.body_publishers.append(publisher)
                self.body_names.append(body_name)
            except KeyError:
                continue

        body_indices = [dataset.body_names.index(name) for name in self.body_names]
        self.body_pos_w = motion_data.body_pos_w[:, body_indices].numpy()
        self.body_quat_w = motion_data.body_quat_w[:, body_indices].numpy()

        self.publish_rate = rate
        self.index = 0
        self.n_steps = dataset.num_steps
        
        # Playback control
        self.paused = False
        self.running = True
        self.lock = threading.Lock()
        
        print(f"Loaded {self.n_steps} frames at {rate} Hz")
        print(f"Joint names: {self.joint_names}")
        print("Controls: Space=Pause/Resume, Left/Right=Navigate (when paused), Esc=Exit")

    def publish_once(self):
        # Publish joint names
        self.joint_names_publisher.publish_names(self.joint_names)

        # Publish joint state with original joint order
        joint_qpos = self.joint_pos[self.index]
        self.joint_publisher.publish_joint_state(joint_qpos)
        
        # Publish body poses
        for i, body_publisher in enumerate(self.body_publishers):
            body_pos = self.body_pos_w[self.index, i]
            body_quat = self.body_quat_w[self.index, i]
            body_publisher.publish_pose(body_pos, body_quat)

    def on_key_press(self, key):
        """Handle keyboard input"""
        with self.lock:
            if key == "space":
                self.paused = not self.paused
                status = "PAUSED" if self.paused else "PLAYING"
                print(f"Playback {status} (frame {self.index}/{self.n_steps-1})")
                
            elif key == "left" and self.paused:
                self.index = (self.index - 5) % self.n_steps
                print(f"Frame {self.index}/{self.n_steps-1}")
                
            elif key == "right" and self.paused:
                self.index = (self.index + 5) % self.n_steps
                print(f"Frame {self.index}/{self.n_steps-1}")
                
            elif key == "esc":
                print("Stopping...")
                self.running = False
                stop_listening()

    def start_keyboard_listener(self):
        """Start keyboard listener in a separate thread"""
        def keyboard_thread():
            try:
                listen_keyboard(
                    on_press=self.on_key_press,
                    until=None,  # Don't stop on any key, we handle it manually
                    # sequential=True
                )
            except Exception as e:
                print(f"Keyboard listener error: {e}")
        
        thread = threading.Thread(target=keyboard_thread, daemon=True)
        thread.start()

    def run(self):
        """Run the publisher in a loop"""
        # Start keyboard listener
        self.start_keyboard_listener()
        
        try:
            while self.running:
                start_time = time.time()
                
                with self.lock:
                    # Always publish current frame
                    self.publish_once()
                    
                    # Only advance if not paused
                    if not self.paused:
                        self.index = (self.index + 1) % self.n_steps
                
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
        self.running = False
        stop_listening()
        
        self.joint_publisher.close()
        self.joint_names_publisher.close()
        for publisher in self.body_publishers:
            publisher.close()
        
        # Cleanup temporary directory
        if hasattr(self, 'tmp_dir') and os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
            print(f"Cleaned up temporary directory {self.tmp_dir}")

    def __del__(self):
        self.cleanup()

def main():
    parser = argparse.ArgumentParser(
        description="Publish pelvis and joint poses using ZMQ with keyboard controls."
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
