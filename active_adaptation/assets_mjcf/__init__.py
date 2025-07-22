import os
import json
from active_adaptation.envs.mujoco import MJArticulationCfg
import active_adaptation.utils.symmetry as symmetry_utils

ROBOTS = {}

PATH = os.path.dirname(__file__)

ROBOTS["g1_29dof"] = MJArticulationCfg(
    mjcf_path=os.path.join(PATH, "g1_23dof", "g1_29dof_nohand.xml"),
    **json.load(open(os.path.join(PATH, "g1_23dof", "g1_29dof_nohand.json"))),
    joint_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
        "left_hip_roll_joint": (-1, "right_hip_roll_joint"),
        "left_hip_yaw_joint": (-1, "right_hip_yaw_joint"),
        "left_knee_joint": (1, "right_knee_joint"),
        "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
        "left_ankle_roll_joint": (-1, "right_ankle_roll_joint"),
        "waist_yaw_joint": (-1, "waist_yaw_joint"),
        "waist_roll_joint": (-1, "waist_roll_joint"),
        "waist_pitch_joint": (1, "waist_pitch_joint"),
        "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
        "left_shoulder_roll_joint": (-1, "right_shoulder_roll_joint"),
        "left_shoulder_yaw_joint": (-1, "right_shoulder_yaw_joint"),
        "left_elbow_joint": (1, "right_elbow_joint"),
        "left_wrist_yaw_joint": (-1, "right_wrist_yaw_joint"),
        "left_wrist_roll_joint": (-1, "right_wrist_roll_joint"),
        "left_wrist_pitch_joint": (1, "right_wrist_pitch_joint"),
    }),
    spatial_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_link": "right_hip_pitch_link",
        "left_hip_roll_link": "right_hip_roll_link",
        "left_hip_yaw_link": "right_hip_yaw_link",
        "left_knee_link": "right_knee_link",
        "left_ankle_pitch_link": "right_ankle_pitch_link",
        "left_ankle_roll_link": "right_ankle_roll_link",
        "pelvis": "pelvis",
        "torso_link": "torso_link",
        "waist_yaw_link": "waist_yaw_link",
        "waist_roll_link": "waist_roll_link",
        "left_shoulder_pitch_link": "right_shoulder_pitch_link",
        "left_shoulder_roll_link": "right_shoulder_roll_link",
        "left_shoulder_yaw_link": "right_shoulder_yaw_link",
        "left_elbow_link": "right_elbow_link",
        "left_wrist_yaw_link": "right_wrist_yaw_link",
        "left_wrist_roll_link": "right_wrist_roll_link",
        "left_wrist_pitch_link": "right_wrist_pitch_link",
        "pelvis_contour_link": "pelvis_contour_link",
        "imu_link": "imu_link",
        "d435_link": "d435_link",
        "head_link": "head_link",
        "logo_link": "logo_link",
        "mid360_link": "mid360_link",
        "waist_support_link": "waist_support_link",
        "left_hand_marker": "right_hand_marker",
    })
)
