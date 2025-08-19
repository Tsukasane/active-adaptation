from dataclasses import MISSING
import os
import copy
import isaaclab.sim as sim_utils
import torch
from isaaclab_assets import H1_CFG
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets import Articulation
from active_adaptation.envs.actuator import HybridActuatorCfg
import active_adaptation.utils.symmetry as symmetry_utils

from .base import ArticulationCfg


ASSET_PATH = os.path.dirname(__file__)

H1_CFG = copy.deepcopy(H1_CFG)
H1_CFG.spawn.usd_path = f"{ASSET_PATH}/H1/h1_minimal.usd"
H1_CFG.actuators = {
    "base_legs": DCMotorCfg(
        joint_names_expr=[".*"],
        effort_limit=300.0,
        saturation_effort=300.0,
        velocity_limit=30.0,
        stiffness={
            ".*hip.*": 200,
            ".*knee.*": 300,
            ".*ankle.*": 40,
            "torso": 300,
            ".*shoulder.*": 100,
            ".*elbow.*": 100
        },
        damping={
            ".*hip.*": 5,
            ".*knee.*": 6,
            ".*ankle.*": 2,
            "torso": 6,
            ".*shoulder.*": 2,
            ".*elbow.*": 2
        },
        friction=0.0,
    )
}

_G1_29DOF_CFG = ArticulationCfg( # no wrist pitch and yaw
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}" + "/g1/{ROBOT_TYPE}.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": -0.15,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.15,
            ".*_elbow_joint": 0.8,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=".*",
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                ".*_ankle_pitch_joint": 35.0,
                ".*_ankle_roll_joint": 35.0,

                ".*waist_yaw_joint": 88.0,
                ".*waist_roll_joint": 35.0,
                ".*waist_pitch_joint": 35.0,

                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,

                ".*_elbow_joint": 25.0,

                ".*_wrist_yaw_joint": 5.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,

            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
                ".*_ankle_pitch_joint": 30.0,
                ".*_ankle_roll_joint": 30.0,

                ".*waist_yaw_joint": 32.0,
                ".*waist_roll_joint": 30.0,
                ".*waist_pitch_joint": 30.0,

                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,

                ".*_elbow_joint": 37.0,

                ".*_wrist_yaw_joint": 22.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
            },

            stiffness=MISSING,
            damping=MISSING,
            armature={
                ".*_hip_pitch_joint": 0.0103,
                ".*_hip_roll_joint": 0.0251,
                ".*_hip_yaw_joint": 0.0103,
                ".*_knee_joint": 0.0251,
                ".*_ankle_.*_joint": 0.003597,
                "waist_.*_joint": 0.0103,
                ".*_shoulder_.*": 0.003597,
                ".*_elbow_joint": 0.003597,
                ".*_wrist_.*_joint": 0.003597,
            },
            friction=0.01,
        ),
    },
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
    })
)

G1_29DOF_CFG = copy.deepcopy(_G1_29DOF_CFG)
G1_29DOF_CFG.actuators["base_legs"].stiffness = {
    ".*_hip_yaw_joint": 150.0,
    ".*_hip_roll_joint": 150.0,
    ".*_hip_pitch_joint": 200.0,
    ".*_knee_joint": 200.0,
    "waist_yaw_joint": 150.0, # unitree_ros
    "waist_roll_joint": 150.0, # unitree_ros
    "waist_pitch_joint": 150.0, # unitree_ros
    # ".*ankle_pitch_joint": 20.0,
    # ".*ankle_roll_joint": 20.0,
    # reduced on 0630
    ".*ankle_pitch_joint": 10.0,
    ".*ankle_roll_joint": 2.0,
    ".*_shoulder_.*": 40.0,
    ".*_elbow_joint": 40.0,
    ".*_wrist_.*_joint": 4.0,
}
G1_29DOF_CFG.actuators["base_legs"].damping = {
    ".*_hip_yaw_joint": 6.0,
    ".*_hip_roll_joint": 6.0,
    ".*_hip_pitch_joint": 6.0,
    ".*_knee_joint": 6.0,
    "waist_yaw_joint": 5.0, # unitree_ros
    "waist_roll_joint": 5.0, # unitree_ros
    "waist_pitch_joint": 5.0, # unitree_ros
    ".*_shoulder_.*": 2.0,
    ".*_elbow_joint": 2.0,
    ".*_ankle_.*_joint": 1.0,
    ".*_wrist_.*_joint": 0.5,
}

G1_GMT_29DOF_CFG = copy.deepcopy(_G1_29DOF_CFG)
G1_GMT_29DOF_CFG.actuators["base_legs"].stiffness = {
    ".*_hip_.*_joint": 100.0,
    ".*_knee_joint": 150.0,
    ".*_ankle_.*_joint": 40.0,
    "waist_.*_joint": 150.0,
    ".*_shoulder_.*_joint": 40.0,
    ".*_elbow_joint": 40.0,
    ".*_wrist_.*_joint": 4.0,
}
G1_GMT_29DOF_CFG.actuators["base_legs"].damping = {
    ".*_hip_.*_joint": 2.0,
    ".*_knee_joint": 4.0,
    ".*_ankle_.*_joint": 2.0,
    "waist_.*_joint": 4.0,
    ".*_shoulder_.*_joint": 5.0,
    ".*_elbow_joint": 5.0,
    ".*_wrist_.*_joint": 0.5,
}

G1_HV_29DOF_CFG = copy.deepcopy(_G1_29DOF_CFG)
G1_HV_29DOF_CFG.actuators["base_legs"].stiffness = {
    ".*_hip_.*_joint": 100.0,
    ".*_knee_joint": 200.0,
    ".*_ankle_.*_joint": 20.0,
    "waist_.*_joint": 200.0,
    ".*_shoulder_.*_joint": 40.0,
    ".*_elbow_joint": 40.0,
    ".*_wrist_.*_joint": 4.0,
}
G1_HV_29DOF_CFG.actuators["base_legs"].damping = {
    ".*_hip_.*_joint": 2.0,
    ".*_knee_joint": 4.0,
    ".*_ankle_.*_joint": 0.2,
    "waist_.*_joint": 4.0,
    ".*_shoulder_.*_joint": 5.0,
    ".*_elbow_joint": 5.0,
    ".*_wrist_.*_joint": 0.5,
}


BOOSTER_T1_CFG = ArticulationCfg( # no wrist pitch and yaw
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}" + "/t1/{ROBOT_TYPE}.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=6,
            solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.72),
        joint_pos={
            # Head
            "head_yaw_joint": 0.0,
            "head_pitch_joint": 0.0,
            # Arm
            ".*_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": -1.35,
            "right_shoulder_roll_joint": 1.35,
            ".*_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -0.5,
            "right_elbow_joint": 0.5,
            # Waist
            "waist_yaw_joint": 0.0,
            # Leg
            ".*_hip_pitch_joint": -0.20,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=".*",
            effort_limit_sim={
                ".*_hip_pitch_joint": 24.0,
                ".*_hip_roll_joint": 30.0,
                ".*_hip_yaw_joint": 30.0,
                ".*_knee_joint": 60.0,
                "waist_yaw_joint": 30.0,

                ".*_ankle_pitch_joint": 24.0,
                ".*_ankle_roll_joint": 15.0,

                ".*_shoulder_pitch_joint": 18.0,
                ".*_shoulder_roll_joint": 18.0,
                ".*_shoulder_yaw_joint": 18.0,
                ".*_elbow_joint": 18.0,

                "head_yaw_joint": 10.0,
                "head_pitch_joint": 10.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 45.0,
                ".*_hip_roll_joint": 30.0,
                ".*_hip_yaw_joint": 30.0,
                ".*_knee_joint": 60.0,
                "waist_yaw_joint": 30.0,

                ".*_ankle_pitch_joint": 18.8,
                ".*_ankle_roll_joint": 12.4,

                ".*_shoulder_pitch_joint": 18.8,
                ".*_shoulder_roll_joint": 18.8,
                ".*_shoulder_yaw_joint": 18.8,
                ".*_elbow_joint": 18.8,

                "head_yaw_joint": 10.0,
                "head_pitch_joint": 10.0,
            },
            stiffness={
                ".*_hip_pitch_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_yaw_joint": 200.0,
                ".*_knee_joint": 200.0,
                "waist_yaw_joint": 200.0,

                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,

                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_joint": 40.0,

                "head_yaw_joint": 10.0,
                "head_pitch_joint": 10.0,
            },
            damping={
                ".*_hip_pitch_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_yaw_joint": 5.0,
                ".*_knee_joint": 5.0,
                "waist_yaw_joint": 5.0,

                ".*_ankle_pitch_joint": 1.0,
                ".*_ankle_roll_joint": 1.0,

                ".*_shoulder_pitch_joint": 10.0,
                ".*_shoulder_roll_joint": 10.0,
                ".*_shoulder_yaw_joint": 10.0,
                ".*_elbow_joint": 10.0,

                "head_yaw_joint": 10.0,
                "head_pitch_joint": 10.0,
            },
            armature=0.01,
            friction=0.01,
        ),
    },
)
