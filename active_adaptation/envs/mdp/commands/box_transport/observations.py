from active_adaptation.envs.mdp.commands.box_transport.command import BoxTransport
from active_adaptation.envs.mdp.base import Observation as BaseObservation

from isaaclab.utils.math import (
    quat_apply_inverse,
    quat_mul,
    quat_conjugate,
    matrix_from_quat,
)
from active_adaptation.utils.math import batchify
quat_apply_inverse = batchify(quat_apply_inverse)

BoxTransportObservation = BaseObservation[BoxTransport]

class target_box_pos_b_xy(BoxTransportObservation):
    def compute(self):
        target_box_pos_w = self.command_manager.box_final_pos_w
        robot_root_pos_w = self.command_manager.robot_root_pos_w
        robot_root_quat_w = self.command_manager.robot_root_quat_w
        target_box_pos_b = quat_apply_inverse(
            robot_root_quat_w, target_box_pos_w - robot_root_pos_w
        )
        return target_box_pos_b[:, :2]

class target_box_pos_b(BoxTransportObservation):
    def compute(self):
        target_box_pos_w = self.command_manager.box_final_pos_w
        robot_root_pos_w = self.command_manager.robot_root_pos_w
        robot_root_quat_w = self.command_manager.robot_root_quat_w
        target_box_pos_b = quat_apply_inverse(
            robot_root_quat_w, target_box_pos_w - robot_root_pos_w
        )
        return target_box_pos_b

class target_box_ori_b(BoxTransportObservation):
    def compute(self):
        target_box_quat_w = self.command_manager.box_final_quat_w
        robot_root_quat_w = self.command_manager.robot_root_quat_w
        target_box_quat_b = quat_mul(
            quat_conjugate(robot_root_quat_w), target_box_quat_w
        )
        target_box_ori_b = matrix_from_quat(target_box_quat_b)
        return target_box_ori_b[:, :2].reshape(self.num_envs, -1)
