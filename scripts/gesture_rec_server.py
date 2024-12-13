#!/usr/bin/env python3
import sys

import torch

import rclpy
from rclpy.node import Node

from situated_hri_interfaces.msg import CategoricalDistribution
from situated_hri_interfaces.srv import ObjectVisRec

sys.path.append('/home/john_duncan/gesture_ws/src/gesture_recognition/third/include') # TODO - get this path programmatically

from hri_cacti_gestures.classifier import LSTMGestureClassifier

class GestureRecServer(Node):

    def __init__(self):

        # ROS objects
        super().__init__('gesture_rec_server')

        # ROS objects
        self.srv = self.create_service(ObjectVisRec, 'clip_object_rec', self.gesture_rec_callback)

        # Gesture data parameters TODO - make ros params
        self.window_length = 24 # approximately 1.5 second
        self.num_pose_keypoints = 17
        self.dim_pose_keypoints = 3
        self.command_list = ['Advance', 'Attention', 'FollowMe', 'Halt', 'MoveForward', 'MoveInReverse', 'Rally']
        self.command_idx = [0, 1, 2, 3, 4, 5, 6]
        self.command_idx_tensor = torch.tensor(self.command_idx)
        self.model_path = '/home/john_duncan/storage/models/syn_gesture_rec_wl_24/lstm_ges_kp_class_2024.12.04_00.47.41.pt'

        self.model_input_dim = self.num_pose_keypoints*self.dim_pose_keypoints
        self.model_output_dim = len(self.command_idx)

        # Torch objects
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LSTMGestureClassifier(self.model_input_dim, self.model_output_dim)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(self.torch_device)

    def gesture_rec_callback(self, req, resp):

        input_tensor = torch.Tensor(req.keypoint_data)
        input_tensor.reshape(1, self.window_length, self.model_input_dim)
        input_tensor.to(torch.float32)
        output_tensor = self.model(input_tensor)

        resp.stamp = req.stamp
        resp.object_id = req.object_id
        resp.comms.variable = "gesture"
        resp.comms.categories = self.command_list
        resp.comms.probabilities = output_tensor.tolist()

def main(args=None):
    rclpy.init(args=args)

    gesture_rec_server = GestureRecServer()
    rclpy.spin(gesture_rec_server)

    gesture_rec_server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()