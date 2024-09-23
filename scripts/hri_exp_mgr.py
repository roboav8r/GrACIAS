#!/usr/bin/env python3

import os
import time
import json
import subprocess
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rclpy.wait_for_message import wait_for_message

import rosbag2_py
from rosidl_runtime_py.utilities import get_message

from foxglove_msgs.msg import SceneUpdate
from tracking_msgs.msg import Tracks3D
from std_srvs.srv import Empty

class HRIExpManager(Node):
    def __init__(self):
        super().__init__('hri_exp_mgr_node')

        # Declare and load ROS parameters from .yaml file
        self.declare_parameter('exp_configs', rclpy.Parameter.Type.STRING_ARRAY )
        self.declare_parameter('mcap_dir', rclpy.Parameter.Type.STRING )
        self.exp_configs = self.get_parameter('exp_configs').get_parameter_value().string_array_value
        self.mcap_dir = Path.home() / self.get_parameter('mcap_dir').get_parameter_value().string_value

        # Non-ros parameters
        self.package_dir = get_package_share_directory('situated_interaction')

        # Create ROS objects
        self.reset_spatial_tracker_client = self.create_client(Empty, 'tbd_tracker_node/reset_tracker')
        self.reset_semantic_tracker_client = self.create_client(Empty, 'semantic_fusion_node/reset')
        self.reconf_semantic_tracker_client = self.create_client(Empty, 'semantic_fusion_node/reconfigure')
        self.empty_req = Empty.Request()
    
    def run_experiments(self):

        # Reconfigure tracker
        for exp in self.exp_configs:

            # Load the experimental configuration for the tracker
            exp_path = os.path.join(self.package_dir,exp)
            self.exp_name = os.path.splitext(os.path.split(exp_path)[-1])[0]
            self.get_logger().info("Loading tracker experiment configuration: %s" % (self.exp_name))
            self.get_logger().info("ros2 param load /semantic_fusion_node %s" % os.path.join(self.package_dir,exp_path))
            subprocess.run(["ros2", "param", "load", "/semantic_fusion_node", os.path.join(self.package_dir,exp_path)])
            
            # Reconfigure semantic node
            self.future = self.reconf_semantic_tracker_client.call_async(self.empty_req)
            rclpy.spin_until_future_complete(self, self.future,timeout_sec=5)
            while self.future.done() is False:
                self.get_logger().info("Could not reconfigure, retrying")
                self.future = self.reconf_semantic_tracker_client.call_async(self.empty_req)
                rclpy.spin_until_future_complete(self, self.future,timeout_sec=5)

            ### Play mcap files as if live
            last_root = None

            # Traverse filepath
            for root, _, files in os.walk(self.mcap_dir):

                # Examine each file
                for file in files:
                    
                    # If it is an mcap, examine
                    if os.path.splitext(file)[-1] == '.mcap':

                        # Get labels based on filepath
                        labels = os.path.normpath(root).split(os.sep)
                        cmd_actual = labels[-1]
                        cmd_mode_actual = labels[-2]
                        role_actual = labels[-3]
                        scene_actual = labels[-4]
                        iteration = os.path.splitext(file)[-2].split('_')[-1]

                        # scene_actual_idx = actual_scene_labels.index(scene_actual)

                        if root != last_root: # Avoid playing mcap file multiple times

                            time.sleep(1)

                            # Reset tracker
                            self.future = self.reset_spatial_tracker_client.call_async(self.empty_req)
                            rclpy.spin_until_future_complete(self, self.future,timeout_sec=5)
                            while self.future.done() is False:
                                self.get_logger().info("Could not reset tracker, retrying")
                                self.future = self.reset_spatial_tracker_client.call_async(self.empty_req)
                                rclpy.spin_until_future_complete(self, self.future,timeout_sec=5)

                            # Reset semantic node
                            self.future = self.reconf_semantic_tracker_client.call_async(self.empty_req)
                            rclpy.spin_until_future_complete(self, self.future,timeout_sec=5)
                            while self.future.done() is False:
                                self.get_logger().info("Could not reset, retrying")
                                self.future = self.reconf_semantic_tracker_client.call_async(self.empty_req)
                                rclpy.spin_until_future_complete(self, self.future,timeout_sec=5)

                            bag_play_cmd = "ros2 bag play %s --clock 1" % root
                            self.get_logger().info(bag_play_cmd)
                            try:
                                result = subprocess.check_output(bag_play_cmd, shell=True, text=True)
                                self.get_logger().info(result)
                            except subprocess.CalledProcessError as e:
                                self.get_logger().info(f"Error executing command: {e}")

                        last_root = root
                
            


def main(args=None):
    rclpy.init(args=args)

    # Create experiment manager
    hri_exp_mgr = HRIExpManager()
    hri_exp_mgr.run_experiments()

    # Shut down the node
    hri_exp_mgr.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()