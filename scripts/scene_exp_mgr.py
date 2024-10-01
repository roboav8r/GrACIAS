#!/usr/bin/env python3

import os
import time
import json
import subprocess
import re

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
from rcl_interfaces.srv import GetParameters

from situated_hri_interfaces.srv import RecordEpoch

from situated_interaction.utils import try_to_declare_parameter

class SceneExpManager(Node):
    def __init__(self):
        super().__init__('scene_exp_mgr_node')

        # Declare and load ROS parameters from .yaml file
        self.declare_parameter('exp_configs', rclpy.Parameter.Type.STRING_ARRAY )
        self.declare_parameter('mcap_dir', rclpy.Parameter.Type.STRING )
        self.exp_configs = self.get_parameter('exp_configs').get_parameter_value().string_array_value
        self.mcap_dir = Path.home() / self.get_parameter('mcap_dir').get_parameter_value().string_value

        # Non-ros parameters
        self.package_dir = get_package_share_directory('situated_interaction')

        # Create ROS objects
        # Reset CLIP msg count (downsampling)
        self.reset_clip_scene_client = self.create_client(Empty, 'clip_scene_rec/reset')
        self.reconf_clip_scene_client = self.create_client(Empty, 'clip_scene_rec/reconfigure')

        # Reset scene estimates from different modalities
        self.reset_bayes_audio_scene_client = self.create_client(Empty, 'bayes_audio_scene_est/reset') 
        self.reset_bayes_clip_scene_client = self.create_client(Empty, 'bayes_clip_scene_est/reset')
        self.reset_bayes_fused_scene_client = self.create_client(Empty, 'bayes_fused_scene_est/reset')

        self.start_recording_client = self.create_client(RecordEpoch, 'record_scene_results_node/record_epoch')
        self.stop_recording_client = self.create_client(Empty, 'record_scene_results_node/stop_recording')
        self.empty_req = Empty.Request()
    
    def run_experiments(self):

        # Reconfigure tracker
        for exp in self.exp_configs:

            # Load the experimental configuration for the tracker
            exp_path = os.path.join(self.package_dir,exp)
            self.exp_name = os.path.splitext(os.path.split(exp_path)[-1])[0]
            self.get_logger().info("Loading scene recognition experiment configuration: %s" % (self.exp_name))
            subprocess.run(["ros2", "param", "load", "/clip_scene_rec", os.path.join(self.package_dir,exp_path)])
            
            # Reconfigure clip scene rec node
            self.future = self.reconf_clip_scene_client.call_async(self.empty_req)
            rclpy.spin_until_future_complete(self, self.future)

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

                        if root != last_root: # Avoid playing mcap file multiple times

                            self.get_logger().info("Resetting nodes and playing data")

                            # Reset scene rec nodes        
                            bayes_audio_future = self.reset_bayes_audio_scene_client.call_async(Empty.Request())
                            rclpy.spin_until_future_complete(self, bayes_audio_future,timeout_sec=10)

                            bayes_clip_future = self.reset_bayes_clip_scene_client.call_async(Empty.Request())
                            rclpy.spin_until_future_complete(self, bayes_clip_future,timeout_sec=10)

                            bayes_fused_future = self.reset_bayes_fused_scene_client.call_async(Empty.Request())
                            rclpy.spin_until_future_complete(self, bayes_fused_future,timeout_sec=10)
                            
                            clip_future = self.reset_clip_scene_client.call_async(Empty.Request())
                            rclpy.spin_until_future_complete(self, clip_future,timeout_sec=10)

                            # Start recording epoch
                            record_epoch_req = RecordEpoch.Request()
                            record_epoch_req.config = exp
                            record_epoch_req.scene = scene_actual
                            record_epoch_req.role = role_actual
                            record_epoch_req.cmd_mode = cmd_mode_actual
                            record_epoch_req.cmd = cmd_actual

                            self.future = self.start_recording_client.call_async(record_epoch_req)
                            rclpy.spin_until_future_complete(self, self.future,timeout_sec=5)
                           
                            bag_play_cmd = "ros2 bag play %s --clock 1" % root
                            self.get_logger().info(bag_play_cmd)
                            try:
                                result = subprocess.check_output(bag_play_cmd, shell=True, text=True)
                                self.get_logger().info(result)
                            except subprocess.CalledProcessError as e:
                                self.get_logger().info(f"Error executing command: {e}")

                        last_root = root

            self.future = self.stop_recording_client.call_async(self.empty_req)
            rclpy.spin_until_future_complete(self, self.future,timeout_sec=5)
                
def main(args=None):
    rclpy.init(args=args)

    # Create experiment manager
    scene_exp_mgr = SceneExpManager()
    scene_exp_mgr.run_experiments()

    # Shut down the node
    scene_exp_mgr.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()