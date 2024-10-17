#!/usr/bin/env python3

import pandas as pd
import numpy as np

import os
from pathlib import Path

import rclpy
from rclpy.node import Node

from std_srvs.srv import Empty

from situated_hri_interfaces.msg import CategoricalDistribution, HierarchicalCommands
from situated_hri_interfaces.srv import RecordEpoch

from situated_interaction.utils import time_to_float

class RecHierarchicalCmdResultsNode(Node):

    def __init__(self):

        super().__init__('rec_hierarchical_cmd_results_node')

        self.hierarchical_cmd_sub = self.create_subscription(HierarchicalCommands, 'hierarchical_commands', self.hierarchical_cmd_callback, 10)

        self.record_epoch_srv = self.create_service(RecordEpoch, '~/record_epoch', self.record_epoch)
        self.stop_record_srv = self.create_service(Empty, '~/stop_recording', self.stop_recording)

        self.declare_parameter('result_dir', rclpy.Parameter.Type.STRING )
        self.result_dir = Path.home() / self.get_parameter('result_dir').get_parameter_value().string_value

        self.results_columns = ['config','scene','role','cmd_mode','cmd','role rec method','cmd rec method','stamp','object id','object class','estimated role','estimated command']
        self.results_df = pd.DataFrame(columns = self.results_columns)
    
    def hierarchical_cmd_callback(self, msg):

        # Check and see if tracked object at 3,0
        obj_id = None
        obj_class = None
        est_role = None
        est_cmd = None

        for cmd in msg.commands:
            # self.get_logger().info('track: class string %s, x %s, y %s, dist %s' % (trk.class_string, trk.pose.pose.position.x, trk.pose.pose.position.y,np.linalg.norm([trk.pose.pose.position.x - 3, trk.pose.pose.position.y - 0])))
            if cmd.class_string=='person' and np.linalg.norm([cmd.pose.position.x - 3, cmd.pose.position.y - 0]) < .5:
                obj_id = cmd.object_id
                obj_class = cmd.class_string

                for kv in cmd.states:
                    if kv.key =='role':
                        est_role = kv.value
                
                est_cmd = cmd.comms


        # Add experiment result to dataframe
        time = rclpy.time.Time.from_msg(msg.header.stamp)
        time_sec, time_nsec = time.seconds_nanoseconds()
        stamp = time_to_float(time_sec, time_nsec)
        result_df = pd.DataFrame([[self.exp_config, self.scene, self.role, self.cmd_mode, self.cmd, self.role_rec_method, self.cmd_rec_method, stamp, obj_id, obj_class, est_role, est_cmd]], columns=self.results_columns)

        self.results_df = pd.concat([self.results_df, result_df],axis=0, ignore_index=True)


    def record_epoch(self, req, resp):

        # Get epoch information
        self.exp_config = req.config
        self.scene = req.scene
        self.role = req.role
        self.cmd_mode = req.cmd_mode
        self.cmd = req.cmd
        self.cmd_rec_method = req.cmd_rec_method
        self.role_rec_method = req.role_rec_method

        self.epoch_start_time = req.epoch_start_time

        return resp

    def stop_recording(self, _, resp):
        self.results_df.to_csv(os.path.join(self.result_dir,'hierarchical_cmd_results.csv'),columns = self.results_columns)
        return resp


def main(args=None):
    rclpy.init(args=args)

    record_hierarchical_cmd_results_node = RecHierarchicalCmdResultsNode()
    rclpy.spin(record_hierarchical_cmd_results_node)

    record_hierarchical_cmd_results_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()