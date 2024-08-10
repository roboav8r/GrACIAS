#!/usr/bin/env python3

import pandas as pd
import numpy as np

import rclpy
from rclpy.node import Node

from std_srvs.srv import Empty

from situated_hri_interfaces.msg import CategoricalDistribution
from situated_hri_interfaces.srv import RecordEpoch

class RecSceneResultsNode(Node):

    def __init__(self):

        super().__init__('rec_scene_results_node')

        self.audio_scene_sub = self.create_subscription(CategoricalDistribution, 'bayes_audio_scene_est/fused_scene_category', lambda msg: self.scene_callback(msg, 'audio'), 10)
        self.clip_scene_sub = self.create_subscription(CategoricalDistribution, 'bayes_clip_scene_est/fused_scene_category', lambda msg: self.scene_callback(msg, 'clip'), 10)
        self.fused_scene_sub = self.create_subscription(CategoricalDistribution, 'bayes_fused_scene_est/fused_scene_category', lambda msg: self.scene_callback(msg, 'fused'), 10)

        self.record_epoch_srv = self.create_service(RecordEpoch, '~/record_epoch', self.record_epoch)
        self.stop_record_srv = self.create_service(Empty, '~/stop_recording', self.stop_recording)

        self.results_columns = ['scene','role','cmd_mode','cmd','iteration','scene_mode','n_updates','t_elapsed','scene_est','scene_conf']
        self.results_df = pd.DataFrame(columns = self.results_columns)
    
        self.scene_est_count_dict = {'audio': 0, 'clip': 0, 'fused': 0}


    def scene_callback(self, msg, scene_mode):
        # self.get_logger().info("From mode %s got msg: %s" % (scene_mode, msg))

        # Increment update count
        self.scene_est_count_dict[scene_mode] += 1
        
        # Add experiment result to dataframe
        # TODO - time elapsed?
        result_df = pd.DataFrame([[self.scene, self.role, self.cmd_mode, self.cmd, self.iteration, scene_mode, self.scene_est_count_dict[scene_mode], 0., msg.categories[np.argmax(msg.probabilities)],msg.probabilities[np.argmax(msg.probabilities)]]], columns=self.results_columns)

        self.results_df = pd.concat([self.results_df, result_df],axis=0, ignore_index=True)


    def record_epoch(self, req, resp):

        # Get epoch information
        self.scene = req.scene
        self.role = req.role
        self.cmd_mode = req.cmd_mode
        self.cmd = req.cmd
        self.iteration = req.iteration

        self.epoch_start_time = req.epoch_start_time

        # Reset estimate counts
        self.scene_est_count_dict = {'audio': 0, 'clip': 0, 'fused': 0}

        self.get_logger().info("Recording epoch: %s/%s/%s/%s/%s" % (self.scene, self.role, self.cmd_mode, self.cmd, self.iteration))

        return resp

    def stop_recording(self, _, resp):

        self.results_df.to_csv('recorded_results.csv',columns = self.results_columns)

        return resp



def main(args=None):
    rclpy.init(args=args)

    record_scene_results_node = RecSceneResultsNode()
    rclpy.spin(record_scene_results_node)

    record_scene_results_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()