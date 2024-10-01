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

        self.declare_parameter('config_name', rclpy.Parameter.Type.STRING)    
        self.scene_config_name =  self.get_parameter('config_name').get_parameter_value().string_value

        self.audio_scene_est_sub = self.create_subscription(CategoricalDistribution, 'audio_scene_category', lambda msg: self.scene_est_count_callback(msg, 'audio'), 10)
        self.clip_scene_est_sub = self.create_subscription(CategoricalDistribution, 'clip_scene_category', lambda msg: self.scene_est_count_callback(msg, 'clip'), 10)

        self.audio_scene_sub = self.create_subscription(CategoricalDistribution, 'bayes_audio_scene_est/fused_scene_category', lambda msg: self.scene_callback(msg, 'audio'), 10)
        self.clip_scene_sub = self.create_subscription(CategoricalDistribution, 'bayes_clip_scene_est/fused_scene_category', lambda msg: self.scene_callback(msg, 'clip'), 10)
        self.fused_scene_sub = self.create_subscription(CategoricalDistribution, 'bayes_fused_scene_est/fused_scene_category', lambda msg: self.scene_callback(msg, 'fused'), 10)

        self.record_epoch_srv = self.create_service(RecordEpoch, '~/record_epoch', self.record_epoch)
        self.stop_record_srv = self.create_service(Empty, '~/stop_recording', self.stop_recording)

        self.results_columns = ['scene','role','cmd_mode','cmd','scene_estimation_mode','n_audio_updates','n_visual_updates','scene_est','scene_conf']
        self.results_df = pd.DataFrame(columns = self.results_columns)
    
        self.scene_est_count_dict = {'audio': 0, 'clip': 0, 'fused': 0}

    def scene_est_count_callback(self, _, scene_mode):
        # Increment update count
        self.scene_est_count_dict[scene_mode] += 1


    def scene_callback(self, msg, scene_mode):
        
        # Add experiment result to dataframe
        result_df = pd.DataFrame([[self.scene, self.role, self.cmd_mode, self.cmd, scene_mode, self.scene_est_count_dict['audio'], self.scene_est_count_dict['clip'], msg.categories[np.argmax(msg.probabilities)],msg.probabilities[np.argmax(msg.probabilities)]]], columns=self.results_columns)

        self.results_df = pd.concat([self.results_df, result_df],axis=0, ignore_index=True)


    def record_epoch(self, req, resp):

        # Get epoch information
        self.scene = req.scene
        self.role = req.role
        self.cmd_mode = req.cmd_mode
        self.cmd = req.cmd

        self.epoch_start_time = req.epoch_start_time

        # Reset estimate counts
        self.scene_est_count_dict = {'audio': 0, 'clip': 0, 'fused': 0}

        self.get_logger().info("Recording epoch: %s/%s/%s/%s" % (self.scene, self.role, self.cmd_mode, self.cmd))

        return resp

    def stop_recording(self, _, resp):

        self.results_df.to_csv("src/situated_interaction/results/exp1a_scene_recognition/scene_results_%s.csv" % self.scene_config_name, columns = self.results_columns)

        return resp



def main(args=None):
    rclpy.init(args=args)

    record_scene_results_node = RecSceneResultsNode()
    rclpy.spin(record_scene_results_node)

    record_scene_results_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()