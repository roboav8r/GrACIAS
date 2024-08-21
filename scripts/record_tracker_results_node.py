#!/usr/bin/env python3

import pandas as pd
import numpy as np

import rclpy
from rclpy.node import Node

from std_srvs.srv import Empty

from situated_hri_interfaces.msg import CategoricalDistribution
from situated_hri_interfaces.srv import RecordEpoch

from tracking_msgs.msg import Detections3D, Tracks3D

class RecTrackerResultsNode(Node):

    def __init__(self):

        super().__init__('rec_hri_results_node')

        self.oakd_det_sub = self.create_subscription(Detections3D, 'converted_img_detections', lambda msg: self.det_callback(msg, 'oakd'), 10)
        self.oakd_tracker_sub = self.create_subscription(Tracks3D, 'oakd_tracker_node/tracks', lambda msg: self.track_callback(msg, 'oakd'), 10)

        self.record_epoch_srv = self.create_service(RecordEpoch, '~/record_epoch', self.record_epoch)
        self.stop_record_srv = self.create_service(Empty, '~/stop_recording', self.stop_recording)

        self.results_columns = ['scene','role','cmd_mode','cmd','iteration','tracker_type','n_det_msgs','n_trk_msgs','n_dets','trk_id','x_est','y_est']
        self.results_df = pd.DataFrame(columns = self.results_columns)
    
        self.n_det_msgs = 0
        self.n_track_msgs = 0

    def det_callback(self, msg, tracker_type):
        self.n_det_msgs += 1

    def track_callback(self, msg, tracker_type):
        # self.get_logger().info("From mode %s got msg: %s" % (scene_mode, msg))

        # Increment update count
        self.n_track_msgs += 1

        # Check and see if tracked object at 3,0
        obj_x = None
        obj_y = None
        obj_id = None
        n_dets = None

        for trk in msg.tracks:
            # self.get_logger().info('track: class string %s, x %s, y %s, dist %s' % (trk.class_string, trk.pose.pose.position.x, trk.pose.pose.position.y,np.linalg.norm([trk.pose.pose.position.x - 3, trk.pose.pose.position.y - 0])))
            if trk.class_string=='person' and np.linalg.norm([trk.pose.pose.position.x - 3, trk.pose.pose.position.y - 0]) < .5:
                obj_x = trk.pose.pose.position.x
                obj_y = trk.pose.pose.position.y
                obj_id = trk.track_id

                for md in msg.metadata:
                    if md.key == 'num_dets_rcvd':
                        n_dets = int(md.value)  

        # Add experiment result to dataframe
        result_df = pd.DataFrame([[self.scene, self.role, self.cmd_mode, self.cmd, self.iteration, tracker_type, self.n_det_msgs, self.n_track_msgs, n_dets, obj_id, obj_x, obj_y]], columns=self.results_columns)

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
        self.n_det_msgs = 0
        self.n_track_msgs = 0

        self.get_logger().info("Recording epoch: %s/%s/%s/%s/%s" % (self.scene, self.role, self.cmd_mode, self.cmd, self.iteration))

        return resp

    def stop_recording(self, _, resp):

        self.results_df.to_csv('track_results.csv',columns = self.results_columns)

        return resp


def main(args=None):
    rclpy.init(args=args)

    record_tracker_results_node = RecTrackerResultsNode()
    rclpy.spin(record_tracker_results_node)

    record_tracker_results_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()