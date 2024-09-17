#!/usr/bin/env python3

# import time
import numpy as np
import gtsam

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.time import Time, Duration
import tf2_ros

from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point
from ar_track_alvar_msgs.msg import AlvarMarker, AlvarMarkers
from tracking_msgs.msg import Tracks3D
from ros_audition.msg import SpeechAzSources
from foxglove_msgs.msg import SceneUpdate

from situated_hri_interfaces.msg import Auth, Comm, Comms, Identity, CategoricalDistribution
from situated_hri_interfaces.srv import ObjectVisRec

from situated_interaction.assignment import compute_az_match, compute_pos_match, compute_az_from_pos, compute_delta_az, solve_assignment_matrix
from situated_interaction.datatypes import DiscreteVariable, SemanticObject
from situated_interaction.output import foxglove_visualization
from situated_interaction.utils import pmf_to_spec, normalize_vector, load_object_params

class SemanticTrackerNode(Node):

    def __init__(self):
        super().__init__('semantic_tracker_node')

        sub_cb_group = MutuallyExclusiveCallbackGroup()
        timer_cb_group = MutuallyExclusiveCallbackGroup()
        client_cb_group = MutuallyExclusiveCallbackGroup()
        
        # Subscribe to object tracks
        self.subscription_tracks = self.create_subscription(
            Tracks3D,
            'tracks',
            self.tracks_callback,
            10, callback_group=sub_cb_group)
        self.subscription_tracks  # prevent unused variable warning

        # Subscribe to localized speech
        self.subscription_speech = self.create_subscription(
            SpeechAzSources,
            'speech_az_sources',
            self.speech_callback,
            10, callback_group=sub_cb_group)
        self.subscription_speech  # prevent unused variable warning

        # Subscribe to AR markers
        self.subscription_ar = self.create_subscription(
            AlvarMarkers,
            'ar_pose_marker',
            self.ar_callback,
            10, callback_group=sub_cb_group)
        self.subscription_ar  # prevent unused variable warning

        # Define pubs
        self.semantic_scene_pub = self.create_publisher(
            SceneUpdate,
            'semantic_scene',
            10)
        
        # Create timer
        self.declare_parameter('loop_time_sec', rclpy.Parameter.Type.DOUBLE)
        self.loop_time_sec = self.get_parameter('loop_time_sec').get_parameter_value().double_value
        self.timer = self.create_timer(self.loop_time_sec, self.timer_callback, callback_group=timer_cb_group)
        self.service_timeout = .25

        # Create service client
        self.clip_client = self.create_client(ObjectVisRec, 'clip_object_rec', callback_group=client_cb_group)
        while not self.clip_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.clip_req = ObjectVisRec.Request()

        # Create transform buffer/listener
        self.declare_parameter('tracker_frame', rclpy.Parameter.Type.STRING)
        self.declare_parameter('mic_frame', rclpy.Parameter.Type.STRING)
        self.declare_parameter('artag_frame', rclpy.Parameter.Type.STRING)
        self.tracker_frame = self.get_parameter('tracker_frame').get_parameter_value().string_value
        self.mic_frame = self.get_parameter('mic_frame').get_parameter_value().string_value
        self.artag_frame = self.get_parameter('artag_frame').get_parameter_value().string_value

        self.tf_buffer = tf2_ros.buffer.Buffer(Duration(seconds=0.5))
        self.tf_listener = tf2_ros.transform_listener.TransformListener(self.tf_buffer, self)
        self.tf_buffer.can_transform(self.mic_frame,self.tracker_frame,Duration(seconds=10)) # Block until mic -> tracker transform available
        self.tf_buffer.can_transform(self.artag_frame,self.tracker_frame,Duration(seconds=10)) # Block until artag -> tracker transform available
        
        # Generate object att/state variable dictionary
        load_object_params(self)

        # Define member variables
        self.semantic_objects = {}
        self.tracks_msg = Tracks3D()
        self.scene_out_msg = SceneUpdate()

    def send_obj_clip_req(self, id, atts_to_est, states_to_est, est_comms):
        self.clip_req = ObjectVisRec.Request()
        self.clip_req.object_id = id
        self.clip_req.class_string = self.semantic_objects[id].class_string
        self.clip_req.attributes_to_estimate = atts_to_est
        self.clip_req.states_to_estimate = states_to_est
        self.clip_req.estimate_comms = est_comms
        self.clip_req.image = self.semantic_objects[id].image
        self.clip_req.stamp = self.semantic_objects[id].stamp
        resp = self.clip_client.call(self.clip_req)

        for att_dist in resp.attributes:
            att = att_dist.variable
            self.semantic_objects[id].attributes[att].update(att_dist.probabilities, Time.from_msg(resp.stamp))

        for state_dist in resp.states:
            state = state_dist.variable
            self.semantic_objects[id].states[state].update(state_dist.probabilities, Time.from_msg(resp.stamp))

        if est_comms:
            self.semantic_objects[id].update_gesture_comms(resp.comms, self)

        self.semantic_objects[id].image_available = False

    def timer_callback(self):
        start_time = self.get_clock().now()

        # For each object, check if state is stale. If so, send state update request.
        for id in self.semantic_objects.keys():

            obj = self.semantic_objects[id]

            # self.get_logger().info("Checking object %s with atts %s, states: %s" % (id, str(obj.attributes),str(obj.states)))

            if obj.new_image_available==False:
                continue

            states_to_est = []
            atts_to_est = []

            for att in obj.attributes:
                if obj.attributes[att].needs_update(start_time):
                    atts_to_est.append(att)
                
            for state in obj.states:
                if obj.states[state].needs_update(start_time):
                    states_to_est.append(state)

            if self.object_params[obj.class_string]['comms']['gesture_descriptions'] != ['']:
                est_comms = True

            if atts_to_est or states_to_est or est_comms:
                self.get_logger().info("atts to est: %s" % atts_to_est)
                self.get_logger().info("states to est: %s" % states_to_est)
                self.send_obj_clip_req(id, atts_to_est, states_to_est, est_comms)

        foxglove_visualization(self)

        # self.get_logger().info("Timer callback time (s): %s" % ((self.get_clock().now() - start_time).nanoseconds/10**9))

    def tracks_callback(self, msg):
        self.tracks_msg = msg

        # Temporary list for this callback
        self.semantic_object_ids = []

        # add new tracked objects from incoming message
        for tracked_object in msg.tracks:

            if tracked_object.class_string not in self.objects_of_interest:
                continue

            # Add 
            self.semantic_object_ids.append(tracked_object.track_id)

            # Initialize object and add to dict if not currently tracked
            if tracked_object.track_id not in self.semantic_objects.keys():
                self.semantic_objects[tracked_object.track_id] = SemanticObject(tracked_object, self.object_params[tracked_object.class_string], self.tracks_msg.header.frame_id) # TODO - pass frame_id some other way
            else:
                # Update existing track
                self.semantic_objects[tracked_object.track_id].update_spatial_state(tracked_object)

        # Remove untracked objects
        objects_temp = self.semantic_objects
        self.semantic_objects = {}
        self.semantic_objects = {key: val for key, val in objects_temp.items() if key in self.semantic_object_ids}

        foxglove_visualization(self)


    def speech_callback(self, msg):

        # If there are recognized speech sources
        if msg.sources:

            # Compute the assignment matrix
            assignment_matrix = np.zeros((len(msg.sources),len(self.semantic_objects.keys())))

            for ii, speech_src in enumerate(msg.sources):
                for jj, object_key in enumerate(self.semantic_objects.keys()):
                    object = self.semantic_objects[object_key]

                    semantic_object_az = compute_az_from_pos(self.tf_buffer,msg.header.frame_id,self.tracker_frame,object)
                    delta_az = compute_delta_az(speech_src.azimuth, semantic_object_az)
                    assignment_matrix[ii,jj] += delta_az

            # Solve assignment matrix
            assignments = solve_assignment_matrix('greedy', assignment_matrix, self.ang_match_threshold)

            # Handle matches
            for assignment in assignments:

                object_idx = assignment[1]
                object_key = self.semantic_objects.keys()[object_idx]
                speech_idx = assignment[0]

                self.semantic_objects[object_key].update_verbal_comms(msg.sources[speech_idx].transcript, msg.sources[speech_idx].confidence, self)

            # Handle objects with no speech
            for jj, object_key in enumerate(self.semantic_objects.keys()):

                if jj not in assignments[:,1]: # If track is unmatched, handle it as a missed detection

                    self.semantic_objects[object_key].update_verbal_comms(msg.sources[speech_idx].transcript, msg.sources[speech_idx].confidence, self)


    def ar_callback(self, msg):
        for marker in msg.markers:

            self.get_logger().info('got ar msg %s' % marker.id)

            if marker.id in self.object_params['person']['comms']['ar_tag_dict'].keys():

                # Get semantic meaning, word type
                type = self.object_params['person']['comms']['ar_tag_dict'][marker.id]['type']
                word = self.object_params['person']['comms']['ar_tag_dict'][marker.id]['word']
                self.get_logger().info('%s: %s\n' % (type, word))

                # Convert to tracker frame
                artag_tracker_tf = self.tf_buffer.lookup_transform(self.tracker_frame,self.artag_frame,time=rclpy.time.Time(),timeout=rclpy.duration.Duration(seconds=.1))
                pos_in_ar_frame = PointStamped()
                pos_in_ar_frame.point.x = marker.pose.pose.position.x
                pos_in_ar_frame.point.y = marker.pose.pose.position.y
                pos_in_ar_frame.point.z = marker.pose.pose.position.z
                pos_in_tracker_frame = do_transform_point(pos_in_ar_frame,artag_tracker_tf)
                self.get_logger().info('Pos in tracker frame: %s\n' % (pos_in_tracker_frame))

                # Compute 
                match_key = compute_pos_match(self, pos_in_tracker_frame.point)
                self.get_logger().info('Matched to person %s\n' % (match_key))

            # TODO fuse all matches, decay non-matched people

def main(args=None):
    rclpy.init(args=args)

    semantic_tracker_node = SemanticTrackerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(semantic_tracker_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        semantic_tracker_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
