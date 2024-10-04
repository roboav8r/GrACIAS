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

from std_srvs.srv import Empty
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

from tracking_msgs.msg import Tracks3D
from ros_audition.msg import SpeechAzSources
from foxglove_msgs.msg import SceneUpdate

from situated_hri_interfaces.msg import Auth, Comm, Comms, Identity, CategoricalDistribution, HierarchicalCommands
from situated_hri_interfaces.srv import ObjectVisRec

from situated_interaction.assignment import compute_az_match, compute_pos_match, compute_az_from_pos, compute_delta_az, compute_delta_pos, solve_assignment_matrix
from situated_interaction.datatypes import DiscreteVariable, SemanticObject
from situated_interaction.output import foxglove_visualization, publish_hierarchical_commands
from situated_interaction.utils import pmf_to_spec, normalize_vector, load_object_params, initialize_sensors, process_sensor_update, delete_sensors, time_to_float

class SemanticTrackerNode(Node):

    def __init__(self):
        super().__init__('semantic_tracker_node')

        self.sub_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.client_cb_group = MutuallyExclusiveCallbackGroup()

        initialize_sensors(self)
        
        # Subscribe to object tracks
        self.subscription_tracks = self.create_subscription(
            Tracks3D,
            'tracks',
            self.tracks_callback,
            10, callback_group=self.sub_cb_group)
        self.subscription_tracks  # prevent unused variable warning

        # Define pubs
        self.semantic_scene_pub = self.create_publisher(
            SceneUpdate,
            'semantic_scene',
            10)

        self.hierarchical_cmd_pub = self.create_publisher(
            HierarchicalCommands,
            'hierarchical_commands',
            10)
        
        # Create main timer
        self.declare_parameter('loop_time_sec', rclpy.Parameter.Type.DOUBLE)
        self.loop_time_sec = self.get_parameter('loop_time_sec').get_parameter_value().double_value
        self.timer = self.create_timer(self.loop_time_sec, self.timer_callback, callback_group=self.timer_cb_group)
        self.service_timeout = .25

        # Create clip service client
        self.clip_client = self.create_client(ObjectVisRec, 'clip_object_rec', callback_group=self.client_cb_group)
        while not self.clip_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.clip_req = ObjectVisRec.Request()

        # Create reset & reconfigure servers
        self.reset_srv = self.create_service(Empty, '~/reset', self.reset_callback)
        self.reconfigure_srv = self.create_service(Empty, '~/reconfigure', self.reconfigure_callback)
        
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
        self.visual_role_rec_futures = []

    def reset_callback(self, _, resp):

        self.get_logger().info("Resetting")
        self.semantic_objects = {}
        self.tracks_msg = Tracks3D()
        self.scene_out_msg = SceneUpdate()

        return resp
    
    def reconfigure_callback(self, _, resp):

        self.get_logger().info("Reconfiguring")

        # Delete sensor subscribers
        delete_sensors(self)

        # Initialize subscribers and object parameters
        initialize_sensors(self)
        load_object_params(self)

        return resp

    def recognize_role(self, id):
        self.clip_req = ObjectVisRec.Request()
        self.clip_req.object_id = id
        self.clip_req.class_string = self.semantic_objects[id].class_string
        self.clip_req.attributes_to_estimate = []
        self.clip_req.states_to_estimate = ['role']
        self.clip_req.estimate_comms = False
        self.clip_req.image = self.semantic_objects[id].image
        self.clip_req.stamp = self.semantic_objects[id].stamp
        future = self.clip_client.call_async(self.clip_req)

        return future

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
    
        if self.role_rec_method=='visual':

            # Check futures
            temp_futures = self.visual_role_rec_futures
            self.visual_role_rec_futures = []

            for (id,future,stamp) in temp_futures:

                if future.done():

                    # Get role likelihood function
                    resp = future.result()

                    self.get_logger().info(f"Future response for object id {id}: {resp}")

                    for state in resp.states:
                        if state.variable=='role':
                            role_probs = state.probabilities
                            break

                    role_obs_idx = np.array(role_probs).argmax()

                    # Compute obs model, update state variable
                    role_var = self.semantic_objects[id].states['role']
                    role_obs_model = gtsam.DiscreteConditional([self.sensor_dict['clip_role_rec']['role_obs_symbol'],len(self.sensor_dict['clip_role_rec']['role_obs_labels'])],[[role_var.var_symbol,len(role_var.var_labels)]],self.sensor_dict['clip_role_rec']['role_obs_spec'])
                    role_likelihood = role_obs_model.likelihood(role_obs_idx)

                    role_var.update(role_likelihood, stamp)

                else:
                    self.visual_role_rec_futures.append((id,future,stamp))

            # Check if objects need update
            start_time = self.get_clock().now()

            # For each object, check if state is stale. If so, send state update request.
            for id in self.semantic_objects.keys():

                obj = self.semantic_objects[id]

                needs_role_update = False

                if (obj.states['role'].last_updated is None):
                    needs_role_update = True
                else: 
                    time_since_update = (start_time - obj.states['role'].last_updated)
                    sec_since_update = time_to_float(time_since_update.seconds,time_since_update.nsec)

                    if sec_since_update > self.sensor_dict['clip_role_rec']['update_threshold']:
                        needs_role_update = True

                if needs_role_update and obj.new_image_available:
                    
                    future = self.recognize_role(id)
                    self.visual_role_rec_futures.append((id,future,start_time))

        foxglove_visualization(self)
        # publish_hierarchical_commands(self)

    def tracks_callback(self, msg):
        self.tracks_msg = msg

        tracked_object_ids = []

        for tracked_object in self.tracks_msg.tracks:

            if tracked_object.class_string not in self.objects_of_interest:
                continue

            tracked_object_ids.append(tracked_object.track_id)

            # Initialize object and add to dict if not currently tracked
            if tracked_object.track_id not in self.semantic_objects.keys():
                self.semantic_objects[tracked_object.track_id] = SemanticObject(tracked_object, self.object_params[tracked_object.class_string])
            else:
                # Update existing track
                self.semantic_objects[tracked_object.track_id].update_spatial_state(tracked_object)

        # Remove untracked objects
        semantic_objects_to_remove = []
        for semantic_obj_key in self.semantic_objects.keys():
            if semantic_obj_key not in tracked_object_ids:
                semantic_objects_to_remove.append(semantic_obj_key)

        for key in semantic_objects_to_remove:
            self.semantic_objects.pop(key)

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

    def ar_callback(self, msg, sensor_name):

        if process_sensor_update(self.sensor_dict[sensor_name])==False:
            return

        # Compute number of comms and role markers to assign
        comm_markers = {}
        role_markers = {}

        # Compute TF
        artag_tracker_tf = self.tf_buffer.lookup_transform(self.tracker_frame,self.artag_frame,time=rclpy.time.Time(),timeout=rclpy.duration.Duration(seconds=1.))

        for marker in msg.markers:
            # self.get_logger().info('%s: got ar msg %s' % (sensor_name, marker.id))

            if marker.id in self.sensor_dict[sensor_name]['ar_tag_dict'].keys():

                # Get semantic meaning, word type
                type = self.sensor_dict[sensor_name]['ar_tag_dict'][marker.id]['type']
                word = self.sensor_dict[sensor_name]['ar_tag_dict'][marker.id]['word']

                # Compute marker position in tracker frame
                pos_in_ar_frame = PointStamped()
                pos_in_ar_frame.point.x = marker.pose.pose.position.x
                pos_in_ar_frame.point.y = marker.pose.pose.position.y
                pos_in_ar_frame.point.z = marker.pose.pose.position.z
                pos_in_tracker_frame = do_transform_point(pos_in_ar_frame,artag_tracker_tf)
                marker_pos = np.array([pos_in_tracker_frame.point.x,pos_in_tracker_frame.point.y,pos_in_tracker_frame.point.z])

                if type == 'command':
                    comm_markers[marker.id] = {}
                    comm_markers[marker.id]['pos'] = marker_pos
                    comm_markers[marker.id]['marker'] = marker

                elif type == 'role':
                    role_markers[marker.id] = {}
                    role_markers[marker.id]['pos'] = marker_pos
                    role_markers[marker.id]['marker'] = marker

        # Initialize, populate, and solve comm assignment matrix
        if self.command_rec_method == 'artag':
            comm_assignment_matrix = np.zeros((len(comm_markers.keys()),len(self.semantic_objects.keys())))
            for marker_idx, marker_key in enumerate(comm_markers.keys()):

                for jj, object_key in enumerate(self.semantic_objects.keys()):
                
                    object = self.semantic_objects[object_key]
                    marker_pos = comm_markers[marker_key]['pos']
                    object_pos = np.array([object.pos_x,object.pos_y,object.pos_z])
                    delta_pos = compute_delta_pos(marker_pos, object_pos)
                    comm_assignment_matrix[marker_idx,jj] += delta_pos

            comm_assignments = solve_assignment_matrix('greedy', comm_assignment_matrix, self.sensor_dict[sensor_name]['match_threshold'])

            # Handle comm matches
            for assignment in comm_assignments:

                object_idx = assignment[1]
                object_key = list(self.semantic_objects.keys())[object_idx]
                marker_idx = assignment[0]
                marker_id = list(comm_markers.keys())[marker_idx]
                comm_word = self.sensor_dict[sensor_name]['ar_tag_dict'][marker_id]['word']

                # Get comms symbol
                comm_var = self.semantic_objects[object_key].comms
                comm_obs_idx = self.sensor_dict[sensor_name]['comm_obs_labels'].index(comm_word)

                comm_obs_model = gtsam.DiscreteConditional([self.sensor_dict[sensor_name]['comm_obs_symbol'],len(self.sensor_dict[sensor_name]['comm_obs_labels'])],[[comm_var.var_symbol,len(comm_var.var_labels)]],self.sensor_dict[sensor_name]['comm_obs_spec'])
                comm_likelihood = comm_obs_model.likelihood(comm_obs_idx)

                comm_var.update(comm_likelihood, msg.header.stamp)

            # Handle objects with no speech
            for jj, object_key in enumerate(self.semantic_objects.keys()):

                if jj not in comm_assignments[:,1]: # If track is unmatched, handle it as a missed detection

                    comm_word = 'none'

                    # Get comms symbol
                    comm_var = self.semantic_objects[object_key].comms
                    comm_obs_idx = self.sensor_dict[sensor_name]['comm_obs_labels'].index(comm_word)

                    comm_obs_model = gtsam.DiscreteConditional([self.sensor_dict[sensor_name]['comm_obs_symbol'],len(self.sensor_dict[sensor_name]['comm_obs_labels'])],[[comm_var.var_symbol,len(comm_var.var_labels)]],self.sensor_dict[sensor_name]['comm_obs_spec'])
                    comm_likelihood = comm_obs_model.likelihood(comm_obs_idx)

                    comm_var.update(comm_likelihood, msg.header.stamp)

        if self.role_rec_method == 'artag':
            # Initialize, populate, and solve role assignment matrix
            role_assignment_matrix = np.zeros((len(role_markers.keys()),len(self.semantic_objects.keys())))
            for marker_idx, marker_key in enumerate(role_markers.keys()):
                for jj, object_key in enumerate(self.semantic_objects.keys()):

                    object = self.semantic_objects[object_key]
                    marker_pos = role_markers[marker_key]['pos']
                    object_pos = np.array([object.pos_x,object.pos_y,object.pos_z])
                    delta_pos = compute_delta_pos(marker_pos, object_pos)
                    role_assignment_matrix[marker_idx,jj] += delta_pos

            role_assignments = solve_assignment_matrix('greedy', role_assignment_matrix, self.sensor_dict[sensor_name]['match_threshold'])

            # Handle role matches
            for assignment in role_assignments:

                object_idx = assignment[1]
                object_key = list(self.semantic_objects.keys())[object_idx]
                marker_idx = assignment[0]
                marker_id = list(role_markers.keys())[marker_idx]
                role_word = self.sensor_dict[sensor_name]['ar_tag_dict'][marker_id]['word']

                # Get variable symbol
                role_var = self.semantic_objects[object_key].states['role']
                role_obs_idx = self.sensor_dict[sensor_name]['role_obs_labels'].index(role_word)

                # Compute obs model, update state variable
                role_obs_model = gtsam.DiscreteConditional([self.sensor_dict[sensor_name]['role_obs_symbol'],len(self.sensor_dict[sensor_name]['role_obs_labels'])],[[role_var.var_symbol,len(role_var.var_labels)]],self.sensor_dict[sensor_name]['role_obs_spec'])
                role_likelihood = role_obs_model.likelihood(role_obs_idx)

                role_var.update(role_likelihood, msg.header.stamp)

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
