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
from ar_track_alvar_msgs.msg import AlvarMarker, AlvarMarkers
from tracking_msgs.msg import Tracks3D
from ros_audition.msg import SpeechAzSources
from foxglove_msgs.msg import SceneUpdate

from situated_hri_interfaces.msg import Auth, Comm, Comms, Identity, CategoricalDistribution
from situated_hri_interfaces.srv import ObjectVisRec

from situated_interaction.assignment import compute_az_match, compute_pos_match, compute_az_from_pos, compute_delta_az, compute_delta_pos, solve_assignment_matrix
from situated_interaction.datatypes import DiscreteVariable, SemanticObject
from situated_interaction.output import foxglove_visualization
from situated_interaction.utils import pmf_to_spec, normalize_vector, load_object_params

class SemanticTrackerNode(Node):

    def __init__(self):
        super().__init__('semantic_tracker_node')

        self.sub_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.client_cb_group = MutuallyExclusiveCallbackGroup()

        self.initialize_sensors()
        
        # Subscribe to object tracks
        self.subscription_tracks = self.create_subscription(
            Tracks3D,
            'tracks',
            self.tracks_callback,
            10, callback_group=self.sub_cb_group)
        self.subscription_tracks  # prevent unused variable warning

        # Subscribe to localized speech
        self.subscription_speech = self.create_subscription(
            SpeechAzSources,
            'speech_az_sources',
            self.speech_callback,
            10, callback_group=self.sub_cb_group)
        self.subscription_speech  # prevent unused variable warning

        # Define pubs
        self.semantic_scene_pub = self.create_publisher(
            SceneUpdate,
            'semantic_scene',
            10)
        
        # Create timer
        self.declare_parameter('loop_time_sec', rclpy.Parameter.Type.DOUBLE)
        self.loop_time_sec = self.get_parameter('loop_time_sec').get_parameter_value().double_value
        self.timer = self.create_timer(self.loop_time_sec, self.timer_callback, callback_group=self.timer_cb_group)
        self.service_timeout = .25

        # # Create clip service client
        # self.clip_client = self.create_client(ObjectVisRec, 'clip_object_rec', callback_group=self.client_cb_group)
        # while not self.clip_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('service not available, waiting again...')
        # self.clip_req = ObjectVisRec.Request()

        # Create reset server
        self.reset_srv = self.create_service(Empty, '~/reset', self.reset_callback)
        
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

    def initialize_sensors(self):
        self.declare_parameter('sensor_names', rclpy.Parameter.Type.STRING_ARRAY)
        self.sensor_names = self.get_parameter('sensor_names').get_parameter_value().string_array_value
        self.sensor_dict = {}

        for sensor_name in self.sensor_names:
            
            self.sensor_dict[sensor_name] = {}

            self.declare_parameter('sensors.%s.type' % sensor_name, rclpy.Parameter.Type.STRING)
            self.declare_parameter('sensors.%s.topic' % sensor_name, rclpy.Parameter.Type.STRING)
            self.declare_parameter('sensors.%s.match_threshold' % sensor_name, rclpy.Parameter.Type.DOUBLE)
            self.sensor_dict[sensor_name]['type'] = self.get_parameter('sensors.%s.type' % sensor_name).get_parameter_value().string_value
            self.sensor_dict[sensor_name]['topic'] = self.get_parameter('sensors.%s.topic' % sensor_name).get_parameter_value().string_value
            self.sensor_dict[sensor_name]['match_threshold'] = self.get_parameter('sensors.%s.match_threshold' % sensor_name).get_parameter_value().double_value

            # TODO - update method, update thresh

            # Create subscriber
            if self.sensor_dict[sensor_name]['type']=='fiducial':
                self.sensor_dict[sensor_name]['sub'] = self.create_subscription(AlvarMarkers, 
                                                                                self.sensor_dict[sensor_name]['topic'],
                                                                                eval("lambda msg: self.ar_callback(msg, \"" + sensor_name + "\")",locals()),
                                                                                10, callback_group=self.sub_cb_group)

                self.declare_parameter('sensors.%s.ar_tag_ids' % sensor_name, rclpy.Parameter.Type.INTEGER_ARRAY)
                self.declare_parameter('sensors.%s.ar_tag_types' % sensor_name, rclpy.Parameter.Type.STRING_ARRAY)
                self.declare_parameter('sensors.%s.ar_tag_words' % sensor_name, rclpy.Parameter.Type.STRING_ARRAY)
                self.sensor_dict[sensor_name]['ar_tag_ids'] = self.get_parameter('sensors.%s.ar_tag_ids' % sensor_name).get_parameter_value().integer_array_value
                self.sensor_dict[sensor_name]['ar_tag_types'] = self.get_parameter('sensors.%s.ar_tag_types' % sensor_name).get_parameter_value().string_array_value
                self.sensor_dict[sensor_name]['ar_tag_words'] = self.get_parameter('sensors.%s.ar_tag_words' % sensor_name).get_parameter_value().string_array_value

                self.sensor_dict[sensor_name]['ar_tag_dict'] = {}
                
                for index, tag_id in enumerate(self.sensor_dict[sensor_name]['ar_tag_ids']):
                    self.sensor_dict[sensor_name]['ar_tag_dict'][tag_id] = {}
                    self.sensor_dict[sensor_name]['ar_tag_dict'][tag_id]['type'] = self.sensor_dict[sensor_name]['ar_tag_types'][index]
                    self.sensor_dict[sensor_name]['ar_tag_dict'][tag_id]['word'] = self.sensor_dict[sensor_name]['ar_tag_words'][index]

            else:
                self.get_logger.info("Invalid sensor type %s" % self.sensor_dict[sensor_name]['type'])


    def reset_callback(self, _, resp):

        self.get_logger().info("Resetting")
        self.semantic_objects = {}
        self.tracks_msg = Tracks3D()
        self.scene_out_msg = SceneUpdate()

        return resp
    
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
        # start_time = self.get_clock().now()

        # # For each object, check if state is stale. If so, send state update request.
        # for id in self.semantic_objects.keys():

        #     obj = self.semantic_objects[id]

        #     # self.get_logger().info("Checking object %s with atts %s, states: %s" % (id, str(obj.attributes),str(obj.states)))

        #     if obj.new_image_available==False:
        #         continue

        #     states_to_est = []
        #     atts_to_est = []

        #     for att in obj.attributes:
        #         if obj.attributes[att].needs_update(start_time):
        #             atts_to_est.append(att)
                
        #     for state in obj.states:
        #         if obj.states[state].needs_update(start_time):
        #             states_to_est.append(state)

        #     if self.object_params[obj.class_string]['comms']['gesture_descriptions'] != ['']:
        #         est_comms = True

        #     if atts_to_est or states_to_est or est_comms:
        #         self.get_logger().info("atts to est: %s" % atts_to_est)
        #         self.get_logger().info("states to est: %s" % states_to_est)
        #         self.send_obj_clip_req(id, atts_to_est, states_to_est, est_comms)

        foxglove_visualization(self)

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


    def ar_callback(self, msg, sensor_name):

        # Compute number of comms and role markers to assign
        n_comm_assignments = 0
        n_role_assignments = 0
        for marker in msg.markers:
            self.get_logger().info('%s: got ar msg %s' % (sensor_name, marker.id))

            if marker.id in self.sensor_dict[sensor_name]['ar_tag_dict'].keys():

                # Get semantic meaning, word type
                type = self.sensor_dict[sensor_name]['ar_tag_dict'][marker.id]['type']
                word = self.sensor_dict[sensor_name]['ar_tag_dict'][marker.id]['word']
                self.get_logger().info('%s: %s\n' % (type, word))

                if self.sensor_dict[sensor_name]['ar_tag_dict'][marker.id]['type'] == 'command':
                    n_comm_assignments +=1

                elif self.sensor_dict[sensor_name]['ar_tag_dict'][marker.id]['type'] == 'role':
                    n_role_assignments +=1

        self.get_logger().info('Have %s objects' % (len(self.semantic_objects.keys())))
        self.get_logger().info('Got %s role and %s comm AR tags\n' % (n_role_assignments, n_comm_assignments))

        # Compute TF
        artag_tracker_tf = self.tf_buffer.lookup_transform(self.tracker_frame,self.artag_frame,time=rclpy.time.Time(),timeout=rclpy.duration.Duration(seconds=.1))

        # Initialize, populate, and solve assignment matrices
        comm_assignment_matrix = np.zeros((n_comm_assignments,len(self.semantic_objects.keys())))
        if n_comm_assignments != 0:
            marker_idx = 0
        
            for marker in msg.markers:
                if marker.id not in self.sensor_dict[sensor_name]['ar_tag_dict'].keys() or self.sensor_dict[sensor_name]['ar_tag_dict'][marker.id]['type'] != 'command':

                    continue

                for jj, object_key in enumerate(self.semantic_objects.keys()):
                
                    pos_in_ar_frame = PointStamped()
                    pos_in_ar_frame.point.x = marker.pose.pose.position.x
                    pos_in_ar_frame.point.y = marker.pose.pose.position.y
                    pos_in_ar_frame.point.z = marker.pose.pose.position.z
                    pos_in_tracker_frame = do_transform_point(pos_in_ar_frame,artag_tracker_tf)

                    object = self.semantic_objects[object_key]
                    marker_pos = np.array([pos_in_tracker_frame.point.x,pos_in_tracker_frame.point.y,pos_in_tracker_frame.point.z])
                    object_pos = np.array([object.pos_x,object.pos_y,object.pos_z])
                    delta_pos = compute_delta_pos(marker_pos, object_pos)
                    comm_assignment_matrix[marker_idx,jj] += delta_pos


        # Solve comm assignment matrix
        comm_assignments = solve_assignment_matrix('greedy', comm_assignment_matrix, self.sensor_dict[sensor_name]['match_threshold'])
        self.get_logger().info('AR Tag comm assignments: %s\n' % (comm_assignments))

        role_assignment_matrix = np.zeros((n_role_assignments,len(self.semantic_objects.keys())))
        role_idx = 0
        if n_role_assignments != 0:
        
            for marker in msg.markers:
                if marker.id not in self.sensor_dict[sensor_name]['ar_tag_dict'].keys() or self.sensor_dict[sensor_name]['ar_tag_dict'][marker.id]['type'] != 'role':
                    continue

                for jj, object_key in enumerate(self.semantic_objects.keys()):
                
                    pos_in_ar_frame = PointStamped()
                    pos_in_ar_frame.point.x = marker.pose.pose.position.x
                    pos_in_ar_frame.point.y = marker.pose.pose.position.y
                    pos_in_ar_frame.point.z = marker.pose.pose.position.z
                    pos_in_tracker_frame = do_transform_point(pos_in_ar_frame,artag_tracker_tf)

                    object = self.semantic_objects[object_key]
                    marker_pos = np.array([pos_in_tracker_frame.point.x,pos_in_tracker_frame.point.y,pos_in_tracker_frame.point.z])
                    object_pos = np.array([object.pos_x,object.pos_y,object.pos_z])
                    delta_pos = compute_delta_pos(marker_pos, object_pos)
                    role_assignment_matrix[role_idx,jj] += delta_pos

                role_idx +=1

        # Solve assignment matrix
        role_assignments = solve_assignment_matrix('greedy', role_assignment_matrix, self.sensor_dict[sensor_name]['match_threshold'])
        self.get_logger().info('AR Tag role assignments: %s\n' % (role_assignments))

        # Handle role matches
        for assignment in role_assignments:

            object_idx = assignment[1]
            object_key = list(self.semantic_objects.keys())[object_idx]
            marker_id = msg.markers[assignment[0]].id
            role = self.sensor_dict[sensor_name]['ar_tag_dict'][marker_id]['word']
            confidence = msg.markers[assignment[0]].confidence

            self.semantic_objects[object_key].update_semantic_state('role', role, confidence, self)

        # note - no penalty for missing role AR tag


        # Handle comm matches
        for assignment in comm_assignments:

            object_idx = assignment[1]
            object_key = list(self.semantic_objects.keys())[object_idx]
            marker_id = msg.markers[assignment[0]].id
            comm = self.sensor_dict[sensor_name]['ar_tag_dict'][marker_id]['word']
            confidence = msg.markers[assignment[0]].confidence

            self.semantic_objects[object_key].update_comms(comm, confidence, self)

        # Handle objects with no speech
        for jj, object_key in enumerate(self.semantic_objects.keys()):

            if jj not in comm_assignments[:,1]: # If track is unmatched, handle it as a missed detection

                self.semantic_objects[object_key].update_comms("\'\'", confidence, self)


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
