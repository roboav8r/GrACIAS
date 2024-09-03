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

from foxglove_msgs.msg import SceneUpdate, SceneEntity, TextPrimitive
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point
from ar_track_alvar_msgs.msg import AlvarMarker, AlvarMarkers
from tracking_msgs.msg import Tracks3D
from ros_audition.msg import SpeechAzSources

from situated_hri_interfaces.msg import Auth, Comm, Comms, Identity, CategoricalDistribution
from situated_hri_interfaces.srv import ObjectVisRec

from situated_interaction.utils import pmf_to_spec, normalize_vector
from situated_interaction.datatypes import DiscreteVariable, SemanticObject

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
        self.clip_client = self.create_client(ObjectVisRec, 'clip_object_rec',callback_group=client_cb_group)
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
        self.artag_tracker_tf = self.tf_buffer.lookup_transform(self.tracker_frame,self.artag_frame,Duration(seconds=.1))

        # self.subscription_auth = self.create_subscription(
        #     Auth,
        #     'authentication',
        #     self.listener_callback_auth,
        #     10)
        # self.subscription_auth  # prevent unused variable warning

        # self.subscription_id = self.create_subscription(
        #     Identity,
        #     'identification',
        #     self.listener_callback_identity,
        #     10)
        # self.subscription_id  # prevent unused variable warning

        # self.subscription_comms = self.create_subscription(
        #     Comms,
        #     'communications',
        #     self.listener_callback_comms,
        #     10)
        # self.subscription_comms  # prevent unused variable warning

        # Define member variables
        self.semantic_objects = {}
        self.tracks_msg = Tracks3D()
        self.scene_out_msg = SceneUpdate()

        # Generate object att/state variable dictionary
        self.object_params = {}
        self.declare_parameter('objects_of_interest', rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('upper_prob_limit', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('lower_prob_limit', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('ang_match_threshold', rclpy.Parameter.Type.DOUBLE)
        self.objects_of_interest = self.get_parameter('objects_of_interest').get_parameter_value().string_array_value
        self.upper_prob_limit = self.get_parameter('upper_prob_limit').get_parameter_value().double_value
        self.lower_prob_limit = self.get_parameter('lower_prob_limit').get_parameter_value().double_value
        self.ang_match_threshold = self.get_parameter('ang_match_threshold').get_parameter_value().double_value

        for obj in self.objects_of_interest:

            self.object_params[obj] = {}

            self.object_params[obj]['upper_prob_limit'] = self.upper_prob_limit
            self.object_params[obj]['lower_prob_limit'] = self.lower_prob_limit

            self.object_params[obj]['attributes'] = {}
            self.declare_parameter(obj + '.attributes.variables', rclpy.Parameter.Type.STRING_ARRAY)
            att_vars = self.get_parameter(obj + '.attributes.variables').get_parameter_value().string_array_value

            for att_var in att_vars:
                if att_var != '':
                    self.declare_parameter(obj + '.attributes.' + att_var + '.labels', rclpy.Parameter.Type.STRING_ARRAY)
                    self.declare_parameter(obj + '.attributes.' + att_var + '.probs', rclpy.Parameter.Type.DOUBLE_ARRAY)
                    self.declare_parameter(obj + '.attributes.' + att_var + '.sensor_model_coeffs', rclpy.Parameter.Type.DOUBLE_ARRAY)
                    self.declare_parameter(obj + '.attributes.' + att_var + '.update_method', rclpy.Parameter.Type.STRING)
                    self.declare_parameter(obj + '.attributes.' + att_var + '.update_threshold', rclpy.Parameter.Type.DOUBLE)

                    self.object_params[obj]['attributes'][att_var] = {}
                    self.object_params[obj]['attributes'][att_var]['labels'] = self.get_parameter(obj + '.attributes.' + att_var + '.labels').get_parameter_value().string_array_value
                    self.object_params[obj]['attributes'][att_var]['probs'] = self.get_parameter(obj + '.attributes.' + att_var + '.probs').get_parameter_value().double_array_value
                    self.object_params[obj]['attributes'][att_var]['sensor_model_coeffs'] = self.get_parameter(obj + '.attributes.' + att_var + '.sensor_model_coeffs').get_parameter_value().double_array_value
                    self.object_params[obj]['attributes'][att_var]['sensor_model_array'] = np.array(self.object_params[obj]['attributes'][att_var]['sensor_model_coeffs']).reshape(-1,len(self.object_params[obj]['attributes'][att_var]['labels']))
                    self.object_params[obj]['attributes'][att_var]['update_method'] = self.get_parameter(obj + '.attributes.' + att_var + '.update_method').get_parameter_value().string_value
                    self.object_params[obj]['attributes'][att_var]['update_threshold'] = self.get_parameter(obj + '.attributes.' + att_var + '.update_threshold').get_parameter_value().double_value

            self.object_params[obj]['states'] = {}
            self.declare_parameter(obj + '.states.variables', rclpy.Parameter.Type.STRING_ARRAY)
            state_vars = self.get_parameter(obj + '.states.variables').get_parameter_value().string_array_value

            for state_var in state_vars:
                if state_var != '':
                    self.declare_parameter(obj + '.states.' + state_var + '.labels', rclpy.Parameter.Type.STRING_ARRAY)
                    self.declare_parameter(obj + '.states.' + state_var + '.probs', rclpy.Parameter.Type.DOUBLE_ARRAY)
                    self.declare_parameter(obj + '.states.' + state_var + '.sensor_model_coeffs', rclpy.Parameter.Type.DOUBLE_ARRAY)
                    self.declare_parameter(obj + '.states.' + state_var + '.update_method', rclpy.Parameter.Type.STRING)
                    self.declare_parameter(obj + '.states.' + state_var + '.update_threshold', rclpy.Parameter.Type.DOUBLE)

                    self.object_params[obj]['states'][state_var] = {}
                    self.object_params[obj]['states'][state_var]['labels'] = self.get_parameter(obj + '.states.' + state_var + '.labels').get_parameter_value().string_array_value
                    self.object_params[obj]['states'][state_var]['probs'] = self.get_parameter(obj + '.states.' + state_var + '.probs').get_parameter_value().double_array_value
                    self.object_params[obj]['states'][state_var]['sensor_model_coeffs'] = self.get_parameter(obj + '.states.' + state_var + '.sensor_model_coeffs').get_parameter_value().double_array_value
                    self.object_params[obj]['states'][state_var]['sensor_model_array'] = np.array(self.object_params[obj]['states'][state_var]['sensor_model_coeffs']).reshape(-1,len(self.object_params[obj]['states'][state_var]['labels']))
                    self.object_params[obj]['states'][state_var]['update_method'] = self.get_parameter(obj + '.states.' + state_var + '.update_method').get_parameter_value().string_value
                    self.object_params[obj]['states'][state_var]['update_threshold'] = self.get_parameter(obj + '.states.' + state_var + '.update_threshold').get_parameter_value().double_value

            # Communication parameters
            self.object_params[obj]['comms'] = {}
            self.declare_parameter(obj + '.comms.labels', rclpy.Parameter.Type.STRING_ARRAY)
            self.object_params[obj]['comms']['labels'] = self.get_parameter(obj + '.comms.labels').get_parameter_value().string_array_value

            # Get AR tag parameters
            self.declare_parameter(obj + '.comms.ar_tag_ids', rclpy.Parameter.Type.INTEGER_ARRAY)
            self.declare_parameter(obj + '.comms.ar_tag_types', rclpy.Parameter.Type.STRING_ARRAY)
            self.declare_parameter(obj + '.comms.ar_tag_words', rclpy.Parameter.Type.STRING_ARRAY)
            self.object_params[obj]['comms']['ar_tag_ids'] = self.get_parameter(obj + '.comms.ar_tag_ids').get_parameter_value().integer_array_value
            self.object_params[obj]['comms']['ar_tag_types'] = self.get_parameter(obj + '.comms.ar_tag_types').get_parameter_value().string_array_value
            self.object_params[obj]['comms']['ar_tag_words'] = self.get_parameter(obj + '.comms.ar_tag_words').get_parameter_value().string_array_value
            self.object_params[obj]['comms']['ar_tag_dict'] = {}
            assert len(self.object_params[obj]['comms']['ar_tag_ids']) == len(self.object_params[obj]['comms']['ar_tag_types'])
            assert len(self.object_params[obj]['comms']['ar_tag_words']) == len(self.object_params[obj]['comms']['ar_tag_types'])
            for index, tag_id in enumerate(self.object_params[obj]['comms']['ar_tag_ids']):
                self.object_params[obj]['comms']['ar_tag_dict'][tag_id] = {}
                self.object_params[obj]['comms']['ar_tag_dict'][tag_id]['type'] = self.object_params[obj]['comms']['ar_tag_types'][index]
                self.object_params[obj]['comms']['ar_tag_dict'][tag_id]['word'] = self.object_params[obj]['comms']['ar_tag_words'][index]

            # Get gesture parameters
            self.declare_parameter(obj + '.comms.gesture_descriptions', rclpy.Parameter.Type.STRING_ARRAY)
            self.declare_parameter(obj + '.comms.transcripts', rclpy.Parameter.Type.STRING_ARRAY)
            self.declare_parameter(obj + '.comms.probs', rclpy.Parameter.Type.DOUBLE_ARRAY)
            self.declare_parameter(obj + '.comms.gesture_sensor_model_coeffs', rclpy.Parameter.Type.DOUBLE_ARRAY)
            self.declare_parameter(obj + '.comms.verbal_sensor_model_coeffs', rclpy.Parameter.Type.DOUBLE_ARRAY)
            self.object_params[obj]['comms']['gesture_descriptions'] = self.get_parameter(obj + '.comms.gesture_descriptions').get_parameter_value().string_array_value
            self.object_params[obj]['comms']['transcripts'] = self.get_parameter(obj + '.comms.transcripts').get_parameter_value().string_array_value
            self.object_params[obj]['comms']['probs'] = self.get_parameter(obj + '.comms.probs').get_parameter_value().double_array_value
            self.object_params[obj]['comms']['gesture_sensor_model_coeffs'] = self.get_parameter(obj + '.comms.gesture_sensor_model_coeffs').get_parameter_value().double_array_value
            self.object_params[obj]['comms']['gesture_sensor_model_array'] = np.array(self.object_params[obj]['comms']['gesture_sensor_model_coeffs']).reshape(-1,len(self.object_params[obj]['comms']['labels']))
            self.object_params[obj]['comms']['verbal_sensor_model_coeffs'] = self.get_parameter(obj + '.comms.verbal_sensor_model_coeffs').get_parameter_value().double_array_value
            self.object_params[obj]['comms']['verbal_sensor_model_array'] = np.array(self.object_params[obj]['comms']['verbal_sensor_model_coeffs']).reshape(-1,len(self.object_params[obj]['comms']['labels']))

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

        # self.get_logger().info('got gesture response labels %s' % gesture_comms.categories)
        # self.get_logger().info('got gesture response probs %s' % gesture_comms.probabilities)

        # self.semantic_objects[id].states[state].update(state_dist.probabilities, Time.from_msg(resp.stamp))

        self.semantic_objects[id].image_available = False

    def compute_az_match(self, az, az_frame):
        similarity_vector = np.zeros(len(self.semantic_objects.keys()))

        # Ensure az in range [-pi, pi] # TODO make this a helper function, or make sure output of az speech rec in -pi,pi
        while np.abs(az) > np.pi:
            az -= np.sign(az)*2*np.pi

        for ii,key in enumerate(self.semantic_objects.keys()):
            # Convert object position into az frame
            tracker_az_tf = self.tf_buffer.lookup_transform(az_frame,self.tracker_frame,Duration(seconds=.1))
            pos_in_tracker_frame = PointStamped()
            pos_in_tracker_frame.point.x = self.semantic_objects[key].pos_x
            pos_in_tracker_frame.point.y = self.semantic_objects[key].pos_y
            pos_in_tracker_frame.point.z = self.semantic_objects[key].pos_z
            pos_in_az_frame = do_transform_point(pos_in_tracker_frame,tracker_az_tf)

            # Compute azimuth in az frame
            obj_az = np.arctan2(pos_in_az_frame.point.y,pos_in_az_frame.point.x) # in range [-pi,pi]

            similarity_vector[ii] = np.linalg.norm([az - obj_az])

        # Return match if below angular match threshold
        if (similarity_vector.size > 0) and (similarity_vector[np.argmin(similarity_vector)] < self.ang_match_threshold):
            return list(self.semantic_objects.keys())[np.argmin(similarity_vector)] if similarity_vector[np.argmin(similarity_vector)] < self.ang_match_threshold else None
        else:
            return None

    def compute_pos_match(self, pos):
        similarity_vector = np.zeros(len(self.semantic_objects.keys()))

        for ii,key in enumerate(self.semantic_objects.keys()): 
            similarity_vector[ii] = np.linalg.norm([pos.x - self.semantic_objects[key].pos_x, pos.y - self.semantic_objects[key].pos_y, pos.z - self.semantic_objects[key].pos_z])

        return list(self.semantic_objects.keys())[np.argmin(similarity_vector)]

    def timer_callback(self):
        start_time = self.get_clock().now()

        # For each object, check if state is stale. If so, send state update request.
        for id in self.semantic_objects.keys():

            obj = self.semantic_objects[id]

            self.get_logger().info("Checking object %s with atts %s, states: %s" % (id, str(obj.attributes),str(obj.states)))

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

        self.visualize()

        self.get_logger().info("Timer callback time (s): %s" % ((self.get_clock().now() - start_time).nanoseconds/10**9))

    def tracks_callback(self, msg):
        self.tracks_msg = msg

        # Temporary list for this callback
        self.semantic_object_ids = []

        # add new tracked objects from incoming message
        for trkd_obj in msg.tracks:

            if trkd_obj.class_string not in self.objects_of_interest:
                continue

            # Add 
            self.semantic_object_ids.append(trkd_obj.track_id)

            # Initialize object and add to dict if not currently tracked
            if trkd_obj.track_id not in self.semantic_objects.keys():
                self.semantic_objects[trkd_obj.track_id] = SemanticObject(trkd_obj, self.object_params[trkd_obj.class_string], self.tracks_msg.header.frame_id) # TODO - pass frame_id some other way
            else:

                # TODO - make generic semantic_object.update function instead of pos updates
                self.semantic_objects[trkd_obj.track_id].pos_x = trkd_obj.pose.pose.position.x
                self.semantic_objects[trkd_obj.track_id].pos_y = trkd_obj.pose.pose.position.y
                self.semantic_objects[trkd_obj.track_id].pos_z = trkd_obj.pose.pose.position.z

                self.semantic_objects[trkd_obj.track_id].new_image_available = trkd_obj.image_available
                self.semantic_objects[trkd_obj.track_id].image = trkd_obj.image

                self.semantic_objects[trkd_obj.track_id].stamp = msg.header.stamp 

        # Remove untracked objects
        objects_temp = self.semantic_objects
        self.semantic_objects = {}
        self.semantic_objects = {key: val for key, val in objects_temp.items() if key in self.semantic_object_ids}

        self.visualize()


    def speech_callback(self, msg):

        if msg.sources:
            for speech_src in msg.sources:

                match_idx = self.compute_az_match(speech_src.azimuth, msg.header.frame_id)

                if match_idx is not None:
                    self.semantic_objects[match_idx].update_verbal_comms(speech_src.transcript, speech_src.confidence, self)

        else: # If no speech, update all objects with null
            for obj_id in self.semantic_objects.keys():
                self.semantic_objects[obj_id].update_verbal_comms('', 1., self)

    def ar_callback(self, msg):
        for marker in msg.markers:

            self.get_logger().info('got ar msg %s' % marker.id)

            # Get semantic meaning, word type
            type = self.object_params['person']['comms']['ar_tag_dict'][marker.id]['type']
            word = self.object_params['person']['comms']['ar_tag_dict'][marker.id]['word']
            self.get_logger().info('%s: %s\n' % (type, word))

            # Convert to tracker frame
            pos_in_ar_frame = PointStamped()
            pos_in_ar_frame.point.x = marker.pose.pose.position.x
            pos_in_ar_frame.point.y = marker.pose.pose.position.y
            pos_in_ar_frame.point.z = marker.pose.pose.position.z
            pos_in_tracker_frame = do_transform_point(pos_in_ar_frame,self.artag_tracker_tf)
            self.get_logger().info('Pos in tracker frame: %s\n' % (pos_in_tracker_frame))

            # TODO assign

        # TODO fuse all matches, decay non-matched people


    # def listener_callback_auth(self, msg):
    #     # self.get_logger().info('Received auth message: "%s"' % msg)

    #     # Compute match, perform update
    #     match_key = self.compute_match(msg.pose.pose.position)
    #     self.semantic_objects[match_key].update(msg, 'authentication')

    # def listener_callback_comms(self, msg):
    #     # self.get_logger().info('Received communication message: "%s"' % msg)

    #     comm_matches = {}
    #     for person_id in self.semantic_object_ids:
    #         comm_matches[person_id] = []

    #     # Compute set of comms that are matched to a person
    #     for comm in msg.comms:
    #         match_key = self.compute_match(comm.pose.position)            
    #         comm_matches[match_key].append(comm)

    #     # Update
    #     self.get_logger().info('person-comm matches: "%s"' % comm_matches)
    #     for key in comm_matches.keys():
    #         self.semantic_objects[key].update(comm_matches[key], 'communication')

    # def listener_callback_identity(self, msg):
    #     # self.get_logger().info('Received identity message: "%s"' % msg)

    #     # Compute match, perform update
    #     match_key = self.compute_match(msg.pose.pose.position)
    #     self.semantic_objects[match_key].update(msg, 'identity')


    def visualize(self):
        self.scene_out_msg = SceneUpdate()

        for idx in self.semantic_objects.keys():
            obj = self.semantic_objects[idx]
            entity_msg = SceneEntity()

            # Populate entity message with header / object data
            entity_msg.frame_id = self.tracks_msg.header.frame_id
            entity_msg.timestamp = self.tracks_msg.header.stamp
            entity_msg.id = str(obj.track_id)
            entity_msg.frame_locked = True
            entity_msg.lifetime.nanosec = 500000000

            text = TextPrimitive()
            text.billboard = True
            text.font_size = 16.
            text.scale_invariant = True
            text.color.a = 1.0
            text.pose.position.x = obj.pos_x
            text.pose.position.y = obj.pos_y
            text.pose.position.z = obj.pos_z
            # text.text = "%s #%s: \nID: %s \nAuth: %s\n" % (obj.class_string, obj.track_id, obj.identity, obj.auth)
            text.text = "%s #%s: \n" % (obj.class_string, obj.track_id)
            
            text.text += '\n'
            for att in obj.attributes:
                text.text += "%s: %s %2.0f%%\n" % (att, obj.attributes[att].var_labels[obj.attributes[att].probs.argmax()], 100*obj.attributes[att].probs(obj.attributes[att].probs.argmax()))

            text.text += '\n'
            for state in obj.states:
                text.text += "%s: %s %2.0f%%\n" % (state, obj.states[state].var_labels[obj.states[state].probs.argmax()], 100*obj.states[state].probs(obj.states[state].probs.argmax()))

            text.text += '\n'
            text.text += "command: %s %2.0f%%\n" % (obj.comm_labels[obj.comm_probs.argmax()], 100*obj.comm_probs(obj.comm_probs.argmax()))
            entity_msg.texts.append(text)

            self.scene_out_msg.entities.append(entity_msg)

        self.semantic_scene_pub.publish(self.scene_out_msg)


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
