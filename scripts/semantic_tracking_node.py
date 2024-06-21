#!/usr/bin/env python3

from string import ascii_lowercase as alc

import numpy as np

import rclpy
from rclpy.node import Node

import gtsam

from tracking_msgs.msg import Tracks3D
from situated_hri_interfaces.msg import Auth, Comm, Comms, Identity, CategoricalDistribution

from foxglove_msgs.msg import SceneUpdate, SceneEntity, TextPrimitive

class SemanticObject():
    def __init__(self, msg, params):

        self.track_id = msg.track_id
        self.class_string = msg.class_string
        self.identity = "Unknown"
        self.auth = False
        self.comms = {}
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        self.pos_z = msg.pose.pose.position.z
        
        if msg.image_available == True:
            self.image_available = True
            self.image = msg.image
        else:
            self.image_available = False
            self.image = None
        
        self.vis_rec_complete = False

        # Initialize attributes and states
        symbol_idx = 0
        self.attributes = {}
        for att in params['attributes']:
            self.attributes[att] = {}
            self.attributes[att]['symbol'] = gtsam.Symbol(alc[symbol_idx],self.track_id)
            self.attributes[att]['labels'] = params['attributes'][att]['labels']
            self.attributes[att]['prob'] = gtsam.DiscreteDistribution((self.attributes[att]['symbol'].key(),len(self.attributes[att]['labels'])), params['attributes'][att]['probs'])
            
            symbol_idx+=1

        self.states = {}
        for state in params['states']:
            self.states[state] = {}
            self.states[state]['symbol'] = gtsam.Symbol(alc[symbol_idx],self.track_id)
            self.states[state]['labels'] = params['states'][state]['labels']
            self.states[state]['prob'] = gtsam.DiscreteDistribution((self.states[state]['symbol'].key(),len(self.states[state]['labels'])), params['states'][state]['probs'])
            self.states[state]['last_updated'] = msg.time_updated
            
            symbol_idx+=1   


    

    # def update(self, ar_msg, type):
    #     if type=='authentication':
    #         if ar_msg.authenticated:
    #             self.auth = True

    #     if type=='communication':
    #         # Increase weight for existing comms in incoming comms
    #         temp_comms = self.comms # dictionary of comm dictionaries
    #         new_comms = ar_msg # list of [situated_hri_interfaces/Comm]
    #         self.comms = {}

    #         # Handle incoming comms
    #         for new_comm in new_comms:

    #             # Incoming message word matches an existing word
    #             if new_comm.comm in temp_comms.keys():

    #                 # Fuse the two and update confidence value to self.comms dict, use parallel addition
    #                 self.comms[new_comm.comm] = 1 - (1 - new_comm.conf)*(1 - temp_comms[new_comm.comm])/((1 - new_comm.conf) + (1 - temp_comms[new_comm.comm]))

    #                 # Remove word from temp_comms
    #                 del temp_comms[new_comm.comm]

    #             elif new_comm.conf > self.det_thresh: # Create new word
    #                 self.comms[new_comm.comm] = new_comm.conf

    #         # Now, find unmatched words in last comm step and reduce confidence
    #         for word in temp_comms.keys():

    #             new_score = temp_comms[word] - self.score_decay
    #             if new_score > self.del_thresh: # Only publish if above delete threshold
    #                 self.comms[word] = new_score

    #     if type=='identity':
    #         self.identity = ar_msg.identity


class SemanticTrackerNode(Node):
    def __init__(self):
        super().__init__('semantic_tracker_node')
        
        self.subscription_tracks = self.create_subscription(
            Tracks3D,
            'tracks',
            self.listener_callback_tracks,
            10)
        self.subscription_tracks  # prevent unused variable warning

        # Define pubs
        self.semantic_scene_pub = self.create_publisher(
            SceneUpdate,
            'semantic_scene',
            10)

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
        self.objects_of_interest = self.get_parameter('objects_of_interest').get_parameter_value().string_array_value

        for obj in self.objects_of_interest:

            self.object_params[obj] = {}

            self.object_params[obj]['attributes'] = {}
            self.declare_parameter(obj + '.attributes.variables', rclpy.Parameter.Type.STRING_ARRAY)
            att_vars = self.get_parameter(obj + '.attributes.variables').get_parameter_value().string_array_value

            for att_var in att_vars:
                self.declare_parameter(obj + '.attributes.' + att_var + '.labels', rclpy.Parameter.Type.STRING_ARRAY)
                self.declare_parameter(obj + '.attributes.' + att_var + '.probs', rclpy.Parameter.Type.DOUBLE_ARRAY)

                self.object_params[obj]['attributes'][att_var] = {}
                self.object_params[obj]['attributes'][att_var]['labels'] = self.get_parameter(obj + '.attributes.' + att_var + '.labels').get_parameter_value().string_array_value
                self.object_params[obj]['attributes'][att_var]['probs'] = self.get_parameter(obj + '.attributes.' + att_var + '.probs').get_parameter_value().double_array_value

            self.object_params[obj]['states'] = {}
            self.declare_parameter(obj + '.states.variables', rclpy.Parameter.Type.STRING_ARRAY)
            state_vars = self.get_parameter(obj + '.states.variables').get_parameter_value().string_array_value

            for state_var in state_vars:
                self.declare_parameter(obj + '.states.' + state_var + '.labels', rclpy.Parameter.Type.STRING_ARRAY)
                self.declare_parameter(obj + '.states.' + state_var + '.probs', rclpy.Parameter.Type.DOUBLE_ARRAY)

                self.object_params[obj]['states'][state_var] = {}
                self.object_params[obj]['states'][state_var]['labels'] = self.get_parameter(obj + '.states.' + state_var + '.labels').get_parameter_value().string_array_value
                self.object_params[obj]['states'][state_var]['probs'] = self.get_parameter(obj + '.states.' + state_var + '.probs').get_parameter_value().double_array_value


    def compute_match(self, pos):
        similarity_vector = np.zeros(len(self.semantic_objects.keys()))

        for ii,key in enumerate(self.semantic_objects.keys()): 
            similarity_vector[ii] = np.linalg.norm([pos.x - self.semantic_objects[key].pos_x, pos.y - self.semantic_objects[key].pos_y, pos.z - self.semantic_objects[key].pos_z])

        return list(self.semantic_objects.keys())[np.argmin(similarity_vector)]

    def listener_callback_tracks(self, msg):
        self.tracks_msg = msg

        # Temporary list for this callback
        self.semantic_object_ids = []

        # add new tracked objects from incoming message
        for trkd_obj in msg.tracks:

            if trkd_obj.class_string not in self.objects_of_interest:
                self.get_logger().info(f"Class string {trkd_obj.class_string} not in {self.objects_of_interest}") 
                continue

            # Add 
            self.semantic_object_ids.append(trkd_obj.track_id)

            # Initialize object and add to dict if not currently tracked
            if trkd_obj.track_id not in self.semantic_objects.keys():
                self.get_logger().info(f"Creating object {trkd_obj.class_string} with params {self.object_params[trkd_obj.class_string]}")
                self.semantic_objects[trkd_obj.track_id] = SemanticObject(trkd_obj, self.object_params[trkd_obj.class_string])

            else:
                # Update existing tracked object
                # TODO - make generic update function instead of pos updates
                self.semantic_objects[trkd_obj.track_id].pos_x = trkd_obj.pose.pose.position.x
                self.semantic_objects[trkd_obj.track_id].pos_y = trkd_obj.pose.pose.position.y
                self.semantic_objects[trkd_obj.track_id].pos_z = trkd_obj.pose.pose.position.z

        # Remove untracked objects
        objects_temp = self.semantic_objects
        self.semantic_objects = {key: val for key, val in objects_temp.items() if key in self.semantic_object_ids}

        self.visualize()


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
            object = self.semantic_objects[idx]
            entity_msg = SceneEntity()

            # Populate entity message with header / object data
            entity_msg.frame_id = self.tracks_msg.header.frame_id
            entity_msg.timestamp = self.tracks_msg.header.stamp
            entity_msg.id = str(object.track_id)
            entity_msg.frame_locked = True
            entity_msg.lifetime.nanosec = 500000000

            text = TextPrimitive()
            text.billboard = True
            text.font_size = 16.
            text.scale_invariant = True
            text.color.a = 1.0
            text.pose.position.x = object.pos_x
            text.pose.position.y = object.pos_y
            text.pose.position.z = object.pos_z
            text.text = "%s #%s: \nID: %s \nAuth: %s\n" % (object.class_string, object.track_id, object.identity, object.auth)
            for word in object.comms.keys():
                text.text += "%s (%.2f) " % (word, object.comms[word]*100)
            text.text
            entity_msg.texts.append(text)

            self.scene_out_msg.entities.append(entity_msg)

        self.semantic_scene_pub.publish(self.scene_out_msg)


def main(args=None):
    rclpy.init(args=args)

    semantic_tracker_node = SemanticTrackerNode()

    try:
        rclpy.spin(semantic_tracker_node)
    except KeyboardInterrupt:
        pass
    finally:
        semantic_tracker_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
