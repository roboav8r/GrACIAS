#!/usr/bin/env python3

# import time
from string import ascii_lowercase as alc

import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.time import Time, Duration

import gtsam

from foxglove_msgs.msg import SceneUpdate, SceneEntity, TextPrimitive
from tracking_msgs.msg import Tracks3D

from situated_hri_interfaces.msg import Auth, Comm, Comms, Identity, CategoricalDistribution
from situated_hri_interfaces.srv import ObjectVisRec

def pmf_to_spec(pmf):

    spec = ''
    for row_idx in range(pmf.shape[0]):
        row = pmf[row_idx,:]
        row_spec = ''
        
        for col_idx in range(len(row)):
            if col_idx == 0: # If empty spec
                row_spec += str(row[col_idx])
            else:
                row_spec += '/' +  str(row[col_idx]) 
        
        if row_idx==0:
            spec += row_spec
        else:
            spec += ' ' + row_spec
        
    return spec

class SemanticObject():
    def __init__(self, msg, params):

        self.track_id = msg.track_id
        self.class_string = msg.class_string
        self.stamp = msg.time_updated
        self.identity = "Unknown"
        self.auth = False
        self.comms = {}
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        self.pos_z = msg.pose.pose.position.z
        
        if msg.image_available == True:
            self.new_image_available = True
            self.image = msg.image
        else:
            self.new_image_available = False
            self.image = None
        
        self.vis_rec_complete = False

        # Initialize attributes, states, and obs models
        symbol_idx = 0
        self.attributes = {}
        for att in params['attributes']:
            self.attributes[att] = {}
            self.attributes[att]['confidence_threshold'] = 0.75
            self.attributes[att]['var_symbol'] = gtsam.symbol(alc[symbol_idx],self.track_id)
            self.attributes[att]['obs_symbol'] = gtsam.symbol(alc[symbol_idx],self.track_id + 200)
            self.attributes[att]['labels'] = params['attributes'][att]['labels']
            self.attributes[att]['probs'] = gtsam.DiscreteDistribution((self.attributes[att]['var_symbol'],len(self.attributes[att]['labels'])), params['attributes'][att]['probs'])
            self.attributes[att]['sensor_model'] = gtsam.DiscreteConditional([self.attributes[att]['obs_symbol'],len(self.attributes[att]['labels'])],[[self.attributes[att]['var_symbol'],len(self.attributes[att]['labels'])]],pmf_to_spec(params['attributes'][att]['sensor_model_array']))


            symbol_idx+=1

        self.states = {}
        for state in params['states']:
            self.states[state] = {}
            self.states[state]['var_symbol'] = gtsam.symbol(alc[symbol_idx],self.track_id)
            self.states[state]['obs_symbol'] = gtsam.symbol(alc[symbol_idx],self.track_id + 200)
            self.states[state]['labels'] = params['states'][state]['labels']
            self.states[state]['probs'] = gtsam.DiscreteDistribution((self.states[state]['var_symbol'],len(self.states[state]['labels'])), params['states'][state]['probs'])
            self.states[state]['sensor_model'] = gtsam.DiscreteConditional([self.states[state]['obs_symbol'],len(self.states[state]['labels'])],[[self.states[state]['var_symbol'],len(self.states[state]['labels'])]],pmf_to_spec(params['states'][state]['sensor_model_array']))
            self.states[state]['last_updated'] = Time().to_msg()
            
            symbol_idx+=1   

        self.state_timeout = params['state_timeout']

    # def update_category(self, type, name, cat_dist):
    #     # TODO - make this a bayes update, not just a replacement



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

        sub_timer_cb_group = MutuallyExclusiveCallbackGroup()
        
        # Subscribe to object tracks
        self.subscription_tracks = self.create_subscription(
            Tracks3D,
            'tracks',
            self.listener_callback_tracks,
            10, callback_group=sub_timer_cb_group)
        self.subscription_tracks  # prevent unused variable warning

        # Define pubs
        self.semantic_scene_pub = self.create_publisher(
            SceneUpdate,
            'semantic_scene',
            10)
        
        # Create timer
        self.declare_parameter('loop_time_sec', rclpy.Parameter.Type.DOUBLE)
        self.loop_time_sec = self.get_parameter('loop_time_sec').get_parameter_value().double_value
        self.timer = self.create_timer(self.loop_time_sec, self.timer_callback, callback_group=sub_timer_cb_group)
        self.service_timeout = .25

        # Create service client
        client_cb_group = ReentrantCallbackGroup()
        self.clip_client = self.create_client(ObjectVisRec, 'clip_object_rec',callback_group=client_cb_group)
        while not self.clip_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.clip_req = ObjectVisRec.Request()

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

            self.declare_parameter(obj + '.state_timeout', rclpy.Parameter.Type.DOUBLE)
            self.object_params[obj]['state_timeout'] = self.get_parameter(obj + '.state_timeout').get_parameter_value().double_value

            self.object_params[obj]['attributes'] = {}
            self.declare_parameter(obj + '.attributes.variables', rclpy.Parameter.Type.STRING_ARRAY)
            att_vars = self.get_parameter(obj + '.attributes.variables').get_parameter_value().string_array_value

            for att_var in att_vars:
                self.declare_parameter(obj + '.attributes.' + att_var + '.labels', rclpy.Parameter.Type.STRING_ARRAY)
                self.declare_parameter(obj + '.attributes.' + att_var + '.probs', rclpy.Parameter.Type.DOUBLE_ARRAY)
                self.declare_parameter(obj + '.attributes.' + att_var + '.sensor_model_coeffs', rclpy.Parameter.Type.DOUBLE_ARRAY)

                self.object_params[obj]['attributes'][att_var] = {}
                self.object_params[obj]['attributes'][att_var]['labels'] = self.get_parameter(obj + '.attributes.' + att_var + '.labels').get_parameter_value().string_array_value
                self.object_params[obj]['attributes'][att_var]['probs'] = self.get_parameter(obj + '.attributes.' + att_var + '.probs').get_parameter_value().double_array_value
                self.object_params[obj]['attributes'][att_var]['sensor_model_coeffs'] = self.get_parameter(obj + '.attributes.' + att_var + '.sensor_model_coeffs').get_parameter_value().double_array_value
                self.object_params[obj]['attributes'][att_var]['sensor_model_array'] = np.array(self.object_params[obj]['attributes'][att_var]['sensor_model_coeffs']).reshape(-1,len(self.object_params[obj]['attributes'][att_var]['labels']))
                
            self.object_params[obj]['states'] = {}
            self.declare_parameter(obj + '.states.variables', rclpy.Parameter.Type.STRING_ARRAY)
            state_vars = self.get_parameter(obj + '.states.variables').get_parameter_value().string_array_value

            for state_var in state_vars:
                self.declare_parameter(obj + '.states.' + state_var + '.labels', rclpy.Parameter.Type.STRING_ARRAY)
                self.declare_parameter(obj + '.states.' + state_var + '.probs', rclpy.Parameter.Type.DOUBLE_ARRAY)
                self.declare_parameter(obj + '.states.' + state_var + '.sensor_model_coeffs', rclpy.Parameter.Type.DOUBLE_ARRAY)

                self.object_params[obj]['states'][state_var] = {}
                self.object_params[obj]['states'][state_var]['labels'] = self.get_parameter(obj + '.states.' + state_var + '.labels').get_parameter_value().string_array_value
                self.object_params[obj]['states'][state_var]['probs'] = self.get_parameter(obj + '.states.' + state_var + '.probs').get_parameter_value().double_array_value
                self.object_params[obj]['states'][state_var]['sensor_model_coeffs'] = self.get_parameter(obj + '.states.' + state_var + '.sensor_model_coeffs').get_parameter_value().double_array_value
                self.object_params[obj]['states'][state_var]['sensor_model_array'] = np.array(self.object_params[obj]['states'][state_var]['sensor_model_coeffs']).reshape(-1,len(self.object_params[obj]['states'][state_var]['labels']))


    def send_obj_clip_req(self, id, est_atts, est_states):
        self.clip_req = ObjectVisRec.Request()
        self.clip_req.object_id = id
        self.clip_req.class_string = self.semantic_objects[id].class_string
        self.clip_req.estimate_attributes = est_atts
        self.clip_req.estimate_states = est_states
        # self.get_logger().info("Semantic object info : %s" % self.semantic_objects[id])
        self.clip_req.image = self.semantic_objects[id].image
        # self.get_logger().info("Sending request")
        resp = self.clip_client.call(self.clip_req)
        self.get_logger().info("Got response %s" % resp)

        # TODO - make this an update and pass the resp in
        for att_dist in resp.attributes:

            att = att_dist.variable
            self.get_logger().info("VARIABLE %s" % att)
            self.get_logger().info("PRIOR %s" % self.semantic_objects[id].attributes[att]['probs'])

            obs = gtsam.DiscreteDistribution([self.semantic_objects[id].attributes[att]['obs_symbol'],len(self.semantic_objects[id].attributes[att]['labels'])], att_dist.probabilities)
            # self.get_logger().info("OBS %s" % obs)
            # obs_factor = gtsam.DecisionTreeFactor(obs)
            # self.get_logger().info("OBS FACTOR %s" % obs_factor)
            # sensor_model_factor = gtsam.DecisionTreeFactor(self.semantic_objects[id].attributes[att]['sensor_model'])
            # self.get_logger().info("SENSOR MODEL FACTOR %s" % sensor_model_factor)
            # likelihood = (obs_factor*sensor_model_factor).sum(1)
            likelihood = self.semantic_objects[id].attributes[att]['sensor_model'].likelihood(obs.argmax())
            # self.get_logger().info("LIKELIHOOD %s" % likelihood)
            self.semantic_objects[id].attributes[att]['probs'] = gtsam.DiscreteDistribution(likelihood*self.semantic_objects[id].attributes[att]['probs'])

            self.get_logger().info("POSTERIOR %s" % self.semantic_objects[id].attributes[att]['probs'])

        for state_dist in resp.states:
            state = state_dist.variable
            self.get_logger().info("VARIABLE %s" % state)
            self.get_logger().info("PRIOR %s" % self.semantic_objects[id].states[state]['probs'])

            obs = gtsam.DiscreteDistribution([self.semantic_objects[id].states[state]['obs_symbol'],len(self.semantic_objects[id].states[state]['labels'])], state_dist.probabilities)
            # obs_factor = gtsam.DecisionTreeFactor(obs)
            # sensor_model_factor = gtsam.DecisionTreeFactor(self.semantic_objects[id].states[state]['sensor_model'])
            # likelihood = (obs_factor*sensor_model_factor).sum(1)
            likelihood = self.semantic_objects[id].states[state]['sensor_model'].likelihood(obs.argmax())
            self.semantic_objects[id].states[state]['probs'] = gtsam.DiscreteDistribution(likelihood*self.semantic_objects[id].states[state]['probs'])
            # self.get_logger().info("POSTERIOR %s" % self.semantic_objects[id].states[state]['probs'])
            self.semantic_objects[id].states[state]['last_updated'] = self.semantic_objects[id].stamp

        self.semantic_objects[id].new_image_available = False

    def compute_match(self, pos):
        similarity_vector = np.zeros(len(self.semantic_objects.keys()))

        for ii,key in enumerate(self.semantic_objects.keys()): 
            similarity_vector[ii] = np.linalg.norm([pos.x - self.semantic_objects[key].pos_x, pos.y - self.semantic_objects[key].pos_y, pos.z - self.semantic_objects[key].pos_z])

        return list(self.semantic_objects.keys())[np.argmin(similarity_vector)]

    def timer_callback(self):
        # self.get_logger().info("\n\n\n********************************TIMER CALLBACK")
        start_time = self.get_clock().now()

        # For each object, check if state is stale. If so, send state update request.
        for id in self.semantic_objects.keys():

            obj = self.semantic_objects[id]

            if obj.new_image_available==False:
                continue

            update_obj_states = False
            update_obj_atts = False

            for att in obj.attributes:
                if (obj.attributes[att]['labels'][obj.attributes[att]['probs'].argmax()]=='unknown'):
                    self.get_logger().info("Object %s %s has unknown attribute %s " % (obj.class_string, id, att))
                    update_obj_atts = True
                elif (obj.attributes[att]['probs'](obj.attributes[att]['probs'].argmax()) < obj.attributes[att]['confidence_threshold']):
                    self.get_logger().info("Object %s %s has low confidence %s < %s" % (obj.class_string, id, obj.attributes[att]['probs'](obj.attributes[att]['probs'].argmax()),obj.attributes[att]['confidence_threshold']) )
                    update_obj_atts = True
                
            for state in obj.states:
                self.get_logger().info("Start time %s, Last update %s, delta %s sec" % (start_time, Time.from_msg(obj.states[state]['last_updated']), (start_time - Time.from_msg(obj.states[state]['last_updated'])).nanoseconds/1e9 ))
                if (start_time - Time.from_msg(obj.states[state]['last_updated'])).nanoseconds/1e9 > obj.state_timeout:
                    self.get_logger().info("Object %s %s has stale state %s (%s sec)" % (obj.class_string, id, state, (start_time - Time.from_msg(obj.states[state]['last_updated'])).nanoseconds/1e9 ))
                    update_obj_states = True

            self.send_obj_clip_req(id, update_obj_atts, update_obj_states)

        self.visualize()

        self.get_logger().info("Timer callback time (s): %s" % ((self.get_clock().now() - start_time).nanoseconds/10**9))

    def listener_callback_tracks(self, msg):
        # self.get_logger().info("Tracks callback")
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
                self.get_logger().info(f"Creating object {trkd_obj.class_string} with params {self.object_params[trkd_obj.class_string]}")
                self.semantic_objects[trkd_obj.track_id] = SemanticObject(trkd_obj, self.object_params[trkd_obj.class_string])

            else:
                # Update existing tracked object
                # self.get_logger().info("Updating %s-%s" % (trkd_obj.class_string, trkd_obj.track_id))

                # self.get_logger().info("Current image avail / checksum %s %s" % (self.semantic_objects[trkd_obj.track_id].new_image_available, sum(self.semantic_objects[trkd_obj.track_id].image.data)))


                # TODO - make generic update function instead of pos updates
                self.semantic_objects[trkd_obj.track_id].pos_x = trkd_obj.pose.pose.position.x
                self.semantic_objects[trkd_obj.track_id].pos_y = trkd_obj.pose.pose.position.y
                self.semantic_objects[trkd_obj.track_id].pos_z = trkd_obj.pose.pose.position.z

                # self.get_logger().info("Incoming image avail / checksum %s %s" % (trkd_obj.image_available, sum(trkd_obj.image.data)))
                self.semantic_objects[trkd_obj.track_id].new_image_available = trkd_obj.image_available
                self.semantic_objects[trkd_obj.track_id].image = trkd_obj.image
                # self.get_logger().info("Updated image avail / checksum %s %s" % (self.semantic_objects[trkd_obj.track_id].new_image_available, sum(self.semantic_objects[trkd_obj.track_id].image.data)))

                self.semantic_objects[trkd_obj.track_id].stamp = msg.header.stamp 

        # Remove untracked objects
        objects_temp = self.semantic_objects
        self.semantic_objects = {}
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
                text.text += "%s: %s %2.0f%%\n" % (att, obj.attributes[att]['labels'][obj.attributes[att]['probs'].argmax()], 100*obj.attributes[att]['probs'](obj.attributes[att]['probs'].argmax()))

            text.text += '\n'
            for state in obj.states:
                text.text += "%s: %s %2.0f%%\n" % (state, obj.states[state]['labels'][obj.states[state]['probs'].argmax()], 100*obj.states[state]['probs'](obj.states[state]['probs'].argmax()))

            # for word in obj.comms.keys():
            #     text.text += "%s (%.2f) " % (word, obj.comms[word]*100)
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
