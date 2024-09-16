#!/usr/bin/env python3

import gtsam
import numpy as np
from string import ascii_lowercase as alc

from rclpy.time import Time

from situated_interaction.utils import normalize_vector, pmf_to_spec

class DiscreteVariable():
    def __init__(self, name, var_type, method, thresh, stamp, labels, sym, idx, idx_offset, init_prob, obs_model_spec, upper_prob_limit, lower_prob_limit):
        # Admin
        self.name = name
        self.var_type = var_type
        self.update_method = method
        self.update_threshold = thresh
        self.last_updated = stamp
    
        # GTSAM
        self.var_labels = labels # TODO - make separate obs_labels for different detectors
        self.obs_labels = labels
        self.var_symbol = gtsam.symbol(sym, idx)
        self.obs_symbol = gtsam.symbol(sym, idx + idx_offset)
        self.probs = gtsam.DiscreteDistribution((self.var_symbol,len(self.var_labels)), init_prob)
        self.obs_model = gtsam.DiscreteConditional([self.obs_symbol,len(self.obs_labels)],[[self.var_symbol,len(self.var_labels)]],obs_model_spec)

        # Params
        self.upper_prob_limit = upper_prob_limit
        self.lower_prob_limit = lower_prob_limit
        
    def needs_update(self, stamp):
        return ((self.update_method=='confidence' and self.probs(self.probs.argmax()) < self.update_threshold) 
            or (self.update_method=='time' and (stamp - self.last_updated).nanoseconds/1e9 > self.update_threshold))

    def update(self, obs_probs, stamp):
        obs = gtsam.DiscreteDistribution([self.obs_symbol,len(self.obs_labels)], obs_probs)
        likelihood = self.obs_model.likelihood(obs.argmax())
        self.probs = gtsam.DiscreteDistribution(likelihood*self.probs)

        normalized_pmf = normalize_vector(self.probs.pmf(), self.upper_prob_limit, self.lower_prob_limit)
        self.probs = gtsam.DiscreteDistribution((self.var_symbol,len(self.var_labels)),normalized_pmf)

        self.last_updated = stamp

class SemanticObject():
    def __init__(self, msg, params, frame_id):

        self.track_id = msg.track_id
        self.frame_id = frame_id
        self.class_string = msg.class_string
        self.stamp = msg.time_updated

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
            self.attributes[att] = DiscreteVariable(att, 'attribute', params['attributes'][att]['update_method'], params['attributes'][att]['update_threshold'], 
                                                        Time.from_msg(self.stamp), params['attributes'][att]['labels'], alc[symbol_idx], 
                                                        self.track_id, 100, params['attributes'][att]['probs'], 
                                                        pmf_to_spec(params['attributes'][att]['sensor_model_array']), params['upper_prob_limit'], params['lower_prob_limit'])
            symbol_idx+=1

        self.states = {}
        for state in params['states']:

            self.states[state] = DiscreteVariable(state, 'state', params['states'][state]['update_method'], params['states'][state]['update_threshold'], 
                                            Time.from_msg(self.stamp), params['states'][state]['labels'], alc[symbol_idx], 
                                            self.track_id, 100, params['states'][state]['probs'], 
                                            pmf_to_spec(params['states'][state]['sensor_model_array']), params['upper_prob_limit'], params['lower_prob_limit'])
            symbol_idx+=1

        self.comm_labels = params['comms']['labels']
        self.comm_transcripts = params['comms']['transcripts']
        self.comm_var_symbol = gtsam.symbol(alc[symbol_idx], symbol_idx)
        self.comm_gesture_obs_symbol = gtsam.symbol(alc[symbol_idx], symbol_idx + 100)
        self.comm_verbal_obs_symbol = gtsam.symbol(alc[symbol_idx], symbol_idx + 101)
        self.comm_probs = gtsam.DiscreteDistribution((self.comm_var_symbol,len(self.comm_labels)), params['comms']['probs'])
        self.comm_gesture_obs_model = gtsam.DiscreteConditional([self.comm_gesture_obs_symbol,len(self.comm_labels)],[[self.comm_var_symbol,len(self.comm_labels)]],pmf_to_spec(params['comms']['gesture_sensor_model_array']))
        self.comm_verbal_obs_model = gtsam.DiscreteConditional([self.comm_verbal_obs_symbol,len(self.comm_labels)],[[self.comm_var_symbol,len(self.comm_labels)]],pmf_to_spec(params['comms']['verbal_sensor_model_array']))
        self.upper_prob_limit = params['upper_prob_limit']
        self.lower_prob_limit = params['lower_prob_limit']

    def update_spatial_state(self, tracked_object_msg):
                
        self.pos_x = tracked_object_msg.pose.pose.position.x
        self.pos_y = tracked_object_msg.pose.pose.position.y
        self.pos_z = tracked_object_msg.pose.pose.position.z

        self.new_image_available = tracked_object_msg.image_available
        self.image = tracked_object_msg.image

        self.stamp = tracked_object_msg.time_updated

    def update_verbal_comms(self, transcript, confidence, parent_node):
        
        # parent_node.get_logger().info("Comms labels %s" % (self.comm_labels))
        # parent_node.get_logger().info("update comms with transcript %s confidence %s" % (transcript, confidence))
        
        try:
            command_idx = self.comm_transcripts.index(transcript)
        except ValueError:
            if transcript=='':
                command_idx = 0 # Handle null commands
            else:
                command_idx = 1 # Handle "other" commands that are not in list
        # parent_node.get_logger().info("Command index %s" % (command_idx))

        likelihood = self.comm_verbal_obs_model.likelihood(command_idx)
        # parent_node.get_logger().info("Likelihood %s" % (likelihood))
        self.comm_probs = gtsam.DiscreteDistribution(likelihood*self.comm_probs)
        # parent_node.get_logger().info("New comm probs %s" % (self.comm_probs))

        normalized_pmf = normalize_vector(self.comm_probs.pmf(), self.upper_prob_limit, self.lower_prob_limit)
        self.comm_probs = gtsam.DiscreteDistribution((self.comm_var_symbol,len(self.comm_labels)),normalized_pmf)
        # parent_node.get_logger().info("Normalized comm probs %s" % (self.comm_probs))

        # TODO
        # TODO - handle null command/empty command
        # TODO - handle "other" command not in list


    def update_gesture_comms(self, gest_dist, parent_node):
        
        command_idx = np.argmax(gest_dist.probabilities)

        likelihood = self.comm_gesture_obs_model.likelihood(command_idx)
        self.comm_probs = gtsam.DiscreteDistribution(likelihood*self.comm_probs)

        normalized_pmf = normalize_vector(self.comm_probs.pmf(), self.upper_prob_limit, self.lower_prob_limit)
        self.comm_probs = gtsam.DiscreteDistribution((self.comm_var_symbol,len(self.comm_labels)),normalized_pmf)
