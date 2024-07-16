#!/usr/bin/env python3

import gtsam
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
        self.comm_obs_symbol = gtsam.symbol(alc[symbol_idx], symbol_idx + 100)
        self.comm_probs = gtsam.DiscreteDistribution((self.comm_var_symbol,len(self.comm_labels)), params['comms']['probs'])
        self.comm_obs_model = gtsam.DiscreteConditional([self.comm_obs_symbol,len(self.comm_labels)],[[self.comm_var_symbol,len(self.comm_labels)]],pmf_to_spec(params['comms']['sensor_model_array']))

    def update_comms(self, transcript, confidence, parent_node):
        parent_node.get_logger().info("update comms with transcript %s confidence %s" % (transcript, confidence))
        
        # TODO - get index of transcript, find likelihood, 

        # likelihood = self.obs_model.likelihood(obs.argmax())
        # self.probs = gtsam.DiscreteDistribution(likelihood*self.probs)

        # normalized_pmf = normalize_vector(self.probs.pmf(), self.upper_prob_limit, self.lower_prob_limit)
        # self.probs = gtsam.DiscreteDistribution((self.var_symbol,len(self.var_labels)),normalized_pmf)
        
        # TODO
        # TODO - handle null command/empty command
        # TODO - handle "other" command not in list


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