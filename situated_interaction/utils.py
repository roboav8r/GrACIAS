#!/usr/bin/env python3

import numpy as np
import gtsam

import rclpy

from ar_track_alvar_msgs.msg import AlvarMarkers

from gesture_recognition_ros2.msg import KeypointsWithBBox
from ros_audition.msg import SpeechAzSources

def process_sensor_update(sensor_params):
    if sensor_params['update_method']=='count':
        return sensor_params['count']%sensor_params['update_threshold']==0
    else:
        return False
    
def try_to_declare_parameter(node, param, type):
    try:
        node.declare_parameter(param, type)
    except rclpy.exceptions.ParameterAlreadyDeclaredException:
        return

def delete_sensors(semantic_fusion_node):
    for sensor_name in semantic_fusion_node.sensor_names:
        
        if sensor_name == '':
            continue
        
        if 'sub' in semantic_fusion_node.sensor_dict[sensor_name].keys():
            semantic_fusion_node.destroy_subscription(semantic_fusion_node.sensor_dict[sensor_name]['sub'])

def initialize_sensors(semantic_fusion_node):
    semantic_fusion_node.sensor_dict = {}
    try_to_declare_parameter(semantic_fusion_node,'role_rec_methods', rclpy.Parameter.Type.STRING_ARRAY)
    try_to_declare_parameter(semantic_fusion_node,'command_rec_methods', rclpy.Parameter.Type.STRING_ARRAY)
    try_to_declare_parameter(semantic_fusion_node,'sensor_names', rclpy.Parameter.Type.STRING_ARRAY)
    try_to_declare_parameter(semantic_fusion_node,'sensor_symbol', rclpy.Parameter.Type.STRING)
    semantic_fusion_node.role_rec_methods = semantic_fusion_node.get_parameter('role_rec_methods').get_parameter_value().string_array_value
    semantic_fusion_node.command_rec_methods = semantic_fusion_node.get_parameter('command_rec_methods').get_parameter_value().string_array_value
    semantic_fusion_node.sensor_names = semantic_fusion_node.get_parameter('sensor_names').get_parameter_value().string_array_value
    semantic_fusion_node.sensor_dict['symbol'] = semantic_fusion_node.get_parameter('sensor_symbol').get_parameter_value().string_value

    observer_idx = 0 # Used for GTSAM observation factors

    for sensor_name in semantic_fusion_node.sensor_names:

        if sensor_name == '':
            continue
        
        semantic_fusion_node.sensor_dict[sensor_name] = {}

        try_to_declare_parameter(semantic_fusion_node,'sensors.%s.type' % sensor_name, rclpy.Parameter.Type.STRING)
        try_to_declare_parameter(semantic_fusion_node,'sensors.%s.topic' % sensor_name, rclpy.Parameter.Type.STRING)
        semantic_fusion_node.sensor_dict[sensor_name]['type'] = semantic_fusion_node.get_parameter('sensors.%s.type' % sensor_name).get_parameter_value().string_value
        semantic_fusion_node.sensor_dict[sensor_name]['topic'] = semantic_fusion_node.get_parameter('sensors.%s.topic' % sensor_name).get_parameter_value().string_value

        # Update method, update thresh
        try_to_declare_parameter(semantic_fusion_node,'sensors.%s.update_method' % sensor_name, rclpy.Parameter.Type.STRING)
        semantic_fusion_node.sensor_dict[sensor_name]['update_method'] = semantic_fusion_node.get_parameter('sensors.%s.update_method' % sensor_name).get_parameter_value().string_value

        if semantic_fusion_node.sensor_dict[sensor_name]['update_method'] == 'count':
            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.update_threshold' % sensor_name, rclpy.Parameter.Type.INTEGER)
            semantic_fusion_node.sensor_dict[sensor_name]['update_threshold'] = semantic_fusion_node.get_parameter('sensors.%s.update_threshold' % sensor_name).get_parameter_value().integer_value
            semantic_fusion_node.sensor_dict[sensor_name]['count'] = 0

        elif semantic_fusion_node.sensor_dict[sensor_name]['update_method'] in ['confidence', 'time']:
            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.update_threshold' % sensor_name, rclpy.Parameter.Type.DOUBLE)
            semantic_fusion_node.sensor_dict[sensor_name]['update_threshold'] = semantic_fusion_node.get_parameter('sensors.%s.update_threshold' % sensor_name).get_parameter_value().double_value

        # Create subscribers, GTSAM observation models
        if semantic_fusion_node.sensor_dict[sensor_name]['type']=='artag':

            # Create subscriber
            semantic_fusion_node.sensor_dict[sensor_name]['sub'] = semantic_fusion_node.create_subscription(AlvarMarkers, 
                                                                            semantic_fusion_node.sensor_dict[sensor_name]['topic'],
                                                                            eval("lambda msg: semantic_fusion_node.ar_callback(msg, \"" + sensor_name + "\")",locals()),
                                                                            10, callback_group=semantic_fusion_node.update_var_cb_group)

            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.match_threshold' % sensor_name, rclpy.Parameter.Type.DOUBLE)
            semantic_fusion_node.sensor_dict[sensor_name]['match_threshold'] = semantic_fusion_node.get_parameter('sensors.%s.match_threshold' % sensor_name).get_parameter_value().double_value

            # Handle AR Tag interpretation
            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.ar_tag_ids' % sensor_name, rclpy.Parameter.Type.INTEGER_ARRAY)
            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.ar_tag_types' % sensor_name, rclpy.Parameter.Type.STRING_ARRAY)
            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.ar_tag_words' % sensor_name, rclpy.Parameter.Type.STRING_ARRAY)
            semantic_fusion_node.sensor_dict[sensor_name]['ar_tag_ids'] = semantic_fusion_node.get_parameter('sensors.%s.ar_tag_ids' % sensor_name).get_parameter_value().integer_array_value
            semantic_fusion_node.sensor_dict[sensor_name]['ar_tag_types'] = semantic_fusion_node.get_parameter('sensors.%s.ar_tag_types' % sensor_name).get_parameter_value().string_array_value
            semantic_fusion_node.sensor_dict[sensor_name]['ar_tag_words'] = semantic_fusion_node.get_parameter('sensors.%s.ar_tag_words' % sensor_name).get_parameter_value().string_array_value
            semantic_fusion_node.sensor_dict[sensor_name]['ar_tag_dict'] = {}
            
            for index, tag_id in enumerate(semantic_fusion_node.sensor_dict[sensor_name]['ar_tag_ids']):
                semantic_fusion_node.sensor_dict[sensor_name]['ar_tag_dict'][tag_id] = {}
                semantic_fusion_node.sensor_dict[sensor_name]['ar_tag_dict'][tag_id]['type'] = semantic_fusion_node.sensor_dict[sensor_name]['ar_tag_types'][index]
                semantic_fusion_node.sensor_dict[sensor_name]['ar_tag_dict'][tag_id]['word'] = semantic_fusion_node.sensor_dict[sensor_name]['ar_tag_words'][index]

            # Generate observation models for role, communication
            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.role_obs_labels' % sensor_name, rclpy.Parameter.Type.STRING_ARRAY)
            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.role_obs_model_coeffs' % sensor_name, rclpy.Parameter.Type.DOUBLE_ARRAY)
            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.comm_obs_labels' % sensor_name, rclpy.Parameter.Type.STRING_ARRAY)
            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.comm_obs_model_coeffs' % sensor_name, rclpy.Parameter.Type.DOUBLE_ARRAY)

            semantic_fusion_node.sensor_dict[sensor_name]['role_obs_symbol'] = gtsam.symbol(semantic_fusion_node.sensor_dict['symbol'], observer_idx)
            semantic_fusion_node.sensor_dict[sensor_name]['role_obs_labels'] = semantic_fusion_node.get_parameter('sensors.%s.role_obs_labels' % sensor_name,).get_parameter_value().string_array_value
            semantic_fusion_node.sensor_dict[sensor_name]['role_obs_model_coeffs'] = semantic_fusion_node.get_parameter('sensors.%s.role_obs_model_coeffs' % sensor_name).get_parameter_value().double_array_value
            semantic_fusion_node.sensor_dict[sensor_name]['role_obs_model_array'] = np.array(semantic_fusion_node.sensor_dict[sensor_name]['role_obs_model_coeffs']).reshape(-1,len(semantic_fusion_node.sensor_dict[sensor_name]['role_obs_labels']))
            semantic_fusion_node.sensor_dict[sensor_name]['role_obs_spec'] = pmf_to_spec(semantic_fusion_node.sensor_dict[sensor_name]['role_obs_model_array'])
            observer_idx += 1
                
            semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_symbol'] = gtsam.symbol(semantic_fusion_node.sensor_dict['symbol'], observer_idx)
            semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_labels'] = semantic_fusion_node.get_parameter('sensors.%s.comm_obs_labels' % sensor_name,).get_parameter_value().string_array_value
            semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_model_coeffs'] = semantic_fusion_node.get_parameter('sensors.%s.comm_obs_model_coeffs' % sensor_name).get_parameter_value().double_array_value
            semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_model_array'] = np.array(semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_model_coeffs']).reshape(-1, len(semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_labels']))
            semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_spec'] = pmf_to_spec(semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_model_array'])
            observer_idx += 1

        elif semantic_fusion_node.sensor_dict[sensor_name]['type']=='vision':

            # Generate observation models for role, communication
            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.role_obs_labels' % sensor_name, rclpy.Parameter.Type.STRING_ARRAY)
            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.role_obs_model_coeffs' % sensor_name, rclpy.Parameter.Type.DOUBLE_ARRAY)

            semantic_fusion_node.sensor_dict[sensor_name]['role_obs_symbol'] = gtsam.symbol(semantic_fusion_node.sensor_dict['symbol'], observer_idx)
            semantic_fusion_node.sensor_dict[sensor_name]['role_obs_labels'] = semantic_fusion_node.get_parameter('sensors.%s.role_obs_labels' % sensor_name,).get_parameter_value().string_array_value
            semantic_fusion_node.sensor_dict[sensor_name]['role_obs_model_coeffs'] = semantic_fusion_node.get_parameter('sensors.%s.role_obs_model_coeffs' % sensor_name).get_parameter_value().double_array_value
            semantic_fusion_node.sensor_dict[sensor_name]['role_obs_model_array'] = np.array(semantic_fusion_node.sensor_dict[sensor_name]['role_obs_model_coeffs']).reshape(-1,len(semantic_fusion_node.sensor_dict[sensor_name]['role_obs_labels']))
            semantic_fusion_node.sensor_dict[sensor_name]['role_obs_spec'] = pmf_to_spec(semantic_fusion_node.sensor_dict[sensor_name]['role_obs_model_array'])
            observer_idx += 1

        elif semantic_fusion_node.sensor_dict[sensor_name]['type']=='verbal':

            # Create subscriber
            semantic_fusion_node.sensor_dict[sensor_name]['sub'] = semantic_fusion_node.create_subscription(SpeechAzSources, 
                                                                            semantic_fusion_node.sensor_dict[sensor_name]['topic'],
                                                                            eval("lambda msg: semantic_fusion_node.speech_az_callback(msg, \"" + sensor_name + "\")",locals()),
                                                                            10, callback_group=semantic_fusion_node.update_var_cb_group)

            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.match_threshold' % sensor_name, rclpy.Parameter.Type.DOUBLE)
            semantic_fusion_node.sensor_dict[sensor_name]['match_threshold'] = semantic_fusion_node.get_parameter('sensors.%s.match_threshold' % sensor_name).get_parameter_value().double_value

            # Generate observation models for role, communication
            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.comm_obs_labels' % sensor_name, rclpy.Parameter.Type.STRING_ARRAY)
            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.comm_obs_model_coeffs' % sensor_name, rclpy.Parameter.Type.DOUBLE_ARRAY)

            semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_symbol'] = gtsam.symbol(semantic_fusion_node.sensor_dict['symbol'], observer_idx)
            semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_labels'] = semantic_fusion_node.get_parameter('sensors.%s.comm_obs_labels' % sensor_name,).get_parameter_value().string_array_value
            semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_model_coeffs'] = semantic_fusion_node.get_parameter('sensors.%s.comm_obs_model_coeffs' % sensor_name).get_parameter_value().double_array_value
            semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_model_array'] = np.array(semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_model_coeffs']).reshape(-1, len(semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_labels']))
            semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_spec'] = pmf_to_spec(semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_model_array'])
            observer_idx += 1

        elif semantic_fusion_node.sensor_dict[sensor_name]['type']=='gesture':

            # Create subscriber
            semantic_fusion_node.sensor_dict[sensor_name]['sub'] = semantic_fusion_node.create_subscription(KeypointsWithBBox, 
                                                                            semantic_fusion_node.sensor_dict[sensor_name]['topic'],
                                                                            eval("lambda msg: semantic_fusion_node.gesture_kp_callback(msg, \"" + sensor_name + "\")",locals()),
                                                                            10, callback_group=semantic_fusion_node.update_var_cb_group)

            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.match_threshold' % sensor_name, rclpy.Parameter.Type.DOUBLE)
            semantic_fusion_node.sensor_dict[sensor_name]['match_threshold'] = semantic_fusion_node.get_parameter('sensors.%s.match_threshold' % sensor_name).get_parameter_value().double_value

            # Generate observation models for role, communication
            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.comm_obs_labels' % sensor_name, rclpy.Parameter.Type.STRING_ARRAY)
            try_to_declare_parameter(semantic_fusion_node,'sensors.%s.comm_obs_model_coeffs' % sensor_name, rclpy.Parameter.Type.DOUBLE_ARRAY)

            semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_symbol'] = gtsam.symbol(semantic_fusion_node.sensor_dict['symbol'], observer_idx)
            semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_labels'] = semantic_fusion_node.get_parameter('sensors.%s.comm_obs_labels' % sensor_name,).get_parameter_value().string_array_value
            semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_model_coeffs'] = semantic_fusion_node.get_parameter('sensors.%s.comm_obs_model_coeffs' % sensor_name).get_parameter_value().double_array_value
            semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_model_array'] = np.array(semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_model_coeffs']).reshape(-1, len(semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_labels']))
            semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_spec'] = pmf_to_spec(semantic_fusion_node.sensor_dict[sensor_name]['comm_obs_model_array'])
            observer_idx += 1

        else:
            semantic_fusion_node.get_logger().info("Invalid sensor type %s" % semantic_fusion_node.sensor_dict[sensor_name]['type'])


def load_object_params(semantic_fusion_node):
    semantic_fusion_node.object_params = {}
    try_to_declare_parameter(semantic_fusion_node,'objects_of_interest', rclpy.Parameter.Type.STRING_ARRAY)
    semantic_fusion_node.objects_of_interest = semantic_fusion_node.get_parameter('objects_of_interest').get_parameter_value().string_array_value

    for obj in semantic_fusion_node.objects_of_interest:

        semantic_fusion_node.object_params[obj] = {}

        if obj == 'person':
            try_to_declare_parameter(semantic_fusion_node,'person.window_length', rclpy.Parameter.Type.INTEGER)
            try_to_declare_parameter(semantic_fusion_node,'person.model_input_dim', rclpy.Parameter.Type.INTEGER)
            semantic_fusion_node.object_params[obj]['window_length'] = semantic_fusion_node.get_parameter('person.window_length').get_parameter_value().integer_value
            semantic_fusion_node.object_params[obj]['model_input_dim'] = semantic_fusion_node.get_parameter('person.model_input_dim').get_parameter_value().integer_value

        # Object attribute parameters
        semantic_fusion_node.object_params[obj]['attributes'] = {}
        try_to_declare_parameter(semantic_fusion_node,obj + '.attributes.variables', rclpy.Parameter.Type.STRING_ARRAY)
        att_vars = semantic_fusion_node.get_parameter(obj + '.attributes.variables').get_parameter_value().string_array_value
        for att_var in att_vars:
            if att_var != '':
                try_to_declare_parameter(semantic_fusion_node,obj + '.attributes.' + att_var + '.labels', rclpy.Parameter.Type.STRING_ARRAY)
                try_to_declare_parameter(semantic_fusion_node,obj + '.attributes.' + att_var + '.probs', rclpy.Parameter.Type.DOUBLE_ARRAY)
                try_to_declare_parameter(semantic_fusion_node,obj + '.attributes.' + att_var + '.upper_prob_limit', rclpy.Parameter.Type.DOUBLE)
                try_to_declare_parameter(semantic_fusion_node,obj + '.attributes.' + att_var + '.lower_prob_limit', rclpy.Parameter.Type.DOUBLE)

                semantic_fusion_node.object_params[obj]['attributes'][att_var] = {}
                semantic_fusion_node.object_params[obj]['attributes'][att_var]['labels'] = semantic_fusion_node.get_parameter(obj + '.attributes.' + att_var + '.labels').get_parameter_value().string_array_value
                semantic_fusion_node.object_params[obj]['attributes'][att_var]['probs'] = semantic_fusion_node.get_parameter(obj + '.attributes.' + att_var + '.probs').get_parameter_value().double_array_value
                semantic_fusion_node.object_params[obj]['attributes'][att_var]['upper_prob_limit'] = semantic_fusion_node.get_parameter(obj + '.attributes.' + att_var + '.upper_prob_limit').get_parameter_value().double_value
                semantic_fusion_node.object_params[obj]['attributes'][att_var]['lower_prob_limit'] = semantic_fusion_node.get_parameter(obj + '.attributes.' + att_var + '.lower_prob_limit').get_parameter_value().double_value

        # Object state parameters
        semantic_fusion_node.object_params[obj]['states'] = {}
        try_to_declare_parameter(semantic_fusion_node,obj + '.states.variables', rclpy.Parameter.Type.STRING_ARRAY)
        state_vars = semantic_fusion_node.get_parameter(obj + '.states.variables').get_parameter_value().string_array_value
        for state_var in state_vars:
            if state_var != '':
                try_to_declare_parameter(semantic_fusion_node,obj + '.states.' + state_var + '.symbol', rclpy.Parameter.Type.STRING)
                try_to_declare_parameter(semantic_fusion_node,obj + '.states.' + state_var + '.labels', rclpy.Parameter.Type.STRING_ARRAY)
                try_to_declare_parameter(semantic_fusion_node,obj + '.states.' + state_var + '.probs', rclpy.Parameter.Type.DOUBLE_ARRAY)
                try_to_declare_parameter(semantic_fusion_node,obj + '.states.' + state_var + '.upper_prob_limit', rclpy.Parameter.Type.DOUBLE)
                try_to_declare_parameter(semantic_fusion_node,obj + '.states.' + state_var + '.lower_prob_limit', rclpy.Parameter.Type.DOUBLE)

                semantic_fusion_node.object_params[obj]['states'][state_var] = {}
                semantic_fusion_node.object_params[obj]['states'][state_var]['symbol'] = semantic_fusion_node.get_parameter(obj + '.states.' + state_var + '.symbol').get_parameter_value().string_value
                semantic_fusion_node.object_params[obj]['states'][state_var]['labels'] = semantic_fusion_node.get_parameter(obj + '.states.' + state_var + '.labels').get_parameter_value().string_array_value
                semantic_fusion_node.object_params[obj]['states'][state_var]['probs'] = semantic_fusion_node.get_parameter(obj + '.states.' + state_var + '.probs').get_parameter_value().double_array_value
                semantic_fusion_node.object_params[obj]['states'][state_var]['upper_prob_limit'] = semantic_fusion_node.get_parameter(obj + '.states.' + state_var + '.upper_prob_limit').get_parameter_value().double_value
                semantic_fusion_node.object_params[obj]['states'][state_var]['lower_prob_limit'] = semantic_fusion_node.get_parameter(obj + '.states.' + state_var + '.lower_prob_limit').get_parameter_value().double_value

        # Communication parameters
        semantic_fusion_node.object_params[obj]['comms'] = {}
        try_to_declare_parameter(semantic_fusion_node,obj + '.comms.labels', rclpy.Parameter.Type.STRING_ARRAY)
        semantic_fusion_node.object_params[obj]['comms']['labels'] = semantic_fusion_node.get_parameter(obj + '.comms.labels').get_parameter_value().string_array_value
        
        if semantic_fusion_node.object_params[obj]['comms']['labels'] != '':
            try_to_declare_parameter(semantic_fusion_node,obj + '.comms.symbol', rclpy.Parameter.Type.STRING)
            try_to_declare_parameter(semantic_fusion_node,obj + '.comms.probs', rclpy.Parameter.Type.DOUBLE_ARRAY)
            try_to_declare_parameter(semantic_fusion_node,obj + '.comms.upper_prob_limit', rclpy.Parameter.Type.DOUBLE)
            try_to_declare_parameter(semantic_fusion_node,obj + '.comms.lower_prob_limit', rclpy.Parameter.Type.DOUBLE)
            semantic_fusion_node.object_params[obj]['comms']['symbol'] = semantic_fusion_node.get_parameter(obj + '.comms.symbol').get_parameter_value().string_value
            semantic_fusion_node.object_params[obj]['comms']['probs'] = semantic_fusion_node.get_parameter(obj + '.comms.probs').get_parameter_value().double_array_value
            semantic_fusion_node.object_params[obj]['comms']['upper_prob_limit'] = semantic_fusion_node.get_parameter(obj + '.comms.upper_prob_limit').get_parameter_value().double_value
            semantic_fusion_node.object_params[obj]['comms']['lower_prob_limit'] = semantic_fusion_node.get_parameter(obj + '.comms.lower_prob_limit').get_parameter_value().double_value

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


def normalize_vector(vec, upper, lower):
    for ii, num in enumerate(vec):
        if num > upper:
            vec[ii] = upper
        elif num < lower:
            vec[ii] = lower
    return vec

def time_to_float(sec, nsec):
    return sec + nsec/1e9