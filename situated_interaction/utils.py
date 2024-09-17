#!/usr/bin/env python3

import numpy as np
import rclpy

def load_object_params(semantic_fusion_node):
    semantic_fusion_node.object_params = {}
    semantic_fusion_node.declare_parameter('objects_of_interest', rclpy.Parameter.Type.STRING_ARRAY)
    semantic_fusion_node.objects_of_interest = semantic_fusion_node.get_parameter('objects_of_interest').get_parameter_value().string_array_value

    for obj in semantic_fusion_node.objects_of_interest:

        semantic_fusion_node.object_params[obj] = {}

        # Object attribute parameters
        semantic_fusion_node.object_params[obj]['attributes'] = {}
        semantic_fusion_node.declare_parameter(obj + '.attributes.variables', rclpy.Parameter.Type.STRING_ARRAY)
        att_vars = semantic_fusion_node.get_parameter(obj + '.attributes.variables').get_parameter_value().string_array_value
        for att_var in att_vars:
            if att_var != '':
                semantic_fusion_node.declare_parameter(obj + '.attributes.' + att_var + '.labels', rclpy.Parameter.Type.STRING_ARRAY)
                semantic_fusion_node.declare_parameter(obj + '.attributes.' + att_var + '.probs', rclpy.Parameter.Type.DOUBLE_ARRAY)
                semantic_fusion_node.declare_parameter(obj + '.attributes.' + att_var + '.upper_prob_limit', rclpy.Parameter.Type.DOUBLE)
                semantic_fusion_node.declare_parameter(obj + '.attributes.' + att_var + '.lower_prob_limit', rclpy.Parameter.Type.DOUBLE)

                semantic_fusion_node.object_params[obj]['attributes'][att_var] = {}
                semantic_fusion_node.object_params[obj]['attributes'][att_var]['labels'] = semantic_fusion_node.get_parameter(obj + '.attributes.' + att_var + '.labels').get_parameter_value().string_array_value
                semantic_fusion_node.object_params[obj]['attributes'][att_var]['probs'] = semantic_fusion_node.get_parameter(obj + '.attributes.' + att_var + '.probs').get_parameter_value().double_array_value
                semantic_fusion_node.object_params[obj]['attributes'][att_var]['upper_prob_limit'] = semantic_fusion_node.get_parameter(obj + '.attributes.' + att_var + '.upper_prob_limit').get_parameter_value().double_value
                semantic_fusion_node.object_params[obj]['attributes'][att_var]['lower_prob_limit'] = semantic_fusion_node.get_parameter(obj + '.attributes.' + att_var + '.lower_prob_limit').get_parameter_value().double_value

        # Object state parameters
        semantic_fusion_node.object_params[obj]['states'] = {}
        semantic_fusion_node.declare_parameter(obj + '.states.variables', rclpy.Parameter.Type.STRING_ARRAY)
        state_vars = semantic_fusion_node.get_parameter(obj + '.states.variables').get_parameter_value().string_array_value
        for state_var in state_vars:
            if state_var != '':
                semantic_fusion_node.declare_parameter(obj + '.states.' + state_var + '.labels', rclpy.Parameter.Type.STRING_ARRAY)
                semantic_fusion_node.declare_parameter(obj + '.states.' + state_var + '.probs', rclpy.Parameter.Type.DOUBLE_ARRAY)
                semantic_fusion_node.declare_parameter(obj + '.states.' + state_var + '.upper_prob_limit', rclpy.Parameter.Type.DOUBLE)
                semantic_fusion_node.declare_parameter(obj + '.states.' + state_var + '.lower_prob_limit', rclpy.Parameter.Type.DOUBLE)

                semantic_fusion_node.object_params[obj]['states'][state_var] = {}
                semantic_fusion_node.object_params[obj]['states'][state_var]['labels'] = semantic_fusion_node.get_parameter(obj + '.states.' + state_var + '.labels').get_parameter_value().string_array_value
                semantic_fusion_node.object_params[obj]['states'][state_var]['probs'] = semantic_fusion_node.get_parameter(obj + '.states.' + state_var + '.probs').get_parameter_value().double_array_value
                semantic_fusion_node.object_params[obj]['states'][state_var]['upper_prob_limit'] = semantic_fusion_node.get_parameter(obj + '.states.' + state_var + '.upper_prob_limit').get_parameter_value().double_value
                semantic_fusion_node.object_params[obj]['states'][state_var]['lower_prob_limit'] = semantic_fusion_node.get_parameter(obj + '.states.' + state_var + '.lower_prob_limit').get_parameter_value().double_value

        # Communication parameters
        semantic_fusion_node.object_params[obj]['comms'] = {}
        semantic_fusion_node.declare_parameter(obj + '.comms.labels', rclpy.Parameter.Type.STRING_ARRAY)
        semantic_fusion_node.object_params[obj]['comms']['labels'] = semantic_fusion_node.get_parameter(obj + '.comms.labels').get_parameter_value().string_array_value
        
        if semantic_fusion_node.object_params[obj]['comms']['labels'] != '':
            semantic_fusion_node.declare_parameter(obj + '.comms.probs', rclpy.Parameter.Type.DOUBLE_ARRAY)
            semantic_fusion_node.declare_parameter(obj + '.comms.upper_prob_limit', rclpy.Parameter.Type.DOUBLE)
            semantic_fusion_node.declare_parameter(obj + '.comms.lower_prob_limit', rclpy.Parameter.Type.DOUBLE)
            semantic_fusion_node.object_params[obj]['comms']['probs'] = semantic_fusion_node.get_parameter(obj + '.comms.probs').get_parameter_value().double_array_value
            semantic_fusion_node.object_params[obj]['comms']['upper_prob_limit'] = semantic_fusion_node.get_parameter(obj + '.comms.upper_prob_limit').get_parameter_value().double_value
            semantic_fusion_node.object_params[obj]['comms']['lower_prob_limit'] = semantic_fusion_node.get_parameter(obj + '.comms.lower_prob_limit').get_parameter_value().double_value


        # # Get AR tag parameters
        # semantic_fusion_node.declare_parameter(obj + '.comms.ar_tag_ids', rclpy.Parameter.Type.INTEGER_ARRAY)
        # semantic_fusion_node.declare_parameter(obj + '.comms.ar_tag_types', rclpy.Parameter.Type.STRING_ARRAY)
        # semantic_fusion_node.declare_parameter(obj + '.comms.ar_tag_words', rclpy.Parameter.Type.STRING_ARRAY)
        # semantic_fusion_node.object_params[obj]['comms']['ar_tag_ids'] = semantic_fusion_node.get_parameter(obj + '.comms.ar_tag_ids').get_parameter_value().integer_array_value
        # semantic_fusion_node.object_params[obj]['comms']['ar_tag_types'] = semantic_fusion_node.get_parameter(obj + '.comms.ar_tag_types').get_parameter_value().string_array_value
        # semantic_fusion_node.object_params[obj]['comms']['ar_tag_words'] = semantic_fusion_node.get_parameter(obj + '.comms.ar_tag_words').get_parameter_value().string_array_value
        # semantic_fusion_node.object_params[obj]['comms']['ar_tag_dict'] = {}
        # assert len(semantic_fusion_node.object_params[obj]['comms']['ar_tag_ids']) == len(semantic_fusion_node.object_params[obj]['comms']['ar_tag_types'])
        # assert len(semantic_fusion_node.object_params[obj]['comms']['ar_tag_words']) == len(semantic_fusion_node.object_params[obj]['comms']['ar_tag_types'])
        # for index, tag_id in enumerate(semantic_fusion_node.object_params[obj]['comms']['ar_tag_ids']):
        #     semantic_fusion_node.object_params[obj]['comms']['ar_tag_dict'][tag_id] = {}
        #     semantic_fusion_node.object_params[obj]['comms']['ar_tag_dict'][tag_id]['type'] = semantic_fusion_node.object_params[obj]['comms']['ar_tag_types'][index]
        #     semantic_fusion_node.object_params[obj]['comms']['ar_tag_dict'][tag_id]['word'] = semantic_fusion_node.object_params[obj]['comms']['ar_tag_words'][index]

        # # Get gesture parameters
        # semantic_fusion_node.declare_parameter(obj + '.comms.gesture_descriptions', rclpy.Parameter.Type.STRING_ARRAY)
        # semantic_fusion_node.declare_parameter(obj + '.comms.transcripts', rclpy.Parameter.Type.STRING_ARRAY)
        # semantic_fusion_node.declare_parameter(obj + '.comms.probs', rclpy.Parameter.Type.DOUBLE_ARRAY)
        # semantic_fusion_node.declare_parameter(obj + '.comms.gesture_sensor_model_coeffs', rclpy.Parameter.Type.DOUBLE_ARRAY)
        # semantic_fusion_node.declare_parameter(obj + '.comms.verbal_sensor_model_coeffs', rclpy.Parameter.Type.DOUBLE_ARRAY)
        # semantic_fusion_node.object_params[obj]['comms']['gesture_descriptions'] = semantic_fusion_node.get_parameter(obj + '.comms.gesture_descriptions').get_parameter_value().string_array_value
        # semantic_fusion_node.object_params[obj]['comms']['transcripts'] = semantic_fusion_node.get_parameter(obj + '.comms.transcripts').get_parameter_value().string_array_value
        # semantic_fusion_node.object_params[obj]['comms']['probs'] = semantic_fusion_node.get_parameter(obj + '.comms.probs').get_parameter_value().double_array_value
        # semantic_fusion_node.object_params[obj]['comms']['gesture_sensor_model_coeffs'] = semantic_fusion_node.get_parameter(obj + '.comms.gesture_sensor_model_coeffs').get_parameter_value().double_array_value
        # semantic_fusion_node.object_params[obj]['comms']['gesture_sensor_model_array'] = np.array(semantic_fusion_node.object_params[obj]['comms']['gesture_sensor_model_coeffs']).reshape(-1,len(semantic_fusion_node.object_params[obj]['comms']['labels']))
        # semantic_fusion_node.object_params[obj]['comms']['verbal_sensor_model_coeffs'] = semantic_fusion_node.get_parameter(obj + '.comms.verbal_sensor_model_coeffs').get_parameter_value().double_array_value
        # semantic_fusion_node.object_params[obj]['comms']['verbal_sensor_model_array'] = np.array(semantic_fusion_node.object_params[obj]['comms']['verbal_sensor_model_coeffs']).reshape(-1,len(semantic_fusion_node.object_params[obj]['comms']['labels']))
    

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