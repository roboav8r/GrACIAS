#!/usr/bin/env python3

# Imports
import os
import time
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import rclpy
from rclpy.node import Node

from situated_hri_interfaces.srv import SceneVisRec, SceneAudioRec

from PIL import Image as PILImage
import cv2
import cv_bridge

import torch, clip

# Parameters
filepath = '../../bags/e1_est_tuning'
actual_scene_labels = ['campus','courtyard','lab','lobby']
cpjku_audio_scene_labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

# Helper functions
def typename(topic_name, topic_types):
    for topic_type in topic_types:
        if topic_type.name == topic_name:
            return topic_type.type
    raise ValueError(f"topic {topic_name} not in bag")

# ROS members
# rclpy.init()
# node = Node('compute_obs_models_node')

### COMPUTE CLIP VISUAL RECOGNITION MODEL
bridge = cv_bridge.CvBridge()
nn_img_size = 416

# CLIP
clip_device = "cuda" if torch.cuda.is_available() else "cpu"
clip_models = ['RN50','ViT-B/32','ViT-L/14'] # available: ['RN50', 'RN101','RN50x4','RN50x16','RN50x64','ViT-B/32','ViT-B/16','ViT-L/14','ViT-L/14@336px']

### COMPUTE CLIP ROLE OBS MODELS
actual_role_labels = ['pedestrian','teammate','supervisor']
est_role_labels = ['pedestrian_est','teammate_est','supervisor_est']

role_descriptions = {'basic_description': ['person','person with orange shirt and orange hat','person with orange shirt and orange hat and yellow vest'],
                     'detailed_description': ['a picture of a person','a picture of a person wearing an orange texas robotics t-shirt and an orange texas hat','a picture of a person wearing an orange texas robotics t-shirt and an orange texas hat and a yellow safety vest']}

clip_role_results = dict()

for model in clip_models:
    clip_model, clip_preprocess = clip.load(model, device=clip_device)

    for desc_name in role_descriptions.keys():

        # Compute text features
        desc_text = role_descriptions[desc_name]
        text_tokens = clip.tokenize(desc_text).to(clip_device)
        text_features = clip_model.encode_text(text_tokens)

        # Reset observation models
        clip_role_obs_model = dict()
        clip_role_inference_times = dict()
        for scene in actual_scene_labels:
            clip_role_obs_model[scene] = np.zeros((len(actual_role_labels),len(desc_text)))
            clip_role_inference_times[scene] = []
        clip_role_obs_model['overall'] = np.zeros((len(actual_role_labels),len(desc_text)))
        clip_role_inference_times['overall'] = []

        # Traverse filepath
        for root, _, files in os.walk(filepath):
        
            # Examine each file
            for file in files:
                
                # If it is an mcap, examine
                if os.path.splitext(file)[-1] == '.mcap':
        
                    # Get labels based on filepath
                    labels = os.path.normpath(root).split(os.sep)
                    command_actual = labels[-1]
                    cmd_mode_actual = labels[-2]
                    role_actual = labels[-3]
                    scene_actual = labels[-4]
                    iteration = os.path.splitext(file)[-2].split('_')[-1]
        
                    scene_actual_idx = actual_scene_labels.index(scene_actual)
                    role_actual_idx = actual_role_labels.index(role_actual)
                    
                    # print('Command: %s\nModality: %s\nRole: %s (%s)\nScene: %s (%s)\n\n' % (command_actual, cmd_mode_actual, role_actual, role_actual_idx, scene_actual, scene_actual_idx))
        
                    # Create reader object           
                    reader = rosbag2_py.SequentialReader()            
                    reader.open(
                        rosbag2_py.StorageOptions(uri=os.path.join(root,file), storage_id="mcap"),
                        rosbag2_py.ConverterOptions(
                            input_serialization_format="cdr", output_serialization_format="cdr"
                        ),
                    )
                    topic_types = reader.get_all_topics_and_types()
        
                    # Iterate through messages
                    img_received = False
                    while reader.has_next():
                        topic, data, timestamp = reader.read_next()
                        msg_type = get_message(typename(topic,topic_types))
        
                        if topic=='/oak/rgb/image_raw':
                            latest_img_msg = deserialize_message(data,msg_type)
                            img_received = True
        
                        if topic=='/oak/nn/spatial_detections':
                            if img_received:
        
                                latest_det_msg = deserialize_message(data,msg_type)
                                for det in latest_det_msg.detections:
        
                                    # look for human detected ~3m in front of camera
                                    dist_to_3m_in_front = np.linalg.norm([det.results[0].pose.pose.position.x,det.results[0].pose.pose.position.y, det.results[0].pose.pose.position.z - 3.])
                                    
                                    if (dist_to_3m_in_front < 0.5 and det.results[0].hypothesis.class_id=='0'):

                                        start_time = time.time() # start clock
                                        
                                        # Extract image snippet of person in front of robot
                                        cv_image = bridge.imgmsg_to_cv2(latest_img_msg, desired_encoding='rgb8')
           
                                        rows = cv_image.shape[0]
                                        cols = cv_image.shape[1]     
                                        scale = np.min([rows,cols])/nn_img_size
                                        col_off = int((cols - np.min([rows,cols]))/2)
                                        row_off = int((rows - np.min([rows,cols]))/2)
        
                                        bbox_center_x = det.bbox.center.position.x
                                        bbox_center_y = det.bbox.center.position.y
                                        bbox_size_x = det.bbox.size.x
                                        bbox_size_y = det.bbox.size.y
        
                                        col_min = col_off + np.max([0, int(scale*(bbox_center_x - bbox_size_x/2))])
                                        row_min = row_off + np.max([0, int(scale*(bbox_center_y - bbox_size_y/2))])
                                        col_max = col_off + np.min([np.min([rows, cols]), int(scale*(bbox_center_x + bbox_size_x/2))])
                                        row_max = row_off + np.min([np.min([rows, cols]), int(scale*(bbox_center_y + bbox_size_y/2))])

                                        # CLIP recognition
                                        pil_image = PILImage.fromarray(cv_image[row_min:row_max,col_min:col_max,:])
                                        clip_image = clip_preprocess(pil_image).unsqueeze(0).to(clip_device)
                                        image_features = clip_model.encode_image(clip_image)

                                        logits_per_image, _ = clip_model(clip_image, text_tokens)
                                        role_probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()[0]

                                        stop_time = time.time() - start_time

                                        clip_role_obs_model[scene_actual][role_actual_idx, np.argmax(role_probs)] +=1
                                        clip_role_obs_model['overall'][role_actual_idx, np.argmax(role_probs)] +=1

                                        clip_role_inference_times[scene_actual].append(stop_time)
                                        clip_role_inference_times['overall'].append(stop_time)

                    del reader

        # Save model to results dict
        key_name = "%s_%s" % (model, desc_name)
        clip_role_results[key_name] = dict()
        clip_role_results[key_name]['model'] = clip_role_obs_model
        clip_role_results[key_name]['times'] = clip_role_inference_times

# Save clip role estimation to files
for model_key in clip_role_results.keys():

    path_key = model_key.replace("/","")
    
    for scene_key in clip_role_results[model_key]['model'].keys():

        # Save times to json file
        with open("../../results/exp0_obs_models/role_recognition/%s/%s_inference_times.json" % (scene_key,path_key),"w") as outfile:
            outfile.write(json.dumps(clip_role_results[model_key]['times']))

        
        raw_df = pd.DataFrame(clip_role_results[model_key]['model'][scene_key], index=actual_role_labels, columns=est_role_labels)
        raw_df.to_csv("../../results/exp0_obs_models/role_recognition/%s/%s_role_obs_model_raw.csv" % (scene_key,path_key))
               
        # Smooth the model
        smoothed_clip_model = raw_df + 1
        smoothed_clip_model_df = pd.DataFrame(smoothed_clip_model,index=actual_role_labels, columns=est_role_labels)
        smoothed_clip_model_df.to_csv("../../results/exp0_obs_models/role_recognition/%s/%s_role_obs_model_smoothed.csv" % (scene_key,path_key))
        
        # Normalize the smoothed model
        norm_vector = np.expand_dims(smoothed_clip_model.sum(axis=1),axis=1)
        
        normalized_clip_model = smoothed_clip_model / norm_vector
        normalized_clip_model_df = pd.DataFrame(normalized_clip_model,index=actual_role_labels, columns=est_role_labels)
        normalized_clip_model_df.to_csv("../../results/exp0_obs_models/role_recognition/%s/%s_role_obs_model_normalized.csv" % (scene_key,path_key))
        


# # Cleanup
# rclpy.shutdown()

