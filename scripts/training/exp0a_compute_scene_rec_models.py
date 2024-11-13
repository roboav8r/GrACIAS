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
rclpy.init()
node = Node('compute_obs_models_node')
audio_scene_rec_client = node.create_client(SceneAudioRec, 'scene_audio_rec_service')

# Initialize observation models and time array
cpjku_audio_scene_model = np.zeros((len(actual_scene_labels),len(cpjku_audio_scene_labels)))
cpjku_inference_times = []

### COMPUTE AUDIO SCENE RECOGNITION MODEL
# Traverse filepath
for root, dirs, files in os.walk(filepath):

    # Examine each file
    for file in files:
        
        # If it is an mcap, examine
        if os.path.splitext(file)[-1] == '.mcap':

            # Get labels based on filepath
            labels = os.path.normpath(root).split(os.sep)
            command_actual = labels[-1]
            modality_actual = labels[-2]
            role_actual = labels[-3]
            scene_actual = labels[-4]

            scene_actual_idx = actual_scene_labels.index(scene_actual)
            
            reader = rosbag2_py.SequentialReader()            
            reader.open(
                rosbag2_py.StorageOptions(uri=os.path.join(root,file), storage_id="mcap"),
                rosbag2_py.ConverterOptions(
                    input_serialization_format="cdr", output_serialization_format="cdr"
                ),
            )
            topic_types = reader.get_all_topics_and_types()

            # Iterate through messages
            while reader.has_next():
                topic, data, timestamp = reader.read_next()
                msg_type = get_message(typename(topic,topic_types))

                # Handle audio scene data
                if topic=='/audio_data':
                    audio_scene_req = SceneAudioRec.Request()
                    audio_scene_req.scene_audio = deserialize_message(data,msg_type)
                    future = audio_scene_rec_client.call_async(audio_scene_req)
                    rclpy.spin_until_future_complete(node, future)
                    
                    est_scene_idx = np.argmax(future.result().scene_class.probabilities)
                    
                    cpjku_audio_scene_model[scene_actual_idx,est_scene_idx] += 1
                    cpjku_inference_times.append(future.result().inference_time)
                
            del reader

# Save audio model and times to output
with open("../../results/exp0_obs_models/scene_recognition/audio_scene_inference_times.json","w") as outfile:
    outfile.write(json.dumps(cpjku_inference_times))

# Save pandas df to csv
scene_labels = ['campus','courtyard','lab','lobby']
audio_obs_labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

audio_model_df = pd.DataFrame(cpjku_audio_scene_model,columns=audio_obs_labels,index=scene_labels)
audio_model_df.to_csv("../../results/exp0_obs_models/scene_recognition/cpjku_audio_scene_obs_model_raw.csv")

# Smooth the model
smoothed_audio_model = cpjku_audio_scene_model + 1
smoothed_audio_model_df = pd.DataFrame(smoothed_audio_model,columns=audio_obs_labels,index=scene_labels)
smoothed_audio_model_df.to_csv("../../results/exp0_obs_models/scene_recognition/cpjku_audio_scene_obs_model_smoothed.csv")

# Normalize the smoothed model
normalized_audio_model = smoothed_audio_model / smoothed_audio_model.sum(axis=1)[:,np.newaxis]
normalized_audio_model_df = pd.DataFrame(normalized_audio_model,columns=audio_obs_labels,index=scene_labels)
normalized_audio_model_df.to_csv("../../results/exp0_obs_models/scene_recognition/cpjku_audio_scene_obs_model_normalized.csv")


### COMPUTE CLIP VISUAL RECOGNITION MODEL
bridge = cv_bridge.CvBridge()
nn_img_size = 416

# CLIP
clip_device = "cuda" if torch.cuda.is_available() else "cpu"
clip_models = ['RN50','ViT-B/32','ViT-L/14'] # available: ['RN50', 'RN101','RN50x4','RN50x16','RN50x64','ViT-B/32','ViT-B/16','ViT-L/14','ViT-L/14@336px']

# text descriptions
scene_descriptions = {'basic_description': ['a college campus','an outdoor courtyard','a lab', 'a lobby'],
                     'social_in_out_desc': ['a picture of a public outdoor college campus with high social activity','a picture of a private outdoor courtyard with low social activity','a picture of a private indoor laboratory with low social activity','a picture of a public indoor lobby with high social activity']}

clip_scene_results = dict()

for model in clip_models:
    clip_model, clip_preprocess = clip.load(model, device=clip_device)

    for desc_name in scene_descriptions.keys():

        # Compute text features
        desc_text = scene_descriptions[desc_name]
        text_tokens = clip.tokenize(desc_text).to(clip_device)
        text_features = clip_model.encode_text(text_tokens)

        # Reset observation model
        clip_scene_obs_model = np.zeros((len(actual_scene_labels),len(desc_text)))
        clip_scene_inference_times = []

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
                            
                            start_time = time.time() # start clock
                            
                            cv_image = bridge.imgmsg_to_cv2(latest_img_msg, desired_encoding='rgb8')

                            # CLIP recognition
                            pil_image = PILImage.fromarray(cv_image)
                            clip_image = clip_preprocess(pil_image).unsqueeze(0).to(clip_device)
                            image_features = clip_model.encode_image(clip_image)

                            logits_per_image, _ = clip_model(clip_image, text_tokens)
                            scene_probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()[0]

                            clip_scene_obs_model[scene_actual_idx, np.argmax(scene_probs)] +=1
                            clip_scene_inference_times.append(time.time() - start_time)
                        
                    del reader
                    
        # Save model to results dict
        key_name = "%s_%s" % (model, desc_name)
        clip_scene_results[key_name] = dict()
        clip_scene_results[key_name]['model'] = clip_scene_obs_model
        clip_scene_results[key_name]['times'] = clip_scene_inference_times

# Save visual scene recognition results
for key in clip_scene_results.keys():

    # Remove / to avoid path errors
    key_path = key.replace("/","")
    
    # Save times to json file
    with open("../../results/exp0_obs_models/scene_recognition/" + key_path + "_inference_times.json","w") as outfile:
        outfile.write(json.dumps(clip_scene_results[key]['times']))

    # Save models to csv files
    scene_labels = ['campus_actual','courtyard_actual','lab_actual','lobby_actual']
    clip_obs_labels = ['campus_est','courtyard_est','lab_est','lobby_est']

    clip_model_df = pd.DataFrame(clip_scene_results[key]['model'],columns=clip_obs_labels,index=scene_labels)
    clip_model_df.to_csv("../../results/exp0_obs_models/scene_recognition/" + key_path + "_scene_obs_model_raw.csv")
    
    # Smooth the model
    smoothed_clip_model = clip_model_df + 1
    smoothed_clip_model_df = pd.DataFrame(smoothed_clip_model,columns=clip_obs_labels,index=scene_labels)
    smoothed_clip_model_df.to_csv("../../results/exp0_obs_models/scene_recognition/" + key_path + "_scene_obs_model_smoothed.csv")
    
    # Normalize the smoothed model
    norm_vector = np.expand_dims(smoothed_clip_model.sum(axis=1),axis=1)
    
    normalized_clip_model = smoothed_clip_model / norm_vector
    normalized_clip_model_df = pd.DataFrame(normalized_clip_model,columns=clip_obs_labels,index=scene_labels)
    normalized_clip_model_df.to_csv("../../results/exp0_obs_models/scene_recognition/" + key_path + "_scene_obs_model_normalized.csv")


# Cleanup
rclpy.shutdown()

