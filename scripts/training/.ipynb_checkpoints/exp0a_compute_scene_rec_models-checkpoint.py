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
with open("../../results/exp0_obs_models/cpjku_audio_scene_inference_times.json","w") as outfile:
    outfile.write(json.dumps(cpjku_inference_times))

# Save pandas df to csv
scene_labels = ['campus','courtyard','lab','lobby']
audio_obs_labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

audio_model_df = pd.DataFrame(cpjku_audio_scene_model,columns=audio_obs_labels,index=scene_labels)
audio_model_df.to_csv("../../results/exp0_obs_models/cpjku_audio_scene_obs_model_raw.csv")

# Smooth the model
smoothed_audio_model = cpjku_audio_scene_model + 1
smoothed_audio_model_df = pd.DataFrame(smoothed_audio_model,columns=audio_obs_labels,index=scene_labels)
smoothed_audio_model_df.to_csv("../../results/exp0_obs_models/cpjku_audio_scene_obs_model_smoothed.csv")

# Normalize the smoothed model
normalized_audio_model = smoothed_audio_model / smoothed_audio_model.sum(axis=1)[:,np.newaxis]
normalized_audio_model_df = pd.DataFrame(normalized_audio_model,columns=audio_obs_labels,index=scene_labels)
normalized_audio_model_df.to_csv("../../results/exp0_obs_models/cpjku_audio_scene_obs_model_normalized.csv")


### COMPUTE CLIP VISUAL RECOGNITION MODEL




# Cleanup
rclpy.shutdown()

