#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import importlib
import librosa
import torchaudio.transforms as T

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from audio_common_msgs.msg import AudioDataStamped
from std_msgs.msg import String

class AudioBeamformerNode(Node):

    def __init__(self):
        super().__init__('bf_voice_node')
        self.subscription = self.create_subscription(AudioDataStamped, 'audio_data', self.audio_data_callback, 10)
        self.audio_scene_publisher = self.create_publisher(String, 'audio_scene', 10)
        
        # Declare parameters with default values
        self.declare_parameter('n_channels', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('audio_sample_rate', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('hop_size', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('frame_size', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('voice_index', rclpy.Parameter.Type.INTEGER_ARRAY)

        # Retrieve parameters
        self.n_channels = self.get_parameter('n_channels').get_parameter_value().integer_value
        self.sample_rate = self.get_parameter('audio_sample_rate').get_parameter_value().integer_value
        self.hop_size = self.get_parameter('hop_size').get_parameter_value().integer_value
        self.frame_size = self.get_parameter('frame_size').get_parameter_value().integer_value
        self.voice_index = self.get_parameter('voice_index').get_parameter_value().integer_array_value

        # Audio data storage
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hop = torch.zeros([self.hop_size, self.n_channels],dtype=torch.float16)
        self.frame = torch.zeros([self.frame_size, self.n_channels],dtype=torch.float16)
        self.voice_frame = torch.zeros([self.frame_size, len(self.voice_index)],dtype=torch.float16) 
        self.hop = self.hop.to(self.torch_device)
        self.frame = self.frame.to(self.torch_device)
        self.voice_frame = self.voice_frame.to(self.torch_device)


    def audio_data_callback(self, msg):

        self.hop = torch.from_numpy(np.frombuffer(msg.audio.data,dtype=np.float16)).view(-1,self.n_channels)

        # Roll the frame, and replace oldest contents with new chunk
        self.frame = torch.roll(self.frame, -self.hop.size(0), 0)
        self.frame[-self.hop.size(0):,:] = -self.hop

        self.voice_frame = self.frame[:,self.voice_index]

        torch.save(self.voice_frame.T,'beamformer_voice_frame.pt')

    
def main(args=None):
    rclpy.init(args=args)
    audio_bf_node = AudioBeamformerNode()
    rclpy.spin(audio_bf_node)
    audio_bf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
