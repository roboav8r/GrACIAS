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

class SceneRecNode(Node):

    def __init__(self):
        super().__init__('scene_rec_node')
        self.subscription = self.create_subscription(AudioDataStamped, 'audio_data', self.audio_data_callback, 10)
        self.audio_scene_publisher = self.create_publisher(String, 'audio_scene', 10)
        
        # Declare parameters with default values
        self.declare_parameter('n_channels', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('sample_rate', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('chunk_length', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('buffer_length', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('channel_index', rclpy.Parameter.Type.INTEGER_ARRAY)

        # Retrieve parameters
        self.n_channels = self.get_parameter('n_channels').get_parameter_value().integer_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.chunk_length = self.get_parameter('chunk_length').get_parameter_value().integer_value
        self.buffer_length = self.get_parameter('buffer_length').get_parameter_value().integer_value
        self.channel_idx = self.get_parameter('channel_index').get_parameter_value().integer_array_value

        # Audio data storage
        self.chunk = torch.zeros([self.chunk_length, self.n_channels],dtype=torch.float16)
        self.buffer = torch.zeros([self.buffer_length, len(self.channel_idx)],dtype=torch.float16)

    def audio_data_callback(self, msg):

        chunk = torch.from_numpy(np.frombuffer(msg.audio.data,dtype=np.float16)).view(-1,self.n_channels)

        self.get_logger().info('Got chunk with size %s' % (str(chunk.size())))
        self.get_logger().info('Buffer has size %s' % (str(self.buffer.size())))

        # Roll the frame, and replace oldest contents with new chunk
        self.buffer = torch.roll(self.buffer, -chunk.size(0), 0)
        self.buffer[-chunk.size(0):,:] = -chunk[:,self.channel_idx]

        # self.get_logger().info('Computed frame with size %s' % (str(self.frame.size())))

        torch.save(self.buffer,'buffer_data_recovered.pt')

    
def main(args=None):
    rclpy.init(args=args)
    audio_proc_node = SceneRecNode()
    rclpy.spin(audio_proc_node)
    audio_proc_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
