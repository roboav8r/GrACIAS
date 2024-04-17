#!/usr/bin/env python3

import os
import json
import yaml
import torch
import numpy as np
from pathlib import Path
import importlib
import librosa
import torchaudio
import torchaudio.transforms as T

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from audio_common_msgs.msg import AudioDataStamped
from std_msgs.msg import String

class VoiceProcNode(Node):

    def __init__(self):
        super().__init__('voice_proc_node')

        self.subscription = self.create_subscription(AudioDataStamped, 'audio_data', self.audio_data_callback, 10)
        self.audio_scene_publisher = self.create_publisher(String, 'voice_data', 10)
        
        # Declare parameters with default values
        self.declare_parameter('n_channels', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('sample_rate', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('frame_size', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('voice_index', rclpy.Parameter.Type.INTEGER_ARRAY)

        # Retrieve parameters
        self.n_channels = self.get_parameter('n_channels').get_parameter_value().integer_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.frame_size = self.get_parameter('frame_size').get_parameter_value().integer_value
        self.voice_idx = self.get_parameter('voice_index').get_parameter_value().integer_array_value

        # Audio data storage
        self.frame = torch.zeros([self.frame_size, self.n_channels],dtype=torch.float16)
        self.voice_channels = torch.zeros([self.frame_size, len(self.voice_idx)],dtype=torch.float16)
        # self.voice_channels = self.voice_channels.to('cuda')
        self.min_voice_len = 16800
        self.n_trailing_frames = 2 # prevent early cutoff. Only stop recording if silent for n consecutive frames.
        self.n_silent = 0 
        self.recording = False

    def audio_data_callback(self, msg):
        # self.get_logger().info("Audio CB 1")

        # self.frame = torch.from_numpy(np.frombuffer(msg.audio.data,dtype=np.float16)).view(-1,self.n_channels)


        chunk = torch.from_numpy(np.frombuffer(msg.audio.data,dtype=np.float16)).view(-1,self.n_channels)

        # self.get_logger().info('Got chunk with size %s' % (str(chunk.size())))

        self.voice_chunk = chunk[:,self.voice_idx]

        # self.voice_channels = self.voice_channels.to('cuda')
        # self.get_logger().info("Audio CB 2")

        # run VAD on voice channels
        self.voice_data = torchaudio.functional.vad(self.voice_chunk.T, self.sample_rate, trigger_level=3.0, pre_trigger_time=0.2)

        self.get_logger().info('Got voice_data with size %s' % (str(self.voice_data.size())))


        # If contains voice data
        if self.voice_data.size(1) != self.min_voice_len:

            self.n_silent = 0

            self.get_logger().info('Voice data detected')

            # if already recording, append to existing voice tensor
            if self.recording:
                self.get_logger().info('Continuing recording')
                self.voice_tensor = torch.cat((self.voice_tensor,self.voice_chunk),0)

            # If not recording, start recording with existing voice chunk
            else:
                self.get_logger().info('Starting recording')
                self.recording = True
                self.voice_tensor = self.voice_chunk
                

        # If it doesn't contain voice data
        else:
            self.get_logger().info('NO voice data detected')
            self.n_silent +=1

            # If recording, stop recording and process
            if self.recording:

                if self.n_silent >= self.n_trailing_frames:

                    self.get_logger().info('Ending recording')
                    self.recording = False
                    torch.save(self.voice_tensor.T,'voice_data.pt')

                else: 
                    self.get_logger().info('Continuing recording')
                    self.voice_tensor = torch.cat((self.voice_tensor,self.voice_chunk),0)

            # If not recording, do nothing
        


        # # Roll the frame, and replace oldest contents with new chunk
        # self.frame = torch.roll(self.frame, -chunk.size(0), 0)
        # self.frame[-chunk.size(0):,:] = -chunk

        # self.get_logger().info('Computed frame with size %s' % (str(self.frame.size())))



        # scene_msg = String()
        # scene_msg.data = "Class: %s, %s%%" % (self.audio_scene_labels[class_idx], conf.item())
        # self.audio_scene_publisher.publish(scene_msg)


def main(args=None):
    rclpy.init(args=args)
    voice_proc_node = VoiceProcNode()
    rclpy.spin(voice_proc_node)
    voice_proc_node.destroy_node()
    rclpy.shutdown()

    


if __name__ == '__main__':
    main()