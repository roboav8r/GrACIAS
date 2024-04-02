#!/usr/bin/env python3

import torch
import torchaudio

import rclpy
from rclpy.node import Node
from audio_common_msgs.msg import AudioDataStamped
# from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
# import cv2
import numpy as np



class AudioProcNode(Node):

    def __init__(self):
        super().__init__('audio_proc_node')
        self.subscription = self.create_subscription(
            AudioDataStamped,
            'audio_data',
            self.audio_data_callback,
            10)
        self.publisher = self.create_publisher(
            Float64MultiArray,
            'wav_float_data',
            10)
        
        # Declare parameters with default values
        self.declare_parameter('n_channels', 6)
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('hop_size', 1600) # .1 seconds
        self.declare_parameter('frame_size', 16000)

        # Retrieve parameters
        self.n_channels = self.get_parameter('n_channels').get_parameter_value().integer_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.hop_size = self.get_parameter('hop_size').get_parameter_value().integer_value
        self.frame_size = self.get_parameter('frame_size').get_parameter_value().integer_value

        # Channel position map
        self.channel_data = {1: {'pos': [.2298, -.2298, 0.]}, 
                             2: {'pos': [.2298, .2298, 0.]}, 
                             3: {'pos': [-.2298, .2298, 0.]}, 
                             4: {'pos': [-.2298, -.2298, 0.]}}
        self.speed_sound = 343.0
        # self.beam_azimuths = [0., 30., 60., 90., 120., 150., 180., 210., 240., 270., 300., 330.]
        self.beam_data = {'12_o_clock': {'az': 0.}, 
                          '11_o_clock': {'az': 30.}, 
                          '10_o_clock': {'az': 60.}, 
                          '1_o_clock': {'az': 300.}, 
                          '2_o_clock': {'az': 330.}}

        # Audio data storage
        self.frame = torch.zeros([self.frame_size*self.n_channels],dtype=torch.float16)
        self.channels = torch.zeros([self.frame_size, self.n_channels],dtype=torch.float16)

    # def form_beams(self):
    #     for az in self.beam_azimuths:
    #         for channel in self.channel_data.keys():

    #             # compute projection of position in az direction

    #             # save max/closest to sound source - this is where signal arrives first

    #             # find per-channel delay time = (dist_max - dist_i)/self.speed_sound
                
    #             # find per-channel sample shift (min should be 0)

    #         self.beams = torch.zeros([self.frame_size - self.max_shift, self.n_channels],dtype=torch.float16):

            # beam = 

    def audio_data_callback(self, msg):

        self.frame = torch.from_numpy(np.frombuffer(msg.audio.data,dtype=np.float16)).view(-1,self.n_channels) # .view(self.n_channels,-1)
        # self.channels = self.frame.view(-1,self.n_channels)
        torch.save(self.frame,'frame_data_recovered.pt')

        # Form beams
        # for beam in beams

        # Process audio data and convert it to image
        # Here, we are just creating a dummy image for demonstration
        dummy_image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)

        # Create a float message
        float_msg = Float64MultiArray()
        for channel in self.channel_data.keys():
            float_msg.data.append(self.frame[-1,channel])

        # Publish the image message
        self.publisher.publish(float_msg)


def main(args=None):
    rclpy.init(args=args)
    audio_proc_node = AudioProcNode()
    rclpy.spin(audio_proc_node)
    audio_proc_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
