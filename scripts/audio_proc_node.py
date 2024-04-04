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

from numba import jit

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
        self.declare_parameter('frame_size', 1600)

        # Retrieve parameters
        self.n_channels = self.get_parameter('n_channels').get_parameter_value().integer_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.frame_size = self.get_parameter('frame_size').get_parameter_value().integer_value

        # Channel position map
        # TODO - add this as param in config file
        # respeaker
        self.channel_data = {1: {'pos': [.02298, -.02298, 0.]}, 
                             2: {'pos': [.02298, .02298, 0.]}, 
                             3: {'pos': [-.02298, .02298, 0.]}, 
                             4: {'pos': [-.02298, -.02298, 0.]}}
        # kinect
        # self.channel_data = {0: {'pos': [0., 0., 0.]},
        #                      1: {'pos': [.04, 0., 0.]},
        #                      2: {'pos': [.02, -.034641016, 0.]}, 
        #                      3: {'pos': [-.02, -.034641016, 0.]},
        #                      4: {'pos': [-.04, 0., 0.]},
        #                      5: {'pos': [-.02, .034641016, 0.]}, 
        #                      6: {'pos': [.02, .034641016, 0.]}}


        self.speed_sound = 343.0
        # TODO - add this as param in config file
        # self.beam_azimuths = [0., 30., 60., 90., 120., 150., 180., 210., 240., 270., 300., 330.]
        self.beam_data = {'12_o_clock': {'az': 0.}, 
                        #   '11_o_clock': {'az': 30.},
                          'front_left': {'az': 45.}, 
                        #   '10_o_clock': {'az': 60.}, 
                        #   '1_o_clock': {'az': 300.},
                          'front_right': {'az': 315.}} 
                        #   '2_o_clock': {'az': 330.}}
        self.max_offset = np.iinfo(np.uint8).min

        # Audio data storage
        # TODO - get datatype from config file
        self.frame = torch.zeros([self.frame_size*self.n_channels],dtype=torch.float16)
        self.channels = torch.zeros([self.frame_size, self.n_channels],dtype=torch.float16)
        # self.frame = torch.zeros([self.frame_size*self.n_channels],dtype=torch.float32)
        # self.channels = torch.zeros([self.frame_size, self.n_channels],dtype=torch.float32)


        # Compute beam parameters
        self.compute_beam_params()

    def beam_projection(self, pos, beam_az):
        pos_angle = np.arctan2(pos[1],pos[0])

        while pos_angle < 0:
            pos_angle += 2*np.pi
        while pos_angle > 2*np.pi:
            pos_angle -= 2*np.pi

        while pos_angle < 0:
            pos_angle += 2*np.pi
        while pos_angle > 2*np.pi:
            pos_angle -= 2*np.pi
        
        delta_angle = beam_az - pos_angle

        return np.sqrt(pos[0]**2 + pos[1]**2)*np.cos(delta_angle)

    def compute_beam_params(self):
        self.get_logger().info('forming beams')

        for beam in self.beam_data.keys():
            self.beam_data[beam]['channel_proj'] = {}
            self.beam_data[beam]['sample_offset'] = {}
            self.beam_data[beam]['max_proj'] = np.finfo(np.float16).min
            self.beam_data[beam]['min_proj'] = np.finfo(np.float16).max

            # Find mic. distance in beam direction
            for channel in self.channel_data.keys():

                # compute projection of mic position in az direction, save to beam dict
                proj = self.beam_projection(self.channel_data[channel]['pos'][:2],np.deg2rad(self.beam_data[beam]['az']))
                self.beam_data[beam]['channel_proj'][channel] = proj

                # save max/closest to sound source - this is where signal arrives first
                self.beam_data[beam]['max_proj'] = np.max([self.beam_data[beam]['max_proj'],proj])
                self.beam_data[beam]['min_proj'] = np.min([self.beam_data[beam]['min_proj'],proj])

            # Find channel sample offsets
            for channel in self.channel_data.keys():
                offset = int((self.beam_data[beam]['channel_proj'][channel] - self.beam_data[beam]['min_proj'])*self.sample_rate/self.speed_sound)
                self.beam_data[beam]['sample_offset'][channel] = offset
                self.max_offset = np.max([self.max_offset,offset])

        # Create empty beam tensor
        self.beams = torch.zeros([self.frame_size - self.max_offset, len(self.beam_data.keys())],dtype=torch.float16)
        self.beams = torch.zeros([self.frame_size - self.max_offset, len(self.beam_data.keys())],dtype=torch.float32)
        
       
        self.get_logger().info(str(self.beam_data))

    def compute_beams(self):
        self.beams = torch.zeros([self.frame_size - self.max_offset, len(self.beam_data.keys())],dtype=torch.float16) # TODO
        
        for ii,beam_key in enumerate(self.beam_data.keys()):
            for channel_key in self.channel_data.keys():

                self.beams[:,ii] += self.frame[(self.max_offset - self.beam_data[beam_key]['sample_offset'][channel_key]):self.frame_size-self.beam_data[beam_key]['sample_offset'][channel_key],channel_key]

            self.beams[:,ii] /= len(self.channel_data.keys())

    def audio_data_callback(self, msg):

        self.frame = torch.from_numpy(np.frombuffer(msg.audio.data,dtype=np.float16)).view(-1,self.n_channels) # .view(self.n_channels,-1) # TODO
        self.compute_beams()
        # self.channels = self.frame.view(-1,self.n_channels)
        torch.save(self.frame,'frame_data_recovered.pt')
        torch.save(self.beams,'beam_data.pt')

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
