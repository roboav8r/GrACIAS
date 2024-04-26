#!/usr/bin/env python3

import torch
import torchaudio
import torch.multiprocessing as mp

import rclpy
from rclpy.node import Node
from audio_common_msgs.msg import AudioDataStamped, AudioInfo

class AudioPublisherNode(Node):
    def __init__(self):
        super().__init__('audio_publisher_node')
        
        # Initialize publishers
        self.audio_data_publisher = self.create_publisher(AudioDataStamped, 'audio_data', 10)
        self.audio_info_publisher = self.create_publisher(AudioInfo, 'audio_info', 10)
        
        # Declare parameters with default values
        self.declare_parameter('n_channels', 6)
        self.declare_parameter('src', 'plughw:1,0')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('hop_size', 1600) # .1 seconds
        self.declare_parameter('frame_size', 1600)
        self.declare_parameter('microphone_frame_id','respeaker_frame')
        self.declare_parameter('codec', 'pcm_s16le')

        # Retrieve parameters
        self.n_channels = self.get_parameter('n_channels').get_parameter_value().integer_value
        self.src = self.get_parameter('src').get_parameter_value().string_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.hop_size = self.get_parameter('hop_size').get_parameter_value().integer_value
        self.frame_size = self.get_parameter('frame_size').get_parameter_value().integer_value
        self.microphone_frame_id = self.get_parameter('microphone_frame_id').get_parameter_value().string_value
        self.codec= self.get_parameter('codec').get_parameter_value().string_value
        self.format = "alsa"
        self.options = {"sample_rate": str(self.sample_rate), "channels": str(self.n_channels)}

        # Create audio_info message
        self.audio_info_msg = AudioInfo()
        self.audio_info_msg.channels = self.n_channels
        self.audio_info_msg.sample_rate = self.sample_rate

        self.audio_data_msg = AudioDataStamped()
        self.audio_data_msg.header.frame_id = self.microphone_frame_id

        # Create frame cache and queue
        # TODO - update for 16/32 bit
        self.frame = torch.zeros([self.frame_size, self.n_channels],dtype=torch.float16)
        # self.frame = torch.zeros([self.frame_size, self.n_channels],dtype=torch.float32)

        # Create stream
        self.get_logger().info("Building StreamReader\nsrc: %s \nformat: %s \noptions: %s \nframes per chunk: %s" % (self.src, self.format, self.options, self.hop_size))
        self.streamer = torchaudio.io.StreamReader(src=self.src, format=self.format, option=self.options)
        self.streamer.add_basic_audio_stream(frames_per_chunk=self.hop_size, decoder='pcm_s32le', sample_rate=self.sample_rate, num_channels=self.n_channels)

        self.get_logger().info(str(self.streamer.get_src_stream_info(0)))

        self.publish_audio_messages()
       
    def publish_audio_messages(self):

        # Build audio dataframe from queue
        for (chunk_a) in self.streamer.stream(timeout=-1, backoff=1.0):

            # Roll the frame, and replace oldest contents with new chunk
            self.frame = torch.roll(self.frame, -chunk_a[0].size(0), 0)
            self.frame[-chunk_a[0].size(0):,:] = -chunk_a[0]

            torch.save(self.frame,'frame_data_original.pt')

            self.audio_data_msg.header.stamp = self.get_clock().now().to_msg()
            self.audio_data_msg.audio.data = self.frame.view(-1).numpy().tobytes()
            self.audio_data_publisher.publish(self.audio_data_msg)
            
            # Publish AudioInfo message
            self.audio_info_publisher.publish(self.audio_info_msg)

def main(args=None):
    rclpy.init(args=args)
    audio_publisher_node = AudioPublisherNode()
    rclpy.spin(audio_publisher_node)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically when the garbage collector destroys the node object)
    del audio_publisher_node.streamer
    audio_publisher_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()