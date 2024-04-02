#!/usr/bin/env python3

import torch
import torchaudio
import torch.multiprocessing as mp

import rclpy
from rclpy.node import Node
from audio_common_msgs.msg import AudioDataStamped, AudioInfo


class ContextCacher:
    """Cache the end of input data and prepend the next input data with it.

    Args:
        segment_length (int): The size of main segment.
            If the incoming segment is shorter, then the segment is padded.
        context_length (int): The size of the context, cached and appended.
    """

    def __init__(self, segment_length: int, context_length: int, n_channels: int):
        self.segment_length = segment_length
        self.context_length = context_length
        self.context = torch.zeros([context_length, n_channels],dtype=torch.float16)
        self.frame = torch.zeros([segment_length, n_channels],dtype=torch.float16)

    def __call__(self, chunk: torch.Tensor, audio_node):
        # audio_node.get_logger().info("chunk size: %s" % str(chunk.size()) )
        # audio_node.get_logger().info("chunk size 0: %s" % str(chunk.size(0)) )

        # audio_node.get_logger().info("last chunk row: %s" % str(chunk[-1,:]) )

        self.frame = torch.roll(self.frame, -chunk.size(0), 0)
        self.frame[-chunk.size(0):,:] = chunk

        # audio_node.get_logger().info("updated frame size: %s" % str(self.frame.size()) )
        # audio_node.get_logger().info("last frame row: %s" % str(self.frame[-1,:]) )        

        return self.frame

class AudioPublisherNode(Node):
    def __init__(self):
        super().__init__('audio_publisher_node')
        
        # Initialize publishers
        self.audio_data_publisher = self.create_publisher(AudioDataStamped, 'audio_data', 10)
        self.audio_info_publisher = self.create_publisher(AudioInfo, 'audio_info', 10)
        
        # Declare parameters with default values
        self.declare_parameter('n_channels', 6)
        self.declare_parameter('src', 'hw:4,0')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('hop_size', 1600) # .1 seconds
        self.declare_parameter('frame_size', 1600)

        # Retrieve parameters
        self.n_channels = self.get_parameter('n_channels').get_parameter_value().integer_value
        self.src = self.get_parameter('src').get_parameter_value().string_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.hop_size = self.get_parameter('hop_size').get_parameter_value().integer_value
        self.frame_size = self.get_parameter('frame_size').get_parameter_value().integer_value
        self.format = "alsa"
        self.options = {"sample_rate": str(self.sample_rate),"channels": str(self.n_channels)}

        # Create audio_info message
        self.audio_info_msg = AudioInfo()
        self.audio_info_msg.channels = self.n_channels
        self.audio_info_msg.sample_rate = self.sample_rate
        # Example additional fields setup
        # audio_info_msg.sample_format = "16-bit"

        # Create frame cacher and queue
        self.cacher = ContextCacher(self.frame_size, self.hop_size, self.n_channels)
        # self.q = []

        # Create stream
        self.get_logger().info("Building StreamReader...")
        self.streamer = torchaudio.io.StreamReader(src=self.src, format=self.format, option=self.options)
        self.streamer.add_basic_audio_stream(frames_per_chunk=self.hop_size, sample_rate=self.sample_rate, num_channels=self.n_channels)

        self.get_logger().info(str(self.streamer.get_src_stream_info(0)))

        # Multithreading
        # self.ctx = mp.get_context("spawn")
        # self.q = self.ctx.Queue()
        # self.p = self.ctx.Process(target=self.stream, args=())

        # Declare timer and publisher
        # timer_period = .1  # seconds
        # self.timer = self.create_timer(timer_period, self.publish_audio_messages)

        self.publish_audio_messages()

    # def stream(self):
    #     self.get_logger().info("stream")
    #     for (chunk_a) in self.streamer.stream(timeout=-1, backoff=1.0):
    #         self.q.append([chunk_a])
    #         self.get_logger().info("streamin'")

    # def populate_audio_data(self):
    #     while self.q:
    #         chunk = self.q.pop(0)      
    #         self.get_logger().info("Chunk size: %s \n" % chunk[0][0].size())
    #         self.audio_frame_data = self.cacher(chunk[0][0])
    #         self.get_logger().info("Frame size: %s \n" % self.audio_frame_data.size())
        
    def publish_audio_messages(self):

        # Build audio dataframe from queue
        for (chunk_a) in self.streamer.stream(timeout=-1, backoff=1.0):
            # self.get_logger().info("Chunk size: %s \n" % str(chunk_a[0].size()))
            
            
            
            # self.get_logger().info("audio chunk data: %s \n" % str(chunk_a))            
            self.audio_frame_data = self.cacher(chunk_a[0],self).view(-1)
            torch.save(self.audio_frame_data,'frame_data_original.pt')
            self.audio_frame_bytes = self.audio_frame_data.numpy().tobytes()
            # self.get_logger().info("audio frame data: %s \n" % str(self.audio_frame_data))

            # self.get_logger().info("Frame data shape: %s \n" % str(self.audio_frame_data.size()))
            # self.get_logger().info("Frame bytes length: %s \n" % str(len(self.audio_frame_bytes)))

        
            # Publish AudioDataStamped message (fill with actual audio data as needed)
            audio_data_msg = AudioDataStamped()
            audio_data_msg.audio.data = self.audio_frame_bytes # Placeholder data
            self.audio_data_publisher.publish(audio_data_msg)
            
            # Publish AudioInfo message
            self.audio_info_publisher.publish(self.audio_info_msg)
            
            # self.get_logger().info(f'Published audio info with {self.n_channels} channels, sample rate {self.sample_rate}, src {self.src}')


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