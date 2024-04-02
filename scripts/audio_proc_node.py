#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from audio_common_msgs.msg import AudioDataStamped
from sensor_msgs.msg import Image
import cv2
import numpy as np

import torch
import torchaudio
import librosa


class AudioProcNode(Node):

    def __init__(self):
        super().__init__('audio_proc_node')
        self.subscription = self.create_subscription(
            AudioDataStamped,
            'audio_data',
            self.audio_data_callback,
            10)
        self.publisher = self.create_publisher(
            Image,
            'image_data_topic',
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

    def audio_data_callback(self, msg):

        self.frame = torch.from_numpy(librosa.util.buf_to_float(np.frombuffer(msg.audio.data))).type(torch.float16) # .view(self.n_channels,-1)
        torch.save(self.frame,'frame_data_recovered.pt')

        # Process audio data and convert it to image
        # Here, we are just creating a dummy image for demonstration
        dummy_image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)

        # Create a sensor_msgs/Image message
        image_msg = Image()
        image_msg.header = msg.header
        image_msg.height = dummy_image.shape[0]
        image_msg.width = dummy_image.shape[1]
        image_msg.encoding = 'bgr8'  # Assuming OpenCV BGR encoding
        image_msg.step = dummy_image.shape[1] * 3  # bytes per row
        image_msg.data = dummy_image.tobytes()

        # Publish the image message
        self.publisher.publish(image_msg)


def main(args=None):
    rclpy.init(args=args)
    audio_proc_node = AudioProcNode()
    rclpy.spin(audio_proc_node)
    audio_proc_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
