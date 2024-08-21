#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
from vision_msgs.msg import Detection3DArray
from tf2_msgs.msg import TFMessage

### Dummy subscriber node for VICON motion capture experiments.
### Since many topics are bridged from ROS1,
### AND since ros1_bridge doesn't bridge unless there is a subscriber,
### this script creates dummy subscribers to experiment topics so that
### the topics are bridged and an experiment .mcap can be recorded.

class MultiSubscriberNode(Node):
    def __init__(self):
        super().__init__('multi_subscriber_node')
        self.subscription_lidar_points = self.create_subscription(
            PointCloud2,
            '/philbart/lidar_points',
            self.common_callback,
            10)
        self.get_logger().info('Subscribers have been started.')

    def common_callback(self, msg):
        self.get_logger().debug('callback')

def main(args=None):
    rclpy.init(args=args)
    multi_subscriber_node = MultiSubscriberNode()
    try:
        rclpy.spin(multi_subscriber_node)
    except KeyboardInterrupt:
        pass
    finally:
        multi_subscriber_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
