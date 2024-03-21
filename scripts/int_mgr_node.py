#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tracking_msgs.msg import Tracks3D
from gracias_interfaces.msg import Auth
from gracias_interfaces.msg import Comm 
from gracias_interfaces.msg import Identity 

class Person():
    def __init__(self, msg):
        self.track_id = msg.track_id
        self.identity = "unknown"
        self.auth = "not authenticated"
        self.comms = []

class InteractionManagerNode(Node):
    def __init__(self):
        super().__init__('interaction_manager_node')
        
        self.subscription_tracked_persons = self.create_subscription(
            Tracks3D,
            'tracked_persons',
            self.listener_callback_tracked_persons,
            10)
        self.subscription_tracked_persons  # prevent unused variable warning

        self.subscription_auth = self.create_subscription(
            Auth,
            'auth',
            self.listener_callback_auth,
            10)
        self.subscription_auth  # prevent unused variable warning

        self.subscription_id = self.create_subscription(
            Identity,
            'identity',
            self.listener_callback_identity,
            10)
        self.subscription_id  # prevent unused variable warning

        self.subscription_comm = self.create_subscription(
            Comm,
            'communication',
            self.listener_callback_comm,
            10)
        self.subscription_comm  # prevent unused variable warning

        # Define member variables
        persons = []

    def listener_callback_tracked_persons(self, msg):
        self.get_logger().info('Received tracked persons: "%s"' % msg)

        tracked_person_ids = []

        # add new tracked persons
        for person in msg.tracks:
            tracked_person_ids.append(person.track_id)


        # remove untracked persons

    def listener_callback_auth(self, msg):
        self.get_logger().info('Received auth message: "%s"' % msg)

    def listener_callback_comm(self, msg):
        self.get_logger().info('Received communication message: "%s"' % msg)

    def listener_callback_identity(self, msg):
        self.get_logger().info('Received identity message: "%s"' % msg)

def main(args=None):
    rclpy.init(args=args)

    interaction_manager_node = InteractionManagerNode()

    try:
        rclpy.spin(interaction_manager_node)
    except KeyboardInterrupt:
        pass
    finally:
        interaction_manager_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
