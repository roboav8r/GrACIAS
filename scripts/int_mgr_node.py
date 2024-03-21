#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tracking_msgs.msg import Tracks3D
from gracias_interfaces.msg import Auth
from gracias_interfaces.msg import Comm 
from gracias_interfaces.msg import Identity

from foxglove_msgs.msg import SceneUpdate, SceneEntity

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
        self.persons = {}
        self.scene_msg = SceneUpdate()

    def listener_callback_tracked_persons(self, msg):
        self.get_logger().info('Received tracked persons: "%s"' % msg)

        self.tracked_person_ids = []

        # add new tracked persons from incoming message
        for trkd_person in msg.tracks:
            self.tracked_person_ids.append(trkd_person.track_id)

            # Manage dict of tracked people; key is track_id
            if trkd_person.track_id not in self.persons.keys():
                self.persons[trkd_person.track_id] = Person(trkd_person)

        # remove untracked persons
        persons_temp = self.persons
        self.persons = {key: val for key, val in persons_temp.items() if key in self.tracked_person_ids}

        self.get_logger().info('Managed persons: "%s"' % self.persons)

        # TODO - visualize people


    def listener_callback_auth(self, msg):
        self.get_logger().info('Received auth message: "%s"' % msg)

    def listener_callback_comm(self, msg):
        self.get_logger().info('Received communication message: "%s"' % msg)

    def listener_callback_identity(self, msg):
        self.get_logger().info('Received identity message: "%s"' % msg)

    # def visualize(self):



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
