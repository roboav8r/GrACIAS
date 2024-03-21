#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tracking_msgs.msg import Tracks3D
from gracias_interfaces.msg import Auth
from gracias_interfaces.msg import Comm 
from gracias_interfaces.msg import Identity

from foxglove_msgs.msg import SceneUpdate, SceneEntity, TextPrimitive

class Person():
    def __init__(self, msg):
        self.track_id = msg.track_id
        self.identity = "Unknown identity"
        self.auth = "Not authenticated"
        self.comms = []
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        self.pos_z = msg.pose.pose.position.z

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
        self.tracks_msg = Tracks3D()
        self.scene_msg = SceneUpdate()

        # Define pubs
        self.publisher_people_scene = self.create_publisher(
            SceneUpdate,
            'people_scene',
            10)


    def listener_callback_tracked_persons(self, msg):
        self.tracks_msg = msg

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

        self.visualize()


    def listener_callback_auth(self, msg):
        self.get_logger().info('Received auth message: "%s"' % msg)

    def listener_callback_comm(self, msg):
        self.get_logger().info('Received communication message: "%s"' % msg)

    def listener_callback_identity(self, msg):
        self.get_logger().info('Received identity message: "%s"' % msg)

    def visualize(self):
        self.scene_msg = SceneUpdate()

        for idx in self.persons.keys():
            person = self.persons[idx]
            entity_msg = SceneEntity()

            # Populate entity message with header / person data
            entity_msg.frame_id = self.tracks_msg.header.frame_id
            entity_msg.timestamp = self.tracks_msg.header.stamp
            entity_msg.id = str(person.track_id)
            entity_msg.frame_locked = True
            entity_msg.lifetime.nanosec = 3000000

            text = TextPrimitive()
            text.billboard = True
            text.font_size = 24.
            text.scale_invariant = True
            text.color.a = 1.0
            text.pose.position.x = person.pos_x
            text.pose.position.y = person.pos_y
            text.pose.position.z = person.pos_z
            text.text = "person #%s: \n%s \n%s" % (person.track_id, person.identity, person.auth)
            for msg in person.comms:
                text.text += "\n%s" % (msg)
            entity_msg.texts.append(text)

            self.scene_msg.entities.append(entity_msg)

        self.publisher_people_scene.publish(self.scene_msg)


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
