#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node

import gtsam

from tracking_msgs.msg import Tracks3D
from gracias_interfaces.msg import Auth, Comm, Comms, Identity

from foxglove_msgs.msg import SceneUpdate, SceneEntity, TextPrimitive

class Person():
    def __init__(self, msg):
        self.track_id = msg.track_id
        self.identity = "Unknown"
        self.auth = False
        self.comms = []
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        self.pos_z = msg.pose.pose.position.z

    def update(self, ar_msg, type):
        if type=='authentication':
            if ar_msg.authenticated:
                self.auth = True

        if type=='communication':
            # Increase weight for existing comms in incoming comms
            pre_comms = self.comms # list of [gracias_interfaces/Comm]
            new_comms = ar_msg # list of [gracias_interfaces/Comm]
            # Reduce weight for existing comms not in incoming comms

        if type=='identity':
            self.identity = ar_msg.identity


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
            'authentication',
            self.listener_callback_auth,
            10)
        self.subscription_auth  # prevent unused variable warning

        self.subscription_id = self.create_subscription(
            Identity,
            'identification',
            self.listener_callback_identity,
            10)
        self.subscription_id  # prevent unused variable warning

        self.subscription_comms = self.create_subscription(
            Comms,
            'communications',
            self.listener_callback_comms,
            10)
        self.subscription_comms  # prevent unused variable warning

        # Define member variables
        self.persons = {}
        self.tracks_msg = Tracks3D()
        self.scene_msg = SceneUpdate()

        # Define pubs
        self.publisher_people_scene = self.create_publisher(
            SceneUpdate,
            'people_scene',
            10)

    def compute_match(self, pos):
        similarity_vector = np.zeros(len(self.persons.keys()))

        for ii,key in enumerate(self.persons.keys()): 
            similarity_vector[ii] = np.linalg.norm([pos.x - self.persons[key].pos_x, pos.y - self.persons[key].pos_y, pos.z - self.persons[key].pos_z])

        return list(self.persons.keys())[np.argmin(similarity_vector)]

    def listener_callback_tracked_persons(self, msg):
        self.tracks_msg = msg

        self.tracked_person_ids = []

        # add new tracked persons from incoming message
        for trkd_person in msg.tracks:
            self.tracked_person_ids.append(trkd_person.track_id)

            # Create person and add to dict if not currently tracked
            if trkd_person.track_id not in self.persons.keys():
                self.persons[trkd_person.track_id] = Person(trkd_person)

            self.persons[trkd_person.track_id].pos_x = trkd_person.pose.pose.position.x
            self.persons[trkd_person.track_id].pos_y = trkd_person.pose.pose.position.y
            self.persons[trkd_person.track_id].pos_z = trkd_person.pose.pose.position.z

        # remove untracked persons
        persons_temp = self.persons
        self.persons = {key: val for key, val in persons_temp.items() if key in self.tracked_person_ids}

        self.visualize()


    def listener_callback_auth(self, msg):
        # self.get_logger().info('Received auth message: "%s"' % msg)

        # Compute match, perform update
        match_key = self.compute_match(msg.pose.pose.position)
        self.persons[match_key].update(msg, 'authentication')

    def listener_callback_comms(self, msg):
        # self.get_logger().info('Received communication message: "%s"' % msg)

        comm_matches = {}

        # Compute set of comms that are matched to a person
        for comm in msg.comms:
            match_key = self.compute_match(comm.pose.position)
            
            if match_key not in comm_matches.keys():
                comm_matches[match_key] = []
            
            comm_matches[match_key].append(comm)

        # Update
        self.get_logger().info('person-comm matches: "%s"' % comm_matches)
        for key in comm_matches.keys():
            self.persons[key].update(comm_matches[key], 'communication')

    def listener_callback_identity(self, msg):
        # self.get_logger().info('Received identity message: "%s"' % msg)

        # Compute match, perform update
        match_key = self.compute_match(msg.pose.pose.position)
        self.persons[match_key].update(msg, 'identity')


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
            entity_msg.lifetime.nanosec = 500000000

            text = TextPrimitive()
            text.billboard = True
            text.font_size = 16.
            text.scale_invariant = True
            text.color.a = 1.0
            text.pose.position.x = person.pos_x
            text.pose.position.y = person.pos_y
            text.pose.position.z = person.pos_z
            text.text = "person #%s: \nID: %s \nAuth: %s" % (person.track_id, person.identity, person.auth)
            for msg in person.comms:
                text.text += "\n%s" % (msg)
            text.text
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
