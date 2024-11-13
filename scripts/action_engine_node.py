#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import Twist, PoseStamped
from situated_hri_interfaces.msg import HierarchicalCommands
from geometry_msgs.msg import Pose, PoseStamped


class CommandProcessor(Node):
    def __init__(self):
        super().__init__('command_processor')

        # Separate callback groups for subscription and timer
        self.subscription_callback_group = MutuallyExclusiveCallbackGroup()
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()

        # Subscription to accept valid commands
        self.subscription = self.create_subscription(
            HierarchicalCommands,
            '/hierarchical_commands',
            self.command_callback,
            10,
            callback_group=self.subscription_callback_group
        )

        # Timer for periodic check
        self.timer = self.create_timer(
            0.1,  # 0.1 second period
            self.timer_callback,
            callback_group=self.timer_callback_group
        )

        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.follow_pose_publisher = self.create_publisher(PoseStamped, '/follow_pose', 10)
        self.waypoint_pose_publisher = self.create_publisher(PoseStamped, '/waypoint_pose', 10)

        # Behavior parameters
        self.drive_speed = 0.1 # meters/sec
        self.follow_x_offset = -1. # meters

        # Initialize state
        self.current_command = "halt"
        self.follow_id = None
        self.target_pose = Pose()
        self.agent_poses = {}

    # TODO
    # def build_route_base_frame(self):

    # TODO
    # def build_route_map_frame(self):

    def command_callback(self, msg):
        self.last_command_header = msg.header
        self.agent_poses = {}

        for command in msg.commands:  # Assuming msg.commands is a list of HierarchicalCommand

            # Save locations of agents
            self.agent_poses[command.object_id] = command.pose

            if command.comms == 'halt':
                self.current_command = "halt"
                # self.publish_halt()

            elif command.comms == 'move-forward' and command.states[0].value == 'supervisor':
                self.current_command = "move-forward"
                # self.publish_move_forward()

            elif command.comms == 'move-in-reverse' and command.states[0].value == 'supervisor':
                self.current_command = "move-in-reverse"
                # self.publish_move_backward()

            elif command.comms == 'follow-me' and command.states[0].value == 'supervisor':
                self.current_command = "follow"
                self.follow_id = command.object_id

            # elif command.comms == 'advance' and command.attribute == 'supervisor':
            #     waypoint_pose = self.get_next_waypoint(command.pose)
            #     self.publish_waypoint_pose(waypoint_pose)

    def timer_callback(self):
        # Check current command state, and publish appropriately
        if self.current_command == 'halt':
            self.publish_halt()

        elif self.current_command == 'move-forward':
            self.publish_move_forward()

        elif self.current_command == 'move-in-reverse':
            self.publish_move_backward()

        elif self.current_command == 'follow':

            # Target position known = compute follow pose & send 
            if self.follow_id in self.agent_poses.keys():
                self.compute_follow_target_pose(self.follow_id)
                self.publish_follow_pose(self.target_pose)
            
            # Don't know where target is = stop
            else:
                self.current_command = "halt"
                self.publish_halt()

    def publish_halt(self):
        twist = Twist()
        self.cmd_vel_publisher.publish(twist)
        self.get_logger().info('Published halt command')

    def publish_move_forward(self):
        twist = Twist()
        twist.linear.x = self.drive_speed
        self.cmd_vel_publisher.publish(twist)
        self.get_logger().info('Published move-forward command')

    def publish_move_backward(self):
        twist = Twist()
        twist.linear.x = -self.drive_speed
        self.cmd_vel_publisher.publish(twist)
        self.get_logger().info('Published move-backward command')

    def publish_follow_pose(self, pose):
        pose_stamped = PoseStamped()
        pose_stamped.pose = pose
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        self.follow_pose_publisher.publish(pose_stamped)
        self.get_logger().info('Published follow-me command')

    def publish_waypoint_pose(self, pose):
        pose_stamped = PoseStamped()
        pose_stamped.pose = pose
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        self.waypoint_pose_publisher.publish(pose_stamped)
        self.get_logger().info('Published advance command')

    def compute_follow_target_pose(self, target_id):

        agent_pose = self.agent_poses[target_id]
        agent_pose_with_offset = agent_pose
        agent_pose_with_offset.position.x -= self.follow_x_offset
        self.target_pose = agent_pose_with_offset

    def get_next_waypoint(self, current_pose):
        # Define logic for calculating next waypoint pose
        # Placeholder: returns the provided current_pose as next waypoint pose
        return current_pose


def main(args=None):
    rclpy.init(args=args)
    node = CommandProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
