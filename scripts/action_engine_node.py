#!/usr/bin/env python3

# import copy

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_msgs.msg import Empty, Bool
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from situated_hri_interfaces.msg import HierarchicalCommands
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from nav_msgs.msg import Path

from tf_transformations import quaternion_from_euler
from tf2_ros import Buffer, TransformListener, TransformException

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
        self.cancel_follow_publisher = self.create_publisher(Empty, '/cancel_follow', 10)
        self.transform_broadcaster = TransformBroadcaster(self)

        # Waypoint navigation members
        self.declare_parameter('path_x_positions', [1.5, 2.0., 3.5, 4.0, 5.5])
        self.declare_parameter('path_source_frame', 'philbart/base_link')
        self.declare_parameter('path_target_frame', 'philbart/map')
        self.path_x_positions = self.get_parameter('path_x_positions').get_parameter_value().double_array_value
        self.path_source_frame = self.get_parameter('path_source_frame').get_parameter_value().string_value
        self.path_target_frame = self.get_parameter('path_target_frame').get_parameter_value().string_value
        
        self.waypoint_path_publisher = self.create_publisher(Path, '/philbart/waypoint_manager/waypoint_plan', 10)
        self.cancel_waypoint_nav_publisher = self.create_publisher(Empty, '/philbart/waypoint_manager/cancel', 10)
        self.pause_waypoint_nav_publisher = self.create_publisher(Bool, '/philbart/waypoint_manager/todo_pause', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Behavior parameters
        self.drive_speed = 0.1 # meters/sec
        self.follow_x_offset = -1. # meters

        # Initialize state
        self.current_command = "halt"
        self.follow_id = None
        self.target_pose = Pose()
        self.target_pose.position.x += self.follow_x_offset
        self.follow_transform = TransformStamped()
        self.target_frame_name = "follow_target_frame"
        self.agent_poses = {}
        self.paused_msg = Bool()
        self.paused_msg.data = True

        self.generate_path()
        self.pause_waypoint_nav_publisher.publish(self.paused_msg)
        self.waypoint_path_publisher.publish(self.nav_path)

    
    def generate_path(self):
        path = Path()
        path.header.frame_id = self.path_target_frame
        path.header.stamp = self.get_clock().now().to_msg()

        for x in self.path_x_positions:
            try:
                # Create PoseStamped in source frame
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = self.path_source_frame
                pose_stamped.header.stamp = self.get_clock().now().to_msg()
                pose_stamped.pose.position.x = x
                pose_stamped.pose.position.y = 0.0
                pose_stamped.pose.position.z = 0.0
                pose_stamped.pose.orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

                # Transform PoseStamped to target frame
                transformed_pose = self.tf_buffer.transform(pose_stamped, self.target_frame)
                path.poses.append(transformed_pose)

            except TransformException as ex:
                self.get_logger().error(f"Transform error: {ex}")
                return

        self.nav_path = path

    def command_callback(self, msg):
        self.last_command_header = msg.header
        self.agent_poses = {}

        for command in msg.commands:  # Assuming msg.commands is a list of HierarchicalCommand

            # Save locations of agents
            self.agent_poses[command.object_id] = command.pose # geometry_msgs/Pose

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
                if self.current_command != "follow":
                    self.current_command = "follow"

                self.follow_id = command.object_id
                self.compute_follow_target(self.follow_id)

                # Update target pose with current stamp and publish
                self.publish_follow_pose(self.target_pose, self.target_frame_name)

            elif command.comms == 'advance' and command.attribute == 'supervisor':
                self.current_command = "advance"

    def timer_callback(self):
        # Check current command state, and publish appropriately
        if self.current_command == 'halt':
            self.publish_halt()

            # Pause navigation
            self.paused_msg.data = True
            self.pause_waypoint_nav_publisher.publish(self.paused_msg)

        elif self.current_command == 'move-forward':
            self.publish_move_forward()

        elif self.current_command == 'move-in-reverse':
            self.publish_move_backward()

        elif self.current_command == 'follow':

            # Target position known = compute follow pose & send 
            if self.follow_id in self.agent_poses.keys():
                self.compute_follow_target(self.follow_id)
                # self.publish_follow_pose(self.target_pose)
                self.publish_target_tf()

            # Don't know where target is = stop
            else:
                self.current_command = "halt"
                self.publish_halt()
                
        elif self.current_command == "advance":
            self.paused_msg.data = False
            self.pause_waypoint_nav_publisher.publish(self.paused_msg)

    def publish_halt(self):
        twist_msg = Twist()
        self.cmd_vel_publisher.publish(twist_msg)

        empty_msg = Empty()
        self.cancel_follow_publisher.publish(empty_msg)

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

    def publish_follow_pose(self, pose, frame_id):
        pose_stamped = PoseStamped()
        pose_stamped.pose = pose
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = frame_id
        self.follow_pose_publisher.publish(pose_stamped)
        self.get_logger().info('Published follow-me command')

    def publish_target_tf(self):
        self.transform_broadcaster.sendTransform(self.follow_transform)

    def publish_waypoint_pose(self, pose):
        pose_stamped = PoseStamped()
        pose_stamped.pose = pose
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        self.waypoint_pose_publisher.publish(pose_stamped)
        self.get_logger().info('Published advance command')

    def compute_follow_target(self, target_id):

        agent_pose = self.agent_poses[target_id]

        # New way = pose pointed directly at agent, with offset
        # range_to_agent = np.linalg.norm([agent_pose.position.x, agent_pose.position.y])
        # range_to_target = range_to_agent + self.follow_x_offset
        yaw_angle_to_agent = np.arctan2(agent_pose.position.y, agent_pose.position.x)
        # self.target_pose = Pose()
        # self.target_pose.position.x = agent_pose.position.x*(range_to_target)/range_to_agent
        # self.target_pose.position.y = agent_pose.position.y*(range_to_target)/range_to_agent

        quat = quaternion_from_euler(0.,0.,yaw_angle_to_agent)
        # self.target_pose.orientation.w = quat[3]
        # self.target_pose.orientation.x = quat[0]
        # self.target_pose.orientation.y = quat[1]
        # self.target_pose.orientation.z = quat[2]

        # Publish transform instead
        self.follow_transform = TransformStamped()
        self.follow_transform.header.stamp = self.get_clock().now().to_msg()
        self.follow_transform.header.frame_id = self.last_command_header.frame_id
        self.follow_transform.child_frame_id = self.target_frame_name

        self.follow_transform.transform.translation.x = agent_pose.position.x
        self.follow_transform.transform.translation.y = agent_pose.position.y
        self.follow_transform.transform.translation.z = 0.

        self.follow_transform.transform.rotation.x = quat[0]
        self.follow_transform.transform.rotation.y = quat[1]
        self.follow_transform.transform.rotation.z = quat[2]
        self.follow_transform.transform.rotation.w = quat[3]
        
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
