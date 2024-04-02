import os

from ament_index_python import get_package_share_directory

from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution
from launch import LaunchDescription
from launch_ros.actions import Node, LoadComposableNodes, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    ld = LaunchDescription()

    # Config files
    # audio_config = os.path.join(
    #     get_package_share_directory('GrACIAS'),
    #     'config',
    #     'audio_config.yaml'
    # )

    # Audio acquisition node
    acq_node = Node(
        package='GrACIAS',
        executable='audio_acq_node.py',
        name='respeaker_acq_node',
        output='screen',
        # remappings=[('/detections','/converted_detections')],
        # parameters=[tracker_config]
    )
    ld.add_action(acq_node)

    # Audio processing node
    proc_node = Node(
        package='GrACIAS',
        executable='audio_proc_node.py',
        name='audio_proc_node',
        output='screen',
        # remappings=[('/detections','/converted_detections')],
        # parameters=[tracker_config]
    )
    ld.add_action(proc_node)


    return ld