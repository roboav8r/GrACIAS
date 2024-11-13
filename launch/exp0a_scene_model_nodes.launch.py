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

    clip_params = os.path.join(
        get_package_share_directory('mm_scene_rec'),
        'config',
        'philbart_clip_params.yaml'
    )
    audio_scene_rec_params = os.path.join(
        get_package_share_directory('mm_scene_rec'),
        'config',
        'philbart_audio_scene_rec_params.yaml'
    )

    # Scene recognition nodes
    clip_rec_node = Node(package = "mm_scene_rec", 
                    executable = "clip_scene_rec_server.py",
                    name = "clip_scene_rec_server",
                    remappings=[('/clip_scene_image','/oak/rgb/image_raw')],
                    parameters=[clip_params]
    )
    ld.add_action(clip_rec_node)
    
    audio_rec_node = Node(
        package='mm_scene_rec',
        executable='audio_scene_rec_server.py',
        name='audio_scene_rec',
        output='screen',
        parameters=[audio_scene_rec_params]
    )
    ld.add_action(audio_rec_node)

    # Foxglove bridge for visualization
    viz_node = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('marmot'),
                'launch/foxglove_bridge_launch.xml'))
    )
    ld.add_action(viz_node)

    return ld