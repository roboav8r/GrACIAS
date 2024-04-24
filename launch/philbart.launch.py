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
    config = os.path.join(
        get_package_share_directory('situated_interaction'),
        'config',
        'philbart_config.yaml'
    )

    tf_node = Node(package = "tf2_ros", 
                    executable = "static_transform_publisher",
                    arguments = [".15", "0", ".55", "0", "-0.261799", "0", "philbart/base_link", "oak-d-base-frame"]
    )
    ld.add_action(tf_node)

    # MaRMOT / multiple object tracking
    oakd_preproc_node = Node(
        package='marmot',
        executable='depthai_preproc',
        name='depthai_preproc_node',
        remappings=[('/depthai_detections','/oak/nn/spatial_detections'), ('/converted_detections','/converted_detections_oakd')],
        parameters=[config]
    )
    ld.add_action(oakd_preproc_node)

    # headset_1_node = Node(
    #     package='marmot',
    #     executable='pose_preproc',
    #     name='headset_1_preproc_node',
    #     remappings=[('/pose_detections','/vrpn_client_node/headset_1/pose'), ('/converted_detections','/converted_detections_headset_1')],
    #     parameters=[exp_config]
    # )
    # ld.add_action(headset_1_node)

    trk_node = Node(
        package='marmot',
        executable='tbd_node.py',
        name='tbd_tracker_node',
        output='screen',
        remappings=[('/detections','/converted_detections')],
        parameters=[config]
    )
    ld.add_action(trk_node)


    # Scene recognition node
    # rec_node = Node(
    #     package='situated_interaction',
    #     executable='scene_rec_node.py',
    #     name='scene_rec_node',
    #     output='screen',
    #     # remappings=[('/detections','/converted_detections')],
    #     parameters=[config]
    # )
    # ld.add_action(rec_node)

    # # Voice processing node
    # voice_node = Node(
    #     package='situated_interaction',
    #     executable='voice_proc_node.py',
    #     name='voice_proc_node',
    #     output='screen',
    #     # remappings=[('/detections','/converted_detections')],
    #     parameters=[audio_config]
    # )
    # ld.add_action(voice_node)

    # Foxglove bridge for visualization
    viz_node = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('marmot'),
                'launch/foxglove_bridge_launch.xml'))
    )
    ld.add_action(viz_node)


    return ld