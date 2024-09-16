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
    oakd_params = os.path.join(
        get_package_share_directory('situated_interaction'),
        'config',
        'oakd_params_exp1.yaml'
    )
    oakd_tracker_params = os.path.join(
        get_package_share_directory('situated_interaction'),
        'config',
        'oakd_img_tracker_exp1.yaml'
    )
    semantic_fusion_params_exp1 = os.path.join(
        get_package_share_directory('situated_interaction'),
        'config',
        'semantic_fusion_params_exp1.yaml'
    )

    # Static TF nodes
    tf_node = Node(package = "tf2_ros", 
                    executable = "static_transform_publisher",
                    arguments = [".15", "0", ".55", "0", "-0.261799", "0", "philbart/base_link", "oak-d-base-frame"]
    )
    ld.add_action(tf_node)

    # Detector preprocessing nodes
    oakd_preproc_node = Node(
        package='marmot',
        executable='depthai_img_preproc',
        name='depthai_img_preproc_node',
        remappings=[('/converted_detections','/converted_img_detections'),('/depthai_detections','/oak/nn/spatial_detections'),('/depthai_img','/oak/rgb/image_raw')],
        parameters=[oakd_params])    
    ld.add_action(oakd_preproc_node)
    
    tracks_rec_node = Node(package = "situated_interaction", 
                    executable = "record_tracker_results_node.py",
                    name = "record_tracker_results_node",
                    output = 'screen'
    )
    ld.add_action(tracks_rec_node)

    # Tracker node
    oakd_trk_node = Node(
        package='marmot',
        executable='tbd_node.py',
        name='oakd_tracker_node',
        output='screen',
        remappings=[('/detections','/converted_img_detections')],
        parameters=[oakd_tracker_params]
    )
    ld.add_action(oakd_trk_node)

    # Semantic fusion node
    semantic_fusion_node = Node(
        package='situated_interaction',
        executable='semantic_tracking_node.py',
        name='semantic_fusion_node',
        output='screen',
        parameters=[semantic_fusion_params_exp1]
    )
    ld.add_action(semantic_fusion_node)

    # Foxglove bridge for visualization
    viz_node = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('marmot'),
                'launch/foxglove_bridge_launch.xml'))
    )
    ld.add_action(viz_node)

    return ld