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

    audio_bayes_est_params = os.path.join(
        get_package_share_directory('mm_scene_rec'),
        'config',
        'audio_bayes_estimator_params.yaml'
    )

    clip_bayes_est_params = os.path.join(
        get_package_share_directory('mm_scene_rec'),
        'config',
        'clip_bayes_estimator_params.yaml'
    )

    fused_bayes_est_params = os.path.join(
        get_package_share_directory('mm_scene_rec'),
        'config',
        'fused_bayes_estimator_params.yaml'
    )

    # Scene recognition nodes
    clip_rec_node = Node(package = "mm_scene_rec", 
                    executable = "clip_scene_rec.py",
                    name = "clip_scene_rec",
                    remappings=[('/clip_scene_image','/oak/rgb/image_raw')],
                    parameters=[clip_params]
    )
    ld.add_action(clip_rec_node)
    
    audio_rec_node = Node(
        package='mm_scene_rec',
        executable='audio_scene_rec.py',
        name='audio_scene_rec',
        output='screen',
        parameters=[audio_scene_rec_params]
    )
    ld.add_action(audio_rec_node)

    audio_bayes_rec_node = Node(package = "mm_scene_rec", 
                    executable = "bayes_scene_est.py",
                    name = "bayes_audio_scene_est",
                    output = 'screen',
                    parameters=[audio_bayes_est_params]
    )
    ld.add_action(audio_bayes_rec_node)

    clip_bayes_rec_node = Node(package = "mm_scene_rec", 
                    executable = "bayes_scene_est.py",
                    name = "bayes_clip_scene_est",
                    output = 'screen',
                    parameters=[clip_bayes_est_params]
    )
    ld.add_action(clip_bayes_rec_node)

    fused_bayes_rec_node = Node(package = "mm_scene_rec", 
                    executable = "bayes_scene_est.py",
                    name = "bayes_fused_scene_est",
                    output = 'screen',
                    parameters=[fused_bayes_est_params]
    )
    ld.add_action(fused_bayes_rec_node)

    scene_rec_node = Node(package = "situated_interaction", 
                    executable = "record_scene_results_node.py",
                    name = "record_scene_results_node",
                    output = 'screen'
    )
    ld.add_action(scene_rec_node)

    # Foxglove bridge for visualization
    viz_node = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('marmot'),
                'launch/foxglove_bridge_launch.xml'))
    )
    ld.add_action(viz_node)

    return ld