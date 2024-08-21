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


    ### CONFIG FILES
    config = os.path.join(
        get_package_share_directory('situated_interaction'),
        'config',
        'philbart_config.yaml'
    )


    ### TF DATA
    tf_node = Node(package = "tf2_ros", 
                    executable = "static_transform_publisher",
                    arguments = [".15", "0", ".55", "0", "-0.261799", "0", "philbart/base_link", "oak-d-base-frame"]
    )
    ld.add_action(tf_node)


    ### SENSORS
    # Vision
    cam_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('depthai_ros_driver'),
                'launch',
                'rgbd_pcl.launch.py'
            ])
        ]),
        launch_arguments={'params_file': config }.items()
    )
    ld.add_action(cam_node)

    # Audio acquisition node
    acq_node = Node(
        package='ros_audition',
        executable='audio_acq_node.py',
        name='audio_acq_node',
        output='screen',
        parameters=[config]
    )
    ld.add_action(acq_node)


    ### UNIMODAL PROCESSING
    # Scene recognition - vision
    clip_rec_node = Node(package = "mm_scene_rec", 
                    executable = "clip_scene_rec.py",
                    name = "clip_scene_rec",
                    remappings=[('/clip_scene_image','/oak/rgb/image_raw')],
                    parameters=[config]
    )
    ld.add_action(clip_rec_node)
    
    # Scene recognition - audition
    audio_rec_node = Node(
        package='mm_scene_rec',
        executable='audio_scene_rec.py',
        name='audio_scene_rec',
        output='screen',
        parameters=[config]
    )
    ld.add_action(audio_rec_node)


    ### MULTIMODAL PROCESSING
    # Scene recognition
    scene_rec_node = Node(package = "mm_scene_rec", 
                    executable = "bayes_scene_est.py",
                    name = "bayes_fused_scene_est",
                    parameters=[config]
    )
    ld.add_action(scene_rec_node)


    ### OUTPUT / VISUALIZATION
    # Foxglove bridge for visualization
    viz_node = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('marmot'),
                'launch/foxglove_bridge_launch.xml'))
    )
    ld.add_action(viz_node)


    return ld