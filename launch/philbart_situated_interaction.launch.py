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
    tracker_config = os.path.join(
        get_package_share_directory('situated_interaction'),
        'config',
        'philbart_tracker.yaml'
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

    # LiDAR to laserscan node
    lidar_to_scan = Node(
            package='pointcloud_to_laserscan', executable='pointcloud_to_laserscan_node',
            remappings=[('cloud_in', 'philbart/lidar_points'),
                        ('scan', 'philbart/scan')],
            parameters=[{
                'target_frame': 'cloud',
                'transform_tolerance': 0.01,
                'min_height': 0.0,
                'max_height': 1.0,
                'angle_min': -1.5708,  # -M_PI/2
                'angle_max': 1.5708,  # M_PI/2
                'angle_increment': 0.0087,  # M_PI/360.0
                'scan_time': 0.3333,
                'range_min': 0.45,
                'range_max': 4.0,
                'use_inf': True,
                'inf_epsilon': 1.0
            }],
            name='pointcloud_to_laserscan'
        )
    ld.add_action(lidar_to_scan)


    ### VISION PROCESSING
    # Scene recognition
    clip_rec_node = Node(package = "mm_scene_rec", 
                    executable = "clip_scene_rec.py",
                    name = "clip_scene_rec",
                    remappings=[('/clip_scene_image','/oak/rgb/image_raw')],
                    parameters=[config]
    )
    ld.add_action(clip_rec_node)

    # Object recognition
    clip_obj_rec_server = Node(package = "situated_interaction", 
                    executable = "clip_vis_rec_server.py",
                    name = "clip_vis_rec_server",
                    # remappings=[('/clip_scene_image','/oak/rgb/image_raw')],
                    parameters=[config]
    )
    ld.add_action(clip_obj_rec_server)

    # Detection preprocessing for tracker
    preproc_node = Node(
        package='marmot',
        executable='depthai_preproc',
        name='depthai_preproc_node',
        remappings=[('/depthai_detections','/oak/nn/spatial_detections')],
        output='screen',
        parameters=[config])    
    ld.add_action(preproc_node)
    

    ### AUDITION PROCESSING
    # Scene recognition
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

    # Multiobject tracking
    trk_node = Node(
        package='marmot',
        executable='tbd_node.py',
        name='tbd_tracker_node',
        output='screen',
        remappings=[('/detections','/converted_detections')],
        parameters=[tracker_config]
    )
    ld.add_action(trk_node)

    # Semantic fusion
    semantic_tracking_node = Node(
        package='situated_interaction',
        executable='semantic_tracking_node.py',
        name='semantic_tracking_node',
        output='screen',
        parameters=[config]
    )
    ld.add_action(semantic_tracking_node)


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