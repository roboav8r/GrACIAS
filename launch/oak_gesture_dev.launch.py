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
    params = os.path.join(
        get_package_share_directory('situated_interaction'),
        'config',
        'oakd_gesture_demo_params.yaml'
    )

    tracker_params = os.path.join(
        get_package_share_directory('situated_interaction'),
        'config',
        'oakd_gesture_tracker_params.yaml'
    )

    # Static TF nodes
    map_oakd_tf_node = Node(package = "tf2_ros", 
                    executable = "static_transform_publisher",
                    arguments = ["0", "0", "1.0", "0", "0", "0", "map", "oak-d-base-frame"]
    )
    ld.add_action(map_oakd_tf_node)

    # Sensor nodes
    cam_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('depthai_ros_driver'),
                'launch',
                'rgbd_pcl.launch.py'
            ])
        ]),
        launch_arguments={'params_file': params }.items()
    )
    ld.add_action(cam_node)


    # Detector preprocessing nodes
    preproc_node = Node(
        package='marmot',
        executable='depthai_img_preproc',
        name='depthai_img_preproc_node',
        remappings=[('/converted_detections','/converted_visual_detections'),('/depthai_detections','/oak/nn/spatial_detections'),('/depthai_img','/oak/rgb/image_raw')],
        output='screen',
        parameters=[params])    
    ld.add_action(preproc_node)


    # Tracker node
    trk_node = Node(
        package='marmot',
        executable='tbd_node.py',
        name='tbd_tracker_node',
        output='screen',
        # remappings=[('/detections','/converted_visual_detections')],
        parameters=[tracker_params]
    )
    ld.add_action(trk_node)

    # Gesture recognition node

    # # Object recognition
    # clip_obj_rec_server = Node(package = "situated_interaction", 
    #                 executable = "clip_vis_rec_server.py",
    #                 name = "clip_vis_rec_server",
    #                 # remappings=[('/clip_scene_image','/oak/rgb/image_raw')],
    #                 parameters=[semantic_tracking_params]
    # )
    # ld.add_action(clip_obj_rec_server)

    # # interaction manager node
    # semantic_tracking_node = Node(
    #     package='situated_interaction',
    #     executable='semantic_tracking_node.py',
    #     name='semantic_tracking_node',
    #     output='screen',
    #     parameters=[semantic_tracking_params]
    # )
    # ld.add_action(semantic_tracking_node)

    # Foxglove bridge for visualization
    viz_node = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('marmot'),
                'launch/foxglove_bridge_launch.xml'))
    )
    ld.add_action(viz_node)

    return ld