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
        'oakd_params.yaml'
    )
    tracker_params = os.path.join(
        get_package_share_directory('marmot'),
        'config',
        'oakd_tracker.yaml'
    )
    ar_params = os.path.join(
        get_package_share_directory('situated_interaction'),
        'config',
        'ar_commands.yaml'
    )
    mic_params = os.path.join(
        get_package_share_directory('ros_audition'),
        'config',
        'c270_config.yaml'
    )
    clip_params = os.path.join(
        get_package_share_directory('mm_scene_rec'),
        'config',
        'clip_params.yaml'
    )
    audio_scene_rec_params = os.path.join(
        get_package_share_directory('mm_scene_rec'),
        'config',
        'audio_scene_rec_params.yaml'
    )

    bayes_est_params = os.path.join(
        get_package_share_directory('mm_scene_rec'),
        'config',
        'bayes_estimator_params.yaml'
    )

    # Static TF node
    tf_node = Node(package = "tf2_ros", 
                    executable = "static_transform_publisher",
                    arguments = ["0", "0", "1.0", "0", "0", "0", "map", "oak-d-base-frame"]
    )
    ld.add_action(tf_node)

    # Sensor node
    cam_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('depthai_ros_driver'),
                'launch',
                'rgbd_pcl.launch.py'
            ])
        ]),
        launch_arguments={'params_file': oakd_params }.items()
    )
    ld.add_action(cam_node)

    audio_acq_node = Node(
        package='ros_audition',
        executable='audio_acq_node.py',
        name='audio_acq_node',
        output='screen',
        parameters=[mic_params]
    )
    ld.add_action(audio_acq_node)

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

    fused_scene_rec_node = Node(package = "mm_scene_rec", 
                    executable = "bayes_scene_est.py",
                    name = "bayes_scene_category_est",
                    output = 'screen',
                    parameters=[bayes_est_params]
    )
    ld.add_action(fused_scene_rec_node)


    # Detector preprocessing node
    preproc_node = Node(
        package='marmot',
        executable='depthai_img_preproc',
        name='depthai_img_preproc_node',
        remappings=[('/depthai_detections','/oak/nn/spatial_detections'),('/depthai_img','/oak/rgb/image_raw')],
        output='screen',
        parameters=[oakd_params])    
    ld.add_action(preproc_node)

    # ar_node = Node(
    #     package='situated_interaction',
    #     executable='ar_preproc',
    #     name='ar_preproc_node',
    #     output='screen',
    #     parameters=[ar_params])    
    # ld.add_action(ar_node)

    # Tracker node
    trk_node = Node(
        package='marmot',
        executable='tbd_node.py',
        name='tbd_tracker_node',
        output='screen',
        remappings=[('/detections','/converted_detections')],
        parameters=[tracker_params]
    )
    ld.add_action(trk_node)

    # # situated_interaction track -> tracked person preprocessor node
    # person_preproc_node = Node(
    #     package='situated_interaction',
    #     executable='track_preproc',
    #     name='track_preproc_node',
    #     output='screen')
    # ld.add_action(person_preproc_node)

    # # interaction manager node
    # int_mgr_node = Node(
    #     package='situated_interaction',
    #     executable='int_mgr_node.py',
    #     name='interaction_manager_node',
    #     output='screen',
    # )
    # ld.add_action(int_mgr_node)



    # Foxglove bridge for visualization
    viz_node = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('marmot'),
                'launch/foxglove_bridge_launch.xml'))
    )
    ld.add_action(viz_node)

    return ld