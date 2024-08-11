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

    # # Config files
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

    # ar_params = os.path.join(
    #     get_package_share_directory('situated_interaction'),
    #     'config',
    #     'ar_commands.yaml'
    # )
    # semantic_tracking_params = os.path.join(
    #     get_package_share_directory('situated_interaction'),
    #     'config',
    #     'semantic_tracking_params.yaml'
    # )
    # audio_params = os.path.join(
    #     get_package_share_directory('ros_audition'),
    #     'config',
    #     'soundsusb_config.yaml'
    # )
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

    # bayes_est_params = os.path.join(
    #     get_package_share_directory('mm_scene_rec'),
    #     'config',
    #     'bayes_estimator_params.yaml'
    # )

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

    # ar_node = Node(
    #     package='situated_interaction',
    #     executable='ar_preproc',
    #     name='ar_preproc_node',
    #     output='screen',
    #     parameters=[ar_params])    
    # ld.add_action(ar_node)

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

    results_rec_node = Node(package = "situated_interaction", 
                    executable = "record_scene_results_node.py",
                    name = "record_scene_results_node",
                    output = 'screen'
    )
    ld.add_action(results_rec_node)

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

    # Object recognition
    # clip_obj_rec_server = Node(package = "situated_interaction", 
    #                 executable = "clip_vis_rec_server.py",
    #                 name = "clip_vis_rec_server",
    #                 # remappings=[('/clip_scene_image','/oak/rgb/image_raw')],
    #                 parameters=[semantic_tracking_params]
    # )
    # ld.add_action(clip_obj_rec_server)

    # 
    # # Sound source localization and beamformer nodes
    # pra_node = Node(
    #     package='ros_audition',
    #     executable='pra_node.py',
    #     name='pra_node',
    #     output='screen',
    #     parameters=[audio_params]
    # )
    # ld.add_action(pra_node)

    # speech_node = Node(
    #     package='ros_audition',
    #     executable='doa_speech_rec_node.py',
    #     name='directional_speech_rec_node',
    #     output='screen',
    #     parameters=[audio_params]
    # )
    # ld.add_action(speech_node)

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