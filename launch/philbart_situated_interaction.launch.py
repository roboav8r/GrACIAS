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

    tf_node = Node(package = "tf2_ros", 
                    executable = "static_transform_publisher",
                    arguments = [".53", "0", ".48", "0", "0", "0", "philbart/base_link", "philbart/front_laser"]
    )
    ld.add_action(tf_node)


    ### SENSORS
    # 2D LiDAR node
    lidar_2d_node = Node(
        package='urg_node',
        executable='urg_node_driver',
        name='urg_node',
        output='screen',
        parameters=[config]
    )
    ld.add_action(lidar_2d_node)

    # LiDAR leg detection
    leg_det_node = Node(
        package='dr_spaam_ros',
        executable='node.py',
        name='dr_spaam_ros',
        output='screen',
        parameters=[config]
    )
    ld.add_action(leg_det_node)

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
    depthai_preproc_node = Node(
        package='marmot',
        executable='depthai_img_preproc',
        name='depthai_img_preproc_node',
        remappings=[('/converted_detections','/converted_vision_detections'),
                    ('/depthai_detections','/oak/nn/spatial_detections'),
                    ('/depthai_img','/oak/rgb/image_raw')],
        output='screen',
        parameters=[config])    
    ld.add_action(depthai_preproc_node)
    
    lidar_preproc_node = Node(
        package='marmot',
        executable='lidar_2d_preproc',
        name='lidar_preproc_node',
        remappings=[('/pose_array_detections','/philbart/dr_spaam_detections'),
                    ('/converted_detections','converted_lidar_detections'),
                    ('/scan','/philbart/scan'),
                    ('/point_cloud','/philbart/lidar_points')],
        output='screen',
        parameters=[config])    
    ld.add_action(lidar_preproc_node)

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

    # Direction of arrival (DOA)/PyRoomAcoustics node
    pra_node = Node(
        package='ros_audition',
        executable='pra_node.py',
        name='pra_node',
        output='screen',
        parameters=[config]
    )
    ld.add_action(pra_node)

    # Directional speech recognition node
    speech_node = Node(
        package='ros_audition',
        executable='doa_speech_rec_node.py',
        name='directional_speech_rec_node',
        output='screen',
        parameters=[config]
    )
    ld.add_action(speech_node)

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
        # remappings=[('/detections','/converted_detections')],
        parameters=[tracker_config]
    )
    ld.add_action(trk_node)

    # Semantic fusion
    semantic_tracking_node = Node(
        package='situated_interaction',
        executable='semantic_tracking_node.py',
        name='semantic_fusion_node',
        output='screen',
        remappings=[('/tracks','/tbd_tracker_node/tracks')],
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