#!/bin/bash

ros2 bag record -s mcap -o bags/$1 \
/ar_pose_marker \
/audio_scene_category \
/clip_scene_category \
/converted_lidar_detections \
/converted_vision_detections \
/dr_spaam_rviz \
/fused_scene_est/fused_scene_category \
/hierarchical_commands \
/oak/rgb/image_raw \
/oak/stereo/camera_info \
/philbart/dr_spaam_detections \
/philbart/hololens/cmd_vel \
/philbart/waypoint_manager/waypoint_plan \
/scan \
/semantic_scene \
/speech_az_sources \
/tbd_tracker_node/scenes \
/tbd_tracker_node/tracks \
/tf \
/tf_static