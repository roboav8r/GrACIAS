#!/bin/bash 
set -e 
echo "Perception entrypoint" 

# Source ROS2 and bridge environments 
source "/opt/ros/humble/setup.bash" 
cd /perception_ws 
source install/setup.bash 

ros2 launch situated_interaction philbart.launch.py