#!/bin/bash 
set -e 
echo "AR bridge entrypoint" 

# Source ROS2 and bridge environments 
source "/opt/ros/$ROS2_DISTRO/setup.bash" 
cd /bridge_ws 
source install/setup.bash 

# Display bridged messages and launch the bridge 
dpkg -l | grep cyclone
ros2 run ros1_bridge dynamic_bridge --print-pairs 
ros2 run ros1_bridge dynamic_bridge 
exec "$@"