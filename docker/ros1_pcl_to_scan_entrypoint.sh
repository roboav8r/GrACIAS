#!/bin/bash 
set -e 
echo "Pointcloud to laserscan entrypoint" 

# Source ROS environment
source "/opt/ros/$ROS1_DISTRO/setup.bash" 
cd /ros1_ws 
source devel/setup.bash 

# Display bridged messages and launch the bridge 
rosrun pointcloud_to_laserscan pointcloud_to_laserscan_node
exec "$@"