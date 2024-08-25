#!/bin/bash 
set -e 
echo "Pointcloud to laserscan entrypoint" 

# Source ROS environment
source "/opt/ros/$ROS1_DISTRO/setup.bash" 
cd /ros1_ws 
source devel/setup.bash 

# Display bridged messages and launch the bridge
# TODO - adjust ROS parameters
rosrun pointcloud_to_laserscan pointcloud_to_laserscan_node cloud_in:=$PCL_TOPIC scan:=$SCAN_TOPIC
exec "$@"