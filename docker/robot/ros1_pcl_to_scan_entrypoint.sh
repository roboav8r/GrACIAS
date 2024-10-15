#!/bin/bash 
set -e 
echo "Pointcloud to laserscan entrypoint" 

# Source ROS environment
source "/opt/ros/$ROS1_DISTRO/setup.bash" 
cd /ros1_ws 
source devel/setup.bash 

roslaunch dr_spaam_ros philbart_pcl_to_laserscan.launch --wait
exec "$@"