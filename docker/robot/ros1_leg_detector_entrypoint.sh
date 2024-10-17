#!/bin/bash 
set -e 
echo "DR-SPAAM leg detector entrypoint" 

# Source ROS environment
source "/opt/ros/$ROS1_DISTRO/setup.bash" 
cd /ros1_ws 
source devel/setup.bash 

# Launch the leg detector
roslaunch dr_spaam_ros philbart_dr_spaam_ros.launch --wait
exec "$@"