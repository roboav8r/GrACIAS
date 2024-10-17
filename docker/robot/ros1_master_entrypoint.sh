#!/bin/bash 
set -e 
echo "ROS1 master entrypoint" 

# Source ROS environment
source "/opt/ros/$ROS1_DISTRO/setup.bash" 
cd /ros1_ws 
source devel/setup.bash 

# Launch the leg detector
roscore
exec "$@"