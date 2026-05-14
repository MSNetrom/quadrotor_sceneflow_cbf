#!/bin/bash
source /opt/ros/noetic/setup.bash  # change "jade" (or your ROS distro) as appropriate

# (Optional) Clean build artifacts.
rm -rf ros1_ws/build ros1_ws/devel

cd ros1_ws
catkin_make

# Source the new setup file.
source devel/setup.bash

echo "ROS1 workspace built."

export CMAKE_PREFIX_PATH="${HOME}/ros1_ws/devel;/opt/ros/noetic:${CMAKE_PREFIX_PATH}"

# Build main package
cd ..
cmake .
make

echo "Build complete."