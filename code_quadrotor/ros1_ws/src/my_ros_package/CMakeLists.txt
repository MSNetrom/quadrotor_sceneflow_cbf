cmake_minimum_required(VERSION 2.8.3)
project(my_ros_package)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  message_filters
  cv_bridge
  scene_flow_msgs
  tf2_ros
  mavros_msgs
)
find_package(Eigen3 3.3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ros_nodes
  CATKIN_DEPENDS roscpp sensor_msgs message_filters cv_bridge scene_flow_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(ros_nodes src/ros_nodes.cpp)
target_link_libraries(ros_nodes ${catkin_LIBRARIES} ${EIGEN3_LIBRARIES})

install(TARGETS ros_nodes
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)