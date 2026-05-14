cmake_minimum_required(VERSION 3.0.2)
project(pi_controller_node)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
)

catkin_package(
  # Exported include directory for header files.
  INCLUDE_DIRS include
  # Library name to be exported.
  LIBRARIES pi_controller_node
  # Catkin packages that this package depends on.
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs geometry_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(pi_controller_node
    src/pi_controller_node.cpp
)

## Specify libraries to link a library or executable target against
target_link_libraries(pi_controller_node
    ${catkin_LIBRARIES}
)

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
install(TARGETS pi_controller_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/pi_controller_node/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

