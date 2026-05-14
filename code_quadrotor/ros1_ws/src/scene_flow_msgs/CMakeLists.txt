cmake_minimum_required(VERSION 2.8.3)
project(scene_flow_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  geometry_msgs
)

# Add message files.
add_message_files(
  FILES
    SceneFlow.msg
)

# Generate messages with dependencies.
generate_messages(
  DEPENDENCIES
    std_msgs
    geometry_msgs
)

# --- Dummy target to force creation of config files ---
add_library(${PROJECT_NAME} INTERFACE)

# Declare the catkin package.
catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs
)

install(DIRECTORY msg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)