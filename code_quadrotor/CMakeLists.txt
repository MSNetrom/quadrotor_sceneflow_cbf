cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0057 NEW)
project(PDFlow)

# Set the CUDA host compiler to gcc-8.
# (Do NOT both set CUDA_HOST_COMPILER and append -ccbin in CUDA_NVCC_FLAGS.)
#set(CUDA_HOST_COMPILER "/usr/bin/gcc-8")
set(CUDA_HOST_COMPILER "/usr/bin/gcc")

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Find required external packages.
find_package(CUDA REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
#set(OpenCV_DIR "/home/arl/PD-Flow-/opencv/bin/lib/cmake/opencv4/")
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# Find catkin and declare ROS dependencies.
find_package(catkin REQUIRED COMPONENTS 
  roscpp 
  my_ros_package 
  pi_controller_node
  std_msgs 
  sensor_msgs 
  cv_bridge
)

# Include directories provided by external packages and catkin.
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${YAML_CPP_INCLUDE_DIR}
)

# Specify source files
set(cuda_src src/pdflow_cudalib.cu src/cud/cuda_logsumexp.cu src/apps/cbf_calc.cu)
#set(cpp_src src/scene_flow.cpp)

# GPU architecture flags.
set(CUDA_ARCH_BIN "87" CACHE STRING "Target GPU arch for Tegra186 (Jetson TX2)")
set(CUDA_ARCH_PTX "87" CACHE STRING "Target GPU arch for Tegra186 (Jetson TX2)")

include(CudaComputeTargetFlags.cmake)
append_target_arch_flags()

# Append additional NVCC flags (do NOT add -ccbin here).
list(APPEND CUDA_NVCC_FLAGS "-O3;-Xcompiler;-fPIC;-std=c++11")

# Compile the CUDA source.
cuda_compile(cuda_objs ${cuda_src})

# Build the PDFlow shared library.
add_library(${PROJECT_NAME} SHARED ${cuda_src} ${cuda_objs})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${catkin_LIBRARIES} cublas)

# Explicitly set the linker language to CXX.
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

#
# -- Package configuration for external consumption --
#

# Export target file.
file(REMOVE "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")
export(TARGETS ${PROJECT_NAME} APPEND FILE "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

set(CMAKECONFIG_INSTALL_DIR lib/cmake/${PROJECT_NAME})
file(RELATIVE_PATH REL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKECONFIG_INSTALL_DIR}"
                               "${CMAKE_INSTALL_PREFIX}/include")

# Build tree configuration.
set(EXPORT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
configure_file(${CMAKE_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake @ONLY IMMEDIATE)

# Install tree configuration.
set(EXPORT_INCLUDE_DIR "\${PROJECT_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(${CMAKE_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake @ONLY)

# Optionally export package for use by other software.
option(EXPORT_${PROJECT_NAME} "Export ${PROJECT_NAME} package for use by other software" ON)
if(EXPORT_${PROJECT_NAME})
    export(PACKAGE ${PROJECT_NAME})
endif()

# Install targets and headers.
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake"
        DESTINATION ${CMAKECONFIG_INSTALL_DIR})

# Create an uninstall target.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY)

# Add ROS includes to your target
target_include_directories(${PROJECT_NAME} PRIVATE
  ${catkin_INCLUDE_DIRS}
)

add_executable(pdflow_app src/apps/my_app.cpp 
src/apps/my_extras.cpp 
src/apps/cbf_solver.cpp 
src/apps/rs_reader_threaded.cpp 
src/apps/vector_median.cpp
src/apps/scene_transform.cpp
src/apps/magnitude_filter.h)
target_link_libraries(pdflow_app 
  ${catkin_LIBRARIES} 
  ${PROJECT_NAME} 
  ${OpenCV_LIBS} 
  ${Eigen3_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  cublas
  realsense2
)
