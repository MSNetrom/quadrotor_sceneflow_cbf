#ifndef RS_READER_THREADED_HPP
#define RS_READER_THREADED_HPP

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <thread>
#include <atomic>
#include <algorithm>
#include <iostream>
#include <geometry_msgs/Vector3.h>
#include "my_ros_package/ros_nodes.hpp"

// A container similar to the ROS CpuImageBuffer to hold the last two synced images.
struct CpuImageBuffer {
    unsigned int newImages = 0;  // Allowed values: 0, 1, or 2

    // Raw newest image
    cv::Mat raw_newest_color_image;
    cv::Mat raw_newest_depth_image;

    // Two-slot buffers for grayscale and depth images.
    cv::Mat gray_buffer[2];
    cv::Mat depth_buffer[2];

    // Timestamps for each image (using steady_clock).
    std::chrono::steady_clock::time_point gray_timestamps[2];
    std::chrono::steady_clock::time_point depth_timestamps[2];
    
    // Angular velocity synchronized with the frames
    geometry_msgs::Vector3 angular_velocity[2];
};

class RSReader {
public:
    // Constructor now takes a reference to RosNode
    RSReader(my_ros_package::RosNode& ros_node);
    // Destructor stops the background thread before object destruction.
    ~RSReader();

    // Retrieve a deep copy of the current CPU image buffer.
    CpuImageBuffer getCpuImageBuffer();

    // Returns camera intrinsics as a tuple: (fx, fy, cx, cy, width, height)
    std::tuple<float, float, float, float, unsigned int, unsigned int> getCameraIntrinsics() const;

    // New: Wait until the initial valid images are available.
    void waitForFirstImages();

private:
    // The main loop, running on its own thread, that reads and processes frames.
    void runLoop();

    // RealSense objects.
    rs2::pipeline            pipe_;
    rs2::config              cfg_;
    rs2::pipeline_profile    profile_;
    rs2_intrinsics           intrinsics_;

    // CPU image buffer along with its associated mutex and condition variable.
    CpuImageBuffer           cpu_buffer_;
    std::mutex               image_mutex_;
    std::condition_variable  new_images_cv_;

    // Thread control.
    std::atomic<bool>        exit_loop_;
    std::thread              run_thread_;

    // Create filtering objects once per thread iteration.
    rs2::spatial_filter spatial_filter_;
    rs2::temporal_filter temporal_filter_;

    // Reference to the ROS node for accessing angular velocity
    my_ros_package::RosNode& ros_node_;
    
    // Cached transform from IMU to camera frame (constant)
    Eigen::Affine3f imu_to_camera_transform_;
};

#endif // RS_READER_THREADED_HPP