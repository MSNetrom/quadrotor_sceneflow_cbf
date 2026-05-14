#ifndef MY_ROS_PACKAGE_ROS_NODES_HPP_
#define MY_ROS_PACKAGE_ROS_NODES_HPP_

#include <ros/ros.h>
#include <scene_flow_msgs/SceneFlow.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/Twist.h>    // For reference control messages.
#include <mavros_msgs/PositionTarget.h>  // Updated include for reference control messages
#include <array>
#include <mutex>                    // For std::mutex
#include <opencv2/core.hpp>         // For cv::Mat
#include <nav_msgs/Odometry.h>       // For odometry messages

// TF2 headers:
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/exceptions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h> // Include this for tf2 conversions
#include <tf2_ros/transform_broadcaster.h>

#include <tf2_ros/static_transform_broadcaster.h>

#include <ros/ros.h>
#include <boost/bind.hpp>
#include <algorithm>  // For std::min
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <geometry_msgs/Twist.h>  // For processing reference control messages.

// Include Eigen for quaternion operations
#include <Eigen/Geometry>

// Include array for vector operations
#include <array>

#include <std_msgs/Float64.h>                // For CBF value messages
#include <geometry_msgs/PointStamped.h>      // For closest point publishing
#include <visualization_msgs/Marker.h>       // For RViz visualization

#include <sensor_msgs/PointCloud2.h>

namespace my_ros_package {

// Bounds the length of a vector to max_length while preserving direction
// Handles zero vectors in a numerically stable way
std::array<float, 3> boundVectorLength(const std::array<float, 3>& vec, float max_length);

class RosNode
{
public:
  explicit RosNode(ros::NodeHandle &nh);
  ~RosNode();

  void publishColorImage(const cv::Mat &image, std::chrono::steady_clock::time_point timestamp);
  void publishDepthImage(const cv::Mat &image, std::chrono::steady_clock::time_point timestamp);
  void publishSceneFlowImage(const cv::Mat &image, std::chrono::steady_clock::time_point timestamp);
  void publishSceneFlowMagnitudeImage(const cv::Mat &image, std::chrono::steady_clock::time_point timestamp);
  void publishSceneFlow(const scene_flow_msgs::SceneFlow &msg);
  void publishControlVector(const std::array<float, 3> &u, const std::array<float, 3> &u_unfiltered);

  // Getter to retrieve the latest reference control as a float array of length 3.
  std::array<float, 3> getLatestReferenceControl() const;
  
  // Getters for odometry information.
  // Add methods to get transformed orientation and angular velocity
  geometry_msgs::Quaternion getLatestOrientationInFrame(const std::string& target_frame);
  geometry_msgs::Vector3 getLatestAngularVelocityInFrame(const std::string& target_frame);

  // Get the camera offset with relation to the base.
  // This method attempts to lookup the transform from "base_link" to "camera0".
  geometry_msgs::TransformStamped getCameraOffset();

  // Get orientation quaternion with roll and pitch zeroed (level frame with yaw preserved)
  geometry_msgs::Quaternion getLevelWithHeadingOrientation() const;
  
  // Get the relative transformation quaternion from body frame to level-with-heading frame
  geometry_msgs::Quaternion getBodyToLevelWithHeadingTransform() const;

  // Get the cached body to level-with-heading transform
  geometry_msgs::TransformStamped getBodyToLevelWithHeadingTransformStamped() const;

  // New method for point cloud visualization
  void publishDepthPointCloud(const cv::Mat &depth_mat, const cv::Mat &xx_mat, const cv::Mat &yy_mat,
                             std::chrono::steady_clock::time_point timestamp);

  // Add method for publishing raw minimum CBF value
  //void publishRawCbfValue(float raw_cbf_value);

  // Add method to publish the largest scene flow vector
  void publishLargestSceneFlowVector(const geometry_msgs::TwistStamped &largest_flow_vector);

  // Add method to publish the raw scene flow
  void publishSceneFlowRaw(const scene_flow_msgs::SceneFlow &msg);

  // Add method to publish the processed scene flow
  void publishSceneFlowProcessed(const scene_flow_msgs::SceneFlow &msg);

  // New methods for getting transformed IMU data (always to camera frame)
  geometry_msgs::Vector3 getAngularVelocityInCameraFrame();
  geometry_msgs::Quaternion getOrientationInCameraFrame();

  // Add methods to publish psi and h values as images
  void publishPsiImage(const cv::Mat &image, std::chrono::steady_clock::time_point timestamp,
                       float min_negative = -1.0f, float max_positive = 9.0f);
  void publishHImage(const cv::Mat &image, std::chrono::steady_clock::time_point timestamp,
                     float min_negative = -1.0f, float max_positive = 89.0f);

  // Add methods to publish unscaled psi and h values as images
  void publishRawPsiImage(const cv::Mat &image, std::chrono::steady_clock::time_point timestamp);
  void publishRawHImage(const cv::Mat &image, std::chrono::steady_clock::time_point timestamp);

private:
  // Callback for receiving reference control commands.
  void referenceControlCallback(const geometry_msgs::Twist::ConstPtr& msg);
  // Callback for receiving odometry messages.
  void odometryCallback(const nav_msgs::Odometry::ConstPtr& msg);

  ros::NodeHandle nh_;

  ros::Publisher scene_flow_pub_raw_;
  ros::Publisher scene_flow_pub_processed_;
  ros::Publisher scene_flow_magnitude_image_pub_;

  ros::Publisher control_vector_viz_pub_;
  ros::Publisher control_vector_viz_raw_pub_;
  ros::Publisher control_vector_viz_unfiltered_pub_;
  ros::Publisher input_viz_pub_;
  ros::Publisher color_image_pub_;
  ros::Publisher depth_image_pub_;
  ros::Publisher setpoint_raw_pub_;
  
  // Point cloud publishers
  ros::Publisher depth_cloud_pub_;
  
  // Add frame IDs for odometry and camera frames
  std::string odometry_frame_id_ = "imu"; // Replace with actual odometry frame
  std::string camera_frame_id_ = "camera0"; // Replace with your camera frame ID
  
  // Subscriber that listens for the latest reference control.
  ros::Subscriber reference_control_sub_;
  // Subscriber for odometry messages.
  ros::Subscriber odometry_sub_;

  // Latest received reference control converted to a float array of size 3.
  std::array<float, 3> latest_reference_control_ = {0.0f, 0.0f, 0.0f};
  geometry_msgs::Twist latest_reference_control_msg_;

  // Mutex to protect access to latest_reference_control_.
  mutable std::mutex reference_mutex_;

  // Latest odometry data for orientation and angular velocity.
  geometry_msgs::Quaternion latest_orientation_;
  geometry_msgs::Vector3 latest_angular_velocity_;

  // Mutex to protect access to odometry data.
  mutable std::mutex odometry_mutex_;

  // --- TF2 members for transform lookup ---
  // Declare the buffer first so that it is available for the listener.
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // TF2 broadcaster for publishing transforms
  tf2_ros::TransformBroadcaster tf_broadcaster_;
  
  // Cached transform from body to level-with-heading frame
  geometry_msgs::TransformStamped body_to_level_heading_transform_;
  
  // Mutex to protect access to the cached transform
  mutable std::mutex transform_mutex_;

  // Cache the transform between IMU and various target frames
  mutable std::map<std::string, Eigen::Affine3d> cached_transforms_;
  mutable std::map<std::string, ros::Time> cached_transform_times_;
  mutable std::mutex cached_transform_mutex_;

  // Cached transform from IMU to camera frame
  Eigen::Affine3d cached_imu_to_camera_transform_;
  bool cached_transform_valid_ = false;

  // Add publishers for psi and h images and raw values
  ros::Publisher psi_image_pub_;
  ros::Publisher h_image_pub_;

  // Add publishers for raw images
  ros::Publisher raw_psi_image_pub_;
  ros::Publisher raw_h_image_pub_;
  
};

}  // namespace my_ros_package

#endif  // MY_ROS_PACKAGE_ROS_NODES_HPP_