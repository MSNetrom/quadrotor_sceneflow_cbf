#include "my_ros_package/ros_nodes.hpp"
#include <Eigen/Geometry>
#include <tf2_ros/transform_broadcaster.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

namespace my_ros_package {

// Bounds the length of a vector to max_length while preserving direction
// Handles zero vectors in a numerically stable way
std::array<float, 3> boundVectorLength(const std::array<float, 3>& vec, float max_length) {
    // Handle invalid max_length
    if (max_length <= 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }

    // Calculate vector length
    float length = std::sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    
    // If length is effectively zero, return zero vector
    if (length < 1e-6f) {
        return {0.0f, 0.0f, 0.0f}; 
    }

    // If length exceeds max, scale the vector down
    if (length > max_length) {
        float scale = max_length / length;
        return {vec[0] * scale, vec[1] * scale, vec[2] * scale};
    }

    // Otherwise return original vector
    return vec;
}

RosNode::RosNode(ros::NodeHandle &nh)
  : nh_(nh),
    // Initialize TF2 members.
    tf_buffer_(),
    tf_listener_(tf_buffer_),
    tf_broadcaster_(),
    body_to_level_heading_transform_(),
    transform_mutex_()
{
  // Initialize the publisher for scene flow messages (queue size 10).
  scene_flow_pub_raw_ = nh_.advertise<scene_flow_msgs::SceneFlow>("/scene_flow_raw", 10);
  scene_flow_pub_processed_ = nh_.advertise<scene_flow_msgs::SceneFlow>("/scene_flow_processed", 10);
  // Add publisher for scene flow magnitude image.
  scene_flow_magnitude_image_pub_ = nh_.advertise<sensor_msgs::Image>("/scene_flow_magnitude_image", 10);

  // Initialize the publisher for control vector messages (TwistStamped, queue size 10).
  control_vector_viz_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("/mavros/setpoint_raw/local_viz", 10);
  input_viz_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("/input_viz", 10);
  control_vector_viz_raw_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("/control_vector_viz_raw", 10);
  control_vector_viz_unfiltered_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("/control_vector_viz_unfiltered", 10);
  
  // Initialize setpoint_raw publisher for acceleration commands
  setpoint_raw_pub_ = nh_.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 10);
  
  // Initialize the publisher for color image messages (queue size 10).
  color_image_pub_ = nh_.advertise<sensor_msgs::Image>("/color_image", 10);

  // Initialize the publisher for depth image messages (queue size 10).
  depth_image_pub_ = nh_.advertise<sensor_msgs::Image>("/depth_image", 10);
  
  // Initialize point cloud publishers
  depth_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/depth_point_cloud", 1);

  // Initialize the subscriber for reference control messages (geometry_msgs::Twist, queue size 10).
  reference_control_sub_ = nh_.subscribe("/cmd_joy", 1, &RosNode::referenceControlCallback, this);

  // Initialize the subscriber for odometry messages (nav_msgs::Odometry) from "/rig_node/odometry".
  //odometry_sub_ = nh_.subscribe("/mavros/local_position/odom", 10, &RosNode::odometryCallback, this);  // odom?
  odometry_sub_ = nh_.subscribe("/rig_node/odometry", 10, &RosNode::odometryCallback, this);

  // Optionally initialize latest_reference_control_ to zero values.
  latest_reference_control_ = {0.0f, 0.0f, 0.0f};
  latest_reference_control_msg_.linear.x = 0.0f;
  latest_reference_control_msg_.linear.y = 0.0f;
  latest_reference_control_msg_.linear.z = 0.0f;
  latest_reference_control_msg_.angular.z = 0.0f;

  // Initialize odometry-related data.
  latest_orientation_.x = 0.0;
  latest_orientation_.y = 0.0;
  latest_orientation_.z = 0.0;
  latest_orientation_.w = 1.0;  // Identity quaternion.
  
  latest_angular_velocity_.x = 0.0;
  latest_angular_velocity_.y = 0.0;
  latest_angular_velocity_.z = 0.0;

  // Initialize frame IDs and cached transform
  cached_transform_valid_ = false;

  // Initialize the publishers for psi and h images
  psi_image_pub_ = nh_.advertise<sensor_msgs::Image>("/psi_image", 10);
  h_image_pub_ = nh_.advertise<sensor_msgs::Image>("/h_image", 10);

  // Initialize the publishers for raw images
  raw_psi_image_pub_ = nh_.advertise<sensor_msgs::Image>("/raw_psi_image", 10);
  raw_h_image_pub_ = nh_.advertise<sensor_msgs::Image>("/raw_h_image", 10);
}

RosNode::~RosNode()
{
  // No threads or dynamic resources to clean up.
}

void RosNode::publishSceneFlowRaw(const scene_flow_msgs::SceneFlow &msg)
{
  scene_flow_pub_raw_.publish(msg);
}

void RosNode::publishSceneFlowProcessed(const scene_flow_msgs::SceneFlow &msg)
{
  scene_flow_pub_processed_.publish(msg);
}

void RosNode::publishControlVector(const std::array<float, 3> &u, const std::array<float, 3> &u_unfiltered)
{

  // Create a bound version of the control vector
  std::array<float, 3> u_bound = boundVectorLength(u, 2.0f);

  {
    std::lock_guard<std::mutex> lock(reference_mutex_);
    mavros_msgs::PositionTarget target;
    target.coordinate_frame = mavros_msgs::PositionTarget::FRAME_BODY_NED;
    
    // Ignore position and velocity
    target.type_mask = mavros_msgs::PositionTarget::IGNORE_PX | 
                       mavros_msgs::PositionTarget::IGNORE_PY |
                       mavros_msgs::PositionTarget::IGNORE_PZ |
                       mavros_msgs::PositionTarget::IGNORE_VX |
                       mavros_msgs::PositionTarget::IGNORE_VY |
                       mavros_msgs::PositionTarget::IGNORE_VZ |
                       mavros_msgs::PositionTarget::IGNORE_YAW;
                       
    target.acceleration_or_force.x = u_bound[0];
    target.acceleration_or_force.y = u_bound[1];  // Negate Y for NED convention
    target.acceleration_or_force.z = u_bound[2];  // Negate Z for NED convention
    target.yaw_rate = latest_reference_control_msg_.angular.z;
    setpoint_raw_pub_.publish(target);
  }

  geometry_msgs::TwistStamped viz_msg;
  viz_msg.header.stamp = ros::Time::now();
  viz_msg.header.frame_id = "base_link";
  viz_msg.twist.linear.x = u_bound[0];
  viz_msg.twist.linear.y = u_bound[1];
  viz_msg.twist.linear.z = u_bound[2];
  viz_msg.twist.angular.z = latest_reference_control_msg_.angular.z;
  control_vector_viz_pub_.publish(viz_msg);

  // Publish the raw control vector
  geometry_msgs::TwistStamped raw_viz_msg;
  raw_viz_msg.header.stamp = ros::Time::now();
  raw_viz_msg.header.frame_id = "base_link";
  raw_viz_msg.twist.linear.x = u[0];
  raw_viz_msg.twist.linear.y = u[1];
  raw_viz_msg.twist.linear.z = u[2];
  raw_viz_msg.twist.angular.z = latest_reference_control_msg_.angular.z;
  control_vector_viz_raw_pub_.publish(raw_viz_msg);

  // Also publish the input action
  {
    std::lock_guard<std::mutex> lock(reference_mutex_);
    geometry_msgs::TwistStamped input_viz_msg;
    input_viz_msg.header.stamp = ros::Time::now();
    input_viz_msg.header.frame_id = "base_link";
    input_viz_msg.twist.linear.x = latest_reference_control_[0];
    input_viz_msg.twist.linear.y = latest_reference_control_[1];
    input_viz_msg.twist.linear.z = latest_reference_control_[2];
    input_viz_msg.twist.angular.z = latest_reference_control_msg_.angular.z;
    input_viz_pub_.publish(input_viz_msg);
  }

  // Publish the unfiltered control vector
  geometry_msgs::TwistStamped unfiltered_viz_msg;
  unfiltered_viz_msg.header.stamp = ros::Time::now();
  unfiltered_viz_msg.header.frame_id = "base_link";
  unfiltered_viz_msg.twist.linear.x = u_unfiltered[0];
  unfiltered_viz_msg.twist.linear.y = u_unfiltered[1];
  unfiltered_viz_msg.twist.linear.z = u_unfiltered[2];
  unfiltered_viz_msg.twist.angular.z = latest_reference_control_msg_.angular.z;
  control_vector_viz_unfiltered_pub_.publish(unfiltered_viz_msg);
  
}

void RosNode::publishColorImage(const cv::Mat &image, std::chrono::steady_clock::time_point timestamp)
{
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", image).toImageMsg();
  msg->header.stamp = ros::Time::now();
  color_image_pub_.publish(msg);
}

void RosNode::publishDepthImage(const cv::Mat &image, std::chrono::steady_clock::time_point timestamp)
{
  cv::Mat image8;
  image.convertTo(image8, CV_8UC1, 255.0 / 65535.0);
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", image8).toImageMsg();
  msg->header.stamp = ros::Time::now();
  depth_image_pub_.publish(msg);
}

// Revised point cloud publishing method without PCL dependency
void RosNode::publishDepthPointCloud(const cv::Mat &depth_mat, const cv::Mat &xx_mat, const cv::Mat &yy_mat,
                                    std::chrono::steady_clock::time_point timestamp)
{
  // Create a point cloud message
  sensor_msgs::PointCloud2 cloud_msg;
  
  // Set up the message
  cloud_msg.header.stamp = ros::Time::now();
  cloud_msg.header.frame_id = "base_link";
  cloud_msg.height = depth_mat.rows;
  cloud_msg.width = depth_mat.cols;
  
  // Set up the fields
  cloud_msg.fields.resize(4); // x, y, z, rgb
  
  cloud_msg.fields[0].name = "x";
  cloud_msg.fields[0].offset = 0;
  cloud_msg.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
  cloud_msg.fields[0].count = 1;
  
  cloud_msg.fields[1].name = "y";
  cloud_msg.fields[1].offset = 4;
  cloud_msg.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
  cloud_msg.fields[1].count = 1;
  
  cloud_msg.fields[2].name = "z";
  cloud_msg.fields[2].offset = 8;
  cloud_msg.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
  cloud_msg.fields[2].count = 1;
  
  cloud_msg.fields[3].name = "rgb";
  cloud_msg.fields[3].offset = 12;
  cloud_msg.fields[3].datatype = sensor_msgs::PointField::FLOAT32;
  cloud_msg.fields[3].count = 1;
  
  // Set up the point cloud properties
  cloud_msg.is_bigendian = false;
  cloud_msg.point_step = 16; // 4 fields * 4 bytes
  cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;
  cloud_msg.is_dense = false;
  
  // Resize the data array to hold our points
  cloud_msg.data.resize(cloud_msg.row_step * cloud_msg.height);
  
  // Create iterators for the point cloud
  sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(cloud_msg, "rgb");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(cloud_msg, "rgb");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(cloud_msg, "rgb");
  
  // Fill the cloud with depth data
  for (int r = 0; r < depth_mat.rows; r++) {
    for (int c = 0; c < depth_mat.cols; c++) {
      float z = depth_mat.at<float>(r, c);
      float x = xx_mat.at<float>(r, c);
      float y = yy_mat.at<float>(r, c);
      
      // Actually assign the x, y, z values to the point cloud
      *iter_x = x;
      *iter_y = y;
      *iter_z = z;
      
      // Set default color (white)
      *iter_r = 255;
      *iter_g = 255;
      *iter_b = 255;
      
      // Move iterators to next point
      ++iter_x; ++iter_y; ++iter_z; 
      ++iter_r; ++iter_g; ++iter_b;
    }
  }
  
  // Publish the point cloud
  depth_cloud_pub_.publish(cloud_msg);
}

void RosNode::publishSceneFlowMagnitudeImage(const cv::Mat &image, std::chrono::steady_clock::time_point timestamp)
{
  // Convert to ROS image message. Use "mono8" encoding for grayscale images.
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();
  msg->header.stamp = ros::Time::now();
  scene_flow_magnitude_image_pub_.publish(msg);
}

// Modify the callback to handle PositionTarget instead of Twist
void RosNode::referenceControlCallback(const geometry_msgs::Twist::ConstPtr& msg)
{
  // First, get our transform
  geometry_msgs::TransformStamped base_link_to_vehicle;
  {
    std::lock_guard<std::mutex> lock(transform_mutex_);
    base_link_to_vehicle = body_to_level_heading_transform_;
  }

  // Convert transform to Eigen
  Eigen::Quaternionf base_link_to_vehicle_quat(
    base_link_to_vehicle.transform.rotation.w,
    base_link_to_vehicle.transform.rotation.x,
    base_link_to_vehicle.transform.rotation.y,
    base_link_to_vehicle.transform.rotation.z
  );

  {
    // Convert Twist linear velocities to acceleration commands 
    // (assuming joystick sends velocity-like commands that we treat as accelerations)
    Eigen::Vector3f acceleration_or_force_eigen(
        msg->linear.x,  // Map to match previous convention
        msg->linear.y,  // Map to match previous convention
        msg->linear.z   // Use z directly
    );

    // Transform from vehicle (level-with-heading) frame to base_link frame
    Eigen::Vector3f acceleration_or_force_base_link_eigen = 
        base_link_to_vehicle_quat * acceleration_or_force_eigen;

    // Lock the mutex to protect shared data.
    std::lock_guard<std::mutex> lock(reference_mutex_);
    
    // Store the transformed acceleration values
    latest_reference_control_[0] = static_cast<float>(acceleration_or_force_base_link_eigen(0));
    latest_reference_control_[1] = static_cast<float>(acceleration_or_force_base_link_eigen(1));
    latest_reference_control_[2] = static_cast<float>(acceleration_or_force_base_link_eigen(2));
    
    // Store the values in a format compatible with the rest of the code
    latest_reference_control_msg_.linear.x = msg->linear.x;
    latest_reference_control_msg_.linear.y = msg->linear.y;
    latest_reference_control_msg_.linear.z = msg->linear.z;
    latest_reference_control_msg_.angular.z = msg->angular.z;
  }
}

// Callback for receiving odometry data that extracts orientation (quaternion) and angular velocity.
void RosNode::odometryCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
  
  {
    std::lock_guard<std::mutex> lock(odometry_mutex_);
    latest_orientation_ = msg->pose.pose.orientation;
    latest_angular_velocity_ = msg->twist.twist.angular;
    // Store the frame_id for future reference
    odometry_frame_id_ = msg->header.frame_id;
  }
  
  // Calculate the relative transform
  geometry_msgs::Quaternion relative_rotation = getBodyToLevelWithHeadingTransform();
  
  // Create and store the transform
  {
    std::lock_guard<std::mutex> lock(transform_mutex_);
    
    body_to_level_heading_transform_.header.stamp = ros::Time::now(); // Use current time instead of message time
    body_to_level_heading_transform_.header.frame_id = "base_link";  // Parent frame (body)
    body_to_level_heading_transform_.child_frame_id = "vehicle";     // Child frame (level-with-heading)
    
    // Set translation to zero (frames share origin)
    body_to_level_heading_transform_.transform.translation.x = 0.0;
    body_to_level_heading_transform_.transform.translation.y = 0.0;
    body_to_level_heading_transform_.transform.translation.z = 0.0;
    
    // Set rotation
    body_to_level_heading_transform_.transform.rotation = relative_rotation;
  }
  
  // Broadcast the transform and add debug info
  tf_broadcaster_.sendTransform(body_to_level_heading_transform_);
}

std::array<float, 3> RosNode::getLatestReferenceControl() const
{
  // Lock the mutex during access.
  std::lock_guard<std::mutex> lock(reference_mutex_);
  return latest_reference_control_;
}

geometry_msgs::Quaternion RosNode::getLatestOrientationInFrame(const std::string& target_frame = "base_link")
{
  std::lock_guard<std::mutex> lock(odometry_mutex_);
  // Create a PoseStamped message with the latest orientation
  geometry_msgs::PoseStamped pose_in;
  pose_in.header.stamp = ros::Time(0); // Use latest available transform
  pose_in.header.frame_id = odometry_frame_id_; // The frame of the odometry data
  pose_in.pose.orientation = latest_orientation_;
  pose_in.pose.position.x = 0;
  pose_in.pose.position.y = 0;
  pose_in.pose.position.z = 0;

  // Transform to the target frame
  geometry_msgs::PoseStamped pose_out;
  tf_buffer_.transform(pose_in, pose_out, target_frame, ros::Duration(1.0));

  // Get the quaternion from the pose
  geometry_msgs::Quaternion quaternion = pose_out.pose.orientation;
  return quaternion;
}

geometry_msgs::Vector3 RosNode::getLatestAngularVelocityInFrame(const std::string& target_frame = "base_link")
{
  std::lock_guard<std::mutex> lock(odometry_mutex_);
  // Create a Vector3Stamped message with the latest angular velocity
  geometry_msgs::Vector3Stamped vel_in;
  vel_in.header.stamp = ros::Time(0); // Use latest available transform
  vel_in.header.frame_id = odometry_frame_id_; // The frame of the odometry data
  vel_in.vector = latest_angular_velocity_;

  // Transform to the target frame
  geometry_msgs::Vector3Stamped vel_out;
  tf_buffer_.transform(vel_in, vel_out, target_frame, ros::Duration(0.0));
  return vel_out.vector;
}

// New method: lookup the transform between "base_link" and "camera0"
// which gives the camera's offset with respect to the base.
geometry_msgs::TransformStamped RosNode::getCameraOffset()
{
  return tf_buffer_.lookupTransform("base_link", "camera0", ros::Time(0), ros::Duration(1.0));
}

geometry_msgs::Quaternion RosNode::getLevelWithHeadingOrientation() const
{
  std::lock_guard<std::mutex> lock(odometry_mutex_);
  
  // Convert ROS quaternion to Eigen quaternion
  Eigen::Quaterniond eigen_quat(
      latest_orientation_.w,
      latest_orientation_.x,
      latest_orientation_.y,
      latest_orientation_.z);
  
  // Get the rotation matrix
  Eigen::Matrix3d rot_matrix = eigen_quat.toRotationMatrix();
  
  // Extract the world-aligned X axis (which corresponds to the projection of
  // the body's X axis onto the world XY plane)
  Eigen::Vector3d x_projected = rot_matrix.col(0);
  x_projected.z() = 0;  // Project onto XY plane
  
  // Ensure we have a valid vector (if vehicle is pointing straight up/down)
  if (x_projected.norm() < 1e-6) {
    // Default to original forward direction if projection is too small
    x_projected = Eigen::Vector3d::UnitX();
  } else {
    x_projected.normalize();
  }
  
  // Create a new orthonormal basis where:
  // - X is the projected vector (preserving heading)
  // - Z is straight up (0,0,1) to ensure level orientation
  // - Y completes the right-handed coordinate system
  Eigen::Vector3d z_world(0, 0, 1);
  Eigen::Vector3d y_level = z_world.cross(x_projected).normalized();
  
  // Create a rotation matrix from these orthogonal vectors
  Eigen::Matrix3d level_rot;
  level_rot.col(0) = x_projected;
  level_rot.col(1) = y_level;
  level_rot.col(2) = z_world;
  
  // Convert to quaternion
  Eigen::Quaterniond level_quat(level_rot);
  
  // Convert back to geometry_msgs::Quaternion
  geometry_msgs::Quaternion result;
  result.w = level_quat.w();
  result.x = level_quat.x();
  result.y = level_quat.y();
  result.z = level_quat.z();
  
  return result;
}

geometry_msgs::Quaternion RosNode::getBodyToLevelWithHeadingTransform() const
{
  // Get level-with-heading quaternion
  geometry_msgs::Quaternion level_heading_msg = getLevelWithHeadingOrientation();
  
  // Declare variables outside of the lock scopes so they remain accessible
  Eigen::Quaterniond body_quat;
  Eigen::Quaterniond level_heading_quat;
  
  // Fill the body quaternion
  {
    std::lock_guard<std::mutex> lock(odometry_mutex_);
    body_quat = Eigen::Quaterniond(
      latest_orientation_.w,
      latest_orientation_.x,
      latest_orientation_.y,
      latest_orientation_.z);
  
  
    // Fill the level heading quaternion
    level_heading_quat = Eigen::Quaterniond(
      level_heading_msg.w,
      level_heading_msg.x,
      level_heading_msg.y,
      level_heading_msg.z);

  }
  
  // Compute the relative rotation: body_quat^(-1) * level_heading_quat
  Eigen::Quaterniond relative_quat = body_quat.inverse() * level_heading_quat;
  
  // Convert back to geometry_msgs::Quaternion
  geometry_msgs::Quaternion result;
  result.w = relative_quat.w();
  result.x = relative_quat.x();
  result.y = relative_quat.y();
  result.z = relative_quat.z();
  
  return result;
}

// Getter for the cached transform
geometry_msgs::TransformStamped RosNode::getBodyToLevelWithHeadingTransformStamped() const
{
  std::lock_guard<std::mutex> lock(transform_mutex_);
  return body_to_level_heading_transform_;
}
// Simplified method that always transforms to camera frame
geometry_msgs::Vector3 RosNode::getAngularVelocityInCameraFrame()
{
    // Check if we need to initialize the transform
    if (!cached_transform_valid_) {
        // This transform should be constant, so we only need to look it up once
        geometry_msgs::TransformStamped transform_stamped = 
            tf_buffer_.lookupTransform(camera_frame_id_, odometry_frame_id_, 
                                      ros::Time(0), ros::Duration(1.0));
        
        // Convert to Eigen for faster computation
        cached_imu_to_camera_transform_.translation() = Eigen::Vector3d(
            transform_stamped.transform.translation.x,
            transform_stamped.transform.translation.y,
            transform_stamped.transform.translation.z);
        
        cached_imu_to_camera_transform_.linear() = Eigen::Quaterniond(
            transform_stamped.transform.rotation.w,
            transform_stamped.transform.rotation.x,
            transform_stamped.transform.rotation.y,
            transform_stamped.transform.rotation.z).toRotationMatrix();
        
        cached_transform_valid_ = true;
        ROS_INFO("Successfully cached IMU to camera transform");
    }
    
    // Apply the cached transform to the angular velocity
    std::lock_guard<std::mutex> lock(odometry_mutex_);
    
    // Get the angular velocity from IMU frame
    Eigen::Vector3d ang_vel(
        latest_angular_velocity_.x,
        latest_angular_velocity_.y,
        latest_angular_velocity_.z);
    
    // Rotate the angular velocity (translation doesn't affect angular velocity)
    Eigen::Vector3d transformed_vel = cached_imu_to_camera_transform_.rotation() * ang_vel;
    
    // Convert back to geometry_msgs::Vector3
    geometry_msgs::Vector3 result;
    result.x = transformed_vel.x();
    result.y = transformed_vel.y();
    result.z = transformed_vel.z();
    return result;
}

void RosNode::publishPsiImage(const cv::Mat &image, std::chrono::steady_clock::time_point timestamp, 
                             float min_negative, float max_positive)
{

  // Create matrices for positive and negative values
  cv::Mat pos_mask, neg_mask;
  cv::compare(image, 0, pos_mask, cv::CMP_GT);  // Mask where values > 0
  cv::compare(image, 0, neg_mask, cv::CMP_LT);  // Mask where values < 0

  // Create normalized values for positive (green) and negative (red) parts
  cv::Mat pos_values = cv::Mat::zeros(image.size(), image.type());
  cv::Mat neg_values = cv::Mat::zeros(image.size(), image.type());
  
  // Normalize positive values (0 to max_positive maps to 0-255)
  image.copyTo(pos_values, pos_mask);
  pos_values = cv::abs(pos_values) * (255.0 / max_positive);
  
  // Normalize negative values (min_negative to 0 maps to 255-0)
  image.copyTo(neg_values, neg_mask);
  neg_values = cv::abs(neg_values) * (255.0 / std::abs(min_negative));  // Take absolute value of min_negative
  
  // Clip values to 0-255 range
  cv::min(pos_values, 255.0, pos_values);
  cv::min(neg_values, 255.0, neg_values);
  
  // Convert to 8-bit
  cv::Mat pos_values_8u, neg_values_8u;
  pos_values.convertTo(pos_values_8u, CV_8UC1);
  neg_values.convertTo(neg_values_8u, CV_8UC1);
  
  // Create the three color channels (BGR)
  std::vector<cv::Mat> channels(3);
  channels[0] = cv::Mat::zeros(image.size(), CV_8UC1);  // Blue = 0
  channels[1] = pos_values_8u;                          // Green = positive values
  channels[2] = neg_values_8u;                          // Red = negative values
  
  // Merge channels into a color image
  cv::Mat color_image;
  cv::merge(channels, color_image);
  
  // Convert to ROS image message
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color_image).toImageMsg();
  msg->header.stamp = ros::Time::now();
  psi_image_pub_.publish(msg);
}

void RosNode::publishHImage(const cv::Mat &image, std::chrono::steady_clock::time_point timestamp, 
                           float min_negative, float max_positive)
{
  // Create matrices for positive and negative values
  cv::Mat pos_mask, neg_mask;
  cv::compare(image, 0, pos_mask, cv::CMP_GT);  // Mask where values > 0
  cv::compare(image, 0, neg_mask, cv::CMP_LT);  // Mask where values < 0
  
  // Create normalized values for positive (green) and negative (red) parts
  cv::Mat pos_values = cv::Mat::zeros(image.size(), image.type());
  cv::Mat neg_values = cv::Mat::zeros(image.size(), image.type());
  
  // Normalize positive values (0 to max_positive maps to 0-255)
  image.copyTo(pos_values, pos_mask);
  pos_values = cv::abs(pos_values) * (255.0 / max_positive);
  
  // Normalize negative values (min_negative to 0 maps to 255-0)
  image.copyTo(neg_values, neg_mask);
  neg_values = cv::abs(neg_values) * (255.0 / std::abs(min_negative));  // Take absolute value of min_negative
  
  // Clip values to 0-255 range
  cv::min(pos_values, 255.0, pos_values);
  cv::min(neg_values, 255.0, neg_values);
  
  // Convert to 8-bit
  cv::Mat pos_values_8u, neg_values_8u;
  pos_values.convertTo(pos_values_8u, CV_8UC1);
  neg_values.convertTo(neg_values_8u, CV_8UC1);
  
  // Create the three color channels (BGR)
  std::vector<cv::Mat> channels(3);
  channels[0] = cv::Mat::zeros(image.size(), CV_8UC1);  // Blue = 0
  channels[1] = pos_values_8u;                          // Green = positive values
  channels[2] = neg_values_8u;                          // Red = negative values
  
  // Merge channels into a color image
  cv::Mat color_image;
  cv::merge(channels, color_image);
  
  // Convert to ROS image message
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color_image).toImageMsg();
  msg->header.stamp = ros::Time::now();
  h_image_pub_.publish(msg);
}

void RosNode::publishRawPsiImage(const cv::Mat &image, std::chrono::steady_clock::time_point timestamp)
{
  // Clone the image to ensure we're not modifying the original
  cv::Mat raw_image = image.clone();
  
  // Convert to ROS image message - use 32FC1 to maintain floating point precision
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", raw_image).toImageMsg();
  msg->header.stamp = ros::Time::now();
  raw_psi_image_pub_.publish(msg);
}

void RosNode::publishRawHImage(const cv::Mat &image, std::chrono::steady_clock::time_point timestamp)
{
  // Clone the image to ensure we're not modifying the original
  cv::Mat raw_image = image.clone();
  
  // Convert to ROS image message - use 32FC1 to maintain floating point precision
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", raw_image).toImageMsg();
  msg->header.stamp = ros::Time::now();
  raw_h_image_pub_.publish(msg);
}

}  // namespace my_ros_package