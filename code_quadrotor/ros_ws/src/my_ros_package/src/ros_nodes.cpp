#include "my_ros_package/ros_nodes.hpp"
#include <chrono>
#include <thread>
#include <memory>
#include <algorithm>  // Ensure algorithm is included for std::min

namespace my_ros_package
{

using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>;

RosNode::RosNode() : Node("ros_image_subscriber")
{
  // Initialize message filter subscribers for RGB and stereo topics.
  // Note: We pass 'this' (our node) so that message_filters can create the subscriptions internally.
  //rgb_sub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::Image>>(this, "/oak/rgb/image_raw");
  //stereo_sub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::Image>>(this, "/oak/stereo/image_raw");
  rgb_sub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::Image>>(this, "/camera/rgb/image_color");
  stereo_sub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::Image>>(this, "/camera/depth/image");

  // Create the synchronizer with a queue size of 3.
  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
    SyncPolicy(10), *rgb_sub_, *stereo_sub_);
  
  // Set maximum time difference to 1 millisecond (1e6 nanoseconds)
  sync_->setMaxIntervalDuration(rclcpp::Duration::from_nanoseconds(1000000));
  
  // Register the combined callback that will be called when a matching pair of messages is found.
  sync_->registerCallback(&RosNode::combined_image_callback, this);

  // Initialize the publisher on (for example) "/processed/image"
  scene_flow_pub_ = this->create_publisher<scene_flow_msgs::SceneFlow>("/scene_flow", 10);
  // (Optional) Use a timer or integrate in your callback to publish messages.
}

void RosNode::pushBuffer(const sensor_msgs::Image::SharedPtr rgb, const sensor_msgs::Image::SharedPtr stereo) {
  {
    std::lock_guard<std::mutex> lock(image_mutex_);

    // Increase the number of new images, making sure both arguments are unsigned.
    image_buffer_.newImages = std::min(image_buffer_.newImages + 1U, 2U);

    // Shift the history: assign the previous image (slot [1]) into slot [0]
    image_buffer_.rgb_buffer[0] = image_buffer_.rgb_buffer[1];
    image_buffer_.stereo_buffer[0] = image_buffer_.stereo_buffer[1];

    // Store the new images by copying the contents from the shared pointer.
    image_buffer_.rgb_buffer[1] = *rgb;  
    image_buffer_.stereo_buffer[1] = *stereo;

    // Shift the timestamps similarly and update with the new timestamp.
    image_buffer_.rgb_timestamps[0] = image_buffer_.rgb_timestamps[1];
    image_buffer_.rgb_timestamps[1] = rclcpp::Time(rgb->header.stamp);
    image_buffer_.stereo_timestamps[0] = image_buffer_.stereo_timestamps[1];
    image_buffer_.stereo_timestamps[1] = rclcpp::Time(stereo->header.stamp);
  }
  // Notify waiting threads that new images are available.
  new_images_cv_.notify_one();
}

void RosNode::combined_image_callback(
  const sensor_msgs::Image::SharedPtr rgb,
  const sensor_msgs::Image::SharedPtr stereo)
{  
  // Log the image timestamps
  RCLCPP_INFO(this->get_logger(), 
              "Image timestamps - RGB: %u.%u, Stereo: %u.%u", 
              rgb->header.stamp.sec, rgb->header.stamp.nanosec,
              stereo->header.stamp.sec, stereo->header.stamp.nanosec);

  pushBuffer(rgb, stereo);
}
void RosNode::waitForFirstImages() {
  RCLCPP_INFO(this->get_logger(), "Waiting for the first synchronized images...");
  // Loop until both images are received.
  while (rclcpp::ok())
  {
    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      // Check if we have received both images (by verifying data is non-empty)
      if (image_buffer_.rgb_buffer[0].data.size() > 0 && 
          image_buffer_.stereo_buffer[0].data.size() > 0)
      {
        RCLCPP_INFO(this->get_logger(), "First synchronized images received!");
        return;
      }
    }
    // Process any pending callbacks.
    rclcpp::spin_some(std::dynamic_pointer_cast<rclcpp::Node>(shared_from_this()));
    rclcpp::sleep_for(std::chrono::milliseconds(100));
  }
}
ImageBuffer RosNode::get_image_buffer() {
  
  // Use a unique_lock for condition variable wait.
  std::unique_lock<std::mutex> lock(image_mutex_);

  // Wait until new images are available.
  new_images_cv_.wait(lock, [this]() { return image_buffer_.newImages > 0; });

  ImageBuffer image_buffer = image_buffer_;
  image_buffer_.newImages = 0;

  return image_buffer;
}

void RosNode::publishSceneFlow(const scene_flow_msgs::SceneFlow &msg) {
  scene_flow_pub_->publish(msg);
}

}  // namespace my_ros_package