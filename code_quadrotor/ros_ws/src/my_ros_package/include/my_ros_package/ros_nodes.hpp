#ifndef MY_ROS_PACKAGE__ROS_NODES_HPP_
#define MY_ROS_PACKAGE__ROS_NODES_HPP_

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include "scene_flow_msgs/msg/scene_flow.hpp"
#include <mutex>
#include <memory>

// Include message_filters headers for synchronization:
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <condition_variable>

namespace my_ros_package
{

  // A container for the last two synchronized images and their timestamps.
  // We now store timestamps as rclcpp::Time for easier assignment from message stamps.
  class ImageBuffer {
  public:
    unsigned int newImages; // 0, 1 or 2

    sensor_msgs::Image rgb_buffer[2];
    sensor_msgs::Image stereo_buffer[2];

    rclcpp::Time rgb_timestamps[2];
    rclcpp::Time stereo_timestamps[2];
  };
    
  class RosNode : public rclcpp::Node {
  public:
    RosNode();

    // This method returns an ImageBuffer that holds copies of images.
    ImageBuffer get_image_buffer();

    void waitForFirstImages();

    // New function to publish scene flow messages.
    void publishSceneFlow(const scene_flow_msgs::SceneFlow &msg);

  private:
    void pushBuffer(const sensor_msgs::Image::SharedPtr rgb, const sensor_msgs::Image::SharedPtr stereo);

    // Buffers for 2 latest images
    ImageBuffer image_buffer_;

    // Callback for synchronized images.
    void combined_image_callback(
      const sensor_msgs::Image::SharedPtr rgb,
      const sensor_msgs::Image::SharedPtr stereo);

    // Mutex to protect image copies.
    std::mutex image_mutex_;

    // Mutax that locks as long as there are 0 new images
    std::condition_variable new_images_cv_;
    
    // Message filter subscribers for each topic.
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> rgb_sub_;
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> stereo_sub_;

    // Define the synchronization policy.
    using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>;

    // Synchronizer which calls our callback.
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // New publisher: for example, to publish processed images or related messages.
    rclcpp::Publisher<scene_flow_msgs::SceneFlow>::SharedPtr scene_flow_pub_;
  };

}  // namespace my_ros_package

#endif  // MY_ROS_PACKAGE__ROS_NODES_HPP_