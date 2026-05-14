#include "pi_controller_node/pi_controller_node.h"

PIController::PIController(ros::NodeHandle& nh, float kp, float ki, float dt)
    : nh_(nh), kp_(kp), ki_(ki), dt_(dt),
      control_reference_{0.0f, 0.0f, 0.0f}, velocity_measurement_(0.0f),
      z_error_integral_(0.0f), control_output_{0.0f, 0.0f, 0.0f}, running_(true)
{
    nh_.param("kp", kp_, kp);
    nh_.param("ki", ki_, ki);
    nh_.param("integration_dt", dt_, dt);

    control_reference_sub_ = nh_.subscribe("control_reference", 10, &PIController::controlReferenceCallback, this);
    measurement_sub_ = nh_.subscribe("/rig_node/base_link/twist", 10, &PIController::measurementCallback, this);

    control_thread_ = std::thread(&PIController::controlLoop, this);
}

PIController::~PIController() {
    running_ = false;
    if (control_thread_.joinable()) {
        control_thread_.join();
    }
}

std::array<float, 3> PIController::getControlOutput() {
    std::lock_guard<std::mutex> lock(mutex_);
    return control_output_;
}

void PIController::controlReferenceCallback(const geometry_msgs::TwistStamped::ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    control_reference_[0] = msg->twist.linear.z;
    control_reference_[1] = 0.0f;
    control_reference_[2] = 0.0f;
}

void PIController::measurementCallback(const geometry_msgs::TwistStamped::ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    velocity_measurement_ = msg->twist.linear.z;
}

void PIController::controlLoop() {
    ros::Rate rate(1.0f / dt_);
    while (ros::ok() && running_) {
        float error;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            error = control_reference_[0] - velocity_measurement_;
            z_error_integral_ += error * dt_;
            control_output_[0] = kp_ * error + ki_ * z_error_integral_;
        }
        rate.sleep();
    }
}

