#ifndef PI_CONTROLLER_NODE_PI_CONTROLLER_NODE_H
#define PI_CONTROLLER_NODE_PI_CONTROLLER_NODE_H

#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/TwistStamped.h>
#include <thread>
#include <mutex>
#include <array>

/**
 * @brief A PI controller that runs in its own thread.
 *
 * The controller subscribes to a control reference (as a Twist message) and to a vehicle's twist message.
 * The velocity measurement is extracted from the z component of msg.twist.linear.
 */
class PIController {
public:
    /**
     * @brief Constructor: initializes the controller with the given NodeHandle.
     *
     * @param nh A reference to the ROS NodeHandle.
     * @param kp Proportional gain.
     * @param ki Integral gain.
     * @param dt Integration period in seconds.
     */
    PIController(ros::NodeHandle& nh, float kp, float ki, float dt);

    /**
     * @brief Destructor: stops the control thread.
     */
    ~PIController();

    /**
     * @brief Returns the computed control output (thread-safe).
     *
     * @return A 3-element array representing the control output.
     */
    std::array<float, 3> getControlOutput();

private:
    /**
     * @brief Callback for updating the control reference.
     *
     * Now the control reference is provided as a TwistStamped message.
     * In this example, we extract the desired velocity from the twist's linear.z component.
     *
     * @param msg Pointer to the received geometry_msgs::TwistStamped message.
     */
    void controlReferenceCallback(const geometry_msgs::TwistStamped::ConstPtr &msg);

    /**
     * @brief Callback for updating the velocity measurement.
     *
     * This callback extracts the velocity measurement from the z component
     * of the twist message (msg->twist.linear.z).
     *
     * @param msg Pointer to the received geometry_msgs::TwistStamped message.
     */
    void measurementCallback(const geometry_msgs::TwistStamped::ConstPtr &msg);

    /**
     * @brief Internal control loop running in a separate thread.
     *
     * This loop calculates the error, integrates it, and computes the control output.
     */
    void controlLoop();

    // ROS node handle and subscribers.
    ros::NodeHandle nh_;
    ros::Subscriber control_reference_sub_;
    ros::Subscriber measurement_sub_;
    
    // Thread for the control loop and mutex for thread-safe data access.
    std::thread control_thread_;
    std::mutex mutex_;
    
    // Controller gains and time step for integration.
    float kp_;
    float ki_;
    float dt_;
    
    // Internal states: control reference, measurement, integrated error, and control output.
    // Here, for consistency with the original design, we use a 3-element array.
    std::array<float, 3> control_reference_;
    float velocity_measurement_;
    float z_error_integral_;
    std::array<float, 3> control_output_;

    // Control flag for stopping the control thread.
    bool running_;
};

#endif // PI_CONTROLLER_NODE_PI_CONTROLLER_NODE_H