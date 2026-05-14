#include "scene_flow_msgs/SceneFlow.h"
#include "my_ros_package/ros_nodes.hpp"
#include "my_extras.h"
#include <ros/ros.h>
#include <thread>
#include <iostream>
#include <memory>  // For std::shared_ptr and std::make_shared
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include "cbf_calc.h"
#include "cbf_solver.h"
#include "rs_reader_threaded.h"
#include "rotation_util.hpp"
#include "vector_median.h"
#include "pi_controller_node/pi_controller_node.h"
#include <Eigen/Dense>
#include "scene_transform.h"
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include "magnitude_filter.h"
#include <yaml-cpp/yaml.h>
#include <fstream>

// Preserves roll and pitch but zeroes out yaw
Eigen::Quaternionf getRollPitchOnlyQuaternion(const Eigen::Quaternionf &fullQuat) 
{
    // Get the "forward" direction of the vehicle in world coordinates
    Eigen::Vector3f vehicleForward = fullQuat * Eigen::Vector3f::UnitX();
    
    // Project this vector onto the XY plane to get the yaw component
    Eigen::Vector3f yawVector = vehicleForward;
    yawVector.z() = 0;
    
    // If the projection is too small, use global forward direction
    if (yawVector.norm() < 1e-6f) {
        yawVector = Eigen::Vector3f::UnitX();
    } else {
        yawVector.normalize();
    }
    
    // Find the rotation from the projected forward to global forward (X-axis)
    // This is the *inverse* of the yaw rotation
    Eigen::Quaternionf yawCorrection = Eigen::Quaternionf::FromTwoVectors(
        yawVector, Eigen::Vector3f::UnitX());
    
    // Apply this correction to the original quaternion to remove the yaw component
    // This leaves only roll and pitch
    return yawCorrection * fullQuat;
}

// Function to apply median filter to a single Eigen matrix
Eigen::MatrixXf applyMedianFilter(const Eigen::MatrixXf& input, int kernel_size) {
    // Convert Eigen matrix to OpenCV Mat
    cv::Mat cv_input;
    cv::eigen2cv(input, cv_input);
    
    // Get min and max values
    double min_val, max_val;
    cv::minMaxLoc(cv_input, &min_val, &max_val);
    
    // Convert to 8-bit for median filtering
    cv::Mat input_uint8;
    double scale = (max_val > min_val) ? (255.0 / (max_val - min_val)) : 1.0;
    double offset = -min_val * scale;
    cv_input.convertTo(input_uint8, CV_8UC1, scale, offset);
    
    // Apply median filter
    cv::Mat filtered_uint8;
    cv::medianBlur(input_uint8, filtered_uint8, kernel_size);
    
    // Convert back to original scale
    cv::Mat filtered_float;
    filtered_uint8.convertTo(filtered_float, CV_32FC1, (max_val - min_val) / 255.0, min_val);
    
    // Convert back to Eigen matrix
    Eigen::MatrixXf output;
    cv::cv2eigen(filtered_float, output);
    
    return output;
}

// Function to load parameters from a YAML configuration file
bool loadConfigurationFile(const std::string& config_file_path,
                          float& min_depth, float& max_depth, float& max_velocity,
                          float& radius, float& alpha1, float& alpha2, float& k, float& u_filter_alpha) {
    try {
        YAML::Node config = YAML::LoadFile(config_file_path);
        
        // Load parameters with defaults if not found
        min_depth = config["cbf"]["min_depth"].as<float>(0.03f);
        max_depth = config["cbf"]["max_depth"].as<float>(3.5f);
        max_velocity = config["cbf"]["max_velocity"].as<float>(15.0f);
        radius = config["cbf"]["radius"].as<float>(1.0f);
        alpha1 = config["cbf"]["alpha1"].as<float>(1.0f);
        alpha2 = config["cbf"]["alpha2"].as<float>(1.0f);
        k = config["cbf"]["k"].as<float>(20.0f);
        u_filter_alpha = config["cbf"]["u_filter_alpha"].as<float>(0.80f);
        
        // Print loaded parameters
        std::cout << "Loaded configuration from: " << config_file_path << std::endl;
        std::cout << "min_depth: " << min_depth << ", max_depth: " << max_depth 
                  << ", max_velocity: " << max_velocity << std::endl;
        std::cout << "radius: " << radius << ", alpha1: " << alpha1 
                  << ", alpha2: " << alpha2 << ", k: " << k << std::endl;
        std::cout << "u_filter_alpha: " << u_filter_alpha << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading configuration file: " << e.what() << std::endl;
        std::cerr << "Using default parameters instead." << std::endl;
        return false;
    }
}

int main(int argc, char *argv[]) {

    // Initialize parameters with default values
    float min_depth = 0.03f;
    float max_depth = 3.5f;
    float max_velocity = 15.0f; // m/s - per channel
    float radius = 1.0f;
    float alpha1 = 1.0f;
    float alpha2 = 1.0f;
    float k = 20.0f; // The softmin parameter
    float u_filter_alpha = 0.80f;
    
    // Try to load configuration from file
    std::string config_path = "config/cbf_params.yaml";

    // Allow overriding config path via command line
    if (argc > 1) {
        config_path = argv[1];
    }
    loadConfigurationFile(config_path, min_depth, max_depth, max_velocity, radius, alpha1, alpha2, k, u_filter_alpha);

    //MagnitudeFilter magnitude_filter(u_filter_alpha);
    StabilizedDirectionFilter stabilized_filter(u_filter_alpha);

    //lets_test_realsense();
    unsigned int solution_rows = 120;
    unsigned int solution_cols = static_cast<unsigned int>(640 * solution_rows / 480.0f);

    CbfCalc cbf_calc(radius, alpha1, solution_rows * solution_cols);

    // Initialize ROS (using ros::init)
    ros::init(argc, argv, "my_app");
    ros::NodeHandle nh;
    my_ros_package::RosNode node(nh);

    // Create the image preparer
    std::shared_ptr<RSReader> rs_reader = std::make_shared<RSReader>(node);
    rs_reader->waitForFirstImages();
    std::shared_ptr<ImagePreparer> image_preparer = std::make_shared<RSImagePreparer>(rs_reader);

    // Create reference control node
    //std::shared_ptr<PIController> pi_controller = std::make_shared<PIController>(nh, 1.0f, 1.0f, 0.01f);

    PD_flow_opencv sceneflow(image_preparer, solution_rows, "output");

    // Start spinning in a separate thread.
    std::thread spin_thread([]() {
         ros::spin();
    });

    // Get base_link to camera0 transform
    geometry_msgs::TransformStamped camera0_to_base_link = node.getCameraOffset();

    Eigen::Affine3f camera0_to_base_link_affine = Eigen::Affine3f::Identity();
    camera0_to_base_link_affine.linear() = Eigen::Quaternionf(camera0_to_base_link.transform.rotation.w, camera0_to_base_link.transform.rotation.x, camera0_to_base_link.transform.rotation.y, camera0_to_base_link.transform.rotation.z).normalized().toRotationMatrix();
    camera0_to_base_link_affine.translation() = Eigen::Vector3f(camera0_to_base_link.transform.translation.x, camera0_to_base_link.transform.translation.y, camera0_to_base_link.transform.translation.z);

    sceneflow.initializeCUDA();

    int counter = 0;
    auto current_time = std::chrono::high_resolution_clock::now();

    // Main loop: capture frames and process them while ROS is running
    while (ros::ok()) {

        // Wait for the new image (this might block until new data is available)
        CpuImageBuffer image_buffer = image_preparer->waitForImage();

        // Get quaternions from node
        geometry_msgs::TransformStamped body_to_level_heading_transform = node.getBodyToLevelWithHeadingTransformStamped();

        // Prepare the frame for the scene flow computation
        auto time_diff = sceneflow.CaptureFrame(image_buffer);

        // Measure the scene flow computation on GPU
        sceneflow.solveSceneFlowGPU();

        // Make eigen quaternion from body_orientation_quaternion
        Eigen::Affine3f base_link_to_vehicle = Eigen::Affine3f::Identity();
        base_link_to_vehicle.linear() = Eigen::Quaternionf(body_to_level_heading_transform.transform.rotation.w, body_to_level_heading_transform.transform.rotation.x, body_to_level_heading_transform.transform.rotation.y, body_to_level_heading_transform.transform.rotation.z).normalized().toRotationMatrix();
        base_link_to_vehicle.translation() = Eigen::Vector3f(body_to_level_heading_transform.transform.translation.x, body_to_level_heading_transform.transform.translation.y, body_to_level_heading_transform.transform.translation.z);

        // Combine the two affine transformations to transform between NED and camera frame
        //Eigen::Affine3f camera_to_vehicle = base_link_to_vehicle * camera0_to_base_link_affine;
        
        // Make eigen vector from angular_velocity

        Eigen::Vector3f angular_velocity_eigen((image_buffer.angular_velocity[1].x + image_buffer.angular_velocity[0].x) / 2.0f,
                                             (image_buffer.angular_velocity[1].y + image_buffer.angular_velocity[0].y) / 2.0f,
                                             (image_buffer.angular_velocity[1].z + image_buffer.angular_velocity[0].z) / 2.0f);

        // Build the skew-symmetric matrix for angular_velocity_eigen:
        Eigen::Matrix3f omega_skew;
        omega_skew <<  0, -angular_velocity_eigen(2),  angular_velocity_eigen(1),
                       angular_velocity_eigen(2),  0, -angular_velocity_eigen(0),
                      -angular_velocity_eigen(1),  angular_velocity_eigen(0),  0;

        // Apply median filter to the scene flow
        Eigen::MatrixXf scene_flow_x = applyMedianFilter(sceneflow.dx[0], 13);
        Eigen::MatrixXf scene_flow_y = applyMedianFilter(sceneflow.dy[0], 13);
        Eigen::MatrixXf scene_flow_z = applyMedianFilter(sceneflow.dz[0], 13);

        // Transform scene flow and position into a single variable with both x, y, and z components
        // For dx, dy, dz fields each of size (120, 160)
        Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> flattened_dx(
            scene_flow_x.data(), 1, scene_flow_x.size());
        Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> flattened_dy(
            scene_flow_y.data(), 1, scene_flow_y.size());
        Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> flattened_dz(
            scene_flow_z.data(), 1, scene_flow_z.size());

        Eigen::MatrixXf velocities(3, flattened_dx.cols());
        velocities.row(0) = flattened_dx.array() / time_diff.count();
        velocities.row(1) = flattened_dy.array() / time_diff.count();
        velocities.row(2) = flattened_dz.array() / time_diff.count();


        // Create an unfiltered velocities matrix
        Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> flattened_dx_unfiltered(
            sceneflow.dx[0].data(), 1, scene_flow_x.size());
        Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> flattened_dy_unfiltered(
            sceneflow.dy[0].data(), 1, scene_flow_y.size());
        Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> flattened_dz_unfiltered(
            sceneflow.dz[0].data(), 1, scene_flow_z.size());

        Eigen::MatrixXf velocities_unfiltered(3, flattened_dx_unfiltered.cols());
        velocities_unfiltered.row(0) = flattened_dx_unfiltered.array() / time_diff.count();
        velocities_unfiltered.row(1) = flattened_dy_unfiltered.array() / time_diff.count();
        velocities_unfiltered.row(2) = flattened_dz_unfiltered.array() / time_diff.count();

        // Create position matrix
        Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> flattened_xx(
            sceneflow.xx[2].data(), 1, sceneflow.xx[2].size());
        Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> flattened_yy(
            sceneflow.yy[2].data(), 1, sceneflow.yy[2].size());
        Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> flattened_depth(
            sceneflow.depth[2].data(), 1, sceneflow.depth[2].size());        

        // Now if the depth is less than 0.01 meters we set it to max_depth
        float max_new_depth = std::max(flattened_depth.maxCoeff(), max_depth);
        Eigen::RowVectorXf adjusted_depth = (flattened_depth.array() <= min_depth)
                                        .select(max_new_depth, flattened_depth.array());

        Eigen::Matrix<float, 3, Eigen::Dynamic> positions(3, flattened_xx.cols());
        positions.row(0) = flattened_xx;
        positions.row(1) = flattened_yy;
        positions.row(2) = adjusted_depth;

        // For the velocity we only apply the rotational part
        //std::cout << "omega_skew: " << angular_velocity_eigen << std::endl;
        Eigen::MatrixXf velocities_base_link = camera0_to_base_link_affine.rotation() * (velocities + omega_skew * positions);
        Eigen::MatrixXf velocities_base_link_unfiltered = camera0_to_base_link_affine.rotation() * velocities_unfiltered;
        // Clamp the velocities to max_velocity
        velocities_base_link = velocities_base_link.cwiseMin(max_velocity).cwiseMax(-max_velocity);


        // Create a mask from the depth condition:
        Eigen::RowVectorXf depth_mask = ((flattened_depth.array() < max_depth) &&
                                        (flattened_depth.array() > min_depth)).cast<float>();

        // Broadcast mask to the velocity matrix dimensions (3 x N)
        // and apply it to the original velocities.
        velocities_base_link = velocities_base_link.array() * depth_mask.replicate(3, 1).array();

        // For the position we apply both the translational and rotational parts
        Eigen::MatrixXf positions_base_link = camera0_to_base_link_affine * positions;

        // Extract the velocity components from Eigen matrices
        Eigen::RowVectorXf dx_flat = velocities_base_link.row(0).eval();
        Eigen::RowVectorXf dy_flat = velocities_base_link.row(1).eval();
        Eigen::RowVectorXf dz_flat = velocities_base_link.row(2).eval();

        // Extract the position components from Eigen matrices
        Eigen::RowVectorXf xx_flat = positions_base_link.row(0).eval();
        Eigen::RowVectorXf yy_flat = positions_base_link.row(1).eval();
        Eigen::RowVectorXf depth_flat = positions_base_link.row(2).eval();

        // Create OpenCV matrices to pass to the CBF calculation
        cv::Mat xx_mat(static_cast<int>(sceneflow.xx[2].cols()), static_cast<int>(sceneflow.xx[2].rows()), CV_32FC1, xx_flat.data());
        cv::Mat yy_mat(static_cast<int>(sceneflow.yy[2].cols()), static_cast<int>(sceneflow.yy[2].rows()), CV_32FC1, yy_flat.data());
        cv::Mat depth_mat(static_cast<int>(sceneflow.depth[2].cols()), static_cast<int>(sceneflow.depth[2].rows()), CV_32FC1, depth_flat.data());
        cv::Mat dx_mat(static_cast<int>(sceneflow.dx[0].cols()), static_cast<int>(sceneflow.dx[0].rows()), CV_32FC1, dx_flat.data());
        cv::Mat dy_mat(static_cast<int>(sceneflow.dy[0].cols()), static_cast<int>(sceneflow.dy[0].rows()), CV_32FC1, dy_flat.data());
        cv::Mat dz_mat(static_cast<int>(sceneflow.dz[0].cols()), static_cast<int>(sceneflow.dz[0].rows()), CV_32FC1, dz_flat.data());

        // Clone and transpose for visualization and CBF calculation
        xx_mat = xx_mat.clone().t();
        yy_mat = yy_mat.clone().t();
        depth_mat = depth_mat.clone().t();
        dx_mat = dx_mat.clone().t();
        dy_mat = dy_mat.clone().t();
        dz_mat = dz_mat.clone().t();

        // Measure the CBF calculation stage
        // Use OpenCV matrices with ptr<float>(0) as expected by the function
        CbfCalcOutputs cbf_calc_outputs = 
            cbf_calc.get_Lg_Lf_and_h(xx_mat.ptr<float>(0), yy_mat.ptr<float>(0), depth_mat.ptr<float>(0),
                                     dx_mat.ptr<float>(0), dy_mat.ptr<float>(0), dz_mat.ptr<float>(0));

        // Measure the softmin calculation stage
        CbfSoftminResults cbf_softmin_results = cbf_calc.getSoftminCombined(k);

        // Calculate the safe control
        std::array<float, 3> u_ref = node.getLatestReferenceControl();
        std::array<float, 3> u_safe = cbf_solver(u_ref, cbf_softmin_results, alpha2);

        // Apply the control with filtering
        //u_safe = control_low_pass_filter.filter(u_safe);
        auto u_safe_filtered = stabilized_filter.filter(u_safe, base_link_to_vehicle);
        
        // Publish the control vector
        node.publishControlVector(u_safe_filtered, u_safe);
        
        // Publish the point cloud with closest and CBF critical points
        node.publishDepthPointCloud(depth_mat, xx_mat, yy_mat,
                                   image_buffer.depth_timestamps[0]);

        // Image publishing
        node.publishColorImage(image_buffer.raw_newest_color_image, image_buffer.gray_timestamps[0]);
        node.publishDepthImage(image_buffer.raw_newest_depth_image, image_buffer.depth_timestamps[0]);

        // Scene flow publishing
        auto generated_scene_flow = generateSceneFlowMsg(positions_base_link, velocities_unfiltered);
        node.publishSceneFlowRaw(generated_scene_flow);
        auto generated_scene_flow_processed = generateSceneFlowMsg(xx_mat, yy_mat, depth_mat, dx_mat, dy_mat, dz_mat);
        node.publishSceneFlowProcessed(generated_scene_flow_processed);
        auto sf_magnitude_image = createSceneFlowMagnitudeImage(dx_mat, dy_mat, dz_mat, 0.3f);
        node.publishSceneFlowMagnitudeImage(sf_magnitude_image, image_buffer.gray_timestamps[0]);

        // Publish CBF and first derivative invariance values
        auto psi_mat = cbf_calc.getPsiMatrix(xx_mat.rows, xx_mat.cols);
        auto h_mat = cbf_calc.getHMatrix(xx_mat.rows, xx_mat.cols);
        node.publishPsiImage(psi_mat, image_buffer.depth_timestamps[0]);
        node.publishHImage(h_mat, image_buffer.depth_timestamps[0]);
        node.publishRawPsiImage(psi_mat, image_buffer.depth_timestamps[0]);
        node.publishRawHImage(h_mat, image_buffer.depth_timestamps[0]);

        counter++;

        // Log and optionally compute frame rate every 10 iterations
        if (counter % 10 == 0) {
            auto new_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(new_time - current_time);
            double fps = 10.0 * 1000.0 / static_cast<double>(duration.count());
            std::cout << "Frame rate: " << fps << " Hz" << std::endl;
            current_time = std::chrono::high_resolution_clock::now();
        }
    }

    sceneflow.freeGPUMemory();

    // Properly join the spin thread before shutting down
    if (spin_thread.joinable()) {
        spin_thread.join();
    }
    ros::shutdown();

    return 0;
}