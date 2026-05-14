#ifndef MY_EXTRAS_H
#define MY_EXTRAS_H

#ifdef _WIN32
    #include <opencv2/core.hpp>
    #include <opencv2/highgui.hpp>
	#include <io.h>
#elif __linux
    #include <opencv2/core/core.hpp>
    #include <opencv2/highgui/highgui.hpp>
    #include <cv_bridge/cv_bridge.h>
	#include <unistd.h>
#endif

//#include <Eigen/src/Core/Matrix.h>
#include <Eigen/Dense>
#include <fstream>
#include <string.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include "my_ros_package/ros_nodes.hpp"
#include "../include/pd_flow/pdflow_cudalib.h"
#include "../src/cud/cuda_logsumexp.h"
#include <ros/ros.h>

#include "my_ros_package/ros_nodes.hpp"
#include "scene_flow_msgs/SceneFlow.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Vector3.h"
#include "npy.hpp"
#include <librealsense2/rs.hpp>
#include "cublas_v2.h"
#include "cbf_calc.h"
#include "rs_reader_threaded.h"
//#include "legend_pdflow.xpm"

#ifdef _WIN32
    #define M_PI 3.14159265f
    #define M_LOG2E 1.44269504088896340736f //log2(e)
    inline float log2(const float x){ return  log(x) * M_LOG2E; }

#elif __linux
    inline int stoi(char *c) {return int(std::strtol(c,NULL,10));}

#endif

//==================================================================
//					PD-Flow class (using openCV)
//==================================================================

cv::Mat createSceneFlowImage(const cv::Mat &temporal_dx,
                             const cv::Mat &temporal_dy,
                             const cv::Mat &temporal_dz);

void resizeAndCrop(const cv::Mat &input, cv::Mat &output, int targetWidth, int targetHeight);

cv::Mat readPFM(const std::string &filename);

// LowPassFilter control output a bit
class ControlLowPassFilter {
    private:
        std::array<float, 3> u_filtered_var = {0.0f, 0.0f, 0.0f};
        float alpha;
        std::chrono::steady_clock::time_point last_time;
    public:
        ControlLowPassFilter(float alpha);
        std::array<float, 3> filter(const std::array<float, 3>& u_ref);
};

// Direction-stabilized filter that works in leveled frame
class StabilizedDirectionFilter {
    private:
        std::array<float, 3> u_filtered_var = {0.0f, 0.0f, 0.0f};
        float alpha;
    public:
        StabilizedDirectionFilter(float alpha);
        std::array<float, 3> filter(const std::array<float, 3>& u_ref, const Eigen::Affine3f& base_link_to_vehicle);
};

cv::Mat createSceneFlowMagnitudeImage(const cv::Mat &temporal_dx,
                                      const cv::Mat &temporal_dy,
                                      const cv::Mat &temporal_dz,
                                      double max_intensity = 0.5);
        

// Class that holds a pair of images
struct ImagePair{
    public:
        unsigned int newImages;
        cv::Mat intensity1;
        cv::Mat depth1;
        cv::Mat intensity2;
        cv::Mat depth2;
        ros::Time rgb_timestamps[2];
};

struct CameraProperties{
    float fx;
    float fy;
    float cx;
    float cy;
    unsigned int res_width;
    unsigned int res_height;
};

// Base class for image preparers, declared as abstract
class ImagePreparer {
    public:
        virtual CpuImageBuffer waitForImage() = 0;
        virtual CameraProperties getCameraProperties() = 0;
        virtual ~ImagePreparer() = default; // Virtual destructor for proper cleanup.
};


class RSImagePreparer : public ImagePreparer{
    private:
        std::shared_ptr<RSReader> rs_reader;
    public:
        RSImagePreparer(std::shared_ptr<RSReader> rs_reader);
        CameraProperties getCameraProperties() override;
        CpuImageBuffer waitForImage() override;
};

class PD_flow_opencv {
public:

    unsigned int cam_mode;	// (1 - 640 x 480, 2 - 320 x 240)
    unsigned int ctf_levels;//Number of levels used in the coarse-to-fine scheme (always dividing by two)
    unsigned int num_max_iter[6];  //Number of iterations at every pyramid level (primal-dual solver)
    float g_mask[25];

    // Use a smart pointer to hold the image preparer interface:
    std::shared_ptr<ImagePreparer> image_preparer;
	
    //Matrices that store the original images
	cv::Mat intensity1;
	cv::Mat depth1;
	cv::Mat intensity2;
	cv::Mat depth2;

	//Aux pointers to copy the RGBD images to CUDA 
	float *I, *Z;

    // Instead of raw CPU pointers, hold GPU representations:
    //cv::cuda::GpuMat I_gpu; // GPU intensity image (grayscale)
    //cv::cuda::GpuMat Z_gpu; // GPU depth image

    // GPU pointers containing wanted results for use by other functions
    //float *r_x_gpu, *r_y_gpu, *r_z_gpu;
    //float *r_dot_x_gpu, *r_dot_y_gpu, *r_dot_z_gpu;

    //Motion field
	float *dxp, *dyp, *dzp;

    std::vector<Eigen::MatrixXf> colour;
    std::vector<Eigen::MatrixXf> colour_old;
    std::vector<Eigen::MatrixXf> depth;
    std::vector<Eigen::MatrixXf> depth_old;
    std::vector<Eigen::MatrixXf> xx;
    std::vector<Eigen::MatrixXf> xx_old;
    std::vector<Eigen::MatrixXf> yy;
    std::vector<Eigen::MatrixXf> yy_old;

    //Motion field
    std::vector<Eigen::MatrixXf> dx;
    std::vector<Eigen::MatrixXf> dy;
    std::vector<Eigen::MatrixXf> dz;

    //Max resolution of the coarse-to-fine scheme.
    unsigned int rows;
    unsigned int cols;

	//Resolution of the input images
	unsigned int width;
	unsigned int height;

    //Optimization Parameters
    float mu, lambda_i, lambda_d;

    std::string output_filename_root;

    //Cuda
    CSF_cuda csf_host, *csf_device;

	//Methods
    void createImagePyramidGPU();
    void solveSceneFlowGPU();
    void freeGPUMemory();
    void initializeCUDA();
	//void showImages();
    cv::Mat createImage() const;
	void showAndSaveResults();

    void initIntensityAndDepth();
    std::chrono::duration<double> CaptureFrame(CpuImageBuffer &image_buffer);
    void showResults();
    void saveResults( const cv::Mat& sf_image ) const;
    scene_flow_msgs::SceneFlow generateSceneFlowMsg(std::chrono::duration<double> time_diff);

    void transformCameraToDrone(float tx, float ty, float tz);

    PD_flow_opencv(std::shared_ptr<ImagePreparer> image_preparer,
    unsigned int solution_rows,
    const std::string& output_filename_root);
};

/**
 * Generate a SceneFlow message from scene flow data using OpenCV matrices.
 * 
 * @param xx_old_mat Matrix of X coordinates of points in the reference frame
 * @param yy_old_mat Matrix of Y coordinates of points in the reference frame
 * @param depth_old_mat Matrix of depth values of points in the reference frame
 * @param dx_mat Matrix of flow vectors in X direction
 * @param dy_mat Matrix of flow vectors in Y direction
 * @param dz_mat Matrix of flow vectors in Z direction
 * @param time_diff Duration between frames used to compute velocity
 * @return SceneFlow message containing points and flow vectors
 */
scene_flow_msgs::SceneFlow generateSceneFlowMsg(
    const cv::Mat& xx_old_mat,
    const cv::Mat& yy_old_mat,
    const cv::Mat& depth_old_mat,
    const cv::Mat& dx_mat,
    const cv::Mat& dy_mat,
    const cv::Mat& dz_mat);

/**
 * Generate a SceneFlow message from scene flow data using compact Eigen matrices.
 * 
 * @param positions 3×N matrix where each column represents a 3D position (x,y,z)
 * @param velocities 3×N matrix where each column represents a 3D velocity vector
 * @return SceneFlow message containing points and flow vectors
 */
scene_flow_msgs::SceneFlow generateSceneFlowMsg(
    const Eigen::Matrix<float, 3, Eigen::Dynamic>& positions,
    const Eigen::Matrix<float, 3, Eigen::Dynamic>& velocities);

#endif  // MY_EXTRAS_H