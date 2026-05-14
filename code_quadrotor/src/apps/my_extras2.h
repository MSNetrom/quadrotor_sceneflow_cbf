/*****************************************************************************
**				Primal-Dual Scene Flow for RGB-D cameras					**
**				----------------------------------------					**
**																			**
**	Copyright(c) 2015, Mariano Jaimez Tarifa, University of Malaga			**
**	Copyright(c) 2015, Mohamed Souiai, Technical University of Munich		**
**	Copyright(c) 2015, MAPIR group, University of Malaga					**
**	Copyright(c) 2015, Computer Vision group, Tech. University of Munich	**
**																			**
**  This program is free software: you can redistribute it and/or modify	**
**  it under the terms of the GNU General Public License (version 3) as		**
**	published by the Free Software Foundation.								**
**																			**
**  This program is distributed in the hope that it will be useful, but		**
**	WITHOUT ANY WARRANTY; without even the implied warranty of				**
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the			**
**  GNU General Public License for more details.							**
**																			**
**  You should have received a copy of the GNU General Public License		**
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.	**
**																			**
*****************************************************************************/


#include "../include/pd_flow/pdflow_cudalib.h"
#include <stdio.h>
#include <string.h>
#include <Eigen/Dense>

#include "my_ros_package/ros_nodes.hpp"
#include "scene_flow_msgs/msg/scene_flow.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/vector3.hpp"

//#ifdef _WIN32
//    inline float log2(const float x){ return  log(x) * M_LOG2E;}

//#elif __linux
//    inline int stoi(char *c) {return int(std::strtol(c,NULL,10));}

//#endif

using namespace std;
using Eigen::MatrixXf;

cv::Mat createTemporalSceneFlowImage(const cv::Mat &temporal_dx, 
                                       const cv::Mat &temporal_dy, 
                                       const cv::Mat &temporal_dz);

void resizeAndCrop(const cv::Mat &input, cv::Mat &output, int targetWidth, int targetHeight);


class PD_flow_2 {
public:

    float fps;              //In Hz
    unsigned int cam_mode;	// (1 - 640 x 480, 2 - 320 x 240)
    unsigned int ctf_levels;//Number of levels used in the coarse-to-fine scheme (always dividing by two)
    unsigned int num_max_iter[6];  //Max number of iterations distributed homogeneously between all levels
    float g_mask[25];

    std::shared_ptr<my_ros_package::RosNode> ros_node;

    //Matrices that store the original images
	cv::Mat intensity1;
	cv::Mat depth1;
	cv::Mat intensity2;
	cv::Mat depth2;
	
    //Matrices that store the original images with the image resolution
    MatrixXf colour_wf;
    MatrixXf depth_wf;

    //Matrices that store the images downsampled
    vector<MatrixXf> colour;
    vector<MatrixXf> colour_old;
    vector<MatrixXf> depth;
    vector<MatrixXf> depth_old;
    vector<MatrixXf> xx;
    vector<MatrixXf> xx_old;
    vector<MatrixXf> yy;
    vector<MatrixXf> yy_old;

    //Motion field
    vector<MatrixXf> dx;
    vector<MatrixXf> dy;
    vector<MatrixXf> dz;

    //Camera properties
    float fovh;     //In radians
    float fovv;     //In radians

    //Max resolution of the coarse-to-fine scheme.
    unsigned int rows;
    unsigned int cols;

    //Optimization Parameters
    float mu, lambda_i, lambda_d;
    //openni::VideoStream 	rgb,dimage;

    //Cuda
    CSF_cuda csf_host, *csf_device;

	//Methods
    void createImagePyramidGPU();
    void solveSceneFlowGPU();
	//bool OpenCamera();
	void CloseCamera();
	ros::Duration CaptureFrame();
    void freeGPUMemory();
    void initializeCUDA();
	//void initializeScene();
	//void updateScene();
	void initializePDFlow();

    PD_flow_2(unsigned int cam_mode_config, unsigned int fps_config, unsigned int rows_config);
};


