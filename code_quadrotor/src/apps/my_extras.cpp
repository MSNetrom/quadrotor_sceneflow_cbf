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

#include "my_extras.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cctype>
#include <locale>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "rs_reader_threaded.h"
// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
        [](unsigned char ch) { return !std::isspace(ch); }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
        [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

// Read a PFM file and return a cv::Mat of type CV_32FC1 (or CV_32FC3 if color)
cv::Mat readPFM(const std::string &filename)
{
    std::ifstream infile(filename, std::ios::binary);
    if (!infile)
        throw std::runtime_error("Could not open file " + filename);

    std::string header;
    std::getline(infile, header);
    trim(header);  // Remove trailing/leading whitespace

    bool color;
    if (header == "PF")
        color = true;
    else if (header == "Pf")
        color = false;
    else
        throw std::runtime_error("Not a PFM file: " + filename);

    // Skip possible comment lines (lines starting with '#')
    std::string line;
    do {
        std::getline(infile, line);
    } while (!line.empty() && line[0]=='#');

    int width, height;
    {
        std::istringstream dims(line);
        dims >> width >> height;
        if (!dims)
            throw std::runtime_error("Malformed PFM header in file " + filename);
    }

    // Read scale factor and determine endianness.
    std::getline(infile, line);
    float scale;
    {
        std::istringstream sscale(line);
        sscale >> scale;
        if (!sscale)
            throw std::runtime_error("Invalid scale in PFM header in file " + filename);
    }
    bool fileLittleEndian = (scale < 0);
    if (fileLittleEndian)
        scale = -scale;  // use positive scale factor from here on

    int numChannels = color ? 3 : 1;
    size_t numFloats = width * height * numChannels;
    std::vector<float> data(numFloats);

    infile.read(reinterpret_cast<char*>(data.data()), numFloats * sizeof(float));
    if (!infile)
        throw std::runtime_error("PFM file ended prematurely: " + filename);

    // Determine the system's endianness.
    auto systemIsLittleEndian = []() -> bool {
        int num = 1;
        return *(char*)&num == 1;
    };

    if (systemIsLittleEndian() != fileLittleEndian) {
        // Swap bytes for each float.
        for (size_t i = 0; i < numFloats; i++) {
            char *floatBytes = reinterpret_cast<char*>(&data[i]);
            std::swap(floatBytes[0], floatBytes[3]);
            std::swap(floatBytes[1], floatBytes[2]);
        }
    }

    int type = color ? CV_32FC3 : CV_32FC1;
    // Create a cv::Mat header using the data buffer (note that PFM stores the image upside down).
    cv::Mat mat(height, width, type, data.data());
    cv::Mat matFlipped;
    cv::flip(mat, matFlipped, 0); // flip vertically
    return matFlipped.clone();   // clone to ensure data ownership
}

bool  fileExists(const std::string& path)
{
    return 0 == access(path.c_str(), 0x00 ); // 0x00 = Check for existence only!
}

void resizeAndCrop(const cv::Mat &input, cv::Mat &output, int targetWidth, int targetHeight) {
    // Calculate the aspect ratios
    double inputAspect  = static_cast<double>(input.cols) / input.rows;
    double targetAspect = static_cast<double>(targetWidth) / targetHeight;

    cv::Mat resized;
    if (inputAspect > targetAspect) {
        // Input is wider than target: scale by height and then crop width.
        int newHeight = targetHeight;
        int newWidth  = static_cast<int>(inputAspect * newHeight);
        cv::resize(input, resized, cv::Size(newWidth, newHeight));
        // Crop the width to targetWidth by centering the crop
        int x = (newWidth - targetWidth) / 2;
        output = resized(cv::Rect(x, 0, targetWidth, targetHeight)).clone();
    } else {
        // Input is taller than target: scale by width and then crop height.
        int newWidth  = targetWidth;
        int newHeight = static_cast<int>(newWidth / inputAspect);
        cv::resize(input, resized, cv::Size(newWidth, newHeight));
        // Crop the height to targetHeight by centering the crop
        int y = (newHeight - targetHeight) / 2;
        output = resized(cv::Rect(0, y, targetWidth, targetHeight)).clone();
    }
}


ControlLowPassFilter::ControlLowPassFilter(float alpha) : alpha(alpha) {}

std::array<float, 3> ControlLowPassFilter::filter(const std::array<float, 3>& u_ref) {

    // Update the filtered control output
    u_filtered_var[0] = u_filtered_var[0] + alpha * (u_ref[0] - u_filtered_var[0]);
    u_filtered_var[1] = u_filtered_var[1] + alpha * (u_ref[1] - u_filtered_var[1]);
    u_filtered_var[2] = u_filtered_var[2] + alpha * (u_ref[2] - u_filtered_var[2]);

    return u_filtered_var;
}

RSImagePreparer::RSImagePreparer(std::shared_ptr<RSReader> rs_reader)
    : rs_reader(rs_reader)
{
    // Get camera intrinsics
    std::tuple<float, float, float, float, unsigned int, unsigned int> intrinsics = rs_reader->getCameraIntrinsics();
}

CameraProperties RSImagePreparer::getCameraProperties(){
    auto intrinsics = rs_reader->getCameraIntrinsics();
    return CameraProperties{
        .fx = std::get<0>(intrinsics),
        .fy = std::get<1>(intrinsics),
        .cx = std::get<2>(intrinsics),
        .cy = std::get<3>(intrinsics),
        .res_width = std::get<4>(intrinsics),
        .res_height = std::get<5>(intrinsics)
    };
}

CpuImageBuffer RSImagePreparer::waitForImage(){
    return rs_reader->getCpuImageBuffer();
}

/*my_ros_package::GpuImageBuffer RosImagePreparer::waitForImage(){
    return ros_node->getGpuImageBuffer();
}*/


PD_flow_opencv::PD_flow_opencv(std::shared_ptr<ImagePreparer> image_preparer,
    unsigned int solution_rows,
    const std::string& output_filename_root) : image_preparer(image_preparer), output_filename_root(output_filename_root)
{
    width = image_preparer->getCameraProperties().res_width;
    height = image_preparer->getCameraProperties().res_height;

    rows = solution_rows;      //Maximum size of the coarse-to-fine scheme - Default 240 (QVGA)
    cols = rows*image_preparer->getCameraProperties().res_width/image_preparer->getCameraProperties().res_height;
    ctf_levels = static_cast<unsigned int>(log2(float(rows/15))) + 1;

	//Iterations of the primal-dual solver at each pyramid level.
	//Maximum value set to 100 at the finest level
	for (int i=5; i>=0; i--)
	{
		if (i >= ctf_levels - 1)
			num_max_iter[i] = 100;	
		else
			num_max_iter[i] = num_max_iter[i+1]-15;
	}

    //Compute gaussian mask
	int v_mask[5] = {1,4,6,4,1};
    for (unsigned int i=0; i<5; i++)
        for (unsigned int j=0; j<5; j++)
            g_mask[i+5*j] = float(v_mask[i]*v_mask[j])/256.f;


    //Reserve memory for the scene flow estimate (the finest)
	dxp = (float *) malloc(sizeof(float)*rows*cols);
	dyp = (float *) malloc(sizeof(float)*rows*cols);
	dzp = (float *) malloc(sizeof(float)*rows*cols);

    // After setting rows, cols, and ctf_levels
    dx.resize(ctf_levels);
    dy.resize(ctf_levels);
    dz.resize(ctf_levels);

    unsigned int s, cols_i, rows_i;
    for (unsigned int i = 0; i < ctf_levels; i++) {
        s = static_cast<unsigned int>(pow(2.f, int(ctf_levels - (i + 1))));
        cols_i = cols / s;
        rows_i = rows / s;
        dx[ctf_levels - i - 1].resize(rows_i, cols_i);
        dy[ctf_levels - i - 1].resize(rows_i, cols_i);
        dz[ctf_levels - i - 1].resize(rows_i, cols_i);
    }

	//Resize pyramid
    const unsigned int pyr_levels = round(log2(width/cols)) + ctf_levels;
    colour.resize(pyr_levels);
    colour_old.resize(pyr_levels);
    depth.resize(pyr_levels);
    depth_old.resize(pyr_levels);
    xx.resize(pyr_levels);
    xx_old.resize(pyr_levels);
    yy.resize(pyr_levels);
    yy_old.resize(pyr_levels);

    for (unsigned int i = 0; i<pyr_levels; i++)
    {
        s = pow(2.f,int(i));
        colour[i].resize(height/s, width/s);
        colour_old[i].resize(height/s, width/s);
        colour[i].setZero();
        colour_old[i].setZero();
        depth[i].resize(height/s, width/s);
        depth_old[i].resize(height/s, width/s);
        depth[i].setZero();
        depth_old[i].setZero();
        xx[i].resize(height/s, width/s);
        xx_old[i].resize(height/s, width/s);
        xx[i].setZero();
        xx_old[i].setZero();
        yy[i].resize(height/s, width/s);
        yy_old[i].resize(height/s, width/s);
        yy[i].setZero();
        yy_old[i].setZero();
    }

    

    //Parameters of the variational method
    lambda_i = 0.04f;
    lambda_d = 0.35f;
    mu = 75.f;

	// Create window
	//cv::namedWindow("SceneFlow", cv::WINDOW_NORMAL);
	//cv::namedWindow("SceneFlow", cv::WINDOW_NORMAL);
	//cv::moveWindow("SceneFlow",width - cols/2,height - rows/2);
}


void PD_flow_opencv::createImagePyramidGPU()
{
    // Pass the GpuMat objects directly. The function will extract the
    // raw pointer and pitch information (via .ptr<float>() and .step).
    csf_host.copyNewFrames(I, Z);

    // Copy scene flow object to device.
    csf_device = ObjectToDevice(&csf_host);

    unsigned int pyr_levels = static_cast<unsigned int>(log2(float(width/cols))) + ctf_levels;
    GaussianPyramidBridge(csf_device, pyr_levels,
                          image_preparer->getCameraProperties().res_width,
                          image_preparer->getCameraProperties().res_height);

    // Copy scene flow object back to host.
    BridgeBack(&csf_host, csf_device);
}

void PD_flow_opencv::solveSceneFlowGPU()
{
    
    unsigned int s;
    unsigned int cols_i, rows_i;
    unsigned int level_image;
    unsigned int num_iter;

    //For every level (coarse-to-fine)
    for (unsigned int i=0; i<ctf_levels; i++)
    {
        s = static_cast<unsigned int>(pow(2.f,int(ctf_levels-(i+1))));
        cols_i = cols/s;
        rows_i = rows/s;
        level_image = ctf_levels - i + static_cast<unsigned int>(log2(float(width/cols))) - 1;

        //=========================================================================
        //                              Cuda - Begin
        //=========================================================================

        //Cuda allocate memory
        csf_host.allocateMemoryNewLevel(rows_i, cols_i, i, level_image);

        //Cuda copy object to device
        csf_device = ObjectToDevice(&csf_host);

        //Assign zeros to the corresponding variables
        AssignZerosBridge(csf_device);

        //Upsample previous solution
        if (i>0)
            UpsampleBridge(csf_device);

        //Compute connectivity (Rij)
		RijBridge(csf_device);
		
		//Compute colour and depth derivatives
        ImageGradientsBridge(csf_device);
        WarpingBridge(csf_device);

        //Compute mu_uv and step sizes for the primal-dual algorithm
        MuAndStepSizesBridge(csf_device);

        //Primal-Dual solver
		for (num_iter = 0; num_iter < num_max_iter[i]; num_iter++)
        {
            GradientBridge(csf_device);
            DualVariablesBridge(csf_device);
            DivergenceBridge(csf_device);
            PrimalVariablesBridge(csf_device);
        }

        //Filter solution
        FilterBridge(csf_device);

        //Compute the motion field
        MotionFieldBridge(csf_device);

        //BridgeBack to host
        BridgeBack(&csf_host, csf_device);

        //Free memory of variables associated to this level
        csf_host.freeLevelVariables();

		//Copy motion field and images to CPU
		csf_host.copyAllSolutions(dx[ctf_levels-i-1].data(), dy[ctf_levels-i-1].data(), dz[ctf_levels-i-1].data(),
                        depth[level_image].data(), depth_old[level_image].data(), colour[level_image].data(), colour_old[level_image].data(),
                        xx[level_image].data(), xx_old[level_image].data(), yy[level_image].data(), yy_old[level_image].data());

        //=========================================================================
        //                              Cuda - end
        //=========================================================================
    }

}

void PD_flow_opencv::freeGPUMemory()
{
    csf_host.freeDeviceMemory();
}

void PD_flow_opencv::initializeCUDA()
{
	
	if (height == 240) {cam_mode = 2;}
	else			   {cam_mode = 1;}

	I = (float *) malloc(sizeof(float)*width*height);
	Z = (float *) malloc(sizeof(float)*width*height);

	//CaptureFrame();
	
	//Read parameters
    csf_host.readParameters(image_preparer->getCameraProperties().res_width, image_preparer->getCameraProperties().res_height, 
                            image_preparer->getCameraProperties().fx, image_preparer->getCameraProperties().fy, 
                            image_preparer->getCameraProperties().cx, image_preparer->getCameraProperties().cy, rows, cols, ctf_levels, g_mask);

    //Allocate memory
    csf_host.allocateDevMemory();
}

/*void PD_flow_opencv::showImages()
{
	const unsigned int dispx = intensity1.cols + 20;
	const unsigned int dispy = intensity1.rows + 20;

	//Show images with OpenCV windows
	cv::namedWindow("I1", cv::WINDOW_AUTOSIZE);
	cv::moveWindow("I1",10,10);
	cv::imshow("I1", intensity1);

	cv::namedWindow("Z1", cv::WINDOW_AUTOSIZE);
	cv::moveWindow("Z1",dispx,10);
	cv::imshow("Z1", depth1);

	cv::namedWindow("I2", cv::WINDOW_AUTOSIZE);
	cv::moveWindow("I2",10,dispy);
	cv::imshow("I2", intensity2);

	cv::namedWindow("Z2", cv::WINDOW_AUTOSIZE);
	cv::moveWindow("Z2",dispx,dispy);
	cv::imshow("Z2", depth2);

	cv::waitKey(30);
}*/

void PD_flow_opencv::initIntensityAndDepth() {
    intensity1 = cv::Mat::zeros(height, width, CV_8UC1);
    depth1 = cv::Mat::zeros(height, width, CV_16UC1);
    intensity2 = cv::Mat::zeros(height, width, CV_8UC1);
    depth2 = cv::Mat::zeros(height, width, CV_16UC1);
}

/*std::chrono::duration<double> PD_flow_opencv::CaptureFrame(CpuImageBuffer &image_buffer) {

    if (I == nullptr) {
        std::cerr << "I is not allocated!" << std::endl;
    }
    // Now, if needed, fill your I and Z arrays as in your file-loading code:
    // Copy intensity (grayscale) pixels into I.
    for (unsigned int u = 0; u < width; u++) {
        for (unsigned int v = 0; v < height; v++) {
            I[v + u * height] = float(image_buffer.gray_buffer[0].at<unsigned char>(v, u));
        }
    }
    
    // If needed, copy the depth floating-point values into Z
    for (unsigned int v = 0; v < height; v++) {
        for (unsigned int u = 0; u < width; u++) {
            Z[v + u * height] = image_buffer.depth_buffer[0].at<float>(v, u);
        }
    }

	createImagePyramidGPU();

	// Now, if needed, fill your I and Z arrays as in your file-loading code:
    // Copy intensity (grayscale) pixels into I.
    for (unsigned int u = 0; u < width; u++) {
        for (unsigned int v = 0; v < height; v++) {
            I[v + u * height] = float(image_buffer.gray_buffer[1].at<unsigned char>(v, u));
        }
    }

	// If needed, copy the depth floating-point values into Z
    for (unsigned int v = 0; v < height; v++) {
        for (unsigned int u = 0; u < width; u++) {
            Z[v + u * height] = image_buffer.depth_buffer[1].at<float>(v, u);
        }
    }

	createImagePyramidGPU();

    // Return the time difference between the two grayscale image timestamps.
    return image_buffer.gray_timestamps[1] - image_buffer.gray_timestamps[0];
}*/

std::chrono::duration<double> PD_flow_opencv::CaptureFrame(CpuImageBuffer &image_buffer) {

    if (I == nullptr) {
        std::cerr << "I is not allocated!" << std::endl;
    }
    cv::Mat transposedGray;
    cv::transpose(image_buffer.gray_buffer[0], transposedGray);
    memcpy(I, transposedGray.ptr<float>(), width * height * sizeof(float));

    // b) Transpose and copy for the depth image.
    cv::Mat transposedDepth;
    cv::transpose(image_buffer.depth_buffer[0], transposedDepth);
    memcpy(Z, transposedDepth.ptr<float>(), width * height * sizeof(float));

    // c) Create the image pyramid for the first frame.
    createImagePyramidGPU();

    cv::Mat transposedGray2;
    cv::transpose(image_buffer.gray_buffer[1], transposedGray2);
    memcpy(I, transposedGray2.ptr<float>(), width * height * sizeof(float));

    // b) Transpose and copy for the depth image.
    cv::Mat transposedDepth2;
    cv::transpose(image_buffer.depth_buffer[1], transposedDepth2);
    memcpy(Z, transposedDepth2.ptr<float>(), width * height * sizeof(float));

    // c) Create the image pyramid for the second frame.
    createImagePyramidGPU();

    // Return the time difference between the two grayscale image timestamps.
    return image_buffer.gray_timestamps[1] - image_buffer.gray_timestamps[0];
}

cv::Mat createSceneFlowImage(const cv::Mat &temporal_dx,
                             const cv::Mat &temporal_dy,
                             const cv::Mat &temporal_dz)
{
    // Ensure that all input matrices are non-empty and have the same dimensions.
    if (temporal_dx.empty() || temporal_dy.empty() || temporal_dz.empty())
        throw std::runtime_error("One or more input matrices are empty.");
    
    if ((temporal_dx.size() != temporal_dy.size()) ||
        (temporal_dx.size() != temporal_dz.size()))
        throw std::runtime_error("All input matrices must have the same dimensions.");
    
    // Get the image dimensions.
    int rows = temporal_dx.rows;
    int cols = temporal_dx.cols;
    
    // Compute the maximum absolute values of each flow component over the entire image.
    float maxDx = 0.f, maxDy = 0.f, maxDz = 0.f;
    for (int v = 0; v < rows; v++) {
        for (int u = 0; u < cols; u++) {
            float absDx = std::abs(temporal_dx.at<float>(v, u));
            float absDy = std::abs(temporal_dy.at<float>(v, u));
            float absDz = std::abs(temporal_dz.at<float>(v, u));
            maxDx = std::max(maxDx, absDx);
            maxDy = std::max(maxDy, absDy);
            maxDz = std::max(maxDz, absDz);
        }
    }
    
    // Avoid division by zero.
    if (maxDx <= 0.f)
        maxDx = 1.f;
    if (maxDy <= 0.f)
        maxDy = 1.f;
    if (maxDz <= 0.f)
        maxDz = 1.f;
    
    // Create the output 8-bit, 3-channel image.
    cv::Mat scene_flow_image(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // Normalize and assign each component to its respective channel.
    // Blue for temporal_dx, green for temporal_dy, red for temporal_dz.
    for (int v = 0; v < rows; v++) {
        for (int u = 0; u < cols; u++) {
            unsigned char blue  = static_cast<unsigned char>(255.f * std::abs(temporal_dx.at<float>(v, u)) / maxDx);
            unsigned char green = static_cast<unsigned char>(255.f * std::abs(temporal_dy.at<float>(v, u)) / maxDy);
            unsigned char red   = static_cast<unsigned char>(255.f * std::abs(temporal_dz.at<float>(v, u)) / maxDz);
            scene_flow_image.at<cv::Vec3b>(v, u) = cv::Vec3b(blue, green, red);
        }
    }
    
    return scene_flow_image;
}

// Constructor for StabilizedDirectionFilter
StabilizedDirectionFilter::StabilizedDirectionFilter(float alpha) : alpha(alpha) {
}

// Filter method that transforms to level frame, filters, and transforms back
std::array<float, 3> StabilizedDirectionFilter::filter(
        const std::array<float, 3>& u_ref, 
        const Eigen::Affine3f& base_link_to_vehicle) {
    
    // Convert control array to Eigen vector for easier manipulation
    Eigen::Vector3f control_body(u_ref[0], u_ref[1], u_ref[2]);
    
    // Transform control from body frame to level frame using the existing transform
    Eigen::Vector3f control_level = base_link_to_vehicle.linear() * control_body;
    
    // Apply low-pass filter in the level frame
    Eigen::Vector3f filtered_level;
    for (int i = 0; i < 3; i++) {
        u_filtered_var[i] = alpha * u_filtered_var[i] + (1.0f - alpha) * control_level[i];
        filtered_level[i] = u_filtered_var[i];
    }
    
    // Transform filtered control back to body frame
    Eigen::Vector3f filtered_body = base_link_to_vehicle.linear().transpose() * filtered_level;
    
    // Return as array
    return {filtered_body[0], filtered_body[1], filtered_body[2]};
}

cv::Mat createSceneFlowMagnitudeImage(const cv::Mat &temporal_dx,
                                      const cv::Mat &temporal_dy,
                                      const cv::Mat &temporal_dz,
                                      double max_intensity)  // Optional parameter
{
    // Ensure that all input matrices are non-empty and have the same dimensions.
    if (temporal_dx.empty() || temporal_dy.empty() || temporal_dz.empty())
        throw std::runtime_error("One or more input matrices are empty.");
    
    if ((temporal_dx.size() != temporal_dy.size()) ||
        (temporal_dx.size() != temporal_dz.size()))
        throw std::runtime_error("All input matrices must have the same dimensions.");
    
    // Calculate squared components
    cv::Mat dx_squared, dy_squared, dz_squared;
    cv::pow(temporal_dx, 2, dx_squared);
    cv::pow(temporal_dy, 2, dy_squared);
    cv::pow(temporal_dz, 2, dz_squared);
    
    // Sum the squares
    cv::Mat sum_of_squares = dx_squared + dy_squared + dz_squared;
    
    // Calculate magnitude (square root of sum of squares)
    cv::Mat magnitude;
    cv::sqrt(sum_of_squares, magnitude);
    
    // Create grayscale output image
    cv::Mat magnitude_image;    
    // Normalize to 0-255 range
    // This will cap values above max_intensity at 255
    magnitude.convertTo(magnitude_image, CV_8UC1, 255.0/max_intensity, 0);
    
    return magnitude_image;
}


/*void PD_flow_opencv::showResults( )
{
	cv::Mat sf_image = createImage( );

	//Show the scene flow as an RGB image	
	//cv::namedWindow("SceneFlow", cv::WINDOW_NORMAL);
    //cv::moveWindow("SceneFlow",width - cols/2,height - rows/2);
	cv::imshow("SceneFlow", sf_image);
}*/

void PD_flow_opencv::showResults( )
{
	cv::Mat sf_image = createImage( );

	//Show the scene flow as an RGB image	
    
	cv::imshow("SceneFlow", sf_image);
	cv::waitKey(1);

	//saveResults( sf_image );
}

void PD_flow_opencv::saveResults( const cv::Mat& sf_image ) const
{
	//Save the scene flow as a text file 
	char	name[500];
	int     nFichero = 0;
	bool    free_name = false;

	while (!free_name)
	{
		nFichero++;
		sprintf(name, "%s_results%02u.txt", output_filename_root.c_str(), nFichero );
		free_name = !fileExists(name);
	}
	
	std::ofstream f_res;
	f_res.open(name);
	printf("Saving the estimated scene flow to file: %s \n", name);

	//Format: (pixel(row), pixel(col), vx, vy, vz)
	for (unsigned int v=0; v<dx[0].rows(); v++)
		for (unsigned int u=0; u<dx[0].cols(); u++)
		{
			f_res << v << " ";
			f_res << u << " ";
			f_res << dx[0](v, u) << " ";
			f_res << dy[0](v, u) << " ";
			f_res << dz[0](v, u) << std::endl;
		}

	f_res.close();

	//Save the RGB representation of the scene flow
	sprintf(name, "%s_representation%02u.png", output_filename_root.c_str(), nFichero);
	printf("Saving the visual representation to file: %s \n", name);
	cv::imwrite(name, sf_image);
}

cv::Mat PD_flow_opencv::createImage() const
{
	//Save scene flow as an RGB image (one colour per direction)
	cv::Mat sf_image(rows, cols, CV_8UC3);

    //Compute the max values of the flow (of its components)
	float maxmodx = 0.f, maxmody = 0.f, maxmodz = 0.f;
	for (unsigned int v=0; v<rows; v++)
		for (unsigned int u=0; u<cols; u++)
		{
            if (fabs(dxp[v + u*rows]) > maxmodx)
                maxmodx = fabs(dxp[v + u*rows]);
            if (fabs(dyp[v + u*rows]) > maxmody)
                maxmody = fabs(dyp[v + u*rows]);
            if (fabs(dzp[v + u*rows]) > maxmodz)
                maxmodz = fabs(dzp[v + u*rows]);
		}

	//Create an RGB representation of the scene flow estimate: 
	for (unsigned int v=0; v<rows; v++)
		for (unsigned int u=0; u<cols; u++)
		{
            sf_image.at<cv::Vec3b>(v,u)[0] = static_cast<unsigned char>(255.f*fabs(dxp[v + u*rows])/maxmodx); //Blue - x
            sf_image.at<cv::Vec3b>(v,u)[1] = static_cast<unsigned char>(255.f*fabs(dyp[v + u*rows])/maxmody); //Green - y
            sf_image.at<cv::Vec3b>(v,u)[2] = static_cast<unsigned char>(255.f*fabs(dzp[v + u*rows])/maxmodz); //Red - z
		}
	
	return sf_image;
}

void PD_flow_opencv::showAndSaveResults( )
{
	cv::Mat sf_image = createImage( );

	//Show the scene flow as an RGB image	
	//cv::namedWindow("SceneFlow", cv::WINDOW_NORMAL);
    //cv::moveWindow("SceneFlow",width - cols/2,height - rows/2);
	cv::imshow("SceneFlow", sf_image);
	cv::waitKey(1);

	saveResults( sf_image );
}

scene_flow_msgs::SceneFlow generateSceneFlowMsg(
    const cv::Mat& xx_mat,
    const cv::Mat& yy_mat,
    const cv::Mat& depth_mat,
    const cv::Mat& dx_mat,
    const cv::Mat& dy_mat,
    const cv::Mat& dz_mat) {
    
    // Create the message
    scene_flow_msgs::SceneFlow msg;
    
    // Clear the arrays, just to be sure
    msg.points.clear();
    msg.flow_vectors.clear();

    // Make sure all matrices have the same dimensions
    if (xx_mat.rows != yy_mat.rows || xx_mat.cols != yy_mat.cols ||
        xx_mat.rows != depth_mat.rows || xx_mat.cols != depth_mat.cols ||
        dx_mat.rows != dy_mat.rows || dx_mat.cols != dy_mat.cols ||
        dz_mat.rows != dy_mat.rows || dz_mat.cols != dy_mat.cols) {
        std::cerr << "Matrix dimensions don't match in generateSceneFlowMsg" << std::endl;
        return msg;
    }

    for (int i = 0; i < xx_mat.rows; i++) {
        for (int j = 0; j < xx_mat.cols; j++) {
            // Optional: Add depth threshold check if needed
            // if (depth_old_mat.at<float>(i, j) > 0.1f) {
                
            // Create and populate a geometry_msgs::Point
            geometry_msgs::Point pt;
            pt.x = static_cast<double>(xx_mat.at<float>(i, j));
            pt.y = static_cast<double>(yy_mat.at<float>(i, j));
            pt.z = static_cast<double>(depth_mat.at<float>(i, j));
            
            // Create and populate the corresponding flow vector
            geometry_msgs::Vector3 vec;
            vec.x = static_cast<double>(dx_mat.at<float>(i, j));
            vec.y = static_cast<double>(dy_mat.at<float>(i, j));
            vec.z = static_cast<double>(dz_mat.at<float>(i, j));
    
            // Append the point and the vector to the message arrays
            msg.points.push_back(pt);
            msg.flow_vectors.push_back(vec);
            // }
        }
    }

    return msg;
}

scene_flow_msgs::SceneFlow generateSceneFlowMsg(
    const Eigen::Matrix<float, 3, Eigen::Dynamic>& positions,
    const Eigen::Matrix<float, 3, Eigen::Dynamic>& velocities) {
    
    // Create the message
    scene_flow_msgs::SceneFlow msg;
    
    // Clear the arrays, just to be sure
    msg.points.clear();
    msg.flow_vectors.clear();

    // Make sure matrices have the same number of columns (points)
    if (positions.cols() != velocities.cols()) {
        std::cerr << "Matrix dimensions don't match in generateSceneFlowMsg" << std::endl;
        return msg;
    }

    // Number of points
    const int numPoints = positions.cols();
    
    for (int i = 0; i < numPoints; i++) {
        // Create and populate a geometry_msgs::Point
        geometry_msgs::Point pt;
        pt.x = static_cast<double>(positions(0, i)); // x coordinate
        pt.y = static_cast<double>(positions(1, i)); // y coordinate
        pt.z = static_cast<double>(positions(2, i)); // z coordinate
        
        // Create and populate the corresponding flow vector
        geometry_msgs::Vector3 vec;
        vec.x = static_cast<double>(velocities(0, i)); // vx
        vec.y = static_cast<double>(velocities(1, i)); // vy
        vec.z = static_cast<double>(velocities(2, i)); // vz
    
        // Append the point and the vector to the message arrays
        msg.points.push_back(pt);
        msg.flow_vectors.push_back(vec);
    }

    return msg;
}

void PD_flow_opencv::transformCameraToDrone(float tx, float ty, float tz) {
    
    // Call the bridge function that applies the translation.
    ApplyPositionChangeBridge(&csf_host, tx, ty, tz);
}