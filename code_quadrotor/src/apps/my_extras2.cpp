
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

#include "my_extras2.h"

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

PD_flow_2::PD_flow_2(unsigned int cam_mode_config, unsigned int fps_config, unsigned int rows_config)
{     
    rows = rows_config;      //Maximum size of the coarse-to-fine scheme - Default 240 (QVGA)
    cols = rows*320/240;
    cam_mode = cam_mode_config;   // (1 - 640 x 480, 2 - 320 x 240), Default - 1
    ctf_levels = round(log2(rows/15)) + 1;
    fovh = M_PI*62.5f/180.f;
    fovv = M_PI*45.f/180.f;
	fps = fps_config;		//In Hz, Default - 30

    // Initialize the ros node
    ros_node = std::make_shared<my_ros_package::RosNode>();

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
	float v_mask[5] = {1.f,4.f,6.f,4.f,1.f};
    for (unsigned int i=0; i<5; i++)
        for (unsigned int j=0; j<5; j++)
            g_mask[i+5*j] = v_mask[i]*v_mask[j]/256.f;

    //Matrices that store the original and filtered images with the image resolution
    colour_wf.resize(480/cam_mode,640/cam_mode);
    depth_wf.resize(480/cam_mode,640/cam_mode);

    //Resize vectors according to levels
    dx.resize(ctf_levels); dy.resize(ctf_levels); dz.resize(ctf_levels);

    const unsigned int width = colour_wf.cols();
    const unsigned int height = colour_wf.rows();
    unsigned int s, cols_i, rows_i;

    for (unsigned int i = 0; i<ctf_levels; i++)
    {
        s = pow(2.f,int(ctf_levels-(i+1)));
        cols_i = cols/s; rows_i = rows/s;
        dx[ctf_levels-i-1].resize(rows_i,cols_i);
        dy[ctf_levels-i-1].resize(rows_i,cols_i);
        dz[ctf_levels-i-1].resize(rows_i,cols_i);
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
}

void PD_flow_2::createImagePyramidGPU()
{
    //Copy new frames to the scene flow object
    csf_host.copyNewFrames(colour_wf.data(), depth_wf.data());

    //Copy scene flow object to device
    csf_device = ObjectToDevice(&csf_host);

    unsigned int pyr_levels = round(log2(640/(cam_mode*cols))) + ctf_levels;
    GaussianPyramidBridge(csf_device, pyr_levels, 640, 480);

    //Copy scene flow object back to host
    BridgeBack(&csf_host, csf_device);
}

void PD_flow_2::solveSceneFlowGPU()
{
    //Define variables
    //CTicTac	clock;

    unsigned int s;
    unsigned int cols_i, rows_i;
    unsigned int level_image;
    unsigned int num_iter;

    //clock.Tic();

    //For every level (coarse-to-fine)
    for (unsigned int i=0; i<ctf_levels; i++)
    {
        const unsigned int width = colour_wf.cols();
        s = pow(2.f,int(ctf_levels-(i+1)));
        cols_i = cols/s;
        rows_i = rows/s;
        level_image = ctf_levels - i + round(log2(width/cols)) - 1;

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

        //BridgeBack
        BridgeBack(&csf_host, csf_device);

        //Free variables of variables associated to this level
        csf_host.freeLevelVariables();

        //Copy motion field and images to CPU
		csf_host.copyAllSolutions(dx[ctf_levels-i-1].data(), dy[ctf_levels-i-1].data(), dz[ctf_levels-i-1].data(),
                        depth[level_image].data(), depth_old[level_image].data(), colour[level_image].data(), colour_old[level_image].data(),
                        xx[level_image].data(), xx_old[level_image].data(), yy[level_image].data(), yy_old[level_image].data());

		//For debugging
        //DebugBridge(csf_device);

        //=========================================================================
        //                              Cuda - end
        //=========================================================================
    }
}

/*bool PD_flow::OpenCamera()
{
	rc = openni::STATUS_OK;

    const char* deviceURI = openni::ANY_DEVICE;

    rc = openni::OpenNI::initialize();

    printf("Opening camera...\n %s\n", openni::OpenNI::getExtendedError());
    rc = device.open(deviceURI);
    if (rc != openni::STATUS_OK)
    {
        printf("Device open failed:\n%s\n", openni::OpenNI::getExtendedError());
        openni::OpenNI::shutdown();
        return 1;
    }

    //								Create RGB and Depth channels
    //========================================================================================
    rc = dimage.create(device, openni::SENSOR_DEPTH);
    rc = rgb.create(device, openni::SENSOR_COLOR);


	//                            Configure some properties (resolution)
	//========================================================================================
    rc = device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);

    options = rgb.getVideoMode();
    if (cam_mode == 1)
        options.setResolution(640,480);
    else
        options.setResolution(320,240);

    rc = rgb.setVideoMode(options);
    rc = rgb.setMirroringEnabled(false);

    options = dimage.getVideoMode();
    if (cam_mode == 1)
        options.setResolution(640,480);
    else
        options.setResolution(320,240);

    rc = dimage.setVideoMode(options);
    rc = dimage.setMirroringEnabled(false);

    //Turn off autoExposure
    rgb.getCameraSettings()->setAutoExposureEnabled(false);
    printf("Auto Exposure: %s \n", rgb.getCameraSettings()->getAutoExposureEnabled() ? "ON" : "OFF");

    //Check final resolution
    options = rgb.getVideoMode();
    printf("Resolution (%d, %d) \n", options.getResolutionX(), options.getResolutionY());

	//								Start channels
	//===================================================================================
    rc = dimage.start();
    if (rc != openni::STATUS_OK)
    {
        printf("Couldn't start depth stream:\n%s\n", openni::OpenNI::getExtendedError());
        dimage.destroy();
    }

    rc = rgb.start();
    if (rc != openni::STATUS_OK)
    {
        printf("Couldn't start rgb stream:\n%s\n", openni::OpenNI::getExtendedError());
        rgb.destroy();
    }

    if (!dimage.isValid() || !rgb.isValid())
    {
        printf("Camera: No valid streams. Exiting\n");
        openni::OpenNI::shutdown();
        return 1;
    }

    return 0;
}*/

void PD_flow_2::CloseCamera()
{
    //rgb.destroy();
    //openni::OpenNI::shutdown();
}

/*void PD_flow::CaptureFrame()
{
    //openni::VideoFrameRef framergb, framed;
    //rgb.readFrame(&framergb);
    //dimage.readFrame(&framed);

    const int height = framergb.getHeight();
    const int width = framergb.getWidth();

    if ((framed.getWidth() != framergb.getWidth()) || (framed.getHeight() != framergb.getHeight()))
        cout << endl << "The RGB and the depth frames don't have the same size.";

    else
    {
        //Read new frame
        const openni::DepthPixel* pDepthRow = (const openni::DepthPixel*)framed.getData();
        const openni::RGB888Pixel* pRgbRow = (const openni::RGB888Pixel*)framergb.getData();
        int rowSize = framergb.getStrideInBytes() / sizeof(openni::RGB888Pixel);

        for (int yc = height-1; yc >= 0; --yc)
        {
            const openni::RGB888Pixel* pRgb = pRgbRow;
            const openni::DepthPixel* pDepth = pDepthRow;
            for (int xc = width-1; xc >= 0; --xc, ++pRgb, ++pDepth)
            {
                colour_wf(yc,xc) = 0.299*pRgb->r + 0.587*pRgb->g + 0.114*pRgb->b;
                depth_wf(yc,xc) = 0.001f*(*pDepth);
            }
            pRgbRow += rowSize;
            pDepthRow += rowSize;
        }
    }
}*/

ros::Duration PD_flow_2::CaptureFrame() {

    // sensor_msgs::Image::SharedPtr
    //auto rgb_image = ros_node->get_latest_rgb_image();
    //auto stereo_image = ros_node->get_latest_stereo_image();

	my_ros_package::ImageBuffer image_buffer = ros_node->get_image_buffer();

    // Convert the images to cv::Mat
	//cv_bridge::CvImagePtr cv_ptr_rgb_1 = cv_bridge::toCvCopy(image_buffer.rgb_buffer[0], "bgr8");
	//cv::Mat rgb_image_mat_1 = cv_ptr_rgb_1->image;

    cv_bridge::CvImagePtr cv_ptr_rgb_2 = cv_bridge::toCvCopy(image_buffer.rgb_buffer[1], "bgr8");
    cv::Mat rgb_image_mat_2 = cv_ptr_rgb_2->image;



    // Convert the stereo image using its actual encoding.
    // Here we assume stereo_image->encoding is "16UC1" 
    // (since depth images usually come encoded in 16 bits).
    //cv_bridge::CvImagePtr cv_ptr_stereo_1 = cv_bridge::toCvCopy(image_buffer.stereo_buffer[0], image_buffer.stereo_buffer[0].encoding);
    //cv::Mat stereo_image_mat_1 = cv_ptr_stereo_1->image;

	cv_bridge::CvImagePtr cv_ptr_stereo_2 = cv_bridge::toCvCopy(image_buffer.stereo_buffer[1], image_buffer.stereo_buffer[1].encoding);
	cv::Mat stereo_image_mat_2 = cv_ptr_stereo_2->image;

	// Show RGB and depth images before processing
	//cv::imshow("RGB Before", rgb_image_mat_1);
	//cv::imshow("Depth Before", stereo_image_mat_1);

	// Print min and max of depth image
	//double minDepth, maxDepth;
	//cv::minMaxLoc(stereo_image_mat_1, &minDepth, &maxDepth);
	//std::cout << "Depth min: " << minDepth << ", max: " << maxDepth << std::endl;

    // Convert RGB to grayscale and resize if needed
    //cv::Mat gray_image_1;
    //cv::cvtColor(rgb_image_mat_1, gray_image_1, cv::COLOR_BGR2GRAY);
    //resizeAndCrop(gray_image_1, intensity1, colour_wf.cols(), colour_wf.rows());

	cv::Mat gray_image_2;
    cv::cvtColor(rgb_image_mat_2, gray_image_2, cv::COLOR_BGR2GRAY);
    resizeAndCrop(gray_image_2, intensity2, colour_wf.cols(), colour_wf.rows());

    // Process the stereo image:
    // If it is a depth image stored as "16UC1", convert it to a float image in meters.
    //cv::Mat depth_float_1;
    //stereo_image_mat_1.convertTo(depth_float_1, CV_32FC1, 1.0);
    //resizeAndCrop(depth_float_1, depth1, colour_wf.cols(), colour_wf.rows());

	cv::Mat depth_float_2;
    stereo_image_mat_2.convertTo(depth_float_2, CV_32FC1, 1.0);
    resizeAndCrop(depth_float_2, depth2, colour_wf.cols(), colour_wf.rows());

	// Show RGB and depth images after processing
	//cv::imshow("RGB After", intensity1);
	//cv::imshow("Depth After", depth1);



    // Now, if needed, fill your I and Z arrays as in your file-loading code:
    // Copy intensity (grayscale) pixels into I.
    /*for (unsigned int u = 0; u < width; u++) {
        for (unsigned int v = 0; v < height; v++) {
            I[v + u * height] = float(intensity1.at<unsigned char>(v, u));
        }
    }
    
    // If needed, copy the depth floating-point values into Z
    for (unsigned int v = 0; v < height; v++) {
        for (unsigned int u = 0; u < width; u++) {
            Z[v + u * height] = depth1.at<float>(v, u);
        }
    }

	createImagePyramidGPU();*/


    const int height = intensity2.rows;
    const int width = intensity2.cols;

    if ((intensity2.cols != depth2.cols) || (intensity2.rows != depth2.rows))
        cout << endl << "The RGB and the depth frames don't have the same size.";

    else
    {

        for (int yc = height-1; yc >= 0; --yc)
        {
            for (int xc = width-1; xc >= 0; --xc)
            {
                colour_wf(yc,xc) = intensity2.at<unsigned char>(yc,xc);
                depth_wf(yc,xc) = depth2.at<float>(yc,xc);
            }
        }
    }

	/*// Now, if needed, fill your I and Z arrays as in your file-loading code:
    // Copy intensity (grayscale) pixels into I.
    for (unsigned int u = 0; u < width; u++) {
        for (unsigned int v = 0; v < height; v++) {
            I[v + u * height] = float(intensity2.at<unsigned char>(v, u));
        }
    }

	// If needed, copy the depth floating-point values into Z
    for (unsigned int v = 0; v < height; v++) {
        for (unsigned int u = 0; u < width; u++) {
            Z[v + u * height] = depth2.at<float>(v, u);
        }
    }

	createImagePyramidGPU();*/
	
	// Return the timedifference between the two frames
	return image_buffer.rgb_timestamps[1] - image_buffer.rgb_timestamps[0];
}

void PD_flow_2::freeGPUMemory()
{
    csf_host.freeDeviceMemory();
}

void PD_flow_2::initializeCUDA()
{
    //Read parameters
    csf_host.readParameters(640, 480, 528.0, 528.0, 320.0, 240.0, rows, cols, ctf_levels, g_mask);

    //Allocate memory
    csf_host.allocateDevMemory();
}

/*void PD_flow::initializeScene()
{
    global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 10000000;
    window.resize(1000,900);
    window.setPos(900,0);
    window.setCameraZoom(4);
    window.setCameraAzimuthDeg(190);
    window.setCameraElevationDeg(30);
	window.setCameraPointingToPoint(1,0,0);
	
	scene = window.get3DSceneAndLock();

    //Point cloud (final)
    opengl::CPointCloudPtr fpoints_gl = opengl::CPointCloud::Create();
    fpoints_gl->setColor(0, 1, 1);
    fpoints_gl->enablePointSmooth();
    fpoints_gl->setPointSize(3.0);
    scene->insert( fpoints_gl );

    //Scene Flow (includes initial point cloud)
    opengl::CVectorField3DPtr sf = opengl::CVectorField3D::Create();
    sf->setPointSize(3.0f);
    sf->setLineWidth(2.0f);
    sf->setPointColor(1,0,0);
    sf->setVectorFieldColor(0,0,1);
    sf->enableAntiAliasing();
    scene->insert( sf );

    //Reference frame
    opengl::CSetOfObjectsPtr reference = opengl::stock_objects::CornerXYZ();
    reference->setPose(CPose3D(0,0,0,0,0,0));
	reference->setScale(0.15f);
    scene->insert( reference );

	//Legend
	utils::CImage img_legend;
	img_legend.loadFromXPM(legend_pdflow_xpm);
	opengl::COpenGLViewportPtr legend = scene->createViewport("legend");
	legend->setViewportPosition(20, 20, 201, 252);
	legend->setImageView(img_legend);

    window.unlockAccess3DScene();
    window.repaint();
}*/

/*void PD_flow::updateScene()
{	
	scene = window.get3DSceneAndLock();

	const unsigned int repr_level = round(log2(colour_wf.getColCount()/cols));

	//Point cloud (final)
    opengl::CPointCloudPtr fpoints_gl = scene->getByClass<opengl::CPointCloud>(0);
    fpoints_gl->clear();
    for (unsigned int v=0; v<rows; v++)
        for (unsigned int u=0; u<cols; u++)
            if (depth[repr_level](v,u) > 0.1f)
                fpoints_gl->insertPoint(depth[repr_level](v,u), xx[repr_level](v,u), yy[repr_level](v,u));


    //Scene flow
	opengl::CVectorField3DPtr sf = scene->getByClass<opengl::CVectorField3D>(0);
	sf->setPointCoordinates(depth_old[repr_level], xx_old[repr_level], yy_old[repr_level]);
    sf->setVectorField(dz[0], dx[0], dy[0]);

    window.unlockAccess3DScene();
    window.repaint();
}*/

void PD_flow_2::initializePDFlow()
{
	//Initialize Visualization
	//initializeScene();

    //Initialize CUDA
    // Sleep for 500 milliseconds
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    initializeCUDA();

	//Start video streaming
    //OpenCamera();

	//Fill empty matrices
    CaptureFrame();
    createImagePyramidGPU();
    CaptureFrame();
    createImagePyramidGPU();
    solveSceneFlowGPU();
}