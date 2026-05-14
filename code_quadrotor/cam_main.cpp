#include <iostream>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

int main() {
    try {

        std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
        
        // Create a pipeline and configure the depth stream
        rs2::pipeline p;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
        p.start(cfg);

        // Allow some frames for warmup
        for (int i = 0; i < 5; i++) {
            p.wait_for_frames();
        }

        // Grab the latest frameset and get a depth frame
        rs2::frameset frames = p.wait_for_frames();
        rs2::depth_frame depth = frames.get_depth_frame();

        if (!depth) {
            std::cerr << "Did not receive a valid depth frame!" << std::endl;
            return EXIT_FAILURE;
        }

        // Get the dimensions and stride of the depth frame
        int width = depth.get_width();
        int height = depth.get_height();
        int stride = depth.get_stride_in_bytes();

        // Create a cv::Mat that points to the frame's data.
        cv::Mat depthMat(cv::Size(width, height), CV_16U, (void*)depth.get_data(), stride);

        // Clone to safely copy data if depth frame expires.
        cv::Mat depthMatCopy = depthMat.clone();

        // For easier visualization, convert the 16-bit depth to an 8-bit image.
        // Here we map a maximum depth of 10 meters to a value of 255.
        cv::Mat depthMat8U;
        depthMatCopy.convertTo(depthMat8U, CV_8U, 255.0/10000);

        // Save the image to disk instead of displaying it.
        if (!cv::imwrite("depth.png", depthMat8U)) {
            std::cerr << "Failed to write image" << std::endl;
            return EXIT_FAILURE;
        }
        
        std::cout << "Depth image saved to 'depth.png'" << std::endl;
    }
    catch (const rs2::error &e) {
        std::cerr << "RealSense error calling " << e.get_failed_function()
                  << "(" << e.get_failed_args() << "): " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}