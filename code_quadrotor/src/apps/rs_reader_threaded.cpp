#include "rs_reader_threaded.h"


// Constructor automatically sets up the RealSense streams, retrieves the camera intrinsics,
// and spawns the background run loop thread.
RSReader::RSReader(my_ros_package::RosNode& ros_node) : exit_loop_(false), ros_node_(ros_node) {
    // Configure the streams:
    // Depth: 640x480, Z16 format, 30 FPS.
    // Color: 640x480, RGB8 format, 30 FPS.
    cfg_.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg_.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);

    // Start streaming.
    profile_ = pipe_.start(cfg_);

    // Retrieve the intrinsics from the color stream.
    auto color_stream = profile_.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    intrinsics_ = color_stream.get_intrinsics();
    std::cout << "Camera Intrinsics:" << std::endl;
    std::cout << "  fx: " << intrinsics_.fx << std::endl;
    std::cout << "  fy: " << intrinsics_.fy << std::endl;
    std::cout << "  cx: " << intrinsics_.ppx << std::endl;
    std::cout << "  cy: " << intrinsics_.ppy << std::endl;
    std::cout << "Resolution: " << intrinsics_.width << " x " << intrinsics_.height << std::endl;


    // The filters are already created as member variables.
    // Optionally, you can set default filter parameters here.
    // Example defaults:
    spatial_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.3f);
    spatial_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 30.f);
    //spatial_filter_.set_option(RS2_OPTION_HOLES_FILL, 2);

    temporal_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.4f);
    temporal_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20.f);
    //temporal_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_FRAMES, 4);


    // Start the run loop on its background thread.
    run_thread_ = std::thread(&RSReader::runLoop, this);
}

// The destructor signals the run loop to exit and joins the background thread.
RSReader::~RSReader() {
    exit_loop_ = true;
    if (run_thread_.joinable()) {
        run_thread_.join();
    }
}

// New: Wait until initial valid images are available in the CPU buffer.
void RSReader::waitForFirstImages() {
    std::unique_lock<std::mutex> lock(image_mutex_);

    while (cpu_buffer_.newImages < 2) {
        // Wait for the next set of frames.
        rs2::frameset frameset = pipe_.wait_for_frames();
        // Get angular velocity synchronized with the frames - directly after frame capture
        geometry_msgs::Vector3 current_angular_velocity = 
            ros_node_.getAngularVelocityInCameraFrame();
        rs2::frame color_frame = frameset.get_color_frame();
        rs2::frame depth_frame = frameset.get_depth_frame();

        if (!color_frame || !depth_frame) {
            continue;
        }
        
        

        // --- Apply Filtering on the Depth Frame ---
        // First, apply the spatial filter to smooth the data with edge preservation.
        depth_frame = spatial_filter_.process(depth_frame);
        // Then, apply the temporal filter to reduce noise over time.
        //depth_frame = temporal_filter_.process(depth_frame);

        // --- Process the Color Frame ---
        int color_width  = color_frame.as<rs2::video_frame>().get_width();
        int color_height = color_frame.as<rs2::video_frame>().get_height();

        // Create a cv::Mat from the color frame (data in RGB order).
        cv::Mat color(cv::Size(color_width, color_height), CV_8UC3,
                      (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

        // Convert the RGB image to grayscale.
        cv::Mat gray;
        cv::cvtColor(color, gray, cv::COLOR_RGB2GRAY);

        // Convert to 32-bit float.
        cv::Mat gray_32f;
        gray.convertTo(gray_32f, CV_32FC1);

        // --- Process the Depth Frame ---
        int depth_width  = depth_frame.as<rs2::video_frame>().get_width();
        int depth_height = depth_frame.as<rs2::video_frame>().get_height();

        // Create a cv::Mat from the depth frame (Z16 format).
        cv::Mat depth(cv::Size(depth_width, depth_height), CV_16U,
                      (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

        cv::Mat depth_32f;
        // Convert to 32-bit float in meters (scale factor 0.001).
        depth.convertTo(depth_32f, CV_32FC1, 0.001f);

        cpu_buffer_.gray_buffer[0]  = gray_32f.clone();
        cpu_buffer_.depth_buffer[0] = depth_32f.clone();

        cpu_buffer_.gray_buffer[1]  = gray_32f.clone();
        cpu_buffer_.depth_buffer[1] = depth_32f.clone();
        
        // Store the raw newest images
        cpu_buffer_.raw_newest_color_image = color.clone();
        cpu_buffer_.raw_newest_depth_image = depth.clone();

        // Store initial angular velocity in both slots
        cpu_buffer_.angular_velocity[0] = current_angular_velocity;
        cpu_buffer_.angular_velocity[1] = current_angular_velocity;

        cpu_buffer_.newImages = 2;
    }
}

// The run loop reads frames from the camera, converts them, and stores the results in the CPU image buffer.
void RSReader::runLoop() {

    int frame_count = 0;
    auto last_time = std::chrono::steady_clock::now();

    while (!exit_loop_) {
        // Wait for the next set of frames.
        rs2::frameset frameset = pipe_.wait_for_frames();
        // Immediately after getting frames, get the latest angular velocity
        // This ensures the best possible synchronization
        geometry_msgs::Vector3 current_angular_velocity = 
            ros_node_.getAngularVelocityInCameraFrame();
        rs2::frame color_frame = frameset.get_color_frame();
        rs2::frame depth_frame = frameset.get_depth_frame();

        if (!color_frame || !depth_frame) {
            continue;
        }
        
        

        // --- Apply Filtering on the Depth Frame ---
        // First, apply the spatial filter to smooth the data with edge preservation.
        //depth_frame = spatial_filter_.process(depth_frame);
        // Then, apply the temporal filter to reduce noise over time.
        //depth_frame = temporal_filter_.process(depth_frame);

        // --- Process the Color Frame ---
        int color_width  = color_frame.as<rs2::video_frame>().get_width();
        int color_height = color_frame.as<rs2::video_frame>().get_height();

        // Create a cv::Mat from the color frame (data in RGB order).
        cv::Mat color(cv::Size(color_width, color_height), CV_8UC3,
                      (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

        // Convert the RGB image to grayscale.
        cv::Mat gray;
        cv::cvtColor(color, gray, cv::COLOR_RGB2GRAY);

        // Convert to 32-bit float.
        cv::Mat gray_32f;
        gray.convertTo(gray_32f, CV_32FC1);

        // --- Process the Depth Frame ---
        int depth_width  = depth_frame.as<rs2::video_frame>().get_width();
        int depth_height = depth_frame.as<rs2::video_frame>().get_height();

        // Create a cv::Mat from the depth frame (Z16 format).
        cv::Mat depth(cv::Size(depth_width, depth_height), CV_16U,
                      (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

        cv::Mat depth_32f;
        // Convert to 32-bit float in meters (scale factor 0.001).
        depth.convertTo(depth_32f, CV_32FC1, 0.001f);

        // --- Update the CPU image buffer ---
        {
            std::lock_guard<std::mutex> lock(image_mutex_);

            // Store the raw newest images
            cpu_buffer_.raw_newest_color_image = color.clone();
            cpu_buffer_.raw_newest_depth_image = depth.clone();

            // Move the older images to the previous slot
            cpu_buffer_.gray_buffer[0]  = cpu_buffer_.gray_buffer[1];
            cpu_buffer_.depth_buffer[0] = cpu_buffer_.depth_buffer[1];

            // Store the latest images in slot 1.
            cpu_buffer_.gray_buffer[1]  = gray_32f.clone();
            cpu_buffer_.depth_buffer[1] = depth_32f.clone();

            // Retrieve sensor timestamps (in microseconds) for both frames.
            double color_sensor_ts = color_frame.get_frame_metadata(RS2_FRAME_METADATA_SENSOR_TIMESTAMP);
            double depth_sensor_ts = depth_frame.get_frame_metadata(RS2_FRAME_METADATA_SENSOR_TIMESTAMP);

            // Use a shared timestamp,
            // either by simply choosing one or by averaging them.
            // Here we average the two sensor timestamps:
            double shared_sensor_ts = (color_sensor_ts + depth_sensor_ts) * 0.5;

            // Convert the shared timestamp to a std::chrono::steady_clock::time_point.
            // Note: The sensor timestamp is in microseconds.
            auto shared_timepoint = std::chrono::steady_clock::time_point(std::chrono::microseconds((int64_t)shared_sensor_ts));


            // Update timestamps with the shared value.
            cpu_buffer_.gray_timestamps[0]  = cpu_buffer_.gray_timestamps[1];
            cpu_buffer_.gray_timestamps[1]  = shared_timepoint;
            cpu_buffer_.depth_timestamps[0] = cpu_buffer_.depth_timestamps[1];
            cpu_buffer_.depth_timestamps[1] = shared_timepoint;
            
            // Update angular velocity with synchronized value
            cpu_buffer_.angular_velocity[0] = cpu_buffer_.angular_velocity[1];
            cpu_buffer_.angular_velocity[1] = current_angular_velocity;

            // Increment newImages counter (capped at 2).
            cpu_buffer_.newImages = std::min(cpu_buffer_.newImages + 1U, 2U);
        }
        new_images_cv_.notify_one();

        // --- FPS Calculation ---
        frame_count++;
        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - last_time;
        if (elapsed_seconds.count() >= 1.0) {
            double fps = frame_count / elapsed_seconds.count();
            frame_count = 0;
            last_time = current_time;
        }

        // --- Display the Center Depth ---
        float center_depth = depth_32f.at<float>(depth_height / 2, depth_width / 2);
    }
}

// Retrieve a deep copy of the current CPU image buffer.
CpuImageBuffer RSReader::getCpuImageBuffer() {
    std::unique_lock<std::mutex> lock(image_mutex_);
    new_images_cv_.wait(lock, [this]() { return cpu_buffer_.newImages > 0; });

    CpuImageBuffer copy;
    copy.newImages = cpu_buffer_.newImages;
    for (int i = 0; i < 2; ++i) {
        copy.raw_newest_color_image = cpu_buffer_.raw_newest_color_image.clone();
        copy.raw_newest_depth_image = cpu_buffer_.raw_newest_depth_image.clone();
        copy.gray_buffer[i]       = cpu_buffer_.gray_buffer[i].clone();
        copy.depth_buffer[i]      = cpu_buffer_.depth_buffer[i].clone();
        copy.gray_timestamps[i]   = cpu_buffer_.gray_timestamps[i];
        copy.depth_timestamps[i]  = cpu_buffer_.depth_timestamps[i];
        copy.angular_velocity[i]  = cpu_buffer_.angular_velocity[i];
    }
    // Reset the newImages counter after retrieving the buffer.
    cpu_buffer_.newImages = 0;
    return copy;
}

// Return camera intrinsics as a tuple: (fx, fy, cx, cy, width, height)
std::tuple<float, float, float, float, unsigned int, unsigned int>
RSReader::getCameraIntrinsics() const {
    return std::make_tuple(intrinsics_.fx, 
                           intrinsics_.fy, 
                           intrinsics_.ppx,  // ppx is used as cx
                           intrinsics_.ppy,  // ppy is used as cy
                           intrinsics_.width, 
                           intrinsics_.height);
}

