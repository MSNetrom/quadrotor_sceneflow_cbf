#include "vector_median.h"

// Calculates the vector median for a group of 3-channel pixels in a fixed-size array.
// 'pixels' is an array of cv::Vec3b values and 'count' is the number of neighbors.
cv::Vec3b vectorMedian(const cv::Vec3b* pixels, int count) {
    double minSum = std::numeric_limits<double>::max();
    cv::Vec3b medianPixel = pixels[0];
    
    for (int i = 0; i < count; i++) {
        double sumDistances = 0.0;
        for (int j = 0; j < count; j++) {
            int diff0 = static_cast<int>(pixels[i][0]) - static_cast<int>(pixels[j][0]);
            int diff1 = static_cast<int>(pixels[i][1]) - static_cast<int>(pixels[j][1]);
            int diff2 = static_cast<int>(pixels[i][2]) - static_cast<int>(pixels[j][2]);
            int distSq = diff0 * diff0 + diff1 * diff1 + diff2 * diff2;
            sumDistances += distSq;
        }
        if (sumDistances < minSum) {
            minSum = sumDistances;
            medianPixel = pixels[i];
        }
    }
    
    return medianPixel;
}

// Applies a vector median filter on a 3-channel image using a fixed-size array (without push_back).
// kernelSize must be an odd number.
cv::Mat vectorMedianFilter(const cv::Mat& src, int kernelSize) {
    CV_Assert(src.channels() == 3);
    CV_Assert(kernelSize % 2 == 1); // Ensure kernelSize is odd.
    
    int radius = kernelSize / 2;
    cv::Mat dst = src.clone();
    
    // Maximum neighbors in a window: kernelSize * kernelSize.
    const int maxNeighbors = kernelSize * kernelSize;
    
    // Iterate over every pixel in the image.
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            // Preallocate an array for the neighbors.
            cv::Vec3b neighbors[maxNeighbors];
            int count = 0;
            
            // Collect neighbor pixels.
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int ny = row + dy;
                    int nx = col + dx;
                    
                    // Check if the neighbor is inside the image bounds.
                    if (ny >= 0 && ny < src.rows && nx >= 0 && nx < src.cols) {
                        neighbors[count] = src.at<cv::Vec3b>(ny, nx);
                        count++; // Increase neighbor count.
                    }
                }
            }
            
            // Compute the vector median from the collected neighbors.
            dst.at<cv::Vec3b>(row, col) = vectorMedian(neighbors, count);
        }
    }
    return dst;
}

// Applies a vector median filter on three separate channels representing a color image.
// Each channel is assumed to be a single-channel image (e.g. CV_8UC1) with the same dimensions.
// The filtered output is split into three separate output images.
void vectorMedianFilterSeparate(const cv::Mat& ch0,
                                const cv::Mat& ch1,
                                const cv::Mat& ch2,
                                int kernelSize,
                                cv::Mat& out0,
                                cv::Mat& out1,
                                cv::Mat& out2) {
    CV_Assert(ch0.size() == ch1.size() && ch0.size() == ch2.size());
    CV_Assert(ch0.type() == ch1.type() && ch0.type() == ch2.type());
    CV_Assert(ch0.channels() == 1 && ch1.channels() == 1 && ch2.channels() == 1);

    // Create output images with the same size and type as the inputs.
    out0 = ch0.clone();
    out1 = ch1.clone();
    out2 = ch2.clone();

    int rows = ch0.rows;
    int cols = ch0.cols;
    int radius = kernelSize / 2;
    const int maxNeighbors = kernelSize * kernelSize;

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            cv::Vec3b neighbors[maxNeighbors];
            int count = 0;

            // Collect neighbor pixels from the three channels.
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int ny = row + dy;
                    int nx = col + dx;
                    if (ny >= 0 && ny < rows && nx >= 0 && nx < cols) {
                        // Build the 3-channel vector from the three separate channel images.
                        uchar v0 = ch0.at<uchar>(ny, nx);
                        uchar v1 = ch1.at<uchar>(ny, nx);
                        uchar v2 = ch2.at<uchar>(ny, nx);
                        neighbors[count] = cv::Vec3b(v0, v1, v2);
                        count++;
                    }
                }
            }
            // Compute the vector median from the collected neighbors.
            cv::Vec3b med = vectorMedian(neighbors, count);
            out0.at<uchar>(row, col) = med[0];
            out1.at<uchar>(row, col) = med[1];
            out2.at<uchar>(row, col) = med[2];
        }
    }
}