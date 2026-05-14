#ifndef VECTOR_MEDIAN_H
#define VECTOR_MEDIAN_H

#include <opencv2/opencv.hpp>
#include <cmath>
#include <limits>
#include <iostream>

// Calculates the vector median for a group of 3-channel pixels in a fixed-size array.
// 'pixels' is an array of cv::Vec3b values and 'count' is the number of neighbors.
cv::Vec3b vectorMedian(const cv::Vec3b* pixels, int count);

// Applies a vector median filter on a 3-channel image using a fixed-size array
// kernelSize must be an odd number
cv::Mat vectorMedianFilter(const cv::Mat& src, int kernelSize);

// Applies a vector median filter on three separate channels representing a color image.
// Each input channel must be a single-channel image with the same dimensions and type (e.g. CV_8UC1).
// The output images (out0, out1, out2) will contain the filtered channels,
// computed by selecting for each pixel the vector (from the image neighborhood)
// whose cumulative distance is minimal.
void vectorMedianFilterSeparate(const cv::Mat& ch0,
                                const cv::Mat& ch1,
                                const cv::Mat& ch2,
                                int kernelSize,
                                cv::Mat& out0,
                                cv::Mat& out1,
                                cv::Mat& out2);

#endif // VECTOR_MEDIAN_H
