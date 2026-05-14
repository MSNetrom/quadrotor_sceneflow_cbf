#pragma once

#include <Eigen/Dense>
#include <array>
#include <cmath>

/**
 * Filters only the magnitude of a vector while preserving its direction.
 * Useful for control signals where maintaining direction is important,
 * but you want to smooth the intensity of the command.
 */
class MagnitudeFilter {
public:
    /**
     * Constructor
     * @param alpha Filter coefficient (0-1). Higher values mean less filtering.
     *        alpha=1.0 means no filtering, alpha=0.0 means output never changes.
     */
    explicit MagnitudeFilter(float alpha = 0.7f) 
        : alpha_(alpha), prev_magnitude_(0.0f) {}
    
    /**
     * Filter a 3D vector by filtering only its magnitude
     * @param input The input vector to filter
     * @return The filtered vector with preserved direction
     */
    Eigen::Vector3f filter(const Eigen::Vector3f& input) {
        // Handle zero input case
        if (input.isZero(1e-6)) {
            prev_magnitude_ = 0.0f;
            return Eigen::Vector3f::Zero();
        }
        
        // Extract magnitude and direction
        float magnitude = input.norm();
        Eigen::Vector3f direction = input.normalized();
        
        // Apply low-pass filter to magnitude only
        float filtered_magnitude = alpha_ * magnitude + (1.0f - alpha_) * prev_magnitude_;
        prev_magnitude_ = filtered_magnitude;
        
        // Return vector with filtered magnitude but original direction
        return direction * filtered_magnitude;
    }
    
    /**
     * Filter a 3D array by filtering only its magnitude
     * @param input The input array to filter
     * @return The filtered array with preserved direction
     */
    std::array<float, 3> filter(const std::array<float, 3>& input) {
        // Convert array to Eigen vector
        Eigen::Vector3f vec(input[0], input[1], input[2]);
        
        // Apply vector filter
        Eigen::Vector3f filtered = filter(vec);
        
        // Convert back to array
        return {filtered[0], filtered[1], filtered[2]};
    }

private:
    float alpha_;          // Filter coefficient
    float prev_magnitude_; // Previous filtered magnitude
}; 