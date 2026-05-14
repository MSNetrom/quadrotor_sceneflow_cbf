#ifndef SCENE_TRANSFORM_H
#define SCENE_TRANSFORM_H

#include <Eigen/Dense>
#include <iostream>

// Computes translational velocities for a batch of points given velocities, angular velocities and relative positions
void computeTranslationalVelocitiesBatch(const Eigen::MatrixXf &velocities,
                                       const Eigen::MatrixXf &angularVelocities, 
                                       const Eigen::MatrixXf &relativePositions,
                                       Eigen::MatrixXf &translationalVelocities);

// Transforms vectors from one frame to NED frame using rotation matrix
void transformToNEDBatch(const Eigen::Matrix3f &R_NED,
                        const Eigen::MatrixXf &inputVectors,
                        Eigen::MatrixXf &nedVectors);

#endif // SCENE_TRANSFORM_H
