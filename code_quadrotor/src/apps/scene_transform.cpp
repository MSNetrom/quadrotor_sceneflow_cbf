#include "scene_transform.h"

// Add positional transform here

void computeTranslationalVelocitiesBatch(const Eigen::MatrixXf &velocities,
                                           const Eigen::MatrixXf &angularVelocities,
                                           const Eigen::MatrixXf &relativePositions,
                                           Eigen::MatrixXf &translationalVelocities)
{
    const Eigen::Index N = velocities.cols();
    translationalVelocities.resize(3, N);

    // Since the angular velocity is the same for all columns,
    // we extract it from the first column of angularVelocities.
    Eigen::Vector3f angVel = angularVelocities.col(0);

    // Compute the cross product for all columns using vectorized array operations.
    // Recall that for a cross product:
    //   (angVel) x (r) = [ angVel.y * r.z - angVel.z * r.y,
    //                       angVel.z * r.x - angVel.x * r.z,
    //                       angVel.x * r.y - angVel.y * r.x ]
    Eigen::ArrayXXf cross(3, N);
    cross.row(0) = angVel.y() * relativePositions.row(2).array()
                 - angVel.z() * relativePositions.row(1).array();
    cross.row(1) = angVel.z() * relativePositions.row(0).array()
                 - angVel.x() * relativePositions.row(2).array();
    cross.row(2) = angVel.x() * relativePositions.row(1).array()
                 - angVel.y() * relativePositions.row(0).array();

    // Compute the final translational velocities for all columns.
    translationalVelocities = velocities - cross.matrix();
}

void transformToNEDBatch(const Eigen::Matrix3f &R_NED,
                         const Eigen::MatrixXf &inputVectors,
                         Eigen::MatrixXf &nedVectors)
{
    // Apply the rotation matrix to the entire batch
    nedVectors = R_NED * inputVectors;
}

