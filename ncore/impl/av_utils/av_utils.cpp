// Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

#include <Eigen/Dense>
#include <npe.h>

#include <iomanip>
#include <array>

Eigen::Quaternionf euler2Quaternion(const float roll, const float pitch, const float yaw)
{
    Eigen::AngleAxisf rollAngle(roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle(pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle(yaw, Eigen::Vector3f::UnitZ());

    Eigen::Quaternionf q = yawAngle * pitchAngle * rollAngle;
    return q;
}

const char* isWithin3DBBoxes_doc = R"igl_Qu8mg5v7(
Computes boolean array of point incidences relative to a list of bbboxes
)igl_Qu8mg5v7";
npe_function(_isWithin3DBBoxes)
npe_doc(isWithin3DBBoxes_doc)
npe_arg(points, dense_float)
npe_arg(bboxes, dense_float)
npe_begin_code()

    // Initialize the buffer and set it to false
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> inBBox;
    inBBox.setConstant(points.rows(), bboxes.rows(), false);

    // Iterate over the bounding boxes and check if the point is within the bbox
    Eigen::Matrix<float, Eigen::Dynamic, 3> transformedPoints;
    transformedPoints.resize(points.rows(), 3);

    for (int i =0; i < bboxes.rows(); i++)
    {
        Eigen::Vector3f center = {bboxes(i,0), bboxes(i,1), bboxes(i,2)};
        Eigen::Vector3f dim = {bboxes(i,3), bboxes(i,4), bboxes(i,5)}; 
        Eigen::Vector3f halfDim = 0.5 * dim;

        // Get the rotation matrix from the euler angles and compute its inverse (we need the inverse transform)
        Eigen::Quaternionf rotQuat = euler2Quaternion(bboxes(i,6), bboxes(i,7), bboxes(i,8));
        Eigen::Matrix3f rotMatrix = rotQuat.normalized().toRotationMatrix().transpose();

        // Compute the inverse translation
        Eigen::Vector3f translation = -rotMatrix * center;

        // Transform the points in the bbox frame
        transformedPoints = (rotMatrix * points.transpose()).transpose().rowwise() + translation.transpose();

        for (int j=0; j < transformedPoints.rows(); j++)
        {
          if (transformedPoints(j, 0) <= halfDim.x() &&
              transformedPoints(j, 0) >= -halfDim.x() &&
              transformedPoints(j, 1) <= halfDim.y() &&
              transformedPoints(j, 1) >= -halfDim.y() &&
              transformedPoints(j, 2) <= halfDim.z() &&
              transformedPoints(j, 2) >= -halfDim.z()) {
            inBBox(j, i) = true;
          }
        }
    }

    return npe::move(inBBox);

npe_end_code()
