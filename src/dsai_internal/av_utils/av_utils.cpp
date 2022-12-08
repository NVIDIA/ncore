// Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

#include "camera.hpp"

#include <Eigen/Dense>
#include <npe.h>

#include <iomanip>
#include <array>

const char* lidarUnwinding_doc = R"igl_Qu8mg5v7(
Unwind the lidar point cloud
)igl_Qu8mg5v7";
npe_function(_lidarUnwinding)
npe_doc(lidarUnwinding_doc)
npe_arg(pointCloud, dense_double)
npe_arg(transforms, npe_matches(pointCloud))
npe_arg(columnIdx, dense_long)
npe_begin_code()

    // Initialize the features and local reference frames 
    npe_Matrix_pointCloud unwoundPC(pointCloud.rows(), 6);
    unwoundPC.setZero();

    for (int i = 0; i < pointCloud.rows(); i++)
    {
        auto c_idx = columnIdx(i, 0);
        
        Eigen::Matrix<double,3,1> startPoint = transforms.template block(c_idx*4,3,3,1);
        Eigen::Matrix<double,3,1> transformedPoint =  transforms.template block(c_idx*4,0,3,3) * pointCloud.row(i).transpose() + startPoint;

        unwoundPC.template block(i,0,1,3) = startPoint.transpose();
        unwoundPC.template block(i,3,1,3) = transformedPoint.transpose();
    }

    return npe::move(unwoundPC); 

npe_end_code()


const char* pixel2WorldRayFTheta_doc = R"igl_Qu8mg5v7(
Compute unprojections of camera pixels to the world rays by considering the rolling shutter information
)igl_Qu8mg5v7";
npe_function(_pixel2WorldRayFTheta)
npe_doc(pixel2WorldRayFTheta_doc)
npe_arg(pixelCoordinates, dense_double)
npe_arg(fwPoly, dense_double)
npe_arg(bwPoly, dense_double)
npe_arg(principalPoint, dense_double)
npe_arg(imgWidth, int)
npe_arg(imgHeight, int)
npe_arg(maxAngle, double)
npe_arg(shutterType, std::string)
npe_arg(TSensorWorld, dense_double)
npe_begin_code()

    // Extract the coefficients of the forward polynomial
    std::array<double, 6> fwPolyCoefficients = {fwPoly(0,0), fwPoly(0,1), fwPoly(0,2), fwPoly(0,3), fwPoly(0,4), fwPoly(0,5)};
    std::array<double, 6> bwPolyCoefficients = {bwPoly(0,0), bwPoly(0,1), bwPoly(0,2), bwPoly(0,3), bwPoly(0,4), bwPoly(0,5)};

    // Initialize the camera model
    FThetaCamera f_theta{imgWidth, imgHeight, principalPoint, fwPolyCoefficients, bwPolyCoefficients, maxAngle, shutterType};

    // Convert the pixel coordinates to a ray in camera space
    Eigen::Matrix<double, Eigen::Dynamic, 3 > cameraRays;
    cameraRays.resize(pixelCoordinates.rows(), 3);
    cameraRays.setOnes();

    f_theta.pixelToCameraRay(pixelCoordinates, cameraRays);

    // Compute the world rays by considering the rolling shutter information
    Eigen::Matrix<double, Eigen::Dynamic, 6 > worldRays;
    worldRays.resize(pixelCoordinates.rows(), 6);
    worldRays.setZero();

    f_theta.cameraToWorldRay(pixelCoordinates, 
                             cameraRays,
                             TSensorWorld,
                             worldRays);

    return npe::move(worldRays); 

npe_end_code()

const char* pixel2WorldRayPinhole_doc = R"igl_Qu8mg5v7(
Compute unprojections of camera pixels to the world rays by considering the rolling shutter information
)igl_Qu8mg5v7";
npe_function(_pixel2WorldRayPinhole)
npe_doc(pixel2WorldRayPinhole_doc)
npe_arg(pixelCoordinates, dense_double)
npe_arg(principalPoint, dense_double)
npe_arg(focalLength, dense_double)
npe_arg(radialPoly, dense_double)
npe_arg(tangentialPoly, dense_double)
npe_arg(imgWidth, int)
npe_arg(imgHeight, int)
npe_arg(shutterType, std::string)
npe_arg(TSensorWorld, dense_double)
npe_begin_code()

    // Initialize the camera model
    PinholeCamera pinhole{imgWidth, imgHeight, principalPoint, focalLength, radialPoly, tangentialPoly, shutterType};

    // Convert the pixel coordinates to a ray in camera space
    Eigen::Matrix<double, Eigen::Dynamic, 3 > cameraRays;
    cameraRays.resize(pixelCoordinates.rows(), 3);
    cameraRays.setOnes();

    pinhole.pixelToCameraRay(pixelCoordinates, cameraRays);

    // Compute the world rays by considering the rolling shutter information
    Eigen::Matrix<double, Eigen::Dynamic, 6 > worldRays;
    worldRays.resize(pixelCoordinates.rows(), 6);
    worldRays.setZero();

    pinhole.cameraToWorldRay(pixelCoordinates, 
                             cameraRays,
                             TSensorWorld,
                             worldRays);

    return npe::move(worldRays); 

npe_end_code()

const char* pixel2CameraRayFTheta_doc = R"igl_Qu8mg5v7(
Compute unprojections of pixels in the camera image plane to camera rays
)igl_Qu8mg5v7";
npe_function(_pixel2CameraRayFTheta)
npe_doc(pixel2CameraRayFTheta_doc)
npe_arg(pixelCoordinates, dense_double)
npe_arg(fwPoly, dense_double)
npe_arg(bwPoly, dense_double)
npe_arg(principalPoint, dense_double)
npe_arg(imgWidth, int)
npe_arg(imgHeight, int)
npe_arg(maxAngle, double)
npe_arg(shutterType, std::string)
npe_begin_code()

    // Extract the coefficients of the forward polynomial
    std::array<double, 6> fwPolyCoefficients = {fwPoly(0,0), fwPoly(0,1), fwPoly(0,2), fwPoly(0,3), fwPoly(0,4), fwPoly(0,5)};
    std::array<double, 6> bwPolyCoefficients = {bwPoly(0,0), bwPoly(0,1), bwPoly(0,2), bwPoly(0,3), bwPoly(0,4), bwPoly(0,5)};

    // Initialize the camera model
    FThetaCamera f_theta{imgWidth, imgHeight, principalPoint, fwPolyCoefficients, bwPolyCoefficients, maxAngle, shutterType};

    // Convert the pixel coordinates to a ray in camera space
    Eigen::Matrix<double, Eigen::Dynamic, 3 > cameraRays;
    cameraRays.resize(pixelCoordinates.rows(), 3);
    cameraRays.setOnes();

    f_theta.pixelToCameraRay(pixelCoordinates, cameraRays);

    return npe::move(cameraRays); 

npe_end_code()

const char* pixel2CameraRayPinhole_doc = R"igl_Qu8mg5v7(
Compute unprojections of pixels in the camera image plane to camera rays
)igl_Qu8mg5v7";
npe_function(_pixel2CameraRayPinhole)
npe_doc(pixel2CameraRayPinhole_doc)
npe_arg(pixelCoordinates, dense_double)
npe_arg(principalPoint, dense_double)
npe_arg(focalLength, dense_double)
npe_arg(radialPoly, dense_double)
npe_arg(tangentialPoly, dense_double)
npe_arg(imgWidth, int)
npe_arg(imgHeight, int)
npe_arg(shutterType, std::string)
npe_begin_code()

    // Initialize the camera model
    PinholeCamera pinhole{imgWidth, imgHeight, principalPoint, focalLength, radialPoly, tangentialPoly, shutterType};

    // Convert the pixel coordinates to a ray in camera space
    Eigen::Matrix<double, Eigen::Dynamic, 3 > cameraRays;
    cameraRays.resize(pixelCoordinates.rows(), 3);
    cameraRays.setOnes();

    pinhole.pixelToCameraRay(pixelCoordinates, cameraRays);

    return npe::move(cameraRays); 

npe_end_code()

const char* cameraRay2PixelFTheta_doc = R"igl_Qu8mg5v7(
Compute projections of camera rays to pixels in the camera image plane
)igl_Qu8mg5v7";
npe_function(_cameraRay2PixelFTheta)
npe_doc(cameraRay2PixelFTheta_doc)
npe_arg(cameraPoints, dense_double)
npe_arg(fwPoly, dense_double)
npe_arg(bwPoly, dense_double)
npe_arg(principalPoint, dense_double)
npe_arg(imgWidth, int)
npe_arg(imgHeight, int)
npe_arg(maxAngle, double)
npe_arg(shutterType, std::string)
npe_begin_code()

    // Extract the coefficients of the forward polynomial
    std::array<double, 6> fwPolyCoefficients = {fwPoly(0,0), fwPoly(0,1), fwPoly(0,2), fwPoly(0,3), fwPoly(0,4), fwPoly(0,5)};
    std::array<double, 6> bwPolyCoefficients = {bwPoly(0,0), bwPoly(0,1), bwPoly(0,2), bwPoly(0,3), bwPoly(0,4), bwPoly(0,5)};

    // Initialize the camera model
    FThetaCamera f_theta{imgWidth, imgHeight, principalPoint, fwPolyCoefficients, bwPolyCoefficients, maxAngle, shutterType};

    // Convert the pixel coordinates to a ray in camera space
    Eigen::Matrix<double, Eigen::Dynamic, 2 > pixelCoordinates;
    Eigen::Matrix<bool, Eigen::Dynamic, 1 > validFlag;
    pixelCoordinates.resize(cameraPoints.rows(), 2);
    validFlag.resize(cameraPoints.rows(), 1);
    
    f_theta.cameraRayToPixel(cameraPoints, pixelCoordinates, validFlag);

    return std::make_tuple(npe::move(pixelCoordinates), npe::move(validFlag));

npe_end_code()

const char* cameraRay2PixelPinhole_doc = R"igl_Qu8mg5v7(
Computes projection of camera rays to pixels in the camera image plane for a pinhole camera
)igl_Qu8mg5v7";
npe_function(_cameraRay2PixelPinhole)
npe_doc(cameraRay2PixelPinhole_doc)
npe_arg(cameraPoints, dense_double)
npe_arg(principalPoint, dense_double)
npe_arg(focalLength, dense_double)
npe_arg(radialPoly, dense_double)
npe_arg(tangentialPoly, dense_double)
npe_arg(imgWidth, int)
npe_arg(imgHeight, int)
npe_arg(shutterType, std::string)
npe_begin_code()

    // Initialize the camera model
    PinholeCamera pinhole{imgWidth, imgHeight, principalPoint, focalLength, radialPoly, tangentialPoly, shutterType};

    // Convert the pixel coordinates to a ray in camera space
    Eigen::Matrix<double, Eigen::Dynamic, 2 > pixelCoordinates;
    Eigen::Matrix<bool, Eigen::Dynamic, 1 > validFlag;
    pixelCoordinates.resize(cameraPoints.rows(), 2);
    validFlag.resize(cameraPoints.rows(), 1);
    
    pinhole.cameraRayToPixel(cameraPoints, pixelCoordinates, validFlag);

    return std::make_tuple(npe::move(pixelCoordinates), npe::move(validFlag));

npe_end_code()


const char* rollingShutterProjectionFTheta_doc = R"igl_Qu8mg5v7(
Compute projections of the world points to the camera image plane by considering the rolling shutter information
)igl_Qu8mg5v7";
npe_function(_rollingShutterProjectionFTheta)
npe_doc(rollingShutterProjectionFTheta_doc)
npe_arg(points, dense_double)
npe_arg(fwPoly, dense_double)
npe_arg(bwPoly, dense_double)
npe_arg(principalPoint, dense_double)
npe_arg(imgWidth, int)
npe_arg(imgHeight, int)
npe_arg(maxAngle, double)
npe_arg(shutterType, std::string)
npe_arg(TWorldSensorStart, dense_double)
npe_arg(TWorldSensorEnd, dense_double)
npe_arg(maxIter, int)
npe_begin_code()

    // Extract the coefficients of the forward polynomial
    std::array<double, 6> fwPolyCoefficients = {fwPoly(0,0), fwPoly(0,1), fwPoly(0,2), fwPoly(0,3), fwPoly(0,4), fwPoly(0,5)};
    std::array<double, 6> bwPolyCoefficients = {bwPoly(0,0), bwPoly(0,1), bwPoly(0,2), bwPoly(0,3), bwPoly(0,4), bwPoly(0,5)};

    // Initialize the camera model
    FThetaCamera f_theta{imgWidth, imgHeight, principalPoint, fwPolyCoefficients, bwPolyCoefficients, maxAngle, shutterType};

    // Initialize the output parameters
    Eigen::Matrix<double, Eigen::Dynamic, 4> transformationMatrices;
    Eigen::Matrix<double, Eigen::Dynamic, 2> pixelCoordinates;
    Eigen::VectorXi validProjec;
    Eigen::VectorXi initialValidIdx;

    // Compute the rolling shutter projection
    f_theta.rollingShutterProjection(points,
                                     TWorldSensorStart,
                                     TWorldSensorEnd, 
                                     maxIter,
                                     transformationMatrices,
                                     pixelCoordinates,
                                     validProjec,
                                     initialValidIdx);

    return std::make_tuple(npe::move(pixelCoordinates), npe::move(transformationMatrices), npe::move(validProjec), npe::move(initialValidIdx));

npe_end_code()


const char* rollingShutterProjectionPinhole_doc = R"igl_Qu8mg5v7(
Compute projections of the world points to the camera image plane by considering the rolling shutter information
)igl_Qu8mg5v7";
npe_function(_rollingShutterProjectionPinhole)
npe_doc(rollingShutterProjectionPinhole_doc)
npe_arg(points, dense_double)
npe_arg(principalPoint, dense_double)
npe_arg(focalLength, dense_double)
npe_arg(radialPoly, dense_double)
npe_arg(tangentialPoly, dense_double)
npe_arg(imgWidth, int)
npe_arg(imgHeight, int)
npe_arg(shutterType, std::string)
npe_arg(TWorldSensorStart, dense_double)
npe_arg(TWorldSensorEnd, dense_double)
npe_arg(maxIter, int)
npe_begin_code()

    // Initialize the camera model
    PinholeCamera pinhole{imgWidth, imgHeight, principalPoint, focalLength, radialPoly, tangentialPoly, shutterType};

    // Initialize the output parameters
    Eigen::Matrix<double, Eigen::Dynamic, 4> transformationMatrices;
    Eigen::Matrix<double, Eigen::Dynamic, 2> pixelCoordinates;
    Eigen::VectorXi validProjec;
    Eigen::VectorXi initialValidIdx;

    // Compute the rolling shutter projection
    pinhole.rollingShutterProjection(points,
                                     TWorldSensorStart,
                                     TWorldSensorEnd, 
                                     maxIter,
                                     transformationMatrices,
                                     pixelCoordinates,
                                     validProjec,
                                     initialValidIdx);

    return std::make_tuple(npe::move(pixelCoordinates), npe::move(transformationMatrices), npe::move(validProjec), npe::move(initialValidIdx));

npe_end_code()


Eigen::Quaternionf euler2Quaternion(const float roll, const float pitch, const float yaw)
{
    Eigen::AngleAxisf rollAngle(roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle(pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle(yaw, Eigen::Vector3f::UnitZ());

    Eigen::Quaternionf q = yawAngle * pitchAngle * rollAngle;
    return q;
}

const char* isWithin3DBoundingBox_doc = R"igl_Qu8mg5v7(
Computes boolean array of point incidences relative to a list of bbboxes
)igl_Qu8mg5v7";
npe_function(_isWithin3DBoundingBox)
npe_doc(isWithin3DBoundingBox_doc)
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
