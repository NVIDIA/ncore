// Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

#include <npe.h>
#include <Eigen/Dense>
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


// Evaluates a polynomial (of degree DEGREE) given it's coefficients at a specific point)
// using numerically stable Horner-scheme (https://en.wikipedia.org/wiki/Horner%27s_method)
template<size_t DEGREE, typename Scalar>
Scalar evaluatePolynomialHornerScheme(Scalar x,
                                      std::array<Scalar, DEGREE+1> const& coefficients)
{
      auto ret = Scalar{0};
      for(auto it = coefficients.rbegin(); it != coefficients.rend(); ++it)
      {
          ret = ret * x + *it;
      }
      return ret;
}

template <typename Derived, typename Derived1, typename Derived2>
void computeBackwardsPolynomial(const Eigen::MatrixBase<Derived>& pixelNorms, 
                                const Eigen::MatrixBase<Derived1>& cameraIntrinsic,
                                Eigen::MatrixBase<Derived2>& alphas){
    // Iterate over all the pixels and evaluate the backwards polynomial
    for (int i = 0; i < pixelNorms.rows(); i++)
    {
        auto const r = pixelNorms(i,0);
        alphas(i, 0) = evaluatePolynomialHornerScheme<4U>(
            r, {cameraIntrinsic(0, 4), cameraIntrinsic(0, 5),
                cameraIntrinsic(0, 6), cameraIntrinsic(0, 7),
                cameraIntrinsic(0, 8)});
    }
}

template <typename Derived, typename Derived1, typename Derived2>
void computeForwardPolynomial(const Eigen::MatrixBase<Derived>& alphas, 
                              const Eigen::MatrixBase<Derived1>& cameraIntrinsic,
                              Eigen::MatrixBase<Derived2>& delta){
    // Iterate over all the angles and evaluate the forward polynomial
    for (int i = 0; i < alphas.rows(); i++)
    {
        auto const r = alphas(i,0);
        delta(i, 0)  = evaluatePolynomialHornerScheme<4U>(
            r, {cameraIntrinsic(0, 9), cameraIntrinsic(0, 10),
                cameraIntrinsic(0, 11), cameraIntrinsic(0, 12),
                cameraIntrinsic(0, 13)});
    }
}

template <typename Derived, typename Derived1, typename Derived2>
static void iteraitiveUndistortPoints(const Eigen::MatrixBase<Derived>& src, 
                                      Eigen::MatrixBase<Derived1>& tgt,
                                      const Eigen::MatrixBase<Derived2>& cameraIntrinsic,
                                      const double eps = 1e-12)

{
    const double k_0 = cameraIntrinsic(0,4);
    const double k_1 = cameraIntrinsic(0,5);
    const double k_2 = cameraIntrinsic(0,6);
    const double k_3 = cameraIntrinsic(0,7);
    const double k_4 = cameraIntrinsic(0,8);

    // Convert pixel coordinate to normalized coordinates
    double x, y, x0, y0;
    x = x0 = (src(0,0) - cameraIntrinsic(0,2)) / cameraIntrinsic(0,0);
    y = y0 = (src(0,1) - cameraIntrinsic(0,3)) / cameraIntrinsic(0,1);

    // Initialize the parameters
    double error = std::numeric_limits<double>::max();
    constexpr int maxIter = 30;

    for(int i = 0; i < maxIter && error > eps; i++)
    {
        double r_2 = x*x + y*y;
        double icD = 1. / (1 + ((k_4 * r_2 + k_1) * r_2 + k_0) * r_2);
        double deltaX = 2 * k_2 * x * y + k_3 * (r_2  + 2 * x * x);
        double deltaY = 2 * k_3 * x * y + k_2 * (r_2  + 2 * y * y);

        double x_prev = x; 
        double y_prev = y;

        double x = (x0 - deltaX)*icD;
        double y = (y0 - deltaY)*icD;

        // Compute the current error
        error = std::pow(x - x_prev, 2) + std::pow(y - y_prev, 2);
    }

    // Save the final estimates
    tgt(0,0) = x;
    tgt(0,1) = y;

}

// Transform the camera rays to the world coordinate system by considering the rolling shutter effect
template <typename Derived, typename Derived1, typename Derived2, typename Derived3, typename Derived4>
void cameraToWorldRay(const Eigen::MatrixBase<Derived>& pixelCoordinates,
                      const Eigen::MatrixBase<Derived1>& cameraRays, 
                      Eigen::MatrixBase<Derived2>& worldRays, 
                      const Eigen::MatrixBase<Derived3>& cameraToWorldTransform, 
                      const Eigen::MatrixBase<Derived4>& cameraToWorldTransformTimestamps, 
                      double imgHeight,
                      double imgWidth,
                      int rollingShutterDirection){

        // Start pose
        Eigen::Matrix<double,3,3> startRotMat = cameraToWorldTransform.template block(0,0,3,3);
        Eigen::Quaterniond startRot(startRotMat);
        Eigen::Matrix<double,3,1> startTrans(cameraToWorldTransform.template block(0,3,3,1));

        // End pose
        Eigen::Matrix<double,3,3> endRotMat = cameraToWorldTransform.template block(4,0,3,3);
        Eigen::Quaterniond endRot(endRotMat);
        Eigen::Matrix<double,3,1> endTrans(cameraToWorldTransform.template block(4,3,3,1));

        // Get the required timestamps
        double startTransformTimestamp = cameraToWorldTransformTimestamps(0,0);
        double endTransformTimestamp = cameraToWorldTransformTimestamps(0,1);
        double deltaTimeStartEnd  = endTransformTimestamp - startTransformTimestamp;

        // Iterate over the pixels and transform the rays to the world coordinate system
        for (int i = 0; i < cameraRays.rows(); i++)
        {
            double projTimestamp{};
            switch(rollingShutterDirection)
            {
                case 1:
                    projTimestamp = startTransformTimestamp + std::floor(pixelCoordinates(i,1)) * deltaTimeStartEnd / (imgHeight - 1);
                    break;
                case 2:
                    projTimestamp = startTransformTimestamp + std::floor(pixelCoordinates(i,0)) * deltaTimeStartEnd / (imgWidth - 1);
                    break;
                case 3:
                    projTimestamp = startTransformTimestamp + (imgHeight - std::ceil(pixelCoordinates(i,1))) * deltaTimeStartEnd / (imgHeight - 1);
                    break;
                case 4:
                    projTimestamp = startTransformTimestamp + (imgWidth - std::ceil(pixelCoordinates(i,0))) * deltaTimeStartEnd / (imgWidth - 1);
                    break;
            }

            double projDeltaT = (projTimestamp - startTransformTimestamp) / deltaTimeStartEnd;

            // Interpolate the pose
            Eigen::Matrix<double,3,1> translation = (1 - projDeltaT) * startTrans.array() + projDeltaT * endTrans.array();
            Eigen::Quaterniond Rot = startRot.slerp(projDeltaT,endRot);

            worldRays.template block(i,0,1,3) = translation.transpose();
            worldRays.template block(i,3,1,3) = (Rot.normalized().toRotationMatrix() * cameraRays.row(i).transpose()).normalized().transpose();
        } 
}

// Transform the camera rays to the world coordinate system by considering the rolling shutter effect
template <typename Derived, typename Derived1, typename Derived2>
void pixelToCameraRay(Eigen::MatrixBase<Derived>& pixelCoordinates,
                      Eigen::MatrixBase<Derived1>& cameraIntrinsic,
                      std::string cameraModel,
                      Eigen::MatrixBase<Derived2>& cameraRays){
    if (cameraModel == "pinhole") 
    {
        // Compute the x and y coordinate according to the pinhole camera model, z is already set to one
        for(int i = 0; i < pixelCoordinates.rows(); i++)
        {
            Eigen::Matrix<double,1,2> worldRay;
            worldRay.setZero();
            iteraitiveUndistortPoints(pixelCoordinates.row(i), worldRay, cameraIntrinsic);

            cameraRays.row(i) << pixelCoordinates(i,2), -worldRay(0,0)*pixelCoordinates(i,2), -worldRay(0,1)*pixelCoordinates(i,2);
        }
    } 
    else if (cameraModel == "f_theta") 
    {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> pixelOffsets;
        pixelOffsets.resize(pixelCoordinates.rows(), pixelCoordinates.cols());

        pixelOffsets.col(0) = pixelCoordinates.col(0).array() - cameraIntrinsic(0,0);
        pixelOffsets.col(1) = pixelCoordinates.col(1).array() - cameraIntrinsic(0,1);

        // Compute the norm of the pixel offsets
        Eigen::Matrix<double, Eigen::Dynamic, 1> pixelNorms;
        pixelNorms.resize(pixelOffsets.rows(), 1);
        pixelNorms = pixelOffsets.rowwise().norm();

        // Evaluate the backward polynomial
        Eigen::Matrix<double, Eigen::Dynamic, 1> alphas;
        alphas.resize(pixelOffsets.rows(), 1);
        computeBackwardsPolynomial(pixelNorms, cameraIntrinsic, alphas);

        // Compute the ray direction and handle the rays perpendicular to the
        // image plane
        for (auto i = 0U; i < cameraRays.rows(); ++i)
        {
            auto const pixelNorm = pixelNorms(i, 0);
            if (pixelNorm < std::numeric_limits<double>::min())
            {
                cameraRays(i, 0)  = 0.f;
                cameraRays(i, 1)  = 0.f;
                cameraRays(i, 2)  = 1.f;
            } 
            else
            {
                auto const alpha  = alphas(i);
                auto const alphaSinePerPixelNorm = std::sin(alpha) / pixelNorm;
                auto const alphaCos = std::cos(alpha);
                cameraRays(i, 0)  = alphaSinePerPixelNorm * pixelOffsets(i, 0);
                cameraRays(i, 1)  = alphaSinePerPixelNorm * pixelOffsets(i, 1);
                cameraRays(i, 2)  = alphaCos;
          }
        }
    } 
    else
    {
		std::cout << "Invalid camera model selected (must be one of [pinhole, f_theta]).\n"; 
    }
}

template <typename Derived, typename Derived1>
void numericallyStable2Norm2D(const Eigen::MatrixBase<Derived>& camPoints,
                              Eigen::MatrixBase<Derived1>& xyNorms)
{
    for (int i = 0; i < camPoints.rows(); i++)
    {
        auto absX = std::abs(camPoints(i,0));
        auto absY = std::abs(camPoints(i,1));

        auto minimum = std::min(absX, absY);
        auto maximum = std::max(absX, absY);

        if(maximum <= 0.0)
        {
            xyNorms(i) = 0.0;
            continue;
        }

        auto oneOverMaximum = 1.0 / maximum;
        auto minMaxRatio = minimum * oneOverMaximum;

        xyNorms(i) = maximum * std::sqrt(1.0 + minMaxRatio * minMaxRatio);
    }
}

// Transform the camera rays to the world coordinate system by considering the rolling shutter effect
template <typename Derived, typename Derived1, typename Derived2, typename Derived3>
void cameraRayToPixel(const Eigen::MatrixBase<Derived>& cameraPoints,
                      const Eigen::MatrixBase<Derived1>& cameraIntrinsic,
                      double imgWidth,
                      double imgHeight,
                      std::string cameraModel,
                      Eigen::MatrixBase<Derived2>& imgPoints,
                      Eigen::MatrixBase<Derived3>& valid){

    if (cameraModel == "pinhole") 
    {
        Eigen::Array<double, Eigen::Dynamic, 1> uNormalized = -cameraPoints.col(1).array() / cameraPoints.col(0).array();
        Eigen::Array<double, Eigen::Dynamic, 1> vNormalized = -cameraPoints.col(2).array() / cameraPoints.col(0).array();

        // Evaluate polynomial using robust Horner scheme (inside-out)
        Eigen::Array<double, Eigen::Dynamic, 1> r2 = uNormalized * uNormalized + vNormalized * vNormalized;
        Eigen::Array<double, Eigen::Dynamic, 1> rD = 1.0 + r2 * (cameraIntrinsic(0,4) + r2 * (cameraIntrinsic(0,5) + r2 * cameraIntrinsic(0,8)));

        // If the radial distortion is too large, the computed coordinates will be unreasonable
        const double kMinRadialDistortion = 0.8;
        const double kMaxRadialDistortion = 1.2;

        // Check if the points are valid
        Eigen::Matrix<bool, Eigen::Dynamic, 1> invalidIDx;
        invalidIDx.resize(rD.rows(),1);

        for (int i = 0; i < rD.rows(); i++)
        {
            if (rD(i,0) <= kMinRadialDistortion || rD(i,0) >= kMaxRadialDistortion)
                invalidIDx(i,0) = true;
            else
                invalidIDx(i,0) = false;
        }

        Eigen::Array<double, Eigen::Dynamic, 1> uND = uNormalized * rD + 2.0 * cameraIntrinsic(6,0) * uNormalized * vNormalized + cameraIntrinsic(7,0) * (r2 + 2.0 * uNormalized * uNormalized);
        Eigen::Array<double, Eigen::Dynamic, 1> vND = vNormalized * rD + cameraIntrinsic(6,0) * (r2 + 2.0 * vNormalized * vNormalized) + 2.0 * cameraIntrinsic(7,0) * uNormalized * vNormalized;

        Eigen::Array<double, Eigen::Dynamic, 1> uD = uND * cameraIntrinsic(0,0) + cameraIntrinsic(0,2);
        Eigen::Array<double, Eigen::Dynamic, 1> vD = vND * cameraIntrinsic(0,1) + cameraIntrinsic(0,3);

        // Set the valid flag of points behind the camera to 0
        for (int i=0;  i < cameraPoints.rows(); i++)
        {
            if (cameraPoints(i,0) < 0.0)
                valid(i,0) = false;
            else
                valid(i,0) = true;
        }

        const double clipping_radius = std::sqrt(imgWidth*imgWidth + imgHeight*imgHeight);
        
        for (int i=0;  i < invalidIDx.rows(); i++)
        {
            if (invalidIDx(i,0))
            {
                uD(i,0) = uNormalized(i,0) * ((double)1.0 /std::sqrt(r2(i,0))) * clipping_radius + cameraIntrinsic(0,2);
                vD(i,0) = vNormalized(i,0) * ((double)1.0 /std::sqrt(r2(i,0))) * clipping_radius + cameraIntrinsic(0,3);
                valid(i,0) = false;
            }
        }

        // Concatenate the points
        imgPoints.col(0) = uD.matrix();
        imgPoints.col(1) = vD.matrix();

        // Check if the points are valid
        for (int i = 0; i < imgPoints.rows(); i++)
        {
            if ( 0 > imgPoints(i,0) ||  imgPoints(i,0) > imgWidth || 0 > imgPoints(i,1) || imgPoints(i,1) > imgHeight )
                valid(i,0) = false;
        }
    }
    else if (cameraModel == "f_theta")
    {
        Eigen::Matrix<double, Eigen::Dynamic, 1> xyNorm;
        xyNorm.resize(cameraPoints.rows(),1);
        numericallyStable2Norm2D(cameraPoints, xyNorm);

        Eigen::Matrix<double, Eigen::Dynamic, 1> cosAlpha;
        Eigen::Matrix<double, Eigen::Dynamic, 1> alpha;
        Eigen::Matrix<double, Eigen::Dynamic, 1> delta;
        cosAlpha.resize(cameraPoints.rows(),1);
        alpha.resize(cameraPoints.rows(),1);
        delta.resize(cameraPoints.rows(),1);

        cosAlpha = (cameraPoints.col(2).transpose() * cameraPoints.rowwise().norm().asDiagonal().inverse()).transpose();
        alpha = cosAlpha.cwiseMin(1.0).cwiseMax(-1.0).array().acos();

        computeForwardPolynomial(alpha, cameraIntrinsic, delta);

        // Recompute the deltas that are not valid
        for (int i = 0; i < alpha.rows(); i++)
        {
            if (alpha(i,0) > cameraIntrinsic(0,16))
            {
                delta(i,0) = cameraIntrinsic(0,14) + (alpha(i,0) - cameraIntrinsic(0,16)) * cameraIntrinsic(0,15);
            }
        }

        // Determine the bad points with a norm of zero, and avoid division by zero
        for (int i = 0; i < xyNorm.rows(); i++)
        {
            if (xyNorm(i,0) <= 0.0)
            {
               xyNorm(i,0) = 1.0;
               delta(i,0) = 0.0;
            }
        }

        // Generate the offset vector
        Eigen::Matrix<double,1,2> offset;
        offset.setZero();
        offset.template block(0,0,1,2) = cameraIntrinsic.template block(0,0,1,2);

        auto scale = delta.array() / xyNorm.array();
        imgPoints = scale.array().replicate(1,2) * cameraPoints.template block(0, 0, cameraPoints.rows(), 2).array();
        imgPoints.rowwise() += offset;

        // Check if the points are valid
        for (int i = 0; i < cameraPoints.rows(); i++)
        {
            if ( 0 <= imgPoints(i,0) &&  imgPoints(i,0) < imgWidth && 0 <= imgPoints(i,1) && imgPoints(i,1) < imgHeight)
                valid(i,0) = true;
            else
                valid(i,0) = false;
        }

    }
    else
    {
        throw std::invalid_argument("Invalid camera model selected (must be one of [pinhole, f_theta])"); 
    }
}

const char* pixel2WorldRay_doc = R"igl_Qu8mg5v7(
Compute unprojections of camera pixels to the world rays by considering the rolling shutter information
)igl_Qu8mg5v7";
npe_function(_pixel2WorldRay)
npe_doc(pixel2WorldRay_doc)
npe_arg(pixelCoordinates, dense_double)
npe_arg(cameraIntrinsic, dense_double)
npe_arg(cameraModel, std::string)
npe_arg(imageHeight, double)
npe_arg(imageWidth, double)
npe_arg(cameraToWorldTransform, dense_double)
npe_arg(cameraToWorldTransformTimestamps, dense_double)
npe_arg(rollingShutterDirection, int)
npe_begin_code()

    // Convert the pixel coordinates to a ray in camera space
    npe_Matrix_pixelCoordinates cameraRays(pixelCoordinates.rows(), 3);
    cameraRays.setOnes();

    pixelToCameraRay(pixelCoordinates, cameraIntrinsic, cameraModel, cameraRays);

    // Compute the world rays by considering the rolling shutter information
    npe_Matrix_pixelCoordinates worldRays(pixelCoordinates.rows(), 6);
    worldRays.setZero();

    cameraToWorldRay(pixelCoordinates, cameraRays, worldRays, cameraToWorldTransform, cameraToWorldTransformTimestamps, 
                      imageHeight, imageWidth, rollingShutterDirection);

    return npe::move(worldRays); 

npe_end_code()


const char* pixel2CameraRay_doc = R"igl_Qu8mg5v7(
Compute unprojections of pixels in the camera image plane to camera rays
)igl_Qu8mg5v7";
npe_function(_pixel2CameraRay)
npe_doc(pixel2CameraRay_doc)
npe_arg(pixelCoordinates, dense_double)
npe_arg(cameraIntrinsic, dense_double)
npe_arg(cameraModel, std::string)
npe_begin_code()

    // Convert the pixel coordinates to a ray in camera space
    npe_Matrix_pixelCoordinates cameraRays(pixelCoordinates.rows(), 3);
    cameraRays.setOnes();

    pixelToCameraRay(pixelCoordinates, cameraIntrinsic, cameraModel, cameraRays);

    return npe::move(cameraRays); 

npe_end_code()


const char* cameraRay2Pixel_doc = R"igl_Qu8mg5v7(
Compute projections of camera rays to pixels in the camera image plane
)igl_Qu8mg5v7";
npe_function(_cameraRay2Pixel)
npe_doc(cameraRay2Pixel_doc)
npe_arg(cameraPoints, dense_double)
npe_arg(cameraIntrinsic, dense_double)
npe_arg(imgWidth, double)
npe_arg(imgHeight, double)
npe_arg(cameraModel, std::string)
npe_begin_code()

    // Convert the pixel coordinates to a ray in camera space
    npe_Matrix_cameraPoints pixelCoordinates(cameraPoints.rows(), 2);

    Eigen::Matrix<bool, Eigen::Dynamic, 1 > validFlag;
    validFlag.resize(cameraPoints.rows(), 1);
    
    cameraRayToPixel(cameraPoints, cameraIntrinsic, imgWidth, imgHeight, cameraModel, pixelCoordinates, validFlag);
    return std::make_tuple(npe::move(pixelCoordinates), npe::move(validFlag));

npe_end_code()


const char* rollingShutterProjection_doc = R"igl_Qu8mg5v7(
Compute projections of the world points to the camera image plane by considering the rolling shutter information
)igl_Qu8mg5v7";
npe_function(_rollingShutterProjection)
npe_doc(rollingShutterProjection_doc)
npe_arg(points, dense_double)
npe_arg(cameraIntrinsic, dense_double)
npe_arg(imgHeight, double)
npe_arg(imgWidth, double)
npe_arg(rollingShutterDirection, int)
npe_arg(worldToCameraTransform, dense_double)
npe_arg(worldToCameraTransformTimestamps, dense_double)
npe_arg(cameraModel, std::string)
npe_arg(maxIter, int)
npe_begin_code()

    // Extract the start and end rotations and translation vectors
    Eigen::Matrix<double,3,3> startRotMat = worldToCameraTransform.template block(0,0,3,3);
    Eigen::Quaterniond startRot(startRotMat);
    Eigen::Matrix<double,3,3> endRotMat = worldToCameraTransform.template block(4,0,3,3);
    Eigen::Quaterniond endRot(endRotMat);

    Eigen::Matrix<double,3,1> startTrans = worldToCameraTransform.template block(0,3,3,1);
    Eigen::Matrix<double,3,1> endTrans = worldToCameraTransform.template block(4,3,3,1);
    
    // Get the required timestamps
    double startTransformTimestamp = worldToCameraTransformTimestamps(0,0);
    double endTransformTimestamp = worldToCameraTransformTimestamps(0,1);
    double deltaTimeStartEnd  = endTransformTimestamp - startTransformTimestamp;

    // Interpolate the pose to mof timestamp  
    Eigen::Matrix<double,3,1> mofTrans = (1 - 0.5) * startTrans.array() + 0.5 * endTrans.array();
    Eigen::Quaterniond mofRot = startRot.slerp(0.5, endRot);

    // Convert the pixel coordinates to a ray in camera space
    npe_Matrix_points camPoints(points.rows(), 3);

    camPoints = (mofRot.normalized().toRotationMatrix() * points.transpose()).transpose();
    camPoints.rowwise() += mofTrans.transpose();

    // Convert the pixel coordinates to a ray in camera space
    npe_Matrix_points initialProjections(camPoints.rows(), 2);
    Eigen::Matrix<bool, Eigen::Dynamic, 1 > validFlag;
    validFlag.resize(camPoints.rows(), 1);
    cameraRayToPixel(camPoints, cameraIntrinsic, imgWidth, imgHeight, cameraModel, initialProjections, validFlag);

    // Get the number of valid elements 
    auto nValid = validFlag.count();
    
    std::vector<int> initialValidIndices;

    // Get the indices of the points that were valid initially 
    for (int i = 0; i < initialProjections.rows(); i++)
    {
        if (validFlag(i,0))
            initialValidIndices.push_back(i);
    }

    // Initialize a transformation matrix
    npe_Matrix_points pixelCoordinates(nValid, 2);
    npe_Matrix_points transformationMatrices(nValid*4, 4);
    transformationMatrices.setZero();

    // Perform the rolling shutter compensation
    int runIdx = 0;
    std::vector<int> validProjections;
    
    // TODO: Implement iterative approach
    for (int i = 0; i < initialProjections.rows(); i++)
    {
        if (validFlag(i,0))
        {
            // Initialize the values
            double x = initialProjections(i,0);
            double y = initialProjections(i,1); 

            Eigen::Matrix<bool, 1, 1> finalValidFlag;
            Eigen::Matrix<double,3,1> projTrans;
            Eigen::Quaterniond projRot = Eigen::Quaterniond::Identity();
            
            // Initialize the parameters
            double error = std::numeric_limits<double>::max();
            double eps = 1e-6;

            for (int iter_idx = 0; (iter_idx < maxIter) && (error > eps); iter_idx++)
            {
                double projTimestamp{};
                switch(rollingShutterDirection)
                {
                    case 1:
                        projTimestamp = startTransformTimestamp + std::floor(y) * deltaTimeStartEnd / (imgHeight - 1);
                        break;
                    case 2:
                        projTimestamp = startTransformTimestamp + std::floor(x) * deltaTimeStartEnd / (imgWidth - 1);
                        break;
                    case 3:
                        projTimestamp = startTransformTimestamp + (imgHeight - std::ceil(y)) * deltaTimeStartEnd / (imgHeight - 1);
                        break;
                    case 4:
                        projTimestamp = startTransformTimestamp + (imgWidth - std::ceil(x)) * deltaTimeStartEnd / (imgWidth - 1);
                        break;
                }
                
                // Interpolate the pose to the row index
                double projDeltaT = (projTimestamp - startTransformTimestamp) / deltaTimeStartEnd;
                projTrans = (1 - projDeltaT) * startTrans.array() + projDeltaT * endTrans.array();
                projRot = startRot.slerp(projDeltaT, endRot);

                // Transform the point to the cam coordinate system
                Eigen::Matrix<double, 1, 3 > tmpCamPoint = (projRot.normalized().toRotationMatrix() * points.row(i).transpose() + projTrans).transpose();

                // Convert the pixel coordinates to a ray in camera space
                Eigen::Matrix<double, 1, 2 > tmpProjection;
                cameraRayToPixel(tmpCamPoint, cameraIntrinsic, imgWidth, imgHeight, cameraModel, tmpProjection, finalValidFlag);

                // Compute the projection error between two iterations
                error = std::pow(tmpProjection(0,0) - x, 2) + std::pow(tmpProjection(0,1) - y, 2);
                
                // update the value
                x = tmpProjection(0,0);
                y = tmpProjection(0,1);
            }

            // Save the pixel coordinates
            pixelCoordinates.row(runIdx) << x, y;

            // Save the transformation
            transformationMatrices.template block(4*runIdx, 0, 3, 3) = projRot.normalized().toRotationMatrix();
            transformationMatrices.template block(4*runIdx, 3, 3, 1) = projTrans;
            transformationMatrices(4*runIdx + 3, 3) = 1.0;

            // Update the index
            if (finalValidFlag(0,0))
                validProjections.push_back(runIdx);
            
            runIdx++;
        }
    }

    Eigen::VectorXi validProjec = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(validProjections.data(), validProjections.size());
    Eigen::VectorXi initialValidIdx = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(initialValidIndices.data(), initialValidIndices.size());

    return std::make_tuple(npe::move(pixelCoordinates), npe::move(transformationMatrices), npe::move(validProjec), npe::move(initialValidIdx));

npe_end_code()

Eigen::Quaternionf euler2Quaternion(const float roll, const float pitch, const float yaw )
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
            if(transformedPoints(j,0) <= halfDim.x() && transformedPoints(j,0) >= -halfDim.x() && transformedPoints(j,1) <= halfDim.y() && transformedPoints(j,1) >= -halfDim.y() && transformedPoints(j,2) <= halfDim.z() && transformedPoints(j,2) >= -halfDim.z())
            {
                inBBox(j,i) = true;
            }
        }
    }

    return npe::move(inBBox);
npe_end_code()
