#include <npe.h>
#include <Eigen/Dense>
#include <iomanip>

const char* lidarUnwinding_doc = R"igl_Qu8mg5v7(
Compute the compact geometric features representation (binned density in the local patch)
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
        auto c_idx = columnIdx.coeff(i, 0);
        
        Eigen::Matrix<double,3,1> startPoint = transforms.template block(c_idx*4,3,3,1);
        Eigen::Matrix<double,3,1> transformedPoint =  transforms.template block(c_idx*4,0,3,3) * pointCloud.row(i).transpose() + startPoint;

        unwoundPC.template block(i,0,1,3) = startPoint.transpose();
        unwoundPC.template block(i,3,1,3) = transformedPoint.transpose();
    }


    return npe::move(unwoundPC); 

npe_end_code()


template <typename Derived, typename Derived1, typename Derived2>
void computeBackwardsPolynomial(Eigen::MatrixBase<Derived>& pixelNorms, 
                           Eigen::MatrixBase<Derived1>& cameraIntrinsic,
                           Eigen::MatrixBase<Derived2>& alphas){

    // Iterate over all the pixels and evaluate the polynomial
    for (int i = 0; i < pixelNorms.rows(); i++){
        auto r = pixelNorms.coeff(i,0);
        alphas(i,0) = cameraIntrinsic.coeff(0, 4) + cameraIntrinsic.coeff(0, 5) * r + 
                                cameraIntrinsic.coeff(0, 6) * r * r + cameraIntrinsic.coeff(0, 7) * r * r * r  +
                                    cameraIntrinsic.coeff(0, 8) *r * r * r * r;
    }



}


template <typename Derived, typename Derived1, typename Derived2>
void computeForwardPolynomial(Eigen::MatrixBase<Derived>& alphas, 
                           Eigen::MatrixBase<Derived1>& cameraIntrinsic,
                           Eigen::MatrixBase<Derived2>& delta){

    // Iterate over all the pixels and evaluate the polynomial
    for (int i = 0; i < alphas.rows(); i++){
        auto r = alphas.coeff(i,0);
        delta(i,0) = cameraIntrinsic.coeff(0, 9) + cameraIntrinsic.coeff(0, 10) * r + 
                                cameraIntrinsic.coeff(0, 11) * r * r + cameraIntrinsic.coeff(0, 12) * r * r * r  +
                                    cameraIntrinsic.coeff(0, 13) *r * r * r * r;
    }
}


// Transform the camera rays to the world coordinate system by considering the rolling shutter effect
template <typename Derived, typename Derived1, typename Derived2, typename Derived3, typename Derived4>
void cameraToWorldRay(Eigen::MatrixBase<Derived>& pixelCoordinates,
                    Eigen::MatrixBase<Derived1>& cameraRays, 
                    Eigen::MatrixBase<Derived2>& worldRays, 
                    Eigen::MatrixBase<Derived3>& poses, 
                    Eigen::MatrixBase<Derived4>& poseTimestamps, 
                    int imageHeight,
                    double eofTimestamp,
                    double rollingShutterDelay,
                    double pixelExposureTime){

        // Start pose
        Eigen::Matrix<double,3,3> startRMat = poses.template block(0,0,3,3);
        Eigen::Quaterniond startR(startRMat);
        Eigen::Matrix<double,3,1> startT(poses.template block(0,3,3,1));
        double startTimestamp = poseTimestamps.coeff(0,0);

        // End pose
        Eigen::Matrix<double,3,3> endRMat = poses.template block(4,0,3,3);
        Eigen::Quaterniond endR(endRMat);
        Eigen::Matrix<double,3,1> endT(poses.template block(4,3,3,1));
        double endTimestamp = poseTimestamps.coeff(0,1);

        // Compute the timestamp of the start of the frame
        double sofTimestamp = eofTimestamp - rollingShutterDelay;
        double firstRowTimestamp = sofTimestamp - pixelExposureTime;
        double lastRowTimestamp = eofTimestamp - pixelExposureTime;
        double dFirstLastRow = lastRowTimestamp - firstRowTimestamp;

        // Iterate over the pixels and transform the rats to the world coordinate system
        for (int i = 0; i < cameraRays.rows(); i++){
            double pixelTimestamp = firstRowTimestamp + pixelCoordinates.coeff(i,1) * dFirstLastRow / (imageHeight - 1.0);

            const double t = (pixelTimestamp - startTimestamp) / (endTimestamp - startTimestamp);

            // Interpolate the translation
            Eigen::Matrix<double,3,1> translation = (1 - t) * startT.array() + t * endT.array();

            // Interpolate the rotation
            Eigen::Quaterniond Rot = startR.slerp(t, endR);

            worldRays.template block(i,0,1,3) = translation.transpose();
            worldRays.template block(i,3,1,3) = (Rot.normalized().toRotationMatrix() * cameraRays.row(i).transpose()).normalized().transpose();
        } 
}

// Transform the camera rays to the world coordinate system by considering the rolling shutter effect
template <typename Derived, typename Derived1, typename Derived2>
void pixelToCameraRay(Eigen::MatrixBase<Derived>& pixelCoordinates,
                      Eigen::MatrixBase<Derived1>& cameraIntrinsic,
                      std::string cameraModel,
                      Eigen::MatrixBase<Derived2>& cameraRays)
{
    if (cameraModel == "pinhole") {
        // Compute the x and y coordinate according to the pinhole camera model, z is already set to one
        cameraRays.col(0) = (pixelCoordinates.col(0).array() + 0.5 - cameraIntrinsic.coeff(0,2)) / cameraIntrinsic.coeff(0,0); 
        cameraRays.col(1) = (pixelCoordinates.col(1).array() + 0.5 - cameraIntrinsic.coeff(0,5)) / cameraIntrinsic.coeff(0,4);

    } else if (cameraModel == "f_theta") {

        // Compute the x and y coordinate according to the f_theta camera model, z is already set to one
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> centeredpixelCoordinates;
        centeredpixelCoordinates.resize(pixelCoordinates.rows(), pixelCoordinates.cols());

        centeredpixelCoordinates.col(0) = pixelCoordinates.col(0).array() + 0.5 - cameraIntrinsic.coeff(0,0);
        centeredpixelCoordinates.col(1) = pixelCoordinates.col(1).array() + 0.5 - cameraIntrinsic.coeff(0,1);

        // Compute the norm of the pixel coordinates
        Eigen::Matrix<double, Eigen::Dynamic, 1> pixelNorms;
        pixelNorms.resize(pixelCoordinates.rows(), 1);
        pixelNorms = centeredpixelCoordinates.rowwise().norm();

        // Evaluate the backward polynomial
        Eigen::Matrix<double, Eigen::Dynamic, 1> alphas;
        alphas.resize(pixelCoordinates.rows(), 1);
        computeBackwardsPolynomial(pixelNorms, cameraIntrinsic, alphas);

        // Compute the ray direction
        cameraRays.col(0) = alphas.array().sin() * centeredpixelCoordinates.array().col(0) / pixelNorms.array();
        cameraRays.col(1) = alphas.array().sin() * centeredpixelCoordinates.array().col(1) / pixelNorms.array();
        cameraRays.col(2) = alphas.array().cos();

        // Handle the rays perpendicular to the image plane
        for(int i=0; i < cameraRays.rows(); i++){
            if (pixelNorms.coeff(i,0) < std::numeric_limits<double>::min())
                cameraRays.row(i) = {0,0,1};
        }

    } else {
		std::cout << "Invalid camera model selected (must be one of [pinhole, f_theta]).\n"; 
    }
}

template <typename Derived, typename Derived1>
void numericallyStable2Norm2D(Eigen::MatrixBase<Derived>& camPoints,
                              Eigen::MatrixBase<Derived1>& xyNorms)
{
    for (int i = 0; i < camPoints.rows(); i++)
    {
        auto absX = std::abs(camPoints.coeff(i,0));
        auto absY = std::abs(camPoints.coeff(i,1));

        auto minimum = std::min(absX, absY);
        auto maximum = std::max(absX, absY);

        if(maximum <= 0.0){
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
void cameraRayToPixel(Eigen::MatrixBase<Derived>& cameraPoints,
                      Eigen::MatrixBase<Derived1>& cameraIntrinsic,
                      double imgWidth,
                      double imgHeight,
                      std::string cameraModel,
                      Eigen::MatrixBase<Derived2>& imgPoints,
                      Eigen::MatrixBase<Derived3>& valid)
{
    if (cameraModel == "pinhole") {
        
        Eigen::Matrix<double, 3, 3> intrinsic;
        intrinsic << cameraIntrinsic.coeff(0,0), cameraIntrinsic.coeff(0,1), cameraIntrinsic.coeff(0,2),
             cameraIntrinsic.coeff(0,3), cameraIntrinsic.coeff(0,4), cameraIntrinsic.coeff(0,5),
             cameraIntrinsic.coeff(0,6), cameraIntrinsic.coeff(0,7), cameraIntrinsic.coeff(0,8);

        imgPoints = (intrinsic * cameraPoints.transpose()).transpose();
        imgPoints = (imgPoints.transpose() * imgPoints.col(2).asDiagonal().inverse()).transpose();
        
        // Check if the points are valid
        for (int i = 0; i < cameraPoints.rows(); i++)
        {
            if ( 0 <= imgPoints.coeff(i,0) &&  imgPoints.coeff(i,0) < imgWidth && 0 <= imgPoints.coeff(i,1) && imgPoints.coeff(i,1) < imgHeight)
                valid(i,0) = true;
            else
                valid(i,0) = false;
        }
    }
    else if (cameraModel == "f_theta"){

        Eigen::Matrix<double, Eigen::Dynamic, 1> xyNorm;
        xyNorm.resize(cameraPoints.rows(),1);
        numericallyStable2Norm2D(cameraPoints, xyNorm);

        Eigen::Matrix<double, Eigen::Dynamic, 1> cos_alpha;
        Eigen::Matrix<double, Eigen::Dynamic, 1> alpha;
        Eigen::Matrix<double, Eigen::Dynamic, 1> delta;
        cos_alpha.resize(cameraPoints.rows(),1);
        alpha.resize(cameraPoints.rows(),1);
        delta.resize(cameraPoints.rows(),1);

        cos_alpha = (cameraPoints.col(2).transpose() * cameraPoints.rowwise().norm().asDiagonal().inverse()).transpose();
        alpha = cos_alpha.cwiseMin(1.0).cwiseMax(-1.0).array().acos();

        computeForwardPolynomial(alpha, cameraIntrinsic, delta);

        // Recompute the deltas that are not valid
        for (int i = 0; i < alpha.rows(); i++){
            if (alpha.coeff(i,0) > cameraIntrinsic.coeff(0,16)){
                delta(i,0) = cameraIntrinsic.coeff(0,14) + (alpha.coeff(i,0) - cameraIntrinsic.coeff(0,16)) * cameraIntrinsic.coeff(0,15);
            }
        }

        // Determine the bad points with a norm of zero, and avoid division by zero
        for (int i = 0; i < xyNorm.rows(); i++){
            if (xyNorm.coeff(i,0) <= 0.0){
               xyNorm(i,0) = 1.0;
               delta(i,0) = 0.0;
            }
        }

        // Generate the offset vector
        Eigen::Matrix<double,1,3> offset;
        offset.setZero();
        offset.template block(0,0,1,2) = cameraIntrinsic.template block(0,0,1,2);


        auto scale = delta.array() / xyNorm.array();
        imgPoints = scale.array().replicate(1,3) * cameraPoints.array();
        imgPoints.rowwise() += offset;

        // Check if the points are valid
        for (int i = 0; i < cameraPoints.rows(); i++)
        {
            if ( 0 <= imgPoints.coeff(i,0) &&  imgPoints.coeff(i,0) < imgWidth && 0 <= imgPoints.coeff(i,1) && imgPoints.coeff(i,1) < imgHeight)
                valid(i,0) = true;
            else
                valid(i,0) = false;
        }

    }
    else{
        std::cout << "Invalid camera model selected (must be one of [pinhole, f_theta]).\n"; 
    }
}

const char* pixel2WorldRay_doc = R"igl_Qu8mg5v7(
Compute the compact geometric features representation (binned density in the local patch)
)igl_Qu8mg5v7";
npe_function(_pixel2WorldRay)
npe_doc(pixel2WorldRay_doc)
npe_arg(pixelCoordinates, dense_double)
npe_arg(cameraIntrinsic, dense_double)
npe_arg(cameraModel, std::string)
npe_arg(imageHeight, double)
npe_arg(rollingShutterDelay, double)
npe_arg(pixelExposureTime, double)
npe_arg(eofTimestamp, double)
npe_arg(poses, dense_double)
npe_arg(poseTimestamps, dense_double)
npe_begin_code()

    // Convert the pixel coordinates to a ray in camera space
    npe_Matrix_pixelCoordinates cameraRays(pixelCoordinates.rows(), 3);
    cameraRays.setOnes();

    pixelToCameraRay(pixelCoordinates, cameraIntrinsic, cameraModel, cameraRays);

    // if (cameraModel == "pinhole") {
    //     // Compute the x and y coordinate according to the pinhole camera model, z is already set to one
    //     pixel2rayPinhole
    //     cameraRays.col(0) = (pixelCoordinates.col(0).array() + 0.5 - cameraIntrinsic.coeff(0,2)) / cameraIntrinsic.coeff(0,0); 
    //     cameraRays.col(1) = (pixelCoordinates.col(1).array() + 0.5 - cameraIntrinsic.coeff(0,5)) / cameraIntrinsic.coeff(0,4);

    // } else if (cameraModel == "ftheta") {

    //     pixel2rayFTheta(pixelCoordinates, cameraRays)
    //     // Compute the x and y coordinate according to the f_theta camera model, z is already set to one
    //     npe_Matrix_pixelCoordinates centeredpixelCoordinates(pixelCoordinates.rows(), pixelCoordinates.cols());
    //     centeredpixelCoordinates.col(0) = pixelCoordinates.col(0).array() + 0.5 - cameraIntrinsic.coeff(0,0);
    //     centeredpixelCoordinates.col(1) = pixelCoordinates.col(1).array() + 0.5 - cameraIntrinsic.coeff(0,1);

    //     // Compute the norm of the pixel coordinates
    //     npe_Matrix_pixelCoordinates pixelNorms(pixelCoordinates.rows(), 1);
    //     pixelNorms = centeredpixelCoordinates.rowwise().norm();

    //     // Evaluate the backward polynomial
    //     npe_Matrix_pixelCoordinates alphas(pixelCoordinates.rows(), 1);
    //     computeBackwardsPolynomial(pixelNorms, cameraIntrinsic, alphas);

    //     // Compute the ray direction
    //     cameraRays.col(0) = alphas.array().sin() * centeredpixelCoordinates.array().col(0) / pixelNorms.array();
    //     cameraRays.col(1) = alphas.array().sin() * centeredpixelCoordinates.array().col(1) / pixelNorms.array();
    //     cameraRays.col(2) = alphas.array().cos();

    //     // Handle the rays perpendicular to the image plane
    //     for(int i=0; i < cameraRays.rows(); i++){
    //         if (pixelNorms.coeff(i,0) < std::numeric_limits<double>::min())
    //             cameraRays.row(i) = {0,0,1};
    //     }

    // } else {
	// 	std::cout << "Invalid camera model selected (must be one of [pinhole, f_theta]).\n"; 
    // }

    // Compute the world rays by considering the rollign shutter information
    npe_Matrix_pixelCoordinates worldRays(pixelCoordinates.rows(), 6);
    worldRays.setZero();

    cameraToWorldRay(pixelCoordinates, cameraRays, worldRays, poses, poseTimestamps, 
                      imageHeight, eofTimestamp, rollingShutterDelay, pixelExposureTime);

    return npe::move(worldRays); 

npe_end_code()

const char* cameraRay2Pixel_doc = R"igl_Qu8mg5v7(
Compute the compact geometric features representation (binned density in the local patch)
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
    npe_Matrix_cameraPoints pixelCoordinates(cameraPoints.rows(), 3);

    Eigen::Matrix<bool, Eigen::Dynamic, 1 > validFlag;
    validFlag.resize(cameraPoints.rows(), 1);
    
    cameraRayToPixel(cameraPoints, cameraIntrinsic, imgWidth, imgHeight, cameraModel, pixelCoordinates, validFlag);
    return std::make_tuple(npe::move(pixelCoordinates), npe::move(validFlag));

npe_end_code()


const char* rollingShutterProjection_doc = R"igl_Qu8mg5v7(
Compute the compact geometric features representation (binned density in the local patch)
)igl_Qu8mg5v7";
npe_function(_rollingShutterProjection)
npe_doc(rollingShutterProjection_doc)
npe_arg(points, dense_double)
npe_arg(cameraIntrinsic, dense_double)
npe_arg(imgHeight, double)
npe_arg(imgWidth, double)
npe_arg(rollingShutterDelay, double)
npe_arg(exposureTime, double)
npe_arg(eofTimestamp, double)
npe_arg(poses, dense_double)
npe_arg(poseTimestamps, dense_double)
npe_arg(cameraModel, std::string)
npe_arg(iterate, bool)
npe_begin_code()

    // Extract the start and end rotations and translation vectors
    Eigen::Matrix<double,3,3> startRotMat = poses.template block(0,0,3,3);
    Eigen::Quaterniond startRot(startRotMat);
    Eigen::Matrix<double,3,3> endRotMat = poses.template block(4,0,3,3);
    Eigen::Quaterniond endRot(endRotMat);

    Eigen::Matrix<double,3,1> startTrans = poses.template block(0,3,3,1);
    Eigen::Matrix<double,3,1> endTrans = poses.template block(4,3,3,1);
    
    double startPoseTimestamp = poseTimestamps.coeff(0,0);
    double endPoseTimestamp = poseTimestamps.coeff(0,1);

    // Get the required timestamps
    double sofTimestamp = eofTimestamp - rollingShutterDelay;
    double firstRowTimestamp = sofTimestamp - exposureTime;
    double lastRowTimestamp = eofTimestamp - exposureTime;
    double deltaTimeFirstLastRow  = lastRowTimestamp - firstRowTimestamp;

    // Interpolate the pose to the eof timestamp and perform the initial projection
    double dt = (eofTimestamp - startPoseTimestamp) / (endPoseTimestamp - startPoseTimestamp);
    Eigen::Matrix<double,3,1> eofTrans = (1 - dt) * startTrans.array() + dt * endTrans.array();
    Eigen::Quaterniond eofRot = startRot.slerp(dt, endRot);

    // Convert the pixel coordinates to a ray in camera space
    npe_Matrix_points camPoints(points.rows(), 3);

    camPoints = (eofRot.normalized().toRotationMatrix() * points.transpose()).transpose();
    camPoints.rowwise() += eofTrans.transpose();

    // Convert the pixel coordinates to a ray in camera space
    npe_Matrix_points initialProjections(camPoints.rows(), 3);
    Eigen::Matrix<bool, Eigen::Dynamic, 1 > validFlag;
    validFlag.resize(camPoints.rows(), 1);
    cameraRayToPixel(camPoints, cameraIntrinsic, imgWidth, imgHeight, cameraModel, initialProjections, validFlag);

    // Get the number of valid elements 
    auto nValid = validFlag.count();
    npe_Matrix_points pixelCoordinates(nValid, 3);

    // Initialize a transformation matrix
    npe_Matrix_points transformationMatrices(nValid*4, 4);
    transformationMatrices.setZero();

    // Perform the rolling shutter compensation
    int runIdx = 0;
    int nMaxIter = 10;
    double nTimeThres = 300; // Threshold on the time
    double updateStep = 50;
    Eigen::Matrix<int, Eigen::Dynamic, 1> validIndices;
    validIndices.resize(nValid, 1);

    for (int i = 0; i < camPoints.rows(); i++)
    {
        if (validFlag.coeff(i,0))
        {
            // Compute the timestamp based on the row index of the projected point
            double projTimestamp = firstRowTimestamp + initialProjections.coeff(i,1) * deltaTimeFirstLastRow / (imgHeight - 1);

            if(iterate)
            {

                continue;
            }
            else{
                
                // Interpolate the pose to the row index
                double projDeltaT = (projTimestamp - startPoseTimestamp) / (endPoseTimestamp - startPoseTimestamp);
                Eigen::Matrix<double,3,1> projTrans = (1 - projDeltaT) * startTrans.array() + projDeltaT * endTrans.array();
                Eigen::Quaterniond projRot = startRot.slerp(projDeltaT, endRot);

                // Transform the point to the cam coordinate system
                Eigen::Matrix<double, 1, 3 > tmpCamPoint = (projRot.normalized().toRotationMatrix() * points.row(i).transpose() + projTrans).transpose();

                // Convert the pixel coordinates to a ray in camera space
                Eigen::Matrix<bool, 1, 1 > finalValidFlag;
                Eigen::Matrix<double, 1, 3 > finalProjection;
                cameraRayToPixel(tmpCamPoint, cameraIntrinsic, imgWidth, imgHeight, cameraModel, finalProjection, finalValidFlag);

                // Save the pixel coordinates
                pixelCoordinates.row(runIdx) = finalProjection;

                // Save the transformation matrix
                transformationMatrices.template block(4*runIdx, 0, 3, 3) = projRot.normalized().toRotationMatrix();
                transformationMatrices.template block(4*runIdx, 3, 3, 1) = projTrans;
                transformationMatrices(4*runIdx + 3, 3) = 1.0;

                // Update the index
                validIndices(runIdx,0) = i;
                runIdx++;
            }
        }
    }

    return std::make_tuple(npe::move(pixelCoordinates), npe::move(transformationMatrices), npe::move(validIndices));


npe_end_code()