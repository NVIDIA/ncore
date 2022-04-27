#define __NPE_FOR_REAL__
#include <npe.h>
#include <npe.h>
#include <Eigen/Dense>
#include <iomanip>

const char* lidarUnwinding_doc = R"igl_Qu8mg5v7(
Compute the compact geometric features representation (binned density in the local patch)
)igl_Qu8mg5v7";

template <typename npe_Map_pointCloud,typename npe_Matrix_pointCloud,typename npe_Scalar_pointCloud,typename npe_Map_transforms,typename npe_Matrix_transforms,typename npe_Scalar_transforms,typename npe_Map_columnIdx,typename npe_Matrix_columnIdx,typename npe_Scalar_columnIdx>
static auto callit__lidarUnwinding(npe_Map_pointCloud pointCloud,npe_Map_transforms transforms,npe_Map_columnIdx columnIdx) {

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

}


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

template <typename npe_Map_pixelCoordinates,typename npe_Matrix_pixelCoordinates,typename npe_Scalar_pixelCoordinates,typename npe_Map_cameraIntrinsic,typename npe_Matrix_cameraIntrinsic,typename npe_Scalar_cameraIntrinsic,typename npe_Map_poses,typename npe_Matrix_poses,typename npe_Scalar_poses,typename npe_Map_poseTimestamps,typename npe_Matrix_poseTimestamps,typename npe_Scalar_poseTimestamps>
static auto callit__pixel2WorldRay(npe_Map_pixelCoordinates pixelCoordinates,npe_Map_cameraIntrinsic cameraIntrinsic,std::string cameraModel,double imageHeight,double rollingShutterDelay,double pixelExposureTime,double eofTimestamp,npe_Map_poses poses,npe_Map_poseTimestamps poseTimestamps) {

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

}

const char* cameraRay2Pixel_doc = R"igl_Qu8mg5v7(
Compute the compact geometric features representation (binned density in the local patch)
)igl_Qu8mg5v7";

template <typename npe_Map_cameraPoints,typename npe_Matrix_cameraPoints,typename npe_Scalar_cameraPoints,typename npe_Map_cameraIntrinsic,typename npe_Matrix_cameraIntrinsic,typename npe_Scalar_cameraIntrinsic>
static auto callit__cameraRay2Pixel(npe_Map_cameraPoints cameraPoints,npe_Map_cameraIntrinsic cameraIntrinsic,double imgWidth,double imgHeight,std::string cameraModel) {

    // Convert the pixel coordinates to a ray in camera space
    npe_Matrix_cameraPoints pixelCoordinates(cameraPoints.rows(), 3);

    Eigen::Matrix<bool, Eigen::Dynamic, 1 > validFlag;
    validFlag.resize(cameraPoints.rows(), 1);
    
    cameraRayToPixel(cameraPoints, cameraIntrinsic, imgWidth, imgHeight, cameraModel, pixelCoordinates, validFlag);
    return std::make_tuple(npe::move(pixelCoordinates), npe::move(validFlag));

}


const char* rollingShutterProjection_doc = R"igl_Qu8mg5v7(
Compute the compact geometric features representation (binned density in the local patch)
)igl_Qu8mg5v7";

template <typename npe_Map_points,typename npe_Matrix_points,typename npe_Scalar_points,typename npe_Map_cameraIntrinsic,typename npe_Matrix_cameraIntrinsic,typename npe_Scalar_cameraIntrinsic,typename npe_Map_poses,typename npe_Matrix_poses,typename npe_Scalar_poses,typename npe_Map_poseTimestamps,typename npe_Matrix_poseTimestamps,typename npe_Scalar_poseTimestamps>
static auto callit__rollingShutterProjection(npe_Map_points points,npe_Map_cameraIntrinsic cameraIntrinsic,double imgHeight,double imgWidth,double rollingShutterDelay,double exposureTime,double eofTimestamp,npe_Map_poses poses,npe_Map_poseTimestamps poseTimestamps,std::string cameraModel,bool iterate) {

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


}
void pybind_output_fun_utils_cpp(pybind11::module& m) {
m.def("_lidarUnwinding", [](pybind11::array pointCloud, pybind11::array transforms, pybind11::array columnIdx) {
#ifdef __NPE_REDIRECT_IO__
pybind11::scoped_ostream_redirect __npe_redirect_stdout__(std::cout, pybind11::module::import("sys").attr("stdout"));
pybind11::scoped_ostream_redirect __npe_redirect_stderr__(std::cerr, pybind11::module::import("sys").attr("stderr"));
#endif
const char _NPE_PY_BINDING_pointCloud_type_s = npe::detail::transform_typechar(static_cast<pybind11::array&>(pointCloud).dtype().type());
ssize_t pointCloud_shape_0 = 0;
ssize_t pointCloud_shape_1 = 0;
if (static_cast<pybind11::array&>(pointCloud).ndim() == 1) {
pointCloud_shape_0 = static_cast<pybind11::array&>(pointCloud).shape()[0];
pointCloud_shape_1 = static_cast<pybind11::array&>(pointCloud).shape()[0] == 0 ? 0 : 1;
} else if (static_cast<pybind11::array&>(pointCloud).ndim() == 2) {
pointCloud_shape_0 = static_cast<pybind11::array&>(pointCloud).shape()[0];
pointCloud_shape_1 = static_cast<pybind11::array&>(pointCloud).shape()[1];
} else if (static_cast<pybind11::array&>(pointCloud).ndim() > 2) {
throw std::invalid_argument("Argument pointCloud has invalid number of dimensions. Must be 1 or 2.");
}
const npe::detail::StorageOrder _NPE_PY_BINDING_pointCloud_so = (static_cast<pybind11::array&>(pointCloud).flags() & NPY_ARRAY_F_CONTIGUOUS) ? npe::detail::ColMajor : (static_cast<pybind11::array&>(pointCloud).flags() & NPY_ARRAY_C_CONTIGUOUS ? npe::detail::RowMajor : npe::detail::NoOrder);
const int _NPE_PY_BINDING_pointCloud_t_id = npe::detail::get_type_id(npe::detail::is_sparse<std::remove_reference<decltype(static_cast<pybind11::array&>(pointCloud))>::type>::value, _NPE_PY_BINDING_pointCloud_type_s, _NPE_PY_BINDING_pointCloud_so);
const char _NPE_PY_BINDING_transforms_type_s = npe::detail::transform_typechar(static_cast<pybind11::array&>(transforms).dtype().type());
ssize_t transforms_shape_0 = 0;
ssize_t transforms_shape_1 = 0;
if (static_cast<pybind11::array&>(transforms).ndim() == 1) {
transforms_shape_0 = static_cast<pybind11::array&>(transforms).shape()[0];
transforms_shape_1 = static_cast<pybind11::array&>(transforms).shape()[0] == 0 ? 0 : 1;
} else if (static_cast<pybind11::array&>(transforms).ndim() == 2) {
transforms_shape_0 = static_cast<pybind11::array&>(transforms).shape()[0];
transforms_shape_1 = static_cast<pybind11::array&>(transforms).shape()[1];
} else if (static_cast<pybind11::array&>(transforms).ndim() > 2) {
throw std::invalid_argument("Argument transforms has invalid number of dimensions. Must be 1 or 2.");
}
const npe::detail::StorageOrder _NPE_PY_BINDING_transforms_so = (static_cast<pybind11::array&>(transforms).flags() & NPY_ARRAY_F_CONTIGUOUS) ? npe::detail::ColMajor : (static_cast<pybind11::array&>(transforms).flags() & NPY_ARRAY_C_CONTIGUOUS ? npe::detail::RowMajor : npe::detail::NoOrder);
const int _NPE_PY_BINDING_transforms_t_id = npe::detail::get_type_id(npe::detail::is_sparse<std::remove_reference<decltype(static_cast<pybind11::array&>(transforms))>::type>::value, _NPE_PY_BINDING_transforms_type_s, _NPE_PY_BINDING_transforms_so);
const char _NPE_PY_BINDING_columnIdx_type_s = npe::detail::transform_typechar(static_cast<pybind11::array&>(columnIdx).dtype().type());
ssize_t columnIdx_shape_0 = 0;
ssize_t columnIdx_shape_1 = 0;
if (static_cast<pybind11::array&>(columnIdx).ndim() == 1) {
columnIdx_shape_0 = static_cast<pybind11::array&>(columnIdx).shape()[0];
columnIdx_shape_1 = static_cast<pybind11::array&>(columnIdx).shape()[0] == 0 ? 0 : 1;
} else if (static_cast<pybind11::array&>(columnIdx).ndim() == 2) {
columnIdx_shape_0 = static_cast<pybind11::array&>(columnIdx).shape()[0];
columnIdx_shape_1 = static_cast<pybind11::array&>(columnIdx).shape()[1];
} else if (static_cast<pybind11::array&>(columnIdx).ndim() > 2) {
throw std::invalid_argument("Argument columnIdx has invalid number of dimensions. Must be 1 or 2.");
}
const npe::detail::StorageOrder _NPE_PY_BINDING_columnIdx_so = (static_cast<pybind11::array&>(columnIdx).flags() & NPY_ARRAY_F_CONTIGUOUS) ? npe::detail::ColMajor : (static_cast<pybind11::array&>(columnIdx).flags() & NPY_ARRAY_C_CONTIGUOUS ? npe::detail::RowMajor : npe::detail::NoOrder);
const int _NPE_PY_BINDING_columnIdx_t_id = npe::detail::get_type_id(npe::detail::is_sparse<std::remove_reference<decltype(static_cast<pybind11::array&>(columnIdx))>::type>::value, _NPE_PY_BINDING_columnIdx_type_s, _NPE_PY_BINDING_columnIdx_so);
if (_NPE_PY_BINDING_pointCloud_type_s!= npe::detail::transform_typechar( npe::detail::NumpyTypeChar::char_double)) {
std::string err_msg = std::string("Invalid scalar type (") + npe::detail::type_to_str(_NPE_PY_BINDING_pointCloud_type_s) + ", " + npe::detail::storage_order_to_str(_NPE_PY_BINDING_pointCloud_so) + std::string(") for argument 'pointCloud'. Expected one of ['float64'].");
throw std::invalid_argument(err_msg);
}
{
int group_matched_type_id = _NPE_PY_BINDING_pointCloud_t_id;
bool found_non_1d = (pointCloud_shape_0 != 1 && pointCloud_shape_1 != 1 && pointCloud_shape_0 != 0 && pointCloud_shape_1 != 0);
std::string match_to_name = "pointCloud";
npe::detail::StorageOrder match_so = _NPE_PY_BINDING_pointCloud_so;
char group_type_s = _NPE_PY_BINDING_pointCloud_type_s;
if (pointCloud_shape_0 != 1 && pointCloud_shape_1 != 1 && pointCloud_shape_0 != 0 && pointCloud_shape_1 != 0) {
if (!found_non_1d) {
group_matched_type_id = _NPE_PY_BINDING_pointCloud_t_id;
found_non_1d = true;
match_to_name = "pointCloud";
match_so = _NPE_PY_BINDING_pointCloud_so;
group_type_s = _NPE_PY_BINDING_pointCloud_type_s;

}
if (_NPE_PY_BINDING_pointCloud_t_id != group_matched_type_id) {
std::string err_msg = std::string("Invalid type (") + npe::detail::type_to_str(_NPE_PY_BINDING_pointCloud_type_s) + ", " + npe::detail::storage_order_to_str(_NPE_PY_BINDING_pointCloud_so) + std::string(") for argument 'pointCloud'. Expected it to match argument '") + match_to_name + std::string("' which is of type (") + npe::detail::type_to_str(group_type_s) + ", " + npe::detail::storage_order_to_str(match_so) + std::string(").");
throw std::invalid_argument(err_msg);
}
} else if (group_type_s != _NPE_PY_BINDING_pointCloud_type_s) {
std::string err_msg = std::string("Invalid type (") + npe::detail::type_to_str(_NPE_PY_BINDING_pointCloud_type_s) + ", " + npe::detail::storage_order_to_str(match_so) + std::string(") for argument 'pointCloud'. Expected it to match argument '") + match_to_name + std::string("' which is of type (") + npe::detail::type_to_str(group_type_s) + ", " +  npe::detail::storage_order_to_str(match_so) + std::string(").");
throw std::invalid_argument(err_msg);
}
if (transforms_shape_0 != 1 && transforms_shape_1 != 1 && transforms_shape_0 != 0 && transforms_shape_1 != 0) {
if (!found_non_1d) {
group_matched_type_id = _NPE_PY_BINDING_transforms_t_id;
found_non_1d = true;
match_to_name = "transforms";
match_so = _NPE_PY_BINDING_transforms_so;
group_type_s = _NPE_PY_BINDING_transforms_type_s;

}
if (_NPE_PY_BINDING_transforms_t_id != group_matched_type_id) {
std::string err_msg = std::string("Invalid type (") + npe::detail::type_to_str(_NPE_PY_BINDING_transforms_type_s) + ", " + npe::detail::storage_order_to_str(_NPE_PY_BINDING_transforms_so) + std::string(") for argument 'transforms'. Expected it to match argument '") + match_to_name + std::string("' which is of type (") + npe::detail::type_to_str(group_type_s) + ", " + npe::detail::storage_order_to_str(match_so) + std::string(").");
throw std::invalid_argument(err_msg);
}
} else if (group_type_s != _NPE_PY_BINDING_transforms_type_s) {
std::string err_msg = std::string("Invalid type (") + npe::detail::type_to_str(_NPE_PY_BINDING_transforms_type_s) + ", " + npe::detail::storage_order_to_str(match_so) + std::string(") for argument 'transforms'. Expected it to match argument '") + match_to_name + std::string("' which is of type (") + npe::detail::type_to_str(group_type_s) + ", " +  npe::detail::storage_order_to_str(match_so) + std::string(").");
throw std::invalid_argument(err_msg);
}
}
if (_NPE_PY_BINDING_columnIdx_type_s!= npe::detail::transform_typechar( npe::detail::NumpyTypeChar::char_long)) {
std::string err_msg = std::string("Invalid scalar type (") + npe::detail::type_to_str(_NPE_PY_BINDING_columnIdx_type_s) + ", " + npe::detail::storage_order_to_str(_NPE_PY_BINDING_columnIdx_so) + std::string(") for argument 'columnIdx'. Expected one of ['int64'].");
throw std::invalid_argument(err_msg);
}
if (_NPE_PY_BINDING_pointCloud_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_columnIdx_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_long_cm)) {
{
typedef npy_double Scalar_pointCloud;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pointCloud;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pointCloud;
typedef npy_double Scalar_transforms;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_transforms;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_transforms;
typedef npy_long Scalar_columnIdx;
typedef Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_columnIdx;
typedef Eigen::Map<Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_columnIdx;
return callit__lidarUnwinding<Map_pointCloud, Matrix_pointCloud, Scalar_pointCloud,Map_transforms, Matrix_transforms, Scalar_transforms,Map_columnIdx, Matrix_columnIdx, Scalar_columnIdx>(Map_pointCloud((Scalar_pointCloud*) static_cast<pybind11::array&>(pointCloud).data(), pointCloud_shape_0, pointCloud_shape_1),Map_transforms((Scalar_transforms*) static_cast<pybind11::array&>(transforms).data(), transforms_shape_0, transforms_shape_1),Map_columnIdx((Scalar_columnIdx*) static_cast<pybind11::array&>(columnIdx).data(), columnIdx_shape_0, columnIdx_shape_1));
}
} else if (_NPE_PY_BINDING_pointCloud_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_columnIdx_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_long_rm)) {
{
typedef npy_double Scalar_pointCloud;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pointCloud;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pointCloud;
typedef npy_double Scalar_transforms;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_transforms;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_transforms;
typedef npy_long Scalar_columnIdx;
typedef Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_columnIdx;
typedef Eigen::Map<Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_columnIdx;
return callit__lidarUnwinding<Map_pointCloud, Matrix_pointCloud, Scalar_pointCloud,Map_transforms, Matrix_transforms, Scalar_transforms,Map_columnIdx, Matrix_columnIdx, Scalar_columnIdx>(Map_pointCloud((Scalar_pointCloud*) static_cast<pybind11::array&>(pointCloud).data(), pointCloud_shape_0, pointCloud_shape_1),Map_transforms((Scalar_transforms*) static_cast<pybind11::array&>(transforms).data(), transforms_shape_0, transforms_shape_1),Map_columnIdx((Scalar_columnIdx*) static_cast<pybind11::array&>(columnIdx).data(), columnIdx_shape_0, columnIdx_shape_1));
}
} else if (_NPE_PY_BINDING_pointCloud_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_columnIdx_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_long_x)) {
{
typedef npy_double Scalar_pointCloud;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pointCloud;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pointCloud;
typedef npy_double Scalar_transforms;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_transforms;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_transforms;
typedef npy_long Scalar_columnIdx;
typedef Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_columnIdx;
Eigen::Index columnIdx_inner_stride = 0;
Eigen::Index columnIdx_outer_stride = 0;
if (columnIdx.ndim() == 1) {
columnIdx_outer_stride = columnIdx.strides(0) / sizeof(npy_long);
} else if (columnIdx.ndim() == 2) {
columnIdx_outer_stride = columnIdx.strides(1) / sizeof(npy_long);
columnIdx_inner_stride = columnIdx.strides(0) / sizeof(npy_long);
}typedef Eigen::Map<Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_columnIdx;
return callit__lidarUnwinding<Map_pointCloud, Matrix_pointCloud, Scalar_pointCloud,Map_transforms, Matrix_transforms, Scalar_transforms,Map_columnIdx, Matrix_columnIdx, Scalar_columnIdx>(Map_pointCloud((Scalar_pointCloud*) static_cast<pybind11::array&>(pointCloud).data(), pointCloud_shape_0, pointCloud_shape_1),Map_transforms((Scalar_transforms*) static_cast<pybind11::array&>(transforms).data(), transforms_shape_0, transforms_shape_1),Map_columnIdx((Scalar_columnIdx*) static_cast<pybind11::array&>(columnIdx).data(), columnIdx_shape_0, columnIdx_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(columnIdx_outer_stride, columnIdx_inner_stride)));
}
} else if (_NPE_PY_BINDING_pointCloud_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_columnIdx_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_long_cm)) {
{
typedef npy_double Scalar_pointCloud;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pointCloud;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pointCloud;
typedef npy_double Scalar_transforms;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_transforms;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_transforms;
typedef npy_long Scalar_columnIdx;
typedef Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_columnIdx;
typedef Eigen::Map<Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_columnIdx;
return callit__lidarUnwinding<Map_pointCloud, Matrix_pointCloud, Scalar_pointCloud,Map_transforms, Matrix_transforms, Scalar_transforms,Map_columnIdx, Matrix_columnIdx, Scalar_columnIdx>(Map_pointCloud((Scalar_pointCloud*) static_cast<pybind11::array&>(pointCloud).data(), pointCloud_shape_0, pointCloud_shape_1),Map_transforms((Scalar_transforms*) static_cast<pybind11::array&>(transforms).data(), transforms_shape_0, transforms_shape_1),Map_columnIdx((Scalar_columnIdx*) static_cast<pybind11::array&>(columnIdx).data(), columnIdx_shape_0, columnIdx_shape_1));
}
} else if (_NPE_PY_BINDING_pointCloud_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_columnIdx_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_long_rm)) {
{
typedef npy_double Scalar_pointCloud;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pointCloud;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pointCloud;
typedef npy_double Scalar_transforms;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_transforms;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_transforms;
typedef npy_long Scalar_columnIdx;
typedef Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_columnIdx;
typedef Eigen::Map<Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_columnIdx;
return callit__lidarUnwinding<Map_pointCloud, Matrix_pointCloud, Scalar_pointCloud,Map_transforms, Matrix_transforms, Scalar_transforms,Map_columnIdx, Matrix_columnIdx, Scalar_columnIdx>(Map_pointCloud((Scalar_pointCloud*) static_cast<pybind11::array&>(pointCloud).data(), pointCloud_shape_0, pointCloud_shape_1),Map_transforms((Scalar_transforms*) static_cast<pybind11::array&>(transforms).data(), transforms_shape_0, transforms_shape_1),Map_columnIdx((Scalar_columnIdx*) static_cast<pybind11::array&>(columnIdx).data(), columnIdx_shape_0, columnIdx_shape_1));
}
} else if (_NPE_PY_BINDING_pointCloud_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_columnIdx_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_long_x)) {
{
typedef npy_double Scalar_pointCloud;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pointCloud;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pointCloud;
typedef npy_double Scalar_transforms;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_transforms;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_transforms;
typedef npy_long Scalar_columnIdx;
typedef Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_columnIdx;
Eigen::Index columnIdx_inner_stride = 0;
Eigen::Index columnIdx_outer_stride = 0;
if (columnIdx.ndim() == 1) {
columnIdx_outer_stride = columnIdx.strides(0) / sizeof(npy_long);
} else if (columnIdx.ndim() == 2) {
columnIdx_outer_stride = columnIdx.strides(1) / sizeof(npy_long);
columnIdx_inner_stride = columnIdx.strides(0) / sizeof(npy_long);
}typedef Eigen::Map<Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_columnIdx;
return callit__lidarUnwinding<Map_pointCloud, Matrix_pointCloud, Scalar_pointCloud,Map_transforms, Matrix_transforms, Scalar_transforms,Map_columnIdx, Matrix_columnIdx, Scalar_columnIdx>(Map_pointCloud((Scalar_pointCloud*) static_cast<pybind11::array&>(pointCloud).data(), pointCloud_shape_0, pointCloud_shape_1),Map_transforms((Scalar_transforms*) static_cast<pybind11::array&>(transforms).data(), transforms_shape_0, transforms_shape_1),Map_columnIdx((Scalar_columnIdx*) static_cast<pybind11::array&>(columnIdx).data(), columnIdx_shape_0, columnIdx_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(columnIdx_outer_stride, columnIdx_inner_stride)));
}
} else if (_NPE_PY_BINDING_pointCloud_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_columnIdx_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_long_cm)) {
{
typedef npy_double Scalar_pointCloud;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pointCloud;
Eigen::Index pointCloud_inner_stride = 0;
Eigen::Index pointCloud_outer_stride = 0;
if (pointCloud.ndim() == 1) {
pointCloud_outer_stride = pointCloud.strides(0) / sizeof(npy_double);
} else if (pointCloud.ndim() == 2) {
pointCloud_outer_stride = pointCloud.strides(1) / sizeof(npy_double);
pointCloud_inner_stride = pointCloud.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pointCloud;
typedef npy_double Scalar_transforms;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_transforms;
Eigen::Index transforms_inner_stride = 0;
Eigen::Index transforms_outer_stride = 0;
if (transforms.ndim() == 1) {
transforms_outer_stride = transforms.strides(0) / sizeof(npy_double);
} else if (transforms.ndim() == 2) {
transforms_outer_stride = transforms.strides(1) / sizeof(npy_double);
transforms_inner_stride = transforms.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_transforms;
typedef npy_long Scalar_columnIdx;
typedef Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_columnIdx;
typedef Eigen::Map<Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_columnIdx;
return callit__lidarUnwinding<Map_pointCloud, Matrix_pointCloud, Scalar_pointCloud,Map_transforms, Matrix_transforms, Scalar_transforms,Map_columnIdx, Matrix_columnIdx, Scalar_columnIdx>(Map_pointCloud((Scalar_pointCloud*) static_cast<pybind11::array&>(pointCloud).data(), pointCloud_shape_0, pointCloud_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pointCloud_outer_stride, pointCloud_inner_stride)),Map_transforms((Scalar_transforms*) static_cast<pybind11::array&>(transforms).data(), transforms_shape_0, transforms_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(transforms_outer_stride, transforms_inner_stride)),Map_columnIdx((Scalar_columnIdx*) static_cast<pybind11::array&>(columnIdx).data(), columnIdx_shape_0, columnIdx_shape_1));
}
} else if (_NPE_PY_BINDING_pointCloud_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_columnIdx_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_long_rm)) {
{
typedef npy_double Scalar_pointCloud;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pointCloud;
Eigen::Index pointCloud_inner_stride = 0;
Eigen::Index pointCloud_outer_stride = 0;
if (pointCloud.ndim() == 1) {
pointCloud_outer_stride = pointCloud.strides(0) / sizeof(npy_double);
} else if (pointCloud.ndim() == 2) {
pointCloud_outer_stride = pointCloud.strides(1) / sizeof(npy_double);
pointCloud_inner_stride = pointCloud.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pointCloud;
typedef npy_double Scalar_transforms;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_transforms;
Eigen::Index transforms_inner_stride = 0;
Eigen::Index transforms_outer_stride = 0;
if (transforms.ndim() == 1) {
transforms_outer_stride = transforms.strides(0) / sizeof(npy_double);
} else if (transforms.ndim() == 2) {
transforms_outer_stride = transforms.strides(1) / sizeof(npy_double);
transforms_inner_stride = transforms.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_transforms;
typedef npy_long Scalar_columnIdx;
typedef Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_columnIdx;
typedef Eigen::Map<Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_columnIdx;
return callit__lidarUnwinding<Map_pointCloud, Matrix_pointCloud, Scalar_pointCloud,Map_transforms, Matrix_transforms, Scalar_transforms,Map_columnIdx, Matrix_columnIdx, Scalar_columnIdx>(Map_pointCloud((Scalar_pointCloud*) static_cast<pybind11::array&>(pointCloud).data(), pointCloud_shape_0, pointCloud_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pointCloud_outer_stride, pointCloud_inner_stride)),Map_transforms((Scalar_transforms*) static_cast<pybind11::array&>(transforms).data(), transforms_shape_0, transforms_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(transforms_outer_stride, transforms_inner_stride)),Map_columnIdx((Scalar_columnIdx*) static_cast<pybind11::array&>(columnIdx).data(), columnIdx_shape_0, columnIdx_shape_1));
}
} else if (_NPE_PY_BINDING_pointCloud_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_columnIdx_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_long_x)) {
{
typedef npy_double Scalar_pointCloud;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pointCloud;
Eigen::Index pointCloud_inner_stride = 0;
Eigen::Index pointCloud_outer_stride = 0;
if (pointCloud.ndim() == 1) {
pointCloud_outer_stride = pointCloud.strides(0) / sizeof(npy_double);
} else if (pointCloud.ndim() == 2) {
pointCloud_outer_stride = pointCloud.strides(1) / sizeof(npy_double);
pointCloud_inner_stride = pointCloud.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pointCloud;
typedef npy_double Scalar_transforms;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_transforms;
Eigen::Index transforms_inner_stride = 0;
Eigen::Index transforms_outer_stride = 0;
if (transforms.ndim() == 1) {
transforms_outer_stride = transforms.strides(0) / sizeof(npy_double);
} else if (transforms.ndim() == 2) {
transforms_outer_stride = transforms.strides(1) / sizeof(npy_double);
transforms_inner_stride = transforms.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_transforms;
typedef npy_long Scalar_columnIdx;
typedef Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_columnIdx;
Eigen::Index columnIdx_inner_stride = 0;
Eigen::Index columnIdx_outer_stride = 0;
if (columnIdx.ndim() == 1) {
columnIdx_outer_stride = columnIdx.strides(0) / sizeof(npy_long);
} else if (columnIdx.ndim() == 2) {
columnIdx_outer_stride = columnIdx.strides(1) / sizeof(npy_long);
columnIdx_inner_stride = columnIdx.strides(0) / sizeof(npy_long);
}typedef Eigen::Map<Eigen::Matrix<npy_long, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_columnIdx;
return callit__lidarUnwinding<Map_pointCloud, Matrix_pointCloud, Scalar_pointCloud,Map_transforms, Matrix_transforms, Scalar_transforms,Map_columnIdx, Matrix_columnIdx, Scalar_columnIdx>(Map_pointCloud((Scalar_pointCloud*) static_cast<pybind11::array&>(pointCloud).data(), pointCloud_shape_0, pointCloud_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pointCloud_outer_stride, pointCloud_inner_stride)),Map_transforms((Scalar_transforms*) static_cast<pybind11::array&>(transforms).data(), transforms_shape_0, transforms_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(transforms_outer_stride, transforms_inner_stride)),Map_columnIdx((Scalar_columnIdx*) static_cast<pybind11::array&>(columnIdx).data(), columnIdx_shape_0, columnIdx_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(columnIdx_outer_stride, columnIdx_inner_stride)));
}
} else {
throw std::invalid_argument("This should never happen but clearly it did. File a github issue at https://github.com/fwilliams/numpyeigen");
}

}, lidarUnwinding_doc, pybind11::arg("pointCloud"), pybind11::arg("transforms"), pybind11::arg("columnIdx"));
m.def("_pixel2WorldRay", [](pybind11::array pixelCoordinates, pybind11::array cameraIntrinsic, std::string cameraModel, double imageHeight, double rollingShutterDelay, double pixelExposureTime, double eofTimestamp, pybind11::array poses, pybind11::array poseTimestamps) {
#ifdef __NPE_REDIRECT_IO__
pybind11::scoped_ostream_redirect __npe_redirect_stdout__(std::cout, pybind11::module::import("sys").attr("stdout"));
pybind11::scoped_ostream_redirect __npe_redirect_stderr__(std::cerr, pybind11::module::import("sys").attr("stderr"));
#endif
const char _NPE_PY_BINDING_pixelCoordinates_type_s = npe::detail::transform_typechar(static_cast<pybind11::array&>(pixelCoordinates).dtype().type());
ssize_t pixelCoordinates_shape_0 = 0;
ssize_t pixelCoordinates_shape_1 = 0;
if (static_cast<pybind11::array&>(pixelCoordinates).ndim() == 1) {
pixelCoordinates_shape_0 = static_cast<pybind11::array&>(pixelCoordinates).shape()[0];
pixelCoordinates_shape_1 = static_cast<pybind11::array&>(pixelCoordinates).shape()[0] == 0 ? 0 : 1;
} else if (static_cast<pybind11::array&>(pixelCoordinates).ndim() == 2) {
pixelCoordinates_shape_0 = static_cast<pybind11::array&>(pixelCoordinates).shape()[0];
pixelCoordinates_shape_1 = static_cast<pybind11::array&>(pixelCoordinates).shape()[1];
} else if (static_cast<pybind11::array&>(pixelCoordinates).ndim() > 2) {
throw std::invalid_argument("Argument pixelCoordinates has invalid number of dimensions. Must be 1 or 2.");
}
const npe::detail::StorageOrder _NPE_PY_BINDING_pixelCoordinates_so = (static_cast<pybind11::array&>(pixelCoordinates).flags() & NPY_ARRAY_F_CONTIGUOUS) ? npe::detail::ColMajor : (static_cast<pybind11::array&>(pixelCoordinates).flags() & NPY_ARRAY_C_CONTIGUOUS ? npe::detail::RowMajor : npe::detail::NoOrder);
const int _NPE_PY_BINDING_pixelCoordinates_t_id = npe::detail::get_type_id(npe::detail::is_sparse<std::remove_reference<decltype(static_cast<pybind11::array&>(pixelCoordinates))>::type>::value, _NPE_PY_BINDING_pixelCoordinates_type_s, _NPE_PY_BINDING_pixelCoordinates_so);
const char _NPE_PY_BINDING_cameraIntrinsic_type_s = npe::detail::transform_typechar(static_cast<pybind11::array&>(cameraIntrinsic).dtype().type());
ssize_t cameraIntrinsic_shape_0 = 0;
ssize_t cameraIntrinsic_shape_1 = 0;
if (static_cast<pybind11::array&>(cameraIntrinsic).ndim() == 1) {
cameraIntrinsic_shape_0 = static_cast<pybind11::array&>(cameraIntrinsic).shape()[0];
cameraIntrinsic_shape_1 = static_cast<pybind11::array&>(cameraIntrinsic).shape()[0] == 0 ? 0 : 1;
} else if (static_cast<pybind11::array&>(cameraIntrinsic).ndim() == 2) {
cameraIntrinsic_shape_0 = static_cast<pybind11::array&>(cameraIntrinsic).shape()[0];
cameraIntrinsic_shape_1 = static_cast<pybind11::array&>(cameraIntrinsic).shape()[1];
} else if (static_cast<pybind11::array&>(cameraIntrinsic).ndim() > 2) {
throw std::invalid_argument("Argument cameraIntrinsic has invalid number of dimensions. Must be 1 or 2.");
}
const npe::detail::StorageOrder _NPE_PY_BINDING_cameraIntrinsic_so = (static_cast<pybind11::array&>(cameraIntrinsic).flags() & NPY_ARRAY_F_CONTIGUOUS) ? npe::detail::ColMajor : (static_cast<pybind11::array&>(cameraIntrinsic).flags() & NPY_ARRAY_C_CONTIGUOUS ? npe::detail::RowMajor : npe::detail::NoOrder);
const int _NPE_PY_BINDING_cameraIntrinsic_t_id = npe::detail::get_type_id(npe::detail::is_sparse<std::remove_reference<decltype(static_cast<pybind11::array&>(cameraIntrinsic))>::type>::value, _NPE_PY_BINDING_cameraIntrinsic_type_s, _NPE_PY_BINDING_cameraIntrinsic_so);
const char _NPE_PY_BINDING_poses_type_s = npe::detail::transform_typechar(static_cast<pybind11::array&>(poses).dtype().type());
ssize_t poses_shape_0 = 0;
ssize_t poses_shape_1 = 0;
if (static_cast<pybind11::array&>(poses).ndim() == 1) {
poses_shape_0 = static_cast<pybind11::array&>(poses).shape()[0];
poses_shape_1 = static_cast<pybind11::array&>(poses).shape()[0] == 0 ? 0 : 1;
} else if (static_cast<pybind11::array&>(poses).ndim() == 2) {
poses_shape_0 = static_cast<pybind11::array&>(poses).shape()[0];
poses_shape_1 = static_cast<pybind11::array&>(poses).shape()[1];
} else if (static_cast<pybind11::array&>(poses).ndim() > 2) {
throw std::invalid_argument("Argument poses has invalid number of dimensions. Must be 1 or 2.");
}
const npe::detail::StorageOrder _NPE_PY_BINDING_poses_so = (static_cast<pybind11::array&>(poses).flags() & NPY_ARRAY_F_CONTIGUOUS) ? npe::detail::ColMajor : (static_cast<pybind11::array&>(poses).flags() & NPY_ARRAY_C_CONTIGUOUS ? npe::detail::RowMajor : npe::detail::NoOrder);
const int _NPE_PY_BINDING_poses_t_id = npe::detail::get_type_id(npe::detail::is_sparse<std::remove_reference<decltype(static_cast<pybind11::array&>(poses))>::type>::value, _NPE_PY_BINDING_poses_type_s, _NPE_PY_BINDING_poses_so);
const char _NPE_PY_BINDING_poseTimestamps_type_s = npe::detail::transform_typechar(static_cast<pybind11::array&>(poseTimestamps).dtype().type());
ssize_t poseTimestamps_shape_0 = 0;
ssize_t poseTimestamps_shape_1 = 0;
if (static_cast<pybind11::array&>(poseTimestamps).ndim() == 1) {
poseTimestamps_shape_0 = static_cast<pybind11::array&>(poseTimestamps).shape()[0];
poseTimestamps_shape_1 = static_cast<pybind11::array&>(poseTimestamps).shape()[0] == 0 ? 0 : 1;
} else if (static_cast<pybind11::array&>(poseTimestamps).ndim() == 2) {
poseTimestamps_shape_0 = static_cast<pybind11::array&>(poseTimestamps).shape()[0];
poseTimestamps_shape_1 = static_cast<pybind11::array&>(poseTimestamps).shape()[1];
} else if (static_cast<pybind11::array&>(poseTimestamps).ndim() > 2) {
throw std::invalid_argument("Argument poseTimestamps has invalid number of dimensions. Must be 1 or 2.");
}
const npe::detail::StorageOrder _NPE_PY_BINDING_poseTimestamps_so = (static_cast<pybind11::array&>(poseTimestamps).flags() & NPY_ARRAY_F_CONTIGUOUS) ? npe::detail::ColMajor : (static_cast<pybind11::array&>(poseTimestamps).flags() & NPY_ARRAY_C_CONTIGUOUS ? npe::detail::RowMajor : npe::detail::NoOrder);
const int _NPE_PY_BINDING_poseTimestamps_t_id = npe::detail::get_type_id(npe::detail::is_sparse<std::remove_reference<decltype(static_cast<pybind11::array&>(poseTimestamps))>::type>::value, _NPE_PY_BINDING_poseTimestamps_type_s, _NPE_PY_BINDING_poseTimestamps_so);
if (_NPE_PY_BINDING_pixelCoordinates_type_s!= npe::detail::transform_typechar( npe::detail::NumpyTypeChar::char_double)) {
std::string err_msg = std::string("Invalid scalar type (") + npe::detail::type_to_str(_NPE_PY_BINDING_pixelCoordinates_type_s) + ", " + npe::detail::storage_order_to_str(_NPE_PY_BINDING_pixelCoordinates_so) + std::string(") for argument 'pixelCoordinates'. Expected one of ['float64'].");
throw std::invalid_argument(err_msg);
}
if (_NPE_PY_BINDING_cameraIntrinsic_type_s!= npe::detail::transform_typechar( npe::detail::NumpyTypeChar::char_double)) {
std::string err_msg = std::string("Invalid scalar type (") + npe::detail::type_to_str(_NPE_PY_BINDING_cameraIntrinsic_type_s) + ", " + npe::detail::storage_order_to_str(_NPE_PY_BINDING_cameraIntrinsic_so) + std::string(") for argument 'cameraIntrinsic'. Expected one of ['float64'].");
throw std::invalid_argument(err_msg);
}
if (_NPE_PY_BINDING_poses_type_s!= npe::detail::transform_typechar( npe::detail::NumpyTypeChar::char_double)) {
std::string err_msg = std::string("Invalid scalar type (") + npe::detail::type_to_str(_NPE_PY_BINDING_poses_type_s) + ", " + npe::detail::storage_order_to_str(_NPE_PY_BINDING_poses_so) + std::string(") for argument 'poses'. Expected one of ['float64'].");
throw std::invalid_argument(err_msg);
}
if (_NPE_PY_BINDING_poseTimestamps_type_s!= npe::detail::transform_typechar( npe::detail::NumpyTypeChar::char_double)) {
std::string err_msg = std::string("Invalid scalar type (") + npe::detail::type_to_str(_NPE_PY_BINDING_poseTimestamps_type_s) + ", " + npe::detail::storage_order_to_str(_NPE_PY_BINDING_poseTimestamps_so) + std::string(") for argument 'poseTimestamps'. Expected one of ['float64'].");
throw std::invalid_argument(err_msg);
}
if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_pixelCoordinates;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1));
}
} else if (_NPE_PY_BINDING_pixelCoordinates_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_pixelCoordinates;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_pixelCoordinates;
Eigen::Index pixelCoordinates_inner_stride = 0;
Eigen::Index pixelCoordinates_outer_stride = 0;
if (pixelCoordinates.ndim() == 1) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
} else if (pixelCoordinates.ndim() == 2) {
pixelCoordinates_outer_stride = pixelCoordinates.strides(1) / sizeof(npy_double);
pixelCoordinates_inner_stride = pixelCoordinates.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_pixelCoordinates;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__pixel2WorldRay<Map_pixelCoordinates, Matrix_pixelCoordinates, Scalar_pixelCoordinates,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_pixelCoordinates((Scalar_pixelCoordinates*) static_cast<pybind11::array&>(pixelCoordinates).data(), pixelCoordinates_shape_0, pixelCoordinates_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(pixelCoordinates_outer_stride, pixelCoordinates_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),cameraModel,imageHeight,rollingShutterDelay,pixelExposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)));
}
} else {
throw std::invalid_argument("This should never happen but clearly it did. File a github issue at https://github.com/fwilliams/numpyeigen");
}

}, pixel2WorldRay_doc, pybind11::arg("pixelCoordinates"), pybind11::arg("cameraIntrinsic"), pybind11::arg("cameraModel"), pybind11::arg("imageHeight"), pybind11::arg("rollingShutterDelay"), pybind11::arg("pixelExposureTime"), pybind11::arg("eofTimestamp"), pybind11::arg("poses"), pybind11::arg("poseTimestamps"));
m.def("_cameraRay2Pixel", [](pybind11::array cameraPoints, pybind11::array cameraIntrinsic, double imgWidth, double imgHeight, std::string cameraModel) {
#ifdef __NPE_REDIRECT_IO__
pybind11::scoped_ostream_redirect __npe_redirect_stdout__(std::cout, pybind11::module::import("sys").attr("stdout"));
pybind11::scoped_ostream_redirect __npe_redirect_stderr__(std::cerr, pybind11::module::import("sys").attr("stderr"));
#endif
const char _NPE_PY_BINDING_cameraPoints_type_s = npe::detail::transform_typechar(static_cast<pybind11::array&>(cameraPoints).dtype().type());
ssize_t cameraPoints_shape_0 = 0;
ssize_t cameraPoints_shape_1 = 0;
if (static_cast<pybind11::array&>(cameraPoints).ndim() == 1) {
cameraPoints_shape_0 = static_cast<pybind11::array&>(cameraPoints).shape()[0];
cameraPoints_shape_1 = static_cast<pybind11::array&>(cameraPoints).shape()[0] == 0 ? 0 : 1;
} else if (static_cast<pybind11::array&>(cameraPoints).ndim() == 2) {
cameraPoints_shape_0 = static_cast<pybind11::array&>(cameraPoints).shape()[0];
cameraPoints_shape_1 = static_cast<pybind11::array&>(cameraPoints).shape()[1];
} else if (static_cast<pybind11::array&>(cameraPoints).ndim() > 2) {
throw std::invalid_argument("Argument cameraPoints has invalid number of dimensions. Must be 1 or 2.");
}
const npe::detail::StorageOrder _NPE_PY_BINDING_cameraPoints_so = (static_cast<pybind11::array&>(cameraPoints).flags() & NPY_ARRAY_F_CONTIGUOUS) ? npe::detail::ColMajor : (static_cast<pybind11::array&>(cameraPoints).flags() & NPY_ARRAY_C_CONTIGUOUS ? npe::detail::RowMajor : npe::detail::NoOrder);
const int _NPE_PY_BINDING_cameraPoints_t_id = npe::detail::get_type_id(npe::detail::is_sparse<std::remove_reference<decltype(static_cast<pybind11::array&>(cameraPoints))>::type>::value, _NPE_PY_BINDING_cameraPoints_type_s, _NPE_PY_BINDING_cameraPoints_so);
const char _NPE_PY_BINDING_cameraIntrinsic_type_s = npe::detail::transform_typechar(static_cast<pybind11::array&>(cameraIntrinsic).dtype().type());
ssize_t cameraIntrinsic_shape_0 = 0;
ssize_t cameraIntrinsic_shape_1 = 0;
if (static_cast<pybind11::array&>(cameraIntrinsic).ndim() == 1) {
cameraIntrinsic_shape_0 = static_cast<pybind11::array&>(cameraIntrinsic).shape()[0];
cameraIntrinsic_shape_1 = static_cast<pybind11::array&>(cameraIntrinsic).shape()[0] == 0 ? 0 : 1;
} else if (static_cast<pybind11::array&>(cameraIntrinsic).ndim() == 2) {
cameraIntrinsic_shape_0 = static_cast<pybind11::array&>(cameraIntrinsic).shape()[0];
cameraIntrinsic_shape_1 = static_cast<pybind11::array&>(cameraIntrinsic).shape()[1];
} else if (static_cast<pybind11::array&>(cameraIntrinsic).ndim() > 2) {
throw std::invalid_argument("Argument cameraIntrinsic has invalid number of dimensions. Must be 1 or 2.");
}
const npe::detail::StorageOrder _NPE_PY_BINDING_cameraIntrinsic_so = (static_cast<pybind11::array&>(cameraIntrinsic).flags() & NPY_ARRAY_F_CONTIGUOUS) ? npe::detail::ColMajor : (static_cast<pybind11::array&>(cameraIntrinsic).flags() & NPY_ARRAY_C_CONTIGUOUS ? npe::detail::RowMajor : npe::detail::NoOrder);
const int _NPE_PY_BINDING_cameraIntrinsic_t_id = npe::detail::get_type_id(npe::detail::is_sparse<std::remove_reference<decltype(static_cast<pybind11::array&>(cameraIntrinsic))>::type>::value, _NPE_PY_BINDING_cameraIntrinsic_type_s, _NPE_PY_BINDING_cameraIntrinsic_so);
if (_NPE_PY_BINDING_cameraPoints_type_s!= npe::detail::transform_typechar( npe::detail::NumpyTypeChar::char_double)) {
std::string err_msg = std::string("Invalid scalar type (") + npe::detail::type_to_str(_NPE_PY_BINDING_cameraPoints_type_s) + ", " + npe::detail::storage_order_to_str(_NPE_PY_BINDING_cameraPoints_so) + std::string(") for argument 'cameraPoints'. Expected one of ['float64'].");
throw std::invalid_argument(err_msg);
}
if (_NPE_PY_BINDING_cameraIntrinsic_type_s!= npe::detail::transform_typechar( npe::detail::NumpyTypeChar::char_double)) {
std::string err_msg = std::string("Invalid scalar type (") + npe::detail::type_to_str(_NPE_PY_BINDING_cameraIntrinsic_type_s) + ", " + npe::detail::storage_order_to_str(_NPE_PY_BINDING_cameraIntrinsic_so) + std::string(") for argument 'cameraIntrinsic'. Expected one of ['float64'].");
throw std::invalid_argument(err_msg);
}
if (_NPE_PY_BINDING_cameraPoints_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_cameraPoints;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraPoints;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraPoints;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
return callit__cameraRay2Pixel<Map_cameraPoints, Matrix_cameraPoints, Scalar_cameraPoints,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic>(Map_cameraPoints((Scalar_cameraPoints*) static_cast<pybind11::array&>(cameraPoints).data(), cameraPoints_shape_0, cameraPoints_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgWidth,imgHeight,cameraModel);
}
} else if (_NPE_PY_BINDING_cameraPoints_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_cameraPoints;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraPoints;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraPoints;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
return callit__cameraRay2Pixel<Map_cameraPoints, Matrix_cameraPoints, Scalar_cameraPoints,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic>(Map_cameraPoints((Scalar_cameraPoints*) static_cast<pybind11::array&>(cameraPoints).data(), cameraPoints_shape_0, cameraPoints_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgWidth,imgHeight,cameraModel);
}
} else if (_NPE_PY_BINDING_cameraPoints_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_cameraPoints;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraPoints;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraPoints;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
return callit__cameraRay2Pixel<Map_cameraPoints, Matrix_cameraPoints, Scalar_cameraPoints,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic>(Map_cameraPoints((Scalar_cameraPoints*) static_cast<pybind11::array&>(cameraPoints).data(), cameraPoints_shape_0, cameraPoints_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgWidth,imgHeight,cameraModel);
}
} else if (_NPE_PY_BINDING_cameraPoints_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_cameraPoints;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraPoints;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraPoints;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
return callit__cameraRay2Pixel<Map_cameraPoints, Matrix_cameraPoints, Scalar_cameraPoints,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic>(Map_cameraPoints((Scalar_cameraPoints*) static_cast<pybind11::array&>(cameraPoints).data(), cameraPoints_shape_0, cameraPoints_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgWidth,imgHeight,cameraModel);
}
} else if (_NPE_PY_BINDING_cameraPoints_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_cameraPoints;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraPoints;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraPoints;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
return callit__cameraRay2Pixel<Map_cameraPoints, Matrix_cameraPoints, Scalar_cameraPoints,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic>(Map_cameraPoints((Scalar_cameraPoints*) static_cast<pybind11::array&>(cameraPoints).data(), cameraPoints_shape_0, cameraPoints_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgWidth,imgHeight,cameraModel);
}
} else if (_NPE_PY_BINDING_cameraPoints_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_cameraPoints;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraPoints;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraPoints;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
return callit__cameraRay2Pixel<Map_cameraPoints, Matrix_cameraPoints, Scalar_cameraPoints,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic>(Map_cameraPoints((Scalar_cameraPoints*) static_cast<pybind11::array&>(cameraPoints).data(), cameraPoints_shape_0, cameraPoints_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgWidth,imgHeight,cameraModel);
}
} else if (_NPE_PY_BINDING_cameraPoints_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_cameraPoints;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraPoints;
Eigen::Index cameraPoints_inner_stride = 0;
Eigen::Index cameraPoints_outer_stride = 0;
if (cameraPoints.ndim() == 1) {
cameraPoints_outer_stride = cameraPoints.strides(0) / sizeof(npy_double);
} else if (cameraPoints.ndim() == 2) {
cameraPoints_outer_stride = cameraPoints.strides(1) / sizeof(npy_double);
cameraPoints_inner_stride = cameraPoints.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraPoints;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
return callit__cameraRay2Pixel<Map_cameraPoints, Matrix_cameraPoints, Scalar_cameraPoints,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic>(Map_cameraPoints((Scalar_cameraPoints*) static_cast<pybind11::array&>(cameraPoints).data(), cameraPoints_shape_0, cameraPoints_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraPoints_outer_stride, cameraPoints_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgWidth,imgHeight,cameraModel);
}
} else if (_NPE_PY_BINDING_cameraPoints_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_cameraPoints;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraPoints;
Eigen::Index cameraPoints_inner_stride = 0;
Eigen::Index cameraPoints_outer_stride = 0;
if (cameraPoints.ndim() == 1) {
cameraPoints_outer_stride = cameraPoints.strides(0) / sizeof(npy_double);
} else if (cameraPoints.ndim() == 2) {
cameraPoints_outer_stride = cameraPoints.strides(1) / sizeof(npy_double);
cameraPoints_inner_stride = cameraPoints.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraPoints;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
return callit__cameraRay2Pixel<Map_cameraPoints, Matrix_cameraPoints, Scalar_cameraPoints,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic>(Map_cameraPoints((Scalar_cameraPoints*) static_cast<pybind11::array&>(cameraPoints).data(), cameraPoints_shape_0, cameraPoints_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraPoints_outer_stride, cameraPoints_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgWidth,imgHeight,cameraModel);
}
} else if (_NPE_PY_BINDING_cameraPoints_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_cameraPoints;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraPoints;
Eigen::Index cameraPoints_inner_stride = 0;
Eigen::Index cameraPoints_outer_stride = 0;
if (cameraPoints.ndim() == 1) {
cameraPoints_outer_stride = cameraPoints.strides(0) / sizeof(npy_double);
} else if (cameraPoints.ndim() == 2) {
cameraPoints_outer_stride = cameraPoints.strides(1) / sizeof(npy_double);
cameraPoints_inner_stride = cameraPoints.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraPoints;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
return callit__cameraRay2Pixel<Map_cameraPoints, Matrix_cameraPoints, Scalar_cameraPoints,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic>(Map_cameraPoints((Scalar_cameraPoints*) static_cast<pybind11::array&>(cameraPoints).data(), cameraPoints_shape_0, cameraPoints_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraPoints_outer_stride, cameraPoints_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgWidth,imgHeight,cameraModel);
}
} else {
throw std::invalid_argument("This should never happen but clearly it did. File a github issue at https://github.com/fwilliams/numpyeigen");
}

}, cameraRay2Pixel_doc, pybind11::arg("cameraPoints"), pybind11::arg("cameraIntrinsic"), pybind11::arg("imgWidth"), pybind11::arg("imgHeight"), pybind11::arg("cameraModel"));
m.def("_rollingShutterProjection", [](pybind11::array points, pybind11::array cameraIntrinsic, double imgHeight, double imgWidth, double rollingShutterDelay, double exposureTime, double eofTimestamp, pybind11::array poses, pybind11::array poseTimestamps, std::string cameraModel, bool iterate) {
#ifdef __NPE_REDIRECT_IO__
pybind11::scoped_ostream_redirect __npe_redirect_stdout__(std::cout, pybind11::module::import("sys").attr("stdout"));
pybind11::scoped_ostream_redirect __npe_redirect_stderr__(std::cerr, pybind11::module::import("sys").attr("stderr"));
#endif
const char _NPE_PY_BINDING_points_type_s = npe::detail::transform_typechar(static_cast<pybind11::array&>(points).dtype().type());
ssize_t points_shape_0 = 0;
ssize_t points_shape_1 = 0;
if (static_cast<pybind11::array&>(points).ndim() == 1) {
points_shape_0 = static_cast<pybind11::array&>(points).shape()[0];
points_shape_1 = static_cast<pybind11::array&>(points).shape()[0] == 0 ? 0 : 1;
} else if (static_cast<pybind11::array&>(points).ndim() == 2) {
points_shape_0 = static_cast<pybind11::array&>(points).shape()[0];
points_shape_1 = static_cast<pybind11::array&>(points).shape()[1];
} else if (static_cast<pybind11::array&>(points).ndim() > 2) {
throw std::invalid_argument("Argument points has invalid number of dimensions. Must be 1 or 2.");
}
const npe::detail::StorageOrder _NPE_PY_BINDING_points_so = (static_cast<pybind11::array&>(points).flags() & NPY_ARRAY_F_CONTIGUOUS) ? npe::detail::ColMajor : (static_cast<pybind11::array&>(points).flags() & NPY_ARRAY_C_CONTIGUOUS ? npe::detail::RowMajor : npe::detail::NoOrder);
const int _NPE_PY_BINDING_points_t_id = npe::detail::get_type_id(npe::detail::is_sparse<std::remove_reference<decltype(static_cast<pybind11::array&>(points))>::type>::value, _NPE_PY_BINDING_points_type_s, _NPE_PY_BINDING_points_so);
const char _NPE_PY_BINDING_cameraIntrinsic_type_s = npe::detail::transform_typechar(static_cast<pybind11::array&>(cameraIntrinsic).dtype().type());
ssize_t cameraIntrinsic_shape_0 = 0;
ssize_t cameraIntrinsic_shape_1 = 0;
if (static_cast<pybind11::array&>(cameraIntrinsic).ndim() == 1) {
cameraIntrinsic_shape_0 = static_cast<pybind11::array&>(cameraIntrinsic).shape()[0];
cameraIntrinsic_shape_1 = static_cast<pybind11::array&>(cameraIntrinsic).shape()[0] == 0 ? 0 : 1;
} else if (static_cast<pybind11::array&>(cameraIntrinsic).ndim() == 2) {
cameraIntrinsic_shape_0 = static_cast<pybind11::array&>(cameraIntrinsic).shape()[0];
cameraIntrinsic_shape_1 = static_cast<pybind11::array&>(cameraIntrinsic).shape()[1];
} else if (static_cast<pybind11::array&>(cameraIntrinsic).ndim() > 2) {
throw std::invalid_argument("Argument cameraIntrinsic has invalid number of dimensions. Must be 1 or 2.");
}
const npe::detail::StorageOrder _NPE_PY_BINDING_cameraIntrinsic_so = (static_cast<pybind11::array&>(cameraIntrinsic).flags() & NPY_ARRAY_F_CONTIGUOUS) ? npe::detail::ColMajor : (static_cast<pybind11::array&>(cameraIntrinsic).flags() & NPY_ARRAY_C_CONTIGUOUS ? npe::detail::RowMajor : npe::detail::NoOrder);
const int _NPE_PY_BINDING_cameraIntrinsic_t_id = npe::detail::get_type_id(npe::detail::is_sparse<std::remove_reference<decltype(static_cast<pybind11::array&>(cameraIntrinsic))>::type>::value, _NPE_PY_BINDING_cameraIntrinsic_type_s, _NPE_PY_BINDING_cameraIntrinsic_so);
const char _NPE_PY_BINDING_poses_type_s = npe::detail::transform_typechar(static_cast<pybind11::array&>(poses).dtype().type());
ssize_t poses_shape_0 = 0;
ssize_t poses_shape_1 = 0;
if (static_cast<pybind11::array&>(poses).ndim() == 1) {
poses_shape_0 = static_cast<pybind11::array&>(poses).shape()[0];
poses_shape_1 = static_cast<pybind11::array&>(poses).shape()[0] == 0 ? 0 : 1;
} else if (static_cast<pybind11::array&>(poses).ndim() == 2) {
poses_shape_0 = static_cast<pybind11::array&>(poses).shape()[0];
poses_shape_1 = static_cast<pybind11::array&>(poses).shape()[1];
} else if (static_cast<pybind11::array&>(poses).ndim() > 2) {
throw std::invalid_argument("Argument poses has invalid number of dimensions. Must be 1 or 2.");
}
const npe::detail::StorageOrder _NPE_PY_BINDING_poses_so = (static_cast<pybind11::array&>(poses).flags() & NPY_ARRAY_F_CONTIGUOUS) ? npe::detail::ColMajor : (static_cast<pybind11::array&>(poses).flags() & NPY_ARRAY_C_CONTIGUOUS ? npe::detail::RowMajor : npe::detail::NoOrder);
const int _NPE_PY_BINDING_poses_t_id = npe::detail::get_type_id(npe::detail::is_sparse<std::remove_reference<decltype(static_cast<pybind11::array&>(poses))>::type>::value, _NPE_PY_BINDING_poses_type_s, _NPE_PY_BINDING_poses_so);
const char _NPE_PY_BINDING_poseTimestamps_type_s = npe::detail::transform_typechar(static_cast<pybind11::array&>(poseTimestamps).dtype().type());
ssize_t poseTimestamps_shape_0 = 0;
ssize_t poseTimestamps_shape_1 = 0;
if (static_cast<pybind11::array&>(poseTimestamps).ndim() == 1) {
poseTimestamps_shape_0 = static_cast<pybind11::array&>(poseTimestamps).shape()[0];
poseTimestamps_shape_1 = static_cast<pybind11::array&>(poseTimestamps).shape()[0] == 0 ? 0 : 1;
} else if (static_cast<pybind11::array&>(poseTimestamps).ndim() == 2) {
poseTimestamps_shape_0 = static_cast<pybind11::array&>(poseTimestamps).shape()[0];
poseTimestamps_shape_1 = static_cast<pybind11::array&>(poseTimestamps).shape()[1];
} else if (static_cast<pybind11::array&>(poseTimestamps).ndim() > 2) {
throw std::invalid_argument("Argument poseTimestamps has invalid number of dimensions. Must be 1 or 2.");
}
const npe::detail::StorageOrder _NPE_PY_BINDING_poseTimestamps_so = (static_cast<pybind11::array&>(poseTimestamps).flags() & NPY_ARRAY_F_CONTIGUOUS) ? npe::detail::ColMajor : (static_cast<pybind11::array&>(poseTimestamps).flags() & NPY_ARRAY_C_CONTIGUOUS ? npe::detail::RowMajor : npe::detail::NoOrder);
const int _NPE_PY_BINDING_poseTimestamps_t_id = npe::detail::get_type_id(npe::detail::is_sparse<std::remove_reference<decltype(static_cast<pybind11::array&>(poseTimestamps))>::type>::value, _NPE_PY_BINDING_poseTimestamps_type_s, _NPE_PY_BINDING_poseTimestamps_so);
if (_NPE_PY_BINDING_points_type_s!= npe::detail::transform_typechar( npe::detail::NumpyTypeChar::char_double)) {
std::string err_msg = std::string("Invalid scalar type (") + npe::detail::type_to_str(_NPE_PY_BINDING_points_type_s) + ", " + npe::detail::storage_order_to_str(_NPE_PY_BINDING_points_so) + std::string(") for argument 'points'. Expected one of ['float64'].");
throw std::invalid_argument(err_msg);
}
if (_NPE_PY_BINDING_cameraIntrinsic_type_s!= npe::detail::transform_typechar( npe::detail::NumpyTypeChar::char_double)) {
std::string err_msg = std::string("Invalid scalar type (") + npe::detail::type_to_str(_NPE_PY_BINDING_cameraIntrinsic_type_s) + ", " + npe::detail::storage_order_to_str(_NPE_PY_BINDING_cameraIntrinsic_so) + std::string(") for argument 'cameraIntrinsic'. Expected one of ['float64'].");
throw std::invalid_argument(err_msg);
}
if (_NPE_PY_BINDING_poses_type_s!= npe::detail::transform_typechar( npe::detail::NumpyTypeChar::char_double)) {
std::string err_msg = std::string("Invalid scalar type (") + npe::detail::type_to_str(_NPE_PY_BINDING_poses_type_s) + ", " + npe::detail::storage_order_to_str(_NPE_PY_BINDING_poses_so) + std::string(") for argument 'poses'. Expected one of ['float64'].");
throw std::invalid_argument(err_msg);
}
if (_NPE_PY_BINDING_poseTimestamps_type_s!= npe::detail::transform_typechar( npe::detail::NumpyTypeChar::char_double)) {
std::string err_msg = std::string("Invalid scalar type (") + npe::detail::type_to_str(_NPE_PY_BINDING_poseTimestamps_type_s) + ", " + npe::detail::storage_order_to_str(_NPE_PY_BINDING_poseTimestamps_so) + std::string(") for argument 'poseTimestamps'. Expected one of ['float64'].");
throw std::invalid_argument(err_msg);
}
if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_points;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_cameraIntrinsic;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poses;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_cm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::ColMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_rm)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor> Matrix_poseTimestamps;
typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::RowMajor>, npe::detail::Alignment::Aligned> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1),cameraModel,iterate);
}
} else if (_NPE_PY_BINDING_points_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_cameraIntrinsic_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poses_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x) && _NPE_PY_BINDING_poseTimestamps_t_id == npe::detail::transform_typeid(npe::detail::TypeId::dense_double_x)) {
{
typedef npy_double Scalar_points;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_points;
Eigen::Index points_inner_stride = 0;
Eigen::Index points_outer_stride = 0;
if (points.ndim() == 1) {
points_outer_stride = points.strides(0) / sizeof(npy_double);
} else if (points.ndim() == 2) {
points_outer_stride = points.strides(1) / sizeof(npy_double);
points_inner_stride = points.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_points;
typedef npy_double Scalar_cameraIntrinsic;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_cameraIntrinsic;
Eigen::Index cameraIntrinsic_inner_stride = 0;
Eigen::Index cameraIntrinsic_outer_stride = 0;
if (cameraIntrinsic.ndim() == 1) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
} else if (cameraIntrinsic.ndim() == 2) {
cameraIntrinsic_outer_stride = cameraIntrinsic.strides(1) / sizeof(npy_double);
cameraIntrinsic_inner_stride = cameraIntrinsic.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_cameraIntrinsic;
typedef npy_double Scalar_poses;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poses;
Eigen::Index poses_inner_stride = 0;
Eigen::Index poses_outer_stride = 0;
if (poses.ndim() == 1) {
poses_outer_stride = poses.strides(0) / sizeof(npy_double);
} else if (poses.ndim() == 2) {
poses_outer_stride = poses.strides(1) / sizeof(npy_double);
poses_inner_stride = poses.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poses;
typedef npy_double Scalar_poseTimestamps;
typedef Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder> Matrix_poseTimestamps;
Eigen::Index poseTimestamps_inner_stride = 0;
Eigen::Index poseTimestamps_outer_stride = 0;
if (poseTimestamps.ndim() == 1) {
poseTimestamps_outer_stride = poseTimestamps.strides(0) / sizeof(npy_double);
} else if (poseTimestamps.ndim() == 2) {
poseTimestamps_outer_stride = poseTimestamps.strides(1) / sizeof(npy_double);
poseTimestamps_inner_stride = poseTimestamps.strides(0) / sizeof(npy_double);
}typedef Eigen::Map<Eigen::Matrix<npy_double, Eigen::Dynamic, Eigen::Dynamic, npe::detail::StorageOrder::NoOrder>, npe::detail::Alignment::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_poseTimestamps;
return callit__rollingShutterProjection<Map_points, Matrix_points, Scalar_points,Map_cameraIntrinsic, Matrix_cameraIntrinsic, Scalar_cameraIntrinsic,Map_poses, Matrix_poses, Scalar_poses,Map_poseTimestamps, Matrix_poseTimestamps, Scalar_poseTimestamps>(Map_points((Scalar_points*) static_cast<pybind11::array&>(points).data(), points_shape_0, points_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(points_outer_stride, points_inner_stride)),Map_cameraIntrinsic((Scalar_cameraIntrinsic*) static_cast<pybind11::array&>(cameraIntrinsic).data(), cameraIntrinsic_shape_0, cameraIntrinsic_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cameraIntrinsic_outer_stride, cameraIntrinsic_inner_stride)),imgHeight,imgWidth,rollingShutterDelay,exposureTime,eofTimestamp,Map_poses((Scalar_poses*) static_cast<pybind11::array&>(poses).data(), poses_shape_0, poses_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poses_outer_stride, poses_inner_stride)),Map_poseTimestamps((Scalar_poseTimestamps*) static_cast<pybind11::array&>(poseTimestamps).data(), poseTimestamps_shape_0, poseTimestamps_shape_1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(poseTimestamps_outer_stride, poseTimestamps_inner_stride)),cameraModel,iterate);
}
} else {
throw std::invalid_argument("This should never happen but clearly it did. File a github issue at https://github.com/fwilliams/numpyeigen");
}

}, rollingShutterProjection_doc, pybind11::arg("points"), pybind11::arg("cameraIntrinsic"), pybind11::arg("imgHeight"), pybind11::arg("imgWidth"), pybind11::arg("rollingShutterDelay"), pybind11::arg("exposureTime"), pybind11::arg("eofTimestamp"), pybind11::arg("poses"), pybind11::arg("poseTimestamps"), pybind11::arg("cameraModel"), pybind11::arg("iterate"));
}

