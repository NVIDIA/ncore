#include "camera.hpp"
#include <Eigen/Dense>
#include <array>
#include <vector>

CameraModel::CameraModel(const int imgWidth_, const double imgHeight_, const Eigen::Matrix<double, 1,2>& principalPoint_, const std::string& shutterType_) 
: imgWidth(imgWidth_), imgHeight(imgHeight_), principalPoint(principalPoint_), shutterType(shutterType_) {}

CameraModel::~CameraModel(){}

void CameraModel::cameraToWorldRay(const Eigen::Matrix<double, Eigen::Dynamic, 2>& pixelCoordinates,
                        const Eigen::Matrix<double, Eigen::Dynamic, 3>& cameraRays, 
                        const Eigen::Matrix<double, 8, 4>& TSensorWorld, 
                        Eigen::Matrix<double, Eigen::Dynamic, 6>& worldRays)
{

    // Start pose
    Eigen::Matrix<double,3,3> startRotMat = TSensorWorld.block(0,0,3,3);
    Eigen::Quaterniond startRot(startRotMat);
    Eigen::Matrix<double,3,1> startTrans(TSensorWorld.block(0,3,3,1));

    // End pose
    Eigen::Matrix<double,3,3> endRotMat = TSensorWorld.block(4,0,3,3);
    Eigen::Quaterniond endRot(endRotMat);
    Eigen::Matrix<double,3,1> endTrans(TSensorWorld.block(4,3,3,1));

    // Iterate over the pixels and transform the rays to the world coordinate system
    for (int i = 0; i < cameraRays.rows(); i++)
    {
        int64_t projTimestamp{};
        if (shutterType == "ROLLING_TOP_TO_BOTTOM")
            {
                projTimestamp = std::floor(pixelCoordinates(i,1)) / (imgHeight - 1);
            }
            else if (shutterType == "ROLLING_LEFT_TO_RIGHT")
            {
                projTimestamp = std::floor(pixelCoordinates(i,0)) / (imgWidth - 1);
            }
            else if (shutterType == "ROLLING_BOTTOM_TO_TOP")
            {
                projTimestamp = (imgHeight - std::ceil(pixelCoordinates(i,1))) / (imgHeight - 1);
            }
            else if (shutterType == "ROLLING_RIGHT_TO_LEFT")
            {
                projTimestamp = (imgWidth - std::ceil(pixelCoordinates(i,0))) / (imgWidth - 1);
            }

        // Interpolate the pose
        Eigen::Matrix<double,3,1> translation = static_cast<double>(1 - projTimestamp) * startTrans.array() + static_cast<double>(projTimestamp) * endTrans.array();
        Eigen::Quaterniond Rot = startRot.slerp(static_cast<double>(projTimestamp), endRot);

        worldRays.block(i,0,1,3) = translation.transpose();
        worldRays.block(i,3,1,3) = (Rot.normalized().toRotationMatrix() * cameraRays.row(i).transpose()).normalized().transpose();
    } 
}

void CameraModel::rollingShutterProjection(const Eigen::Matrix<double, Eigen::Dynamic, 3>& points,
                                const Eigen::Matrix<double, 8, 4>& TWorldSensor, 
                                const int maxIter,
                                Eigen::Matrix<double, Eigen::Dynamic, 4>& transformationMatrices,
                                Eigen::Matrix<double, Eigen::Dynamic, 2>& pixelCoordinates,
                                Eigen::VectorXi& validProjec,
                                Eigen::VectorXi& initialValidIdx)
{
    Eigen::Matrix<double,3,3> startRotMat = TWorldSensor.block(0,0,3,3);
    Eigen::Quaterniond startRot(startRotMat);
    Eigen::Matrix<double,3,3> endRotMat = TWorldSensor.block(4,0,3,3);
    Eigen::Quaterniond endRot(endRotMat);

    Eigen::Matrix<double,3,1> startTrans = TWorldSensor.block(0,3,3,1);
    Eigen::Matrix<double,3,1> endTrans = TWorldSensor.block(4,3,3,1);

    // Interpolate the pose to mof timestamp  
    Eigen::Matrix<double,3,1> mofTrans = (1 - 0.5) * startTrans.array() + 0.5 * endTrans.array();
    Eigen::Quaterniond mofRot = startRot.slerp(0.5, endRot);

    // Convert the pixel coordinates to a ray in camera space
    Eigen::Matrix<double, Eigen::Dynamic, 3> camPoints;
    camPoints.resize(points.rows(), 3);

    camPoints = (mofRot.normalized().toRotationMatrix() * points.transpose()).transpose();
    camPoints.rowwise() += mofTrans.transpose();

    // Convert the pixel coordinates to a ray in camera space
    Eigen::Matrix<double, Eigen::Dynamic, 2 > initialProjections;
    initialProjections.resize(camPoints.rows(), 2);

    Eigen::Matrix<bool,Eigen::Dynamic, 1 > validFlag;
    validFlag.resize(camPoints.rows(), 1);

    // Project camera rays to pixels
    cameraRayToPixel(camPoints, initialProjections, validFlag);

    // Get the number of valid elements 
    auto nValid = validFlag.count();
    std::vector<int> initialValidIndices;

    // Get the indices of the points that were valid initially 
    for (int i = 0; i < initialProjections.rows(); i++)
    {
        if (validFlag(i,0))
            initialValidIndices.push_back(i);
    }

    // Resize the output parameters
    pixelCoordinates.resize(nValid, 2);
    transformationMatrices.resize(nValid*4, 4);
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

            Eigen::Matrix<bool, Eigen::Dynamic, 1> finalValidFlag;
            finalValidFlag.resize(1, 1);

            Eigen::Matrix<double,3,1> projTrans;
            Eigen::Quaterniond projRot = Eigen::Quaterniond::Identity();
            
            // Initialize the parameters
            double error = std::numeric_limits<double>::max();
            double eps = 1e-6;

            for (int iter_idx = 0; (iter_idx < maxIter) && (error > eps); iter_idx++)
            {
                int64_t projTimestamp{};
                if (shutterType == "ROLLING_TOP_TO_BOTTOM")
                {
                    projTimestamp = std::floor(y) / (imgHeight - 1);
                }
                else if (shutterType == "ROLLING_LEFT_TO_RIGHT")
                {
                    projTimestamp = std::floor(x) / (imgWidth - 1);
                }
                else if (shutterType == "ROLLING_BOTTOM_TO_TOP")
                {
                    projTimestamp = (imgHeight - std::ceil(y)) / (imgHeight - 1);
                }
                else if (shutterType == "ROLLING_RIGHT_TO_LEFT")
                {
                    projTimestamp = (imgWidth - std::ceil(x)) / (imgWidth - 1);
                }
                    
                // Interpolate the pose to the row index
                projTrans = static_cast<double>(1 - projTimestamp) * startTrans.array() + static_cast<double>(projTimestamp) * endTrans.array();
                projRot = startRot.slerp(projTimestamp, endRot);

                // Transform the point to the cam coordinate system
                Eigen::Matrix<double, 1, 3 > tmpCamPoint = (projRot.normalized().toRotationMatrix() * points.row(i).transpose() + projTrans).transpose();

                // Convert the pixel coordinates to a ray in camera space
                Eigen::Matrix<double, Eigen::Dynamic , 2 > tmpProjection;
                tmpProjection.resize(1, 2);
                cameraRayToPixel(tmpCamPoint, tmpProjection, finalValidFlag);

                // Compute the projection error between two iterations
                error = std::pow(tmpProjection(0,0) - x, 2) + std::pow(tmpProjection(0,1) - y, 2);
                
                // update the value
                x = tmpProjection(0,0);
                y = tmpProjection(0,1);
            }

            // Save the pixel coordinates
            pixelCoordinates.row(runIdx) << x, y;

            // Save the transformation
            transformationMatrices.block(4*runIdx, 0, 3, 3) = projRot.normalized().toRotationMatrix();
            transformationMatrices.block(4*runIdx, 3, 3, 1) = projTrans;
            transformationMatrices(4*runIdx + 3, 3) = 1.0;

            // Update the index
            if (finalValidFlag(0,0))
                validProjections.push_back(runIdx);
            
            runIdx++;
        }
    }

    validProjec = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(validProjections.data(), validProjections.size());
    initialValidIdx = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(initialValidIndices.data(), initialValidIndices.size());
}


FThetaCamera::FThetaCamera(const int imgWidth_,  const int imgHeight_, const Eigen::Matrix<double, 1,2>& principalPoint_,
                const std::array<double, 6>& fwPolyCoefficients_, const std::array<double, 6>& bwPolyCoefficients_,
                const double maxAngle_, const std::string shutterType_):
    CameraModel(imgWidth_, imgHeight_, principalPoint_, shutterType_), bwPolyCoefficients(bwPolyCoefficients_), 
                                    fwPolyCoefficients(fwPolyCoefficients_), maxAngle(maxAngle_) {}

FThetaCamera::~FThetaCamera(){}

void FThetaCamera::cameraRayToPixel(const Eigen::Matrix<double, Eigen::Dynamic, 3>& cameraPoints,
                        Eigen::Matrix<double, Eigen::Dynamic, 2>& imgPoints,
                        Eigen::Matrix<bool, Eigen::Dynamic, 1 >& valid)
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

    computeForwardPolynomial(alpha, delta);

    // Determine the bad points with a norm of zero, and avoid division by zero
    for (int i = 0; i < xyNorm.rows(); i++)
    {
        if (xyNorm(i,0) <= 0.0)
        {
            xyNorm(i,0) = 1.0;
            delta(i,0) = 0.0;
        }
    }

    // Compute the image coordinates
    auto scale = delta.array() / xyNorm.array();
    imgPoints = scale.array().replicate(1,2) * cameraPoints.block(0, 0, cameraPoints.rows(), 2).array();
    imgPoints.rowwise() += principalPoint.row(0);

    // Check if the points are valid
    for (int i = 0; i < cameraPoints.rows(); i++)
    {
        if ( 0.0 <= imgPoints(i,0) &&  imgPoints(i,0) < static_cast<double>(imgWidth) && 0 <= imgPoints(i,1) && imgPoints(i,1) < static_cast<double>(imgHeight) && alpha(i,0 ) <= maxAngle)
            valid(i,0) = true;
        else
            valid(i,0) = false;
    }
}

void FThetaCamera::pixelToCameraRay(const Eigen::Matrix<double, Eigen::Dynamic, 2>& pixelCoordinates,
                            Eigen::Matrix<double, Eigen::Dynamic, 3>& cameraRays) 
{

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> pixelOffsets;
    pixelOffsets.resize(pixelCoordinates.rows(), pixelCoordinates.cols());

    pixelOffsets.col(0) = pixelCoordinates.col(0).array() - principalPoint(0,0);
    pixelOffsets.col(1) = pixelCoordinates.col(1).array() - principalPoint(0,1);

    // Compute the norm of the pixel offsets
    Eigen::Matrix<double, Eigen::Dynamic, 1> pixelNorms;
    pixelNorms.resize(pixelOffsets.rows(), 1);
    pixelNorms = pixelOffsets.rowwise().norm();

    // Evaluate the backward polynomial
    Eigen::Matrix<double, Eigen::Dynamic, 1> alphas;
    alphas.resize(pixelOffsets.rows(), 1);
    computeBackwardsPolynomial(pixelNorms, alphas);

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
            auto const alpha = alphas(i);
            auto const alphaSinePerPixelNorm = std::sin(alpha) / pixelNorm;
            auto const alphaCos = std::cos(alpha);
            cameraRays(i, 0)  = alphaSinePerPixelNorm * pixelOffsets(i, 0);
            cameraRays(i, 1)  = alphaSinePerPixelNorm * pixelOffsets(i, 1);
            cameraRays(i, 2)  = alphaCos;
        }
    }
}

     
void FThetaCamera::computeForwardPolynomial(const Eigen::Matrix<double, Eigen::Dynamic, 1>& alphas, 
                                Eigen::Matrix<double, Eigen::Dynamic, 1>& delta)
{
    // Iterate over all the angles and evaluate the forward polynomial
    for (int i = 0; i < alphas.rows(); i++)
    {
        auto const r = alphas(i,0);
        delta(i, 0)  = evaluatePolynomialHornerScheme<5U>(r, fwPolyCoefficients);
    }
}

void FThetaCamera::computeBackwardsPolynomial(const Eigen::Matrix<double, Eigen::Dynamic, 1>& pixelNorms, 
                                Eigen::Matrix<double, Eigen::Dynamic, 1>& alphas)
{
    // Iterate over all the pixels and evaluate the backwards polynomial
    for (int i = 0; i < pixelNorms.rows(); i++)
    {
        auto const r = pixelNorms(i,0);
        alphas(i, 0) = evaluatePolynomialHornerScheme<5U>(r, bwPolyCoefficients);
    }
}

void FThetaCamera::numericallyStable2Norm2D(const Eigen::Matrix<double, Eigen::Dynamic, 3>& camPoints,
                                      Eigen::Matrix<double, Eigen::Dynamic, 1>& xyNorms)
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

// Evaluates a polynomial (of degree DEGREE) given it's coefficients at a specific point)
// using numerically stable Horner-scheme (https://en.wikipedia.org/wiki/Horner%27s_method)
template<size_t DEGREE, typename Scalar>
Scalar FThetaCamera::evaluatePolynomialHornerScheme(Scalar x,
                                        std::array<Scalar, DEGREE+1> const& coefficients)
{
        auto ret = Scalar{0};
        for(auto it = coefficients.rbegin(); it != coefficients.rend(); ++it)
        {
            ret = ret * x + *it;
        }
        return ret;
}


PinholeCamera::PinholeCamera(const int imgWidth_, const int imgHeight_,  const Eigen::Matrix<double, 1,2>& principalPoint_,
                const Eigen::Matrix<double, 1,2>& focalLength_, const Eigen::Matrix<double, 1,3>& radialPoly_,
                const Eigen::Matrix<double, 1,2>& tangentialPoly_, const std::string shutterType_):
    CameraModel(imgWidth_, imgHeight_, principalPoint_, shutterType_), focalLength(focalLength_), 
                                    radialPoly(radialPoly_), tangentialPoly(tangentialPoly_) {}

PinholeCamera::~PinholeCamera(){}

// Transform the camera rays to pixel coordinates for pinhole camera
void PinholeCamera::cameraRayToPixel(const Eigen::Matrix<double, Eigen::Dynamic, 3>& cameraPoints,
                        Eigen::Matrix<double, Eigen::Dynamic, 2>& imgPoints,
                        Eigen::Matrix<bool, Eigen::Dynamic, 1 >& valid)
{


    Eigen::Array<double, Eigen::Dynamic, 1> uNormalized = cameraPoints.col(0).array() / cameraPoints.col(2).array();
    Eigen::Array<double, Eigen::Dynamic, 1> vNormalized = cameraPoints.col(1).array() / cameraPoints.col(2).array();

    Eigen::Array<double, Eigen::Dynamic, 1> r2 = uNormalized * uNormalized + vNormalized * vNormalized;
    Eigen::Array<double, Eigen::Dynamic, 1> rD = 1.0 + r2 * (radialPoly(0,0) + r2 * (radialPoly(0,1) + r2 * radialPoly(0,2)));

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

    Eigen::Array<double, Eigen::Dynamic, 1> uND = uNormalized * rD + 2.0 * tangentialPoly(0,0) * uNormalized * vNormalized + tangentialPoly(0,1) * (r2 + 2.0 * uNormalized * uNormalized);
    Eigen::Array<double, Eigen::Dynamic, 1> vND = vNormalized * rD + tangentialPoly(0,0) * (r2 + 2.0 * vNormalized * vNormalized) + 2.0 * tangentialPoly(0,1) * uNormalized * vNormalized;

    Eigen::Array<double, Eigen::Dynamic, 1> uD = uND * focalLength(0,0) + principalPoint(0,0);
    Eigen::Array<double, Eigen::Dynamic, 1> vD = vND * focalLength(0,1) + principalPoint(0,1);

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
            uD(i,0) = uNormalized(i,0) * ((double)1.0 / std::sqrt(r2(i,0))) * clipping_radius + principalPoint(0,0);
            vD(i,0) = vNormalized(i,0) * ((double)1.0 / std::sqrt(r2(i,0))) * clipping_radius + principalPoint(0,1);
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

void PinholeCamera::pixelToCameraRay(const Eigen::Matrix<double, Eigen::Dynamic, 2>& pixelCoordinates,
                            Eigen::Matrix<double, Eigen::Dynamic, 3>& cameraRays)
{
    // Compute the x and y coordinate according to the pinhole camera model, z is already set to one
    for(int i = 0; i < pixelCoordinates.rows(); i++)
    {
        Eigen::Matrix<double,Eigen::Dynamic,2> worldRay;
        worldRay.resize(1,2);
        worldRay.setZero();
        iteraitiveUndistortPoints(pixelCoordinates.row(i), worldRay);

        cameraRays.row(i) << worldRay(0,0), worldRay(0,1), 1.0;
    }
} 

void PinholeCamera::iteraitiveUndistortPoints(const Eigen::Matrix<double, Eigen::Dynamic, 2>& src, 
                                Eigen::Matrix<double, Eigen::Dynamic, 2>& tgt,
                                const double eps)
{
    // Convert pixel coordinate to normalized coordinates
    double x, y, x0, y0;
    x = x0 = (src(0,0) - principalPoint(0,0)) / focalLength(0,0);
    y = y0 = (src(0,1) - principalPoint(0,1)) / focalLength(0,1);

    // Initialize the parameters
    double error = std::numeric_limits<double>::max();
    constexpr int maxIter = 30;

    for(int i = 0; i < maxIter && error > eps; i++)
    {
        double r_2 = x*x + y*y;
        double icD = 1. / (1 + ((radialPoly(0,2) * r_2 + radialPoly(0,1)) * r_2 + radialPoly(0,0)) * r_2);
        double deltaX = 2 * tangentialPoly(0,0) * x * y + tangentialPoly(0,1) * (r_2  + 2 * x * x);
        double deltaY = 2 * tangentialPoly(0,1) * x * y + tangentialPoly(0,0) * (r_2  + 2 * y * y);

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