#pragma once
#include <Eigen/Dense>
#include <array>
 
class CameraModel {
    protected:
        int imgWidth;
        int imgHeight;
        Eigen::Matrix<double, 1,2> principalPoint;
        std::string shutterType;

    public:
        CameraModel(const int imgWidth_, const double imgHeight_, const Eigen::Matrix<double, 1,2>& principalPoint_, const std::string& shutterType_);

        virtual ~CameraModel();

        virtual void cameraRayToPixel(const Eigen::Matrix<double, Eigen::Dynamic, 3>& cameraPoints,
                                Eigen::Matrix<double, Eigen::Dynamic, 2 >& imgPoints,
                                Eigen::Matrix<bool, Eigen::Dynamic, 1 >& valid) = 0;

        virtual void pixelToCameraRay(const Eigen::Matrix<double, Eigen::Dynamic, 2>& pixelCoordinates,
                                    Eigen::Matrix<double, Eigen::Dynamic, 3>& cameraRays) = 0;


        void cameraToWorldRay(const Eigen::Matrix<double, Eigen::Dynamic, 2>& pixelCoordinates,
                              const Eigen::Matrix<double, Eigen::Dynamic, 3>& cameraRays, 
                              const Eigen::Matrix<double, 8, 4>& TSensorWorld, 
                              Eigen::Matrix<double, Eigen::Dynamic, 6>& worldRays);


        void rollingShutterProjection(const Eigen::Matrix<double, Eigen::Dynamic, 3>& points,
                                      const Eigen::Matrix<double, 8, 4>& TWorldSensor, 
                                      const int maxIter,
                                      Eigen::Matrix<double, Eigen::Dynamic, 4>& transformationMatrices,
                                      Eigen::Matrix<double, Eigen::Dynamic, 2>& pixelCoordinates,
                                      Eigen::VectorXi& validProjec,
                                      Eigen::VectorXi& initialValidIdx);
};

class FThetaCamera : public CameraModel 
{
    protected:
        std::array<double, 6> bwPolyCoefficients;
        std::array<double, 6> fwPolyCoefficients;
        double maxAngle;

    public:
        FThetaCamera(const int imgWidth_,  const int imgHeight_, const Eigen::Matrix<double, 1,2>& principalPoint_,
                     const std::array<double, 6>& fwPolyCoefficients_, const std::array<double, 6>& bwPolyCoefficients_,
                     const double maxAngle_, const std::string shutterType_);

        ~FThetaCamera();

        void cameraRayToPixel(const Eigen::Matrix<double, Eigen::Dynamic, 3>& cameraPoints,
                              Eigen::Matrix<double, Eigen::Dynamic, 2>& imgPoints,
                              Eigen::Matrix<bool, Eigen::Dynamic, 1 >& valid) override;

        void pixelToCameraRay(const Eigen::Matrix<double, Eigen::Dynamic, 2>& pixelCoordinates,
                                    Eigen::Matrix<double, Eigen::Dynamic, 3>& cameraRays) override;


    private: 
        void computeForwardPolynomial(const Eigen::Matrix<double, Eigen::Dynamic, 1>& alphas, 
                                      Eigen::Matrix<double, Eigen::Dynamic, 1>& delta);

        void computeBackwardsPolynomial(const Eigen::Matrix<double, Eigen::Dynamic, 1>& pixelNorms, 
                                        Eigen::Matrix<double, Eigen::Dynamic, 1>& alphas);

        void numericallyStable2Norm2D(const Eigen::Matrix<double, Eigen::Dynamic, 3>& camPoints,
                                      Eigen::Matrix<double, Eigen::Dynamic, 1>& xyNorms);

        // Evaluates a polynomial (of degree DEGREE) given it's coefficients at a specific point)
        // using numerically stable Horner-scheme (https://en.wikipedia.org/wiki/Horner%27s_method)
        template<size_t DEGREE, typename Scalar>
        Scalar evaluatePolynomialHornerScheme(Scalar x,
                                              std::array<Scalar, DEGREE+1> const& coefficients);
};

class PinholeCamera : public CameraModel 
{
    protected:
        Eigen::Matrix<double, 1,2> focalLength;
        Eigen::Matrix<double, 1,3> radialPoly;
        Eigen::Matrix<double, 1,2> tangentialPoly;

    public:
        PinholeCamera(const int imgWidth_, const int imgHeight_,  const Eigen::Matrix<double, 1,2>& principalPoint_,
                     const Eigen::Matrix<double, 1,2>& focalLength_, const Eigen::Matrix<double, 1,3>& radialPoly_,
                     const Eigen::Matrix<double, 1,2>& tangentialPoly_, const std::string shutterType_);

        ~PinholeCamera();

        // Transform the camera rays to pixel coordinates for pinhole camera
        void cameraRayToPixel(const Eigen::Matrix<double, Eigen::Dynamic, 3>& cameraPoints,
                              Eigen::Matrix<double, Eigen::Dynamic, 2>& imgPoints,
                              Eigen::Matrix<bool, Eigen::Dynamic, 1 >& valid) override;

        virtual void pixelToCameraRay(const Eigen::Matrix<double, Eigen::Dynamic, 2>& pixelCoordinates,
                                    Eigen::Matrix<double, Eigen::Dynamic, 3>& cameraRays);

    private:

        void iteraitiveUndistortPoints(const Eigen::Matrix<double, Eigen::Dynamic, 2>& src, 
                                       Eigen::Matrix<double, Eigen::Dynamic, 2>& tgt,
                                       const double eps = 1e-12);
};