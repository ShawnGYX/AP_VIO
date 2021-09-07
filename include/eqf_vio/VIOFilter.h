#pragma once

#include <memory>
#include <ostream>

#include "eqf_vio/CSVReader.h"
#include "eqf_vio/EqFMatrices.h"
#include "eqf_vio/IMUVelocity.h"
#include "eqf_vio/VIOGroup.h"
#include "eqf_vio/VIOState.h"
#include "eqf_vio/VisionMeasurement.h"

constexpr int FilterBaseDim = IMUVelocity::CompDim + VIOSensorState::CompDim;

class VIOFilter {
  protected:
    EqFCoordinateSuite const* coordinateSuite = &EqFCoordinateSuite_euclid;

    Eigen::Matrix<double, 6, 1> inputBias = Eigen::Matrix<double, 6, 1>::Zero();
    VIOState xi0;
    VIOGroup X = VIOGroup::Identity();
    Eigen::MatrixXd Sigma = Eigen::MatrixXd::Identity(FilterBaseDim, FilterBaseDim);

    bool initialisedFlag = false;
    double currentTime = -1;
    IMUVelocity currentVelocity = IMUVelocity::Zero();

    IMUVelocity accumulatedVelocity = IMUVelocity::Zero();
    double accumulatedTime = 0.0;

    bool integrateUpToTime(const double& newTime, const bool doRiccati = true);
    void addNewLandmarks(const VisionMeasurement& measurement);
    void addNewLandmarks(std::vector<Landmark>& newLandmarks);
    void removeOldLandmarks(const std::vector<int>& measurementIds);
    void removeOutliers(VisionMeasurement& measurement);
    void removeLandmarkByIndex(const int& idx);
    void removeLandmarkById(const int& id);
    double getMedianSceneDepth() const;

    Eigen::Matrix3d getLandmarkCovById(const int& id) const;
    Eigen::Matrix2d getOutputCovById(const int& id, const Eigen::Vector2d& y, const GIFT::GICameraPtr& camPtr) const;

  public:
    // Settings
    struct Settings;
    std::unique_ptr<VIOFilter::Settings> settings;

    // Setup
    VIOFilter() = default;
    VIOFilter(const VIOFilter::Settings& settings);
    void initialiseFromIMUData(const IMUVelocity& imuVelocity);
    void setState(const VIOState& xi);

    // Input
    void processIMUData(const IMUVelocity& imuVelocity);
    void processVisionData(const VisionMeasurement& measurement);

    // Output
    double getTime() const;
    bool isInitialised() const { return initialisedFlag; };
    VisionMeasurement getBearingPredictions(const GIFT::GICameraPtr& camPtr, const double& stamp = -1);
    VIOState stateEstimate() const;
    Eigen::MatrixXd stateCovariance() const;
    friend CSVLine& operator<<(CSVLine& line, const VIOFilter& filter);
};

CSVLine& operator<<(CSVLine& line, const VIOFilter& filter);