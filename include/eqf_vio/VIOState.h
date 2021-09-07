#pragma once

#include "liepp/SE3.h"

#include "CSVReader.h"
#include "eigen3/Eigen/Eigen"
#include "eqf_vio/Geometry.h"
#include "eqf_vio/IMUVelocity.h"
#include "eqf_vio/VisionMeasurement.h"

#include <functional>
#include <memory>
#include <ostream>
#include <vector>

struct Landmark {
    Eigen::Vector3d p;
    int id = -1;

    constexpr static int CompDim = 3;
};

struct VIOSensorState {
    // The constant size part of the VIO total state
    liepp::SE3d pose;
    Eigen::Vector3d velocity;
    liepp::SE3d cameraOffset;

    Eigen::Vector3d gravityDir() const;

    constexpr static int CompDim = 6 + 3 + 6;
};

struct VIOState {
    VIOSensorState sensor;
    std::vector<Landmark> cameraLandmarks;

    std::vector<int> getIds() const;

    const int Dim() const;
    constexpr static int CompDim = Eigen::Dynamic;
};

extern const CoordinateChart<VIOSensorState> sensorChart_std;
extern const CoordinateChart<VIOSensorState> sensorChart_normal;
extern const CoordinateChart<Landmark> pointChart_euclid;
extern const CoordinateChart<Landmark> pointChart_invdepth;
extern const CoordinateChart<Landmark> pointChart_normal;

const CoordinateChart<VIOState> constructVIOChart(
    const CoordinateChart<VIOSensorState>& sensorBundleChart, const CoordinateChart<Landmark>& pointChart);

extern const CoordinateChart<VIOState> VIOChart_euclid;
extern const CoordinateChart<VIOState> VIOChart_invdepth;
extern const CoordinateChart<VIOState> VIOChart_normal;

const Eigen::MatrixXd coordinateDifferential_invdepth_euclid(const VIOState& xi0);
const Eigen::MatrixXd coordinateDifferential_normal_euclid(const VIOState& xi0);

Eigen::Vector2d e3ProjectSphere(const Eigen::Vector3d& eta);
Eigen::Vector3d e3ProjectSphereInv(const Eigen::Vector2d& y);
Eigen::Matrix<double, 2, 3> e3ProjectSphereDiff(const Eigen::Vector3d& eta);
Eigen::Matrix<double, 3, 2> e3ProjectSphereInvDiff(const Eigen::Vector2d& y);

extern const EmbeddedCoordinateChart<3, 2> sphereChart_stereo;
extern const EmbeddedCoordinateChart<3, 2> sphereChart_normal;

CSVLine& operator<<(CSVLine& line, const VIOSensorState& sensor);
CSVLine& operator<<(CSVLine& line, const VIOState& state);

[[nodiscard]] VIOState integrateSystemFunction(const VIOState& state, const IMUVelocity& velocity, const double& dt);
[[nodiscard]] VisionMeasurement measureSystemState(const VIOState& state, const GIFT::GICameraPtr& cameraPtr);