#pragma once

#include "CSVReader.h"
#include "eqf_vio/VIOState.h"
#include "liepp/SE3.h"
#include "liepp/SOT3.h"

struct VIOGroup {
    liepp::SE3d A;               // related to IMU pose
    liepp::SE3d B;               // related to camera offset
    Eigen::Vector3d w;           // related to imu frame velocity
    std::vector<liepp::SOT3d> Q; // related to landmark bearing and depth
    std::vector<int> id;

    [[nodiscard]] VIOGroup operator*(const VIOGroup& other) const;
    [[nodiscard]] static VIOGroup Identity(const std::vector<int>& id = {});
    [[nodiscard]] VIOGroup inverse() const;
    [[nodiscard]] bool hasNaN() const;
};

CSVLine& operator<<(CSVLine& line, const VIOGroup& X);

struct VIOAlgebra {
    liepp::se3d U_A;
    liepp::se3d U_B;
    Eigen::Vector3d u_w;
    std::vector<Eigen::Vector4d> W;
    std::vector<int> id;

    [[nodiscard]] VIOAlgebra operator*(const double& c) const;
    [[nodiscard]] VIOAlgebra operator-() const;
    [[nodiscard]] VIOAlgebra operator+(const VIOAlgebra& other) const;
    [[nodiscard]] VIOAlgebra operator-(const VIOAlgebra& other) const;
};
[[nodiscard]] VIOAlgebra operator*(const double& c, const VIOAlgebra& lambda);

[[nodiscard]] VIOSensorState sensorStateGroupAction(const VIOGroup& X, const VIOSensorState& sensor);
[[nodiscard]] VIOState stateGroupAction(const VIOGroup& X, const VIOState& state);
[[nodiscard]] VisionMeasurement outputGroupAction(const VIOGroup& X, const VisionMeasurement& measurement);

[[nodiscard]] VIOAlgebra liftVelocity(const VIOState& state, const IMUVelocity& velocity);
[[nodiscard]] VIOGroup liftVelocityDiscrete(const VIOState& state, const IMUVelocity& velocity, const double& dt);

[[nodiscard]] VIOGroup VIOExp(const VIOAlgebra& lambda);