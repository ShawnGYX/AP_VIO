#pragma once

#include "CSVReader.h"
#include "eigen3/Eigen/Eigen"

constexpr double GRAVITY_CONSTANT = 9.80665;

struct IMUVelocity {
    double stamp;
    Eigen::Vector3d omega;
    Eigen::Vector3d accel;

    static IMUVelocity Zero();

    IMUVelocity() = default;
    IMUVelocity(const Eigen::Matrix<double, 6, 1>& vec);

    IMUVelocity operator+(const IMUVelocity& other) const;
    IMUVelocity operator+(const Eigen::Matrix<double, 6, 1>& vec) const;
    IMUVelocity operator-(const Eigen::Matrix<double, 6, 1>& vec) const;
    IMUVelocity operator*(const double& c) const;

    constexpr static int CompDim = 6;
};

CSVLine& operator<<(CSVLine& line, const IMUVelocity& imu);
CSVLine& operator>>(CSVLine& line, IMUVelocity& imu);