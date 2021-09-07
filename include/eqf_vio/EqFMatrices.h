#pragma once

#include "eqf_vio/VIOGroup.h"
#include <functional>

struct EqFCoordinateSuite {
    // State and output space charts
    const CoordinateChart<VIOState>& stateChart;

    // Matrices for Riccati term
    const std::function<Eigen::MatrixXd(const VIOGroup&, const VIOState&, const IMUVelocity&)> stateMatrixA;
    const std::function<Eigen::MatrixXd(const VIOGroup&, const VIOState&)> inputMatrixB;
    const std::function<Eigen::Matrix<double, 2, 3>(
        const Eigen::Vector3d& q0, const liepp::SOT3d& QHat, const GIFT::GICameraPtr& camPtr, const Eigen::Vector2d& y)>
        outputMatrixCi;

    const Eigen::MatrixXd outputMatrixC(const VIOState& xi0, const VIOGroup& X, const VisionMeasurement& y) const;

    // Innovation lift
    const std::function<VIOAlgebra(const Eigen::VectorXd&, const VIOState&)> liftInnovation;
    const std::function<VIOGroup(const Eigen::VectorXd&, const VIOState&)> liftInnovationDiscrete;
    const std::function<Eigen::VectorXd(
        const Eigen::VectorXd&, const VIOState&, const VIOGroup&, const Eigen::MatrixXd&)>
        bundleLift;
};

extern const EqFCoordinateSuite EqFCoordinateSuite_euclid;
extern const EqFCoordinateSuite EqFCoordinateSuite_invdepth;
extern const EqFCoordinateSuite EqFCoordinateSuite_normal;
