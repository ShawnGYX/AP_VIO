#include <numeric>

#include "eigen3/Eigen/QR"
#include "opencv2/imgproc.hpp"

#include "eqvio/EqFMatrices.h"
#include "eqvio/VIOFilter.h"
#include "eqvio/VIOFilterSettings.h"

using namespace Eigen;
using namespace std;
using namespace liepp;

VectorXd bundleLift2(const VectorXd& baseInn, const VectorXd& fibreInn, const VIOState& Xi0);

Matrix<double, FilterBaseDim, FilterBaseDim> constructBaseSigma(const VIOFilter::Settings& settings) {
    Matrix<double, FilterBaseDim, FilterBaseDim> Sigma;
    Sigma.setZero();
    Sigma.block<3, 3>(0, 0) = Matrix3d::Identity() * settings.initialBiasOmegaVariance;
    Sigma.block<3, 3>(3, 3) = Matrix3d::Identity() * settings.initialBiasAccelVariance;
    Sigma.block<3, 3>(6, 6) = Matrix3d::Identity() * settings.initialAttitudeVariance;
    Sigma.block<3, 3>(9, 9) = Matrix3d::Identity() * settings.initialPositionVariance;
    Sigma.block<3, 3>(12, 12) = Matrix3d::Identity() * settings.initialVelocityVariance;
    Sigma.block<3, 3>(15, 15) = Matrix3d::Identity() * settings.initialCameraAttitudeVariance;
    Sigma.block<3, 3>(18, 18) = Matrix3d::Identity() * settings.initialCameraPositionVariance;
    return Sigma;
}

void removeRows(MatrixXd& mat, int startRow, int numRows) {
    int rows = mat.rows();
    int cols = mat.cols();

    assert(startRow + numRows <= rows);
    mat.block(startRow, 0, rows - numRows - startRow, cols) =
        mat.block(startRow + numRows, 0, rows - numRows - startRow, cols);
    mat.conservativeResize(rows - numRows, NoChange);
}

void removeCols(MatrixXd& mat, int startCol, int numCols) {
    int rows = mat.rows();
    int cols = mat.cols();

    assert(startCol + numCols <= cols);
    mat.block(0, startCol, rows, cols - numCols - startCol) =
        mat.block(0, startCol + numCols, rows, cols - numCols - startCol);
    mat.conservativeResize(NoChange, cols - numCols);
}

VIOFilter::VIOFilter(const VIOFilter::Settings& settings) {
    this->settings = make_unique<VIOFilter::Settings>(settings);
    Sigma = constructBaseSigma(settings);

    xi0.sensor.pose.setIdentity();
    xi0.sensor.velocity.setZero();
    xi0.sensor.cameraOffset = settings.cameraOffset;

    inputBias.setZero();

    if (settings.coordinateChoice == CoordinateChoice::Euclidean) {
        coordinateSuite = &EqFCoordinateSuite_euclid;
    } else if (settings.coordinateChoice == CoordinateChoice::InvDepth) {
        coordinateSuite = &EqFCoordinateSuite_invdepth;
    } else if (settings.coordinateChoice == CoordinateChoice::Normal) {
        coordinateSuite = &EqFCoordinateSuite_normal;
    }
}

void VIOFilter::processIMUData(const IMUVelocity& imuVelocity) {
    IMUVelocity unbiasedVelocity = imuVelocity - inputBias;
    if (!initialisedFlag) {
        initialiseFromIMUData(unbiasedVelocity);
    }

    integrateUpToTime(imuVelocity.stamp, !settings->fastRiccati);

    // Update the velocity and time
    currentVelocity = unbiasedVelocity;
    currentTime = imuVelocity.stamp;
}

void VIOFilter::initialiseFromIMUData(const IMUVelocity& imuVelocity) {
    xi0.sensor.pose.setIdentity();
    xi0.sensor.velocity.setZero();
    initialisedFlag = true;

    // Compute the attitude from the gravity vector
    // accel \approx g R^\top e_3,
    // e_3 \approx R accel.normalized()

    const Vector3d& approxGravity = imuVelocity.accel.normalized();
    xi0.sensor.pose.R = SO3d::SO3FromVectors(approxGravity, Vector3d::Unit(2));
}

void VIOFilter::setState(const VIOState& xi) {
    xi0 = xi;
    X = VIOGroup::Identity(xi.getIds());

    const int N = xi.cameraLandmarks.size();
    Sigma = MatrixXd::Identity(FilterBaseDim + 3 * N, FilterBaseDim + 3 * N);
    Sigma.block<FilterBaseDim, FilterBaseDim>(0, 0) = constructBaseSigma(*settings);
    Sigma.block(FilterBaseDim, FilterBaseDim, 3 * N, 3 * N) *= settings->initialPointVariance;

    initialisedFlag = true;
}

bool VIOFilter::integrateUpToTime(const double& newTime, const bool doRiccati) {
    if (currentTime < 0)
        return false;

    const double dt = newTime - currentTime;
    if (dt <= 0)
        return false;

    accumulatedTime += dt;
    accumulatedVelocity = accumulatedVelocity + currentVelocity * dt;

    const int N = xi0.cameraLandmarks.size();
    const VIOState currentState = stateEstimate();

    if (doRiccati) {
        assert(!X.hasNaN());
        // Lift the velocity and compute the Riccati process matrices
        MatrixXd PMat = MatrixXd::Identity(Sigma.rows(), Sigma.cols());
        PMat.block<3, 3>(0, 0) *= settings->biasOmegaProcessVariance;
        PMat.block<3, 3>(3, 3) *= settings->biasAccelProcessVariance;
        PMat.block<3, 3>(6, 6) *= settings->attitudeProcessVariance;
        PMat.block<3, 3>(9, 9) *= settings->positionProcessVariance;
        PMat.block<3, 3>(12, 12) *= settings->velocityProcessVariance;
        PMat.block<3, 3>(15, 15) *= settings->cameraAttitudeProcessVariance;
        PMat.block<3, 3>(18, 18) *= settings->cameraPositionProcessVariance;
        PMat.block(FilterBaseDim, FilterBaseDim, 3 * N, 3 * N) *= settings->pointProcessVariance;

        accumulatedVelocity = accumulatedVelocity * (1.0 / accumulatedTime);
        const MatrixXd A0t = coordinateSuite->stateMatrixA(X, xi0, accumulatedVelocity);

        // Compute the Riccati velocity matrix
        const MatrixXd Bt = coordinateSuite->inputMatrixB(X, xi0);
        Matrix<double, 6, 6> R = Matrix<double, 6, 6>::Identity();
        R.block<3, 3>(0, 0) *= settings->velOmegaVariance;
        R.block<3, 3>(3, 3) *= settings->velAccelVariance;

        // Create Bias filter matrices
        MatrixXd A0tBiased = MatrixXd::Zero(A0t.rows() + 6, A0t.cols() + 6);
        A0tBiased.block(6, 6, A0t.rows(), A0t.cols()) = A0t;
        A0tBiased.block(6, 0, Bt.rows(), Bt.cols()) = -Bt;
        // const MatrixXd A0tBiasedExp = (A0tBiased * dt).exp();
        const MatrixXd A0tBiasedExp =
            MatrixXd::Identity(A0tBiased.rows(), A0tBiased.cols()) + A0tBiased * accumulatedTime;
        MatrixXd BtBiased = MatrixXd::Zero(Bt.rows() + 6, Bt.cols());
        BtBiased.block(6, 0, Bt.rows(), Bt.cols()) = Bt;

        // Sigma += dt * (PMat + Bt * R * Bt.transpose() + A0tBiased * Sigma + Sigma * A0tBiased.transpose());
        Sigma = accumulatedTime * (PMat + BtBiased * R * BtBiased.transpose()) +
                A0tBiasedExp * Sigma * A0tBiasedExp.transpose();
        assert(!Sigma.hasNaN());

        accumulatedVelocity = IMUVelocity::Zero();
        accumulatedTime = 0.0;
    }

    // Integrate the equations
    VIOGroup liftedVelocity;
    if (settings->useDiscreteVelocityLift) {
        liftedVelocity = liftVelocityDiscrete(currentState, currentVelocity, dt);
    } else {
        const auto liftedVelocityAlg = liftVelocity(currentState, currentVelocity);
        liftedVelocity = VIOExp(dt * liftedVelocityAlg);
    }
    assert(!liftedVelocity.hasNaN());
    X = X * liftedVelocity;
    assert(!X.hasNaN());

    currentTime = newTime;
    return true;
}

void VIOFilter::processVisionData(const VisionMeasurement& measurement) {
    // Use the stored velocity input to bring the filter up to the current timestamp
    bool integrationFlag = integrateUpToTime(measurement.stamp, true);
    if (!integrationFlag || !initialisedFlag)
        return;

    removeOldLandmarks(measurement.getIds());
    assert(measurement.camCoordinates.size() >= X.id.size());
    for (int i = X.id.size() - 1; i >= 0; --i) {
        assert(measurement.camCoordinates.count(X.id[i]) > 0);
    }

    VisionMeasurement matchedMeasurement = measurement;
    removeOutliers(matchedMeasurement);
    addNewLandmarks(matchedMeasurement);

    assert(matchedMeasurement.camCoordinates.size() == X.id.size());
    for (int i = X.id.size() - 1; i >= 0; --i) {
        assert(matchedMeasurement.camCoordinates.count(X.id[i]) > 0);
    }

    if (matchedMeasurement.camCoordinates.empty())
        return;

    // --------------------------
    // Compute the EqF innovation
    // --------------------------
    const VisionMeasurement estimatedMeasurement = measureSystemState(stateEstimate(), measurement.cameraPtr);
    const VisionMeasurement measurementResidual = matchedMeasurement - estimatedMeasurement;
    const MatrixXd Ct = coordinateSuite->outputMatrixC(xi0, X, matchedMeasurement);
    const int N = xi0.cameraLandmarks.size();
    const MatrixXd QMat = settings->measurementVariance * MatrixXd::Identity(2 * N, 2 * N);

    // Create the bias matrix
    MatrixXd CtBiased = MatrixXd::Zero(Ct.rows(), Ct.cols() + 6);
    CtBiased.block(0, 6, Ct.rows(), Ct.cols()) = Ct;

    // Use the discrete update form
    MatrixXd SInv = (CtBiased * Sigma * CtBiased.transpose() + QMat).inverse();
    const MatrixXd K = Sigma * CtBiased.transpose() * SInv;

    const VectorXd yTilde = measurementResidual;
    const VectorXd baseInnovationBiased = K * yTilde;
    const VectorXd& baseInnovationEqF = baseInnovationBiased.block(6, 0, baseInnovationBiased.rows() - 6, 1);
    const VectorXd& baseInnovationBias = baseInnovationBiased.block<6, 1>(0, 0);

    VectorXd Gamma;
    if (settings->useBundleLiftType1) {
        Gamma = coordinateSuite->bundleLift(
            baseInnovationEqF, xi0, X, Sigma.block(6, 6, FilterBaseDim - 6 + 3 * N, FilterBaseDim - 6 + 3 * N));
    } else {
        Gamma = baseInnovationEqF;
    }
    assert(!Gamma.hasNaN());

    VIOGroup Delta;
    if (settings->useDiscreteInnovationLift) {
        Delta = coordinateSuite->liftInnovationDiscrete(Gamma, xi0);
    } else {
        Delta = VIOExp(coordinateSuite->liftInnovation(Gamma, xi0));
    }
    assert(!Delta.hasNaN());

    inputBias = inputBias + baseInnovationBias;
    X = Delta * X;
    Sigma = Sigma - K * CtBiased * Sigma;

    assert(!Sigma.hasNaN());
    assert(!X.hasNaN());
    // assert(Sigma.eigenvalues().real().minCoeff() > 0);
}

VIOState VIOFilter::stateEstimate() const { return stateGroupAction(this->X, this->xi0); }

VisionMeasurement VIOFilter::getBearingPredictions(const GIFT::GICameraPtr& camPtr, const double& stamp) {
    if (stamp > 0) {
        bool integrationFlag = integrateUpToTime(stamp, true);
    }
    return measureSystemState(stateEstimate(), camPtr);
}

Eigen::MatrixXd VIOFilter::stateCovariance() const {
    // TODO: Propagate Sigma to the local tangent space
    return Sigma;
}

CSVLine& operator<<(CSVLine& line, const VIOFilter& filter) { return line << filter.xi0 << filter.X << filter.Sigma; }

double VIOFilter::getTime() const { return currentTime; }

void VIOFilter::addNewLandmarks(std::vector<Landmark>& newLandmarks) {
    // Initialise all landmarks to the median scene depth
    const double medianDepth = getMedianSceneDepth();
    for_each(newLandmarks.begin(), newLandmarks.end(), [&medianDepth](Landmark& blm) { blm.p *= medianDepth; });
    xi0.cameraLandmarks.insert(xi0.cameraLandmarks.end(), newLandmarks.begin(), newLandmarks.end());

    vector<int> newIds(newLandmarks.size());
    transform(newLandmarks.begin(), newLandmarks.end(), newIds.begin(), [](const Landmark& blm) { return blm.id; });
    X.id.insert(X.id.end(), newIds.begin(), newIds.end());

    vector<SOT3d> newTransforms(newLandmarks.size());
    for (SOT3d& newTf : newTransforms) {
        newTf.setIdentity();
    }
    X.Q.insert(X.Q.end(), newTransforms.begin(), newTransforms.end());

    const int newN = newLandmarks.size();
    const int ogSize = Sigma.rows();
    Sigma.conservativeResize(ogSize + 3 * newN, ogSize + 3 * newN);
    Sigma.block(ogSize, 0, 3 * newN, ogSize).setZero();
    Sigma.block(0, ogSize, ogSize, 3 * newN).setZero();
    Sigma.block(ogSize, ogSize, 3 * newN, 3 * newN) =
        MatrixXd::Identity(3 * newN, 3 * newN) * settings->initialPointVariance;
}

void VIOFilter::addNewLandmarks(const VisionMeasurement& measurement) {
    // Grab all the new landmarks
    std::vector<Landmark> newLandmarks;
    for (const pair<int, Vector2d>& cc : measurement.camCoordinates) {
        const int& ccId = cc.first;
        if (none_of(X.id.begin(), X.id.end(), [&ccId](const int& i) { return i == ccId; })) {
            Vector3d bearing = measurement.cameraPtr->undistortPoint(cc.second);
            newLandmarks.emplace_back(Landmark{bearing, ccId});
        }
    }
    if (newLandmarks.empty())
        return;
    addNewLandmarks(newLandmarks);
}

void VIOFilter::removeOldLandmarks(const vector<int>& measurementIds) {
    // Determine which indices have been lost
    vector<int> lostIndices(X.id.size());
    iota(lostIndices.begin(), lostIndices.end(), 0);
    if (lostIndices.empty())
        return;

    const auto lostIndicesEnd = remove_if(lostIndices.begin(), lostIndices.end(), [&](const int& lidx) {
        const int& oldId = X.id[lidx];
        return any_of(
            measurementIds.begin(), measurementIds.end(), [&oldId](const int& measId) { return measId == oldId; });
    });
    lostIndices.erase(lostIndicesEnd, lostIndices.end());

    if (lostIndices.empty())
        return;

    // Remove the origin state and transforms and Sigma bits corresponding to these indices.
    reverse(lostIndices.begin(), lostIndices.end()); // Should be in descending order now
    for (const int li : lostIndices) {
        removeLandmarkByIndex(li);
    }
}

void VIOFilter::removeLandmarkByIndex(const int& idx) {
    xi0.cameraLandmarks.erase(xi0.cameraLandmarks.begin() + idx);
    X.id.erase(X.id.begin() + idx);
    X.Q.erase(X.Q.begin() + idx);
    removeRows(Sigma, FilterBaseDim + 3 * idx, 3);
    removeCols(Sigma, FilterBaseDim + 3 * idx, 3);
}

void VIOFilter::removeLandmarkById(const int& id) {
    const auto it = find_if(
        xi0.cameraLandmarks.begin(), xi0.cameraLandmarks.end(), [&id](const Landmark& lm) { return lm.id == id; });
    assert(it != xi0.cameraLandmarks.end());
    const int idx = distance(xi0.cameraLandmarks.begin(), it);
    removeLandmarkByIndex(idx);
}

void VIOFilter::removeOutliers(VisionMeasurement& measurement) {
    const VIOState xiHat = stateEstimate();
    const VisionMeasurement yHat = measureSystemState(xiHat, measurement.cameraPtr);
    // Remove if the difference between the true and expected measurement exceeds a threshold
    assert(measurement.camCoordinates.size() >= yHat.camCoordinates.size());
    std::set<int> proposedOutliers;
    for_each(
        yHat.camCoordinates.begin(), yHat.camCoordinates.end(),
        [this, &measurement, &proposedOutliers](const pair<int, Vector2d>& lm) {
            assert(measurement.camCoordinates.count(lm.first) > 0);
            double bearingErrorAbs = (measurement.camCoordinates[lm.first] - lm.second).norm();
            if (bearingErrorAbs > settings->outlierThresholdAbs) {
                proposedOutliers.emplace(lm.first);
            }
        });
    const VisionMeasurement& measurementResidual = measurement - yHat;
    for (const auto& cc : measurementResidual.camCoordinates) {
        const int& lmId = cc.first;
        const Vector2d& yTilde_i = cc.second;
        const Matrix2d outputCov = getOutputCovById(lmId, measurement.camCoordinates[lmId], measurement.cameraPtr);
        double bearingErrorProb = yTilde_i.transpose() * outputCov.inverse() * yTilde_i;
        if (bearingErrorProb > settings->outlierThresholdProb) {
            proposedOutliers.emplace(lmId);
        }
    }

    for_each(proposedOutliers.begin(), proposedOutliers.end(), [this, &measurement](const int& lmId) {
        removeLandmarkById(lmId);
        measurement.camCoordinates.erase(lmId);
    });
}

VectorXd bundleLift2(const VectorXd& baseInn, const VectorXd& fibreInn, const VIOState& Xi0) {
    const int N = Xi0.cameraLandmarks.size();
    VectorXd Gamma = VectorXd(15 + 3 * N);

    const Vector3d& eta0 = Xi0.sensor.gravityDir();

    // Rotation
    Gamma.segment<3>(0) =
        -SO3d::skew(eta0) * sphereChart_stereo.invDiff0(eta0) * baseInn.segment<2>(0) + eta0 * fibreInn(3);
    // Translation
    Gamma.segment<3>(3) = Xi0.sensor.pose.R.inverse() * fibreInn.segment<3>(0);
    // Velocity, camera offset, and points
    Gamma.segment(6, 9 + 3 * N) = baseInn.segment(2, 9 + 3 * N);

    return Gamma;
}

double VIOFilter::getMedianSceneDepth() const {
    const vector<Landmark> landmarks = this->stateEstimate().cameraLandmarks;
    vector<double> depthsSquared(landmarks.size());
    transform(landmarks.begin(), landmarks.end(), depthsSquared.begin(), [](const Landmark& blm) {
        return blm.p.squaredNorm();
    });
    const auto midway = depthsSquared.begin() + depthsSquared.size() / 2;
    nth_element(depthsSquared.begin(), midway, depthsSquared.end());
    double medianDepth = settings->initialSceneDepth;
    if (!(midway == depthsSquared.end())) {
        medianDepth = pow(*midway, 0.5);
    }

    return medianDepth;
}

Eigen::Matrix3d VIOFilter::getLandmarkCovById(const int& id) const {
    const auto it = find_if(
        xi0.cameraLandmarks.begin(), xi0.cameraLandmarks.end(), [&id](const Landmark& lm) { return lm.id == id; });
    assert(it != xi0.cameraLandmarks.end());
    const int i = distance(xi0.cameraLandmarks.begin(), it);
    return Sigma.block<3, 3>(FilterBaseDim + 3 * i, FilterBaseDim + 3 * i);
}

Eigen::Matrix2d VIOFilter::getOutputCovById(const int& id, const Vector2d& y, const GIFT::GICameraPtr& camPtr) const {
    const Matrix3d lmCov = getLandmarkCovById(id);
    const auto it = find_if(
        xi0.cameraLandmarks.begin(), xi0.cameraLandmarks.end(), [&id](const Landmark& lm) { return lm.id == id; });
    assert(it != xi0.cameraLandmarks.end());
    const auto it_X = find_if(X.id.begin(), X.id.end(), [&it](const int& i) { return i == it->id; });
    assert(it_X != X.id.end());
    const SOT3d& Q_i = X.Q[distance(X.id.begin(), it_X)];
    const Matrix<double, 2, 3> C0i = coordinateSuite->outputMatrixCi(it->p, Q_i, camPtr, y);
    const Matrix2d landmarkCov = C0i * lmCov * C0i.transpose();
    return landmarkCov;
}