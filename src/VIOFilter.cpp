#include <numeric>

#include "eigen3/Eigen/QR"
#include "opencv2/imgproc.hpp"

#include "eqvio/LoopTimer.h"
#include "eqvio/VIOFilter.h"
#include "eqvio/VIOFilterSettings.h"
#include "eqvio/mathematical/EqFMatrices.h"

using namespace Eigen;
using namespace std;
using namespace liepp;

VIOFilter::VIOFilter(const VIOFilter::Settings& settings) {
    this->settings = make_unique<VIOFilter::Settings>(settings);
    filterState.Sigma = this->settings->constructInitialStateCovariance();

    filterState.xi0.sensor.inputBias.setZero();
    filterState.xi0.sensor.pose.setIdentity();
    filterState.xi0.sensor.velocity.setZero();
    filterState.xi0.sensor.cameraOffset = settings.cameraOffset;

    filterState.coordinateSuite = getCoordinates(this->settings->coordinateChoice);
}

void VIOFilter::processIMUData(const IMUVelocity& imuVelocity) {
    if (!initialisedFlag) {
        initialiseFromIMUData(imuVelocity);
    }
    velocityBuffer.emplace_back(imuVelocity);
}

void VIOFilter::initialiseFromIMUData(const IMUVelocity& imuVelocity) {
    filterState.xi0.sensor.inputBias.setZero();
    filterState.xi0.sensor.pose.setIdentity();
    filterState.xi0.sensor.velocity.setZero();
    initialisedFlag = true;

    // Compute the attitude from the gravity vector
    // acc \approx g R^\top e_3,
    // e_3 \approx R acc.normalized()

    const Vector3d& approxGravity = imuVelocity.acc.normalized();
    filterState.xi0.sensor.pose.R = SO3d::SO3FromVectors(approxGravity, Vector3d::Unit(2));
    filterState.currentTime = imuVelocity.stamp;
}

void VIOFilter::setState(const VIOState& xi) {
    filterState.xi0 = xi;
    filterState.X = VIOGroup::Identity(xi.getIds());

    const int N = xi.cameraLandmarks.size();
    filterState.Sigma = MatrixXd::Identity(VIOSensorState::CompDim + 3 * N, VIOSensorState::CompDim + 3 * N);
    filterState.Sigma.block<VIOSensorState::CompDim, VIOSensorState::CompDim>(0, 0) =
        settings->constructInitialStateCovariance();
    filterState.Sigma.block(VIOSensorState::CompDim, VIOSensorState::CompDim, 3 * N, 3 * N) *=
        settings->initialPointVariance;

    initialisedFlag = true;
}

bool VIOFilter::integrateUpToTime(const double& newTime) {
    if (newTime <= filterState.currentTime || filterState.currentTime < 0)
        return false;

    filterState.processIMUData(
        velocityBuffer, newTime, settings->constructInputGainMatrix(),
        settings->constructStateGainMatrix(filterState.xi0.cameraLandmarks.size()), settings->fastRiccati,
        settings->useDiscreteVelocityLift);

    // Clear velocity buffer from the front
    auto it = std::find_if(velocityBuffer.begin(), velocityBuffer.end(), [this](const IMUVelocity& imuVel) {
        return imuVel.stamp >= this->filterState.currentTime;
    });
    if (it != velocityBuffer.begin()) {
        --it;
        velocityBuffer.erase(velocityBuffer.begin(), it);
    }

    return true;
}

void VIOFilter::processVisionData(const VisionMeasurement& measurement) {
    // Use the stored velocity input to bring the filter up to the current timestamp
    loopTimer.startTiming("propagation");
    bool integrationFlag = integrateUpToTime(measurement.stamp);
    if (!integrationFlag || !initialisedFlag)
        return;
    loopTimer.endTiming("propagation");

    loopTimer.startTiming("preprocessing");
    if (settings->removeLostLandmarks) {
        removeOldLandmarks(measurement.getIds());
        assert(measurement.camCoordinates.size() >= filterState.X.id.size());
        for (int i = filterState.X.id.size() - 1; i >= 0; --i) {
            assert(measurement.camCoordinates.count(filterState.X.id[i]) > 0);
        }
    }

    VisionMeasurement matchedMeasurement = measurement;
    removeOutliers(matchedMeasurement);
    addNewLandmarks(matchedMeasurement);

    if (settings->removeLostLandmarks) {
        assert(matchedMeasurement.camCoordinates.size() == filterState.X.id.size());
        for (int i = filterState.X.id.size() - 1; i >= 0; --i) {
            assert(matchedMeasurement.camCoordinates.count(filterState.X.id[i]) > 0);
        }
    }
    loopTimer.endTiming("preprocessing");

    if (matchedMeasurement.camCoordinates.empty())
        return;

    // --------------------------
    // Compute the EqF innovation
    // --------------------------
    loopTimer.startTiming("correction");

    filterState.processVisionData(
        matchedMeasurement, settings->constructOutputGainMatrix(matchedMeasurement.camCoordinates.size()),
        settings->useEquivariantOutput, settings->useDiscreteInnovationLift);

    filterState.removeInvalidLandmarks();
    loopTimer.endTiming("correction");

    assert(!filterState.Sigma.hasNaN());
    assert(!filterState.X.hasNaN());
    // assert(filterState.Sigma.eigenvalues().real().minCoeff() > 0);
}

VIOState VIOFilter::stateEstimate() const { return filterState.stateEstimate(); }

VisionMeasurement VIOFilter::getFeaturePredictions(const GIFT::GICameraPtr& camPtr, const double& stamp) {
    if (settings->useFeaturePredictions) {
        return measureSystemState(filterState.predictState(stamp, velocityBuffer), camPtr);
    }
    return VisionMeasurement();
}

CSVLine& operator<<(CSVLine& line, const VIOFilter& filter) { return line << filter.filterState; }

double VIOFilter::getTime() const { return filterState.currentTime; }

void VIOFilter::addNewLandmarks(const VisionMeasurement& measurement) {
    // Grab all the new landmarks
    std::vector<Landmark> newLandmarks;
    for (const pair<const int, Vector2d>& cc : measurement.camCoordinates) {
        const int& ccId = cc.first;
        if (none_of(filterState.X.id.begin(), filterState.X.id.end(), [&ccId](const int& i) { return i == ccId; })) {
            Vector3d bearing = measurement.cameraPtr->undistortPoint(cc.second);
            newLandmarks.emplace_back(Landmark{bearing, ccId});
        }
    }
    if (newLandmarks.empty())
        return;

    // Initialise all landmarks to the median scene depth
    double initialDepth = settings->useMedianDepth ? getMedianSceneDepth() : settings->initialSceneDepth;
    for_each(newLandmarks.begin(), newLandmarks.end(), [&initialDepth](Landmark& blm) { blm.p *= initialDepth; });
    const int newN = newLandmarks.size();
    const Eigen::MatrixXd newLandmarksCov =
        Eigen::MatrixXd::Identity(3 * newN, 3 * newN) * settings->initialPointVariance;
    filterState.addNewLandmarks(newLandmarks, newLandmarksCov);
}

void VIOFilter::removeOldLandmarks(const vector<int>& measurementIds) {
    // Determine which indices have been lost
    vector<int> lostIndices(filterState.X.id.size());
    iota(lostIndices.begin(), lostIndices.end(), 0);
    if (lostIndices.empty())
        return;

    const auto lostIndicesEnd = remove_if(lostIndices.begin(), lostIndices.end(), [&](const int& lidx) {
        const int& oldId = filterState.X.id[lidx];
        return any_of(
            measurementIds.begin(), measurementIds.end(), [&oldId](const int& measId) { return measId == oldId; });
    });
    lostIndices.erase(lostIndicesEnd, lostIndices.end());

    if (lostIndices.empty())
        return;

    // Remove the origin state and transforms and filterState.Sigma bits corresponding to these indices.
    reverse(lostIndices.begin(), lostIndices.end()); // Should be in descending order now
    for (const int li : lostIndices) {
        filterState.removeLandmarkByIndex(li);
    }
}

void VIOFilter::removeOutliers(VisionMeasurement& measurement) {
    const size_t maxOutliers = (1.0 - settings->featureRetention) * measurement.camCoordinates.size();
    const VIOState xiHat = stateEstimate();
    const VisionMeasurement yHat = measureSystemState(xiHat, measurement.cameraPtr);
    // Remove if the difference between the true and expected measurement exceeds a threshold
    std::vector<int> proposedOutliers;
    std::map<int, double> absoluteOutliers;
    for (const auto& [lmId, yHat_i] : yHat.camCoordinates) {
        if (measurement.camCoordinates.count(lmId) == 0) {
            continue;
        }
        double bearingErrorAbs = (measurement.camCoordinates.at(lmId) - yHat_i).norm();
        if (bearingErrorAbs > settings->outlierThresholdAbs) {
            absoluteOutliers[lmId] = bearingErrorAbs;
            proposedOutliers.emplace_back(lmId);
        }
    }

    std::map<int, double> probabilisticOutliers;
    const VisionMeasurement& measurementResidual = measurement - yHat;
    for (const auto& [lmId, yTilde_i] : measurementResidual.camCoordinates) {
        if (absoluteOutliers.count(lmId) || measurement.camCoordinates.count(lmId) == 0) {
            continue;
        }
        const Matrix2d outputCov =
            filterState.getOutputCovById(lmId, measurement.camCoordinates[lmId], measurement.cameraPtr);
        double bearingErrorProb = yTilde_i.transpose() * outputCov.inverse() * yTilde_i;
        if (bearingErrorProb > settings->outlierThresholdProb) {
            probabilisticOutliers[lmId] = bearingErrorProb;
            proposedOutliers.emplace_back(lmId);
        }
    }

    // Prioritise which landmarks to discard
    std::sort(
        proposedOutliers.begin(), proposedOutliers.end(),
        [&absoluteOutliers, &probabilisticOutliers](const int& lmId1, const int& lmId2) {
            if (absoluteOutliers.count(lmId1)) {
                if (absoluteOutliers.count(lmId2)) {
                    return absoluteOutliers.at(lmId1) < absoluteOutliers.at(lmId2);
                } else {
                    return false;
                }
            } else {
                if (absoluteOutliers.count(lmId2)) {
                    return true;
                } else {
                    return probabilisticOutliers.at(lmId1) < probabilisticOutliers.at(lmId2);
                }
            }
        });
    std::reverse(proposedOutliers.begin(), proposedOutliers.end());
    if (proposedOutliers.size() > maxOutliers) {
        proposedOutliers.erase(proposedOutliers.begin() + maxOutliers, proposedOutliers.end());
    }

    for_each(proposedOutliers.begin(), proposedOutliers.end(), [this, &measurement](const int& lmId) {
        filterState.removeLandmarkById(lmId);
        measurement.camCoordinates.erase(lmId);
    });
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