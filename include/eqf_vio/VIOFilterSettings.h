#pragma once

#include <memory>
#include <ostream>
#include <string>

#include "LieYaml.h"
#include "yaml-cpp/yaml.h"

#include "VIOFilter.h"
#include "common.h"

enum class CoordinateChoice { Euclidean, InvDepth, Normal };

static CoordinateChoice coordinateSelection(const YAML::Node& ccNode) {
    std::string choice;
    safeConfig(ccNode, "settings:coordinateChoice", choice);

    if (choice == "Euclidean")
        return CoordinateChoice::Euclidean;
    else if (choice == "InvDepth")
        return CoordinateChoice::InvDepth;
    else if (choice == "Normal")
        return CoordinateChoice::Normal;
    else
        throw std::runtime_error("Invalid coordinate choice. Valid choices are Euclidean, InvDepth, Normal.");
}

struct VIOFilter::Settings {
    double biasOmegaProcessVariance = 0.001;
    double biasAccelProcessVariance = 0.001;
    double attitudeProcessVariance = 0.001;
    double positionProcessVariance = 0.001;
    double velocityProcessVariance = 0.001;
    double cameraAttitudeProcessVariance = 0.001;
    double cameraPositionProcessVariance = 0.001;
    double pointProcessVariance = 0.001;

    double velOmegaVariance = 0.1;
    double velAccelVariance = 0.1;

    double measurementVariance = 0.1;
    double outlierThresholdAbs = 1e8;
    double outlierThresholdProb = 1e8;

    double initialAttitudeVariance = 1.0;
    double initialPositionVariance = 1.0;
    double initialVelocityVariance = 1.0;
    double initialCameraAttitudeVariance = 0.1;
    double initialCameraPositionVariance = 0.1;
    double initialPointVariance = 1.0;
    double initialBiasOmegaVariance = 1.0;
    double initialBiasAccelVariance = 1.0;
    double initialSceneDepth = 1.0;

    bool useInnovationLift = true;
    bool useBundleLiftType1 = true;
    bool useDiscreteInnovationLift = true;
    bool useDiscreteVelocityLift = true;
    bool fastRiccati = false;
    CoordinateChoice coordinateChoice = CoordinateChoice::Euclidean;
    liepp::SE3d cameraOffset = liepp::SE3d::Identity();

    Settings() = default;
    Settings(const YAML::Node& configNode);
};

inline VIOFilter::Settings::Settings(const YAML::Node& configNode) {
    // Configure gain matrices

    safeConfig(configNode, "processVariance:biasGyr", biasOmegaProcessVariance);
    safeConfig(configNode, "processVariance:biasAcc", biasAccelProcessVariance);
    safeConfig(configNode, "processVariance:attitude", attitudeProcessVariance);
    safeConfig(configNode, "processVariance:position", positionProcessVariance);
    safeConfig(configNode, "processVariance:velocity", velocityProcessVariance);
    safeConfig(configNode, "processVariance:point", pointProcessVariance);
    safeConfig(configNode, "processVariance:cameraAttitude", cameraAttitudeProcessVariance);
    safeConfig(configNode, "processVariance:cameraPosition", cameraPositionProcessVariance);

    safeConfig(configNode, "measurementVariance:feature", measurementVariance);
    safeConfig(configNode, "measurementVariance:featureOutlierAbs", outlierThresholdAbs);
    safeConfig(configNode, "measurementVariance:featureOutlierProb", outlierThresholdProb);

    safeConfig(configNode, "velocityVariance:gyr", velOmegaVariance);
    safeConfig(configNode, "velocityVariance:acc", velAccelVariance);

    // Configure initial gains
    safeConfig(configNode, "initialVariance:attitude", initialAttitudeVariance);
    safeConfig(configNode, "initialVariance:position", initialPositionVariance);
    safeConfig(configNode, "initialVariance:velocity", initialVelocityVariance);
    safeConfig(configNode, "initialVariance:point", initialPointVariance);
    safeConfig(configNode, "initialVariance:biasGyr", initialBiasOmegaVariance);
    safeConfig(configNode, "initialVariance:biasAcc", initialBiasAccelVariance);
    safeConfig(configNode, "initialVariance:cameraAttitude", initialCameraAttitudeVariance);
    safeConfig(configNode, "initialVariance:cameraPosition", initialCameraPositionVariance);

    // Configure computation methods
    safeConfig(configNode, "settings:useInnovationLift", useInnovationLift);
    safeConfig(configNode, "settings:useBundleLiftType1", useBundleLiftType1);
    safeConfig(configNode, "settings:useDiscreteInnovationLift", useDiscreteInnovationLift);
    safeConfig(configNode, "settings:useDiscreteVelocityLift", useDiscreteVelocityLift);
    safeConfig(configNode, "settings:fastRiccati", fastRiccati);
    coordinateChoice = coordinateSelection(configNode);

    // Configure initial settings
    safeConfig(configNode, "initialValue:sceneDepth", initialSceneDepth);
    safeConfig(configNode, "initialValue:cameraOffset", cameraOffset);
}
