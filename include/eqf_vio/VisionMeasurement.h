#pragma once

#include "CSVReader.h"
#include "Geometry.h"

#include "GIFT/camera/camera.h"
#include "eigen3/Eigen/Eigen"

#include <map>
#include <memory>

struct VisionMeasurement {
    double stamp;
    std::map<int, Eigen::Vector2d> camCoordinates;
    GIFT::GICameraPtr cameraPtr;

    constexpr static int CompDim = Eigen::Dynamic;

    std::vector<int> getIds() const;

    operator Eigen::VectorXd() const;
};

VisionMeasurement operator-(const VisionMeasurement& y1, const VisionMeasurement& y2);

CSVLine& operator<<(CSVLine& line, const VisionMeasurement& vision);
CSVLine& operator>>(CSVLine& line, VisionMeasurement& vision);
