#include "eqvio/mathematical/Geometry.h"
#include "eigen3/Eigen/Cholesky"
#include <random>

Eigen::MatrixXd
numericalDifferential(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f, const Eigen::VectorXd& x, double h) {
    if (h < 0) {
        h = std::cbrt(std::numeric_limits<double>::epsilon());
    }
    Eigen::MatrixXd Df(f(x).rows(), x.rows());
    for (int j = 0; j < Df.cols(); ++j) {
        const Eigen::VectorXd ej = Eigen::VectorXd::Unit(Df.cols(), j);
        Df.col(j) = (f(x + h * ej) - f(x - h * ej)) / (2 * h);
    }
    return Df;
}

Eigen::VectorXd sampleGaussianDistribution(const Eigen::MatrixXd& covariance) {
    static std::random_device rd;
    static std::mt19937 rng = std::mt19937(rd());
    static std::normal_distribution<> dist{0, 1};
    const int n = covariance.rows();
    Eigen::VectorXd x(n);

    for (int i = 0; i < n; ++i) {
        x(i) = dist(rng);
    }

    Eigen::MatrixXd L = covariance.llt().matrixL();
    Eigen::VectorXd sample = L * x;

    return sample;
}

std::vector<Eigen::VectorXd> sampleGaussianDistribution(const Eigen::MatrixXd& covariance, const size_t& numSamples) {
    static std::random_device rd;
    static std::mt19937 rng = std::mt19937(rd());
    static std::normal_distribution<> dist{0, 1};

    const int n = covariance.rows();
    Eigen::MatrixXd L = covariance.llt().matrixL();

    std::vector<Eigen::VectorXd> samples(numSamples);
    for (Eigen::VectorXd& sample : samples) {
        Eigen::VectorXd x(n);
        for (int i = 0; i < n; ++i) {
            x(i) = dist(rng);
        }
        sample = L * x;
    }

    return samples;
}