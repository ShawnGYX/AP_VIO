#pragma once

#include "eigen3/Eigen/Core"

#include <functional>
#include <type_traits>

#if EIGEN_MAJOR_VERSION == 3 && EIGEN_MINOR_VERSION <= 9
namespace Eigen {
template <typename _Scalar, int _Rows> using Vector = Matrix<_Scalar, _Rows, 1>;
};
#endif

#if SUPPORT_CONCEPTS
#include <concepts>
// Concept of manifold requires that the class has an int member called `CompDim'.
template <typename T>
concept Manifold = requires(T t) {
    std::is_same<decltype(&T::CompDim, char(0)), int>::value;
};
#endif

#if SUPPORT_CONCEPTS
template <Manifold M> struct CoordinateChart {
#else
template <typename M> struct CoordinateChart {
#endif
    const std::function<Eigen::Matrix<double, M::CompDim, 1>(const M&, const M&)> chart;
    const std::function<M(const Eigen::Vector<double, M::CompDim>&, const M&)> chartInv;

    Eigen::Vector<double, M::CompDim> operator()(const M& xi, const M& xi0) const { return chart(xi, xi0); };
    M inv(const Eigen::Vector<double, M::CompDim>& x, const M& xi0) const { return chartInv(x, xi0); };
};

template <int EDim, int Dim = Eigen::Dynamic> struct EmbeddedCoordinateChart {
    using EManifold = Eigen::Matrix<double, EDim, 1>;
    const std::function<Eigen::Matrix<double, Dim, 1>(const EManifold&, const EManifold&)> chart;
    const std::function<EManifold(const Eigen::Vector<double, Dim>&, const EManifold&)> chartInv;
    const std::function<Eigen::Matrix<double, Dim, EDim>(const EManifold&)> chartDiff0;
    const std::function<Eigen::Matrix<double, EDim, Dim>(const EManifold&)> chartInvDiff0;

    Eigen::Vector<double, Dim> operator()(const EManifold& xi, const EManifold& xi0) const { return chart(xi, xi0); };
    EManifold inv(const Eigen::Vector<double, Dim>& x, const EManifold& xi0) const { return chartInv(x, xi0); };

    Eigen::Matrix<double, Dim, EDim> diff0(const EManifold& xi0) const { return chartDiff0(xi0); };
    Eigen::Matrix<double, EDim, Dim> invDiff0(const EManifold& xi0) const { return chartInvDiff0(xi0); };
};

Eigen::MatrixXd numericalDifferential(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f, const Eigen::VectorXd& x, double h = -1.0);