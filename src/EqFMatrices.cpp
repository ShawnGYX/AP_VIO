#include "eqf_vio/EqFMatrices.h"

using namespace Eigen;
using namespace std;
using namespace liepp;

const Eigen::MatrixXd
EqFCoordinateSuite::outputMatrixC(const VIOState& xi0, const VIOGroup& X, const VisionMeasurement& y) const {
    // Rows and their corresponding output components
    // [2i, 2i+2): Landmark measurement i

    // Cols and their corresponding state components
    // I am using zero indexing and half open ranges
    // [0,2): Gravity vector (deviation from e3)
    // [2,5) Body-fixed velocity
    // [5+3i,5+3(i+1)): Body-fixed landmark i

    const int M = xi0.cameraLandmarks.size();
    const vector<int> ids = y.getIds();
    const int N = ids.size();
    MatrixXd C0 = MatrixXd::Zero(2 * N, VIOSensorState::CompDim + Landmark::CompDim * M);

    for (int i = 0; i < M; ++i) {
        const int& idNum = xi0.cameraLandmarks[i].id;
        const Vector3d& qi0 = xi0.cameraLandmarks[i].p;
        const auto it_y = find(ids.begin(), ids.end(), idNum);
        const auto it_Q = find(X.id.begin(), X.id.end(), idNum);
        assert(it_Q != X.id.end());
        const int k = distance(X.id.begin(), it_Q);
        if (it_y != ids.end()) {

            assert(*it_y == *it_Q);
            assert(X.id[k] == idNum);

            const int j = distance(ids.begin(), it_y);
            C0.block<2, 3>(2 * j, VIOSensorState::CompDim + 3 * i) =
                outputMatrixCi(qi0, X.Q[k], y.cameraPtr, y.camCoordinates.at(idNum));
        }
    }

    assert(!C0.hasNaN());
    return C0;
}

/*
-------------------------------------

 Euclidean implementation

-------------------------------------
*/

Eigen::MatrixXd EqFStateMatrixA_euclid(const VIOGroup& X, const VIOState& xi0, const IMUVelocity& imuVel);
Eigen::MatrixXd EqFInputMatrixB_euclid(const VIOGroup& X, const VIOState& xi0);
Eigen::Matrix<double, 2, 3> EqFOutputMatrixCi_euclid(
    const Vector3d& q0, const SOT3d& QHat, const GIFT::GICameraPtr& camPtr, const Eigen::Vector2d& y);

VIOAlgebra liftInnovation_euclid(const Eigen::VectorXd& baseInnovation, const VIOState& xi0);
VIOGroup liftInnovationDiscrete_euclid(const Eigen::VectorXd& totalInnovation, const VIOState& xi0);
Eigen::VectorXd bundleLift_euclid(
    const Eigen::VectorXd& baseInnovation, const VIOState& xi0, const VIOGroup& X, const Eigen::MatrixXd& Sigma);

const EqFCoordinateSuite EqFCoordinateSuite_euclid{
    VIOChart_euclid,       EqFStateMatrixA_euclid,        EqFInputMatrixB_euclid, EqFOutputMatrixCi_euclid,
    liftInnovation_euclid, liftInnovationDiscrete_euclid, bundleLift_euclid};

VIOAlgebra liftInnovation_euclid(const Eigen::VectorXd& totalInnovation, const VIOState& xi0) {
    assert(totalInnovation.size() == xi0.Dim());
    VIOAlgebra Delta;

    // Delta_A
    Delta.U_A = totalInnovation.block<6, 1>(0, 0);

    // Delta w
    const Vector3d& gamma_v = totalInnovation.block<3, 1>(6, 0);
    Delta.u_w = -gamma_v - SO3d::skew(Delta.U_A.block<3, 1>(0, 0)) * xi0.sensor.velocity;

    // Delta_B
    Delta.U_B = totalInnovation.segment<6>(9) + xi0.sensor.cameraOffset.inverse().Adjoint() * Delta.U_A;

    // Delta q_i
    const int N = xi0.cameraLandmarks.size();
    Delta.id.resize(N);
    Delta.W.resize(N);
    for (int i = 0; i < N; ++i) {
        const Vector3d& gamma_qi0 = totalInnovation.segment<3>(15 + 3 * i);
        const Vector3d& qi0 = xi0.cameraLandmarks[i].p;

        // Rotation part
        Delta.W[i].block<3, 1>(0, 0) = -qi0.cross(gamma_qi0) / qi0.squaredNorm();
        // scale part
        Delta.W[i](3) = -qi0.dot(gamma_qi0) / qi0.squaredNorm();
        // id number
        Delta.id[i] = xi0.cameraLandmarks[i].id;
    }

    return Delta;
}

Eigen::VectorXd bundleLift_euclid(
    const Eigen::VectorXd& baseInnovation, const VIOState& xi0, const VIOGroup& X, const MatrixXd& Sigma) {
    // Lift the innovation to the total space using weighted least-squares

    const VIOState xiHat = stateGroupAction(X, xi0);
    const int N = xi0.cameraLandmarks.size();
    const Vector3d& eta0 = xi0.sensor.gravityDir();

    // Construct the default  Delta
    const Vector2d& gamma_gravity = baseInnovation.block<2, 1>(0, 0);
    se3d DeltaU;
    DeltaU.block<3, 1>(0, 0) = -SO3d::skew(eta0) * sphereChart_stereo.invDiff0(eta0) * gamma_gravity;
    DeltaU.block<3, 1>(3, 0) = Vector3d::Zero();

    // The unknown parts of the innovation are those corresponding to the vehicle yaw and position

    // Create some lambdas for constructing const matrices
    auto constructKPara = [](const Vector3d& eta) {
        Matrix<double, 6, 4> KPara = Matrix<double, 6, 4>::Zero();
        KPara.block<3, 1>(0, 0) = eta;
        KPara.block<3, 3>(3, 1) = Matrix3d::Identity();
        return KPara;
    };
    auto constructKPerp = [](const Vector3d& eta) {
        Matrix<double, 6, 6> KPerp = Matrix<double, 6, 6>::Zero();
        KPerp.block<3, 3>(0, 0) = Matrix3d::Identity() - eta * eta.transpose();
        KPerp.block<3, 3>(3, 1) = Matrix3d::Zero();
        return KPerp;
    };

    // Use some sensible variable names
    const SO3d R_C = xiHat.sensor.pose.R * xiHat.sensor.cameraOffset.R;
    const Matrix3d R_CTransMat = R_C.inverse().asMatrix();
    const Matrix<double, 6, 6> AdP0 = xi0.sensor.pose.Adjoint();
    const Matrix<double, 6, 4> KPara = constructKPara(eta0);
    const Matrix<double, 6, 6> KPerp = constructKPerp(eta0);
    const Matrix<double, 6, 1> DeltaUFixed = KPerp * DeltaU;

    // Set up the components of a least squares problem
    Matrix<double, Dynamic, 4> coeffMat = Matrix<double, Dynamic, 4>(3 * N, 4);
    VectorXd observationVec(3 * N);
    MatrixXd weightingTransferD = MatrixXd::Zero(11 + 3 * N, 3 * N);

    // Populate the least squares components
    for (int i = 0; i < N; ++i) {
        const Vector3d& gamma_qi0 = baseInnovation.segment<3>(11 + 3 * i);
        const Vector3d& pHat_i = xiHat.sensor.pose * xiHat.sensor.cameraOffset * xiHat.cameraLandmarks[i].p;

        // Populate the observation vector
        const Vector3d alpha = -(R_C * (X.Q[i].inverse() * gamma_qi0));
        Matrix<double, 3, 6> pHatMat;
        pHatMat.block<3, 3>(0, 0) = -SO3d::skew(pHat_i);
        pHatMat.block<3, 3>(0, 3) = Matrix3d::Identity();
        const Vector3d obsVecBlock = alpha - pHatMat * AdP0 * DeltaUFixed;
        observationVec.block<3, 1>(3 * i, 0) = obsVecBlock;

        // Populate the coefficient matrix
        Matrix<double, 3, 4> coeffMatBlock = pHatMat * AdP0 * KPara;
        coeffMat.block<3, 4>(3 * i, 0) = coeffMatBlock;

        // Populate the weighting transfer matrix
        weightingTransferD.block<3, 3>(11 + 3 * i, 3 * i) = X.Q[i].asMatrix3() * R_CTransMat;
    }

    // Compute the weighted least-squares solution
    const MatrixXd weightMat = weightingTransferD.transpose() * Sigma.inverse() * weightingTransferD;
    const Vector4d WLSSolution = (coeffMat.transpose() * weightMat * coeffMat)
                                     .householderQr()
                                     .solve(coeffMat.transpose() * weightMat * observationVec);
    DeltaU = DeltaUFixed + KPara * WLSSolution;

    // We must re-compute the velocity component of Delta
    VectorXd liftedInnovation = VectorXd(15 + 3 * N);
    liftedInnovation.block<6, 1>(0, 0) = DeltaU;
    liftedInnovation.segment(6, 9 + 3 * N) = baseInnovation.segment(2, 9 + 3 * N);

    return liftedInnovation;
}

VIOGroup liftInnovationDiscrete_euclid(const Eigen::VectorXd& totalInnovation, const VIOState& xi0) {
    // Lift the innovation discretely
    VIOGroup lift;
    lift.A = SE3d::exp(totalInnovation.segment<6>(0));
    lift.w = xi0.sensor.velocity - lift.A.R * (xi0.sensor.velocity + totalInnovation.segment<3>(6));
    lift.B =
        xi0.sensor.cameraOffset.inverse() * lift.A * xi0.sensor.cameraOffset * SE3d::exp(totalInnovation.segment<6>(9));

    // Lift for each of the points
    const int N = xi0.cameraLandmarks.size();
    lift.id.resize(N);
    lift.Q.resize(N);
    for (int i = 0; i < N; ++i) {
        const Vector3d& qi = xi0.cameraLandmarks[i].p;
        const Vector3d& Gamma_qi = totalInnovation.segment<3>(15 + 3 * i);
        const Vector3d qi1 = (qi + Gamma_qi);
        lift.Q[i].R = SO3d::SO3FromVectors(qi1.normalized(), qi.normalized());
        lift.Q[i].a = qi.norm() / qi1.norm();

        lift.id[i] = xi0.cameraLandmarks[i].id;
    }

    assert(!lift.hasNaN());

    return lift;
}

Eigen::MatrixXd EqFStateMatrixA_euclid(const VIOGroup& X, const VIOState& xi0, const IMUVelocity& imuVel) {
    const int N = xi0.cameraLandmarks.size();
    MatrixXd A0t = MatrixXd::Zero(xi0.Dim(), xi0.Dim());

    // Rows / Cols and their corresponding state components
    // I am using zero indexing and half open ranges
    // [0,6): Pose attitude and position
    // [6,9): Body-fixed velocity
    // [9,15): Camera Offset from IMU
    // [15+3i,15+3(i+1)): Body-fixed landmark i

    // Effect of velocity on translation
    A0t.block<3, 3>(3, 6) = Matrix3d::Identity();

    // Effect of gravity cov on velocity cov
    A0t.block<3, 3>(6, 0) = -GRAVITY_CONSTANT * SO3d::skew(xi0.sensor.gravityDir());

    const VIOState xi_hat = stateGroupAction(X, xi0);
    const se3d U_I = (se3d() << imuVel.omega, xi_hat.sensor.velocity).finished();

    // Effect of camera offset cov on self
    // Formula is \ad(\Ad_{\mr{T}}^{-1} \Ad_{\hat{A}} U_I)
    A0t.block<6, 6>(9, 9) = SE3d::adjoint(xi0.sensor.cameraOffset.inverse().Adjoint() * X.A.Adjoint() * U_I);

    // Effect of velocity cov on landmarks cov
    const Matrix3d R_IC = xi_hat.sensor.cameraOffset.R.asMatrix();
    const Matrix3d R_Ahat = X.A.R.asMatrix();
    for (int i = 0; i < N; ++i) {
        const Matrix3d Qhat_i = X.Q[i].R.asMatrix() * X.Q[i].a;
        A0t.block<3, 3>(VIOSensorState::CompDim + 3 * i, 6) = -Qhat_i * R_IC.transpose() * R_Ahat.transpose();
    }

    // Effect of camera offset cov on landmarks cov
    const Matrix<double, 6, 6> commonTerm =
        X.B.inverse().Adjoint() * SE3d::adjoint(xi0.sensor.cameraOffset.inverse().Adjoint() * X.A.Adjoint() * U_I);
    for (int i = 0; i < N; ++i) {
        Matrix<double, 3, 6> temp;
        temp << SO3d::skew(xi0.cameraLandmarks[i].p) * X.Q[i].R.asMatrix(), -X.Q[i].a * X.Q[i].R.asMatrix();
        A0t.block<3, 6>(VIOSensorState::CompDim + 3 * i, 9) = temp * commonTerm;
    }

    // Effect of landmark cov on landmark cov
    const se3d U_C = xi_hat.sensor.cameraOffset.inverse().Adjoint() * U_I;
    const Vector3d v_C = U_C.block<3, 1>(3, 0);
    for (int i = 0; i < N; ++i) {
        const Matrix3d Qhat_i = X.Q[i].R.asMatrix() * X.Q[i].a;
        const Vector3d& qhat_i = xi_hat.cameraLandmarks[i].p;
        const Matrix3d A_qi =
            -Qhat_i * (SO3d::skew(qhat_i) * SO3d::skew(v_C) - 2 * v_C * qhat_i.transpose() + qhat_i * v_C.transpose()) *
            Qhat_i.inverse() * (1 / qhat_i.squaredNorm());
        A0t.block<3, 3>(VIOSensorState::CompDim + 3 * i, VIOSensorState::CompDim + 3 * i) = A_qi;
    }

    assert(!A0t.hasNaN());

    return A0t;
}

Eigen::Matrix<double, 2, 3> EqFOutputMatrixCi_euclid(
    const Vector3d& q0, const SOT3d& QHat, const GIFT::GICameraPtr& camPtr, const Eigen::Vector2d& y) {
    const Vector3d qHat = QHat.inverse() * q0;
    const Vector3d yHat = qHat.normalized();

    // Get the exponential map for epsilon
    Matrix<double, 4, 3> m2g;
    m2g.block<3, 3>(0, 0) = -SO3d::skew(q0);
    m2g.block<1, 3>(3, 0) = -q0.transpose();
    m2g = m2g / q0.squaredNorm();

    auto DRho = [&camPtr](const Vector3d& yVec) {
        Matrix<double, 3, 4> DRhoVec;
        DRhoVec << SO3d::skew(yVec), Vector3d::Zero();
        Matrix<double, 2, 4> DRho = camPtr->projectionJacobian(yVec) * DRhoVec;
        return DRho;
    };

    const Vector3d yVec = camPtr->undistortPoint(y);

    Matrix<double, 2, 3> Cti = 0.5 * (DRho(yVec) + DRho(yHat)) * QHat.inverse().Adjoint() * m2g;
    return Cti;
}

Eigen::MatrixXd EqFInputMatrixB_euclid(const VIOGroup& X, const VIOState& xi0) {
    const int N = xi0.cameraLandmarks.size();
    MatrixXd Bt = MatrixXd::Zero(xi0.Dim(), 6);

    // Rows and their corresponding state components
    // I am using zero indexing and half open ranges
    // [0,6): Pose attitude and position
    // [6,9): Body-fixed velocity
    // [9,15): Camera Offset from IMU
    // [15+3i,15+3(i+1)): Body-fixed landmark i

    // Cols and their corresponding output components
    // [0, 3): Angular velocity omega
    // [3, 6): Linear acceleration accel

    const VIOState xi_hat = stateGroupAction(X, xi0);

    // Attitude
    const Matrix3d R_A = X.A.R.asMatrix();
    Bt.block<3, 3>(0, 0) = R_A;

    // Position
    Bt.block<3, 3>(3, 0) = SO3d::skew(X.A.x) * R_A;

    // Body fixed velocity
    Bt.block<3, 3>(6, 0) = R_A * SO3d::skew(xi_hat.sensor.velocity);
    Bt.block<3, 3>(6, 3) = R_A;

    // Landmarks
    const Matrix3d RT_IC = xi_hat.sensor.cameraOffset.R.inverse().asMatrix();
    const Vector3d x_IC = xi_hat.sensor.cameraOffset.x;
    for (int i = 0; i < N; ++i) {
        const Matrix3d Qhat_i = X.Q[i].R.asMatrix() * X.Q[i].a;
        const Vector3d& qhat_i = xi_hat.cameraLandmarks[i].p;
        Bt.block<3, 3>(VIOSensorState::CompDim + 3 * i, 0) =
            Qhat_i * (SO3d::skew(qhat_i) * RT_IC + RT_IC * SO3d::skew(x_IC));
    }

    assert(!Bt.hasNaN());

    return Bt;
}

/*
-------------------------------------

 Inverse depth implementation

-------------------------------------
*/
Eigen::MatrixXd EqFStateMatrixA_invdepth(const VIOGroup& X, const VIOState& xi0, const IMUVelocity& imuVel);
Eigen::MatrixXd EqFInputMatrixB_invdepth(const VIOGroup& X, const VIOState& xi0);
Eigen::Matrix<double, 2, 3> EqFOutputMatrixCi_invdepth(
    const Vector3d& q0, const SOT3d& QHat, const GIFT::GICameraPtr& camPtr, const Eigen::Vector2d& y);

VIOAlgebra liftInnovation_invdepth(const Eigen::VectorXd& totalInnovation, const VIOState& xi0);
VIOGroup liftInnovationDiscrete_invdepth(const Eigen::VectorXd& totalInnovation, const VIOState& xi0);
Eigen::VectorXd bundleLift_invdepth(
    const Eigen::VectorXd& baseInnovation, const VIOState& xi0, const VIOGroup& X, const Eigen::MatrixXd& Sigma);

const EqFCoordinateSuite EqFCoordinateSuite_invdepth{
    VIOChart_invdepth,       EqFStateMatrixA_invdepth,        EqFInputMatrixB_invdepth, EqFOutputMatrixCi_invdepth,
    liftInnovation_invdepth, liftInnovationDiscrete_invdepth, bundleLift_invdepth};

Eigen::MatrixXd EqFStateMatrixA_invdepth(const VIOGroup& X, const VIOState& xi0, const IMUVelocity& imuVel) {
    const int N = xi0.cameraLandmarks.size();
    MatrixXd A0t = MatrixXd::Zero(xi0.Dim(), xi0.Dim());

    // Rows / Cols and their corresponding state components
    // I am using zero indexing and half open ranges
    // [0,6): Pose attitude and position
    // [6,9): Body-fixed velocity
    // [9,15): Camera Offset from IMU
    // [15+3i,15+3(i+1)): Body-fixed landmark i

    // Effect of velocity on translation
    A0t.block<3, 3>(3, 6) = Matrix3d::Identity();

    // Effect of gravity cov on velocity cov
    A0t.block<3, 3>(6, 0) = -GRAVITY_CONSTANT * SO3d::skew(xi0.sensor.gravityDir());

    const VIOState xi_hat = stateGroupAction(X, xi0);
    const se3d U_I = (se3d() << imuVel.omega, xi_hat.sensor.velocity).finished();

    // Effect of camera offset cov on self
    // Formula is \ad(\Ad_{\mr{T}}^{-1} \Ad_{\hat{A}} U_I)
    A0t.block<6, 6>(9, 9) = SE3d::adjoint(xi0.sensor.cameraOffset.inverse().Adjoint() * X.A.Adjoint() * U_I);

    const auto conv_euc2ind = [](const Vector3d& q0) {
        const double& rho0i = 1. / q0.norm();
        const Vector3d& y0i = q0 * rho0i;
        Matrix3d conv_M;
        conv_M.block<2, 3>(0, 0) =
            rho0i * sphereChart_stereo.chartDiff0(y0i) * (Matrix3d::Identity() - y0i * y0i.transpose());
        conv_M.block<1, 3>(2, 0) = -rho0i * rho0i * y0i.transpose();
        return conv_M;
    };
    const auto conv_ind2euc = [](const Vector3d& q0) {
        const double& rho0i = 1. / q0.norm();
        const Vector3d& y0i = q0 * rho0i;
        Matrix3d conv_M;
        conv_M.block<3, 2>(0, 0) = sphereChart_stereo.chartInvDiff0(y0i) / rho0i;
        conv_M.block<3, 1>(0, 2) = -y0i / (rho0i * rho0i);
        return conv_M;
    };

    // Effect of velocity cov on landmarks cov
    const Matrix3d R_IC = xi_hat.sensor.cameraOffset.R.asMatrix();
    const Matrix3d R_Ahat = X.A.R.asMatrix();
    for (int i = 0; i < N; ++i) {
        const Vector3d& q0 = xi0.cameraLandmarks[i].p;
        const Matrix3d Qhat_i = X.Q[i].R.asMatrix() * X.Q[i].a;
        A0t.block<3, 3>(VIOSensorState::CompDim + 3 * i, 6) =
            -conv_euc2ind(q0) * Qhat_i * R_IC.transpose() * R_Ahat.transpose();
    }

    // Effect of camera offset cov on landmarks cov
    const Matrix<double, 6, 6> commonTerm =
        X.B.inverse().Adjoint() * SE3d::adjoint(xi0.sensor.cameraOffset.inverse().Adjoint() * X.A.Adjoint() * U_I);
    for (int i = 0; i < N; ++i) {
        const Vector3d& q0 = xi0.cameraLandmarks[i].p;
        Matrix<double, 3, 6> temp;
        temp << SO3d::skew(q0) * X.Q[i].R.asMatrix(), -X.Q[i].a * X.Q[i].R.asMatrix();

        A0t.block<3, 6>(VIOSensorState::CompDim + 3 * i, 9) = conv_euc2ind(q0) * temp * commonTerm;
    }

    // Effect of landmark cov on landmark cov
    const se3d U_C = xi_hat.sensor.cameraOffset.inverse().Adjoint() * U_I;
    const Vector3d v_C = U_C.block<3, 1>(3, 0);
    for (int i = 0; i < N; ++i) {
        const Vector3d& q0 = xi0.cameraLandmarks[i].p;
        const Matrix3d Qhat_i = X.Q[i].R.asMatrix() * X.Q[i].a;
        const Vector3d& qhat_i = xi_hat.cameraLandmarks[i].p;
        const Matrix3d A_qi =
            -Qhat_i * (SO3d::skew(qhat_i) * SO3d::skew(v_C) - 2 * v_C * qhat_i.transpose() + qhat_i * v_C.transpose()) *
            Qhat_i.inverse() * (1 / qhat_i.squaredNorm());
        A0t.block<3, 3>(VIOSensorState::CompDim + 3 * i, VIOSensorState::CompDim + 3 * i) =
            conv_euc2ind(q0) * A_qi * conv_ind2euc(q0);
    }

    assert(!A0t.hasNaN());

    return A0t;
}

Eigen::MatrixXd EqFInputMatrixB_invdepth(const VIOGroup& X, const VIOState& xi0) {

    const auto conv_euc2ind = [](const Vector3d& q0) {
        const double& rho0i = 1. / q0.norm();
        const Vector3d& y0i = q0 * rho0i;
        Matrix3d conv_M;
        conv_M.block<2, 3>(0, 0) =
            rho0i * sphereChart_stereo.chartDiff0(y0i) * (Matrix3d::Identity() - y0i * y0i.transpose());
        conv_M.block<1, 3>(2, 0) = -rho0i * rho0i * y0i.transpose();
        return conv_M;
    };

    const int N = xi0.cameraLandmarks.size();
    MatrixXd Bt = MatrixXd::Zero(xi0.Dim(), 6);

    // Rows and their corresponding state components
    // I am using zero indexing and half open ranges
    // [0,6): Pose attitude and position
    // [6,9): Body-fixed velocity
    // [9,15): Camera Offset from IMU
    // [15+3i,15+3(i+1)): Body-fixed landmark i

    // Cols and their corresponding output components
    // [0, 3): Angular velocity omega
    // [3, 6): Linear acceleration accel

    const VIOState xi_hat = stateGroupAction(X, xi0);

    // Attitude
    const Matrix3d R_A = X.A.R.asMatrix();
    Bt.block<3, 3>(0, 0) = R_A;

    // Position
    Bt.block<3, 3>(3, 0) = SO3d::skew(X.A.x) * R_A;

    // Body fixed velocity
    Bt.block<3, 3>(6, 0) = R_A * SO3d::skew(xi_hat.sensor.velocity);
    Bt.block<3, 3>(6, 3) = R_A;

    // Landmarks
    const Matrix3d RT_IC = xi_hat.sensor.cameraOffset.R.inverse().asMatrix();
    const Vector3d x_IC = xi_hat.sensor.cameraOffset.x;
    for (int i = 0; i < N; ++i) {
        const Vector3d& q0 = xi0.cameraLandmarks[i].p;
        const Matrix3d Qhat_i = X.Q[i].R.asMatrix() * X.Q[i].a;
        const Vector3d& qhat_i = xi_hat.cameraLandmarks[i].p;
        Bt.block<3, 3>(VIOSensorState::CompDim + 3 * i, 0) =
            conv_euc2ind(q0) * Qhat_i * (SO3d::skew(qhat_i) * RT_IC + RT_IC * SO3d::skew(x_IC));
    }

    assert(!Bt.hasNaN());

    return Bt;
}

VIOAlgebra liftInnovation_invdepth(const Eigen::VectorXd& totalInnovation, const VIOState& xi0) {
    const MatrixXd& totalM = coordinateDifferential_invdepth_euclid(xi0);
    return liftInnovation_euclid(totalM.inverse() * totalInnovation, xi0);
}

VIOGroup liftInnovationDiscrete_invdepth(const Eigen::VectorXd& totalInnovation, const VIOState& xi0) {
    // const MatrixXd& totalM = coordinateDifferential_invdepth_euclid(xi0);
    // return liftInnovationDiscrete_euclid(totalM.inverse() * totalInnovation, xi0);
    const VectorXd totalInnovationEuclid = VIOChart_euclid(VIOChart_invdepth.inv(totalInnovation, xi0), xi0);
    return liftInnovationDiscrete_euclid(totalInnovationEuclid, xi0);
}

Eigen::VectorXd bundleLift_invdepth(
    const Eigen::VectorXd& baseInnovation, const VIOState& xi0, const VIOGroup& X, const Eigen::MatrixXd& Sigma) {
    const MatrixXd& M = coordinateDifferential_invdepth_euclid(xi0);
    const MatrixXd& totalM = coordinateDifferential_invdepth_euclid(xi0);
    return totalM * bundleLift_euclid(M.inverse() * baseInnovation, xi0, X, M.inverse() * Sigma * M);
}

Eigen::Matrix<double, 2, 3> EqFOutputMatrixCi_invdepth(
    const Vector3d& q0, const SOT3d& QHat, const GIFT::GICameraPtr& camPtr, const Eigen::Vector2d& y) {
    const double& r0 = q0.norm();
    const Vector3d& y0 = q0 / r0;

    Eigen::Matrix3d ind2euc;
    ind2euc.block<3, 2>(0, 0) = r0 * sphereChart_stereo.chartInvDiff0(y0);
    ind2euc.block<3, 1>(0, 2) = -r0 * q0;

    Eigen::Matrix<double, 2, 3> C0i = EqFOutputMatrixCi_euclid(q0, QHat, camPtr, y) * ind2euc;
    return C0i;
}

/*
-------------------------------------

 Normal implementation

-------------------------------------
*/
Eigen::MatrixXd EqFStateMatrixA_normal(const VIOGroup& X, const VIOState& xi0, const IMUVelocity& imuVel);
Eigen::MatrixXd EqFInputMatrixB_normal(const VIOGroup& X, const VIOState& xi0);
Eigen::Matrix<double, 2, 3> EqFOutputMatrixCi_normal(
    const Vector3d& q0, const SOT3d& QHat, const GIFT::GICameraPtr& camPtr, const Eigen::Vector2d& y);

VIOAlgebra liftInnovation_normal(const Eigen::VectorXd& baseInnovation, const VIOState& xi0);
VIOAlgebra liftInnovation_normal(const Eigen::VectorXd& totalInnovation, const VIOState& xi0);
VIOGroup liftInnovationDiscrete_normal(const Eigen::VectorXd& totalInnovation, const VIOState& xi0);
Eigen::VectorXd bundleLift_normal(
    const Eigen::VectorXd& baseInnovation, const VIOState& xi0, const VIOGroup& X, const Eigen::MatrixXd& Sigma);

const EqFCoordinateSuite EqFCoordinateSuite_normal{
    VIOChart_normal,       EqFStateMatrixA_normal,        EqFInputMatrixB_normal, EqFOutputMatrixCi_normal,
    liftInnovation_normal, liftInnovationDiscrete_normal, bundleLift_normal};

Eigen::MatrixXd EqFStateMatrixA_normal(const VIOGroup& X, const VIOState& xi0, const IMUVelocity& imuVel) {
    const MatrixXd& M = coordinateDifferential_normal_euclid(xi0);
    return M * EqFStateMatrixA_euclid(X, xi0, imuVel) * M.inverse();
}

Eigen::MatrixXd EqFInputMatrixB_normal(const VIOGroup& X, const VIOState& xi0) {
    const MatrixXd& M = coordinateDifferential_normal_euclid(xi0);
    return M * EqFInputMatrixB_euclid(X, xi0);
}

VIOAlgebra liftInnovation_normal(const Eigen::VectorXd& baseInnovation, const VIOState& xi0) {
    const MatrixXd& M = coordinateDifferential_normal_euclid(xi0);
    return liftInnovation_euclid(M.inverse() * baseInnovation, xi0);
}

VIOGroup liftInnovationDiscrete_normal(const Eigen::VectorXd& totalInnovation, const VIOState& xi0) {
    return liftInnovationDiscrete_euclid(VIOChart_euclid(VIOChart_normal.inv(totalInnovation, xi0), xi0), xi0);
}

Eigen::VectorXd bundleLift_normal(
    const Eigen::VectorXd& baseInnovation, const VIOState& xi0, const VIOGroup& X, const Eigen::MatrixXd& Sigma) {
    const MatrixXd& M = coordinateDifferential_normal_euclid(xi0);
    const MatrixXd& totalM = coordinateDifferential_normal_euclid(xi0);
    return totalM * bundleLift_euclid(M.inverse() * baseInnovation, xi0, X, M.inverse() * Sigma * M);
}

Eigen::Matrix<double, 2, 3> EqFOutputMatrixCi_normal(
    const Vector3d& q0, const SOT3d& QHat, const GIFT::GICameraPtr& camPtr, const Eigen::Vector2d& y) {
    const Vector3d& y0 = q0.normalized();
    const Vector3d& yHat = QHat.R.inverse() * y0;
    Eigen::Matrix<double, 2, 3> C0i = Eigen::Matrix<double, 2, 3>::Zero();
    C0i.block<2, 2>(0, 0) =
        camPtr->projectionJacobian(yHat) * QHat.R.asMatrix().transpose() * sphereChart_normal.chartInvDiff0(q0);
    return C0i;
}