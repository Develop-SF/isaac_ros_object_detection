/*
 * Filename: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src/detection2_d_array_vlm_filter.cpp
 * Path: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src
 * Created Date: Monday, November 3rd 2025, 11:03:24 am
 * Author: Wen-Yu Chien
 * Description: Kalman Filter for ByteTrack object tracking
 * Copyright (c) 2025 Copyright (c) 2025 Shinfang Global
 */

#include "isaac_ros_yolov8/kalman_filter.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{

KalmanFilterXYAH::KalmanFilterXYAH()
: std_weight_position_(1.0f / 20.0f),
  std_weight_velocity_(1.0f / 160.0f)
{
  constexpr int ndim = 4;
  constexpr float dt = 1.0f;

  // Create Kalman filter model matrices
  motion_mat_ = Eigen::Matrix<float, 8, 8>::Identity();
  for (int i = 0; i < ndim; ++i) {
    motion_mat_(i, ndim + i) = dt;
  }

  update_mat_ = Eigen::Matrix<float, 4, 8>::Zero();
  for (int i = 0; i < ndim; ++i) {
    update_mat_(i, i) = 1.0f;
  }
}

std::pair<KalmanFilterXYAH::StateVector, KalmanFilterXYAH::StateMatrix>
KalmanFilterXYAH::initiate(const MeasurementVector & measurement)
{
  StateVector mean = StateVector::Zero();
  mean.head<4>() = measurement;

  StateVector std;
  std << 2 * std_weight_position_ * measurement(3),
    2 * std_weight_position_ * measurement(3),
    1e-2f,
    2 * std_weight_position_ * measurement(3),
    10 * std_weight_velocity_ * measurement(3),
    10 * std_weight_velocity_ * measurement(3),
    1e-5f,
    10 * std_weight_velocity_ * measurement(3);

  StateMatrix covariance = std.array().square().matrix().asDiagonal();

  return {mean, covariance};
}

std::pair<KalmanFilterXYAH::StateVector, KalmanFilterXYAH::StateMatrix>
KalmanFilterXYAH::predict(const StateVector & mean, const StateMatrix & covariance)
{
  StateVector std;
  std << std_weight_position_ * mean(3),
    std_weight_position_ * mean(3),
    1e-2f,
    std_weight_position_ * mean(3),
    std_weight_velocity_ * mean(3),
    std_weight_velocity_ * mean(3),
    1e-5f,
    std_weight_velocity_ * mean(3);

  StateMatrix motion_cov = std.array().square().matrix().asDiagonal();

  StateVector new_mean = motion_mat_ * mean;
  StateMatrix new_covariance = motion_mat_ * covariance * motion_mat_.transpose() + motion_cov;

  return {new_mean, new_covariance};
}

std::pair<KalmanFilterXYAH::MeasurementVector, KalmanFilterXYAH::MeasurementMatrix>
KalmanFilterXYAH::project(const StateVector & mean, const StateMatrix & covariance)
{
  MeasurementVector std;
  std << std_weight_position_ * mean(3),
    std_weight_position_ * mean(3),
    1e-1f,
    std_weight_position_ * mean(3);

  MeasurementMatrix innovation_cov = std.array().square().matrix().asDiagonal();

  MeasurementVector projected_mean = update_mat_ * mean;
  MeasurementMatrix projected_cov =
    update_mat_ * covariance * update_mat_.transpose() + innovation_cov;

  return {projected_mean, projected_cov};
}

std::pair<std::vector<KalmanFilterXYAH::StateVector>,
  std::vector<KalmanFilterXYAH::StateMatrix>>
KalmanFilterXYAH::multi_predict(
  const std::vector<StateVector> & means,
  const std::vector<StateMatrix> & covariances)
{
  std::vector<StateVector> new_means;
  std::vector<StateMatrix> new_covariances;
  new_means.reserve(means.size());
  new_covariances.reserve(covariances.size());

  for (size_t i = 0; i < means.size(); ++i) {
    auto [predicted_mean, predicted_cov] = predict(means[i], covariances[i]);
    new_means.push_back(predicted_mean);
    new_covariances.push_back(predicted_cov);
  }

  return {new_means, new_covariances};
}

std::pair<KalmanFilterXYAH::StateVector, KalmanFilterXYAH::StateMatrix>
KalmanFilterXYAH::update(
  const StateVector & mean,
  const StateMatrix & covariance,
  const MeasurementVector & measurement)
{
  auto [projected_mean, projected_cov] = project(mean, covariance);

  // Compute Kalman gain: K = P * H^T * (H * P * H^T + R)^-1
  // P * H^T = covariance * update_mat_.transpose() -> (8x8) * (8x4) = (8x4)
  Eigen::Matrix<float, 8, 4> PHt = covariance * update_mat_.transpose();

  // Cholesky decomposition of projected_cov (innovation covariance)
  Eigen::LLT<MeasurementMatrix> cholesky(projected_cov);
  
  // Solve for Kalman gain using Cholesky decomposition
  // K = PHt * inv(projected_cov)
  // We solve: projected_cov * K^T = PHt^T for K^T, then transpose
  Eigen::Matrix<float, 4, 8> KT = cholesky.solve(PHt.transpose());
  Eigen::Matrix<float, 8, 4> kalman_gain = KT.transpose();

  MeasurementVector innovation = measurement - projected_mean;

  StateVector new_mean = mean + kalman_gain * innovation;
  StateMatrix new_covariance = covariance - kalman_gain * projected_cov * kalman_gain.transpose();

  return {new_mean, new_covariance};
}

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

