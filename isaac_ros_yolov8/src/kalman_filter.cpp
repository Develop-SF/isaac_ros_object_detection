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
: std_weight_position_(1.0 / 20.0),
  std_weight_velocity_(1.0 / 160.0)
{
  // Initialize motion matrix (state transition matrix)
  // State is 8-dimensional: [x, y, a, h, vx, vy, va, vh]
  motion_mat_ = Eigen::MatrixXd::Identity(8, 8);
  // Position updates with velocity (assuming dt = 1)
  for (int i = 0; i < 4; ++i) {
    motion_mat_(i, i + 4) = 1.0;
  }

  // Initialize measurement matrix
  // We measure [x, y, a, h] from detections
  update_mat_ = Eigen::MatrixXd::Zero(4, 8);
  for (int i = 0; i < 4; ++i) {
    update_mat_(i, i) = 1.0;
  }
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilterXYAH::initiate(
  const Eigen::Vector4d & measurement)
{
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(8);
  mean.head<4>() = measurement;

  // Initialize covariance
  Eigen::VectorXd std = Eigen::VectorXd(8);
  std(0) = 2.0 * std_weight_position_ * measurement(3);  // x
  std(1) = 2.0 * std_weight_position_ * measurement(3);  // y
  std(2) = 1e-2;  // aspect ratio
  std(3) = 2.0 * std_weight_position_ * measurement(3);  // height
  std(4) = 10.0 * std_weight_velocity_ * measurement(3);  // vx
  std(5) = 10.0 * std_weight_velocity_ * measurement(3);  // vy
  std(6) = 1e-5;  // va
  std(7) = 10.0 * std_weight_velocity_ * measurement(3);  // vh

  Eigen::MatrixXd covariance = std.array().square().matrix().asDiagonal();

  return {mean, covariance};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilterXYAH::predict(
  const Eigen::VectorXd & mean,
  const Eigen::MatrixXd & covariance)
{
  // Process noise
  Eigen::VectorXd std = Eigen::VectorXd(8);
  std(0) = std_weight_position_ * mean(3);
  std(1) = std_weight_position_ * mean(3);
  std(2) = 1e-2;
  std(3) = std_weight_position_ * mean(3);
  std(4) = std_weight_velocity_ * mean(3);
  std(5) = std_weight_velocity_ * mean(3);
  std(6) = 1e-5;
  std(7) = std_weight_velocity_ * mean(3);

  Eigen::MatrixXd motion_cov = std.array().square().matrix().asDiagonal();

  // Predict
  Eigen::VectorXd predicted_mean = motion_mat_ * mean;
  Eigen::MatrixXd predicted_covariance = 
    motion_mat_ * covariance * motion_mat_.transpose() + motion_cov;

  return {predicted_mean, predicted_covariance};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilterXYAH::update(
  const Eigen::VectorXd & mean,
  const Eigen::MatrixXd & covariance,
  const Eigen::Vector4d & measurement)
{
  auto [projected_mean, projected_cov] = project(mean, covariance);

  // Kalman gain
  Eigen::Matrix<double, 8, 4> kalman_gain = 
    covariance * update_mat_.transpose() * projected_cov.inverse();

  // Innovation
  Eigen::Vector4d innovation = measurement - projected_mean;

  // Update
  Eigen::VectorXd new_mean = mean + kalman_gain * innovation;
  Eigen::MatrixXd new_covariance = 
    covariance - kalman_gain * projected_cov * kalman_gain.transpose();

  return {new_mean, new_covariance};
}

std::pair<Eigen::Vector4d, Eigen::Matrix4d> KalmanFilterXYAH::project(
  const Eigen::VectorXd & mean,
  const Eigen::MatrixXd & covariance)
{
  // Measurement noise
  Eigen::Vector4d std;
  std(0) = std_weight_position_ * mean(3);
  std(1) = std_weight_position_ * mean(3);
  std(2) = 1e-1;
  std(3) = std_weight_position_ * mean(3);

  Eigen::Matrix4d innovation_cov = std.array().square().matrix().asDiagonal();

  Eigen::Vector4d projected_mean = update_mat_ * mean;
  Eigen::Matrix4d projected_cov = 
    update_mat_ * covariance * update_mat_.transpose() + innovation_cov;

  return {projected_mean, projected_cov};
}

std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> 
KalmanFilterXYAH::multi_predict(
  const std::vector<Eigen::VectorXd> & means,
  const std::vector<Eigen::MatrixXd> & covariances)
{
  std::vector<Eigen::VectorXd> predicted_means;
  std::vector<Eigen::MatrixXd> predicted_covariances;
  predicted_means.reserve(means.size());
  predicted_covariances.reserve(covariances.size());

  for (size_t i = 0; i < means.size(); ++i) {
    auto [pred_mean, pred_cov] = predict(means[i], covariances[i]);
    predicted_means.push_back(pred_mean);
    predicted_covariances.push_back(pred_cov);
  }

  return {predicted_means, predicted_covariances};
}

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

