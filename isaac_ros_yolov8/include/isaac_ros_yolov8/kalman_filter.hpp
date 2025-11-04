/*
 * Filename: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/include/isaac_ros_yolov8/kalman_filter.hpp
 * Path: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/include/isaac_ros_yolov8
 * Created Date: Tuesday, November 4th 2025, 12:00:00 pm
 * Author: Wen-Yu Chien
 * Description: Kalman Filter for ByteTrack object tracking
 * Copyright (c) 2025 Shinfang Global
 */

#ifndef ISAAC_ROS_YOLOV8__KALMAN_FILTER_HPP_
#define ISAAC_ROS_YOLOV8__KALMAN_FILTER_HPP_

#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{

/**
 * @brief Kalman filter for tracking bounding boxes in image space using x-y-aspect-height representation
 * 
 * The state is represented as [x, y, a, h, vx, vy, va, vh] where:
 * - (x, y) is the center position
 * - a is the aspect ratio (width/height)
 * - h is the height
 * - vx, vy, va, vh are the respective velocities
 */
class KalmanFilterXYAH
{
public:
  KalmanFilterXYAH();

  /**
   * @brief Initialize a new track from the first detection
   * @param measurement Measurement vector [x, y, a, h]
   * @return Pair of (mean state vector, covariance matrix)
   */
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> initiate(const Eigen::Vector4d & measurement);

  /**
   * @brief Predict the next state
   * @param mean Current state mean vector
   * @param covariance Current state covariance matrix
   * @return Pair of (predicted mean, predicted covariance)
   */
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> predict(
    const Eigen::VectorXd & mean,
    const Eigen::MatrixXd & covariance);

  /**
   * @brief Update the state with a new measurement
   * @param mean Current state mean vector
   * @param covariance Current state covariance matrix
   * @param measurement New measurement [x, y, a, h]
   * @return Pair of (updated mean, updated covariance)
   */
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> update(
    const Eigen::VectorXd & mean,
    const Eigen::MatrixXd & covariance,
    const Eigen::Vector4d & measurement);

  /**
   * @brief Project state distribution to measurement space
   * @param mean State mean vector
   * @param covariance State covariance matrix
   * @return Pair of (projected mean, projected covariance)
   */
  std::pair<Eigen::Vector4d, Eigen::Matrix4d> project(
    const Eigen::VectorXd & mean,
    const Eigen::MatrixXd & covariance);

  /**
   * @brief Predict multiple tracks in batch
   * @param means Vector of state mean vectors
   * @param covariances Vector of state covariance matrices
   * @return Pair of (predicted means, predicted covariances)
   */
  std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> multi_predict(
    const std::vector<Eigen::VectorXd> & means,
    const std::vector<Eigen::MatrixXd> & covariances);

private:
  double std_weight_position_;
  double std_weight_velocity_;

  Eigen::MatrixXd motion_mat_;   // State transition matrix
  Eigen::MatrixXd update_mat_;   // Measurement function
};

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_YOLOV8__KALMAN_FILTER_HPP_

