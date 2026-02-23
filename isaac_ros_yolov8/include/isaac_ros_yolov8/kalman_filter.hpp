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
#include <vector>

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{

/**
 * @brief Kalman Filter for tracking bounding boxes in image space using XYAH representation
 * 
 * Implements a Kalman filter for tracking bounding boxes in image space. The 8-dimensional
 * state space (x, y, a, h, vx, vy, va, vh) contains the bounding box center position (x, y),
 * aspect ratio a, height h, and their respective velocities. Object motion follows a constant
 * velocity model, and bounding box location (x, y, a, h) is taken as a direct observation
 * of the state space (linear observation model).
 */
class KalmanFilterXYAH
{
public:
  using StateVector = Eigen::Matrix<float, 8, 1>;
  using StateMatrix = Eigen::Matrix<float, 8, 8>;
  using MeasurementVector = Eigen::Matrix<float, 4, 1>;
  using MeasurementMatrix = Eigen::Matrix<float, 4, 4>;

  /**
   * @brief Construct a new Kalman Filter XYAH object
   */
  KalmanFilterXYAH();

  /**
   * @brief Create a track from an unassociated measurement
   * 
   * @param measurement Bounding box coordinates (x, y, a, h) with center position (x, y),
   *                   aspect ratio a, and height h
   * @return std::pair<StateVector, StateMatrix> Mean vector and covariance matrix
   */
  std::pair<StateVector, StateMatrix> initiate(const MeasurementVector & measurement);

  /**
   * @brief Run Kalman filter prediction step
   * 
   * @param mean The 8-dimensional mean vector of the object state at the previous time step
   * @param covariance The 8x8 covariance matrix of the object state at the previous time step
   * @return std::pair<StateVector, StateMatrix> Predicted mean and covariance
   */
  std::pair<StateVector, StateMatrix> predict(
    const StateVector & mean,
    const StateMatrix & covariance);

  /**
   * @brief Project state distribution to measurement space
   * 
   * @param mean The state's mean vector (8 dimensional)
   * @param covariance The state's covariance matrix (8x8 dimensional)
   * @return std::pair<MeasurementVector, MeasurementMatrix> Projected mean and covariance
   */
  std::pair<MeasurementVector, MeasurementMatrix> project(
    const StateVector & mean,
    const StateMatrix & covariance);

  /**
   * @brief Run Kalman filter prediction step (vectorized version)
   * 
   * @param means The Nx8 dimensional mean matrix of the object states
   * @param covariances The Nx8x8 covariance matrix of the object states
   * @return std::pair<std::vector<StateVector>, std::vector<StateMatrix>> Predicted means and covariances
   */
  std::pair<std::vector<StateVector>, std::vector<StateMatrix>> multi_predict(
    const std::vector<StateVector> & means,
    const std::vector<StateMatrix> & covariances);

  /**
   * @brief Run Kalman filter correction step
   * 
   * @param mean The predicted state's mean vector (8 dimensional)
   * @param covariance The state's covariance matrix (8x8 dimensional)
   * @param measurement The 4 dimensional measurement vector (x, y, a, h)
   * @return std::pair<StateVector, StateMatrix> Corrected mean and covariance
   */
  std::pair<StateVector, StateMatrix> update(
    const StateVector & mean,
    const StateMatrix & covariance,
    const MeasurementVector & measurement);

private:
  Eigen::Matrix<float, 8, 8> motion_mat_;     ///< Motion matrix
  Eigen::Matrix<float, 4, 8> update_mat_;     ///< Update matrix
  float std_weight_position_;                  ///< Standard deviation weight for position
  float std_weight_velocity_;                  ///< Standard deviation weight for velocity
};

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_YOLOV8__KALMAN_FILTER_HPP_

