/*
 * Filename: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src/detection2_d_array_vlm_filter.cpp
 * Path: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src
 * Created Date: Monday, November 3rd 2025, 11:03:24 am
 * Author: Wen-Yu Chien
 * Description: Matching functions for ByteTrack object tracking
 * Copyright (c) 2025 Copyright (c) 2025 Shinfang Global
 */

#ifndef ISAAC_ROS_YOLOV8__MATCHING_HPP_
#define ISAAC_ROS_YOLOV8__MATCHING_HPP_

#include <Eigen/Dense>
#include <vector>
#include <array>

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{

// Forward declaration
class STrack;

namespace matching
{

/**
 * @brief Calculate IoU (Intersection over Union) between two bounding boxes
 * @param tlwh1 First bbox in top-left-width-height format
 * @param tlwh2 Second bbox in top-left-width-height format
 * @return IoU value between 0 and 1
 */
double calc_iou(const std::array<double, 4> & tlwh1, const std::array<double, 4> & tlwh2);

/**
 * @brief Calculate IoU distance matrix between tracks and detections
 * @param tracks Vector of track objects
 * @param detections Vector of detection objects
 * @return Distance matrix (1 - IoU)
 */
Eigen::MatrixXd iou_distance(
  const std::vector<STrack *> & tracks,
  const std::vector<STrack *> & detections);

/**
 * @brief Fuse detection scores with distance matrix
 * @param cost_matrix Distance matrix
 * @param detections Vector of detection objects
 * @return Fused cost matrix
 */
Eigen::MatrixXd fuse_score(
  const Eigen::MatrixXd & cost_matrix,
  const std::vector<STrack *> & detections);

/**
 * @brief Linear assignment using Hungarian algorithm
 * @param cost_matrix Cost matrix
 * @param thresh Threshold for matching
 * @return Tuple of (matches, unmatched_tracks, unmatched_detections)
 */
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>> 
linear_assignment(const Eigen::MatrixXd & cost_matrix, double thresh);

}  // namespace matching
}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_YOLOV8__MATCHING_HPP_

