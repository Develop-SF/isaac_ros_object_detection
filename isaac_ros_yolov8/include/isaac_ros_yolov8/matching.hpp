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
#include <tuple>

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
 * @brief Compute IoU distance matrix between two lists of tracks
 * 
 * @param atracks List of tracks A
 * @param btracks List of tracks B
 * @return Eigen::MatrixXf Distance matrix where distance = 1 - IoU
 */
Eigen::MatrixXf iou_distance(
  const std::vector<STrack *> & atracks,
  const std::vector<STrack *> & btracks);

/**
 * @brief Fuse detection scores with cost matrix
 * 
 * @param cost_matrix The original cost matrix
 * @param detections List of detections
 * @return Eigen::MatrixXf Fused cost matrix
 */
Eigen::MatrixXf fuse_score(
  const Eigen::MatrixXf & cost_matrix,
  const std::vector<STrack *> & detections);

/**
 * @brief Perform linear assignment using Hungarian algorithm
 * 
 * @param cost_matrix The cost matrix for assignment
 * @param thresh Threshold for valid assignments
 * @return std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
 *         Matches, unmatched tracks A, unmatched tracks B
 */
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
linear_assignment(const Eigen::MatrixXf & cost_matrix, float thresh);

/**
 * @brief Compute IoU between two bounding boxes in xyxy format
 * 
 * @param bbox1 First bounding box [x1, y1, x2, y2]
 * @param bbox2 Second bounding box [x1, y1, x2, y2]
 * @return float IoU value
 */
float compute_iou(const Eigen::Vector4f & bbox1, const Eigen::Vector4f & bbox2);

}  // namespace matching
}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_YOLOV8__MATCHING_HPP_

