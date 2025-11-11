/*
 * Filename: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src/detection2_d_array_vlm_filter.cpp
 * Path: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src
 * Created Date: Monday, November 3rd 2025, 11:03:24 am
 * Author: Wen-Yu Chien
 * Description: Matching functions for ByteTrack object tracking
 * Copyright (c) 2025 Copyright (c) 2025 Shinfang Global
 */

#include "isaac_ros_yolov8/matching.hpp"
#include "isaac_ros_yolov8/strack.hpp"
#include <algorithm>
#include <limits>
#include <set>

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{
namespace matching
{

float compute_iou(const Eigen::Vector4f & bbox1, const Eigen::Vector4f & bbox2)
{
  float x1_min = bbox1(0);
  float y1_min = bbox1(1);
  float x1_max = bbox1(2);
  float y1_max = bbox1(3);

  float x2_min = bbox2(0);
  float y2_min = bbox2(1);
  float x2_max = bbox2(2);
  float y2_max = bbox2(3);

  // Compute intersection area
  float inter_x_min = std::max(x1_min, x2_min);
  float inter_y_min = std::max(y1_min, y2_min);
  float inter_x_max = std::min(x1_max, x2_max);
  float inter_y_max = std::min(y1_max, y2_max);

  float inter_width = std::max(0.0f, inter_x_max - inter_x_min);
  float inter_height = std::max(0.0f, inter_y_max - inter_y_min);
  float inter_area = inter_width * inter_height;

  // Compute union area
  float area1 = (x1_max - x1_min) * (y1_max - y1_min);
  float area2 = (x2_max - x2_min) * (y2_max - y2_min);
  float union_area = area1 + area2 - inter_area;

  if (union_area <= 0.0f) {
    return 0.0f;
  }

  return inter_area / union_area;
}

Eigen::MatrixXf iou_distance(
  const std::vector<STrack *> & atracks,
  const std::vector<STrack *> & btracks)
{
  if (atracks.empty() || btracks.empty()) {
    return Eigen::MatrixXf::Zero(atracks.size(), btracks.size());
  }

  Eigen::MatrixXf cost_matrix(atracks.size(), btracks.size());

  for (size_t i = 0; i < atracks.size(); ++i) {
    for (size_t j = 0; j < btracks.size(); ++j) {
      Eigen::Vector4f bbox1 = atracks[i]->xyxy();
      Eigen::Vector4f bbox2 = btracks[j]->xyxy();
      float iou = compute_iou(bbox1, bbox2);
      cost_matrix(i, j) = 1.0f - iou;  // Distance = 1 - IoU
    }
  }

  return cost_matrix;
}

Eigen::MatrixXf fuse_score(
  const Eigen::MatrixXf & cost_matrix,
  const std::vector<STrack *> & detections)
{
  if (detections.empty() || cost_matrix.cols() == 0) {
    return cost_matrix;
  }

  Eigen::MatrixXf fused_matrix = cost_matrix;

  for (int j = 0; j < fused_matrix.cols() && j < static_cast<int>(detections.size()); ++j) {
    fused_matrix.col(j) *= (1.0f - detections[j]->score);
  }

  return fused_matrix;
}

// Hungarian algorithm implementation for linear assignment
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
linear_assignment(const Eigen::MatrixXf & cost_matrix, float thresh)
{
  std::vector<std::pair<int, int>> matches;
  std::vector<int> unmatched_a;
  std::vector<int> unmatched_b;

  if (cost_matrix.rows() == 0 || cost_matrix.cols() == 0) {
    for (int i = 0; i < cost_matrix.rows(); ++i) {
      unmatched_a.push_back(i);
    }
    for (int j = 0; j < cost_matrix.cols(); ++j) {
      unmatched_b.push_back(j);
    }
    return {matches, unmatched_a, unmatched_b};
  }

  // Simple greedy assignment algorithm
  // For a more accurate implementation, consider using a proper Hungarian algorithm library
  std::set<int> matched_a;
  std::set<int> matched_b;

  // Create a list of all potential assignments with their costs
  std::vector<std::tuple<float, int, int>> assignments;
  for (int i = 0; i < cost_matrix.rows(); ++i) {
    for (int j = 0; j < cost_matrix.cols(); ++j) {
      if (cost_matrix(i, j) < thresh) {
        assignments.push_back({cost_matrix(i, j), i, j});
      }
    }
  }

  // Sort by cost (ascending)
  std::sort(
    assignments.begin(), assignments.end(),
    [](const auto & a, const auto & b) {return std::get<0>(a) < std::get<0>(b);});

  // Greedily assign tracks
  for (const auto & [cost, i, j] : assignments) {
    if (matched_a.find(i) == matched_a.end() && matched_b.find(j) == matched_b.end()) {
      matches.push_back({i, j});
      matched_a.insert(i);
      matched_b.insert(j);
    }
  }

  // Find unmatched tracks
  for (int i = 0; i < cost_matrix.rows(); ++i) {
    if (matched_a.find(i) == matched_a.end()) {
      unmatched_a.push_back(i);
    }
  }

  for (int j = 0; j < cost_matrix.cols(); ++j) {
    if (matched_b.find(j) == matched_b.end()) {
      unmatched_b.push_back(j);
    }
  }

  return {matches, unmatched_a, unmatched_b};
}

}  // namespace matching
}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

