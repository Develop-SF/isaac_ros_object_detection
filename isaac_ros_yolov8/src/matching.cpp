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
#include <vector>

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{
namespace matching
{

double calc_iou(const std::array<double, 4> & tlwh1, const std::array<double, 4> & tlwh2)
{
  // tlwh format: [top, left, width, height]
  double left1 = tlwh1[1];
  double top1 = tlwh1[0];
  double right1 = left1 + tlwh1[2];
  double bottom1 = top1 + tlwh1[3];

  double left2 = tlwh2[1];
  double top2 = tlwh2[0];
  double right2 = left2 + tlwh2[2];
  double bottom2 = top2 + tlwh2[3];

  // Calculate intersection
  double inter_left = std::max(left1, left2);
  double inter_top = std::max(top1, top2);
  double inter_right = std::min(right1, right2);
  double inter_bottom = std::min(bottom1, bottom2);

  double inter_width = std::max(0.0, inter_right - inter_left);
  double inter_height = std::max(0.0, inter_bottom - inter_top);
  double inter_area = inter_width * inter_height;

  // Calculate union
  double area1 = tlwh1[2] * tlwh1[3];
  double area2 = tlwh2[2] * tlwh2[3];
  double union_area = area1 + area2 - inter_area;

  if (union_area <= 0.0) {
    return 0.0;
  }

  return inter_area / union_area;
}

Eigen::MatrixXd iou_distance(
  const std::vector<STrack *> & tracks,
  const std::vector<STrack *> & detections)
{
  if (tracks.empty() || detections.empty()) {
    return Eigen::MatrixXd::Ones(tracks.size(), detections.size());
  }

  Eigen::MatrixXd cost_matrix(tracks.size(), detections.size());

  for (size_t i = 0; i < tracks.size(); ++i) {
    for (size_t j = 0; j < detections.size(); ++j) {
      double iou = calc_iou(tracks[i]->tlwh(), detections[j]->tlwh());
      cost_matrix(i, j) = 1.0 - iou;  // Convert IoU to distance
    }
  }

  return cost_matrix;
}

Eigen::MatrixXd fuse_score(
  const Eigen::MatrixXd & cost_matrix,
  const std::vector<STrack *> & detections)
{
  if (detections.empty()) {
    return cost_matrix;
  }

  Eigen::MatrixXd fused_cost = cost_matrix;

  for (int j = 0; j < cost_matrix.cols(); ++j) {
    for (int i = 0; i < cost_matrix.rows(); ++i) {
      fused_cost(i, j) = cost_matrix(i, j) * (1.0 - detections[j]->score());
    }
  }

  return fused_cost;
}

// Simple greedy assignment algorithm (can be replaced with Hungarian algorithm for better results)
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>> 
linear_assignment(const Eigen::MatrixXd & cost_matrix, double thresh)
{
  std::vector<std::pair<int, int>> matches;
  std::vector<int> unmatched_a;
  std::vector<int> unmatched_b;

  if (cost_matrix.size() == 0) {
    for (int i = 0; i < cost_matrix.rows(); ++i) {
      unmatched_a.push_back(i);
    }
    for (int j = 0; j < cost_matrix.cols(); ++j) {
      unmatched_b.push_back(j);
    }
    return {matches, unmatched_a, unmatched_b};
  }

  std::set<int> matched_a;
  std::set<int> matched_b;

  // Greedy assignment: find minimum cost associations
  std::vector<std::tuple<double, int, int>> costs;
  for (int i = 0; i < cost_matrix.rows(); ++i) {
    for (int j = 0; j < cost_matrix.cols(); ++j) {
      if (cost_matrix(i, j) < thresh) {
        costs.push_back({cost_matrix(i, j), i, j});
      }
    }
  }

  // Sort by cost
  std::sort(costs.begin(), costs.end());

  // Assign greedily
  for (const auto & [cost, i, j] : costs) {
    if (matched_a.find(i) == matched_a.end() && matched_b.find(j) == matched_b.end()) {
      matches.push_back({i, j});
      matched_a.insert(i);
      matched_b.insert(j);
    }
  }

  // Find unmatched
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

