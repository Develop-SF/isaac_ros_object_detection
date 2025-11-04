// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_yolov8/byte_tracker_node.hpp"
#include "isaac_ros_yolov8/matching.hpp"

#include <algorithm>
#include <set>
#include <unordered_set>

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{

ByteTrackerNode::ByteTrackerNode(const rclcpp::NodeOptions & options)
: Node("byte_tracker_node", options),
  frame_id_(0)
{
  // Declare parameters
  track_high_thresh_ = declare_parameter<double>("track_high_thresh", 0.6);
  track_low_thresh_ = declare_parameter<double>("track_low_thresh", 0.1);
  new_track_thresh_ = declare_parameter<double>("new_track_thresh", 0.7);
  match_thresh_ = declare_parameter<double>("match_thresh", 0.8);
  max_time_lost_ = declare_parameter<int>("max_time_lost", 30);
  fuse_score_ = declare_parameter<bool>("fuse_score", true);

  // Initialize Kalman filter
  kalman_filter_ = std::make_shared<KalmanFilterXYAH>();

  // Create subscriber and publisher
  detection_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
    "detections_input", 10,
    std::bind(&ByteTrackerNode::detectionCallback, this, std::placeholders::_1));

  tracked_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>("tracked_detections", 10);

  RCLCPP_INFO(
    get_logger(),
    "ByteTracker initialized with track_high_thresh=%.2f, track_low_thresh=%.2f, "
    "new_track_thresh=%.2f, match_thresh=%.2f, max_time_lost=%d",
    track_high_thresh_, track_low_thresh_, new_track_thresh_, match_thresh_, max_time_lost_);
}

ByteTrackerNode::~ByteTrackerNode() = default;

void ByteTrackerNode::detectionCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
{
  auto tracked_msg = update(*msg);
  tracked_pub_->publish(tracked_msg);
}

vision_msgs::msg::Detection2DArray ByteTrackerNode::update(
  const vision_msgs::msg::Detection2DArray & detections_msg)
{
  frame_id_++;

  // Separate detections by confidence
  vision_msgs::msg::Detection2DArray high_detections;
  vision_msgs::msg::Detection2DArray low_detections;

  for (const auto & det : detections_msg.detections) {
    double score = det.results.empty() ? 0.0 : det.results[0].hypothesis.score;
    if (score >= track_high_thresh_) {
      high_detections.detections.push_back(det);
    } else if (score >= track_low_thresh_) {
      low_detections.detections.push_back(det);
    }
  }

  // Initialize tracks from detections
  std::vector<STrack> detections = init_track(high_detections);
  std::vector<STrack> detections_second = init_track(low_detections);

  // Separate tracked and unconfirmed tracks
  std::vector<STrack> unconfirmed;
  std::vector<STrack> tracked_stracks;

  for (auto & track : tracked_stracks_) {
    if (!track.is_activated()) {
      unconfirmed.push_back(track);
    } else {
      tracked_stracks.push_back(track);
    }
  }

  // Combine tracked and lost tracks for matching
  std::vector<STrack> strack_pool = joint_stracks(tracked_stracks, lost_stracks_);

  // Predict current locations with Kalman filter
  std::vector<STrack *> strack_pool_ptrs;
  for (auto & track : strack_pool) {
    strack_pool_ptrs.push_back(&track);
  }
  STrack::multi_predict(strack_pool_ptrs, kalman_filter_);

  // First association with high score detections
  std::vector<STrack *> detections_ptrs;
  for (auto & det : detections) {
    detections_ptrs.push_back(&det);
  }

  Eigen::MatrixXd dists = get_dists(strack_pool_ptrs, detections_ptrs);
  auto [matches, u_track, u_detection] = 
    matching::linear_assignment(dists, match_thresh_);

  std::vector<STrack> activated_stracks;
  std::vector<STrack> refind_stracks;
  std::vector<STrack> lost_stracks;
  std::vector<STrack> removed_stracks;

  // Update matched tracks
  for (const auto & [itracked, idet] : matches) {
    STrack & track = strack_pool[itracked];
    STrack & det = detections[idet];
    
    if (track.state() == TrackState::Tracked) {
      track.update(det, frame_id_);
      activated_stracks.push_back(track);
    } else {
      track.re_activate(det, frame_id_, false);
      refind_stracks.push_back(track);
    }
  }

  // Second association with low score detections
  std::vector<STrack> r_tracked_stracks;
  for (int i : u_track) {
    if (strack_pool[i].state() == TrackState::Tracked) {
      r_tracked_stracks.push_back(strack_pool[i]);
    }
  }

  std::vector<STrack *> r_tracked_ptrs;
  for (auto & track : r_tracked_stracks) {
    r_tracked_ptrs.push_back(&track);
  }

  std::vector<STrack *> detections_second_ptrs;
  for (auto & det : detections_second) {
    detections_second_ptrs.push_back(&det);
  }

  Eigen::MatrixXd dists_second = matching::iou_distance(r_tracked_ptrs, detections_second_ptrs);
  auto [matches_second, u_track_second, u_detection_second] = 
    matching::linear_assignment(dists_second, 0.5);

  for (const auto & [itracked, idet] : matches_second) {
    STrack & track = r_tracked_stracks[itracked];
    STrack & det = detections_second[idet];
    
    if (track.state() == TrackState::Tracked) {
      track.update(det, frame_id_);
      activated_stracks.push_back(track);
    } else {
      track.re_activate(det, frame_id_, false);
      refind_stracks.push_back(track);
    }
  }

  // Mark unmatched tracked as lost
  for (int it : u_track_second) {
    STrack & track = r_tracked_stracks[it];
    if (track.state() != TrackState::Lost) {
      track.mark_lost();
      lost_stracks.push_back(track);
    }
  }

  // Deal with unconfirmed tracks
  std::vector<STrack> detections_remain;
  for (int i : u_detection) {
    detections_remain.push_back(detections[i]);
  }

  std::vector<STrack *> unconfirmed_ptrs;
  for (auto & track : unconfirmed) {
    unconfirmed_ptrs.push_back(&track);
  }

  std::vector<STrack *> detections_remain_ptrs;
  for (auto & det : detections_remain) {
    detections_remain_ptrs.push_back(&det);
  }

  Eigen::MatrixXd dists_unconf = get_dists(unconfirmed_ptrs, detections_remain_ptrs);
  auto [matches_unconf, u_unconfirmed, u_detection_remain] = 
    matching::linear_assignment(dists_unconf, 0.7);

  for (const auto & [itracked, idet] : matches_unconf) {
    unconfirmed[itracked].update(detections_remain[idet], frame_id_);
    activated_stracks.push_back(unconfirmed[itracked]);
  }

  for (int it : u_unconfirmed) {
    STrack & track = unconfirmed[it];
    track.mark_removed();
    removed_stracks.push_back(track);
  }

  // Initialize new tracks
  for (int inew : u_detection_remain) {
    STrack & track = detections_remain[inew];
    if (track.score() < new_track_thresh_) {
      continue;
    }
    track.activate(kalman_filter_, frame_id_);
    activated_stracks.push_back(track);
  }

  // Update state: remove tracks that have been lost for too long
  for (auto & track : lost_stracks_) {
    if (frame_id_ - track.end_frame() > max_time_lost_) {
      track.mark_removed();
      removed_stracks.push_back(track);
    }
  }

  // Update internal state
  std::vector<STrack> tracked_stracks_keep;
  for (const auto & track : tracked_stracks_) {
    if (track.state() == TrackState::Tracked) {
      tracked_stracks_keep.push_back(track);
    }
  }

  tracked_stracks_ = joint_stracks(tracked_stracks_keep, activated_stracks);
  tracked_stracks_ = joint_stracks(tracked_stracks_, refind_stracks);
  lost_stracks_ = sub_stracks(lost_stracks_, tracked_stracks_);
  
  for (const auto & track : lost_stracks) {
    lost_stracks_.push_back(track);
  }
  
  lost_stracks_ = sub_stracks(lost_stracks_, removed_stracks_);
  
  auto [tracked_clean, lost_clean] = remove_duplicate_stracks(tracked_stracks_, lost_stracks_);
  tracked_stracks_ = tracked_clean;
  lost_stracks_ = lost_clean;

  for (const auto & track : removed_stracks) {
    removed_stracks_.push_back(track);
  }

  // Limit removed_stracks size
  if (removed_stracks_.size() > 1000) {
    removed_stracks_.erase(
      removed_stracks_.begin(),
      removed_stracks_.begin() + (removed_stracks_.size() - 999));
  }

  // Create output message
  vision_msgs::msg::Detection2DArray output_msg;
  output_msg.header = detections_msg.header;

  for (const auto & track : tracked_stracks_) {
    if (track.is_activated()) {
      output_msg.detections.push_back(track.to_detection());
    }
  }

  RCLCPP_DEBUG(
    get_logger(),
    "Frame %d: %zu tracked, %zu lost, %zu removed",
    frame_id_, tracked_stracks_.size(), lost_stracks_.size(), removed_stracks_.size());

  return output_msg;
}

std::vector<STrack> ByteTrackerNode::init_track(
  const vision_msgs::msg::Detection2DArray & detections)
{
  std::vector<STrack> tracks;
  tracks.reserve(detections.detections.size());

  for (const auto & det : detections.detections) {
    tracks.emplace_back(det);
  }

  return tracks;
}

Eigen::MatrixXd ByteTrackerNode::get_dists(
  const std::vector<STrack *> & tracks,
  const std::vector<STrack *> & detections)
{
  Eigen::MatrixXd dists = matching::iou_distance(tracks, detections);
  
  if (fuse_score_) {
    dists = matching::fuse_score(dists, detections);
  }
  
  return dists;
}

std::vector<STrack> ByteTrackerNode::joint_stracks(
  const std::vector<STrack> & tlista,
  const std::vector<STrack> & tlistb)
{
  std::unordered_set<int> exists;
  std::vector<STrack> res;
  res.reserve(tlista.size() + tlistb.size());

  for (const auto & t : tlista) {
    exists.insert(t.track_id());
    res.push_back(t);
  }

  for (const auto & t : tlistb) {
    if (exists.find(t.track_id()) == exists.end()) {
      exists.insert(t.track_id());
      res.push_back(t);
    }
  }

  return res;
}

std::vector<STrack> ByteTrackerNode::sub_stracks(
  const std::vector<STrack> & tlista,
  const std::vector<STrack> & tlistb)
{
  std::unordered_set<int> track_ids_b;
  for (const auto & t : tlistb) {
    track_ids_b.insert(t.track_id());
  }

  std::vector<STrack> res;
  res.reserve(tlista.size());

  for (const auto & t : tlista) {
    if (track_ids_b.find(t.track_id()) == track_ids_b.end()) {
      res.push_back(t);
    }
  }

  return res;
}

std::pair<std::vector<STrack>, std::vector<STrack>> ByteTrackerNode::remove_duplicate_stracks(
  const std::vector<STrack> & stracksa,
  const std::vector<STrack> & stracksb)
{
  std::vector<STrack *> ptrs_a;
  std::vector<STrack *> ptrs_b;
  std::vector<STrack> copy_a = stracksa;
  std::vector<STrack> copy_b = stracksb;

  for (auto & t : copy_a) {
    ptrs_a.push_back(&t);
  }
  for (auto & t : copy_b) {
    ptrs_b.push_back(&t);
  }

  Eigen::MatrixXd pdist = matching::iou_distance(ptrs_a, ptrs_b);

  std::set<int> dupa;
  std::set<int> dupb;

  for (int p = 0; p < pdist.rows(); ++p) {
    for (int q = 0; q < pdist.cols(); ++q) {
      if (pdist(p, q) < 0.15) {
        int timep = stracksa[p].frame_id() - stracksa[p].start_frame();
        int timeq = stracksb[q].frame_id() - stracksb[q].start_frame();
        
        if (timep > timeq) {
          dupb.insert(q);
        } else {
          dupa.insert(p);
        }
      }
    }
  }

  std::vector<STrack> resa;
  std::vector<STrack> resb;

  for (size_t i = 0; i < stracksa.size(); ++i) {
    if (dupa.find(i) == dupa.end()) {
      resa.push_back(stracksa[i]);
    }
  }

  for (size_t i = 0; i < stracksb.size(); ++i) {
    if (dupb.find(i) == dupb.end()) {
      resb.push_back(stracksb[i]);
    }
  }

  return {resa, resb};
}

void ByteTrackerNode::reset()
{
  tracked_stracks_.clear();
  lost_stracks_.clear();
  removed_stracks_.clear();
  frame_id_ = 0;
  STrack::reset_id();
}

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolov8::ByteTrackerNode)

