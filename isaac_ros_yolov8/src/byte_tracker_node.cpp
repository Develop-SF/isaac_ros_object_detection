/*
 * Filename: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src/byte_tracker_node.cpp
 * Path: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src/byte_tracker_node.cpp
 * Created Date: Monday, November 3rd 2025, 11:03:24 am
 * Author: Wen-Yu Chien
 * Description: ByteTrack Node for multi-object tracking
 * Copyright (c) 2025 Copyright (c) 2025 Shinfang Global
 */

#include "isaac_ros_yolov8/byte_tracker_node.hpp"
#include "isaac_ros_yolov8/matching.hpp"
#include <rclcpp_components/register_node_macro.hpp>
#include <algorithm>
#include <set>

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{

// BYTETracker implementation
BYTETracker::BYTETracker(
  float track_high_thresh,
  float track_low_thresh,
  float new_track_thresh,
  int track_buffer,
  float match_thresh,
  bool fuse_score,
  int frame_rate)
: frame_id_(0),
  max_time_lost_(static_cast<int>(frame_rate / 30.0 * track_buffer)),
  track_high_thresh_(track_high_thresh),
  track_low_thresh_(track_low_thresh),
  new_track_thresh_(new_track_thresh),
  match_thresh_(match_thresh),
  fuse_score_(fuse_score)
{
  BaseTrack::reset_id();
}

std::vector<std::shared_ptr<STrack>> BYTETracker::init_track(
  const std::vector<std::vector<float>> & detections)
{
  std::vector<std::shared_ptr<STrack>> tracks;
  for (const auto & det : detections) {
    // Detection format: [x1, y1, x2, y2, score, class_id]
    if (det.size() < 6) {
      continue;
    }

    // Convert xyxy to xywh
    float x1 = det[0];
    float y1 = det[1];
    float x2 = det[2];
    float y2 = det[3];
    float score = det[4];
    int cls = static_cast<int>(det[5]);

    float cx = (x1 + x2) / 2.0f;
    float cy = (y1 + y2) / 2.0f;
    float w = x2 - x1;
    float h = y2 - y1;

    std::vector<float> xywh = {cx, cy, w, h};
    tracks.push_back(
      std::make_shared<STrack>(xywh, score, cls, static_cast<int>(tracks.size())));
  }
  return tracks;
}

Eigen::MatrixXf BYTETracker::get_dists(
  std::vector<STrack *> & tracks,
  std::vector<STrack *> & detections)
{
  Eigen::MatrixXf dists = matching::iou_distance(tracks, detections);
  if (fuse_score_) {
    dists = matching::fuse_score(dists, detections);
  }
  return dists;
}

std::vector<std::shared_ptr<STrack>> BYTETracker::joint_stracks(
  const std::vector<std::shared_ptr<STrack>> & tlista,
  const std::vector<std::shared_ptr<STrack>> & tlistb)
{
  std::set<int> exists;
  std::vector<std::shared_ptr<STrack>> res;

  for (const auto & t : tlista) {
    exists.insert(t->track_id);
    res.push_back(t);
  }

  for (const auto & t : tlistb) {
    if (exists.find(t->track_id) == exists.end()) {
      exists.insert(t->track_id);
      res.push_back(t);
    }
  }

  return res;
}

std::vector<std::shared_ptr<STrack>> BYTETracker::sub_stracks(
  const std::vector<std::shared_ptr<STrack>> & tlista,
  const std::vector<std::shared_ptr<STrack>> & tlistb)
{
  std::set<int> track_ids_b;
  for (const auto & t : tlistb) {
    track_ids_b.insert(t->track_id);
  }

  std::vector<std::shared_ptr<STrack>> res;
  for (const auto & t : tlista) {
    if (track_ids_b.find(t->track_id) == track_ids_b.end()) {
      res.push_back(t);
    }
  }

  return res;
}

std::pair<std::vector<std::shared_ptr<STrack>>, std::vector<std::shared_ptr<STrack>>>
BYTETracker::remove_duplicate_stracks(
  const std::vector<std::shared_ptr<STrack>> & stracksa,
  const std::vector<std::shared_ptr<STrack>> & stracksb)
{
  std::vector<STrack *> a_ptrs, b_ptrs;
  for (const auto & t : stracksa) {
    a_ptrs.push_back(t.get());
  }
  for (const auto & t : stracksb) {
    b_ptrs.push_back(t.get());
  }

  Eigen::MatrixXf pdist = matching::iou_distance(a_ptrs, b_ptrs);

  std::set<int> dupa, dupb;
  for (int i = 0; i < pdist.rows(); ++i) {
    for (int j = 0; j < pdist.cols(); ++j) {
      if (pdist(i, j) < 0.15f) {
        int timep = stracksa[i]->frame_id - stracksa[i]->start_frame;
        int timeq = stracksb[j]->frame_id - stracksb[j]->start_frame;
        if (timep > timeq) {
          dupb.insert(j);
        } else {
          dupa.insert(i);
        }
      }
    }
  }

  std::vector<std::shared_ptr<STrack>> resa, resb;
  for (size_t i = 0; i < stracksa.size(); ++i) {
    if (dupa.find(i) == dupa.end()) {
      resa.push_back(stracksa[i]);
    }
  }
  for (size_t j = 0; j < stracksb.size(); ++j) {
    if (dupb.find(j) == dupb.end()) {
      resb.push_back(stracksb[j]);
    }
  }

  return {resa, resb};
}

std::vector<std::vector<float>> BYTETracker::update(
  const std::vector<std::vector<float>> & detections)
{
  frame_id_++;

  std::vector<std::shared_ptr<STrack>> activated_stracks;
  std::vector<std::shared_ptr<STrack>> refind_stracks;
  std::vector<std::shared_ptr<STrack>> lost_stracks;
  std::vector<std::shared_ptr<STrack>> removed_stracks;

  // Split detections by confidence threshold
  std::vector<std::vector<float>> detections_high, detections_low;
  for (const auto & det : detections) {
    if (det.size() < 6) {continue;}
    float score = det[4];
    if (score >= track_high_thresh_) {
      detections_high.push_back(det);
    } else if (score > track_low_thresh_ && score < track_high_thresh_) {
      detections_low.push_back(det);
    }
  }

  // Initialize tracks from detections
  auto detections_first = init_track(detections_high);
  auto detections_second = init_track(detections_low);

  // Separate tracked and unconfirmed tracks
  std::vector<std::shared_ptr<STrack>> unconfirmed;
  std::vector<std::shared_ptr<STrack>> tracked_stracks;
  for (auto & track : tracked_stracks_) {
    if (!track->is_activated) {
      unconfirmed.push_back(track);
    } else {
      tracked_stracks.push_back(track);
    }
  }

  // Step 2: First association with high score detections
  auto strack_pool = joint_stracks(tracked_stracks, lost_stracks_);

  // Predict current locations
  std::vector<STrack *> strack_pool_ptrs;
  for (auto & t : strack_pool) {
    strack_pool_ptrs.push_back(t.get());
  }
  STrack::multi_predict(strack_pool_ptrs, kalman_filter_);

  std::vector<STrack *> detections_first_ptrs;
  for (auto & t : detections_first) {
    detections_first_ptrs.push_back(t.get());
  }

  Eigen::MatrixXf dists = get_dists(strack_pool_ptrs, detections_first_ptrs);
  auto [matches, u_track, u_detection] = matching::linear_assignment(dists, match_thresh_);

  for (const auto & [itracked, idet] : matches) {
    auto & track = strack_pool[itracked];
    auto & det = detections_first[idet];
    if (track->state == TrackState::Tracked) {
      track->update(*det, frame_id_);
      activated_stracks.push_back(track);
    } else {
      track->re_activate(*det, frame_id_, false);
      refind_stracks.push_back(track);
    }
  }

  // Step 3: Second association with low score detections
  std::vector<std::shared_ptr<STrack>> r_tracked_stracks;
  for (int i : u_track) {
    if (strack_pool[i]->state == TrackState::Tracked) {
      r_tracked_stracks.push_back(strack_pool[i]);
    }
  }

  std::vector<STrack *> r_tracked_stracks_ptrs;
  for (auto & t : r_tracked_stracks) {
    r_tracked_stracks_ptrs.push_back(t.get());
  }

  std::vector<STrack *> detections_second_ptrs;
  for (auto & t : detections_second) {
    detections_second_ptrs.push_back(t.get());
  }

  Eigen::MatrixXf dists_second = matching::iou_distance(
    r_tracked_stracks_ptrs,
    detections_second_ptrs);
  auto [matches2, u_track2, u_detection2] = matching::linear_assignment(dists_second, 0.5f);

  for (const auto & [itracked, idet] : matches2) {
    auto & track = r_tracked_stracks[itracked];
    auto & det = detections_second[idet];
    if (track->state == TrackState::Tracked) {
      track->update(*det, frame_id_);
      activated_stracks.push_back(track);
    } else {
      track->re_activate(*det, frame_id_, false);
      refind_stracks.push_back(track);
    }
  }

  for (int it : u_track2) {
    auto & track = r_tracked_stracks[it];
    if (track->state != TrackState::Lost) {
      track->mark_lost();
      lost_stracks.push_back(track);
    }
  }

  // Deal with unconfirmed tracks
  std::vector<std::shared_ptr<STrack>> detections_left;
  for (int i : u_detection) {
    detections_left.push_back(detections_first[i]);
  }

  std::vector<STrack *> unconfirmed_ptrs;
  for (auto & t : unconfirmed) {
    unconfirmed_ptrs.push_back(t.get());
  }

  std::vector<STrack *> detections_left_ptrs;
  for (auto & t : detections_left) {
    detections_left_ptrs.push_back(t.get());
  }

  Eigen::MatrixXf dists_unconf = get_dists(unconfirmed_ptrs, detections_left_ptrs);
  auto [matches3, u_unconfirmed, u_detection3] = matching::linear_assignment(dists_unconf, 0.7f);

  for (const auto & [itracked, idet] : matches3) {
    unconfirmed[itracked]->update(*detections_left[idet], frame_id_);
    activated_stracks.push_back(unconfirmed[itracked]);
  }

  for (int it : u_unconfirmed) {
    auto & track = unconfirmed[it];
    track->mark_removed();
    removed_stracks.push_back(track);
  }

  // Step 4: Init new stracks
  for (int inew : u_detection3) {
    auto & track = detections_left[inew];
    if (track->score < new_track_thresh_) {
      continue;
    }
    track->activate(kalman_filter_, frame_id_);
    activated_stracks.push_back(track);
  }

  // Step 5: Update state
  for (auto & track : lost_stracks_) {
    if (frame_id_ - track->end_frame > max_time_lost_) {
      track->mark_removed();
      removed_stracks.push_back(track);
    }
  }

  tracked_stracks_ = sub_stracks(tracked_stracks_, removed_stracks);
  tracked_stracks_ = joint_stracks(tracked_stracks_, activated_stracks);
  tracked_stracks_ = joint_stracks(tracked_stracks_, refind_stracks);

  lost_stracks_ = sub_stracks(lost_stracks_, tracked_stracks_);
  lost_stracks_ = joint_stracks(lost_stracks_, lost_stracks);
  lost_stracks_ = sub_stracks(lost_stracks_, removed_stracks);

  auto [tracked_cleaned, lost_cleaned] = remove_duplicate_stracks(tracked_stracks_, lost_stracks_);
  tracked_stracks_ = tracked_cleaned;
  lost_stracks_ = lost_cleaned;

  removed_stracks_.insert(removed_stracks_.end(), removed_stracks.begin(), removed_stracks.end());
  if (removed_stracks_.size() > 1000) {
    removed_stracks_.erase(
      removed_stracks_.begin(),
      removed_stracks_.begin() + (removed_stracks_.size() - 999));
  }

  // Return tracked results
  std::vector<std::vector<float>> results;
  for (const auto & track : tracked_stracks_) {
    if (track->is_activated) {
      results.push_back(track->result());
    }
  }

  return results;
}

void BYTETracker::reset()
{
  tracked_stracks_.clear();
  lost_stracks_.clear();
  removed_stracks_.clear();
  frame_id_ = 0;
  BaseTrack::reset_id();
}

// ByteTrackerNode implementation
ByteTrackerNode::ByteTrackerNode(const rclcpp::NodeOptions & options)
: Node("byte_tracker_node", options)
{
  // Declare parameters
  track_high_thresh_ = this->declare_parameter<double>("track_high_thresh", 0.6);
  track_low_thresh_ = this->declare_parameter<double>("track_low_thresh", 0.1);
  new_track_thresh_ = this->declare_parameter<double>("new_track_thresh", 0.7);
  track_buffer_ = this->declare_parameter<int>("track_buffer", 30);
  match_thresh_ = this->declare_parameter<double>("match_thresh", 0.8);
  fuse_score_ = this->declare_parameter<bool>("fuse_score", false);
  frame_rate_ = this->declare_parameter<int>("frame_rate", 30);

  // Initialize tracker
  tracker_ = std::make_unique<BYTETracker>(
    track_high_thresh_,
    track_low_thresh_,
    new_track_thresh_,
    track_buffer_,
    match_thresh_,
    fuse_score_,
    frame_rate_);

  // Create subscriber and publisher
  detections_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
    "detections_input", 10,
    std::bind(&ByteTrackerNode::detections_callback, this, std::placeholders::_1));

  tracked_detections_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
    "tracked_detections", 10);

  RCLCPP_INFO(this->get_logger(), "ByteTrackerNode initialized");
}

void ByteTrackerNode::detections_callback(
  const vision_msgs::msg::Detection2DArray::SharedPtr msg)
{
  // Convert detections to tracker format
  std::vector<std::vector<float>> detections;
  for (const auto & det : msg->detections) {
    if (det.results.empty()) {
      continue;
    }

    float x_center = det.bbox.center.position.x;
    float y_center = det.bbox.center.position.y;
    float width = det.bbox.size_x;
    float height = det.bbox.size_y;

    float x1 = x_center - width / 2.0f;
    float y1 = y_center - height / 2.0f;
    float x2 = x_center + width / 2.0f;
    float y2 = y_center + height / 2.0f;

    float score = det.results[0].hypothesis.score;
    float class_id = std::stof(det.results[0].hypothesis.class_id);

    detections.push_back({x1, y1, x2, y2, score, class_id});
  }

  // Update tracker
  auto tracked_results = tracker_->update(detections);

  // Convert results back to Detection2DArray
  auto tracked_msg = std::make_unique<vision_msgs::msg::Detection2DArray>();
  tracked_msg->header = msg->header;

  for (const auto & result : tracked_results) {
    // Result format: [x1, y1, x2, y2, track_id, score, class_id, idx]
    if (result.size() < 8) {
      continue;
    }

    vision_msgs::msg::Detection2D detection;
    detection.header = msg->header;

    float x1 = result[0];
    float y1 = result[1];
    float x2 = result[2];
    float y2 = result[3];
    int track_id = static_cast<int>(result[4]);
    float score = result[5];
    int class_id = static_cast<int>(result[6]);

    detection.bbox.center.position.x = (x1 + x2) / 2.0;
    detection.bbox.center.position.y = (y1 + y2) / 2.0;
    detection.bbox.size_x = x2 - x1;
    detection.bbox.size_y = y2 - y1;

    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
    hypothesis.hypothesis.class_id = std::to_string(class_id);
    hypothesis.hypothesis.score = score;

    detection.results.push_back(hypothesis);
    detection.id = std::to_string(track_id);

    tracked_msg->detections.push_back(detection);
  }

  tracked_detections_pub_->publish(std::move(tracked_msg));
}

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolov8::ByteTrackerNode)
