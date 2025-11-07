/*
 * Filename: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src/detection2_d_array_vlm_filter.cpp
 * Path: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src
 * Created Date: Monday, November 3rd 2025, 11:03:24 am
 * Author: Wen-Yu Chien
 * Description: STrack class for ByteTrack object tracking
 * Copyright (c) 2025 Copyright (c) 2025 Shinfang Global
 */

#include "isaac_ros_yolov8/strack.hpp"

#include <algorithm>

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{

int STrack::_count = 0;

STrack::STrack(const vision_msgs::msg::Detection2D & detection)
: track_id_(-1),
  frame_id_(-1),
  start_frame_(-1),
  end_frame_(-1),
  tracklet_len_(0),
  state_(TrackState::New),
  is_activated_(false),
  score_(0.0),
  frames_without_update_(0)
{
  // Extract bounding box in tlwh format
  double center_x = detection.bbox.center.position.x;
  double center_y = detection.bbox.center.position.y;
  double width = detection.bbox.size_x;
  double height = detection.bbox.size_y;

  _tlwh[0] = center_y - height / 2.0;  // top
  _tlwh[1] = center_x - width / 2.0;   // left
  _tlwh[2] = width;
  _tlwh[3] = height;

  // Extract score and class
  if (!detection.results.empty()) {
    score_ = detection.results[0].hypothesis.score;
    class_id_ = detection.results[0].hypothesis.class_id;
  }

  mean_ = Eigen::VectorXd::Zero(8);
  covariance_ = Eigen::MatrixXd::Zero(8, 8);
}

void STrack::predict()
{
  frames_without_update_++;  // Increment counter for frames without updates
  
  Eigen::VectorXd mean_state = mean_;
  if (state_ != TrackState::Tracked) {
    mean_state(7) = 0.0;  // Set velocity to zero for non-tracked states
  }
  
  // Store original state for drift limiting
  Eigen::Vector4d original_xyah = mean_.head<4>();
  std::array<double, 4> original_tlwh = _tlwh;
  
  auto [pred_mean, pred_cov] = kalman_filter_->predict(mean_state, covariance_);
  mean_ = pred_mean;
  covariance_ = pred_cov;

  // Update tlwh from predicted state
  Eigen::Vector4d xyah = mean_.head<4>();
  std::array<double, 4> predicted_tlwh = xyah_to_tlwh(xyah);
  
  // Limit drift: prevent aspect ratio and size from changing too dramatically
  // This is especially important for tracks that haven't been updated recently
  if (state_ != TrackState::Tracked || frames_without_update_ > 3) {
    // For lost tracks or new tracks, limit the prediction drift
    double aspect_ratio_change = std::abs(xyah(2) - original_xyah(2)) / original_xyah(2);
    double height_change = std::abs(xyah(3) - original_xyah(3)) / original_xyah(3);
    
    // If changes are too dramatic (>50% for aspect ratio, >100% for height), limit them
    if (aspect_ratio_change > 0.5) {
      double sign = (xyah(2) > original_xyah(2)) ? 1.0 : -1.0;
      xyah(2) = original_xyah(2) * (1.0 + sign * 0.5);
      mean_(2) = xyah(2);
    }
    
    if (height_change > 1.0) {
      double sign = (xyah(3) > original_xyah(3)) ? 1.0 : -1.0;
      xyah(3) = original_xyah(3) * (1.0 + sign * 1.0);
      mean_(3) = xyah(3);
    }
    
    // Recalculate tlwh with limited drift
    _tlwh = xyah_to_tlwh(xyah);
  } else {
    _tlwh = predicted_tlwh;
  }
}

void STrack::activate(std::shared_ptr<KalmanFilterXYAH> kalman_filter, int frame_id)
{
  kalman_filter_ = kalman_filter;
  track_id_ = next_id();

  Eigen::Vector4d xyah = tlwh_to_xyah(_tlwh);
  auto [init_mean, init_cov] = kalman_filter_->initiate(xyah);
  mean_ = init_mean;
  covariance_ = init_cov;

  tracklet_len_ = 0;
  state_ = TrackState::Tracked;
  is_activated_ = true;
  frame_id_ = frame_id;
  start_frame_ = frame_id;
  frames_without_update_ = 0;  // Reset counter since we have a new measurement
}

void STrack::re_activate(const STrack & new_track, int frame_id, bool new_id)
{
  Eigen::Vector4d xyah = tlwh_to_xyah(new_track._tlwh);
  auto [upd_mean, upd_cov] = kalman_filter_->update(mean_, covariance_, xyah);
  mean_ = upd_mean;
  covariance_ = upd_cov;

  // Update tlwh from updated state
  _tlwh = xyah_to_tlwh(mean_.head<4>());

  tracklet_len_ = 0;
  state_ = TrackState::Tracked;
  is_activated_ = true;
  frame_id_ = frame_id;

  if (new_id) {
    track_id_ = next_id();
  }

  score_ = new_track.score_;
  class_id_ = new_track.class_id_;
}

void STrack::update(const STrack & new_track, int frame_id)
{
  frame_id_ = frame_id;
  tracklet_len_++;
  frames_without_update_ = 0;  // Reset counter since we have a new measurement

  Eigen::Vector4d xyah = tlwh_to_xyah(new_track._tlwh);
  auto [upd_mean, upd_cov] = kalman_filter_->update(mean_, covariance_, xyah);
  mean_ = upd_mean;
  covariance_ = upd_cov;

  // Update tlwh from updated state
  _tlwh = xyah_to_tlwh(mean_.head<4>());

  state_ = TrackState::Tracked;
  is_activated_ = true;

  score_ = new_track.score_;
  class_id_ = new_track.class_id_;
}

void STrack::mark_lost()
{
  state_ = TrackState::Lost;
  end_frame_ = frame_id_;
}

void STrack::mark_removed()
{
  state_ = TrackState::Removed;
}

Eigen::Vector4d STrack::tlwh_to_xyah(const std::array<double, 4> & tlwh)
{
  // tlwh: [top, left, width, height]
  // xyah: [center_x, center_y, aspect_ratio, height]
  Eigen::Vector4d xyah;
  xyah(0) = tlwh[1] + tlwh[2] / 2.0;  // center_x
  xyah(1) = tlwh[0] + tlwh[3] / 2.0;  // center_y
  xyah(2) = tlwh[2] / tlwh[3];        // aspect_ratio
  xyah(3) = tlwh[3];                  // height
  return xyah;
}

std::array<double, 4> STrack::xyah_to_tlwh(const Eigen::Vector4d & xyah)
{
  // xyah: [center_x, center_y, aspect_ratio, height]
  // tlwh: [top, left, width, height]
  std::array<double, 4> tlwh;
  double width = xyah(2) * xyah(3);
  tlwh[0] = xyah(1) - xyah(3) / 2.0;  // top
  tlwh[1] = xyah(0) - width / 2.0;    // left
  tlwh[2] = width;
  tlwh[3] = xyah(3);
  return tlwh;
}

void STrack::multi_predict(
  std::vector<STrack *> & stracks,
  std::shared_ptr<KalmanFilterXYAH> kalman_filter)
{
  if (stracks.empty()) {
    return;
  }

  std::vector<Eigen::VectorXd> multi_mean;
  std::vector<Eigen::MatrixXd> multi_covariance;
  multi_mean.reserve(stracks.size());
  multi_covariance.reserve(stracks.size());

  for (auto * st : stracks) {
    Eigen::VectorXd mean_state = st->mean_;
    if (st->state_ != TrackState::Tracked) {
      mean_state(7) = 0.0;
    }
    multi_mean.push_back(mean_state);
    multi_covariance.push_back(st->covariance_);
  }

  auto [pred_means, pred_covs] = kalman_filter->multi_predict(multi_mean, multi_covariance);

  for (size_t i = 0; i < stracks.size(); ++i) {
    // Increment frames without update counter
    stracks[i]->frames_without_update_++;
    
    // Store original state for drift limiting
    Eigen::Vector4d original_xyah = stracks[i]->mean_.head<4>();
    
    stracks[i]->mean_ = pred_means[i];
    stracks[i]->covariance_ = pred_covs[i];
    
    Eigen::Vector4d xyah = pred_means[i].head<4>();
    
    // Limit drift for non-tracked or tracks without recent updates
    if (stracks[i]->state_ != TrackState::Tracked || stracks[i]->frames_without_update_ > 3) {
      double aspect_ratio_change = std::abs(xyah(2) - original_xyah(2)) / original_xyah(2);
      double height_change = std::abs(xyah(3) - original_xyah(3)) / original_xyah(3);
      
      if (aspect_ratio_change > 0.5) {
        double sign = (xyah(2) > original_xyah(2)) ? 1.0 : -1.0;
        xyah(2) = original_xyah(2) * (1.0 + sign * 0.5);
        stracks[i]->mean_(2) = xyah(2);
      }
      
      if (height_change > 1.0) {
        double sign = (xyah(3) > original_xyah(3)) ? 1.0 : -1.0;
        xyah(3) = original_xyah(3) * (1.0 + sign * 1.0);
        stracks[i]->mean_(3) = xyah(3);
      }
    }
    
    stracks[i]->_tlwh = xyah_to_tlwh(xyah);
  }
}

void STrack::multi_gmc(std::vector<STrack *> & stracks, const Eigen::Matrix<double, 2, 3> & H)
{
  if (stracks.empty()) {
    return;
  }

  Eigen::Matrix2d R = H.block<2, 2>(0, 0);
  Eigen::Vector2d t = H.col(2);

  // Create 8x8 rotation matrix for state space
  Eigen::MatrixXd R8x8 = Eigen::MatrixXd::Zero(8, 8);
  for (int i = 0; i < 4; ++i) {
    R8x8.block<2, 2>(i * 2, i * 2) = R;
  }

  for (auto * st : stracks) {
    st->mean_ = R8x8 * st->mean_;
    st->mean_.head<2>() += t;
    st->covariance_ = R8x8 * st->covariance_ * R8x8.transpose();
    st->_tlwh = xyah_to_tlwh(st->mean_.head<4>());
  }
}

std::array<double, 4> STrack::tlwh() const
{
  return _tlwh;
}

std::array<double, 4> STrack::xyxy() const
{
  std::array<double, 4> xyxy;
  xyxy[0] = _tlwh[1];                    // x1
  xyxy[1] = _tlwh[0];                    // y1
  xyxy[2] = _tlwh[1] + _tlwh[2];        // x2
  xyxy[3] = _tlwh[0] + _tlwh[3];        // y2
  return xyxy;
}

vision_msgs::msg::Detection2D STrack::to_detection() const
{
  vision_msgs::msg::Detection2D detection;

  // Set bounding box
  detection.bbox.center.position.x = _tlwh[1] + _tlwh[2] / 2.0;
  detection.bbox.center.position.y = _tlwh[0] + _tlwh[3] / 2.0;
  detection.bbox.size_x = _tlwh[2];
  detection.bbox.size_y = _tlwh[3];

  // Set tracking ID
  detection.id = std::to_string(track_id_);

  // Set class and score
  vision_msgs::msg::ObjectHypothesisWithPose hyp;
  hyp.hypothesis.class_id = class_id_;
  hyp.hypothesis.score = score_;
  detection.results.push_back(hyp);

  return detection;
}

void STrack::reset_id()
{
  _count = 0;
}

int STrack::next_id()
{
  return ++_count;
}

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

