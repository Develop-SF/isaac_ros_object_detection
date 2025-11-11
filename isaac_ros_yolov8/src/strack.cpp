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

// Initialize static member
int BaseTrack::_count = 0;

// BaseTrack implementation
BaseTrack::BaseTrack()
: track_id(0),
  is_activated(false),
  state(TrackState::New),
  frame_id(0),
  start_frame(0),
  end_frame(0)
{
}

int BaseTrack::next_id()
{
  _count++;
  return _count;
}

void BaseTrack::reset_id()
{
  _count = 0;
}

void BaseTrack::mark_lost()
{
  state = TrackState::Lost;
}

void BaseTrack::mark_removed()
{
  state = TrackState::Removed;
}

// Helper function to convert xywh to ltwh (left-top-width-height)
static Eigen::Vector4f xywh2ltwh(const std::vector<float> & xywh)
{
  Eigen::Vector4f ltwh;
  ltwh(0) = xywh[0] - xywh[2] / 2.0f;  // left = center_x - width/2
  ltwh(1) = xywh[1] - xywh[3] / 2.0f;  // top = center_y - height/2
  ltwh(2) = xywh[2];                    // width
  ltwh(3) = xywh[3];                    // height
  return ltwh;
}

// STrack implementation
STrack::STrack(const std::vector<float> & xywh, float score, int cls, int idx)
: BaseTrack(),
  score(score),
  cls(cls),
  idx(idx),
  tracklet_len(0),
  angle(nullptr),
  kalman_filter_(nullptr)
{
  // xywh format: [center_x, center_y, width, height] or [center_x, center_y, width, height, angle]
  _tlwh = xywh2ltwh(xywh);

  if (xywh.size() == 5) {
    angle = std::make_unique<float>(xywh[4]);
  }

  mean_ = KalmanFilterXYAH::StateVector::Zero();
  covariance_ = KalmanFilterXYAH::StateMatrix::Zero();
}

void STrack::predict()
{
  KalmanFilterXYAH::StateVector mean_state = mean_;
  if (state != TrackState::Tracked) {
    mean_state(7) = 0.0f;
  }
  auto [new_mean, new_cov] = kalman_filter_->predict(mean_state, covariance_);
  mean_ = new_mean;
  covariance_ = new_cov;
}

void STrack::multi_predict(
  std::vector<STrack *> & stracks,
  KalmanFilterXYAH & kalman_filter)
{
  if (stracks.empty()) {
    return;
  }

  std::vector<KalmanFilterXYAH::StateVector> multi_mean;
  std::vector<KalmanFilterXYAH::StateMatrix> multi_covariance;

  for (auto * st : stracks) {
    KalmanFilterXYAH::StateVector mean_state = st->mean_;
    if (st->state != TrackState::Tracked) {
      mean_state(7) = 0.0f;
    }
    multi_mean.push_back(mean_state);
    multi_covariance.push_back(st->covariance_);
  }

  auto [new_means, new_covs] = kalman_filter.multi_predict(multi_mean, multi_covariance);

  for (size_t i = 0; i < stracks.size(); ++i) {
    stracks[i]->mean_ = new_means[i];
    stracks[i]->covariance_ = new_covs[i];
  }
}

void STrack::multi_gmc(std::vector<STrack *> & stracks, const Eigen::Matrix<float, 2, 3> & H)
{
  if (stracks.empty()) {
    return;
  }

  Eigen::Matrix2f R = H.block<2, 2>(0, 0);
  Eigen::Matrix<float, 8, 8> R8x8 = Eigen::Matrix<float, 8, 8>::Zero();
  for (int i = 0; i < 4; ++i) {
    R8x8.block<2, 2>(i * 2, i * 2) = R;
  }

  Eigen::Vector2f t = H.col(2);

  for (auto * st : stracks) {
    st->mean_ = R8x8 * st->mean_;
    st->mean_.head<2>() += t;
    st->covariance_ = R8x8 * st->covariance_ * R8x8.transpose();
  }
}

void STrack::activate(KalmanFilterXYAH & kalman_filter, int frame_id)
{
  kalman_filter_ = &kalman_filter;
  track_id = next_id();

  auto [mean, cov] = kalman_filter.initiate(convert_coords(_tlwh));
  mean_ = mean;
  covariance_ = cov;

  tracklet_len = 0;
  state = TrackState::Tracked;
  if (frame_id == 1) {
    is_activated = true;
  }
  this->frame_id = frame_id;
  start_frame = frame_id;
}

void STrack::re_activate(STrack & new_track, int frame_id, bool new_id)
{
  auto [mean, cov] = kalman_filter_->update(mean_, covariance_, convert_coords(new_track.tlwh()));
  mean_ = mean;
  covariance_ = cov;

  tracklet_len = 0;
  state = TrackState::Tracked;
  is_activated = true;
  this->frame_id = frame_id;

  if (new_id) {
    track_id = next_id();
  }

  score = new_track.score;
  cls = new_track.cls;
  idx = new_track.idx;
  if (new_track.angle) {
    angle = std::make_unique<float>(*new_track.angle);
  } else {
    angle.reset();
  }
}

void STrack::update(STrack & new_track, int frame_id)
{
  this->frame_id = frame_id;
  tracklet_len++;

  Eigen::Vector4f new_tlwh = new_track.tlwh();
  auto [mean, cov] = kalman_filter_->update(mean_, covariance_, convert_coords(new_tlwh));
  mean_ = mean;
  covariance_ = cov;

  state = TrackState::Tracked;
  is_activated = true;

  score = new_track.score;
  cls = new_track.cls;
  idx = new_track.idx;
  if (new_track.angle) {
    angle = std::make_unique<float>(*new_track.angle);
  } else {
    angle.reset();
  }
}

Eigen::Vector4f STrack::tlwh() const
{
  if (mean_(0) == 0.0f && mean_(1) == 0.0f && mean_(2) == 0.0f && mean_(3) == 0.0f) {
    return _tlwh;
  }

  Eigen::Vector4f ret = mean_.head<4>();
  ret(2) *= ret(3);  // width = aspect_ratio * height
  ret.head<2>() -= ret.tail<2>() / 2.0f;  // top-left = center - size/2
  return ret;
}

Eigen::Vector4f STrack::xyxy() const
{
  Eigen::Vector4f ret = tlwh();
  ret.tail<2>() += ret.head<2>();  // bottom-right = top-left + size
  return ret;
}

Eigen::Vector4f STrack::xywh() const
{
  Eigen::Vector4f ret = tlwh();
  ret.head<2>() += ret.tail<2>() / 2.0f;  // center = top-left + size/2
  return ret;
}

std::vector<float> STrack::result() const
{
  Eigen::Vector4f coords = angle ? xywh() : xyxy();
  
  std::vector<float> res;
  res.push_back(coords(0));
  res.push_back(coords(1));
  res.push_back(coords(2));
  res.push_back(coords(3));
  
  if (angle) {
    res.push_back(*angle);
  }
  
  res.push_back(static_cast<float>(track_id));
  res.push_back(score);
  res.push_back(static_cast<float>(cls));
  res.push_back(static_cast<float>(idx));
  
  return res;
}

Eigen::Vector4f STrack::tlwh_to_xyah(const Eigen::Vector4f & tlwh)
{
  Eigen::Vector4f ret = tlwh;
  ret.head<2>() += ret.tail<2>() / 2.0f;  // center = top-left + size/2
  ret(2) /= ret(3);  // aspect_ratio = width / height
  return ret;
}

Eigen::Vector4f STrack::convert_coords(const Eigen::Vector4f & tlwh) const
{
  return tlwh_to_xyah(tlwh);
}

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

