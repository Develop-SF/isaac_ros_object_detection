/*
 * Filename: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src/detection2_d_array_vlm_filter.cpp
 * Path: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src
 * Created Date: Monday, November 3rd 2025, 11:03:24 am
 * Author: Wen-Yu Chien
 * Description: STrack class for ByteTrack object tracking
 * Copyright (c) 2025 Copyright (c) 2025 Shinfang Global
 */

#ifndef ISAAC_ROS_YOLOV8__STRACK_HPP_
#define ISAAC_ROS_YOLOV8__STRACK_HPP_

#include "isaac_ros_yolov8/kalman_filter.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{

/**
 * @brief Track state enumeration
 */
enum class TrackState
{
  New = 0,
  Tracked = 1,
  Lost = 2,
  Removed = 3
};

/**
 * @brief Base class for object tracking
 */
class BaseTrack
{
public:
  BaseTrack();
  virtual ~BaseTrack() = default;

  /**
   * @brief Get the next unique track ID
   * @return int Next track ID
   */
  static int next_id();

  /**
   * @brief Reset the track ID counter
   */
  static void reset_id();

  /**
   * @brief Mark the track as lost
   */
  void mark_lost();

  /**
   * @brief Mark the track as removed
   */
  void mark_removed();

  int track_id;         ///< Unique track identifier
  bool is_activated;    ///< Whether the track has been activated
  TrackState state;     ///< Current state of the track
  int frame_id;         ///< Current frame ID
  int start_frame;      ///< Frame where the track started
  int end_frame;        ///< Frame where the track ended

private:
  static int _count;    ///< Global track counter
};

/**
 * @brief Single object tracking representation using Kalman filtering
 * 
 * This class stores all information regarding individual tracklets and performs
 * state updates and predictions based on Kalman filter.
 */
class STrack : public BaseTrack
{
public:
  /**
   * @brief Construct a new STrack object
   * 
   * @param xywh Bounding box in format [x, y, w, h] or [x, y, w, h, angle]
   * @param score Confidence score
   * @param cls Class label
   * @param idx Detection index
   */
  STrack(const std::vector<float> & xywh, float score, int cls, int idx);

  /**
   * @brief Predict the next state using Kalman filter
   */
  void predict();

  /**
   * @brief Perform multi-object predictive tracking using Kalman filter
   * 
   * @param stracks List of tracks to predict
   * @param kalman_filter Shared Kalman filter instance
   */
  static void multi_predict(
    std::vector<STrack *> & stracks,
    KalmanFilterXYAH & kalman_filter);

  /**
   * @brief Update track states using homography matrix (for camera motion compensation)
   * 
   * @param stracks List of tracks to update
   * @param H Homography matrix (2x3)
   */
  static void multi_gmc(std::vector<STrack *> & stracks, const Eigen::Matrix<float, 2, 3> & H);

  /**
   * @brief Activate a new tracklet
   * 
   * @param kalman_filter Kalman filter instance
   * @param frame_id Current frame ID
   */
  void activate(KalmanFilterXYAH & kalman_filter, int frame_id);

  /**
   * @brief Reactivate a previously lost tracklet
   * 
   * @param new_track New track with updated information
   * @param frame_id Current frame ID
   * @param new_id Whether to assign a new track ID
   */
  void re_activate(STrack & new_track, int frame_id, bool new_id = false);

  /**
   * @brief Update the state of a matched track
   * 
   * @param new_track New track with updated information
   * @param frame_id Current frame ID
   */
  void update(STrack & new_track, int frame_id);

  /**
   * @brief Get bounding box in tlwh format (top-left-width-height)
   * 
   * @return Eigen::Vector4f Bounding box [x, y, w, h]
   */
  Eigen::Vector4f tlwh() const;

  /**
   * @brief Get bounding box in xyxy format (top-left-bottom-right)
   * 
   * @return Eigen::Vector4f Bounding box [x1, y1, x2, y2]
   */
  Eigen::Vector4f xyxy() const;

  /**
   * @brief Get bounding box in xywh format (center-width-height)
   * 
   * @return Eigen::Vector4f Bounding box [cx, cy, w, h]
   */
  Eigen::Vector4f xywh() const;

  /**
   * @brief Get the tracking result
   * 
   * @return std::vector<float> Result in format [x1, y1, x2, y2, track_id, score, cls, idx]
   */
  std::vector<float> result() const;

  float score;              ///< Confidence score
  int cls;                  ///< Class label
  int idx;                  ///< Detection index
  int tracklet_len;         ///< Length of the tracklet
  std::unique_ptr<float> angle;  ///< Optional angle for oriented bounding boxes

private:
  /**
   * @brief Convert tlwh to xyah format for Kalman filter
   * 
   * @param tlwh Bounding box in tlwh format
   * @return Eigen::Vector4f Bounding box in xyah format
   */
  static Eigen::Vector4f tlwh_to_xyah(const Eigen::Vector4f & tlwh);

  /**
   * @brief Convert coordinates for Kalman filter
   * 
   * @param tlwh Bounding box in tlwh format
   * @return Eigen::Vector4f Bounding box in xyah format
   */
  Eigen::Vector4f convert_coords(const Eigen::Vector4f & tlwh) const;

  Eigen::Vector4f _tlwh;                       ///< Internal tlwh representation
  KalmanFilterXYAH * kalman_filter_;           ///< Pointer to Kalman filter
  KalmanFilterXYAH::StateVector mean_;         ///< Mean state estimate
  KalmanFilterXYAH::StateMatrix covariance_;   ///< Covariance matrix
};

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_YOLOV8__STRACK_HPP_

