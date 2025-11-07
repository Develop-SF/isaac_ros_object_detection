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

#include <Eigen/Dense>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "isaac_ros_yolov8/kalman_filter.hpp"
#include "vision_msgs/msg/detection2_d.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{

enum class TrackState
{
  New = 0,
  Tracked = 1,
  Lost = 2,
  Removed = 3
};

/**
 * @brief Single object tracking representation using Kalman filtering
 */
class STrack
{
public:
  /**
   * @brief Construct a new STrack from a detection
   * @param detection Detection2D message
   */
  explicit STrack(const vision_msgs::msg::Detection2D & detection);

  /**
   * @brief Predict the next state using Kalman filter
   */
  void predict();

  /**
   * @brief Activate a new tracklet
   * @param kalman_filter Shared Kalman filter
   * @param frame_id Current frame ID
   */
  void activate(std::shared_ptr<KalmanFilterXYAH> kalman_filter, int frame_id);

  /**
   * @brief Re-activate a previously lost track
   * @param new_track New detection to re-activate with
   * @param frame_id Current frame ID
   * @param new_id Whether to assign a new track ID
   */
  void re_activate(const STrack & new_track, int frame_id, bool new_id = false);

  /**
   * @brief Update the state with a new detection
   * @param new_track New detection
   * @param frame_id Current frame ID
   */
  void update(const STrack & new_track, int frame_id);

  /**
   * @brief Mark track as lost
   */
  void mark_lost();

  /**
   * @brief Mark track as removed
   */
  void mark_removed();

  /**
   * @brief Convert tlwh to xyah format for Kalman filter
   * @param tlwh Bounding box in top-left-width-height format
   * @return xyah format [center_x, center_y, aspect_ratio, height]
   */
  static Eigen::Vector4d tlwh_to_xyah(const std::array<double, 4> & tlwh);

  /**
   * @brief Convert xyah back to tlwh format
   * @param xyah Bounding box in xyah format
   * @return tlwh format [top, left, width, height]
   */
  static std::array<double, 4> xyah_to_tlwh(const Eigen::Vector4d & xyah);

  /**
   * @brief Static method to perform multi-track prediction
   * @param stracks Vector of tracks to predict
   * @param kalman_filter Shared Kalman filter
   */
  static void multi_predict(
    std::vector<STrack *> & stracks,
    std::shared_ptr<KalmanFilterXYAH> kalman_filter);

  /**
   * @brief Apply camera motion compensation using homography
   * @param stracks Vector of tracks
   * @param H Homography matrix (2x3)
   */
  static void multi_gmc(std::vector<STrack *> & stracks, const Eigen::Matrix<double, 2, 3> & H);

  /**
   * @brief Get the bounding box in tlwh format
   * @return Bounding box [top, left, width, height]
   */
  std::array<double, 4> tlwh() const;

  /**
   * @brief Get the bounding box in xyxy format
   * @return Bounding box [x1, y1, x2, y2]
   */
  std::array<double, 4> xyxy() const;

  /**
   * @brief Convert current state to Detection2D message
   * @return Detection2D message
   */
  vision_msgs::msg::Detection2D to_detection() const;

  // Getters
  int track_id() const {return track_id_;}
  int frame_id() const {return frame_id_;}
  int start_frame() const {return start_frame_;}
  int end_frame() const {return end_frame_;}
  TrackState state() const {return state_;}
  bool is_activated() const {return is_activated_;}
  double score() const {return score_;}
  std::string class_id() const {return class_id_;}
  int tracklet_len() const {return tracklet_len_;}
  int frames_without_update() const {return frames_without_update_;}

  // Setters
  void set_track_id(int id) {track_id_ = id;}

  // Global track ID counter
  static void reset_id();
  static int next_id();

private:
  std::array<double, 4> _tlwh;  // Private storage for tlwh
  std::shared_ptr<KalmanFilterXYAH> kalman_filter_;
  Eigen::VectorXd mean_;
  Eigen::MatrixXd covariance_;

  int track_id_;
  int frame_id_;
  int start_frame_;
  int end_frame_;
  int tracklet_len_;

  TrackState state_;
  bool is_activated_;

  double score_;
  std::string class_id_;
  int frames_without_update_;  // Counter for frames without measurement updates

  static int _count;
};

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_YOLOV8__STRACK_HPP_

