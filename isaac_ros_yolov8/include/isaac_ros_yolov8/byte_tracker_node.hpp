/*
 * Filename: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src/detection2_d_array_vlm_filter.cpp
 * Path: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src
 * Created Date: Monday, November 3rd 2025, 11:03:24 am
 * Author: Wen-Yu Chien
 * Description: ByteTrack Node for multi-object tracking
 * Copyright (c) 2025 Copyright (c) 2025 Shinfang Global
 */

#ifndef ISAAC_ROS_YOLOV8__BYTE_TRACKER_NODE_HPP_
#define ISAAC_ROS_YOLOV8__BYTE_TRACKER_NODE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

#include "isaac_ros_yolov8/strack.hpp"
#include "isaac_ros_yolov8/kalman_filter.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{

/**
 * @brief BYTETracker algorithm implementation
 * 
 * Implements the BYTETracker algorithm for multi-object tracking.
 * Maintains tracked, lost, and removed tracks over frames.
 */
class BYTETracker
{
public:
  /**
   * @brief Construct a new BYTETracker object
   * 
   * @param track_high_thresh High threshold for track detection confidence
   * @param track_low_thresh Low threshold for track detection confidence
   * @param new_track_thresh Threshold for initializing new tracks
   * @param track_buffer Number of frames to keep lost tracks
   * @param match_thresh Matching threshold
   * @param fuse_score Whether to fuse scores in matching
   * @param frame_rate Frame rate of the video
   */
  BYTETracker(
    float track_high_thresh = 0.6f,
    float track_low_thresh = 0.1f,
    float new_track_thresh = 0.7f,
    int track_buffer = 30,
    float match_thresh = 0.8f,
    bool fuse_score = false,
    int frame_rate = 30);

  ~BYTETracker() = default;

  /**
   * @brief Update tracker with new detections
   * 
   * @param detections Detection results [x1, y1, x2, y2, score, class_id]
   * @return std::vector<std::vector<float>> Tracked results
   */
  std::vector<std::vector<float>> update(const std::vector<std::vector<float>> & detections);

  /**
   * @brief Reset the tracker
   */
  void reset();

private:
  /**
   * @brief Initialize tracks from detections
   * 
   * @param detections Detection results
   * @return std::vector<std::shared_ptr<STrack>> List of initialized tracks
   */
  std::vector<std::shared_ptr<STrack>> init_track(
    const std::vector<std::vector<float>> & detections);

  /**
   * @brief Get distance matrix between tracks and detections
   * 
   * @param tracks List of tracks
   * @param detections List of detections
   * @return Eigen::MatrixXf Distance matrix
   */
  Eigen::MatrixXf get_dists(
    std::vector<STrack *> & tracks,
    std::vector<STrack *> & detections);

  /**
   * @brief Joint two lists of tracks
   * 
   * @param tlista First list
   * @param tlistb Second list
   * @return std::vector<std::shared_ptr<STrack>> Combined list without duplicates
   */
  static std::vector<std::shared_ptr<STrack>> joint_stracks(
    const std::vector<std::shared_ptr<STrack>> & tlista,
    const std::vector<std::shared_ptr<STrack>> & tlistb);

  /**
   * @brief Subtract second list from first list
   * 
   * @param tlista First list
   * @param tlistb Second list
   * @return std::vector<std::shared_ptr<STrack>> Filtered list
   */
  static std::vector<std::shared_ptr<STrack>> sub_stracks(
    const std::vector<std::shared_ptr<STrack>> & tlista,
    const std::vector<std::shared_ptr<STrack>> & tlistb);

  /**
   * @brief Remove duplicate tracks
   * 
   * @param stracksa First list
   * @param stracksb Second list
   * @return std::pair<std::vector<std::shared_ptr<STrack>>, std::vector<std::shared_ptr<STrack>>>
   *         Deduplicated lists
   */
  static std::pair<std::vector<std::shared_ptr<STrack>>, std::vector<std::shared_ptr<STrack>>>
  remove_duplicate_stracks(
    const std::vector<std::shared_ptr<STrack>> & stracksa,
    const std::vector<std::shared_ptr<STrack>> & stracksb);

  std::vector<std::shared_ptr<STrack>> tracked_stracks_;   ///< Successfully activated tracks
  std::vector<std::shared_ptr<STrack>> lost_stracks_;      ///< Lost tracks
  std::vector<std::shared_ptr<STrack>> removed_stracks_;   ///< Removed tracks

  int frame_id_;                          ///< Current frame ID
  int max_time_lost_;                     ///< Maximum frames for lost tracks
  KalmanFilterXYAH kalman_filter_;        ///< Kalman filter instance

  // Parameters
  float track_high_thresh_;
  float track_low_thresh_;
  float new_track_thresh_;
  float match_thresh_;
  bool fuse_score_;
};

/**
 * @brief ROS2 node for ByteTracker
 * 
 * Subscribes to Detection2DArray messages and publishes tracked detections
 * with tracking IDs.
 */
class ByteTrackerNode : public rclcpp::Node
{
public:
  explicit ByteTrackerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  ~ByteTrackerNode() = default;

private:
  /**
   * @brief Callback for detection messages
   * 
   * @param msg Detection2DArray message
   */
  void detections_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg);

  // Subscriber and publisher
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detections_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr tracked_detections_pub_;

  // ByteTracker instance
  std::unique_ptr<BYTETracker> tracker_;

  // Parameters
  float track_high_thresh_;
  float track_low_thresh_;
  float new_track_thresh_;
  int track_buffer_;
  float match_thresh_;
  bool fuse_score_;
  int frame_rate_;
};

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_YOLOV8__BYTE_TRACKER_NODE_HPP_

