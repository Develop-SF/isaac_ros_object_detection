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
#include "sensor_msgs/msg/image.hpp"

#include "isaac_ros_yolov8/kalman_filter.hpp"
#include "isaac_ros_yolov8/strack.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{

/**
 * @brief ByteTrack node for multi-object tracking
 * 
 * This node implements the ByteTrack algorithm for real-time multi-object tracking.
 * It subscribes to Detection2DArray messages and publishes tracked detections with
 * consistent IDs across frames.
 */
class ByteTrackerNode : public rclcpp::Node
{
public:
  explicit ByteTrackerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~ByteTrackerNode();

private:
  /**
   * @brief Callback for detection messages
   * @param msg Detection2DArray message
   */
  void detectionCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg);

  /**
   * @brief Main tracking update step
   * @param detections_msg Input detections
   * @return Tracked detections
   */
  vision_msgs::msg::Detection2DArray update(
    const vision_msgs::msg::Detection2DArray & detections_msg);

  /**
   * @brief Initialize tracks from detections
   * @param detections Input detections
   * @return Vector of STrack objects
   */
  std::vector<STrack> init_track(const vision_msgs::msg::Detection2DArray & detections);

  /**
   * @brief Get distance matrix between tracks and detections
   * @param tracks Vector of track pointers
   * @param detections Vector of detection pointers
   * @return Distance matrix
   */
  Eigen::MatrixXd get_dists(
    const std::vector<STrack *> & tracks,
    const std::vector<STrack *> & detections);

  /**
   * @brief Combine two track lists without duplicates
   * @param tlista First track list
   * @param tlistb Second track list
   * @return Combined track list
   */
  static std::vector<STrack> joint_stracks(
    const std::vector<STrack> & tlista,
    const std::vector<STrack> & tlistb);

  /**
   * @brief Subtract tracks in tlistb from tlista
   * @param tlista First track list
   * @param tlistb Tracks to remove
   * @return Filtered track list
   */
  static std::vector<STrack> sub_stracks(
    const std::vector<STrack> & tlista,
    const std::vector<STrack> & tlistb);

  /**
   * @brief Remove duplicate tracks based on IoU
   * @param stracksa First track list
   * @param stracksb Second track list
   * @return Pair of filtered track lists
   */
  static std::pair<std::vector<STrack>, std::vector<STrack>> remove_duplicate_stracks(
    const std::vector<STrack> & stracksa,
    const std::vector<STrack> & stracksb);

  /**
   * @brief Reset tracker state
   */
  void reset();

  /**
   * @brief Check if tracker should be reset due to scene change
   * @param detections_msg Current detections
   * @return True if tracker should be reset
   */
  bool shouldResetTracker(const vision_msgs::msg::Detection2DArray & detections_msg);

  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;
  // ROS subscribers and publishers
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detection_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr tracked_pub_;

  // Tracker parameters
  double track_high_thresh_;    // High confidence threshold for first association
  double track_low_thresh_;     // Low confidence threshold for second association
  double new_track_thresh_;     // Threshold for creating new tracks
  double match_thresh_;         // IoU threshold for matching
  int max_time_lost_;           // Maximum frames to keep lost tracks
  bool fuse_score_;             // Whether to fuse detection scores with IoU

  // Scene change detection parameters
  double scene_change_thresh_;         // Threshold for match ratio to detect scene change
  int min_detections_for_reset_;       // Minimum detections required to consider reset
  int frames_without_match_thresh_;    // Frames without good matches before reset

  // Tracker state
  std::vector<STrack> tracked_stracks_;
  std::vector<STrack> lost_stracks_;
  std::vector<STrack> removed_stracks_;
  int frame_id_;

  // Scene change detection state
  int frames_without_good_match_;      // Counter for frames without good matches
  int last_detection_count_;           // Previous frame detection count

  // Kalman filter
  std::shared_ptr<KalmanFilterXYAH> kalman_filter_;
};

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_YOLOV8__BYTE_TRACKER_NODE_HPP_
