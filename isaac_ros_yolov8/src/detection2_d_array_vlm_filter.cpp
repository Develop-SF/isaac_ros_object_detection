/*
 * Filename: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src/detection2_d_array_vlm_filter.cpp
 * Path: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src
 * Created Date: Monday, November 3rd 2025, 11:03:24 am
 * Author: Wen-Yu Chien
 * Description: Isaac ROS VLM BBOX Selector
 * Copyright (c) 2025 Copyright (c) 2025 Shinfang Global
 */

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_yolov8/vlm_utils.hpp"

#include "httplib.h"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{

// Minimum image dimension required by Qwen2.5VL model
constexpr int VLM_MIN_IMAGE_SIZE = 28;

using ExactTimePolicy = message_filters::sync_policies::ExactTime<
  sensor_msgs::msg::Image,
  vision_msgs::msg::Detection2DArray>;

class Detection2DArrayVLMFilter : public rclcpp::Node
{
public:
  explicit Detection2DArrayVLMFilter(const rclcpp::NodeOptions & options)
  : Node("detection2_d_vlm_filter_node", options),
    input_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")),
    output_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")),
    desired_class_id_(declare_parameter<std::string>("desired_class_id", "")),
    desired_class_name_(declare_parameter<std::string>("desired_class_name", "soy sauce")),
    vlm_prompt_(declare_parameter<std::string>("vlm_prompt", "What is this? soy sauce or oil or seasoning")),
    vlm_model_(declare_parameter<std::string>("vlm_model", "qwen2.5vl:7b")),
    vlm_url_(declare_parameter<std::string>("vlm_url", "http://localhost:11434")),
    timeout_seconds_(std::max<int>(declare_parameter<int>("timeout", 300), 0)),
    max_token_(std::max<int>(declare_parameter<int>("max_token", 0), 0)),
    vlm_sampling_rate_(std::max<double>(declare_parameter<double>("vlm_sampling_rate", 0.2), 0.01)),
    track_id_(""),  // Start with empty track_id to prevent publishing before VLM processing
    track_id_miss_count_(0),
    vlm_processing_(false),
    shutdown_requested_(false)
  {
    // Create callback groups
    detection_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    image_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    
    // Options for subscriptions
    rclcpp::SubscriptionOptions detection_options;
    detection_options.callback_group = detection_callback_group_;
    
    rclcpp::SubscriptionOptions image_options;
    image_options.callback_group = image_callback_group_;
    
    filtered_detection2_d_pub_ = create_publisher<vision_msgs::msg::Detection2D>(
      "vlm_selected_detections", output_qos_);
    
    vlm_reason_pub_ = create_publisher<std_msgs::msg::String>(
      "vlm_reason", output_qos_);

    // Subscribe to detections for fast republishing
    detections_only_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
      "detection2_d_array", input_qos_, 
      std::bind(&Detection2DArrayVLMFilter::bboxesCallback, this, std::placeholders::_1),
      detection_options);

    // Subscribe to image for periodic VLM sampling (throttled)
    image_sub_simple_ = create_subscription<sensor_msgs::msg::Image>(
      "image", input_qos_,
      std::bind(&Detection2DArrayVLMFilter::imageSamplingCallback, this, std::placeholders::_1),
      image_options);

    vlmInit();
    
    // Start VLM worker thread
    vlm_worker_thread_ = std::thread(&Detection2DArrayVLMFilter::vlmWorkerThread, this);

    // Register parameter callback for runtime updates
    param_callback_handle_ = add_on_set_parameters_callback(
      std::bind(&Detection2DArrayVLMFilter::parametersCallback, this, std::placeholders::_1));
    
    RCLCPP_INFO(get_logger(), "Node initialized. Target class: '%s', VLM sampling rate: %.2f Hz", 
                desired_class_name_.c_str(), vlm_sampling_rate_);
  }

  ~Detection2DArrayVLMFilter()
  {
    // Signal shutdown and wait for worker thread
    {
      std::lock_guard<std::mutex> lock(vlm_queue_mutex_);
      shutdown_requested_ = true;
    }
    vlm_queue_cv_.notify_all();
    
    if (vlm_worker_thread_.joinable()) {
      vlm_worker_thread_.join();
    }
  }

private:
  struct TrackImagePayload
  {
    std::string track_id;
    std::string image_base64;
  };

  // VLM task structure for async processing
  struct VLMTask {
    sensor_msgs::msg::Image::ConstSharedPtr image_msg;
    vision_msgs::msg::Detection2DArray::ConstSharedPtr detections_msg;
  };

  // Async VLM worker thread
  void vlmWorkerThread()
  {
    RCLCPP_INFO(get_logger(), "VLM worker thread started");
    
    while (!shutdown_requested_) {
      VLMTask task;
      
      // Wait for a task
      {
        std::unique_lock<std::mutex> lock(vlm_queue_mutex_);
        vlm_queue_cv_.wait(lock, [this] { 
          return !vlm_task_queue_.empty() || shutdown_requested_; 
        });
        
        if (shutdown_requested_) {
          break;
        }
        
        if (!vlm_task_queue_.empty()) {
          task = std::move(vlm_task_queue_.front());
          vlm_task_queue_.pop();
        } else {
          continue;
        }
      }
      
      // Process task asynchronously
      processVLMTask(task);
      
      vlm_processing_.store(false);
    }
    
    RCLCPP_INFO(get_logger(), "VLM worker thread stopped");
  }

  // Throttled image sampling callback - runs at vlm_sampling_rate_ Hz
  void imageSamplingCallback(const sensor_msgs::msg::Image::ConstSharedPtr & image_msg)
  {
    // Throttle based on vlm_sampling_rate_
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
      now - last_vlm_sample_time_).count();
    
    if (elapsed < (1.0 / vlm_sampling_rate_)) {
      return;  // Skip this frame
    }
    
    last_vlm_sample_time_ = now;
    
    // Skip if VLM is already processing
    if (vlm_processing_.load()) {
      RCLCPP_DEBUG(get_logger(), "VLM still processing, skipping frame");
      return;
    }
    
    // Get latest detections (non-blocking, just cache the latest)
    vision_msgs::msg::Detection2DArray::ConstSharedPtr latest_detections;
    {
      std::lock_guard<std::mutex> lock(cached_detections_mutex_);
      if (!cached_detections_) {
        RCLCPP_DEBUG(get_logger(), "No cached detections available");
        return;
      }
      latest_detections = cached_detections_;
    }
    
    // Create VLM task
    VLMTask task;
    task.image_msg = image_msg;
    task.detections_msg = latest_detections;
    
    // Queue the task (with size limit to prevent backpressure)
    {
      std::lock_guard<std::mutex> lock(vlm_queue_mutex_);
      if (vlm_task_queue_.size() >= 2) {
        RCLCPP_DEBUG(get_logger(), "VLM queue full, skipping frame to prevent backpressure");
        return;
      }
      vlm_task_queue_.push(std::move(task));
    }
    
    vlm_processing_.store(true);
    vlm_queue_cv_.notify_one();
  }

  // Process VLM task in worker thread
  void processVLMTask(const VLMTask & task)
  {
    const auto candidate_detections = collectCandidateDetections(*task.detections_msg);
    if (candidate_detections.empty()) {
      return;
    }

    cv_bridge::CvImageConstPtr bridge;
    try {
      bridge = cv_bridge::toCvShare(task.image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception & ex) {
      RCLCPP_WARN(get_logger(), "Falling back to native image encoding: %s", ex.what());
      try {
        bridge = cv_bridge::toCvShare(task.image_msg, task.image_msg->encoding);
      } catch (const cv_bridge::Exception & inner_ex) {
        RCLCPP_ERROR(get_logger(), "Failed to convert image: %s", inner_ex.what());
        return;
      }
    }

    cv::Mat image = bridge->image;
    if (image.empty()) {
      RCLCPP_WARN(get_logger(), "Received empty image frame");
      return;
    }

    std::vector<TrackImagePayload> track_image_payloads;
    track_image_payloads.reserve(candidate_detections.size());

    for (size_t idx = 0; idx < candidate_detections.size(); ++idx) {
      const auto * detection_ptr = candidate_detections[idx];
      auto roi_opt = detectionToRoi(*detection_ptr, image.cols, image.rows);
      if (!roi_opt) {
        continue;
      }

      const cv::Rect roi = *roi_opt;
      cv::Mat cropped = image(roi).clone();
      if (cropped.empty()) {
        continue;
      }

      // Ensure the cropped image meets minimum size requirements
      cv::Mat resized_crop = ensureMinimumSize(cropped);
      if (resized_crop.empty()) {
        continue;
      }

      std::vector<unsigned char> buffer;
      if (!cv::imencode(".jpg", resized_crop, buffer)) {
        continue;
      }

      std::string track_id = detection_ptr->id;
      if (track_id.empty()) {
        track_id = "det_" + std::to_string(idx);
      }

      track_image_payloads.push_back(TrackImagePayload{track_id, base64_encode(buffer)});
    }

    if (track_image_payloads.empty()) {
      return;
    }

    std::string selected_track_id;
    
    if (desired_class_id_.empty()) {
      selected_track_id = track_image_payloads.front().track_id;
      std::lock_guard<std::mutex> lock(track_id_mutex_);
      track_id_ = selected_track_id;
    } else {
      // Query VLM for each detection in parallel using threads
      std::vector<VLMJsonResponse> all_responses;
      all_responses.resize(track_image_payloads.size());
      auto total_start_time = std::chrono::steady_clock::now();
      
      std::vector<std::thread> query_threads;
      query_threads.reserve(track_image_payloads.size());
      
      for (size_t i = 0; i < track_image_payloads.size(); ++i) {
        query_threads.emplace_back([this, i, &track_image_payloads, &all_responses]() {
          const auto & payload = track_image_payloads[i];
          RCLCPP_INFO(get_logger(), "Querying VLM for Track ID: %s", payload.track_id.c_str());
          
          auto vlm_response = queryVlmSingle(vlm_prompt_, payload.track_id, payload.image_base64);
          
          if (vlm_response) {
            VLMJsonResponse parsed = parseVLMResponse(*vlm_response);
            if (parsed.valid) {
              all_responses[i] = parsed;
            }
          }
        });
      }
      
      // Wait for all queries to complete
      for (auto & thread : query_threads) {
        if (thread.joinable()) {
          thread.join();
        }
      }
      
      // Process results and find match
      for (const auto & parsed : all_responses) {
        if (!parsed.valid) continue;
        
        if (selected_track_id.empty() && containsCaseInsensitive(parsed.object_name, desired_class_name_)) {
          selected_track_id = parsed.id;
          RCLCPP_INFO(get_logger(), "Found match! Track ID: %s is %s", 
                      selected_track_id.c_str(), parsed.object_name.c_str());
          
          // Publish reason
          std_msgs::msg::String reason_msg;
          reason_msg.data = "Match found - Track ID " + parsed.id + ": " + parsed.object_name;
          vlm_reason_pub_->publish(reason_msg);
        }
      }
      
      auto total_end_time = std::chrono::steady_clock::now();
      auto total_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end_time - total_start_time).count();
      
      RCLCPP_INFO(get_logger(), "VLM total processing time (parallel): %ld ms for %zu detections", 
                  total_duration_ms, track_image_payloads.size());
      
      // Filter valid responses
      std::vector<VLMJsonResponse> valid_responses;
      for (const auto & resp : all_responses) {
        if (resp.valid) valid_responses.push_back(resp);
      }
      
      if (selected_track_id.empty() && !valid_responses.empty()) {
        // No match - publish what was detected
        std::ostringstream no_match_reason;
        no_match_reason << "No '" << desired_class_name_ << "' found. Detected: ";
        for (size_t i = 0; i < valid_responses.size(); ++i) {
          if (i > 0) no_match_reason << ", ";
          no_match_reason << "Track ID " << valid_responses[i].id << ": " << valid_responses[i].object_name;
        }
        
        std_msgs::msg::String reason_msg;
        reason_msg.data = no_match_reason.str();
        vlm_reason_pub_->publish(reason_msg);
        return;
      }
      
      // Update track ID
      if (!selected_track_id.empty()) {
        std::lock_guard<std::mutex> lock(track_id_mutex_);
        track_id_ = selected_track_id;
        RCLCPP_INFO(get_logger(), "Updated track_id to: %s", selected_track_id.c_str());
      }
    }
  }

  void vlmInit()
  {
    std::lock_guard<std::mutex> lock(vlm_mutex_);
    vlm_client_ = std::make_unique<httplib::Client>(vlm_url_);
    if (timeout_seconds_ > 0) {
      vlm_client_->set_connection_timeout(timeout_seconds_, 0);
      vlm_client_->set_read_timeout(timeout_seconds_, 0);
    }
    RCLCPP_INFO(get_logger(), "Initialized VLM client for %s", vlm_url_.c_str());
  }

  rcl_interfaces::msg::SetParametersResult parametersCallback(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;

    for (const auto & param : parameters) {
      if (param.get_name() == "desired_class_name") {
        std::string new_value = param.as_string();
        result.successful = true;
        
        // Clear track_id when target class changes and cancel ongoing VLM requests
        {
          std::lock_guard<std::mutex> lock(track_id_mutex_);
          if (new_value != desired_class_name_) {
            track_id_.clear();
            RCLCPP_INFO(get_logger(), 
                        "Target class updated: '%s' -> '%s' (track_id cleared)",
                        desired_class_name_.c_str(), new_value.c_str());
            
            // Clear pending VLM tasks
            {
              std::lock_guard<std::mutex> vlm_lock(vlm_queue_mutex_);
              while (!vlm_task_queue_.empty()) {
                vlm_task_queue_.pop();
              }
              RCLCPP_INFO(get_logger(), "Cleared VLM task queue");
            }
          }
        }
        
        desired_class_name_ = new_value;
        result.reason = "Desired class name updated: '" + desired_class_name_ + "'";
        
      } else if (param.get_name() == "vlm_prompt") {
        vlm_prompt_ = param.as_string();
        RCLCPP_INFO(get_logger(), "VLM prompt updated: '%s'", vlm_prompt_.c_str());
        result.reason = "VLM prompt updated: '" + vlm_prompt_ + "'";
      }
    }

    return result;
  }

	// System 2: careful thinking and make decision
  void imageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr & detections_msg)
  {
    const auto candidate_detections = collectCandidateDetections(*detections_msg);
    if (candidate_detections.empty()) {
      return;
    }

    cv_bridge::CvImageConstPtr bridge;
    try {
      bridge = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception & ex) {
      RCLCPP_WARN(get_logger(), "Falling back to native image encoding: %s", ex.what());
      try {
        bridge = cv_bridge::toCvShare(image_msg, image_msg->encoding);
      } catch (const cv_bridge::Exception & inner_ex) {
        RCLCPP_ERROR(get_logger(), "Failed to convert image: %s", inner_ex.what());
        return;
      }
    }

    cv::Mat image = bridge->image;
    if (image.empty()) {
      RCLCPP_WARN(get_logger(), "Received empty image frame");
      return;
    }

    std::vector<TrackImagePayload> track_image_payloads;  // track_id and cropped image
    track_image_payloads.reserve(candidate_detections.size());

    for (size_t idx = 0; idx < candidate_detections.size(); ++idx) {
      const auto * detection_ptr = candidate_detections[idx];
      auto roi_opt = detectionToRoi(*detection_ptr, image.cols, image.rows);
      if (!roi_opt) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Skipping detection %zu due to invalid ROI", idx);
        continue;
      }

      const cv::Rect roi = *roi_opt;
      cv::Mat cropped = image(roi).clone();
      if (cropped.empty()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Skipping detection %zu because cropped image is empty", idx);
        continue;
      }

      // Ensure the cropped image meets minimum size requirements
      cv::Mat resized_crop = ensureMinimumSize(cropped);
      if (resized_crop.empty()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Skipping detection %zu because resized image is empty", idx);
        continue;
      }

      std::vector<unsigned char> buffer;
      if (!cv::imencode(".jpg", resized_crop, buffer)) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Failed to encode cropped image for detection %zu", idx);
        continue;
      }

      std::string track_id = detection_ptr->id;
      if (track_id.empty()) {
        track_id = "det_" + std::to_string(idx);
      }

      track_image_payloads.push_back(TrackImagePayload{track_id, base64_encode(buffer)});
    }

    if (track_image_payloads.empty()) {
      RCLCPP_WARN(get_logger(), "Skipping VLM query because no detection crops were valid");
      return;
    }

    std::string selected_track_id;
    std::string combined_reason;
    
    if (desired_class_id_.empty()) {
      // No class filter, use the first detection
      selected_track_id = track_image_payloads.front().track_id;
      std::lock_guard<std::mutex> lock(track_id_mutex_);
      track_id_ = selected_track_id;
    } else {
      // Query VLM for each detection individually and collect all responses
      std::vector<VLMJsonResponse> all_responses;
      
      auto total_start_time = std::chrono::steady_clock::now();
      
      for (const auto & payload : track_image_payloads) {
        RCLCPP_INFO(get_logger(), "Querying VLM for Track ID: %s", payload.track_id.c_str());
        
        auto query_start_time = std::chrono::steady_clock::now();
        auto vlm_response = queryVlmSingle(vlm_prompt_, payload.track_id, payload.image_base64);
        auto query_end_time = std::chrono::steady_clock::now();
        
        auto query_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          query_end_time - query_start_time).count();
        
        if (!vlm_response) {
          RCLCPP_WARN(get_logger(), "VLM response unavailable for Track ID: %s (query time: %ld ms)", 
                      payload.track_id.c_str(), query_duration_ms);
          continue;
        }
        
        RCLCPP_INFO(get_logger(), "VLM raw response for Track ID %s (query time: %ld ms): '%s'", 
                    payload.track_id.c_str(), query_duration_ms, vlm_response->c_str());
        
        // Parse JSON response
        VLMJsonResponse parsed = parseVLMResponse(*vlm_response);
        if (parsed.valid) {
          RCLCPP_INFO(get_logger(), "VLM parsed - id: '%s', object: '%s', reason: '%s'", 
                      parsed.id.c_str(), parsed.object_name.c_str(), parsed.concise_reason.c_str());
          
          all_responses.push_back(parsed);
          
          // Check if this detection matches the desired class name
          if (selected_track_id.empty() && containsCaseInsensitive(parsed.object_name, desired_class_name_)) {
            selected_track_id = parsed.id;
            combined_reason = "Match found - Track ID " + parsed.id + ": " + parsed.object_name + 
                            (parsed.concise_reason.empty() ? "" : " (" + parsed.concise_reason + ")");
            
            RCLCPP_INFO(get_logger(), "Found match! Track ID: %s is %s", 
                        selected_track_id.c_str(), parsed.object_name.c_str());
          }
        } else {
          RCLCPP_WARN(get_logger(), "Failed to parse JSON response for Track ID: %s", payload.track_id.c_str());
        }
      }
      
      auto total_end_time = std::chrono::steady_clock::now();
      auto total_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end_time - total_start_time).count();
      
      RCLCPP_INFO(get_logger(), "VLM total processing time: %ld ms for %zu detections (avg: %.1f ms/detection)", 
                  total_duration_ms, track_image_payloads.size(), 
                  track_image_payloads.empty() ? 0.0 : static_cast<double>(total_duration_ms) / track_image_payloads.size());
      
      if (selected_track_id.empty()) {
        // No match found, but publish what was detected
        if (!all_responses.empty()) {
          std::ostringstream no_match_reason;
          no_match_reason << "No '" << desired_class_name_ << "' found. Detected: ";
          for (size_t i = 0; i < all_responses.size(); ++i) {
            if (i > 0) no_match_reason << ", ";
            no_match_reason << "Track ID " << all_responses[i].id << ": " << all_responses[i].object_name;
          }
          combined_reason = no_match_reason.str();
          
          // Publish the reason explaining what was detected
          std_msgs::msg::String reason_msg;
          reason_msg.data = combined_reason;
          vlm_reason_pub_->publish(reason_msg);
          
          RCLCPP_WARN(get_logger(), "%s", combined_reason.c_str());
        }
        
        // Don't publish any detection bbox, but keep previous track_id for bboxesCallback
        return;
      }
      
      // Match found - publish the reason to a separate topic
      std_msgs::msg::String reason_msg;
      reason_msg.data = combined_reason;
      vlm_reason_pub_->publish(reason_msg);
      
      // Update the stored track ID for bboxesCallback to use
      {
        std::lock_guard<std::mutex> lock(track_id_mutex_);
        track_id_ = selected_track_id;
      }
      
      RCLCPP_INFO(get_logger(), "Updated track_id to: %s", selected_track_id.c_str());
    }

    // Find and publish the detection with the selected track_id
    const vision_msgs::msg::Detection2D * selected_detection = nullptr;
    for (size_t idx = 0; idx < candidate_detections.size(); ++idx) {
      if (candidate_detections[idx]->id == selected_track_id) {
        selected_detection = candidate_detections[idx];
        break;
      }
    }

    if (selected_detection) {
      vision_msgs::msg::Detection2D detection_to_publish = *selected_detection;
      if (detection_to_publish.header.stamp.nanosec == 0 && detection_to_publish.header.stamp.sec == 0) {
        detection_to_publish.header = detections_msg->header;
      }
      
      // Publish the detection
      filtered_detection2_d_pub_->publish(detection_to_publish);
      RCLCPP_INFO(get_logger(), "Published detection from synchronized callback with track ID: %s", 
                  detection_to_publish.id.c_str());
    } else {
      RCLCPP_WARN(get_logger(), "Selected track_id '%s' not found in current detections", 
                  selected_track_id.c_str());
    }
  }

	// System 1: fast republish
  void bboxesCallback(const vision_msgs::msg::Detection2DArray::ConstSharedPtr & detections_msg)
  {
    if (!detections_msg) {
      RCLCPP_WARN(get_logger(), "Received null detections message");
      return;
    }

    // Cache detections for VLM processing
    {
      std::lock_guard<std::mutex> lock(cached_detections_mutex_);
      cached_detections_ = detections_msg;
    }

    std::string current_track_id;
    {
      std::lock_guard<std::mutex> lock(track_id_mutex_);
      current_track_id = track_id_;
    }

    // If track_id is empty, it means no target object was found by VLM
    // Don't publish any detection bbox
    if (current_track_id.empty()) {
      RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 2000,
        "No target object found by VLM, skipping detection publish");
      return;
    }

    for (const auto & detection : detections_msg->detections) {
      if (detection.id == current_track_id) {
        vision_msgs::msg::Detection2D detection_to_publish = detection;
        if (detection_to_publish.header.stamp.nanosec == 0 && detection_to_publish.header.stamp.sec == 0) {
          detection_to_publish.header = detections_msg->header;
        }

        filtered_detection2_d_pub_->publish(detection_to_publish);
        // Reset the miss counter when we successfully find the track
        {
          std::lock_guard<std::mutex> lock(track_id_mutex_);
          track_id_miss_count_ = 0;
        }
        // RCLCPP_INFO(get_logger(), "Published detection with track ID: %s", track_id_.c_str());
        return;
      }
    }

    // If we reach here, the track ID was not found in the current detections
    // Use a grace period before clearing to handle temporary tracking losses
    {
      std::lock_guard<std::mutex> lock(track_id_mutex_);
      track_id_miss_count_++;
      
      // Only clear after missing for multiple consecutive frames (grace period)
      if (track_id_miss_count_ > 10) {  // ~0.3 seconds at 30 FPS
        track_id_.clear();
        track_id_miss_count_ = 0;
        RCLCPP_WARN(get_logger(),
          "Track ID %s not found for %d frames, cleared track_id", 
          current_track_id.c_str(), track_id_miss_count_);
      } else {
        RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 1000,
          "Track ID %s temporarily lost (miss count: %d/10)", 
          current_track_id.c_str(), track_id_miss_count_);
      }
    }
  }

  // collect the detections that match the desired class id
  std::vector<const vision_msgs::msg::Detection2D *> collectCandidateDetections(
    const vision_msgs::msg::Detection2DArray & detections_msg) const
  {
    std::vector<const vision_msgs::msg::Detection2D *> matching_class_detections;
    matching_class_detections.reserve(detections_msg.detections.size());

    // If no specific class ID is desired, process all detections
    if (desired_class_id_.empty()) {
      for (const auto & detection : detections_msg.detections) {
        matching_class_detections.push_back(&detection);
      }
      return matching_class_detections;
    }

    // If a specific class ID is desired, only return detections with that class ID
    for (const auto & detection : detections_msg.detections) {
      for (const auto & result : detection.results) {
        if (result.hypothesis.class_id == desired_class_id_) {
          matching_class_detections.push_back(&detection);
          break;
        }
      }
    }

    // Return matching detections (empty if no matches found)
    // This prevents VLM processing when the desired class is not detected
    return matching_class_detections;
  }

  // Ensure image meets minimum size requirement for VLM processing
  // Resizes with padding to preserve aspect ratio
  static cv::Mat ensureMinimumSize(const cv::Mat & input)
  {
    if (input.empty()) {
      return input;
    }

    // Check if resize is needed
    if (input.cols >= VLM_MIN_IMAGE_SIZE && input.rows >= VLM_MIN_IMAGE_SIZE) {
      return input;
    }

    // Calculate scale factor to make smallest dimension = VLM_MIN_IMAGE_SIZE
    const double scale = static_cast<double>(VLM_MIN_IMAGE_SIZE) / 
                         std::min(input.cols, input.rows);
    
    const int new_width = std::max(VLM_MIN_IMAGE_SIZE, 
                                    static_cast<int>(std::round(input.cols * scale)));
    const int new_height = std::max(VLM_MIN_IMAGE_SIZE, 
                                     static_cast<int>(std::round(input.rows * scale)));

    // Resize the image
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

    // If one dimension is still too small (due to extreme aspect ratio), add padding
    const int final_width = std::max(new_width, VLM_MIN_IMAGE_SIZE);
    const int final_height = std::max(new_height, VLM_MIN_IMAGE_SIZE);

    if (final_width > new_width || final_height > new_height) {
      // Create a canvas with padding (use gray color for padding)
      cv::Mat padded = cv::Mat(final_height, final_width, input.type(), cv::Scalar(128, 128, 128));
      
      // Center the resized image
      const int x_offset = (final_width - new_width) / 2;
      const int y_offset = (final_height - new_height) / 2;
      
      resized.copyTo(padded(cv::Rect(x_offset, y_offset, new_width, new_height)));
      return padded;
    }

    return resized;
  }

  // convert the detection to a region of interest
  std::optional<cv::Rect> detectionToRoi(
    const vision_msgs::msg::Detection2D & detection,
    int image_width,
    int image_height) const
  {
    if (image_width <= 0 || image_height <= 0) {
      return std::nullopt;
    }

    const double bbox_width = detection.bbox.size_x;
    const double bbox_height = detection.bbox.size_y;
    if (bbox_width <= 1.0 || bbox_height <= 1.0) {
      return std::nullopt;
    }

    const double center_x = detection.bbox.center.position.x;
    const double center_y = detection.bbox.center.position.y;

    const int x_min = std::clamp(
      static_cast<int>(std::round(center_x - bbox_width * 0.5)), 0, image_width);
    const int y_min = std::clamp(
      static_cast<int>(std::round(center_y - bbox_height * 0.5)), 0, image_height);
    const int x_max = std::clamp(
      static_cast<int>(std::round(center_x + bbox_width * 0.5)), 0, image_width);
    const int y_max = std::clamp(
      static_cast<int>(std::round(center_y + bbox_height * 0.5)), 0, image_height);

    if (x_max <= x_min || y_max <= y_min) {
      return std::nullopt;
    }

    return cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min);
  }

  // Query the VLM model with a single detection using Ollama structured outputs
  std::optional<std::string> queryVlmSingle(
    const std::string & prompt,
    const std::string & track_id,
    const std::string & image_base64)
  {
    // Create a thread-local client for parallel queries
    httplib::Client local_client(vlm_url_);
    if (timeout_seconds_ > 0) {
      local_client.set_connection_timeout(timeout_seconds_, 0);
      local_client.set_read_timeout(timeout_seconds_, 0);
    }

    // Build JSON payload with Ollama structured outputs format
    std::ostringstream payload;
    payload << "{\"model\":\"" << json_escape(vlm_model_) << "\","
            << "\"messages\":[{\"role\":\"user\",\"content\":\"" 
            << json_escape(prompt) << "\","
            << "\"images\":[\"" << json_escape(image_base64) << "\"]}],"
            << "\"stream\":false,"
            << "\"format\":{"
            << "\"type\":\"object\","
            << "\"properties\":{"
            << "\"id\":{\"type\":\"string\"},"
            << "\"object_name\":{\"type\":\"string\"},"
            << "\"concise_reason\":{\"type\":\"string\"}"
            << "},"
            << "\"required\":[\"id\",\"object_name\",\"concise_reason\"]"
            << "}";

    if (max_token_ > 0) {
      payload << R"(,"options":{"num_predict":)" << max_token_ << "}";
    }

    payload << "}";

    RCLCPP_DEBUG_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "VLM request payload: %s", payload.str().c_str());

    httplib::Headers headers = {
      {"Content-Type", "application/json"}
    };

    auto response = local_client.Post("/api/chat", headers, payload.str(), "application/json");
    if (!response) {
      RCLCPP_INFO(get_logger(), "VLM request failed: %s", httplib::to_string(response.error()).c_str());
      return std::nullopt;
    }

    if (response->status != 200) {
      RCLCPP_INFO(
        get_logger(),
        "VLM returned status %d: %s",
        response->status,
        response->body.c_str());
      return std::nullopt;
    }

    const std::string output_text = extract_output_text(response->body);
    if (!output_text.empty()) {
      logStatistics(response->body);
      // Inject the track_id into the response since VLM doesn't know it
      std::string modified_response = output_text;
      // Replace "id":"..." with "id":"<track_id>"
      size_t id_pos = modified_response.find("\"id\"");
      if (id_pos != std::string::npos) {
        size_t colon_pos = modified_response.find(":", id_pos);
        if (colon_pos != std::string::npos) {
          size_t start_quote = modified_response.find("\"", colon_pos);
          if (start_quote != std::string::npos) {
            size_t end_quote = modified_response.find("\"", start_quote + 1);
            if (end_quote != std::string::npos) {
              modified_response.replace(start_quote + 1, end_quote - start_quote - 1, track_id);
            }
          }
        }
      }
      return modified_response;
    }

    logStatistics(response->body);
    return response->body;
  }

  // log the statistics of the VLM model
  void logStatistics(const std::string & body)
  {
    const ResponseStatistics stats = compute_statistics(body);
    if (!has_statistics(stats)) {
      return;
    }

    RCLCPP_DEBUG(
      get_logger(),
      "VLM stats - input tokens: %lld, output tokens: %lld, latency(ms): %.2f",
      stats.input_tokens,
      stats.output_tokens,
      stats.latency_ms);
  }

  // check if the text contains the pattern
  static bool containsCaseInsensitive(const std::string & text, const std::string & pattern)
  {
    if (pattern.empty()) {
      return true;
    }

    std::string text_lower = text;
    std::string pattern_lower = pattern;
    std::transform(
      text_lower.begin(), text_lower.end(), text_lower.begin(),
      [](unsigned char c) {return static_cast<char>(std::tolower(c));});
    std::transform(
      pattern_lower.begin(), pattern_lower.end(), pattern_lower.begin(),
      [](unsigned char c) {return static_cast<char>(std::tolower(c));});

    return text_lower.find(pattern_lower) != std::string::npos;
  }

  // the response from the VLM model
  struct VLMJsonResponse {
    std::string id;
    std::string object_name;
    std::string concise_reason;
    bool valid;
  };

  // parse the response from the VLM model
  static VLMJsonResponse parseVLMResponse(const std::string & response)
  {
    VLMJsonResponse result;
    result.valid = false;

    // Simple JSON parser for the expected format: {"id": "...", "object_name": "...", "concise_reason": "..."}
    size_t id_pos = response.find("\"id\"");
    size_t obj_pos = response.find("\"object_name\"");
    size_t reason_pos = response.find("\"concise_reason\"");

    if (id_pos == std::string::npos || obj_pos == std::string::npos) {
      return result;
    }

    // Extract "id" field
    size_t id_colon = response.find(":", id_pos);
    if (id_colon != std::string::npos) {
      size_t id_quote_start = response.find("\"", id_colon);
      if (id_quote_start != std::string::npos) {
        size_t id_quote_end = response.find("\"", id_quote_start + 1);
        if (id_quote_end != std::string::npos) {
          result.id = response.substr(id_quote_start + 1, id_quote_end - id_quote_start - 1);
        }
      }
    }

    // Extract "object_name" field
    size_t obj_colon = response.find(":", obj_pos);
    if (obj_colon != std::string::npos) {
      size_t obj_quote_start = response.find("\"", obj_colon);
      if (obj_quote_start != std::string::npos) {
        size_t obj_quote_end = response.find("\"", obj_quote_start + 1);
        if (obj_quote_end != std::string::npos) {
          result.object_name = response.substr(obj_quote_start + 1, obj_quote_end - obj_quote_start - 1);
        }
      }
    }

    // Extract "concise_reason" field (optional)
    if (reason_pos != std::string::npos) {
      size_t reason_colon = response.find(":", reason_pos);
      if (reason_colon != std::string::npos) {
        size_t reason_quote_start = response.find("\"", reason_colon);
        if (reason_quote_start != std::string::npos) {
          size_t reason_quote_end = response.find("\"", reason_quote_start + 1);
          if (reason_quote_end != std::string::npos) {
            result.concise_reason = response.substr(reason_quote_start + 1, reason_quote_end - reason_quote_start - 1);
          }
        }
      }
    }

    result.valid = !result.id.empty() && !result.object_name.empty();
    return result;
  }

  // Parameters
  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;
  std::string desired_class_id_;
  std::string desired_class_name_;
  std::string vlm_prompt_;
  std::string vlm_model_;
  std::string vlm_url_;
  int timeout_seconds_;
  int max_token_;
  double vlm_sampling_rate_;
  std::string track_id_;

  // Publishers
  rclcpp::Publisher<vision_msgs::msg::Detection2D>::SharedPtr filtered_detection2_d_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr vlm_reason_pub_;
  
  // Subscribers
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detections_only_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_simple_;

  // VLM client
  std::mutex vlm_mutex_;
  std::unique_ptr<httplib::Client> vlm_client_;
  
  // Async VLM processing
  std::mutex vlm_queue_mutex_;
  std::condition_variable vlm_queue_cv_;
  std::queue<VLMTask> vlm_task_queue_;
  std::thread vlm_worker_thread_;
  std::atomic<bool> vlm_processing_;
  std::atomic<bool> shutdown_requested_;
  std::chrono::steady_clock::time_point last_vlm_sample_time_;
  
  // Detection caching
  std::mutex cached_detections_mutex_;
  vision_msgs::msg::Detection2DArray::ConstSharedPtr cached_detections_;
  
  // Track ID storage
  std::mutex track_id_mutex_;
  int track_id_miss_count_;  // Counter for consecutive frames without finding track_id
  
  // Callback groups
  rclcpp::CallbackGroup::SharedPtr detection_callback_group_;
  rclcpp::CallbackGroup::SharedPtr image_callback_group_;
  
  // Parameter callback handle
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
};

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolov8::Detection2DArrayVLMFilter)
