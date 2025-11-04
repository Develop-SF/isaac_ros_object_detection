/*
 * Filename: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src/detection2_d_array_vlm_filter.cpp
 * Path: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src
 * Created Date: Monday, November 3rd 2025, 11:03:24 am
 * Author: Wen-Yu Chien
 * Description: Isaac ROS VLM BBOX Selector
 * Copyright (c) 2025 Copyright (c) 2025 Shinfang Global
 */

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
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
    track_id_("0")
  {
    // Create callback groups to prevent execution conflicts
    sync_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    detection_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    
    // Options for the subscription
    rclcpp::SubscriptionOptions detection_options;
    detection_options.callback_group = detection_callback_group_;
    
    filtered_detection2_d_pub_ = create_publisher<vision_msgs::msg::Detection2D>(
      "vlm_selected_detections", output_qos_);
    
    vlm_reason_pub_ = create_publisher<std_msgs::msg::String>(
      "vlm_reason", output_qos_);

    setupSynchronizer();

    // Use callback group for the detection-only subscription
    detections_only_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
      "detection2_d_array", input_qos_, 
      std::bind(&Detection2DArrayVLMFilter::bboxesCallback, this, std::placeholders::_1),
      detection_options);

    vlmInit();

    // Register parameter callback for runtime updates
    param_callback_handle_ = add_on_set_parameters_callback(
      std::bind(&Detection2DArrayVLMFilter::parametersCallback, this, std::placeholders::_1));
    
    RCLCPP_INFO(get_logger(), "Node initialized. Target class: '%s'", desired_class_name_.c_str());
  }

private:
  struct TrackImagePayload
  {
    std::string track_id;
    std::string image_base64;
  };

  void setupSynchronizer()
  {
    const auto qos_profile = input_qos_.get_rmw_qos_profile();

    // Create subscription options with the synchronizer callback group
    rclcpp::SubscriptionOptions sync_options;
    sync_options.callback_group = sync_callback_group_;

    // Create message filter subscribers with callback group
    image_sub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::msg::Image>>(
      this, "image", qos_profile, sync_options);
    detections_sub_ = std::make_unique<message_filters::Subscriber<vision_msgs::msg::Detection2DArray>>(
      this, "detection2_d_array", qos_profile, sync_options);

    synchronizer_ = std::make_shared<message_filters::Synchronizer<ExactTimePolicy>>(
      ExactTimePolicy(sync_queue_size_), *image_sub_, *detections_sub_);
    synchronizer_->registerCallback(
      std::bind(
        &Detection2DArrayVLMFilter::imageCallback,
        this,
        std::placeholders::_1,
        std::placeholders::_2));
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
        
        // Clear track_id when target class changes
        {
          std::lock_guard<std::mutex> lock(track_id_mutex_);
          if (new_value != desired_class_name_) {
            track_id_.clear();
            RCLCPP_INFO(get_logger(), 
                        "Target class updated: '%s' -> '%s' (track_id cleared)",
                        desired_class_name_.c_str(), new_value.c_str());
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

  void imageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr & detections_msg)
  {
    const auto candidate_detections = collectCandidateDetections(*detections_msg);
    if (candidate_detections.empty()) {
      std::lock_guard<std::mutex> lock(track_id_mutex_);
      track_id_.clear();
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

      // Debug: Display cropped image
      cv::resize(cropped, cropped, cv::Size(240, 640), 0, 0, cv::INTER_LINEAR);
      cv::imshow("Debug Image " + detection_ptr->id, cropped);
      cv::waitKey(1);

      std::vector<unsigned char> buffer;
      if (!cv::imencode(".jpg", cropped, buffer)) {
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
        
        // Clear track_id so bboxesCallback doesn't publish anything
        {
          std::lock_guard<std::mutex> lock(track_id_mutex_);
          track_id_.clear();
        }
        
        // Don't publish any detection bbox
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

  void bboxesCallback(const vision_msgs::msg::Detection2DArray::ConstSharedPtr & detections_msg)
  {
    if (!detections_msg) {
      RCLCPP_WARN(get_logger(), "Received null detections message");
      return;
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
        RCLCPP_INFO(get_logger(), "Published detection with track ID: %s", track_id_.c_str());
        return;
      }
    }

    // If we reach here, the track ID was not found in the current detections
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
      "Track ID %s not found in current detections", current_track_id.c_str());

    // Publish the best detection if track ID not found and assign -1 to indicate no tracking
    {
      std::lock_guard<std::mutex> lock(track_id_mutex_);
      track_id_ = "-1";
      current_track_id = track_id_;
    }
    
    vision_msgs::msg::Detection2D detection_to_publish;
    // pick the first detection as a fallback
    if (!detections_msg->detections.empty()) {
      detection_to_publish = detections_msg->detections[0];
      if (detection_to_publish.header.stamp.nanosec == 0 && detection_to_publish.header.stamp.sec == 0) {
        detection_to_publish.header = detections_msg->header;
      }
      detection_to_publish.id = current_track_id;
      filtered_detection2_d_pub_->publish(detection_to_publish);
      // RCLCPP_INFO(get_logger(), "Published fallback detection with track ID -1");
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Published fallback detection with track ID -1");
    }
  }

  // collect the detections that match the desired class id
  std::vector<const vision_msgs::msg::Detection2D *> collectCandidateDetections(
    const vision_msgs::msg::Detection2DArray & detections_msg) const
  {
    std::vector<const vision_msgs::msg::Detection2D *> matching_class_detections;
    std::vector<const vision_msgs::msg::Detection2D *> fallback_detections;
    matching_class_detections.reserve(detections_msg.detections.size());
    fallback_detections.reserve(detections_msg.detections.size());

    for (const auto & detection : detections_msg.detections) {
      fallback_detections.push_back(&detection);

      if (desired_class_id_.empty()) {
        continue;
      }

      for (const auto & result : detection.results) {
        if (result.hypothesis.class_id == desired_class_id_) {
          matching_class_detections.push_back(&detection);
          break;
        }
      }
    }

    if (!desired_class_id_.empty() && !matching_class_detections.empty()) {
      return matching_class_detections;
    }

    return fallback_detections;
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
    std::lock_guard<std::mutex> lock(vlm_mutex_);
    if (!vlm_client_) {
      RCLCPP_INFO(get_logger(), "VLM client not initialized");
      return std::nullopt;
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

    auto response = vlm_client_->Post("/api/chat", headers, payload.str(), "application/json");
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

  static constexpr int sync_queue_size_ = 10;

  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;
  std::string desired_class_id_;
  std::string desired_class_name_;
  std::string vlm_prompt_;
  std::string vlm_model_;
  std::string vlm_url_;
  std::string image_topic_name_;
  int timeout_seconds_;
  int max_token_;
  std::string track_id_;

  rclcpp::Publisher<vision_msgs::msg::Detection2D>::SharedPtr filtered_detection2_d_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr vlm_reason_pub_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detections_only_sub_;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> image_sub_;
  std::unique_ptr<message_filters::Subscriber<vision_msgs::msg::Detection2DArray>> detections_sub_;
  std::shared_ptr<message_filters::Synchronizer<ExactTimePolicy>> synchronizer_;

  std::mutex vlm_mutex_;
  std::mutex track_id_mutex_;
  std::unique_ptr<httplib::Client> vlm_client_;
  
  // Callback groups for execution control
  rclcpp::CallbackGroup::SharedPtr sync_callback_group_;
  rclcpp::CallbackGroup::SharedPtr detection_callback_group_;
  
  // Parameter callback handle
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
};

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolov8::Detection2DArrayVLMFilter)
