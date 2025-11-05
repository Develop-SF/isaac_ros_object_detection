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

#include "isaac_ros_yolov8/yolov8_decoder_node.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"


#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>

#include "vision_msgs/msg/detection2_d_array.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "isaac_ros_common/qos.hpp"


namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{
YoloV8DecoderNode::YoloV8DecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("yolov8_decoder_node", options),
  input_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")),
  output_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")),
  nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosTensorListView>>(
      this,
      "tensor_sub",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
      std::bind(&YoloV8DecoderNode::InputCallback, this,
      std::placeholders::_1),
      nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig(),
      input_qos_)},
  pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
      "detections_output", output_qos_)},
  tensor_name_{declare_parameter<std::string>("tensor_name", "output_tensor")},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.25)},
  nms_threshold_{declare_parameter<double>("nms_threshold", 0.45)},
  num_classes_{declare_parameter<int64_t>("num_classes", 80)},
  network_width_{declare_parameter<int64_t>("network_width", 640)},
  network_height_{declare_parameter<int64_t>("network_height", 640)}
{

  // Camera info topic parameter and subscription
  camera_info_topic_ = declare_parameter<std::string>("camera_info_topic", camera_info_topic_);
  camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    camera_info_topic_, 10, std::bind(&YoloV8DecoderNode::CameraInfoCallback, this, std::placeholders::_1));
}

YoloV8DecoderNode::~YoloV8DecoderNode() = default;

void YoloV8DecoderNode::CameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  // Store original image dimensions for mapping detections
  original_width_ = static_cast<int>(msg->width);
  original_height_ = static_cast<int>(msg->height);
}

void YoloV8DecoderNode::InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
{
  auto tensor = msg.GetNamedTensor(tensor_name_);
  size_t buffer_size{tensor.GetTensorSize()};
  std::vector<float> results_vector{};
  results_vector.resize(buffer_size);
  cudaMemcpy(results_vector.data(), tensor.GetBuffer(), buffer_size, cudaMemcpyDefault);
  // Safety check: Verify tensor data is valid
  if (buffer_size == 0 || results_vector.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Invalid tensor: empty or zero-sized buffer");
    return;
  }

  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  std::vector<int> indices;
  std::vector<int> classes;

  //  Output dimensions = [1, 84, 8400]
  int out_dim = 8400;  // Default anchor count for YOLOv8 models
  
  // Estimate tensor dimensions from buffer size
  // Buffer size = sizeof(float) * num_predictions * (4 + num_classes)
  const size_t float_count = buffer_size / sizeof(float);
  const int estimated_features = static_cast<int>(float_count / out_dim);
  const int estimated_classes = estimated_features - 4;  // 4 values are for bbox coordinates
  
  // Use the smaller of the configured classes or estimated classes to avoid out-of-bounds
  int classes_to_use = estimated_classes > 0 ? 
      std::min(static_cast<int>(num_classes_), estimated_classes) : 
      static_cast<int>(num_classes_);
      
  RCLCPP_DEBUG(this->get_logger(), "Tensor buffer size: %zu bytes, estimated features: %d, using %d classes", 
              buffer_size, estimated_features, classes_to_use);
  
  float * results_data = reinterpret_cast<float *>(results_vector.data());

  // Model input (resized) dimensions from parameters
  const int64_t resized_width = network_width_;
  const int64_t resized_height = network_height_;

  // Compute scale and padding to map model-space boxes to original image if camera info is available
  float scale = 1.0f;
  float padding_x = 0.0f;
  float padding_y = 0.0f;
  if (original_width_ > 0 && original_height_ > 0) {
    scale = std::min(static_cast<float>(resized_width) / static_cast<float>(original_width_),
                     static_cast<float>(resized_height) / static_cast<float>(original_height_));
    padding_x = (static_cast<float>(resized_width) - (original_width_ * scale)) / 2.0f;
    padding_y = (static_cast<float>(resized_height) - (original_height_ * scale)) / 2.0f;
  }

  for (int i = 0; i < out_dim; i++) {
    float x = *(results_data + i);
    float y = *(results_data + (out_dim * 1) + i);
    float w = *(results_data + (out_dim * 2) + i);
    float h = *(results_data + (out_dim * 3) + i);

    float x1 = (x - (0.5 * w));
    float y1 = (y - (0.5 * h));
    float width = w;
    float height = h;

    std::vector<float> conf;
    // Limit class access to the valid range
    for (int j = 0; j < classes_to_use; j++) {
      // Safely access confidence values: offset = base + (out_dim * (4 + class_index)) + anchor_index
      const size_t offset = out_dim * (4 + j) + i;
      if (offset < float_count) {
        conf.push_back(*(results_data + offset));
      } else {
        // If we're out of bounds, add a zero confidence
        conf.push_back(0.0f);
      }
    }

    std::vector<float>::iterator ind_max_conf;
    ind_max_conf = std::max_element(std::begin(conf), std::end(conf));
    int max_index = distance(std::begin(conf), ind_max_conf);
    float val_max_conf = *max_element(std::begin(conf), std::end(conf));

    // Map to original image coordinates if camera info provided; otherwise keep model-space coords
    float x1_scaled = x1;
    float y1_scaled = y1;
    float width_scaled = width;
    float height_scaled = height;
    if (original_width_ > 0 && original_height_ > 0 && scale > 0.0f) {
      x1_scaled = (x1 - padding_x) / scale;
      y1_scaled = (y1 - padding_y) / scale;
      width_scaled = width / scale;
      height_scaled = height / scale;
    }
    bboxes.push_back(cv::Rect(x1_scaled, y1_scaled, width_scaled, height_scaled));
    indices.push_back(i);
    scores.push_back(val_max_conf);
    classes.push_back(max_index);
  }

  RCLCPP_DEBUG(this->get_logger(), "Count of bboxes: %lu", bboxes.size());
  cv::dnn::NMSBoxes(bboxes, scores, confidence_threshold_, nms_threshold_, indices, 5);
  RCLCPP_DEBUG(this->get_logger(), "# boxes after NMS: %lu", indices.size());

  vision_msgs::msg::Detection2DArray final_detections_arr;

  for (size_t i = 0; i < indices.size(); i++) {
    int ind = indices[i];
    vision_msgs::msg::Detection2D detection;

    geometry_msgs::msg::Pose center;
    geometry_msgs::msg::Point position;
    geometry_msgs::msg::Quaternion orientation;

    // 2D object Bbox
    vision_msgs::msg::BoundingBox2D bbox;
    float w = bboxes[ind].width;
    float h = bboxes[ind].height;
    float x_center = bboxes[ind].x + (0.5 * w);
    float y_center = bboxes[ind].y + (0.5 * h);
    detection.bbox.center.position.x = x_center;
    detection.bbox.center.position.y = y_center;
    detection.bbox.size_x = w;
    detection.bbox.size_y = h;


    // Class probabilities
    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = std::to_string(classes.at(ind));
    hyp.hypothesis.score = scores.at(ind);
    detection.results.push_back(hyp);

    detection.header.stamp.sec = msg.GetTimestampSeconds();
    detection.header.stamp.nanosec = msg.GetTimestampNanoseconds();

    final_detections_arr.detections.push_back(detection);
  }

  final_detections_arr.header.stamp.sec = msg.GetTimestampSeconds();
  final_detections_arr.header.stamp.nanosec = msg.GetTimestampNanoseconds();
  pub_->publish(final_detections_arr);
}

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolov8::YoloV8DecoderNode)
