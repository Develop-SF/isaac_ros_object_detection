/*
 * Filename: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src/detection2_d_array_vlm_filter.cpp
 * Path: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src
 * Created Date: Monday, November 3rd 2025, 11:03:24 am
 * Author: Wen-Yu Chien
 * Description: Isaac ROS VLM BBOX Selector
 * Copyright (c) 2025 Copyright (c) 2025 Shinfang Global
 */

#include <limits>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "isaac_ros_common/qos.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{

/*
ROS2 node that selects a single object from a vision_msgs::msg::Detection2DArray
based on VLM (Vision Language Model) criteria. This is modified from
isaac_ros_foundationpose::detection2_d_array_filter_node.cpp
*/
class Detection2DArrayVLMFilter : public rclcpp::Node
{
public:
  explicit Detection2DArrayVLMFilter(const rclcpp::NodeOptions & options)
  : Node("detection2_d_vlm_filter_node", options),
    input_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")),
    output_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")),
    desired_class_id_(declare_parameter<std::string>("desired_class_id", "")),
    detection2_d_array_sub_{create_subscription<vision_msgs::msg::Detection2DArray>(
      "detection2_d_array", input_qos_,
      std::bind(
        &Detection2DArrayVLMFilter::boundingBoxArrayCallback, this, std::placeholders::_1))},
    filtered_detection2_d_pub_{create_publisher<vision_msgs::msg::Detection2D>("detection2_d",
      output_qos_)},
		track_id_("0")
  {
  }

private:
  void boundingBoxArrayCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
  {
    const vision_msgs::msg::Detection2D * best_detection = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();

		// container to hold the desired detections
		std::vector<vision_msgs::msg::Detection2D> desired_detections;

		// keep the desired class id only if specified
		if (!desired_class_id_.empty()) {
			for (const auto & detection : msg->detections) {
				if (detection.results.empty()) {
					continue;
				}

				// check the results.hypothesis.class_id against desired_class_id_
				for (const auto & results : detection.results) {
					if (results.hypothesis.class_id == desired_class_id_) {
						desired_detections.push_back(detection);
						break;  // no need to check other hypothesis for this detection
					}
				}
			}
		}

		// find the detection with the highest score among the desired detections
		for (const auto & detection : desired_detections) {
			const auto & hypothesis = detection.results[0].hypothesis;
			if (hypothesis.class_id == desired_class_id_) {
				if (hypothesis.score > best_score) {
					best_score = hypothesis.score;
					best_detection = &detection;
				}
			}
		}

		// publish the best detection if found
    if (best_detection != nullptr) {
			track_id_ = !best_detection->id.empty() ? best_detection->id : track_id_;

      filtered_detection2_d_pub_->publish(*best_detection);
			RCLCPP_INFO(
				this->get_logger(), "[DEBUG] Published detection with track ID: %s", track_id_.c_str());
    } else {
      track_id_.clear();
    }
  }

  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;
  std::string desired_class_id_;
  std::string track_id_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detection2_d_array_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2D>::SharedPtr filtered_detection2_d_pub_;
};

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolov8::Detection2DArrayVLMFilter)
