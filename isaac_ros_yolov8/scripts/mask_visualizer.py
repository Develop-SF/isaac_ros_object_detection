#!/usr/bin/env python3
'''
Filename: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/scripts/mask_visualizer.py
Path: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/scripts
Description: Visualizer node for YOLOv8 object detection results with mask overlay
Created Date: Monday, November 10th 2025, 10:33:59 am
Author: Wen-Yu Chien

Copyright (c) 2025 Copyright (c) 2025 Shinfang Global
'''

import os 
import cv2
import cv_bridge
import message_filters
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class MaskVisualizer(Node):
    def __init__(self):
        super().__init__('mask_visualizer')
        self._bridge = cv_bridge.CvBridge()
        self._processed_image_pub = self.create_publisher(
            Image, 'invert_mask', 10)

        self._masks_subscription = message_filters.Subscriber(
            self,
            Image,
            'yolo_segmentation')
        self._image_subscription = message_filters.Subscriber(
            self,
            Image,
            '/color_image_resized')

        self.time_synchronizer = message_filters.TimeSynchronizer(
            [self._masks_subscription, self._image_subscription],
            10)

        self.time_synchronizer.registerCallback(self.masks_callback)

    def masks_callback(self, masks_msg, img_msg):
        cv2_img = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        cv2_masks = self._bridge.imgmsg_to_cv2(masks_msg, desired_encoding='mono8')

        # draw the pixels from image where mask is zero
        cv2_img[cv2_masks == 0] = 0
        processed_img_msg = self._bridge.cv2_to_imgmsg(cv2_img, encoding='bgr8')
        self._processed_image_pub.publish(processed_img_msg)

def main(args=None):
    rclpy.init(args=args)
    mask_visualizer = MaskVisualizer()
    rclpy.spin(mask_visualizer)
    mask_visualizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
