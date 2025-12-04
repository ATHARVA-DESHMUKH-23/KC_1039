#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		    Krishi coBot (KC) Theme (eYRC 2025-26)
*        		===============================================
*
*  This script should be used to implement Task 1B of Krishi coBot (KC) Theme (eYRC 2025-26).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          [ 1039 ]
# Author List:		[Atharva Deshmukh, Sumit Shelwane, Rajvardhan Deshmukh]
# Filename:		    task1b_boiler_plate.py
# Functions:
#			        [ Comma separated list of functions in this file ]
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/aligned_depth_to_color/image_raw, /etc... ]



import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
import cv2
import numpy as np


# runtime parameters
SHOW_IMAGE = False
DISABLE_MULTITHREADING = False

class FruitsTF(Node):
    """
    ROS2 Boilerplate for fruit detection and TF publishing.
    Students should implement detection logic inside the sections.
    """

    def __init__(self):
        super().__init__('fruits_tf')
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None

        # callback group handling
        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        # Subscriptions - use actual topics seen on your system
        self.create_subscription(Image, '/camera/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depthimagecb, 10, callback_group=self.cb_group)

        # Timer for periodic processing
        self.create_timer(0.2, self.process_image, callback_group=self.cb_group)

        # 🔹 New: debug image publisher
        self.debug_img_pub = self.create_publisher(Image, '/fruits/debug_image', 10)

        if SHOW_IMAGE:
            cv2.namedWindow('fruits_tf_view', cv2.WINDOW_NORMAL)

        self.get_logger().info("FruitsTF boilerplate node started.")

    # ---------------- Callbacks ----------------
    def depthimagecb(self, data):
        """
        Depth callback: convert to CV image and store shape & dtype info for debugging.
        """
        try:
            # convert (keep passthrough so we can inspect dtype)
            depth_cv = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            
            self.depth_image = depth_cv
        except Exception as e:
            self.get_logger().error(f"Depth conversion error: {e}")



    def colorimagecb(self, data):
        """
        Color callback: convert to BGR8 and store; log shape once.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
            
            self.cv_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Color conversion error: {e}")

        ############ ADD YOUR CODE HERE ############



    def bad_fruit_detection(self, rgb_image):
        '''
        Detect bad (greyish-white) fruits in the image frame,
        but only within the left‐side tray region.
        Returns a list of dicts with center, distance, angle, width, and id.
        '''
        bad_fruits = []
        if rgb_image is None:
            return bad_fruits

        # 1. Define ROI polygon for the tray on the left side
        h, w = rgb_image.shape[:2]
        # Adjust these points to match your tray location in pixels
        tray_roi = np.array([[
            (0, 0),            # top-left corner
            (w//2, 0),         # mid-top
            (w//2, h),         # mid-bottom
            (0, h)             # bottom-left
        ]], dtype=np.int32)

        # Create blank mask then fill ROI polygon
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, tray_roi, 255)

        #just to see selected area of tray
        # if SHOW_IMAGE:
        #     overlay = rgb_image.copy()
        #     color_mask = cv2.merge([roi_mask, np.zeros_like(roi_mask), np.zeros_like(roi_mask)])  # red overlay
        #     debug_view = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
        #     cv2.imshow("Tray ROI Debug View", cv2.resize(debug_view, (640, 360)))
        #     cv2.waitKey(1)

        # 2. Preprocess & convert to HSV
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        # 3. Apply your bad‐fruit HSV threshold
        lower_gray = np.array([0, 0, 80])
        upper_gray = np.array([180, 60, 255])

        lower_green = np.array([35, 60, 60])
        upper_green = np.array([85, 255, 255])
        
        ## 3.5 indivaual masking 
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # 4. Restrict both mask to ROI
        gray_mask = cv2.bitwise_and(gray_mask, gray_mask, mask=roi_mask)
        green_mask = cv2.bitwise_and(green_mask, green_mask, mask=roi_mask)
        
        ## 4.5 Combine — fruit valid only if green is near gray
        kernel = np.ones((15, 15), np.uint8)
        green_dilated = cv2.dilate(green_mask, kernel, iterations=1)
        combined_mask = cv2.bitwise_and(gray_mask, green_dilated)

        # 5. Morphological clean‐up
        kernel_small = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_small)

        # 6. Find contours & process only within ROI
        # it traces the outlines
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fruit_id = 1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (500 < area < 10000):
                continue

            x, y, w_box, h_box = cv2.boundingRect(cnt)
            cX = x + w_box // 2
            cY = y + h_box // 2

            # Verify center lies inside ROI mask
            if roi_mask[cY, cX] == 0:
                continue

            # Distance calculation (as before)...
            distance = 0.5
            if self.depth_image is not None:
                dh, dw = self.depth_image.shape[:2]
                if 0 <= cX < dw and 0 <= cY < dh:
                    depth_val = float(self.depth_image[cY, cX])
                    depth_m = depth_val * 0.001 if self.depth_image.dtype == np.uint16 else (depth_val * 0.001 if depth_val > 10.0 else depth_val)
                    if depth_m > 0 and not np.isnan(depth_m):
                        distance = depth_m

            fruit_info = {
                'center': (cX, cY),
                'distance': distance,
                'angle': 0.0,
                'bbox': (x, y, w_box, h_box),
                'id': fruit_id
            }
            bad_fruits.append(fruit_info)
            fruit_id += 1

            # Visualize only within ROI
            # if SHOW_IMAGE:
            #     cv2.rectangle(rgb_image, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            #     cv2.putText(rgb_image, "bad fruit", (x, y - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return bad_fruits

        
    def process_image(self):
        """
        Timer-driven loop for periodic image processing.
        """
        if self.cv_image is None or self.depth_image is None:
            return

        # Camera intrinsics (from /camera_info)
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 642.724365234375
        centerCamY = 361.9780578613281
        focalX = 915.3003540039062
        focalY = 914.0320434570312

        # ---- 1) Detection on a clean copy ----
        rgb_for_detection = self.cv_image.copy()
        bad_fruits = self.bad_fruit_detection(rgb_for_detection)

        # ---- 2) TF publishing ----
        import tf2_ros
        from geometry_msgs.msg import TransformStamped
        import math
        import tf_transformations

        if not hasattr(self, 'tf_broadcaster'):
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        team_id = 1039

        # Camera fixed transform relative to base
        CAMERA_TO_BASE = {
            'x': 0.18,
            'y': 0.007,
            'z': 1.09,
            'roll': 0.0,
            'pitch': math.radians(42.0),
            'yaw': 0.0
        }

        dx = CAMERA_TO_BASE['x']
        dy = CAMERA_TO_BASE['y']
        dz = CAMERA_TO_BASE['z']
        roll = CAMERA_TO_BASE['roll']
        pitch = CAMERA_TO_BASE['pitch']
        yaw = CAMERA_TO_BASE['yaw']

        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll),  math.cos(roll)]
        ])
        Ry = np.array([
            [ math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw),  math.cos(yaw), 0],
            [0, 0, 1]
        ])
        R = Rz @ Ry @ Rx

        # Orientation of fruit frames (kept same as your code)
        fruit_roll = np.pi       # flip Z
        fruit_pitch = 0.0
        fruit_yaw = -np.pi / 2.0
        q = tf_transformations.quaternion_from_euler(fruit_roll, fruit_pitch, fruit_yaw)

        for fruit in bad_fruits:
            cX, cY = fruit['center']
            distance = fruit['distance']

            # pixel + depth -> camera frame
            x_cam = float(distance * (cX - centerCamX) / focalX)
            y_cam = float(distance * (cY - centerCamY) / focalY)
            z_cam = float(distance)

            # camera -> robot mapping
            x_r = z_cam
            y_r = -x_cam
            z_r = -y_cam

            cam_point = np.array([[x_r], [y_r], [z_r]])
            base_point = R @ cam_point + np.array([[dx], [dy], [dz]])
            x_base, y_base, z_base = base_point.flatten()

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'base_link'
            t.child_frame_id = f"{team_id}_bad_fruit_{fruit['id']}"

            # ⚠️ remove -1.22 first; tune later if needed
            t.transform.translation.x = x_base-1.22
            t.transform.translation.y = y_base
            t.transform.translation.z = z_base
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            self.tf_broadcaster.sendTransform(t)

        # ---- 3) Build visualization image for RViz / debug ----
        try:
            vis_img = self.cv_image.copy()

            # AOI polygon (tray ROI)
            h, w = vis_img.shape[:2]
            tray_roi = np.array([[
                (0, 0),
                (w // 2, 0),
                (w // 2, h),
                (0, h)
            ]], dtype=np.int32)

            overlay = vis_img.copy()
            cv2.polylines(overlay, tray_roi, isClosed=True, color=(0, 0, 255), thickness=2)
            alpha = 0.3
            vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)

            # Draw fruit boxes & centers on vis_img
            for fruit in bad_fruits:
                x_box, y_box, w_box, h_box = fruit['bbox']
                cX, cY = fruit['center']

                cv2.rectangle(vis_img, (x_box, y_box),
                            (x_box + w_box, y_box + h_box),
                            (0, 255, 0), 2)
                cv2.circle(vis_img, (cX, cY), 4, (0, 255, 0), -1)
                cv2.putText(vis_img, f"bad_fruit_{fruit['id']}",
                            (x_box, y_box - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

            # 🔹 Publish debug image as ROS topic
            debug_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
            self.debug_img_pub.publish(debug_msg)

            # Optional local OpenCV window
            if SHOW_IMAGE:
                display_img = cv2.resize(vis_img, (640, 360))
                cv2.imshow('fruits_tf_view', display_img)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Debug image error: {e}")

                    

       
def main(args=None):
    rclpy.init(args=args)
    node = FruitsTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down FruitsTF")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()