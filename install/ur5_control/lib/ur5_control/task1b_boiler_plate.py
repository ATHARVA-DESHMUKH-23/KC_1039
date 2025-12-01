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
SHOW_IMAGE = True
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

        # INSTRUCTIONS & HELP :
        #   -> Use `data` variable to convert ROS Image message to CV2 Image type
        #   -> HINT: You may use CvBridge to do the same
        #   -> Store the converted image into `self.cv_image`
        #   -> Check if you need any rotation or flipping of the image 
        #      (as input data may be oriented differently than expected).
        #      You may use cv2 functions such as `cv2.flip` or `cv2.rotate`.

        ############################################


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
            (0, h//3.2),            # top-left corner
            (w//3, h//3.2),         # mid-top
            (w//3, h//1.8),         # mid-bottom
            (0, h//1.8)             # bottom-left
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
        lower_bad = np.array([0, 0, 80])
        upper_bad = np.array([180, 60, 255])
        mask = cv2.inRange(hsv, lower_bad, upper_bad)

        # 4. Restrict mask to ROI
        mask = cv2.bitwise_and(mask, mask, mask=roi_mask)

        # 5. Morphological clean‐up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 6. Find contours & process only within ROI
        # it traces the outlines
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fruit_id = 1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (1000 < area < 5000):
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
        '''
        Description: Timer-driven loop for periodic image processing.
        Returns: None
        '''
        if self.cv_image is None or self.depth_image is None:
            return
            
        # Camera intrinsics (from /camera_info)
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 642.724365234375
        centerCamY = 361.9780578613281
        focalX = 915.3003540039062
        focalY = 914.0320434570312

        bad_fruits = self.bad_fruit_detection(self.cv_image)

        import tf2_ros
        from geometry_msgs.msg import TransformStamped
        import math
        if not hasattr(self, 'tf_broadcaster'):
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        team_id = 1039  

        # ---- Camera fixed transform relative to base ----
        CAMERA_TO_BASE = {
            'x': 0.18,   # camera is 35 cm forward of base
            'y': 0.007,    # centered
            'z': 1.09,   # 40 cm above base
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

        # Rotation matrix for RPY (in case you mount camera tilted later)
        Rx = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll),  math.cos(roll)]])
        Ry = np.array([[ math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw),  math.cos(yaw), 0],
                    [0, 0, 1]])
        R = Rz @ Ry @ Rx  # Combined rotation

        for fruit in bad_fruits:
            cX, cY = fruit['center']
            distance = fruit['distance']

            # 1️⃣ Convert pixel + depth -> camera frame coordinates
            x_cam = float(distance * (cX - centerCamX) / focalX)
            y_cam = float(distance * (cY - centerCamY) / focalY)
            z_cam = float(distance)

            # 2️⃣ Convert camera frame → robot’s frame orientation
            #    Your custom mapping (camera: Z forward, X right, Y down)
            x_r = z_cam
            y_r = -x_cam
            z_r = -y_cam

            # 3️⃣ Apply fixed transform (camera → base)
            cam_point = np.array([[x_r], [y_r], [z_r]])
            base_point = R @ cam_point + np.array([[dx], [dy], [dz]])
            x_base, y_base, z_base = base_point.flatten()

            #just transformaing tf to match 
            import tf_transformations

            # roll (X), pitch (Y), yaw (Z)
            roll = np.pi      # flip Z down
            pitch = 0
            yaw = -np.pi/2     # rotate XY 90 deg clockwise

            q = tf_transformations.quaternion_from_euler(roll, pitch, yaw)


            # 4️⃣ Publish TF in base_link frame
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'base_link'
            t.child_frame_id = f"{team_id}_bad_fruit_{fruit['id']}"
            t.transform.translation.x = x_base-1.22
            t.transform.translation.y = y_base
            t.transform.translation.z = z_base
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.tf_broadcaster.sendTransform(t)

            # Draw bounding boxes
            if SHOW_IMAGE:
                # Unpack the stable box
                x_box, y_box, w_box, h_box = fruit['bbox']

                # Draw a green rectangle
                cv2.rectangle(self.cv_image,
                            (x_box, y_box),
                            (x_box + w_box, y_box + h_box),
                            (0, 255, 0), 2)

                # Draw a filled green dot at the center
                cX, cY = fruit['center']
                cv2.circle(self.cv_image, (cX, cY), 4, (0, 255, 0), -1)

                # Label
                cv2.putText(self.cv_image, "bad_fruit",
                            (x_box, y_box - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        # Show image (debug)
        if SHOW_IMAGE:
            try:
                display_img = cv2.resize(self.cv_image, (640, 360))
                cv2.imshow('fruits_tf_view', display_img)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().error(f"OpenCV display error: {e}")

                

       
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