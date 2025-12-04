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

# Team ID:          [ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:		    bonus_task2.py
# Functions:
#			        [ Comma separated list of functions in this file ]
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/aligned_depth_to_color/image_raw, /etc... ]


################### IMPORT MODULES #######################

import rclpy
import sys
import cv2
import math
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage, Image


##################### FUNCTION DEFINITIONS #######################

def calculate_rectangle_area(coordinates):
    '''
    Description:    Function to calculate area or detected aruco

    Args:
        coordinates (list):     coordinates of detected aruco (4 set of (x,y) coordinates)

    Returns:
        area        (float):    area of detected aruco
        width       (float):    width of detected aruco
    '''

    ############ Function VARIABLES ############

    # You can remove these variables after reading the instructions. These are just for sample.


    ############ ADD YOUR CODE HERE ############
    # Convert to NumPy array
    pts = np.array(coordinates, dtype=np.float32)

    # Calculate polygon area
    area = cv2.contourArea(pts)

    # Calculate side lengths
    width1 = np.linalg.norm(pts[0] - pts[1])
    width2 = np.linalg.norm(pts[2] - pts[3])
    height1 = np.linalg.norm(pts[1] - pts[2])
    height2 = np.linalg.norm(pts[3] - pts[0])

    # Average width and height
    width = (width1 + width2 + height1 + height2) / 4.0
    ############################################

    return area, width


def detect_aruco(image):
    '''
    Description:    Function to perform aruco detection and return each detail of aruco detected 
                    such as marker ID, distance, angle, width, center point location, etc.

    Args:
        image                   (Image):    Input image frame received from respective camera topic

    Returns:
        center_aruco_list       (list):     Center points of all aruco markers detected
        distance_from_rgb_list  (list):     Distance value of each aruco markers detected from RGB camera
        angle_aruco_list        (list):     Angle of all pose estimated for aruco marker
        width_aruco_list        (list):     Width of all detected aruco markers
        ids                     (list):     List of all aruco marker IDs detected in a single frame 
    '''

    ############ Function VARIABLES ############

    # ->  You can remove these variables if needed. These are just for suggestions to let you get started

    # Use this variable as a threshold value to detect aruco markers of certain size.
    # Ex: avoid markers/boxes placed far away from arm's reach position  
    aruco_area_threshold = 1500

    # The camera matrix is defined as per camera info loaded from the plugin used. 
    # You may get this from /camer_info topic when camera is spawned in gazebo.
    # Make sure you verify this matrix once if there are calibration issues.
    cam_mat = np.array([[915.3003540039062, 0.0,642.724365234375], [0.0, 914.0320434570312, 361.9780578613281], [0.0, 0.0, 1.0]])

    # The distortion matrix is currently set to 0. 
    # We will be using it
    # during Stage 2 hardware as Intel Realsense Camera provides these camera info.
    dist_mat = np.array([0.0,0.0,0.0,0.0,0.0])

    # We are using 150x150 aruco marker size
    size_of_aruco_m = 0.13

    # You can remove these variables after reading the instructions. These are just for sample.
    center_aruco_list = []
    distance_from_rgb_list = []
    angle_aruco_list = []
    width_aruco_list = []
    ids = []
 
    ############ ADD YOUR CODE HERE ############


    # INSTRUCTIONS & HELP : 

    #	->  Convert input BGR image to GRAYSCALE for aruco detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #   ->  Use these aruco parameters-
    #       ->  Dictionary: 4x4_50 (4x4 only until 50 aruco IDs)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    #   ->  Detect aruco marker in the image and store 'corners' and 'ids'
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    #       ->  HINT: Handle cases for empty markers detection. 
    corners, ids, rejected = detector.detectMarkers(gray)
    center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list = [], [], [], []

    #   ->  Draw detected marker on the image frame which will be shown later

    #   ->  Loop over each marker ID detected in frame and calculate area using function defined above (calculate_rectangle_area(coordinates))

    #   ->  Remove tags which are far away from arm's reach positon based on some threshold defined

    #   ->  Calculate center points aruco list using math and distance from RGB camera using pose estimation of aruco marker
    #       ->  HINT: You may use numpy for center points and 'estimatePoseSingleMarkers' from cv2 aruco library for pose estimation
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for i, corner in enumerate(corners):
            pts = corner[0]
            (area, width) = calculate_rectangle_area(pts)
            if area < aruco_area_threshold:  # threshold
                continue

            cX = int(np.mean(pts[:, 0]))
            cY = int(np.mean(pts[:, 1]))
            center_aruco_list.append((cX, cY))
            width_aruco_list.append(width)

            # Pose estimation
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, size_of_aruco_m, cam_mat, dist_mat)
            # print("tvec shape:", tvec)
            # print("rvec shape:", rvec)
            distance = np.linalg.norm(tvec[0][0])
            distance_from_rgb_list.append(distance)
            angle_aruco_list.append(rvec[0][0][2]) 
            cv2.drawFrameAxes(image, cam_mat, dist_mat, rvec, tvec, 0.1)
    
    #   ->  Draw frame axes from coordinates received using pose estimation
    #       ->  HINT: You may use 'cv2.drawFrameAxes'

    ############################################

    return center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids


##################### CLASS DEFINITION #######################

class aruco_tf(Node):
    '''
    ___CLASS___

    Description:    Class which servers purpose to define process for detecting aruco marker and publishing tf on pose estimated.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class aruco_tf
                        All classes have a function called __init__(), which is always executed when the class is being initiated.
                        The __init__() function is called automatically every time the class is being used to create a new object.
                        You can find more on this topic here -> https://www.w3schools.com/python/python_classes.asp
        '''

        super().__init__('aruco_tf_publisher')                                          # registering node

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(Image, '/camera/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depthimagecb, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        image_processing_rate = 0.2                                                     # rate of time to process image (seconds)
        self.bridge = CvBridge()                                                        # initialise CvBridge object for image conversion
        self.tf_buffer = tf2_ros.buffer.Buffer()                                        # buffer time used for listening transforms
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)                                    # object as transform broadcaster to send transform wrt some frame_id
        self.timer = self.create_timer(image_processing_rate, self.process_image)       # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)
        
        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None                                                         # depth image variable (from depthimagecb())


    def depthimagecb(self, data):
        '''
        Description:    Callback function for aligned depth camera topic. 
                        Use this function to receive image depth data and convert to CV2 image

        Args:
            data (Image):    Input depth image frame received from aligned depth camera topic

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data,desired_encoding="passthrough")
            
        except CvBridgeError as e:
            self.get_logge().error(f"Depth image conversion failed: {e}")
        

        ############################################


    def colorimagecb(self, data):
        '''
        Description:    Callback function for colour camera raw topic.
                        Use this function to receive raw image data and convert to CV2 image

        Args:
            data (Image):    Input coloured raw image frame received from image_raw camera topic

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
            # If image is mirrored, uncomment:
            # cv_image = cv2.flip(cv_image, 1)
            self.cv_image = cv_image        
        except CvBridgeError as e:
            self.get_logger().error(f"Color image conversion failed: {e}")

        ############################################


    def process_image(self):
        '''
        Description:    Timer function used to detect aruco markers and publish tf on estimated poses.

        Args:
        Returns:
        '''

        ############ Function VARIABLES ############

        # These are the variables defined from camera info topic such as image pixel size, focalX, focalY, etc.
        # Make sure you verify these variable values once. As it may affect your result.
        # You can find more on these variables here -> http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
        
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 642.724365234375
        centerCamY = 361.9780578613281
        focalX = 915.3003540039062
        focalY = 914.0320434570312
            

        ############ ADD YOUR CODE HERE ############
        if self.cv_image is None or self.depth_image is None:
            return

        centers, distances, angles, widths, ids = detect_aruco(self.cv_image)
        if ids is None:
            return
        # INSTRUCTIONS & HELP : 

        #	->  Get aruco center, distance from rgb, angle, width and ids list from 'detect_aruco_center' defined above
        
        #   ->  Loop over detected box ids received to calculate position and orientation transform to publish TF 
        for i, marker_id in enumerate(ids.flatten()):
            cX, cY = centers[i]
            distance = distances[i]
            angle = angles[i]

        #   ->  Use this equation to correct the input aruco angle received from cv2 aruco function 'estimatePoseSingleMarkers' here
        #       It's a correction formula- 
        #       angle_aruco = (0.788*angle_aruco) - ((angle_aruco**2)/3160)
            angle = (0.788 * angle) - ((angle ** 2) / 3160)

        #   ->  Then calculate quaternions from roll pitch yaw (where, roll and pitch are 0 while yaw is corrected aruco_angle)
            r = R.from_euler('zyx', [angle, 0, np.deg2rad(315)])
            quat = r.as_quat()  # x, y, z, w
        #   ->  Use center_aruco_list to get realsense depth and log them down. (divide by 1000 to convert mm to m)
            
        #   ->  Use this formula to rectify x, y, z based on focal length, center value and size of image
        #       x = distance_from_rgb * (sizeCamX - cX - centerCamX) / focalX
        #       y = distance_from_rgb * (sizeCamY - cY - centerCamY) / focalY
        #       z = distance_from_rgb
        #       where, 
        #               cX, and cY from 'center_aruco_list'
        #               distance_from_rgb is depth of object calculated in previous step
        #               sizeCamX, sizeCamY, centerCamX, centerCamY, focalX and focalY are defined above
            x = distance * (sizeCamX - cX - centerCamX) / focalX
            y = distance * (sizeCamY - cY - centerCamY) / focalY
            z = distance

        #   ->  Now, mark the center points on image frame using cX and cY variables with help of 'cv2.cirle' function 
            cv2.circle(self.cv_image, (cX, cY), 5, (0, 0, 255), -1)

            #naming according to Aruco Ids
            if(marker_id==3):
                names="fertilizer_1"
            elif(marker_id==6):
                names="cart"
            else: names = "obj_"+str(marker_id)
        #   ->  Here, till now you receive coordinates from camera_link to aruco marker center position. 
        #       So, publish this transform w.r.t. camera_link using Geometry Message - TransformStamped 
        #       so that we will collect it's position w.r.t base_link in next step.
        #       Use the following frame_id-
        #           frame_id = 'camera_link'
        #           child_frame_id = 'cam_<marker_id>'          Ex: cam_20, where 20 is aruco marker ID
            t=TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id ="camera_link"
            t.child_frame_id = f"1039_{names}"
            
            if(marker_id==3):
                t.transform.translation.x = z-0.15
                t.transform.translation.y = x+0.14
                t.transform.translation.z = y-0.05
            else:
                t.transform.translation.x = z
                t.transform.translation.y = x
                t.transform.translation.z = y

            if marker_id == 6:
                # Multiply by a 180° rotation around Z
                flip_r = R.from_euler('x', np.pi)
                r_flipped = flip_r * r
                quat = r_flipped.as_quat()

            t.transform.rotation.x = quat[2]
            t.transform.rotation.y = quat[0]
            t.transform.rotation.z = quat[1]
            t.transform.rotation.w = quat[3]
            self.br.sendTransform(t)
        #   ->  Then finally lookup transform between base_link and obj frame to publish the TF
        #       You may use 'lookup_transform' function to pose of obj frame w.r.t base_link 
        #   ->  And now publish TF between object frame and base_link
        #       Use the following frame_id-
        #           frame_id = 'base_link'
        #           child_frame_id = 'obj_<marker_id>'          Ex: obj_20, where 20 is aruco marker ID
        
            try:
                trans = self.tf_buffer.lookup_transform('base_link', f'1039_{names}', rclpy.time.Time())
                base_t = TransformStamped()
                base_t.header.stamp = self.get_clock().now().to_msg()
                base_t.header.frame_id = 'base_link'
                base_t.child_frame_id = f'1039_{names}'
                base_t.transform = trans.transform
                self.br.sendTransform(base_t)
            except Exception as e:
                self.get_logger().error(f"Failed to lookup or broadcast transform for marker {marker_id}: {e}")

        #   ->  At last show cv2 image window having detected markers drawn and center points located using 'cv2.imshow' function.
        #       Refer MD book on portal for sample image -> https://portal.e-yantra.org/

        #   ->  NOTE:   The Z axis of TF should be pointing inside the box (Purpose of this will be known in task 1C)
        #               Also, auto eval script will be judging angular difference as well. So, make sure that Z axis is inside the box (Refer sample images on Portal - MD book)
        # cv2.imshow("Aruco Detection", self.cv_image)
        cv2.waitKey(1)

        ############################################


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spin around for the aruco_tf class to perform it's task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('aruco_tf_process')                    # creating ROS node

    node.get_logger().info('Node created: Aruco tf process')        # logging information

    aruco_tf_class = aruco_tf()                                     # creating a new object for class 'aruco_tf'

    rclpy.spin(aruco_tf_class)                                      # spining on the object to make it alive in ROS 2 DDS

    aruco_tf_class.destroy_node()                                   # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':
    '''
    Description:    If the python interpreter is running that module (the source file) as the main program, 
                    it sets the special __name__ variable to have a value “__main__”. 
                    If this file is being imported from another module, __name__ will be set to the module’s name.
                    You can find more on this here -> https://www.geeksforgeeks.org/what-does-the-if-__name__-__main__-do/
    '''

    main()


