#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
import numpy as np
import cv2
import math
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class LiDARShapeDetector(Node):
    def __init__(self):
        super().__init__('lidar_shape_detector')

        # Subscribers
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.detection_pub = self.create_publisher(String, '/detection_status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/lidar_shapes', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.detect_marker_pub = self.create_publisher(Marker, '/detection_points', 10)
        self.det_marker_id = 0


        # Movement-based gating
        self.last_shape_pos = None   # (x, y) of last VALID detection
        self.distance_threshold = 0.8  # 0.5 m between detections

        # Square double-tap flag
        self.square_flag = False

        # Hold state (2s stop after detection)
        self.is_holding = False
        self.hold_until = None
        
        self.pending_action = None
        # durations (seconds) — tweak if needed
        self.move_duration = 5   # move forward for 1 second before publishing
        self.hold_duration = 2.0   # hold for 2 seconds after publishing
        self.move_speed = 0.50     # forward speed while moving (m/s)

        # Timer to enforce stop during hold
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("LiDAR Shape Detector Node Started")

    def publish_shape_markers(self, shapes, contours):
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        mid = 0

        cy_center = 300  # same as used in extract_shapes()

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

            # compute centroid
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Convert centroid from image canvas → lidar frame
            scale = 100.0
            lx = (cx - 300) / scale
            ly = -(cy - 300) / scale

            # Determine pos/neg based on your canvas logic
            contour_side = "pos" if cy < cy_center else "neg"

            # Pick correct shape for this contour
            assigned = "unknown"
            for name, side in shapes.items():
                if side == contour_side:
                    assigned = name
                    break

            # Transform LiDAR frame point → odom frame
            lx, ly = self.transform_point(lx, ly)
            if lx is None:
                continue

            # Create marker
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = stamp
            m.ns = "lidar_shapes"
            m.id = mid
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            m.pose.position.x = lx
            m.pose.position.y = ly
            m.pose.position.z = 0.1

            m.scale.x = 0.22
            m.scale.y = 0.22
            m.scale.z = 0.22

            m.color.a = 1.0

            # color coding
            if "square" in assigned:
                m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0  # yellow
            elif "triangle" in assigned:
                m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0  # green
            elif "pentagon" in assigned:
                m.color.r, m.color.g, m.color.b = 0.0, 0.4, 1.0  # blue
            elif "wall" in assigned:
                m.color.r, m.color.g, m.color.b = 1.0, 0.0, 0.0  # red
            else:
                m.color.r, m.color.g, m.color.b = 0.4, 0.4, 0.4  # gray

            markers.markers.append(m)
            mid += 1

        self.marker_pub.publish(markers)

    def transform_point(self, x, y):
        try:
            trans = self.tf_buffer.lookup_transform(
                'odom', 'ebot_base_link', rclpy.time.Time()
            )
            tx = trans.transform.translation.x
            ty = trans.transform.translation.y
            yaw = self.yaw_from_quaternion(trans.transform.rotation)

            # rotate + translate
            wx = tx + x * math.cos(yaw) - y * math.sin(yaw)
            wy = ty + x * math.sin(yaw) + y * math.cos(yaw)
            return wx, wy
        except:
            return None, None

    def yaw_from_quaternion(self, q):
        siny = 2.0 * (q.w*q.z + q.x*q.y)
        cosy = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        return math.atan2(siny, cosy)

    def publish_detection_point(self, status, x, y):
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "detection_points"
        m.id = self.det_marker_id
        self.det_marker_id += 1

        m.type = Marker.SPHERE
        m.action = Marker.ADD

        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.2

        m.scale.x = 0.35
        m.scale.y = 0.35
        m.scale.z = 0.35
        m.color.a = 1.0

        if status == "FERTILIZER_REQUIRED":
            m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0
        elif status == "BAD_HEALTH":
            m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0
        elif status == "DOCK_STATION":
            m.color.r, m.color.g, m.color.b = 0.0, 0.4, 1.0
        else:
            m.color.r, m.color.g, m.color.b = 1.0, 1.0, 1.0

        # IMPORTANT: publish Marker directly, not MarkerArray
        self.detect_marker_pub.publish(m)

    def delay_then(self, seconds, callback):
        # ROS2 Humble has no oneshot timer → create timer then cancel inside wrapper
        def wrapper():
            try:
                callback()
            finally:
                timer.cancel()   # cancel after first execution

        timer = self.create_timer(seconds, wrapper)

    # ---------------- TF & movement gating ----------------

    def get_robot_position(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'odom', 'ebot_base_link', rclpy.time.Time()
            )
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            return x, y
        except Exception as e:
            # self.get_logger().warn(f"TF lookup failed: {e}")
            return None, None

    def movement_allowed(self, shape_coords=None):
        if shape_coords is not None:
            sx, sy = shape_coords
            if self.last_shape_pos is None:
                self.last_shape_pos = (sx, sy)
                return True
            last_x, last_y = self.last_shape_pos
            dist = math.hypot(sx - last_x, sy - last_y)
            if dist >= self.distance_threshold:
                self.last_shape_pos = (sx, sy)
                return True
            return False

        # fallback to robot pose behavior:
        x, y = self.get_robot_position()
        if x is None:
            return False
        if self.last_shape_pos is None:
            self.last_shape_pos = (x, y)
            return True
        last_x, last_y = self.last_shape_pos
        if math.hypot((x - last_x), (y - last_y)) >= self.distance_threshold:
            self.last_shape_pos = (x, y)
            return True
        return False
    # ---------------- Core callback ----------------

    def lidar_callback(self, msg: LaserScan):
        # If we are currently holding after a detection → ignore scans
        if self.is_holding:
            return

        # Convert polar LiDAR data to Cartesian coordinates
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        # Remove invalid readings
        mask = np.isfinite(ranges)
        angles = angles[mask]
        ranges = ranges[mask]

        # Limit to side sectors only (your earlier logic)
        deg = np.degrees(angles)
        mask_deg = ((deg >= 60) & (deg <= 90)) | ((deg >= -90) & (deg <= -60))
        angles = angles[mask_deg]
        ranges = ranges[mask_deg]

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        points = np.vstack((x, y)).T

        # Detect shapes
        self.detect_shapes(points, msg.header)

    # ---------------- Shape extraction (contours) ----------------

    def extract_shapes(self, contours):
        """
        Returns dict: { "square": "pos", "triangle": "neg", "unknown0": "pos", ... }
        pos/neg based on vertical position in the 600x600 canvas.
        """
        shapes = {}

        cy_center = 300
        unknown_id = 0  # unique ids for unknowns

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80 or area > 30000:
                continue

            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            contour_mean_y = np.mean(cnt[:, :, 1])
            pos = "pos" if contour_mean_y < cy_center else "neg"

            # classify shape
            shape = "unknown"

            # wall
            if (len(approx) == 4 and 20 < w < 60 and h < 20 and 200 < y < 500):
                shape = "wall"

            # triangle (your tuned ranges)
            elif len(approx) == 6 and 45 < w < 55 and 30 < h < 40 and 650 < area < 750:
                shape = "triangle"

            # square
            elif len(approx) == 8 and 35 < w < 45 and h < 45:
                shape = "square"

            # pentagon
            elif len(approx) == 7:
                shape = "pentagon"

            # store
            if shape == "unknown":
                key = f"unknown{unknown_id}"
                unknown_id += 1
                shapes[key] = pos
            else:
                shapes[shape] = pos

        return shapes

    # ---------------- Drawing ----------------

    def draw_shapes(self, img, contours, shapes):
        cy_center = 300

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80 or area > 30000:
                continue

            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            contour_mean_y = np.mean(cnt[:, :, 1])
            pos = "pos" if contour_mean_y < cy_center else "neg"

            # best-effort mapping by side
            shape = None
            for name, side in shapes.items():
                if side == pos:
                    shape = name
                    break

            if shape is None:
                shape = "unknown"

            # Draw
            if shape.startswith("unknown"):
                cv2.drawContours(img, [approx], -1, 180, 1)
                cv2.putText(img, shape, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, 180, 1)
            else:
                cv2.drawContours(img, [approx], -1, 127, 2)
                cv2.putText(img, f"{shape} ({pos})", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

    # ---------------- Detection reporting + 2s stop ----------------

    def report_detection(self, status: str):
        """
        status: one of FERTILIZER_REQUIRED, BAD_HEALTH, DOCK_STATION
        1. Get robot (x, y)
        2. Publish to /detection_status: "Status,x,y"
        3. Start 2s hold with zero cmd_vel
        """
        x, y = self.get_robot_position()
        if x is None:
            # If TF failed, still publish with 0,0 to avoid missing detection
            x, y = 0.0, 0.0

        msg = String()
        msg.data = f"{status},{x:.2f},{y:.2f}"
        self.detection_pub.publish(msg)
        self.get_logger().info(f"Detection: {msg.data}")
        
        # publish visual marker in RViz
        self.publish_detection_point(status, x, y)
        
        # Start 2 second hold
        self.is_holding = True
        self.hold_until = self.get_clock().now() + Duration(seconds=2.0)

    def control_loop(self):
        """
        Runs at 20 Hz. Handles:
        - pending_action phases (move -> publish -> hold),
        - and the existing hold behavior which keeps robot stopped for 2s after publish.
        """
        now = self.get_clock().now()

        # If we're already in a 'hold' caused by a completed detection, keep robot stopped.
        if self.is_holding:
            twist = Twist()  # all zeros
            self.cmd_vel_pub.publish(twist)
            if now >= self.hold_until:
                self.is_holding = False
            return

        # If there is a pending action scheduled by detection, run its state machine
        if self.pending_action is not None:
            phase = self.pending_action.get('phase')

            if phase == 'move':
                # publish a forward velocity until deadline
                # send cmd_vel forward (in robot frame: linear.x positive)
                t = Twist()
                t.linear.x = self.move_speed
                t.angular.z = 0.0
                self.cmd_vel_pub.publish(t)

                # check if move time is done
                if now >= self.pending_action['deadline']:
                    # stop robot (publish zero immediately)
                    self.cmd_vel_pub.publish(Twist())
                    # next phase: publish & enter hold
                    self.pending_action['phase'] = 'publish'
                    # immediate publish will happen in this cycle (no additional delay)
                    # mark the publish time (use now)
                    self.pending_action['publish_time'] = now

            elif phase == 'publish':
                status = self.pending_action['status']
                stored = self.pending_action.get('shape_coords')
                if stored is not None:
                    sx, sy = stored
                else:
                    # fallback: best-effort TF now or robot pose
                    sx, sy = self.get_robot_position()
                    if sx is None:
                        self.get_logger().warning("No TF and no stored shape coords — publishing 0,0")
                        sx, sy = 0.0, 0.0

                msg = String()
                msg.data = f"{status},{sx:.2f},{sy:.2f}"
                self.detection_pub.publish(msg)
                self.get_logger().info(f"Detection: {msg.data}")
                self.publish_detection_point(status, sx, sy)

                self.is_holding = True
                self.hold_until = now + Duration(seconds=self.hold_duration)
                self.pending_action = None
                return
            
            else:
                # Unknown phase — clear to be safe
                self.get_logger().warn("Unknown pending_action phase, clearing")
                self.pending_action = None
                # ensure robot stopped
                self.cmd_vel_pub.publish(Twist())
                return

            # while moving/pending, do not process new scans (we allow lidar_callback to keep running;
            # control loop just manages velocities). Return to let loop iterate.
            return

        # If no pending_action and not in holding, nothing special to do here.
        # (lidar_callback still runs and will schedule a pending_action when it sees a valid detection.)
        return

    # ---------------- Shape validity logic (walls, opposite side, etc.) ----------------
    def schedule_delayed_publish(self, status, shape_coords=None):
        """
        Schedule the simple sequence:
        move forward for self.move_duration,
        then publish current odom + status,
        then hold for self.hold_duration.
        If another pending action exists, ignore new requests (simple suppression).
        """
        # do not overwrite an already scheduled action
        if self.pending_action is not None:
            self.get_logger().debug(f"Pending action present, ignoring new schedule for {status}")
            return

        now = self.get_clock().now()
        self.pending_action = {
            'status': status,
            'phase': 'move',
            'deadline': now + Duration(seconds=self.move_duration),
            'requested_at': now,
            'shape_coords': shape_coords
        }


    def validate_shapes(self, shapes, shape_world_positions=None):
        if shape_world_positions is None:
            shape_world_positions = {}
        # Need wall for all logic
        if "wall" not in shapes:
            self.square_flag = False
            return

        wall_side = shapes["wall"]

        # ==== TRIANGLE & SQUARE ====
        for name, side in shapes.items():
            if name == "wall":
                continue

            opposite = (side == "pos" and wall_side == "neg") or \
                       (side == "neg" and wall_side == "pos")

            if not opposite:
                continue

            # ---- DOUBLE-TAP logic for square ----
            if name == "square":
                if not self.square_flag:
                    # first detection → arm the flag
                    self.square_flag = True
                else:
                    # second detection → check movement + report BAD_HEALTH
                    if self.movement_allowed():
                        coords = shape_world_positions.get(name, None)
                        self.schedule_delayed_publish("BAD_HEALTH")
                        #self.delay_then(5.0, lambda: self.report_detection("BAD_HEALTH"))
                        
                    self.square_flag = False
                continue

            # ---- Triangle instantly valid ----
            if name == "triangle":
                if self.movement_allowed():
                    self.schedule_delayed_publish("FERTILIZER_REQUIRED")
                    #self.delay_then(2.0, lambda: self.report_detection("FERTILIZER_REQUIRED"))
                self.square_flag = False
                continue

        # ==== PENTAGON VALIDATION ====
        if "pentagon" in shapes:
            p_side = shapes["pentagon"]

            opposite_to_wall = (
                (p_side == "pos" and wall_side == "neg") or
                (p_side == "neg" and wall_side == "pos")
            )

            if opposite_to_wall:
                unknown_count = 0
                for n, s in shapes.items():
                    if "unknown" in n and s == p_side:
                        unknown_count += 1

                if unknown_count >= 2:
                    if self.movement_allowed():
                        self.schedule_delayed_publish("DOCK_STATION")
                        #self.delay_then(5.0, lambda: self.report_detection("DOCK_STATIONs"))

    # ---------------- High-level LiDAR → image → shapes ----------------

    def detect_shapes(self, points, header):
        scale = 100
        img = np.zeros((600, 600), dtype=np.uint8)
        cx, cy = 300, 300

        # Plot points into image
        for px, py in points:
            ix = int(cx + px * scale)
            iy = int(cy - py * scale)
            if 0 <= ix < 600 and 0 <= iy < 600:
                img[iy, ix] = 255

        # Preprocess
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=2)

        # Contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Classify shapes (unchanged)
        shapes = self.extract_shapes(contours)

        # For each contour compute centroid and transform to odom immediately.
        # We'll build a map: shape_name -> (x_odom, y_odom)
        shape_world_positions = {}

        cy_center = 300
        unknown_id = 0
        # We'll iterate again over contours and try to match by side/pos to the shapes dict.
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80 or area > 30000:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            ix = int(M['m10'] / M['m00'])
            iy = int(M['m01'] / M['m00'])
            contour_mean_y = np.mean(cnt[:, :, 1])
            pos = "pos" if contour_mean_y < cy_center else "neg"

            matched_is_square = False
            for name, side in shapes.items():
                if name == "square" and side == pos:
                    matched_is_square = True
                    break

            if not matched_is_square:
                continue  # we only care about square here
            px = (ix - cx) / scale
            py = (cy - iy) / scale  # sign flip


            p = PointStamped()
            p.header.stamp = header.stamp
            p.header.frame_id = header.frame_id if header.frame_id else 'ebot_base_link'
            p.point.x = px
            p.point.y = py
            p.point.z = 0.0

            try:
                p_odom = self.tf_buffer.transform(p, 'odom', timeout=Duration(seconds=0.1))

                shape_world_positions['square'] = (p_odom.point.x, p_odom.point.y)
            except Exception as e:
               
                self.get_logger().debug(f"TF transform failed for square centroid: {e}")
               

    
        self.draw_shapes(img, contours, shapes)


        self.validate_shapes(shapes, shape_world_positions)

        self.publish_shape_markers(shapes, contours)

def main(args=None):
    rclpy.init(args=args)
    node = LiDARShapeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()