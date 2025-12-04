#!/usr/bin/env python3
"""
Standalone eBot Task1A Navigator
- Navigate through 3 waypoints
- Avoid obstacles using LiDAR
- Faster and more robust than previous version
"""

# --- Required libraries ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math
import time
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point



# ===== Parameters we can tune =====
WAYPOINTS = [
    ( 0.26, -6.206, 0.012 ),
    (0.26, -1.95, 1.57 ),
    (-1.48, -0.67, -1.57),
    ( -1.53, -6.61, -1.57)
]

POS_TOLERANCE = 0.3         # meters
YAW_TOLERANCE = math.radians(10)  # radians (~10 deg)
CONTROL_HZ = 20.0           # faster control loop

KP_LINEAR = 1.0             # increase linear gain
KP_ANGULAR = 2.0            # increase angular gain
MAX_LINEAR_SPEED = 1.0      # faster max linear speed
MAX_ANGULAR_SPEED = 2.0     # faster max angular speed

OBSTACLE_FRONT_DIST = 0.55
FRONT_SECTOR_DEG = 30

STUCK_TIMEOUT = 3.0         # recovery if stuck for this long

# ================================

def normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

class EBotNavigator(Node):
    def __init__(self):
        super().__init__('ebot_nav_task1a')
        self.current_goal_idx = 0
        self.pose_x = None
        self.pose_y = None
        self.pose_yaw = None
        self.scan = None
        self.last_move_time = time.time()

        # ROS publishers/subscribers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_cb, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/ebot_debug', 10)


        self.timer = self.create_timer(1.0 / CONTROL_HZ, self.control_loop)

    def scan_cb(self, msg):
        self.scan = msg

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pose_x = p.x
        self.pose_y = p.y
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y**2 + q.z**2)
        self.pose_yaw = math.atan2(siny, cosy)

       # --- Helper functions ---
    def _min_distance_in_sector(self, start_angle, end_angle):
        if self.scan is None:
            return float('inf')
        angle = self.scan.angle_min
        min_r = float('inf')
        for r in self.scan.ranges:
            if math.isfinite(r) and self.scan.range_min <= r <= self.scan.range_max:
                if start_angle <= angle <= end_angle:
                    min_r = min(min_r, r)
            angle += self.scan.angle_increment
        return min_r

    def detect_front_obstacle(self):
        a = math.radians(FRONT_SECTOR_DEG)
        d = self._min_distance_in_sector(-a, a)
        return d, d < OBSTACLE_FRONT_DIST

    def sector_left_right(self):
        left = self._min_distance_in_sector(math.radians(30), math.radians(150))
        right = self._min_distance_in_sector(math.radians(-150), math.radians(-30))
        return left, right
    
    def _create_sector(self, frame, start_angle, end_angle, radius, r, g, b, mid):
        """Creates a colored sector marker (triangle fan). Returns a Marker."""
        m = Marker()
        m.header.frame_id = frame
        # use zero time to avoid TF extrapolation errors
        m.header.stamp = rclpy.time.Time().to_msg()
        m.ns = "sectors"
        m.id = mid
        # TRIANGLE_LIST draws triangles; points should be a multiple of 3
        m.type = Marker.TRIANGLE_LIST
        m.action = Marker.ADD

        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0

        m.color.a = 0.35
        m.color.r = r
        m.color.g = g
        m.color.b = b

        # center of robot
        cx = 0.0
        cy = 0.0

        # generate triangle fan in small steps
        pts = []
        step = math.radians(5)
        angle = start_angle
        # ensure we handle small negative/positive ranges correctly
        while angle < end_angle:
            a1 = angle
            a2 = min(end_angle, angle + step)

            x1 = radius * math.cos(a1)
            y1 = radius * math.sin(a1)
            x2 = radius * math.cos(a2)
            y2 = radius * math.sin(a2)

            # triangle: center, p1, p2
            pts.append(Point(x=cx, y=cy, z=0.01))
            pts.append(Point(x=x1, y=y1, z=0.01))
            pts.append(Point(x=x2, y=y2, z=0.01))

            angle += step

        m.points = pts
        return m

    def publish_debug_markers(self, gx, gy, start=None, end=None):
        """Publish a MarkerArray containing: waypoint (odom), heading arrow (ebot_base_link),
           front line, front sector, and optional clearance sector."""
        markers = MarkerArray()
        stamp = rclpy.time.Time().to_msg()  # single stamp reused

        # ---- 1. Target waypoint marker (in odom frame) ----
        wp = Marker()
        wp.header.frame_id = "odom"
        wp.header.stamp = stamp
        wp.ns = "waypoints"
        wp.id = 1
        wp.type = Marker.SPHERE
        wp.action = Marker.ADD
        wp.pose.position.x = gx
        wp.pose.position.y = gy
        wp.pose.position.z = 0.1
        wp.scale.x = 0.3
        wp.scale.y = 0.3
        wp.scale.z = 0.3
        wp.color.a = 1.0
        wp.color.r = 1.0
        wp.color.g = 0.2
        wp.color.b = 0.2
        markers.markers.append(wp)

        # ---- 2. Heading arrow (ebot_base_link frame) ----
        hd = Marker()
        hd.header.frame_id = "ebot_base_link"
        hd.header.stamp = stamp
        hd.ns = "heading"
        hd.id = 2
        hd.type = Marker.ARROW
        hd.action = Marker.ADD
        # arrow anchored at robot origin, pointing along +X of ebot_base_link
        hd.pose.position.x = 0.0
        hd.pose.position.y = 0.0
        hd.pose.position.z = 0.12
        hd.scale.x = 0.8   # length
        hd.scale.y = 0.08
        hd.scale.z = 0.08
        hd.color.a = 1.0
        hd.color.r = 0.1
        hd.color.g = 0.8
        hd.color.b = 0.1
        markers.markers.append(hd)

        # ---- 3. Front obstacle line ----
        front = Marker()
        front.header.frame_id = "ebot_base_link"
        front.header.stamp = stamp
        front.ns = "front_obstacle"
        front.id = 3
        front.type = Marker.LINE_STRIP
        front.action = Marker.ADD
        front.scale.x = 0.05
        front.color.a = 1.0
        front.color.r = 1.0
        front.color.g = 1.0
        front.color.b = 0.0
        front.points = [
            Point(x=0.0, y=0.0, z=0.12),
            Point(x=OBSTACLE_FRONT_DIST, y=0.0, z=0.12)
        ]
        markers.markers.append(front)

        # ---- 4. FRONT OBSTACLE SECTOR (red) ----
        front_sector = self._create_sector(
            frame="ebot_base_link",
            start_angle=-math.radians(FRONT_SECTOR_DEG),
            end_angle= math.radians(FRONT_SECTOR_DEG),
            radius=OBSTACLE_FRONT_DIST,
            r=1.0, g=0.0, b=0.0,
            mid=10
        )
        # re-use our stamp to avoid TF extrapolation
        front_sector.header.stamp = stamp
        markers.markers.append(front_sector)

        # ---- 5. Clearance sector toward goal (optional, yellow) ----
        if start is not None and end is not None:
            clearance_sector = self._create_sector(
                frame="ebot_base_link",
                start_angle=start,
                end_angle=end,
                radius=0.8,
                r=1.0, g=1.0, b=0.0,
                mid=11
            )
            clearance_sector.header.stamp = stamp
            markers.markers.append(clearance_sector)

        # publish once
        self.marker_pub.publish(markers)

    # --- Main control loop ---
    def control_loop(self):
        if self.pose_x is None or self.scan is None:
            return

        if self.current_goal_idx >= len(WAYPOINTS):
            self.publish_stop()
            return

        gx, gy, gyaw = WAYPOINTS[self.current_goal_idx]
        dx = gx - self.pose_x
        dy = gy - self.pose_y
        distance = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)
        angle_err = normalize_angle(angle_to_goal - self.pose_yaw)
        yaw_err = normalize_angle(gyaw - self.pose_yaw)

        front_min, obstacle_in_front = self.detect_front_obstacle()
        twist = Twist()

        clearance_start = None
        clearance_end = None

        if obstacle_in_front:
            left_min, right_min = self.sector_left_right()
            if left_min >= right_min:
                twist.angular.z = +2.0
            else:
                twist.angular.z = -2.0
            twist.linear.x = 0.0
            self.last_move_time = time.time()
        else:
            if distance > POS_TOLERANCE:
                # prioritize heading correction first
                if abs(angle_err) > 0.1:
                    # Check clearance in waypoint direction before turning
                    clearance_start = angle_err - math.radians(50)
                    clearance_end   = angle_err + math.radians(50)
                    sector_clear = self._min_distance_in_sector(clearance_start, clearance_end)

                    
                    if sector_clear > 0.7:
                        # safe to rotate toward goal
                        twist.linear.x = 0.0
                        twist.angular.z = max(-MAX_ANGULAR_SPEED,
                                              min(MAX_ANGULAR_SPEED, KP_ANGULAR * angle_err))
                    else:
                        # keep moving forward in row
                        twist.linear.x = min(MAX_LINEAR_SPEED, KP_LINEAR * 0.5)
                        twist.angular.z = 0.0
                else:
                    twist.linear.x = min(MAX_LINEAR_SPEED, KP_LINEAR * distance)
                    twist.angular.z = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, KP_ANGULAR * angle_err))
                self.last_move_time = time.time()
            else:
                # align final orientation
                if abs(yaw_err) > YAW_TOLERANCE:
                    twist.linear.x = 0.0
                    twist.angular.z = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, KP_ANGULAR * yaw_err))
                else:
                    self.publish_stop()
                    self.current_goal_idx += 1
                    time.sleep(0.3)
                    return

            # stuck recovery
            if time.time() - self.last_move_time > STUCK_TIMEOUT:
                twist.angular.z = 1.0
                twist.linear.x = 0.0
                self.last_move_time = time.time()
        
        self.publish_debug_markers(gx, gy, clearance_start, clearance_end)
        self.cmd_pub.publish(twist)

    def publish_stop(self):
        t = Twist()
        self.cmd_pub.publish(t)

def main(args=None):
    rclpy.init(args=args)
    node = EBotNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown requested")
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

