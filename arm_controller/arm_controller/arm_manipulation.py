#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np
import time

# TF imports
import tf_transformations
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64MultiArray

# magnet attach/detach imports
from linkattacher_msgs.srv import AttachLink, DetachLink

def quat_conjugate(q):
    # q = [x,y,z,w]
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_mul(q1, q2):
    # q1,q2 = [x,y,z,w]
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    x =  w1*x2 + x1*w2 + y1*z2 - z1*y2
    y =  w1*y2 - x1*z2 + y1*w2 + z1*x2
    z =  w1*z2 + x1*y2 - y1*x2 + z1*w2
    w =  w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.array([x,y,z,w])

def quat_to_axis_angle(q):
    # q = [x,y,z,w] unit quaternion
    w = q[3]
    # numeric clamp
    w = np.clip(w, -1.0, 1.0)
    theta = 2.0 * np.arccos(w)
    s = np.sqrt(max(0.0, 1 - w*w))
    if s < 1e-6:
        # angle ~ 0, axis arbitrary (use vector part)
        return np.array([0.0, 0.0, 0.0]), 0.0
    axis = q[0:3] / s
    return axis, theta


class CartesianPoseFollower(Node):
    def __init__(self):
        super().__init__('cartesian_pose_follower')
        self.pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(Float64MultiArray, '/delta_joint_cmds', 10)
        
        # Magnet control service clients
        self.attach_cli = self.create_client(AttachLink, '/attach_link')
        self.detach_cli = self.create_client(DetachLink, '/detach_link')

        # Wait for services to be ready
        while not self.attach_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('/attach_link service not available, waiting...')
        while not self.detach_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('/detach_link service not available, waiting...')

        # TF buffer & listener to read ee pose relative to base
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer_period = 0.05  # 20 Hz control loop
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        # Gains (tune)
        self.Kp_pos = 2.0    # (m/s per m of error)
        self.Kp_rot = 1.5    # (rad/s per rad of orientation error)

        # Limits
        self.v_max = 0.2     # m/s
        self.w_max = 0.8     # rad/s

        # Termination thresholds
        self.pos_tol = 0.007  # meters
        self.ang_tol = 0.03   # radians (~1.7 deg)

        # Stages:
        # waiting_for_pick -> picking -> lifting -> placing -> (after fertiliser) fruit_sequence
        # fruit_sequence stages: goto_tray -> wait_for_fruit -> goto_fruit -> attach -> lift_fruit -> goto_dustbin -> detach -> return_tray
        self.stage = "waiting_for_pick"

        # frames (team id prefix included in TFs you publish)
        self.pick_frame = '1039_fertiliser_can'
        self.place_frame = '1039_cart'
        self.fruit_tf_name = '1039_bad_fruit_1'  # always target this; upstream detection reindexes dynamically

        # Fixed known points (tray and dustbin) as provided
        #(np.array([0.1,  0.501, 0.415]), np.array([0.029, 0.997, 0.045,0.033])),
        self.tray_pos = np.array([0.1,  0.501, 0.415])
        self.tray_quat = np.array([0.029, 0.997, 0.045, 0.033])
        self.dustbin_pos = np.array([-0.806,  0.010, 0.382])
        self.dustbin_quat = np.array([-0.684, 0.726, 0.05, 0.008])

        self.lift_height_after_pick = 0.10  # 10 cm lift after picking fruit

        self.curr_goal_idx = 0
        self.last_goal_time = time.time()
        self.max_goal_time = 100.0  # safety timeout per goal (s)

        self.base_frame = 'base_link'
        self.ee_frame = 'tool0'

        # fruit sequence helpers
        self.fruit_wait_cycles = 0
        self.fruit_wait_limit = int(0.2 / self.timer_period)  # number of cycles to wait before checking absence (5s)
        self.fruit_missing_confirm_cycles = int(1.0 / self.timer_period)  # cycles of "no TF" before concluding empty
        self.fruit_missing_counter = 0

        self.current_goal = None

        # saved cart pose for reliable placement (set just after a successful attach)
        self.saved_cart_pose = None  # tuple (pos(np.array), quat(np.array), timestamp)
        self.wait_cart_retry = 6     # how many attempts to check for cart TF (0.05s per control loop; adjust)
        self.wait_cart_sleep = 0.2   # seconds between retries when actively waiting
        self.scan_on_miss = True     # enable small scan to help reacquire marker


        self.get_logger().info('CartesianPoseFollower started (20 Hz).')
    
    
    def wait_for_and_save_cart(self, max_total_time=5.0):
        """
        Try to read cart TF repeatedly up to max_total_time seconds.
        On success store to self.saved_cart_pose = (pos, quat, time.time()) and return True.
        Optionally do a small base scan if repeated misses occur.
        """
        start = time.time()
        attempts = 0
        while time.time() - start < max_total_time:
            pos, quat = self.get_goal_from_tf(self.place_frame, max_age_sec=1.0)
            if pos is not None:
                self.saved_cart_pose = (pos.copy(), quat.copy(), time.time())
                self.get_logger().info(f"Saved cart pose for placement (age 0s).")
                return True

            attempts += 1
            # After a few attempts, optionally do a tiny scan to assist reacquire
            if self.scan_on_miss and attempts % 4 == 0:
                self.get_logger().info("Cart TF not seen — performing small scan to reacquire.")
                try:
                    self.rotate_base_joint(angle_deg=8.0, step_size=2.0)
                    time.sleep(0.05)
                    self.rotate_base_joint(angle_deg=-8.0, step_size=2.0)
                except Exception as e:
                    self.get_logger().warn(f"Scan failed: {e}")

            time.sleep(self.wait_cart_sleep)

        self.get_logger().warn(f"Failed to find cart TF within {max_total_time:.1f}s.")
        return False

    # -------------------------------------------------------
    # Helper: get goal points tf
    # -------------------------------------------------------
    def get_goal_from_tf(self, frame_name, max_age_sec=1.0):
        try:
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                self.base_frame, frame_name, rclpy.time.Time())

            # ✅ Check if TF is too old
            now = self.get_clock().now()
            tf_time = rclpy.time.Time.from_msg(trans.header.stamp)
            age = (now - tf_time).nanoseconds * 1e-9

            if age > max_age_sec:
                self.get_logger().warn(f"{frame_name} TF is stale (age={age:.2f}s), ignoring.")
                return None, None

            t = trans.transform.translation
            q = trans.transform.rotation
            pos = np.array([t.x, t.y, t.z])
            quat = np.array([q.x, q.y, q.z, q.w])

            # override orientation for fruit TF
            if frame_name == '1039_bad_fruit_1':
                quat = np.array([0.029, 0.997, 0.045, 0.033])

            return pos, quat
        except Exception:
            return None, None
        
        
    # -------------------------------------------------------
    # Helper: rotate base joint
    # -------------------------------------------------------
    def rotate_base_joint(self, angle_deg=90.0, step_size=1.0):
        self.get_logger().info(f'Rotating base joint by {angle_deg} degrees (anticlockwise)...')
        delta = Float64MultiArray()
        sign = 1.0 if angle_deg > 0 else -1.0
        steps = int(abs(angle_deg) / step_size)
        delta_step = [sign * 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]  # only base joint

        for _ in range(steps):
            delta.data = delta_step
            self.joint_pub.publish(delta)
            time.sleep(0.05)  # smooth movement
        self.get_logger().info('Base rotation complete.')

    # -------------------------------------------------------
    # TF Pose Reader
    # -------------------------------------------------------
    def read_current_pose(self):
        try:
            trans: TransformStamped = self.tf_buffer.lookup_transform(self.base_frame, self.ee_frame, rclpy.time.Time())
        except Exception:
            return None, None

        t = trans.transform.translation
        q = trans.transform.rotation
        pos = np.array([t.x, t.y, t.z])
        quat = np.array([q.x, q.y, q.z, q.w])
        return pos, quat
    # -------------------------------------------------------
    # Magnet control helpers
    # -------------------------------------------------------
    def attach_object(self, object_name: str):
        req = AttachLink.Request()
        req.model1_name = object_name
        req.link1_name = 'body'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'

        self.get_logger().info(f"Attaching object: {object_name}")
        future = self.attach_cli.call_async(req)
        # do not block; short sleep after caller will verify

    def detach_object(self, object_name: str):
        req = DetachLink.Request()
        req.model1_name = object_name
        req.link1_name = 'body'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'

        self.get_logger().info(f"Detaching object: {object_name}")
        future = self.detach_cli.call_async(req)
        # do not block; short sleep after caller will verify

    # -------------------------------------------------------
    # Helpers to set goals
    # -------------------------------------------------------
    def set_goal(self, pos: np.ndarray, quat: np.ndarray):
        self.current_goal = (pos, quat)
        self.last_goal_time = time.time()

    # -------------------------------------------------------
    # Main control loop
    # -------------------------------------------------------
    def control_loop(self):
        # shutdown if done
        if self.stage == "done":
            self.stop_and_finish()
            return

        # -------------------
        # Stage logic and dynamic goal selection
        # -------------------
        # (A) Fertiliser handling (existing flow) ----------------
        if self.stage == "waiting_for_pick":
            goal_pos, goal_quat = self.get_goal_from_tf(self.pick_frame)
            if goal_pos is None:
                self.get_logger().info("Waiting for can position TF...")
                return
            self.stage = "picking"
            self.get_logger().info("Can TF found — moving to pick position.")
            self.set_goal(goal_pos, goal_quat)
            return

        elif self.stage == "picking":
            # current_goal already set
            pass

        elif self.stage == "lifting":
            # current_goal already set
            pass

        elif self.stage == "placing":
            # current_goal already set
            pass

        # (B) Fruit sequence ----------------
        elif self.stage == "fruit_sequence":
            # sub-state flow managed by sub-stage in self.sub_stage
            # initialize sub-stage if not present
            if not hasattr(self, 'sub_stage'):
                self.sub_stage = 'goto_tray'
                self.get_logger().info('Entering fruit_sequence: goto_tray')
                self.set_goal(self.tray_pos, self.tray_quat)
                # reset counters
                self.fruit_missing_counter = 0
                self.fruit_wait_cycles = 0
                return

            # goto_tray: move to tray safe point before searching
            if self.sub_stage == 'goto_tray':
                # goal already set to tray
                pass

            # wait_for_fruit: once at tray, wait a short time and then check for fruit TF
            elif self.sub_stage == 'wait_for_fruit':
                # stay at tray for a small stable period then check TF
                self.fruit_wait_cycles += 1
                if self.fruit_wait_cycles < self.fruit_wait_limit:
                    # keep waiting
                    return
                # then check TF
                fruit_pos, fruit_quat = self.get_goal_from_tf(self.fruit_tf_name)
                if fruit_pos is None:
                    self.fruit_missing_counter += 1
                    self.get_logger().info(f"No fruit TF ({self.fruit_missing_counter}/{self.fruit_missing_confirm_cycles}) - still waiting...")
                    if self.fruit_missing_counter >= self.fruit_missing_confirm_cycles:
                        self.get_logger().info("No more fruits detected. Fruit clearing complete.")
                        self.stage = "done"
                        return
                    else:
                        # wait more cycles; stay in wait_for_fruit
                        self.fruit_wait_cycles = 0
                        return
                else:
                    # fruit detected, proceed to go to fruit
                    fruit_pos = fruit_pos + np.array([0.0, 0.0, 0.03])
                    self.get_logger().info(f"Fruit TF found — moving to offset pick position ({fruit_pos}).")
                    
                    self.set_goal(fruit_pos, fruit_quat)
                    self.get_logger().info(f"current goal ({self.current_goal}).")
                    self.sub_stage = 'goto_fruit'
                    return

            elif self.sub_stage == 'goto_fruit':
                # approachable: goal is already fruit pos
                pass

            elif self.sub_stage == 'attach':
                # after reaching fruit, will attach and then set lift goal
                pass

            elif self.sub_stage == 'lift_fruit':
                # lift goal already set
                pass

            elif self.sub_stage == 'goto_dustbin':
                # goal already set to dustbin
                pass

            elif self.sub_stage == 'detach':
                # detach then set return_tray
                pass

            elif self.sub_stage == 'return_tray':
                # goal already set to tray
                pass

        else:
            # unknown stage
            self.get_logger().warn(f'Unknown stage: {self.stage}')
            return

        # -------------------
        # Read current ee pose and compute control commands
        # -------------------
        curr_pos, curr_quat = self.read_current_pose()
        if curr_pos is None:
            self.get_logger().warn('TF for end-effector not available yet; waiting for transforms.')
            return

        goal_pos, goal_quat = self.current_goal
        e_pos = goal_pos - curr_pos
        dist = np.linalg.norm(e_pos)

        q_curr_conj = quat_conjugate(curr_quat)
        q_err = quat_mul(goal_quat, q_curr_conj)
        axis, angle = quat_to_axis_angle(q_err)
        e_ang = axis * angle

        # -------------------
        # Check if goal reached
        # -------------------
        if dist < self.pos_tol and np.linalg.norm(e_ang) < self.ang_tol:
            # Fertiliser flow: existing states
            if self.stage in ["picking", "lifting", "placing"]:
                if self.stage == "picking":
                    self.get_logger().info("Reached can — checking distance for pickup.")
                    if dist < 0.10:
                        self.attach_object("fertiliser_can")
                        time.sleep(1.0)

                        # --- WAIT FOR CART TF and SAVE IT BEFORE LIFTING ---
                        got_cart = self.wait_for_and_save_cart(max_total_time=5.0)
                        if not got_cart:
                            # decide fallback behavior:
                            # 1) skip waiting and proceed to place using cached cart pose if any
                            # 2) or give up and return to tray / safe point.
                            if self.saved_cart_pose is not None:
                                self.get_logger().warn("Using previously saved cart pose (older) for placement.")
                            else:
                                self.get_logger().warn("No cart pose available — returning to tray and will retry later.")
                                # go to tray safe point and continue fruit sequence or finish fertiliser flow
                                # prefer saved cart pose if available and recent
                                if self.saved_cart_pose is not None and (time.time() - self.saved_cart_pose[2]) < 10.0:
                                    drop_pos, drop_quat = self.saved_cart_pose[0], self.saved_cart_pose[1]
                                    self.get_logger().info("Using saved cart pose for placing.")
                                else:
                                    drop_pos, drop_quat = self.get_goal_from_tf(self.place_frame, max_age_sec=15.0)
                                    if drop_pos is None:
                                        self.get_logger().info("Waiting for cart TF...")
                                        return

                                if drop_pos is None:
                                    # if even this fails, fallback: return to tray and mark as waiting
                                    self.stage = "placing"
                                    self.set_goal(self.tray_pos, self.tray_quat)
                                    return
                                else:
                                    self.saved_cart_pose = (drop_pos, drop_quat, time.time())

                        # Now we have either freshly saved pose or an older saved_cart_pose.
                        # Proceed to lift in current tool pose and then later in placing use saved_cart_pose.
                        curr_pos, curr_quat = self.read_current_pose()
                        if curr_pos is not None:
                            lift_vector = np.array([0.0, 0.0, -0.2])
                            R = tf_transformations.quaternion_matrix(curr_quat)[:3, :3]
                            lift_in_base = R @ lift_vector
                            lifted_pos = curr_pos + lift_in_base
                            self.set_goal(lifted_pos, curr_quat)
                            self.stage = "lifting"
                            return
                    else:
                        self.get_logger().warn("Not close enough to pick fertiliser (<0.1m). Skipping attach.")
                        # fallthrough to place at cart anyway


                elif self.stage == "lifting":
                    self.get_logger().info("Lift complete — moving to cart position.")
                    drop_pos, drop_quat = self.get_goal_from_tf(self.place_frame,max_age_sec=15.0)
                    if drop_pos is None:
                        self.get_logger().info("Waiting for cart TF...")
                        return
                    drop_pos = drop_pos + np.array([0.0, 0.0, 0.20])
                    self.set_goal(drop_pos, drop_quat)
                    self.stage = "placing"
                    return

                elif self.stage == "placing":
                    self.get_logger().info("Reached drop position — releasing fertiliser.")
                    self.detach_object("fertiliser_can")
                    time.sleep(1.0)
                    # After fertiliser placement, switch to fruit sequence
                    self.get_logger().info("Fertiliser placed — switching to fruit sequence.")
                    self.stage = "fruit_sequence"
                    # initialize sub_stage and move to tray
                    self.sub_stage = 'goto_tray'
                    self.set_goal(self.tray_pos, self.tray_quat)
                    # reset counters
                    self.fruit_missing_counter = 0
                    self.fruit_wait_cycles = 0
                    return

            # Fruit sequence reached a sub-goal
            if self.stage == "fruit_sequence":
                if self.sub_stage == 'goto_tray':
                    self.get_logger().info("Reached tray safe point — waiting for fruit TF.")
                    
                    self.sub_stage = 'wait_for_fruit'
                    self.fruit_wait_cycles = 0
                    return

                elif self.sub_stage == 'wait_for_fruit':
                    # Should not happen: wait_for_fruit handles TF queries before we set goal
                    return

                elif self.sub_stage == 'goto_fruit':
                    self.get_logger().info("Reached fruit position — attempting attach.")
                    # ensure close enough
                    if dist < 0.10:
                        self.attach_object("bad_fruit")
                        time.sleep(1.0)
                        # after attach, lift by configured height in base frame
                        curr_pos, curr_quat = self.read_current_pose()
                        if curr_pos is not None:
                            # lift in tool frame negative z, rotated into base
                            lift_vector = np.array([0.0, 0.0, -self.lift_height_after_pick])
                            R = tf_transformations.quaternion_matrix(curr_quat)[:3, :3]
                            lift_in_base = R @ lift_vector
                            lifted_pos = curr_pos + lift_in_base
                            self.set_goal(lifted_pos, curr_quat)
                            self.sub_stage = 'lift_fruit'
                            self.get_logger().info(f"Lifted fruit to {lifted_pos}")
                            return
                        else:
                            self.get_logger().warn("Couldn't read EE pose after attach — proceeding to dustbin anyway.")
                    else:
                        self.get_logger().warn("Not close enough to attach fruit (<0.1m). Skipping attach and returning to tray.")
                        # return to tray 
                        self.sub_stage = 'return_tray'
                        self.set_goal(self.tray_pos, self.tray_quat)
                        return

                elif self.sub_stage == 'lift_fruit':
                    # after lift, go directly to dustbin (no 20cm offset per user)
                    self.get_logger().info("Lift complete — moving to dustbin to drop fruit.")
                    self.sub_stage = 'goto_dustbin'
                    self.set_goal(self.dustbin_pos, self.dustbin_quat)
                    return

                elif self.sub_stage == 'goto_dustbin':
                    self.get_logger().info("Reached dustbin — releasing fruit.")
                    self.detach_object("bad_fruit")
                    time.sleep(1.0)
                    # After detach, return to tray to look for next fruit
                    self.sub_stage = 'return_tray'
                    self.set_goal(self.tray_pos, self.tray_quat)
                    # small reset to allow tf publisher to reindex; fruit detection should update TFs
                    self.fruit_wait_cycles = 0
                    return

                elif self.sub_stage == 'return_tray':
                    self.get_logger().info("Returned to tray — ready for next fruit.")
                    # go to wait_for_fruit to check for next fruit
                    self.sub_stage = 'wait_for_fruit'
                    self.fruit_wait_cycles = 0
                    return

        # -------------------
        # Normal control law (publish twist)
        # -------------------
        v_cmd = self.Kp_pos * e_pos
        w_cmd = self.Kp_rot * e_ang

        # Saturate
        if np.linalg.norm(v_cmd) > self.v_max and np.linalg.norm(v_cmd) > 0.0:
            v_cmd *= self.v_max / np.linalg.norm(v_cmd)
        if np.linalg.norm(w_cmd) > self.w_max and np.linalg.norm(w_cmd) > 0.0:
            w_cmd *= self.w_max / np.linalg.norm(w_cmd)

        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = v_cmd
        twist.angular.x, twist.angular.y, twist.angular.z = w_cmd
        self.pub.publish(twist)

    # -------------------------------------------------------
    # Stop node
    # -------------------------------------------------------
    def stop_and_finish(self):
        self.get_logger().info('All goals processed — sending final stop and shutting down.')
        self.pub.publish(Twist())
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CartesianPoseFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted, stopping.')
        node.pub.publish(Twist())
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
