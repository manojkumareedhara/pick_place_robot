#!/usr/bin/env python3
"""
PANDA SIDE-GRASP PICK & PLACE (FULL) - for cuboid/box using your CubeDetector topics.

Requires topics (from your detector):
  /detected_object_position     geometry_msgs/PointStamped   (centroid) in panda_link0
  /detected_object_dimensions   geometry_msgs/PointStamped   (width, depth, height) in panda_link0
  /detected_object_boundary     geometry_msgs/PolygonStamped (footprint poly) in panda_link0
  /joint_states                 sensor_msgs/JointState

Core robustness:
- Perception updates ONLY in HOME
- Lock centroid + dims + boundary for ONE attempt
- If fail: backoff -> HOME -> reacquire fresh detections
- Side grasp:
    * Compute box yaw via PCA on boundary points
    * Pick a side normal and approach from outside in XY
    * Close at mid-height (centroid.z)
    * Verify grasp by finger closure
    * Lift and place
"""

import math
import time
from collections import deque
from enum import Enum

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Point, PointStamped, PolygonStamped
from sensor_msgs.msg import JointState

from pymoveit2 import MoveIt2, GripperInterface
from pymoveit2.robots import panda


class RobotState(Enum):
    HOME = 0
    LOCKED = 1
    PICKING = 2
    PLACING = 3


def quaternion_from_euler(roll: float, pitch: float, yaw: float):
    """
    Convert roll/pitch/yaw to quaternion (x,y,z,w).
    """
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qx, qy, qz, qw]


class PandaSidePickPlace(Node):
    # ---------- perception stability ----------
    STABLE_FRAMES_REQUIRED = 6
    POS_STD_THRESH = 0.003  # m

    # ---------- grasp geometry ----------
    # How far outside the object we start (approach point)
    APPROACH_DIST = 0.14     # m (horizontal approach)
    # How far inside from the face we stop before closing (small)
    FINAL_INSET = 0.008      # m
    # Height of side grasp = centroid.z (you can offset)
    GRASP_Z_OFFSET = 0.0

    # ---------- motion ----------
    VEL_FAST = 0.15
    ACC_FAST = 0.15
    VEL_SLOW = 0.05
    ACC_SLOW = 0.05
    VEL_LIFT_VERIFY = 0.03
    ACC_LIFT_VERIFY = 0.03
    MIN_ABSOLUTE_Z = 0.05

    # Lift heights
    LIFT_VERIFY_DZ = 0.06
    LIFT_DZ = 0.15

    # Close dwell
    CLOSE_DWELL_SEC = 0.30

    # ---------- grasp success ----------
    EMPTY_GRASP_THRESHOLD = 0.004

    # ---------- retries ----------
    MAX_ATTEMPTS_PER_OBJECT = 6
    BACKOFF_Z = 0.10
    REACQUIRE_COOLDOWN_SEC = 0.25

    # ---------- place target ----------
    PLACE_X = 0.55
    PLACE_Y = -0.15
    PLACE_Z = 0.20
    PLACE_APPROACH_Z = 0.18
    PLACE_DESCENT_STEPS = 8

    # ---------- side grasp EE orientation ----------
    # This is the BIG tuning knob if side grasp “misses”.
    # These Euler angles define the gripper’s baseline orientation; yaw is added based on box yaw.
    #
    # Common Panda setups:
    #   - roll=pi, pitch=+pi/2 often makes gripper approach sideways with fingers vertical-ish.
    # If it approaches with the wrong face, try pitch=-pi/2 or roll=0.
    SIDE_GRASP_ROLL = math.pi
    SIDE_GRASP_PITCH = math.pi / 2

    def __init__(self):
        super().__init__("panda_side_pick_place")

        self.cb_group = ReentrantCallbackGroup()
        self.state = RobotState.HOME

        # --- live perception buffers (HOME only) ---
        self.centroid_buf = deque(maxlen=self.STABLE_FRAMES_REQUIRED)
        self.dims_buf = deque(maxlen=self.STABLE_FRAMES_REQUIRED)
        self.boundary_msg = None

        # --- locked perception (one attempt) ---
        self.locked_centroid = None  # PointStamped
        self.locked_dims = None      # PointStamped (width, depth, height)
        self.locked_boundary = None  # PolygonStamped

        # attempt tracking
        self.attempts = 0
        self.reacquire_until_time = 0.0

        # finger joints feedback
        self.finger = {"panda_finger_joint1": None, "panda_finger_joint2": None}

        # MoveIt2
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=panda.joint_names(),
            base_link_name=panda.base_link_name(),
            end_effector_name=panda.end_effector_name(),
            group_name=panda.MOVE_GROUP_ARM,
            callback_group=self.cb_group,
        )
        self.moveit2.max_velocity = self.VEL_FAST
        self.moveit2.max_acceleration = self.ACC_FAST

        self.gripper = GripperInterface(
            node=self,
            gripper_joint_names=panda.gripper_joint_names(),
            open_gripper_joint_positions=panda.OPEN_GRIPPER_JOINT_POSITIONS,
            closed_gripper_joint_positions=panda.CLOSED_GRIPPER_JOINT_POSITIONS,
            gripper_group_name=panda.MOVE_GROUP_GRIPPER,
            callback_group=self.cb_group,
            gripper_command_action_name="gripper_action_controller/gripper_cmd",
        )

        # Subscribers (your detector topics)
        self.create_subscription(PointStamped, "/detected_object_position", self.centroid_cb, 10)
        self.create_subscription(PointStamped, "/detected_object_dimensions", self.dims_cb, 10)
        self.create_subscription(PolygonStamped, "/detected_object_boundary", self.boundary_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_cb, 50)

        # Home joints
        self.home_joints = [
            0.0,
            math.radians(-45.0),
            0.0,
            math.radians(-135.0),
            0.0,
            math.radians(90.0),
            math.radians(45.0),
        ]

        self.go_home()
        self.get_logger().info("Side-grasp pick&place ready. Waiting for stable detection in HOME...")

        self.create_timer(0.1, self.step, callback_group=self.cb_group)

    # ---------------- callbacks ----------------
    def centroid_cb(self, msg: PointStamped):
        if self.state != RobotState.HOME:
            return
        if time.time() < self.reacquire_until_time:
            return
        self.centroid_buf.append((msg.point.x, msg.point.y, msg.point.z, msg.header.stamp))

    def dims_cb(self, msg: PointStamped):
        if self.state != RobotState.HOME:
            return
        if time.time() < self.reacquire_until_time:
            return
        # width=x, depth=y, height=z
        self.dims_buf.append((msg.point.x, msg.point.y, msg.point.z, msg.header.stamp))

    def boundary_cb(self, msg: PolygonStamped):
        if self.state != RobotState.HOME:
            return
        if time.time() < self.reacquire_until_time:
            return
        if msg.polygon.points:
            self.boundary_msg = msg

    def joint_cb(self, msg: JointState):
        idx = {n: i for i, n in enumerate(msg.name)}
        for j in self.finger:
            if j in idx:
                self.finger[j] = msg.position[idx[j]]

    # ---------------- basic motion ----------------
    def go_home(self):
        self.moveit2.max_velocity = self.VEL_FAST
        self.moveit2.max_acceleration = self.ACC_FAST
        self.moveit2.move_to_configuration(self.home_joints)
        self.moveit2.wait_until_executed()
        self.state = RobotState.HOME

    def open_gripper(self):
        self.gripper.open()
        self.gripper.wait_until_executed()

    def close_gripper(self):
        self.gripper.close()
        self.gripper.wait_until_executed()

    def move_pose(self, x, y, z, quat_xyzw, fast=True):
        if fast:
            self.moveit2.max_velocity = self.VEL_FAST
            self.moveit2.max_acceleration = self.ACC_FAST
        else:
            self.moveit2.max_velocity = self.VEL_SLOW
            self.moveit2.max_acceleration = self.ACC_SLOW
        self.moveit2.move_to_pose(position=[float(x), float(y), float(z)], quat_xyzw=quat_xyzw)
        self.moveit2.wait_until_executed()

    def get_ee_pos(self):
        fk = self.moveit2.compute_fk()
        if fk is None:
            return None
        return fk.pose.position

    def backoff_vertical(self):
        ee = self.get_ee_pos()
        if ee is None:
            return
        self.move_pose(ee.x, ee.y, ee.z + self.BACKOFF_Z, quat_xyzw=[0.0, 0.0, 0.0, 1.0], fast=True)

    # ---------------- perception lock ----------------
    def stable(self) -> bool:
        if len(self.centroid_buf) < self.STABLE_FRAMES_REQUIRED:
            return False
        if len(self.dims_buf) < self.STABLE_FRAMES_REQUIRED:
            return False
        if self.boundary_msg is None:
            return False

        c = np.array([[x, y, z] for (x, y, z, _) in self.centroid_buf], dtype=np.float32)
        d = np.array([[x, y, z] for (x, y, z, _) in self.dims_buf], dtype=np.float32)

        # centroid stability
        if np.max(np.std(c, axis=0)) > self.POS_STD_THRESH:
            return False

        # dims stability (looser)
        if np.max(np.std(d, axis=0)) > 0.01:
            return False

        return True

    def lock(self):
        c = np.mean(np.array([[x, y, z] for (x, y, z, _) in self.centroid_buf], dtype=np.float32), axis=0)
        d = np.mean(np.array([[x, y, z] for (x, y, z, _) in self.dims_buf], dtype=np.float32), axis=0)

        self.locked_centroid = PointStamped()
        self.locked_centroid.header.frame_id = self.boundary_msg.header.frame_id
        self.locked_centroid.point.x = float(c[0])
        self.locked_centroid.point.y = float(c[1])
        self.locked_centroid.point.z = float(c[2])

        self.locked_dims = PointStamped()
        self.locked_dims.header.frame_id = self.boundary_msg.header.frame_id
        self.locked_dims.point.x = float(d[0])  # width
        self.locked_dims.point.y = float(d[1])  # depth
        self.locked_dims.point.z = float(d[2])  # height

        self.locked_boundary = self.boundary_msg

        self.centroid_buf.clear()
        self.dims_buf.clear()
        self.boundary_msg = None

        self.state = RobotState.LOCKED
        self.attempts += 1

        self.get_logger().info(
            f"LOCKED attempt {self.attempts}/{self.MAX_ATTEMPTS_PER_OBJECT} "
            f"cen=({self.locked_centroid.point.x:.3f},{self.locked_centroid.point.y:.3f},{self.locked_centroid.point.z:.3f}) "
            f"dims=({self.locked_dims.point.x:.3f},{self.locked_dims.point.y:.3f},{self.locked_dims.point.z:.3f})"
        )

    def clear_lock_and_buffers(self):
        self.locked_centroid = None
        self.locked_dims = None
        self.locked_boundary = None
        self.centroid_buf.clear()
        self.dims_buf.clear()
        self.boundary_msg = None

    # ---------------- box yaw from boundary (PCA) ----------------
    def compute_box_yaw(self):
        b = self.locked_boundary
        if b is None or not b.polygon.points or len(b.polygon.points) < 3:
            return None

        pts = np.array([(p.x, p.y) for p in b.polygon.points], dtype=np.float32)
        mean = np.mean(pts, axis=0)
        centered = pts - mean

        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        major = eigvecs[:, int(np.argmax(eigvals))]
        yaw = math.atan2(float(major[1]), float(major[0]))
        return yaw

    def side_grasp_quat(self, yaw):
        # yaw is box yaw; we align gripper yaw with box yaw (or +/- 90 depending on your setup)
        # Here we use yaw directly; if fingers are “wrong direction”, try yaw + pi/2.
        return quaternion_from_euler(self.SIDE_GRASP_ROLL, self.SIDE_GRASP_PITCH, yaw)

    # ---------------- grasp checks ----------------
    def grasp_succeeded(self) -> bool:
        j1 = self.finger["panda_finger_joint1"]
        j2 = self.finger["panda_finger_joint2"]
        if j1 is None or j2 is None:
            self.get_logger().warn("No finger feedback -> assuming success")
            return True
        avg = 0.5 * (abs(j1) + abs(j2))
        self.get_logger().info(f"Finger avg: {avg:.4f}")
        return avg > self.EMPTY_GRASP_THRESHOLD

    # ---------------- side grasp plan ----------------
    def attempt_side_grasp_once(self) -> bool:
        """
        Try side grasp from BOTH sides of the box.
        """
        if self.locked_centroid is None or self.locked_dims is None or self.locked_boundary is None:
            return False

        yaw = self.compute_box_yaw()
        if yaw is None:
            self.get_logger().warn("No yaw from boundary -> cannot side grasp")
            return False

        quat = self.side_grasp_quat(yaw)

        cen = self.locked_centroid.point
        dims = self.locked_dims.point

        # choose half of the smaller footprint dimension as approach offset
        half_short = 0.5 * min(float(dims.x), float(dims.y))
        side_offset = max(0.0, half_short - self.FINAL_INSET)

        # normal direction for “a face” (use box yaw)
        nx = math.cos(yaw)
        ny = math.sin(yaw)

        grasp_z = float(cen.z) + self.GRASP_Z_OFFSET
        grasp_z = max(grasp_z, self.MIN_ABSOLUTE_Z)

        # Try both faces (+normal and -normal)
        for sign in (+1.0, -1.0):
            tx = float(cen.x) + sign * side_offset * nx
            ty = float(cen.y) + sign * side_offset * ny

            # Approach point is further out along same normal
            ax = tx + sign * self.APPROACH_DIST * nx
            ay = ty + sign * self.APPROACH_DIST * ny

            self.get_logger().info(
                f"Side grasp try sign={int(sign)} yaw={yaw:.3f} "
                f"approach=({ax:.3f},{ay:.3f},{grasp_z:.3f}) target=({tx:.3f},{ty:.3f},{grasp_z:.3f})"
            )

            # Execute:
            self.state = RobotState.PICKING
            self.open_gripper()
            time.sleep(0.05)

            # 1) go to approach point (fast)
            self.move_pose(ax, ay, grasp_z, quat, fast=True)

            # 2) slide in (slow)
            self.move_pose(tx, ty, grasp_z, quat, fast=False)

            # 3) close and dwell
            self.close_gripper()
            time.sleep(self.CLOSE_DWELL_SEC)

            if not self.grasp_succeeded():
                self.get_logger().warn("Empty grasp -> backoff and try other side")
                # back off outward a bit (reverse the approach)
                self.move_pose(ax, ay, grasp_z, quat, fast=True)
                continue

            # 4) lift verify slowly
            self.moveit2.max_velocity = self.VEL_LIFT_VERIFY
            self.moveit2.max_acceleration = self.ACC_LIFT_VERIFY
            self.moveit2.move_to_pose(
                position=[tx, ty, grasp_z + self.LIFT_VERIFY_DZ],
                quat_xyzw=quat
            )
            self.moveit2.wait_until_executed()

            if not self.grasp_succeeded():
                self.get_logger().warn("Lost grasp during verify -> backoff")
                self.move_pose(ax, ay, grasp_z + self.LIFT_VERIFY_DZ, quat, fast=True)
                continue

            # 5) full lift
            self.move_pose(tx, ty, grasp_z + self.LIFT_DZ, quat, fast=True)
            return True

        return False

    # ---------------- place routine ----------------
    def place(self):
        self.state = RobotState.PLACING

        px, py, pz = self.PLACE_X, self.PLACE_Y, self.PLACE_Z
        above_z = pz + self.PLACE_APPROACH_Z

        # keep same quat as last (doesn’t matter much for place)
        # Use a neutral yaw to avoid weird wrist flips
        quat = quaternion_from_euler(self.SIDE_GRASP_ROLL, self.SIDE_GRASP_PITCH, 0.0)

        # go above place
        self.move_pose(px, py, above_z, quat, fast=True)

        # descend in steps
        self.moveit2.max_velocity = self.VEL_SLOW
        self.moveit2.max_acceleration = self.ACC_SLOW
        for i in range(self.PLACE_DESCENT_STEPS):
            t = i / (self.PLACE_DESCENT_STEPS - 1) if self.PLACE_DESCENT_STEPS > 1 else 1.0
            z = above_z * (1.0 - t) + pz * t
            self.moveit2.move_to_pose(position=[px, py, float(z)], quat_xyzw=quat)
            self.moveit2.wait_until_executed()
            time.sleep(0.02)

        self.open_gripper()
        time.sleep(0.10)

        # retreat
        self.move_pose(px, py, above_z, quat, fast=True)

    # ---------------- reacquire ----------------
    def reacquire(self, reason: str):
        self.get_logger().warn(f"Reacquire: {reason}")
        # backoff up
        self.backoff_vertical()
        # clear everything & go home
        self.clear_lock_and_buffers()
        self.go_home()
        # small delay so buffers refill with new data
        self.reacquire_until_time = time.time() + self.REACQUIRE_COOLDOWN_SEC

    # ---------------- main step ----------------
    def step(self):
        if self.state != RobotState.HOME:
            return

        if self.attempts >= self.MAX_ATTEMPTS_PER_OBJECT:
            self.get_logger().error("Max attempts reached. Resetting and waiting for fresh stable detection.")
            self.attempts = 0
            self.clear_lock_and_buffers()
            self.reacquire_until_time = time.time() + self.REACQUIRE_COOLDOWN_SEC
            return

        if time.time() < self.reacquire_until_time:
            return

        if not self.stable():
            return

        # lock for ONE attempt
        self.lock()

        ok = self.attempt_side_grasp_once()
        if not ok:
            self.reacquire("side grasp failed")
            return

        # place and reset
        self.place()
        self.get_logger().info("Pick-and-place SUCCESS. Returning HOME.")
        self.attempts = 0
        self.clear_lock_and_buffers()
        self.go_home()
        self.reacquire_until_time = time.time() + self.REACQUIRE_COOLDOWN_SEC


def main():
    rclpy.init()
    node = PandaSidePickPlace()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
