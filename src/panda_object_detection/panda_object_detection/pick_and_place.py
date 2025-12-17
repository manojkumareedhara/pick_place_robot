#!/usr/bin/env python3


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
    STABLE_FRAMES_REQUIRED = 6
    POS_STD_THRESH = 0.003  

    APPROACH_DIST = 0.14    
    FINAL_INSET = 0.008     
    GRASP_Z_OFFSET = 0.0

    VEL_FAST = 0.15
    ACC_FAST = 0.15
    VEL_SLOW = 0.05
    ACC_SLOW = 0.05
    VEL_LIFT_VERIFY = 0.03
    ACC_LIFT_VERIFY = 0.03
    MIN_ABSOLUTE_Z = 0.05

    LIFT_VERIFY_DZ = 0.06
    LIFT_DZ = 0.15

    CLOSE_DWELL_SEC = 0.30

    EMPTY_GRASP_THRESHOLD = 0.004

    MAX_ATTEMPTS_PER_OBJECT = 6
    BACKOFF_Z = 0.10
    REACQUIRE_COOLDOWN_SEC = 0.25

    PLACE_X = 0.55
    PLACE_Y = -0.15
    PLACE_Z = 0.20
    PLACE_APPROACH_Z = 0.18
    PLACE_DESCENT_STEPS = 8

    SIDE_GRASP_ROLL = math.pi
    SIDE_GRASP_PITCH = math.pi / 2

    def __init__(self):
        super().__init__("panda_side_pick_place")

        self.cb_group = ReentrantCallbackGroup()
        self.state = RobotState.HOME

        self.centroid_buf = deque(maxlen=self.STABLE_FRAMES_REQUIRED)
        self.dims_buf = deque(maxlen=self.STABLE_FRAMES_REQUIRED)
        self.boundary_msg = None

        self.locked_centroid = None  
        self.locked_dims = None     
        self.locked_boundary = None  

        self.attempts = 0
        self.reacquire_until_time = 0.0

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

        self.create_subscription(PointStamped, "/detected_object_position", self.centroid_cb, 10)
        self.create_subscription(PointStamped, "/detected_object_dimensions", self.dims_cb, 10)
        self.create_subscription(PolygonStamped, "/detected_object_boundary", self.boundary_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_cb, 50)

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

    def stable(self) -> bool:
        if len(self.centroid_buf) < self.STABLE_FRAMES_REQUIRED:
            return False
        if len(self.dims_buf) < self.STABLE_FRAMES_REQUIRED:
            return False
        if self.boundary_msg is None:
            return False

        c = np.array([[x, y, z] for (x, y, z, _) in self.centroid_buf], dtype=np.float32)
        d = np.array([[x, y, z] for (x, y, z, _) in self.dims_buf], dtype=np.float32)

        if np.max(np.std(c, axis=0)) > self.POS_STD_THRESH:
            return False

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
        self.locked_dims.point.x = float(d[0]) 
        self.locked_dims.point.y = float(d[1]) 
        self.locked_dims.point.z = float(d[2]) 

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

        return quaternion_from_euler(self.SIDE_GRASP_ROLL, self.SIDE_GRASP_PITCH, yaw)

    def grasp_succeeded(self) -> bool:
        j1 = self.finger["panda_finger_joint1"]
        j2 = self.finger["panda_finger_joint2"]
        if j1 is None or j2 is None:
            self.get_logger().warn("No finger feedback -> assuming success")
            return True
        avg = 0.5 * (abs(j1) + abs(j2))
        self.get_logger().info(f"Finger avg: {avg:.4f}")
        return avg > self.EMPTY_GRASP_THRESHOLD

    def attempt_side_grasp_once(self) -> bool:
        
        if self.locked_centroid is None or self.locked_dims is None or self.locked_boundary is None:
            return False

        yaw = self.compute_box_yaw()
        if yaw is None:
            self.get_logger().warn("No yaw from boundary, cannot side grasp")
            return False

        quat = self.side_grasp_quat(yaw)

        cen = self.locked_centroid.point
        dims = self.locked_dims.point

        half_short = 0.5 * min(float(dims.x), float(dims.y))
        side_offset = max(0.0, half_short - self.FINAL_INSET)

        nx = math.cos(yaw)
        ny = math.sin(yaw)

        grasp_z = float(cen.z) + self.GRASP_Z_OFFSET
        grasp_z = max(grasp_z, self.MIN_ABSOLUTE_Z)

        for sign in (+1.0, -1.0):
            tx = float(cen.x) + sign * side_offset * nx
            ty = float(cen.y) + sign * side_offset * ny

            ax = tx + sign * self.APPROACH_DIST * nx
            ay = ty + sign * self.APPROACH_DIST * ny

            self.get_logger().info(
                f"Side grasp try sign={int(sign)} yaw={yaw:.3f} "
                f"approach=({ax:.3f},{ay:.3f},{grasp_z:.3f}) target=({tx:.3f},{ty:.3f},{grasp_z:.3f})"
            )

            self.state = RobotState.PICKING
            self.open_gripper()
            time.sleep(0.05)

            self.move_pose(ax, ay, grasp_z, quat, fast=True)

            self.move_pose(tx, ty, grasp_z, quat, fast=False)

            self.close_gripper()
            time.sleep(self.CLOSE_DWELL_SEC)

            if not self.grasp_succeeded():
                self.get_logger().warn("Empty grasp -> backoff and try other side")
                self.move_pose(ax, ay, grasp_z, quat, fast=True)
                continue

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

            self.move_pose(tx, ty, grasp_z + self.LIFT_DZ, quat, fast=True)
            return True

        return False

    
    def place(self):
        self.state = RobotState.PLACING

        px, py, pz = self.PLACE_X, self.PLACE_Y, self.PLACE_Z
        above_z = pz + self.PLACE_APPROACH_Z


        quat = quaternion_from_euler(self.SIDE_GRASP_ROLL, self.SIDE_GRASP_PITCH, 0.0)

        self.move_pose(px, py, above_z, quat, fast=True)

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

        self.move_pose(px, py, above_z, quat, fast=True)

    def reacquire(self, reason: str):
        self.get_logger().warn(f"Reacquire: {reason}")
        self.backoff_vertical()
        self.clear_lock_and_buffers()
        self.go_home()
        self.reacquire_until_time = time.time() + self.REACQUIRE_COOLDOWN_SEC

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

        self.lock()

        ok = self.attempt_side_grasp_once()
        if not ok:
            self.reacquire("side grasp failed")
            return

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
