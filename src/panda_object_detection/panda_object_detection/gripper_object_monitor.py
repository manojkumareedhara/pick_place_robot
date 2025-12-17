#!/usr/bin/env python3
"""
Monitor and compare gripper finger positions with detected object position.
Calculates distance and checks if gripper has reached the object.

Usage:
ros2 run your_package gripper_object_monitor.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
import tf2_ros
from tf2_ros import TransformException
import numpy as np
import math


class GripperObjectMonitor(Node):
    def __init__(self):
        super().__init__('gripper_object_monitor')
        
        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # State variables
        self.object_position = None
        self.gripper_center_position = None
        self.left_finger_position = None
        self.right_finger_position = None
        self.joint_states = None
        
        # Subscribers
        self.object_sub = self.create_subscription(
            PointStamped,
            '/detected_object_position',
            self.object_callback,
            10
        )
        
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Parameters
        self.declare_parameter('threshold_distance', 0.05)  # 5cm threshold
        self.declare_parameter('monitor_rate', 2.0)  # Hz
        
        # Timer for periodic monitoring
        rate = self.get_parameter('monitor_rate').value
        self.timer = self.create_timer(1.0 / rate, self.monitor_callback)
        
        self.get_logger().info('Gripper-Object Monitor started')
        self.get_logger().info('Monitoring object and gripper positions...')
        
    def object_callback(self, msg: PointStamped):
        """Store detected object position"""
        if msg.header.frame_id == 'panda_link0':
            self.object_position = np.array([
                msg.point.x,
                msg.point.y,
                msg.point.z
            ])
    
    def joint_state_callback(self, msg: JointState):
        """Store joint states for gripper info"""
        self.joint_states = msg
    
    def get_transform_position(self, target_frame, source_frame):
        """Get position of source_frame in target_frame coordinates"""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            return np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
        except TransformException as e:
            return None
    
    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        if pos1 is None or pos2 is None:
            return None
        return np.linalg.norm(pos1 - pos2)
    
    def get_gripper_state(self):
        """Get gripper opening state from joint states"""
        if self.joint_states is None:
            return None, None
        
        try:
            left_idx = self.joint_states.name.index('panda_finger_joint1')
            right_idx = self.joint_states.name.index('panda_finger_joint2')
            
            left_pos = self.joint_states.position[left_idx]
            right_pos = self.joint_states.position[right_idx]
            
            # Total gripper opening (sum of both fingers)
            gripper_opening = left_pos + right_pos
            
            return left_pos, right_pos, gripper_opening
        except (ValueError, IndexError):
            return None, None, None
    
    def monitor_callback(self):
        """Periodic monitoring and reporting"""
        
        # Get gripper positions from TF
        self.gripper_center_position = self.get_transform_position(
            'panda_link0', 'panda_hand'
        )
        self.left_finger_position = self.get_transform_position(
            'panda_link0', 'panda_leftfinger'
        )
        self.right_finger_position = self.get_transform_position(
            'panda_link0', 'panda_rightfinger'
        )
        
        # Check if we have all necessary data
        if self.object_position is None:
            self.get_logger().info('⏳ Waiting for object detection...', 
                                   throttle_duration_sec=5.0)
            return
        
        if self.gripper_center_position is None:
            self.get_logger().warn('⚠️  Cannot get gripper position from TF',
                                   throttle_duration_sec=5.0)
            return
        
        # Calculate distances
        distance_to_center = self.calculate_distance(
            self.object_position, self.gripper_center_position
        )
        
        distance_to_left = None
        distance_to_right = None
        if self.left_finger_position is not None:
            distance_to_left = self.calculate_distance(
                self.object_position, self.left_finger_position
            )
        if self.right_finger_position is not None:
            distance_to_right = self.calculate_distance(
                self.object_position, self.right_finger_position
            )
        
        # Get gripper state
        left_joint, right_joint, gripper_opening = self.get_gripper_state()
        
        # Get threshold
        threshold = self.get_parameter('threshold_distance').value
        
        # Generate report
        self.print_report(
            distance_to_center,
            distance_to_left,
            distance_to_right,
            gripper_opening,
            threshold
        )
    
    def print_report(self, dist_center, dist_left, dist_right, 
                     gripper_opening, threshold):
        """Print formatted monitoring report"""
        
        print("\n" + "="*70)
        print("  GRIPPER-OBJECT POSITION MONITOR")
        print("="*70)
        
        # Object position
        print(f"\n📍 Object Position (panda_link0):")
        print(f"   X: {self.object_position[0]:7.4f} m")
        print(f"   Y: {self.object_position[1]:7.4f} m")
        print(f"   Z: {self.object_position[2]:7.4f} m")
        
        # Gripper center position
        if self.gripper_center_position is not None:
            print(f"\n🤖 Gripper Center (panda_hand):")
            print(f"   X: {self.gripper_center_position[0]:7.4f} m")
            print(f"   Y: {self.gripper_center_position[1]:7.4f} m")
            print(f"   Z: {self.gripper_center_position[2]:7.4f} m")
        
        # Distances
        print(f"\n📏 Distances to Object:")
        print(f"   From gripper center: {dist_center:7.4f} m", end="")
        if dist_center <= threshold:
            print(" ✓ REACHED")
        else:
            print(f" ✗ ({(dist_center - threshold)*100:.1f} cm away)")
        
        if dist_left is not None:
            print(f"   From left finger:    {dist_left:7.4f} m")
        if dist_right is not None:
            print(f"   From right finger:   {dist_right:7.4f} m")
        
        # Relative position
        if self.gripper_center_position is not None:
            delta = self.object_position - self.gripper_center_position
            print(f"\n📐 Object Relative to Gripper:")
            print(f"   ΔX: {delta[0]:+7.4f} m {'(left)' if delta[0] < 0 else '(right)'}")
            print(f"   ΔY: {delta[1]:+7.4f} m {'(back)' if delta[1] < 0 else '(forward)'}")
            print(f"   ΔZ: {delta[2]:+7.4f} m {'(below)' if delta[2] < 0 else '(above)'}")
        
        # Gripper state
        if gripper_opening is not None:
            print(f"\n✋ Gripper State:")
            print(f"   Opening: {gripper_opening*1000:6.2f} mm", end="")
            if gripper_opening < 0.01:  # Less than 10mm
                print(" (CLOSED)")
            elif gripper_opening > 0.07:  # More than 70mm
                print(" (OPEN)")
            else:
                print(" (PARTIAL)")
        
        # Status
        print(f"\n🎯 Status:")
        if dist_center <= threshold:
            print(f"   ✓ Gripper has REACHED the object!")
            print(f"   Distance: {dist_center*100:.2f} cm (threshold: {threshold*100:.1f} cm)")
        else:
            print(f"   ✗ Gripper has NOT reached the object yet")
            print(f"   Distance: {dist_center*100:.2f} cm (need: {threshold*100:.1f} cm)")
            print(f"   Remaining: {(dist_center - threshold)*100:.1f} cm")
        
        print("="*70 + "\n")


def main(args=None):
    rclpy.init(args=args)
    
    node = GripperObjectMonitor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down monitor...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()