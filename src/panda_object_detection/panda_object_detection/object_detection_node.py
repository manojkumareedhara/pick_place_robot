#!/usr/bin/env python3
"""
Robust RGB-D Cube Detector for Gazebo Simulation

Optimized for detecting and tracking colored cubes on a table.

Publishes:
- /detected_object_position    (PointStamped): centroid in panda_link0
- /detected_object_top         (PointStamped): top surface grasp point
- /detected_object_height      (Float32): cube height
- /detected_object_dimensions  (PointStamped): width, depth, height
- /detected_object_boundary    (PolygonStamped): 3D footprint boundary
- /debug_image                 (Image): visualization

Run:
  ros2 run panda_object_detection cube_detector.py
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PolygonStamped, Point32
from std_msgs.msg import Float32

from cv_bridge import CvBridge
import numpy as np
import cv2

import tf2_ros
from tf2_geometry_msgs import do_transform_point
import message_filters
from collections import deque


class CubeDetector(Node):
    """Robust detector for colored cubes using RGB-D camera."""
    
    def __init__(self):
        super().__init__("cube_detector")

        self.bridge = CvBridge()

        # ===================== PARAMETERS =====================
        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_info_received = False
        
        # Depth filtering
        self.declare_parameter("min_depth", 0.20)
        self.declare_parameter("max_depth", 1.50)
        self.declare_parameter("min_area", 400)
        
        # Color detection (RED by default for common test cubes)
        self.declare_parameter("lower_hue1", 0)
        self.declare_parameter("upper_hue1", 10)
        self.declare_parameter("lower_hue2", 170)
        self.declare_parameter("upper_hue2", 180)
        self.declare_parameter("min_saturation", 100)
        self.declare_parameter("min_value", 50)
        
        # Depth percentiles for robust measurement
        self.declare_parameter("top_percentile", 10.0)     # top surface
        self.declare_parameter("bottom_percentile", 90.0)  # bottom surface
        
        # Morphology
        self.declare_parameter("morph_kernel", 5)
        
        # Frames
        self.declare_parameter("camera_frame", "camera_link_optical")
        self.declare_parameter("base_frame", "panda_link0")
        
        # ===================== TF SETUP =====================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # ===================== TEMPORAL SMOOTHING =====================
        self.centroid_buffer = deque(maxlen=5)
        self.top_buffer = deque(maxlen=5)
        
        # ===================== SUBSCRIBERS =====================
        # Synchronized RGB and Depth
        self.rgb_sub = message_filters.Subscriber(
            self, Image, "/camera/image_raw"
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, "/camera/depth/image_raw"
        )
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1,
        )
        self.ts.registerCallback(self.rgbd_callback)
        
        # Camera info
        self.create_subscription(
            CameraInfo,
            "/camera/depth/camera_info",
            self.camera_info_callback,
            10,
        )
        
        # ===================== PUBLISHERS =====================
        self.centroid_pub = self.create_publisher(
            PointStamped, "/detected_object_position", 10
        )
        self.top_pub = self.create_publisher(
            PointStamped, "/detected_object_top", 10
        )
        self.height_pub = self.create_publisher(
            Float32, "/detected_object_height", 10
        )
        self.dimensions_pub = self.create_publisher(
            PointStamped, "/detected_object_dimensions", 10
        )
        self.boundary_pub = self.create_publisher(
            PolygonStamped, "/detected_object_boundary", 10
        )
        self.debug_image_pub = self.create_publisher(
            Image, "/debug_image", 10
        )
        
        self.get_logger().info("Cube detector initialized. Waiting for camera info...")

    # ===================== CAMERA INFO =====================
    def camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics."""
        if self.camera_info_received:
            return
        
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])
        self.camera_info_received = True
        
        self.get_logger().info(
            f"Camera intrinsics received: "
            f"fx={self.fx:.1f}, fy={self.fy:.1f}, "
            f"cx={self.cx:.1f}, cy={self.cy:.1f}"
        )

    # ===================== MAIN CALLBACK =====================
    def rgbd_callback(self, rgb_msg: Image, depth_msg: Image):
        """Process synchronized RGB-D images."""
        if not self.camera_info_received:
            return

        # Convert images
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        # Sanitize depth
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        # Convert mm to m if needed
        if depth[depth > 0].size > 0 and np.max(depth[depth > 0]) > 10.0:
            depth *= 0.001

        # Detect object
        mask = self.detect_colored_object(rgb, depth)
        if mask is None:
            self.publish_debug_image(rgb, None, None, None)
            return

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.publish_debug_image(rgb, None, None, None)
            return

        # Filter by area
        min_area = float(self.get_parameter("min_area").value)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not valid_contours:
            self.publish_debug_image(rgb, None, None, None)
            return

        # Get largest contour (closest/biggest cube)
        contour = max(valid_contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)

        # Process detection
        self.process_cube(hull, depth, rgb, rgb_msg.header)

    # ===================== COLOR DETECTION =====================
    def detect_colored_object(self, rgb, depth):
        """
        Detect colored object using HSV color space and depth filtering.
        
        Returns:
            Binary mask of detected object, or None if nothing found
        """
        min_depth = float(self.get_parameter("min_depth").value)
        max_depth = float(self.get_parameter("max_depth").value)

        # Convert to HSV
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

        # Red color detection (wraps around hue)
        lh1 = int(self.get_parameter("lower_hue1").value)
        uh1 = int(self.get_parameter("upper_hue1").value)
        lh2 = int(self.get_parameter("lower_hue2").value)
        uh2 = int(self.get_parameter("upper_hue2").value)
        ms = int(self.get_parameter("min_saturation").value)
        mv = int(self.get_parameter("min_value").value)

        # Create color masks
        mask1 = cv2.inRange(hsv, (lh1, ms, mv), (uh1, 255, 255))
        mask2 = cv2.inRange(hsv, (lh2, ms, mv), (uh2, 255, 255))
        color_mask = cv2.bitwise_or(mask1, mask2)

        # Depth filtering
        depth_mask = ((depth > min_depth) & (depth < max_depth)).astype(np.uint8) * 255

        # Combine masks
        combined = cv2.bitwise_and(color_mask, depth_mask)

        # Morphological operations to clean up
        k = int(self.get_parameter("morph_kernel").value)
        k = max(3, k | 1)  # ensure odd and >= 3
        kernel = np.ones((k, k), np.uint8)

        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

        # Check if enough pixels detected
        if np.count_nonzero(combined) < 100:
            return None

        return combined

    # ===================== CUBE PROCESSING =====================
    def process_cube(self, hull, depth, rgb, header):
        """
        Process detected cube and publish all information.
        
        Args:
            hull: Convex hull of detected contour
            depth: Depth image
            rgb: RGB image
            header: Image header with timestamp
        """
        cam_frame = self.get_parameter("camera_frame").value
        base_frame = self.get_parameter("base_frame").value

        # Create mask from hull
        mask = np.zeros(depth.shape, dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)

        # Get depth values inside hull
        depths = depth[mask == 255]
        depths = depths[depths > 0]

        if depths.size < 50:
            self.publish_debug_image(rgb, hull, None, None)
            return

        # Compute robust depth statistics
        top_p = float(self.get_parameter("top_percentile").value)
        bottom_p = float(self.get_parameter("bottom_percentile").value)

        z_top = float(np.percentile(depths, top_p))
        z_bottom = float(np.percentile(depths, bottom_p))
        z_centroid = float(np.median(depths))
        height = float(max(0.0, z_bottom - z_top))

        # Compute 2D centroid
        M = cv2.moments(hull)
        if abs(M["m00"]) < 1e-6:
            self.publish_debug_image(rgb, hull, None, None)
            return

        u_c = int(M["m10"] / M["m00"])
        v_c = int(M["m01"] / M["m00"])

        # Compute top grasp point (center of top surface)
        # Use pixels near top depth
        top_layer = (mask == 255) & (depth > 0) & (depth <= (z_top + 0.015))
        ys, xs = np.where(top_layer)

        if xs.size >= 20:
            u_top = int(np.mean(xs))
            v_top = int(np.mean(ys))
        else:
            u_top, v_top = u_c, v_c

        # Convert to 3D camera frame
        # Centroid point
        cx_cam = float((u_c - self.cx) * z_centroid / self.fx)
        cy_cam = float((v_c - self.cy) * z_centroid / self.fy)
        cz_cam = float(z_centroid)

        # Top grasp point
        tx_cam = float((u_top - self.cx) * z_top / self.fx)
        ty_cam = float((v_top - self.cy) * z_top / self.fy)
        tz_cam = float(z_top)

        # Transform to base frame
        stamp_time = rclpy.time.Time.from_msg(header.stamp)

        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame=base_frame,
                source_frame=cam_frame,
                time=stamp_time,
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            self.publish_debug_image(rgb, hull, (u_c, v_c), (u_top, v_top))
            return

        # Create and transform centroid point
        centroid_cam = PointStamped()
        centroid_cam.header.frame_id = cam_frame
        centroid_cam.header.stamp = header.stamp
        centroid_cam.point.x = cx_cam
        centroid_cam.point.y = cy_cam
        centroid_cam.point.z = cz_cam

        # Create and transform top point
        top_cam = PointStamped()
        top_cam.header.frame_id = cam_frame
        top_cam.header.stamp = header.stamp
        top_cam.point.x = tx_cam
        top_cam.point.y = ty_cam
        top_cam.point.z = tz_cam

        centroid_base = do_transform_point(centroid_cam, tf)
        top_base = do_transform_point(top_cam, tf)

        # Temporal smoothing
        self.centroid_buffer.append((
            centroid_base.point.x,
            centroid_base.point.y,
            centroid_base.point.z
        ))
        self.top_buffer.append((
            top_base.point.x,
            top_base.point.y,
            top_base.point.z
        ))

        c_avg = np.mean(np.array(self.centroid_buffer, dtype=np.float32), axis=0)
        t_avg = np.mean(np.array(self.top_buffer, dtype=np.float32), axis=0)

        # Publish smoothed centroid
        centroid_out = PointStamped()
        centroid_out.header.frame_id = base_frame
        centroid_out.header.stamp = header.stamp
        centroid_out.point.x = float(c_avg[0])
        centroid_out.point.y = float(c_avg[1])
        centroid_out.point.z = float(c_avg[2])

        # Publish smoothed top
        top_out = PointStamped()
        top_out.header.frame_id = base_frame
        top_out.header.stamp = header.stamp
        top_out.point.x = float(t_avg[0])
        top_out.point.y = float(t_avg[1])
        top_out.point.z = float(t_avg[2])

        # Compute dimensions
        x, y, w, h = cv2.boundingRect(hull)
        x1_3d = float((x - self.cx) * z_centroid / self.fx)
        y1_3d = float((y - self.cy) * z_centroid / self.fy)
        x2_3d = float(((x + w) - self.cx) * z_centroid / self.fx)
        y2_3d = float(((y + h) - self.cy) * z_centroid / self.fy)
        
        width = float(abs(x2_3d - x1_3d))
        depth_dim = float(abs(y2_3d - y1_3d))

        dims = PointStamped()
        dims.header.frame_id = base_frame
        dims.header.stamp = header.stamp
        dims.point.x = width
        dims.point.y = depth_dim
        dims.point.z = height

        # Publish height
        hmsg = Float32()
        hmsg.data = height

        # Publish all
        self.centroid_pub.publish(centroid_out)
        self.top_pub.publish(top_out)
        self.height_pub.publish(hmsg)
        self.dimensions_pub.publish(dims)

        # Publish boundary
        self.publish_boundary(hull, depth, header, tf)

        # Debug visualization
        self.publish_debug_image(rgb, hull, (u_c, v_c), (u_top, v_top))

    # ===================== BOUNDARY =====================
    def publish_boundary(self, hull, depth, header, tf_cam_to_base):
        """Publish 3D boundary polygon in base frame."""
        cam_frame = self.get_parameter("camera_frame").value
        base_frame = self.get_parameter("base_frame").value

        poly = PolygonStamped()
        poly.header.frame_id = base_frame
        poly.header.stamp = header.stamp

        for p in hull:
            u, v = int(p[0][0]), int(p[0][1])
            
            # Bounds check
            if v < 0 or v >= depth.shape[0] or u < 0 or u >= depth.shape[1]:
                continue

            z = float(depth[v, u])
            if z <= 0.0:
                continue

            # Convert to 3D camera frame
            x = float((u - self.cx) * z / self.fx)
            y = float((v - self.cy) * z / self.fy)

            # Create point in camera frame
            cam_pt = PointStamped()
            cam_pt.header.frame_id = cam_frame
            cam_pt.header.stamp = header.stamp
            cam_pt.point.x = x
            cam_pt.point.y = y
            cam_pt.point.z = z

            # Transform to base frame
            base_pt = do_transform_point(cam_pt, tf_cam_to_base)

            # Add to polygon
            p32 = Point32()
            p32.x = float(base_pt.point.x)
            p32.y = float(base_pt.point.y)
            p32.z = float(base_pt.point.z)
            poly.polygon.points.append(p32)

        if poly.polygon.points:
            self.boundary_pub.publish(poly)

    # ===================== DEBUG VISUALIZATION =====================
    def publish_debug_image(self, rgb, hull, centroid_uv, top_uv):
        """Publish debug image with overlays."""
        debug = rgb.copy()

        if hull is not None:
            # Draw hull
            cv2.drawContours(debug, [hull], -1, (0, 255, 0), 2)

        if centroid_uv is not None:
            # Draw centroid (blue)
            cv2.circle(debug, centroid_uv, 6, (255, 0, 0), -1)
            cv2.putText(
                debug, "CENTROID", 
                (centroid_uv[0] + 10, centroid_uv[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
            )

        if top_uv is not None:
            # Draw top grasp point (red)
            cv2.circle(debug, top_uv, 6, (0, 0, 255), -1)
            cv2.putText(
                debug, "GRASP", 
                (top_uv[0] + 10, top_uv[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )

        # Publish
        try:
            msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            self.debug_image_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"Debug image publish failed: {e}")


def main():
    rclpy.init()
    node = CubeDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()