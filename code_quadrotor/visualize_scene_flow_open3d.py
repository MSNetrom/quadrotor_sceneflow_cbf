#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from scene_flow_msgs.msg import SceneFlow
import numpy as np
import open3d as o3d

class SceneFlowListener(Node):
    def __init__(self):
        super().__init__('scene_flow_listener')
        # Subscribe to the /scene_flow topic
        self.subscription = self.create_subscription(
            SceneFlow,
            '/scene_flow',
            self.scene_flow_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Variables to store the first message's data
        self.latest_points = None  # Will be an (N, 3) NumPy array
        self.latest_vectors = None  # Will be an (N, 3) NumPy array
        self._first_message_received = False

    def scene_flow_callback(self, msg):
        if not self._first_message_received:
            # Check that data is present
            if len(msg.points) == 0 or len(msg.flow_vectors) == 0:
                self.get_logger().warn("Empty scene flow message received.")
                return

            # Convert ROS message data to NumPy arrays
            self.latest_points = np.array([[pt.x, pt.y, pt.z] for pt in msg.points])
            self.latest_vectors = np.array([[vec.x, vec.y, vec.z] for vec in msg.flow_vectors])
            self._first_message_received = True

            self.get_logger().info(
                f"Received scene flow with {self.latest_points.shape[0]} points and {self.latest_vectors.shape[0]} vectors."
            )
            # Shutdown ROS spinning as we only need the first message for visualization
            rclpy.shutdown()

def main(args=None):
    # Initialize ROS
    rclpy.init(args=args)
    listener = SceneFlowListener()
    # Spin until the first message is received
    rclpy.spin(listener)

    # After we have the data from the first message, continue with Open3D visualization
    if listener.latest_points is None or listener.latest_vectors is None:
        print("No scene flow message received. Exiting...")
        return

    pts = listener.latest_points
    vecs = listener.latest_vectors

    # Filter out points where depth (z) is less than 0.1
    depth_mask = pts[:, 2] >= 0.1
    pts_filtered = pts[depth_mask]
    vecs_filtered = vecs[depth_mask]

    # Optionally sample every Nth point/vector if the data is too dense.
    pts_sampled = pts_filtered[::]
    vecs_sampled = vecs_filtered[::]

    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_sampled)

    # Build an Open3D LineSet to represent the vectors as arrows/lines.
    arrow_points = []
    arrow_lines = []
    arrow_colors = []

    # For each sampled point and its corresponding vector, add a line from the point
    # to the point plus the vector.
    for p, v in zip(pts_sampled, vecs_sampled):
        start = p
        # Optionally adjust scale for visualization of the vector length
        scale = 1
        end = p + scale * v

        idx_start = len(arrow_points)
        arrow_points.append(start)
        idx_end = len(arrow_points)
        arrow_points.append(end)
        arrow_lines.append([idx_start, idx_end])
        # Color the arrow red
        arrow_colors.append([1.0, 0.0, 0.0])

    # Create an Open3D LineSet for the arrows
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(arrow_points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(arrow_lines))
    line_set.colors = o3d.utility.Vector3dVector(np.array(arrow_colors))

    # Instead of using draw_geometries, use the Visualizer class to enable camera control.
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Scene Flow Vector Field",
        width=800,
        height=600,
        left=50,
        top=50
    )
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)

    # Get the ViewControl object and camera parameters.
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    # --- Set custom camera parameters ---
    # Define the desired camera position, target (look-at) and up vector.
    camera_position = np.array([0.0, 0.0, 0.0])  # Use floats!
    lookat = np.array([0.0, 0.0, -1.0])
    up_direction = np.array([0.0, 1.0, 0.0])

    # Compute the new camera coordinate frame.
    # Forward vector (from camera towards target)
    forward = (lookat - camera_position)
    forward /= np.linalg.norm(forward)
    # Right vector:
    right = np.cross(forward, up_direction)
    right /= np.linalg.norm(right)
    # Recompute the orthogonal up vector
    up = np.cross(right, forward)

    # Build the rotation matrix from world to camera coordinates.
    # Open3D expects an extrinsic matrix that transforms points from world space
    # into camera space. One way to compute this is to form a 4x4 matrix:
    R = np.stack((right, up, -forward), axis=0)  # 3x3 rotation
    T = -R @ camera_position.reshape(3, 1)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3:] = T

    # Assign our modified extrinsic matrix to the camera parameters.
    param.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(param)
    # --- End of camera parameter adjustment ---

    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    main() 