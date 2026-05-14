#!/usr/bin/env python3
import rospy
from scene_flow_msgs.msg import SceneFlow
import numpy as np
import open3d as o3d

def main():
    rospy.init_node('scene_flow_listener', anonymous=True)
    
    rospy.loginfo("Waiting for first scene flow message on /scene_flow...")
    try:
        msg = rospy.wait_for_message('/scene_flow', SceneFlow, timeout=10.0)
    except rospy.ROSException as e:
        rospy.logerr("Timeout waiting for /scene_flow message: {}".format(e))
        return

    # Check that the received message actually contains data.
    if len(msg.points) == 0 or len(msg.flow_vectors) == 0:
        rospy.logwarn("Empty scene flow message received.")
        return

    # Convert ROS message data to NumPy arrays.
    pts = np.array([[pt.x, pt.y, pt.z] for pt in msg.points])
    vecs = np.array([[vec.x, vec.y, vec.z] for vec in msg.flow_vectors])
    
    rospy.loginfo("Received scene flow with {} points and {} vectors.".format(pts.shape[0], vecs.shape[0]))

    # Filter out points where depth (z) is less than 0.1
    depth_mask = pts[:, 2] >= 0.1
    pts_filtered = pts[depth_mask]
    vecs_filtered = vecs[depth_mask]

    if pts_filtered.shape[0] == 0:
        rospy.logwarn("No points passed the depth filter. Using original points.")
        pts_filtered = pts
        vecs_filtered = vecs

    # Optionally, sample every Nth point/vector if the data is too dense.
    pts_sampled = pts_filtered[::]
    vecs_sampled = vecs_filtered[::]

    # Create an Open3D point cloud.
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
        # Optionally adjust scale for visualization of the vector length.
        scale = 1
        end = p + scale * v

        idx_start = len(arrow_points)
        arrow_points.append(start)
        idx_end = len(arrow_points)
        arrow_points.append(end)
        arrow_lines.append([idx_start, idx_end])
        # Color the arrow red.
        arrow_colors.append([1.0, 0.0, 0.0])

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

    # Only build and add the line set if arrow points exist.
    if arrow_points:
        line_set = o3d.geometry.LineSet()
        arrow_points_np = np.vstack(arrow_points).astype(np.float64)  # Shape (N, 3)
        arrow_lines_np  = np.array(arrow_lines, dtype=np.int32)       # Shape (M, 2)
        arrow_colors_np = np.vstack(arrow_colors).astype(np.float64)   # Shape (M, 3)
    
        line_set.points = o3d.utility.Vector3dVector(arrow_points_np)
        line_set.lines = o3d.utility.Vector2iVector(arrow_lines_np)
        line_set.colors = o3d.utility.Vector3dVector(arrow_colors_np)
        vis.add_geometry(line_set)
    else:
        rospy.logwarn("No arrow points generated; skipping arrow visualization.")

    # Get the ViewControl object and camera parameters.
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    # --- Set custom camera parameters ---
    # Define the desired camera position, target (look-at) and up vector.
    camera_position = np.array([0.0, 0.0, 0.0])
    lookat = np.array([0.0, 0.0, -1.0])
    up_direction = np.array([0.0, 1.0, 0.0])

    # Compute the new camera coordinate frame.
    forward = (lookat - camera_position)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up_direction)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # Build the rotation matrix from world to camera coordinates.
    R = np.stack((right, up, -forward), axis=0)
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