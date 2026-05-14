#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from scene_flow_msgs.msg import SceneFlow
import numpy as np
import threading
import matplotlib
matplotlib.use('TkAgg')  # Force an interactive backend; change to 'Qt5Agg' if needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection
import time

class SceneFlowVisualizer(Node):
    def __init__(self):
        super().__init__('scene_flow_visualizer')
        self.subscription = self.create_subscription(
            SceneFlow,
            '/scene_flow',
            self.scene_flow_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Store the latest scene flow data (as NumPy arrays)
        self.latest_points = None  # Will be an (N, 3) array
        self.latest_vectors = None  # Will be an (N, 3) array

    def scene_flow_callback(self, msg):
        # Check that there is data in both arrays
        if len(msg.points) == 0 or len(msg.flow_vectors) == 0:
            return

        # Convert the point and vector arrays from the message to numpy arrays
        self.latest_points = np.array([[pt.x, pt.y, pt.z] for pt in msg.points])
        self.latest_vectors = np.array([[vec.x, vec.y, vec.z] for vec in msg.flow_vectors])
        # Print min and max of the points and vectors
        self.get_logger().info(
            f"Min point: {np.min(self.latest_points)}, Max point: {np.max(self.latest_points)}"
        )
        self.get_logger().info(
            f"Min vector: {np.min(self.latest_vectors)}, Max vector: {np.max(self.latest_vectors)}"
        )
        self.get_logger().info(
            f"Received {self.latest_points.shape[0]} points and {self.latest_vectors.shape[0]} vectors"
        )

def main(args=None):
    rclpy.init(args=args)
    visualizer_node = SceneFlowVisualizer()

    # Run ROS spinning in a separate thread so that the main thread can update the plot
    ros_thread = threading.Thread(target=rclpy.spin, args=(visualizer_node,), daemon=True)
    ros_thread.start()

    # Set up an interactive Matplotlib 3D plot for visualization
    plt.ion()  # turn on interactive mode
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Scene Flow Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


    time.sleep(5)

    try:
        while rclpy.ok():
            ax.cla()  # clear the axes for fresh drawing
            ax.set_title("Scene Flow Visualization")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            # Set fixed axis limits (modify as needed)
            ax.set_xlim(-1, 2)
            ax.set_ylim(-1, 2)
            ax.set_zlim(-1, 9)

            ax.view_init(elev=90, azim=90)

            # If we have received a scene flow message, update the plot
            if visualizer_node.latest_points is not None and visualizer_node.latest_vectors is not None:
                pts = visualizer_node.latest_points
                vecs = visualizer_node.latest_vectors

                # Log the shape of the points:
                visualizer_node.get_logger().info(f"Plotting {pts.shape[0]} points.")

                # Sample every 10th element:
                pts_sampled = pts[::50]
                vecs_sampled = vecs[::50]

                # Plot the sampled points in blue
                ax.scatter(pts_sampled[:, 0], pts_sampled[:, 1], pts_sampled[:, 2], c='b', marker='o', s=2)

                # Plot the sampled flow vectors as red arrows
                ax.quiver(
                    pts_sampled[:, 0], pts_sampled[:, 1], pts_sampled[:, 2],
                    vecs_sampled[:, 0], vecs_sampled[:, 1], vecs_sampled[:, 2],
                    color='r'
                )

            plt.draw()
            #plt.pause(0.1)  # update every 100 ms
            plt.show()
            input("Press Enter to continue...")
    except KeyboardInterrupt:
        print("Shutting down visualization...")
    finally:
        visualizer_node.destroy_node()
        rclpy.shutdown()
        plt.close('all')

if __name__ == '__main__':
    main()