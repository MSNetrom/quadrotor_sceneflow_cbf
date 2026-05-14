#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from scene_flow_msgs.msg import SceneFlow  # Import your custom message

class SceneFlowListener(Node):
    def __init__(self):
        super().__init__('scene_flow_listener')
        self.subscription = self.create_subscription(
            SceneFlow,
            '/scene_flow',
            self.listener_callback,
            10)
        self.subscription  # Prevent unused variable warning

    def listener_callback(self, msg):
        # Log header information
        self.get_logger().info("Received a SceneFlow message!")
        self.get_logger().info(
            f"Header Stamp: sec={msg.header.stamp.sec}, nanosec={msg.header.stamp.nanosec}"
        )
        self.get_logger().info(f"Header Frame ID: {msg.header.frame_id}")

        # Log how many points and vectors are in the message
        num_points = len(msg.points)
        num_vectors = len(msg.flow_vectors)
        self.get_logger().info(
            f"Received {num_points} points and {num_vectors} flow vectors."
        )

        if num_points > 0 and num_vectors > 0:
            # For example, display the first point and its corresponding flow vector.
            first_point = msg.points[0]
            first_vector = msg.flow_vectors[0]
            self.get_logger().info(
                f"First Point -> x: {first_point.x}, y: {first_point.y}, z: {first_point.z}"
            )
            self.get_logger().info(
                f"First Flow Vector -> x: {first_vector.x}, y: {first_vector.y}, z: {first_vector.z}"
            )

def main(args=None):
    rclpy.init(args=args)
    node = SceneFlowListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
