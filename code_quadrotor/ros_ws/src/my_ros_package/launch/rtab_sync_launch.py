import os
from launch import LaunchDescription
import launch_ros.actions

def generate_launch_description():
    # Create a node for RTAB-Map's synchronization functionality.
    rgbd_sync_node = launch_ros.actions.Node(
        package='rtabmap_sync',
        executable='rgbd_sync',
        name='rtab_sync',
        output='screen',
        parameters=[{
            "approx_sync": True  # Set to True for approximate synchronization; change as needed.
        }],
        remappings=[
            # Remap the expected topic names to match your new sensor topics:
            ("rgb/image", "/camera/rgb/image_color"),
            ("depth/image", "/camera/depth/image"),
            ("rgb/camera_info", "/camera/rgb/camera_info"),
            ("depth/camera_info", "/camera/depth/camera_info"),
            # The synchronized composite message is published on this topic.
            ("rgbd_image", "/rtabmap/rgbd_image")
        ]
    )

    return LaunchDescription([rgbd_sync_node])