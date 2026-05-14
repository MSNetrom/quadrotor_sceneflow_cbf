#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def process_bag_to_video(bagpath, output_video_path, topic='/scene_flow', frame_step=10):
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (800, 600))
    
    # Get ROS2 typestore
    typestore = get_typestore(Stores.ROS2_HUMBLE)  # Use appropriate ROS2 distribution
    
    # Process the bag file
    with AnyReader([Path(bagpath)], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic]
        
        if not connections:
            print(f"No messages found on topic {topic}")
            return
        
        frame_count = 0
        video_frame_count = 0
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            frame_count += 1
            
            # Skip frames based on the frame_step parameter
            if frame_count % frame_step != 0:
                continue
                
            video_frame_count += 1
            print(f"Processing frame {frame_count} (video frame {video_frame_count})")
            
            # Deserialize the message
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            # Convert message data to NumPy arrays
            pts = np.array([[pt.x, pt.y, pt.z] for pt in msg.points])
            vecs = np.array([[vec.x, vec.y, vec.z] for vec in msg.flow_vectors])
            
            # Skip if empty
            if len(pts) == 0 or len(vecs) == 0:
                continue
                
            # Filter out points where depth (z) is less than 0.1
            depth_mask = pts[:, 2] >= 0.1
            pts_filtered = pts[depth_mask]
            vecs_filtered = vecs[depth_mask]
            
            # Optional: Downsample for better performance
            if len(pts_filtered) > 2000:  # Only downsample if there are many points
                step = max(1, len(pts_filtered) // 2000)
                indices = np.arange(0, len(pts_filtered), step)
                pts_filtered = pts_filtered[indices]
                vecs_filtered = vecs_filtered[indices]
            
            # Create a new figure for this frame
            fig = plt.figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot points
            ax.scatter(pts_filtered[:, 0], pts_filtered[:, 1], pts_filtered[:, 2], 
                       c='lightgrey', s=2, alpha=0.5)
            
            # Plot vectors as arrows
            # For better performance, only plot a subset of arrows
            if len(pts_filtered) > 500:
                arrow_step = max(1, len(pts_filtered) // 500)
                arrow_indices = np.arange(0, len(pts_filtered), arrow_step)
                pts_arrows = pts_filtered[arrow_indices]
                vecs_arrows = vecs_filtered[arrow_indices]
            else:
                pts_arrows = pts_filtered
                vecs_arrows = vecs_filtered
                
            # Calculate vector magnitudes for coloring
            magnitudes = np.linalg.norm(vecs_arrows, axis=1)
            normalized_magnitudes = magnitudes / (np.max(magnitudes) if np.max(magnitudes) > 0 else 1.0)
            
            # Plot each arrow
            for i, (p, v) in enumerate(zip(pts_arrows, vecs_arrows)):
                # Use magnitude for color (red->yellow spectrum)
                color = (1.0, normalized_magnitudes[i], 0)  # (r,g,b) from red to yellow
                ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2], 
                          color=color, length=1.0, normalize=True, alpha=0.8)
            
            # Set axis limits and labels
            min_xyz = np.min(pts_filtered, axis=0) - 1.0
            max_xyz = np.max(pts_filtered, axis=0) + 1.0
            
            ax.set_xlim([min_xyz[0], max_xyz[0]])
            ax.set_ylim([min_xyz[1], max_xyz[1]])
            ax.set_zlim([min_xyz[2], max_xyz[2]])
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Set background color
            ax.set_facecolor((0.1, 0.1, 0.1))
            fig.patch.set_facecolor((0.1, 0.1, 0.1))
            
            # Set title
            ax.set_title(f'Scene Flow - Frame {frame_count}', color='white')
            
            # Set white labels
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.zaxis.label.set_color('white')
            ax.tick_params(colors='white')
            
            # Set viewing angle
            ax.view_init(elev=20, azim=-35)
            
            # Capture the figure as an image
            fig.canvas.draw()
            # Get the ARGB buffer from the figure
            buf = fig.canvas.tostring_argb()
            ncols, nrows = fig.canvas.get_width_height()
            # The buffer is in ARGB format, so we need to process it to get RGB
            img = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)
            # Reorder from ARGB to BGR (discarding alpha channel)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            # Write the frame to video
            video_writer.write(img)
            
            # Close the figure to free memory
            plt.close(fig)
    
    # Cleanup
    video_writer.release()
    print(f"Video saved to {output_video_path}, processed {video_frame_count} frames out of {frame_count} total")

if __name__ == '__main__':
    bag_path = '../sceneflow_cbf/jacamar_2025-05-08-14-59-52_0.bag'  # Your bag file path
    output_path = 'scene_flow_visualization.mp4'
    process_bag_to_video(bag_path, output_path, frame_step=30)  # Process every 10th frame