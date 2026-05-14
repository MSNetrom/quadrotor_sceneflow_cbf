# SceneFlow.msg
#
# A header (with timestamp and frame)
std_msgs/Header header

# An array of 3D points where scene flow is evaluated.
geometry_msgs/Point[] points

# An array of corresponding 3D vectors.
geometry_msgs/Vector3[] flow_vectors