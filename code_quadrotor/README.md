# Quadrotor Onboard Pipeline

C++/CUDA code for real-time scene flow estimation and CBF-based collision avoidance, running onboard an NVIDIA Jetson Orin NX.

## Overview

The pipeline captures RGB-D frames from an Intel RealSense D445, estimates dense scene flow, evaluates a composite control barrier function, and outputs safe velocity corrections. Everything runs onboard at 20+ Hz.

Key source files in `src/apps/`:

| File | Description |
|---|---|
| `main_scene_flow_impair.cpp` | Offline scene flow from image pairs |
| `my_app.cpp` | Main onboard application |
| `cbf_calc.cu` / `cbf_calc.h` | CUDA CBF computation |
| `cbf_solver.cpp` / `cbf_solver.h` | CBF QP solver |
| `scene_flow_impair.cpp` / `.h` | Scene flow estimation core |
| `magnitude_filter.h` | Scene flow magnitude filtering |
| `rs_reader_threaded.cpp` / `.h` | Threaded RealSense capture |

CBF parameters are configured via YAML files in `config/`.

## Dependencies

- **CUDA** (required)
- **ROS Noetic** + catkin
- **OpenCV**
- **Eigen3**
- **yaml-cpp**
- **Intel RealSense SDK**

## Setup

### CUDA

Verify your GPU is detected:

```bash
nvcc -o device_query device_query.cpp && ./device_query
```

You may need these exports:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

On Jetson you may need to select the right CUDA version:

```bash
sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-11.4 50
sudo update-alternatives --config cuda
```

### Intel RealSense

1. Clone [realsense-ros (ros1-legacy)](https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy) and follow its instructions.
2. Clone [librealsense](https://github.com/IntelRealSense/librealsense) and build with CUDA support using the provided `my_libuvc_installation.sh` script (copy it into `librealsense/scripts/` and run it).
3. Alternatively follow the [Jetson installation guide](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md) with RSUSB backend.

### OpenCV with CUDA

If needed, build OpenCV with CUDA support. See `opencv/build_commands.md` if present, or follow [this guide](https://gist.github.com/minhhieutruong0705/8f0ec70c400420e0007c15c98510f133).

## Building

```bash
source /opt/ros/noetic/setup.bash
./rebuild.sh
```

The default CUDA target architecture is `87` (Jetson Orin NX, compute 8.7). To build for a different GPU, change `CUDA_ARCH_BIN` and `CUDA_ARCH_PTX` in `CMakeLists.txt`:

```cmake
# Jetson Orin NX (default)
set(CUDA_ARCH_BIN "87" ...)

# e.g. RTX 3090
set(CUDA_ARCH_BIN "86" ...)
```

## Running

Start the camera:

```bash
roslaunch realsense2_camera rs_camera.launch \
  depth_width:=640 depth_height:=480 depth_fps:=10 \
  enable_sync:=true align_depth:=true
```

Then run the pipeline:

```bash
./pdflow_app
```

## Scene Flow Origin

The scene flow estimator is adapted from [PD-Flow](https://github.com/MarianoJT88/PD-Flow) by Jaimez et al. (ICRA 2015). See the root [README](../README.md) for full citation.

## License

GPLv3 — see [`GPL LICENSE.txt`](GPL%20LICENSE.txt).
