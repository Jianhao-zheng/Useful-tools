# Visualize pointcloud and camera trajectory incrementally from SLAM input

A Python tool for visualizing RGB-D camera sequences with trajectory information using Open3D. This visualizer handles synchronized RGB-D data with pose information, making it particularly suitable for SLAM and visual odometry datasets like TUM RGB-D.

## Data Format

The visualizer expects data in [TUM-RGBD format](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats):

```
dataset_folder/
├── rgb.txt
├── depth.txt
├── groundtruth.txt
├── rgb/
│   ├── rgb_name1.png
│   ├── rgb_name2.png
│   └── ...
└── depth/
    ├── depth_name1.png
    ├── depth_name2.png
    └── ...
```

## Usage

1. Specify <path/to/dataset> and camera intrinsics in ```run.py```:
```python
folder = "example/rgbd_dataset_freiburg1_desk"

intrinsics = CameraIntrinsics(
    width=640,
    height=480,
    ppx=318.6,  # Principal point X
    ppy=255.3,  # Principal point Y
    fx=517.3,   # Focal length X
    fy=516.5,   # Focal length Y
    depth_scale=1/5000.0,  # Depth scale factor
)
```

2. Optionally set the frame stride for visualization if your sequence is very long
```python
visualizer = Open3DVisualizer(
    data_folder=folder,
    intrinsics=intrinsics,
    stride=<your preferred stride>  # Optional: frame stride for visualization
)
```

3. Simply run ```python run.py```. Note that the default parameters in the script is for TUM-RDGBD fr1/desk, you can download the dataset [here](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)

## Controls

- **Key "A"**: Advance to next frame
- **Key "Esc"**: End the system in advance
- **Mouse Left Button**: Rotate view
- **Mouse Right Button**: Pan view
- **Mouse Wheel**: Zoom in/out

## Visualization Features

1. **Point Clouds**: 
   - Generated from synchronized RGB-D data
   - Full resolution for point cloud generation
   - Filtered by depth range (0.1m to 10m)
   - Downsampled for efficient visualization

2. **Camera Trajectory**: 
   - Wireframe camera model showing position and orientation
   - Configurable scale and color
   - Persistent visualization between frames

3. **Coordinate Axes**:
   - Red: X-axis
   - Green: Y-axis
   - Blue: Z-axis
   - Configurable size with arrow heads

4. **Image Display**:
   - Real-time RGB image preview
   - Automatic resizing for display while maintaining aspect ratio
   - Maximum display size configurable (default: 480px)

## Acknowledgement

This tool is built based on the visualizer from [Droid-SLAM](https://github.com/princeton-vl/DROID-SLAM)