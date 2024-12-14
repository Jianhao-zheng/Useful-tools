import open3d as o3d
import numpy as np
import cv2
import os
import json
from dataclasses import dataclass
from typing import Tuple, Union
from scipy.spatial.transform import Rotation as R

# Camera visualization constants
CAM_POINTS = np.array(
    [
        [0, 0, 0],
        [-1, -1, 1.5],
        [1, -1, 1.5],
        [1, 1, 1.5],
        [-1, 1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5],
    ]
)

CAM_LINES = np.array(
    [[1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]]
)


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    ppx: float  # Principal point X
    ppy: float  # Principal point Y
    fx: float  # Focal length X
    fy: float  # Focal length Y
    depth_scale: float


class Open3DVisualizer:
    def __init__(
        self,
        data_folder: str,
        intrinsics: CameraIntrinsics,
        stride: int = 5,
    ):
        # Load data
        self.rgb_info  = np.loadtxt(
            os.path.join(data_folder, "rgb.txt"), dtype=np.unicode_, delimiter=" "
        )
        self.depth_info = np.loadtxt(
            os.path.join(data_folder, "depth.txt"), dtype=np.unicode_, delimiter=" "
        )
        self.pose_data = np.loadtxt(
            os.path.join(data_folder, "groundtruth.txt"), dtype=np.float64, delimiter=" "
        )

        tstamp_image = self.rgb_info[:, 0].astype(np.float64)
        tstamp_depth = self.depth_info[:, 0].astype(np.float64)
        tstamp_pose = self.pose_data[:, 0].astype(np.float64)
        self.associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        self.sequence_length = len(self.associations)
        self.intrinsics = intrinsics
        self.data_folder = data_folder
        self.stride = stride

        # State variables
        self.current_frame = 0
        self.last_added_frame = -1
        self.last_camera = None
        self.visualizer = None

        self.initial_view_set = False  # Add this line
        self.saved_view = None

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.05):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    @staticmethod
    def resize_with_aspect_ratio(image: np.ndarray, max_size: int = 480) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio if either dimension exceeds max_size.

        Args:
            image: Input image array
            max_size: Maximum allowed dimension

        Returns:
            np.ndarray: Resized image if necessary, original image otherwise
        """
        height, width = image.shape[:2]

        # If both dimensions are smaller than max_size, return original image
        if width <= max_size and height <= max_size:
            return image

        # Calculate the scaling factor to maintain aspect ratio
        scale = min(max_size / width, max_size / height)

        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize image
        resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        return resized

    @staticmethod
    def create_camera_actor(
        color: Union[Tuple[float, float, float], bool] = True, scale: float = 0.2
    ) -> o3d.geometry.LineSet:
        """
        Creates a camera visualization actor.

        Args:
            color: Either a boolean (True for default color) or RGB tuple (0-1 range)
            scale: Scale factor for camera size

        Returns:
            o3d.geometry.LineSet: Camera visualization geometry
        """
        camera_actor = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
            lines=o3d.utility.Vector2iVector(CAM_LINES),
        )

        if isinstance(color, bool):
            # Default color when True is passed
            camera_color = (1.0, 0.5, 0.0)  # Orange color
        else:
            # Use provided color tuple
            camera_color = color

        camera_actor.paint_uniform_color(camera_color)
        return camera_actor

    def create_origin_and_axis(self, size: float = 1.0) -> o3d.geometry.LineSet:
        """
        Create world coordinate axis visualization using only LineSet.

        Args:
            size: Length of each axis arrow

        Returns:
            o3d.geometry.LineSet: Combined axis visualization
        """
        points = []
        lines = []
        colors = []

        # Create axes
        axis_points = [
            [0, 0, 0],
            [size, 0, 0],  # X axis
            [0, 0, 0],
            [0, size, 0],  # Y axis
            [0, 0, 0],
            [0, 0, size],  # Z axis
        ]

        axis_lines = [[0, 1], [2, 3], [4, 5]]  # Connect points for each axis

        # Add arrow heads (small additional lines)
        arrow_length = size * 0.1
        arrow_points = [
            # X axis arrow
            [size - arrow_length, arrow_length, 0],
            [size, 0, 0],
            [size - arrow_length, -arrow_length, 0],
            # Y axis arrow
            [arrow_length, size - arrow_length, 0],
            [0, size, 0],
            [-arrow_length, size - arrow_length, 0],
            # Z axis arrow
            [0, arrow_length, size - arrow_length],
            [0, 0, size],
            [0, -arrow_length, size - arrow_length],
        ]

        # Combine all points and create lines for arrows
        points = axis_points + arrow_points
        lines.extend(axis_lines)

        # Add arrow head lines
        base_idx = len(axis_points)
        for i in range(3):  # For each axis
            idx = base_idx + i * 3
            lines.extend(
                [
                    [idx, idx + 1],  # First arrow line
                    [idx + 1, idx + 2],  # Second arrow line
                ]
            )

        # Create colors list (RGB for X, Y, Z)
        axis_colors = [
            [1, 0, 0],  # X axis - Red
            [0, 1, 0],  # Y axis - Green
            [0, 0, 1],  # Z axis - Blue
        ]

        colors = []
        for i, color in enumerate(axis_colors):
            colors.append(color)  # Main axis
        for i, color in enumerate(axis_colors):
            colors.append(color)  # Arrow head first line
            colors.append(color)  # Arrow head second line

        # Create LineSet
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array(points)),
            lines=o3d.utility.Vector2iVector(np.array(lines)),
        )

        # Assign colors to lines
        line_set.colors = o3d.utility.Vector3dVector(np.array(colors))

        return line_set

    def create_point_cloud(
        self, rgb: np.ndarray, depth: np.ndarray, pose: np.ndarray
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """Creates point cloud from RGB and depth data."""
        img_stride = 8
        u, v = np.meshgrid(range(self.intrinsics.width), range(self.intrinsics.height))
        u = u[::img_stride, ::img_stride].reshape(-1)
        v = v[::img_stride, ::img_stride].reshape(-1)

        depth = depth[::img_stride, ::img_stride].reshape(-1)
        rgb = rgb[::img_stride, ::img_stride, :].reshape((-1, 3))

        # Filter points by depth
        valid_depth = (depth > 0.1) & (depth < 10)
        rgb = rgb[valid_depth, ::-1] / 255.0

        # Create point cloud in camera coordinates
        points_c = np.vstack(
            (
                (u - self.intrinsics.ppx) * depth / self.intrinsics.fx,
                (v - self.intrinsics.ppy) * depth / self.intrinsics.fy,
                depth,
                np.ones_like(depth),
            )
        )
        points_c = points_c[:, valid_depth]

        # Transform points to world coordinates
        points_world = (pose @ points_c)[:3].transpose()

        # Create and return point clouds
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        # Create camera position marker
        pcd_camera = o3d.geometry.PointCloud()
        pcd_camera.points = o3d.utility.Vector3dVector(pose[:3, 3].reshape((1, -1)))
        pcd_camera.colors = o3d.utility.Vector3dVector(np.array([[0.0, 1.0, 0.0]]))

        return pcd, pcd_camera

    def load_frame_data(
        self, frame_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads RGB, depth, and pose data for a given frame."""
        # self.associations[frame_idx] is a tuple of 3 synchronized 
        #  (img_idx, depth_idx, pose_idx)
        rgb_info = self.rgb_info[self.associations[frame_idx][0]]
        depth_info = self.depth_info[self.associations[frame_idx][1]]
        pose_vec = self.pose_data[self.associations[frame_idx][2]]

        # assert float(rgb_info[0]) == pose_vec[0], "Frame mismatch"

        # Load RGB image
        rgb_path = os.path.join(self.data_folder, rgb_info[1])
        rgb = cv2.imread(rgb_path)

        cv2.imshow("img", self.resize_with_aspect_ratio(rgb, max_size=480) / 255.0)
        cv2.waitKey(1)

        # Load depth image
        depth_path = os.path.join(self.data_folder, depth_info[1])
        depth = (
            cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) * self.intrinsics.depth_scale
        )

        # Create pose matrix
        pose = self.pose_matrix_from_quaternion(pose_vec[1:])

        return rgb, depth, pose

    @staticmethod
    def pose_matrix_from_quaternion(pose_vec: np.ndarray) -> np.ndarray:
        """Creates 4x4 pose matrix from quaternion vector."""
        pose = np.eye(4)
        pose[:3, :3] = R.from_quat(pose_vec[3:]).as_matrix()
        pose[:3, 3] = pose_vec[:3]
        return pose

    def animation_callback(self, vis: o3d.visualization.Visualizer) -> None:
        """Callback for updating visualization."""
        # Check if we've reached the end
        if self.current_frame >= self.sequence_length:
            print("Reached end of sequence. Closing visualizer...")
            vis.close()
            return False

        if self.current_frame > self.last_added_frame:
            # Save current view parameters if already set
            if self.initial_view_set:
                self.saved_view = (
                    vis.get_view_control().convert_to_pinhole_camera_parameters()
                )

            # Load frame data
            rgb, depth, pose = self.load_frame_data(self.current_frame)

            # Update camera visualization
            if self.last_camera is not None:
                vis.remove_geometry(self.last_camera)

            cam_color = (1.0, 0.7, 0.7)
            cam_actor = self.create_camera_actor(color=cam_color, scale=0.2)
            cam_actor.transform(pose)

            if self.current_frame > -1:
                vis.add_geometry(cam_actor)
                self.last_camera = cam_actor

            # Create and add point clouds
            pcd, pcd_camera = self.create_point_cloud(rgb, depth, pose)
            if self.current_frame > -1:
                vis.add_geometry(pcd)
                vis.add_geometry(pcd_camera)

            # Add coordinate axes only once
            if not self.initial_view_set:
                vis.add_geometry(self.create_origin_and_axis(size=0.5))
                self.initial_view_set = True

            # Restore previous view if it exists
            if self.saved_view is not None:
                vis.get_view_control().convert_from_pinhole_camera_parameters(
                    self.saved_view
                )

            vis.poll_events()
            vis.update_renderer()
            self.last_added_frame = self.current_frame

    def next_frame(self, vis: o3d.visualization.Visualizer) -> bool:
        """
        Advances to next frame. Returns False to stop visualization when reaching the end.
        """
        next_frame = self.current_frame + self.stride

        # Check if we've reached the end of the sequence
        if next_frame >= self.sequence_length:
            print("Reached end of sequence. Closing visualizer...")
            vis.close()
            return False

        self.current_frame = next_frame
        return True

    def run(self) -> None:
        """Runs the visualizer."""
        self.visualizer = o3d.visualization.VisualizerWithKeyCallback()
        self.visualizer.register_animation_callback(self.animation_callback)
        self.visualizer.register_key_callback(ord("A"), self.next_frame)

        self.visualizer.create_window(height=540, width=960)
        self.visualizer.run()
        self.visualizer.destroy_window()


def main():
    folder = "example/rgbd_dataset_freiburg1_desk"

    # intrinsic for tum-rgbd fr1
    intrinsics = CameraIntrinsics(
        width=640,
        height=480,
        ppx=318.6,
        ppy=255.3,
        fx=517.3,
        fy=516.5,
        depth_scale=1/5000.0,
    )

    # Create and run visualizer
    visualizer = Open3DVisualizer(folder, intrinsics, stride=1)
    visualizer.run()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
