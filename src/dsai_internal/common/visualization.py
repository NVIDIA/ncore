# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import numpy as np
import open3d as o3d
import open3d.ml.tf as ml3d

from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.spatial.transform import Rotation as R
from multimethod import multimethod

from src.dsai_internal.data.data import FrameLabel3

from src.dsai_internal.common.nvidia_utils import LabelProcessor as NvidiaLabelProcessor

def rgba(r):
    """Generates a color based on range.

	Args:
		r: the range value of a given point.
	Returns:
		The color for a given range
	"""
    c = plt.get_cmap('jet')((r % 50.0) / 50.0)
    c = list(c)
    c[-1] = 0.5  # alpha
    return c


def plot_image(camera_image):
    """Plot a cmaera image."""
    plt.figure(figsize=(20, 12))
    plt.imshow(camera_image)
    plt.grid(visible=False)


def plot_points_on_image(projected_points, camera_image, title, rgba_func =rgba, point_size=5.0):
    """Plots points on a camera image.

    Args:
        projected_points: [N, 3] numpy array. The inner dims are
            [camera_x, camera_y, range].
        camera_image: jpeg encoded camera image.
        rgba_func: a function that generates a color from a range value.
        point_size: the point size.

    """
    plot_image(camera_image)

    xs = []
    ys = []
    colors = []

    for point in projected_points:
        xs.append(point[0])  # width, col
        ys.append(point[1])  # height, row
        colors.append(rgba_func(point[2]))

    plt.title(title)
    plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")
    plt.axis('off')
    plt.grid(visible=False)


class LabelVisualizer:
    # Get the color map for the NVIDIA labels
    COLOR_MAP_LABELS = cm.get_cmap('tab20')

    # TODO: currently only works for NVIDIA classes
    LABELCLASS_STRING_TO_LABELCLASS_ID = NvidiaLabelProcessor.LABELCLASS_STRING_TO_LABELCLASS_ID
    LABELCLASS_ID_TO_LABELCLASS_STRING = NvidiaLabelProcessor.LABELCLASS_ID_TO_LABELCLASS_STRING

    def __init__(self) -> None:
        """ 
        Visualizes the point cloud together with the labels. Currently only supports NVIDIA (classes)
        """

        # Initialize the visualizer and the data variables
        self.vis = ml3d.vis.Visualizer()
        self.data: list = []
        self.bounding_boxes: list = []

        # Initialize the classes
        self.lut = ml3d.vis.LabelLUT()
        for id, (key, value) in enumerate(self.LABELCLASS_STRING_TO_LABELCLASS_ID.items()):
            self.lut.add_label(value, key, self.COLOR_MAP_LABELS(id)[:3])

    @multimethod
    def add_pc(self, xyz: np.ndarray, intensity: np.ndarray, dynamic_flag: np.ndarray, timestamp: np.ndarray, frame_id: int) -> None:
        ''' Adds a single lidar point cloud to the visualizer (V2 data) '''

        # normalize timestamps to floating point [0,1]
        timestamp = (timestamp - timestamp.min())/ (timestamp.max() - timestamp.min())
        timestamp[np.isnan(timestamp)] = 0 # first spin has same timestamp for all points

        self.data.append({
            'name': str(frame_id),
            'points': xyz.astype(np.float32),
            'intensity': intensity.astype(np.float32),
            'dynamic_flag': dynamic_flag,
            'timestamp': timestamp.astype(np.float32)
        })

    @multimethod
    def add_labels(self, frame_labels: list[FrameLabel3]) -> None:
        ''' Registers frame-label bounding boxes (V2 data) '''
        for frame_label in frame_labels:
            self._add_bbox(bbox=frame_label.bbox3.to_array(),
                           label_class=frame_label.label_class,
                           identifier=frame_label.track_id,
                           confidence=frame_label.confidence)

    def _add_bbox(self, bbox: np.ndarray, label_class: str, identifier: str, confidence: float = 1.0) -> None:
        # TODO: This orientation seems correct to me, but we should double check it as the definition is weird

        orientation = R.from_euler('xyz', bbox[6:9], degrees=False).as_matrix()
        self.bounding_boxes.append(
            ml3d.vis.BoundingBox3D(center=bbox[:3],
                                   front=orientation[:, 0],
                                   up=orientation[:, 2],
                                   left=orientation[:, 1],
                                   size=np.array([bbox[4], bbox[5], bbox[3]]),
                                   label_class=label_class,
                                   confidence=confidence,
                                   identifier=identifier))

    def show(self) -> None:
        self.vis.visualize(self.data, self.lut, self.bounding_boxes)
