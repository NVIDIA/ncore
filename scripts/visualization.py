# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d.ml.tf as ml3d

from matplotlib import cm
from matplotlib import pyplot as plt
from multimethod import multimethod
from scipy.spatial.transform import Rotation as R

from ncore.impl.data.types import BBox3


def rgba(r):
    """Generates a color based on range.

    Args:
            r: the range value of a given point.
    Returns:
            The color for a given range
    """
    c = plt.get_cmap("jet")((r % 50.0) / 50.0)
    c = list(c)
    c[-1] = 0.5  # alpha
    return c


def plot_image(camera_image):
    """Plot a camera image."""
    plt.figure(figsize=(20, 12))
    plt.imshow(camera_image)
    plt.grid(visible=False)


def plot_points_on_image(
    projected_points, camera_image, title="", rgba_func=rgba, point_size=5.0, show=True, save_path: Optional[str] = None
):
    """Plots points on a camera image.

    Args:
        projected_points: [N, 3] numpy array. The inner dims are
            [camera_x, camera_y, range].
        camera_image: jpeg encoded camera image.
        title: if given, the title to add to the plot.
        rgba_func: a function that generates a color from a range value.
        point_size: the point size.
        show: whether to show the plot.
        save_path: filename to store the plot to if provided.

    """
    plt.clf()
    plot_image(camera_image)

    xs = []
    ys = []
    colors = []

    for point in projected_points:
        xs.append(point[0])  # width, col
        ys.append(point[1])  # height, row
        colors.append(rgba_func(point[2]))

    if len(title):
        plt.title(title)

    plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")
    plt.axis("off")
    plt.grid(visible=False)

    if show:
        plt.show()

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)


class LabelVisualizer:
    # Get the color map for the NVIDIA labels
    COLOR_MAP_LABELS = cm.get_cmap("tab20")

    # TODO: currently only works for NVIDIA classes
    LABELCLASS_STRING_TO_LABELCLASS_ID: dict[str, int] = {
        "unknown": 0,
        "automobile": 1,
        "pedestrian": 2,
        "sign": 3,
        "CYCLIST": 4,
        "heavy_truck": 5,
        "bus": 6,
        "other_vehicle": 7,
        "motorcycle": 8,
        "motorcycle_with_rider": 9,
        "person": 10,
        "rider": 11,
        "bicycle_with_rider": 12,
        "bicycle": 13,
        "stroller": 14,
        "person_group": 15,
        "unclassifiable_vehicle": 16,
        "cycle": 17,
        "trailer": 18,
        "protruding_object": 19,
        "animal": 20,
        "train_or_tram_car": 21,
    }
    LABELCLASS_ID_TO_LABELCLASS_STRING: dict[int, str] = {v: k for k, v in LABELCLASS_STRING_TO_LABELCLASS_ID.items()}

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
    def add_pc(
        self,
        frame_id: int,
        xyz: np.ndarray,
        intensity: np.ndarray,
        timestamp: np.ndarray,
        dynamic_flag: Optional[np.ndarray],
        semantic_class: Optional[np.ndarray],
    ) -> None:
        """Adds a single lidar point cloud to the visualizer"""

        pc = {
            "name": str(frame_id),
            "points": xyz.astype(np.float32),
            "intensity": intensity.astype(np.float32),
        }

        # normalize timestamps to floating point [0,1]
        timestamp_normalized = (timestamp - timestamp.min()) / (timestamp.max() - timestamp.min())
        timestamp_normalized[np.isnan(timestamp_normalized)] = 0  # first spin could have same timestamp for all points
        pc["timestamp"] = timestamp_normalized.astype(np.float32)

        if semantic_class is not None:
            pc["semantic_class"] = semantic_class

        if dynamic_flag is not None:
            pc["dynamic_flag"] = dynamic_flag

        self.data.append(pc)

    @dataclass
    class BBox3Label:
        """Description of a 3D frame-associated label"""

        track_id: str  #: Unique identifier of the object's track this label is associated with
        label_class: str  #: String-representation of the class associated with this label
        bbox3: BBox3  #: Bounding-box coordinates of the object relative to the frame's end-of-frame coordinate system
        timestamp_us: (
            int  #: The timestamp associated with the centroid of the label (possibly an accurate in-frame time)
        )
        confidence: Optional[float]  #: If available, the confidence score of the label [0..1]

        def __post_init__(self):
            # Sanity checks
            assert isinstance(self.track_id, str)
            assert isinstance(self.label_class, str)
            assert isinstance(self.bbox3, BBox3)
            assert isinstance(self.timestamp_us, int)
            assert isinstance(self.confidence, (type(None), float))

    @multimethod
    def add_labels(self, frame_labels: list[BBox3Label]) -> None:
        """Registers frame-label bounding boxes to be visualized."""
        for frame_label in frame_labels:
            self._add_bbox(
                bbox=frame_label.bbox3.to_array(),
                label_class=frame_label.label_class,
                identifier=f"{frame_label.track_id}_{frame_label.timestamp_us}",
                confidence=frame_label.confidence if frame_label.confidence else 1.0,
            )

    def _add_bbox(self, bbox: np.ndarray, label_class: str, identifier: str, confidence: float = 1.0) -> None:
        orientation = R.from_euler("xyz", bbox[6:9], degrees=False).as_matrix()
        self.bounding_boxes.append(
            ml3d.vis.BoundingBox3D(
                center=bbox[:3],
                front=orientation[:, 0],
                up=orientation[:, 2],
                left=orientation[:, 1],
                size=np.array([bbox[4], bbox[5], bbox[3]]),
                label_class=label_class,
                confidence=confidence,
                identifier=identifier,
            )
        )

    def show(self) -> None:
        self.vis.visualize(self.data, self.lut, self.bounding_boxes)
