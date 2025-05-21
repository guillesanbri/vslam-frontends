# TODO: Limit maximum number of keypoints (adaptative grid)
# TODO: Add NMS to FAST
# TODO: Unit tests

from typing import List

import numpy as np

from .fast import get_fast_keypoints
from .brief import get_brief_descriptors
from .types import Feature
from .utils import patch_within_bounds


def get_keypoints_orientation(image, keypoints, patch_size=31) -> List[float]:
    """
    Return a list of angles for each keypoint in radians.
    Uses intensity centroid.
    """
    half = patch_size // 2
    orientations = []

    for kp in keypoints:
        if not patch_within_bounds(kp.x, kp.y, half, image):
            orientations.append(0.0)
            continue

        patch = image[
            kp.y - half : kp.y + half + 1, kp.x - half : kp.x + half + 1
        ].astype(np.float32)

        coords = np.arange(-half, half + 1)
        X, Y = np.meshgrid(coords, coords)

        m00 = np.sum(patch)
        if m00 == 0:
            orientations.append(0.0)
            continue

        m10 = np.sum(X * patch)
        m01 = np.sum(Y * patch)

        theta = np.arctan2(m01, m10)
        orientations.append(theta)

    return orientations


def get_orb_features(image) -> List[Feature]:
    """Returns keypoints and descriptors using Oriented FAST and Rotated BRIEF"""
    # detect keypoints
    keypoints = get_fast_keypoints(image)
    print(f"Num kps: {len(keypoints)}")
    # compute orientations and update keypoints
    keypoints_orientation = get_keypoints_orientation(image, keypoints)
    for kp, kp_o in zip(keypoints, keypoints_orientation):
        kp.angle = kp_o
    # compute descriptors and rotate them
    descriptors = get_brief_descriptors(image, keypoints, rotated=True)
    return [Feature(kp, d) for kp, d in zip(keypoints, descriptors)]
