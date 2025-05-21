from functools import lru_cache
from typing import List

import numpy as np

from .types import Descriptor
from .utils import patch_within_bounds


def generate_brief_pairs(n, patch_size, seed=42):
    # TODO: This gets called every frame, precompute
    rng = np.random.default_rng(seed)
    half = patch_size // 2
    # return n tuples (x1, y1, x2, y2)
    return rng.integers(-half, half + 1, size=(n, 4))


def get_brief_descriptors(
    image, keypoints, n=256, patch_size=31, rotated=False, seed=42
) -> List[Descriptor]:

    pairs = generate_brief_pairs(n, patch_size, seed)
    half = patch_size // 2
    descriptors = []

    pairs = np.array(pairs)
    x1s, y1s, x2s, y2s = pairs.T

    # for each kp, check n random pairs. The descriptor will have
    for kp in keypoints:
        if not patch_within_bounds(kp.x, kp.y, half, image):
            descriptors.append(None)
            continue

        patch = image[kp.y - half : kp.y + half + 1, kp.x - half : kp.x + half + 1]

        if rotated:
            cos, sin = np.cos(kp.angle), np.sin(kp.angle)
            x1r = np.round(x1s * cos - y1s * sin).astype(int)
            y1r = np.round(x1s * sin + y1s * cos).astype(int)
            x2r = np.round(x2s * cos - y2s * sin).astype(int)
            y2r = np.round(x2s * sin + y2s * cos).astype(int)
        else:
            x1r, y1r, x2r, y2r = x1s, y1s, x2s, y2s

        # mask for valid indices
        valid = (
            (np.abs(x1r) <= half)
            & (np.abs(y1r) <= half)
            & (np.abs(x2r) <= half)
            & (np.abs(y2r) <= half)
        )

        x1r += half
        y1r += half
        x2r += half
        y2r += half

        desc = np.zeros(n, dtype=np.uint8)
        p1 = patch[y1r[valid], x1r[valid]]
        p2 = patch[y2r[valid], x2r[valid]]
        desc[valid] = p1 < p2
        descriptors.append(Descriptor(bits=desc.astype(np.uint8)))
    return descriptors
