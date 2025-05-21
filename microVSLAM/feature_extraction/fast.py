from typing import List

import numpy as np

from .types import Keypoint

CIRCLE_OFFSETS = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
]


def fast_check(img, x, y, thr) -> bool:
    brighter = 0
    darker = 0
    center_value = img[y][x]

    # compass coordinates
    first_pass_indices = [1, 5, 9, 13]
    for i in first_pass_indices:
        i_value = img[y + CIRCLE_OFFSETS[i - 1][1]][x + CIRCLE_OFFSETS[i - 1][0]]
        if i_value > center_value + thr:
            brighter += 1
        elif i_value < center_value - thr:
            darker += 1

    second_pass_indices = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]
    if brighter >= 3 or darker >= 3:
        # full check
        for i in second_pass_indices:
            i_value = img[y + CIRCLE_OFFSETS[i - 1][1]][x + CIRCLE_OFFSETS[i - 1][0]]
            if i_value > center_value + thr:
                brighter += 1
            elif i_value < center_value - thr:
                darker += 1

        # TODO: Check for continuity, this is not correct
        return brighter >= 9 or darker >= 9

    return False


def get_fast_keypoints(image, thr=10) -> List[Keypoint]:
    kps = []
    h, w = image.shape
    img_float = image.astype(np.float32)
    for x in range(3, w - 3):
        for y in range(3, h - 3):
            if fast_check(img_float, x, y, thr):
                kps.append(Keypoint(x=x, y=y))
    return kps
