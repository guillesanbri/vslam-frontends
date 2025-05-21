import os
import cv2

from .base import BaseStereoSource


class EuRoCMAVDataset(BaseStereoSource):
    def __init__(self, root_dir, max_frames=None):
        super().__init__()
        self.left_dir = os.path.join(root_dir, "cam0", "data")
        self.right_dir = os.path.join(root_dir, "cam1", "data")
        # load image names
        self.left_images = sorted(os.listdir(self.left_dir))
        self.right_images = sorted(os.listdir(self.right_dir))

        assert len(self.left_images) == len(
            self.right_images
        ), "Number of images must be the same."

        self.max_frames = max_frames or len(self.left_images)

    def __len__(self):
        return min(self.max_frames, len(self.left_images))

    def get_frame(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range!")

        left_path = os.path.join(self.left_dir, self.left_images[idx])
        right_path = os.path.join(self.right_dir, self.right_images[idx])

        left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

        return left_img, right_img
