import cv2
import numpy as np

from sources.dataset_euroc_mav import EuRoCMAVDataset


def main():
    dataset_path = "../sample_data/EuRoC_MAV/mav0"
    source = EuRoCMAVDataset(dataset_path)

    print("Playing sequences, press ESC to exit...")
    for l, r in source:
        if l.shape != r.shape:
            print("Image shapes do not match!")
            exit()

        pair = np.hstack((l, r))
        cv2.imshow("Stereo Pair", pair)

        k = cv2.waitKey(30)
        if k == 27:
            break


if __name__ == "__main__":
    main()
