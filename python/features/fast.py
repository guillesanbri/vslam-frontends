import cv2
import time
import numpy as np

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


def fast(img, thr=10):

    print("=" * 80)
    print("Scratch FAST Detector")
    print("-" * 80)

    kp = []
    h, w = img.shape
    img_float = img.astype(np.float32)
    t0 = time.perf_counter()
    for x in range(3, w - 3):
        for y in range(3, h - 3):
            if fast_check(img_float, x, y, thr):
                kp.append((x, y))

    # TODO: add nms

    t1 = time.perf_counter()
    print(f"Number of kp found (nms): {len(kp)}, took {(t1 - t0)*1000:.3f} ms")

    kp = [
        cv2.KeyPoint(_kp[0], _kp[1], size=7) for _kp in kp
    ]  # size is diameter used in drawKeypoints
    scratch_fast_nms = cv2.drawKeypoints(img, kp, None, (0, 255, 0))
    cv2.imshow("Scratch FAST original img", img)
    cv2.imshow("Scratch FAST with NMS", scratch_fast_nms)


def opencv_fast(img):
    fast = cv2.FastFeatureDetector_create()
    print("=" * 80)
    print("OpenCV FAST Detector")
    print("-" * 80)

    t0 = time.perf_counter()
    kp = fast.detect(img, None)
    t1 = time.perf_counter()
    print(f"Number of kp found (nms): {len(kp)}, took {(t1 - t0)*1000:.3f} ms")
    opencv_fast_nms = cv2.drawKeypoints(img, kp, None, (0, 255, 0))

    sorted_kp = sorted(kp, key=lambda k: k.response, reverse=True)
    sorted_kp = sorted_kp[:100]
    opencv_fast_nms_top_100 = cv2.drawKeypoints(img, sorted_kp, None, (0, 255, 0))

    fast.setNonmaxSuppression(0)
    t0 = time.perf_counter()
    kp = fast.detect(img, None)
    t1 = time.perf_counter()
    print(f"Number of kp found (no nms): {len(kp)}, took {(t1 - t0)*1000:.3f} ms")
    opencv_fast_no_nms = cv2.drawKeypoints(img, kp, None, (0, 255, 0))

    print("=" * 80)
    cv2.imshow("OpenCV FAST original img", img)
    cv2.imshow("OpenCV FAST with NMS", opencv_fast_nms)
    cv2.imshow("OpenCV FAST (top 100)", opencv_fast_nms_top_100)
    cv2.imshow("OpenCV FAST without NMS", opencv_fast_no_nms)


def main():
    img = cv2.imread(
        "../../sample_data/EuRoC_MAV/mav0/cam0/data/1403636579763555584.png"
    )
    if img is None:
        print("Img not found!")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    opencv_fast(img)
    fast(img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
