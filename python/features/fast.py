import cv2
import time


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

    opencv_fast(img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
