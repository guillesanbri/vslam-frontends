import cv2


def main():
    img = cv2.imread("../sample_data/EuRoC_MAV/mav0/cam0/data/1403636579763555584.png")
    if img is None:
        print("Img not found!")
        return
    cv2.imshow("Sample img", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
