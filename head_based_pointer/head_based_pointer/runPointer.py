from HeadPoseEstimator import HeadPoseEstimator
import cv2


def main():
    # return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return
    headPoseEstimator = HeadPoseEstimator()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            euler_angle = headPoseEstimator.get_head_pose_in_euler_angle(frame)
            print('\r%.2f |%.2f | %.2f ' % (euler_angle[0, 0], euler_angle[1, 0], euler_angle[2, 0]), end = '\r')

if __name__ == '__main__':
    main()
