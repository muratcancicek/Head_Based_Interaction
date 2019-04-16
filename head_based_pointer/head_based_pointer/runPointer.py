from InputEstimators.HeadPoseEstimators.RegTreesHeadPoseEstimator import RegTreesHeadPoseEstimator
from InputEstimators.FacialLandmarkDetectors.DLIBFacialLandmarkDetector import DLIBFacialLandmarkDetector
from InputEstimators.FaceDetectors.DLIBFrontalFaceDetector import DLIBFrontalFaceDetector
from DemoHandler import DemoHandler

import cv2


def main():
    #estimator = RegTreesHeadPoseEstimator()
    estimator = DLIBFacialLandmarkDetector()
    #estimator = DLIBFrontalFaceDetector()
    #DemoHandler().printDemo(estimator)
    DemoHandler().streamDemo(estimator)

if __name__ == '__main__':
    main()
