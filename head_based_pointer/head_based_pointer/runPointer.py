from InputEstimators.HeadPoseEstimators.RegTreesHeadPoseEstimator import RegTreesHeadPoseEstimator
from InputEstimators.FacialLandmarkDetectors.DLIBFacialLandmarkDetector import DLIBFacialLandmarkDetector
from InputEstimators.FaceDetectors.DLIBFrontalFaceDetector import DLIBFrontalFaceDetector
from InputEstimators.FaceDetectors.CV2Res10SSDFaceDetector import CV2Res10SSDFaceDetector
from DemoHandler import DemoHandler

import cv2


def main():
    #estimator = RegTreesHeadPoseEstimator()
    estimator = CV2Res10SSDFaceDetector()

    faceDetector = DLIBFrontalFaceDetector()

    estimator = DLIBFacialLandmarkDetector(faceDetector)

    DemoHandler().play(estimator, printing = True, displaying = True)#)False

if __name__ == '__main__':
    main()
