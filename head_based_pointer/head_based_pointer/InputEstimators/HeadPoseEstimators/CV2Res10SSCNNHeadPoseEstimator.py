# The code is derived from the following repository:
# https://github.com/yinguobing/head-pose-estimation

from InputEstimators.HeadPoseEstimators.PoseCalculators.YinsKalmanFilteredHeadPoseCalculator import YinsKalmanFilteredHeadPoseCalculator
from InputEstimators.FacialLandmarkDetectors.YinsCNNBasedFacialLandmarkDetector import YinsCNNBasedFacialLandmarkDetector
from InputEstimators.FaceDetectors.CV2Res10SSDFaceDetector import CV2Res10SSDFaceDetector
from InputEstimators.HeadPoseEstimators.HeadPoseEstimatorABC import HeadPoseEstimatorABC
from imutils import face_utils
import numpy as np
import dlib
import cv2

class CV2Res10SSCNNHeadPoseEstimator(HeadPoseEstimatorABC):
    def __init__(self, faceDetector = None, landmarkDetector = None, poseCalculator = None, face_landmark_path = None, inputFramesize = (720, 1280), *args, **kwargs):
        if landmarkDetector == None:
            if faceDetector == None:
                faceDetector = CV2Res10SSDFaceDetector(squaringFaceBox = True)
            landmarkDetector = YinsCNNBasedFacialLandmarkDetector(faceDetector)
        if poseCalculator == None:
            poseCalculator = YinsKalmanFilteredHeadPoseCalculator(inputFramesize = inputFramesize)
        self._headPose3D = np.zeros((3,))
        super().__init__(faceDetector, landmarkDetector, poseCalculator, *args, **kwargs)
    
    def calculateHeadPose(self, frame):
        self.__facial_landmarks = self._landmarkDetector.detectFacialLandmarks(frame)
        if len(self.__facial_landmarks) == 0:
            return self._headPose3D
        else:
            self._headPose3D = self._poseCalculator.calculatePose(self.__facial_landmarks)
            return self._headPose3D
            
    def _calculateHeadPoseWithAnnotations(self, frame):
        self._headPose3D = self.calculateHeadPose(frame)
        self.__projectionPoints = self._poseCalculator.calculateProjectionPoints(self.__facial_landmarks)
        return self._headPose3D, self.__projectionPoints, self.__facial_landmarks

    @property
    def headPose(self):
        return self._headPose3D