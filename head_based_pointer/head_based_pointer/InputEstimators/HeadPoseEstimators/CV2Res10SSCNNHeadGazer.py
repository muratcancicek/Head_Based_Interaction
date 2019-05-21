# The code is derived from the following repository:
# https://github.com/yinguobing/head-pose-estimation

from InputEstimators.HeadPoseEstimators.PoseCalculators.MuratcansHeadGazeCalculator import MuratcansHeadGazeCalculator
from InputEstimators.HeadPoseEstimators.CV2Res10SSCNNHeadPoseEstimator import CV2Res10SSCNNHeadPoseEstimator
from imutils import face_utils
import numpy as np
import dlib
import cv2

class CV2Res10SSCNNHeadGazer(CV2Res10SSCNNHeadPoseEstimator):
    def __init__(self, faceDetector = None, landmarkDetector = None, poseCalculator = None, face_landmark_path = None, inputFramesize = (640, 360), *args, **kwargs):
        if poseCalculator == None:
            poseCalculator = MuratcansHeadGazeCalculator(inputFramesize = inputFramesize)
        super().__init__(faceDetector, landmarkDetector, poseCalculator, face_landmark_path, inputFramesize, *args, **kwargs)
        
    def calculateHeadPose(self, frame):
        self.__facial_landmarks = self._landmarkDetector.detectFacialLandmarks(frame)
        if len(self.__facial_landmarks) == 0:
            return self._headPose3D
        else:
            self._headPose3D = self._poseCalculator.calculatePose(self.__facial_landmarks)
            return self._headPose3D
                    
    def calculateHeadGaze(self, frame):
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
    