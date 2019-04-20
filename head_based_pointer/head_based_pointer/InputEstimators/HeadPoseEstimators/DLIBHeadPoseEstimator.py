from InputEstimators.FacialLandmarkDetectors.DLIBFacialLandmarkDetector import DLIBFacialLandmarkDetector
from InputEstimators.HeadPoseEstimators.PoseCalculators.CV2_PnP_HeadPoseCalculator import CV2_PnP_HeadPoseCalculator
from InputEstimators.FaceDetectors.DLIBFrontalFaceDetector import DLIBFrontalFaceDetector
from InputEstimators.HeadPoseEstimators.HeadPoseEstimatorABC import HeadPoseEstimatorABC
from imutils import face_utils
import numpy as np
import dlib
import cv2

class DLIBHeadPoseEstimator(HeadPoseEstimatorABC):
    def __init__(self, faceDetector = None, landmarkDetector = None, poseCalculator = None, face_landmark_path = None, *args, **kwargs):
        if landmarkDetector == None:
            if faceDetector == None:
                faceDetector = DLIBFrontalFaceDetector()
            landmarkDetector = DLIBFacialLandmarkDetector(faceDetector)
        if poseCalculator == None:
            poseCalculator = CV2_PnP_HeadPoseCalculator()
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