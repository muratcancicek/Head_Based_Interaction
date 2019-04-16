from InputEstimators.FacialLandmarkDetectors.DLIBFacialLandmarkDetector import DLIBFacialLandmarkDetector
from InputEstimators.HeadPoseEstimators.CV2_PnP_PoseCalculator import CV2_PnP_PoseCalculator
from InputEstimators.FaceDetectors.DLIBFrontalFaceDetector import DLIBFrontalFaceDetector
from InputEstimators.HeadPoseEstimators.HeadPoseEstimatorABC import HeadPoseEstimatorABC
from imutils import face_utils
import numpy as np
import dlib
import cv2

class RegTreesHeadPoseEstimator(HeadPoseEstimatorABC):
    def __init__(self, face_landmark_path = None, *args, **kwargs):

        if face_landmark_path == None:
            self.__face_landmark_path = 'C:/cStorage/Datasets/Dlib/shape_predictor_68_face_landmarks.dat'

        self.__detector = DLIBFrontalFaceDetector()
        self.__predictor = DLIBFacialLandmarkDetector(faceDetector = self.__detector, face_landmark_path = self.__face_landmark_path)
        self.__poseCalculator = CV2_PnP_PoseCalculator()
        self._headPose3D = np.zeros((3,))
        super().__init__(*args, **kwargs)
    
    def calculateHeadPose(self, frame):
        self.__facial_landmarks = self.__predictor.detectFacialLandmarks(frame)
        if len(self.__facial_landmarks) == 0:
            return self._headPose3D
        else:
            self._headPose3D = self.__poseCalculator.calculatePoseFromShape(self.__facial_landmarks)
            return self._headPose3D
            
    def _calculateHeadPoseWithAnnotations(self, frame):
        self.__facial_landmarks = self.__predictor.detectFacialLandmarks(frame)
        self._headPose3D = self.__poseCalculator.calculatePoseFromShape(self.__facial_landmarks)
        self.__projectionPoints = self.__poseCalculator.calculateProjectionPointsFromShape(self.__facial_landmarks)
        return self._headPose3D, self.__projectionPoints, self.__facial_landmarks

    @property
    def headPose(self):
        return self._headPose3D