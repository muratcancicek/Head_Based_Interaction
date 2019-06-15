# The code is derived from the following repository:
# https://github.com/yinguobing/head-pose-estimation

from InputEstimators.HeadPoseEstimators.PoseCalculators.MuratcansHeadGazeCalculator import MuratcansHeadGazeCalculator
from InputEstimators.HeadPoseEstimators.YinsHeadPoseEstimator import YinsHeadPoseEstimator
from imutils import face_utils
import numpy as np
import dlib
import cv2

class MuratcansHeadGazer(YinsHeadPoseEstimator):
    def __init__(self, faceDetector = None, landmarkDetector = None, poseCalculator = None, face_landmark_path = None, inputFramesize = (1920, 1080), *args, **kwargs):
        if poseCalculator == None:
            poseCalculator = MuratcansHeadGazeCalculator(inputFramesize = inputFramesize)
        self._pPoints = np.zeros((1, 2))
        self._gazingFrameSize = inputFramesize
        self._halfFrameHeight = inputFramesize[1]/2
        super().__init__(faceDetector, landmarkDetector, poseCalculator, face_landmark_path, inputFramesize, *args, **kwargs)
        
    def calculateHeadPose(self, frame):
        self._landmarks = self._landmarkDetector.detectFacialLandmarks(frame)
        if len(self._landmarks) == 0:
            return self._headPose3D
        else:
            self._headPose3D = self._poseCalculator.calculatePose(self._landmarks)
            return self._headPose3D
                    
    def calculateHeadGaze(self, frame):
        self._landmarks = self._landmarkDetector.detectFacialLandmarks(frame)
        if len(self._landmarks) != 0:
            self._halfFrameHeight = frame.shape[0]/2
            g = self._poseCalculator.calculateHeadGazeWithProjectionPoints(self._landmarks) 
            self._headPose3D, self._pPoints = g
            return self._headPose3D
            
    def _calculateHeadPoseWithAnnotations(self, frame):
        self._headPose3D = self.calculateHeadGaze(frame)
        return self._headPose3D, self._pPoints, self._landmarks
    
    def getGazingFrameDimensions(self):
        #return int(1920), int(self._halfFrameHeight + 1080)
        #print(self._gazingFrameSize)
        return self._gazingFrameSize