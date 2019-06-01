from InputEstimators.InputEstimatorABC import InputEstimatorABC
from abc import abstractmethod
import numpy

class HeadPoseEstimatorABC(InputEstimatorABC):

    def __init__(self, faceDetector = None, landmarkDetector = None, poseCalculator = None, *args, **kwargs):
        self._faceDetector = faceDetector
        self._landmarkDetector = landmarkDetector
        self._poseCalculator = poseCalculator
        self._headPose3D = numpy.zeros((3,))
        
    @abstractmethod
    def calculateHeadPose(self, frame):
        raise NotImplementedError
    
    @abstractmethod
    def _calculateHeadPoseWithAnnotations(self, frame):
        raise NotImplementedError
    
    def estimateInputValues(self, frame):
        return self.calculateHeadPose(frame)
    
    def estimateInputValuesWithAnnotations(self, frame):
        return self._calculateHeadPoseWithAnnotations(frame)

    @property
    def inputValues(self):
        return self._headPose3D

    def returns3D(self):
        return True
