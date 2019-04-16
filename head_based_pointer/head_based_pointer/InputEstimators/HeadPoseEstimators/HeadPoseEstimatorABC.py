from InputEstimators.InputEstimatorABC import InputEstimatorABC
from abc import abstractmethod
import numpy

class HeadPoseEstimatorABC(InputEstimatorABC):

    def __init__(self, *args, **kwargs):
        self._headPose3D = numpy.zeros((3,))
        
    @abstractmethod
    def calculateHeadPose(self, frame):
        return NotImplemented
    
    @abstractmethod
    def _calculateHeadPoseWithAnnotations(self, frame):
        return NotImplemented

    def estimateInputValues(self, frame):
        return self.calculateHeadPose(frame)
    
    def estimateInputValuesWithAnnotations(self, frame):
        return self._calculateHeadPoseWithAnnotations(frame)

    @property
    def inputValues(self):
        return self._headPose3D