from InputEstimators.InputEstimatorABC import InputEstimatorABC
from abc import abstractmethod
import numpy

class FacialLandmarkDetectorABC(InputEstimatorABC):

    def __init__(self, *args, **kwargs):
        self._facialLandmarks = []
        self._inputLandmark = numpy.zeros((3,))
        self._faceDetector = None
        
    @classmethod
    @abstractmethod
    def detectFacialLandmarks(self, frame):
        return NotImplemented
    
    @abstractmethod
    def _findInputLandmark(self, frame):
        return NotImplemented
    
    @abstractmethod
    def _findInputLandmarkWithAnnotations(self, frame):
        return NotImplemented

    @property
    def facialLandmarks(self):
            return self._facialLandmarks

    def estimateInputValues(self, frame):
        return self._findInputLandmark(frame)
    
    def estimateInputValuesWithAnnotations(self, frame):
        return self._findInputLandmarkWithAnnotations(frame)

    @property
    def inputValues(self):
        return self._inputLandmark