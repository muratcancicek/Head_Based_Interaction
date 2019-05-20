# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from InputEstimators.InputEstimatorABC import InputEstimatorABC
from abc import abstractmethod
import numpy

class FacialLandmarkDetectorABC(InputEstimatorABC):

    def __init__(self, faceDetector = None, inputLandmarkIndex = 30, *args, **kwargs):
        self._facialLandmarks = []
        self._inputLandmarkIndex = inputLandmarkIndex
        self._inputLandmark = numpy.zeros((3,))
        self._faceDetector = faceDetector
        super().__init__(*args, **kwargs)
        
    @classmethod
    @abstractmethod
    def detectFacialLandmarks(self, frame):
        raise NotImplementedError
    
    @property
    def facialLandmarks(self):
            return self._facialLandmarks

    def findInputLandmarkLocation(self, frame):
        landmarks = self.detectFacialLandmarks(frame)
        if len(landmarks) > self._inputLandmarkIndex:
            self._facialLandmarks = landmarks
            self._inputLandmark[:2] = self._facialLandmarks[self._inputLandmarkIndex]
        return self._inputLandmark
            
    def estimateInputValues(self, frame):
        self._updateBoundariesForInputValues(0, frame.shape[1], 0, frame.shape[0], 0, 0)
        return self.findInputLandmarkLocation(frame)
    
    def findInputLandmarkLocationWithAnnotations(self, frame):
        return self.findInputLandmarkLocation(frame), self._faceDetector.getProjectionPoints(), self._facialLandmarks

    def estimateInputValuesWithAnnotations(self, frame):
        self._updateBoundariesForInputValues(0, frame.shape[1], 0, frame.shape[0], 0, 0)
        return self.findInputLandmarkLocationWithAnnotations(frame)

    @property
    def inputValues(self):
        return self._inputLandmark

    def returns3D(self):
        return False