from InputEstimators.InputEstimatorABC import InputEstimatorABC
from abc import abstractmethod
import numpy as np

class FaceDetectorABC(InputEstimatorABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self._faceBox = []
        self._faceLocation = np.zeros((3,))
        super().__init__(*args, **kwargs)

    @abstractmethod
    def detectFaceBox(self, frame):
        return NotImplemented
        
    @abstractmethod
    def findFaceLocation(self, frame):
        return NotImplemented
    
    @abstractmethod
    def getProjectionPoints(self):
        return NotImplemented

    @abstractmethod
    def findFaceLocationWithAnnotations(self, frame):
        return NotImplemented

    @property
    def faceBox(self):
        return self._faceBox
            
    def estimateInputValues(self, frame):
        return self.findFaceLocation(frame)

    def estimateInputValuesWithAnnotations(self, frame):
        return self.findFaceLocationWithAnnotations(frame)

    @property
    def inputValues(self):
        return self._faceLocation
               