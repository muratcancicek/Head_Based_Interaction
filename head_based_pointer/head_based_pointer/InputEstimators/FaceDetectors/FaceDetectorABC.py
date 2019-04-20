from InputEstimators.InputEstimatorABC import InputEstimatorABC
from abc import abstractmethod
import numpy as np

class FaceDetectorABC(InputEstimatorABC):

    @abstractmethod
    def __init__(self, squaringFaceBox = False, *args, **kwargs):
        self._faceBox = None
        self._squaringFaceBox = squaringFaceBox
        self._faceLocation = np.zeros((3,))
        super().__init__(*args, **kwargs)
        
    @staticmethod
    @abstractmethod
    def _decodeFaceBox(self, detection):
        return NotImplemented
        
    @abstractmethod
    def _detectFaceBox(self, frame):
        return NotImplemented

    def detectFaceBox(self, frame):
        self._faceBox = self._detectFaceBox(frame)
        if self._faceBox != None and self._squaringFaceBox:
            self._faceBox = self._faceBox.getSquareFaceBoxOnFrame(frame)
        return self._faceBox

    def detectFaceImage(self, frame):
        self._faceBox = self.detectFaceBox(frame)
        if self._faceBox == None:
            return self._faceBox
        else:
            return self._faceBox.getFaceImageFromFrame(frame)

    def findFaceLocation(self, frame):
        self._faceBox = self.detectFaceBox(frame)
        if self._faceBox == None:
            return self._faceLocation
        else:
            self._faceLocation[0] = self._faceBox.location[0]
            self._faceLocation[1] = self._faceBox.location[1]
        return self._faceLocation
    
    def getProjectionPoints(self):
        if self._faceBox == None:
            return []
        return self._faceBox.getProjectionPoints()

    def findFaceLocationWithAnnotations(self, frame):
        return self.findFaceLocation(frame), self.getProjectionPoints(), [self._faceLocation[:2].astype(int)]

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