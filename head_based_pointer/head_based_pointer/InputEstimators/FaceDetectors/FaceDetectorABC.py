from InputEstimators.InputEstimatorABC import InputEstimatorABC
from abc import abstractmethod
import numpy as np

class FaceDetectorABC(InputEstimatorABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self._faceBox = None
        self._faceLocation = np.zeros((3,))
        super().__init__(*args, **kwargs)
        
    @abstractmethod
    def _decodeFaceBox(self, detection):
        return NotImplemented
        
    @abstractmethod
    def detectFaceBox(self, frame):
        return NotImplemented
        
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
        return self.findFaceLocation(frame), self.getProjectionPoints(), []

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
               
    class FaceBox(object):
        def __init__(self, left, top, right, bottom, *args, **kwargs):
            self.left = left
            self.top = top
            self.right = right
            self.bottom = bottom
            self._tl_corner = (left, top)
            self._tr_corner = (right, top)
            self._bl_corner = (left, bottom)
            self._br_corner = (right, bottom)
            self.location = (left + abs(right - left)/2, top + abs(bottom - top)/2)
            super().__init__(*args, **kwargs)
    
        def getProjectionPoints(self):
            corners = [self._tl_corner, self._tr_corner, self._br_corner, self._bl_corner]
            return [(corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[3]), (corners[3], corners[0])]
