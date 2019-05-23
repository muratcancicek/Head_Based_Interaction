# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from InputEstimators.HeadPoseEstimators.HeadPoseEstimatorABC import HeadPoseEstimatorABC
from InputEstimators.HeadPoseEstimators.MuratcansHeadGazer import MuratcansHeadGazer
from InputEstimators.FacialLandmarkDetectors.FacialLandmarkDetectorABC import FacialLandmarkDetectorABC
from InputEstimators.FaceDetectors.FaceDetectorABC import FaceDetectorABC


from abc import ABC, abstractmethod
from CommonTools.Boundary import Boundary
import numpy

class MappingABC(ABC):

    def __init__(self, inputEstimator, outputBoundaries, *args, **kwargs):
        self._inputEstimator = inputEstimator
        self._initializeInputCalculator()
        self._inputValues = numpy.zeros((3,))
        self._outputValues = numpy.zeros((2,))
        self._outputBoundaries = outputBoundaries
        super().__init__()
    
    def _calculateInputValuesFromFaceBox(self):
        #if self._lastFaceBox == None:
        #    self._lastFaceBox = self._inputEstimator.faceBox
        #    self._minX, self._minY = 0, 0
        #self._inputValues[:2] = self._inputValues[:2] - (self._minX, self._minY)
        #if self._inputBoundaries.isIn(self._inputValues):
        #    print('\r', 'p','\t', end= '\r' )
        #    return self._inputValues
        ##print('\r', 'o','\t', end= '\r' )
        #minX, maxX = self._lastFaceBox.left, self._lastFaceBox.right
        #minY, maxY = self._lastFaceBox.top, self._lastFaceBox.bottom
        #width, height = maxX - minY, maxY - minY
        #self._minX, maxX = minX - 2 * width, maxX + 2 * width
        #self._minY, maxY = minY - 1/2 * height, maxY + 1/2 * height
        #minX, maxX, minY, maxY =  0, maxX - minX, 0, maxY  - minY 
        #print('\r', 'o', self._inputValues, minX, maxX, minY, maxY,'\t', end= '\r' )
        #self._inputBoundaries = Boundary(minX, maxX, minY, maxY, 0, 0)
        #self._lastFaceBox = self._inputEstimator.faceBox
        return self._inputValues
    
    def _calculateInputValuesFromNose(self):
        minX, maxX = self._Landmarks[49, 0], self._Landmarks[53, 0]
        minY = (self._Landmarks[1, 1] + self._Landmarks[15, 1])/2
        maxY = (self._Landmarks[4, 1] + self._Landmarks[12, 1])/2
        self._inputValues[:2] = self._inputValues[:2] - (minX, minY)
        minX, maxX, minY, maxY =  0, maxX - minX, 0, maxY  - minY
        self._inputBoundaries = Boundary(minX, maxX, minY, maxY)
        return self._inputValues

    def _recalculateInputValues(self):
        raise NotImplementedError
           
    def _initializeInputCalculator(self):
        self._outputDependsAnnotations = False
        self._inputBoundaries = Boundary()
        if isinstance(self._inputEstimator, FaceDetectorABC):
            self._recalculateInputValues = self._calculateInputValuesFromFaceBox
            self._lastFaceBox = None
            self._inputBoundaries = Boundary(0, 0, 0, 0, 0, 0)
        elif isinstance(self._inputEstimator, FacialLandmarkDetectorABC):
            self._recalculateInputValues = self._calculateInputValuesFromNose
            self._outputDependsAnnotations = True
        elif isinstance(self._inputEstimator, MuratcansHeadGazer):
            self._inputBoundaries = Boundary(0, 1920, 0, 1260)
            self._recalculateInputValues = self._calculateInputValuesFromFaceBox
        elif isinstance(self._inputEstimator, HeadPoseEstimatorABC):
            self._recalculateInputValues = self._calculateInputValuesFromFaceBox

    def _estimateInput(self, frame):
        if self._outputDependsAnnotations:
            annos = self._inputEstimator.estimateInputValuesWithAnnotations(frame)
            self._inputValues, self._pPoints, self._Landmarks = annos
        else:
            self._inputValues = self._inputEstimator.estimateInputValues(frame)
        return self._inputValues 

    @abstractmethod
    def _calculate(self):
        raise NotImplementedError
        
    def _updateOutputValues(self):
        self._calculate()
        self._outputValues = self._outputBoundaries.keepInside(self._outputValues)
        return self._outputValues

    def calculateOutputValues(self, frame):
        self._estimateInput(frame)
        self._recalculateInputValues()
        return self._updateOutputValues()

    def calculateOutputValuesWithAnnotations(self, frame):
        self._outputDependsAnnotations = True
        self._inputBoundaries = Boundary(0, frame.shape[0], 0 , frame.shape[1])
        self.calculateOutputValues(frame)
        return self._outputValues, self._inputValues, self._pPoints, self._Landmarks

    @property
    def inputValues(self):
        return self._inputValues

    @property
    def outputValues(self):
        return self._outputValues

    def getEstimator(self):
        return self._inputEstimator