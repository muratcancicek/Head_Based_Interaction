# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from InputEstimators.HeadPoseEstimators.HeadPoseEstimatorABC import HeadPoseEstimatorABC
from InputEstimators.HeadPoseEstimators.MuratcansHeadGazer import MuratcansHeadGazer
from InputEstimators.FacialLandmarkDetectors.FacialLandmarkDetectorABC import FacialLandmarkDetectorABC
from InputEstimators.FaceDetectors.FaceDetectorABC import FaceDetectorABC
from InputEstimators.FaceDetectors.FaceBox import FaceBox

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
        currentFaceBox = self._inputEstimator.faceBox
        left, right = currentFaceBox.left, currentFaceBox.right
        top, bottom = currentFaceBox.top, currentFaceBox.bottom
        if self._inputBoundaries == None:
                self._inputBoundaries = Boundary(left, right, top, bottom)
                self._faceBoxForInput = currentFaceBox
        x, y = self._inputValues[:2]
        update = False
        if not self._inputBoundaries.isInRanges(x = x):
            update = True
            if x - self._faceBoxForInput.location[0] > 0:
                left, right = x - (right - left), x
            else:
                left, right = x, x + (right - left)
            top = self._faceBoxForInput.top
            bottom = self._faceBoxForInput.bottom
        if not self._inputBoundaries.isInRanges(y = y):
            update = True
            if y - self._faceBoxForInput.location[1] > 0:
                top, bottom = y - (bottom - top), y
            else:
                top, bottom = y, y + (bottom - top)
            left = self._faceBoxForInput.left
            right = self._faceBoxForInput.right
        if update:
            self._inputBoundaries = Boundary(left, right, top, bottom)
            self._faceBoxForInput = FaceBox(int(left), int(top),
                                           int(right), int(bottom))
        self._pPoints = self._faceBoxForInput.getProjectionPoints()
        return self._inputValues
        
    def _calculateInputValuesFromNose(self):
        minX, maxX = self._Landmarks[49, 0], self._Landmarks[53, 0]
        minY = (self._Landmarks[1, 1] + self._Landmarks[15, 1])/2
        maxY = (self._Landmarks[4, 1] + self._Landmarks[12, 1])/2
        self._inputValues[:2] = self._inputValues[:2] - (minX, minY)
        minX, maxX, minY, maxY =  0, maxX - minX, 0, maxY  - minY
        self._inputBoundaries = Boundary(minX, maxX, minY, maxY)
        return self._inputValues
    
    def _calculateInputValuesFromHeadPose(self):
        return self._inputValues

    def _calculateInputValuesFromHeadGaze(self):
        return self._inputValues

    def _recalculateInputValues(self):
        raise NotImplementedError
           
    def _initializeInputCalculator(self):
        self._outputDependsAnnotations = False
        self._inputBoundaries = Boundary()
        if isinstance(self._inputEstimator, FaceDetectorABC):
            self._recalculateInputValues = self._calculateInputValuesFromFaceBox
            self._inputBoundaries = None
        elif isinstance(self._inputEstimator, FacialLandmarkDetectorABC):
            self._recalculateInputValues = self._calculateInputValuesFromNose
            self._outputDependsAnnotations = True
        elif isinstance(self._inputEstimator, HeadPoseEstimatorABC):
            self._inputBoundaries = Boundary(40, 60, -10, 5)
            self._recalculateInputValues = self._calculateInputValuesFromHeadPose
        elif isinstance(self._inputEstimator, MuratcansHeadGazer):
            width, height = self._inputEstimator.getGazingFrameDimensions()
            self._inputBoundaries = Boundary(-640, 1280, 0, height)
            self._recalculateInputValues = self._calculateInputValuesFromHeadGaze

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

    def getOutputBoundaries(self):
        return self._outputBoundaries

    def getInputBoundaries(self):
        return self._inputBoundaries
