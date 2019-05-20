# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from abc import ABC, abstractmethod
from CommonTools.Boundary import Boundary
import numpy

class MappingABC(ABC):

    def __init__(self, inputEstimator, outputBoundaries, *args, **kwargs):
        self._inputEstimator = inputEstimator
        self._inputValues = numpy.zeros((3,))
        self._inputBoundaries = self._inputEstimator.getBoundariesForInputValues()
        self._outputValues = numpy.zeros((2,))
        self._outputBoundaries = outputBoundaries
        super().__init__()
        
    @abstractmethod
    def _calculate(self):
        raise NotImplementedError
        
    def _updateOutputValues(self):
        self._calculate()
        self._outputValues = self._outputBoundaries.keepInside(self._outputValues)
        return self._outputValues

    def calculateOutputValues(self, frame):
        self._inputBoundaries = self._inputEstimator.getBoundariesForInputValues()
        self._inputValues = self._inputEstimator.estimateInputValues(frame)
        return self._updateOutputValues()

    def calculateOutputValuesWithAnnotations(self, frame):
        annotations = self._inputEstimator.estimateInputValuesWithAnnotations(frame)
        self._inputBoundaries = self._inputEstimator.getBoundariesForInputValues()
        self._inputValues, projectionPoints, facial_landmarks = annotations
        self._updateOutputValues()
        return self._outputValues, self._inputValues, projectionPoints, facial_landmarks

    @property
    def inputValues(self):
        return self._inputValues

    @property
    def outputValues(self):
        return self._outputValues

    def getEstimator(self):
        return self._inputEstimator
