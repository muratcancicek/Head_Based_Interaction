from abc import ABC, abstractmethod
import numpy

class InputEstimatorABC(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self._inputValues = numpy.zeros((3,))
        
    @abstractmethod
    def estimateInputValues(self, frame):
        return self._inputValues

    @abstractmethod
    def estimateInputValuesWithAnnotations(self, frame):
        return self._inputValues, [], []

    @property
    @abstractmethod
    def inputValues(self):
        return self._inputValues