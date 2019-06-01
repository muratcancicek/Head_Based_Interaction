# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from abc import ABC, abstractmethod
from CommonTools.Boundary import Boundary
import numpy

class InputEstimatorABC(ABC):

    @abstractmethod
    def __init__(self, boundary = None, *args, **kwargs):
        self._inputValues = numpy.zeros((3,))
        
    @abstractmethod
    def estimateInputValues(self, frame):
        raise NotImplementedError

    @abstractmethod
    def estimateInputValuesWithAnnotations(self, frame):
        raise NotImplementedError

    @property
    @abstractmethod
    def inputValues(self):
        return self._inputValues

    @property
    @abstractmethod
    def returns3D(self):
        raise NotImplementedError