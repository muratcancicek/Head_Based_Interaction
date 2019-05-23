# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from HeadCursorMapping.MappingABC import MappingABC
from InputEstimators.HeadPoseEstimators.MuratcansHeadGazer import MuratcansHeadGazer
from InputEstimators.HeadPoseEstimators.HeadPoseEstimatorABC import HeadPoseEstimatorABC
from abc import abstractmethod
import numpy

class StaticMapping(MappingABC):
        
    def _calculate(self):
        inputRanges = self._inputBoundaries.getRanges()
        outputRanges = self._outputBoundaries.getRanges()
        #print(inputRanges, outputRanges,'\t') 
        #print('\r', inputRanges, outputRanges,'\t', end= '\r' ) 
        for i, v in enumerate(self._inputValues[:len(self._outputValues)]):
            #if inputRanges[i] == float('inf'):
            #    print('amk')
            self._outputValues[i] = v/inputRanges[i] * outputRanges[i]
        if isinstance(self._inputEstimator, HeadPoseEstimatorABC) and not isinstance(self._inputEstimator, MuratcansHeadGazer):
            self._outputValues = numpy.flip(self._outputValues)
        return self._outputValues