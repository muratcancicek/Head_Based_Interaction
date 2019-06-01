# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from HeadCursorMapping.MappingABC import MappingABC
from InputEstimators.HeadPoseEstimators.MuratcansHeadGazer import MuratcansHeadGazer
from InputEstimators.HeadPoseEstimators.HeadPoseEstimatorABC import HeadPoseEstimatorABC
from abc import abstractmethod

class StaticMapping(MappingABC):
        
    def _calculate(self):
        inputRanges = self._inputBoundaries.getRanges()
        outputRanges = self._outputBoundaries.getRanges()
        ratios = self._inputBoundaries.getVolumeAbsRatio(self._inputValues)
        if isinstance(self._inputEstimator, HeadPoseEstimatorABC) and \
                   not isinstance(self._inputEstimator, MuratcansHeadGazer):
            t = ratios[0]; ratios[0] = ratios[1]; ratios[1] = t
        i = self._outputValues.shape[0]
        self._outputValues = ratios[:i] * outputRanges[:i]
        return self._outputValues