# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from HeadCursorMapping.MappingABC import MappingABC
from InputEstimators.HeadPoseEstimators.HeadPoseEstimatorABC import HeadPoseEstimatorABC
from abc import abstractmethod
import numpy

class DynamicMapping(MappingABC):
    def __init__(self, inputEstimator, outputBoundaries, xSpeed = 5, ySpeed = 5,
                acceleration = 2, smoothness = 8, motionThreshold = 4, *args, **kwargs):
        super().__init__(inputEstimator, outputBoundaries, *args, **kwargs)
        if smoothness < 2: smoothness = 2
        self._inputValueQueue = None
        self._outputValueQueue = None
        self._speed = numpy.array((xSpeed, ySpeed))
        self._acceleration = acceleration
        self._smoothness = smoothness
        self._motionThreshold = motionThreshold
        
    def _initializeQueues(self):
        if not self._inputValueQueue is None: return
        self._inputValueQueue = numpy.zeros((self._smoothness, 3))
        self._outputValueQueue = numpy.zeros((self._smoothness, 2))
        for i in range(self._inputValueQueue.shape[0]):
           self._inputValueQueue[i] = self._inputValues
        outputRanges = self._outputBoundaries.getRanges()[:2]
        for i in range(self._outputValueQueue.shape[0]):
           self._outputValueQueue[i] = outputRanges
        self._outputValueQueue = self._outputValueQueue/2
        return

    def _calculate(self):
        if isinstance(self._inputEstimator, HeadPoseEstimatorABC):
            t = self._inputValues[0]
            self._inputValues[0] = self._inputValues[1] 
            self._inputValues[1] = t
        self._initializeQueues()
        inputRanges = self._inputBoundaries.getRanges()
        outputRanges = self._outputBoundaries.getRanges()

        self._inputValueQueue[:-1, :] = self._inputValueQueue[1:, :]
        self._inputValueQueue[-1, :] = self._inputValues
        self._inputValues = self._inputValueQueue.mean(axis = 0)

        direction = self._inputValues - self._inputValueQueue[-2, :] 
        direction = (direction[:2]/inputRanges[:2] * self._speed)*outputRanges[:2]

        self._outputValueQueue[:-1, :] = self._outputValueQueue[1:, :]
        self._outputValueQueue[-1, :] = self._outputValueQueue[-2, :] - direction
        self._outputValues = self._outputValueQueue.mean(axis = 0)
        
        if isinstance(self._inputEstimator, HeadPoseEstimatorABC):
            self._outputValues[1] -= 150

        return self._outputValues