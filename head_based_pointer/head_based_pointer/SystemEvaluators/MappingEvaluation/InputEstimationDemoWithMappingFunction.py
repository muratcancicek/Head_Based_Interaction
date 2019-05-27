# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from SystemEvaluators.InputEstimationEvaluation.InputEstimationDemo import InputEstimationDemo
import cv2, numpy

class InputEstimationDemoWithMappingFunction(InputEstimationDemo):
        
    def __init__(self, mappingFunc, *args, **kwargs):
        self._mappingFunc = mappingFunc
        super().__init__(mappingFunc.getEstimator(), *args, **kwargs)

    def _addValues(self, frame):
        pos = (20, 20)
        labels = ['inX', 'inY', 'inZ']
        colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
        frame = self._addValuesLineByLine(frame, self._inputValues, labels, pos, colors)
        labels = ['outX', 'outY', 'outZ']
        pos = (frame.shape[1] - 180, 20)
        colors = ((0, 0, 255), (0, 255, 0))
        return self._addValuesLineByLine(frame, self._outputValues, labels, pos, colors)
    
    def _addPointer(self, frame):
        boundaries = self._mappingFunc.getOutputBoundaries()
        (height, width, depth) = frame.shape
        (xRange, yRange, _) = boundaries.getRanges()
        if xRange != width or yRange != height:
            xRange, yRange = boundaries.getVolumeAbsRatio(self._outputValues)
            x, y = int(xRange*width), int(yRange*height)
        else:
            x, y = self._outputValues.astype(int)
        cv2.circle(frame, (x, y), 1, (0, 255, 235), 12, cv2.LINE_AA)
        return frame

    def _getProcessedFrame(self, frame):
        if not self._landmarks is None and self._showLandmarks:
            frame = self._addLandmarks(frame)
        if not self._pPoints is None and self._showBoxes:
            frame = self._addBox(frame)
        frame = self._addPointer(frame)
        if self._demoName != 'Demo':
            frame = self._addDemoName(frame)
        if not self._inputValues is None and self._showValues: 
            frame = self._addValues(frame)
        return frame

    def _makeInOutLogText(self):
        inText = self._makeLogText(self._inputValues)
        outText = self._makeLogText(self._outputValues.astype(int), 'd', 6)
        logText = 'in: %s | out: %s' % (inText, outText)
        return logText
    
    def getLogText(self, frame):
        self._outputValues = self._mappingFunc.calculateOutputValues(frame)
        return self._makeInOutLogText()
        
    def getLogTextAndProcessedFrame(self, frame):
        annos = self._mappingFunc.calculateOutputValuesWithAnnotations(frame)
        self._outputValues, self._inputValues, self._pPoints, self._landmarks = annos
        frame = self._getProcessedFrame(frame)
        logText = self._makeInOutLogText()
        return logText, frame