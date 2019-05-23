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
        (x, y) = self._outputValues.astype(int)
        cv2.circle(frame, (x, y), 1, (0, 255, 235), 12, cv2.LINE_AA)
        return frame

    def _getProcessedFrame(self, frame):
        if not self._landmarks is None and self._showLandmarks:
            frame = self._addLandmarks(frame)


        if not self._pPoints is None and self._showBoxes:
            frame = self._addBox(frame)

        frame = self._addPointer(frame)
        #if isinstance(self._estimator, MuratcansHeadGazer):
        #    frame = cv2.resize(frame, (w, h))

        if self._demoName != 'Demo':
            frame = self._addDemoName(frame)
        if not self._inputValues is None and self._showValues: 
            frame = self._addValues(frame)
        return frame
    
    def getLogText(self, frame):
        self._outputValues = self._mappingFunc.calculateOutputValues(frame)
        return self._makeLogText(self._outputValues)
        
    def getLogTextAndProcessedFrame(self, frame):
        annotations = self._mappingFunc.calculateOutputValuesWithAnnotations(frame)
        self._outputValues, self._inputValues, self._pPoints, self._landmarks = annotations
        frame = self._getProcessedFrame(frame)
        logText = self._makeLogText(self._outputValues)
        return logText, frame