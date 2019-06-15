# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from InputEstimators.HeadPoseEstimators.MuratcansHeadGazer import MuratcansHeadGazer
from InteractionDemos.DemoABC import DemoABC
import cv2, numpy as np

class InputEstimationDemo(DemoABC):
    
    def __init__(self, estimator, demoName = 'Demo', showValues = True, showBoxes = True, showLandmarks = True, *args, **kwargs):
        self._estimator = estimator
        self._demoName = demoName
        self._showValues = showValues 
        self._showBoxes = showBoxes 
        self._showLandmarks = showLandmarks
        super().__init__(*args, **kwargs)

    def _addText(self, frame, text, pos, color, fontScale = 0.75):
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness=2)
        return frame

    def _addValuesLineByLine(self, frame, values, labels, position, colors):
        for v, l, c in zip(values, labels, colors):
            text = "{:s}: {:7.2f}".format(l, v)
            frame = self._addText(frame, text, position, c)
            position = (position[0], position[1]+30)
        return frame

    def _addValues(self, frame):
        pos = (20, 20)
        labels = ['X', 'Y', 'Z']
        colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
        return self._addValuesLineByLine(frame, self._inputValues, labels, pos, colors)
    
    def _addBox(self, frame):
        color = (255, 255, 255)
        cv2.polylines(frame, [self._pPoints], True, color, 2, cv2.LINE_AA)
        if self._estimator.returns3D():
            pPoints = []
            for start, end in [(1,6), (2, 7), (3, 8)]:
                p = (tuple(self._pPoints[start]), tuple(self._pPoints[end]))
                pPoints.append(p)
            for start, end in pPoints:
                cv2.line(frame, start, end, color, 2, cv2.LINE_AA)
        return frame

    def _addLandmarks(self, frame):
        for i, (x, y) in enumerate(self._landmarks):
            inputLandmark = (len(self._landmarks) == 1 or i == 30)
            if inputLandmark and not self._estimator.returns3D():
                cv2.circle(frame, (x, y), 1, (0, 135, 235), 4, cv2.LINE_AA)
            else:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1, cv2.LINE_AA)
        return frame

    def _addDemoName(self, frame):
        position = (int(frame.shape[1]/46), frame.shape[0] - 10)
        frame = self._addText(frame, self._demoName, position, (0, 255, 0), 0.5)
        return frame

    def _getProcessedFrame(self, frame):
        if not self._landmarks is None and self._showLandmarks:
            frame = self._addLandmarks(frame)
        if not self._pPoints is None and self._showBoxes:
            frame = self._addBox(frame)
        if self._demoName != 'Demo':
            frame = self._addDemoName(frame)
        if not self._inputValues is None and self._showValues: 
            frame = self._addValues(frame)
        return frame
        
    def _makeLogText(self, values, format = '.2f', space = 11):
        values = [(('%'+format).rjust(space) % i)[-(space-2):] for i in values]
        logText = ''
        for v in values: logText += '%s ' % v
        return logText
    
    def getLogTextAndProcessedFrame(self, frame):
        annotations = self._estimator.estimateInputValuesWithAnnotations(frame)
        self._inputValues, self._pPoints, self._landmarks = annotations
        frame = self._getProcessedFrame(frame)
        logText = self._makeLogText(self._inputValues)
        return logText, frame
        
    def getLogText(self, frame):
        self._inputValues = self._estimator.estimateInputValues(frame)
        return self._makeLogText(self._inputValues)