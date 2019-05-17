# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from SystemEvaluators.DemoABC import DemoABC
from paths import DemoVideos_Folder, Experiments_Folder
import cv2

class InputEstimationDemo(DemoABC):
    
    def __init__(self, estimator, demoName = 'Demo', showValues = True, showBoxes = True, showLandmarks = True):
        self.__estimator = estimator
        self.__demoName = demoName
        self.__showValues = showValues 
        self.__showBoxes = showBoxes 
        self.__showLandmarks = showLandmarks
        self.__inputValues = [0, 0, 0]
        super().__init__()

    def __addValues(self, frame, inputValues):
        cv2.putText(frame, "X: " + "{:7.2f}".format(inputValues[0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)
        cv2.putText(frame, "Y: " + "{:7.2f}".format(inputValues[1]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness=2)
        cv2.putText(frame, "Z: " + "{:7.2f}".format(inputValues[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)
        return frame

    def __addBox(self, frame, projectionPoints):
        color = (255, 255, 255)
        cv2.polylines(frame, [projectionPoints], True, color, 2, cv2.LINE_AA)
        if self.__estimator.returns3D():
            projectionPoints = [(tuple(projectionPoints[start]), tuple(projectionPoints[end])) for start, end in [(1,6), (2, 7), (3, 8)]]
            for start, end in projectionPoints:
                cv2.line(frame, start, end, color, 2, cv2.LINE_AA)
        return frame

    def __addLandmarks(self, frame, facial_landmarks):
        for i, (x, y) in enumerate(facial_landmarks):
            if (len(facial_landmarks) == 1 or i == 30) and not self.__estimator.returns3D():
                cv2.circle(frame, (x, y), 1, (0, 135, 235), 4, cv2.LINE_AA)
            else:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1, cv2.LINE_AA)
        return frame

    def __getProcessedFrame(self, frame, inputValues = None, projectionPoints = None, facial_landmarks = None):
        if self.__demoName != 'Demo':
            location = (int(frame.shape[1]/6), frame.shape[0] - 10)
            cv2.putText(frame, self.__demoName, location, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness=2)

        if not inputValues is None and self.__showValues: 
            frame = self.__addValues(frame, inputValues)

        if not projectionPoints is None and self.__showBoxes:
            frame = self.__addBox(frame, projectionPoints)

        if not facial_landmarks is None and self.__showLandmarks:
            frame = self.__addLandmarks(frame, facial_landmarks)

        return frame
    
    def __makeLogText(self, inputValues):
        inputValues = [('%.2f'.rjust(11) % i)[-9:] for i in inputValues]
        return '%s |%s |%s ' % (inputValues[0], inputValues[1], inputValues[2])
    
    def getLogTextAndProcessedFrame(self, frame):
        inputValues, projectionPoints, facial_landmarks = self.__estimator.estimateInputValuesWithAnnotations(frame)
        frame = self.__getProcessedFrame(frame, inputValues, projectionPoints, facial_landmarks)
        logText = self.__makeLogText(inputValues)
        return logText, frame
        
    def getLogText(self, frame):
        inputValues = self.__estimator.estimateInputValues(frame)
        return self.__makeLogText(inputValues)