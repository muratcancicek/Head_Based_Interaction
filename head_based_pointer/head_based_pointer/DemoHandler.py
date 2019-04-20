import cv2

class DemoHandler(object):
    
    def __init__(self):
        self.__inputValues = [0, 0, 0]
        super().__init__()

    def getProcessedFrame(self, frame, inputValues = None, projectionPoints = None, facial_landmarks = None, showValues = True, showBoxes = True, showLandmarks = True):
        if not inputValues is None and showValues: 
            cv2.putText(frame, "X: " + "{:7.2f}".format(inputValues[0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)
            cv2.putText(frame, "Y: " + "{:7.2f}".format(inputValues[1]), (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness=2)
            cv2.putText(frame, "Z: " + "{:7.2f}".format(inputValues[2]), (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)
        inputValues2D = True
        if not projectionPoints is None and showBoxes:
            color = (255, 255, 255)
            cv2.polylines(frame, [projectionPoints], True, color, 2, cv2.LINE_AA)
            if len(projectionPoints) >= 8:
                inputValues2D = False
                projectionPoints = [(tuple(projectionPoints[start]), tuple(projectionPoints[end])) for start, end in [(1,6), (2, 7), (3, 8)]]
                for start, end in projectionPoints:
                    cv2.line(frame, start, end, color, 2, cv2.LINE_AA)

        if not facial_landmarks is None and showLandmarks:
            for i, (x, y) in enumerate(facial_landmarks):
                if (len(facial_landmarks) == 1 or i == 30) and inputValues2D:
                    cv2.circle(frame, (x, y), 1, (0, 165, 255), 4, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1, cv2.LINE_AA)

        return frame

    def __getVideoRecorder(self, cap):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2) .CV_FOURCC(does not exist
        video_writer = cv2.VideoWriter("outputF.avi", fourcc, 20, (int(cap.get(3)), int(cap.get(4))))
        return video_writer
    
    def __getVInputValuesWithProcessedFrame(self, estimator, frame, showValues = True, showBoxes = True, showLandmarks = True):
        inputValues, projectionPoints, facial_landmarks = estimator.estimateInputValuesWithAnnotations(frame)
        frame = self.getProcessedFrame(frame, inputValues, projectionPoints, facial_landmarks, showValues, showBoxes, showLandmarks)
        return inputValues, frame

    def __updatedPrintedInputValues(self, inputValues):
        print('\r%.2f |%.2f | %.2f ' % (inputValues[0], inputValues[1], inputValues[2]), end = '\r')

        
    def __play(self, estimator, videoSource = 0, printing = True, displaying = False, recording = False, showValues = True, showBoxes = True, showLandmarks = True):
        cap = cv2.VideoCapture(videoSource)
        if not cap.isOpened():
            print("Unable to connect to camera.")
            return
        if recording:
            videoRecorder = self.__getVideoRecorder(cap)
            
        while cap.isOpened():
            ret, frame = cap.read()
            if videoSource == 0:
                frame = cv2.flip(frame,1)
            if not ret or cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(10) == 27:
                break
                self.__updatedPrintedInputValues(self.__inputValues)
                print('Done')
            else:
                if displaying or recording:
                    self.__inputValues, frame = self.__getVInputValuesWithProcessedFrame(estimator, frame, showValues, showBoxes, showLandmarks)
                elif printing:
                    self.__inputValues = estimator.estimateInputValues(frame)

                if printing:
                    self.__updatedPrintedInputValues(self.__inputValues)
                if recording:
                    videoRecorder.write(frame)
                if displaying:
                    cv2.imshow("demo", frame)

    def play(self, estimator, videoSource = 0, printing = True, displaying = False, recording = False, showValues = True, showBoxes = True, showLandmarks = True):
        self.__inputValues = [0, 0, 0]
        try:
            self.__play(estimator, videoSource, printing, displaying, recording, showValues, showBoxes, showLandmarks)
        except KeyboardInterrupt:
            self.__updatedPrintedInputValues(self.__inputValues)
            print('Done')
            return


def main():
    inputValuesEstimator = HeadPoseEstimator()
    inputValuesEstimator.streamDemo()
    #inputValuesEstimator.recordDemo()

if __name__ == '__main__':
    main()

