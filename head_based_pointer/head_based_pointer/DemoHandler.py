import cv2

class DemoHandler(object):
    
    def __init__(self):
        self.__inputValues = [0, 0, 0]
        super().__init__()

    def getProcessedFrame(self, frame, inputValues = None, projectionPoints = None, facial_landmarks = None):
        if len(inputValues) > 0: 
            cv2.putText(frame, "X: " + "{:7.2f}".format(inputValues[0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)
            cv2.putText(frame, "Y: " + "{:7.2f}".format(inputValues[1]), (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness=2)
            cv2.putText(frame, "Z: " + "{:7.2f}".format(inputValues[2]), (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)

        if len(projectionPoints) > 0:
            for start, end in projectionPoints:
                cv2.line(frame, start, end, (0, 0, 255))

        if len(facial_landmarks) > 0:
            for (x, y) in facial_landmarks:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        return frame

    def __getVideoRecorder(self, cap):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2) .CV_FOURCC(does not exist
        video_writer = cv2.VideoWriter("outputF.avi", fourcc, 20, (int(cap.get(3)), int(cap.get(4))))
        return video_writer
    
    def __getVInputValuesWithProcessedFrame(self, estimator, frame):
        frame = cv2.flip(frame,1)
        inputValues, projectionPoints, facial_landmarks = estimator.estimateInputValuesWithAnnotations(frame)
        frame = self.getProcessedFrame(frame, inputValues, projectionPoints, facial_landmarks)
        return inputValues, frame

    def __updatedPrintedInputValues(self, inputValues):
        print('\r%.2f |%.2f | %.2f ' % (inputValues[0], inputValues[1], inputValues[2]), end = '\r')

        
    def __play(self, estimator, printing = True, displaying = False, recording = False):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Unable to connect to camera.")
            return
        if recording:
            videoRecorder = self.__getVideoRecorder(cap)
            
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if displaying or recording:
                    self.__inputValues, frame = self.__getVInputValuesWithProcessedFrame(estimator, frame)
                elif printing:
                    self.__inputValues = estimator.estimateInputValues(frame)

            if printing:
                self.__updatedPrintedInputValues(self.__inputValues)
            if recording:
                videoRecorder.write(frame)
            if displaying:
                cv2.imshow("demo", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                self.__updatedPrintedInputValues(self.__inputValues)
                print('Done')


    def play(self, estimator, printing = True, displaying = False, recording = False):
        self.__inputValues = [0, 0, 0]
        try:
            self.__play(estimator, printing, displaying, recording)
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

