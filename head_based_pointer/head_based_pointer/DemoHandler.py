import cv2

class DemoHandler(object):
    
    def __init__(self):
        super().__init__()
    def get_frame_with_annotations(self, frame, inputValues = None, projectionPoints = None, facial_landmarks = None):
        if len(inputValues) > 0: 
            cv2.putText(frame, "X: " + "{:7.2f}".format(inputValues[0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
            cv2.putText(frame, "Y: " + "{:7.2f}".format(inputValues[1]), (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
            cv2.putText(frame, "Z: " + "{:7.2f}".format(inputValues[2]), (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)

        if len(projectionPoints) > 0:
            for start, end in projectionPoints:
                cv2.line(frame, start, end, (0, 0, 255))

        if len(facial_landmarks) > 0:
            for (x, y) in facial_landmarks:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        return frame
    
    def printDemo(self, estimator):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Unable to connect to camera.")
            return
        try:
            inputValues = [0, 0, 0]
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    inputValues = estimator.estimateInputValues(frame)
                    print('\r%.2f |%.2f | %.2f ' % (inputValues[0], inputValues[1], inputValues[2]), end = '\r')
        except KeyboardInterrupt:
            print('%.2f |%.2f | %.2f ' % (inputValues[0], inputValues[1], inputValues[2]))
            print('Done')
            return


    def streamDemo(self, estimator):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Unable to connect to camera.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                inputValues, projectionPoints, facial_landmarks = estimator.estimateInputValuesWithAnnotations(frame)
                frame = self.get_frame_with_annotations(frame, inputValues, projectionPoints, facial_landmarks)

            cv2.imshow("demo", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def recordDemo(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Unable to connect to camera.")
            return

        # video recorder
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2) .CV_FOURCC(does not exist
        video_writer = cv2.VideoWriter("outputF.avi", fourcc, 20, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame,1)
                frame = self.get_frame_with_annotations(frame, showDots = True, showLines = True, showValues = True)
            video_writer.write(frame)
            cv2.imshow("demo", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def play(self, estimator, printing = True, displaying = False, recording = False):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Unable to connect to camera.")
            return
        if recording:
        # video recorder
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2) .CV_FOURCC(does not exist
            video_writer = cv2.VideoWriter("outputF.avi", fourcc, 20, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if displaying:
                    frame = cv2.flip(frame,1)
                    frame = self.get_frame_with_annotations(frame, showDots = True, showLines = True, showValues = True)
            video_writer.write(frame)
            cv2.imshow("demo", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def main():
    inputValuesEstimator = HeadPoseEstimator()
    inputValuesEstimator.streamDemo()
    #inputValuesEstimator.recordDemo()

if __name__ == '__main__':
    main()

