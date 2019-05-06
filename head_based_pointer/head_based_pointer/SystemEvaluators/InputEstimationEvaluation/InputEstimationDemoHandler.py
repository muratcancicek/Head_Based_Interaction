# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from paths import DemoVideos_Folder, Experiments_Folder
import cv2

class InputEstimationDemoHandler(object):
    
    def __init__(self, videoSource = 0, windowTitle = 'Demo', outputVideo = None, outputFile = None, showValues = True, showBoxes = True, showLandmarks = True):
        if outputVideo == None: 
            self.__outputVideo = Experiments_Folder + 'outputVideo.avi'
        else: 
            self.__outputVideo = outputVideo
        if outputFile == None: 
            self.__outputFile = Experiments_Folder + 'outputValues.txt'
        else: 
            self.__outputFile = outputFile

        self.__videoSource = videoSource
        self.__windowTitle = windowTitle
        self.__showValues = showValues 
        self.__showBoxes = showBoxes 
        self.__showLandmarks = showLandmarks
        self.__inputValues = [0, 0, 0]
        super().__init__()

    def getProcessedFrame(self, frame, inputValues = None, projectionPoints = None, facial_landmarks = None):
        if self.__windowTitle != 'Demo':
            cv2.putText(frame, self.__windowTitle, (600, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness=2)
        if not inputValues is None and self.__showValues: 
            cv2.putText(frame, "X: " + "{:7.2f}".format(inputValues[0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)
            cv2.putText(frame, "Y: " + "{:7.2f}".format(inputValues[1]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness=2)
            cv2.putText(frame, "Z: " + "{:7.2f}".format(inputValues[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)
        inputValues2D = True
        if not projectionPoints is None and self.__showBoxes:
            color = (255, 255, 255)
            cv2.polylines(frame, [projectionPoints], True, color, 2, cv2.LINE_AA)
            if len(projectionPoints) >= 8:
                inputValues2D = False
                projectionPoints = [(tuple(projectionPoints[start]), tuple(projectionPoints[end])) for start, end in [(1,6), (2, 7), (3, 8)]]
                for start, end in projectionPoints:
                    cv2.line(frame, start, end, color, 2, cv2.LINE_AA)

        if not facial_landmarks is None and self.__showLandmarks:
            for i, (x, y) in enumerate(facial_landmarks):
                if (len(facial_landmarks) == 1 or i == 30) and inputValues2D:
                    cv2.circle(frame, (x, y), 1, (0, 135, 235), 4, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1, cv2.LINE_AA)

        return frame

    def __getVideoRecorder(self, cap):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2) .CV_FOURCC(does not exist
        video_writer = cv2.VideoWriter(self.__outputVideo, fourcc, 20, (int(cap.get(3)/2), int(cap.get(4))))
        return video_writer
    
    def __getVInputValuesWithProcessedFrame(self, estimator, frame):
        inputValues, projectionPoints, facial_landmarks = estimator.estimateInputValuesWithAnnotations(frame)
        frame = self.getProcessedFrame(frame, inputValues, projectionPoints, facial_landmarks)
        return inputValues, frame

    def __updatePrintedInputValues(self, inputValues):
        print('\r%.2f | %.2f | %.2f' % (inputValues[0], inputValues[1], inputValues[2]), end = '\r')
        
    def __endPrinting(self):
        self.__updatePrintedInputValues(self.__inputValues)
        print('\nDone')
        
    def __play(self, estimator, printing = True, displaying = False, recording = False, writing = False):
        cap = cv2.VideoCapture(self.__videoSource)
        if not cap.isOpened():
            print("Unable to connect to camera.")
            return

        if recording:
            print('ids')
            videoRecorder = self.__getVideoRecorder(cap)
        if writing:
            file = open(self.__outputFile, 'w')
        while cap.isOpened():
            ret, frame = cap.read()
            if self.__videoSource == 0:
                frame = cv2.flip(frame,1)
            #Temporary if statement for evaluating mirrored videos with doubled length
            if not ret or cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(10) == 27:
                self.__updatePrintedInputValues(self.__inputValues)
                self.__endPrinting()
                if writing:
                    file.close()
                break
            else:
                #print(frame.shape)
                if frame.shape[1] == 2560:
                    frame = frame[:, int(frame.shape[1]/2):]
                    #print(frame.shape)
                if displaying or recording:
                    self.__inputValues, frame = self.__getVInputValuesWithProcessedFrame(estimator, frame)
                elif printing:
                    self.__inputValues = estimator.estimateInputValues(frame)

                if printing:
                    self.__updatePrintedInputValues(self.__inputValues)
                if recording:
                    videoRecorder.write(frame)
                if displaying:
                    cv2.imshow(self.__windowTitle, frame)
                if writing:
                    file.write('%.2f, %.2f, %.2f\n' % (self.__inputValues[0], self.__inputValues[1], self.__inputValues[2])) 

    def play(self, estimator, printing = True, displaying = False, recording = False, writing = False, windowTitle = None, outputVideo = None, outputFile = None):
        if outputVideo != None: 
            self.__outputVideo = outputVideo
            self.__windowTitle = outputVideo
        if windowTitle != None: self.__windowTitle = windowTitle
        if outputFile != None: self.__outputFile = outputFile
        if recording: job = 'Recording'
        elif writing: job = 'Writing'
        elif displaying: job = 'Displaying'
        else: job = 'Printing'
        print('%s %s' % (job, self.__windowTitle))
        try:
            self.__play(estimator, printing, displaying, recording, writing)
        except KeyboardInterrupt:
            self.__endPrinting()
            return

    def print(self, estimator):
        self.play(estimator)

    def writeValues(self, estimator, outputFile = None):
        self.play(estimator, writing = True, outputFile = outputFile)
        
    def display(self, estimator, windowTitle = None):
        self.play(estimator, displaying = True, writing = True, windowTitle = windowTitle)

    def displayNWrite(self, estimator, windowTitle = None, outputFile = None):
        self.play(estimator, displaying = True, writing = True, windowTitle = windowTitle, outputFile = outputFile)

    def record(self, estimator, windowTitle = None, outputVideo = None):
        self.play(estimator, displaying = True, recording = True, windowTitle = windowTitle, outputVideo = outputVideo)

    def recordNWrite(self, estimator, windowTitle = None, outputVideo = None, outputFile = None):
        self.play(estimator, displaying = True, recording = True, writing = True, windowTitle = windowTitle, outputVideo = outputVideo, outputFile = outputFile)

    def silentRecord(self, estimator, estimatorTitle = None, outputVideo = None):
        self.play(estimator, recording = True, windowTitle = estimatorTitle, outputVideo = outputVideo)

    def silentRecordWithoutPrinting(self, estimator, estimatorTitle = None, outputVideo = None):
        self.play(estimator, printing = False, recording = True, windowTitle = estimatorTitle, outputVideo = outputVideo)

