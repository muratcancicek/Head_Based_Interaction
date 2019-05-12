# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from paths import DemoVideos_Folder, Experiments_Folder
import cv2

class DemoPlayer(object):
    
    def __init__(self, videoSource = 'Cam', windowTitle = 'Demo', outputVideo = None, outputFile = None, recordingSize = None):
        if videoSource == 'Cam': 
            self.__videoSource = 0
        else:
            self.__videoSource = videoSource
        self.__recordingSize = recordingSize
        self.__outputVideo = outputVideo
        self.__windowTitle = windowTitle
        self.__outputFile = outputFile
        self.__logText = ''
        super().__init__()

    def __getVideoRecorder(self, cap, recordingSize = None):
        if recordingSize is None and int(cap.get(3)) == 2560:
            recordingSize = (int(cap.get(3)/2), int(cap.get(4)))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2) .CV_FOURCC(does not exist
        print(recordingSize, self.__outputVideo)
        video_writer = cv2.VideoWriter(self.__outputVideo, fourcc, 20, recordingSize)
        return video_writer, recordingSize
    
    def __updatePrintedInputValues(self):
        print('\r%s' % (self.__logText), end = '\r')
        
    def __endPrinting(self):
        self.__updatePrintedInputValues()
        print('\nDone')
        
    def __playFrame(self, demo, frame, printing = True, displaying = False, recording = False, writing = False):
        if self.__videoSource in [0, 1]:
            frame = cv2.flip(frame,1)
            x = int(frame.shape[0]/8)
            frame = frame[x:-x, :]
    #Temporary if statement for evaluating mirrored videos with doubled length
        if frame.shape[1] == 2560:
            frame = frame[:, :int(frame.shape[1]/2)]
                    
        if displaying or recording:
            if frame.shape[0] != self.__recordingSize[1] or frame.shape[1] != self.__recordingSize[0]:
                frame = cv2.resize(frame, self.__recordingSize)
            self.__logText = str(frame.shape)
            self.__logText, frame = demo.getLogTextAndProcessedFrame(frame)
        elif printing:
            pass
            self.__logText = demo.getLogText(frame)

        if printing:
            self.__updatePrintedInputValues()
        if recording:
            videoRecorder.write(frame)
        if displaying:
            cv2.imshow(self.__windowTitle, frame)
        if writing:
            file.write(self.__logText) 
        
    def __play(self, demo, printing = True, displaying = False, recording = False, writing = False):
       # cap = cv2.VideoCapture(self.__videoSource + cv2.CAP_DSHOW)
        cap = cv2.VideoCapture(self.__videoSource)

        if not cap.isOpened():
            print("Unable to connect to camera or open video.")
            return
        if recording:
            videoRecorder, recordingSize = self.__getVideoRecorder(cap, recordingSize)
        if writing:
            file = open(self.__outputFile, 'w')

        if self.__recordingSize is None:
            self.__recordingSize = (int(cap.get(3)), int(3*cap.get(4)/4))
                        

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret or cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(10) == 27:
                self.__endPrinting()
                if writing:
                    file.close()
                break
            else:
                self.__playFrame(demo, frame, printing, displaying, recording, writing)

    def play(self, demo, printing = True, displaying = False, recording = False, writing = False):
        if recording: job = 'Recording'
        elif writing: job = 'Writing'
        elif displaying: job = 'Displaying'
        else: job = 'Printing'
        print('%s %s' % (job, self.__windowTitle))
        try:
            self.__play(demo, printing, displaying, recording, writing)
        except KeyboardInterrupt:
            self.__endPrinting()
            return

    def print(self, demo):
        self.play(demo)

    def writeValues(self, demo):
        self.play(demo, writing = True)
        
    def display(self, demo):
        self.play(demo, displaying = True)

    def displayNWrite(self, demo):
        self.play(demo, displaying = True, writing = True)

    def record(self, demo):
        self.play(demo, displaying = True, recording = True)

    def recordNWrite(self, demo):
        self.play(demo, displaying = True, recording = True, writing = True)

    def silentRecord(self, demo):
        self.play(demo, recording = True)

    def silentRecordWithoutPrinting(self, demo):
        self.play(demo, printing = False, recording = True)


