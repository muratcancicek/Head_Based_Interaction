# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from paths import DemoVideos_Folder, Experiments_Folder
import cv2

class DemoPlayer(object):
    
    def __init__(self, videoSource = 'Cam', windowTitle = 'Demo', outputVideo = None, outputFilePath = None, outputSize = None, grid = (1,1)):
        if videoSource == 'Cam': 
            self.__videoSource = 0
        else:
            self.__videoSource = videoSource
        self.__outputSize = outputSize
        self.__outputVideo = outputVideo
        self.__windowTitle = windowTitle
        self.__outputFilePath = outputFilePath
        self.__logText = ''
        self.__grid = grid
        self.__videoRecorder = None
        self.__file = None
        super().__init__()

    def __calculateInputGrid(self, cap):
        width_multiplier = cap.get(3)/16
        height_scale = cap.get(4)/width_multiplier
        height_ratio = height_scale
        col_num = 1
        while not height_ratio.is_integer():
            height_ratio += height_scale
            col_num += 1
        row_num = 1
        if height_ratio % 9 == 0 and height_ratio % 12 == 0:
            row_num = int(height_ratio / 9)
            self.__inputHeight_scale = 9
        elif height_ratio % 9 == 0:
            row_num = int(height_ratio / 9)
            self.__inputHeight_scale = 9
        elif height_ratio % 12 == 0:
            row_num = int(height_ratio / 12)
            self.__inputHeight_scale = 12
        self.__inputGrid = (col_num, row_num)
        return self.__inputGrid
    
    def __checkhorizontalSideBars(self, frame):
        black = frame[:30, :30].mean() == 0
        self.__hashorizontalSideBars = self.__inputHeight_scale == 12 and black
        return self.__hashorizontalSideBars

    def __calculateOutputSize(self, cap):
        if self.__outputSize is None:
            if  self.__hashorizontalSideBars:
                self.__outputSize = (int(cap.get(3)), int(3*cap.get(4)/4))
            else:
                self.__outputSize = (int(cap.get(3)/self.__inputGrid[0]), int(cap.get(4)/self.__inputGrid[1]))

    def __getVideoRecorder(self, cap):
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        print(self.__fps, self.__outputSize, self.__outputVideo)
        self.__video_writer = cv2.VideoWriter(self.__outputVideo, fourcc, 15, self.__outputSize)
        return self.__video_writer
    
    def __updatePrintedInputValues(self):
        print('\r%s' % (self.__logText), end = '\r')
        
    def __endPrinting(self):
        self.__updatePrintedInputValues()
        print('\nDone')
    
    def __preprocessFrame(self, frame):
        if self.__videoSource in [0, 1]:
            frame = cv2.flip(frame,1)
            if self.__hashorizontalSideBars:
                x = int(frame.shape[0]/8)
                frame = frame[x:-x, :]
                self.__inputHeight_scale = 9
        if not self.__inputGrid == (1,1):
            frame = frame[:int(frame.shape[0]/self.__inputGrid[1]), :int(frame.shape[1]/self.__inputGrid[0])]
        return frame
    
    def __playFrame(self, demo, frame, printing = True, displaying = False, recording = False, writing = False):
        frame = self.__preprocessFrame(frame)
        if displaying or recording:
            if frame.shape[0] != self.__outputSize[1] or frame.shape[1] != self.__outputSize[0]:
                frame = cv2.resize(frame, self.__outputSize)
            self.__logText = str(frame.shape)
            self.__logText, frame = demo.getLogTextAndProcessedFrame(frame)
        elif printing:
            pass
            self.__logText = demo.getLogText(frame)

        if printing:
            self.__updatePrintedInputValues()
        if recording:
            self.__videoRecorder.write(frame)
        if displaying:
            cv2.imshow(self.__windowTitle, frame)
        if writing:
            self.__file.write(self.__logText) 
        
    def __kill(self, ret, writing):
        kill = False
        if not ret or cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(10) == 27:
            kill = True
            self.__endPrinting()
            if writing and not self.__file is None:
                self.__file.close()
        return kill

    def __start(self, writing = False):
       # cap = cv2.VideoCapture(self.__videoSource + cv2.CAP_DSHOW)
        cap = cv2.VideoCapture(self.__videoSource)
        if not cap.isOpened():
            print("Unable to connect to camera or open video.")
            return None, None, None

        self.__fps = cap.get(cv2.CAP_PROP_FPS)
        self.__calculateInputGrid(cap)
        
        ret, frame = cap.read()
        if self.__kill(ret, writing): return None, None, None
        self.__checkhorizontalSideBars(frame)
        self.__calculateOutputSize(cap)    
        return cap, ret, frame
        
    def __play(self, demo, printing = True, displaying = False, recording = False, writing = False):
        cap, ret, frame = self.__start(writing)
        if not cap: return

        if recording:
            self.__videoRecorder = self.__getVideoRecorder(cap)
        if writing:
            self.__file = open(self.__outputFilePath, 'w')

        while cap.isOpened():
            ret, frame = cap.read()
            if self.__kill(ret, writing):
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


