# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from paths import DemoVideos_Folder, Experiments_Folder
import numpy as np
import cv2

class DemoPlayer(object):
    
    def __init__(self, videoSource = 'Cam', windowTitle = 'Demo', outputVideo = None, outputFilePath = None, outputSize = None):
        if videoSource == 'Cam': 
            self.__videoSource = 0
        else:
            self.__videoSource = videoSource
        self.__outputSize = outputSize
        self.__outputVideo = outputVideo
        self.__windowTitle = windowTitle
        self.__outputFilePath = outputFilePath
        self.__logText = ''
        self.__videoRecorder = None
        self.__file = None
        self.__printing = False
        self.__displaying = False
        self.__recording = False
        self.__writing = False
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
    
    def __updatePrintedInputValues(self):
        print('\r%s' % (self.__logText), end = '\r')
        
    def __endPrinting(self):
        self.__updatePrintedInputValues()
        print('\nDone')
    
    def __kill(self, ret):
        kill = False
        if not ret or cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(10) == 27:
            kill = True
            self.__endPrinting()
            if self.__writing and not self.__file is None:
                self.__file.close()
        return kill

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
        return
    
    def __start(self):
        # cap = cv2.VideoCapture(self.__videoSource + cv2.CAP_DSHOW)
        cap = cv2.VideoCapture(self.__videoSource)
        if not cap.isOpened():
            print("Unable to connect to camera or open video.")
            return None, None, None

        self.__fps = cap.get(cv2.CAP_PROP_FPS)
        self.__calculateInputGrid(cap)
        
        ret, frame = cap.read()
        if self.__kill(ret): return None, None, None
        self.__checkhorizontalSideBars(frame)
        self.__calculateOutputSize(cap)  
        if self.__recording:
            self.__videoRecorder = self.__getVideoRecorder(cap)
        if self.__writing:
            self.__file = open(self.__outputFilePath, 'w')  
        return cap, ret, frame
      
    def __getVideoRecorder(self, cap):
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        print(self.__fps, self.__outputSize, self.__outputVideo)
        self.__video_writer = cv2.VideoWriter(self.__outputVideo, fourcc, 15, self.__outputSize)
        return self.__video_writer
    
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

    def __matchFrameWithOutputSize(self, frame):
        if frame.shape[0] != self.__outputSize[1] or frame.shape[1] != self.__outputSize[0]:
            frame = cv2.resize(frame, self.__outputSize)
        return frame

    def __runDemoOnFrame(self, demo, frame):
        if self.__displaying or self.__recording:
            frame = self.__matchFrameWithOutputSize(frame)
            logText, frame = demo.getLogTextAndProcessedFrame(frame)
        elif self.__printing:
            logText = demo.getLogText(frame)
        return logText, frame

    @staticmethod
    def __getProperGrid(actual_cell_num):
        cols, rows, w = 1, 1, True
        while actual_cell_num > cols*rows:
            if w: cols += 1
            else: rows += 1
            w = not w
        cell_num = cols*rows
        empty_cells = cell_num - actual_cell_num
        return cell_num, cols, rows, empty_cells
    
    @staticmethod
    def __generateEmptyFramesLike(empty_cells, modelFrame):
        emptyFrames = []
        for i in range(empty_cells):
            blackCell = np.zeros_like(modelFrame)
            emptyFrames.append(blackCell)
        return emptyFrames

    def __generateOutputFrameAsGrid(self, outputFrames):
        frames_count = len(outputFrames)
        grid = self.__getProperGrid(frames_count) 
        cell_num, col_num, row_num, empty_cells = grid
        emptyFrames = self.__generateEmptyFramesLike(empty_cells, outputFrames[0])
        outputFrames.extend(emptyFrames)
        rows = []
        for i in range(0, cell_num, col_num):
            r = np.concatenate(outputFrames[i:i+col_num], axis=1)
            rows.append(r)
        if row_num > 1:
            finalFrame = np.concatenate(rows)
        else:
            finalFrame = rows[0]
        return finalFrame

    def __runEachDemoOnFrame(self, givenDemos, inputFrame):
        outputFrames = []
        finalLogText = ''
        for demo in givenDemos:
            logText, frame = self.__runDemoOnFrame(demo, inputFrame.copy())
            finalLogText = finalLogText + ' - ' + logText
            outputFrames.append(frame)
        finalFrame = self.__generateOutputFrameAsGrid(outputFrames)
        return finalLogText, finalFrame

    def __processFrame(self, demo, frame):
        if self.__hasMultipleDemos:
            self.__logText, frame = self.__runEachDemoOnFrame(demo, frame)
        else:
            self.__logText, frame = self.__runDemoOnFrame(demo, frame)
        return frame

    def __playFrame(self, demo, frame):
        frame = self.__preprocessFrame(frame)
        frame = self.__processFrame(demo, frame)
        if self.__printing:
            self.__updatePrintedInputValues()
        if self.__recording:
            self.__videoRecorder.write(frame)
        if self.__displaying:
            cv2.imshow(self.__windowTitle, frame)
        if self.__writing:
            self.__file.write(self.__logText) 
        return
        
    def __play(self, demo):
        cap, ret, frame = self.__start()
        if not cap: return
        self.__hasMultipleDemos = isinstance(demo, (list, tuple))
        while cap.isOpened():
            ret, frame = cap.read()
            if self.__kill(ret):
                break
            else:
                self.__playFrame(demo, frame)

    def play(self, demo, printing = True, displaying = False, recording = False, writing = False):
        self.__printing = printing
        self.__displaying = displaying
        self.__recording = recording
        self.__writing = writing

        if recording: job = 'Recording'
        elif writing: job = 'Writing'
        elif displaying: job = 'Displaying'
        else: job = 'Printing'
        print('%s %s' % (job, self.__windowTitle))

        try:
            self.__play(demo)
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


