# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from paths import DemoVideos_Folder, Experiments_Folder
import numpy as np
import cv2

class DemoPlayer(object):
    
    def __init__(self, videoSource = 'Cam', windowTitle = 'Demo', outputVideo = None, outputFilePath = None, outputSize = None):
        if videoSource == 'Cam': 
            self.__videoSource = 0
        else:
            if isinstance(videoSource, int):
                self.__videoSource = videoSource
            elif videoSource.isdigit():
                videoSource = int(videoSource)
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
        self.__fps = 6
        super().__init__()
  
    @staticmethod
    def __setInputResolution(cap, x,y):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
        return 

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
    
    def __updateLogText(self):
        print('\r%s' % (self.__logText), end = '\r')
        pass
        
    def __endPrinting(self):
        self.__updateLogText()
        print('\nDone')
    
    def __kill(self, ret):
        kill = False
        #if 'q' == chr(cv2.waitKey(0) & 255):
        #    kill = True
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
    
    def __calculateOutputGrid(self, demos, firstFrame):
        self.__hasMultipleDemos = isinstance(demos, (list, tuple))
        if self.__hasMultipleDemos:
            frames_count = len(demos)
        else:
            frames_count = 1
        self.__outputGrid = DemoPlayer.__getProperGrid(frames_count) 
        return self.__outputGrid

    def __matchFrameWithOutputSize(self, frame):
        if frame.shape[0] != self.__outputSize[1] or frame.shape[1] != self.__outputSize[0]:
            frame = cv2.resize(frame, self.__outputSize)
        return frame
    
    @staticmethod
    def __generateEmptyFramesLike(empty_cells, modelFrame):
        emptyFrames = []
        for i in range(empty_cells):
            blackCell = np.zeros_like(modelFrame)
            emptyFrames.append(blackCell)
        return emptyFrames
    
    def __calculateOutputSize(self, firstFrame):
        h, w = firstFrame.shape[0], firstFrame.shape[1]
        if self.__outputSize is None:
            if  self.__hashorizontalSideBars:
                self.__outputSize = (w, int(3*h/4))
            else:
                self.__outputSize = (int(w/self.__inputGrid[0]),
                                    int(h/self.__inputGrid[1]))
        if self.__hasMultipleDemos:
            frame = self.__matchFrameWithOutputSize(firstFrame)
            n, col_num, row_num, empty_cells = self.__outputGrid
            self.__outputSize = (self.__outputSize[0]*col_num,
                                 self.__outputSize[1]*row_num)
            #if n == 1: n = 2
            #self.__fps = self.__fps / n
            self.__emptyFrames = DemoPlayer.__generateEmptyFramesLike(empty_cells, frame)
        return self.__outputSize
      
    def __getVideoRecorder(self, fourcc = None):
        if fourcc == None: fourcc = cv2.VideoWriter_fourcc(*'mpeg') 
        print(self.__fps, self.__outputSize, self.__outputVideo)
        self.__video_writer = cv2.VideoWriter(self.__outputVideo, int(fourcc), self.__fps, self.__outputSize)
        return self.__video_writer
    
    def __start(self, demo):
        cap = cv2.VideoCapture(self.__videoSource, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Unable to connect to camera or open video.")
            return None, None, None
        
        self.__setInputResolution(cap, 1920, 1080)
        self.__calculateInputGrid(cap)
        
        ret, firstFrame = cap.read()
        if self.__kill(ret): return None, None, None
        self.__checkhorizontalSideBars(firstFrame)
        self.__calculateOutputGrid(demo, firstFrame)
        self.__calculateOutputSize(firstFrame)  
        if self.__recording:
            self.__videoRecorder = self.__getVideoRecorder()
        if self.__writing:
            self.__file = open(self.__outputFilePath, 'w')  
        return cap, ret, firstFrame
    
    def __preprocessFrame(self, frame):
        if isinstance(self.__videoSource, int):
            if self.__videoSource < cv2.CAP_DSHOW:
                frame = cv2.flip(frame,1)
                if self.__hashorizontalSideBars:
                    x = int(frame.shape[0]/8)
                    frame = frame[x:-x, :]
                    self.__inputHeight_scale = 9
        if not self.__inputGrid == (1,1):
            frame = frame[:int(frame.shape[0]/self.__inputGrid[1]), 
                          :int(frame.shape[1]/self.__inputGrid[0])]
        return frame

    def __runDemoOnFrame(self, demo, frame):
        if self.__displaying or self.__recording:
            logText, frame = demo.getLogTextAndProcessedFrame(frame)
        elif self.__printing:
            logText = demo.getLogText(frame)
        return logText, frame
    
    def __generateOutputFrameAsGrid(self, outputFrames):
        cell_num, col_num, row_num, empty_cells = self.__outputGrid
        outputFrames.extend(self.__emptyFrames)
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
        return finalLogText[3:], finalFrame

    def __processFrame(self, demo, frame):
        if self.__hasMultipleDemos:
            self.__logText, frame = self.__runEachDemoOnFrame(demo, frame)
        else:
            self.__logText, frame = self.__runDemoOnFrame(demo, frame)
        frame = self.__matchFrameWithOutputSize(frame)
        return frame

    def __playFrame(self, demo, frame):
        frame = self.__preprocessFrame(frame)
        frame = self.__processFrame(demo, frame)
        if self.__printing:
            self.__updateLogText()
        if self.__recording:
            self.__videoRecorder.write(frame)
        if self.__displaying:
            cv2.imshow(self.__windowTitle, frame)
        if self.__writing:
            self.__file.write(self.__logText+'\n') 
        return
        
    def __play(self, demo):
        cap, ret, frame = self.__start(demo)
        if not cap: return

        while cap.isOpened():
            ret, frame = cap.read()

            if self.__kill(ret):
                cap.release()
                cv2.destroyAllWindows()
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
        #print('%s %s at %d fps.' % (job, self.__windowTitle, self.__fps))

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


