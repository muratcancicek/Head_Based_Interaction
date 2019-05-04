from paths import Experiments_Folder
import numpy as np
import cv2


class InputOutputHandler(object):
    
    def __init__(self, videoSource = 0, windowTitle = 'Demo', outputVideoName = 'Exp'):
        self.__outputVideo = Experiments_Folder + outputVideoName + '.avi'
        self.__videoSource = videoSource
        self.__windowTitle = windowTitle
        super().__init__()

    def __getVideoRecorder(self, cap, width = 0, height = 0):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if width == 0: width = int(cap.get(3))
        if height == 0: height = int(cap.get(4))
        video_writer = cv2.VideoWriter(self.__outputVideo, fourcc, 20, (width, height))
        return video_writer

    def __updatePrintedInputValues(self, inputValues):
        print('\r%s' % (inputValues), end = '\r')
        
    def __endPrinting(self):
        self.__updatePrintedInputValues('')
        print('\nDone')
        
    def __play(self, dotWalker, printing = True, displaying = False, recording = False):
        cap = cv2.VideoCapture(self.__videoSource, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print("Unable to connect to camera.")
            return
        
        frame2 = dotWalker.getUpdatedFrame()
        if recording:
            videoRecorder = self.__getVideoRecorder(cap, frame2.shape[1]*2, frame2.shape[0])
        while cap.isOpened():
            ret, frame = cap.read()
            #if self.__videoSource == 0:
            frame1 = cv2.flip(frame, 1)
            frame2 = dotWalker.getUpdatedFrame()
            #frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))
            if frame1.shape != frame2.shape:
                temp = np.zeros_like(frame2)
                temp[:frame1.shape[0], :frame1.shape[1]] = frame1 
                frame1 = temp
            frame = np.concatenate((frame1, frame2), 1)
            cv2.line(frame, (frame1.shape[1], 0), (frame1.shape[1], frame1.shape[0]), (255, 255, 255), 2, cv2.LINE_AA)
            #print(frame.shape)
            if not ret or cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(10) == 27:
                self.__updatePrintedInputValues('matrix')
                self.__endPrinting()
                break
            else:
                if recording:
                    videoRecorder.write(frame)
                cv2.imshow(self.__windowTitle, frame2)
                self.__updatePrintedInputValues('matrix')

    def play(self, dotWalker, printing = True, displaying = False, recording = False, windowTitle = None, outputVideo = None):
        if outputVideo != None: 
            self.__outputVideo = outputVideo
            self.__windowTitle = outputVideo
        if windowTitle != None: self.__windowTitle = windowTitle
        if recording: job = 'Recording'
        elif displaying: job = 'Displaying'
        else: job = 'Printing'
        print('%s %s' % (job, self.__windowTitle))
        try:
            self.__play(dotWalker, printing, displaying, recording)
        except KeyboardInterrupt:
            self.__endPrinting()
            return

def main():
    handler = getDemoHandlerForReplayingSource()
    inputValuesEstimator = HeadPoseEstimator()
    inputValuesEstimator.streamDemo()
    #inputValuesEstimator.recordDemo()

if __name__ == '__main__':
    main()

