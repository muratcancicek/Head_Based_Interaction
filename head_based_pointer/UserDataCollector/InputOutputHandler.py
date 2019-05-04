from paths import DemoVideos_Folder
import cv2
import numpy as np


class InputOutputHandler(object):
    
    def __init__(self, videoSource = 0, windowTitle = 'Demo', outputVideo = None):
        if outputVideo == None: self.__outputVideo = DemoVideos_Folder + 'outputVideo.avi'
        self.__videoSource = videoSource
        self.__windowTitle = windowTitle
        super().__init__()

    def getProcessedFrame(self, frame):
        cv2.circle(frame, (320, 240), 20, (0, 255, 0), -1, cv2.LINE_AA)
        return frame

    def __getVideoRecorder(self, cap):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(self.__outputVideo, fourcc, 20, (int(2*cap.get(3)), int(cap.get(4))))
        return video_writer

    def __updatePrintedInputValues(self, inputValues):
        print('\r%s' % (inputValues), end = '\r')
        
    def __endPrinting(self):
        self.__updatePrintedInputValues('')
        print('\nDone')
        
    def __play(self, dotWalker, printing = True, displaying = False, recording = False):
        cap = cv2.VideoCapture(self.__videoSource, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Unable to connect to camera.")
            return

        if recording:
            videoRecorder = self.__getVideoRecorder(cap)
        while cap.isOpened():
            ret, frame = cap.read()
            #if self.__videoSource == 0:
            frame = cv2.flip(frame, 1)
            frame2 = dotWalker.getUpdatedFrame()
            frame = np.concatenate((frame, frame2), 1)
            #print(frame.shape)
            if not ret or cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(10) == 27:
                self.__updatePrintedInputValues('matrix')
                self.__endPrinting()
                break
            else:
                if recording:
                    videoRecorder.write(frame)
                cv2.imshow(self.__windowTitle, frame)
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

