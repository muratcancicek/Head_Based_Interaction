from InputOutputHandler import InputOutputHandler
from DotWalker import DotWalker

def getDemoHandlerForRealTimeEstimation():
    return InputOutputHandler(videoSource = 1)

def getDemoHandlerForReplayingSource(videoSource = None):
    return InputOutputHandler(videoSource = videoSource)

def main():
    handler = getDemoHandlerForRealTimeEstimation()
    handler.play(DotWalker(), displaying = True, recording = True)

if __name__ == '__main__':
    main()

