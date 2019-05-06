from InputOutputHandler import InputOutputHandler
from DotWalker import DotWalker

def main():
    outputFileName = 'Exp003'
    handler = InputOutputHandler(videoSource = 1, outputVideoName = outputFileName)
    size = (720, 1280, 3) # (1080, 1920, 3) # (480, 640, 3) # 
    walker = DotWalker(size = size, maxFrameCount = 2000, outputFileName = outputFileName)
    handler.play(walker, displaying = True, recording = True)

if __name__ == '__main__':
    main()

