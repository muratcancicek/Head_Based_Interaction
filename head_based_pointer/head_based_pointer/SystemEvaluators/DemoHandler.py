# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from InputEstimators.HeadPoseEstimators.PoseCalculators.YinsKalmanFilteredHeadPoseCalculator import YinsKalmanFilteredHeadPoseCalculator
from InputEstimators.HeadPoseEstimators.PoseCalculators.AnthropometricHeadPoseCalculator import AnthropometricHeadPoseCalculator
from InputEstimators.FacialLandmarkDetectors.YinsCNNBasedFacialLandmarkDetector import YinsCNNBasedFacialLandmarkDetector
from InputEstimators.HeadPoseEstimators.CV2Res10SSCNNHeadPoseEstimator import CV2Res10SSCNNHeadPoseEstimator
from InputEstimators.FacialLandmarkDetectors.DLIBFacialLandmarkDetector import DLIBFacialLandmarkDetector
from InputEstimators.FaceDetectors.TFMobileNetSSDFaceDetector import TFMobileNetSSDFaceDetector
from InputEstimators.HeadPoseEstimators.DLIBHeadPoseEstimator import DLIBHeadPoseEstimator
from InputEstimators.FaceDetectors.DLIBFrontalFaceDetector import DLIBFrontalFaceDetector
from InputEstimators.FaceDetectors.HaarCascadeFaceDetector import HaarCascadeFaceDetector
from InputEstimators.FaceDetectors.CV2Res10SSDFaceDetector import CV2Res10SSDFaceDetector

from SystemEvaluators.InputEstimationEvaluation.InputEstimationDemo import InputEstimationDemo
from SystemEvaluators.DemoPlayer import DemoPlayer
from paths import InputEstimatorsDemo_Folder, Experiments_Folder

def getDemoHandlerForReplayingSource(videoSource = None):
    if videoSource == None:
       videoSource = InputEstimatorsDemo_Folder + 'SourceVideo.avi' # Darkest
    return InputEstimationDemo(estimator, showValues = True, showBoxes = True, showLandmarks = True)
    return DemoPlayer(videoSource = videoSource)

def getDefaultEstimators():
    return {'DLIBFaceDetector': DLIBFrontalFaceDetector(), 
            'ResSSDFaceDetector': CV2Res10SSDFaceDetector(squaringFaceBox = True),
            'HaarCascadeFaceDetector': HaarCascadeFaceDetector(),
            'TFMobileNetSSDFaceDetector': TFMobileNetSSDFaceDetector(squaringFaceBox = True),
            'DLIBLandmarkDetector': DLIBFacialLandmarkDetector(),
            'YinsCNNBasedLandmarkDetector': YinsCNNBasedFacialLandmarkDetector(), 
            'DLIBHeadPoseEstimator': DLIBHeadPoseEstimator(), 
            'YinsHeadPoseEstimator' : CV2Res10SSCNNHeadPoseEstimator()}

def displayGivenEstimators(source, estimators, outputSize = None, grid = (1,1)):
    for estimatorName, estimator in estimators.items():
        player = DemoPlayer(videoSource = source, windowTitle = estimatorName, outputSize = outputSize, grid = grid)
        demo = InputEstimationDemo(estimator, demoName = estimatorName)
        player.display(demo) 

def recordGivenEstimators(source, estimators, expFolder = Experiments_Folder, outputSize = None, grid = (1,1)):
    for estimatorName, estimator in estimators.items():
        outputVideo = expFolder + estimatorName + '.avi'
        player = DemoPlayer(videoSource = source, windowTitle = estimatorName, outputVideo = outputVideo, outputSize = outputSize, grid = grid)
        demo = InputEstimationDemo(estimator, demoName = estimatorName)
        player.record(demo) 

def writeGivenEstimators(source, estimators, expFolder = Experiments_Folder, outputSize = None, grid = (1,1)):
    for estimatorName, estimator in estimators.items():
        outputFile = expFolder + estimatorName + '.txt'
        player = DemoPlayer(videoSource = source, windowTitle = estimatorName, outputFilePath = outputFile, outputSize = outputSize, grid = grid)
        demo = InputEstimationDemo(estimator, demoName = estimatorName)
        player.displayNWrite(demo) 

def recordNWriteGivenEstimators(source, estimators, expName, expFolder = Experiments_Folder, outputSize = None, grid = (1,1)):
    for estimatorName, estimator in estimators.items():
        estimatorName = expName + '_' + estimatorName
        outputVideo = expFolder + expName + '/' + estimatorName + '.avi'
        outputFile = expFolder + expName + '/' + estimatorName + '.txt'
        player = DemoPlayer(videoSource = source, windowTitle = estimatorName, outputVideo = outputVideo, outputFilePath = outputFile, outputSize = outputSize, grid = grid)
        demo = InputEstimationDemo(estimator, estimatorName, False, False, False)
        player.recordNWrite(demo)  

def playExperiment(expName, expFolder = Experiments_Folder, outputSize = None):
    source =  expFolder + expName + '/' + expName + '.avi'    
    estimators = { 'YinsHeadPoseEstimator' : CV2Res10SSCNNHeadPoseEstimator()} 
    #estimators = { 'DLIBHeadPoseEstimator': DLIBHeadPoseEstimator()} # getDefaultEstimators() #     inputFramesize=outputSize
    recordNWriteGivenEstimators(source, estimators, expName, outputSize = outputSize)
    
def play():
    #outputSize = (1280, 720) # (640, 480) # , outputSize = outputSize
    playExperiment('Exp888', expFolder = Experiments_Folder)

def play3():
    #source =  Experiments_Folder + 'Exp001/Exp001.avi'
    #handler = getDemoHandlerForReplayingSource(source), outputSize = (1280, 720)
    handler = DemoPlayer(videoSource = source)
    #estimator = TFMobileNetSSDFaceDetector()
    estimator = YinsCNNBasedFacialLandmarkDetector()
    demo = InputEstimationDemo(estimator, 'module')
    handler.display(demo)
    #handler.print(demo)

def play2():
    #source =  Experiments_Folder + 'Exp000/Exp000.avi', grid = (2,1)
    source =  0 #Experiments_Folder + 'Exp001/Exp001.avi'
    estimators = {'YinsHeadPoseEstimator' : CV2Res10SSCNNHeadPoseEstimator()} #, 'DLIBHeadPoseEstimator': DLIBHeadPoseEstimator() getDefaultEstimators() # , 
    #writeGivenEstimators(source, estimators, 'Exp888')
    recordNWriteGivenEstimators(source, estimators, 'Exp888')

    