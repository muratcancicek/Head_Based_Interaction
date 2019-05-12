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

def displayGivenEstimators(source, estimators):
    for estimatorName, estimator in estimators.items():
        player = DemoPlayer(videoSource = source, windowTitle = estimatorName)
        demo = InputEstimationDemo(estimator, demoName = estimatorName)
        player.display(demo) 

def recordGivenEstimators(source, estimators):
    for estimatorName, estimator in estimators.items():
        outputVideo = Experiments_Folder + estimatorName + '.avi'
        player = DemoPlayer(videoSource = source, windowTitle = estimatorName, outputVideo = outputVideo)
        demo = InputEstimationDemo(estimator, demoName = estimatorName)
        player.record(estimator) 

def writeGivenEstimators(handler, estimators, expName):
    for estimatorName, estimator in estimators.items():
        estimatorName = expName + '_' + estimatorName
        outputFile = Experiments_Folder + estimatorName + '.txt'
        handler.displayNWrite(estimator, windowTitle = estimatorName, outputFile = outputFile) 

def recordNWriteGivenEstimators(handler, estimators, expName, expFolder = Experiments_Folder, recordingSize = None):
    for estimatorName, estimator in estimators.items():
        estimatorName = expName + '_' + estimatorName
        outputVideo = expFolder + expName + '/' + estimatorName + '.avi'
        outputFile = expFolder + expName + '/' + estimatorName + '.txt'
        handler.recordNWrite(estimator, windowTitle = estimatorName, outputVideo = outputVideo, outputFile = outputFile, recordingSize = recordingSize) 

def playExperiment(expName, expFolder = Experiments_Folder, recordingSize = None):
    #anthPoseCalculator = AnthropometricHeadPoseCalculator()
    #yinsPoseCalculator = YinsKalmanFilteredHeadPoseCalculator()
        
    source =  expFolder + expName + '/' + expName + '.avi'
    
    estimators = { 'YinsHeadPoseEstimator' : CV2Res10SSCNNHeadPoseEstimator(inputFramesize=recordingSize)} 
    #estimators = { 'DLIBHeadPoseEstimator': DLIBHeadPoseEstimator()} # getDefaultEstimators() # 
    
    recordNWriteGivenEstimators(handler, estimators, expName, recordingSize = recordingSize)
    
def play2():
    recordingSize = (1280, 720) # (640, 480) # 
    playExperiment('Exp002', expFolder = Experiments_Folder, recordingSize = recordingSize)

def play3():
    #source =  Experiments_Folder + 'Exp001/Exp001.avi'
    #handler = getDemoHandlerForReplayingSource(source)
    handler = DemoPlayer(videoSource = source, recordingSize = (1280, 720))
    #estimator = TFMobileNetSSDFaceDetector()
    estimator = YinsCNNBasedFacialLandmarkDetector()
    demo = InputEstimationDemo(estimator, 'module')
    handler.display(demo)
    #handler.print(demo)

def play():
    source =  Experiments_Folder + 'Exp000/Exp000.avi'
    estimators = {'YinsHeadPoseEstimator' : CV2Res10SSCNNHeadPoseEstimator()} # getDefaultEstimators() # , 
    displayGivenEstimators(source, estimators)

    