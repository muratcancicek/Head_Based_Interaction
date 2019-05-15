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

def getDemoPlayerForReplayingSource(videoSource = None, outputSize = None):
    if videoSource == None:
       videoSource = InputEstimatorsDemo_Folder + 'SourceVideo.avi' # Darkest
    return DemoPlayer(videoSource = videoSource, outputSize = outputSize)

def getDefaultEstimators():
    return {'DLIBFaceDetector': DLIBFrontalFaceDetector(), 
            'ResSSDFaceDetector': CV2Res10SSDFaceDetector(squaringFaceBox = True),
            'HaarCascadeFaceDetector': HaarCascadeFaceDetector(),
            'TFMobileNetSSDFaceDetector': TFMobileNetSSDFaceDetector(squaringFaceBox = True),
            'DLIBLandmarkDetector': DLIBFacialLandmarkDetector(),
            'YinsCNNBasedLandmarkDetector': YinsCNNBasedFacialLandmarkDetector(), 
            'DLIBHeadPoseEstimator': DLIBHeadPoseEstimator(), 
            'YinsHeadPoseEstimator' : CV2Res10SSCNNHeadPoseEstimator()}

def displayGivenEstimators(source, estimators, outputSize = None):
    for estimatorName, estimator in estimators.items():
        player = DemoPlayer(videoSource = source, windowTitle = estimatorName, outputSize = outputSize)
        demo = InputEstimationDemo(estimator, demoName = estimatorName)
        player.display(demo) 

def recordGivenEstimators(source, estimators, expFolder = Experiments_Folder, outputSize = None):
    for estimatorName, estimator in estimators.items():
        outputVideo = expFolder + estimatorName + '.avi'
        player = DemoPlayer(videoSource = source, windowTitle = estimatorName, outputVideo = outputVideo, outputSize = outputSize)
        demo = InputEstimationDemo(estimator, demoName = estimatorName)
        player.record(demo) 

def writeGivenEstimators(source, estimators, expFolder = Experiments_Folder, outputSize = None):
    for estimatorName, estimator in estimators.items():
        outputFile = expFolder + estimatorName + '.txt'
        player = DemoPlayer(videoSource = source, windowTitle = estimatorName, outputFilePath = outputFile, outputSize = outputSize)
        demo = InputEstimationDemo(estimator, demoName = estimatorName)
        player.displayNWrite(demo) 

def recordNWriteGivenEstimators(source, estimators, expName, expFolder = Experiments_Folder, outputSize = None):
    for estimatorName, estimator in estimators.items():
        estimatorName = expName + '_' + estimatorName
        outputVideo = expFolder + expName + '/' + estimatorName + '.avi'
        outputFile = expFolder + expName + '/' + estimatorName + '.txt'
        player = DemoPlayer(videoSource = source, windowTitle = estimatorName, outputVideo = outputVideo, outputFilePath = outputFile, outputSize = outputSize)
        demo = InputEstimationDemo(estimator, estimatorName) #, False, False, False
        player.recordNWrite(demo)  

def playExperiment(expName, expFolder = Experiments_Folder, outputSize = None):
    source =  expFolder + expName + '/' + expName + '.avi'
    estimators = getDefaultEstimators() 
    #{'YinsHeadPoseEstimator' : CV2Res10SSCNNHeadPoseEstimator()} 
    #estimators = {'DLIBHeadPoseEstimator': DLIBHeadPoseEstimator()} 
    recordNWriteGivenEstimators(source, estimators, expName, outputSize = outputSize)
    
def getDemosForGivenEstimators(estimators):
    demos = []
    for estimatorName, estimator in estimators.items():
        demos.append(InputEstimationDemo(estimator, estimatorName))
    return demos
    
def getDemosForTwoExampleEstimators():
    estimator1 = TFMobileNetSSDFaceDetector()
    estimator2 = YinsCNNBasedFacialLandmarkDetector()
    demo1 = InputEstimationDemo(estimator1, 'YinsCNNBasedFacialLandmarkDetector')
    demo2 = InputEstimationDemo(estimator2, 'TFMobileNetSSDFaceDetector')
    return [demo1, demo2]

def getDemosForDefaultEstimators():
     return getDemosForGivenEstimators(getDefaultEstimators())

def displayTogetherGivenEstimators(source, estimators, windowTitle = 'Demo', outputSize = None):
    demos = getDemosForGivenEstimators(estimators)
    player = DemoPlayer(videoSource = source, windowTitle = windowTitle, outputSize = outputSize)
    player.display(demos) 

def recordTogetherGivenEstimators(source, estimators, outputVideoName = 'OutputVideo', expFolder = Experiments_Folder, windowTitle = 'Demo', outputSize = None):
    demos = getDemosForGivenEstimators(estimators)
    outputVideo = expFolder + outputVideoName + '.avi'
    player = DemoPlayer(videoSource = source, windowTitle = windowTitle, outputVideo = outputVideo, outputSize = outputSize)
    player.record(demos) 

def writeTogetherGivenEstimators(source, estimators, outputFileName = 'OutputFile', expFolder = Experiments_Folder, windowTitle = 'Demo', outputSize = None):
    demos = getDemosForGivenEstimators(estimators)
    outputFile = expFolder + outputFileName + '.txt'
    player = DemoPlayer(videoSource = source, windowTitle = windowTitle, outputFilePath = outputFile, outputSize = outputSize)
    player.displayNWrite(demos) 

def recordNWriteTogetherGivenEstimators(source, estimators, expName, outputVideoName = 'OutputVideo', outputFileName = 'OutputFile', expFolder = Experiments_Folder, windowTitle = 'Demo', outputSize = None):
    outputVideo = expFolder + expName + '/' + outputVideoName + '.avi'
    outputFile = expFolder + expName + '/' + outputFileName + '.txt'
    player = DemoPlayer(videoSource = source, windowTitle = windowTitle, outputVideo = outputVideo, outputFilePath = outputFile, outputSize = outputSize)
    demos = getDemosForGivenEstimators(estimators)
    player.recordNWrite(demos)  


def play3():
    outputSize = (1280, 720) # (640, 480) # 
    playExperiment('Exp888', expFolder = Experiments_Folder, outputSize = outputSize)

def play():
    source = Experiments_Folder + 'Exp888/Exp888.avi'#, outputSize = (1280, 720) 0 
    player = getDemoPlayerForReplayingSource(source)
    demos = getDemosForTwoExampleEstimators()
    #demos = getDemosForDefaultEstimators()
    player.display(demos)

def play2():
    #source =  Experiments_Folder + 'Exp000/Exp000.avi', grid = (2,1)
    source =  0 #Experiments_Folder + 'Exp001/Exp001.avi'
    estimators = {'YinsHeadPoseEstimator' : CV2Res10SSCNNHeadPoseEstimator(), 'DLIBHeadPoseEstimator': DLIBHeadPoseEstimator() } #getDefaultEstimators() # , 
    #writeGivenEstimators(source, estimators, 'Exp888')
    recordNWriteTogetherGivenEstimators(source, estimators, 'Exp888')

    