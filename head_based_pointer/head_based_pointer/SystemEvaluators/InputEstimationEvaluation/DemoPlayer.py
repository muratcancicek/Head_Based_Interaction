# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from InputEstimators.HeadPoseEstimators.PoseCalculators.YinsKalmanFilteredHeadPoseCalculator import YinsKalmanFilteredHeadPoseCalculator
from InputEstimators.HeadPoseEstimators.PoseCalculators.AnthropometricHeadPoseCalculator import AnthropometricHeadPoseCalculator
from InputEstimators.FacialLandmarkDetectors.YinsCNNBasedFacialLandmarkDetector import YinsCNNBasedFacialLandmarkDetector
from InputEstimators.HeadPoseEstimators.CV2Res10SSCNNHeadPoseEstimator import CV2Res10SSCNNHeadPoseEstimator
from SystemEvaluators.InputEstimationEvaluation.InputEstimationDemoHandler import InputEstimationDemoHandler
from InputEstimators.FacialLandmarkDetectors.DLIBFacialLandmarkDetector import DLIBFacialLandmarkDetector
from InputEstimators.FaceDetectors.TFMobileNetSSDFaceDetector import TFMobileNetSSDFaceDetector
from InputEstimators.HeadPoseEstimators.DLIBHeadPoseEstimator import DLIBHeadPoseEstimator
from InputEstimators.FaceDetectors.DLIBFrontalFaceDetector import DLIBFrontalFaceDetector
from InputEstimators.FaceDetectors.HaarCascadeFaceDetector import HaarCascadeFaceDetector
from InputEstimators.FaceDetectors.CV2Res10SSDFaceDetector import CV2Res10SSDFaceDetector
from paths import InputEstimatorsDemo_Folder, Experiments_Folder

def getDemoHandlerForRealTimeEstimation():
    return InputEstimationDemoHandler(videoSource = 0, showValues = True, showBoxes = True, showLandmarks = True)

def getDemoHandlerForRecordingFromCam():
    return InputEstimationDemoHandler(videoSource = 0, showValues = False, showBoxes = False, showLandmarks = False)

def getDemoHandlerForReplayingSource(videoSource = None):
    if videoSource == None:
       videoSource = InputEstimatorsDemo_Folder + 'SourceVideo.avi' # Darkest
    return InputEstimationDemoHandler(videoSource = videoSource, showValues = True, showBoxes = True, showLandmarks = True)

def getDefaultEstimators():
    return {'DLIBFaceDetector': DLIBFrontalFaceDetector(), 
            'ResSSDFaceDetector': CV2Res10SSDFaceDetector(squaringFaceBox = True),
            'HaarCascadeFaceDetector': HaarCascadeFaceDetector(),
            'TFMobileNetSSDFaceDetector': TFMobileNetSSDFaceDetector(squaringFaceBox = True),
            'DLIBLandmarkDetector': DLIBFacialLandmarkDetector(),
            'YinsCNNBasedLandmarkDetector': YinsCNNBasedFacialLandmarkDetector(), 
            'DLIBHeadPoseEstimator': DLIBHeadPoseEstimator(),# #}
            'YinsHeadPoseEstimator' : CV2Res10SSCNNHeadPoseEstimator()}

def displayGivenEstimators(handler, estimators):
    for estimatorName, estimator in estimators.items():
        handler.display(estimator, windowTitle = estimatorName) 

def recordGivenEstimators(handler, estimators):
    for estimatorName, estimator in estimators.items():
        outputVideo = InputEstimatorsDemo_Folder + estimatorName + '.avi'
        handler.record(estimator, windowTitle = estimatorName, outputVideo = outputVideo) 
        #handler.silentRecord(estimator, estimatorTitle = estimatorName, outputVideo = outputVideo)
        #handler.silentRecordWithoutPrinting(estimator, estimatorTitle = estimatorName, outputVideo = outputVideo) 

def writeGivenEstimators(handler, estimators, expName):
    for estimatorName, estimator in estimators.items():
        estimatorName = expName + '_' + estimatorName
        outputFile = Experiments_Folder + estimatorName + '.txt'
        handler.displayNWrite(estimator, windowTitle = estimatorName, outputFile = outputFile) 

def recordNWriteGivenEstimators(handler, estimators, expName):
    for estimatorName, estimator in estimators.items():
        estimatorName = expName + '_' + estimatorName
        outputVideo = Experiments_Folder + estimatorName + '.avi'
        outputFile = Experiments_Folder + estimatorName + '.txt'
        handler.recordNWrite(estimator, windowTitle = estimatorName, outputVideo = outputVideo, outputFile = outputFile) 

def play():
    #anthPoseCalculator = AnthropometricHeadPoseCalculator()
    #yinsPoseCalculator = YinsKalmanFilteredHeadPoseCalculator()
        
    source =  Experiments_Folder + 'Exp001/Exp001.avi'
    handler = getDemoHandlerForReplayingSource(source) # getDemoHandlerForRealTimeEstimation() # list()[:2]
    
    estimators = {'White': DLIBFrontalFaceDetector()} # getDefaultEstimators() # 
    
    #displayGivenEstimators(handler, estimators)recordNW
    recordNWriteGivenEstimators(handler, estimators, 'Exp000')
    

def play2():
    source =  Experiments_Folder + 'Exp001/Exp001.avi'
    handler = getDemoHandlerForReplayingSource(source)
    #handler = getDemoHandlerForRealTimeEstimation()
    #estimator = TFMobileNetSSDFaceDetector()
    estimator = YinsCNNBasedFacialLandmarkDetector()
    print(type(estimator))
    handler.display(estimator)
    