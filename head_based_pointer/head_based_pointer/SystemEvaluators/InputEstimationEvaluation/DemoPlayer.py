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
from paths import InputEstimatorsDemo_Folder

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
            'YinsCNNBasedlandmarkDetector': YinsCNNBasedFacialLandmarkDetector(), 
            'DLIBHeadPoseEstimator': DLIBHeadPoseEstimator(), 
            'YinsHeadPoseEstimator ': CV2Res10SSCNNHeadPoseEstimator()}

def displayGivenEstimators(handler, estimators):
    for estimatorName, estimator in estimators.items():
        handler.display(estimator, windowTitle = estimatorName) 

def recordGivenEstimators(handler, estimators):
    for estimatorName, estimator in estimators.items():
        outputVideo = InputEstimatorsDemo_Folder + estimatorName + '.avi'
        handler.record(estimator, windowTitle = estimatorName, outputVideo = outputVideo) 
        #handler.silentRecord(estimator, estimatorTitle = estimatorName, outputVideo = outputVideo)
        #handler.silentRecordWithoutPrinting(estimator, estimatorTitle = estimatorName, outputVideo = outputVideo) 

def play2():
    #anthPoseCalculator = AnthropometricHeadPoseCalculator()
    #yinsPoseCalculator = YinsKalmanFilteredHeadPoseCalculator()
        
    handler = getDemoHandlerForReplayingSource() # getDemoHandlerForRealTimeEstimation() # list()[:2]
    
    estimators = {'TFMobileNetSSDFaceDetector': TFMobileNetSSDFaceDetector(squaringFaceBox = True)} # getDefaultEstimators()
    
    #displayGivenEstimators(handler, estimators)
    recordGivenEstimators(handler, estimators)
    

def play():
    #handler = getDemoHandlerForReplayingSource()
    handler = getDemoHandlerForRealTimeEstimation()
    estimator = TFMobileNetSSDFaceDetector()
    handler.display(estimator)
    