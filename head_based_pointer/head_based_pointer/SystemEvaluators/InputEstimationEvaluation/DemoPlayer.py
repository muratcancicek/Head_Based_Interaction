from InputEstimators.HeadPoseEstimators.PoseCalculators.YinsKalmanFilteredHeadPoseCalculator import YinsKalmanFilteredHeadPoseCalculator
from InputEstimators.FacialLandmarkDetectors.TF_FrozenCNNBasedFacialLandmarkDetector import TF_FrozenCNNBasedFacialLandmarkDetector
from InputEstimators.HeadPoseEstimators.PoseCalculators.AnthropometricHeadPoseCalculator import AnthropometricHeadPoseCalculator
from InputEstimators.HeadPoseEstimators.CV2Res10SSCNNHeadPoseEstimator import CV2Res10SSCNNHeadPoseEstimator
from SystemEvaluators.InputEstimationEvaluation.InputEstimationDemoHandler import InputEstimationDemoHandler
from InputEstimators.FacialLandmarkDetectors.DLIBFacialLandmarkDetector import DLIBFacialLandmarkDetector
from InputEstimators.HeadPoseEstimators.DLIBHeadPoseEstimator import DLIBHeadPoseEstimator
from InputEstimators.FaceDetectors.DLIBFrontalFaceDetector import DLIBFrontalFaceDetector
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

def play():
    faceDetector = DLIBFrontalFaceDetector()
    #faceDetector = CV2Res10SSDFaceDetector(squaringFaceBox = True) 
    
    landmarkDetector = DLIBFacialLandmarkDetector(faceDetector) 
    #landmarkDetector = TF_FrozenCNNBasedFacialLandmarkDetector(faceDetector) #

    poseCalculator = AnthropometricHeadPoseCalculator()
    #poseCalculator = CV2_PnP_with_KF_HeadPoseCalculator()
    
    estimator = DLIBHeadPoseEstimator(faceDetector, landmarkDetector, poseCalculator)
    #estimator = CV2Res10SSCNNHeadPoseEstimator(faceDetector, landmarkDetector, poseCalculator)
    
    #estimator = faceDetector
    #estimator = landmarkDetector
        
    handler = getDemoHandlerForReplayingSource() # getDemoHandlerForRealTimeEstimation() # 
    
    handler.record(estimator) #display(estimator) 
    