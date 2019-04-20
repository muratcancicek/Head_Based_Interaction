from InputEstimators.HeadPoseEstimators.DLIBHeadPoseEstimator import DLIBHeadPoseEstimator
from InputEstimators.HeadPoseEstimators.CV2Res10SSCNNHeadPoseEstimator import CV2Res10SSCNNHeadPoseEstimator
from InputEstimators.FacialLandmarkDetectors.DLIBFacialLandmarkDetector import DLIBFacialLandmarkDetector
from InputEstimators.FacialLandmarkDetectors.TF_FrozenCNNBasedFacialLandmarkDetector import TF_FrozenCNNBasedFacialLandmarkDetector
from InputEstimators.HeadPoseEstimators.PoseCalculators.CV2_PnP_HeadPoseCalculator import CV2_PnP_HeadPoseCalculator
from InputEstimators.HeadPoseEstimators.PoseCalculators.CV2_PnP_with_KF_HeadPoseCalculator import CV2_PnP_with_KF_HeadPoseCalculator
from InputEstimators.FaceDetectors.DLIBFrontalFaceDetector import DLIBFrontalFaceDetector
from InputEstimators.FaceDetectors.CV2Res10SSDFaceDetector import CV2Res10SSDFaceDetector
from DemoHandler import DemoHandler

def main():
    #faceDetector = DLIBFrontalFaceDetector()
    faceDetector = CV2Res10SSDFaceDetector(squaringFaceBox = True) 
    
    landmarkDetector = DLIBFacialLandmarkDetector(faceDetector) 
    landmarkDetector = TF_FrozenCNNBasedFacialLandmarkDetector(faceDetector) #

    #poseCalculator = CV2_PnP_HeadPoseCalculator()
    poseCalculator = CV2_PnP_with_KF_HeadPoseCalculator()
    
    estimator = DLIBHeadPoseEstimator(faceDetector, landmarkDetector, poseCalculator)
    #estimator = CV2Res10SSCNNHeadPoseEstimator(faceDetector, landmarkDetector, poseCalculator)
    
    #estimator = faceDetector
    #estimator = landmarkDetector
    
    videoSource = 'SourceVideo.avi' # 0 #
    DemoHandler().play(estimator, videoSource = videoSource, printing = True, displaying = True, recording = True, showValues = True, showBoxes = True, showLandmarks = True)
        
    #DemoHandler().play(estimator, videoSource = videoSource, printing = True, displaying = True, recording = True, showValues = False, showBoxes = False, showLandmarks = False)

if __name__ == '__main__':
    main()
