from InputEstimators.HeadPoseEstimators.RegTreesHeadPoseEstimator import RegTreesHeadPoseEstimator
from InputEstimators.FacialLandmarkDetectors.DLIBFacialLandmarkDetector import DLIBFacialLandmarkDetector
from InputEstimators.FacialLandmarkDetectors.TF_FrozenCNNBasedFacialLandmarkDetector import TF_FrozenCNNBasedFacialLandmarkDetector
from InputEstimators.FaceDetectors.DLIBFrontalFaceDetector import DLIBFrontalFaceDetector
from InputEstimators.FaceDetectors.CV2Res10SSDFaceDetector import CV2Res10SSDFaceDetector
from DemoHandler import DemoHandler

def main():
    #faceDetector = DLIBFrontalFaceDetector()
    #faceDetector = CV2Res10SSDFaceDetector(squaringFaceBox = True) 
    
    #estimator = DLIBFacialLandmarkDetector(faceDetector) 
    estimator = TF_FrozenCNNBasedFacialLandmarkDetector() #faceDetector

    #estimator = RegTreesHeadPoseEstimator()


    #estimator = faceDetector

    DemoHandler().play(estimator, printing = True, displaying = True)#)False

if __name__ == '__main__':
    main()
