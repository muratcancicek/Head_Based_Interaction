# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
import InputEstimators.FaceDetectors as FaceModule
import InputEstimators.FacialLandmarkDetectors as LandmarkModule
import InputEstimators.HeadPoseEstimators as HeadPoseModule
from CommonTools.ArgParser import getArgsWithRawEstimators
import EstimatorLists

def buildEstimator(estimator, faceDetector, landmarkDetector):
    EstimatorModule = None
    if estimator in EstimatorLists.faceDetectors:
        EstimatorModule = FaceModule
    elif estimator in EstimatorLists.landmarkDetectors:
        EstimatorModule = LandmarkModule
    elif estimator in EstimatorLists.headPoseEstimators:
        EstimatorModule = HeadPoseModule
    eval('import EstimatorModule.%s' % estimator)
    eval('from FaceModule import %s' % faceDetector)
    eval('from LandmarkModule import %s' % landmarkDetector)

    eval('estimator = %s' % estimator)
    eval('faceDetector = %s' % faceDetector)
    eval('landmarkDetector = %s' % landmarkDetector)
    print(faceDetector)

def run():
    args, combos = getArgsWithRawEstimators()
    #buildEstimator(*combos[0][1:])
