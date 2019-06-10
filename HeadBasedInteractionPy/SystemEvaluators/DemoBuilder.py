# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
from SystemEvaluators.MappingEvaluation.MappingDemoHandler import displayTogetherGivenEstimators, displayTogetherGivenMappingFunctions
from CommonTools.ArgParser import getArgsWithInstances
from CommonTools.Boundary import Boundary
import EstimatorLists

def getFrom(name, module):
    if name != 'None': 
        exec('from %s.%s import %s' % (module, name, name), globals())
    return eval(name)

def buildFaceDetector(estimator):
    estimator = getFrom(estimator, EstimatorLists.FaceModule)
    return estimator()

def buildLandmarkDetector(estimator, faceDetector):
    estimator = getFrom(estimator, EstimatorLists.HeadPoseModule)
    if faceDetector:
        faceDetector = faceDetector(squaringFaceBox = True)
    return estimator(faceDetector = faceDetector)

def buildHeadPoseEstimator(estimator, faceDetector, landmarkDetector):
    estimator = getFrom(estimator, EstimatorLists.HeadPoseModule)
    if faceDetector:
        faceDetector = faceDetector(squaringFaceBox = True)
    if landmarkDetector:
        landmarkDetector = landmarkDetector(faceDetector = faceDetector)
    return estimator(landmarkDetector = landmarkDetector)

def buildEstimator(estimator, faceDetector, landmarkDetector):
    faceDetector = getFrom(faceDetector, EstimatorLists.FaceModule)
    landmarkDetector = getFrom(landmarkDetector, EstimatorLists.LandmarkModule)

    if estimator in EstimatorLists.faceDetectors:
        return buildFaceDetector(estimator)
    elif estimator in EstimatorLists.landmarkDetectors:
        return buildLandmarkDetector(estimator, faceDetector)
    elif estimator in EstimatorLists.headPoseEstimators:
        return buildHeadPoseEstimator(estimator, faceDetector, landmarkDetector)

def generateEstimatorTag(codes):
    est, fd, ld = codes
    est = EstimatorLists.estimatorDict[est]
    if fd == '_' and ld == '_':
        return est
    elif ld == '_':
        return est + '(FD=%s)' % fd
    elif fd == '_':
        return est + '(LD=%s)' % ld
    else:
        return est + '(FD=%s,LD=%s)' % (fd, ld)

def buildEstimatorWithTag(components):
    codes = [c[0] for c in components]
    names = [c[1] for c in components]
    estimator = buildEstimator(*names)
    tag = generateEstimatorTag(codes)
    return estimator, tag

def playInputDemos(instances):
    estimators = [buildEstimatorWithTag(i) for i in instances]
    estimators = {tag: mapping for mapping, tag in estimators}
    displayTogetherGivenEstimators(0, estimators)

def buildMappingFunction(name, estimator):
    mapping = getFrom(name, EstimatorLists.MappingModule)
    outputSize = (640, 360)
    boundary = Boundary(0, outputSize[0], 0, outputSize[1])
    return mapping(estimator, boundary)

def buildMappingFunctionWithTag(components):
    name, estComponents = components[0], components[1:]
    estimator, estTag = buildEstimatorWithTag(estComponents)
    mapping = buildMappingFunction(name, estimator)
    mappingTag = '%s_with_%s' % (estTag, name)
    return mapping, mappingTag

def playMappingDemos(instances):
    mappingFunctions = [buildMappingFunctionWithTag(i) for i in instances]
    mappingFunctions = {tag: mapping for mapping, tag in mappingFunctions}
    displayTogetherGivenMappingFunctions(0, mappingFunctions)

def run():
    args, instances = getArgsWithInstances()
    if args.module == 'Mapping':
        playMappingDemos(instances)
    else:
        playInputDemos(instances)
    #print(mappingFunctions)