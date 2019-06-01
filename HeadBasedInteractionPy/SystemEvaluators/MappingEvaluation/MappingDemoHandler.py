# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from SystemEvaluators.MappingEvaluation.InputEstimationDemoWithMappingFunction import InputEstimationDemoWithMappingFunction
from InputEstimators.HeadPoseEstimators.MuratcansHeadGazer import MuratcansHeadGazer
from SystemEvaluators.InputEstimationEvaluation.InputEstDemoHandler import *
from HeadCursorMapping.StaticMapping import StaticMapping
from HeadCursorMapping.DynamicMapping import DynamicMapping
from CommonTools.Boundary import Boundary

def getDemosForGivenMappingFunctions(mappingFunctions):
    demos = []
    for mappingFunctionName, mappingFunction in mappingFunctions.items():
        demos.append(InputEstimationDemoWithMappingFunction(mappingFunction, mappingFunctionName))
    return demos
    
def getDemosForThreeExampleMappingFunctions(outputSize = (640, 360)):
    boundary = Boundary(0, outputSize[0], 0, outputSize[1])
    mappingFunction1 = StaticMapping(TFMobileNetSSDFaceDetector(), boundary)
    mappingFunction2 = StaticMapping(YinsCNNBasedFacialLandmarkDetector(), boundary)
    mappingFunction3 = StaticMapping(CV2Res10SSCNNHeadPoseEstimator(), boundary)
    demo1 = InputEstimationDemoWithMappingFunction(mappingFunction1, 'TFMobileNetSSDFaceDetector')
    demo2 = InputEstimationDemoWithMappingFunction(mappingFunction2, 'YinsCNNBasedFacialLandmarkDetector')
    demo3 = InputEstimationDemoWithMappingFunction(mappingFunction3, 'CV2Res10SSCNNHeadPoseEstimator')
    return [demo1, demo2, demo3]

def getDemosForDefaultMappingFunctions():
     return getDemosForGivenMappingFunctions(getDefaultMappingFunctions())

def displayTogetherGivenMappingFunctions(source, mappingFunctions, windowTitle = 'Demo', outputSize = None):
    demos = getDemosForGivenMappingFunctions(mappingFunctions)
    player = DemoPlayer(videoSource = source, windowTitle = windowTitle, outputSize = outputSize)
    player.display(demos) 

def recordTogetherGivenMappingFunctions(source, mappingFunctions, outputVideoName = 'OutputVideo', expFolder = Experiments_Folder, windowTitle = 'Demo', outputSize = None):
    demos = getDemosForGivenMappingFunctions(mappingFunctions)
    outputVideo = expFolder + outputVideoName + '.avi'
    player = DemoPlayer(videoSource = source, windowTitle = windowTitle, outputVideo = outputVideo, outputSize = outputSize)
    player.record(demos) 

def writeTogetherGivenMappingFunctions(source, mappingFunctions, outputFileName = 'OutputFile', expFolder = Experiments_Folder, windowTitle = 'Demo', outputSize = None):
    demos = getDemosForGivenMappingFunctions(mappingFunctions)
    outputFile = expFolder + outputFileName + '.txt'
    player = DemoPlayer(videoSource = source, windowTitle = windowTitle, outputFilePath = outputFile, outputSize = outputSize)
    player.displayNWrite(demos) 

def recordNWriteTogetherGivenMappingFunctions(source, mappingFunctions, expName, outputVideoName = 'OutputVideo', outputFileName = 'OutputFile', expFolder = Experiments_Folder, windowTitle = 'Demo', outputSize = None):
    outputVideo = expFolder + expName + '/' + outputVideoName + '.avi'
    outputFile = expFolder + expName + '/' + outputFileName + '.txt'
    player = DemoPlayer(videoSource = source, windowTitle = windowTitle, outputVideo = outputVideo, outputFilePath = outputFile, outputSize = outputSize)
    demos = getDemosForGivenMappingFunctions(mappingFunctions)
    player.recordNWrite(demos)  

def playExperimentAtOnce(expName, outputVideoName = 'OutputVideo', outputFileName = 'OutputFile', expFolder = Experiments_Folder, outputSize = None):
    source =  expFolder + expName + '/' + expName + '.avi'
    mappingFunctions = getDefaultMappingFunctions() 
    recordNWriteTogetherGivenMappingFunctions(source, mappingFunctions, expName, outputVideoName, outputFileName, expFolder, outputSize)
    
def playMapping3():
    #outputSize = (1280, 720) # (640, 480) # , outputSize = outputSize
    playExperimentAtOnce('Exp000', expFolder = Experiments_Folder)

def playMapping2():
    source = Experiments_Folder + 'Exp999/Exp999.avi'#, outputSize = (1280, 720) 0 
    player = getDemoPlayerForReplayingSource(source)
    demos = getDemosForThreeExampleMappingFunctions()
    #demos = getDemosForDefaultMappingFunctions()
    player.display(demos)

def playMapping():
    source = 0 # Experiments_Folder + 'Exp999/Exp999.avi' # 'Exp000/Exp000.avi' # 'Exp001/Exp001.avi'
    #outputSize = (1920, 1080)
    outputSize = (640, 360)
    #outputSize = (1080, 720)
    boundary = Boundary(0, outputSize[0], 0, outputSize[1])
    mappingStr = 'StaticMappingOn' # 'DynamicMappingOn' # 
    Mapping = DynamicMapping if mappingStr == 'DynamicMappingOn' else StaticMapping # 
    mappingFunctions = {
                         #mappingStr + 'MblNtSSDBox': Mapping(TFMobileNetSSDFaceDetector(squaringFaceBox = True), boundary), 
                         #mappingStr + 'YinsLMarks': Mapping(YinsCNNBasedFacialLandmarkDetector(), boundary), 
                         #mappingStr + 'YinsHPose': Mapping(YinsHeadPoseEstimator(), boundary),
                         mappingStr + 'MrtcnsGaze': Mapping(MuratcansHeadGazer(), boundary)
                       }
    #mappingFunctions = getDefaultMappingFunctions() (720, 480)
    #outputSize = (1280, 720) record  
    names = [k for k, i in mappingFunctions.items()]
    s = ''
    for n in names:
       s += (n + '_')
    s = s[:-1]
    expName = 'Exp999'
    name = expName+'_'+s
    displayTogetherGivenMappingFunctions(source, mappingFunctions, name, outputSize = outputSize)
    #recordNWriteTogetherGivenMappingFunctions(source, mappingFunctions, expName, name, name, outputSize = outputSize)

    