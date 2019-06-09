# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
from argparse import ArgumentParser, RawTextHelpFormatter
from EstimatorLists import *
import sys

def getEstimatorsInfo():
    faceDetDes = ''
    for c, est in zip(faceDetectorCodes, faceDetectors):
        faceDetDes += '\'%s\': %s\n' % (c, est)

    lndmrkDetDes = ''
    for c, est in zip(landmarkDetectorCodes, landmarkDetectors):
        lndmrkDetDes += '\'%s\': %s\n' % (c, est)

    poseEstDes = ''
    for c, est in zip(headPoseEstimatorCodes, headPoseEstimators):
        poseEstDes += '\'%s\': %s\n' % (c, est)

    estimatorDes = faceDetDes + lndmrkDetDes + poseEstDes
    return faceDetDes, lndmrkDetDes, poseEstDes, estimatorDes

def getFinalParser():
    faceDetDes, lndmrkDetDes, poseEstDes, estimatorDes = getEstimatorsInfo()
        
    description = 'Head-based Interaction Demos'
    parser = ArgumentParser(description = description,
                           formatter_class = RawTextHelpFormatter)

    demos = ['Input', 'Mapping']
    demoHelp = 'Select a module to run'
    parser.add_argument('module', help = demoHelp, choices = demos)
    
    estHelp = 'Select InputEstimators to run ' \
         '(Multiple estimators run simultaneously).\n' \
         'Find the available estimators below:\n' + estimatorDes
    est = 'estimators'
    parser.add_argument(est, metavar = 'Est', nargs= '+', 
                        help = estHelp, choices = estimatorCodes)
    
    mappings = ['Static', 'Dynamic']
    estHelp = 'Select MappingFunctions to apply ' \
         '(Required when using \'Mapping\' module and \n' \
         'You must pass as many functions as InputEstimators).\n' \
         'Find the available MappingFunctions below:\n' \
         '\'%s\' : Static Mapping Function\n' \
         '\'%s\': Dynamic Mapping Function' % (*mappings,)
    mf = '--mappingFunctions'
    parser.add_argument('-mf', mf, metavar = 'MF', nargs='+', 
                        help = estHelp, choices = mappings, 
                        required = 'Mapping' in sys.argv)

    fdHelp = 'Select FaceDetectors to change ' \
         '(You must pass as many detectors as InputEstimators).\n' \
         'Find the available detectors below:\n' \
         '\'_\'     : Pass to keep the estimator\'s default ' \
         'FaceDetector\n' + faceDetDes
    fd = '--faceDetectors'
    parser.add_argument('-fd', fd, metavar = 'FD', nargs='+', 
                        help = fdHelp, choices = ['_']+faceDetectorCodes)

    ldHelp = 'Select LandmarkDetectors to change ' \
         '(You must pass as many detectors as InputEstimators).\n' \
         'Find the available detectors below:\n' \
         '\'_\'     : Pass to keep the estimator\'s default ' \
         'LandmarkDetector\n' + lndmrkDetDes
    ld = '--landmarkDetectors'
    parser.add_argument('-ld', ld, metavar = 'LD', nargs='+', 
                        help = ldHelp, choices = ['_']+landmarkDetectorCodes)

    return parser

def getSafeArgs():
    parser = getFinalParser()
    args = parser.parse_args()

    num_ests = len(args.estimators)

    if args.module == 'Mapping' and len(args.mappingFunctions) != num_ests:
        parser.error('You must pass as many MappingFunctions as ' \
                      'InputEstimators when using -mf/--mappingFunctions')

    if args.module == 'Input' and args.mappingFunctions:
        parser.error('Input Estimators module does not accept ' \
                      'Mapping Functions, do not use -mf/--mappingFunctions ' \
                      'or use \'Mapping\' module instead of \'Input\'.')
    
    faceDetIndices = [i for i in range(num_ests)
                      if args.estimators[i][-2:] == 'FD']
    if args.faceDetectors:
        if len(args.faceDetectors) != num_ests:
            parser.error('You must pass as many FaceDetectors as ' \
                          'InputEstimators when using -fd/--faceDetectors')

        for i in faceDetIndices:
            if not args.faceDetectors[i] in [args.estimators[i], '_']:
                parser.error('%s cannot be FaceDetector of %s, ' \
                              'try to use \'%s\' itself or \'_\', ' \
                              'or set \'%s\' as the estimator.' \
                             % (args.faceDetectors[i], args.estimators[i],
                               args.estimators[i], args.faceDetectors[i]))
            
    landmarkDetIndices = [i for i in range(num_ests)
                      if args.estimators[i][-2:] == 'LD']
    if args.landmarkDetectors:
        if len(args.landmarkDetectors) != num_ests:
            parser.error('You must pass as many FaceDetectors as ' \
                          'InputEstimators when using -ld/--landmarkDetectors')
            
        for i in faceDetIndices:
            if args.landmarkDetectors[i] != '_':
                parser.error('%s cannot have a LandmarkDetector, ' \
                              'pass \'_\' instead of %s.' \
                             % (args.estimators[i], args.landmarkDetectors[i]))
                
        for i in landmarkDetIndices:
            if not args.landmarkDetectors[i] in [args.estimators[i], '_']:
                parser.error('%s cannot be LandmarkDetector of %s, ' \
                              'try to use \'%s\' itself or \'_\', ' \
                              'or set \'%s\' as the estimator.' \
                             % (args.landmarkDetectors[i], args.estimators[i],
                               args.estimators[i], args.landmarkDetectors[i]))
    return args

def getArgsWithRawEstimators():
    args = getSafeArgs()
    estDict = estimatorDict.copy()
    estDict['_'] = 'None'
    
    givenEstimators = [estDict[e] for e in args.estimators]
    num_ests = len(args.estimators)
    
    givenFaceDetectors = [None] * num_ests
    if args.faceDetectors:
        givenFaceDetectors = [estDict[d] for d in args.faceDetectors]

    givenLandmarkDetectors = [None] * num_ests
    if args.landmarkDetectors:
        givenLandmarkDetectors = [estDict[d] for d in args.landmarkDetectors]

    combos = []
    for i in range(num_ests):
        if args.module == 'Mapping':
            mapping = args.mappingFunctions[i] + 'Mapping'
            combos.append((mapping, givenEstimators[i], 
                           givenFaceDetectors[i], givenLandmarkDetectors[i]))
        else:
            combos.append((givenEstimators[i], faceDetectors[i],
                          givenLandmarkDetectors[i]))
        print(combos[i])
    
    return args, combos