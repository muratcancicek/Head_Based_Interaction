# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

faceDetectors = ['CV2Res10SSDFaceDetector', 
                 'DLIBFrontalFaceDetector', 
                 'HaarCascadeFaceDetector', 
                 'TFMobileNetSSDFaceDetector']
faceDetectorCodes = ['CResFD', 'DlibFD', 'HCasFD', 'TMobFD']
faceDetectorDict = {c: est for c, est in zip(faceDetectorCodes, faceDetectors)}

landmarkDetectors = ['DLIBFacialLandmarkDetector', 
                     'YinsCNNBasedFacialLandmarkDetector']
landmarkDetectorCodes = ['DlibLD', 'YinsLD']
landmarkDetectorDict = {c: est for c, est 
                        in zip(landmarkDetectorCodes, landmarkDetectors)}

headPoseEstimators = ['DLIBHeadPoseEstimator', 
                      'YinsHeadPoseEstimator', 
                      'MuratcansHeadGazer']
headPoseEstimatorCodes = ['DlibHP', 'YinsHP', 'MursHG']
headPoseEstimatorDict = {c: est for c, est 
                         in zip(headPoseEstimatorCodes, headPoseEstimators)}

estimators = faceDetectors + landmarkDetectors + headPoseEstimators
estimatorCodes = faceDetectorCodes + \
                 landmarkDetectorCodes + \
                 headPoseEstimatorCodes

estimatorDict = {**faceDetectorDict, 
                 **landmarkDetectorDict, 
                 **headPoseEstimatorDict}

