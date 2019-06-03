# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

import os

#if 'COMPUTERNAME' in os.environ:
DATASETS_Folder = 'C:/cStorage/Datasets/'
DLIB_face_landmark_model_path = DATASETS_Folder + 'Dlib/shape_predictor_68_face_landmarks.dat'
CV2Res10SSD_frozen_Folder = DATASETS_Folder + 'CV2Nets/CV2Res10SSD/'
TF_Models_Folder = DATASETS_Folder + 'TF_Models/'
TFMobileNetSSDFaceDetector_tf_model_path = TF_Models_Folder + 'TFMobileNetSSDFaceDetector/frozen_inference_graph_face.pb'
YinsFacialLandmarkDetector_tf_model_path = TF_Models_Folder + 'YinsCNNBasedFacialLandmarkDetector/frozen_inference_graph.pb'
CV2Res10SSD_frozen_face_model_path = CV2Res10SSD_frozen_Folder + 'face68_model.txt'
DemoVideos_Folder = 'D:/GoogleDrive/Studying/Graduate/PhD Research/HeadbasedPointer/DemoVideos/'
InputEstimatorsDemo_Folder = DemoVideos_Folder + 'InputEstimatorDemos/'
HaarCascade_FrontalFace_path =  DATASETS_Folder + 'CV2Nets/HaarCascade/haarcascade_frontalface_default.xml'
OfflineProjectFolder = 'C:/Users/MSI2/Documents/Offline_Projects/HeadBasedPointer/'
DemoVideos_Offline_Folder = OfflineProjectFolder + 'DemoVideos/'
Experiments_Folder = OfflineProjectFolder + 'Experiments/'
