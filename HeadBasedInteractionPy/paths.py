# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

import os

InputOutputFolders = 'InputOutputFolders/'
DATASETS_Folder = InputOutputFolders + 'ModelData/'

DLIB_face_landmark_model_path = DATASETS_Folder + 'Dlib/shape_predictor_68_face_landmarks.dat'

HaarCascade_FrontalFace_path =  DATASETS_Folder + 'CV2Nets/HaarCascade/haarcascade_frontalface_default.xml'

CV2Res10SSD_frozen_Folder = DATASETS_Folder + 'CV2Nets/CV2Res10SSD/'
CV2Res10SSD_frozen_face_model_path = CV2Res10SSD_frozen_Folder + 'face68_model.txt'

TF_Models_Folder = DATASETS_Folder + 'TF_Models/'
TFMobileNetSSDFaceDetector_tf_model_path = TF_Models_Folder + 'TFMobileNetSSDFaceDetector/frozen_inference_graph_face.pb'
YinsFacialLandmarkDetector_tf_model_path = TF_Models_Folder + 'YinsCNNBasedFacialLandmarkDetector/frozen_inference_graph.pb'

DemoVideos_Folder = InputOutputFolders + 'DemoVideos/'
InputEstimatorsDemo_Folder = DemoVideos_Folder + 'InputEstimatorDemos/'

if 'COMPUTERNAME' in os.environ:
    if os.environ['COMPUTERNAME'] == "MSI2":
        OfflineProjectFolder = 'C:/Users/MSI2/Documents/Offline_Projects/HeadBasedPointer/'
        DemoVideos_Offline_Folder = OfflineProjectFolder + 'DemoVideos/'
        Experiments_Folder = OfflineProjectFolder + 'Experiments/'

