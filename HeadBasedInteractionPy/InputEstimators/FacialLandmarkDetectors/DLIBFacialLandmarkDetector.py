# The code is derived from the following repository:
# https://github.com/lincolnhard/head-pose-estimation

from InputEstimators.FaceDetectors.DLIBFrontalFaceDetector import DLIBFrontalFaceDetector
from InputEstimators.FacialLandmarkDetectors.FacialLandmarkDetectorABC import FacialLandmarkDetectorABC
from paths import DLIB_face_landmark_model_path
from imutils import face_utils
import dlib

class DLIBFacialLandmarkDetector(FacialLandmarkDetectorABC):
    
    def __init__(self, faceDetector = None, inputLandmarkIndex = 30, face_landmark_path = None, *args, **kwargs):
        if faceDetector == None:
            faceDetector = DLIBFrontalFaceDetector()
        super().__init__(faceDetector, inputLandmarkIndex, *args, **kwargs)

        if face_landmark_path == None:
            self.__face_landmark_path = DLIB_face_landmark_model_path
        else:
            self.__face_landmark_path = face_landmark_path

        self._landmarkDetector = dlib.shape_predictor(self.__face_landmark_path)
        
    def detectFacialLandmarks(self, frame):
        faceBox = self._faceDetector.detectFaceBox(frame)       
        if faceBox == None:
            return self.facialLandmarks
        else:
            faceBox = dlib.rectangle(faceBox.left, faceBox.top, faceBox.right, faceBox.bottom)
            shape = self._landmarkDetector(frame, faceBox)
            shape = face_utils.shape_to_np(shape)
            self._facialLandmarks = shape
            return self.facialLandmarks