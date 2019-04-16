from InputEstimators.FaceDetectors.DLIBFrontalFaceDetector import DLIBFrontalFaceDetector
from InputEstimators.FacialLandmarkDetectors.FacialLandmarkDetectorABC import FacialLandmarkDetectorABC
from imutils import face_utils
import dlib

class DLIBFacialLandmarkDetector(FacialLandmarkDetectorABC):
    
    def __init__(self, faceDetector = None, face_landmark_path = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if faceDetector == None:
            self._faceDetector = DLIBFrontalFaceDetector()
        else:
            self._faceDetector = faceDetector

        if face_landmark_path == None:
            self.__face_landmark_path = 'C:/cStorage/Datasets/Dlib/shape_predictor_68_face_landmarks.dat'
        else:
            self.__face_landmark_path = face_landmark_path

        self.__predictor = dlib.shape_predictor(self.__face_landmark_path)
        
    def detectFacialLandmarks(self, frame):
        face_rects = self._faceDetector.detectFaceBox(frame)        
        if len(face_rects) <= 0:
            return self.facialLandmarks
        else:
            shape = self.__predictor(frame, face_rects[0])
            shape = face_utils.shape_to_np(shape)
            self._facialLandmarks = shape
            return self.facialLandmarks

    def _findInputLandmark(self, frame):
        self._inputLandmark[:2] = self.detectFacialLandmarks(frame)[30]
        return self._inputLandmark
    
    def _findInputLandmarkWithAnnotations(self, frame):
        return self._findInputLandmark(frame), self._faceDetector.getProjectionPoints(), self._facialLandmarks
