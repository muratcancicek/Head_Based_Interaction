from InputEstimators.FaceDetectors.FaceDetectorABC import FaceDetectorABC
from InputEstimators.FaceDetectors.FaceBox import FaceBox
import dlib

class DLIBFrontalFaceDetector(FaceDetectorABC):
    def __init__(self, *args, **kwargs):
        self.__detector = dlib.get_frontal_face_detector()
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def _decodeFaceBox(detection):
        return FaceBox(detection.left(), detection.top(), detection.right(), detection.bottom())

    def _detectFaceBox(self, frame):
        face_rects = self.__detector(frame, 0)
        if len(face_rects) <= 0:
            return self._faceBox
        else:
            self._faceBox = self._decodeFaceBox(face_rects[0])
        return self._faceBox