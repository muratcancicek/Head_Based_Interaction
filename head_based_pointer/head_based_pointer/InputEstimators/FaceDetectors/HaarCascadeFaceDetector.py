# The code is derived from the following repository:
# https://github.com/lincolnhard/head-pose-estimation

from InputEstimators.FaceDetectors.FaceDetectorABC import FaceDetectorABC
from InputEstimators.FaceDetectors.FaceBox import FaceBox
from paths import HaarCascade_FrontalFace_path
import dlib
import cv2

class HaarCascadeFaceDetector(FaceDetectorABC):
    def __init__(self, haarcascade_frontalface_path = None, *args, **kwargs):
        if haarcascade_frontalface_path == None:
           haarcascade_frontalface_path = HaarCascade_FrontalFace_path
        self.__detector = cv2.CascadeClassifier(haarcascade_frontalface_path)
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def _decodeFaceBox(detection):
        (x,y,w,h) = detection
        return FaceBox(x, y, x+w, y+h)

    def _detectFaceBox(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = self.__detector.detectMultiScale(gray, 1.5, 2)
        if len(face_rects) <= 0:
            return self._faceBox
        else:
            self._faceBox = self._decodeFaceBox(face_rects[0])
        return self._faceBox