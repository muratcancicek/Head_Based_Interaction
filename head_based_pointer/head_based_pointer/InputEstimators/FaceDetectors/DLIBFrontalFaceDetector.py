from InputEstimators.FaceDetectors.FaceDetectorABC import FaceDetectorABC
import dlib, cv2, numpy as np

class DLIBFrontalFaceDetector(FaceDetectorABC):
    def __init__(self, *args, **kwargs):
        self.__detector = dlib.get_frontal_face_detector()
        super().__init__(*args, **kwargs)

    def detectFaceBox(self, frame):
        self._faceBox = self.__detector(frame, 0)
        return self._faceBox
        
    def findFaceLocation(self, frame):
        face_rects = self.__detector(frame, 0)
        if len(face_rects) <= 0:
            return self._faceLocation
        else:
            self._faceBox = face_rects
            self._faceLocation[0] = self._faceBox[0].center().x
            self._faceLocation[1] = self._faceBox[0].center().y
            return self._faceLocation

    def getProjectionPoints(self):
        if len(self._faceBox) <= 0:
            return []
        face = self._faceBox[0]
        corners = [face.tl_corner(), face.tr_corner(), face.br_corner(), face.bl_corner()]
        corners = [(c.x, c.y) for c in corners]
        return [(corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[3]), (corners[3], corners[0])]

    def findFaceLocationWithAnnotations(self, frame):
        return self.findFaceLocation(frame), self.getProjectionPoints(), []