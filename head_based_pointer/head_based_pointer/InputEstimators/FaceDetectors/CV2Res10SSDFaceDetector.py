from InputEstimators.FaceDetectors.FaceDetectorABC import FaceDetectorABC
from InputEstimators.FaceDetectors.FaceBox import FaceBox
import cv2

class CV2Res10SSDFaceDetector(FaceDetectorABC):
    def __init__(self, confidence_threshold = 0.90, dnn_proto_text_path = None, dnn_model_path = None, *args, **kwargs):
        if dnn_proto_text_path == None:
            dnn_proto_text_path = 'C:/cStorage/Datasets/CV2Nets/CV2Res10SSD/deploy.prototxt'
        if dnn_model_path == None:
            dnn_model_path = 'C:/cStorage/Datasets/CV2Nets/CV2Res10SSD/res10_300x300_ssd_iter_140000.caffemodel'
        self.__detector = cv2.dnn.readNetFromCaffe(dnn_proto_text_path, dnn_model_path)
        self.__confidence_threshold = confidence_threshold  
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def _decodeFaceBox(detection):
        (rows, cols, _), detection = detection
        x_left_bottom = int(detection[3] * cols)
        y_left_bottom = int(detection[4] * rows)
        x_right_top = int(detection[5] * cols)
        y_right_top = int(detection[6] * rows)
        return FaceBox(x_left_bottom, y_right_top, x_right_top, y_left_bottom)

    def _detectFaceBox(self, frame):
        confidences = []
        faceBoxDetections = []

        self.__detector.setInput(cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))
        detections = self.__detector.forward()

        for detection in detections[0, 0, :, :]:
            confidence = detection[2]
            if confidence > self.__confidence_threshold:
                confidences.append(confidence)
                faceBoxDetections.append((frame.shape, detection))

        if len(faceBoxDetections) > 0: 
            self._faceBox = self._decodeFaceBox(faceBoxDetections[0])
        return self._faceBox
        