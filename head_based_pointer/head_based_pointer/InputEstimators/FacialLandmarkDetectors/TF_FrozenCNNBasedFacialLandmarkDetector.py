from InputEstimators.FaceDetectors.CV2Res10SSDFaceDetector import CV2Res10SSDFaceDetector
from InputEstimators.FacialLandmarkDetectors.FacialLandmarkDetectorABC import FacialLandmarkDetectorABC
import cv2, numpy as np, tensorflow as tf

class TF_FrozenCNNBasedFacialLandmarkDetector(FacialLandmarkDetectorABC):

    @staticmethod
    def loadTFGraph(tf_model_path):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(tf_model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def __init__(self, faceDetector = None, inputLandmarkIndex = 30, tf_model_path = None, *args, **kwargs):
        if faceDetector == None:
            faceDetector = CV2Res10SSDFaceDetector(squaringFaceBox = True)
        super().__init__(faceDetector, inputLandmarkIndex, *args, **kwargs)
        if tf_model_path == None:
            tf_model_path = 'C:/cStorage/Datasets/CV2Nets/CV2Res10SSD/frozen_inference_graph.pb'

        self.graph = self.loadTFGraph(tf_model_path)
        self.sess = tf.Session(graph = self.graph)
        self._cnn_input_size = 128

    def __detectFaceImage(self, frame):
        faceBox = self._faceDetector.detectFaceBox(frame)
        if faceBox == None:
            squaredFaceImage = frame
        else:
            squaredFaceImage = faceBox.getSquaredFaceImageFromFrame(frame)
        squaredFaceImage = cv2.resize(squaredFaceImage, (self._cnn_input_size, self._cnn_input_size))
        return cv2.cvtColor(squaredFaceImage, cv2.COLOR_BGR2RGB)

    def detectFacialLandmarks(self, frame):
        faceImage = self.__detectFaceImage(frame)

        logits_tensor = self.graph.get_tensor_by_name('logits/BiasAdd:0')
        predictions = self.sess.run(logits_tensor, feed_dict={'input_image_tensor:0': faceImage})

        marks = np.array(predictions).flatten()
        marks = np.reshape(marks, (-1, 2))
        #marks = marks * frame.shape[:2]
        #print(marks.shape)
        marks[:, 0] *= self._faceDetector.faceBox.width
        marks[:, 1] *= self._faceDetector.faceBox.height
        marks[:, 0] += self._faceDetector.faceBox.left
        marks[:, 1] += self._faceDetector.faceBox.top
        self._facialLandmarks = marks.astype(int)
        return self._facialLandmarks