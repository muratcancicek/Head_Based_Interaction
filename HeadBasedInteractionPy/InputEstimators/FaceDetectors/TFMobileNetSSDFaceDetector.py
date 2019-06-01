# The code is derived from the following repository:
# https://github.com/yeephycho/tensorflow-face-detection

from InputEstimators.FaceDetectors.FaceDetectorABC import FaceDetectorABC
from InputEstimators.FaceDetectors.FaceBox import FaceBox
from paths import TFMobileNetSSDFaceDetector_tf_model_path
import tensorflow as tf
import numpy as np
import cv2

class TFMobileNetSSDFaceDetector(FaceDetectorABC):
    @staticmethod
    def __convert_label_map_to_categories(label_map, max_num_classes, use_display_name=True):
      categories = []
      list_of_ids_already_added = []
      if not label_map:
        label_id_offset = 1
        for class_id in range(max_num_classes):
          categories.append({'id': class_id + label_id_offset, 'name': 'category_{}'.format(class_id + label_id_offset) })
        return categories
      for item in label_map.item:
        if not 0 < item.id <= max_num_classes:
          logging.info('Ignore item %d since it falls outside of requested label range.', item.id)
          continue
        if use_display_name and item.HasField('display_name'):
          name = item.display_name
        else:
          name = item.name
        if item.id not in list_of_ids_already_added:
          list_of_ids_already_added.append(item.id)
          categories.append({'id': item.id, 'name': name})
      return categories
  
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
    
    @staticmethod
    def _decodeFaceBox(detection):
        (rows, cols, _), detection = detection
        x_left_bottom = int(detection[1] * cols)
        y_left_bottom = int(detection[2] * rows)
        x_right_top = int(detection[3] * cols)
        y_right_top = int(detection[0] * rows)
        return FaceBox(x_left_bottom, y_right_top, x_right_top, y_left_bottom)

    def __init__(self, confidence_threshold = 0.90, model_path = None, *args, **kwargs):
        if model_path == None:
            model_path = TFMobileNetSSDFaceDetector_tf_model_path
      
        self.__graph = self.loadTFGraph(model_path)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.__sess = tf.Session(graph = self.__graph, config=config)

        super().__init__(*args, **kwargs)

    def _detectFirstFaceBox(self, frame):
        image_tensor = self.__graph.get_tensor_by_name('image_tensor:0')
        boxes = self.__graph.get_tensor_by_name('detection_boxes:0')
        boxes = self.__sess.run([boxes], feed_dict={image_tensor: frame})
        return [b for b in boxes[0][0]]

    def _detectAllFaceBoxesWithScores(self, frame):
        image_tensor = self.__graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.__graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.__graph.get_tensor_by_name('detection_scores:0')
        classes = self.__graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.__graph.get_tensor_by_name('num_detections:0')
        outputTensors = [boxes, scores, classes, num_detections]
        # Actual detection.
        outputs = self.__sess.run(outputTensors, feed_dict={image_tensor: frame})
        (boxes, scores, classes, num_detections) = outputs
        return [b for b in boxes[0]]

    def _detectFaceBox(self, frame):
        frameShape = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, axis=0)
        #faceBoxDetections = self._detectAllFaceBoxesWithScores(frame)
        faceBoxDetections = self._detectFirstFaceBox(frame)
        faceBoxDetections = [(frameShape, b) for b in faceBoxDetections]
        if len(faceBoxDetections) > 0: 
            self._faceBox = self._decodeFaceBox(faceBoxDetections[0])
        return self._faceBox
        