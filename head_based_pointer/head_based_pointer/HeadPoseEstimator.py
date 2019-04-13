import cv2
import dlib
import numpy as np
from imutils import face_utils

class HeadPoseEstimator():
    def __init__(self, face_landmark_path = None):

        if face_landmark_path == None:
            self.__face_landmark_path = 'C:/cStorage/Datasets/Dlib/shape_predictor_68_face_landmarks.dat'

        self.__K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
             0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
             0.0, 0.0, 1.0]
        self.__D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

        self.__cam_matrix = np.array(self.__K).reshape(3, 3).astype(np.float32)
        self.__dist_coeffs = np.array(self.__D).reshape(5, 1).astype(np.float32)

        self.__object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                                 [1.330353, 7.122144, 6.903745],
                                 [-1.330353, 7.122144, 6.903745],
                                 [-6.825897, 6.760612, 4.402142],
                                 [5.311432, 5.485328, 3.987654],
                                 [1.789930, 5.393625, 4.413414],
                                 [-1.789930, 5.393625, 4.413414],
                                 [-5.311432, 5.485328, 3.987654],
                                 [2.005628, 1.409845, 6.165652],
                                 [-2.005628, 1.409845, 6.165652],
                                 [2.774015, -2.080775, 5.048531],
                                 [-2.774015, -2.080775, 5.048531],
                                 [0.000000, -3.116408, 6.097667],
                                 [0.000000, -7.415691, 4.070434]])

        self.__reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                   [10.0, 10.0, -10.0],
                                   [10.0, -10.0, -10.0],
                                   [10.0, -10.0, 10.0],
                                   [-10.0, 10.0, 10.0],
                                   [-10.0, 10.0, -10.0],
                                   [-10.0, -10.0, -10.0],
                                   [-10.0, -10.0, 10.0]])

        self.__line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]]

        self.__detector = dlib.get_frontal_face_detector()
        self.__predictor = dlib.shape_predictor(self.__face_landmark_path)
        
        self.__euler_angle = np.zeros((3, 1))

    def __get_head_pose_from_shape(self, shape):
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])

        _, rotation_vec, translation_vec = cv2.solvePnP(self.__object_pts, image_pts, self.__cam_matrix, self.__dist_coeffs)

        reprojectdst, _ = cv2.projectPoints(self.__reprojectsrc, rotation_vec, translation_vec, self.__cam_matrix,
                                            self.__dist_coeffs)

        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, self.__euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        return reprojectdst, self.__euler_angle
    
    def get_head_pose_with_annotations(self, frame):
        face_rects = self.__detector(frame, 0)

        if len(face_rects) <= 0:
            return None
        else:
            shape = self.__predictor(frame, face_rects[0])
            shape = face_utils.shape_to_np(shape)

            reprojectdst, self.__euler_angle = self.__get_head_pose_from_shape(shape)

            return shape, reprojectdst, self.__euler_angle
    
    def get_head_pose_in_euler_angle(self, frame):
        headPose = self.get_head_pose_with_annotations(frame)
        if headPose:
            shape, reprojectdst, self.__euler_angle = headPose
        return self.__euler_angle        

    def get_line_pairs(self):
        return self.__line_pairs
    
    def get_frame_with_annotations(self, frame, showDots = True, showLines = True, showValues = True):
        
        headPose = self.get_head_pose_with_annotations(frame)
        if headPose:
            shape, reprojectdst, self.__euler_angle = headPose
            if showDots:
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            if showLines:
                for start, end in self.__line_pairs:
                    cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

        if showValues:
            cv2.putText(frame, "X: " + "{:7.2f}".format(self.__euler_angle[0, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)
            cv2.putText(frame, "Y: " + "{:7.2f}".format(self.__euler_angle[1, 0]), (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)
            cv2.putText(frame, "Z: " + "{:7.2f}".format(self.__euler_angle[2, 0]), (20, 140), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)
        return frame

    def streamDemo(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Unable to connect to camera.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = self.get_frame_with_annotations(frame, showDots = True, showLines = True, showValues = True)

            cv2.imshow("demo", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def main():
    headPoseEstimator = HeadPoseEstimator()
    headPoseEstimator.streamDemo()

if __name__ == '__main__':
    main()
