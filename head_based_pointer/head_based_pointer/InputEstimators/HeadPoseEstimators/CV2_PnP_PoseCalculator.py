import cv2, numpy as np

class CV2_PnP_PoseCalculator(object):

    def __init__(self):

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
        
        self.__rotation_vec = None
        self.__translation_vec = None

    def calculatePoseFromShape(self, shape):
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],

                                shape[48], shape[54], shape[57], shape[8]])
        _, self.__rotation_vec, self.__translation_vec = cv2.solvePnP(self.__object_pts, image_pts, self.__cam_matrix, self.__dist_coeffs)

        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(self.__rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, self.__translation_vec))
        _, _, _, _, _, _, self.__euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        return self.__euler_angle.reshape((3,))
    
    def calculateProjectionPointsFromShape(self, shape, recalculatePose = False):
        if recalculatePose:
                self.calculatePoseFromShape(shape)
        self.__projectionPoints, _ = cv2.projectPoints(self.__reprojectsrc, self.__rotation_vec, self.__translation_vec,
                                                self.__cam_matrix, self.__dist_coeffs)
        self.__projectionPoints = tuple(map(tuple, self.__projectionPoints.reshape(8, 2)))
        self.__projectionPoints = [(self.__projectionPoints[start], self.__projectionPoints[end]) for start, end in self.__line_pairs]
        return self.__projectionPoints