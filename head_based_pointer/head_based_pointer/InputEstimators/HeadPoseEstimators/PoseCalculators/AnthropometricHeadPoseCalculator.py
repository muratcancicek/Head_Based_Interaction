# The code is derived from the following repository:
# https://github.com/lincolnhard/head-pose-estimation

from InputEstimators.HeadPoseEstimators.PoseCalculators.PoseCalculatorABC import PoseCalculatorABC
from InputEstimators.InputEstimatorABC import InputEstimatorABC
from abc import abstractmethod
import cv2, numpy as np

class AnthropometricHeadPoseCalculator(PoseCalculatorABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rectCorners3D = self._get_3d_points(rear_size = 8, rear_depth = 7, front_size = 10, front_depth = 14)

        self.__K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
             0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
             0.0, 0.0, 1.0]
        self.__D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

        self._camera_matrix = np.array(self.__K).reshape(3, 3).astype(np.float32)
        self._dist_coeffs = np.array(self.__D).reshape(5, 1).astype(np.float32)

        self._faceModelPoints = np.float32([[6.825897, 6.760612, 4.402142],
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
        
        self._rotation_vector = None
        self._translation_vector = None

    def calculatePose(self, shape):
        shape_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])

        _, self._rotation_vector, self._translation_vector = cv2.solvePnP(self._faceModelPoints, shape_pts, self._camera_matrix, self._dist_coeffs)

        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(self._rotation_vector)
        pose_mat = cv2.hconcat((rotation_mat, self._translation_vector))
        _, _, _, _, _, _, self._pose = cv2.decomposeProjectionMatrix(pose_mat)
        self._pose[1] *= -1
        return self._pose.reshape((3,))