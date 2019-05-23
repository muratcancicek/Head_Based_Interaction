# The code is derived from the following repository:
# https://github.com/yinguobing/head-pose-estimation

from InputEstimators.HeadPoseEstimators.PoseCalculators.YinsKalmanFilteredHeadPoseCalculator import YinsKalmanFilteredHeadPoseCalculator
import cv2, numpy as np

class MuratcansHeadGazeCalculator(YinsKalmanFilteredHeadPoseCalculator):
       
    def __init__(self, face_model_path = None, inputFramesize = (720, 1280), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._front_depth = 500
        self._rectCorners3D = self._get_3d_points(rear_size = 10, rear_depth = 0, 
                                                  front_size = 10, front_depth = self._front_depth)
        
    def calculateProjectionPointsAsGaze(self, shape, recalculatePose = False):
        if recalculatePose:
                self.calculatePose(shape)
        if not (self._rotation_vector is None or self._translation_vector is None):
            self._front_depth = -self._translation_vector[2, 0]
            self._rectCorners3D = self._get_3d_points(rear_size = 1, rear_depth = 0, 
                                                    front_size = 1, front_depth = self._front_depth)
            point_2d, _ = cv2.projectPoints(self._rectCorners3D, self._rotation_vector, 
                                            self._translation_vector, self._camera_matrix,
                                           self._dist_coeffs)
            self._projectionPoints = np.int32(point_2d.reshape(-1, 2))
        return self._projectionPoints

    def calculateHeadGazeWithProjectionPoints(self, shape):
        self._pose = self.calculatePose(shape)
        self.calculateProjectionPointsAsGaze(shape)
        self._pose[:2] = self._projectionPoints[-1, :] 
        self._pose[2] = 0
        self._pose[0] += 640
        return self._pose, self._projectionPoints
