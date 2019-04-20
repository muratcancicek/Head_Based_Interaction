from abc import ABC, abstractmethod
import numpy as np
import cv2

class PoseCalculatorABC(ABC):
    @staticmethod
    def _get_3d_points(rear_size = 7.5, rear_depth = 0, front_size = 10.0, front_depth = 10.0):
        point_3d = []
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))
                
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
        return point_3d

    def __init__(self, *args, **kwargs):
        self._pose = np.zeros((3,))
        self._rectCorners3D = self._get_3d_points()
        self._projectionPoints = None
        super().__init__(*args, **kwargs)
        
    @abstractmethod
    def calculatePose(self, shape):
        return NotImplemented
    
    def calculateProjectionPoints(self, shape, recalculatePose = False):
        if recalculatePose:
                self.calculatePose(shape)
        if not (self._rotation_vector is None or self._translation_vector is None):
            point_2d, _ = cv2.projectPoints(self._rectCorners3D, self._rotation_vector, self._translation_vector, self._camera_matrix, self._dist_coeffs)
            self._projectionPoints = np.int32(point_2d.reshape(-1, 2))
        return self._projectionPoints

    @property
    def pose(self):
        return self._pose