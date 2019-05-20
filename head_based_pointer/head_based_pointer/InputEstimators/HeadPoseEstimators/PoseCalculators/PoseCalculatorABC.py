from abc import ABC, abstractmethod
import numpy as np
import cv2

class PoseCalculatorABC(ABC):
    @staticmethod
    def _get_3d_points(rear_size = 7.5, rear_depth = 0, front_size = 10.0, front_depth = 10.0):
        point_3d = []
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        #point_3d.append((rear_size, rear_size, rear_depth))
        #point_3d.append((rear_size, -rear_size, rear_depth))
        #point_3d.append((-rear_size, -rear_size, rear_depth))
                
        #point_3d.append((-front_size, -front_size, front_depth))
        #point_3d.append((-front_size, front_size, front_depth))
        #point_3d.append((front_size, front_size, front_depth))
        #point_3d.append((front_size, -front_size, front_depth))
        #point_3d.append((-front_size, -front_size, front_depth))
        
        point_3d.append((0, 0, 0))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype='float32').reshape(-1, 3)
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
            self._rectCorners3D[-1, :2] = self._translation_vector[:2].reshape(-1, 2)
            point_2d, _ = cv2.projectPoints(self._rectCorners3D, self._rotation_vector, self._translation_vector, self._camera_matrix, self._dist_coeffs)
            #t = np.array(self._translation_vector, dtype=np.float).reshape(-1, 3)#[(0, 0, 0)], tvectype(self._rectCorners3D), point_2d[-1], 
            #p, _ = cv2.projectPoints(t, self._rotation_vector, self._translation_vector, self._dist_coeffs)
            point_2d = point_2d.reshape(-1, 2).astype('float32')
            #print(point_2d)
            retval, cameraMatrix, distCoeffs, rvecs, tvec = cv2.calibrateCamera([self._rectCorners3D], [point_2d], (640, 360)
                                                                                , self._camera_matrix, self._dist_coeffs)

            print('\r', cameraMatrix[0], tvec[0].reshape(-1, 3), end= '\r' )
            self._projectionPoints = np.int32(point_2d.reshape(-1, 2))
        return self._projectionPoints

    @property
    def pose(self):
        return self._pose