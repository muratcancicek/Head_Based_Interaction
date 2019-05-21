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

    def calculatePoseWithMatrices(self, shape):
        pose = self.solve_pose_by_68_points(shape)
        # Stabilize the pose.
        stabile_pose = []
        pose_np = np.array(pose).flatten()
        for value, ps_stb in zip(pose_np, self._pose_stabilizers):
            ps_stb.update([value])
            stabile_pose.append(ps_stb.state[0])
        rotation_vector, translation_vector = np.reshape(stabile_pose, (-1, 3))
        
        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat((rotation_mat, translation_vector))
        cameraMatrix, rotMatrix, transVect, rotMatrixX, \
        rotMatrixY, rotMatrixZ, pose = cv2.decomposeProjectionMatrix(pose_mat)
        self._pose[0] = pose[0] * (-1)
        self._pose[1] = pose[1]
        self._pose[2] = (pose[2] - 180) if pose[2] > 0 else pose[2] + 180
        self._pose = self._pose.reshape((3,))
        return cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, self._pose 
    
    def calculatePose(self, shape):
        self.calculatePoseWithMatrices(shape)
        return self._pose
    
    def redefine3DProjectionPoints(rear_size = 50, rear_depth = 0, front_size = 50, front_depth = 100):
        self._rectCorners3D = self._get_3d_points(rear_size, rear_depth, front_size, front_depth)
        

    def calculateProjectionPoints(self, shape, recalculatePose = False):
        if recalculatePose:
                self.calculatePose(shape)
        if not (self._rotation_vector is None or self._translation_vector is None):
            point_2d, _ = cv2.projectPoints(self._rectCorners3D, self._rotation_vector, 
                                            self._translation_vector, self._camera_matrix,
                                           self._dist_coeffs)
            self._projectionPoints = np.int32(point_2d.reshape(-1, 2))
        return self._projectionPoints

    def calculateProjectionPointsWithGaze(self, shape, recalculatePose = False):
        def pArr(a):
           return str(a).replace('\n', ' ')
        if recalculatePose:
                self.calculatePose(shape)
        if not (self._rotation_vector is None or self._translation_vector is None):
            self._front_depth = -self._translation_vector[2, 0]
            point_2d, _ = cv2.projectPoints(self._rectCorners3D, self._rotation_vector, 
                                            self._translation_vector, self._camera_matrix,
                                           self._dist_coeffs)
                        
            #print('\r', f1, f2, c1, c2, end= '\r' )
            print('\r', point_2d[-1], end= '\r' )
            

            
            self._projectionPoints = np.int32(point_2d.reshape(-1, 2))
        return self._projectionPoints