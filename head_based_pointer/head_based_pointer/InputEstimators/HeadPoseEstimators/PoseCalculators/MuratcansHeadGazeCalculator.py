# The code is derived from the following repository:
# https://github.com/yinguobing/head-pose-estimation

from InputEstimators.HeadPoseEstimators.PoseCalculators.YinsKalmanFilteredHeadPoseCalculator import YinsKalmanFilteredHeadPoseCalculator
import cv2, numpy as np

class MuratcansHeadGazeCalculator(YinsKalmanFilteredHeadPoseCalculator):
       
    def __init__(self, face_model_path = None, inputFramesize = (720, 1280), *args, **kwargs):
        super().__init__(face_model_path, inputFramesize, *args, **kwargs)
        print(inputFramesize)
        self._translation_vector = np.array([[-14.97821226], [-10.62040383], [-200]])#-2053.03596872
        
        self._front_depth = 500
        self._rectCorners3D = self._get_3d_points(rear_size = 10, rear_depth = 0, 
                                                  front_size = 10, front_depth = self._front_depth)
        self._objectPointsVec = [self._faceModelPoints]*7
        self._imagePointsVec = []
        #print(self._faceModelPoints)
        
    def calculatePose(self, shape):
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
        return self._pose

    def calibrateCamera(self, imagePoints):
        ip = imagePoints.astype('float32')        
        print(imagePoints)
        self._imagePointsVec.append(ip)
        n = 7
        if len(self._imagePointsVec) < n+1:
            return
        self._imagePointsVec.pop(0)
        print(ip.shape, self._faceModelPoints.shape, len(self._objectPointsVec), len(self._imagePointsVec))
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(self._objectPointsVec, self._imagePointsVec, (640, 360), 
                                                                             self._camera_matrix, self._dist_coeffs,
                                                                             flags=(cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT))
        self._camera_matrix, self._dist_coeffs = cameraMatrix, distCoeffs
        self._rotation_vector, self._translation_vector = rvecs[0], tvecs[0]

    def calculateProjectionPointsAsGaze(self, shape, recalculatePose = False):
        if recalculatePose:
                self.calculatePose(shape)
        if not (self._rotation_vector is None or self._translation_vector is None):
            self._front_depth = -self._translation_vector[2, 0]
            self._rectCorners3D = self._get_3d_points(rear_size = 1, rear_depth = 0, 
                                                    front_size = 1, front_depth = self._front_depth)
            #self.calibrateCamera(shape)
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
        return self._pose, self._projectionPoints
