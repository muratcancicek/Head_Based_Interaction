from InputEstimators.InputEstimatorABC import InputEstimatorABC
from InputEstimators.HeadPoseEstimators.PoseCalculators.PoseCalculatorABC import PoseCalculatorABC
from abc import abstractmethod
import cv2, numpy as np

class CV2_PnP_with_KF_HeadPoseCalculator(PoseCalculatorABC):
    @staticmethod
    def __getCameraMatrix(size):
        focal_length = size[1]
        camera_center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, camera_center[0]],
             [0, focal_length, camera_center[1]],
             [0, 0, 1]], dtype="double")
        return camera_matrix

    @staticmethod
    def __get_pose_stabilizers():
        Stabilizer = CV2_PnP_with_KF_HeadPoseCalculator.Stabilizer
        stabilizers = []
        for _ in range(6):
            stabilizers.append(Stabilizer(state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1) )
        return stabilizers

    @staticmethod
    def __get_full_model_points(filename):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        # model_points *= 4
        model_points[:, -1] *= -1
        return model_points
    
    def __init__(self, face_model_path = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if face_model_path == None:
            face_model_path = 'C:/cStorage/Datasets/CV2Nets/CV2Res10SSD/face68_model.txt'
        self._faceModelPoints = self.__get_full_model_points(face_model_path)
        self._rectCorners3D = self._get_3d_points(rear_size = 75, rear_depth = 0, front_size = 100, front_depth = 100)

        size = [480, 640]

        # Camera internals
        self._camera_matrix = self.__getCameraMatrix(size)

        # Assuming no lens distortion
        self._dist_coeffs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self._rotation_vector = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self._translation_vector = np.array([[-14.97821226], [-10.62040383], [-2053.03596872]])
        
        self.__pose_stabilizers = self.__get_pose_stabilizers()

    def solve_pose_by_68_points(self, image_points): 
        image_points = image_points.astype(np.float32)
        (_, rotation_vector, translation_vector) = cv2.solvePnP(self._faceModelPoints, image_points, self._camera_matrix, self._dist_coeffs,
                                                    rvec=self._rotation_vector, tvec=self._translation_vector, useExtrinsicGuess=True)
        return (rotation_vector, translation_vector)

    def calculatePose(self, shape):
        pose = self.solve_pose_by_68_points(shape)
        # Stabilize the pose.
        stabile_pose = []
        pose_np = np.array(pose).flatten()
        for value, ps_stb in zip(pose_np, self.__pose_stabilizers):
            ps_stb.update([value])
            stabile_pose.append(ps_stb.state[0])
        rotation_vector, translation_vector = np.reshape(stabile_pose, (-1, 3))
        
        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat((rotation_mat, translation_vector))
        _, _, _, _, _, _, pose = cv2.decomposeProjectionMatrix(pose_mat)
        self._pose[0] = pose[0] * (-1)
        self._pose[1] = pose[1]
        self._pose[2] = (pose[2] - 180) if pose[2] > 0 else pose[2] + 180
        return self._pose.reshape((3,))
    
    class Stabilizer:
        """Using Kalman filter as a point stabilizer."""

        def __init__(self, state_num=4, measure_num=2, cov_process=0.0001, cov_measure=0.1):
            """Initialization"""
            # Currently we only support scalar and point, so check user input first.
            assert state_num == 4 or state_num == 2, "Only scalar and point supported, Check state_num please."

            # Store the parameters.
            self.state_num = state_num
            self.measure_num = measure_num

            # The filter itself.
            self.filter = cv2.KalmanFilter(state_num, measure_num, 0)

            # Store the state.
            self.state = np.zeros((state_num, 1), dtype=np.float32)

            # Store the measurement result.
            self.measurement = np.array((measure_num, 1), np.float32)

            # Store the prediction.
            self.prediction = np.zeros((state_num, 1), np.float32)

            # Kalman parameters setup for scalar.
            if self.measure_num == 1:
                self.filter.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)

                self.filter.measurementMatrix = np.array([[1, 1]], np.float32)

                self.filter.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * cov_process

                self.filter.measurementNoiseCov = np.array( [[1]], np.float32) * cov_measure

            # Kalman parameters setup for point.
            if self.measure_num == 2:
                self.filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                         [0, 1, 0, 1],
                                                         [0, 0, 1, 0],
                                                         [0, 0, 0, 1]], np.float32)

                self.filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                          [0, 1, 0, 0]], np.float32)

                self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1]], np.float32) * cov_process

                self.filter.measurementNoiseCov = np.array([[1, 0],
                                                            [0, 1]], np.float32) * cov_measure

        def update(self, measurement):
            """Update the filter"""
            # Make kalman prediction
            self.prediction = self.filter.predict()

            # Get new measurement
            if self.measure_num == 1:
                self.measurement = np.array([[np.float32(measurement[0])]])
            else:
                self.measurement = np.array([[np.float32(measurement[0])],
                                             [np.float32(measurement[1])]])

            # Correct according to mesurement
            self.filter.correct(self.measurement)

            # Update state value.
            self.state = self.filter.statePost

        def set_q_r(self, cov_process=0.1, cov_measure=0.001):
            """Set new value for processNoiseCov and measurementNoiseCov."""
            if self.measure_num == 1:
                self.filter.processNoiseCov = np.array([[1, 0],
                                                        [0, 1]], np.float32) * cov_process
                self.filter.measurementNoiseCov = np.array(
                    [[1]], np.float32) * cov_measure
            else:
                self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1]], np.float32) * cov_process
                self.filter.measurementNoiseCov = np.array([[1, 0],
                                                            [0, 1]], np.float32) * cov_measure