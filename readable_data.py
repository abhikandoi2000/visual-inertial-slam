import numpy as np
from utils import *


if __name__ == '__main__':
	filename = "./data/0022.npz"
	t, features, linear_velocity, \
		rotational_velocity, K, b, cam_T_imu = load_data(filename)

	how_many = 6
	print('t.shape', t.shape)
	print((t[:,:how_many].reshape((how_many,))*100000000).astype(np.int64))
	print(t[0,5]-t[0,0])
	print('\n\n')
	# print(features.shape)
	print('linear_velocity.shape', linear_velocity.shape)
	print(linear_velocity[:,:how_many])
	print('\n\n')
	print('rotational_velocity.shape', rotational_velocity.shape)
	print(rotational_velocity[:,:how_many])
	print('\n\n')
	# print(K, b)
	# print(cam_T_imu)

	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
	#visualize_trajectory_2d(world_T_imu,show_ori=True)
