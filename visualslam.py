import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

from utils import load_data, visualize_trajectory_2d

# Tunable parameters
INITIAL_POSE_COVARIANCE_FACTOR = 0
MOTION_NOISE_COVARIANCE_FACTOR = 0

def hatmap_so3(theta):
    assert theta.shape[0] == 3, 'invalid vector theta, must have exactly three values'
    theta = theta.reshape((3,))
    theta_hat = np.array([[         0,  -theta[2],  theta[1]],
                          [  theta[2],          0, -theta[0]],
                          [ -theta[1],   theta[0],        0]])
    return theta_hat

# see slide 23, lecture 12
def hatmap_se3(theta):
    assert theta.shape[0] == 6, 'invalid vector theta, must have exactly six values'
    theta = theta.reshape((6,))
    theta_hat = np.array([[         0,  -theta[5],  theta[4], theta[0]],
                          [  theta[5],          0, -theta[3], theta[1]],
                          [ -theta[4],   theta[3],         0, theta[2]],
                          [         0,          0,         0,       0]])

    return theta_hat

def hatmap_curly(theta):
    assert theta.shape[0] == 6, 'invalid vector theta, must have exactly six values'
    theta = theta.reshape((6,))
    theta_curlyhat = np.array([[         0,  -theta[5],  theta[4],         0, -theta[2],  theta[1]],
                               [  theta[5],          0, -theta[3],  theta[2],         0, -theta[0]],
                               [ -theta[4],   theta[3],         0, -theta[1],  theta[0],         0],
                               [         0,          0,         0,         0, -theta[5],  theta[4]],
                               [         0,          0,         0,  theta[5],         0, -theta[3]],
                               [         0,          0,         0, -theta[4],  theta[3],         0]])
    return theta_curlyhat

def sample_noise(mean, cov):
    epsilon = np.random.multivariate_normal(mean, cov).T
    return epsilon

def sample_pose(mean, cov, pose_mu):
    epsilon = sample_noise(mean, cov)
    pose = np.matmul(expm(hatmap_se3(epsilon)), pose_mu)
    return pose

# SAMPLE dimensions
# t_vec                   (1, 1106)
# linear_velocity_vec     (3, 1106)
# rotational_velocity_vec (3, 1106)
def doOnlyLocalization(t_vec, linear_velocity_vec, rotational_velocity_vec, timesteps):
    mean = np.zeros(6)
    cov = INITIAL_POSE_COVARIANCE_FACTOR * np.identity(6)  # diagonal covariance
    pose_mu_t_t = np.identity(4)
    pose_cov_t_t = cov
    pose0_0 = sample_pose(mean, cov, pose_mu_t_t)

    pose_t_t = pose0_0

    assert pose0_0.shape == (4, 4), \
        'pose0_0 has invalid shape of {}'.format(pose0_0.shape)

    print('pose0_0:', pose0_0)

    N_pose = t_vec.shape[1]
    t_vec = t_vec.reshape((N_pose,))

    # IMU's pose in world frame over time, 4x4xN_pose matrix
    # N_pose is the number of pose
    world_T_imu = np.zeros((4, 4, N_pose))

    pose_mu_matrix = np.zeros((4, 4, N_pose))

    prev_timestamp = t_vec[0]
    for idx, curr_timestamp in enumerate(t_vec[:timesteps]):
        # 1. PREDICTION step
        tau = curr_timestamp - prev_timestamp
        prev_timestamp = curr_timestamp
        assert tau < 1, 'tau seems too large, tau={}'.format(tau)
        assert tau >= 0, 'tau cannot be negative'

        linear_velocity_v_t = linear_velocity_vec[:,idx]
        rotational_velocity_omega_t = rotational_velocity_vec[:,idx]
        # rotational_velocity_omega_t = np.array([0,0,0]).reshape((3,))
        control_u_t = np.concatenate((linear_velocity_v_t, rotational_velocity_omega_t), axis=0)
        
        assert linear_velocity_v_t.shape         == (3,), \
            'linear velocity has invalid shape of {}'.format(linear_velocity_v_t.shape)
        assert rotational_velocity_omega_t.shape == (3,), \
            'rotational velocity has invalid shape of {}'.format(rotational_velocity_omega_t.shape)
        assert control_u_t.shape                 == (6,), \
            'control input has invalid shape of {}'.format(control_u_t.shape)

        # pose mean and covariance matrix
        pose_mu_tplus1_t = np.matmul(expm(-tau*hatmap_se3(control_u_t)), pose_mu_t_t)
        assert pose_mu_tplus1_t.shape == (4, 4), \
            'pose_mu_tplus1_t has invalid shape of {}'.format(pose_mu_tplus1_t.shape)
        
        control_noise_cov_t = MOTION_NOISE_COVARIANCE_FACTOR * np.identity(6)
        assert control_noise_cov_t.shape == (6, 6), \
            'control_noise_cov_t has invalid shape of {}'.format(control_noise_cov_t.shape)

        control_u_t_curlymap = hatmap_curly(control_u_t)
        expmap = expm(-tau*control_u_t_curlymap)
        pose_cov_tplus1_t = np.matmul(expmap, np.matmul(pose_cov_t_t, expmap.T)) + control_noise_cov_t
        assert pose_cov_tplus1_t.shape == (6, 6), \
            'pose_cov_tplus1_t has invalid shape of {}'.format(pose_cov_tplus1_t.shape)

        # append pose to world_T_imu matrix for visualization
        mean = np.zeros(6)
        pose_tplus1_t = sample_pose(mean, pose_cov_tplus1_t, pose_mu_tplus1_t)

        w_t = np.random.multivariate_normal(np.zeros(6), control_noise_cov_t).T
        pose_tplus1_t = np.matmul(expm(-tau*hatmap_se3(control_u_t+w_t)), pose_t_t)

        pose_t_t = pose_tplus1_t

        # print('pose_tplus1_t:', pose_tplus1_t)
        # print('(x, y) is ({}, {})'.format(pose_tplus1_t[0,3], pose_tplus1_t[1,3]))
        # print('\n')

        world_T_imu[:,:,idx] = np.linalg.inv(pose_tplus1_t)
        pose_mu_matrix[:,:,idx] = pose_mu_tplus1_t

        # 2. UPDATE step
        pose_mu_t_t  = pose_mu_tplus1_t  # TODO: correct using EKF
        pose_cov_t_t = pose_cov_tplus1_t # TODO: correct using EKF
    return world_T_imu, pose_mu_matrix

if __name__ == '__main__':
    datasets = ['0022', '0027', '0034']
    dataset_idx = 2
    dataset_name = datasets[dataset_idx]
    print('Working with dataset', dataset_name)
    print('============================')
    filename = "./data/{}.npz".format(dataset_name)
    t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

    # 0022 => 800 steps
    # 0027 => 1106 steps
    # 0034 => 1224 steps
    print(t.shape)
    # print(features.shape)
    print(linear_velocity.shape)
    print(rotational_velocity.shape)
    # print(K, b)
    # print(cam_T_imu)
    timesteps=t.shape[1]
    world_T_imu, pose_mu_matrix = doOnlyLocalization(t, linear_velocity, rotational_velocity, timesteps=timesteps)

    assert pose_mu_matrix.shape == (4, 4, t.shape[1]), \
            'pose_mu_matrix has invalid shape of {}'.format(pose_mu_matrix.shape)

    filename = "pose_mu_matrix_{}.npy".format(dataset_name)
    np.save(filename, pose_mu_matrix)
    print('saved pose_mu_matrix to file {}'.format(filename))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    pose = world_T_imu[:,:,:timesteps]
    n_pose = pose.shape[2]
    ax.plot(-pose[1,3,:],pose[0,3,:], 'r-',label='{} path'.format(dataset_name))
    ax.scatter(-pose[1,3,0],pose[0,3,0], marker='s',label="start")
    ax.scatter(-pose[1,3,-1],pose[0,3,-1], marker='o',label="end")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.show(block=True)
    # visualize_trajectory_2d(world_T_imu[:,:,:timesteps], show_ori=True)
