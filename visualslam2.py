import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

from utils import load_data, visualize_trajectory_2d

# Tunable parameters
INITIAL_POSE_COVARIANCE_FACTOR = 0
MOTION_NOISE_COVARIANCE_FACTOR = 0
MEASUREMENT_NOISE_COVARIANCE_FACTOR = 2

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


# return indices of elements that are not [-1, -1, -1, -1]
def get_observed_feature_indices(z_t, debug=False):
    M = z_t.shape[1]
    not_observable = np.array([-1., -1., -1., -1.]).reshape((4,))
    indices = []
    for idx in np.arange(M):
        observation = z_t[:,idx].reshape((4,))
        is_observed = not np.array_equal(observation, not_observable)
        if is_observed:
            indices.append(idx)
    if debug: 
        print('observed {} out of {}'.format(len(indices), M))
    return indices

def get_prq_matrices(z_t_new_features, M_stereo, opt_T_world):
    count = z_t_new_features.shape[1]
    u_L = z_t_new_features[0,:]
    v_L = z_t_new_features[1,:]
    u_R = z_t_new_features[2,:]
    disparity = np.subtract(u_L, u_R)
    assert disparity.shape == (count,), \
        'disparity has invalid shape of {}'.format(disparity.shape)

    a14 = opt_T_world[0,3]
    a24 = opt_T_world[1,3]
    a34 = opt_T_world[2,3]
    fs_u = M_stereo[0,0]
    c_u = M_stereo[0,2]
    fs_v = M_stereo[1,1]
    c_v = M_stereo[1,2]
    r = -M_stereo[2,3]/disparity
    p = np.divide(np.multiply((u_L - c_u), r), fs_u)
    p = np.subtract(p, a14)
    q = np.divide(np.multiply((v_L - c_v), r), fs_v)
    q = np.subtract(q, a24)
    r =  np.subtract(r, a34) # DON'T move this line above the calculation of p and q
    return p, q, r

def dpi_by_dq(q):
    assert q.shape == (4,), 'q has invalid shape of {}'.format(q.shape)
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    mat = np.array([[1, 0, -q1/q3, 0],
                    [0, 1, -q2/q3, 0],
                    [0, 0,      0, 0],
                    [0, 0, -q4/q3, 1]])
    mat = np.divide(mat, q3)
    return mat

def do_pi(q):
    return np.divide(q, q[2])

# SAMPLE dimensions
# t_vec                   (1, 1106)
# pose_mu_matrix          (4, 4, 1106)
# features                (4, 395, 1106), ie dealing with 395 features
def doOnlyMapping(t_vec, pose_mu_matrix, features, opt_T_imu, M_stereo, timesteps, update=True):
    print('\nrunning doOnlyMapping for {} timesteps'.format(timesteps))
    print('=======================================================')
    M = features.shape[1]
    assert M > 0 and M < 5000, 'M is out or range (0, 5000), is {}'.format(M)

    # mean of landmark's world coordinates x, y, z
    landmark_mu_t = np.zeros((3*M,)) # in meters
    landmark_cov_t = np.eye(3*M)

    # 1 means observed, 0 means not yet
    observed_map = np.zeros((M, ))

    N_pose = t_vec.shape[1]
    t_vec = t_vec.reshape((N_pose,))

    # used in calculating H_t
    P_T = np.concatenate((np.eye(3), np.zeros((3,1))), axis=1).T

    for idx, curr_timestamp in enumerate(t_vec[:timesteps]):
        if idx % 50 == 0:
            print('running update step number {}'.format(idx+1))
        # 1. UPDATE step
        #   a. if landmark/feature was never seen before, 
        #      find its world position and use it
        #      to update the landmark_mu_t matrix

        z_t = features[:,:,idx]
        assert z_t.shape == (4, M), 'z_t has invalid shape of {}'.format(z_t.shape)

        # find indices of features we've never observed before
        observed_feature_indices = get_observed_feature_indices(z_t)
        observed_map[observed_feature_indices[0]] = 1
        all_unobserved_indices = np.where(observed_map == 0)[0]  # gives the indexes in observed_map for which value = 0
        never_observed_indices = \
            np.intersect1d(all_unobserved_indices, observed_feature_indices)

        # find the world coordinates of these never observed features
        # to update landmark_mu_t matrix
        z_t_new_features = z_t[:,never_observed_indices]
        U_t = pose_mu_matrix[:,:,idx]
        opt_T_world = np.matmul(opt_T_imu, U_t)
        p, q, r = get_prq_matrices(z_t_new_features, M_stereo, opt_T_world)
        A = opt_T_world[0:3,0:3]

        for x in np.arange(len(never_observed_indices)):
            index = never_observed_indices[x]
            b = np.array([p[x], q[x], r[x]]).reshape((3,))
            try:
                soln = np.linalg.solve(A, b)
            except np.linalg.LinAlgError as e:
                print('got np.linalg.LinAlgError, because A is singular')
                print('A', A)
                print('b', b)
                raise e
            landmark_mu_t[3*index:3*index+3] = soln

        # mark current features as observed
        observed_map[never_observed_indices] = 1

        if not update:
            continue

        #   b. find H_t, ie the jacobian of observation model (no noise)
        #      w.r.t. the world positions of the landmarks
        N_t = len(observed_feature_indices) # no. of landmarks observed now
        H_t = np.zeros((4*N_t, 3*M))

        term3 = np.matmul(opt_T_world, P_T)
        assert term3.shape == (4,3), 'term3 has invalid shape of {}'.format(term3.shape)

        landmark_mu_t_observed = np.zeros((4, N_t))
        
        I_kronecker_V = np.zeros((4*N_t, 4*N_t))
        V = MEASUREMENT_NOISE_COVARIANCE_FACTOR * np.identity(4)
        for x in np.arange(N_t):
            index = observed_feature_indices[x]

            landmark_mu_t_j = landmark_mu_t[3*index:3*index+3]
            landmark_mu_t_j_under = np.concatenate((landmark_mu_t_j, np.array([1])), axis=0)
            landmark_mu_t_observed[:,x] = landmark_mu_t_j_under.reshape(4,)
            assert landmark_mu_t_j_under.shape == (4,), \
                'landmark_mu_t_j_under has invalid shape of {}'.format(landmark_mu_t_j_under.shape)
            q = np.matmul(opt_T_world, landmark_mu_t_j_under)
            H_t_i_j = np.matmul(M_stereo, np.matmul(dpi_by_dq(q), term3))
            H_t[4*x:4*x+4, 3*index:3*index+3] = H_t_i_j

            I_kronecker_V[4*x:4*x+4, 4*x:4*x+4] = V
        # kalman gain
        bracket_term = np.add(np.matmul(H_t, np.matmul(landmark_cov_t, H_t.T)), I_kronecker_V)
        K_t = np.matmul(landmark_cov_t, np.matmul(H_t.T, np.linalg.inv(bracket_term)))

        z_t_observed_flat = z_t[:,observed_feature_indices].reshape((-1,), order='F')
        assert z_t_observed_flat.shape == (4*N_t,), \
            'z_t_observed_flat has invalid shape of {}'.format(z_t_observed_flat.shape)
        assert z_t_observed_flat[1] == z_t[:,observed_feature_indices][1,0], \
            'z_t_observed_flat is not properly ordered'
        q = np.matmul(opt_T_world, landmark_mu_t_observed) # shape is 4xN_t
        z_t_observed_hat = np.matmul(M_stereo, np.divide(q, q[2,:].reshape((1,N_t))))
        z_t_observed_hat_flat = z_t_observed_hat.reshape((-1,), order='F')
        innovation_term = z_t_observed_flat - z_t_observed_hat_flat
        # landmark_mu_tplus1
        landmark_mu_tplus1 = landmark_mu_t + np.matmul(K_t, innovation_term)
        # landmark_cov_tplus1
        landmark_cov_tplus1 = np.matmul(np.subtract(np.eye(3*M), np.matmul(K_t, H_t)), landmark_cov_t)

        landmark_mu_t = landmark_mu_tplus1
        landmark_cov_t = landmark_cov_tplus1

    if not update:
        print('NOTE: no EKF updating was performed!!!')
    return landmark_mu_t.reshape((3, M), order='F')

# how_many is out of 10
def pick_landmarks(how_many, features, seed=74):
    assert features.shape[0] == 4, \
        'each landmark observed should have 4 pixel coordinate values'
    features_count = features.shape[1]

    picked_indices = []

    # seed the random number generator so it uses the same values everytime
    np.random.seed(seed)
    for x in np.arange(0, features_count, 10):
        left = features_count - x
        high = 10 if left >= 10 else left
        indices = x + np.random.choice(high, how_many, replace=False)
        picked_indices.extend(indices.tolist())
    picked_indices.sort()
    return features[:,picked_indices,:]


def test_pick_landmarks():
    how_many = 2
    features = np.arange(4*15*2).reshape((4,15,2))    
    picked = pick_landmarks(how_many, features, seed=74)
    # [0, 5, 11, 14]
    is_working = np.array_equal(features[:,0,:], picked[:,0,:]) and \
                np.array_equal(features[:,5,:], picked[:,1,:]) and \
                np.array_equal(features[:,11,:], picked[:,2,:]) and \
                np.array_equal(features[:,14,:], picked[:,3,:])
    return is_working

if __name__ == '__main__':
    # 0022 => 800 steps, 3220 landmarks
    # 0027 => 1106 steps, 3950 landmarks
    # 0034 => 1224 steps, 4815 landmarks
    datasets = ['0022', '0027', '0034']
    dataset_idx = 2
    dataset_name = datasets[dataset_idx]

    print('Working with dataset', dataset_name)
    print('============================')

    filename = "./data/{}.npz".format(dataset_name)
    t, features, linear_velocity, rotational_velocity, \
        K, b, opt_T_imu = load_data(filename)

    filename = "pose_mu_matrix_{}.npy".format(dataset_name)
    print('loaded pose_mu_matrix from file {}'.format(filename))
    pose_mu_matrix = np.load(filename)

    assert pose_mu_matrix.shape == (4, 4, t.shape[1]), \
            'pose_mu_matrix has invalid shape of {}'.format(pose_mu_matrix.shape)

    
    assert test_pick_landmarks() == True, 'pick_landmarks didnt behave as expected'


    print(t.shape)
    print(features.shape)
    print(features[:,0,0])
    # print(linear_velocity.shape)
    # print(rotational_velocity.shape)
    print(K, b)
    print(opt_T_imu)


    how_many = 1 # 1 every 10 features
    picked_features = pick_landmarks(how_many, features)
    M = picked_features.shape[1] # total number of features/landmarks
    print('picked_features.shape', picked_features.shape)

    fs_u = K[0,0]
    fs_v = K[1,1]
    c_u  = K[0,2]
    c_v  = K[1,2]
    M_stereo = np.array([[fs_u,    0, c_u,       0],
                         [   0, fs_v, c_v,       0],
                         [fs_u,    0, c_u, -fs_u*b],
                         [   0, fs_v, c_v,      0]])

   
    timesteps=t.shape[1]
    # timesteps = 300
    landmark_position_world = doOnlyMapping(t, pose_mu_matrix, picked_features, \
                                            opt_T_imu, M_stereo, timesteps=timesteps, update=True)

    assert landmark_position_world.shape == (3, M), \
            'landmark_position_world has invalid shape of {}'\
            .format(landmark_position_world.shape)

    world_T_imu = np.zeros((4, 4, timesteps))
    for idx in np.arange(timesteps):
        U_t = pose_mu_matrix[:,:,idx]
        world_T_imu[:,:,idx] = np.linalg.inv(U_t)
    all_landmarks = []
    for idx in np.arange(M):
        x = landmark_position_world[0, idx]
        y = landmark_position_world[1, idx]
        all_landmarks.append((-y, x))
    # print(all_landmarks)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(*zip(*all_landmarks),label='{} landmarks'.format(dataset_name))
    ax.plot(-world_T_imu[1,3,:],world_T_imu[0,3,:], 'r-',label='{} path'.format(dataset_name))
    ax.scatter(-world_T_imu[1,3,0],world_T_imu[0,3,0], marker='s',label="start")
    ax.scatter(-world_T_imu[1,3,-1],world_T_imu[0,3,-1], marker='o',label="end")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.show(block=True)

    # visualize_trajectory_2d(world_T_imu[:,:,:timesteps], show_ori=True)
