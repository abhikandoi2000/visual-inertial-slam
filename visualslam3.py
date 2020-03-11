import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

from utils import load_data, visualize_trajectory_2d

# Tunable parameters
HOW_MANY = 5
OUT_OF = 50
DATASET_IDX = 2 # 0, 1, 2


LANDMARK_POSITION_COVARIANCE_FACTOR = 5
POSE_COVARIANCE_FACTOR = 0.1
MEASUREMENT_NOISE_COVARIANCE_FACTOR = 4
LINEAR_MOTION_NOISE_COVARIANCE_FACTOR = 0.002
ANGULAR_MOTION_NOISE_COVARIANCE_FACTOR = 0.001


def hatmap_so3(theta):
    assert theta.shape[0] == 3, 'invalid vector theta, must have exactly three values'
    theta = theta.reshape((3,))
    theta_hat = np.array([[         0,  -theta[2],  theta[1]],
                          [  theta[2],          0, -theta[0]],
                          [ -theta[1] ,   theta[0],        0]])
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

def weird_dot(k):
    assert k.shape == (4,), 'k has invalid shape of {}'.format(k.shape)
    mat = np.zeros((4, 6))
    mat[:3,:3] = np.eye(3)
    mat[:3,3:6] = hatmap_so3(-k[:3].reshape(3,))
    # print('weird_dot', mat)
    return mat

# SAMPLE dimensions
# t_vec                   (1, 1106)
# linear_velocity_vec     (3, 1106)
# rotational_velocity_vec (3, 1106)
# features                (4, 395, 1106), ie dealing with 395 features
def doSLAM(t_vec, linear_velocity_vec, rotational_velocity_vec, \
            features, opt_T_imu, M_stereo, \
            timesteps, update=True):
    print('\nrunning doSLAM for {} timesteps'.format(timesteps))
    print('=======================================================')
    M = features.shape[1]
    assert M > 0 and M < 5000, 'M is out or range (0, 5000), is {}'.format(M)

    # mean of pose, ie pose of world wrt imu
    pose_mu_t_t = np.identity(4)
    # mean of landmark's world coordinates x, y, z
    landmark_mu_t = np.zeros((3*M,)) # in meters
    # joint covariance
    joint_cov_t = np.zeros((3*M+6, 3*M+6))
    joint_cov_t[3*M:3*M+6,3*M:3*M+6] = POSE_COVARIANCE_FACTOR * np.eye(6)

    control_noise_cov = np.zeros((6, 6))
    control_noise_cov[:3,:3] = LINEAR_MOTION_NOISE_COVARIANCE_FACTOR * np.eye(3)
    control_noise_cov[3:,3:] = ANGULAR_MOTION_NOISE_COVARIANCE_FACTOR * np.eye(3)
    assert control_noise_cov.shape == (6, 6), \
        'control_noise_cov has invalid shape of {}'.format(control_noise_cov.shape)

    # 1 means observed, 0 means not yet
    observed_map = np.zeros((M, ))

    N_pose = t_vec.shape[1]
    t_vec = t_vec.reshape((N_pose,))

    # IMU's pose in world frame over time, 4x4xN_pose matrix
    # N_pose is the number of pose
    world_T_imu = np.zeros((4, 4, N_pose))

    # used in calculating H_tplus1_t
    P_T = np.concatenate((np.eye(3), np.zeros((3,1))), axis=1).T

    prev_timestamp = t_vec[0]
    for idx, curr_timestamp in enumerate(t_vec[:timesteps]):
        if idx % 50 == 0:
            print('running loop for step number {}'.format(idx+1))
        # 1. PREDICTION step
        #    use code from part a to predict the SE(3) pose
        #    using the motion model
        tau = curr_timestamp - prev_timestamp
        prev_timestamp = curr_timestamp
        assert tau < 1, 'tau seems too large, tau={}'.format(tau)
        assert tau >= 0, 'tau cannot be negative'

        linear_velocity_v_t = linear_velocity_vec[:,idx]
        rotational_velocity_omega_t = rotational_velocity_vec[:,idx]
        control_u_t = np.concatenate((linear_velocity_v_t, rotational_velocity_omega_t), axis=0)

        assert linear_velocity_v_t.shape         == (3,), \
            'linear velocity has invalid shape of {}'.format(linear_velocity_v_t.shape)
        assert rotational_velocity_omega_t.shape == (3,), \
            'rotational velocity has invalid shape of {}'.format(rotational_velocity_omega_t.shape)
        assert control_u_t.shape                 == (6,), \
            'control input has invalid shape of {}'.format(control_u_t.shape)

        # predicted pose mean
        pose_mu_tplus1_t = np.matmul(expm(-tau*hatmap_se3(control_u_t)), pose_mu_t_t)
        assert pose_mu_tplus1_t.shape == (4, 4), \
            'pose_mu_tplus1_t has invalid shape of {}'.format(pose_mu_tplus1_t.shape)

        control_u_t_curlymap = hatmap_curly(control_u_t)
        expmap = expm(-tau*control_u_t_curlymap)
        pose_cov_t_t = joint_cov_t[3*M:3*M+6,3*M:3*M+6]
        assert pose_cov_t_t.shape == (6, 6), \
            'pose_cov_t_t has invalid shape of {}'.format(pose_cov_t_t.shape)
        pose_cov_tplus1_t = np.matmul(expmap, np.matmul(pose_cov_t_t, expmap.T)) + control_noise_cov
        assert pose_cov_tplus1_t.shape == (6, 6), \
            'pose_cov_tplus1_t has invalid shape of {}'.format(pose_cov_tplus1_t.shape)

        # put back in joint_cov_t
        joint_cov_t[3*M:3*M+6,3*M:3*M+6] = pose_cov_tplus1_t

        # 2. UPDATE step
        #   a. if landmark/feature was never seen before, 
        #      find its world position and use it
        #      to update the landmark_mu_t matrix

        z_t = features[:,:,idx]
        assert z_t.shape == (4, M), 'z_t has invalid shape of {}'.format(z_t.shape)

        # find indices of features we've never observed before
        observed_feature_indices = get_observed_feature_indices(z_t)
        all_unobserved_indices = np.where(observed_map == 0)[0]  # gives the indexes in observed_map for which value = 0
        never_observed_indices = \
            np.intersect1d(all_unobserved_indices, observed_feature_indices)

        # find the world coordinates of these never observed features
        # to update landmark_mu_t matrix
        z_t_new_features = z_t[:,never_observed_indices]
        opt_T_world = np.matmul(opt_T_imu, pose_mu_tplus1_t)
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
            assert np.array_equal(landmark_mu_t[3*index:3*index+3], np.zeros((3,))), \
                'landmark_mu_t[3*index:3*index+3] is not all zeros by default, should be'
            landmark_mu_t[3*index:3*index+3] = soln

            assert np.array_equal(joint_cov_t[3*index:3*index+3,3*index:3*index+3], np.zeros((3, 3))), \
                'joint_cov_t[3*index:3*index+3,3*index:3*index+3] is not all zeros by default, should be'
            joint_cov_t[3*index:3*index+3,3*index:3*index+3] = \
                    LANDMARK_POSITION_COVARIANCE_FACTOR*np.eye(3)

        # mark current features as observed
        observed_map[never_observed_indices] = 1

        if not update:
            continue

        #   b. find H_tplus1_t, ie the jacobian of observation model (no noise)
        #      w.r.t. the world positions of the landmarks
        N_tplus1 = len(observed_feature_indices) # no. of landmarks observed now
        H_tplus1_t = np.zeros((4*N_tplus1, 3*M+6))

        term3 = np.matmul(opt_T_world, P_T)
        assert term3.shape == (4,3), 'term3 has invalid shape of {}'.format(term3.shape)

        landmark_mu_t_observed = np.zeros((4, N_tplus1))
        
        I_kronecker_V = np.zeros((4*N_tplus1, 4*N_tplus1))
        V = MEASUREMENT_NOISE_COVARIANCE_FACTOR * np.identity(4)
        for x in np.arange(N_tplus1):
            index = observed_feature_indices[x]

            landmark_mu_t_j = landmark_mu_t[3*index:3*index+3]
            landmark_mu_t_j_under = np.concatenate((landmark_mu_t_j, np.array([1])), axis=0)
            landmark_mu_t_observed[:,x] = landmark_mu_t_j_under.reshape(4,)
            assert landmark_mu_t_j_under.shape == (4,), \
                'landmark_mu_t_j_under has invalid shape of {}'.format(landmark_mu_t_j_under.shape)
            q = np.matmul(opt_T_world, landmark_mu_t_j_under)
            H_t_i_j = np.matmul(M_stereo, np.matmul(dpi_by_dq(q), term3))
            H_tplus1_t[4*x:4*x+4, 3*index:3*index+3] = H_t_i_j

            weird_dot_val = weird_dot(np.matmul(pose_mu_tplus1_t, landmark_mu_t_j_under))
            assert weird_dot_val.shape == (4, 6), \
                'weird_dot_val has invalid shape of {}'.format(weird_dot_val.shape)
            H_i_tplus1_t = np.matmul(M_stereo, np.matmul(dpi_by_dq(q), np.matmul(opt_T_imu, weird_dot_val)))
            assert H_i_tplus1_t.shape == (4,6), \
                'H_i_tplus1_t has invalid shape of {}'.format(H_i_tplus1_t.shape)

            H_tplus1_t[4*x:4*x+4, 3*M:3*M+6] = H_i_tplus1_t

            I_kronecker_V[4*x:4*x+4, 4*x:4*x+4] = V
        # joint kalman gain
        bracket_term = np.add(np.matmul(H_tplus1_t, np.matmul(joint_cov_t, H_tplus1_t.T)), I_kronecker_V)
        assert bracket_term.shape == (4*N_tplus1, 4*N_tplus1), \
            'bracket_term has invalid shape of {}'.format(bracket_term.shape)
        try:
            bracket_term_inv = np.linalg.inv(bracket_term)
        except np.linalg.LinAlgError as e:
            print('bracket_term matrix has rank of {}'.format(np.linalg.matrix_rank(bracket_term)))
            print('H_tplus1_t matrix has max element {} and min element {}'\
                    .format(np.max(H_tplus1_t), np.min(H_tplus1_t)))
            print('H_tplus1_t matrix has max element at index {}'\
                    .format(np.unravel_index(H_tplus1_t.argmax(), H_tplus1_t.shape)))
            print('joint_cov_t matrix has max element {} and min element {}'\
                    .format(np.max(joint_cov_t), np.min(joint_cov_t)))
            print('bracket_term matrix has max element {} and min element {}'\
                    .format(np.max(bracket_term), np.min(bracket_term)))
            raise e
        # finally:
            # bracket_term_max = np.max(bracket_term)
            # bracket_term_inv = 
        K_tplus1_t = np.matmul(joint_cov_t, np.matmul(H_tplus1_t.T, bracket_term_inv))

        assert K_tplus1_t.shape == (3*M+6, 4*N_tplus1), \
            'K_tplus1_t has invalid shape of {}'.format(K_tplus1_t.shape)

        z_t_observed_flat = z_t[:,observed_feature_indices].reshape((-1,), order='F')
        assert z_t_observed_flat.shape == (4*N_tplus1,), \
            'z_t_observed_flat has invalid shape of {}'.format(z_t_observed_flat.shape)
        assert z_t_observed_flat[1] == z_t[:,observed_feature_indices][1,0], \
            'z_t_observed_flat is not properly ordered'
        q = np.matmul(opt_T_world, landmark_mu_t_observed) # shape is 4xN_t
        z_t_observed_hat = np.matmul(M_stereo, np.divide(q, q[2,:].reshape((1, N_tplus1))))
        z_t_observed_hat_flat = z_t_observed_hat.reshape((-1,), order='F')
        innovation_term = z_t_observed_flat - z_t_observed_hat_flat
        assert innovation_term.shape == (4*N_tplus1,), \
            'innovation_term has invalid shape of {}'.format(innovation_term.shape)
        kalman_innovation = np.matmul(K_tplus1_t, innovation_term)
        assert kalman_innovation.shape == (3*M+6,), \
            'kalman_innovation has invalid shape of {}'.format(kalman_innovation.shape)

        pose_mu_tplus1_tplus1 = np.matmul(expm(hatmap_se3(kalman_innovation[3*M:3*M+6])), pose_mu_tplus1_t)
        landmark_mu_tplus1_tplus1 = landmark_mu_t + kalman_innovation[:3*M]
        joint_cov_tplus1_tplus1 = np.matmul(np.subtract(np.eye(3*M+6), np.matmul(K_tplus1_t, H_tplus1_t)), joint_cov_t)
        
        world_T_imu[:,:,idx] = np.linalg.inv(pose_mu_tplus1_tplus1)

        # update these for next timestep
        pose_mu_t_t = pose_mu_tplus1_tplus1
        landmark_mu_t = landmark_mu_tplus1_tplus1
        joint_cov_t = joint_cov_tplus1_tplus1

    if not update:
        print('NOTE: no EKF updating was performed!!!')
    return landmark_mu_t.reshape((3, M), order='F'), world_T_imu

# how_many to select per out_of
def pick_landmarks(how_many, out_of, features, seed=74):
    assert features.shape[0] == 4, \
        'each landmark observed should have 4 pixel coordinate values'
    features_count = features.shape[1]

    picked_indices = []

    # seed the random number generator so it uses the same values everytime
    np.random.seed(seed)
    for x in np.arange(0, features_count, out_of):
        left = features_count - x
        high = out_of if left >= out_of else left
        indices = x + np.random.choice(high, how_many, replace=False)
        picked_indices.extend(indices.tolist())
    picked_indices.sort()
    return features[:,picked_indices,:]


def test_pick_landmarks():
    how_many = 2
    features = np.arange(4*15*2).reshape((4,15,2))    
    picked = pick_landmarks(how_many, 10, features, seed=74)
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
    dataset_idx = DATASET_IDX
    dataset_name = datasets[dataset_idx]

    print('Working with dataset', dataset_name)
    print('============================')

    filename = "./data/{}.npz".format(dataset_name)
    t, features, linear_velocity, rotational_velocity, \
        K, b, opt_T_imu = load_data(filename)

    assert test_pick_landmarks() == True, 'pick_landmarks didnt behave as expected'

    print(t.shape)
    print(features.shape)
    print(features[:,0,0])
    print(linear_velocity.shape)
    print(rotational_velocity.shape)
    print(K, b)
    print(opt_T_imu)

    picked_features = pick_landmarks(HOW_MANY, OUT_OF, features)
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
    landmark_position_world, world_T_imu = doSLAM(t, linear_velocity, rotational_velocity, \
                                            picked_features, opt_T_imu, M_stereo, \
                                            timesteps=timesteps, update=True)

    assert landmark_position_world.shape == (3, M), \
            'landmark_position_world has invalid shape of {}'\
            .format(landmark_position_world.shape)

    all_landmarks = []
    for idx in np.arange(M):
        x = landmark_position_world[0, idx]
        y = landmark_position_world[1, idx]
        all_landmarks.append((-y, x))

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
