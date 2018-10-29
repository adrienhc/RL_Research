#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import joblib   #used to save large amounts of data, computation efficiently and transparently dump(), restore data load()  -- interface application / disk -- also pipeline, //lized programming 
import tensorflow as tf
import numpy as np
from mpi4py import MPI
import os, errno


def callback(localv, globalv):           #used to save the network's variables -- By default Every 10 policy runs
    if localv['iters_so_far'] % 10 != 0:  #change frequency of saves with mod operator
        return
    save_dict = {}
    variables = localv['pi'].get_variables()  #get network's variables and save them in a dictonnary  name -> value
    for i in range(len(variables)):
        cur_val = variables[i].eval()
        save_dict[variables[i].name] = cur_val

    save_dir = logger.get_dir() + '/' + (str(localv['env'].env.env.assist_schedule).replace(' ', ''))  #in the following dir
    try:
        os.makedirs(save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    joblib.dump(save_dict, save_dir + '/policy_params_' + str(localv['iters_so_far']) + '.pkl', compress=True)  #we are actually saving a compressed dir name -> value  of policy parameters
    joblib.dump(save_dict, logger.get_dir() + '/policy_params' + '.pkl', compress=True)


def train_mirror(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_mirror_policy, pposgd_mirror
    U.make_session(num_cpu=2).__enter__() #originally 1 CPU    --  creates session with X CPUs
    set_global_seeds(seed)
    env = gym.make(env_id) #creates gym env -- Here it is walker3d.py
    env.env.assist_timeout = 100.0

    def policy_fn(name, ob_space, ac_space):
        return mlp_mirror_policy.MlpMirrorPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                 hid_size=64, num_hid_layers=3, gmm_comp=1,
                                                 mirror_loss=True,
                                                 observation_permutation=np.array(
                                                     [0.0001, -1, 2, -3, -4, -5, -6, 7, 14, -15, -16, 17, 18, -19, 8,
                                                      -9, -10, 11, 12, -13,
                                                      20, 21, -22, 23, -24, -25, -26, -27, 28, 35, -36, -37, 38, 39,
                                                      -40, 29, -30, -31, 32, 33,
                                                      -34, 42, 41, 43]),
                                                 action_permutation=np.array(
                                                     [-0.0001, -1, 2, 9, -10, -11, 12, 13, -14, 3, -4, -5, 6, 7, -8]))  # initializes a random MLP Mirror Policy aka the good stuff

    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    env.seed(seed + MPI.COMM_WORLD.Get_rank())
    gym.logger.setLevel(logging.WARN)

    previous_params = None  #interesting -- can i use it to use prev params as starting point?
    iter_num = 0
    last_iter = False

    # if initialize from previous runs
    '''previous_params = joblib.load('data/ppo_DartWalker3d-v111_energy04_vel1_1s_mirror4_velrew3_damping5_anklesprint100_5_rotpen1_rew01xinit_stagedcurriculum/policy_params.pkl')  
    env.env.env.assist_schedule = [[0.0,np.array([250.,125.])],[3.0,np.array([125.,62.5])],[6.0,[62.5,31.25]]]''' #sets params from a previous run -- choose starting schedule too!

    previous_params = joblib.load('data/Walker_0_to_1/[[0.0,array([0.,0.])],[3.0,array([0.,0.])],[6.0,[0.0,0.0]]]/policy_params_6400.pkl')  
    env.env.env.assist_schedule = [[0.0,np.array([2000,2000])],[3.0,np.array([1500,1500])],[6.0,[1125.0,1125.0]]]

    joblib.dump(str(env.env.env.__dict__), logger.get_dir() + '/env_specs.pkl', compress=True)

    reward_threshold = None
    while True:                             #do the actual training
        if not last_iter:
            rollout_length_threshold = env.env.env.assist_schedule[2][0] / env.env.env.dt   #if not last iter, update rew length threshold
        else:
            rollout_length_threshold = None
        opt_pi, rew = pposgd_mirror.learn(env, policy_fn,
                                          max_timesteps=num_timesteps,
                                          timesteps_per_batch=int(2500),
                                          clip_param=0.2, entcoeff=0.0,
                                          optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                                          gamma=0.99, lam=0.95, schedule='linear',
                                          callback=callback,
                                          sym_loss_weight=4.0,
                                          positive_rew_enforce=False,
                                          init_policy_params=previous_params,
                                          reward_drop_bound=500,
                                          rollout_length_thershold=rollout_length_threshold,
                                          policy_scope='pi' + str(iter_num),
                                          return_threshold=reward_threshold,
                                          )              # Use Neural Net, to get new policy and reward  -- Notice Callback function to save policy
        if iter_num == 0:
            reward_threshold = 0.7 * rew  #if new schedule, reward to attain is 70% of old reward
        if last_iter:                     #if last iter, no threshold
            reward_threshold = None
        iter_num += 1

        opt_variable = opt_pi.get_variables()   #get policy's GLOBAL variables
        previous_params = {}
        for i in range(len(opt_variable)): 
            cur_val = opt_variable[i].eval()   #curr variable = variable's VALUE
            previous_params[opt_variable[i].name] = cur_val  # Previous = dictionnary  var name -> var value
                                                             # update the assist schedule
        for s in range(len(env.env.env.assist_schedule) - 1):
            env.env.env.assist_schedule[s][1] = np.copy(env.env.env.assist_schedule[s + 1][1])
        env.env.env.assist_schedule[-1][1][0] *= 0.75
        env.env.env.assist_schedule[-1][1][1] *= 0.75
        if env.env.env.assist_schedule[-1][1][0] < 5.0:
            env.env.env.assist_schedule[-1][1][0] = 0.0
        if env.env.env.assist_schedule[-1][1][1] < 5.0:
            env.env.env.assist_schedule[-1][1][1] = 0.0
        zero_assist = True
        for s in range(len(env.env.env.assist_schedule) - 1):  #check if we are in last iteration (all 0 schedules) -- if yes set last iter to true and notify the user
            for v in env.env.env.assist_schedule[s][1]:
                if v != 0.0:
                    zero_assist = False
        print('Current Schedule: ', env.env.env.assist_schedule)
        logger.log("Current Schedule: %s"%env.env.env.assist_schedule) #added by me. keep track of schedule in log file
        if zero_assist:
            last_iter = True
            print('Entering Last Iteration!')
            logger.log("Entering Last Iteration") #added by me, helps know in logger when hit last schedule

    env.close()  #when done close envt


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    #parses through CL Args
    parser.add_argument('--env', help='environment ID', default='DartWalker3d-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    logger.reset()
    logger.configure('data/ppo_' + args.env + str(
        args.seed) + '_energy04_vel1_1s_mirror4_velrew3_ab4_anklesprint100_5_rotpen0_rew05xinit_stagedcurriculum4s75s34ratio')   #setup logger
    train_mirror(args.env, num_timesteps=int(5000 * 4 * 800), seed=args.seed)    #starts the training!  --   16 000 000 iterations


if __name__ == '__main__':
    main()
