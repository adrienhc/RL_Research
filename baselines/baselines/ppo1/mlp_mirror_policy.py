from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np

class MlpMirrorPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):                 #initializes NN calling _init(...) under specific scope (name)
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True, gmm_comp=1, mirror_loss=False, observation_permutation=[],action_permutation=[]):
        assert isinstance(ob_space, gym.spaces.Box)
        if mirror_loss:
            assert gaussian_fixed_var # assume fixed std for now

        self.pdtype = pdtype = make_pdtype(ac_space, gmm_comp)   #pd = probability distribution -- distrib of possible actions
        sequence_length = None
        self.mirror_loss = mirror_loss
        if mirror_loss:
            # construct permutation matrices
            obs_perm_mat = np.zeros((len(observation_permutation), len(observation_permutation)), dtype=np.float32)  #implements mirror loss using permutation matrices
            act_perm_mat = np.zeros((len(action_permutation), len(action_permutation)), dtype=np.float32)            #is it about // limbs learning same behavior?
            for i, perm in enumerate(observation_permutation):                                                       #to swap rows / cols of a matrix  
                obs_perm_mat[i][int(np.abs(perm))] = np.sign(perm)                                                   # PA  /  AP  -- Permutation P swaps rows / cols of A 
            for i, perm in enumerate(action_permutation):
                act_perm_mat[i][int(np.abs(perm))] = np.sign(perm)

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)    #obs gathered from the env, use the Root Means Square of obs matrix as input to NN

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)  # Standard Normal Distrib of Obs, clipped btw [-5, 5] !   
                                                                                      #Z = (X - μ)/σ where Z is the value on the standard normal distribution, 
                                                                                      #X is the value on the original distribution, 
                                                                                      #μ is the mean of the original distribution, and 
                                                                                      #σ is the standard deviation of the original distribution.
        last_out = obz #input
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]     #prediction V network (predicts value V of state)  -- value function

        last_out = obz #input
        params = []
        for i in range(num_hid_layers):
            rt, pw, pb = U.dense_wparams(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0))  #policy function NN -- policy itself, what action given obervations? (aka state)
            last_out = tf.nn.tanh(rt)
            params.append([pw, pb])
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):   #usually True
            if gmm_comp == 1:                                              #usually True
                mean, pw, pb = U.dense_wparams(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))  #final is the Gaussian distrib of all possible actions
                params.append([pw, pb])
                self.mean = mean
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())  #log of std dev of actions distrib
                pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)                                                        # probability distrib of actions! in given distribution
            else:
                means = U.dense(last_out, (pdtype.param_shape()[0] - gmm_comp)//2, "polfinal", U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd",
                                         initializer=tf.constant(np.ones((1, (pdtype.param_shape()[0] - gmm_comp)//2), dtype=np.float32)*(-1.0)))
                weights = tf.nn.softmax(U.dense(last_out, gmm_comp, "gmmweights", U.normc_initializer(0.01)))
                pdparam = U.concatenate([means, means*0.0+logstd, weights], axis=1)
        elif gmm_comp == 1:
            pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))
        else:
            meanstd = U.dense(last_out, pdtype.param_shape()[0]-gmm_comp, "polfinal", U.normc_initializer(0.01))
            weights = tf.nn.softmax(U.dense(last_out, gmm_comp, "gmmweights", U.normc_initializer(0.01)))
            pdparam = U.concatenate([meanstd, weights], axis=1)

        if mirror_loss:
            mirrored_obz = tf.matmul(obz, obs_perm_mat)    #for mirrorred loss, input is permutated Observations!  
            last_val = mirrored_obz                         #as said in paper -- "encourage gait symmetry by measuring the symmetry of ACTIONS (instead os states) thus avoiding issue of delayed reward"
            for i in range(len(params)-1):
                last_val = tf.nn.tanh(tf.matmul(last_val, params[i][0])+params[i][1])  #wanna minimize error of both Action and Mirrored Action 
            mean_mir_obs = tf.matmul(last_val, params[-1][0])+params[-1][1]                             #aka action in State and Mirrored State
            self.mirrored_mean = tf.matmul(mean_mir_obs, act_perm_mat)

        if gmm_comp == 1:  #usually True
            self.pd = pdtype.pdfromflat(pdparam)   #flattens probability distrib params
        else:
            self.pd = pdtype.pdfromflat([pdparam, gmm_comp])

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())  #take action == sample from Diag Gaussian probalility distribution of actions 
        self._act = U.function([stochastic, ob], [ac, self.vpred])  #wrapper function that when given stochastic and obs  ->  returns action it sampled and predicted value of state based on obs



    def act(self, stochastic, ob, mirror_act = False):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

