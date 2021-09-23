#!/usr/bin/env python
# coding: utf-8

from mce_irl_pomdps import parser_pomdp
from mce_irl_pomdps import irl_pomdp_solver as irl_solver
import numpy as np
import stormpy

import pickle

# For reproducibility
np.random.seed(201)

datafile = "test_phoenix_data.pkl"

# Load the data file for robot trajectory
mFile = open(datafile, 'rb')
mData = pickle.load(mFile)

# Get the robot trajectory
robot_obs_traj = mData['robot_obs']
state_dict = mData['state_dict']
obs_dict = mData['obs_dict']

prism_file = mData['prism_file']

# Load the pomdp model
pomdp_r_1 = parser_pomdp.PrismModel(prism_file, counter_type=stormpy.pomdp.PomdpMemoryPattern.selective_counter, memory_len=1, export=True)
# pomdp_r_4 = parser_pomdp.PrismModel(prism_file, counter_type=stormpy.pomdp.PomdpMemoryPattern.fixed_counter, memory_len=4, export=False)

print(pomdp_r_1.pomdp)
# print(pomdp_r_4.pomdp)

# Set the parameter for the trust region
irl_solver.trustRegion = {'red' : lambda x : ((x - 1) / 1.5 + 1),
                          'aug' : lambda x : min(1.5,(x-1)*1.25+1),
                          'lim' : 1+1e-3}

# Options for the solver
options_opt = irl_solver.OptOptions(mu=1e3, mu_spec=1, mu_rew=1.0, maxiter=100, maxiter_weight=100,
                                    graph_epsilon=1e-6, discount=0.999, verbose=True, verbose_solver=False)

# True reward in the POMDP environment
weight = {'goal' : 100, 'road' : 1, 'gravel' : 1, 'grass' : 0.1, 'time' : 1.1}

# Build the solver for different memory size
irlPb_1 = irl_solver.IRLSolver(pomdp_r_1, init_trust_region=1.01, max_trust_region=1.5, options=options_opt)


# Get the optimal policy for memory size 1 and save such policy and the associated performances
pol_val_grb_1 = irlPb_1.from_reward_to_policy_via_scp(weight, save_info=(20, 'rock_mem1_fwd', weight))
pol_val_mdp = irlPb_1.from_reward_to_optimal_policy_mdp_lp(weight, gamma=options_opt.discount, save_info=(-1,'rock_mdp_fwd', weight))

obs_based = False
traj_mdp_5, rew = pomdp_r_1.simulate_policy(pol_val_mdp, weight, 10, 100, obs_based=obs_based, stop_at_accepting_state=True)
print(traj_mdp_5)
print(rew[0])