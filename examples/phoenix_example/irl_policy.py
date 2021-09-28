#!/usr/bin/env python
# coding: utf-8

from mce_irl_pomdps import parser_pomdp
from mce_irl_pomdps import irl_pomdp_solver as irl_solver
import numpy as np
import stormpy

import pickle


def obs_action_from_traj(pomdp_instance, m_trajs, state_dict):
    """ Find the list of observation action in term of the pomdp instance created by prism
    """
    # Save the result of the conversion
    m_rest = list()
    # Iterate through all trajectories in the agent trajectory
    for traj in m_trajs:
        # Save a temporary list containing the (obs, act) in 
        m_temp_list = list()
        for (state, act) in traj:
            state_id = state_dict[state]
            res_obs = None
            res_act = None
            for state_val, string_val in pomdp_instance.state_string.items():
                _state_val = int(string_val.split('\t')[0].split('=')[1])
                if state_id == _state_val:
                    (res_obs,_), = pomdp_instance._obs_state_distr[state_val].items()
                    for _act, _act_label in pomdp_instance.action_string[state_val].items():
                        assert len(_act_label) == 1, 'Label length should be 1'
                        if act in _act_label:
                            res_act = _act
                            break
                    assert res_act is not None
                    break
            assert res_obs is not None and res_act is not None, 'Not found the observation and action associated to {}, {}'.format(state, act)
            m_temp_list.append((res_obs, res_act))
        m_rest.append(m_temp_list)
    return m_rest
           
# For reproducibility
np.random.seed(201)

datafile = "phoenix_scen1_r4uncert_data.pkl"

# Load the data file for robot trajectory
mFile = open(datafile, 'rb')
mData = pickle.load(mFile)

# Get the robot trajectory
robot_obs_traj = mData['robot_obs']
state_dict = mData['state_dict']
obs_dict = mData['obs_dict']

prism_file = mData['prism_file']

# Load the pomdp model
pomdp_r_1 = parser_pomdp.PrismModel(prism_file, counter_type=stormpy.pomdp.PomdpMemoryPattern.selective_counter, memory_len=1, export=False)
# pomdp_r_4 = parser_pomdp.PrismModel(prism_file, counter_type=stormpy.pomdp.PomdpMemoryPattern.fixed_counter, memory_len=4, export=False)

print(pomdp_r_1.pomdp)
# print(pomdp_r_4.pomdp)

# Set the parameter for the trust region
irl_solver.trustRegion = {'red' : lambda x : ((x - 1) / 1.5 + 1),
                          'aug' : lambda x : min(1.5,(x-1)*1.25+1),
                          'lim' : 1+1e-2}

# Options for the solver
options_opt = irl_solver.OptOptions(mu=1e3, mu_spec=1, mu_rew=1.0, maxiter=100, maxiter_weight=100,
                                    graph_epsilon=1e-6, discount=0.998, verbose=True, verbose_solver=False)

# True reward in the POMDP environment
weight = {'goal' : 5, 'road' : 1, 'gravel' : 1, 'grass' : 0.1, 'time' : 1.5}

# Build the solver for different memory size
irlPb_1 = irl_solver.IRLSolver(pomdp_r_1, init_trust_region=1.5, max_trust_region=1.5, options=options_opt)


# Get the optimal policy for memory size 1 and save such policy and the associated performances
# pol_val_grb_1 = irlPb_1.from_reward_to_policy_via_scp(weight, save_info=(20, 'phoenix_mem1_fwd', weight))
pol_val_mdp = irlPb_1.from_reward_to_optimal_policy_mdp_lp(weight, gamma=options_opt.discount, save_info=(-1,'phoenix_mdp_fwd', weight))

# obs_based = False
# traj_mdp_5, rew = pomdp_r_1.simulate_policy(pol_val_mdp, weight, 10, 100, obs_based=obs_based, stop_at_accepting_state=True)
# print(traj_mdp_5)
# print(rew[0])

print(robot_obs_traj)
print(pomdp_r_1.state_string)
# choice_lab = pomdp_r_1.pomdp.choice_labeling
# obs_val = pomdp_r_1.pomdp.observation_valuations
# print(pomdp_r_1.action_string)
# print(pomdp_r_1.obs_string)

print(obs_action_from_traj(pomdp_r_1, robot_obs_traj, state_dict))
