#!/usr/bin/env python
# coding: utf-8

from mce_irl_pomdps import parser_pomdp
from mce_irl_pomdps import irl_pomdp_solver as irl_solver
import numpy as np
import stormpy

import pickle

def obs_action_from_traj(pomdp_instance, m_trajs, state_dict):
    """ Find the list of observation action in term of the pomdp instance created by prism
        :param 
    """
    # Save the result of the conversion
    m_rest = list()
    # Iterate through all trajectories in the agent trajectory
    for traj in m_trajs:
        # Save a temporary list containing the (obs, act) in 
        m_temp_list = list()
        # Iterate through the state (i,j,feat) action in that list
        for (state, act) in traj:
            # Transform the (i,j,feat) state representation into state used in the prism file
            state_id = state_dict[state]
            res_obs = None
            res_act = None
            # Iterate tr=hrough the pomdp labelling function to find the storm state corresponding to this state
            for state_val, (string_val, obs_id, act_reprs) in pomdp_instance._label_state_action_nomem.items():
                _state_val = int(string_val.split('\t')[0].split('=')[1])
                # If we find the correct state identifier
                if state_id == _state_val:
                    # Save the corresponding observation for computing the reward feature
                    res_obs = obs_id
                    # Itearate through the actions associated to this state to find the corresponding action
                    for _act, _act_label in act_reprs.items():
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

def convert_stormstate_to_phoenixstate(trajs_storm, state_dict, pomdp_instance):
    state2feat = { k : (i,j,*feat) for (i,j,*feat), k in state_dict.items()}
    m_rest = list()
    for traj in trajs_storm:
        m_temp_list = list()
        for (state, act) in traj:
             (string_val, obs_id, act_reprs) = pomdp_instance._label_state_action_nomem[state]
             _state_val = int(string_val.split('\t')[0].split('=')[1])
             if _state_val < 0 or _state_val >= len(state2feat):
                continue
             m_temp_list.append(state2feat[_state_val])
        m_rest.append(m_temp_list)
    return m_rest


# For reproducibility
np.random.seed(201)

# datafile = "phoenix_scen1_r4uncert_data.pkl"
datafile = "phoenix_scen1_r5featobs_data.pkl"

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
# pomdp_r_5 = parser_pomdp.PrismModel(prism_file, counter_type=stormpy.pomdp.PomdpMemoryPattern.fixed_counter, memory_len=4, export=False)

print(pomdp_r_1.pomdp)
# print(pomdp_r_4.pomdp)

# Set the parameter for the trust region
irl_solver.trustRegion = {'red' : lambda x : ((x - 1) / 1.5 + 1),
                          'aug' : lambda x : min(1.5,(x-1)*1.25+1),
                          'lim' : 1+1e-2}

# Options for the solver
options_opt = irl_solver.OptOptions(mu=1e3, mu_spec=1, mu_rew=1.0, maxiter=100, maxiter_weight=100,
                                    graph_epsilon=1e-8, discount=0.98, verbose=True, verbose_solver=False)

# True reward in the POMDP environment
weight = {'goal' : 50, 'road' : 0.1, 'gravel' : 0.5, 'grass' : -0.1, 'time' : 0.4}

# Build the solver for different memory size
irlPb_1 = irl_solver.IRLSolver(pomdp_r_1, init_trust_region=1.01, max_trust_region=1.5, options=options_opt)

# Get the optimal policy for memory size 1 and save such policy and the associated performances
pol_val_mdp = irlPb_1.from_reward_to_optimal_policy_mdp_lp(weight, gamma=options_opt.discount, save_info=(-1,'phoenix_mdp_fwd', weight))
pol_val_grb_1 = irlPb_1.from_reward_to_policy_via_scp(weight, save_info=(20, 'phoenix_mem1_fwd', weight))

# Simulate the obtained policies on the MDP to get optimal trajectories
obs_based = False
stat_mdp_val = dict()
_, rew_mdp = pomdp_r_1.simulate_policy(pol_val_mdp, weight, 10, 100, obs_based=obs_based, stop_at_accepting_state=False, stat=stat_mdp_val)
stat_mdp_val['phoenix_traj'] = convert_stormstate_to_phoenixstate(stat_mdp_val['state_evol'], state_dict, pomdp_r_1)

# Simulate the obtained policies on the POMDP to get optimal trajectories
obs_based = True
stat_pomdp_val = dict()
trajExpert_nosi, _ = pomdp_r_1.simulate_policy(parser_pomdp.correct_policy(pol_val_grb_1), weight, 10, 100, obs_based=obs_based, stop_at_accepting_state=True)
_, rew_pomdp = pomdp_r_1.simulate_policy(parser_pomdp.correct_policy(pol_val_grb_1), weight, 10, 100, obs_based=obs_based, stop_at_accepting_state=False, stat=stat_pomdp_val)
stat_pomdp_val['phoenix_traj'] = convert_stormstate_to_phoenixstate(stat_pomdp_val['state_evol'], state_dict, pomdp_r_1)


# COmpute the feature of the expert trajectories
expert_trajectory = obs_action_from_traj(pomdp_r_1, robot_obs_traj, state_dict)
feat_expert = irl_solver.compute_feature_from_trajectory_and_rewfeat(expert_trajectory, irlPb_1._pomdp._reward_feat_nomem, irlPb_1._options.discount)
# feat_expert = irlPb_1.compute_feature_from_trajectory(trajExpert_nosi)

# Print the attained featurres values
print(feat_expert)

# Trust region contraction and expansion
irl_solver.trustRegion = {'red' : lambda x : ((x - 1) / 1.5 + 1),
                          'aug' : lambda x : min(1.5,(x-1)*1.25+1),
                          'lim' : 1+1e-2}

options_opt = irl_solver.OptOptions(mu=1e4, mu_spec=1e1, mu_rew=1, maxiter=100, max_update=2, 
                                    maxiter_weight=150, rho_weight=1, verbose_solver=False,
                                    graph_epsilon=1e-8, discount=0.98, verbose=False, verbose_weight=True)
# Decreasing step size in the gradient updates
irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1 / np.power(iterVal+1, 0.5)


# Learn from the expert on a single memory without side information
irlPb_1._options = options_opt
_, pol_expert_nosi = irlPb_1.solve_irl_pomdp_given_traj(feat_expert, save_info=(20, 'phoenix_mem1_nosi', weight))

# Simulate the obtained policies on the POMDP to get optimal trajectories 
obs_based = True
stat_pomdp_exp_nosi_val = dict()
_, rew_pomdp_irl_nosi = pomdp_r_1.simulate_policy(parser_pomdp.correct_policy(pol_expert_nosi), weight, 10, 100, obs_based=obs_based, stop_at_accepting_state=False, stat=stat_pomdp_exp_nosi_val)
stat_pomdp_exp_nosi_val['phoenix_traj'] = convert_stormstate_to_phoenixstate(stat_pomdp_exp_nosi_val['state_evol'], state_dict, pomdp_r_1)


# Build the case with side information
pomdp_r_1_si = parser_pomdp.PrismModel(prism_file, ["P=? [F \"goal\"]"], counter_type=stormpy.pomdp.PomdpMemoryPattern.selective_counter, memory_len=1, export=False)
print(pomdp_r_1_si)

options_opt = irl_solver.OptOptions(mu=1e4, mu_spec=1e1, mu_rew=1, maxiter=100, max_update= 2, 
                                    maxiter_weight=150, rho_weight= 1, verbose_solver=False,
                                    graph_epsilon=1e-8, discount=0.98, verbose=False, verbose_weight=True)


# Build the solver for different memory size
irlPb_1_si = irl_solver.IRLSolver(pomdp_r_1_si, init_trust_region=1.01, sat_thresh=0.98, max_trust_region=1.5, options=options_opt)

# COmpute the feature of the expert trajectories
expert_trajectory = obs_action_from_traj(pomdp_r_1_si, robot_obs_traj, state_dict)
feat_expert = irl_solver.compute_feature_from_trajectory_and_rewfeat(expert_trajectory, irlPb_1_si._pomdp._reward_feat_nomem, irlPb_1_si._options.discount)
# feat_expert = irl_solver.compute_feature_from_trajectory_and_rewfeat(expert_trajectory, irlPb_1._pomdp._reward_feat_nomem, irlPb_1._options.discount)

# Learn from the MDP demonstrations on a single memory
irlPb_1_si._options = options_opt
_, pol_expert_si = irlPb_1_si.solve_irl_pomdp_given_traj(feat_expert, save_info=(20, 'phoenix_mem1_si', weight))

# # Simulate the obtained policies on the POMDP to get optimal trajectories 
obs_based = True
stat_pomdp_exp_si_val = dict()
_, rew_pomdp_irl_si = pomdp_r_1.simulate_policy(parser_pomdp.correct_policy(pol_expert_si), weight, 10, 100, obs_based=obs_based, stop_at_accepting_state=False, stat=stat_pomdp_exp_si_val)
stat_pomdp_exp_si_val['phoenix_traj'] = convert_stormstate_to_phoenixstate(stat_pomdp_exp_si_val['state_evol'], state_dict, pomdp_r_1_si)


save_result_file = open(prism_file.split('.')[0]+'_traj_res.pkl', 'wb')
mDictSave = {'pol_expert_si' : pol_expert_si, 'stat_pomdp_exp_si_val' : stat_pomdp_exp_si_val, 'rew_pomdp_irl_si' : rew_pomdp_irl_si,
            'pol_expert_nosi' : pol_expert_nosi, 'stat_pomdp_exp_nosi_val' : stat_pomdp_exp_nosi_val, 'rew_pomdp_irl_nosi' : rew_pomdp_irl_nosi,
            'pol_val_mdp' : pol_val_mdp, 'stat_mdp_val' : stat_mdp_val, 'rew_mdp' : rew_mdp,
            'pol_val_pomdp' : pol_val_grb_1, 'stat_pomdp_val' : stat_pomdp_val, 'rew_pomdp' : rew_pomdp}
pickle.dump(mDictSave, save_result_file)
