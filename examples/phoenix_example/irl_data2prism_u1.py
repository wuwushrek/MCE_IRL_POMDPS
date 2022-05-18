#!/usr/bin/env python
# coding: utf-8

from mce_irl_pomdps import parser_pomdp
from mce_irl_pomdps import irl_pomdp_solver as irl_solver
import numpy as np
import stormpy

import pickle
import tikzplotlib

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
		skip_traj = False
		# Iterate through the state (i,j,feat) action in that list
		for (state, act) in traj:
			# Transform the (i,j,feat) state representation into state used in the prism file
			state_id = state_dict[state]
			res_obs = None
			res_act = None
			# Iterate through the pomdp labelling function to find the storm state corresponding to this state
			for state_val, (string_val, obs_id, act_reprs) in pomdp_instance._label_state_action_nomem.items():
				_state_val = int(string_val.split('\t')[0].split('=')[1])
				# If we find the correct state identifier
				if state_id == _state_val:
					# Save the corresponding observation for computing the reward feature
					res_obs = obs_id
					# Itearate through the actions associated to this state to find the corresponding action
					for _act, _act_label in act_reprs.items():
						if len(_act_label) <= 0:
							skip_traj = True
							continue
						assert len(_act_label) == 1, 'Label length should be 1 -> {}'.format(_act_label)
						if act in _act_label:
							res_act = _act
							break
					assert skip_traj or (res_act is not None)
					break
			assert skip_traj or res_obs is not None and res_act is not None, 'Not found the observation and action associated to {}, {}'.format(state, act)
			m_temp_list.append((res_obs, res_act))
		if not skip_traj:
			m_rest.append(m_temp_list)
	return m_rest

def convert_stormstate_to_phoenixstate(trajs_storm, state_dict, pomdp_instance):
	state2feat = { k : (i,j,feat) for (i,j,feat), k in state_dict.items()}
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
np.random.seed(101)

# datafile = "phoenix_scen1_r4uncert_data.pkl"
datafile = "phoenix_u1_r5_s0.1_f0_data.pkl"
# Others parameters
# outfile = 'phoenix_u1_r5_s0.1_f0_mdppol'
outfile = 'phoenix_u1_r5_s0.1_f0_pomdppol'

# Load the data file for robot trajectory
mFile = open(datafile, 'rb')
mData = pickle.load(mFile)

# Get the robot trajectory
robot_obs_traj = mData['robot_obs']
state_dict = mData['state_dict']
obs_dict = mData['obs_dict']

prism_file = mData['prism_file']

max_traj = 300 # Number of run
dur_traj = 300 # Length of each run

optimal_mdp = False
optimal_pomdp = False


# Load the pomdp model
# pomdp_r = parser_pomdp.PrismModel(prism_file, counter_type=stormpy.pomdp.PomdpMemoryPattern.selective_counter, memory_len=1, export=False)
# # pomdp_r = parser_pomdp.PrismModel(prism_file, counter_type=stormpy.pomdp.PomdpMemoryPattern.fixed_counter, memory_len=4, export=False)
# sat_thresh = 0

pomdp_r = parser_pomdp.PrismModel(prism_file, ["P=? [\"notgravel\" U \"goal\"]"], counter_type=stormpy.pomdp.PomdpMemoryPattern.selective_counter, memory_len=1, export=False)
# pomdp_r = parser_pomdp.PrismModel(prism_file, counter_type=stormpy.pomdp.PomdpMemoryPattern.fixed_counter, memory_len=4, export=False)
sat_thresh = 0.8

# Options for the solver
options_opt = irl_solver.OptOptions(mu=1e3, mu_spec=1, mu_rew=1.0, maxiter=150, maxiter_weight=150,
                                    graph_epsilon=1e-8, discount=0.998, verbose=True, verbose_solver=False)


print(pomdp_r.pomdp)

# Set the parameter for the trust region
irl_solver.trustRegion = {'red' : lambda x : ((x - 1) / 1.5 + 1),
                          'aug' : lambda x : min(1.5,(x-1)*1.25+1),
                          'lim' : 1+1e-2}


irlPb_1 = irl_solver.IRLSolver(pomdp_r, init_trust_region=1.05, max_trust_region=1.5, sat_thresh=sat_thresh, options=options_opt)


# True reward in the POMDP environment
weight = {'goal' : 50, 'road' : 0.2, 'gravel' : 30, 'grass' : 2, 'time' : 0.5}


# # Get the optimal policy for memory size 1 and save such policy and the associated performances
# obs_based = True
# pol_val_grb_1 = irlPb_1.from_reward_to_policy_via_scp(weight, save_info=(20, '_ph_mem1_fwd', weight))

# # Get the optimal policy if the agent has full observability
if optimal_mdp:
	obs_based = False
	pol_val_grb_1 = irlPb_1.from_reward_to_optimal_policy_mdp_lp(weight, gamma=options_opt.discount, save_info=(-1,'_ph_mdp_fwd', weight))
	pol_val_grb_1 = parser_pomdp.correct_policy(pol_val_grb_1)

if optimal_pomdp:
	obs_based = True
	pol_val_grb_1 = irlPb_1.from_reward_to_policy_via_scp(weight, save_info=(20,'_ph_pomdp_fwd', weight))
	pol_val_grb_1 = parser_pomdp.correct_policy(pol_val_grb_1)


if optimal_mdp or optimal_pomdp:
	stat_pol = dict()
	_, rew_pol = pomdp_r.simulate_policy(pol_val_grb_1, weight, max_traj, dur_traj, obs_based=obs_based, stop_at_accepting_state=False, stat=stat_pol)
	stat_pol['phoenix_traj'] = convert_stormstate_to_phoenixstate(stat_pol['state_evol'], state_dict, pomdp_r)
	save_result_file = open(outfile+'.pkl', 'wb')
	pickle.dump({'stat' : stat_pol, 'policy' : pol_val_grb_1, 'reward' : rew_pol}, save_result_file)
	exit()


# COmpute the feature of the expert trajectories
expert_trajectory = obs_action_from_traj(pomdp_r, robot_obs_traj, state_dict)
feat_expert = irl_solver.compute_feature_from_trajectory_and_rewfeat(expert_trajectory, irlPb_1._pomdp._reward_feat_nomem, irlPb_1._options.discount)
# feat_expert = irlPb_1.compute_feature_from_trajectory(trajExpert_nosi)

# Print the attained featurres values
print(feat_expert)

# Options for the solver
options_opt = irl_solver.OptOptions(mu=1e3, mu_spec=1, mu_rew=1.0, maxiter=4, maxiter_weight=30, verbose_weight=True,
                                    graph_epsilon=1e-8, discount=0.998, verbose=False, verbose_solver=False)


# Set the parameter for the trust region
irl_solver.trustRegion = {'red' : lambda x : ((x - 1) / 1.5 + 1),
                          'aug' : lambda x : min(1.5,(x-1)*1.25+1),
                          'lim' : 1+1e-2}

# Decreasing step size in the gradient updates
irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1 / np.power(iterVal+1, 0.5)


# Learn from the expert on a single memory without side information
irlPb_1._options = options_opt
_, pol_val_grb_1 = irlPb_1.solve_irl_pomdp_given_traj(feat_expert, save_info=(20, outfile+'_log', weight))
pol_val_grb_1 = parser_pomdp.correct_policy(pol_val_grb_1)

obs_based = True
stat_pol = dict()
_, rew_pol = pomdp_r.simulate_policy(pol_val_grb_1, weight, max_traj, dur_traj, obs_based=obs_based, stop_at_accepting_state=False, stat=stat_pol)
stat_pol['phoenix_traj'] = convert_stormstate_to_phoenixstate(stat_pol['state_evol'], state_dict, pomdp_r)
save_result_file = open(outfile+'.pkl', 'wb')
pickle.dump({'stat' : stat_pol, 'policy' : pol_val_grb_1, 'reward' : rew_pol}, save_result_file)