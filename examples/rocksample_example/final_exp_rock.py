#!/usr/bin/env python
# coding: utf-8

from mce_irl_pomdps import parser_pomdp
from mce_irl_pomdps import irl_pomdp_solver as irl_solver
import numpy as np
import stormpy

# For reproducibility
np.random.seed(201)


# Build pomdps with different memory size
pomdp_r_1 = parser_pomdp.PrismModel("rocks_5_2.pm", counter_type=stormpy.pomdp.PomdpMemoryPattern.selective_counter, memory_len=1, export=False)
pomdp_r_5 = parser_pomdp.PrismModel("rocks_5_2.pm", counter_type=stormpy.pomdp.PomdpMemoryPattern.fixed_counter, memory_len=10, export=False)

# Set the parameter for the trust region
irl_solver.trustRegion = {'red' : lambda x : ((x - 1) / 1.5 + 1),
                          'aug' : lambda x : min(1.5,(x-1)*1.25+1),
                          'lim' : 1+1e-3}

# Options for the solver
options_opt = irl_solver.OptOptions(mu=1e3, mu_spec=1, mu_rew=1.0, maxiter=100, maxiter_weight=100,
                                    graph_epsilon=1e-6, discount=0.999, verbose=True, verbose_solver=False)

# True reward in the POMDP environment
weight = {'finish' : 100, 'bad' : 4, 'time' : 0.1}


# Build the solver for different memory size
irlPb_1 = irl_solver.IRLSolver(pomdp_r_1, init_trust_region=1.01, max_trust_region=1.5, options=options_opt)
irlPb_5 = irl_solver.IRLSolver(pomdp_r_5, init_trust_region=1.01, max_trust_region=1.5, options=options_opt)

# Get the optimal policy for memory size 1 and save such policy and the associated performances
pol_val_grb_1 = irlPb_1.from_reward_to_policy_via_scp(weight, save_info=(20, 'rock_mem1_fwd', weight))
# # Get the optimal policy for memory size 1 and save such policy and the associated performances
pol_val_grb_5 = irlPb_5.from_reward_to_policy_via_scp(weight, save_info=(20, 'rock_mem10_fwd', weight))
# Get the optimal policy if the agent has full observability
pol_val_mdp = irlPb_1.from_reward_to_optimal_policy_mdp_lp(weight, gamma=options_opt.discount, save_info=(-1,'rock_mdp_fwd', weight))

# Generate Trajectory of different length using the state-based policy from the MDP and observation-based from MDP
obs_based = True
pol_val_grb_5 = parser_pomdp.correct_policy(pol_val_grb_5) # Correct the policy for numerical instabilities
traj_pomdp_mem_5, _ = pomdp_r_5.simulate_policy(pol_val_grb_5, weight, 10, 500, obs_based=obs_based, stop_at_accepting_state=True)
obs_based = False
traj_mdp_5, _ = pomdp_r_1.simulate_policy(pol_val_mdp, weight, 10, 500, obs_based=obs_based, stop_at_accepting_state=True)


# COmpute the feature expectation of the trajectorie
feat_pomdp_mem5_5 =irlPb_5.compute_feature_from_trajectory(traj_pomdp_mem_5)
feat_mdp_5 =irlPb_1.compute_feature_from_trajectory(traj_mdp_5)

# Print the feature of the pomdp and mdp
print(feat_pomdp_mem5_5)
print(feat_mdp_5)

# Trust region contraction and expansion
irl_solver.trustRegion = {'red' : lambda x : ((x - 1) / 1.5 + 1),
                          'aug' : lambda x : min(1.5,(x-1)*1.25+1),
                          'lim' : 1+1e-4}

# Solver option
options_opt = irl_solver.OptOptions(mu=1e3, mu_spec=1e1, mu_rew=1, maxiter=100, max_update=2, 
                                    maxiter_weight=300, rho_weight= 1, verbose_solver=False,
                                    graph_epsilon=1e-6, discount=0.999, verbose=False, verbose_weight=True)

# Decreasing step size in the gradient updates
irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1 / np.power(iterVal+1, 0.5)


# Learn from the MDP demonstrations on a single memory
irlPb_1._options = options_opt
_, pol_mdp_mem1_5 = irlPb_1.solve_irl_pomdp_given_traj(feat_mdp_5, save_info=(20, 'rock_mem1_trajsize10mdp_irl', weight))
# Learn from the MDP demonstrations on a memory len 10
irlPb_5._options = options_opt
_, pol_mdp_mem5_5 = irlPb_5.solve_irl_pomdp_given_traj(feat_mdp_5, save_info=(20, 'rock_mem10_trajsize10mdp_irl', weight))

# Learn from the POMDP demonstrations on a single memory
irlPb_1._options = options_opt
_, pol_pomdp_mem1_5 = irlPb_1.solve_irl_pomdp_given_traj(feat_pomdp_mem5_5, save_info=(20, 'rock_mem1_trajsize10pomdp_irl', weight))
# Learn from the POMDP demonstrations on a memory len 10
irlPb_5._options = options_opt
_, pol_pomdp_mem5_5 = irlPb_5.solve_irl_pomdp_given_traj(feat_pomdp_mem5_5, save_info=(20, 'rock_mem10_trajsize10pomdp_irl', weight))

# feat_pomdp_mem5_5 = {'bad': -0.39790449500299907, 'time': -7.46518944379174, 'finish': 0.9906398983721957}
# feat_mdp_5 = {'bad': 0.0, 'time': -5.37993954864368, 'finish': 0.9928233552551552}

###############################################################################
# Build the model with side information
pomdp_r_1_si = parser_pomdp.PrismModel("rocks_5_2.pm",  ["P=? [F \"goal\"]"], counter_type=stormpy.pomdp.PomdpMemoryPattern.selective_counter, memory_len=1, export=False)
pomdp_r_5_si = parser_pomdp.PrismModel("rocks_5_2.pm", ["P=? [F \"goal\"]"], counter_type=stormpy.pomdp.PomdpMemoryPattern.fixed_counter, memory_len=10, export=False)


options_opt = irl_solver.OptOptions(mu=1e4, mu_spec=1e1, mu_rew=1, maxiter=100, max_update= 2, 
									maxiter_weight=50, rho_weight= 1, verbose_solver=False,
									graph_epsilon=1e-6, discount=0.999, verbose=False, verbose_weight=True)
# Build the solver for different memory size
irlPb_1_si = irl_solver.IRLSolver(pomdp_r_1_si, init_trust_region=1.01, sat_thresh=0.995, max_trust_region=1.5, options=options_opt)
irlPb_5_si = irl_solver.IRLSolver(pomdp_r_5_si, init_trust_region=1.01, sat_thresh=0.995, max_trust_region=1.5, options=options_opt)

# Learn from the MDP demonstrations on a single memory
irlPb_1_si._options = options_opt
_, pol_mdp_mem1_5_si = irlPb_1_si.solve_irl_pomdp_given_traj(feat_mdp_5, save_info=(20, 'rock_mem1_trajsize10mdp_irl_si', weight))
_, pol_pomdp_mem1_5_si = irlPb_1_si.solve_irl_pomdp_given_traj(feat_pomdp_mem5_5, save_info=(20, 'rock_mem1_trajsize10pomdp_irl_si', weight))

irlPb_5_si._options = options_opt
_, pol_mdp_mem10_5_si = irlPb_5_si.solve_irl_pomdp_given_traj(feat_mdp_5, save_info=(20, 'rock_mem10_trajsize10mdp_irl_si', weight))
_, pol_pomdp_mem10_5_si = irlPb_5_si.solve_irl_pomdp_given_traj(feat_pomdp_mem5_5, save_info=(20, 'rock_mem10_trajsize10pomdp_irl_si', weight))