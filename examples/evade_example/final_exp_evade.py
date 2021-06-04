#!/usr/bin/env python
# coding: utf-8

from mce_irl_pomdps import parser_pomdp
from mce_irl_pomdps import irl_pomdp_solver as irl_solver
import numpy as np
import stormpy

# For reproducibility
np.random.seed(201)


# Build pomdps with different memory size
pomdp_r_1 = parser_pomdp.PrismModel("evade_5_2.pm", counter_type=stormpy.pomdp.PomdpMemoryPattern.selective_counter, memory_len=1, export=False)

# Set the parameter for the trust region
irl_solver.trustRegion = {'red' : lambda x : ((x - 1) / 1.5 + 1),
                          'aug' : lambda x : min(1.5,(x-1)*1.25+1),
                          'lim' : 1+1e-3}

# Options for the solver
options_opt = irl_solver.OptOptions(mu=1e3, mu_spec=1, mu_rew=1, maxiter=100, maxiter_weight=100,
                                    graph_epsilon=0, discount=0.999, verbose=True, verbose_solver=False)

# True reward in the POMDP environment
weight = { 'crash_state' : 10, 'finish' : 20, 'time' : 0.1}


# Build the solver for memoryless policy
irlPb_1 = irl_solver.IRLSolver(pomdp_r_1, init_trust_region=1.01, max_trust_region=1.5, options=options_opt)

# Get the optimal policy for memory size 1 and save such policy and the associated performances
pol_val_grb_1 = irlPb_1.from_reward_to_policy_via_scp(weight, save_info=(20, 'evade_mem1_fwd', weight))
# Get the optimal policy if the agent has full observability
pol_val_mdp = irlPb_1.from_reward_to_optimal_policy_mdp_lp(weight, gamma=options_opt.discount, save_info=(-1,'evade_mdp_fwd', weight))

# Generate Trajectories using the state-based policy from the MDP and observation-based from POMDP
obs_based = True
pol_val_grb_1 = parser_pomdp.correct_policy(pol_val_grb_1) # Correct the policy for numerical instabilities
traj_pomdp_mem_1, _ = pomdp_r_1.simulate_policy(pol_val_grb_1, weight, 10, 500, obs_based=obs_based, stop_at_accepting_state=True)
obs_based = False
traj_mdp_1, _ = pomdp_r_1.simulate_policy(pol_val_mdp, weight,10, 500, obs_based=obs_based, stop_at_accepting_state=True)


# COmpute the feature expectation of the trajectorie
feat_pomdp_mem1_15 =irlPb_1.compute_feature_from_trajectory(traj_pomdp_mem_1)
feat_mdp_15 =irlPb_1.compute_feature_from_trajectory(traj_mdp_1)

# Print the attained featurres values
print(feat_pomdp_mem1_15)
print(feat_mdp_15)

# Trust region contraction and expansion
irl_solver.trustRegion = {'red' : lambda x : ((x - 1) / 1.5 + 1),
                          'aug' : lambda x : min(1.5,(x-1)*1.25+1),
                          'lim' : 1+1e-4}

options_opt = irl_solver.OptOptions(mu=1e3, mu_spec=1e1, mu_rew=1, maxiter=100, max_update=2, 
                                    maxiter_weight=300, rho_weight=1, verbose_solver=False,
                                    graph_epsilon=1e-6, discount=0.999, verbose=False, verbose_weight=True)
# Decreasing step size in the gradient updates
irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1 / np.power(iterVal+1, 0.5)


# Learn from the MDP demonstrations on a single memory
irlPb_1._options = options_opt
_, pol_mdp_mem1_15 = irlPb_1.solve_irl_pomdp_given_traj(feat_mdp_15, save_info=(20, 'evade_mem1_trajsize10mdp_irl', weight))
_, pol_pomdp_mem1_15 = irlPb_1.solve_irl_pomdp_given_traj(feat_pomdp_mem1_15, save_info=(20, 'evade_mem1_trajsize10pomdp_irl', weight))

# Build the model with side information
pomdp_r_1_si = parser_pomdp.PrismModel("evade_5_2.pm",  ["P=? [\"notbad\" U \"goal\"]"], counter_type=stormpy.pomdp.PomdpMemoryPattern.selective_counter, memory_len=1, export=False)


options_opt = irl_solver.OptOptions(mu=1e4, mu_spec=1e1, mu_rew=1, maxiter=100, max_update= 2, 
									maxiter_weight=300, rho_weight= 1, verbose_solver=False,
									graph_epsilon=1e-6, discount=0.999, verbose=False, verbose_weight=True)
# Build the solver for different memory size
irlPb_1_si = irl_solver.IRLSolver(pomdp_r_1_si, init_trust_region=1.01, sat_thresh=0.98, max_trust_region=1.5, options=options_opt)

# Learn from the MDP demonstrations on a single memory
irlPb_1_si._options = options_opt
_, pol_mdp_mem1_5_si = irlPb_1_si.solve_irl_pomdp_given_traj(feat_mdp_15, save_info=(20, 'evade_mem1_trajsize10mdp_irl_si', weight))
_, pol_pomdp_mem1_5_si = irlPb_1_si.solve_irl_pomdp_given_traj(feat_pomdp_mem1_15, save_info=(20, 'evade_mem1_trajsize10pomdp_irl_si', weight))
