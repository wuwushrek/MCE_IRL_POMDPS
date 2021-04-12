#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mce_irl_pomdps import parser_pomdp
from mce_irl_pomdps import irl_pomdp_solver as irl_solver
import numpy as np


# In[2]:


# Set seed for reproductibility
np.random.seed(201)
# Open and build the prism file under different side information
pomdp_nosideinfo = parser_pomdp.PrismModel("maze_stochastic.pm", [], export=False)
pomdp_reachavoidsideinfo = parser_pomdp.PrismModel("maze_stochastic.pm", ["P=? [F \"target\"]", "P=? [G !\"poison_light\"]"], export=True)


# In[8]:


# Options for the solver
options_opt = irl_solver.OptOptions(mu=1e2, mu_spec=0, maxiter=20, maxiter_weight=20,
                      graph_epsilon=0, discount=0.999, verbose=True)
# Build Instances of the IRL problem
irlPb_nosideinfo = irl_solver.IRLSolver(pomdp_nosideinfo, init_trust_region=4, options=options_opt)
# True reward in the POMDP environment
weight = { 'poisonous' : 10, 'total_time' : 0.1, 'goal_reach' : 50}


# In[9]:


# Find the optimal policy that maximizes the expected reward using Gurobi non convex solver
# The maximum expected reward obtained is 196.03264535140676
pol_val_opt = irlPb_nosideinfo.from_reward_to_optimal_policy_nonconvex_grb(weight)
pol_val_opt = {o : {a : 0 if val<0 else val for a, val in actDict.items()} for o, actDict in pol_val_opt.items()}


# In[10]:


# Find the optimal policy that maximizes the expected reward using the sequential convex programming scheme
# The maximum expected reward obtained is 195.33685851754532
pol_val_scp = irlPb_nosideinfo.from_reward_to_policy_via_scp(weight)
pol_val_scp = {o : {a : 0 if val<0 else val for a, val in actDict.items()} for o, actDict in pol_val_scp.items()}


# In[11]:


# Compute the optimal policy on the underlying MDP
# The maximum expected reward obtained is 394.1186174404245
pol_val_mdp = irlPb_nosideinfo.from_reward_to_optimal_policy_mdp_lp(weight, gamma=options_opt.discount)
pol_val_mdp = {o : {a : 0 if val<0 else val for a, val in actDict.items()} for o, actDict in pol_val_mdp.items()}


# In[71]:


# Generate Trajectory of different length using the state-based policy from the MDP
obs_based = False
traj_mdp_15, rewData_15 = pomdp_nosideinfo.simulate_policy(pol_val_mdp, weight, 30, 50, 
                                            obs_based=obs_based, stop_at_accepting_state=True)
traj_mdp_30, rewData_30 = pomdp_nosideinfo.simulate_policy(pol_val_mdp, weight, 100, 50, 
                                            obs_based=obs_based, stop_at_accepting_state=True)


# In[13]:


# Define some parameters when optimizing the problems
# Set the parameter for the step size in the update of the weight
irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1.0/np.power(iterVal, 0.6)
# Set the parameter for the trust region
irl_solver.trustRegion = {'red' : lambda x : ((x - 1) / 1.5 + 1),
                          'aug' : lambda x : min(10,(x-1)*1.25+1),
                          'lim' : 1+1e-4}
# Set the parameter for minimum state visitation count to be considered as zero
irl_solver.ZERO_NU_S = 1e-8


# In[14]:


# Parameter for the optimization
irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1.0/np.power(iterVal, 1.5)
options_opt = irl_solver.OptOptions(mu=1e4, mu_spec=1e4, maxiter=100, maxiter_weight=100,
                      graph_epsilon=0, discount=0.999, verbose=False, verbose_weight=True)
irlPb1 = irl_solver.IRLSolver(pomdp_nosideinfo, sat_thresh=0, init_trust_region=2, rew_eps=1e-4, options=options_opt)


# In[15]:


irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1.0/np.power(iterVal, 0.6)
weight_mdp_30, pol_mdp_30 = irlPb1.solve_irl_pomdp_given_traj(traj_mdp_30)


# In[16]:


irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1.0/np.power(iterVal, 0.6)
weight_mdp_15, pol_mdp_15 = irlPb1.solve_irl_pomdp_given_traj(traj_mdp_15)


# In[ ]:


# irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1.0/np.power(iterVal, 1.5)
# weight_pomdp_15, pol_pomdp_15 = irlPb1.solve_irl_pomdp_given_traj(traj_pomdp_15)


# In[ ]:


# irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1.0/np.power(iterVal, 1.55)
# weight_pomdp_30, pol_pomdp_30 = irlPb1.solve_irl_pomdp_given_traj(traj_pomdp_30)


# In[18]:


irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1.0/np.power(iterVal, 1.5)
options_opt = irl_solver.OptOptions(mu=1e4, mu_spec=1e4, maxiter=100, maxiter_weight=100,
                      graph_epsilon=0, discount=0.999, verbose=False, verbose_weight=True)
irlPb3 = irl_solver.IRLSolver(pomdp_reachavoidsideinfo, sat_thresh=0.95, init_trust_region=2, rew_eps=1e-4, options=options_opt)


# In[19]:


irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1.0/np.power(iterVal, 0.6)
weight_avoidreach_mdp_30, pol_avoidreach_mdp_30 = irlPb3.solve_irl_pomdp_given_traj(traj_mdp_30)


# In[20]:


irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1.0/np.power(iterVal, 0.6)
weight_avoidreach_mdp_15, pol_avoidreach_mdp_15 = irlPb3.solve_irl_pomdp_given_traj(traj_mdp_15)


# In[ ]:


# irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1.0/np.power(iterVal, 0.6)
# weight_avoidreach_pomdp_15, pol_avoidreach_pomdp_15 = irlPb3.solve_irl_pomdp_given_traj(traj_pomdp_15)


# In[ ]:


# irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1.0/np.power(iterVal, 0.6)
# weight_avoidreach_pomdp_30, pol_avoidreach_pomdp_30 = irlPb3.solve_irl_pomdp_given_traj(traj_pomdp_30)


# In[77]:


import matplotlib
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

def plot_pol(pol_val, color='red', nb_run=10, nb_iter_run=30, label='dum', alpha=0.5, is_obs=True, plot_std=False):
    obs_based = True
    _, rewData = pomdp_nosideinfo.simulate_policy(pol_val, weight, nb_run, nb_iter_run, 
                        obs_based=is_obs, stop_at_accepting_state=False)
    arr_rewData = np.cumsum(np.array(rewData), axis=1)
    mean_rew = np.mean(arr_rewData, axis = 0)
    min_rew = np.min(arr_rewData, axis=0)
    max_rew = np.max(arr_rewData, axis=0)
    std_rew = np.std(arr_rewData, axis=0)
    axis_x = [i for i in range(mean_rew.shape[0])]
    plt.plot(axis_x, mean_rew, color=color, label=label)
    if plot_std:
        plt.fill_between(axis_x, np.maximum(min_rew,mean_rew-std_rew), np.minimum(max_rew,mean_rew+std_rew), color=color, alpha=alpha)


# In[75]:


np.random.seed(501)
nb_run = 300
max_iter_per_run = 250
plt.figure()
plot_pol(pol_val_mdp, color='blue', nb_run=nb_run, nb_iter_run=max_iter_per_run, is_obs=False, label='Optimal policy on the MDP', alpha=1)
plot_pol(pol_val_opt, color='green', nb_run=nb_run, nb_iter_run=max_iter_per_run, is_obs=True, label='Optimal policy on the POMDP', alpha=0.8)
# plot_pol(pol_val_scp, color='red', nb_run=nb_run, nb_iter_run=max_iter_per_run, is_obs=True)
plot_pol(pol_avoidreach_mdp_15, color='orange', nb_run=nb_run, nb_iter_run=max_iter_per_run, is_obs=True, label='Learned policy with side information', alpha = 0.6)
plot_pol(pol_mdp_30, color='cyan', nb_run=nb_run, nb_iter_run=max_iter_per_run, is_obs=True, label='Learned policy with no side information', alpha=0.2)
plt.ylabel('Mean Accumulated reward')
plt.xlabel('Time steps')
plt.grid(True)
plt.legend(ncol=2, bbox_to_anchor=(0,1), loc='lower left', columnspacing=1.0)
plt.tight_layout()
# plt.show()


# In[78]:


np.random.seed(501)
nb_run = 300
max_iter_per_run = 2500
plt.figure()
plot_pol(pol_val_mdp, color='blue', nb_run=nb_run, nb_iter_run=max_iter_per_run, is_obs=False, label='Optimal policy on the MDP', alpha=1, plot_std=True)
plot_pol(pol_val_opt, color='green', nb_run=nb_run, nb_iter_run=max_iter_per_run, is_obs=True, label='Optimal policy on the POMDP', alpha=0.8, plot_std=True)
# plot_pol(pol_val_scp, color='red', nb_run=nb_run, nb_iter_run=max_iter_per_run, is_obs=True)
plot_pol(pol_avoidreach_mdp_15, color='orange', nb_run=nb_run, nb_iter_run=max_iter_per_run, is_obs=True, label='Learned policy with side information', alpha = 0.6, plot_std=True)
plot_pol(pol_mdp_30, color='cyan', nb_run=nb_run, nb_iter_run=max_iter_per_run, is_obs=True, label='Learned policy with no side information', alpha=0.2, plot_std=True)
plt.ylabel('Mean Accumulated reward')
plt.xlabel('Time steps')
plt.grid(True)
plt.legend(ncol=2, bbox_to_anchor=(0,1), loc='lower left', columnspacing=1.0)
plt.tight_layout()
plt.show()

