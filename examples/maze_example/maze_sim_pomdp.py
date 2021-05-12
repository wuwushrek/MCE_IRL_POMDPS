#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mce_irl_pomdps import parser_pomdp
from mce_irl_pomdps import irl_pomdp_solver as irl_solver
import numpy as np


# In[2]:


# Set seed for reproductibility -> IncREMENT MEMORY LEN FOR HIGHER PERFORMANCE
np.random.seed(201)
pomdp_r_init = parser_pomdp.PrismModel("maze_stochastic.pm", ["P=? [F \"target\"]"], memory_len=3, export=True)


# In[3]:


# Options for the solver
options_opt = irl_solver.OptOptions(mu=1e3, mu_spec=1e4, maxiter=100, maxiter_weight=100,
                      graph_epsilon=0, discount=0.999, verbose=True)
# True reward in the POMDP environment
weight = { 'poisonous' : 10, 'total_time' : 0.1, 'goal_reach' : 50}


# In[4]:


# Build the instance without side information
pomdp_r_init._has_sideinfo = False # Ignore the side information for the first part
irlPb_nosi = irl_solver.IRLSolver(pomdp_r_init, init_trust_region=4, options=options_opt)


# In[5]:


# Find the optimal policy that maximizes the expected reward using Gurobi non convex solver on the problem with no side information
# pol_val_grb_nosi = irlPb_nosi.from_reward_to_optimal_policy_nonconvex_grb(weight)


# In[ ]:


# Find the optimal policy that maxpol_val_nosideinfoimizes the expected reward using the sequential convex programming scheme
pol_val_grb_nosi = irlPb_nosi.from_reward_to_policy_via_scp(weight)


# In[6]:


# Compute the optimal policy on the underlying MDP
pol_val_mdp = irlPb_nosi.from_reward_to_optimal_policy_mdp_lp(weight, gamma=options_opt.discount)


# In[7]:


# Generate Trajectory of different length using the state-based policy from the MDP
obs_based = True
traj_pomdp_30, rewData_pomdp_30 = pomdp_r_init.simulate_policy(pol_val_grb_nosi, weight, 5, 1000,
                                            obs_based=obs_based, stop_at_accepting_state=True)


# In[8]:


features_traj_pomdp_30=irlPb_nosi.compute_feature_from_trajectory(traj_pomdp_30)
print(features_traj_pomdp_30)
#print(a)

pomdp_r = parser_pomdp.PrismModel("maze_stochastic.pm", ["P=? [F \"target\"]"], memory_len=1, export=True)
# In[9]:


# Define some parameters when optimizing the problems
# Set the parameter for the trust region
irl_solver.trustRegion = {'red' : lambda x : ((x - 1) / 1.5 + 1),
                          'aug' : lambda x : min(1.5,(x-1)*1.5+1),
                          'lim' : 1+1e-8}
# Set the parameter for minimum state visitation count to be considered as zero
irl_solver.ZERO_NU_S = 1e-8


# In[10]:


# Parameter for the optimization\
pomdp_r._has_sideinfo = False
irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1.0  /np.power(iterVal+1, 0.5)
options_opt = irl_solver.OptOptions(mu=1e3, mu_spec=1e4, maxiter=1000, maxiter_weight=100, rho=0e-4, rho_weight= 1,
                      graph_epsilon=0, discount=0.999, verbose=False, verbose_weight=True)
irlPb1 = irl_solver.IRLSolver(pomdp_r, sat_thresh=0, init_trust_region=1.5, rew_eps=1e-4, options=options_opt)
features_traj_pomdp_30=irlPb_nosi.compute_feature_from_trajectory(traj_pomdp_30)
weight_pomdp_30, pol_pomdp_30 = irlPb1.solve_irl_pomdp_given_traj(features_traj_pomdp_30)


# In[11]:


# Parameter for the optimization\
pomdp_r._has_sideinfo = True
irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1.0  /np.power(iterVal+1, 0.5)
options_opt = irl_solver.OptOptions(mu=1e3, mu_spec=1e4, maxiter=1000, maxiter_weight=100,rho=0e-4, rho_weight= 1,
                      graph_epsilon=0, discount=0.999, verbose=False, verbose_weight=True)
irlPb2 = irl_solver.IRLSolver(pomdp_r, sat_thresh=0.8, init_trust_region=1.5, rew_eps=1e-4, options=options_opt)
features_traj_pomdp_30=irlPb_nosi.compute_feature_from_trajectory(traj_pomdp_30)
weight_pomdp_30_si, pol_pomdp_30_si = irlPb2.solve_irl_pomdp_given_traj(features_traj_pomdp_30)

pomdp_r_mem = parser_pomdp.PrismModel("maze_stochastic.pm", ["P=? [F \"target\"]"], memory_len=3, export=True)
# In[9]:
print(features_traj_pomdp_30)
print(pomdp_r_mem)

# Define some parameters when optimizing the problems
# Set the parameter for the trust region
irl_solver.trustRegion = {'red' : lambda x : ((x - 1) / 1.5 + 1),
                          'aug' : lambda x : min(1.5,(x-1)*1.5+1),
                          'lim' : 1+1e-8}
# Set the parameter for minimum state visitation count to be considered as zero
irl_solver.ZERO_NU_S = 1e-8


# In[10]:


# Parameter for the optimization\
pomdp_r_mem._has_sideinfo = False
irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1.0  /np.power(iterVal+1, 0.5)
options_opt = irl_solver.OptOptions(mu=1e3, mu_spec=1e4, maxiter=100, maxiter_weight=100, rho=0e-4, rho_weight= 1,
                      graph_epsilon=1e-6, discount=0.999, verbose=False, verbose_weight=True)
irlPb1_mem = irl_solver.IRLSolver(pomdp_r_mem, sat_thresh=0, init_trust_region=1.5, rew_eps=1e-4, options=options_opt)
features_traj_pomdp_30=irlPb_nosi.compute_feature_from_trajectory(traj_pomdp_30)
weight_pomdp_30_3mem, pol_pomdp_30_3mem = irlPb1_mem.solve_irl_pomdp_given_traj(features_traj_pomdp_30)


# In[11]:

print(features_traj_pomdp_30)

# Parameter for the optimization\
pomdp_r_mem._has_sideinfo = True
irl_solver.gradientStepSize = lambda iterVal, diffFeat : 1.0  /np.power(iterVal+1, 0.5)
options_opt = irl_solver.OptOptions(mu=1e3, mu_spec=1e4, maxiter=100, maxiter_weight=100, rho=0e-4, rho_weight= 1,
                      graph_epsilon=1e-6, discount=0.999, verbose=False, verbose_weight=True)
irlPb2_mem = irl_solver.IRLSolver(pomdp_r_mem, sat_thresh=0.8, init_trust_region=1.5, rew_eps=1e-4, options=options_opt)
features_traj_pomdp_30=irlPb_nosi.compute_feature_from_trajectory(traj_pomdp_30)
weight_pomdp_30_si_3mem, pol_pomdp_30_si_3mem = irlPb2_mem.solve_irl_pomdp_given_traj(features_traj_pomdp_30)

print(features_traj_pomdp_30)

# In[20]:

#pol_pomdp_30_opt = irlPb1.from_reward_to_policy_via_scp(weight_pomdp_30)
#pol_pomdp_30_si_opt = irlPb2.from_reward_to_policy_via_scp(weight_pomdp_30_si)
#pol_pomdp_30_opt_3mem = irlPb1_mem.from_reward_to_policy_via_scp(weight_pomdp_30_3mem)
#pol_pomdp_30_si_opt_3mem = irlPb2_mem.from_reward_to_policy_via_scp(weight_pomdp_30_si_3mem)



# pomdp_reachavoidsideinfo
np.random.seed(501)
nb_run = 300
max_iter_per_run = 2000
_, rewDataPomdpOpt = pomdp_r_init.simulate_policy(pol_val_grb_nosi, weight, nb_run, max_iter_per_run,
                        obs_based=True, stop_at_accepting_state=True)
_, rewDataMdpOpt = pomdp_r_init.simulate_policy(pol_val_mdp, weight, nb_run, max_iter_per_run,
                        obs_based=False, stop_at_accepting_state=True)
_, rewDataPomdpNoSi = pomdp_r.simulate_policy(pol_pomdp_30, weight, nb_run, max_iter_per_run,
                        obs_based=True, stop_at_accepting_state=True)
_, rewDataPomdpSi = pomdp_r.simulate_policy(pol_pomdp_30_si, weight, nb_run, max_iter_per_run,
                        obs_based=True, stop_at_accepting_state=True)

_, rewDataPomdpNoSi_3mem = pomdp_r_mem.simulate_policy(pol_pomdp_30_3mem, weight, nb_run, max_iter_per_run,
                        obs_based=True, stop_at_accepting_state=True)
_, rewDataPomdpSi_3mem = pomdp_r_mem.simulate_policy(pol_pomdp_30_si_3mem, weight, nb_run, max_iter_per_run,
                        obs_based=True, stop_at_accepting_state=True)

# In[21]:


# discountArray = np.array([options_opt.discount**i for i in range(max_iter_per_run)])
discountArray = np.array([1 for i in range(max_iter_per_run)])


# In[22]:


import matplotlib
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

def plot_pol(rewData, cData=-1, color='red', label='dum', alpha=0.5, plot_std=False):
    rewData = np.array(rewData) * discountArray
    arr_rewData = np.cumsum(rewData, axis=1)
    mean_rew = np.mean(arr_rewData, axis = 0)
    min_rew = np.min(arr_rewData, axis=0)
    max_rew = np.max(arr_rewData, axis=0)
    std_rew = np.std(arr_rewData, axis=0)
    axis_x = np.array([i for i in range(mean_rew.shape[0])])
#     print(mean_rew.shape, cData)
    plt.plot(axis_x[:cData], mean_rew[:cData], color=color, label=label)
    if plot_std:
        plt.fill_between(axis_x[:cData], np.maximum(min_rew,mean_rew-std_rew)[:cData], np.minimum(max_rew,mean_rew+std_rew)[:cData], color=color, alpha=alpha)


# In[23]:


nData = 150
plt.figure()
plot_pol(rewDataMdpOpt, nData, color='blue', label='Optimal policy on the MDP', alpha=1, plot_std=False)
plot_pol(rewDataPomdpOpt, nData, color='green', label='Optimal policy on the POMDP', alpha=0.8, plot_std=False)
# plot_pol(pol_val_scp, color='red', nb_run=nb_run, nb_iter_run=max_iter_per_run, is_obs=True)
plot_pol(rewDataPomdpNoSi, nData, color='orange', label='Learned policy with no side information', alpha = 0.6, plot_std=False)
plot_pol(rewDataPomdpSi, nData, color='yellow', label='Learned policy with side information', alpha=0.6, plot_std=False)
plot_pol(rewDataPomdpNoSi_3mem, nData, color='purple',label='Learned policy with no side information, mem=3', alpha = 0.6, plot_std=False)
plot_pol(rewDataPomdpSi_3mem, nData, color='red', label='Learned policy with side information, mem=3', alpha=0.6, plot_std=False)
# plot_pol(rewDataSideInfoLp, color='cyan', label='Learned policy with side information,0.7', alpha=0.2)
plt.ylabel('Mean Accumulated reward')
plt.xlabel('Time steps')
plt.grid(True)
plt.legend(ncol=2, bbox_to_anchor=(0,1), loc='lower left', columnspacing=1.0)
plt.tight_layout()
plt.show()

