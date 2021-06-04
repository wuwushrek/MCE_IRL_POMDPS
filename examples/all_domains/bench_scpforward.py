from mce_irl_pomdps import parser_pomdp
from mce_irl_pomdps import irl_pomdp_solver as irl_solver

from mce_irl_pomdps.utils import from_prism_to_pomdp
import numpy as np
import stormpy


pomdp_r_init = parser_pomdp.PrismModel("evade_5_2.pm", memory_len=1, 
		counter_type= stormpy.pomdp.PomdpMemoryPattern.selective_counter, export=False)
weight = { 'crash_state' : 10, 'finish' : 100}

pomdp_r_init = parser_pomdp.PrismModel("evade_5_2_s.pm", memory_len=1, 
		counter_type= stormpy.pomdp.PomdpMemoryPattern.selective_counter, export=False)
weight = { 'crash_state' : 10, 'finish' : 100}

# pomdp_r_init = parser_pomdp.PrismModel("evade_10_2.pm", memory_len=1, 
# 		counter_type= stormpy.pomdp.PomdpMemoryPattern.selective_counter, export=False)
# weight = { 'crash_state' : 10, 'finish' : 100}

# pomdp_r_init = parser_pomdp.PrismModel("avoid_4_2.pm", memory_len=1, 
# 		counter_type= stormpy.pomdp.PomdpMemoryPattern.selective_counter, export=False)
# weight = { 'crash_state' : 10, 'finish' : 10, 'avoid' : 1}

# pomdp_r_init = parser_pomdp.PrismModel("avoid_4_2_s.pm", memory_len=1, 
# 		counter_type= stormpy.pomdp.PomdpMemoryPattern.selective_counter, export=False)
# weight = { 'crash_state' : 10, 'finish' : 10, 'avoid' : 1}

# pomdp_r_init = parser_pomdp.PrismModel("avoid_7_2.pm", memory_len=1, 
# 		counter_type= stormpy.pomdp.PomdpMemoryPattern.selective_counter, export=False)
# weight = { 'crash_state' : 10, 'finish' : 10, 'avoid' : 1}

# pomdp_r_init = parser_pomdp.PrismModel("intercept_5_2.pm", memory_len=1, 
# 		counter_type= stormpy.pomdp.PomdpMemoryPattern.selective_counter, export=False)
# weight = {'finish' : 20, 'left' : 5}

# pomdp_r_init = parser_pomdp.PrismModel("intercept_5_2_s.pm", memory_len=1, 
# 		counter_type= stormpy.pomdp.PomdpMemoryPattern.selective_counter, export=False)
# weight = {'finish' : 20, 'left' : 5}

# pomdp_r_init = parser_pomdp.PrismModel("intercept_10_2.pm", memory_len=1, 
# 		counter_type= stormpy.pomdp.PomdpMemoryPattern.selective_counter, export=False)
# weight = {'finish' : 20, 'left' : 5}

# pomdp_r_init = parser_pomdp.PrismModel("obstacle_25.pm", memory_len=1, 
# 		counter_type= stormpy.pomdp.PomdpMemoryPattern.fixed_counter, export=False)
# weight = {'finish' : 20, 'crash_state' : 2}

# pomdp_r_init = parser_pomdp.PrismModel("rocks_5_2.pm", memory_len=5, 
# 		counter_type= stormpy.pomdp.PomdpMemoryPattern.selective_counter, export=False)
# weight = {'finish' : 20, 'bad' : 2}

# pomdp_r_init = parser_pomdp.PrismModel("maze_stochastic.pm", memory_len=1, 
# 		counter_type= stormpy.pomdp.PomdpMemoryPattern.selective_counter, export=False)
# weight = { 'poisonous' : 10, 'total_time' : 0.1, 'goal_reach' : 50}

# pomdp_r_init = parser_pomdp.PrismModel("maze_stochastic.pm", memory_len=3, 
# 		counter_type= stormpy.pomdp.PomdpMemoryPattern.selective_counter, export=False)
# weight = { 'poisonous' : 10, 'total_time' : 0.1, 'goal_reach' : 50}

# pomdp_r_init = parser_pomdp.PrismModel("maze_stochastic.pm", memory_len=10, 
# 		counter_type= stormpy.pomdp.PomdpMemoryPattern.selective_counter, export=False)
# weight = { 'poisonous' : 10, 'total_time' : 0.1, 'goal_reach' : 50}

print(pomdp_r_init._path_prism)
print(pomdp_r_init.pomdp)

# Discount factor
discount=0.999

# Trust region parameters
irl_solver.trustRegion = {'red' : lambda x : ((x - 1) / 1.5 + 1),
                          'aug' : lambda x : min(1.5,(x-1)*1.25+1),
                          'lim' : 1+1e-4}

# Options for the solver
options_opt = irl_solver.OptOptions(mu=1e3, mu_spec=1e1, maxiter=100, maxiter_weight=100,
					  graph_epsilon=1e-6, discount=discount, verbose=True, verbose_solver=False)


irlPb_nosi = irl_solver.IRLSolver(pomdp_r_init, init_trust_region=1.01, options=options_opt)
pol_val_grb_nosi = irlPb_nosi.from_reward_to_policy_via_scp(weight)