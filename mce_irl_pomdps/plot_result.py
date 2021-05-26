import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mce_irl_pomdps import parser_pomdp

import json

def parse_data(exp_name, label, max_run=1000, max_iter_per_run=50, 
				seed=201, color='red', linestyle='solid', 
				include_errobar=True, errorbariter=2, errorbarsize=6,
				linesize=3, markertype=None, markersize=None,
				save_result= dict()):
	""" Simulate a given policy and plot the evolution of reward with the
		true weight instead of the learned weight
		:param exp_name : the name/path of the file containing the policy, weights and statistics
						of the learner. Typically, it has the files 
						{exp_name}_pol.json, {exp_name}_weight.json, {exp_name}_stats.json must exists
		:param label : The label associated to this data
		:param max_run : The number of run when simulating the policy
		:param max_iter_per_tun : The maximum number of iteraction with the environment in a run
		:param seed : A seed for reproducibility of the simulation
		:param color : The color of this plot
		:param linestyle : The style of the lines
		:param include_errobar : Specify if the errorbar should be included
		:param errorbariter : Specify the number of points between each bar plots
		:param errorbarsize : Specify the size of the cap of the error bars
		:param linesize : Specify the size of each line in the plot
		:param markertype : If marker are used, it specifies the type of the marker
		:param markersize : specify the size of the markers
		:param save_result : A dictionary containing the results of the simulation
	"""
	# Read the file
	with open(exp_name+'_stats.json', 'r') as f:
		read_stats = json.load(f)
		obs_based = read_stats['obs_based']
		print('###########################  ', label, ': STATS  ########################### \n', 
				json.dumps(read_stats, indent=4, sort_keys=True))
	with open(exp_name+'_pol.json', 'r') as f:
		des_policy = json.load(f)
	with open(exp_name+'_weight.json', 'r') as f:
		weight = json.load(f)
		weight_true = weight['true weight']
		weight_learned = weight['learned weight']
		print('###########################  ', label, ': WEIGHT  ########################### \n', 
				json.dumps(weight, indent=4, sort_keys=True))

	# Optain the parameters of the prism model and build the prism model
	prism_path = read_stats['path_prism']
	spec_formula = read_stats['formula']
	mem_length = read_stats['mem']
	counter_type = parser_pomdp.memoryTypeDict[read_stats['counter_type']]
	# discount = read_stats['discount'] if 'discount' in read_stats else discount
	pomdp_r = parser_pomdp.PrismModel(prism_path, spec_formula, memory_len=mem_length, counter_type=counter_type, export=False)
	print('###########################  ', label, ': POMDP  ########################### \n', 
				pomdp_r.pomdp)
	# Parse the keys of the policy to be integer instead of strings
	exp_pol = parser_pomdp.correct_policy({ int(obs) : { int(a) : val for a, val in actSet.items()} for obs, actSet in des_policy.items()})

	# Simulate the policy
	stat_sim = dict()
	_, rewData = pomdp_r.simulate_policy(exp_pol, weight_true, max_run, max_iter_per_run, seed=seed,
						obs_based=obs_based, stop_at_accepting_state=True, stat=stat_sim)

	# Cumulative sum of the reward for each run
	# arr_rewData = np.cumsum(np.array(rewData)[:, :stat_sim['max_len']], axis=1)
	arr_rewData = np.cumsum(np.array(rewData), axis=1)
	mean_rew = np.mean(arr_rewData, axis = 0)
	# min_rew = np.min(arr_rewData, axis=0)
	# max_rew = np.max(arr_rewData, axis=0)
	std_rew = np.std(arr_rewData, axis=0)
	# axis_x = np.array([i for i in range(mean_rew.shape[0])])
	save_result[exp_name] = (mean_rew, std_rew, label, color, include_errobar, linestyle, linesize, 
								errorbariter, errorbarsize, markertype, markersize)



def plot_data(fig, dict_result):
	""" Plot on the figure the accumulated reward
		:param fig : An instance of plt figure to draw the data on
		:dict_result : THe results of parse_data and contains the parameters 
						required to plot and differentiate between the different results
	"""
	# Plot the data
	ax = plt.gca()
	for elemName, (mean_rew, std_rew, label, color, include_errobar, linestyle, linesize, errorbariter, errorbarsize, markertype, markersize) in dict_result.items():
		len_a = mean_rew.shape[0]
		axis_x = range(len_a)
		ax.plot(axis_x, mean_rew, color=color, label=label, 
					linestyle=linestyle, linewidth=linesize, marker=markertype, markersize=markersize)
		if include_errobar:
			err_val = std_rew
			ax.errorbar(axis_x[0:len_a:errorbariter], mean_rew[0:len_a:errorbariter], 
				yerr = err_val[0:len_a:errorbariter], color=color, fmt='none', capsize=errorbarsize)
