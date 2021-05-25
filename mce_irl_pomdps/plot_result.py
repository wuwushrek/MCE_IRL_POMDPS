import numpy as np

from mce_irl_pomdps import parser_pomdp

import json

def plot_data(exp_name, label, fig, max_run=1000, max_iter_per_run=50, 
				seed=201, color='red', linestyle='solid', 
				include_errobar=True, errorbariter=2, errorbarsize=6,
				linesize=3, markertype=None, markersize=None):
	""" Simulate a given policy and plot the evolution of reward with the
		true weight instead of the learned weight
		:param exp_name : the name/path of the file containing the policy, weights and statistics
						of the learner. Typically, it has the files 
						{exp_name}_pol.json, {exp_name}_weight.json, {exp_name}_stats.json must exists
		:param label : The label associated to this data
		:param fig : An instance of plt figure to draw the data on
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
	min_rew = np.min(arr_rewData, axis=0)
	max_rew = np.max(arr_rewData, axis=0)
	std_rew = np.std(arr_rewData, axis=0)
	axis_x = np.array([i for i in range(mean_rew.shape[0])])

	# Plot the data
	ax = fig.gca()
	ax.plot(axis_x, mean_rew, color=color, label=label, 
				linestyle=linestyle, linewidth=linesize, marker=markertype, markersize=markersize)
	# err_val = np.maximum(min_rew,mean_rew-std_rew) - np.minimum(max_rew,mean_rew+std_rew)
	
	if include_errobar:
		err_val = std_rew
		ax.errorbar(axis_x[0:axis_x.shape[0]:errorbariter], mean_rew[0:axis_x.shape[0]:errorbariter], 
			yerr = err_val[0:axis_x.shape[0]:errorbariter], color=color, fmt='none', capsize=errorbarsize)