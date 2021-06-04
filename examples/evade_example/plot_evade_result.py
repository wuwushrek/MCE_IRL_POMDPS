import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import multiprocessing
from multiprocessing import Pool, cpu_count, Manager
import tikzplotlib

from mce_irl_pomdps.plot_result import parse_data, plot_data



def compare_perf_irl(output_file ='maze_demo_from_mdp_irl', demo_from_mdp=True, show=True):
	""" Compare the different learnt policies
	"""
	# Seed for reproductibility
	seed = 201
	# NUmber of run and number of interactions in each run
	max_run = 3000
	max_iter_per_run = 300
	# Data information and how to they should be vizualize
	labelMDP = r'$\mathrm{Opt. \ policy \  on \ MDP}$'
	labelPOMDP = r'$\mathrm{Opt. \ FSC \  on \ POMDP, M=1}$'
	if demo_from_mdp:
		file_names = ['evade_mdp_fwd', 'evade_mem1_fwd', 'evade_mem1_trajsize10mdp_irl', 'evade_mem1_trajsize10mdp_irl_si']
		labelMDP = r'$\mathrm{Opt. \ policy \  on \ MDP, \ N=10}$'
	else:
		file_names = ['evade_mdp_fwd', 'evade_mem1_fwd',
						'evade_mem1_trajsize10pomdp_irl', 'evade_mem1_trajsize10pomdp_irl_si']
		labelPOMDP = r'$\mathrm{Opt. \ FSC \  on \ POMDP, \ M=1, N=10}$'
	labels = [	labelMDP, 
				labelPOMDP,
				r'$\mathrm{Learned \ policy, \ |n|=1}$',
				r'$\mathrm{Guided \ policy, \ |n|=1}$',]
	color_values = ['blue', 'red', 'green', 'olive']
	linestyles = ['solid', 'solid', 'solid', 'solid']
	xaxislabel = r'$\mathrm{Time \ steps}$'
	yaxislabel = r'$\mathrm{Mean \ accumulated \ reward}$'
	# Plots parameters
	linesize = 2
	markertype=None
	markersize=None
	# Error bar parameters
	include_errobar = False
	errorbarsize = 3*linesize
	errorbariter = 5 # <---- To modify if more bars are need

	# The dictionary to store the results
	manager = Manager()
	sim_result = manager.dict()

	# Regroup the functions arguments
	args_set = list()
	for fname, label, color, linestyle in zip(file_names, labels, color_values, linestyles):
		args_set.append((fname, label, max_run, max_iter_per_run, 
				seed, color, linestyle, include_errobar, errorbariter, 
				errorbarsize, linesize, markertype, markersize, sim_result))
		# plot_data(fname, label, fig, max_run, max_iter_per_run, 
		# 		seed, color, linestyle, include_errobar, errorbariter, 
		# 		errorbarsize, linesize, markertype, markersize)

	# Parrallelize the plotting
	with Pool(multiprocessing.cpu_count()) as pool:
		pool.starmap(parse_data, args_set)

	# Plot the results
	# THe figure to draw on
	fig = plt.figure()
	plot_data(fig, sim_result)
	plt.xlabel(xaxislabel)
	plt.ylabel(yaxislabel)
	plt.grid(True)
	plt.legend(ncol=3, bbox_to_anchor=(0,1), loc='lower left', columnspacing=1.0)
	
	tikzplotlib.clean_figure(fig=fig)
	tikzplotlib.save(output_file+".tex", figure=fig)
	fig.savefig(output_file+'.svg', dpi=300, transparent=True)

	if show:
		plt.show()

if __name__ == '__main__':
	compare_perf_irl(output_file ='evade_demo_from_mdp_irl', demo_from_mdp=True, show=False)
	compare_perf_irl(output_file ='evade_demo_from_pomdp_irl', demo_from_mdp=False, show=True)