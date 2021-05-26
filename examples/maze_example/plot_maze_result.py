import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import multiprocessing
from multiprocessing import Pool, cpu_count, Manager
import tikzplotlib

from mce_irl_pomdps.plot_result import parse_data, plot_data


def compare_perf_forward(output_file = 'maze_memory_impact_fwd', show=True):
	""" Show solution of the forward problem under different memory
	"""
	# Seed for reproductibility
	seed = 201
	# NUmber of run and number of interactions in each run
	max_run = 2000
	max_iter_per_run = 100
	# Data information and how to they should be vizualize
	file_names = ['maze_mdp_fwd', 'maze_mem1_fwd', 'maze_mem5_fwd', 'maze_mem10_fwd', 'maze_mem15_fwd' ]
	labels = [r'$\mathrm{Opt. \ policy \  on \ MDP}$', r'$\mathrm{Opt. \ FSC \ on  \ POMDP, |n|=1}$',
				r'$\mathrm{Opt. \ FSC \ on  \ POMDP, |n|=5}$', r'$\mathrm{Opt. \ FSC \ on  \ POMDP, |n|=10}$',
				r'$\mathrm{Opt. \ FSC \ on  \ POMDP, |n|=15}$']
	color_values = ['blue', 'gray', 'indigo', 'green', 'red']
	xaxislabel = r'$\mathrm{Time \ steps}$'
	yaxislabel = r'$\mathrm{Mean \ accumulated \ reward}$'
	# Plots parameters
	linestyle = 'solid'
	linesize = 2
	markertype=None
	markersize=None
	# Error bar parameters
	include_errobar = True
	errorbarsize = 3*linesize
	errorbariter = 5 # <---- To modify if more bars are need

	# The dictionary to store the results
	manager = Manager()
	sim_result = manager.dict()

	# Regroup the functions arguments
	args_set = list()
	for fname, label, color in zip(file_names, labels, color_values):
		args_set.append((fname, label, max_run, max_iter_per_run, 
				seed, color, linestyle, include_errobar, errorbariter, 
				errorbarsize, linesize, markertype, markersize, sim_result))
		# plot_data(fname, label, fig, max_run, max_iter_per_run, 
		# 		seed, color, linestyle, include_errobar, errorbariter, 
		# 		errorbarsize, linesize, markertype, markersize)
	# print(sim_result)
	# # # Parrallelize the plotting
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

def compare_perf_forward(output_file = 'maze_demo_from_mdp_irl', show=True):
	""" Compare the different learnt policies
	"""
	output_file = 'maze_memory_impact_fwd'
	# Seed for reproductibility
	seed = 201
	# NUmber of run and number of interactions in each run
	max_run = 2000
	max_iter_per_run = 100
	# Data information and how to they should be vizualize
	file_names = ['maze_mdp_fwd', 'maze_mem15_fwd', 'maze_mem1_trajsize5mdp_irl', 'maze_mem1_trajsize100mdp_irl',
					'maze_mem10_trajsize5mdp_irl', 'maze_mem10_trajsize100mdp_irl',
					'maze_mem1_trajsize5mdp_irl_si', 'maze_mem1_trajsize100mdp_irl_si',
					'maze_mem10_trajsize5mdp_irl_si', 'maze_mem10_trajsize100mdp_irl_si' ]
	labels = [r'$\mathrm{Opt. \ policy \  on \ MDP}$', r'$\mathrm{Opt. \ FSC \ on  \ POMDP, |n|=15}$',
				r'$\mathrm{Learned policy, \ |n|=1,N=5}$', r'$N=100$',
				r'$\mathrm{Learned policy, \ |n|=10,N=5}$', r'$N=100$',
				r'$\mathrm{Guided policy, \ |n|=1,N=5}$', r'$N=100$',
				r'$\mathrm{Guided policy, \ |n|=10,N=5}$', r'$N=100$']
	color_values = ['blue', 'red', 'gray', 'gray', 'green', 'green', 'indigo', 'indigo', 'darkorange', 'darkorange']
	linestyles = ['solid', 'solid', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid']
	xaxislabel = r'$\mathrm{Time \ steps}$'
	yaxislabel = r'$\mathrm{Mean \ accumulated \ reward}$'
	# Plots parameters
	linesize = 2
	markertype=None
	markersize=None
	# Error bar parameters
	include_errobar = True
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
	# print(sim_result)
	# # # Parrallelize the plotting
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
	compare_perf_forward()