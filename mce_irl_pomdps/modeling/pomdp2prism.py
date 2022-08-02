import math
import numpy as np
import json

from itertools import product

from tqdm.auto import tqdm

def load_json_pomdp(filedir):
	""" Load the json file containing the pomdp representations
	"""
	pomdp_dict = json.load(open(filedir,"r"))
	init_state = pomdp_dict.pop('init_state')
	end_state = pomdp_dict.pop('end_state')
	#[TODO Franck] Should remove this when Christian fix it
	# Remove all outgoing transitions from end_state
	pomdp_dict.pop(end_state, None)

	return init_state, end_state, pomdp_dict

def parse_set_states(pomdp_dict):
	""" Extract the number of states and the set of states of 
		the pomdp
	"""
	stateSet = set()
	for s, aDict in pomdp_dict.items():
		stateSet.add(s)
		for a, nextStates in aDict.items():
			for s2, _ in nextStates.items():
				stateSet.add(s2)
	return len(stateSet), stateSet

def parse_set_obs(pomdp_dict):
	""" Extract the number of observation and the set of 
		observations of the pomdp
	"""
	obsSet = set()
	state_wo_obs_set = set()
	for s, aDict in pomdp_dict.items():
		for a, nextStates in aDict.items():
			for s2, val in nextStates.items():
				if isinstance(val, list):
					for k in val[1]:
						obsSet.add(k)
				else:
					# A trick to fix the missing value in the data set
					# [TODO Franck] Change it when the pomdp format is fixed
					nextStates[s2] = [val, { "Grass": 0.3333333333333333,
											 "Gravel": 0.3333333333333333,
											 "Road": 0.3333333333333333
											 }
									 ]
					state_wo_obs_set.add(s2)
	# print(state_wo_obs_set)
	print(len(state_wo_obs_set))
	return len(obsSet), obsSet

def parse_action_set(pomdp_dict):
	""" Obtain the set of actions in the environment
	"""
	action_set = set()
	for s, aDict in pomdp_dict.items():
		action_set.update(aDict.keys())
	return len(action_set), action_set

def build_prism_model_from_json_pomdp(filedir):
	""" This function create the prism model corresponding to the mission defined in Phoenix environment
	"""
	# Load the file
	init_state, end_state, pomdp_dict = load_json_pomdp(filedir)

	# Open the output file
	filename = "{}.prism".format(filedir.split('.json')[0])
	resFile = open(filename, 'w')

	# Parse the POMDP representation of the map
	nstate, set_states = parse_set_states(pomdp_dict)
	states_id = {k : v for k, v in enumerate(set_states)}
	id_states = {v : k for k, v in states_id.items()}
	init_state_id = id_states[init_state]
	end_state_id = id_states[end_state]

	nobs, set_obs = parse_set_obs(pomdp_dict)
	obs_id = {k : v for k, v in enumerate(set_obs)}
	id_obs = {v : k for k, v in obs_id.items()}

	nact, set_act = parse_action_set(pomdp_dict)

	# Create headers for the file
	text_model = '\n// A POMDP representation of a local map in Phoenix environment\n'

	# Create the observable of this environment
	text_model += 'pomdp\n\n\n'

	text_model += 'observables\n'
	text_model += '\tobs\n'
	text_model += 'endobservables\n\n'

	# For each feature, define the observations that characterize the feature
	text_model += '// Specify the observation corresponding to the different features\n'
	road_obs = [k for k, v in obs_id.items() if 'Road' in v]
	text_model += '\nformula road = {};\n'.format('|'.join([ '(obs = {})'.format(obs) for obs in road_obs]))
	gravel_obs = [k for k, v in obs_id.items() if 'Gravel' in v]
	text_model += '\nformula gravel = {};\n'.format('|'.join([ '(obs = {})'.format(obs) for obs in gravel_obs]))
	grass_obs = [k for k, v in obs_id.items() if 'Grass' in v]
	text_model += '\nformula grass = {};\n'.format('|'.join([ '(obs = {})'.format(obs) for obs in grass_obs]))


	# Create the main module
	text_model += '\nmodule phoenix\n\n'

	text_model += '\tstate : [0..{}] init {};\n'.format(nstate, init_state_id)
	text_model += '\tobs : [-1..{}] init -1;\n\n'.format(nobs)

	# Now iterate through the data set and write the transition
	def join_prob(nextprob):
		new_nextprob = {k : nextprob[0]*v for k, v in nextprob[1].items()}
		return new_nextprob

	# Add movement of the agent
	text_model += '\n\t// Moving around the Phoenix environment\n\n'
	for s, aDict in tqdm(pomdp_dict.items()):
		for a, nextStates in aDict.items():
			text_model += '\t[{}] state={} -> {};\n'.format(a, id_states[s], 
				' + '.join( [ '{}: (state\'={}) & (obs\'={})'.format(prob, id_states[snext], id_obs[o]) \
								for snext, nextprob in nextStates.items() for o, prob in join_prob(nextprob).items()
							] 
						) 
			)

	text_model += '\n\t// Sink point --> Loop here\n'
	for act in set_act:
		text_model += '\t[{}]  (state = {}) -> (state\' = {}) & (obs\' = {});\n'.format(act, end_state_id, nstate, nobs)
		text_model += '\t[{}]  (state = {}) -> (state\' = {}) & (obs\' = {});\n'.format(act, nstate, nstate, nobs)

	text_model += '\nendmodule\n\n\n'


	# Define the reward module for the road feature
	text_model += '// Rewards for being on the road\n'
	text_model += 'rewards "road"\n'
	for act in set_act:
		text_model += '\t[{}]  road : 1;\n'.format(act)
	text_model += 'endrewards\n\n'

	# Define the reward module for the grass feature
	text_model += '// Rewards for being on the grass\n'
	text_model += 'rewards "grass"\n'
	for act in set_act:
		text_model += '\t[{}]  grass : -1;\n'.format(act)
	text_model += 'endrewards\n\n'

	# Define the reward module for the gravel feature
	text_model += '// Rewards for being on the gravel\n'
	text_model += 'rewards "gravel"\n'
	for act in set_act:
		text_model += '\t[{}] gravel : -1;\n'.format(act)
	text_model += 'endrewards\n\n'

	# # Define some label
	# text_model += 'label "sink"  = obs={};\n'.format(nobs)
	# # text_model += 'label "road"  = road;\n'
	# # text_model += 'label "grass"  = grass;\n'
	# text_model += 'label "notgravel"  = !gravel;\n'

	# Write the text in the output file
	resFile.write(text_model)
	resFile.close()

	# # Now save the data need for the IRL problem
	# pickFile = open("{}_data.pkl".format(outfile), 'wb')
	# msaves = {'robot_obs' : robot_obs, 'robot_pos' : robot_pos, 'obs_dict' : m_obs_dict, 'state_dict' : dictState,
	# 			'n_row' : n_row, 'n_col' : n_col, 'focus_zone' : focus_zone, 'obs_radius' : obs_radius, 'id_traj' : id_traj, 
	# 			'goalset' : goalset, 'initset' : initset, 'prism_file' : filename}
	# pickle.dump(msaves, pickFile)
	# pickFile.close()


build_prism_model_from_json_pomdp('../../examples/real_robot/pomdps_json/demo_0.json')
# init_state, end_state, json_example = load_json_pomdp('../../examples/real_robot/pomdps_json/demo_0.json')
# ns, list_states = parse_set_states(json_example)
# no, list_obs = parse_set_obs(json_example)
# no, list_obs = parse_set_obs(json_example)

# print(ns, no)
# print(list_obs)

# Why is (0,0) not having an observation?
# We have state where the transition do not provide any observation??
# Initial states 