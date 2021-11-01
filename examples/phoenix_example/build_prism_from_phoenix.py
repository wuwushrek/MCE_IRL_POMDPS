import math
import numpy as np
import pickle

from itertools import product

from tqdm.auto import tqdm

UNKNOWN_FEATURE = 'unknown'

def load_map_file(traj_dir):
	""" Load the Trajectory and phoenix map environment from a pickle file
		:param traj_dir : Name of the pickle file containing the representation of the Phoenix environment
	"""
	# Open the pickle file and load it
	traj_file = open(traj_dir, 'rb')
	traj_data = pickle.load(traj_file)
	print(traj_data.keys())
	print(traj_data['model_info'])
	print(type(traj_data['data']))

	# We only care about the data field of the environment
	traj_data = traj_data['data']

	# Check if the map configuration are valid and obtain the map lwidth and height
	n_row = None
	n_col = None
	n_feat = None

	# Perform some printing to check if the map file make some sense
	for elem in traj_data:
		print('Demo ID      : {}'.format(elem['demonstration_id']))
		print('(height, width)  : ({}, {})'.format(elem['map_height'], elem['map_width'] ))
		print('Features : {}'.format(elem['features']))
		print('Features type    : {}'.format(elem['feature_type']))
		print('Traj length  : {} | start = {}, end = {}'.format(len(elem['trajectory']), elem['trajectory'][0],elem['trajectory'][-1] ))
		assert elem['feature_map_numpy'].shape[:-1] == (elem['map_height'], elem['map_width']), 'Feature map does not match map dimension'
		print('------------------------------------------------')
		assert (n_row is None and n_col is None) or (n_row, n_col) == elem['feature_map_numpy'].shape[:-1], 'Dimensions should match accross demonstrations'
		n_row, n_col = elem['feature_map_numpy'].shape[:-1]
		assert n_feat is None or n_feat == len(elem['features']), 'Number of features should match accross demonstrations'
		n_feat = len(elem['features'])


	# Now handle the robot position and state
	m_robot_state_evol = list()
	for elem in traj_data:
		demo_id = elem['demonstration_id']
		robot_trajectory = elem['trajectory'] 
		m_traj = list()
		for(x,y) in robot_trajectory:
			feat_name = [ feat for (r, feat) in zip(elem['feature_map_numpy'][y,x], elem['features']) if r == 1.0]
			feat_name = feat_name[0] if len(feat_name)>0 else UNKNOWN_FEATURE
			m_traj.append((y,x,feat_name)) # State are in row, column definition
		m_robot_state_evol.append((demo_id, m_traj))
			
	return (n_row, n_col, n_feat), traj_data, m_robot_state_evol

	# # Parse the agent trajectories and save them
	# m_robot = np.zeros((len(traj_data), n_row, n_col))
	# m_robot_row = list() # Save the agent row positions
	# m_robot_col = list() # Save the agent col positions

	# for i, elem in enumerate(traj_data):
	# 	row_val, col_val = list(), list()

	# 	# x,y are given such that x is the index on the column axis and y the index on the row axis
	# 	for (x,y) in elem['trajectory']:
	# 		m_robot[i, y, x] = 1
	# 		row_val.append(y)
	# 		col_val.append(x)

	# 	# Append the robot row and column to the resuls
	# 	m_robot_row.append(row_val)
	# 	m_robot_col.append(col_val)

	# return (n_row, n_col, n_feat), traj_data, (m_robot, m_robot_row, m_robot_col)


def build_map(m_data, n_row, n_col, south_west_center=(0,0), id_traj=[0], eps_bias=10):
	""" Take a set of map from the Phoenix environment and construct a dictionary such that
		for each grid cell (i,j), the value of the dictionary is a probability over the possible
		features that might be present at the cell (i,j)
		:param m_data : SPpecify the set of slam roadmap from phoenix environment --> map of the same zone (same shape)
						Typically this parameter is one of the output of load_map_file
		:param n_row : The number of row of interest in each grid of m_data
		:param n_col : The number of column of interest in each grid of m_data
		:param south_west_center : Specify the south west point of the grid of interest (in term of number row, number of col)
		:param id_traj : Specify if the dictionary should be built from a single map or all the provided map
		:param eps_bias : Non-existant feature for a state count for (1/eps_bias) * the probability of the feature with lowest occurence
	"""
	# Store the subset of trajectories we are interested in
	traj_data = list()
	for elem in m_data:
		if elem['demonstration_id'] in id_traj:
			traj_data.append(elem.copy())

	# Store the desired dictionary
	final_map = dict() # save the modified map
	m_unique_feat = set()

	# Iterate through the selected map
	for elem in traj_data:
		# Get the sub map given by the south west center and the specified n_row and n_col
		fmap_np = elem['feature_map_numpy'][south_west_center[0]:(south_west_center[0]+n_row), south_west_center[1]:(south_west_center[1]+n_col)]
		
		# The set of unique feature in the map
		mUniqueValue = set()

		# Iterate through the reduced map
		for i in tqdm(range(fmap_np.shape[0])):
			for j in tqdm(range(fmap_np.shape[1]),leave=False):
				# Get the current feature of this cell
				feat_name = [ feat for (r, feat) in zip(fmap_np[i,j], elem['features']) if r == 1.0]
				feat_name = feat_name[0] if len(feat_name)>0 else UNKNOWN_FEATURE
				
				# Add the obtained feature in the set of unique features
				mUniqueValue.add(feat_name)
				m_unique_feat.add(feat_name)

				# In case this cell is not registered yet
				if (i,j) not in final_map:
					final_map[(i,j)] = dict()

				# Count the number of time this feature has been seen in this cell
				final_map[(i,j)][feat_name] = final_map[(i,j)].get(feat_name, 0) + 1

				# # If the feature is not unknown then remove unknown from the possible feature
				# if feat_name != UNKNOWN_FEATURE:
				# 	final_map[(i,j)].pop(UNKNOWN_FEATURE, None)
				# # If the feature is unknown but the cell contains a known value then remove unknown
				# if feat_name == UNKNOWN_FEATURE and len(final_map[(i,j)]) > 1:
				# 	final_map[(i,j)].pop(UNKNOWN_FEATURE, None)

		# Print the unique feature in this map
		print('[Map {}] Unique elements : {}'.format(elem['demonstration_id'], mUniqueValue))

	# Quick hack to have non existant state -> Add low probabilities for non existant position, feature combination
	if eps_bias is not None and eps_bias > 0:
		for (i,j), feat in final_map.items():
			min_val_feat_scaled = np.min(np.array([v for _,v in feat.items()])) / eps_bias
			for feat_val in m_unique_feat:
				if feat_val not in feat:
					feat[feat_val] = min_val_feat_scaled

	# Return the result
	return final_map


def build_pomdp(n_row, n_col, dict_feature, obs_radius=4):
	""" Build the pomdp associated with the mdp build from the different maps
		:param n_row : The number of row of interest in each grid of m_data
		:param n_col : The number of column of interest in each grid of m_data
		:param dict_feature : A dictionary providing the mdp (for each cell the probabilities of finding each feature)
		:param obs_radius : Provide the partially part in the model --> Radius of observation of the agent
	"""
	# Define the action set
	actionSet = {'stay' : (0,0), 'north' : (1,0), 'south' : (-1,0), 'east' : (0,1), 'west' : (0,-1),
				'north_east' : (1,1), 'north_west' : (1,-1), 'south_east' : (-1,1), 'south_west' : (-1,-1)}

	# Move function returns the index when taking north, south, east, and west
	move = lambda i, j, i0, j0 : ( min(max(i+i0, 0), n_row-1), max(min(j+j0, n_col-1), 0) )

	# Transition matrix function
	trans_dict = {}

	# Set of states and its unique identifer
	state_set = set()
	num_transition = 0

	# Iterate over the feature of each cell to build the underlying MDP
	for (i,j), featv in tqdm(dict_feature.items(), total=len(dict_feature)):
		# Total number of features for this cell
		total_count = sum([ v for x,v in featv.items()])
		# Iterate through the feature of this cell
		for feat_value, feat_count in tqdm(featv.items(), total=len(featv), leave=False):
			# Define the dictionary for a specific feat_vaue
			trans_dict[(i, j, feat_value)] = {act : dict() for act,_ in actionSet.items()}
			state_set.add((i, j, feat_value))
			# (i,j,feat_value) is the current state --> Go through the action set
			for act, act_repr in actionSet.items():
				# Next state if taking any of the actions in the action set
				next_ij = move(i, j, *act_repr)
				# print(i, j, act, next_ij)
				# Check the possible features for the next state
				next_ij_feat = dict_feature[next_ij]
				# COunt the total number of features in the next state
				total_count_next = sum([ v for x,v in next_ij_feat.items()])
				# Build the transition to the next state(*next_ij, next_ij_feat)
				for next_feat_val, next_feat_prob in next_ij_feat.items():
					trans_dict[(i, j, feat_value)][act][(*next_ij, next_feat_val)] = float(next_feat_prob) / total_count_next
					state_set.add((*next_ij, next_feat_val))
					num_transition += 1
				# trans_dict[(i, j, feat_value)] = { act : (*move(i,j, *actionSet[act]), dict_feature[move(i,j,*actionSet[act])]) for act in actionSet }

	# Build the observation function/map
	obsSet = set() # Store the unique observation id (all the features in a fixed radius )
	obsFullDict = dict() # Store the full observation dictionary
	unique_obs = 0 # Counter of unique observation functions

	# Create a unique identifier
	id_obs = dict()
	id_obs_reverse = dict()

	# This observation function specifies that the agent knows his position with a fixed uncertaincy around his actual position
	for i in tqdm(range(0, n_row, obs_radius)):
		for j in tqdm(range(0, n_col, obs_radius), leave=False):
			# Get the element at the top left corner
			corner_left_up = (i,j)
			# Get the element at the lower right corner
			corner_right_down = move(i, j, obs_radius, obs_radius)
			# Iterate through all the state around the fixed radius of (i,j)
			for k in range(corner_left_up[0], corner_right_down[0]+1):
				for l in range(corner_left_up[1], corner_right_down[1]+1):
					for feat_val, occ_feat in dict_feature[(k,l)].items():
						# If the observation (zone, current feature) has not already been defined
						if (i,j, feat_val) not in obsSet:
							# Encode a unique indentifier for the current zone around (i,j) and the sensing of the feature at current position
							id_obs[unique_obs] = (i,j, feat_val)
							# Specify a way to reciver the identifier fron the observation (i,j,featkl)
							id_obs_reverse[(i,j, feat_val)] = unique_obs
							# Enforce that the current state corresponds to the obtained observation
							obsFullDict[(k, l, feat_val)] = unique_obs
							# Add this observation in the set of observations
							obsSet.add((i,j, feat_val))
							# Increment the number of unique observation
							unique_obs += 1
						else:
							# If the observation is already defined -> Just copy it here
							obsFullDict[(k, l, feat_val)] = id_obs_reverse[(i,j, feat_val)]

	
	print('Number of states : {}'.format(len(state_set)))
	print('Number of transitions : {}'.format(num_transition))
	print('Number of observations : {} | {}'.format(unique_obs, len(id_obs)))

	return (trans_dict, state_set), (obsFullDict, id_obs), (actionSet, move)


def build_state_trajectories(obsFullDict, traj_robot, id_traj, actionSet, focus_zone):
	""" Build the pair of observation action taken by each robot in each trajectories
		:param obsFullDict : A dictionary specifying the observation identifier for each states
		:param traj_robot : The set of trajectories of the robot
		:param id_traj : THe index of the trajectories we are interested into
		:param actionSet : The dictionary specifying the actions and how they translate in the gridworld
		:param focus_zone : The area of interest in the gridworld
	"""
	pos_evol = list()
	obs_evol = list()

	for (indx, robot_traj) in traj_robot:
		if indx not in id_traj:
			continue
		curr_obs_evol = list()
		curr_pos_evol = list()

		# Get the starting point in the focus_zone
		start_index = -1
		for it, (i,j,featv) in enumerate(robot_traj):
			# Check if the current position is inside the boundaries
			if i >= focus_zone[0] and j >= focus_zone[1] and i < focus_zone[2] and j < focus_zone[3]:
				start_index = it
				break
		assert start_index >= 0

		# Get the ending point in the focus zone
		end_index = start_index
		for it, (i,j,featv) in enumerate(robot_traj):
			if it <= start_index:
				continue
			# Check if the current position is outside the boundaries
			if not(i >= focus_zone[0] and j >= focus_zone[1] and i < focus_zone[2] and j < focus_zone[3]):
				end_index = it
				break
			# If it is the last iteration and all the points are inside the boundaries save the length as the last index
			if it == len(robot_traj)-1:
				end_index = len(robot_traj)
		assert end_index - start_index > 1
		# print(end_index, start_index)
		# Get the piece of trajectory of interest
		robot_traj = robot_traj[start_index:end_index]
		# Iterate to get the observation and the action taken at each time step
		for (i,j,featv), (next_i, next_j, next_featv) in zip(robot_traj[:-1],robot_traj[1:]):
			if obsFullDict is None:
				curr_pos_evol.append( (i-focus_zone[0],j-focus_zone[1], featv) )
				continue
			curr_obs = (i-focus_zone[0], j-focus_zone[1], featv) # obsFullDict[(i-focus_zone[0], j-focus_zone[1], featv)]
			diff_pos = (next_i-i, next_j-j)
			curr_act = None
			for act, val in actionSet.items():
				if val == diff_pos:
					curr_act = act
					break

			assert curr_act is not None
			curr_obs_evol.append((curr_obs, curr_act))
			curr_pos_evol.append((i-focus_zone[0],j-focus_zone[1], featv))
		obs_evol.append(curr_obs_evol)
		pos_evol.append(curr_pos_evol)

	return obs_evol, pos_evol


def build_prism_model(pomdp_repr, extra_args, actionSet, outfile = 'phoenix'):
	""" This function create the prism model corresponding to the mission defined in Phoenix environment
	"""
	filename = "{}.prism".format(outfile)
	resFile = open(filename, 'w')

	# Parse the extra arguments of the function
	n_row, n_col, focus_zone, obs_radius, id_traj, goalset, initset, robot_obs, robot_pos = extra_args

	# Make sure there is no repeat in the goal set
	goalset = set(goalset)

	# Parse the POMDP representation of the map
	final_map, trans_dict, state_set, m_obs_dict, id_obs = pomdp_repr

	# Create headers for the file
	text_model = '\n// A POMDP representation of a local map in Phoenix environment\n'
	text_model += '// The local map is of size ({} , {}) and start at (low_row, low_col, high_row, high_col) = ({} , {}, {}, {})\n'.format(n_row, n_col, focus_zone[0], focus_zone[1], focus_zone[2], focus_zone[3])
	text_model += '// Observation radius : {} | Trajectories considered : {}\n'.format(obs_radius, id_traj)
	text_model += '// The Goal set of interest : {}\n'.format(goalset)
	text_model += '// The init set of interest : {}\n\n'.format(initset)

	# Create the observable of this environment
	text_model += 'pomdp\n\n\n'

	text_model += '// Can observe its position with an uncertaincy radius r = {}\n'.format(obs_radius)
	text_model += 'observables\n'
	text_model += '\tobs\n'
	text_model += 'endobservables\n\n'

	# Dictionary to save the relation between the row, col, feat and an unique state identifier
	dictState = {(i,j,featv) : k for k, (i,j,featv) in enumerate(m_obs_dict.keys())}
	nstate = len(dictState)
	nobs = len(id_obs)

	# Add a description of the observables
	text_model += '// The observation to state translation\n'
	for (i,j,featv), obs_id in m_obs_dict.items():
		text_model += '// obs = {} ---> state = {} | row = {}, col = {}, feat = {}\n'.format(obs_id, dictState[(i,j,featv)], i, j, featv)


	# Define formula for the goal set
	text_model += '\n\nformula done = {};\n'.format(' | '.join(['(state = {})'.format(dictState[(row, col, feat)]) for (row, col, feat) in goalset] ) )
	text_model += 'observable "amdone" = done;\n\n'

	# For each feature, define the observations that characterize the feature
	text_model += '// Specify the observation corresponding to the different features\n'
	road_obs = set([ obs for (i,j,featv), obs in m_obs_dict.items() if featv == 'road'])
	if len(road_obs) > 0:
		text_model += '\nformula road = {};\n'.format('|'.join([ '(obs = {})'.format(obs) for obs in road_obs]))
	gravel_obs = set([ obs for (i,j,featv), obs in m_obs_dict.items() if featv == 'gravel'])
	if len(gravel_obs) > 0:
		text_model += '\nformula gravel = {};\n'.format('|'.join([ '(obs = {})'.format(obs) for obs in gravel_obs]))
	grass_obs = set([ obs for (i,j,featv), obs in m_obs_dict.items() if featv == 'grass'])
	if len(grass_obs) > 0:
		text_model += '\nformula grass = {};\n'.format('|'.join([ '(obs = {})'.format(obs) for obs in grass_obs]))


	# Create the main module
	text_model += '\nmodule phoenix\n\n'

	text_model += '\tstate : [-1..{}];\n'.format(nstate+1)
	text_model += '\tobs : [-1..{}];\n\n'.format(nobs+1)

	text_model += '\t// Initialization\n'
	text_model += '\t[] state=-1 -> {};\n'.format(' \n\t\t\t\t\t+ '.join([ '1/{} : (state\'= {}) & (obs\'= {})'.format(len(initset), dictState[(i,j,featv)], m_obs_dict[(i,j,featv)]) for (i,j,featv) in initset ] ) )

	# Add movement of the agent
	text_model += '\n\t// Moving around the Phoenix environment\n\n'
	for (i,j, featv), trans_info in trans_dict.items():
		if (i,j,featv) in goalset:
			continue
		for act, next_state_dist in trans_info.items():
			text_model += '\t[{}] state={} -> {};\n'.format(act, dictState[(i,j,featv)], ' + '.join( ['{}: (state\'={}) & (obs\'={})'.format(prob, dictState[(next_i, next_j, next_featv)], m_obs_dict[(next_i, next_j, next_featv)]) for (next_i, next_j, next_featv), prob in next_state_dist.items()] ) )

	# Add the endless loop at the end goal point
	text_model += '\n\t // Ensure that end goal points reach a sink point\n'
	for (i,j,featv) in goalset:
		for act in actionSet:
			text_model += '\t[{}] state={} -> (state\' = {}) & (obs\' = {});\n'.format(act, dictState[(i,j,featv)], nstate, nobs)
	# for act in actionSet:
	# 	text_model += '\t[{}] done -> (state\' = {}) & (obs\' = {});\n'.format(act, nstate, nobs)

	text_model += '\n\t// Sink point --> Loop here\n'
	for act in actionSet:
		text_model += '\t[{}]  (state = {}) -> (state\' = {}) & (obs\' = {});\n'.format(act, nstate, nstate, nobs)

	text_model += '\nendmodule\n\n\n'

	# Define the reward module for the goal set
	text_model += '// Rewards for reaching one of the goal set\n'
	text_model += 'rewards "goal"\n'
	for act in actionSet:
		text_model += '\t[{}]  done : 1;\n'.format(act)
	text_model += 'endrewards\n\n'

	# Define the reward module for the road feature
	text_model += '// Rewards for being on the road\n'
	text_model += 'rewards "road"\n'
	for act in actionSet:
		text_model += '\t[{}]  !done & road : 1;\n'.format(act)
	text_model += 'endrewards\n\n'

	# Define the reward module for the grass feature
	text_model += '// Rewards for being on the grass\n'
	text_model += 'rewards "grass"\n'
	for act in actionSet:
		text_model += '\t[{}]  !done & grass : 1;\n'.format(act)
	text_model += 'endrewards\n\n'

	# Define the reward module for the gravel feature
	text_model += '// Rewards for being on the gravel\n'
	text_model += 'rewards "gravel"\n'
	for act in actionSet:
		text_model += '\t[{}]  !done & gravel : -1;\n'.format(act)
	text_model += 'endrewards\n\n'

	# Define the reward module for elapsed time
	text_model += '// Rewards for time elapsed\n'
	text_model += 'rewards "time"\n'
	for act in actionSet:
		text_model += '\t[{}]  !done & !(obs = {}) : -1;\n'.format(act, nobs)
	text_model += 'endrewards\n\n'

	# Define some label
	text_model += 'label "goal"  = obs={};\n'.format(nobs)
	text_model += 'label "road"  = road;\n'
	text_model += 'label "grass"  = grass;\n'
	text_model += 'label "gravel"  = gravel;\n'

	# Write the text in the output file
	resFile.write(text_model)
	resFile.close()

	# Now save the data need for the IRL problem
	pickFile = open("{}_data.pkl".format(outfile), 'wb')
	msaves = {'robot_obs' : robot_obs, 'robot_pos' : robot_pos, 'obs_dict' : m_obs_dict, 'state_dict' : dictState,
				'n_row' : n_row, 'n_col' : n_col, 'focus_zone' : focus_zone, 'obs_radius' : obs_radius, 'id_traj' : id_traj, 
				'goalset' : goalset, 'initset' : initset, 'prism_file' : filename}
	pickle.dump(msaves, pickFile)
	pickFile.close()



traj_dir = 'aaai_experiment_09-17.pickle'
# (n_row, n_col, n_feat), traj_data, (m_robot, m_robot_row, m_robot_col) = load_map_file(traj_dir)
(n_row, n_col, n_feat), traj_data, m_robot_state_evol = load_map_file(traj_dir)

# Origin point of the sub-grid
# south_west_center = (0, 0) # Row first then column
south_west_center = (80, 100) # Row first then column

# Define the uncertain observation
obs_radius = 4

# Specify the trajectory of interest
id_traj = [0,1,2,3,4,5,6,7,8,9]
# id_traj = [0]
eps_bias = None

# Number of row and column
# n_row_focus = n_row
# n_col_focus = n_col
n_row_focus = 40
n_col_focus = 80
focus_zone = (south_west_center[0], south_west_center[1], south_west_center[0]+n_row_focus, south_west_center[1]+n_col_focus)
# focus_init = (0, 0, n_row, n_col)
focus_init_row = 100-south_west_center[0]
focus_init_col = 100-south_west_center[1]
focus_init_nrow = 5
focus_init_ncol = 5
focus_init = (focus_init_row, focus_init_col, focus_init_row+focus_init_nrow, focus_init_col+focus_init_ncol)

# Build the feature distribution on the focused map
final_map = build_map(traj_data, n_row_focus, n_col_focus, south_west_center=south_west_center, id_traj=id_traj, eps_bias=eps_bias)
# print(final_map[(20,27)])
# Compute the transition matrix and the set of states and observation from the model
m_obs_dict = None
m_action_set = None
(trans_dict, state_set), (m_obs_dict, id_obs), (m_action_set, move) = build_pomdp(n_row_focus, n_col_focus, final_map, obs_radius=obs_radius)

robot_obs_evol, robot_pos_evol = build_state_trajectories(m_obs_dict, m_robot_state_evol, id_traj, m_action_set, focus_zone)
goal_set = [ m_traj[-1] for m_traj in robot_pos_evol] 
init_set = [ (i,j,featv) for (i, j, featv) in m_obs_dict.keys() if (i>=focus_init[0] and j>=focus_init[1] and i< focus_init[2] and j < focus_init[3])]
# init_set = []
# print(robot_obs_evol)
# print(robot_pos_evol)

# # Build the POMDP file
build_prism_model((final_map, trans_dict, state_set, m_obs_dict, id_obs), 
                    (n_row_focus, n_col_focus, focus_zone, obs_radius, id_traj, goal_set, init_set, robot_obs_evol, robot_pos_evol), 
                    m_action_set, 'phoenix_scen1_r4uncert_40x80_100x100')
