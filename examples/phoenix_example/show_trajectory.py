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
		# assert elem['feature_map_numpy'].shape[:-1] == (elem['map_height'], elem['map_width']), 'Feature map does not match map dimension'
		print('------------------------------------------------')
		if n_row is None or n_col is None:
			n_row, n_col = elem['map_height'], elem['map_width']
		else:
			n_row = elem['map_height'] if n_row > elem['map_height'] else n_row
			n_col = elem['map_width'] if n_col > elem['map_width'] else n_col
		# assert (n_row is None and n_col is None) or (n_row, n_col) == elem['feature_map_numpy'].shape[:-1], 'Dimensions should match accross demonstrations'
		# n_row, n_col = elem['feature_map_numpy'].shape[:-1]
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


def build_map(m_data, n_row, n_col, south_west_center=(0,0), id_traj=[0], eps_bias=None):
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

				# If the feature is not unknown then remove unknown from the possible feature
				if feat_name != UNKNOWN_FEATURE:
					final_map[(i,j)].pop(UNKNOWN_FEATURE, None)
				# If the feature is unknown but the cell contains a known value then remove unknown
				if feat_name == UNKNOWN_FEATURE and len(final_map[(i,j)]) > 1:
					final_map[(i,j)].pop(UNKNOWN_FEATURE, None)

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


traj_dir = 'aaai_experiment_final.pickle'
# (n_row, n_col, n_feat), traj_data, (m_robot, m_robot_row, m_robot_col) = load_map_file(traj_dir)
(n_row, n_col, n_feat), traj_data, m_robot_state_evol = load_map_file(traj_dir)

# Origin point of the sub-grid
# south_west_center = (0, 0) # Row first then column
south_west_center = (25, 25) # Row first then column

# Define the uncertain observation
obs_radius = 4

# Specify the trajectory of interest
# id_traj = [0,1,2,3,4,5,6,7,8,9]
id_traj = [0,1,2,3,4,5,6,7,8,9]
eps_bias = None

# Number of row and column
# n_row_focus = n_row
# n_col_focus = n_col
n_row_focus = 35
n_col_focus = 60
focus_zone = (south_west_center[0], south_west_center[1], south_west_center[0]+n_row_focus, south_west_center[1]+n_col_focus)
# focus_init = (0, 0, 4, 4)
focus_init_row = 15
focus_init_col = 12
focus_init_nrow = 4
focus_init_ncol = 4
focus_init = (focus_init_row, focus_init_col, focus_init_row+focus_init_nrow, focus_init_col+focus_init_ncol)

# Build the feature distribution on the focused map
final_map = build_map(traj_data, n_row_focus, n_col_focus, south_west_center=south_west_center, id_traj=id_traj, eps_bias=eps_bias)
# print(final_map[(20,27)])
# Compute the transition matrix and the set of states and observation from the model
m_obs_dict = None
m_action_set = None
(trans_dict, state_set), (m_obs_dict, id_obs), (m_action_set, move) = build_pomdp(n_row_focus, n_col_focus, final_map, obs_radius=obs_radius)
# obs_states = dict()
# for (i,j, featv), obs in m_obs_dict.items():
# 	obs_states[(i,j, featv)] = set()
# 	for (i1,j1, featv1), obs1 in m_obs_dict.items():
# 		if (i1,j1, featv1) == (i,j, featv) or (i1,j1, featv1) in obs_states[(i,j, featv)]:
# 			continue
# 		if obs1 == obs:
# 			obs_states[(i,j, featv)].add((i1,j1, featv1))

robot_obs_evol, robot_pos_evol = build_state_trajectories(m_obs_dict, m_robot_state_evol, id_traj, m_action_set, focus_zone)
goal_set = [ m_traj[-1] for m_traj in robot_pos_evol] 
init_set = [ (i,j,featv) for (i, j, featv) in m_obs_dict.keys() if (i>=focus_init[0] and j>=focus_init[1] and i< focus_init[2] and j < focus_init[3])]
# init_set = []
# print(robot_obs_evol)
# print(robot_pos_evol)


####################################################################################################
############################################## Do SOME PLOTTING ####################################

from matplotlib import pyplot as plt, cm
import matplotlib.patches as patches

def arrowplot(axes, x, y, nArrs=30, mutateSize=10, color='gray', markerStyle='o'): 
	'''arrowplot : plots arrows along a path on a set of axes
		axes   :  the axes the path will be plotted on
		x      :  list of x coordinates of points defining path
		y      :  list of y coordinates of points defining path
		nArrs  :  Number of arrows that will be drawn along the path
		mutateSize :  Size parameter for arrows
		color  :  color of the edge and face of the arrow head
		markerStyle : Symbol
	
		Bugs: If a path is straight vertical, the matplotlab FanceArrowPatch bombs out.
		  My kludge is to test for a vertical path, and perturb the second x value
		  by 0.1 pixel. The original x & y arrays are not changed
	
		MHuster 2016, based on code by 
	'''
	# recast the data into numpy arrays
	x = np.array(x, dtype='f')
	y = np.array(y, dtype='f')
	nPts = len(x)

	# Plot the points first to set up the display coordinates
	# axes.plot(x,y, markerStyle, ms=2, color=color)

	# get inverse coord transform
	inv = axes.transData.inverted()

	# transform x & y into display coordinates
	# Variable with a 'D' at the end are in display coordinates
	xyDisp = np.array(axes.transData.transform(list(zip(x,y))))
	xD = xyDisp[:,0]
	yD = xyDisp[:,1]

	# drD is the distance spanned between pairs of points
	# in display coordinates
	dxD = xD[1:] - xD[:-1]
	dyD = yD[1:] - yD[:-1]
	drD = np.sqrt(dxD**2 + dyD**2)

	# Compensating for matplotlib bug
	dxD[np.where(dxD==0.0)] = 0.1

	# rtotS is the total path length
	rtotD = np.sum(drD)

	# based on nArrs, set the nominal arrow spacing
	arrSpaceD = rtotD / nArrs
	m_pacthes = []

	# Loop over the path segments
	iSeg = 0
	while iSeg < nPts - 1:
		# Figure out how many arrows in this segment.
		# Plot at least one.
		nArrSeg = max(1, int(drD[iSeg] / arrSpaceD + 0.5))
		xArr = (dxD[iSeg]) / nArrSeg # x size of each arrow
		segSlope = dyD[iSeg] / dxD[iSeg]
		# Get display coordinates of first arrow in segment
		xBeg = xD[iSeg]
		xEnd = xBeg + xArr
		yBeg = yD[iSeg]
		yEnd = yBeg + segSlope * xArr
		# Now loop over the arrows in this segment
		for iArr in range(nArrSeg):
			# Transform the oints back to data coordinates
			xyData = inv.transform(((xBeg, yBeg),(xEnd,yEnd)))
			# Use a patch to draw the arrow
			# I draw the arrows with an alpha of 0.5
			p = patches.FancyArrowPatch( 
				xyData[0], xyData[1], 
				arrowstyle='simple',
				mutation_scale=mutateSize,
				color=color, alpha=0.5)
			m_pacthes.append(axes.add_patch(p))
			# Increment to the next arrow
			xBeg = xEnd
			xEnd += xArr
			yBeg = yEnd
			yEnd += segSlope * xArr
		# Increment segment number
		iSeg += 1
	return m_pacthes


# Plot the obtained focused final map
# Visualize the maps and the robots trajectories
color_dist = {UNKNOWN_FEATURE : np.array([0, 0, 0]), 'grass' : np.array([11,102,35])/255.0,  'gravel' : np.array([1.0, 0, 0]), 'road' : np.array([0.5,0.5,0.5])}
traj_spacing = 2
mutateSize = 5
nArrows = 1
colorTrajectories = 'blue'
initColor = 'cyan'
endColor = 'yellow'
mdpColor = 'black'
pomdpColor = 'magenta'
nosiColor = 'slateblue'
siColor = 'salmon'

# Rebuild the local image for showing the zone of interest
m_local_map = np.zeros((n_row_focus, n_col_focus, 3))
for i in range(n_row_focus):
	for j in range(n_col_focus):
		feat_dict = final_map[i,j]	
		total_count = sum( v for key,v in feat_dict.items())
		m_local_map[i,j] = np.zeros(3)
		# print (i, j, feat_dict)
		for elem, v in feat_dict.items():
			m_local_map[i,j] += (float(v)/total_count) * color_dist[elem]

# Create the figure and draw the image inside the figure
plt.figure(figsize=(12,12))

# Plot the resulting map and define correctly the axes
info_color = plt.imshow(m_local_map, interpolation='none')
# plt.gca().set_xticklabels([ int(val+south_west_center[1]) for val in plt.gca().get_xticks()])
# plt.gca().set_yticklabels([ int(val+south_west_center[0]) for val in plt.gca().get_yticks()])

# Plot the agent trajectories used to do the IRL problem
for ind_traj_robot in robot_pos_evol:
	x = np.array([ j for (i, j, fv) in ind_traj_robot])
	y = np.array([ i for (i, j, fv) in ind_traj_robot])
	arrowplot(plt.gca(), x[0::traj_spacing+1], y[0::traj_spacing+1], nArrs=nArrows, mutateSize=mutateSize, color=colorTrajectories, markerStyle='o')
	# plt.plot(x[0:-1:20], y[0:-1:20])

# Plot the initial and end zone
plt.scatter([ j for (i,j,featv) in init_set], [ i for (i,j,featv) in init_set], color=initColor)
plt.scatter([ j for (i,j,featv) in goal_set], [ i for (i,j,featv) in goal_set], color=endColor)


plt.show()
################################################################################################