import math
import numpy as np
import pickle

from itertools import product

from matplotlib import pyplot as plt, cm

# Load the pickle file containing the trajectories of the agent and the grid configuration
traj_dir = 'aaai_experiment_09-17.pickle'
traj_file = open(traj_dir, 'rb')
traj_data = pickle.load(traj_file)
# print(traj_data)

print(traj_data.keys())
print(traj_data['model_info'])
print(type(traj_data['data']))
traj_data = traj_data['data']

# Check if the map configuration are valid
n_row = None
n_col = None
n_feat = None
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

# # Origin point of the sub-grid
# south_west_center = (190, 350) # Row first then column

# # Number of row and column
# n_row_focus = 50
# n_col_focus = 100

# # Square for the delimitation of zone of interest
# x_delim = [south_west_center[0], south_west_center[0] + n_col_focus, south_west_center[0] + n_col_focus, south_west_center[0], south_west_center[0]]
# y_delim = [south_west_center[1], south_west_center[1], south_west_center[1] + n_row_focus, south_west_center[1] + n_row_focus, south_west_center[1]]

# # Build the resulting map
# my_map = build_map(traj_data, n_row_focus, n_col_focus, south_west_center=south_west_center, id_traj=0)
# print(my_map)

# # Build the transition matrix for a single map
# obs_radius = 4
# trans_dict, obsFullDict, id_obs = build_pomdp_no_merge(n_row_focus, n_col_focus, my_map, obs_radius=obs_radius)

# Visualize the maps and the robots trajectories
color_dist = {'unknown' : np.array([0, 0, 0]), 'grass' : np.array([0, 1, 0]),  'gravel' : np.array([0.5, 0.5, 0.5]), 'road' : np.array([0.0,0.0,1.0]),}
			  # 'grass+gravel' : np.array([1, 0, 0]), 'grass+gravel+unknown' : np.array([0, 0, 1]),
			  # 'grass+unknown' : np.array([0.4, 0.5, 0]), 'gravel+unknown' : np.array([0.5, 0.5, 1]),} # Composed features should be in alphabetic orders for unicity
ncols_fig = 3
nrows_fig = math.ceil((len(traj_data)+1) / ncols_fig) # THe last +1 is for the merged map
sizeSubFig_x = 5
sizeSubFig_y = 5

final_map = dict() # save the modified map
fig, axs = plt.subplots(nrows=nrows_fig, ncols=ncols_fig, figsize=(ncols_fig*sizeSubFig_x,nrows_fig*sizeSubFig_y), sharex=True, sharey=True)
ax_counter=0    # Count the number of axis we iterated on
for elem, ax in zip(traj_data, axs.flatten()):
	ax_counter += 1
	fmap_np = elem['feature_map_numpy'].copy()
	mUniqueValue = set()
	for i in range(fmap_np.shape[0]):
		for j in range(fmap_np.shape[1]):
			# mUniqueValue.add(tuple(fmap_np[i,j]))
			feat_name = [ feat for (r, feat) in zip(fmap_np[i,j], elem['features']) if r == 1.0]
			feat_name = feat_name[0] if len(feat_name)>0 else 'unknown'
			fmap_np[i,j] = color_dist[feat_name]
			mUniqueValue.add(feat_name)
			if (i,j) not in final_map:
				final_map[(i,j)] = set()
			final_map[(i,j)].add(feat_name)
	print('Unique elements  : {}'.format(mUniqueValue))
	# ax.imshow(fmap_np, origin='lower')
	ax.imshow(fmap_np, origin='upper')
	# ax.plot([ x for (x,y) in elem['trajectory']], elem['trajectory'], [ y for (x,y) in elem['trajectory']], elem['trajectory'], color='magenta', linewidth=2)
	# xvalues, yvalues = np.array([ x for (x,y) in elem['trajectory']]), np.array([ y for (x,y) in elem['trajectory']])
	# ax.quiver(xvalues[:-1], yvalues[:-1], xvalues[1:]-xvalues[:-1], yvalues[1:]-yvalues[:-1], color='black', linewidths=5, scale_units='xy', angles='xy')
	# ax.plot([ x for (x,y) in elem['trajectory']], [ y for (x,y) in elem['trajectory']], color='magenta', linewidth=2)
	# # ax.plot([x for (x,y) in m_traj], [y for (x,y) in m_traj], color='magenta', linewidth=2, linestyle='dashed')
	# ax.plot(x_delim, y_delim, color='red', linewidth=2)
	ax.set_title('Demo {}'.format(elem['demonstration_id']))
	ax.grid()

# Merge all the maps above and plot the resulting
rgb_merge_map = np.zeros((n_row, n_col, n_feat))
legend_name = set()
for i in range(n_row):
	for j in range(n_col):
		new_feat = '+'.join(sorted([x for x in final_map[(i,j)]]))
		legend_name.add(new_feat)
		if new_feat not in color_dist:
			color_dist[new_feat] = sum( color_dist[x] for x in final_map[(i,j)]) / len(final_map[(i,j)])
		rgb_merge_map[i,j] = color_dist[new_feat]

# # Just a trick to plot legend with the image
last_ax = axs.flatten()[ax_counter]
for feat_name in legend_name:
	last_ax.plot(0,0,"-", color=color_dist[feat_name], linewidth=5, label=feat_name)

# Plot the merged map
print(len(legend_name))

# last_ax.imshow(rgb_merge_map, origin='lower')
last_ax.imshow(rgb_merge_map, origin='upper')
# last_ax.plot(x_delim, y_delim, color='red', linewidth=2)
last_ax.set_title('Merged map')
last_ax.grid()


handles, labels = last_ax.get_legend_handles_labels()
plt.figlegend(handles, labels, loc= 'center', bbox_to_anchor=(0.5, 0.95), ncol=4, fancybox=True)
# plt.savefig('all_map.png', dpi=1000)

plt.figure(figsize=(8,8))
for feat_name in legend_name:
	plt.plot(0,0,"-", color=color_dist[feat_name], linewidth=5, label=feat_name)
plt.imshow(rgb_merge_map, origin='upper')
# plt.plot(x_delim, y_delim, color='red', linewidth=2)
# plt.plot([ x for (x,y) in elem['trajectory']], [ y for (x,y) in elem['trajectory']], color='magenta', linewidth=2)

# plt.plot([x for (x,y) in m_traj], [y for (x,y) in m_traj], color='magenta', linewidth=2, linestyle='dashed')
# plt.title('Map resulting from merging each map')

plt.grid()
plt.legend(loc= 'center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True)
# plt.savefig('merged_map.png', dpi=1000)


# # Just a trick to plot legend with the image
# for feat_name in mUniqueValue:
#   # feat_name = [ feat for (r, feat) in zip(rgb_val, elem['features']) if r == 1.0]
#   # feat_name = feat_name[0] if len(feat_name)>0 else 'unknown'
#   mColor = color_dist[feat_name]
#   plt.plot(0,0,"-", color=mColor, label=feat_name)

# plt.imshow(fmap_np, origin='lower')
# plt.grid()
# plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4)

# for elem in traj_data:
#   fmap_np = elem['feature_map_numpy'].copy()
#   mUniqueValue = set()
#   for i in range(fmap_np.shape[0]):
#       for j in range(fmap_np.shape[1]):
#           # mUniqueValue.add(tuple(fmap_np[i,j]))
#           feat_name = [ feat for (r, feat) in zip(fmap_np[i,j], elem['features']) if r == 1.0]
#           feat_name = feat_name[0] if len(feat_name)>0 else 'unknown'
#           fmap_np[i,j] = color_dist[feat_name]
#           mUniqueValue.add(feat_name)
#   print('Unique elements  : {}'.format(mUniqueValue))
#   plt.figure(figsize=(8,6))
#   # Just a trick to plot legend with the image
#   for feat_name in mUniqueValue:
#       # feat_name = [ feat for (r, feat) in zip(rgb_val, elem['features']) if r == 1.0]
#       # feat_name = feat_name[0] if len(feat_name)>0 else 'unknown'
#       mColor = color_dist[feat_name]
#       plt.plot(0,0,"-", color=mColor, label=feat_name)
#   plt.imshow(fmap_np, origin='lower')
#   plt.grid()
#   plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4)


plt.show()