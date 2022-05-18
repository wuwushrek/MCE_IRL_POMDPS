import numpy as np
from matplotlib import pyplot as plt, cm
import matplotlib.patches as patches

from data2prism_u1 import *

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

traj_dir = 'data_for_franck'
# (n_row, n_col, n_feat), traj_data, (m_robot, m_robot_row, m_robot_col) = load_map_file(traj_dir)
(n_row, n_col, n_feat), traj_data, m_robot_state_evol, featlist= load_map_file(traj_dir)

# Origin point of the sub-grid
south_west_center = (25, 60) # Row first then column
# south_west_center = (0, 0) # Row first then column

# Define the uncertain observation
obs_radius = 5

# Specify the trajectory of interest
id_traj = 0

# Number of row and column
# n_row_focus, n_col_focus = n_row, n_col
n_row_focus = 35
n_col_focus = 35
focus_zone = (south_west_center[0], south_west_center[1], south_west_center[0]+n_row_focus, south_west_center[1]+n_col_focus)

# focus_init = (0, 0, 4, 4)
focus_init_row = 0
focus_init_col = 30
focus_init_nrow = 4
focus_init_ncol = 4
focus_init = (focus_init_row, focus_init_col, focus_init_row+focus_init_nrow, focus_init_col+focus_init_ncol)

# Build the feature distribution on the focused map
final_map = build_map(traj_data, n_row_focus, n_col_focus, south_west_center=south_west_center, id_traj=id_traj)

robot_obs_evol, robot_pos_evol = build_state_trajectories(final_map, m_robot_state_evol, None, focus_zone)

goal_set = [ m_traj[-1] for m_traj in robot_pos_evol] 
# m_init_set = [ (i,j,featv) for (i, j), featv in final_map.items() if (i>=focus_init[0] and j>=focus_init[1] and i< focus_init[2] and j < focus_init[3])]
m_init_set = [(2,33),(3,33), (2,34), (3,34), (12,0), (13,0), (12,1), (13,1)]
init_set = [(i,j,final_map[i,j]) for (i,j) in m_init_set]

# Plot the obtained focused final map
# Visualize the maps and the robots trajectories
color_dist = {UNKNOWN_FEATURE : np.array([0, 0, 0]), 'grass' : np.array([11,102,35])/255.0,  'gravel' : np.array([1.0, 0, 0]), 'road' : np.array([0.5,0.5,0.5])}
traj_spacing = 1
mutateSize = 5
nArrows = 1
colorTrajectories = 'blue'
initColor = 'cyan'
endColor = 'yellow'

# Rebuild the local image for showing the zone of interest
m_local_map = np.zeros((n_row_focus, n_col_focus, 3))
for i in range(n_row_focus):
	for j in range(n_col_focus):
		m_local_map[i,j] = color_dist[final_map[i,j]]

# Create the figure and draw the image inside the figure
plt.figure(figsize=(12,12))

for feat_name in (featlist+[UNKNOWN_FEATURE]):
	print(feat_name)
	plt.plot(0,0,"-", color=color_dist[feat_name], linewidth=5, label=feat_name)
# Plot the resulting map and define correctly the axes
info_color = plt.imshow(m_local_map, interpolation='none')


# Plot the agent trajectories used to do the IRL problem
for ind_traj_robot in robot_pos_evol:
	x = np.array([ j for (i, j, fv) in ind_traj_robot])
	y = np.array([ i for (i, j, fv) in ind_traj_robot])
	arrowplot(plt.gca(), x[0::traj_spacing+1], y[0::traj_spacing+1], nArrs=nArrows, mutateSize=mutateSize, color=colorTrajectories, markerStyle='o')
	# plt.plot(x[0:-1:20], y[0:-1:20])

plt.legend(loc= 'center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True)
# # Plot the initial and end zone
# plt.scatter([ j for (i,j,featv) in init_set], [ i for (i,j,featv) in init_set], color=initColor)
# plt.scatter([ j for (i,j,featv) in goal_set], [ i for (i,j,featv) in goal_set], color=endColor)
import tikzplotlib
# tikzplotlib.clean_figure()
tikzplotlib.save('partmap.tex')

plt.show()