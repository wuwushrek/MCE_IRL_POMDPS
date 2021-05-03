from gym_minigrid.minigrid import *
import numpy as np

maze_repr_14 = \
[['#', '#', '#', '#', '#', '#', '#'],
 ['#', '0', '1', '2', '3', '4', '#'],
 ['#', '5', '#', '6', '#', '7', '#'],
 ['#', '8', '#', '9', '#', '10','#'],
 ['#', '11','#', '13','#', '12','#'],
 ['#', '#', '#', '14','#', '#', '#']]

goal_maze_14 = ['14']
lava_maze_14 = ['11']
soft_lava_maze_14 = ['12'] 


class LavaSoft(WorldObj):
	def __init__(self):
		super().__init__('lava', 'purple')

	def can_overlap(self):
		return True

	def render(self, img):
		c = (204, 102, 255)

		# Background color
		fill_coords(img, point_in_rect(0, 1, 0, 1), c)

		# Little waves
		for i in range(3):
			ylo = 0.3 + 0.2 * i
			yhi = 0.4 + 0.2 * i
			fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0,0,0))
			fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0,0,0))
			fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0,0,0))
			fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0,0,0))

class WallOther(WorldObj):
	def __init__(self):
		super().__init__('wall', 'purple')

	def can_overlap(self):
		return True

	def render(self, img):
		c = (30, 30, 30)

		# Background color
		fill_coords(img, point_in_rect(0, 1, 0, 1), c)

		# Little waves
		for i in range(12):
			yv = 0.1 * i
			fill_coords(img, point_in_line(0, yv, 1, yv, r=0.03), (0,0,0))

class WallMod(WorldObj):
	def __init__(self, color='purple'):
		super().__init__('wall', color)

	def see_behind(self):
		return False

	def render(self, img):
		fill_coords(img, point_in_rect(0, 1, 0, 1), np.array([30, 30, 30]))

class MazeDemo(MiniGridEnv):
	""" A maze representation from a prism file mode
	"""

	def __init__(self, maze_repr, goal, lava, soft_lava=list(),
					start_pos=None, agent_start_dir=0, seed=1337):
		self.parse_grid_repr(maze_repr)
		self.agent_start_pos = None if start_pos is None else self.state_to_pos[start_pos]
		self.agent_start_dir = agent_start_dir
		self.goals = set()
		self.lavas = set()
		self.soft_lavas = set()
		for state in goal:
			self.goals.add(self.state_to_pos[state])
		for state in lava:
			self.lavas.add(self.state_to_pos[state])
		for state in soft_lava:
			self.soft_lavas.add(self.state_to_pos[state])
		super().__init__(
			width = self.width_maze,
			height = self.height_maze,
			max_steps = 1000,
			seed= seed,
			see_through_walls=True)

	def _gen_grid(self, width, height):
		self.grid = Grid(width, height)
		# self.grid.wall_rect(0, 0, width, height)
		# Set the goal points
		for (i,j) in self.goals:
			self.put_obj(Goal(), i,j)
		# Set the lava points
		for (i,j) in self.lavas:
			self.put_obj(Lava(), i, j)
		# Set the soft lava points
		for (i,j) in self.soft_lavas:
			self.put_obj(LavaSoft(), i, j)
		# Set the walls
		for (i,j) in self.wall_list:
			self.put_obj(WallOther(), i, j)

		# Place the agent
		if self.agent_start_pos is not None:
			self.agent_pos = self.agent_start_pos
			self.agent_dir = self.agent_start_dir
		else:
			self.place_agent()

		self.mission = "Get to the green goal square while avoiding as much as possible purple lava and getting stuck in red lava"

	def place_agent(self, top=None, size=None, rand_dir=True, max_tries=math.inf):
		lKey = list(self.pos_to_state.keys())
		while True:
			indVal = self._rand_int(0, len(lKey))
			curr_pos = lKey[indVal]
			if curr_pos in self.goals or curr_pos in self.lavas or curr_pos in self.soft_lavas:
				continue
			break
		self.agent_pos = curr_pos
		if rand_dir:
			self.agent_dir = self._rand_int(0, 4)
		return curr_pos

	def parse_grid_repr(self, maze_repr):
		self.height_maze = len(maze_repr)
		self.width_maze = len(maze_repr[0])
		self.state_to_pos = dict()
		self.pos_to_state = dict()
		self.wall_list = set()
		for j, x_list in enumerate(maze_repr):
			for i, yVal in enumerate(x_list):
				if yVal.isdigit():
					self.state_to_pos[yVal] = (i,j)
					self.pos_to_state[(i,j)] = yVal
				else:
					assert yVal == '#', 'Walls are only represented by #'
					self.wall_list.add((i,j))

	def update_pos(self, next_state):
		(next_i, next_j) = self.state_to_pos[next_state]
		(curr_i, curr_j) = self.agent_pos
		if next_i-curr_i == 1:
			self.agent_dir = 0
		if next_i-curr_i == -1:
			self.agent_dir = 2
		if next_j-curr_j == 1:
			self.agent_dir = 1
		if next_j-curr_j == -1:
			self.agent_dir = 3
		self.agent_pos = (next_i, next_j)

	def get_neighbor_walls(self):
		res_list = list()
		(pos_i, pos_j) = self.agent_pos
		for (k,l) in [(1,0), (-1,0),(0,1), (0,-1)]:
			if (pos_i+k,pos_j+l) in self.wall_list:
				res_list.append((pos_i+k,pos_j+l))
		return res_list

	def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS):
		"""
		Render the whole-grid human view
		"""

		if close:
			if self.window:
				self.window.close()
			return

		if mode == 'human' and not self.window:
			import gym_minigrid.window
			self.window = gym_minigrid.window.Window('gym_minigrid')
			self.window.show(block=False)

		# Compute which cells are visible to the agent
		vis_mask = self.get_neighbor_walls()
		print(vis_mask)
		print(self.agent_pos)

		# Mask of which cells to highlight
		highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

		# For each cell in the visibility mask
		for (vis_i, vis_j) in vis_mask:
			# Mark this cell to be highlighted
			highlight_mask[vis_i, vis_j] = True

		# Render the whole grid
		img = self.grid.render(
			tile_size,
			self.agent_pos,
			self.agent_dir,
			highlight_mask=highlight_mask if highlight else None
		)

		if mode == 'human':
			self.window.set_caption(self.mission)
			self.window.show_img(img)

		return img


if __name__ == "__main__":
	from gym_minigrid.window import Window
	import matplotlib.pyplot as plt
	rseed = np.random.randint(100)
	mDemo = MazeDemo(maze_repr_14, goal_maze_14, lava_maze_14,
						soft_lava_maze_14, start_pos='0', seed=rseed)
	img = mDemo.render('rgb_array', tile_size=32, highlight=True)
	window = Window('Maze gridworld')
	window.set_caption(mDemo.mission)
	window.show_img(img)
	listTraj = ['1', '2', '3', '4', '7', '10', '12']
	for v in listTraj:
		mDemo.update_pos(v)
		img = mDemo.render('rgb_array', tile_size=32, highlight=True)
		window.show_img(img)
		plt.pause(0.01)

	window.show(block=True)
