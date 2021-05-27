import numpy as np
import stormpy

def from_prism_to_pomdp(path_prism, weight, discount, outputfile):
	""" Convert a prism file model (in the desired format of this package)
		into the traditional .pomdp model
		:param discount : The discount factor
		:param weight : The true weight associated to the features functions
						defined in the prism model
	"""
	prism_program = stormpy.parse_prism_program(path_prism)

	# Enable building rewards, state-valuation and labels
	options = stormpy.BuilderOptions([])
	options.set_build_all_reward_models(True)
	options.set_build_state_valuations(True)
	options.set_build_choice_labels(True)
	options.set_build_all_labels(True)
	options.set_build_with_choice_origins(True)
	options.set_build_observation_valuations(True)

	# Build the pomdp model
	pomdp = stormpy.build_sparse_model_with_options(prism_program, options)
	assert pomdp.model_type == stormpy.ModelType.POMDP, "The file does not represent a POMDP"
	assert len(pomdp.reward_models.keys())>0, "No feature rewards provided"
	for r_id, r_val in pomdp.reward_models.items():
		assert (not r_val.has_state_rewards), " Only state action rewards are allowed"
		assert r_val.has_state_action_rewards, " Only state action rewards are allowed"
		assert (not r_val.has_transition_rewards), " Only state action rewards are allowed"
	assert pomdp.has_choice_labeling, 'No action naming'
	choice_lab = pomdp.choice_labeling
	print(pomdp)

	# # # Export the pomdp representation
	# path_pmc = "export_origin_pomdp_test"
	# stormpy.export_to_drn(pomdp , path_pmc)

	# String represenation of the resulting model
	pomdp_out = ''
	pomdp_out += "discount: {}\n".format(discount)
	pomdp_out += "values: reward\n"
	pomdp_out += "states: {}\n".format(pomdp.nr_states)

	# Get the set of actions
	set_actions = set()
	for state in pomdp.states:
		curr_obs = pomdp.get_observation(state.id)
		for action in state.actions:
			# Get the choice index
			c_index = pomdp.get_choice_index(state.id, action.id)
			for act_name in choice_lab.get_labels_of_choice(c_index):
				set_actions.add(act_name)

	# Add the set of actions
	pomdp_out += "actions: {}\n".format(' '.join(act for act in set_actions))
	pomdp_out += "observations: {}\n".format(pomdp.nr_observations)
	pomdp_out += '\n'

	# Add the intial state distribution
	pomdp_out += "start: {}\n".format(' '.join(['1' if pomdp.initial_states[0] == s.id else '0' for s in pomdp.states]))
	pomdp_out += '\n'

	# Get the transitions and set of actions and rewards
	pomdp_out += "T : * : * : * 0.0\n"
	pomdp_out += 'R : * : * : * : * 0.0\n\n'
	for state in pomdp.states:
		curr_obs = pomdp.get_observation(state.id)
		# pomdp_out += 'T : * : {} : {} 1.0\n'.format(state.id, state.id)
		new_act_set = set_actions.copy()
		for action in state.actions:
			# Get the choice index
			c_index = pomdp.get_choice_index(state.id, action.id)
			curr_action_set = choice_lab.get_labels_of_choice(c_index)
			assert len(curr_action_set) <= 1, 'Each choice should have a single name'
			curr_action_str = None
			for act_name in curr_action_set:
				curr_action_str = act_name
			if curr_action_str is None:
				new_act_set = set()
			else:
				new_act_set -= set([curr_action_str])
			# Compute the reward obtained
			rew_val = 0
			for r_id, r_val in pomdp.reward_models.items():
				rew_val += r_val.get_state_action_reward(c_index) * weight[r_id]
			if curr_action_str is None: # Might be redundant but it is fine
				pomdp_out += 'R : * : {} : * : * {}\n'.format(state.id, rew_val)
			else:
				pomdp_out += 'R : {} : {} : * : * {}\n'.format(curr_action_str, state.id, rew_val)

			for trans in action.transitions:
				next_state = trans.column
				next_obs = pomdp.get_observation(next_state)
				prob_trans = trans.value()
				if curr_action_str is None:
					pomdp_out += 'T : * : {} : {} {}\n'.format(state.id, next_state, prob_trans)
				else:
					pomdp_out += 'T : {} : {} : {} {}\n'.format(curr_action_str, state.id, next_state, prob_trans)
		for action in new_act_set:
			pomdp_out += 'T : {} : {} : {} 1.0\n'.format(action, state.id, state.id)
		pomdp_out += '\n'


	# Add the Observation probabilities
	for state in pomdp.states:
		pomdp_out += 'O : * : {} : {} 1.0\n'.format(state.id, pomdp.get_observation(state.id))
	pomdp_out += '\n'

	# Output the file 
	outf = open(outputfile, 'w')
	outf.write(pomdp_out)

# weight_v = { 'poisonous' : 10, 'total_time' : 0.1, 'goal_reach' : 50}
# from_prism_to_pomdp('maze_stochastic.pm', weight_v, 0.999, 'maze_stochastic.pomdp')