from abc import ABC, abstractmethod

import stormpy
import stormpy.core
import stormpy.info

import stormpy.pomdp

import stormpy._config as config

import numpy as np
import stormpy.simulator

memoryTypeDict = {	'PomdpMemoryPattern.selective_counter' : stormpy.pomdp.PomdpMemoryPattern.selective_counter,
					'PomdpMemoryPattern.fixed_counter' : stormpy.pomdp.PomdpMemoryPattern.fixed_counter,
					'PomdpMemoryPattern.selective_ring' : stormpy.pomdp.PomdpMemoryPattern.selective_ring,
					'PomdpMemoryPattern.fixed_ring' : stormpy.pomdp.PomdpMemoryPattern.fixed_ring,
					'PomdpMemoryPattern.full' : stormpy.pomdp.PomdpMemoryPattern.full,
					'PomdpMemoryPattern.trivial' : stormpy.pomdp.PomdpMemoryPattern.trivial
				}

class POMDP(ABC):
	""" An abstract class modelling a POMDP.
		Typically, this class will be used as an interface
		to interact with a pomdp model written in .prism or
		any other model formats.
		The class provides convenient utilities functions to
		interact with a POMDP. Such functions include 
		getting the reward model, the state that satisfy a
		given Formula with probability one and the states that
		do not satisfy the formula.
	"""

	@property
	@abstractmethod
	def n_state(self):
		""" Getter method. Return the number of state in the 
			POMDP model
		"""
		pass

	@property
	@abstractmethod
	def n_obs(self):
		""" Getter method. Return the number of observations in the
			POMDP model
		"""
		pass

	@property
	@abstractmethod
	def n_trans(self):
		""" Getter method. Return the number of transitions in the
			POMDP model
		"""
		pass

	@abstractmethod
	def write_to_dict(self, resDict):
		""" Save some key components of this POMDP as kets values 
			of the given dictionary resDict
		"""
		pass

	@property
	@abstractmethod
	def prob0E(self):
		""" Set of states with probability zero to satisfy the
			specifications
			state_id in prob0E to check if a state is in prob0E.
			for state_id in prob1A: To iterate
		"""
		pass

	@property
	@abstractmethod
	def prob1A(self):
		""" Set of states with probability 1 to satisfy the
			specifications
			state_id in prob1A to check if a state is in prob1A.
			for state_id in prob1A: To iterate
		"""
		pass

	@property
	@abstractmethod
	def has_sideinfo(self):
		""" Boolean specifying if side information was given or not
		"""
		pass

	@property
	@abstractmethod
	def pred(self):
		""" Return the predecesors of each state with the corresponding actions and
			transition probabilities
			pred[state_id] = list((next_state_k, action_k, prob_k)) such that
			from next_state_k, taking action action_k, with prob_k we end at state_id
		"""
		pass

	@property
	@abstractmethod
	def succ(self):
		""" Return the successor of each state with the corresponding actions and
			transition probabilities
			succ[state_id] = list((next_state_k, action_k, prob_k)) such that
			from state_id, taking action action_k, with prob_k we end at next_state_k
		"""
		pass

	@property
	@abstractmethod
	def reward_features(self):
		""" Return a dictionary of observation- action reward for
			each feature
			reward_features[name][(obs_id,action)] is the reward value
		"""
		pass

	@property
	@abstractmethod
	def states_act(self):
		""" Return for each state the admissible actions
			states_act[state_id] = set(a1, a2,...)
		"""
		pass

	@property
	@abstractmethod
	def obs_act(self):
		""" Return for each observation the admissible actions
			obs_act[obs_id] = set(a1, a2, a3, ...)
		"""
		pass

	@property
	@abstractmethod
	def states(self):
		""" Return for the set of states of the POMDP as a list
		"""
		pass

	@property
	@abstractmethod
	def obs(self):
		""" Return for the set of observations of the POMDP as a list
		"""
		pass

	@property
	@abstractmethod
	def state_init_prob(self):
		""" Give the set of initial states and their probabilties
			init_state_distr.get(state_id, 0) return the corresponding 
			probability if state_id is an initial state and 0 otherwise
		"""
		pass

	@property
	@abstractmethod
	def obs_state_distr(self):
		""" Give the set of initial states and their probabilties
			_obs_state_distr[state_id][obs_id] IS the 
			probability of perceiving observation obs_id at state_id.
		"""
		pass

	@abstractmethod
	def string_repr_state(self, state_id):
		""" Give the string representation of the state id in the format it was
			encoded by the user inside the PRISM file or whatever format is used.
		"""
		pass

	@abstractmethod
	def simulate_policy(self, sigma, weight, max_run, max_iter_per_run, 
						obs_based=True, stop_at_accepting_state=True):
		""" Given an observation-based and randomized policy, compute the trajectories induced
			by the policy until it reaches a trapping state
			:param sigma : an observation based policy
			:weight : coefficient associated to each feature reward
			:max_run : The total number of run
			:max_iter_per_run : The maximum iteration per run
			:param obs_based : True if the policy is observation based
			:param stop_at_accepting_state : When reaching an accepting state, stopy the current run
		"""
		pass
	
	

class PrismModel(POMDP):
	""" Instantiate a POMDP model written in the prism format
	"""
	def __init__(self, path_prism, formulas=[], memory_len=1, 
					counter_type= stormpy.pomdp.PomdpMemoryPattern.selective_counter,
					export=True, savePomdp=True):
		""" Parse the prism file to build the POMDP model, then
			compute the DTMC induced by the given formula to obtain both
			the set of states that guarantees the satisfaction of the
			given formulas and the set of states that falsify teh formula
			:param path_prism : the path to the prism file
			:param formulas : A list of string representing formulas to satisfied
				Each formula should be probabilistic path formula representing either 
				Always, Eventually, or Until 
			:param memory_len : The memmory of the FSC controller
			:param counter_type : The type of the memory counter,
									stormpy.pomdp.PomdpMemoryPattern.selective_counter
									stormpy.pomdp.PomdpMemoryPattern.fixed_counter
									stormpy.pomdp.PomdpMemoryPattern.selective_ring
									stormpy.pomdp.PomdpMemoryPattern.fixed_ring
									stormpy.pomdp.PomdpMemoryPattern.full
									stormpy.pomdp.PomdpMemoryPattern.trivial
			:param export : True if enable exporting the built POMDP and the PMC
			:param savePomdp : True if the class should keep track of the prism POMDP instance
		"""

		# Save the parameters that define the prism file
		self._path_prism = path_prism
		self._formulas = formulas
		self._export = export
		self.memory_len = memory_len
		self.counter_type = counter_type

		# Variable saving if side info was given or not
		self._has_sideinfo = (len(self._formulas) > 0)

		# Do we save the POMDP
		self.savePomdp = savePomdp

		# Build the model
		prism_program, _ = self.build_model()

		# Build the state satysfying the spec with prob 1 and 
		# the state not satysfying the sepc
		pomdp, m_choice_label, m_observation_valuation = self.build_prob01_prop(prism_program)

		# Build the observation, state, state-action, and state-observation mapping
		self.build_model_sets(pomdp, m_choice_label, m_observation_valuation)

		# Build the reward model
		self.build_reward_model(pomdp)

		# Save a string representation
		# self._str_repr = self.get_string(pomdp)
		self._str_repr = ""

		if savePomdp: # Save the storm pomdp model if required
			self.pomdp = pomdp
			self.prism_program = prism_program


	def build_model(self):
		""" Build the POMDP model from the prism file
		"""
		# Build the program
		prism_program = stormpy.parse_prism_program(self._path_prism)

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
		# print(pomdp)
		assert pomdp.model_type == stormpy.ModelType.POMDP, "The file does not represent a POMDP"
		assert len(pomdp.reward_models.keys())>0, "No feature rewards provided"
		for r_id, r_val in pomdp.reward_models.items():
			assert (not r_val.has_state_rewards), " Only state action rewards are allowed"
			assert r_val.has_state_action_rewards, " Only state action rewards are allowed"
			assert (not r_val.has_transition_rewards), " Only state action rewards are allowed"

		# Export the pomdp representation
		if self._export:
			path_pmc = "export_origin_pomdp_"  + self._path_prism
			stormpy.export_to_drn(pomdp , path_pmc)

		return prism_program, pomdp

	def build_model_sets(self, pomdp, m_choice_label, m_observation_valuation):
		""" Initialize the data for the model
		"""
		# Save the number of states
		self._n_state = pomdp.nr_states
		self._n_obs = pomdp.nr_observations
		self._n_trans = pomdp.nr_transitions

		# Define the state space
		self._states = list()
		for state in pomdp.states:
			self._states.append(state.id)

		# Define the initial states distribution -> Storm provide a single initial state after factoring
		self._state_init_prob = dict()
		assert len(pomdp.initial_states) == 1, "Model should have a single initial state using STORM"
		self._state_init_prob[pomdp.initial_states[0]] = 1.0

		# Define the observation space
		self._obs = [i for i in range(pomdp.nr_observations)]
		self._obs_state_distr = dict()
		for state_id in self._states:
			obs_id = pomdp.get_observation(state_id)
			self._obs_state_distr[state_id] = {obs_id : 1.0}

		# Define the state valuations
		state_val = pomdp.state_valuations
		self.state_string = dict()
		for state in pomdp.states:
			self.state_string[state.id] = state_val.get_string(state.id)

		self.action_string = dict()
		self.obs_string = dict()

		# Define the full list of state action and observation action
		self._states_act = dict() # Save for each state the allowed actions
		self._obs_act = dict() # Save for each observations the allowed actions
		self._pred_state = dict() # Save the predecessor of each state
		self._succ_state = dict() # Save the successor of each state
		for state in pomdp.states:
			self._states_act[state.id] = set()
			self._succ_state[state.id] = list()
			obs_id = pomdp.get_observation(state.id)
			if m_observation_valuation is not None:
			    self.obs_string[obs_id] = m_observation_valuation.get_string(obs_id)
			if obs_id not in self._obs_act:
				self._obs_act[obs_id] = set()
			for action in state.actions:
			    c_index = pomdp.get_choice_index(state.id, action.id)
			    if m_choice_label is not None:
			        if state.id not in self.action_string:
			            self.action_string[state.id] = dict()
			        self.action_string[state.id][action.id] = m_choice_label.get_labels_of_choice(c_index)
			    self._states_act[state.id].add(action.id)
			    self._obs_act[obs_id].add(action.id)
			    for trans in action.transitions:
			        if trans.column not in self._pred_state:
			            self._pred_state[trans.column] = list()
			        self._pred_state[trans.column].append((state.id, action.id, trans.value()))
			        self._succ_state[state.id].append((trans.column, action.id, trans.value()))


	def build_reward_model(self, pomdp):
		""" Compute and save the observation-action reward for 
			each feature reward function
			This function assumes that in the prism file, the reward was 
			an observation-action based reward. 
			Specifically, each observation, action of each feature has a unique reward.
			Note that this is not necessary the case if the feature are written as 
			state-action based rewards
		"""
		self._reward_features = dict()
		for r_id, r_val in pomdp.reward_models.items():
			self._reward_features[r_id] = dict()
			for state in pomdp.states:
				obs_id = pomdp.get_observation(state.id)
				for action in state.actions:
					c_index = pomdp.get_choice_index(state.id, action.id)
					curr_rew = r_val.get_state_action_reward(c_index)
					assert ((obs_id, action.id) not in self._reward_features[r_id]) or \
							self._reward_features[r_id][(obs_id, action.id)] == curr_rew, \
								"Observation reward not unique"
					self._reward_features[r_id][(obs_id, action.id)] = curr_rew
						

	def build_prob01_prop(self, prism_program):
		""" Build thet set of states that are guaranteed to satisfy the specifications
			with probability 1 and the set that are guaranteed to not satisfy the 
			specifications.
			This function assumes that the specification formulas are given
			as a list of Logic formula. Each element in the list is a path formula
			that can either be Always phi, Eventually phi, or phi Until psi. No other type of
			logical formulas are allowed by now.
		"""
		listProp = list() # Store all the properties given by the user
		typeFormula = list() # Store if the formula at the adequate index of listProp is of type Always or not
		
		# Build the properties from the given formula
		for formula in self._formulas:
			# Parse the formula fro; the prism model
			props = stormpy.parse_properties_for_prism_program(formula, prism_program)
			# Iterate over the resulting properties and check each property type
			for p in props:
				# Check if the formula is a probability operator
				assert p.raw_formula.is_probability_operator, "Side information should be given as a probability operator"
				
				# Check if the formula are always, eventually, and until
				assert p.raw_formula.subformula.is_eventually_formula or \
					p.raw_formula.subformula.is_until_formula or \
					isinstance(p.raw_formula.subformula, stormpy.logic.GloballyFormula),\
					" Side information should be eventually, Always, or Until operator"
				
				# if the formula is G phi, we need to transform it into ! (F !phi)
				if isinstance(p.raw_formula.subformula, stormpy.logic.GloballyFormula):
					# Transform the formula into eventually
					str_rep_sub = 'P=? [F !('+p.raw_formula.subformula.subformula.__str__()+')]'
					# Parse the formula to obtain the adequate properties
					prop_neg_always = stormpy.parse_properties_for_prism_program(str_rep_sub, prism_program)
					# Add the resulting properties
					for p_neg in prop_neg_always:
						listProp.append(p_neg.raw_formula)
						# Specify that the property was intially an Always operator
						typeFormula.append(True)
				else: # If the formula is not G phi, nothing extra is needed
					typeFormula.append(False)
					listProp.append(p.raw_formula)
		
		# Enable building rewards, state-valuation and labels
		options = stormpy.BuilderOptions(listProp)
		options.set_build_all_reward_models(True)
		options.set_build_state_valuations(True)
		options.set_build_choice_labels(True)
		options.set_build_all_labels(True)
		options.set_build_with_choice_origins(True)
		options.set_build_observation_valuations(True)

		# Build the pomdp model with all the parsed properties
		pomdp_t = stormpy.build_sparse_model_with_options(prism_program, options)

		m_choice_label = None
		if pomdp_t.has_choice_labeling():
		    m_choice_label = pomdp_t.choice_labeling
		m_observation_valuation = None
		if pomdp_t.has_observation_valuations():
		    m_observation_valuation = pomdp_t.observation_valuations

		# Make its representation canonic to obtain later the pMC
		pomdp = stormpy.pomdp.make_canonic(pomdp_t)

		# Export the pomdp representation if enable
		if self._export:
			path_pmc = "export_pomdp_formula_"  + self._path_prism
			stormpy.export_to_drn(pomdp , path_pmc)

		# Construct the memory for the FSC
		memory_builder = stormpy.pomdp.PomdpMemoryBuilder()
		# memory = memory_builder.build(stormpy.pomdp.PomdpMemoryPattern.fixed_counter, self.memory_len)
		memory = memory_builder.build(self.counter_type, self.memory_len)
		# apply the memory onto the POMDP to get the cartesian product
		pomdp = stormpy.pomdp.unfold_memory(pomdp , memory, add_memory_labels=True, keep_state_valuations=True)

		# Export the pomdp representation if enable
		if self._export:
			path_pmc = "export_pomdp_formula_mem_"  + self._path_prism
			stormpy.export_to_drn(pomdp , path_pmc)
		
		# Apply the unknown FSC to obtain a pmc from the POMDP
		pmc = stormpy.pomdp.apply_unknown_fsc(pomdp, stormpy.pomdp.PomdpFscApplicationMode.simple_linear)
		
		# Export the PMC representation if enable
		if self._export:
			path_pmc = "export_pmc_formula_"  + self._path_prism
			export_options = stormpy.core.DirectEncodingOptions()
			export_options.allow_placeholders = False
			stormpy.export_to_drn(pmc, path_pmc, export_options)

		# Compute the 01 probability
		list01Prob = list()
		for prop in listProp: # Iterate over all properties
			if prop.subformula.is_eventually_formula: # Check if the formula is eventually
				phi_states = stormpy.BitVector(pomdp.nr_states, True)
				psi_states = stormpy.model_checking(pmc, prop.subformula.subformula).get_truth_values()
			else: # In case the formula is an until formula, evaluate left and right terms via the checker
				phi_states = stormpy.model_checking(pmc, prop.subformula.left_subformula).get_truth_values()
				psi_states = stormpy.model_checking(pmc, prop.subformula.right_subformula).get_truth_values()
			list01Prob.append(stormpy.compute_prob01max_states(pmc, phi_states, psi_states))

		# Store the resulting prob0E and prob1A:
		self._prob0E = set()
		self._prob1A = set()
		firstIter = True # Temporary variable to figure out if first iteration
		for (prob0E, prob1A), isAlways in zip(list01Prob, typeFormula):
			prob0E_state, prob1A_state = set(), set()
			# Save the state with prob 1 and prob 0 in the set above
			for state in pomdp.states:
				if prob0E.get(state.id):
					prob0E_state.add(state.id)
				if prob1A.get(state.id):
					prob1A_state.add(state.id)
			if isAlways: # If the formula is always, swap the 1A and 0E
				prob1A_state, prob0E_state = prob0E_state, prob1A_state
			if firstIter: # First iteration needs to be initialized consistenly
				self._prob1A = prob1A_state
				firstIter = False
			self._prob0E = prob0E_state | self._prob0E
			self._prob1A = prob1A_state & self._prob1A
		return pomdp, m_choice_label, m_observation_valuation

	@property
	def n_state(self):
		return self._n_state

	@property
	def n_obs(self):
		return self._n_obs

	@property
	def n_trans(self):
		return self._n_trans

	@property
	def prob0E(self):
		return self._prob0E if self._has_sideinfo else set()

	@property
	def prob1A(self):
		return self._prob1A if self._has_sideinfo else set()

	@property
	def has_sideinfo(self):
		return self._has_sideinfo

	@property
	def pred(self):
		return self._pred_state

	@property
	def succ(self):
		return self._succ_state

	@property
	def reward_features(self):
		return self._reward_features

	@property
	def states_act(self):
		return self._states_act

	@property
	def obs_act(self):
		return self._obs_act

	@property
	def states(self):
		return self._states

	@property
	def obs(self):
		return self._obs

	@property
	def state_init_prob(self):
		return self._state_init_prob

	@property
	def obs_state_distr(self):
		return self._obs_state_distr

	def string_repr_state(self, state_id):
		return self.state_string[state_id]

	def write_to_dict(self, resDict):
		resDict['ns'] = self.n_state
		resDict['no'] = self.n_obs
		resDict['nt'] = self.n_trans
		resDict['path_prism'] = self._path_prism
		resDict['mem'] = self.memory_len
		resDict['counter_type'] = str(self.counter_type)
		resDict['formula'] = self._formulas

	def simulate_policy(self, sigma, weight, max_run, max_iter_per_run, seed=None,
							obs_based=True, stop_at_accepting_state=True, stat=dict(), fun_based=False):
		assert self.savePomdp, 'POMDP was not saved in memory for simulation'
		rand_seed = np.random.randint(0, 10000) if seed is None else seed 
		simulator = stormpy.simulator.create_simulator(self.pomdp, seed=rand_seed)
		res_traj = list()
		rew_list = list()
		stat['seed'] = rand_seed
		stat['obs_based'] = obs_based
		stat['max_len'] = 0
		for i in range(max_run):
			# Initialize the simulator
			obs, reward = simulator.restart()
			current_state = simulator._report_state()
			# Save the sequence of observation action of this trajectory
			seq_obs = list()
			acc_reward = list() # Accumulated reward
			firstDone = True # First time reaching done
			for j in range(max_iter_per_run):
				# Get the list of available actions
				actList = [a for a in simulator.available_actions()]
				# Add the observaion, action to the sequence
				if obs_based:
					# Pick an action in the set of random actions with probability given by the policy
					assert not fun_based or (obs_based and fun_based)
					sigma_obs = sigma[obs] if not fun_based else sigma(obs)
					act = np.random.choice(np.array([a for a in sigma_obs]), 
								p=np.array([probA for a, probA in sigma_obs.items()]))
				else:
					# Pick an action in the set of random actions with probability given by the
					act = np.random.choice(np.array([a for a in sigma[current_state]]), 
								p=np.array([probA for a, probA in sigma[current_state].items()]))
				seq_obs.append((obs, act))
				# Update the reward function
				#if firstDone:
				acc_reward.append(sum(w_val*self._reward_features[r_name][(obs,act)] for r_name, w_val in weight.items()))
				# Update the state of the simulator
				obs, reward = simulator.step(actList[act])
				current_state = simulator._report_state()
				# Check if reaching a looping state
				if simulator.is_done() and stop_at_accepting_state:
					if firstDone:
						firstDone = False
					else:

						break
			stat['max_len'] = np.maximum(stat['max_len'], len(acc_reward))
			#append zeros to the reward after simulation is over
			while len(acc_reward) < max_iter_per_run:
				acc_reward.append(0.0)
			#print(acc_reward,"after")
			res_traj.append(seq_obs)
			rew_list.append(acc_reward)
			# print('---------------------------------------------------------')
			# print('[Run: {}, Number iteration: {}, Reward attained: {} ]'.format(i, j, sum(acc_reward)))
			# print('Sequence : {}'.format(seq_obs))
		return res_traj, rew_list


	def get_string(self, pomdp):
		""" String represenattion of this POMDP
		"""
		# Print the STorm representation of the POMDP
		str_repr = ""
		str_repr += "POMDP Model\n"
		str_repr += pomdp.__str__()

		# Get parameteres used in the model file
		state_val = pomdp.state_valuations
		
		# Print the intiial state
		for state in pomdp.initial_states:
			str_repr += "Initial state: {} {}\n".format(state, state_val.get_string(state))
		
		# Print the observation modelStateValuation
		for state in pomdp.states:
			str_repr += 'State id: {} {}, observation: {} \n'.format(
				state.id, state_val.get_string(state.id),
				pomdp.get_observation(state.id))

		# Print all the states actions transition with their valuations and rewards 
		for state_repr in pomdp.states:
			for action in state_repr.actions:
				# Get the choice index
				c_index = pomdp.get_choice_index(state_repr.id, action.id)
				dictR = dict()
				# Reward model
				for r_id, r_val in pomdp.reward_models.items():
					dictR[r_id] = r_val.get_state_action_reward(c_index)
				str_repr += '-----------------------\n'
				str_repr += "State id: {} {}, Action id: {}, Rewards: {}\n".format(
					state_repr.id, state_val.get_string(state_repr.id), 
					action.id, dictR)
				for trans in action.transitions:
					str_repr += 'State id: {} {}, Action id: {} ---> Next state: {} {}, with prob {}\n'.format(
							state_repr.id, state_val.get_string(state_repr.id), 
							action.id,
							trans.column, state_val.get_string(trans.column), np.around(trans.value(),3)
							)
		str_repr += 'Memory policy size : {}\n'.format(self.memory_len)
		return str_repr

	# def get_string(self, pomdp):
	#   """ String represenattion of this POMDP
	#   """
	#   # Print the STorm representation of the POMDP
	#   str_repr = ""
	#   str_repr += "POMDP Model\n"
	#   str_repr += pomdp.__str__()

	#   # Get parameteres used in the model file
	#   print(dir(stormpy.SparsePomdp))
	#   state_val = pomdp.state_valuations
	#   choice_lab = pomdp.choice_labeling
	#   obs_val = pomdp.observation_valuations
		
	#   # Print the intiial state
	#   for state in pomdp.initial_states:
	#       str_repr += "Initial state: {} {}\n".format(state, state_val.get_string(state))
		
	#   # Print the observation modelStateValuation
	#   for state in pomdp.states:
	#       str_repr += 'State id: {} {}, observation: {} {} \n'.format(
	#           state.id, state_val.get_string(state.id),
	#           pomdp.get_observation(state.id),
	#           obs_val.get_string(pomdp.get_observation(state.id)))

	#   # Print all the states actions transition with their valuations and rewards 
	#   for state_repr in pomdp.states:
	#       for action in state_repr.actions:
	#           # Get the choice index
	#           c_index = pomdp.get_choice_index(state_repr.id, action.id)
	#           print('state, action, Choice index : ', state_repr.id, action.id, c_index)
	#           dictR = dict()
	#           # Reward model
	#           for r_id, r_val in pomdp.reward_models.items():
	#               dictR[r_id] = r_val.get_state_action_reward(c_index)
	#           str_repr += '-----------------------\n'
	#           str_repr += "State id: {} {}, Action id: {} {}, Rewards: {}\n".format(
	#               state_repr.id, state_val.get_string(state_repr.id), 
	#               action.id, choice_lab.get_labels_of_choice(c_index), dictR)
	#           for trans in action.transitions:
	#               str_repr += 'State id: {} {}, Action id: {} {} ---> Next state: {} {}, with prob {}\n'.format(
	#                       state_repr.id, state_val.get_string(state_repr.id), 
	#                       action.id, choice_lab.get_labels_of_choice(c_index),
	#                       trans.column, state_val.get_string(trans.column), np.around(trans.value(),3)
	#                       )
	#   return str_repr

	def __str__(self):
		return self._str_repr

def correct_policy(pol):
	""" A basic utilities function to deal with numerical issues of computed policies
		Basically, it set to zero negative values that are close to zero (due to numerical issue of optimizer)
		and it makes sure the resulting policy for each observation sums up to 1
	"""
	res_pol = dict()
	for o, actDict in pol.items():
		res_pol[o] = dict()
		neg_val = dict()
		sum_val = 0
		for a, val in actDict.items():
			# Borderline case negative and greater than 1
			if val < 0:
				res_pol[o][a] = 0
			elif val > 1:
				res_pol[o][a] = 1
			else:
				res_pol[o][a] = val
			sum_val += res_pol[o][a]
		for a, val in actDict.items():
			# Normalization
			res_pol[o][a] = val/sum_val
		# diff_val = 1 - sum_val
		# # print(o, ' : Diff : ', diff_val)
		# for a, val in actDict.items():
		# 	if res_pol[o][a] + diff_val < 0:
		# 		res_pol[o][a] = 0
		# 		diff_val += res_pol[o][a]
		# 	elif res_pol[o][a] + diff_val > 1:
		# 		res_pol[o][a] = 1
		# 		diff_val -= (1-res_pol[o][a])
		# 		assert diff_val == 0
		# 	else:
		# 		res_pol[o][a] += diff_val
		# 		diff_val = 0
	return res_pol

if __name__ == "__main__":
	# Customize maze with loop removed for the poison state
	pModel = PrismModel("maze_stochastic.pm", ["P=? [F \"target\"]", "P=? [G !\"poison\"]"])
	print(pModel)

	# Some test
	print('number state, obs : {}, {}'.format(pModel.n_state, pModel.n_obs))
	print('Prob0E: {}'.format(pModel.prob0E))
	print('Prob1A: {}'.format(pModel.prob1A))
	print('has_sideinfo: {}'.format(pModel.has_sideinfo))
	print('successor list---------')
	for state, listM in pModel.succ.items():
		for (next_state, action, prob) in listM:
			print('State id {}, Action id {}, -> Next state {} with Prob {}'.format(state, action, next_state, prob))

	print('Predecessor list---------')
	for state, listM in pModel.pred.items():
		for (pred_state, action, prob) in listM:
			print('State id {}, Action id {}, -> Next state {} with Prob {}'.format(pred_state, action, state, prob))
	
	for f_name, rew_val in pModel.reward_features.items():
		for (obs, act), r_val in rew_val.items():
			print('{} : ({}, {}) --> {}'.format(f_name, obs, act, r_val))

	for state, acts in pModel.states_act.items():
		print('State: {}, actions: {}'.format(state, acts))

	for obs, acts in pModel.obs_act.items():
		print('Obs: {}, actions {}'.format(obs, acts))

	print('Set of states: {}'.format(pModel.states))
	print('Set of observations: {}'.format(pModel.obs))
	print('Initial states: {}'.format(pModel.state_init_prob))
	print('Prob[Obs|state] : {}'.format(pModel.obs_state_distr))

	# opts = stormpy.BuilderOptions()
	# print(dir(stormpy.logic))
	# print(dir(opts))
	# print('--------------------')
	# print(dir(stormpy.PrismProgram))
	# print('--------------------')
	# print(dir(stormpy))
	# print(dir(stormpy.SparsePomdp))
	# print('--------------------')
	# # print(dir(options))
	# print('--------------------')
	# print(dir(stormpy.pomdp))
	# str_repr += "Number of states: {}\n".format(self.pomdp.nr_states)
	# str_repr += "Number of transitions: {}\n".format(self.pomdp.nr_transitions)
	# str_repr += "Number of observations: {}\n".format(self.pomdp.nr_observations)
	# print(type(r_val), type(r_val.state_action_rewards))
	# print(r_val.state_action_rewards, len(r_val.state_action_rewards))
	# print(r_id, r_val.get_state_action_reward(c_index))
