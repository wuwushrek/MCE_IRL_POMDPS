import numpy as np
import gurobipy as gp

from .parser_pomdp import POMDP, PrismModel
import json
import time

# A structure to set the rules for reducing and augmenting the trust region
trustRegion = {'red': lambda x: ((x - 1) / 1.5 + 1),
			   'aug': lambda x: min(10, (x - 1) * 1.25 + 1),
			   'lim': 1 + 1e-4}
gradientStepSize = lambda iterVal, gradVal: 1.0 / np.power(iterVal+2, 0.5)
ZERO_NU_S = 1e-8

def compute_feature_from_trajectory_and_rewfeat(traj, rewfeat, discount):
	""" Given a set of trajectories <-> a set of observation action sequences that describes
		expert trajecties, and a set of features on the pomdp (or the product pomdp),
		This function provide the desired expected feature induced by the trajectory
		:param traj : A set of observation-action sequences
	"""
	featMatch = dict()
	i_size = 1.0 / len(traj)
	for r_name, rew in rewfeat.items():
		featMatch[r_name] = i_size * \
							sum([rew[(o, a)] * (discount ** i) \
								 for seq_v in traj \
								 for i, (o, a) in enumerate(seq_v)])

	return featMatch

# Class for setting up options for the optimization problem
class OptOptions:
	def __init__(self, mu=1e4, mu_spec=1e4, mu_rew=1, rho=1, rho_weight=1.0, maxiter=100, max_update=None, 
				maxiter_weight=20, graph_epsilon=1e-3, discount=0.9, 
				verbose=True, verbose_weight=True, verbose_solver=None):
		"""
		Returns the float representation for a constant value
		:param mu: parameter for putting penalty on slack variables type: float
		:param mu_spec : parameter for putting penalty on spec slac variables
		:param rho : parameter for infeasbility of feature meatching constraint
		:param maxiter: max number of iterations type: integer
		:param max_update : Max number of correct updates in the SCP method
		:param maxiter_weight : max number of iteration for finding the weight
		:param graph_epsilon: the min probability of taking an action at each observation type: float
		:param silent: print stuff from gurobi or not type: boolean
		:param discount: discount factor type:float
		:param verbose: Enable some logging
		:return:
		"""
		self.mu = mu
		self.mu_spec = mu_spec
		self.rho = rho
		self.rho_weight = rho_weight
		self.maxiter = maxiter
		if max_update is None:
			self.max_update = maxiter
		else:
			self.max_update = max_update
		self.mu_rew = mu_rew
		self.quotmu = self.mu_rew / self.mu
		self.maxiter_weight = maxiter_weight
		self.graph_epsilon = graph_epsilon
		self.discount = discount
		self.verbose = verbose
		self.verbose_weight = verbose_weight
		self.verbose_solver = self.verbose if verbose_solver is None else verbose_solver
		if graph_epsilon < 0:
			raise RuntimeError("graph epsilon should be larger than 0")
		if discount < 0 or discount >= 1:
			raise RuntimeError("discount factor should be between 0 and 1")
		if mu <= 0:
			raise RuntimeError("mu should be larger than 0")


class IRLSolver:
	""" Class that encodes the max entropy inverse reinforcement learning on POMDPs
		as sequential linear optimization problems with trust regions to alleviate
		the error introduced by the linearization. Each LP subproblems are
		solved using Gurobi 9
		The class also provides a function to find a policy that maximize the expected
		rewards on the POMDP given a formula.
	"""

	def __init__(
			self,
			pomdp: POMDP,
			sat_thresh: float = 0.95,
			init_trust_region: float = trustRegion['aug'](4),
			max_trust_region: float = 4,
			rew_eps: float = 1e-6,
			options: OptOptions = OptOptions(),
	) -> None:

		# Attributes to check performances of building, solving the problems
		self.total_solve_time = 0  # Total time elapsed
		self.init_encoding_time = 0  # Time for encoding the full problem
		self.update_constraint_time = 0  # Total time for updating the constraints and objective function
		self.checking_policy_time = 0  # Total time for finding nu_s and nu_sa gievn a policy

		# Save the initial trust region
		self._trust_region = init_trust_region
		self._max_trust_region = max_trust_region
		self._sat_thresh = sat_thresh
		self._options = options
		self._rew_eps = rew_eps
		self._pomdp = pomdp

	def from_reward_to_optimal_policy_nonconvex_grb(self, weight):
		""" Given the weight for each feature functions in the POMDP model,
			compute the optimal policy that maximizes the expected reward
			while satisfying the specifications on the POMDP
			:param weight : A dictionary with its keys being the reward feature name
							and its value be a dictionary with (obs, act) as the key
							and the associated reward at the value
		"""
		# Create the optimization problem
		mOpt = gp.Model('Optimal Policy with NonConvex Gurobi Solver')
		self.total_solve_time = 0  # Total time elapsed
		self.init_encoding_time = 0  # Time for encoding the full problem
		nu_s, nu_s_spec, nu_s_a, nu_s_a_spec, sigma, slack_spec = \
			self.init_optimization_problem(mOpt, noLinearization=True)

		# Define the parameters used by Gurobi for this problem
		mOpt.Params.OutputFlag = self._options.verbose_solver
		mOpt.Params.Presolve = 2  # More aggressive presolve step
		mOpt.Params.FeasibilityTol = 1e-6
		mOpt.Params.OptimalityTol = 1e-6
		mOpt.Params.BarConvTol = 1e-6
		mOpt.Params.NonConvex = 2 # THe optimization is nonconvex

		# Build the objective function -> Negative expected reward
		linExprCost = gp.LinExpr(self.compute_expected_reward(nu_s_a, weight))

		# Add to the cost the penalization from the cost of staisfying the spec
		if self._pomdp.has_sideinfo:
			linExprCost.add(slack_spec, -self._options.mu_spec)

		# Set the objective function -> Negative sign to compensate the outout of compute_expected_reward
		mOpt.setObjective(linExprCost, gp.GRB.MAXIMIZE)

		# Solve the problem
		curr_time = time.time()
		mOpt.optimize()
		self.total_solve_time += time.time() - curr_time

		# Do some printing
		if self._options.verbose and mOpt.status == gp.GRB.OPTIMAL:
			print('[Time used to build the full Model : {}]'.format(self.init_encoding_time))
			print('[Total solving time : {}]'.format(self.total_solve_time))
			print('[Optimal expected reward : {}]'.format(mOpt.objVal / self._options.quotmu))
			if self._pomdp.has_sideinfo:
				print('[Satisfaction of the formula = {}]'.format(sum(nu_s_spec[s].x for s in self._pomdp.prob1A)))
				print('[Slack value spec = {}]'.format(slack_spec.x))
				print('[Number of steps : {}]'.format(sum(nu_s_val.x for s, nu_s_val in nu_s_spec.items())))
			# print('[Optimal policy : {}]'.format({o: {a: p.x for a, p in actVal.items()} for o, actVal in sigma.items()}))
		return {o: {a: val.x for a, val in actList.items()} for o, actList in sigma.items()}

	def solve_irl_pomdp_given_traj(self, featMatch, includeQuadCost=False, save_info = None):
		""" Solve the IRL problem given the feature matching expectation of the
			sample trajectory
			:param featMatch : feature counts from the expert
			:includeQuadCost : Specify if adding a quadratic cost to penalize not matching the features function
		"""
		# Get the expected feature reward
		featMatching = featMatch if includeQuadCost else None

		# Dummy initialization of the weight
		weight = {r_name: 1.0 for r_name, rew in self._pomdp.reward_features.items()}
		rew_pol_dict = {r_name: 0.0 for r_name, rew in self._pomdp.reward_features.items()}

		#rmspropdict
		rmsprop = {r_name: 0.0 for r_name, rew in self._pomdp.reward_features.items()}

		#momentum
		momentum = {r_name: 0.0 for r_name, rew in self._pomdp.reward_features.items()}

		# Create and compute solution of the scp extra_args = (ent_cost, nu_s_k, nu_s_a_k, nu_s_spec_k, nu_s_a_spec_k)
		pol, _, extra_args = self.compute_maxent_policy_via_scp(weight, init_problem=True, featMatch=featMatching, trust_prev=None)
		nu_s_a = extra_args[2]

		# Save the stats of the solver and weights
		stats_solver = dict()
		self._pomdp.write_to_dict(stats_solver)

		#set initial trust region for IRL part
		trust_reg_val= self._trust_region

		for i in range(self._options.maxiter_weight):
			# Store the difference between the expected feature by the policy and the matching feature
			diff_value = 0
			diff_value_dict = dict()

			# Save the new weight
			new_weight = dict()
			step_size = gradientStepSize(i + 1, diff_value_dict)

			if self._options.verbose_weight:
				print("---------------- Printing visitation iteration {} ---------------- ".format(i))
			
			for r_name, val in weight.items():
				rew = self._pomdp.reward_features[r_name]

				# Get the reward attained by the policy
				rew_pol = sum([rew[(o, a)] * p * nu_s_a_val \
							   for s, nu_s_a_t in nu_s_a.items() \
							   for a, nu_s_a_val in nu_s_a_t.items()
							   for o, p in self._pomdp.obs_state_distr[s].items()
							   ])
				rew_pol_dict[r_name]=rew_pol

				# Get the reward by the feature epectation
				rew_demo = featMatch[r_name]

				# Save the sum of the difference to detect convergence
				diff_value += np.abs(rew_demo - rew_pol)
				diff_value_dict[r_name] = rew_demo - rew_pol

				#incorporate rms prop to make sure that the gradients may not vary in magnitude
				rmsprop[r_name]=0.9*rmsprop[r_name]+(1-0.9)*np.power(diff_value_dict[r_name],2)

				momentum[r_name]=(0.0)*momentum[r_name]+(1-0.0)*(step_size/np.power(rmsprop[r_name] + 1e-8,0.5))*diff_value_dict[r_name]
				#rmsprop[r_name]=0.9*rmsprop[r_name]+(0.1)*(diff_value_dict[r_name])

				if self._options.verbose_weight:
					print(rew_pol,rew_demo,r_name, '| rmsprop: ', rmsprop[r_name])

			# Update the weight values
			for r_name, gradVal in diff_value_dict.items():
				# Gradient step update, scaled by the quadratic cost, and the magnitude of the feature
				# new_weight[r_name] = weight[r_name] + self._options.rho * (gradVal/abs(featMatch[r_name]))  # Gradient step size ?
				new_weight[r_name] = weight[r_name] + self._options.rho_weight *momentum[r_name]

				#this update is a way to "normalize" the features, the "rmsprop" part is related to RMSprop
				#new_weight[r_name] = weight[r_name] + self._options.rho_weight * step_size * gradVal / ((abs(rew_pol_dict[r_name]))*(rmsprop[r_name] + 1e-8) ** 0.5)

			# new_weight[r_name] = weight[r_name] + self._options.rho_weight * step_size * gradVal /abs(featMatch[r_name]+rew_pol_dict[r_name])
				#featMatch[r_name]=0.99*featMatch[r_name]+0.01*rew_pol_dict[r_name]
				#new_weight[r_name] = weight[r_name] + self._options.rho_weight * gradVal /abs(featMatch[r_name])
				# momentum[r_name] = self._options.rho_weight * step_size * gradVal

			if np.abs(diff_value) <= self._rew_eps:  # Check if the desired accuracy was attained
				if self._options.verbose_weight:
					# print('---------------- Weight iteration {} -----------------'.format(i))
					print('[Diff with feature matching] : {} ]'.format(diff_value))
					print('[Weight value] : {} ]'.format(weight))
				break

			# Save the policy if needed
			if save_info is not None and ((i == self._options.maxiter_weight-1) or (i > 0 and i % save_info[0] == 0)):
				stats_solver['time'] = self.total_solve_time+self.update_constraint_time+self.checking_policy_time
				stats_solver['rew'] = extra_args[0]
				stats_solver['sat_spec'] = sum(extra_args[3][s] for s in self._pomdp.prob1A) if self._pomdp.has_sideinfo else 0
				stats_solver['discount'] = self._options.discount
				stats_solver['obs_based'] = True
				for r_name, gradVal in diff_value_dict.items():
					stats_solver['diff_with_feat__{}'.format(r_name)] = \
						{'learned feature' : rew_pol_dict[r_name], 'expert feature' : featMatch[r_name], 'gradient' : gradVal, 'rmsprop' : rmsprop[r_name]}
				with open(save_info[1]+'_pol.json', 'w') as fp:
					json.dump(pol, fp, sort_keys=True, indent=4)
				with open(save_info[1]+'_weight.json', 'w') as fp:
					json.dump({'true weight' : save_info[2], 'learned weight' : weight}, fp, sort_keys=True, indent=4)
				with open(save_info[1]+'_stats.json', 'w') as fp:
					json.dump(stats_solver, fp, sort_keys=True, indent=4)

			# Update new weight
			weight = new_weight

			# Compute the new policy based on the obtained weight and previous trust region
			pol, trust_reg_opt, extra_args = self.compute_maxent_policy_via_scp(weight, init_problem=False,
												featMatch=featMatching, initPolicy=pol, 
												trust_prev=trust_reg_val, init_visit=extra_args)
			nu_s_a = extra_args[2]
			trust_reg_val=trust_reg_opt

			# Do some printing
			if self._options.verbose_weight:
				# print('---------------- Weight iteration {} -----------------'.format(i))
				print('[Diff with feature matching] : {} ]'.format(diff_value))
				print('[New weight value] : {} ]'.format(weight))
				print("Update time : {}s, Checking time : {}s, Solve time: {}s".format(
					self.update_constraint_time, self.checking_policy_time, self.total_solve_time))
		return weight, pol

	def compute_feature_from_trajectory(self, traj):
		""" Given a set of trajectories <-> a set of observation action sequences that describes
			expert trajecties, provide the desired expected feature induced by the trajectory
			:param traj : A set of observation-action sequences
		"""
		return compute_feature_from_trajectory_and_rewfeat(traj , self._pomdp.reward_features, self._options.discount)

	def compute_maxent_policy_via_scp(self, weight, init_problem=True, featMatch=None, 
										initPolicy=None, trust_prev=None, init_visit=None):
		""" Given the current weight for each feature functions in the POMDP model,
			and the feature expected reward, compute the optimal policy
			that maximizes the max causal entropy
			:param weight : the coefficient associated to each feature function
			:init_problem : True If the optimization problem hasn't been initialize before
			:featMatch : The vector of feature matching, if not given then
						||(feat_match - expected reward)||^2 is not added to the cost function
			:initPolicy : The starting policy of the scp method, uniform policy if None 
			:trust_prev: current trust region from the previous IRL iteration
			:init_visit : Store the associated nu_s, nu_s_a, ent_cost (entropy+spec sat) to initPolicy
		"""
		# Create the optimization problem
		if init_problem:
			self.scpOpt = gp.Model('Optimal Policy with Sequential convex programming Gurobi Solver')
			self.bellmanOpt = gp.Model('Optimal Policy with Sequential convex programming Gurobi Solver')
			self.total_solve_time = 0  # Total time elapsed
			self.init_encoding_time = 0  # Time for encoding the full problem
			self.update_constraint_time = 0
			self.checking_policy_time = 0

			self.nu_s, self.nu_s_spec, self.nu_s_a, self.nu_s_a_spec, self.sigma, self.slack_spec, \
			self.slack_nu_p, self.slack_nu_n, self.slack_nu_p_spec, self.slack_nu_n_spec, \
			self.constrLin, self.constrLinSpec, self.constrTrustReg, \
			self.nu_s_ver, self.nu_s_ver_spec, self.bellConstr, self.bellConstrDict = \
				self.init_optimization_problem(self.scpOpt,
											   noLinearization=False, checkOpt=self.bellmanOpt)

			if self._options.verbose:
				print('[Time used to build the full Model : {}]'.format(self.init_encoding_time))

			# Define the parameters used by Gurobi for the linearized problem
			self.scpOpt.Params.OutputFlag = self._options.verbose_solver
			self.scpOpt.Params.Presolve = 2  # More aggressive presolve step
			self.scpOpt.Params.Method = 2 # The problem is not really a QP
			self.scpOpt.Params.Crossover = 0
			self.scpOpt.Params.CrossoverBasis = 0
			# self.scpOpt.Params.NumericFocus = 3  # Maximum numerical focus
			self.scpOpt.Params.BarHomogeneous = 1  # No need for, our problem is always feasible/bound
			# self._encoding.Params.ScaleFlag = 3
			# self.scpOpt.Params.FeasibilityTol = 1e-6
			self.scpOpt.Params.OptimalityTol = 1e-6
			self.scpOpt.Params.BarConvTol = 1e-6

			# Define the parameters used by Gurobi for the auxialiry program
			self.bellmanOpt.Params.OutputFlag = self._options.verbose_solver
			self.bellmanOpt.Params.Presolve = 2 
			self.bellmanOpt.Params.Method = 2
			self.bellmanOpt.Params.Crossover = 0
			self.bellmanOpt.Params.CrossoverBasis = 0
			# self.bellmanOpt.Params.NumericFocus = 3  # Maximum numerical focus
			self.bellmanOpt.Params.BarHomogeneous = 1
			# self._encoding.Params.ScaleFlag = 3
			# self.bellmanOpt.Params.FeasibilityTol = 1e-6
			self.bellmanOpt.Params.OptimalityTol = 1e-6
			self.bellmanOpt.Params.BarConvTol = 1e-6

		trust_region = self._trust_region

		#initialize trust region from the previous IRL step, if it exists
		if trust_prev is not None:
			trust_region=trust_prev

		# Initialize policy at iteration k
		if initPolicy is not None:
			policy_k = initPolicy
		else:
			policy_k = {o: {a: 1.0 / len(actList) for a in actList} for o, actList in self._pomdp.obs_act.items()}

		# If the initial state and action visitation count are not given
		if init_visit is None:
			# Initialize the state and state-action visitation count based on the policy
			ent_cost, spec_cost, nu_s_k, nu_s_a_k, nu_s_spec_k, nu_s_a_spec_k = \
				self.verify_solution(self.bellmanOpt, self.nu_s_ver, policy_k, self.bellConstr,
								 nu_s_spec=self.nu_s_ver_spec, constrBellmanSpec=self.bellConstrDict)

			if self._pomdp.has_sideinfo and spec_cost < self._sat_thresh:
				ent_cost += (spec_cost - self._sat_thresh) * self._options.mu * self._options.mu_spec
		else:
			assert initPolicy is not None
			(ent_cost, nu_s_k, nu_s_a_k, nu_s_spec_k, nu_s_a_spec_k) = init_visit
			spec_cost = sum(nu_s_spec_k[s] for s in self._pomdp.prob1A) if self._pomdp.has_sideinfo else 0

		# Store the current entropy + spec cost
		latest_ent_spec = ent_cost

		# Add the cost associated to the linear expected discounted reward given nu_s_a ( multiply back by mu)
		ent_cost += sum(coeff * val * self._options.mu for coeff, val in self.compute_expected_reward(nu_s_a_k, weight))

		# Add ||(feat_match- expected reward)||^2 if feat_match is given
		if featMatch is not None:
			ent_cost -= self.compute_quad_featmatch_value(nu_s_a_k, featMatch)*(self._options.rho/2)

		# if self._options.verbose:
		# 	print("[Initialization] Entropy cost {}, Spec SAT : {}".format(ent_cost, spec_cost))
		# 	print("[Initialization] Number of steps : {}".format(sum(nu_s_val for s, nu_s_val in nu_s_spec_k.items())))

		# Get the total expected reward given theta (it is divided by self._options.mu)
		linExprReward = self.compute_expected_reward(self.nu_s_a, weight)

		# Get the cost term ||(feat_match- expected reward)||^2 if feat_match is given
		quadExprReward = 0
		if featMatch is not None:
			quadExprReward = self.compute_quad_featmatch(self.nu_s_a, featMatch)*(self._options.rho/2)

		# A counter to count the correct updates
		count_correct_update = 0

		# Save if the past step was rejected
		past_rejected = False

		for i in range(self._options.maxiter):
			# Update the set of linearized constraints
			curr_time = time.time()
			self.update_constr_and_trust_region(self.scpOpt, self.constrLin, self.constrLinSpec,
												self.constrTrustReg, nu_s_k, nu_s_spec_k, policy_k,
												self.nu_s, self.nu_s_spec, self.sigma, 
												trust_region, only_trust=past_rejected)

			# Set the current objective based on past solution
			if not past_rejected:
				penCostList = self.compute_entropy_cost(nu_s_k, nu_s_a_k, self.nu_s,
													self.nu_s_a, self.slack_nu_p, self.slack_nu_n, self.slack_nu_p_spec,
													self.slack_nu_n_spec, self.slack_spec)
				if featMatch is not None:
					# Something more efficient than adding the quadratic term like this-> TODO
					self.scpOpt.setObjective(gp.LinExpr([*linExprReward, *penCostList]) - quadExprReward, gp.GRB.MAXIMIZE)
				else:
					self.scpOpt.setObjective(gp.LinExpr([*linExprReward, *penCostList]), gp.GRB.MAXIMIZE)

			self.update_constraint_time += time.time() - curr_time
			
			# Solve the optimization problem
			curr_time = time.time()
			self.scpOpt.optimize()
			self.total_solve_time += time.time() - curr_time

			next_policy = {o: {a: self.sigma[o][a].x for a in actList} for o, actList in self._pomdp.obs_act.items()}

			curr_time = time.time()
			ent_cost_n, spec_cost_n, nu_s_k_n, nu_s_a_k_n, nu_s_spec_k_n, nu_s_a_spec_k_n = \
				self.verify_solution(self.bellmanOpt, self.nu_s_ver, next_policy, self.bellConstr,
									 nu_s_spec=self.nu_s_ver_spec, constrBellmanSpec=self.bellConstrDict)
			self.checking_policy_time += time.time() - curr_time
			#print("true val from policy",ent_cost_n)

			if self._pomdp.has_sideinfo and spec_cost_n - self._sat_thresh < -1e-6:  # The spec properties are not satisfied
				ent_cost_n += (spec_cost_n - self._sat_thresh) * self._options.mu * self._options.mu_spec
			latest_ent_spec_n = ent_cost_n

			# Add the actual reward
			ent_cost_n += sum(
				coeff * val * self._options.mu for coeff, val in self.compute_expected_reward(nu_s_a_k_n, weight))

			if featMatch is not None:
				ent_cost_n -= self.compute_quad_featmatch_value(nu_s_a_k_n, featMatch) * (self._options.rho/2)

			# Check if the new policy improves over the last obtained policy
			if ent_cost_n > ent_cost:
				policy_k = next_policy
				nu_s_k, nu_s_spec_k = nu_s_k_n, nu_s_spec_k_n
				nu_s_a_k, nu_s_a_spec_k = nu_s_a_k_n, nu_s_a_spec_k_n
				ent_cost, spec_cost = ent_cost_n, spec_cost_n
				latest_ent_spec = latest_ent_spec_n
				trust_region = np.minimum(trustRegion['aug'](trust_region), self._max_trust_region)
				past_rejected = False

				#only update policy within the max_updates (and in case there's specs, terminate with policy that satisfies the spec)
				count_correct_update += 1
				if count_correct_update >= self._options.max_update and \
					( (not self._pomdp.has_sideinfo) or (spec_cost - self._sat_thresh >= -1e-4)):
					break

			else:
				trust_region = trustRegion['red'](trust_region)
				past_rejected = True
				if self._options.verbose:
					print("[Iter {}: ----> Reject the current step]".format(i))

			if self._options.verbose:
				# print("[Iter {}]: Finding the state and state-action visitation count given a policy".format(i))
				# print("[Iter {}]: Optimal policy: {}".format(i, policy_k))
				print("[Iter {}]: Entropy cost {}, Spec SAT : {}, Trust region : {}".format(i, ent_cost, spec_cost, trust_region))
				if self._pomdp.has_sideinfo:
					print("[Iter {}]: Number of steps : {}".format(i, sum(
						nu_s_val for s, nu_s_val in nu_s_spec_k.items())))
				print("[Iter {}]: Update time : {}s, Checking time : {}s, Solve time: {}s".format(i,
																								  self.update_constraint_time,
																								  self.checking_policy_time,
																								  self.total_solve_time))

			if trust_region < trustRegion['lim']:
				if self._options.verbose:
					print("[Iter {}: ----> Min trust value reached]".format(i))
				#just to prevent trust region from shrinking forever
				trust_region = self._trust_region
				# trust_region = trustRegion['aug'](trust_region)
				break

		if self._options.verbose_weight:
			print("[No Iter {}]: Entropy + spec {}, Reward {}, Spec SAT : {}, Trust region : {}".format(i, 
					latest_ent_spec, (ent_cost-latest_ent_spec)/self._options.mu_rew, spec_cost, trust_region))

		return policy_k, trust_region, (latest_ent_spec, nu_s_k, nu_s_a_k, nu_s_spec_k, nu_s_a_spec_k)

	def from_reward_to_policy_via_scp(self, weight, initPolicy=None, save_info=None):
		"""Given the weight for each feature functions in the POMDP model,
			compute the optimal policy that maximizes the expected reward
			while satisfying the specifications
			:param weight : A dictionary with its keys being the reward feature name
							and its value be a dictionary with (obs, act) as the key
							and the associated reward at the value
		"""
		# Create the optimization problem

		self.scpOpt = gp.Model('Optimal Policy with Sequential convex programming Gurobi Solver')
		self.bellmanOpt = gp.Model('Feasible solution to the bellman constraints ')
		self.total_solve_time = 0  # Total time elapsed
		self.init_encoding_time = 0  # Time for encoding the full problem
		self.update_constraint_time = 0
		self.checking_policy_time = 0

		self.nu_s, self.nu_s_spec, self.nu_s_a, self.nu_s_a_spec, self.sigma, self.slack_spec, \
		self.slack_nu_p, self.slack_nu_n, self.slack_nu_p_spec, self.slack_nu_n_spec, \
		self.constrLin, self.constrLinSpec, self.constrTrustReg, \
		self.nu_s_ver, self.nu_s_ver_spec, self.bellConstr, self.bellConstrDict = \
			self.init_optimization_problem(self.scpOpt,
										   noLinearization=False, checkOpt=self.bellmanOpt)
		
		if self._options.verbose:
			print('[Time used to build the full Model : {}]'.format(self.init_encoding_time))

		# Define the parameters used by Gurobi for the linearized problem
		self.scpOpt.Params.OutputFlag = self._options.verbose_solver
		self.scpOpt.Params.Presolve = 2  # More aggressive presolve step
		self.scpOpt.Params.Method = 2 # The problem is not really a QP
		self.scpOpt.Params.Crossover = 0
		self.scpOpt.Params.CrossoverBasis = 0
		# self.scpOpt.Params.NumericFocus = 3  # Maximum numerical focus
		self.scpOpt.Params.BarHomogeneous = 1  # No need for, our problem is always feasible/bound
		# # self.scpOpt.Params.ScaleFlag = 3
		# self.scpOpt.Params.FeasibilityTol = 1e-6
		self.scpOpt.Params.OptimalityTol = 1e-6
		self.scpOpt.Params.BarConvTol = 1e-6

		# Define the parameters used by Gurobi for the auxialiry program
		self.bellmanOpt.Params.OutputFlag = self._options.verbose_solver
		self.bellmanOpt.Params.Method = 2 
		self.bellmanOpt.Params.Presolve = 2
		self.bellmanOpt.Params.Crossover = 0
		self.bellmanOpt.Params.CrossoverBasis = 0
		# self.bellmanOpt.Params.NumericFocus = 3  # Maximum numerical focus
		self.bellmanOpt.Params.BarHomogeneous = 1
		# # self._encoding.Params.ScaleFlag = 3
		# self.bellmanOpt.Params.FeasibilityTol = 1e-6
		self.bellmanOpt.Params.OptimalityTol = 1e-6
		self.bellmanOpt.Params.BarConvTol = 1e-6

		# Initialize policy at iteration k
		if initPolicy is not None:
			policy_k = initPolicy
		else:
			policy_k = {o: {a: 1.0 / len(actList) for a in actList} for o, actList in self._pomdp.obs_act.items()}
		# policy_k = {o: {a: 1.0 / len(actList) for a in actList} for o, actList in self._pomdp.obs_act.items()}

		# Initialize the state and state-action visitation count based on the policy
		ent_cost, spec_cost, nu_s_k, nu_s_a_k, nu_s_spec_k, nu_s_a_spec_k = \
			self.verify_solution(self.bellmanOpt, self.nu_s_ver, policy_k, self.bellConstr,
								 nu_s_spec=self.nu_s_ver_spec, constrBellmanSpec=self.bellConstrDict)
		
		ent_cost = 0  # No need for entropy here
		if self._pomdp.has_sideinfo and spec_cost < self._sat_thresh:
			ent_cost += (spec_cost - self._sat_thresh) * self._options.mu * self._options.mu_spec
		ent_cost += sum(
			coeff * val * self._options.mu for coeff, val in self.compute_expected_reward(nu_s_a_k, weight))

		if self._options.verbose:
			print("[Initialization] Reward attained {}, Spec SAT : {}".format(ent_cost, spec_cost))
			print("[Initialization] Number of steps : {}".format(sum(nu_s_val for s, nu_s_val in nu_s_spec_k.items())))

		# Initial trust region
		trust_region = self._trust_region

		# Save the optimal reward
		opt_reward = 0

		# Save the stats of the solver
		stats_solver = dict()
		self._pomdp.write_to_dict(stats_solver)

		# Get the total expected reward given theta
		# linExprReward = [(-c, val) for c, val in self.compute_expected_reward(self.nu_s_a, weight)]
		linExprReward = [(c, val) for c, val in self.compute_expected_reward(self.nu_s_a, weight)]
		for i in range(self._options.maxiter):

			# Update the set of linearized constraints
			curr_time = time.time()
			self.update_constr_and_trust_region(self.scpOpt, self.constrLin, self.constrLinSpec,
												self.constrTrustReg, nu_s_k, nu_s_spec_k, policy_k,
												self.nu_s, self.nu_s_spec, self.sigma, trust_region)

			# Set the current objective based on past solution
			penCostList = self.compute_entropy_cost(nu_s_k, nu_s_a_k, self.nu_s,
													self.nu_s_a, self.slack_nu_p, self.slack_nu_n, self.slack_nu_p_spec,
													self.slack_nu_n_spec, self.slack_spec, addEntropy=False)
			self.scpOpt.setObjective(gp.LinExpr([*linExprReward, *penCostList]), gp.GRB.MAXIMIZE)
			self.update_constraint_time += time.time() - curr_time

			# Solve the optimization problem
			curr_time = time.time()
			self.scpOpt.optimize()
			self.total_solve_time += time.time() - curr_time

			next_policy = {o: {a: self.sigma[o][a].x for a in actList} for o, actList in self._pomdp.obs_act.items()}

			curr_time = time.time()
			ent_cost_n, spec_cost_n, nu_s_k_n, nu_s_a_k_n, nu_s_spec_k_n, nu_s_a_spec_k_n = \
				self.verify_solution(self.bellmanOpt, self.nu_s_ver, next_policy, self.bellConstr,
									 nu_s_spec=self.nu_s_ver_spec, constrBellmanSpec=self.bellConstrDict)
			ent_cost_n = 0
			self.checking_policy_time += time.time() - curr_time

			# Check if the new policy improves over the last obtained policy
			if self._pomdp.has_sideinfo and spec_cost_n - self._sat_thresh < -1e-6:  # The spec properties are not satisfied
				ent_cost_n += (spec_cost_n - self._sat_thresh) * self._options.mu * self._options.mu_spec
			
			# Add the actual reward
			curr_rew = sum(coeff * val * self._options.mu for coeff, val in self.compute_expected_reward(nu_s_a_k_n, weight))
			ent_cost_n += curr_rew

			if ent_cost_n > ent_cost:
				policy_k = next_policy
				nu_s_k, nu_s_spec_k = nu_s_k_n, nu_s_spec_k_n
				nu_s_a_k, nu_s_a_spec_k = nu_s_a_k_n, nu_s_a_spec_k_n
				ent_cost, spec_cost = ent_cost_n, spec_cost_n
				opt_reward = curr_rew
				trust_region = np.minimum(trustRegion['aug'](trust_region), self._max_trust_region)
			else:
				trust_region = trustRegion['red'](trust_region)
				if self._options.verbose:
					print("[Iter {}: ----> Reject the current step]".format(i))

			if self._options.verbose or (trust_region < trustRegion['lim']) or (i==self._options.maxiter-1):
				# print("[Iter {}]: Finding the state and state-action visitation count given a policy".format(i))
				# print("[Iter {}]: Optimal policy: {}".format(i, policy_k))
				print("[Iter {}]: Reward attained {}, Spec SAT : {}, Trust region : {}".format(i, curr_rew/self._options.mu_rew, spec_cost, trust_region))
				if self._pomdp.has_sideinfo:
					print("[Iter {}]: Number of steps : {}".format(i, sum(
						nu_s_val for s, nu_s_val in nu_s_spec_k.items())))
				print("[Iter {}]: Update time : {}s, Checking time : {}s, Solve time: {}s".format(i,
																								  self.update_constraint_time,
																								  self.checking_policy_time,
																								  self.total_solve_time))
				# print("[Iter {}]: Trust region : {}".format(i, trust_region))

			# In case it is required to save the policy
			if save_info is not None and ( (i == self._options.maxiter) or (i > 0 and i % save_info[0] == 0)):
				stats_solver['time'] = self.total_solve_time+self.update_constraint_time+self.checking_policy_time
				stats_solver['rew'] = opt_reward / self._options.mu_rew
				stats_solver['sat_spec'] = spec_cost
				stats_solver['n_iter'] = i+1
				stats_solver['discount'] = self._options.discount
				stats_solver['obs_based'] = True
				with open(save_info[1]+'_pol.json', 'w') as fp:
					json.dump(policy_k, fp, sort_keys=True, indent=4)
				with open(save_info[1]+'_weight.json', 'w') as fp:
					json.dump({'true weight' : save_info[2], 'learned weight' : weight}, fp, sort_keys=True, indent=4)
				with open(save_info[1]+'_stats.json', 'w') as fp:
					json.dump(stats_solver, fp, sort_keys=True, indent=4)

			if trust_region < trustRegion['lim']:
				if self._options.verbose:
					print("[Iter {}: ----> Min trust value reached]".format(i))
				break

		return policy_k

	def from_reward_to_optimal_policy_mdp_lp(self, weight, gamma=1, save_info=None):
		""" Given the weight for each feature functions in the underlying MDP model,
			compute the optimal policy that maximizes the expected reward
			while satisfying the specifications
			:param weight : A dictionary with its keys being the reward feature name
							and its value be a dictionary with (obs, act) as the key
								and the associated reward at the value
		"""
		# Sanity check
		assert not (gamma < 1 and self._pomdp.has_sideinfo)

		# Create the optimization problem
		mOpt = gp.Model('Optimal Policy of the MDP with Gurobi Solver')
		self.total_solve_time = 0  # Total time elapsed
		self.init_encoding_time = 0  # Time for encoding the full problem
		self.update_constraint_time = 0
		self.checking_policy_time = 0

		stats_solver = dict()
		self._pomdp.write_to_dict(stats_solver)

		if self._options.verbose:
			print('Initialize Linear subproblem to be solved at iteration k')

		# Util functions for one/two-dimension dictionaries of positve gurobi variable
		buildVar1D = lambda pb, data, idV: {s: pb.addVar(lb=0, name='{}[{}]'.format(idV, s)) for s in data}
		buildVar2D = lambda pb, data, idV: {s: {a: pb.addVar(lb=0, name='{}[{},{}]'.format(idV, s, a)) for a in dataVal}
											for s, dataVal in data.items()}

		# Store the current time for compute time logging
		curr_time = time.time()

		# Store the state visitation count
		nu_s = buildVar1D(mOpt, self._pomdp.states, 'nu')
		# Store the state action visitation count
		nu_s_a = buildVar2D(mOpt, self._pomdp.states_act, 'nu')
		# Add the constraints between the state visitation count and the state-action visitation count
		self.constr_state_action_to_state_visition(mOpt, nu_s, nu_s_a, name='vis_count')
		# Add the bellman equation constraints
		self.constr_bellman_flow(mOpt, nu_s, nu_s_a=nu_s_a, sigma=None, gamma=gamma, name='bellman')

		# Create a slack variable for statisfiability of the spec if any
		slack_spec = mOpt.addVar(lb=0, name='s2') if self._pomdp.has_sideinfo else 0

		# Constraint for satisfaction of the formula
		if self._pomdp.has_sideinfo:
			mOpt.addLConstr(gp.LinExpr([*((1, nu_s[s]) for s in self._pomdp.prob1A), (1, slack_spec)]),
							gp.GRB.GREATER_EQUAL, self._sat_thresh, name='sat_constr')
		self.init_encoding_time += time.time() - curr_time

		# Define the parameters used by Gurobi for this problem
		mOpt.Params.OutputFlag = self._options.verbose_solver
		mOpt.Params.Presolve = 2  # More aggressive presolve step
		# mOpt.Params.FeasibilityTol = 1e-8
		# mOpt.Params.OptimalityTol = 1e-6
		# mOpt.Params.BarConvTol = 1e-6

		# Build the objective function -> Negative expected reward
		linExprCost = gp.LinExpr(self.compute_expected_reward(nu_s_a, weight))
		# Add to the cost the penalization from the cost of staisfying the spec
		if self._pomdp.has_sideinfo:
			linExprCost.add(slack_spec, self._options.mu_spec)

		# Set the objective function -> Negative sign to compensate the outout of compute_expected_reward
		mOpt.setObjective(linExprCost, gp.GRB.MAXIMIZE)

		# Solve the problem
		curr_time = time.time()
		mOpt.optimize()
		self.total_solve_time += time.time() - curr_time

		# Do some printing
		if mOpt.status == gp.GRB.OPTIMAL:
			print('[Time used to build the full Model : {}]'.format(self.init_encoding_time))
			print('[Total solving time : {}]'.format(self.total_solve_time))
			print('[Optimal expected reward : {}]'.format(mOpt.objVal / self._options.quotmu))
			if self._pomdp.has_sideinfo:
				print('[Satisfaction of the formula = {}]'.format(sum(nu_s[s].x for s in self._pomdp.prob1A)))
				print('[Slack value spec = {}]'.format(slack_spec.x))
				print('[Number of steps : {}]'.format(sum(nu_s_val.x for s, nu_s_val in nu_s.items())))
			# print('[Optimal policy : {}]'.format(
			# 	{s: {a: (p.x / nu_s[s].x if nu_s[s].x > ZERO_NU_S else 1.0 / len(actVal)) for a, p in actVal.items()}
			# 	 for s, actVal in nu_s_a.items()}))
		res_pol = {s: {a: (p.x / nu_s[s].x if nu_s[s].x > ZERO_NU_S else 1.0 / len(actVal)) for a, p in actVal.items()} for
				s, actVal in nu_s_a.items()}

		# In case it is required to save the policy
		if save_info is not None:
			stats_solver['time'] = self.total_solve_time+self.update_constraint_time+self.checking_policy_time
			stats_solver['rew'] = mOpt.objVal / self._options.quotmu
			stats_solver['discount'] = self._options.discount
			stats_solver['obs_based'] = False
			with open(save_info[1]+'_pol.json', 'w') as fp:
				json.dump(res_pol, fp, sort_keys=True, indent=4)
			with open(save_info[1]+'_weight.json', 'w') as fp:
				json.dump({'true weight' : save_info[2], 'learned weight' : weight}, fp, sort_keys=True, indent=4)
			with open(save_info[1]+'_stats.json', 'w') as fp:
				json.dump(stats_solver, fp, sort_keys=True, indent=4)
		return res_pol

	def init_optimization_problem(self, mOpt, noLinearization=False, checkOpt=None):
		""" Initialize the linearized subproblem to solve at iteration k
			and parametrized the constraints induced by the linearization
			such that they can be modified without being recreated/deleted later
			:param m_opt : A gurobi model
			:param noLinearization : Useful when solving the problem using sequential convex optimization
			:param checkOpt : If not None, it is a gurobi model for finding feasible solution of the bellman equation
		"""
		if self._options.verbose:
			print('Initialize Linear subproblem to be solved at iteration k')

		# Util functions for one/two-dimension dictionaries of positve gurobi variable
		buildVar1D = lambda pb, data, idV: {s: pb.addVar(lb=0, name='{}[{}]'.format(idV, s)) for s in data}
		buildVar2D = lambda pb, data, idV: {s: {a: pb.addVar(lb=0, name='{}[{},{}]'.format(idV, s, a)) for a in dataVal}
											for s, dataVal in data.items()}

		# Store the current time for compute time logging
		curr_time = time.time()

		# Store the state visitation count
		nu_s = buildVar1D(mOpt, self._pomdp.states, 'nu')
		nu_s_spec = buildVar1D(mOpt, self._pomdp.states, 'nu_s') if self._pomdp.has_sideinfo else dict()

		# Store the state action visitation count
		nu_s_a = buildVar2D(mOpt, self._pomdp.states_act, 'nu')
		nu_s_a_spec = buildVar2D(mOpt, self._pomdp.states_act, 'nu_s') if self._pomdp.has_sideinfo else dict()

		# Policy variable as a function of obs and state
		sigma = buildVar2D(mOpt, self._pomdp.obs_act, 'sig')

		# Add the constraints implied by the policy -> sum_a sigma[o,a] == 1
		for o, actDict in sigma.items():
			mOpt.addLConstr(gp.LinExpr([(1, sigma_o_a) for a, sigma_o_a in actDict.items()]),
							gp.GRB.EQUAL, 1,
							name='sum_pol[{}]'.format(o)
							)
			for a, sigma_o_a in actDict.items():
				mOpt.addConstr(sigma_o_a >= self._options.graph_epsilon)

		# Add the constraints between the state visitation count and the state-action visitation count
		self.constr_state_action_to_state_visition(mOpt, nu_s, nu_s_a, name='vis_count')
		if self._pomdp.has_sideinfo:
			self.constr_state_action_to_state_visition(mOpt, nu_s_spec, nu_s_a_spec, name='vis_count_spec')

		# Add the bellman equation constraints
		self.constr_bellman_flow(mOpt, nu_s, nu_s_a=nu_s_a, sigma=None, gamma=self._options.discount, name='bellman')
		if self._pomdp.has_sideinfo:
			self.constr_bellman_flow(mOpt, nu_s_spec, nu_s_a=nu_s_a_spec, sigma=None, gamma=1, name='bellman_spec')

		# Create a slack variable for statisfiability of the spec if any
		slack_spec = mOpt.addVar(lb=0, name='s2') if self._pomdp.has_sideinfo else 0

		# Constraint for satisfaction of the formula
		if self._pomdp.has_sideinfo:
			mOpt.addLConstr(gp.LinExpr([*((1, nu_s_spec[s]) for s in self._pomdp.prob1A), (1, slack_spec)]),
							gp.GRB.GREATER_EQUAL, self._sat_thresh, name='sat_constr')

		if noLinearization:
			# Add the bilinear constraint into the problem
			self.constr_bilinear(mOpt, nu_s, nu_s_a, sigma, name='bilConstr')
			if self._pomdp.has_sideinfo:
				self.constr_bilinear(mOpt, nu_s_spec, nu_s_a_spec, sigma, name='bilConstr_spec')
			self.init_encoding_time += time.time() - curr_time
			return nu_s, nu_s_spec, nu_s_a, nu_s_a_spec, sigma, slack_spec

		# If linearization of the nonconvex constraint is enbale, create the slack variables
		# Create the slack variables that will be used for linearizing constraints
		slack_nu_p = buildVar2D(mOpt, self._pomdp.states_act, 's1p')
		slack_nu_n = buildVar2D(mOpt, self._pomdp.states_act, 's1n')
		slack_nu_p_spec = buildVar2D(mOpt, self._pomdp.states_act, 's2p') if self._pomdp.has_sideinfo else dict()
		slack_nu_n_spec = buildVar2D(mOpt, self._pomdp.states_act, 's2n') if self._pomdp.has_sideinfo else dict()

		# Add the parametrized linearized constraint
		constrLin = self.constr_linearized_bilr(mOpt, nu_s, nu_s_a, sigma, slack_nu_p, slack_nu_n, name='LinBil')
		constrLinSpec = self.constr_linearized_bilr(mOpt, nu_s_spec, nu_s_a_spec, sigma, slack_nu_p_spec,
													slack_nu_n_spec, name='LinBil_spec') \
			if self._pomdp.has_sideinfo else dict()

		# Add the parameterized trust region constraint on the policy
		constrTrustReg = self.constr_trust_region(mOpt, sigma)

		# Create variables of the problem to find admissible visitation count given a policy
		assert checkOpt is not None
		nu_s_ver = buildVar1D(checkOpt, self._pomdp.states, 'nu')
		nu_s_ver_spec = buildVar1D(checkOpt, self._pomdp.states, 'nu_s') if self._pomdp.has_sideinfo else dict()

		# Create a dummy policy to initialize parameterized the bellman constraint
		dummy_pol = {o: {a: 1.0 / len(actList) for a in actList} for o, actList in self._pomdp.obs_act.items()}

		# Add the bellman constraint knowing the policy
		bellConstr = self.constr_bellman_flow(checkOpt, nu_s_ver, nu_s_a=None, sigma=dummy_pol,
											  gamma=self._options.discount, name='bellman')
		bellConstrDict = dict()
		if self._pomdp.has_sideinfo:
			bellConstrDict = self.constr_bellman_flow(checkOpt, nu_s_ver_spec, nu_s_a=None, sigma=dummy_pol, gamma=1,
													  name='bellman_spec')

		# Save the encoding compute time the of the problem
		self.init_encoding_time += time.time() - curr_time
		return nu_s, nu_s_spec, nu_s_a, nu_s_a_spec, sigma, slack_spec, \
			   slack_nu_p, slack_nu_n, slack_nu_p_spec, slack_nu_n_spec, \
			   constrLin, constrLinSpec, constrTrustReg, \
			   nu_s_ver, nu_s_ver_spec, bellConstr, bellConstrDict

	def nacc(self, pred_s, s, tProb, gamma):
		""" Return True if the given state s is not in prob1A and prob0E
		"""
		# if self._pomdp.has_sideinfo:
		# 	return s not in self._pomdp.prob1A and s not in self._pomdp.prob0E
		# else:
		# 	return True
		if self._pomdp.has_sideinfo and gamma == 1: # side info and gamma == 1
			# gamma * (tProb if self.nacc(pred_s) else 0)
			# return tProb if (pred_s != s or tProb < 1) else 0
			return tProb if (pred_s not in self._pomdp.prob1A and pred_s not in self._pomdp.prob0E) else 0
		else: # If not side info or gamma < 1, accepting state shouldn't be ignored in bellman constraint
			return gamma * tProb

	def constr_state_action_to_state_visition(self, mOpt, nu_s, nu_s_a, name='vis_count'):
		""" Encode the constraint between nu_s and nu_s_a
			Basically, compute nu_s = sum_a nu_s_a
			:param mOpt : the Gurobi model of the problem
			:param nu_s : state visitation count
			:param nu_s_a : state-action visitation count
		"""
		for s, nu_s_v in nu_s.items():
			mOpt.addLConstr(gp.LinExpr([*((1, nu_s_a_val) for a, nu_s_a_val in nu_s_a[s].items()), (-1, nu_s_v)]),
							gp.GRB.EQUAL, 0,
							name='{}[{}]'.format(name, s)
							)

	def constr_bellman_flow(self, mOpt, nu_s, nu_s_a=None, sigma=None, gamma=1, name='bellman'):
		""" Compute for all states the constraints by the bellman equation.
			This function allows only one of nu_s_a or sigma to be None
			:param mOpt : The gurobi model of the problem
			:param nu_s : state visitation count
			:param nu_s_a : state-action visitation count
			:param sigma : policy (not a Gurobi) variable
			:param gamma : the discount factor
		"""
		assert (nu_s_a is None and sigma is not None) or (nu_s_a is not None and sigma is None)

		dictConstr = dict()
		for s, nu_s_v in nu_s.items():
			# Probabilities of perceiving an obs from state s
			obs_distr = self._pomdp.obs_state_distr[s]
			if sigma is not None:  # If the policy is given
				dicCoeff = self.extract_varcoeff_from_bellman(s, sigma, gamma)
				val_expr = gp.LinExpr([
					*((sum(coeffV), nu_s[pred_s]) for pred_s, coeffV in dicCoeff.items()),
					(-1, nu_s_v)
				])
			else:  # if the state-action visitation count is given
				val_expr = gp.LinExpr([*((self.nacc(pred_s, s, tProb, gamma), nu_s_a[pred_s][a]) \
										 for (pred_s, a, tProb) in self._pomdp.pred.get(s, [])
										 ),
									   (-1, nu_s_v)
									   ])
			# Add the linear constraint -> RHS corresponds to the probability the state is an initial state
			dictConstr[s] = mOpt.addLConstr(val_expr, gp.GRB.EQUAL,
											-self._pomdp.state_init_prob.get(s, 0), name="{}[{}]".format(name, s))
		return dictConstr

	def constr_bilinear(self, mOpt, nu_s, nu_s_a, sigma, name='bilinear'):
		""" Enforce  the biliear constraint
			:param mOpt : The gurobi model of the problem
			:param nu_s : state visitation count
			:param nu_s_a : state-action visitation count
			:param sigma : policy variable
		"""
		for s, nu_s_v in nu_s.items():
			obs_distr = self._pomdp.obs_state_distr[s]
			for a, nu_s_a_val in nu_s_a[s].items():
				mOpt.addConstr(nu_s_a_val - nu_s_v * sum(sigma[o][a] * p for o, p in obs_distr.items()) == 0,
							   name='{}[{},{}]'.format(name, s, a))

	def constr_linearized_bilr(self, mOpt, nu_s, nu_s_a, sigma, slack_nu_p, slack_nu_n, name='LinBil'):
		""" Return a parametrized linearized constraint of the
			bilinear constraint involving nu_s, nu_s_a, and sigma
			Each term sigma[o][a] is associated with the coefficient O(o|s)*nu_s^k
			Each term  nu_s[s] is associated with the coefficient sum_o O(o|s) sigma[o|s]^k
			The RHS (constant) is associated with the value
			:param mOpt : The gurobi model of the problem
			:param nu_s : state visitation count
			:param nu_s_a : state-action visitation count
			:param sigma : policy variable
			:param slack_nu_p : (pos) slack variable to render the linear constraint feasible
			:param slack_nu_n : (neg) slack variable to render the linear constraint feasible
		"""
		dictConstr = list()
		for s, nu_s_v in nu_s.items():
			obs_distr = self._pomdp.obs_state_distr[s]
			for a, nu_s_a_val in nu_s_a[s].items():
				val_expr = gp.LinExpr([*((1, sigma[o][a]) for o, p in obs_distr.items()), \
									   (1, nu_s_v), (-1, nu_s_a_val), \
									   (1, slack_nu_p[s][a]), (-1, slack_nu_n[s][a])])
				dictConstr.append(((s, a), mOpt.addLConstr(val_expr, gp.GRB.EQUAL, 0,
													 name='{}[{},{}]'.format(name, s, a)))
								)
		return dictConstr

	def constr_trust_region(self, mOpt, sigma):
		""" Parametrized the linear constraint by the trust region and returned
			the saved constraint to be modified later in the algorithm
			:param mOpt : Gurobi model
			:param sigma : observation based policy
		"""
		dictConstr = list()
		for o, aDict in sigma.items():
			for a, pol_val in aDict.items():
				dictConstr.append(((o, a, True), mOpt.addLConstr(pol_val,
														   gp.GRB.GREATER_EQUAL, 0,
														   name='trust_inf[{},{}]'.format(o, a)))
								)
				dictConstr.append(((o, a, False),mOpt.addLConstr(pol_val,
															gp.GRB.LESS_EQUAL, 1.0,
															name='trust_sup[{},{}]'.format(o, a))
								))
		return dictConstr

	def compute_expected_reward(self, nu_s_a, theta):
		""" Provide a linear expression for the expected reward
			given a weight theta and the feature matching
			theta^T(feat_match- expected reward)
			:param nu_s_a : the state-action visitation count
			:param theta : the weight for the feature function
			:param featmatch : the expected feature matching
		"""
		val_expr = [(rew[(o, a)] * p * theta[rName] * self._options.quotmu, nu_s_a_val) \
					for rName, rew in self._pomdp.reward_features.items() \
					for s, nu_s_a_t in nu_s_a.items() \
					for a, nu_s_a_val in nu_s_a_t.items()
					for o, p in self._pomdp.obs_state_distr[s].items()
					]
		# if feat_match is not None:
		#   val_expr.addConstant(
		#       sum(theta_val*feat_val \
		#               for r_name, theta_val in theta.items() \
		#                   for (o,a), feat_val in feat_match[r_name].items())
		#   )
		return val_expr

	def compute_quad_featmatch(self, nu_s_a, feat_match):
		""" Compute the gurobi quadratic expression for
			||(feat_match- expected reward)||^2
			:param nu_s_a : the state-action visitation count gurobi variable
			:param featmatch : the expected feature matching
		"""
		quadTerm = 0
		for rname, featVal in feat_match.items():
			rewDict = self._pomdp.reward_features[rname]
			tempV = gp.LinExpr(
				[(rewDict[(o, a)] * p, nu_s_a_val) \
				 for s, nu_s_a_t in nu_s_a.items() \
				 for a, nu_s_a_val in nu_s_a_t.items()
				 for o, p in self._pomdp.obs_state_distr[s].items()]
			)
			# Add - feat_match

			tempV.addConstant(-featVal)

			quadTerm += tempV * tempV
		return quadTerm

	def compute_quad_featmatch_value(self, nu_s_a, feat_match):
		""" Compute the  actual value of
			||(feat_match- expected reward)||^2, given the value of the state-action
			visitation count and the feat match
			:param nu_s_a : the state-action visitation count
			:param featmatch : the expected feature matching
		"""
		quadTerm = 0
		for rname, featVal in feat_match.items():
			rewDict = self._pomdp.reward_features[rname]
			#print(nu_s_a,"quad_featmatch")
			tempV = sum(
				[(rewDict[(o, a)] * p * nu_s_a_val) \
				 for s, nu_s_a_t in nu_s_a.items() \
				 for a, nu_s_a_val in nu_s_a_t.items()
				 for o, p in self._pomdp.obs_state_distr[s].items()]
			)
			tempV += -featVal
			quadTerm += tempV * tempV
		return quadTerm

	def update_linearized_bil_sonstr(self, mOpt, constrLin, nu_s_past, sigma_past, nu_s, sigma):
		""" Update the constraints implied by the linearization of the bilinear constraint
			around the solution of the past iterations
		"""
		for ((s, a), constrV) in constrLin:
			curr_nu = nu_s_past[s]
			obs_distr = self._pomdp.obs_state_distr[s]
			prod_obs_prob_sigma_past = sum(p * sigma_past[o][a] for o, p in obs_distr.items())
			constrV.RHS = curr_nu * prod_obs_prob_sigma_past
			# Update all the coefficients associated to sigma[o][a]
			for o, p in obs_distr.items():
				mOpt.chgCoeff(constrV, sigma[o][a], curr_nu * p)
			# Update the coefficient associated to nu_s[s]
			mOpt.chgCoeff(constrV, nu_s[s], prod_obs_prob_sigma_past)

	def update_constr_and_trust_region(self, mOpt, constrLin, constrLinSpec, constrTrust,
									   nu_s_past, nu_s_past_spec, sigma_past,
									   nu_s, nu_s_spec, sigma, currTrust, only_trust=False):
		""" Update the constraint implied by the linearization of the bilinear constraint and
			the trust region constraint using the solutions of the past iteration
			:param mOpt : A gurobi model of the problem
			:param constrLin : The linearized constraints
			:param constrLinSpec : The linearized constraints for the specifications
			:param constrTrust : The trust region constraints
			:nu_s_past : state visitation count obtained at last iteration
			:nu_s_past_spec : state visitation count obtained at last iteration for the spec
			:sigma_past : policy obtained at the last iteration
			:nu_s : state visitation to be obtained at the current iteration
			:nu_s_spec : state visitation to be obtained at the current iteration from the spec
			:sigma : policy to be obtained at the current iteration
			:currTrust : the current trust region
		"""
		# Start by updating the constraint by the trust region
		inv_trust = 1.0 / currTrust
		for ((o, a, tRhs), constrV) in constrTrust:
			# tRhs is True correspons to lower bound of the trust region and inversely
			constrV.RHS = sigma_past[o][a] * ( inv_trust if tRhs else currTrust)

		# If update only the trust region is allowed then return
		if only_trust:
			return

		# Now update the constraints from the linearization
		self.update_linearized_bil_sonstr(mOpt, constrLin, nu_s_past, sigma_past, nu_s, sigma)
		if self._pomdp.has_sideinfo:
			self.update_linearized_bil_sonstr(mOpt, constrLinSpec, nu_s_past_spec, sigma_past, nu_s_spec, sigma)

	def compute_entropy_cost(self, nu_s_past, nu_s_a_past, nu_s, nu_s_a, slack_nu_p, slack_nu_n,
							 slack_nu_p_spec=dict(), slack_nu_n_spec=dict(), slack_spec=None,
							 addEntropy=True):
		""" Return a linearized entropy function around the solution of the past
			iteration given by nu_s_oast and nu_s_a_past.
			:param nu_s_past : state visitation at the past iteration
			:param nu_s_a_past : state action visitation at the past iteration
			:param nu_s : state visitation
			:param nu_s_a : state action visitation
			.
			.
		"""
		# Save the pair coeff,variable for each term in the lienar cost function
		listCostTerm = list()
		if self._pomdp.has_sideinfo:  # Cost associated to violating the spec constraints
			listCostTerm.append((-self._options.mu_spec, slack_spec))

		for s, nu_s_val in nu_s.items():
			# Don't consider states with zero probability of satisfying the spec
			# if self._pomdp.has_sideinfo and (s in self._pomdp.prob1A or s in self._pomdp.prob0E):
			# 	continue
			nu_s_past_val, nu_s_a_past_sval, slack_nu_p_sval, slack_nu_n_sval = nu_s_past[s], nu_s_a_past[s], slack_nu_p[s], slack_nu_n[s]
			if self._pomdp.has_sideinfo:
				slack_nu_p_spec_sval, slack_nu_n_spec_sval = slack_nu_p_spec[s], slack_nu_n_spec[s]
			for a, nu_s_a_val in nu_s_a[s].items():
				nu_s_a_past_saval = nu_s_a_past_sval[a]
				# First, add the entropy terms
				if nu_s_past_val > 0 and nu_s_a_past_saval > 0 and addEntropy:
					nu_ratio = nu_s_a_past_saval / nu_s_past_val
					listCostTerm.append((nu_ratio / self._options.mu, nu_s_val))
					listCostTerm.append((-(np.log(nu_ratio) + 1) / self._options.mu, nu_s_a_val))

				# Then add the cost associated to the linearization errors
				listCostTerm.append((-1, slack_nu_p_sval[a]))
				listCostTerm.append((-1, slack_nu_n_sval[a]))
				if self._pomdp.has_sideinfo:
					listCostTerm.append((-1, slack_nu_p_spec_sval[a]))
					listCostTerm.append((-1, slack_nu_n_spec_sval[a]))

		return listCostTerm

	def extract_varcoeff_from_bellman(self, state, policy_val, gamma=1, addSpecCoeff=False):
		""" Utility function that provides given a policy, the coefficien of each
			variable nu_s[pred(state)] (in the linear expression) for all predecessor of state
			:param state : the current state of the pomdp
			:param policy_val : the underlying policy
			:param gamma : discount factor
		"""
		dictPredCoeff = dict()  # Variable to store the coefficient associated to nu_s[pred(state)]
		# Get the from the past optimal value, the coefficient for each variable in the bellman constraint
		for (pred_s, a, tProb) in self._pomdp.pred.get(state, []):
			if pred_s not in dictPredCoeff:

				dictPredCoeff[pred_s] = list() if not addSpecCoeff else (list(), list())
			for o, p in self._pomdp.obs_state_distr[pred_s].items():
				# dictPredCoeff[pred_s].append(policy_val[o][a] * p * gamma * (tProb if self.nacc(pred_s) else 0))
				if not addSpecCoeff:
					dictPredCoeff[pred_s].append(policy_val[o][a] * p * self.nacc(pred_s, state, tProb, gamma))
				else:
					(l1, l2) = dictPredCoeff[pred_s]
					l1.append(policy_val[o][a] * p * self.nacc(pred_s, state, tProb, gamma))
					l2.append(policy_val[o][a] * p * self.nacc(pred_s, state, tProb, 1))
		return dictPredCoeff

	def verify_solution(self, mOpt, nu_s, policy_val, constrBellman,
						nu_s_spec=dict(), constrBellmanSpec=dict()):
		""" Given a policy that is solution of the past iteration,
			this function computes the corresponding state and state-action
			visitation count that satisfy the bellman equation flow.
			Then, it computes the objective attained by this policy.
			:param mOpt : The gurobi model encoding the solution of the bellman flow
			:param nu_s : state visitation count of mOpt problem
			:param policy_val : The optimal policy obtaine durint this iteration
			:param constrBellman : constraint representing the bellman equation
		"""
		# Update the optimization problem with the given policy
		for s in nu_s:
			dicCoeff = self.extract_varcoeff_from_bellman(s, policy_val, gamma=self._options.discount, 
															addSpecCoeff=self._pomdp.has_sideinfo)
			for pred_s, coeffV in dicCoeff.items():
				sum_coeff = sum(coeffV[0]) if self._pomdp.has_sideinfo else sum(coeffV)
				mOpt.chgCoeff(constrBellman[s], nu_s[pred_s], sum_coeff - (1 if pred_s == s else 0))
				if self._pomdp.has_sideinfo:  # Do a scale back to gamma = 1
					sum_coeff_undis = sum(coeffV[1]) # sum_coeff / self._options.discount
					mOpt.chgCoeff(constrBellmanSpec[s], nu_s_spec[pred_s], \
								  sum_coeff_undis - (1 if pred_s == s else 0))

		# Now solve the problem to get the corresponding state, state-action visitation count
		mOpt.setObjective(0, gp.GRB.MINIMIZE)
		mOpt.optimize()

		# Save the resulting state and state_action value
		res_nu_s = {s: val.x for s, val in nu_s.items()}
		res_nu_s_a = {s: {a: res_nu_s[s] * sum(p * policy_val[o][a] for o, p in self._pomdp.obs_state_distr[s].items()) \
						  for a in actList} \
					  for s, actList in self._pomdp.states_act.items()
					  }
		# Save the resulting state and state_action value for specifications
		res_nu_s_spec = dict()
		res_nu_s_a_spec = dict()
		if self._pomdp.has_sideinfo:
			res_nu_s_spec = {s: val.x for s, val in nu_s_spec.items()}
			res_nu_s_a_spec = {
				s: {a: res_nu_s_spec[s] * sum(p * policy_val[o][a] for o, p in self._pomdp.obs_state_distr[s].items()) \
					for a in actList} \
				for s, actList in self._pomdp.states_act.items()
				}

		# Get the incurred cost
		spec_cost = 0
		if self._pomdp.has_sideinfo:
			spec_cost = sum(res_nu_s_spec[s] for s in self._pomdp.prob1A)

		# ent_cost = sum(0 if (res_nu_s_a_val <= ZERO_NU_S or res_nu_s_val <= ZERO_NU_S)
		ent_cost = sum(0 if (res_nu_s_a_val <= ZERO_NU_S or res_nu_s_val <= ZERO_NU_S)
					   else (-np.log(res_nu_s_a_val / res_nu_s_val) * res_nu_s_a_val) \
					   for s, res_nu_s_val in res_nu_s.items()\
					   for a, res_nu_s_a_val in res_nu_s_a[s].items()
					   )
		return ent_cost, spec_cost, res_nu_s, res_nu_s_a, res_nu_s_spec, res_nu_s_a_spec


if __name__ == "__main__":
	# Set a random seed for reproducibility
	np.random.seed(201)

	# Customize maze with loop removed for the poison state
	pModel = PrismModel("parametric_maze_stochastic.pm", ["P=? [F \"target\"]"])

	# Comoute policy that maximize the max entropy
	mOptions = OptOptions(mu=1e4, mu_spec=1e4, maxiter=100, maxiter_weight=100,
						  graph_epsilon=0, discount=0.9, verbose=True)
	irlPb = IRLSolver(pModel, sat_thresh=0.95, init_trust_region=1.25, options=mOptions)
	weight = {r_name: 0.0 for r_name, rew in pModel.reward_features.items()}
	irlPb.compute_maxent_policy_via_scp(weight, init_problem=True)

	# Find the policy that maximizes the expected reward
	_pModel = PrismModel("parametric_maze_stochastic.pm")
	mOptions = OptOptions(mu=1, mu_spec=1e4, maxiter=100, maxiter_weight=20,
						  graph_epsilon=0, discount=0.9, verbose=True)
	# Build an instance of the IRL problem
	irlPb = IRLSolver(_pModel, sat_thresh=0.95, init_trust_region=1.25, options=mOptions)
	weight = {'poisonous': 10, 'total_time': 1, 'goal_reach': 100}
	# # pol_val = irlPb.from_reward_to_optimal_policy_nonconvex_grb(weight)
	# # pol_val = irlPb.from_reward_to_policy_via_scp(weight)
	# # obs_based = True
	pol_val = irlPb.from_reward_to_optimal_policy_mdp_lp(weight, gamma=0.95)
	obs_based = False

	trajData, rewData = pModel.simulate_policy(pol_val, weight, 500, 1000,
											   obs_based=obs_based, stop_at_accepting_state=True)

	# Find the weight and solution
	mOptions = OptOptions(mu=1e4, mu_spec=1e4, maxiter=200, maxiter_weight=50, rho= 1.0,
						  graph_epsilon=0, discount=0.9, verbose=False, verbose_weight=True)
	# Build an instance of the IRL problem
	irlPb = IRLSolver(pModel, sat_thresh=0.95, init_trust_region=1.1, rew_eps=1e-3, options=mOptions)
	irlPb.solve_irl_pomdp_given_traj(trajData, includeQuadCost=True)
