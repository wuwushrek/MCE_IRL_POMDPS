import numpy as np
import gurobi as gp

from .parser_pomdp import POMDP, PrismModel
import time

#Class for setting up options for the optimization problem
class OptOptions:
	def __init__(self, mu=1e4, mu_spec=1e4, maxiter=100, 
					graph_epsilon=1e-3, discount=0.9, verbose=True):
		"""
		Returns the float representation for a constant value
		:param mu: parameter for putting penalty on slack variables type: float
		:param maxiter: max number of iterations type: integer
		:param graph_epsilon: the min probability of taking an action at each observation type: float
		:param silent: print stuff from gurobi or not type: boolean
		:param discount: discount factor type:float
		:param verbose: Enable some logging
		:return:
		"""
		self.mu = mu
		self.mu_spec=mu_spec
		self.maxiter = maxiter
		self.graph_epsilon = graph_epsilon
		self.discount=discount
		self.verbose = verbose
		if graph_epsilon<0:
			raise RuntimeError("graph epsilon should be larger than 0")
		if discount<0 or discount>=1:
			raise RuntimeError("discount factor should be between 0 and 1")
		if mu<=0:
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
		init_trust_region: float = 4.0,
		options: OptOptions = OptOptions(),
		) -> None:

		# Attributes to check performances of building, solving the problems
		self.total_solve_time = 0       # Total time elapsed
		self.init_encoding_time = 0     # Time for encoding the full problem
		self.update_constraint_time = 0 # Total time for updating the constraints
		self.checking_policy_time = 0   # Total time for finding nu_s and nu_sa gievn a policy

		# Save the initial trust region
		self._trust_region = init_trust_region
		self._sat_thresh = sat_thresh
		self._options = options
		self._pomdp = pomdp

		# # Initialize the LInearized subproblem
		# self._lin_encoding = gp.Model('Linearized problem')
		# self.init_linearized_problem(pomdp, options, init_trust_region, sat_thresh)

	def compute_optimal_policy_nonconvex_grb(self, weight):
		""" Given the weight for each feature functions in the POMDP model,
			compute the optimal policy that maximizes the expected reward
			while satisfying
			:param weight : A dictionary with its keys being the reward feature name
							and its value be a dictionary with (obs, act) as the key
								and the associated reward at the value
		"""
		# Create the optimization problem
		mOpt = gp.Model('Optimal Policy with NonConvex Gurobi Solver')
		self.total_solve_time = 0       # Total time elapsed
		self.init_encoding_time = 0     # Time for encoding the full problem
		nu_s, nu_s_spec, nu_s_a, nu_s_a_spec, sigma, slack_spec = \
					self.init_optimization_problem(mOpt, noLinearization=True)

		# Define the parameters used by Gurobi for this problem
		mOpt.Params.OutputFlag = self._options.verbose
		mOpt.Params.Presolve = 2 # More aggressive presolve step
		mOpt.Params.FeasibilityTol = 1e-6
		mOpt.Params.OptimalityTol = 1e-6
		mOpt.Params.BarConvTol = 1e-6
		mOpt.Params.NonConvex = 2

		# Build the objective function 
		linExprCost = self.compute_expected_reward(nu_s_a, weight, feat_match=None)
		# Add to the cost the penalization from the cost of staisfying the spec
		if self._pomdp.has_sideinfo:
			linExprCost.add(slack_spec,-self._options.mu_spec)

		# Set the objective function
		mOpt.setObjective(linExprCost, gp.GRB.MAXIMIZE)

		# Solve the problem
		curr_time = time.time()
		mOpt.optimize()
		self.total_solve_time += time.time()-curr_time

		# Do some printing
		if self._options.verbose and mOpt.status == gp.GRB.OPTIMAL:
			print ('[Time used to build the full Model : {}]'.format(self.init_encoding_time))
			print('[Total solving time : {}]'.format(self.total_solve_time))
			print('[Optimal expected reward : {}]'.format(mOpt.objVal))
			if self._pomdp.has_sideinfo:
				print('[Satisfaction of the formula = {}'.format( sum(nu_s_spec[s].x for s in self._pomdp.prob1A) ))
				print('[Slack value spec = {}]'.format(slack_spec.x))
				print('[Number of steps : {}]'.format(sum( nu_s_val.x for s, nu_s_val in nu_s_spec.items())))
			print('[Optimal policy : {}]'.format({ o : {p.x for a, p in actVal.items()} for o, actVal in sigma.items()}))



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
		buildVar1D = lambda pb, data, idV : { s : pb.addVar(lb=0, name='{}[{}]'.format(idV,s)) for s in data}
		buildVar2D = lambda pb, data, idV : { s : { a : pb.addVar(lb=0, name='{}[{},{}]'.format(idV,s,a)) for a in dataVal } 
												   for s, dataVal in data.items() }

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
			mOpt.addLConstr(gp.LinExpr([(1,sigma_o_a) for a, sigma_o_a in actDict.items()]),
							gp.GRB.EQUAL, 1, 
							name='sum_pol[{}]'.format(o)
							)

		# Add the constraints between the state visitation count and the state-action visitation count
		self.constr_state_action_to_state_visition(mOpt, nu_s, nu_s_a, name='vis_count')
		if self._pomdp.has_sideinfo:
			self.constr_state_action_to_state_visition(mOpt, nu_s_spec, nu_s_a_spec, name='vis_count_spec')

		# Add the bellman equation constraints
		self.constr_bellman_flow(mOpt, nu_s, nu_s_a=nu_s_a, sigma=None, gamma=self._options.discount, name='bellman')
		if self._pomdp.has_sideinfo:
			self.constr_bellman_flow(mOpt, nu_s_spec, nu_s_a=nu_s_a_spec, sigma=None, gamma=1.0, name='bellman_spec')

		# Create a slack variable for statisfiability of the spec if any
		slack_spec = mOpt.addVar(lb=0, name='s2') if self._pomdp.has_sideinfo else 0

		# Constraint for satisfaction of the formula
		if self._pomdp.has_sideinfo:
			mOpt.addLConstr(gp.LinExpr([*((1,nu_s_spec[s]) for s in self._pomdp.prob1A), (1,slack_spec)]),
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
		constrLinSpec = self.constr_linearized_bilr(mOpt, nu_s_spec, nu_s_a_spec, sigma, slack_nu_p_spec, slack_nu_n_spec,name='LinBil_spec') \
							if self._pomdp.has_sideinfo else dict()

		# Add the parameterized trust region constraint on the policy
		constrTrustReg = self.constr_trust_region(mOpt, sigma)

		# Create variables of the problem to find admissible visitation count given a policy
		assert checkOpt is not None
		nu_s_ver = buildVar1D(checkOpt, self._pomdp.states, 'nu')
		nu_s_ver_spec = buildVar1D(checkOpt, self._pomdp.states, 'nu_s') if self._pomdp.has_sideinfo else dict()

		# Create a dummy policy to initialize parameterized the bellman constraint
		dummy_pol = { s : { a : 1.0/len(actList) for a in actList} for s, actList in self._pomdp.obs_act.items()}

		# Add the bellman constraint knowing the policy
		self.constr_bellman_flow(checkOpt, nu_s_ver, nu_s_a=None, sigma=dummy_pol, gamma=self._options.discount,name='bellman')
		if self._pomdp.has_sideinfo:
			self.constr_bellman_flow(checkOpt, nu_s_ver_spec, nu_s_a=None, sigma=dummy_pol, gamma=1.0,name='bellman_spec')

		# Save the encoding compute time the of the problem
		self.init_encoding_time += time.time() - curr_time
		return 	nu_s, nu_s_spec, nu_s_a, nu_s_a_spec, sigma, slack_spec, \
				slack_nu_p, slack_nu_n, slack_nu_p_spec, slack_nu_n_spec,\
				constrLin, constrLinSpec, constrTrustReg,\
				nu_s_ver, nu_s_ver_spec

		# # Define the parameters used by Gurobi for this problem
		# self._lin_encoding.Params.OutputFlag = self._options.verbose
		# self._lin_encoding.Params.Presolve = 2 # More aggressive presolve step
		# # self._encoding.Params.Method = 2 # The problem is not really a QP
		# self._encoding.Params.Crossover = 0
		# self._encoding.Params.CrossoverBasis = 0
		# self._lin_encoding.Params.NumericFocus = 3 # Maximum numerical focus
		# self._encoding.Params.BarHomogeneous = 1 # No need for, our problem is always feasible/bound
		# # self._encoding.Params.ScaleFlag = 3
		# self._lin_encoding.Params.FeasibilityTol = 1e-6
		# self._lin_encoding.Params.OptimalityTol = 1e-6
		# self._lin_encoding.Params.BarConvTol = 1e-6

	def constr_state_action_to_state_visition(self, mOpt, nu_s, nu_s_a, name='vis_count'):
		""" Encode the constraint between nu_s and nu_s_a
			Basically, compute nu_s = sum_a nu_s_a
			:param mOpt : the Gurobi model of the problem
			:param nu_s : state visitation count
			:param nu_s_a : state-action visitation count
		"""
		for s, nu_s_v in nu_s.items():
			mOpt.addLConstr(gp.LinExpr([*((1,nu_s_a_val) for a,nu_s_a_val in nu_s_a[s].items()),(-1,nu_s_v)]),
							gp.GRB.EQUAL, 0, 
							name='{}[{}]'.format(name,s)
							)


	def constr_bellman_flow(self, mOpt, nu_s, nu_s_a=None, sigma=None, gamma=1.0, name='bellman'):
		""" Compute for all states the constraints by the bellman equation.
			This function allows only one of nu_s_a or sigma to be None
			:param mOpt : The gurobi model of the problem
			:param nu_s : state visitation count
			:param nu_s_a : state-action visitation count
			:param sigma : policy (not a Gurobi) variable
			:param gamma : the discount factor
		"""
		assert (nu_s_a is None and sigma is not None) or (nu_s_a is not None and sigma is None)
		
		# Quick util function to check if non accepting state
		nacc = lambda s : (s not in self._pomdp.prob1A and s not in self._pomdp.prob0E)
		
		dictConstr = dict()
		for s, nu_s_v in nu_s.items():
			# Probabilities of perceiving an obs from state s
			obs_distr = self._pomdp.obs_state_distr[s]
			if sigma is not None: # If the policy is given
				val_expr = gp.LinExpr([
								*((sigma[o][a]*p*gamma*(tProb if nacc(pred_s) else 0), nu_s[pred_s])\
									for (pred_s, a, tProb) in self._pomdp.pred.get(s,[])  \
										for o,p in self._pomdp.obs_state_distr[pred_s].items()
								 ), (-1,nu_s_v)]
							)
			else: # if the state-action visitation count is given
				val_expr = gp.LinExpr([*( ( gamma*(tProb if nacc(pred_s) else 0), nu_s_a[pred_s][a] ) \
											for (pred_s, a, tProb) in self._pomdp.pred.get(s,[]) 
										), 
										(-1, nu_s_v)
									   ])
			# Add the linear constraint
			dictConstr[s] = mOpt.addLConstr(val_expr, gp.GRB.EQUAL, 
									-self._pomdp.state_init_prob.get(s,0), name="{}[{}]".format(name,s))
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
				mOpt.addConstr(nu_s_a_val - nu_s_v * sum(sigma[o][a]*p for o,p in obs_distr.items()) == 0,
								name='{}[{},{}]'.format(name,s,a))

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
		dictConstr = dict()
		for s, nu_s_v in nu_s.items():
			obs_distr = self._pomdp.obs_state_distr[s]
			for a, nu_s_a_val in nu_s_a[s].items():
				val_expr = gp.LinExpr([*((1,sigma[o][a]) for o,p in obs_distr.items()),\
										 (1,nu_s_v), (-1,nu_s_a_val),\
										 (1,slack_nu_p[s][a]), (-1,slack_nu_n[s][a])])
				dictConstr[(s,a)] = mOpt.addLConstr(val_expr, gp.GRB.EQUAL, 0, 
												name='{}[{},{}]'.format(name,s,a))
		return dictConstr

	def constr_trust_region(self, mOpt, sigma):
		""" Parametrized the linear constraint by the trust region and returned
			the saved constraint to be modified later in the algorithm
			:param mOpt : Gurobi model
			:param sigma : observation based policy
		"""
		dictConstr = dict()
		for o, aDict in sigma.items():
			for a, pol_val in aDict.items():
				dictConstr[(o,a,True)] = mOpt.addLConstr(pol_val, 
										gp.GRB.GREATER_EQUAL, 0, 
										name='trust_inf[{},{}]'.format(o,a))
				dictConstr[(o,a,False)] = mOpt.addLConstr(pol_val, 
										gp.GRB.LESS_EQUAL, 1.0, 
										name='trust_sup[{},{}]'.format(o,a))
		return dictConstr

	def compute_expected_reward(self, nu_s_a, theta, feat_match=None):
		""" Provide a linear expression for the expected reward
			given a weight theta and the feature matching
			:param nu_s_a : the state-action visitation count
			:param theta : the weight for the feature function
			:param featmatch : the expected feature matching
		"""
		print(self._pomdp.reward_features)
		val_expr = gp.LinExpr(
					[( rew[(o,a)]*p*theta[rName][(o,a)] , nu_s_a_val )\
						for rName, rew in self._pomdp.reward_features.items() \
							for s, nu_s_a_t in nu_s_a.items() \
								for a, nu_s_a_val in nu_s_a_t.items()
									for o, p in self._pomdp.obs_state_distr[s].items()
					]
					)
		if feat_match is not None:
			val_expr.addConstant(
				-sum(theta_val*feat_match[r_name][(o,a)] \
						for r_name, theta_state_act in theta.items() \
							for (o,a), theta_val in theta_state_act.items())
			)
		return val_expr

	def update_constr_and_trust_region(self, mOpt, constrLin, constrTrust, nu_s_past, sigma_past, nu_s, sigma, currTrust):
		""" Update the constraint implied by the linearization of the bilinear constraint and
			the trust region constraint using the solutions of the past iteration
			:param mOpt : A gurobi model of the problem
			:param constrLin : The linearized constraints
			:param constrTrust : The trust region constraints
			:nu_s_past : state visitation count obtained at last iteration
			:sigma_past : policy obtained at the last iteration
			:nu_s : state visitation to be obtained at the current iteration
			:sigma : policy to be obtained at the current iteration
			:currTrust : the current trust region
		"""
		# Start by updating the constraint by the trust region
		for (o, a, tRhs), constrV in constrLin.items():
			constrV.RHS = sigma_past[o][a] * (1.0/currTrust if tRhs else currTrust)

		# Now update the constraints from the linearization
		for (s, a), constrV in constrTrust.items():
			obs_distr = self._pomdp.obs_state_distr[s]
			prod_obs_prob_sigma_past = sum( p*sigma_past[o][a] for o,p in obs_distr.items())
			constrV.RHS = nu_s_past[s][a] * prod_obs_prob_sigma_past
			# Update all the coefficients associated to sigma[o][a]
			for o,p in obs_distr.items():
				mOpt.chgCoeff(constrV, sigma[o][a], nu_s_past[s]*p)
			# Update the coefficient associated to nu_s[s]
			mOpt.chgCoeff(constrV, nu_s[s], prod_obs_prob_sigma_past)
		# Model update
		mOpt.update()

	def compute_entropy_cost(self, matchReward, nu_s_past, nu_s_a_past, nu_s, nu_s_a, slack_nu_p, slack_nu_n,
								slack_nu_p_spec=dict(), slack_nu_n_spec=dict(), slack_spec=None):
		""" Return a linearized entropy function around the solution of the past
			iteration given by nu_s_oast and nu_s_a_past.
			If matchReward is not None, the linearized function is added to  
			matchReward = theta^T(expected feature reward - demonstration feature)
			:param matchReward : A linear expression of the reward expression
			:param nu_s_past : state visitation at the past iteration
			:param nu_s_a_past : state action visitation at the past iteration
			:param nu_s : state visitation 
			:param nu_s_a : state action visitation
			.
			.
		"""
		# Save the pair coeff,variable for each term in the lienar cost function
		listCostTerm = list()
		if self._pomdp.has_sideinfo: # Cost associated to violating the spec constraints
			listCostTerm.append((-self._options.mu_spec, slack_spec))

		for s, actionList in self._pomdp.states_act.items():
			# Don't consider accepting states
			if s in self._pomdp.prob1A or s in self._pomdp.prob0E:
				continue

			for a in actionList:
				# First, add the entropy terms
				if nu_s_past[s] > 0: 
					nu_ratio = nu_s_a_past[s][a]/nu_s_past[s]
					listCostTerm.append( (nu_ratio/self._options.mu, nu_s[s]) )
					listCostTerm.append( (-(np.log(nu_ratio)+1)/self._options.mu, nu_s_a[s][a]) )
				
				# Then add the cost associated to the linearization errors
				listCostTerm.append((-1, slack_nu_p[s][a]))
				listCostTerm.append((-1, slack_nu_n[s][a]))
				if self._pomdp.has_sideinfo:
					listCostTerm.append((-1, slack_nu_p_spec[s][a]))
					listCostTerm.append((-1, slack_nu_n_spec[s][a]))

		# Add the linear term above
		coeffT, varT = zip(*listCostTerm)
		matchReward.addTerms(coeffT, varT)
		return matchReward


	# def verify_solution(self, mOpt, nu_s, policy_val, constrBellman, 
	# 						nu_s_spec=dict(), constrBellmanSpec=dict()):
	# 	""" Given a policy that is solution of the past iteration,
	# 		this function computes the corresponding state and state-action
	# 		visitation count that satisfy the bellman equation flow.
	# 		Then, it computes the objective attained by this policy.
	# 		:param mOpt : The gurobi model encoding the solution of the bellman flow equation
	# 		:param nu_s : state visitation count of mOpt problem
	# 		:param policy_val : The optimal policy obtaine durint this iteration
	# 		:param constrBellman : constraint representing the bellman equation
	# 	"""
	# 	# Quick util function to check if non accepting state
	# 	nacc = lambda s : (s not in self._pomdp.prob1A and s not in self._pomdp.prob0E)

	# 	# Update the optimization problem with the given policy
	# 	for s, constrV in constrBellman.items():
	# 		# Probabilities of perceiving an obs from state s
	# 		obs_distr = self._pomdp.obs_state_distr[s]
	# 		coeffV = sum( sum(policy_val[o][a]*p for o,p in obs_distr.items()) *\
	# 				  			self._options.discount*(tProb if nacc(pred_s) else 0) \
	# 								for (pred_s,a,tProb) in self._pomdp.pred[s])
	# 		mOpt.chgCoeff(constrV, nu_s[s], )


if __name__ == "__main__":
	# Customize maze with loop removed for the poison state
	pModel = PrismModel("parametric_maze_stochastic.pm", ["P=? [F \"target\"]"])
	print(pModel.obs_state_distr)
	mOptions = OptOptions()
	irlPb = IRLSolver(pModel, sat_thresh=0.95, init_trust_region=4, options=mOptions)
	mOpt = gp.Model('test')
	checkOpt = gp.Model()
	res = irlPb.init_optimization_problem(mOpt, noLinearization=False, checkOpt=checkOpt)
	mOpt.update()
	checkOpt.update()
	mOpt.display()
	print('Bellman subproblem-----------------')
	checkOpt.display()

	weight = { r_name :{(o,a) : 1.0 for (o,a) in rew} for r_name, rew in pModel.reward_features.items()}
	
	print(weight)
	irlPb.compute_optimal_policy_nonconvex_grb(weight)