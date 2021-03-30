import stormpy
import stormpy.core
import stormpy.info

import pycarl
import pycarl.core

import stormpy.examples
import stormpy.examples.files

import stormpy.pomdp

import stormpy._config as config
from gurobipy import *

import math
import time

#Class for setting up options before querying
class QcqpOptions():
    def __init__(self, mu, mu_spec,maxiter, graph_epsilon, silent,discount):
        """
        Returns the float representation for a constant value
        :param mu: parameter for putting penalty on slack variables type: float
        :param maxiiter: max number of iterations type: integer
        :param graph_epsilon: the min probability of taking an action at each observarion type: float
        :param silent: print stuff from gurobi or not type: boolean
        :param discount: discount factor type:float
        :return:
        """
        self.mu = mu
        self.mu_spec=mu_spec
        self.maxiter = maxiter
        self.graph_epsilon = graph_epsilon
        self.silent = silent
        self.discount=discount
        if graph_epsilon<0:
            raise RuntimeError("graph epsilon should be larger than 0")
        if discount<0 or discount>=1:
            raise RuntimeError("discount factor should be between 0 and 1")
        if mu<=0:
            raise RuntimeError("mu should be larger than 0")



#Class for returning results and objectives
class QcqpResult():
    def __init__(self, value_at_initial, parameter_values):
        """
        Returns the float representation for a constant value
        :param value_at_initial: objective value at initial state after optimization type: float
        :param parameter_values: the values of your parameter or policy type: stormpy dict
        :return:
        """
        self.value_at_initial = value_at_initial
        self.parameter_values = parameter_values




#class for incremental encoding
class QcqpSolver_affine_simple_fun():
    def __init__(self):
        #timers for bunch of stuff
        self.solver_timer = 0.0
        self.encoding_timer = 0.0
        self.robust_encoding_timer = 0.0
        self.model_check_timer =0.0
        self.constraint_timer = 0.0

        #a dictionary for storing floats
        self._constants_floats = dict()

        #this is going to be the gurobi problem
        self._encoding = None

        self.iterations = 0
        #initializations for mu, solver parameters, number of observations
        self._mu=None
        self.solver_params = None
        self.num_obs=None
        self._mu_spec=None

        #i liked 4 as trust region
        self.trust_region=1.3

        self.solver_output=[]

        self._obj_best=-1e20

    def _obs_init(self,model):
        """
        Preprocessing function to compute number of of actions for each state, and predecessors of states
        :param model: a stormpy pomdp model
        :return:
        """
        print("obs_init")
        pass
        #compute the number of observations
        self.num_obs=max(model.observations)+1

        #list of bookkeeping how many actions for each obs
        self.num_actions_for_obs=[0 for _ in range(self.num_obs)]

        #list of checking which state gets which obs
        self.model_states_observations=model.observations
        for state in (model.states):

            #number of actions for each state and obs
            self.num_actions_for_obs[self.model_states_observations[state.id]]=len(state.actions)

        #compute predecessors of each state
        self.predecessor= [[] for _ in range((model.nr_states))]
        self.state_list= [state for state in model.states]

        for state in model.states:

            for action in state.actions:
                for transition in action.transitions:
                    if not state in self.predecessor[transition.column]:
                        # add state to predecessors
                        self.predecessor[transition.column].append(state)

    def _float_repr(self, constant_val):
        """
        Returns the float representation for a constant value
        :param constant_val:
        :return:
        """
        if constant_val.is_one():
            return 1.0
        elif constant_val.is_minus_one():
            return -1.0

        v = self._constants_floats.get(constant_val, float(constant_val))
        self._constants_floats[constant_val] = v
        return v

    def _create_encoding(self,model):
        """
        Creates the gurobi encoding given a stormpy pomdp
        :model: stormpy model of POMDP
        :return:
        """
        print("create_encoding")

        numstate = model.nr_states
        numobs=model.nr_observations
        #print(numstate,numobs)

        #initialize gurobi model
        self._encoding = Model("qcp")
        self._encoding.setParam('OutputFlag', not self._options.silent)

        # occupancy measure variables xVars: state actions, xVars_state: states
        self._xVars = [[self._encoding.addVar(lb=0,name="xVars_"+str(state)+"_"+str(action)) for action in state.actions] for state in model.states]


        self._xVars_state = [self._encoding.addVar(lb=0,name="xVars_state_"+str(state))  for state in model.states]

        # occupancy measure variables xVars: state actions, xVars_state: states
        self._xVars_spec = [[self._encoding.addVar(lb=0,name="xVars_spec"+str(state)+"_"+str(action)) for action in state.actions] for state in model.states]
        self._xVars_state_spec = [self._encoding.addVar(lb=0,name="xVars_state_spec_"+str(state))  for state in model.states]

        #slack variables for convexified constraints
        self._tau_pos = [[self._encoding.addVar(lb=0,name="tau_pos_"+str(state)+"_"+str(action)) for action in state.actions] for state in model.states]
        self._tau_neg = [[self._encoding.addVar(lb=0,name="tau_neg_"+str(state)+"_"+str(action)) for action in state.actions] for state in model.states]
        self._tau_pos_spec = [[self._encoding.addVar(lb=0,name="tau_pos_spec_"+str(state)+"_"+str(action)) for action in state.actions] for state in model.states]
        self._tau_neg_spec = [[self._encoding.addVar(lb=0,name="tau_neg_spec_"+str(state)+"_"+str(action)) for action in state.actions] for state in model.states]

        #policy variables for each obs and action
        self._policy = [[self._encoding.addVar(lb=0,name="policy_"+str(i)+"_"+str(action)) for action in range(self.num_actions_for_obs[i])] for i in range(self.num_obs)]
        #slack variable for spec
        self._spec_tau=self._encoding.addVar(lb=0,name="spec_slack")
        #update gurobi problem
        self._encoding.update()

        #gurobi params
        self._encoding.Params.OutputFlag = 1
        self._encoding.Params.Presolve = 2
        self._encoding.Params.Method = 0
        self._encoding.Params.Crossover = 0
        self._encoding.Params.CrossoverBasis = 0
        self._encoding.Params.NumericFocus = 3
        self._encoding.Params.BarHomogeneous = 1
      #  self._encoding.Params.ScaleFlag = 3
        self._encoding.Params.FeasibilityTol = 1e-6
        self._encoding.Params.OptimalityTol = 1e-6
        self._encoding.Params.BarConvTol = 1e-6
        self._encoding.update()


    def _model_init(self,model,iter_num):
        """
        Given a policy, initialize the occupancy measures
        :model: stormpy model of POMDP
        :return:
        """
        print("model_init")

        #auxillary gurobi problem
        self._aux_encoding = Model("qcp")

        #self._policy_aux denotes the previous best policy
        self._policy_aux = [[0 for action in range(self.num_actions_for_obs[i])] for i in range(self.num_obs)]

        #occupancy measures for states, as we already have a policy
        self._xVars_aux = [self._aux_encoding.addVar(lb=0) for state in model.states]
        self._xVars_aux_spec = [self._aux_encoding.addVar(lb=0) for state in model.states]

        self._aux_encoding.update()

        # update policy, will update the way that I update
        for obs in range(self.num_obs):
            for action in range(self.num_actions_for_obs[obs]):
                # save the best policy
                self._policy_aux[obs][action]=self._policy_Init[obs][action]
                try:
                    self._policy_Init[obs][action] = self._policy[obs][action].x
                except AttributeError:
                    pass
        #print("solver policy:",self._policy_Init)
        #top tier occupancy measure constraints
        for state in model.states:
            cons = 0
            cons_spec=0
            #this is for non-accepting states
            # if not self._prob1A.get(state) and not self._prob0E.get(state):
                #x(s,a)
            cons += self._xVars_aux[state.id]
            cons_spec +=self._xVars_aux_spec[state.id]
            for succ in self.predecessor[state.id]:
                #print(state,succ)
                if not self._prob1A.get(succ) and not self._prob0E.get(succ):
                    for action2 in succ.actions:
                        for transition in action2.transitions:
                            #P(s',a,s)*x(s',a)
                            if transition.column==state.id:
                                #print(state,succ,action2,transition.value())
                                cons += -transition.value() * self._options.discount* self._policy_Init[self.model_states_observations[succ.id]][action2.id]\
                                        * self._xVars_aux[succ.id]
                                cons_spec += -transition.value() * self._policy_Init[self.model_states_observations[succ.id]][action2.id]\
                                        * self._xVars_aux_spec[succ.id]
            #if init state, add 1
            if state.id==(model.initial_states[0]):
                cons=cons-1
                cons_spec=cons_spec-1
            #constraint
            self._aux_encoding.addConstr(cons==0)
            self._aux_encoding.addConstr(cons_spec==0)

            #this is for accepting states
            # if self._prob1A.get(state) or self._prob0E.get(state):
            #
            #     cons += self._xVars_aux[state.id]
            #     cons_spec +=self._xVars_aux_spec[state.id]
            #
            #     for succ in self.predecessor[state.id]:
            #         #if not self._prob1A.get(succ) and not self._prob0E.get(succ):
            #         for action2 in succ.actions:
            #             for transition in action2.transitions:
            #                 if transition.column==state.id:
            #                     #print(state,succ,action2,transition.value())
            #                     cons += -transition.value() * self._options.discount* self._policy_Init[self.model_states_observations[succ.id]][action2.id]\
            #                             * self._xVars_aux[succ.id]
            #                     cons_spec += -transition.value() * self._policy_Init[self.model_states_observations[succ.id]][action2.id]\
            #                             * self._xVars_aux_spec[succ.id]
            #     if state.id==(model.initial_states[0]):
            #         cons=cons-1
            #         cons_spec=cons_spec-1
            #
            #         #print(state,"initstate")
            #     self._aux_encoding.addConstr(cons==0)
            #     self._aux_encoding.addConstr(cons_spec==0)

        #update model
        self._aux_encoding.update()
        #feasability problem, object 0
        self._aux_encoding.setObjective(0, GRB.MINIMIZE)

        self._aux_encoding.update()

        print('Solving...')
        #solve with gurobi
        self._aux_encoding.optimize()

        self._aux_obj=0
        spec_aux=0

        for state in model.states:
            for action in state.actions:
                if self._xVars_aux[state.id].x>0:
                    term_aux=math.log(self._xVars_aux[state.id].x)-math.log(self._xVars_aux[state.id].x
                                        *self._policy_Init[self.model_states_observations[state.id]][action.id])
                    self._aux_obj+=term_aux*self._xVars_aux[state.id].x*self._policy_Init[self.model_states_observations[state.id]][action.id]
            if self._prob1A.get(state):
                spec_aux=spec_aux+self._xVars_aux_spec[state.id].x

        if spec_aux<self._threshold:
            self._aux_obj+=(spec_aux-self._threshold)*self._mu*self._mu_spec
        print("spec_aux",spec_aux)
        if self._aux_obj>self._obj_best:

            # update policy, self._policy_aux is the one that I save
            for obs in range(self.num_obs):
                for action in range(self.num_actions_for_obs[obs]):
                    self._policy_aux[obs][action] = self._policy_Init[obs][action]

            self._obj_best=self._aux_obj
            #update state-action occupancy measuref
            for state in model.states:
                for action in state.actions:
                    self._xInit[state.id][action.id]=self._xVars_aux[state.id].x*self._policy_Init[self.model_states_observations[state.id]][action.id]
                    self._xInit_spec[state.id][action.id]=self._xVars_aux_spec[state.id].x*self._policy_Init[self.model_states_observations[state.id]][action.id]

    #        print(self._xInit)
            #update state occupancy measure
            for state in model.states:
                self._xInit_state[state.id] = self._xVars_aux[state.id].x
                self._xInit_state_spec[state.id] = self._xVars_aux_spec[state.id].x


            if iter_num>=0:

                self.trust_region=self.trust_region=min(10,(self.trust_region-1)*1.5+1)
        else:
            # go back to the original policy
            for obs in range(self.num_obs):
                for action in range(self.num_actions_for_obs[obs]):
                    self._policy_Init[obs][action] = self._policy_aux[obs][action]
            self.trust_region = ((self.trust_region - 1) / 1.5 + 1)
        print("obj_aux:",self._aux_obj)
        print("obj_best:",self._obj_best)
        state_total = 0
        for state in model.states:
            state_total += self._xInit_state_spec[state.id]
        print("num steps:",state_total)
        print("updated policy",self._policy_Init)
        #print(self._xInit_state_spec)
        #print(self._xInit_spec)

    def _model_constraints(self,model):
        """
        Given a model, construct constraints that are related to underlying MDP
        :model: stormpy model of POMDP
        :return:
        """
        pass

        print("model constraints")

        #policy is well-defined, i.e., sums up to 1
        for obs in range(self.num_obs):
            cons=0
            for action in range(self.num_actions_for_obs[obs]):
                cons=cons+self._policy[obs][action]
                self._encoding.addConstr(self._policy[obs][action]>=self._options.graph_epsilon)
            self._encoding.addConstr(cons == 1)

        #sum_a x(s,a)=x(s)
        for state in model.states:
            cons=0
            cons_spec=0

            for action in state.actions:
                cons += self._xVars[state.id][action.id]
                cons_spec +=self._xVars_spec[state.id][action.id]
            self._encoding.addConstr(cons == self._xVars_state[state.id])
            self._encoding.addConstr(cons_spec == self._xVars_state_spec[state.id])

        #occupancy measure constraints
        for state in model.states:
            cons=0
            cons_spec=0
            #if not self._prob1A.get(state) and not self._prob0E.get(state):

            for action in state.actions:
                cons += self._xVars[state.id][action.id]
                cons_spec += self._xVars_spec[state.id][action.id]

            for succ in self.predecessor[state.id]:
                if not self._prob1A.get(succ) and not self._prob0E.get(succ):
                    for action2 in succ.actions:
                        for transition in action2.transitions:
                            if transition.column==state.id:
                                #print(state,succ,action2,transition.value())
                                cons += -transition.value() * self._options.discount*self._xVars[succ.id][action2.id]
                                cons_spec += -transition.value() *self._xVars_spec[succ.id][action2.id]

            if state.id==(model.initial_states[0]):
                cons=cons-1
                cons_spec=cons_spec-1
                #print(state,"initstate")
            #print(cons)
            self._encoding.addConstr(cons==0)
            self._encoding.addConstr(cons_spec==0)

            # if self._prob1A.get(state) or self._prob0E.get(state):
            #
            #
            #     for action in state.actions:
            #         cons += self._xVars[state.id][action.id]
            #         cons_spec += self._xVars_spec[state.id][action.id]
            #
            #     for succ in self.predecessor[state.id]:
            #         if not self._prob1A.get(succ) and not self._prob0E.get(succ):
            #             for action2 in succ.actions:
            #                 for transition in action2.transitions:
            #                     if transition.column==state.id:
            #                         #print(state,succ,action2,transition.value())
            #                         cons += -transition.value() * self._options.discount*self._xVars[succ.id][action2.id]
            #                         cons_spec += -transition.value() *self._xVars_spec[succ.id][action2.id]
            #     if state.id==(model.initial_states[0]):
            #         cons=cons-1
            #         cons_spec=cons_spec-1
            #
            #         #print(state,"initstate")
            #     #print(cons)
            #
            #     self._encoding.addConstr(cons==0)
            #     self._encoding.addConstr(cons_spec==0)

        cons_threshold=0
        #specification constraints
        for state in model.states:
            if self._prob1A.get(state):
                cons_threshold+= self._xVars_state_spec[state.id]
        self._encoding.addConstr(cons_threshold+self._spec_tau>=self._threshold)



    def _set_objective(self,model):
        """
        Sets the objective for the convexified problem
        :model: stormpy model of POMDP
        :return:
        """
        print("add obj")
        self._objective = 0.0

        #penalty for violating spec
        self._objective=self._objective-self._spec_tau*self._mu_spec
        # Adding terms to the objective

        # for obs in states:
                #occup_our_pol=0
        #     for state in pre(obs):
                    #for act in state.actions:
        #         occup_our_pol+self._xInit[state][act]
        #     self._objective + self._objective + F_REWARD(OBS,ACT)*(f_expert(obs,act)-self._xInit[OBS(state)][act]
        for state in model.states:
            if not self._prob1A.get(state) and not self._prob0E.get(state):

                for action in state.actions:
                    #relative entropy, i.e. x(s,a)*log(x(s,a)/x(s))

                    #grad with respect to x(s,a) which is -log(x(s,a)/x(s))-1
                    #multiple then with x(s,a)
                    if self._xInit_state[state.id] > 0:

                        self._objective = self._objective + (self._xVars[state.id][action.id] *
                (- math.log(self._xInit[state.id][action.id]/self._xInit_state[state.id])-1))\
                                          /self._mu

                        #grad with respect to x(s) which is x(s,a)/x(s)
                        #multiple then with x(s)
                        self._objective = self._objective + (self._xVars_state[state.id]*self._xInit[state.id][action.id]\
                                          /self._xInit_state[state.id])/self._mu


                    #objectives for slacks, self._mu is how much you penalize deviations
                    self._objective = self._objective - self._tau_pos[state.id][action.id]

                    self._objective = self._objective - self._tau_neg[state.id][action.id]

                    #objectives for slacks, self._mu is how much you penalize deviations
                    self._objective = self._objective - self._tau_pos_spec[state.id][action.id]

                    self._objective = self._objective - self._tau_neg_spec[state.id][action.id]

    def _nonconvex_constraints(self,model):
        """
        Adds the convexified constraints for x(s)*pol(o(s),a)=x(s,a)
        linearize x(s)*pol(o(s),a), and add the constraint
        :model: stormpy model of POMDP
        :return:
        """
        print("nonconvex cons")

        for state in model.states:
            for action in state.actions:
                cons=0
                #x(s)*hat(pol)(o(s),a)
                cons=cons+self._xVars_state[state.id]*self._policy_Init[self.model_states_observations[state.id]][action.id]

                #hat(x)(s)*pol(o(s),a)
                if self._xInit_state[state.id]>0e-4:
                    cons=cons+self._xInit_state[state.id]*self._policy[self.model_states_observations[state.id]][action.id]

                    #hat(x)(s)*hat(pol)(o(s),a)
                    cons=cons-self._xInit_state[state.id]*self._policy_Init[self.model_states_observations[state.id]][action.id]

                #add everything, and slacks
                self._remove_set.append(self._encoding.addConstr(self._xVars[state.id][action.id]==cons+
                        self._tau_pos[state.id][action.id]-self._tau_neg[state.id][action.id]))


                cons_spec=0

                #x(s)*hat(pol)(o(s),a)
                cons_spec=cons_spec+self._xVars_state_spec[state.id]*self._policy_Init[self.model_states_observations[state.id]][action.id]
                if self._xInit_state_spec[state.id]>0e-4:
                #hat(x)(s)*pol(o(s),a)
                    cons_spec=cons_spec+self._xInit_state_spec[state.id]*self._policy[self.model_states_observations[state.id]][action.id]

                    #hat(x)(s)*hat(pol)(o(s),a)
                    cons_spec=cons_spec-self._xInit_state_spec[state.id]*self._policy_Init[self.model_states_observations[state.id]][action.id]

                #add everything, and slacks
                self._remove_set.append(self._encoding.addConstr(self._xVars_spec[state.id][action.id]==cons_spec+
                        self._tau_pos_spec[state.id][action.id]-self._tau_neg_spec[state.id][action.id]))

        pass

    def _trust_region_constraints(self,model):
        """
        Adds the trust region constraints on the variables are in convexified constraints
        :model: stormpy model of POMDP
        :return:
        """
        print("trust region cons")

        for obs in range(self.num_obs):
            cons=0
            for action in range(self.num_actions_for_obs[obs]):
                #policy
                self._remove_set.append(self._encoding.addConstr(self._policy[obs][action]<=self._policy_Init[obs][action]*self.trust_region))

                self._remove_set.append(self._encoding.addConstr(self._policy[obs][action]>=self._policy_Init[obs][action]/self.trust_region))
        # if self.trust_region<1.2:
        #     for state in model.states:
        #         #state occupancy
        #         self._remove_set.append(self._encoding.addConstr(self._xVars_state[state.id] <= self._xInit_state[state.id] * self.trust_region))
        #         self._remove_set.append(self._encoding.addConstr(self._xVars_state[state.id] >= self._xInit_state[state.id] / self.trust_region))
        #         #state occupancy
        #         self._remove_set.append(self._encoding.addConstr(self._xVars_state_spec[state.id] <= self._xInit_state_spec[state.id] * self.trust_region))
        #         self._remove_set.append(self._encoding.addConstr(self._xVars_state_spec[state.id] >= self._xInit_state_spec[state.id] / self.trust_region))
        #         # for action in state.actions:
                #     #state-action occupancy
                #     self._remove_set.append(self._encoding.addConstr(self._xVars_spec[state.id][action.id]<=self._xInit_spec[state.id][action.id]*self.trust_region))
                #     self._remove_set.append(self._encoding.addConstr(self._xVars_spec[state.id][action.id]>=self._xInit_spec[state.id][action.id]/self.trust_region))
                #     self._remove_set.append(self._encoding.addConstr(self._xVars[state.id][action.id]<=self._xInit[state.id][action.id]*self.trust_region))
                #     self._remove_set.append(self._encoding.addConstr(self._xVars[state.id][action.id]>=self._xInit[state.id][action.id]/self.trust_region))

        pass


    def run(self, model,properties, prob0E, prob1A, threshold,  options,model_check):
        """
        Runs the SCP procedure by a series of calls to gurobi.

        :param model: The model
        :type model: a stormpy dtmc/mdp
        :param properties: The properties as an iterable over stormpy.properties
        :param prob0E: set of states with prob0
        :param prob0E: set of states with prob1
        :param threshold: The threshold
        :type threshold: float
        :param direction: Are we looking for a value below or above
        :type direction: a string, either "above" or "below"
        :param options: Further options with which the algorithm should run
        :param model_check: boolean value if we model check at each iteration
        :return:
        """
        #storing items

        #prob0 states
        self._prob0E = prob0E
        #prob1 states
        self._prob1A = prob1A
        #stormpy property
        self._properties= properties
        #spec threshold
        self._threshold= threshold
        #solver options
        self._options= options
        #boolean for model checking
        self._model_check = model_check
        #stormpy model
        self._model= model
        #penalty for slack parameters
        self._mu = options.mu
        self._mu_spec = options.mu_spec
        #obs_init function
        self._obs_init(model)
        #create encoding
        self._create_encoding(model)

        #inits for occupancy measures and policies
        self._xInit = [[0 for action in state.actions] for state in model.states]
        self._xInit_spec = [[0 for action in state.actions] for state in model.states]
        self._xInit_state = [0 for state in model.states]
        self._xInit_state_spec = [0 for state in model.states]
        self._policy_Init = [[1/(self.num_actions_for_obs[i]) for action in range(self.num_actions_for_obs[i])] for i in range(self.num_obs)]
        #constraints to remove
        self._remove_set = []
        # initialize model
        self._model_init(model,-1)
        # set MDP constraints
        self._model_constraints(model)
        # set objective
        self._set_objective(model)
        # set gurobi objective
        self._encoding.setObjective(self._objective, GRB.MAXIMIZE)
        for i in range(options.maxiter):

            #set nonconvex constraints
            self._nonconvex_constraints(model)

            #build trust region constraints
            self._trust_region_constraints(model)
            self._set_objective(model)
            self._encoding.setObjective(self._objective, GRB.MAXIMIZE)

            self._encoding.update()
            #print(self._encoding.display())
            print('solving...')
            start = time.time()
            self._encoding.optimize()
            end = time.time()
            self.solver_timer += (end - start)
            print("Solver time :" + str(end - start))
            print("total solver time:",self.solver_timer)
            #print("policy,init",self._policy_Init)
            #print("xinit",self._xInit_state)
            #print("xinit_spec",self._xInit_state_spec)
            print("trust region",self.trust_region)
            print("spec violation",self._spec_tau)
            #initialize model
            self._model_init(model,i)


            self._encoding.remove(self._remove_set)
            self._remove_set = []

            self._encoding.update()

            if self.trust_region<1+1e-4:
                break


def example_parametric_models_01():
    # Check support for parameters
    if not config.storm_with_pars:
        print("Support parameters is missing. Try building storm-pars.")
        return

    import stormpy.pars
    from pycarl.formula import FormulaType, Relation
    if stormpy.info.storm_ratfunc_use_cln():
        import pycarl.cln.formula
    else:
        import pycarl.gmp.formula


    ####
    # How to apply an unknown FSC to obtain a pMC from a pPOMDP
    path = "parametric_maze_stochastic.pm"
    prism_program = stormpy.parse_prism_program(path)
    formula_str = "R=? [F \"goal\" | \"bad\"]"

    formula_str = "P=? [F \"target\"]"
    opts = stormpy.BuilderOptions()
    print(dir(opts))
    print(dir(stormpy.PrismProgram))
    print(dir(stormpy))
    properties = stormpy.parse_properties_for_prism_program(formula_str, prism_program)
    # construct the pPOMDP
    #pomdp = stormpy.build_parametric_model(prism_program, properties)
    options = stormpy.BuilderOptions([p.raw_formula for p in properties])
    options.set_build_all_reward_models(True)
    options.set_build_state_valuations(True)
    options.set_build_choice_labels(True)
    options.set_build_all_labels(True)
    options.set_build_with_choice_origins(True)

    print(dir(options))
    pomdp = stormpy.build_sparse_model_with_options(prism_program, options)
    # make its representation canonic.
    path_pmc = "export_" + str("pre_canon_")  + path
    stormpy.export_to_drn(pomdp, path_pmc)
    pomdp = stormpy.pomdp.make_canonic(pomdp)
    path_pmc = "export_" + str("after_canon_")  + path
    stormpy.export_to_drn(pomdp, path_pmc)
    # construct the memory for the FSC
    # in this case, a selective counter with two states
    memory_builder = stormpy.pomdp.PomdpMemoryBuilder()
    memory = memory_builder.build(stormpy.pomdp.PomdpMemoryPattern.selective_counter, 1)
    # apply the memory onto the POMDP to get the cartesian product
    pomdp = stormpy.pomdp.unfold_memory(pomdp, memory)
    # make the POMDP simple. This step is optional but often beneficial
    #pomdp = stormpy.pomdp.make_simple(pomdp)
    # apply the unknown FSC to obtain a pmc from the POMDP
    print(dir(stormpy.pomdp))
    pmc = stormpy.pomdp.apply_unknown_fsc(pomdp, stormpy.pomdp.PomdpFscApplicationMode.simple_linear)
    path_pmc = "export_" + str("after_simple_")  + path
    stormpy.export_to_drn(pomdp, path_pmc)
    export_pmc = True # Set to True to export the pMC as drn.
    if export_pmc:
        export_options = stormpy.core.DirectEncodingOptions()
        export_options.allow_placeholders = False
        stormpy.export_to_drn(pmc, "export_" + str("after_simple_pmc")  + path, export_options)
    properties = stormpy.parse_properties(formula_str)
    threshold = 0.95
    prob0E, prob1A = stormpy.prob01max_states(pmc, properties[0].raw_formula.subformula)
    print(prob0E,prob1A)
    direction = "below"  # can be "below" or "above"
    options = QcqpOptions(mu=1e4, mu_spec=1e4,maxiter=100, graph_epsilon=1e-3, silent=False,discount=0.9)

    # result = solver.run(reward_model_name ,model_rew, parameters_rew, rew0, rew_threshold, direction, options)
    solver = QcqpSolver_affine_simple_fun()
    result = solver.run(pomdp, properties, prob0E, prob1A, threshold, options,True)

def get_prob01States(model, formulas):
    parameters = model.collect_probability_parameters()
    instantiator = stormpy.pars.PDtmcInstantiator(model)
    print(parameters)

    point = dict()
    for p in parameters:
        point[p] = stormpy.RationalRF(0.4)

        print(p)
    instantiated_model = instantiator.instantiate(point)
    assert instantiated_model.nr_states == model.nr_states
    assert not instantiated_model.has_parameters
    pathform = formulas[0].raw_formula.subformula
    assert type(pathform) == stormpy.logic.EventuallyFormula
    labelform = pathform.subformula
    labelprop = stormpy.core.Property("label-prop", labelform)
    phiStates = stormpy.BitVector(instantiated_model.nr_states, True)
    psiStates = stormpy.model_checking(instantiated_model, labelprop).get_truth_values()
    (prob0, prob1) = stormpy.compute_prob01_states(model, phiStates, psiStates)
    return prob0, prob1

if __name__ == '__main__':
    example_parametric_models_01()
