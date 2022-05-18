'''
mdp.py
An MDP Consists of:
S - A set of states
A - A set of actions
T(s_prime | a, s) - A tranistion function
R(...) - A reward function
gamma - A discount factor
'''

# Core Libs
import copy
import pprint
# External Libs
import numpy as np

class MDP_GridWorld():

    def __init__(self, n_rows, n_cols, agent_dynamics, slip_probability=0, gamma=0.9):
        # 0. Setup
        self.slip_probability = slip_probability
        self.gamma = gamma

        # 1. Construct the state space
        self.state_space = self._construct_state_space(n_rows, n_cols)

        # 2. Construct the action space
        self.action_space = self._construct_action_space(agent_dynamics)

        # 3. Construct the transition function
        self.transition_function = self._construct_transition_function(self.state_space, self.action_space, agent_dynamics, slip_probability)
        
        for state, actions in self.transition_function.items():
            for action, prob_dist in actions.items():
                if sum(prob_dist.values()) != 1.0:
                    print("ERROR!")
                    print(state, action)
                    print(prob_dist)
                    print(sum(prob_dist.values()))
                    print()

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.transition_function)


    def _construct_state_space(self, n_rows, n_cols):
        state_space = np.zeros((n_rows, n_cols))
        return state_space


    def _construct_action_space(self, agent_dynamics):
        if agent_dynamics == "full":
            action_space = ["N", "S", "W", "E", "NW", "NE", "SW", "SE"]
        elif agent_dynamics == "elementary":
            action_space = ["N", "S", "W", "E"]
        else:
            raise ValueError("No valid action space for agent_dynamics provided")

        return action_space


    def _construct_transition_function(self, state_space, action_space, agent_dynamics, slip_probability):
        # 0. Setup
        transition_function = {}
        actions = {
            "N": (-1, 0),
            "S": (+1, 0),
            "W": (0, -1),
            "E": (0, +1)
        }

        # 1. Construct transition function based on agent_dynamics
        for i, row in enumerate(state_space):
            for j, col in enumerate(row):
                prob_dists = {}
                for key, value in actions.items():
                    prob_dists[key] = self.construct_transition_prob_dist_for_curr_state(self.state_space, (i,j), key, slip_probability)
                
                transition_function[(i, j)] = prob_dists

        return transition_function


    def construct_transition_prob_dist_for_curr_state(self, state_space, curr_state, agent_action, slip_probability):
        '''
        desc:
            constructs a prob dist over next states from the current state
        params:
            state_idx: (x,y) - tuple of ints
            agent_action: "..." - 1 leter string
        notes:
            transitions are in the following form for each state: 
            {
                (2, 2): {
                    "N": {
                        (1,2): 0.8,
                        (3,2): 0.2/4,
                        (2,1): 0.2/4,
                        (2,3): 0.2/4
                    }
                    ...
                }
            }
        '''

        actions = {
            "N": (-1, 0),
            "S": (+1, 0),
            "W": (0, -1),
            "E": (0, +1)
        }
        transition_prob_dist = {}
        s_prime_prob = 0
        s_prime_prob_wall_bump = 0

        for key, value in actions.items():
            try:
                # get the indices of the next state based on the agent's action
                row_idx = curr_state[0] + value[0]
                col_idx = curr_state[1] + value[1]
                
                # in some cases an idx may be -1. rather than go to the last elem in the array, bump into the wall.
                assert row_idx >= 0
                assert col_idx >= 0

                # make sure it exists, if not go to except IndexError block
                state_space[row_idx][col_idx] 

                # if the next key is the agent action, asign 1-slip_prob probability mass
                if key == agent_action:
                    s_prime_prob = 1-slip_probability

                # if not split the remaining probability mass
                else:
                    s_prime_prob = slip_probability/(len(actions)-1)

                transition_prob_dist[(row_idx, col_idx)] = s_prime_prob

            except (AssertionError, IndexError) as e:
                # this exception occurs when an agent bumps into a wall

                # put the agent back in the current state
                row_idx = curr_state[0]
                col_idx = curr_state[1]

                # case where agent's desired action bumps into wall
                if key == agent_action:
                    s_prime_prob_wall_bump += 1-slip_probability
                # case where agent slips and bumps into a wall
                else:
                    s_prime_prob_wall_bump += slip_probability/(len(actions)-1)
            
                transition_prob_dist[(row_idx, col_idx)] = s_prime_prob_wall_bump

        return transition_prob_dist

# TODO
# Construct POMDP based on observation data struct seen in "nest.py" line 3
class POMDP_Gridworld():

    def __init__(self, n_rows, n_cols, agent_dynamics, slip_probability=0, gamma=0.9, perception_accuracy=0.75, featureList = ["Grass", "Gravel", "Road"]):
        # 0. Setup
        self.slip_probability = slip_probability
        self.gamma = gamma
        self.perception_accuracy = perception_accuracy
        self.featureList = featureList

        # 1. Construct the state space
        self.state_space = self._construct_state_space(n_rows, n_cols)

        # 2. Construct the action space
        self.action_space = self._construct_action_space(agent_dynamics)

        # 3. Construct the observation space
        self.observation_space = self._construct_observation_space(self.state_space, perception_accuracy, self.featureList)

        # 4. Construct the transition function
        self.transition_function = self._construct_transition_function(self.state_space, self.action_space, agent_dynamics, slip_probability)
        
        for state, actions in self.transition_function.items():
            for action, prob_dist in actions.items():
                if sum(prob_dist.values()) != 1.0:
                    print("ERROR!")
                    print(state, action)
                    print(prob_dist)
                    print(sum(prob_dist.values()))
                    print()

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.transition_function)

    def _construct_observation_space(self, state_space, perception_accuracy, featureList):
        '''Format of feature List:
        # featureList= [
        #     "Grass",
        #     "Gravel",
        #     "Road"
        # ]'''
        
        observations = {}

        for i, row in enumerate(state_space):
            for j, col in enumerate(row):
                for feature in featureList:
                    temp_list = featureList.copy()
                    temp_list.remove(feature)
                    observations[(i, j, feature)] = {
                                                    feature: perception_accuracy, 
                                                    temp_list[0]: (1-perception_accuracy)/(len(featureList)-1),
                                                    temp_list[1]: (1-perception_accuracy)/(len(featureList)-1)}
        return observations
                                                    
   
    def _construct_state_space(self, n_rows, n_cols):
        state_space = np.zeros((n_rows, n_cols))
        return state_space
    

    def _construct_action_space(self, agent_dynamics):
        if agent_dynamics == "full":
            action_space = ["N", "S", "W", "E", "NW", "NE", "SW", "SE"]
        elif agent_dynamics == "elementary":
            action_space = ["N", "S", "W", "E"]
        else:
            raise ValueError("No valid action space for agent_dynamics provided")

        return action_space

    def _construct_transition_function(self, state_space, action_space, agent_dynamics, slip_probability):
        # 0. Setup
        transition_function = {}
        actions = {
            "N": (-1, 0),
            "S": (+1, 0),
            "W": (0, -1),
            "E": (0, +1)
        }

        # 1. Construct transition function based on agent_dynamics
        for i, row in enumerate(state_space):
            for j, col in enumerate(row):
                prob_dists = {}
                for key, value in actions.items():
                    prob_dists[key] = self.construct_transition_prob_dist_for_curr_state(self.state_space, (i,j), key, slip_probability)
                
                transition_function[(i, j)] = prob_dists

        return transition_function


    def construct_transition_prob_dist_for_curr_state(self, state_space, curr_state, agent_action, slip_probability):
        '''
        desc:
            constructs a prob dist over next states from the current state
        params:
            state_idx: (x,y) - tuple of ints
            agent_action: "..." - 1 leter string
        notes:
            transitions are in the following form for each state: 
            {
                (2, 2): {
                    "N": {
                        (1,2): 0.8,
                        (3,2): 0.2/4,
                        (2,1): 0.2/4,
                        (2,3): 0.2/4
                    }
                    ...
                }
            }
        '''

        actions = {
            "N": (-1, 0),
            "S": (+1, 0),
            "W": (0, -1),
            "E": (0, +1)
        }
        transition_prob_dist = {}
        s_prime_prob = 0
        s_prime_prob_wall_bump = 0

        for key, value in actions.items():
            try:
                # get the indices of the next state based on the agent's action
                row_idx = curr_state[0] + value[0]
                col_idx = curr_state[1] + value[1]
                
                # in some cases an idx may be -1. rather than go to the last elem in the array, bump into the wall.
                assert row_idx >= 0
                assert col_idx >= 0

                # make sure it exists, if not go to except IndexError block
                state_space[row_idx][col_idx] 

                # if the next key is the agent action, asign 1-slip_prob probability mass
                if key == agent_action:
                    s_prime_prob = 1-slip_probability

                # if not split the remaining probability mass
                else:
                    s_prime_prob = slip_probability/(len(actions)-1)

                transition_prob_dist[(row_idx, col_idx)] = s_prime_prob

            except (AssertionError, IndexError) as e:
                # this exception occurs when an agent bumps into a wall

                # put the agent back in the current state
                row_idx = curr_state[0]
                col_idx = curr_state[1]

                # case where agent's desired action bumps into wall
                if key == agent_action:
                    s_prime_prob_wall_bump += 1-slip_probability
                # case where agent slips and bumps into a wall
                else:
                    s_prime_prob_wall_bump += slip_probability/(len(actions)-1)
            
                transition_prob_dist[(row_idx, col_idx)] = s_prime_prob_wall_bump

        return transition_prob_dist

# Main
if __name__ == "__main__":
    MDP_GridWorld(3, 3, "elementary", 0.2)
    POMDP_Gridworld(3,3, "elementary", 0.2, 0.99, 0.75, ["Grass", "Gravel", "Road"])


# EOF