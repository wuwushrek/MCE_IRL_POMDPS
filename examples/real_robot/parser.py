# Set Path
import sys
sys.path.append('../')
# Core Libs
import pickle
from pprint import pprint
import json
import copy
# External Libs
import numpy as np
# Internal Libs
import mce_irl_pomdps.modeling.pomdp as pomdp

############################## SCRIPT SETUP #############################################################
def setup_observation_map(robot_map):
    # Step 0 - Setup
    obs_map = robot_map['data'][demo_id]['feature_map_numpy']
    
    no_grass = 0
    no_gravel = 0
    no_road = 0
    empty = 0
    # 'features': ['grass', 'gravel', 'road']
    map_shape = np.shape(robot_map['data'][0]['feature_map_numpy'])

    # Step 1 -  Construct the observation map
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):        
            ## grass
            if obs_map[i][j][0] == 1.0:
                obs_map[i][j] = 1
                no_grass += 1
            ## gravel
            elif obs_map[i][j][1] == 1.0:
                obs_map[i][j] = 2
                no_gravel += 1
            ## road
            elif obs_map[i][j][2] == 1.0:           
                obs_map[i][j] = 3
                no_road += 1
            ## no feature
            else:              
                obs_map[i][j] = 0
                empty += 1
    
    # print(no_grass, no_gravel, no_road, empty)

    # Return
    return obs_map

######################################################################################################


############################## FUNCTIONS #############################################################
def dump_json(output_file, data):
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


######################################################################################################


############################## MAIN ##################################################################
if __name__ == "__main__":

    # Step 0 - Setup
    ## open parsed data
    with open('trajectory_data/data_for_franck.pickle', 'rb') as f:
        raw_map = pickle.load(f)
    ## make a copy
    robot_map = copy.deepcopy(raw_map)

    # Step 1 - Create POMDPs
    number_of_demos = 10
    for n in range(number_of_demos):
        demo_id = n
        obs_map = setup_observation_map(robot_map)

        print(f"### Constructing POMDP # {demo_id}")
        demo_pomdp = pomdp.POMDP_Gridworld(obs_map, 85, 110, "elementary", 0.2, 0.99, 0.75, ["Grass", "Gravel", "Road"])
        
        data_to_dump = demo_pomdp.pomdp
        dump_loc = f"pomdps_json/demo_{demo_id}.json"
        dump_json(dump_loc, data_to_dump)
        
        print(f"... POMDP saved to {dump_loc}")

        # print(pomdp.grass, p)
        # print(pomdp.grass, pomdp.gravel, pomdp.road, pomdp.empty)
    
######################################################################################################


# EOF