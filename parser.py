import pickle
import numpy as np
from pprint import pprint
import mce_irl_pomdps.modeling.pomdp as pomdp
import json

with open('/home/aryaman/MCE_IRL_POMDPS/Data/data_for_franck.pickle', 'rb') as f:
    map = pickle.load(f)

data = map.copy()
# print(dict.items(data))
map_shape = np.shape(data['data'][0]['feature_map_numpy'])
# print(map_shape)

for n in range(10):
    demo_id = n
    obs_map = data['data'][demo_id]['feature_map_numpy']
    # pprint(np.shape(data['data'][0]['feature_map_numpy']))
    # print(data['data'][0]['feature_map_numpy'][0][0])
    # pprint(data['data'][0])
    no_grass = 0
    no_gravel = 0
    no_road = 0
    empty = 0
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):        
            # 'features': ['grass', 'gravel', 'road']
            if obs_map[i][j][0] == 1.0:
                # grass            
                obs_map[i][j] = 1
                no_grass += 1

            elif obs_map[i][j][1] == 1.0:
                # gravel
                obs_map[i][j] = 2
                no_gravel += 1

            elif obs_map[i][j][2] == 1.0:
                # road            
                obs_map[i][j] = 3
                no_road += 1

            else:
                # no feature            
                obs_map[i][j] = 0
                empty += 1

    # pprint(obs_map)

    # np.save("demo_1", obs_map)yy

    demo_pomdp = pomdp.POMDP_Gridworld(obs_map, 85, 110, "elementary", 0.2, 0.99, 0.75, ["Grass", "Gravel", "Road"])
    # pprint(demo_pomdp.pomdp)
    # pprint(type(demo_pomdp.pomdp))
    jsonString = json.dumps(demo_pomdp.pomdp)
    jsonFile = open("demo_"+str(demo_id)+".json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()
# print(pomdp.grass, p)
# print(pomdp.grass, pomdp.gravel, pomdp.road, pomdp.empty)
# print(no_grass, no_gravel, no_road, empty)