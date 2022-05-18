# Observation model (constructed from GT map)
# O(o | s_prime)
observations = {
    (1, 1, "Grass") : {
        "Grass": 0.75,
        "Gravel": 0.25/2,
        "Road": 0.25/2
    },
    (1, 1, "Gravel") : {
        "Grass": 0.25/2,
        "Gravel": 0.75,
        "Road": 0.25/2
    },
    ...
}

# Transition model
# T(s_prime | s, a)
transitions = {
    (1,1): {
        "N": {
            (1,2): 0.8,
            (3,2): 0.2/3,
            (2,1): 0.2/3,
            (2,3): 0.2/3
        },
        "S": {
            (1,2): 0.2/3,
            (3,2): 0.2/3,
            (2,1): 0.8,
            (2,3): 0.2/3
        },
        "W": {
            (1,2): 0.8,
            (3,2): 0.2/3,
            (2,1): 0.2/3,
            (2,3): 0.2/3
        },
        "E": {
            (1,2): 0.8,
            (3,2): 0.2/3,
            (2,1): 0.2/3,
            (2,3): 0.2/3
        }
    }
}