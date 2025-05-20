import numpy as np  

PLACE_POS_SHELF = {
    "world": np.array([ 1.1688, -0.1884,  0.8387]),
    "robot": None
}

HELD_POS = {
    "world": None,
    "robot": np.array([ 0.30, -0.30,  0.57]),
} 

HELD_POS_LEDGE = {
    "world": None,
    "robot": np.array([ 0.30, -0.30,  0.90]),
} 

# sideways-forward tensor([-0.7009,  0.1864, -0.2130,  0.6548]))