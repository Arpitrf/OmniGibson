import numpy as np

data_collection_configs = {
    
    "place_in_shelf_forward_box": {
        "start_state": "place_in_shelf_start_state_forward_box.json",
        "obj_name": "box",
        "num_episodes": 500,
        "held_pos": {
            "world": None,
            "robot": np.array([ 0.30, -0.30,  0.57]),
        }, 
    },

    "place_in_shelf_down_box": {
        "start_state": "place_in_shelf_start_state_down_box.json",
        "obj_name": "box",
        "num_episodes": 300,
        "held_pos": {
            "world": None,
            "robot": np.array([ 0.30, -0.30,  0.57]),
        },
        
    },
    
    "place_in_shelf_forward_can": {
        "start_state": "place_in_shelf_start_state_forward_can.json",
        "obj_name": "can_of_baking_mix",
        "num_episodes": 500,
        "held_pos": {
            "world": None,
            "robot": np.array([ 0.30, -0.30,  0.57]),
        },
    },

    "place_in_shelf_down_can": {
        "start_state": "place_in_shelf_start_state_down_can.json",
        "obj_name": "can_of_baking_mix",
        "num_episodes": 300,
        "held_pos": {
            "world": None,
            "robot": np.array([ 0.30, -0.30,  0.57]),
        },
    },

    "place_in_drawer_forward_box": {
        "start_state": "place_in_drawer_start_state_forward_box.json",
        "obj_name": "box",
        "num_episodes": 200,
        "held_pos": {
            "world": None,
            "robot": np.array([ 0.60, -0.30,  0.97]),
        },
    },

    "place_in_drawer_down_box": {
        "start_state": "place_in_drawer_start_state_down_box.json",
        "obj_name": "box",
        "num_episodes": 200,
        "held_pos": {
            "world": None,
            "robot": np.array([ 0.60, -0.30,  0.90]),
        },
        
    },

    "place_in_drawer_forward_can": {
        "start_state": "place_in_drawer_start_state_forward_can.json",
        "obj_name": "can_of_baking_mix",
        "num_episodes": 200,
        "held_pos": {
            "world": None,
            "robot": np.array([ 0.60, -0.30,  0.97]),
        },
    },

    "place_in_drawer_down_can": {
        "start_state": "place_in_drawer_start_state_down_can.json",
        "obj_name": "can_of_baking_mix",
        "num_episodes": 200,
        "held_pos": {
            "world": None,
            "robot": np.array([ 0.60, -0.30,  0.90]),
        },
    },

    "place_on_ledge_forward_box": {
        "start_state": "place_on_ledge_start_state_forward_box.json",
        "obj_name": "box",
        "num_episodes": 400,
        "held_pos": {
            "world": None,
            "robot": np.array([ 0.60, -0.30,  0.97]),
        },
    },

    "place_in_sink_forward_pan": {
        "start_state": "place_in_sink_start_state_forward_pan.json",
        "obj_name": "saucepan",
        "num_episodes": 400,
        "held_pos": {
            "world": None,
            "robot": np.array([ 0.60, -0.30,  0.97]),
        },
    },

    "place_in_shelf_forward_pan": {
        "start_state": "place_in_shelf_start_state_forward_pan.json",
        "obj_name": "saucepan",
        "num_episodes": 400,
        "held_pos": {
            "world": None,
            "robot": np.array([ 0.60, -0.30,  0.97]),
        },
    },

    "open_drawer_rntwkg": {
        "gripper_friction": 0.5,
        "grasp_modes": ["front", "top"],
        "num_episodes": 800,
    },
    
    "open_drawer_pkdnbu": {
        "gripper_friction": 2.0,
        "grasp_modes": ["front"],
        "num_episodes": 800,
    },

    "open_fridge_hivvdf": {
        "gripper_friction": 50.0,
        "grasp_modes": ["front"],
        "num_episodes": 472,
    },

}