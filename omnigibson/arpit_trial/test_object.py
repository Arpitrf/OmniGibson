import os
import yaml

import omnigibson as og
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from omnigibson.utils.asset_utils import decrypt_file

# 592
# 'drawer_unit'
# 'todoac'
# todoph
# bottom_cabinet_slgzfc_0

# decrypt_file('/home/arpit/test_projects/OmniGibson/data/datasets/og_dataset/objects/fridge/dszchb/usd/dszchb.encrypted.usd', '/home/arpit/test_projects/OmniGibson/fridge_dszchb.usd')

config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"

# Create and load this object into the simulator
rot_euler = [0.0, 0.0, -60.0]
rot_quat = np.array(R.from_euler('XYZ', rot_euler, degrees=True).as_quat())
obj_cfg = dict(
    type="DatasetObject",
    name="obj",
    category="fridge",
    model="dszchb",
    position=[2.0, 0, 1.0],
    orientation=rot_quat,
    )
config["objects"] = [obj_cfg]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]

# # Place the object so it rests on the floor
# obj = env.scene.object_registry("name", "obj")
# center_offset = obj.get_position() - obj.aabb_center + np.array([0, 0, obj.aabb_extent[2] / 2.0])
# obj.set_position(center_offset)


for _ in range(50):
    og.sim.step()

obs, obs_info = env.get_obs()
seg_semantic = obs['robot0']['robot0:eyes:Camera:0']['seg_semantic']
seg_instance = obs['robot0']['robot0:eyes:Camera:0']['seg_instance']
seg_instance_id = obs['robot0']['robot0:eyes:Camera:0']['seg_instance_id']
print("seg_instance_id.shape: ", seg_instance_id.shape)
fig, ax = plt.subplots(1,3)
ax[0].imshow(seg_semantic)
ax[1].imshow(seg_instance)
ax[2].imshow(seg_instance_id)
plt.show()

for _ in range(5000):
    og.sim.step()
