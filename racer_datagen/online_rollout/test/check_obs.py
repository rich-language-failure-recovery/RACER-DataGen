import os
import pickle

from rlbench.backend.observation import Observation
from racer_datagen.utils.const_utils import BASE_PATH

obs_path = f"{BASE_PATH}/runs/rvt/augmented_data_heuristic/train/put_item_in_drawer/0/obs.pkl"

with open(obs_path, "rb") as f:
    obs = pickle.load(f)

print(type(obs))

# for key in obs.keys():
    # if "inter" in key:
    #     obs: Observation = obs[key]
    #     print(obs.get_low_dim_data())
    #     print(obs.gripper_open)
    #     break

# obs: Observation = obs[list(obs.keys())[-1]]
# print(obs.)
print(obs[list(obs.keys())[-1]].gripper_open)
