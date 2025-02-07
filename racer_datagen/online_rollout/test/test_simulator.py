import numpy as np
from racer_datagen.utils.const_utils import *
from racer_datagen.online_rollout.base.simulator import RLBenchSim
import cv2

sim = RLBenchSim(
    task_name="close_jar",
    dataset_root=f"{BASE_PATH}/data/test",
    episode_length=30,
    record_every_n=-1
)
obs = sim.reset(episode_num=0)
print(np.shape(obs["front_rgb"])) # (3, 512, 512)
print(np.shape(obs["low_dim_state"]))
print(np.shape(obs["ignore_collisions"]))
# low_dim_state

front_rgb = obs["front_rgb"].transpose(1, 2, 0)
# cv2 bgr
front_rgb = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR)
# save image
cv2.imwrite(f"{BASE_PATH}/online_rollout/test/current.png", front_rgb)
sim.close()