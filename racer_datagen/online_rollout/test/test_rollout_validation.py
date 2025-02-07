import os 
import time
import copy
import pickle
import logging
logging.basicConfig(level=logging.ERROR) 

from racer_datagen.utils.const_utils import *
from racer_datagen.utils.rvt_utils import RLBENCH_TASKS
from racer_datagen.online_rollout.base.utils import logger
from racer_datagen.data_aug.base.action import Action
from racer_datagen.data_aug.base.waypoint_and_perturbation import WayPoint, Perturbation
from racer_datagen.online_rollout.base.constants import *
from racer_datagen.online_rollout.base.rollout_agent import RolloutAgent
from racer_datagen.online_rollout.base.rollout_validation import RolloutValidation, find_action_difference

# Tasks
# "put_item_in_drawer",
# "reach_and_drag",
# "turn_tap",
# "slide_block_to_color_target",
# "open_drawer",
# "put_groceries_in_cupboard",
# "place_shape_in_shape_sorter",
# "put_money_in_safe",
# "push_buttons",
# "close_jar",
# "stack_blocks",
# "place_cups",
# "place_wine_at_rack_location",
# "light_bulb_in",
# "sweep_to_dustpan_of_size",
# "insert_onto_square_peg",
# "meat_off_grill",
# "stack_cups",

save_path = f"{BASE_PATH}/runs/debug/"
kwargs = {
    "task_name":"place_cups",
    "episode_num":30,
    "dataset_root":f"{BASE_PATH}/data/train",
    "save_dir":save_path,
    "heuristics_cfg_root":f"{BASE_PATH}/data_aug/configs",
    "model_path":f"{BASE_PATH}/runs/rvt/model_14.pth",
    "device":0
}

def main():
    N = 5   # set this 0 to see expert demo only
    M = 1
    debug = False
    HEURISTIC = True # model if False
    TOTAL_M = None

    rollout = RolloutAgent(**kwargs, debug=debug)
    rollout_validator = RolloutValidation(N=N, M=M)

    # if you want to test expert rollout
    rollout.env_reset()
    rollout._expert_rollout(rollout.keywpts[0], rollout.keywpts[-1], is_reset=False, is_record=False)
    expert_success = rollout.is_success()
    print("SUCCCCCESS:", expert_success)

    if not expert_success:
        print("Expert failed. Rollout terminated.")
        return
    rollout.env_reset()
    if HEURISTIC:
        validated_episode = rollout_validator.validate_heuristic_rollout(rollout, is_record=True)
        TOTAL_M = M
    else:
        validated_episode = rollout_validator.validate_model_rollout(rollout, is_record=True)
        TOTAL_M = 1
    
    annotated_valid_episode = rollout.rollout_episode(validated_episode, TOTAL_M, prompt_gpt=False)
    with open(annotated_valid_episode.save_path, 'wb') as file:
        pickle.dump(annotated_valid_episode, file)
        print(f"Annotated episode saved in {annotated_valid_episode.save_path}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
