import random
from racer_datagen.utils.const_utils import *
from racer_datagen.utils.rvt_utils import RLBENCH_TASKS
from racer_datagen.online_rollout.base.rollout_agent import RolloutAgent

episode_num = random.randint(0, 99)

rollout = RolloutAgent(
    # stack_cups, insert_onto_square_peg, place_shape_in_shape_sorter, slide_block_to_color_target, put_item_in_drawer
    task_name="insert_onto_square_peg",
    episode_num=15,
    dataset_root=f"{BASE_PATH}/data/train",
    save_dir=f"{BASE_PATH}/runs/rvt/testing_v2",   # CHANGE THIS!!
    heuristics_cfg_root=f"{BASE_PATH}/data_aug/configs",
    model_path=f"{BASE_PATH}/runs/rvt/model_14.pth",
    device=0,
    debug=False,
)

rollout.rollout_heuristic_only(N=5, num_perturb=1)
# success = rollout.rollout_expert_only()
# print(success)
# rollout.rollout_cmd_only()
# rollout.rollout_model_only()

print(episode_num)