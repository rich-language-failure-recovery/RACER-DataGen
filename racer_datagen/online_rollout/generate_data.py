"""
task_list = [
    "put_item_in_drawer",             # 0                 
    "reach_and_drag",                 # 1                 
    "turn_tap",                       # 2  --> [0:3]      
    "slide_block_to_color_target",    # 3                 
    "open_drawer",                    # 4                 
    "put_groceries_in_cupboard",      # 5  --> [3:6]      
    "place_shape_in_shape_sorter",    # 6                 
    "put_money_in_safe",              # 7                 
    "push_buttons",                   # 8  --> [6:9]      
    "close_jar",                      # 9                 
    "stack_blocks",                   # 10                
    "place_cups",                     # 11 --> [9:12]     
    "place_wine_at_rack_location",    # 12                
    "light_bulb_in",                  # 13                
    "sweep_to_dustpan_of_size",       # 14 --> [12:15]    
    "insert_onto_square_peg",         # 15                
    "meat_off_grill",                 # 16                
    "stack_cups",                     # 17 --> [15:18]
]
"""


import os
import time
import pickle
import argparse

from racer_datagen.utils.const_utils import *
from racer_datagen.utils.rvt_utils import RLBENCH_TASKS
from racer_datagen.online_rollout.base.utils import logger
from racer_datagen.data_aug.base.action import Action
from racer_datagen.data_aug.base.waypoint_and_perturbation import WayPoint, Perturbation
from racer_datagen.data_aug.failing_expert_demos import FAILED_EXPERT_TRAIN_DEMOS
from racer_datagen.online_rollout.base.constants import *
from racer_datagen.online_rollout.base.rollout_agent import RolloutAgent
from racer_datagen.online_rollout.base.rollout_validation import RolloutValidation, find_action_difference
from racer_datagen.prompts.episode_to_json_converter import episode_to_json_converter
from racer_datagen.prompts.save_gif_with_gpt import save_gif_with_gpt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate data for RACER")
    parser.add_argument('--start_idx', type=int, default=0, help='Start index of the task list')
    parser.add_argument('--end_idx', type=int, default=None, help='End index of the task list')
    parser.add_argument('--num_of_ep', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--ep_offset', type=int, default=0, help='Offset for episode numbering')
    parser.add_argument('--M', type=int, default=1, help='Number of heuristic perturbations')
    parser.add_argument('--train', action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def total_num_of_rollouts(M: int, validate_heuristic: bool, validate_model: bool, rollout_expert: bool):
    # assert(validate_heuristic or validate_model)
    total = 0
    if validate_heuristic:
        total += M      # M heuristic perturbations
    if validate_model:
        total += 1      # 1 model perturbation
    if rollout_expert:
        total += 1      # 1 expert rollout
    return total

def main():
    args = parse_arguments()

    N = 20          # number of validated perturbations to generate for each keypoint
    M = args.M      # number of validated perturbations to store for each keypoint
    number_of_episodes = args.num_of_ep
    task_list = RLBENCH_TASKS[args.start_idx : args.end_idx] # start_idx is inclusive, end_idx is exclusive

    split = "train" if args.train else "val"
    save_path = f"{BASE_PATH}/runs/{split}"
    
    # just for initializing RolloutAgent (task_name, episode_num) doesn't matter; 
    # these are updated every loop through env_reset
    kwargs = {
        "task_name":"close_jar",
        "episode_num":1,
        "dataset_root":f"{BASE_PATH}/data/{split}",
        "save_dir":save_path,
        "heuristics_cfg_root":f"{BASE_PATH}/data_aug/configs",
        "model_path":f"{BASE_PATH}/runs/rvt/model_14.pth",
        "device":0
    }

    VALIDATE_EXPERT = True                      # check the expert demo first
    VALIDATE_HEURISTIC_PERTURBATION = True      # check if heuristic perturbation and recovery can succeed
    VALIDATE_MODEL_PERTURBATION = False         # check if model perturbation and recovery can succeed
    ROLLOUT_EXPERT = False                      # rollout the expert demo
    PROMPT_GPT = False                           # prompt GPT to generate language description

    TOTAL_M = total_num_of_rollouts(M, VALIDATE_HEURISTIC_PERTURBATION, VALIDATE_MODEL_PERTURBATION, ROLLOUT_EXPERT)

    results = []

    kwargs["model_path"] = None # comment this out to load model parameters
    rollout = RolloutAgent(**kwargs, debug=False)
    rollout_validator = RolloutValidation(N, M)

    for task in task_list:
        for ep in list(range(number_of_episodes)):
            ep += args.ep_offset

            if os.path.exists(os.path.join(save_path, task, str(ep), "language_description.json")):
                continue

            print("======================================")
            print(f"Task: {task} | Episode: {ep}")
            print("======================================")

            if not os.path.exists(os.path.join(save_path, "retry")):
                os.makedirs(os.path.join(save_path, "retry"))
            if not os.path.exists(os.path.join(save_path, "log")):
                os.makedirs(os.path.join(save_path, "log"))

            with open(os.path.join(save_path, "log", f"log_{task}.txt"), "a") as f:
                # put current time in the file
                f.write(f"Time: {time.ctime()} | Task: {task} | Episode: {ep} \n")

            # 1. Check if expert keypoints can succeed
            if VALIDATE_EXPERT:
                print("Validating expert rollout...")
                if split == "train" and task in FAILED_EXPERT_TRAIN_DEMOS and ep in FAILED_EXPERT_TRAIN_DEMOS[task]:
                    with open(os.path.join(save_path, "retry", f"failed_episode_{task}.txt"), "a") as f:
                        f.write(f"Time: {time.ctime()} | Task: {task} | Episode: {ep} | Expert known to fail \n")
                    continue

                for _ in range(3):
                    rollout.env_reset(task_name=task, episode_num=ep, is_record=False)
                    rollout._expert_rollout(rollout.keywpts[0], rollout.keywpts[-1], is_reset=False, is_record=False)
                    expert_success = rollout.is_success()
                    if expert_success:
                        break
                
                # don't save the data for failed expert demo (note in failed_episode_{task}.txt)
                if not expert_success:
                    with open(os.path.join(save_path, "retry", f"failed_episode_{task}.txt"), "a") as f:
                        f.write(f"Time: {time.ctime()} | Task: {task} | Episode: {ep} | Expert failed \n")
                    continue
            else:
                rollout.env_reset(task_name=task, episode_num=ep, is_record=False)
            
            # 2. Validate heuristic perturbation
            if VALIDATE_HEURISTIC_PERTURBATION:
                rollout.env_reset()
                validated_heuristic_episode = rollout_validator.validate_heuristic_rollout(rollout, is_record=False)
            
            # 3. Validate model perturbation
            if VALIDATE_MODEL_PERTURBATION:
                rollout.env_reset()
                validated_model_episode = rollout_validator.validate_model_rollout(rollout, is_record=False)

            # 4. Merge heuristic and model perturbations
            if VALIDATE_HEURISTIC_PERTURBATION and VALIDATE_MODEL_PERTURBATION:   
                validated_heuristic_episode.merge_episodes(validated_model_episode)
            
            # 5. Actual rollout and annotate using GPT.
            if VALIDATE_HEURISTIC_PERTURBATION:
                annotated_valid_episode = rollout.rollout_episode(validated_heuristic_episode, TOTAL_M, prompt_gpt=PROMPT_GPT)
            else:
                annotated_valid_episode = rollout.rollout_episode(validated_model_episode, TOTAL_M, prompt_gpt=PROMPT_GPT)
            
            # save annotated_valid_episode as annotated_episode.pkl
            annotated_valid_episode.save_path = os.path.join(rollout.ep_save_path, 'annotated_episode.pkl')
            with open(annotated_valid_episode.save_path, "wb") as f:
                pickle.dump(annotated_valid_episode, f)

            if annotated_valid_episode.success:
                if PROMPT_GPT:
                    episode_to_json_converter(rollout.ep_save_path, save_path, episode=annotated_valid_episode)
                    
                    if VALIDATE_HEURISTIC_PERTURBATION:
                        if M != 0:
                            num = M
                        else:
                            num = 1

                        for m in range(num):
                            image_dir = os.path.join(rollout.ep_save_path, "front_rgb")
                            save_gif_with_gpt(
                                task_name=task,
                                lang_goal=annotated_valid_episode.lang_goal,
                                episode_num=ep, 
                                ep_save_path=rollout.ep_save_path,
                                input_file=os.path.join(rollout.ep_save_path, "language_description.json"), 
                                output_file=os.path.join(rollout.ep_save_path, "output", f"{m}", "sorted_output.json"), 
                                image_dir=image_dir, 
                                filenames=[file_name for file_name in os.listdir(image_dir)], 
                                perturb_idx=m, 
                                rollout_type="heuristic"
                            )
            else:
                # save into text file the failed episode
                with open(os.path.join(save_path, "retry", f"failed_episode_{task}.txt"), "a") as f:
                        f.write(f"* Time: {time.ctime()} | Task: {task} | Episode: {ep} | Success: {annotated_valid_episode.success} -> Retry \n")
            results.append(f"Task: {task} | Episode: {ep} | Success: {annotated_valid_episode.success}")
        print(f"Log path for failed episodes to retry: {os.path.join(save_path, 'retry', f'failed_episode_{task}.txt')}")
        print(f"Log path: {os.path.join(save_path, 'log', f'log_{task}.txt')}")
    rollout.close()

if __name__ == "__main__":
    # 1. Run expert to see if it can succeed.
    # 2. If success, we validate heuristic and rvt perturbations on the expert demo.
    # 3. Merge the heuristic and rvt perturbations and save the annotated episode after rolling out.
    main()