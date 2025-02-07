"""
LLaVA prompt
- task goal
- previous instruction
- heuristic lang (action transition)
- actual prompt
- image

{ 
    "id": "000000123456", 
    "image": "000000123456.png", 
    "conversations": [ 
        { 
            "from": "human", 
            "value": "task goal + current context + question + \n<image>" 
        }, 
        { 
            "from": "gpt", 
            "value": "failure prediction + current instruction", 
            "heuristic_instruction": "xxx" 
            "label": "start / ongoing / success / recoverable failure / catastrophy failure / end of episode",
            "robot_delta_state": {
                "gripper": "gripper opened / closed",
                "pose_rage" : "moved right, upward and forward / a little .. / didnt move",
                "collision": false,
            }
        } 
    ] 
},
"""

import os
import json
import pickle
from collections import defaultdict

import numpy as np
from racer_datagen.utils.const_utils import *
from racer_datagen.utils.rvt_utils import RLBENCH_TASKS
from racer_datagen.data_aug.base.action import Action
from racer_datagen.online_rollout.base.utils import DIRECTIONS, AXES, ROTATION_SMALL_THRES, ROTATION_LARGE_THRES, TRANSLATION_SMALL_THRES, TRANSLATION_LARGE_THRES
from rlbench.backend.observation import Observation as Obs


# Keypoints
# 1. You are a robot supervisor.
# 2. Your role is to guide the robot to complete the task by giving instructions.
# 3. You can detect failure and correct via instructions.
# 4. You can detect catastrophic failure and call for human help.
SYSTEM_PROMPT = \
"""\
You are a helpful language and vision assistant that can understand the visual content of how a robot arm is interacting with objects in the environment. Your role is decompose the given task goal into subgoals as instructions to guide the robot arm and to complete the task step by step. You are able to understand whether the current robot arm has successfully followed the previous instruction or not. You can detect if the robot has failed to complete the last instruction and provide a new corrective instruction to recover from the failure. Once the robot completes each subgoal, you can provide the next instruction to guide the robot to the next subgoal, until the end of the task. If the robot encounters a catastrophic failure, you should call human for help.\
"""

QUESTION_PROMPT = \
"""\
Based on the visual observation and the context, what would you instruct the robot to do next?\
"""

def build_user_prompt(task_goal, prev_instr=None, prev_action=None):
    if prev_instr is not None:
        PREV_INSTR = \
        f"""\
        In the previous step, the robot arm was given the following instruction: {prev_instr}. Then the robot {prev_action}.\n
        """
    
    question = \
        f"""\
        <image>\n
        Task goal: {task_goal}
        {PREV_INSTR if prev_instr is not None else ""}
        {QUESTION_PROMPT}
        """
    
    return question

# The robot either successfully follows the instruction or fails.
# Failure can be recoverable or catastrophic.
NEXT_INSTRUCTION = "The next instruction is:"
ASSISTANT_START = f"The robot started the task. {NEXT_INSTRUCTION}"
ASSISTANT_ONGOING = f"The robot successfully followed the previous instruction. {NEXT_INSTRUCTION}"
ASSISTANT_SUBGOAL_FAILURE = f"The robot made a recoverable failure. {NEXT_INSTRUCTION}"
ASSISTANT_SUBGOAL_SUCCESS = f"The robot successfully followed the previous instruction. {NEXT_INSTRUCTION}"
ASSISTANT_TASK_SUCCESS = f"The robot completed the task. {NEXT_INSTRUCTION}"


LLAVA_RESPONSE_DICT_INIT = {"from": "gpt", "label": "", "gpt_instruction": "", "heuristic_instruction": ""}


top = 1

TASKS = [RLBENCH_TASKS[0]]

# REAL_WORLD_TASKS = [
#     "open_drawer",
#     "pick_and_place_fruit",
#     "push_buttons",
#     "put_item_in_shelf"
# ]

# TASKS = REAL_WORLD_TASKS

def compute_perturb_score(perturb: Obs, expert: Obs):
    perturb_action = Action.from_numpy(np.hstack((perturb.gripper_pose, perturb.gripper_open, perturb.ignore_collisions)))
    expert_action = Action.from_numpy(np.hstack((expert.gripper_pose, expert.gripper_open, expert.ignore_collisions)))
    delta_action = expert_action.delta_action(perturb_action, expert_action)
    
    translation_cost = np.linalg.norm(delta_action['translation'])
    rotation_cost = np.linalg.norm(delta_action['rotation'])

    cost = {
        "translation": translation_cost,
        "rotation": rotation_cost / 360.
    }

    return cost

def action_transition_heuristic_lang(prev_obs: Obs, curr_obs: Obs):
    prev_action = Action.from_numpy(np.hstack((prev_obs.gripper_pose, prev_obs.gripper_open, prev_obs.ignore_collisions)))
    curr_action = Action.from_numpy(np.hstack((curr_obs.gripper_pose, curr_obs.gripper_open, curr_obs.ignore_collisions)))
    delta_action = curr_action.delta_action(prev_action, curr_action)

    sentence_parts = []

    # Position descriptions
    movements = []
    for direction, axis, sign in DIRECTIONS:
        translation_component = delta_action['translation'][axis]
        if sign * translation_component > TRANSLATION_SMALL_THRES:
            desc = f"moved {direction}"
            if abs(translation_component) < TRANSLATION_LARGE_THRES:
                desc += " a little bit"
            movements.append(desc)
    if not movements:
        movements.append("didn't move its gripper")

    sentence_parts.append(", ".join(movements))

    # Rotation description
    rotation = None
    if np.any(np.abs(delta_action['rotation']) > ROTATION_SMALL_THRES):
        if all(np.abs(delta_action['rotation']) > ROTATION_SMALL_THRES):
            rotation = "rotated the gripper"
        elif np.abs(delta_action['rotation'][2]) > ROTATION_LARGE_THRES:
            rotation = "rotated the gripper about z-axis"
    else:
        rotation = "didn't rotate the gripper"
    
    if rotation:
        sentence_parts.append(rotation)

    # Gripper change
    if delta_action['gripper'] != 0:
        gripper_change = "opened the gripper" if delta_action['gripper'] == 1 else "closed the gripper"
    else:
        gripper_change = "kept the gripper open" if curr_action.gripper_open == 1 else "kept the gripper closed"
    sentence_parts.append(gripper_change)

    # Collision plan
    collision_description = ""
    if delta_action['collision'] != 0:
        collision_description = "that can allow collisions" if delta_action['collision'] == 1 else "that avoids any collision"
        collision_description = f"by planning a motion path {collision_description}"

    # Join parts with proper handling of "and"
    complete_sentence = "Then the robot " + sentence_parts[0]
    if len(sentence_parts) > 1:
        for part in sentence_parts[1:]:
            if part == gripper_change:
                complete_sentence += f", and {part}"
            else:
                complete_sentence += f", {part}"

    # Append collision description last if it exists
    if collision_description:
        complete_sentence += f" {collision_description}"

    complete_sentence += "."

    return complete_sentence

for split in ["train"]:
    num_episodes = 100 if split == "train" else 25

    dataset_list = [f"{split}"]
    # dataset_list = [f"augmented_data_heuristic/real_robot"]

    MIN_THRES_TRANSLATION = 0.03
    MIN_THRES_ROTATION = 0.1
    top_perturbations = False

    for dataset_name in dataset_list:
        save_path = f"{BASE_PATH}/runs/{dataset_name}"

        for task in TASKS:
            for ep in range(num_episodes):
                perturbed_expert_count = 0
                filtered_perturbed_expert_count = 0
                episode_path = os.path.join(save_path, task, str(ep))
                if not os.path.exists(episode_path):
                    continue

                lang_path = os.path.join(save_path, task, str(ep), "language_description.json")

                if os.path.exists(lang_path):
                    with open(lang_path, "r") as f:
                        lang_desc = json.load(f)
                        task_goal = lang_desc["task_goal"]
                        subgoals = lang_desc["subgoal"]
                        expert_steps = list(subgoals.keys())
                        llava_prompt_response_pairs = []
                        num_p = len(subgoals["0_expert"]["gpt-lang"]) # number of perturbations (either 1 or 5)
                        LLAVA_PROMPT_DICT_INIT = {"from": "human", "task_goal": task_goal, "previous_instruction": ""}
                        
                        for i, expert_step in enumerate(expert_steps):
                            perturbation_scores = {"perturb_list": [], "inter_list": [], "score_translation": [], "score_rotation": []}
                            perturbations = []
                            intermediates = []
                            scores_translation = []
                            scores_rotation = []
                            for p in range(num_p):
                                key_mapping_path = os.path.join(save_path, task, str(ep), "output", str(p), "key_mapping.json")
                                q = p
                                llava_json = {"id": "", "image": "", "conversations": []}
                                llava_prompt = LLAVA_PROMPT_DICT_INIT.copy()
                                llava_response = LLAVA_RESPONSE_DICT_INIT.copy()
                                
                                # first process start -> next expert
                                if i == 0:
                                    llava_json["id"] = f"TASK_{task}__EP_{ep}__FROM_start__AT_{expert_step}_{p}"
                                    llava_json["image"] = os.path.join(dataset_name, task, str(ep), "front_rgb", f"{expert_step}.png")
                                    llava_prompt["previous_instruction"] = "The robot is about to start the task."

                                    llava_response["label"] = "start"
                                    gpt = subgoals[expert_step]["gpt-lang"][p]
                                    heuristic = subgoals[expert_step]["heuristic-lang"] # there's only one heuristic lang for expert->expert
                                    llava_response["gpt_instruction"] = f"{ASSISTANT_START} {gpt}"
                                    llava_response["heuristic_instruction"] = f"{ASSISTANT_START} {heuristic}"
                                    llava_response["gpt_instruction_no_label"] = f"{NEXT_INSTRUCTION} {gpt}"
                                    llava_response["heuristic_instruction_no_label"] = f"{NEXT_INSTRUCTION} {heuristic}"

                                    llava_json["conversations"] = [llava_prompt, llava_response]
                                    llava_prompt_response_pairs.append(llava_json)

                                elif i < len(expert_steps):
                                    
                                    if "dense" in expert_step:
                                        continue

                                    if subgoals[expert_step]["augmentation"] != {} and len(subgoals[expert_step]["augmentation"]) > p:
                                        perturbed_expert_count += 1
                                        # recoverable failure state -> intermediate or current expert (handled already from the data itself)
                                        llava_json = {"id": "", "image": "", "conversations": []}
                                        llava_prompt = LLAVA_PROMPT_DICT_INIT.copy()
                                        llava_response = LLAVA_RESPONSE_DICT_INIT.copy()
                                        perturb_name = list(subgoals[expert_step]["augmentation"].keys())
                                        unique_id = len(subgoals[expert_step]["augmentation"][perturb_name[q]]["gpt-lang"]) - 1
                                        llava_json["image"] = os.path.join(dataset_name, task, str(ep), "front_rgb", f"{perturb_name[q]}.png")
                                        idx = subgoals[expert_step]["augmentation"][perturb_name[q]]["idx"]

                                        # naming convention here
                                        llava_json["id"] = f"TASK_{task}__EP_{ep}__FROM_{expert_steps[i-1]}_{p}__AT_{perturb_name[q]}_{unique_id}"
                                        llava_prompt["previous_instruction"] = subgoals[expert_steps[i-1]]["gpt-lang"][p]
                                        
                                        llava_response["label"] = "subgoal failure"
                                        gpt = subgoals[expert_step]["augmentation"][perturb_name[q]]["gpt-lang"][0]
                                        heuristic = subgoals[expert_step]["augmentation"][perturb_name[q]]["heuristic-lang"]
                                        llava_response["gpt_instruction"] = f"{ASSISTANT_SUBGOAL_FAILURE} {gpt}"
                                        llava_response["heuristic_instruction"] = f"{ASSISTANT_SUBGOAL_FAILURE} {heuristic}"
                                        llava_response["gpt_instruction_no_label"] = f"{NEXT_INSTRUCTION} {gpt}"
                                        llava_response["heuristic_instruction_no_label"] = f"{NEXT_INSTRUCTION} {heuristic}"
                                        llava_json["conversations"] = [llava_prompt, llava_response]

                                        if not (i == len(expert_steps) - 1 and task in ["insert_onto_square_peg", "meat_off_grill", "place_shape_in_shape_sorter", "put_item_in_drawer", "turn_tap"]) and not (i == len(expert_steps) - 2 and task in ["put_groceries_in_cupboard"]):
                                            perturbations.append(llava_json)

                                        # intermediate -> current expert (if there was an intermediate step)
                                        if subgoals[expert_step]["augmentation"][perturb_name[q]].get("correction", {}) != {}:
                                            llava_json = {"id": "", "image": "", "conversations": []}
                                            llava_prompt = LLAVA_PROMPT_DICT_INIT.copy()
                                            llava_response = LLAVA_RESPONSE_DICT_INIT.copy()
                                            perturb_name = list(subgoals[expert_step]["augmentation"].keys())
                                            inter_name = list(subgoals[expert_step]["augmentation"][perturb_name[q]]["correction"].keys())[0]
                                            unique_id = len(subgoals[expert_step]["augmentation"][perturb_name[q]]["correction"][inter_name]["gpt-lang"]) - 1

                                            llava_json["image"] = os.path.join(dataset_name, task, str(ep), "front_rgb", f"{inter_name}.png")
                                            llava_json["id"] = f"TASK_{task}__EP_{ep}__FROM_{perturb_name[q]}_{q}__AT_{inter_name}_{unique_id}"
                                            llava_prompt["previous_instruction"] = subgoals[expert_step]["augmentation"][perturb_name[q]]["gpt-lang"][0]

                                            llava_response["label"] = "ongoing"
                                            gpt = subgoals[expert_step]["augmentation"][perturb_name[q]]["correction"][inter_name]["gpt-lang"][0]
                                            heuristic = subgoals[expert_step]["augmentation"][perturb_name[q]]["correction"][inter_name]['heuristic-lang']
                                            llava_response["gpt_instruction"] = f"{ASSISTANT_ONGOING} {gpt}"
                                            llava_response["heuristic_instruction"] = f"{ASSISTANT_ONGOING} {heuristic}"
                                            llava_response["gpt_instruction_no_label"] = f"{NEXT_INSTRUCTION} {gpt}"
                                            llava_response["heuristic_instruction_no_label"] = f"{NEXT_INSTRUCTION} {heuristic}"
                                            llava_json["conversations"] = [llava_prompt, llava_response]
                                            # llava_prompt_response_pairs.append(llava_json)
                                            if not (i == len(expert_steps) - 1 and task in ["insert_onto_square_peg", "meat_off_grill", "place_shape_in_shape_sorter", "put_item_in_drawer", "turn_tap"]) and not (i == len(expert_steps) - 2 and task in ["put_groceries_in_cupboard"]):
                                                intermediates.append(llava_json)
                                        
                                    # add a direct curr expert -> next expert
                                    llava_json = {"id": "", "image": "", "conversations": []}
                                    llava_prompt = LLAVA_PROMPT_DICT_INIT.copy()
                                    llava_response = LLAVA_RESPONSE_DICT_INIT.copy()
                                    llava_json["image"] = os.path.join(dataset_name, task, str(ep), "front_rgb", f"{expert_step}.png")
                                    
                                    if subgoals[expert_step]["augmentation"] != {} and len(subgoals[expert_step]["augmentation"]) > p and subgoals[expert_step]["augmentation"][perturb_name[q]].get("correction", {}) != {}:
                                        # perturb -> (intermediate -> curr expert) -> next expert

                                        llava_prompt["previous_instruction"] = subgoals[expert_step]["augmentation"][perturb_name[q]]["correction"][inter_name]["gpt-lang"][0]
                                        llava_json["id"] = f"TASK_{task}__EP_{ep}__FROM_{inter_name}_0__AT_{expert_step}_{p}"

                                        llava_response["label"] = "subgoal success"
                                        gpt = subgoals[expert_step]["gpt-lang"][p]
                                        heuristic = subgoals[expert_step]["heuristic-lang"]
                                        llava_response["gpt_instruction"] = f"{ASSISTANT_SUBGOAL_SUCCESS} {gpt}"
                                        llava_response["heuristic_instruction"] = f"{ASSISTANT_SUBGOAL_SUCCESS} {heuristic}"
                                        llava_response["gpt_instruction_no_label"] = f"{NEXT_INSTRUCTION} {gpt}"
                                        llava_response["heuristic_instruction_no_label"] = f"{NEXT_INSTRUCTION} {heuristic}"
                                        llava_json["conversations"] = [llava_prompt, llava_response]
                                        if not (i == len(expert_steps) - 1 and task in ["insert_onto_square_peg", "meat_off_grill", "place_shape_in_shape_sorter", "put_item_in_drawer", "turn_tap"]) and not (i == len(expert_steps) - 2 and task in ["put_groceries_in_cupboard"]):
                                            llava_prompt_response_pairs.append(llava_json)
                                        
                                    elif subgoals[expert_step]["augmentation"] != {} and len(subgoals[expert_step]["augmentation"]) > p and subgoals[expert_step]["augmentation"][perturb_name[q]].get("correction", {}) == {}:
                                        # (perturb -> curr expert) -> next expert
                                        llava_prompt["previous_instruction"] = subgoals[expert_step]["augmentation"][perturb_name[q]]["gpt-lang"][0]
                                        llava_json["id"] = f"TASK_{task}__EP_{ep}__FROM_{perturb_name[q]}_{q}__AT_{expert_step}_{p}"

                                        llava_response["label"] = "subgoal success"
                                        gpt = subgoals[expert_step]["gpt-lang"][p]
                                        heuristic = subgoals[expert_step]["heuristic-lang"]
                                        llava_response["gpt_instruction"] = f"{ASSISTANT_SUBGOAL_SUCCESS} {gpt}"
                                        llava_response["heuristic_instruction"] = f"{ASSISTANT_SUBGOAL_SUCCESS} {heuristic}"
                                        llava_response["gpt_instruction_no_label"] = f"{NEXT_INSTRUCTION} {gpt}"
                                        llava_response["heuristic_instruction_no_label"] = f"{NEXT_INSTRUCTION} {heuristic}"
                                        llava_json["conversations"] = [llava_prompt, llava_response]
                                        if not (i == len(expert_steps) - 1 and task in ["insert_onto_square_peg", "meat_off_grill", "place_shape_in_shape_sorter", "put_item_in_drawer", "turn_tap"]) and not (i == len(expert_steps) - 2 and task in ["put_groceries_in_cupboard"]):
                                            llava_prompt_response_pairs.append(llava_json)

                                    llava_json = {"id": "", "image": "", "conversations": []}
                                    llava_prompt = LLAVA_PROMPT_DICT_INIT.copy()
                                    llava_response = LLAVA_RESPONSE_DICT_INIT.copy()
                                    src_p = p
                                    dst_p = p + 1 if p + 1 < num_p else 0

                                    llava_json["id"] = f"TASK_{task}__EP_{ep}__FROM_{expert_steps[i-1]}_{src_p}__AT_{expert_step}_{dst_p}"
                                    llava_json["image"] = os.path.join(dataset_name, task, str(ep), "front_rgb", f"{expert_step}.png")
                                    # (prev expert -> curr expert) -> next expert
                                    if isinstance(subgoals[expert_steps[i-1]]["gpt-lang"], list):
                                        llava_prompt["previous_instruction"] = subgoals[expert_steps[i-1]]["gpt-lang"][src_p]
                                    elif isinstance(subgoals[expert_steps[i-1]]["gpt-lang"], str):
                                        llava_prompt["previous_instruction"] = subgoals[expert_steps[i-1]]["gpt-lang"]

                                    if i == len(expert_steps) - 1:
                                        llava_response["label"] = "task success"
                                        heuristic = "End of episode."
                                    else:
                                        llava_response["label"] = "subgoal success"
                                        heuristic = subgoals[expert_step]["heuristic-lang"]
                                    
                                    gpt = subgoals[expert_step]["gpt-lang"][dst_p]
                                    llava_response["gpt_instruction"] = f"{ASSISTANT_SUBGOAL_SUCCESS} {gpt}"
                                    llava_response["heuristic_instruction"] = f"{ASSISTANT_SUBGOAL_SUCCESS} {heuristic}"
                                    llava_response["gpt_instruction_no_label"] = f"{NEXT_INSTRUCTION} {gpt}"
                                    llava_response["heuristic_instruction_no_label"] = f"{NEXT_INSTRUCTION} {heuristic}"
                                    llava_json["conversations"] = [llava_prompt, llava_response]
                                    llava_prompt_response_pairs.append(llava_json)
        

                            if top_perturbations:
                                if len(perturbations) > 0:
                                    perturbation_scores["perturb_list"] = perturbations
                                    if len(intermediates) > 0:
                                        perturbation_scores["inter_list"] = intermediates
                                    perturbation_scores["score_translation"] = scores_translation
                                    perturbation_scores["score_rotation"] = scores_rotation

                                    if len(intermediates) > 0:
                                        for i, (perturbation, intermediate, score_trans, score_rot) in enumerate(zip(perturbation_scores["perturb_list"], perturbation_scores["inter_list"], perturbation_scores["score_translation"], perturbation_scores["score_rotation"])):
                                            if score_trans > MIN_THRES_TRANSLATION or score_rot > MIN_THRES_ROTATION:
                                                filtered_perturbed_expert_count += 1
                                                llava_prompt_response_pairs.append(perturbation)
                                                llava_prompt_response_pairs.append(intermediate)
                                                # print(perturbation_scores["perturb_list"][i]['id'], round(score_trans,4), round(score_rot, 4), filtered_perturbed_expert_count)
                                    else:
                                        for i, (perturbation, score_trans, score_rot) in enumerate(zip(perturbation_scores["perturb_list"], perturbation_scores["score_translation"], perturbation_scores["score_rotation"])):
                                            if score_trans > MIN_THRES_TRANSLATION or score_rot > MIN_THRES_ROTATION:
                                                filtered_perturbed_expert_count += 1
                                                llava_prompt_response_pairs.append(perturbation)
                                                # print(perturbation_scores["perturb_list"][i]['id'], round(score_trans, 4), round(score_rot, 4), filtered_perturbed_expert_count)

                                    # for i in range(5):
                                    #     print(perturbation_scores["perturb_list"][i]['id'])
                                    # print(perturbation_scores['score_translation'])
                                    # print(perturbation_scores['score_rotation'])

                            else:
                                llava_prompt_response_pairs.extend(perturbations)
                                llava_prompt_response_pairs.extend(intermediates)

                        out_path = os.path.join(save_path, task, str(ep), f"llava.json")
                        with open(out_path, "w") as f:
                            json.dump(llava_prompt_response_pairs, f, indent=4)

                        print(out_path, len(llava_prompt_response_pairs), f"{perturbed_expert_count}  ->  {filtered_perturbed_expert_count}")
                        # print(f"perturbed_expert_count: {perturbed_expert_count} | filtered_perturbations_count: {filtered_perturbed_expert_count}")


            #     break
            # break