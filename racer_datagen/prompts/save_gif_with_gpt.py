import os
import re
import json
import shutil
from PIL import Image
import imageio
import numpy as np
from functools import partial
from racer_datagen.utils.const_utils import *
from racer_datagen.online_rollout.base.constants import *
from racer_datagen.online_rollout.base.utils import logger, append_text_underneath_image, sort_key

np.set_printoptions(precision=4, suppress=True)

def flatten_subgoals(subgoals):
    flat_subgoals = {}

    def extract_subgoals(subgoal_dict, prefix=''):
        for key, subgoal in subgoal_dict.items():
            new_key = f"{prefix}{key}"
            subgoal_copy = {k: v for k, v in subgoal.items() if k not in ['augmentation', 'correction']}
            flat_subgoals[new_key] = subgoal_copy
            if 'augmentation' in subgoal:
                extract_subgoals(subgoal['augmentation'])
            if 'correction' in subgoal:
                extract_subgoals(subgoal['correction'])

    extract_subgoals(subgoals)
    return flat_subgoals

def sort_and_filter_flat_subgoals(flat_subgoals, perturb_idx=None):
    # Filter subgoals first if perturb_idx is specified
    if perturb_idx is not None:
        filtered_subgoals = {key: val for key, val in flat_subgoals.items() if f"_{perturb_idx}" in key or "expert" in key or "dense" in key}
    else:
        filtered_subgoals = flat_subgoals

    # Custom sorting function to sort by numeric prefix and specific order
    def sort_key(item):
        key = item[0]
        num = int(re.search(r'\d+', key).group())
        if "perturb" in key:
            return (num, 0)
        elif "intermediate" in key:
            return (num, 1)
        else:  # "expert" in key
            return (num, 2)

    sorted_subgoals = sorted(filtered_subgoals.items(), key=sort_key)
    return dict(sorted_subgoals)

def save_gif_with_gpt(task_name, lang_goal, episode_num, ep_save_path, input_file, output_file, image_dir, filenames, perturb_idx, rollout_type):
    for f in filenames:
        split_name = os.path.splitext(f)[0].split('_')
        if rollout_type is None and len(split_name) > 3 and split_name[-1] == str(perturb_idx):
            rollout_type = split_name[2] # heuristic or rvt or other model names
    if rollout_type is None:
        rollout_type = WptType.EXPERT
    sorted_files = sorted(filenames, key=partial(sort_key, type=rollout_type))
    filtered_files = []
    for f in sorted_files:
        split_name = os.path.splitext(f)[0].split('_')
        if split_name[1] == 'expert' or split_name[-1] == str(perturb_idx) or split_name[1] == "dense":
            filtered_files.append(f)

    # Read the input JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Sort the subgoals
    flat_subgoals = flatten_subgoals(data['subgoal'])

    # Update the original data with sorted subgoals
    data['subgoal'] = sort_and_filter_flat_subgoals(flat_subgoals, perturb_idx)

    # Save the sorted dictionary back to a JSON file
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Sorted data has been saved to {output_file}")

    tmp2_img_paths = []
    # make new dir
    tmp2_dir = os.path.join(ep_save_path, "output", f"{perturb_idx}", "tmp2")
    if not os.path.exists(tmp2_dir):
        os.makedirs(tmp2_dir)


    # action_list = []
    # with open(os.path.join(ep_save_path, "actions.csv"), 'r') as file:
    #     for line in file:
    #         values = line.strip().split(',')
    #         np_array = np.array(values, dtype=float)
    #         action_list.append(np_array)

    # read json file  os.path.join(ep_save_path, "output", f"{perturb_idx}", "prompt.json")
    with open(os.path.join(ep_save_path, "output", f"{perturb_idx}", "prompt.json"), 'r') as file:
        prompt = json.load(file)

    i = 0
    for f in filtered_files:
        text = []

        parts = f.split('_')
        keypoint_number = int(parts[0])
        action_type = f.split('_')[1].split('.')[0]

        robot_pose_dict = prompt[f"{i}"]["current_timestep"].get("robot_pose", {})
        if robot_pose_dict == {}:
            robot_pose_dict = prompt[f"{i-1}"]["desired_state_at_next_timestep"].get("robot_pose", {})

        text += [f"{task_name} | ep {episode_num}"]
        text += [f"{lang_goal}"]
        text += [f"idx {i} | kypt {keypoint_number} | {action_type}"]

        if robot_pose_dict != {}:
            gripper = "gripper open" if robot_pose_dict["gripper_open"] else "gripper close"
            collision = "ignore collision" if robot_pose_dict["ignore_collision"] else "consider collision"
            text += [
                robot_pose_dict["position"],
                robot_pose_dict["orientation"], 
                f"{gripper} | {collision} \n"
            ]

        keys = list(data['subgoal'].keys())
        if task_name == "place_cups":
            if "dense" in f:

                text += ["Continue."]
            else:
                if len(data['subgoal'][keys[i]]["gpt-lang"]) > 1:
                    text += [data['subgoal'][keys[i]]["gpt-lang"][perturb_idx]]
                else:
                    text += data['subgoal'][keys[i]]["gpt-lang"]
        else:
            if len(data['subgoal'][keys[i]]["gpt-lang"]) > 1:
                text += [data['subgoal'][keys[i]]["gpt-lang"][perturb_idx]]
            else:
                text += data['subgoal'][keys[i]]["gpt-lang"]
        image = Image.open(os.path.join(image_dir, f))
        if isinstance(text, str):
            text = [text]
        annotated_image = append_text_underneath_image(np.array(image), texts=text, max_text_height=500)

        tmp_img_path = os.path.join(tmp2_dir, os.path.basename(f"{keys[i]}.png"))
        Image.fromarray(annotated_image).save(tmp_img_path)
        tmp2_img_paths.append(tmp_img_path)

        output_filename = os.path.join(ep_save_path, "output", f"{perturb_idx}", "gif", f"annotated_{rollout_type}_gpt.gif")
        imageio.mimsave(os.path.join(tmp2_dir, output_filename), [imageio.imread(image_path) for image_path in tmp2_img_paths], duration=1000)

        if "dense" not in f:
            i += 1
    print(f"Saved {output_filename}")
    
    shutil.rmtree(tmp2_dir)






def main():
    task_name = "open_drawer"
    episode_num = 5
    input_file = f'{BASE_PATH}/runs/rvt/0523_gpt_lang_check/{task_name}/{episode_num}/language_description.json'
    output_file = f'{BASE_PATH}/runs/rvt/0523_gpt_lang_check/{task_name}/{episode_num}/output/0/sorted_output.json'
    image_dir = f'{BASE_PATH}/runs/rvt/0523_gpt_lang_check/{task_name}/{episode_num}/front_rgb'
    filenames = [file_name for file_name in os.listdir(image_dir)]
    perturb_idx = 0
    rollout_type = "heuristic"
    save_gif_with_gpt(task_name, episode_num, input_file, output_file, image_dir, filenames, perturb_idx, rollout_type)



if __name__ == "__main__":
    main()