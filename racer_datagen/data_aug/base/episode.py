
import os
import re
import copy
import json
import time
import shutil
import imageio
import numpy as np
from PIL import Image

from typing import Union, List
from functools import partial
from collections.abc import Iterator
from dataclasses import dataclass

from rlbench.demo import Demo

from .action import Action
from .waypoint_and_perturbation import WayPoint
from racer_datagen.online_rollout.base.utils import logger, append_text_underneath_image, sort_key
from racer_datagen.online_rollout.base.constants import *
from racer_datagen.libs.peract.helpers.demo_loading_utils import keypoint_discovery, keypoint_discovery_v2, is_gripper_open, is_gripper_open_v2

from racer_datagen.prompts.main_prompt import *
from racer_datagen.prompts.task_specific_descriptions import *
from racer_datagen.utils.const_utils import *

from racer_datagen.prompts.api import AzureConfig, ChatAPI

np.set_printoptions(precision=4, suppress=True)

@dataclass
class Episode:
    waypoints: list[WayPoint]
    task_name: str
    episode_num: int = None
    lang_goal: str = ""             # language goal for the task & epidode
    success: bool = True            # whether the expert demo is successful
    post_processed: bool = False    # set True after update_info() is called
    save_path: str = None           # save path to the perturbed episode rollout data
    
    def retrieve_index(self, id: int, type: str) -> int:
        for i, wp in enumerate(self.waypoints):
            if wp.id == id and wp.type == type:
                return i
        return None
    
    def __getitem__(self, index: Union[int, List[int]]) -> Union[WayPoint, List[WayPoint]]:
        if isinstance(index, list):
            return [self.waypoints[i] for i in index]
        else:
            return self.waypoints[index]

    def __len__(self) -> int:
        return len(self.waypoints)
    
    def retrieve_keypoints(self, start_index: int=0, end_index: int=1e5):
        # retrieve the keypoint index between start_index and the end_index (end_index not included)
        assert start_index >= 0 and start_index < len(self), \
            "start_index should be within the range of the episode"
        
        if start_index < end_index:
            start, step, end = start_index, 1, min(len(self), end_index)
        else:
            start, step, end = start_index, -1, max(-1, end_index)
                
        for i in range(start, end, step):
            if self.waypoints[i].type == 'keypoint':
                yield i
    
    def all_keypoints_idx(self):
        return [i for i, wpt in enumerate(self.waypoints) if wpt.type == 'keypoint']

    def sample_previous_intermediate_waypoints(self, keypoint_index: int, lowerbound:float=2/4, upperbound:float=3/4) -> WayPoint:
        # sample backward, take the middle 1/3 waypoints
        assert self.waypoints[keypoint_index].type == 'keypoint', "The index should be a keypoint"
        
        possible_indices = []
        for i in range(keypoint_index-1, -1, -1):
            if self.waypoints[i].type == 'intermediate':
                possible_indices.append(i)
            elif self.waypoints[i].type == 'keypoint':
                break
        possible_indices_len = len(possible_indices)
        # take the middle 1/3 waypoints
        possible_indices = possible_indices[int(lowerbound * possible_indices_len) : int(upperbound * possible_indices_len)]
        sample_index = np.random.choice(possible_indices)
        return self.waypoints[sample_index]

    def return_(self):
        indices, keypoints = [], []
        for idx, wpt in enumerate(self.waypoints):
            if wpt.type == "keypoint":
                indices.append(idx)
                keypoints.append(wpt)
        return keypoints, indices
    
    def remove_all_perturbations(self):
        keypoint_idx = self.all_keypoints_idx()
        for idx in keypoint_idx:
            self.waypoints[idx].perturbations.clear()
    
    def print_info(self):
        for idx, step in enumerate(list(self.retrieve_keypoints())):
            kwpt = self.waypoints[step]
            print(f"idx {idx} expert step {step}:", kwpt.action.to_interaction(), " | ", kwpt.verbose)
            for i, perturb_wpt in enumerate(kwpt.perturbations):
                print(f"\tperturb {i}:", perturb_wpt.mistake.action.to_interaction())
                if perturb_wpt.correction is not None:
                    print(f"\tcorrect {i}:", perturb_wpt.correction.action.to_interaction())
                else:
                    print(f"\tcorrect {i}:", None)

    def iterate_one_keypoint(self, wpt_id: int, num_perturb=1, verbose=False, perturb_idx: int = None) -> Iterator[WayPoint]:
        if len(self.waypoints[wpt_id].perturbations) == 0:
            if verbose:
                print(f"\tNo perturbation at keypoint {wpt_id}, returning original keypoint action.")
            self.waypoints[wpt_id].info.update({WptInfoKey.WPT_TYPE: WptType.EXPERT, WptInfoKey.WPT_ID: wpt_id})
            yield self.waypoints[wpt_id]
        else:
            if verbose:
                verbose_str = ', '.join(f"{key}: {BLUE}{value}{RESET}" if key == 'is' else f"{key}: {value}" for key, value in self.waypoints[wpt_id].verbose.items())
                print(f"\tStart perturbation at keypoint {wpt_id} {verbose_str}")
            perturbations = self.waypoints[wpt_id].perturbations
            start_idx = perturb_idx if perturb_idx is not None else 0
            end_idx = perturb_idx + 1 if perturb_idx is not None else num_perturb

            for idx in range(start_idx, min(end_idx, len(perturbations))):
                perturb = perturbations[idx]
                if verbose:
                    print(f"\tmake {RED}mistake{RESET} at keypoint {wpt_id} for perturb {idx}")
                perturb.mistake.info.update({WptInfoKey.WPT_TYPE: WptType.PERTURB, WptInfoKey.WPT_ID: wpt_id, WptInfoKey.PERTURB_IDX: idx})
                yield perturb.mistake
                if perturb.correction is not None:
                    if verbose:
                        print(f"\tmake {GREEN}intermediate{RESET} correction at keypoint {wpt_id} for perturb {idx}")
                    perturb.correction.info.update({WptInfoKey.WPT_TYPE: WptType.INTERMEDIATE, WptInfoKey.WPT_ID: wpt_id, WptInfoKey.PERTURB_IDX: idx})
                    yield perturb.correction
                else:
                    if verbose:
                        print(f"\tno {GREEN}intermediate{RESET} correction at keypoint {wpt_id} for perturb {idx}")
            if verbose:
                print(f"\treturn to {BLUE}expert{RESET} action at keypoint {wpt_id}")
            self.waypoints[wpt_id].info.update({WptInfoKey.WPT_TYPE: WptType.EXPERT, WptInfoKey.WPT_ID: wpt_id})
            yield self.waypoints[wpt_id]

    def update_info(self) -> Iterator[WayPoint]:
        for wpt_id in self.all_keypoints_idx():
            self.waypoints[wpt_id].info.update({WptInfoKey.WPT_TYPE: WptType.EXPERT, WptInfoKey.WPT_ID: wpt_id})
            for i, perturb_wpt in enumerate(self.waypoints[wpt_id].perturbations):
                perturb_wpt.mistake.info.update({WptInfoKey.WPT_TYPE: WptType.PERTURB, WptInfoKey.WPT_ID: wpt_id, WptInfoKey.PERTURB_IDX: i})
                if perturb_wpt.correction is not None:
                    perturb_wpt.correction.info.update({WptInfoKey.WPT_TYPE: WptType.INTERMEDIATE, WptInfoKey.WPT_ID: wpt_id, WptInfoKey.PERTURB_IDX: i})
        self.post_processed = True

    def iterate_all_keypoints(self, num_perturb=1, verbose=False, perturb_idx=None) -> Iterator[WayPoint]:
        for step in list(self.retrieve_keypoints()):
            for keypoint in self.iterate_one_keypoint(step, num_perturb, verbose, perturb_idx=perturb_idx):
                yield keypoint

    # Note: does not print perturb.correction that is None
    def summarize_episode(self, num_perturb=1):
        print(f"Task: {self.task_name}")
        print(f"Goal: {self.lang_goal}")
        for keywpt in self.iterate_all_keypoints(num_perturb):
            keywpt_idx = keywpt.info[WptInfoKey.WPT_ID]
            if keywpt.info[WptInfoKey.WPT_TYPE] == WptType.EXPERT:
                print(f"idx {keywpt_idx} expert step {keywpt.info[WptInfoKey.WPT_ID]}: keywpt", keywpt.action.to_interaction(), " | ", keywpt.verbose)
                lang_str = " ".join(keywpt.info.get('lang', []))
                print(f"\t{keywpt.info['transition_type']} | {lang_str}")
            elif keywpt.info[WptInfoKey.WPT_TYPE] == WptType.PERTURB:
                print(f"\tperturb {keywpt.info[WptInfoKey.PERTURB_IDX]} {keywpt.info[WptInfoKey.PERTURB_TYPE]}:", keywpt.action.to_interaction())
                lang_str = " ".join(keywpt.info.get('lang', []))
                print(f"\t\t{keywpt.info['transition_type']} | {lang_str}")
            elif keywpt.info[WptInfoKey.WPT_TYPE] == WptType.INTERMEDIATE:
                print(f"\tcorrect {keywpt.info[WptInfoKey.PERTURB_IDX]}:", keywpt.action.to_interaction())
                lang_str = " ".join(keywpt.info.get('lang', []))
                print(f"\t\t{keywpt.info['transition_type']} | {lang_str}")
            else:
                print(f"Unknown type {keywpt.info[WptInfoKey.WPT_TYPE]}")
            # print(f"\t {keypoint.info['scene_info']}")
        print(f"Success: {self.success}")

    # merge an additional episode to the current episode (both of same expert demo episode)
    def merge_episodes(self, additional_episode: 'Episode'):
        assert self.task_name == additional_episode.task_name, "Task name should be the same"
        assert self.episode_num == additional_episode.episode_num, "Episode number should be the same"
        assert self.post_processed is True, "The episode should be post-processed"
        assert additional_episode.post_processed is True, "The additional episode should be post-processed"
        add_episode = copy.deepcopy(additional_episode)
        for wpt, add_wpt in zip(self.waypoints, add_episode.waypoints):
            new_perturb_idx = len(wpt.perturbations) # important to update the perturb_idx
            for add_perturb in add_wpt.perturbations:
                add_perturb.mistake.info.update({WptInfoKey.PERTURB_IDX: new_perturb_idx})
                if add_perturb.correction is not None:
                    add_perturb.correction.info.update({WptInfoKey.PERTURB_IDX: new_perturb_idx})
                new_perturb_idx += 1
            wpt.perturbations.extend(add_wpt.perturbations)

    @classmethod
    def from_demos(cls, task_name:str, demo: Demo, ep_num:int, lang_goal:str) -> "Episode":
        if task_name == "put_groceries_in_cupboard":
            episode_keypoints = [0] + keypoint_discovery_v2(demo)
        # elif task_name == "place_cups":
        #     episode_keypoints = keypoint_discovery(demo, method='dense')
        else:
            episode_keypoints = [0] + keypoint_discovery(demo)
            if task_name == "put_item_in_drawer":
                episode_keypoints = sorted([idx for idx in episode_keypoints if idx != 1])

        wpts = []
        for idx, d in enumerate(demo):
            if idx in episode_keypoints or idx == 0:
                type = 'keypoint'
            else:
                type = 'intermediate'

            if task_name == "put_groceries_in_cupboard":
                action = Action(
                    translation=d.gripper_pose[:3],
                    rotation=Action.array_to_quat(d.gripper_pose[3:], style='xyzw'),
                    gripper_open=is_gripper_open(d.gripper_joint_positions),
                    ignore_collision=False
                )
                if idx == len(demo) - 1:
                    action.gripper_open = False
            else:
                action = Action(
                    translation=d.gripper_pose[:3],
                    rotation=Action.array_to_quat(d.gripper_pose[3:], style='xyzw'),
                    gripper_open=bool(d.gripper_open), # is_gripper_open(d.gripper_joint_positions)
                    ignore_collision=bool(d.ignore_collisions)
                )
            
            wp = WayPoint(idx, type, action)
            wpts.append(wp)
        ep = Episode(
            waypoints=wpts, 
            task_name=task_name,
            episode_num=ep_num,
            lang_goal=lang_goal)

        # if task_name == "place_cups":
        #     from racer_datagen.data_aug.heuristic_augmentor import Heuristic
        #     augmentor = Heuristic(task_name, cfg_path=f"{BASE_PATH}/data_aug/configs/{task_name}.yaml")
        #     ep_copy = augmentor.heuristic_perturb(copy.deepcopy(ep), N=0)
        #     kwpt_list = list(ep_copy.retrieve_keypoints())
        #     n_sample = 4
        #     every = 6
        #     for idx, step in enumerate(kwpt_list):
        #         curr_kwpt = ep_copy.waypoints[step]
        #         if idx <= len(kwpt_list) - 3:
        #             next_kwpt = ep_copy.waypoints[kwpt_list[idx+1]]
        #             next_next_kwpt = ep_copy.waypoints[kwpt_list[idx+2]]
        #             if next_kwpt.verbose.get("is", None) == "alignment step" and next_next_kwpt.verbose.get("is", None) == "general place":
        #                 start_dense = max((kwpt_list[idx] + (kwpt_list[idx+1] - kwpt_list[idx]) // 3, kwpt_list[idx+1] - n_sample * every))
        #                 print(start_dense, kwpt_list[idx+1]-1, len(ep_copy))
        #                 dense_idx = np.linspace(start_dense, kwpt_list[idx+1]-1, n_sample, dtype=int).tolist()
        #                 ep.waypoints[step].dense = ep_copy[dense_idx]
        #             if curr_kwpt.verbose.get("is", None) is None and next_kwpt.verbose.get("is", None) == "general place":
        #                 dense_idx = np.linspace(step+1 + (kwpt_list[idx+1] - step) // 2, kwpt_list[idx+1]-1, 5, dtype=int).tolist()
        #                 ep.waypoints[step].dense = ep_copy[dense_idx]
        #         print(f"idx {idx} expert step {step}:", curr_kwpt.action.to_interaction(), " | ", curr_kwpt.verbose)
        #         for i, perturb_wpt in enumerate(curr_kwpt.perturbations):
        #             print(f"\tperturb {i}:", perturb_wpt.mistake.action.to_interaction())
        #             if perturb_wpt.correction is not None:
        #                 print(f"\tcorrect {i}:", perturb_wpt.correction.action.to_interaction())
        #             else:
        #                 print(f"\tcorrect {i}:", None)
            # ep = ep_copy
        ep.update_info()
        return ep
    
def clean_json_string(json_string):
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip()
    
# save gif for perturbed episode or expert episode
def save_annotated_gif(episode_folder, annotated_episode: Episode = None, perturb_num: int = 0, rollout_type: str = None, prompt_gpt: bool = False):
    print("===== Saving videos =====")
    image_dir = os.path.join(episode_folder, "front_rgb")
    print(image_dir)
    
    if perturb_num != 0 and perturb_num is not None:
        for perturb_idx in range(perturb_num):
            rollout_type = None
            out_dir = os.path.join(episode_folder, "output", str(perturb_idx))
            os.makedirs(out_dir, exist_ok=True)
            tmp_dir = os.path.join(out_dir, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            gif_dir = os.path.join(out_dir, "gif")
            os.makedirs(gif_dir, exist_ok=True)

            filenames = [file_name for file_name in os.listdir(image_dir)]

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
                if split_name[1] == 'expert' or split_name[-1] == str(perturb_idx):
                    filtered_files.append(f)

            if annotated_episode.task_name == "place_cups":
                filtered_files_dense = []
                for f in sorted_files:
                    split_name = os.path.splitext(f)[0].split('_')
                    if split_name[1] == "dense":
                        filtered_files_dense.append(f)

            # first iterate through to find when the robot grasps an object
            last_known_pose = {}
            step_target_starts_moving = {}

            for step in annotated_episode.all_keypoints_idx():
                for wpt in annotated_episode.iterate_one_keypoint(step, perturb_idx=perturb_idx):
                    scene_info = wpt.info.get(WptInfoKey.SCENE_INFO, {})
                    scene_text = []                        
                    for key, value in scene_info.items():
                        if "visual" in key:
                            continue
                        else:
                            object_name = key.split('_')[0]
                            if object_name in last_known_pose:
                                if np.linalg.norm(np.array(last_known_pose[object_name]) - np.array(value['pose'][0:3])) > 0.1:
                                    step_target_starts_moving["step"] = step
                                    step_target_starts_moving["object"] = object_name
                            last_known_pose[key] = value['pose'][0:3]



            tmp_img_paths = []
            raw_text_for_prompting = ""
            prepare_prompt = {}
            
            j = 0
            k = 0
            last_known_pose = {}
            update_scene_info = False
            key_mapping = {}
            place_cups_dict = {}

            for step in annotated_episode.all_keypoints_idx():
                for wpt in annotated_episode.iterate_one_keypoint(step, perturb_idx=perturb_idx):                    
                    img_path = filtered_files[j]
                    # print(img_path, wpt.info[WptInfoKey.WPT_TYPE], wpt.info[WptInfoKey.WPT_ID])
                    image = Image.open(os.path.join(image_dir, img_path))
                    parts = img_path.split('_')
                    keypoint_number = int(parts[0])
                    action_type = img_path.split('_')[1].split('.')[0]
                    action = wpt.action.to_numpy()
                    gripper = "gripper close" if action[7] == 0 else "gripper open"
                    collision = "collision ignore" if action[8] == 1 else "collision consider"

                    prepare_prompt[f"{j}"] = {}
                    # prepare_prompt[f"{j}"]["wpt_id"] = f"{wpt.id}"

                    
                    key_mapping[f"{j}"] = img_path

                    if annotated_episode is not None:
                        # ROBOT STATE INFO
                        idx = wpt.info.get(WptInfoKey.PERTURB_IDX, "")
                        text = [
                            f"{annotated_episode.lang_goal}", f"idx {j}, kypt: {keypoint_number} | {action_type}" + f" {rollout_type} {idx}", "",
                            f"{action[0:3]}", f"{action[3:7]}", f"{gripper} & {collision}", ""
                        ]


                        if update_scene_info:
                            prepare_prompt[f"{j}"]["updated_scene_info"] = {}
                            prepare_prompt[f"{j}"]["updated_scene_info"]["objects"] = moved_object
                            update_scene_info = False
                        
                        # print(wpt.info.get(WptInfoKey.WPT_TYPE), wpt.info.get(WptInfoKey.VERBOSE))

                        if wpt.info.get(WptInfoKey.VERBOSE, None) is not None:
                            if wpt.info.get(WptInfoKey.VERBOSE).get("is", None) == "alignment step":
                                if prepare_prompt[f"{j-1}"]["current_timestep"]["status"] != "success":
                                    if prepare_prompt[f"{j-1}"]["current_timestep"].get("failure_reasoning", None) is None:
                                        prepare_prompt[f"{j-1}"]["current_timestep"]["failure_reasoning"] = "failed to align"
                                    else:
                                        if prepare_prompt[f"{j-1}"]["current_timestep"].get("failure_reasoning", None) is not None:
                                            prepare_prompt[f"{j-1}"]["current_timestep"]["failure_reasoning"] = "failed to align, " + prepare_prompt[f"{j-1}"]["current_timestep"]["failure_reasoning"]
                                else:
                                    # print(prepare_prompt[f"{j-1}"]["current_timestep"])
                                    # print(prepare_prompt[f"{j-2}"]["current_timestep"])
                                    if prepare_prompt[f"{j-2}"]["current_timestep"].get("failure_reasoning", None) is not None:
                                        prepare_prompt[f"{j-2}"]["current_timestep"]["failure_reasoning"] = "failed to align, " + prepare_prompt[f"{j-2}"]["current_timestep"]["failure_reasoning"]
                            if wpt.info.get(WptInfoKey.VERBOSE).get("is", None) == "general grasp":
                                if prepare_prompt[f"{j-1}"]["current_timestep"]["status"] != "success" and prepare_prompt[f"{j-1}"]["current_timestep"].get("failure_reasoning", None) is not None:
                                    prepare_prompt[f"{j-1}"]["current_timestep"]["failure_reasoning"] = "failed to grasp, " + prepare_prompt[f"{j-1}"]["current_timestep"]["failure_reasoning"]
                                else:
                                    if j > 2:
                                        if prepare_prompt[f"{j-2}"]["current_timestep"].get("failure_reasoning", None) is not None:
                                            prepare_prompt[f"{j-2}"]["current_timestep"]["failure_reasoning"] = "failed to grasp, " + prepare_prompt[f"{j-2}"]["current_timestep"]["failure_reasoning"]
                            if wpt.info.get(WptInfoKey.VERBOSE).get("is", None) == "general place":
                                if prepare_prompt[f"{j-1}"]["current_timestep"]["status"] != "success" and prepare_prompt[f"{j-1}"]["current_timestep"].get("failure_reasoning", None) is not None:
                                    prepare_prompt[f"{j-1}"]["current_timestep"]["failure_reasoning"] = "failed to place, " + prepare_prompt[f"{j-1}"]["current_timestep"]["failure_reasoning"]
                                else:
                                    if j > 2:
                                        if prepare_prompt[f"{j-2}"]["current_timestep"].get("failure_reasoning", None) is not None:
                                            prepare_prompt[f"{j-2}"]["current_timestep"]["failure_reasoning"] = "failed to place, " + prepare_prompt[f"{j-2}"]["current_timestep"]["failure_reasoning"]

                        # SCENE INFO
                        scene_info = wpt.info.get(WptInfoKey.SCENE_INFO, {})
                        scene_text = []
                        moved_object = []
                        moved_object_pos = None
                        object_true_color = {}
                        if step == 0:
                            for key, value in scene_info.items():
                                if "visual" in key:
                                    object_name = key.split('_')[0]
                                    true_color = value.get('color_name', None)
                                    object_true_color[object_name] = true_color
                                else:
                                    continue
                            for key, value in scene_info.items():
                                if "visual" in key:
                                    continue
                                else:
                                    if key in object_true_color:
                                        scene_text.append(f"{object_true_color[key]} {key}: {value['pose'][0:3]}")
                                    else:
                                        scene_text.append(f"{value.get('color_name', None)} {key}: {value['pose'][0:3]}")
                                    last_known_pose[key] = value['pose'][0:3]
                                    print(f"initial pose: {key} {value['pose'][0:3]}")
                        else:
                            for key, value in scene_info.items():
                                if "visual" in key:
                                    continue
                                else:
                                    if last_known_pose.get(key, None) is not None:
                                        object_name = key.split('_')[0]
                                        if np.linalg.norm(np.array(last_known_pose[key]) - np.array(value['pose'][0:3])) > 0.01:
                                            last_known_pose[key] = value['pose'][0:3]
                                            if object_name in object_true_color:
                                                scene_text.append(f"{object_true_color[object_name]} {key}: {value['pose'][0:3]}")
                                            else:
                                                scene_text.append(f"{value['color_name']} {key}: {value['pose'][0:3]}")
                                            moved_object.append(f"{key}: {str(np.round(value['pose'][0:3], 3).tolist())}")
                                            if annotated_episode.task_name == "slide_block_to_color_target":
                                                moved_object_pos = np.round(value['pose'][0:3], 3).tolist()
                        if scene_text != []:     
                            text += ["--scene info--"]
                            text += scene_text
                            text += [""]

                        if annotated_episode.task_name == "slide_block_to_color_target":
                            for key, value in scene_info.items():
                                print(key, value)
                                object_name = key.split('_')[0]
                                if object_name == "block":
                                    block_pos = value['pose'][0:3]
                                    if moved_object:
                                        print(moved_object)
                                        block_pos = moved_object_pos
                            print("HERE WE GOO", block_pos, moved_object)
                        
                        # (GPT) write the initial scene info
                        if j == 0:
                            # only at the first step
                            prepare_prompt[f"{j}"]["initial_scene_info"] = {}
                            prepare_prompt[f"{j}"]["initial_scene_info"]["objects"] = {
                                key: str(np.round(value['pose'][0:3], 3).tolist()) for key, value in scene_info.items() if "visual" not in key
                            }
                            
                        # ANNOTATED LANGUAGE
                        if j < len(filtered_files) - 1:
                            # these texts are instructions, which means that the next image is the result of this current instruction (as if a command was given)
                            if wpt.info.get(WptInfoKey.FAILURE_REASON, None) is not None:
                                text += [""]
                                text += ["--failure reasoning and reflection--"]
                                text += wpt.info.get(WptInfoKey.FAILURE_REASON, [])
                            text += [""]
                            text += ["--instructions--"]
                            text += wpt.info.get(WptInfoKey.LANG, [])
                            if WptInfoKey.VERBOSE in wpt.info:
                                text += [value for value in [wpt.info[WptInfoKey.VERBOSE].get(key) for key in ['is', 'reason']] if value is not None]
                        else:
                            text += wpt.info.get(WptInfoKey.LANG, [])
                            text += ["goal reached!"]
                        text += ["========================"]
                    else:
                        text = [f"idx {j}, kypt: {keypoint_number} | {action_type}", f"{action[0:3]}", f"{action[3:7]}", f"{gripper} & {collision}"]
                    
                    # (GPT) write the current timestep's robot pose
                    prepare_prompt[f"{j}"]["current_timestep"] = {}
                    if j == 0:
                        prepare_prompt[f"{j}"]["current_timestep"]["status"] = "init"
                        prepare_prompt[f"{j}"]["current_timestep"]["instruction_in_spatial_language"] = wpt.info.get(WptInfoKey.LANG, [])
                    else:

                        if wpt.info.get(WptInfoKey.FAILURE_REASON, None) is not None:
                            prepare_prompt[f"{j}"]["current_timestep"]["status"] = "recoverable failure"
                            prepare_prompt[f"{j}"]["current_timestep"]["failure_reasoning"] = ", ".join(wpt.info.get(WptInfoKey.FAILURE_REASON, []))

                        else:
                            prepare_prompt[f"{j}"]["current_timestep"]["status"] = "success"
                            # prepare_prompt[f"{j}"]["current_timestep"]["failure_reasoning"] = None
                    if wpt.info.get(WptInfoKey.CURRENT_POSE, None) is not None:
                        prepare_prompt[f"{j}"]["current_timestep"]["robot_pose"] = {
                            "position": str(wpt.info.get(WptInfoKey.CURRENT_POSE).get("pos")),
                            "orientation": str(wpt.info.get(WptInfoKey.CURRENT_POSE).get("ori")),
                            "gripper_open": wpt.info.get(WptInfoKey.CURRENT_POSE).get("gripper_open"),
                            "ignore_collision": wpt.info.get(WptInfoKey.CURRENT_POSE).get("ignore_collision")
                        }

                    if j != len(filtered_files) - 1:
                        prepare_prompt[f"{j}"]["current_timestep"]["instruction_in_spatial_language"] = wpt.info.get(WptInfoKey.LANG, [])
                        prepare_prompt[f"{j}"]["desired_state_at_next_timestep"] = {}
                        if wpt.info.get(WptInfoKey.NEXT_POSE, None) is not None:
                            # # if task is slide_block_to_color_target
                            if annotated_episode.task_name == "slide_block_to_color_target":
                                def return_side_of_block(gripper_pos, block_pos):
                                    delta_pos = np.array(gripper_pos) - np.array(block_pos)
                                    # get max arg
                                    if np.argmax(np.abs(delta_pos)) == 0:
                                        if delta_pos[0] > 0:
                                            side = "move the gripper to the back of the block"
                                        else:
                                            side = "move the gripper to the front of the block"
                                    elif np.argmax(np.abs(delta_pos)) == 1:
                                        if delta_pos[1] < 0:
                                            side = "move the gripper to the left of the block"
                                        else:
                                            side = "move the gripper to the right of the block"
                                    else:
                                        side = ""
                                    print(side, delta_pos)
                                    return side
                                print("HERE WEEEEEEE GOOOOOOOOOOOOO")
                                # if prepare_prompt[f"{j}"]["current_timestep"]["status"] != "recoverable failure" or j == 0:
                                if j == 0 or j == 4:
                                    prepare_prompt[f"{j}"]["desired_state_at_next_timestep"]["robot_pos_relative_to_block"] = {
                                        "relative_gripper_pos_to_block": return_side_of_block(wpt.info.get(WptInfoKey.NEXT_POSE).get("pos"), block_pos)
                                    }
                                
                            prepare_prompt[f"{j}"]["desired_state_at_next_timestep"]["robot_pose"] = {
                                "position": str(wpt.info.get(WptInfoKey.NEXT_POSE).get("pos")),
                                "orientation": str(wpt.info.get(WptInfoKey.NEXT_POSE).get("ori")),
                                "gripper_open": wpt.info.get(WptInfoKey.NEXT_POSE).get("gripper_open"),
                                "ignore_collision": wpt.info.get(WptInfoKey.NEXT_POSE).get("ignore_collision")
                            }
                        if moved_object:
                            update_scene_info = True
                            prepare_prompt[f"{j}"]["desired_state_at_next_timestep"]["moved_object"] = moved_object
                    else:
                        prepare_prompt[f"{j}"]["current_timestep"]["status"] = "end of episode"
                        prepare_prompt[f"{j}"]["current_timestep"]["instruction_in_spatial_language"] = ["goal reached"]

                    for t in text:
                        raw_text_for_prompting += f"{t}\n"
                    raw_text_for_prompting += "\n"

                    annotated_image = append_text_underneath_image(np.array(image), text)
                    tmp_img_path = os.path.join(tmp_dir, os.path.basename(img_path))
                    Image.fromarray(annotated_image).save(tmp_img_path)
                    tmp_img_paths.append(tmp_img_path)
                    j += 1

                    if annotated_episode.task_name == "place_cups":
                        if len(wpt.dense) > 0:
                            # print(len(wpt.dense))
                            # print(filtered_files_dense, len(filtered_files_dense))
                            for idx, dense_wpt in enumerate(wpt.dense):
                                img_path = filtered_files_dense[k]
                                # print(img_path, k, len(wpt.dense), dense_wpt)
                                image = Image.open(os.path.join(image_dir, img_path))
                                parts = img_path.split('_')
                                keypoint_number = int(parts[0])
                                action_type = img_path.split('_')[1].split('.')[0]
                                action = dense_wpt.action.to_numpy()
                                gripper = "gripper close" if action[7] == 0 else "gripper open"
                                collision = "collision ignore" if action[8] == 1 else "collision consider"
                                text = [f"idx {keypoint_number} | {action_type}", f"{action[0:3]}", f"{action[3:7]}", f"{gripper} & {collision}"]
                                annotated_image = append_text_underneath_image(np.array(image), text)
                                tmp_img_path = os.path.join(tmp_dir, os.path.basename(img_path))
                                Image.fromarray(annotated_image).save(tmp_img_path)
                                tmp_img_paths.append(tmp_img_path)

                                # get image filename without extension
                                img_name = os.path.splitext(img_path)[0]
                                place_cups_dict[img_name] = {
                                    "idx": keypoint_number,
                                    "label": "ongoing",
                                    "heuristic-lang": "Continue.",
                                    "gpt-lang": "Continue."
                                }
                                print(img_name)
                                k += 1

            if annotated_episode.task_name == "place_cups":
                with open(os.path.join(out_dir, "place_cups_dense.json"), "w") as f:
                    json.dump(place_cups_dict, f, indent=4)
            
            if rollout_type ==  PerturbType.HEURISTIC or rollout_type == PerturbType.RVT:
                output_filename = os.path.join(gif_dir, f"annotated_{rollout_type}_{perturb_idx}.gif")
            if rollout_type == WptType.EXPERT:
                output_filename = os.path.join(gif_dir, f"annotated_{rollout_type}.gif")
            imageio.mimsave(os.path.join(gif_dir, output_filename), [imageio.imread(image_path) for image_path in tmp_img_paths], duration=1000)

            with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
                f.write(raw_text_for_prompting)

            with open(os.path.join(out_dir, "prompt.json"), "w") as f:
                f.write(json.dumps(prepare_prompt, indent=4))
            
            with open(os.path.join(out_dir, "key_mapping.json"), "w") as f:
                json.dump(key_mapping, f, indent=4)


            config = AzureConfig()
            chat_api = ChatAPI(config)
            with open(os.path.join(out_dir, "gpt_prompt.txt"), "w") as f:
                system_text = build_prompts(
                    notes=NOTES_SHORT, # NOTES_DETAILED
                    example="",
                    robot_setup=ROBOT_SETUP,
                    env_setup=ENV_SETUP,
                )
                user_text = build_input_json(
                    task_name=annotated_episode.task_name, 
                    lang_goal=annotated_episode.lang_goal, 
                    task_description=globals()[annotated_episode.task_name.upper()], 
                    input_json=json.dumps(prepare_prompt, indent=4)
                )
                print(os.path.join(out_dir, "gpt_prompt.txt"))
                f.write(system_text + user_text)
            chat_api.message = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
            # print(chat_api.message)
            print(f"PROMPT_GPT: {prompt_gpt}")
            if prompt_gpt:
                attempts = 0
                max_attempts = 3
                success = False
                print("Prompting GPT-4 for the response...")

                while attempts < max_attempts and not success:
                    try:
                        response = chat_api.get_system_response()
                        print("Assistant:", response)
                        chat_api.add_assistant_message(response)

                        with open(os.path.join(out_dir, "gpt_output.json"), "w") as f:
                            json.dump(json.loads(clean_json_string(response)), f, indent=4)
                            print(os.path.join(out_dir, "gpt_output.json"))
                        
                        with open(os.path.join(out_dir, "gpt_response.json"), "w") as f:
                            f.write(json.dumps(chat_api.message, indent=4))
                            print(os.path.join(out_dir, "gpt_response.json"))
                            
                        success = True
                    except Exception as e:
                        attempts += 1
                        print(f"Attempt {attempts} failed: {e}")
                        time.sleep(10)

                if not success:
                    print("All attempts to get the response failed. Please check the internet connection and fix manually later.")

            shutil.rmtree(tmp_dir)

        for f in sorted(os.listdir(gif_dir)):
            print(os.path.join(gif_dir, f))
        
        return os.path.join(gif_dir, output_filename)
    else:
        filenames = [file_name for file_name in os.listdir(image_dir)]
        if rollout_type != "rvt":
            sorted_files = sorted(filenames, key=partial(sort_key, type=WptType.EXPERT))
        else:
            sorted_files = sorted(filenames, key=partial(sort_key, type=PerturbType.RVT))
        filtered_files = []
        tmp_dir = os.path.join(episode_folder, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        gif_dir = os.path.join(episode_folder, "gif")
        os.makedirs(gif_dir, exist_ok=True)
        for f in sorted_files:
            split_name = os.path.splitext(f)[0].split('_')
            if split_name[1] == PerturbType.RVT or split_name[1] == "perturb":
                filtered_files.append(f)
        print(filtered_files, sorted_files)
        tmp_img_paths = []
        for i, f in enumerate(filtered_files):
            image = Image.open(os.path.join(image_dir, f))
            parts = f.split('_')
            keypoint_number = int(parts[0])
            if rollout_type == PerturbType.RVT:
                keypoint_number = i
            if annotated_episode is not None:
                action_type = f.split('_')[1].split('.')[0]
                if isinstance(annotated_episode.waypoints[keypoint_number].action, np.ndarray):
                    action = annotated_episode.waypoints[keypoint_number].action
                else:
                    action = annotated_episode.waypoints[keypoint_number].action.to_numpy()
                gripper = "gripper close" if action[7] == 0 else "gripper open"
                collision = "collision ignore" if action[8] == 1 else "collision consider"
                text = [f"idx {keypoint_number} | {action_type}", f"{action[0:3]}", f"{action[3:7]}", f"{gripper} & {collision}"]
            else:
                text = ""
            annotated_image = append_text_underneath_image(np.array(image), text)
            tmp_img_path = os.path.join(tmp_dir, os.path.basename(f))
            Image.fromarray(annotated_image).save(tmp_img_path)
            tmp_img_paths.append(tmp_img_path)

        imageio.mimsave(os.path.join(gif_dir, "annotated_expert.gif"), [imageio.imread(image_path) for image_path in tmp_img_paths], duration=1000)
        print(os.path.join(gif_dir, "annotated_expert.gif"))
        return os.path.join(gif_dir, "annotated_expert.gif")

