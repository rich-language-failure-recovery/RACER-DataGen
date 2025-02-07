import os
import csv
import cv2
import copy
import pickle
import shutil
from PIL import Image
from typing import List, Optional

import numpy as np
from numpy import ndarray

from peract_colab.rlbench.utils import get_stored_demo

from rlbench.backend.const import *
from rlbench.backend import utils as rlbench_utils
from rlbench.backend.observation import Observation
from racer_datagen.libs.peract.helpers.demo_loading_utils import keypoint_discovery, keypoint_discovery_v2

from yarr.utils.transition import Transition

from .simulator import RLBenchSim
from .utils import logger, OBS_ATTRS_not_for_PKL, append_text_underneath_image

from racer_datagen.online_rollout.base.constants import *
from racer_datagen.online_rollout.base.agent import RolloutAction

np.set_printoptions(precision=4, suppress=True)

class RolloutBase:
    def __init__(
        self,
        task_name: str,
        episode_num: int,
        dataset_root: str,          # testset dir
        save_dir: str,             # save path
        debug: bool = True,
    ):
        self.sim = RLBenchSim(
            task_name=task_name,
            dataset_root=dataset_root,
            episode_length=25,
            record_every_n=-1,
        )

        self.dataset_root = dataset_root
        self.task_name = task_name
        self.episode_num = episode_num        
        self.save_dir = save_dir        
        self.debug = debug
        self.expert_demo_low_dim_path = os.path.join(self.dataset_root, self.task_name, f"all_variations/episodes/episode{str(self.episode_num)}", "low_dim_obs.pkl")
        

    def env_reset(self, task_name:Optional[str]=None, episode_num:Optional[int]=None, is_record=True):
        if task_name is not None and task_name != self.task_name:
            self.sim.env.set_new_task(task_name)
            self.task_name = task_name
            self.expert_demo_low_dim_path = os.path.join(self.dataset_root, self.task_name, f"all_variations/episodes/episode{str(self.episode_num)}", "low_dim_obs.pkl")
        
        if episode_num is not None:
            self.episode_num = episode_num
            self.expert_demo_low_dim_path = os.path.join(self.dataset_root, self.task_name, f"all_variations/episodes/episode{str(self.episode_num)}", "low_dim_obs.pkl")

        obs_dict, _ = self.sim.reset(episode_num=self.episode_num)
        self.transition = Transition(obs_dict, 0.0, False, {})
     
        # reset the expert demo and lang as per new task and episode
        print(f"{self.dataset_root}/{self.task_name}/all_variations/episodes/{self.episode_num}")
        self.expert_demo = get_stored_demo(f"{self.dataset_root}/{self.task_name}/all_variations/episodes", self.episode_num, False)
        if self.task_name == "put_groceries_in_cupboard":
            self.keywpts = [0] + keypoint_discovery_v2(self.expert_demo)
            # self.keywpts = [num for num in self.keywpts if num > 65]
        # elif self.task_name == "place_cups":
        #     self.keywpts = keypoint_discovery(self.expert_demo, method="dense")
        else:
            self.keywpts = [0] + keypoint_discovery(self.expert_demo)
            if self.task_name == "put_item_in_drawer":
                if 1 in self.keywpts:
                    self.keywpts = sorted(list(set(self.keywpts) - set([1])))
        self.lang = self.sim.env._lang_goal
        print(f"language goal: {self.lang}") 

        # set up new episode save path as per new task and episode
        if is_record:
            self.ep_save_path = self.make_dir(os.path.join(self.save_dir, self.task_name, str(self.episode_num)))
            self.actions_path = os.path.join(self.ep_save_path, "actions.csv")
            self.debug_img_path = os.path.join(self.save_dir, "current.png")
            if self.debug:
                print(self.debug_img_path)
            self.record_debug(obs_dict["front_rgb"], np.array(START_ACTION))
        return obs_dict
    
    def step(self, rollout_action: RolloutAction, is_record=True) -> Transition:
        action = rollout_action.action
        if action is None:
            logger.warning("Action is None, skip this step")
            return
        
        assert action.shape == (9,), f"Action should be (9, ) array, but got {action.shape}"
        assert action[-1] in [0.0, 1.0], f"ignore_collision should be 0 or 1, but got {action[-1]}"
        assert action[-2] in [0.0, 1.0], f"gripper_close should be 0 or 1, but got {action[-2]}"

        action[:3] = self._make_xyz_within_task_bound(action[:3])
        action[3:7] = self._normalize_quaternion(action[3:7])
        self.prev_action = action
        transition = self.sim.step(action)
        self.transition = transition

        # For some reason from rlbench sim, the final step's gripper open is not updated. So manual fix.
        # print(transition.terminal, rollout_action.action[-2], transition.info['obs'].gripper_open)
        if transition.terminal and (self.task_name == "close_jar" or self.task_name == "put_groceries_in_cupboard") and rollout_action.info[WptInfoKey.WPT_TYPE] == WptType.EXPERT:
            if transition.info['obs'].gripper_open == 0.0:
                transition.info['obs'].gripper_open = 1.0
            # print(transition.info['obs'].gripper_open)

        if is_record:
            if 'perturb_idx' in rollout_action.info:
                # file naming format: {wpt_id}_{wpt_type}_{perturb_type}_{idx}
                filename = (
                    f"{rollout_action.info[WptInfoKey.WPT_ID]}_{rollout_action.info[WptInfoKey.WPT_TYPE]}_"
                    f"{rollout_action.info[WptInfoKey.PERTURB_TYPE]}_{rollout_action.info[WptInfoKey.PERTURB_IDX]}"
                )
            # elif rollout_action.info[WptInfoKey.WPT_TYPE] == "dense":
            #     filename = f"{rollout_action.info[WptInfoKey.WPT_ID]}_{rollout_action.info[WptInfoKey.WPT_TYPE]}_{rollout_action.info[WptInfoKey.DENSE_ID]}"
            else:
                filename = f"{rollout_action.info[WptInfoKey.WPT_ID]}_{rollout_action.info[WptInfoKey.WPT_TYPE]}"

            if rollout_action.info['wpt_type'] == 'expert':                
                with open(self.expert_demo_low_dim_path, 'rb') as file:
                    expert_demo_low_dim = pickle.load(file)
                    expert_demo_low_dim = expert_demo_low_dim._observations[rollout_action.info['wpt_id']]

                    # if self.task_name != "put_groceries_in_cupboard":
                    #     assert expert_demo_low_dim.ignore_collisions == rollout_action.action[-1], f"ignore_collision should be {expert_demo_low_dim.ignore_collisions}, but got {rollout_action.action[-1]}" 
                    # # compare two arrays but to some threshold
                    # assert np.allclose(expert_demo_low_dim.gripper_pose,  transition.info['obs'].gripper_pose, atol=5e-1), f"gripper_pose should be {expert_demo_low_dim.gripper_pose}, but got {transition.info['obs'].gripper_pose}" 

                    # assert expert_demo_low_dim.gripper_open == rollout_action.action[-2], f"gripper_open should be {expert_demo_low_dim.gripper_open}, but got {rollout_action.action[-2]}"
                    # assert np.allclose(expert_demo_low_dim.misc['front_camera_extrinsics'], transition.info['obs'].misc['front_camera_extrinsics'], atol=1e-3), f"camera_extrinsics should be {expert_demo_low_dim.misc['front_camera_extrinsics']}, but got {transition.info['obs'].misc['front_camera_extrinsics']}"
                    # assert np.allclose(expert_demo_low_dim.misc['front_camera_intrinsics'], transition.info['obs'].misc['front_camera_intrinsics'], atol=1e-3), f"camera_intrinsics should be {expert_demo_low_dim.misc['front_camera_intrinsics']}, but got {transition.info['obs'].misc['front_camera_intrinsics']}"
                    # # check left_shoulder_camera
                    # assert np.allclose(expert_demo_low_dim.misc['left_shoulder_camera_extrinsics'], transition.info['obs'].misc['left_shoulder_camera_extrinsics'], atol=1e-3), f"camera_extrinsics should be {expert_demo_low_dim.misc['left_shoulder_camera_extrinsics']}, but got {transition.info['obs'].misc['left_shoulder_camera_extrinsics']}"
                    # assert np.allclose(expert_demo_low_dim.misc['left_shoulder_camera_intrinsics'], transition.info['obs'].misc['left_shoulder_camera_intrinsics'], atol=1e-3), f"camera_intrinsics should be {expert_demo_low_dim.misc['left_shoulder_camera_intrinsics']}, but got {transition.info['obs'].misc['left_shoulder_camera_intrinsics']}"
                    # # check right_shoulder_camera
                    # assert np.allclose(expert_demo_low_dim.misc['right_shoulder_camera_extrinsics'], transition.info['obs'].misc['right_shoulder_camera_extrinsics'], atol=1e-3), f"camera_extrinsics should be {expert_demo_low_dim.misc['right_shoulder_camera_extrinsics']}, but got {transition.info['obs'].misc['right_shoulder_camera_extrinsics']}"
                    # assert np.allclose(expert_demo_low_dim.misc['right_shoulder_camera_intrinsics'], transition.info['obs'].misc['right_shoulder_camera_intrinsics'], atol=1e-3), f"camera_intrinsics should be {expert_demo_low_dim.misc['right_shoulder_camera_intrinsics']}, but got {transition.info['obs'].misc['right_shoulder_camera_intrinsics']}"
                    # assert np.allclose(expert_demo_low_dim.gripper_joint_positions, transition.info['obs'].gripper_joint_positions, atol=0.1), f"gripper_joint_positions should be {expert_demo_low_dim.gripper_joint_positions}, but got {transition.info['obs'].gripper_joint_positions}"
                    # # check wrist_camera
                    # # assert np.allclose(expert_demo_low_dim.misc['wrist_camera_extrinsics'], transition.info['obs'].misc['wrist_camera_extrinsics'], atol=1e-1), f"camera_extrinsics should be {expert_demo_low_dim.misc['wrist_camera_extrinsics']}, but got {transition.info['obs'].misc['wrist_camera_extrinsics']}"
                    # assert np.allclose(expert_demo_low_dim.misc['wrist_camera_intrinsics'], transition.info['obs'].misc['wrist_camera_intrinsics'], atol=1e-3), f"camera_intrinsics should be {expert_demo_low_dim.misc['wrist_camera_intrinsics']}, but got {transition.info['obs'].misc['wrist_camera_intrinsics']}"

            if transition.info['error_status'] == "error":
                print(f"Error status: {transition.info['error_status']}")
            else:
                transition.info['obs'].ignore_collisions = rollout_action.action[-1]
                print(filename)
                self.record_data(transition.info['obs'], filename, debug=False)
                transition.info.pop('obs', None) # otherwise the Observation object will be saved in the transition
                self.record_action(action)
                self.record_debug(transition.observation["front_rgb"], action)
        return transition
    
    def _normalize_quaternion(self, quaternion: ndarray):
        # check if the quaternion is valid, if not normalize it
        assert quaternion.shape == (4,), f"Quaternion should be (4, ) array, but got {quaternion.shape}"
        if not np.isclose(np.linalg.norm(quaternion), 1.0):
            logger.warning(f"Quaternion is not normalized, normalize it")
        quaternion = quaternion / np.linalg.norm(quaternion)
        return quaternion
    
    def _make_xyz_within_task_bound(self, xyz: ndarray):
        TASK_BOUND = [-0.075, -0.455, 0.752, 0.510, 0.455, 1.485]
        assert xyz.shape == (3,), f"XYZ should be (3, ) array, but got {xyz.shape}"
        if not (np.all(xyz >= TASK_BOUND[:3]) and np.all(xyz <= TASK_BOUND[3:])):
            logger.warning(f"XYZ is out of task bound, clip it to {TASK_BOUND}")
            xyz = np.clip(xyz, TASK_BOUND[:3], TASK_BOUND[3:])
        return xyz
    

    def record_debug(self, rgb: ndarray, action: ndarray):
        if not self.debug: return
        if rgb.shape[0] == 3: # (3, 512, 512) -> (512, 512, 3)
            rgb = rgb.transpose(1, 2, 0)
        rgb = append_text_underneath_image(rgb, self._action_str_list(action, return_raw_list=False))
        self.save_single_img(self.debug_img_path, rgb)
        input("Continue...") # place holder to stop the process
    
    def record_data(self, obs: Observation, keypoint_state: str, debug: bool = False):
        self.save_rgb_and_depth_img(obs, self.ep_save_path, keypoint_state, debug)
        self.save_low_dim(obs, self.ep_save_path, keypoint_state, debug)

    def record_action(self, action: ndarray):        
        with open(self.actions_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(np.round(action,4))

    def is_success(self):
        return self.sim.is_success()

    def close(self):
        self.sim.close()
    
    def _action_str_list(self, action: ndarray, return_raw_list=True):
        return_list = [f"T: {action[:3]}", f"R: {action[3:7]}"]
        if action[-2]:
            return_list.append("Gripper: open")
        else:
            return_list.append("Gripper: close")
        if action[-1]:
            return_list[-1] += " Collision: ignore"
        else:
            return_list[-1] += " Collision: consider"
        if return_raw_list:
            action_list = [f"{a:0.5f}" for a in action.tolist()]
            action_list_str = "raw:" + " ".join(action_list)
            return_list.append(action_list_str)
        return return_list

    @staticmethod
    def make_dir(dir_path):
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                if file not in PERTURB_EPISODE_PKL_LIST:
                    if os.path.isdir(os.path.join(dir_path, file)):
                        shutil.rmtree(os.path.join(dir_path, file))
                    else:
                        os.remove(os.path.join(dir_path, file))
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    
    @staticmethod
    def save_single_img(img_path:str, front_rgb: ndarray):
        front_rgb = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, front_rgb)


    @staticmethod
    def save_rgb_and_depth_img(obs: Optional[Observation], save_path: str, keypoint_state: str, debug: bool = False):
    
        if obs is None:
            return
        
        def make_dir(dir):
            if not os.path.exists(dir):
                os.makedirs(dir)
        if debug:
            front_rgb_path = os.path.join(save_path, FRONT_RGB_FOLDER)
            make_dir(front_rgb_path)
            front_rgb = Image.fromarray(obs.front_rgb)
            front_rgb.save(os.path.join(front_rgb_path, f"{keypoint_state}.png"))
            return
        else:
            # Save image data first, and then None the image data, and pickle
            for attr_rgb, attr_depth in [
                (LEFT_SHOULDER_RGB_FOLDER, LEFT_SHOULDER_DEPTH_FOLDER),  
                (RIGHT_SHOULDER_RGB_FOLDER, RIGHT_SHOULDER_DEPTH_FOLDER),
                (WRIST_RGB_FOLDER, WRIST_DEPTH_FOLDER),
                (FRONT_RGB_FOLDER, FRONT_DEPTH_FOLDER)            
            ]:
                folder_rgb = os.path.join(save_path, attr_rgb)
                make_dir(folder_rgb)
                rgb = Image.fromarray(getattr(obs, attr_rgb))
                rgb.save(os.path.join(folder_rgb, f"{keypoint_state}.png"))
                
                folder_depth = os.path.join(save_path, attr_depth)
                make_dir(folder_depth)
                depth = rlbench_utils.float_array_to_rgb_image(getattr(obs, attr_depth), scale_factor=DEPTH_SCALE)
                depth.save(os.path.join(folder_depth, f"{keypoint_state}.png"))


    @staticmethod
    def save_low_dim(obs: Optional[Observation], save_path: str, keypoint_state: str, debug: bool = False):
        if obs is None:
            return

        if not debug:
            low_dim_obs_path = os.path.join(save_path, "obs.pkl")

            if os.path.exists(low_dim_obs_path):
                with open(low_dim_obs_path, 'rb') as file:
                    low_dim_obs_dict = pickle.load(file)
            else:
                low_dim_obs_dict = {}

            obs_copy = copy.deepcopy(obs)

            for attr in OBS_ATTRS_not_for_PKL:
                setattr(obs_copy, attr, None)
                
            low_dim_obs_dict[keypoint_state] = obs_copy

            with open(os.path.join(save_path, "obs.pkl"), 'wb') as file:
                pickle.dump(low_dim_obs_dict, file)

