import copy
import logging


import cv2
import textwrap

import numpy as np

from typing import TYPE_CHECKING
from racer_datagen.libs.peract.helpers.custom_rlbench_env import CustomRLBenchEnv
from racer_datagen.online_rollout.base.constants import *

from yarr.agents.agent import ActResult, VideoSummary, TextSummary
from yarr.utils.transition import Transition

from pyrep.errors import IKError, ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError
from yarr.utils.process_str import change_case
from racer_datagen.utils.rvt_utils import RLBENCH_TASKS
from rlbench.backend.utils import task_file_to_task_class



if TYPE_CHECKING:
    from racer_datagen.data_aug.base.action import Action

class CustomRLRenchEnv2(CustomRLBenchEnv):
    def __init__(self, *args, **kwargs):
        super(CustomRLRenchEnv2, self).__init__(*args, **kwargs)
        self._task_classes = [task_file_to_task_class(task) for task in RLBENCH_TASKS]
        self._task_name_to_idx = {change_case(tc.__name__):i for i, tc in enumerate(self._task_classes)}

    
    def reset(self) -> dict:
        super().reset()
        self._record_current_episode = False
        return self._previous_obs_dict
    
    def set_new_task(self, task_name: str):
        # Adapted from YARR/yarr/envs/rlbench_env.py MultiTaskRLBenchEnv class
        assert task_name in RLBENCH_TASKS, f"Task {task_name} not found in RLBENCH_TASKS"
        self._active_task_id = self._task_name_to_idx[task_name]
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)

        descriptions, _ = self._task.reset()
        self._lang_goal = descriptions[0] # first description variant
    
    def reset_to_demo(self, i, not_load_image=True):
        self._i = 0
        self._task.set_variation(-1)
        d = self._task.get_demos(
            1, live_demos=False, 
            image_paths=not_load_image,  # not load image
            random_selection=False, from_episode_number=i)[0]

        self._task.set_variation(d.variation_number)
        desc, obs = self._task.reset_to_demo(d)
        obs_copy = copy.deepcopy(obs)
        self._lang_goal = desc[0]

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = False
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict, obs_copy
    
    def step(self, act_result: ActResult) -> Transition:
        action = act_result.action
        success = False
        obs = self._previous_obs_dict  # in case action fails.
        error_status = "success"
        info = {}

        try:
            obs, reward, terminal = self._task.step(action)
            privileged_info = self._task.get_privileged_info()
            info.update({'scene_info': privileged_info})
            obs_copy = copy.deepcopy(obs)
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = 0.0
            obs_copy = None

            if isinstance(e, IKError):
                print("IKError")
                self._error_type_counts['IKError'] += 1
            elif isinstance(e, ConfigurationPathError):
                print("ConfigurationPathError")
                self._error_type_counts['ConfigurationPathError'] += 1
            elif isinstance(e, InvalidActionError):
                print("InvalidActionError")
                error_status = "error"
                self._error_type_counts['InvalidActionError'] += 1
            else:
                print("Unknown error")

            self._last_exception = e
        
        info.update({'error_status': error_status, 'obs': obs_copy})
        summaries = []
        self._i += 1
        if ((terminal or self._i == self._episode_length) and self._record_current_episode):
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            summaries.append(VideoSummary('episode_rollout_' + ('success' if success else 'fail'), vid, fps=30))

            # error summary
            error_str = f"Errors - IK : {self._error_type_counts['IKError']}, " \
                        f"ConfigPath : {self._error_type_counts['ConfigurationPathError']}, " \
                        f"InvalidAction : {self._error_type_counts['InvalidActionError']}"
            if not success and self._last_exception is not None:
                error_str += f"\n Last Exception: {self._last_exception}"
                self._last_exception = None

            summaries.append(TextSummary('errors', f"Success: {success} | " + error_str))
        return Transition(obs, reward, terminal, info=info, summaries=summaries)


def sort_key(filename, type='heuristic'):
    parts = filename.split('_')
    keypoint_number = int(parts[0])
    action_type = parts[1].split('.')[0]

    # Define a custom order for the action types
    if type == PerturbType.HEURISTIC:
        action_order = {WptType.PERTURB: 0, WptType.INTERMEDIATE: 1, WptType.DENSE: 2, WptType.EXPERT: 3}
        action_priority = action_order.get(action_type, 3)  # Default to a high value if the action type is unknown
    elif type == WptType.EXPERT:
        action_order = {WptType.EXPERT: 0}
        action_priority = action_order.get(action_type, 1)
    elif type == PerturbType.CMD:
        action_order = {PerturbType.CMD: 0}
        action_priority = action_order.get(action_type, 1)
    elif type == PerturbType.RVT:
        action_order = {WptType.PERTURB: 0, WptType.INTERMEDIATE: 1, WptType.DENSE: 2, WptType.EXPERT: 3}
        action_priority = action_order.get(action_type, 2)
    return (keypoint_number, action_priority)

def append_text_underneath_image(image: np.ndarray, texts: list[str], max_text_height=600, target_size=(512, 512), min_wrap_width=40):
    """Appends multiple texts underneath an image, each on its own line.

    Optionally upscales the image to the target size before appending text.
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines. Ensures a consistent image size by using a fixed text area height.

    :param image: The image to append text underneath.
    :param texts: A list of strings to display, each on a new line.
    :param max_text_height: Maximum height for the text area.
    :param target_size: The desired size (width, height) to which the image should be resized, if necessary.
    :return: A new image with texts appended underneath.
    """
    # Check if image needs to be resized
    if image.shape[:2] != target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    
    h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros((max_text_height, w, c), dtype=np.uint8)  # Use fixed height for text area

    y = 15  # Start a little lower to avoid clipping the text
    for text in texts:
        if text == "":
            # If the text is an empty string, add a blank line (increase y position)
            y += 15  # Adjust the vertical space for the empty line as needed
        else:
            # Ensure wrap width is not too small
            wrap_width = max(int(w / (cv2.getTextSize(" ", font, font_size, font_thickness)[0][0])), min_wrap_width)
            lines = textwrap.wrap(text, width=wrap_width)
            for line in lines:
                if y + 10 < max_text_height:  # Only add text if there's space
                    textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
                    x = 10
                    cv2.putText(
                        blank_image,
                        line,
                        (x, y),
                        font,
                        font_size,
                        (255, 255, 255),
                        font_thickness,
                        lineType=cv2.LINE_AA,
                    )
                    y += textsize[1] + 10

    final_image = np.concatenate((image, blank_image), axis=0)
    return final_image

def target_object_in_scene(task_name: str, scene_info: dict) -> str:
    # TODO: find the object to track in language from scene info dict
    # possibly could do this per task scene
    pass

def annotate_transition_heuristic(curr_action: 'Action', next_action: 'Action', failure_reasoning: bool = False, expert_action: 'Action' = None, scene_info: dict = None):
    if failure_reasoning:
        assert(expert_action is not None)
    heuristic_language = []
    # curr_action is the current state (after stepping through the sim)
    if not failure_reasoning:
        delta_action = curr_action.delta_action(curr_action, next_action)
        # Iterate over each direction, axis, and sign
        for direction, axis, sign in DIRECTIONS:
            translation_component = delta_action['translation'][axis]
            if sign * translation_component > TRANSLATION_SMALL_THRES:
                magnitude = "by large distance" if abs(translation_component) > TRANSLATION_LARGE_THRES else "a bit more"
                heuristic_language.append(f"move {direction} along {AXES[axis]}-axis {magnitude}") # {translation_component:.2f}

        # annotate rotation change
        if np.any(np.abs(delta_action['rotation']) > ROTATION_SMALL_THRES):
            if (np.abs(delta_action['rotation'][0]) > ROTATION_SMALL_THRES and 
                np.abs(delta_action['rotation'][1]) > ROTATION_SMALL_THRES and 
                np.abs(delta_action['rotation'][2]) > ROTATION_SMALL_THRES):
                heuristic_language.append("rotate gripper")
            elif np.abs(delta_action['rotation'][2]) > ROTATION_LARGE_THRES:
                heuristic_language.append("rotate gripper about z-axis")
            else:
                for axis, diff in zip(AXES, delta_action['rotation']):
                    if diff > ROTATION_SMALL_THRES:
                        heuristic_language.append(f"rotate gripper about {axis}-axis")

        # annotate gripper change
        if delta_action['gripper'] != 0:
            gripper_desc = "open gripper" if delta_action['gripper'] == 1 else "close gripper"
            heuristic_language.append(gripper_desc)
        else:
            if curr_action.gripper_open == 1:
                heuristic_language.append("keep gripper open")
            else:
                heuristic_language.append("keep gripper closed")

        # annotate collision change
        if delta_action['collision'] != 0:
            collision_desc = "allow collision" if delta_action['collision'] == 1 else "avoid collision"
            heuristic_language.append(collision_desc)
    # Computes expert_action - curr_action = delta_action
    # This is the actual error compared to the expert action
    # if delta_action > 0 then the agent moved too much past the expert action
    # else the agent moved too little compared to the expert
    else:
        delta_action = curr_action.delta_action(curr_action, next_action)
        action_error = curr_action.delta_action(next_action, expert_action)
        for direction, axis, sign in DIRECTIONS:
            # the failure desciption's language directionality is determined by delta action between current and next
            if np.sign(delta_action['translation'][axis]) != sign:
                continue
            if np.abs(action_error['translation'][axis]) < 0.01:
                continue
            # how much it deviates from the expert action along the predetermined direction is determined by action error between next and expert
            if curr_action.T[axis] < next_action.T[axis] < expert_action.T[axis] or curr_action.T[axis] > next_action.T[axis] > expert_action.T[axis]:
                howmuch = "too little"
            else:
                howmuch = "too much"
            heuristic_language.append(f"moved {direction} along {AXES[axis]}-axis {howmuch}") #err: {action_error['translation'][axis]:.2f}
        # heuristic_language.insert(0, "failed due to the following action:")

        # annotate rotation change
        if np.any(np.abs(action_error['rotation']) > ROTATION_SMALL_THRES):
            if (np.abs(action_error['rotation'][0]) > ROTATION_SMALL_THRES and 
                np.abs(action_error['rotation'][1]) > ROTATION_SMALL_THRES and 
                np.abs(action_error['rotation'][2]) > ROTATION_SMALL_THRES):
                heuristic_language.append(f"gripper orientation is misaligned")
            elif np.abs(action_error['rotation'][2]) > ROTATION_LARGE_THRES:
                heuristic_language.append(f"gripper orientation is misaligned about z-axis")
            else:
                for axis, diff in zip(AXES, np.abs(action_error['rotation'])):
                    if diff > ROTATION_LARGE_THRES:
                        heuristic_language.append(f"gripper orientation is misaligned about {axis}-axis")
    return heuristic_language

logger = logging.getLogger()

OBS_ATTRS_save_for_PKL = [
    "gripper_open", "gripper_pose", "gripper_joint_positions", "ignore_collisions", "misc"
]

OBS_ATTRS_not_for_PKL = [
    "left_shoulder_rgb", "left_shoulder_depth", "left_shoulder_point_cloud", "left_shoulder_mask",
    "right_shoulder_rgb", "right_shoulder_depth", "right_shoulder_point_cloud", "right_shoulder_mask",
    "overhead_rgb", "overhead_depth", "overhead_point_cloud", "overhead_mask",
    "wrist_rgb", "wrist_depth", "wrist_point_cloud", "wrist_mask",
    "front_rgb", "front_depth", "front_point_cloud", "front_mask",
    "joint_velocities", "joint_positions", "joint_forces", "gripper_matrix", "gripper_touch_forces", "task_low_dim_state"
]

AXES = ["x", "y", "z"]

TRANSLATION_SMALL_THRES = 0.01
TRANSLATION_LARGE_THRES = 0.05
ROTATION_SMALL_THRES = 5
ROTATION_LARGE_THRES = 20

# Directions: (name, axis, +ve sign = 1)
# backward = closer to VLM's view & forward = further away from VLM's perspective
DIRECTIONS = [("backward", 0, 1), ("forward", 0, -1), ("right", 1, 1), ("left", 1, -1), ("down", 2, -1), ("up", 2, 1)]