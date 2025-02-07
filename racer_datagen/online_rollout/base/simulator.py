"""
abstract for simulator and model agent input observations
"""

from numpy import ndarray

from rlbench.backend.const import *
from rlbench.backend.utils import task_file_to_task_class

from .utils import CustomRLRenchEnv2
from racer_datagen.libs.peract.helpers import utils
from racer_datagen.utils.peract_utils import (
    CAMERAS,
    IMAGE_SIZE,
)

from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from racer_datagen.utils.rlbench_planning import (
    EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning,
)
from yarr.agents.agent import ActResult
from yarr.utils.transition import Transition

class Simulator:
    
    def reset(self) -> dict:
        r"""resets the simulator and returns the initial observations.

        :return: initial observations from simulator.
        """
        raise NotImplementedError
    
    def step(self, action, *args, **kwargs) -> dict:
        r"""Perform an action in the simulator and return observations.

        :param action: action to be performed inside the simulator.
        :return: observations after taking action in simulator.
        """
        raise NotImplementedError
    
    def close(self) -> None:
        raise NotImplementedError

    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RLBenchSim(Simulator):
    def __init__(
        self,  
        task_name: str,
        dataset_root: str,
        episode_length: int=25,
        record_every_n: int = -1, # -1 means no recording
    ):
        self.task_name = task_name
        self.dataset_root = dataset_root
        self.episode_length = episode_length
        self.record_every_n = record_every_n

        self.setup_env()


        
    def reset(self, episode_num: int = 0, not_load_image: bool = True) -> dict:
        obs_dict, obs = self.env.reset_to_demo(episode_num, not_load_image)
        return obs_dict, obs
    
    def setup_env(self):
        camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
        obs_config = utils.create_obs_config(CAMERAS, camera_resolution, method_name="")

        gripper_mode = Discrete()
        arm_action_mode = EndEffectorPoseViaPlanning()
        action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)
        self.env = CustomRLRenchEnv2(
            task_class=task_file_to_task_class(self.task_name),
            observation_config=obs_config,
            action_mode=action_mode,
            dataset_root=self.dataset_root,
            episode_length=self.episode_length,
            headless=True,
            time_in_state=True,
            include_lang_goal_in_obs=True,
            record_every_n=self.record_every_n
        )
        self.env.eval = True
        self.env.launch()

    def step(self, action: ndarray) -> Transition:
        # action is (9, ) array, 3 for pose, 4 for quaternion, 1 for gripper, 1 for ignore_collision
        wrap_action = ActResult(action=action)
        transition = self.env.step(wrap_action) # get Transition(obs, reward, terminal, info, summaries)
        
        # TODO: HOW TO HANDLE INVALID ACTION ERROR??

        # print scene info
        if 'scene_info' in transition.info:
            scene_info = transition.info['scene_info']
            # print(f"\tscene info: {scene_info}")

        if transition.info['error_status'] == "error":
            print(f"Error: action was {action}")
            # this is currently handled in generate_data.py
        if isinstance(transition, tuple):
            transition = transition[0]    
        self.transition = transition
        return transition
    
    def is_success(self) -> bool:
        # always called when simulation ends
        score = self.transition.reward
        print(f"Score: {score}") # reward of last transition
        return True if score == 100.0 else False
        
    def close(self):
        self.env.shutdown()
    
    