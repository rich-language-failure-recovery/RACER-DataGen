from typing import List, Optional
from numpy import ndarray
import numpy as np

from .rollout import RolloutBase
from .agent import ExpertAgent, HeuristicAgent, ModelRVTAgent, CmdAgent, RolloutAction
from .utils import sort_key, annotate_transition_heuristic
from .constants import *
from racer_datagen.data_aug.base.episode import Episode, save_annotated_gif
from racer_datagen.data_aug.base.waypoint_and_perturbation import WayPoint



class RolloutAgent(RolloutBase):
    def __init__(
        self,
        task_name: str,
        episode_num: int,
        dataset_root: str,          # testset dir
        save_dir: str,              # save path
        heuristics_cfg_root: str,   # params for heuristic agent
        model_path: Optional[str],            # params for model agent
        device: int,                # params for model agent
        debug: bool = True,         # set this False is use multiple process
    ):
        super().__init__(task_name, episode_num, dataset_root, save_dir, debug)
        self.heuristics_cfg_root = heuristics_cfg_root
        self.env_reset(is_record=False)
        
        self.cmd_agent = CmdAgent()     
        if model_path is not None:
            self.model_agent = ModelRVTAgent(
                model_path=model_path, 
                device=device,
                debug_log_dir=save_dir,
            )

    def env_reset(self, is_record=True, task_name:Optional[str]=None, episode_num:Optional[int]=None):
        super().env_reset(task_name=task_name, episode_num=episode_num, is_record=is_record)
        self.expert_agent = ExpertAgent(self.task_name, self.episode_num, self.lang, self.expert_demo)
        self.heuristic_agent = HeuristicAgent(self.task_name, self.episode_num, self.lang, self.expert_demo, self.heuristics_cfg_root)   

    def _expert_rollout(self, start_kwpt: int, end_kwpt: int, is_reset=False, is_record=True):
        # rollout expert from start_step to end_step
        # start_step and end_step are inclusive, are index of densse frames
        assert start_kwpt <= end_kwpt, f"start_step {start_kwpt} should be less than end_step {end_kwpt}"
        assert start_kwpt in self.keywpts, f"start_step {start_kwpt} not found in keypoints"
        assert end_kwpt in self.keywpts, f"end_step {end_kwpt} not found in keypoints"
        print(f"start expert rollout from {start_kwpt} to {end_kwpt}")
        
        if is_reset: 
            self.env_reset(is_record)
        start_idx_in_keywpts = self.keywpts.index(start_kwpt)
        end_idx_in_keywpts = self.keywpts.index(end_kwpt)
        for i in range(start_idx_in_keywpts, end_idx_in_keywpts+1):
            for action in self.expert_agent.act_on_keywpt_id(self.keywpts[i]):
                # action_str = "\n\t".join(self._action_str_list(action.action))
                # print(f"keypoint {self.keywpts[i]}, action: \n\t{action_str}")
                self.step(action, is_record)
            

    def validate(self, valid_kwpt: int, perturb_action: ndarray, correct_action: List[ndarray],  is_record=False) -> bool:
        # only validate the perturbation in the valid_kwpt, other remains the same as expert demo
        assert valid_kwpt in self.keywpts, f"valid_kwpt {valid_kwpt} not found in keypoints"
        valid_idx_in_keywpts = self.keywpts.index(valid_kwpt)
        assert 0 < valid_idx_in_keywpts <= len(self.keywpts)-1, f"valid_kwpt {valid_kwpt} should be in the expert demo"
        
        # rollout the expert demo for first piece
        self._expert_rollout(0, self.keywpts[valid_idx_in_keywpts-1], True, is_record)
        
        # rollout the perturbation
        action_str = "\n\t".join(self._action_str_list(perturb_action))
        print(f"perturb for {valid_kwpt}")
        # print(f"perturb for {valid_kwpt}, action: \n\t{action_str}")
        perturb_action = RolloutAction(perturb_action, {WptInfoKey.WPT_TYPE: WptType.PERTURB, WptInfoKey.WPT_ID: valid_kwpt})
        self.step(perturb_action, is_record)
        if self.transition.info['error_status'] == "error": # this is RLBench Transition
            return False # move on to next perturbation idx
        for idx, action in enumerate(correct_action):
            action_str = "\n\t".join(self._action_str_list(action))
            print(f"correct for {valid_kwpt}")
            # print(f"correct for {valid_kwpt}, action: \n\t{action_str}")
            if idx == len(correct_action)-1:
                action_info = {WptInfoKey.WPT_TYPE: WptType.EXPERT, WptInfoKey.WPT_ID: valid_kwpt}
            else:
                action_info = {WptInfoKey.WPT_TYPE: WptType.INTERMEDIATE, WptInfoKey.WPT_ID: valid_kwpt}
            rollout_action = RolloutAction(action, action_info)
            self.step(rollout_action, is_record)
        
        # rollout the expert demo for the rest of the pieces
        self._expert_rollout(self.keywpts[valid_idx_in_keywpts], self.keywpts[-1], False, is_record)
        # determine if the perturbation is valid
        return self.sim.is_success()

    # rollout given any episode object (for single episode)
    def rollout_episode(self, episode: Episode, num_perturb: int, is_record: bool = True, lang_annotate: bool = True, prompt_gpt: bool = False) -> Episode:
        self.env_reset(is_record)
        episode.print_info()

        for i, keywpt_id in enumerate(self.keywpts):
            print("keypoint id:", keywpt_id)
            prev_keywpt_id = self.keywpts[i-1] if i > 0 else None
            next_keywpt_id = self.keywpts[i+1] if i < len(self.keywpts)-1 else None

            for keypoint in episode.iterate_one_keypoint(keywpt_id, num_perturb, verbose=True):
                action_str = "\n\t".join(self._action_str_list(keypoint.action.to_numpy()))
                print(f"\t{action_str}")
                transition = self.step(RolloutAction(keypoint.action.to_numpy(), keypoint.info), is_record)
                info = {**keypoint.info, **transition.info, **episode[keywpt_id].verbose} # agent_type, wpt_type, wpt_id, perturb_type, perturb_idx, verbose, error_status, scene_info

                if lang_annotate:
                    if keypoint.info[WptInfoKey.WPT_TYPE] == WptType.EXPERT:
                        # at expert state, agent is commanded to go to the next expert keypoint
                        if next_keywpt_id is not None:
                            next_action = episode[next_keywpt_id].action
                            lang = annotate_transition_heuristic(keypoint.action, next_action, scene_info=info[WptInfoKey.SCENE_INFO])
                            info.update({
                                WptInfoKey.TRANSITION_TYPE: TransitionType.SUCCESS, 
                                WptInfoKey.LANG: lang,
                                WptInfoKey.CURRENT_POSE: {
                                    "pos": np.round(keypoint.action.T, 3).tolist(),
                                    "ori": np.round(keypoint.action.quat_to_array(keypoint.action.R), 3).tolist(),
                                    "gripper_open": keypoint.action.gripper_open,
                                    "ignore_collision": keypoint.action.ignore_collision
                                },
                                WptInfoKey.NEXT_POSE: {
                                    "pos": np.round(next_action.T, 3).tolist(),
                                    "ori": np.round(next_action.quat_to_array(next_action.R), 3).tolist(),
                                    "gripper_open": next_action.gripper_open,
                                    "ignore_collision": next_action.ignore_collision
                                },
                                WptInfoKey.VERBOSE: keypoint.verbose
                            })
                            episode[keywpt_id].info.update(info)
                        else:
                            info.update({
                                WptInfoKey.TRANSITION_TYPE: TransitionType.SUCCESS, 
                                WptInfoKey.LANG: ["end of episode"],
                                WptInfoKey.VERBOSE: keypoint.verbose
                            })
                            episode[keywpt_id].info.update(info)
                            print("episode terminated")

                    elif keypoint.info[WptInfoKey.WPT_TYPE] == WptType.PERTURB:
                        # the agent is given the info on why its action was wrong
                        prev_action = episode[prev_keywpt_id].action
                        expert_action = episode[keywpt_id].action
                        failure_reasoning = annotate_transition_heuristic(prev_action, keypoint.action, failure_reasoning=True, 
                                                                          expert_action=expert_action, scene_info=info[WptInfoKey.SCENE_INFO])

                        # at perturb state, agent is commanded to go to the correction keypoint
                        perturb_idx = keypoint.info[WptInfoKey.PERTURB_IDX]
                        if episode[keywpt_id].perturbations[perturb_idx].correction is not None:
                            next_action = episode[keywpt_id].perturbations[perturb_idx].correction.action
                        else:
                            next_action = episode[keywpt_id].action
                        lang = annotate_transition_heuristic(keypoint.action, next_action, scene_info=info[WptInfoKey.SCENE_INFO])
                        info.update({
                            WptInfoKey.TRANSITION_TYPE: TransitionType.RECOVERABLE_FAILURE, 
                            WptInfoKey.FAILURE_REASON: failure_reasoning, 
                            WptInfoKey.LANG: lang,
                            WptInfoKey.CURRENT_POSE: {
                                "pos": np.round(keypoint.action.T, 3).tolist(),
                                "ori": np.round(keypoint.action.quat_to_array(keypoint.action.R), 3).tolist(),
                                "gripper_open": keypoint.action.gripper_open,
                                "ignore_collision": keypoint.action.ignore_collision
                            },
                            WptInfoKey.NEXT_POSE: {
                                "pos": np.round(next_action.T, 3).tolist(),
                                "ori": np.round(next_action.quat_to_array(next_action.R), 3).tolist(),
                                "gripper_open": next_action.gripper_open,
                                "ignore_collision": next_action.ignore_collision
                            },
                            WptInfoKey.VERBOSE: keypoint.verbose
                        })
                        episode[keywpt_id].perturbations[perturb_idx].mistake.info.update(info)
                    
                    elif keypoint.info[WptInfoKey.WPT_TYPE] == WptType.INTERMEDIATE:
                        # at intermediate state, agent is commanded to go to the current expert keypoint
                        perturb_idx = keypoint.info[WptInfoKey.PERTURB_IDX]
                        next_action = episode[keywpt_id].action
                        lang = annotate_transition_heuristic(keypoint.action, next_action)
                        info.update({
                            WptInfoKey.TRANSITION_TYPE: TransitionType.ONGOING, 
                            WptInfoKey.LANG: lang,
                            WptInfoKey.CURRENT_POSE: {
                                "pos": np.round(keypoint.action.T, 3).tolist(),
                                "ori": np.round(keypoint.action.quat_to_array(keypoint.action.R), 3).tolist(),
                                "gripper_open": keypoint.action.gripper_open,
                                "ignore_collision": keypoint.action.ignore_collision
                            },
                            WptInfoKey.NEXT_POSE: {
                                "pos": np.round(next_action.T, 3).tolist(),
                                "ori": np.round(next_action.quat_to_array(next_action.R), 3).tolist(),
                                "gripper_open": next_action.gripper_open,
                                "ignore_collision": next_action.ignore_collision
                            },
                            WptInfoKey.VERBOSE: keypoint.verbose
                        })
                        episode[keywpt_id].perturbations[perturb_idx].correction.info.update(info)
                    
                    else:
                        raise ValueError(f"Unknown waypoint type {keypoint.info[WptInfoKey.WPT_TYPE]}")
                if len(keypoint.dense) != 0:
                    for dense_keypoint in keypoint.dense:
                        if dense_keypoint.info.get(WptInfoKey.WPT_TYPE, None) != "expert":
                            dense_keypoint.info[WptInfoKey.WPT_TYPE] = "dense"
                            dense_keypoint.info[WptInfoKey.WPT_ID] = dense_keypoint.id
                            print("==========================================================>")
                        transition = self.step(RolloutAction(dense_keypoint.action.to_numpy(), dense_keypoint.info), is_record)

        episode.success = self.sim.is_success()
        episode.lang_goal = self.lang
        if episode.success:
            save_annotated_gif(self.ep_save_path, episode, num_perturb, prompt_gpt=prompt_gpt)
        
        return episode
    

    def rollout_model_only(self, is_record=True) -> bool:
        print("Start rollout with model only...")
        self.model_agent.reset()
        # which is better? original expert length = len(self.keywpts)-1 or maximum 25
        episode = self._model_rollout(0, 25, True, is_record) # returns an episode or an error string
        # self._model_rollout(1, 25, False, is_record)
        if episode != "error":
            save_annotated_gif(self.ep_save_path, annotated_episode=episode, rollout_type="rvt") # TODO: change this  
        # self.close()
        return episode

            
    def _model_rollout(self, start_idx_in_keywpts: int, end_idx_in_keywpts:int, is_reset=False, is_record=True):        
        # model auto rollout from start_idx_in_keywpts to end_idx_in_keywpts (inclusive)
        print(f"start model rollout from {start_idx_in_keywpts} to {end_idx_in_keywpts}")
        if is_reset: self.env_reset(is_record)

        episode = Episode(self.task_name, self.episode_num, self.lang, self.expert_demo)
        wpt_list = []
        for idx_in_keywpts in range(start_idx_in_keywpts, end_idx_in_keywpts):
            obs = self.transition.observation
            action = self.model_agent.act(obs)
            action_str = "\n\t".join(self._action_str_list(action.action))
            # if idx_in_keywpts <= len(self.keywpts)-1:
            # print(f"Prediction {self.keywpts[idx_in_keywpts]}, action: \n\t{action_str}")
            print(f"Prediction {idx_in_keywpts}, action: \n\t{action_str}")
            # action.info['wpt_id'] = idx_in_keywpts # TODO: which is better
            action.info[WptInfoKey.WPT_ID] = idx_in_keywpts
            action.info[WptInfoKey.WPT_TYPE] = WptType.PERTURB
            transition = self.step(action, is_record)
            if transition.info['error_status'] == "error":
                print("Ending the episode due to error")
                return "error"
            
            wpt = WayPoint(id=idx_in_keywpts, type="model", action=action.action, info=action.info)
            wpt_list.append(wpt)
        
        episode.waypoints = wpt_list
        episode.success = self.sim.is_success()
        return episode
            # if self.transition.terminal:
            #     break
    
    
    def rollout_expert_only(self, is_record=True) -> None:
        print("Start rollout with expert only...")
        self.env_reset(is_record)
        self.expert_agent.episode.print_info()
        self._expert_rollout(self.keywpts[0], self.keywpts[-1], False, is_record)
        save_annotated_gif(self.ep_save_path, self.expert_agent.episode)
        self.close()
        return self.sim.is_success()
    
    def rollout_expert_only_multi_task_multi_episode(self, task_list, ep_list, is_record=True) -> List[str]:
        results = []
        fail_after_retries = []  # List to store episodes that fail twice
        for task in task_list:
            for ep in ep_list:
                repeat = True  # Control variable to handle repetition
                attempts = 0  # Track the number of attempts for this task and episode
                max_attempts = 0
                while repeat:
                    print(task, ep)
                    self.env_reset(task_name=task, episode_num=ep, is_record=is_record)
                    self._expert_rollout(self.keywpts[0], self.keywpts[-1], False, is_record)
                    
                    success = self.sim.is_success()
                    # path_to_gif = save_annotated_gif(self.ep_save_path, self.expert_agent.episode)
                    path_to_gif = None
                    result = f"Task: {self.task_name} | {self.lang} | Episode: {ep} | Success: {success}"
                    results.append(result)

                    # Check if it should repeat
                    if not success and attempts < max_attempts:
                        attempts += 1  # Increment the attempts counter
                        print(f"Repeating Task: {task} | Episode: {ep} due to failure.")
                    else:
                        if not success:
                            fail_after_retries.append(f"Task: {task} | {self.lang} | Episode: {ep} failed after retry | {path_to_gif}")  # Append to the failed_twice list if it fails again
                        repeat = False  # Prevent further repetition

        self.close()
        print("Episodes that failed despite repeating:", fail_after_retries)
        return results, fail_after_retries
    
    def rollout_cmd_only(self, is_record=True) -> None:
        print("Start rollout with interactive terminal input only...")
        self.env_reset(is_record)
        self.expert_agent.episode.print_info()
        for _, keywpt_id in enumerate(range(100)):
            for action in self.cmd_agent.act_on_keywpt_id(keywpt_id):
                self.step(action, is_record)
                if self.sim.is_success():
                    print("success")
        self.close()
        print(self.sim.is_success())
    
    def rollout_heuristic_only(self, is_record=True, N=2, num_perturb=1) -> None:
        print("Start rollout with heuristic perturbation only...")
        self.env_reset(is_record)
        self.heuristic_agent.perturb_episode(N=N)
        self.heuristic_agent.episode.print_info()
        for _, keywpt_id in enumerate(self.keywpts):
            print("keypoint id:", keywpt_id)
            for action in self.heuristic_agent.act_on_keywpt_id(keywpt_id, num_perturb):
                action_str = "\n\t".join(self._action_str_list(action.action))
                print(f"current action: \n\t{action_str}")
                self.step(action, is_record)
        self.close()

        save_annotated_gif(self.ep_save_path, self.heuristic_agent.episode, num_perturb)
        
        return self.sim.is_success()
    
    
    def mix_rollout_on_keysteps(self, is_record=True):
        """
        blend all agents together, we can switch by inputing the agent number
        rollout based the keypoint steps, not the dense steps
        observations are even not needed for keystep-based rollout
        """
        assert not self.is_validator, "set is_validator to False to run this mix rollout"
        print("Start mix rollout, input 0 for expert (by default), 1 for heuristic, 2 for cmd", end=";")
        print(" If press enter, the rollout will maintain the current agent unless press other numbers.")
        
        self.env_reset(is_record)
        self.heuristic_agent.perturb_episode(N=1)
        current_agent = self.cmd_agent
        for _, keywpt_id in enumerate(self.keywpts[1:]):
            user_input = input("Continue or Change >>>")
            if user_input.strip() == "0":
                current_agent = self.expert_agent
            elif user_input.strip() == "1":
                current_agent = self.heuristic_agent
            elif user_input.strip() == "2":
                current_agent = self.cmd_agent
            else:
                pass
            for action in current_agent.act_on_keywpt_id(keywpt_id):
                self.step(action, is_record)
        self.close()