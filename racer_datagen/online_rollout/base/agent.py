import os
from typing import Dict
import numpy as np
import torch

from racer_datagen.data_aug.base.episode import Episode

from .utils import logger
from .constants import PerturbType, WptInfoKey
from yarr.agents.agent import ActResult
from racer_datagen.data_aug.heuristic_augmentor import Heuristic

class RolloutAction:
    def __init__(self, action: np.ndarray, info: dict = {}):
        self.action = action
        self.info = info

class Agent:
    r"""Abstract class for defining agents which act inside :ref:`core.env.Env`.

    This abstract class standardizes agents to allow seamless benchmarking.
    """
    type: str = "agent"
    def reset(self) -> None:
        r"""Called before starting a new episode in environment."""
        raise NotImplementedError

    def act(
        self, input_obs,
    ):
        # mainly for model-based agents
        raise NotImplementedError
    
    def act_on_keywpt_id(self, wpt_idx):
        # for discrete control 
        # input a keypoint index like 57, return the action at the time step
        # mainly for expert/heuristic/cmd interactive agents
        raise NotImplementedError


class ExpertAgent(Agent):
    r"""Agent that acts according to expert demonstration."""
    
    def __init__(self, task_name, episode_num, lang_goal, demos):
        self.episode = Episode.from_demos(task_name, demos, ep_num=episode_num, lang_goal=lang_goal)
            
    def act_on_keywpt_id(self, wpt_id):
        assert wpt_id < len(self.episode), "wpt_id should be less than the number of waypoints in the episode."
        for keypoint in self.episode.iterate_one_keypoint(wpt_id, verbose=True):
            if keypoint.verbose.get("is", None) == "alignment step":
                keypoint.action.ignore_collision = False # always consider collision for alignment steps
            yield RolloutAction(keypoint.action.to_numpy(), keypoint.info)
            if len(keypoint.dense) != 0:
                for dense_keypoint in keypoint.dense:
                    if dense_keypoint.info.get(WptInfoKey.WPT_TYPE, None) != "expert":
                        dense_keypoint.info[WptInfoKey.WPT_TYPE] = "dense"
                        dense_keypoint.info[WptInfoKey.WPT_ID] = dense_keypoint.id
                        print("==========================================================>")
                    yield RolloutAction(dense_keypoint.action.to_numpy(), dense_keypoint.info)
            
class HeuristicAgent(Agent):
    r"""Agent that acts according to heuristic perturbation."""
    
    def __init__(self, task_name, episode_num, lang_goal, demos, cfg_root):
        self.episode = Episode.from_demos(task_name, demos, ep_num=episode_num, lang_goal=lang_goal)
        self.augmentor = Heuristic(task_name, cfg_path=os.path.join(cfg_root, f"{task_name}.yaml"))
        
    def perturb_episode(self, N=100):
        self.episode = self.augmentor.heuristic_perturb(self.episode, N=N)
        self.episode[0].perturbations.clear()
        self.episode.update_info()
    
    def perturbed_wpt_idx(self):
        return list(self.augmentor.retrieve_perturbed_keypoints(self.episode))
    
    def act_on_keywpt_id(self, wpt_id, num_perturb=1):
        assert wpt_id < len(self.episode), "wpt_id should be less than the number of waypoints in the episode."
        for keypoint in self.episode.iterate_one_keypoint(wpt_id, num_perturb, verbose=True):
            yield RolloutAction(keypoint.action.to_numpy(), keypoint.info)


class CmdAgent(Agent):
    r"""Agent that acts according to command line input."""
    
    def act_on_keywpt_id(self, wpt_id):
        user_action_str = input("Enter proposed action:")
        user_action = np.fromstring(user_action_str, dtype=float, sep=' ')
        while len(user_action) != 9:
            print("  == Invalid action length. Please enter 9 values with space to split them.==")
            user_action_str = input("Enter proposed action:")
            user_action = np.fromstring(user_action_str, dtype=float, sep=' ')
        yield RolloutAction(user_action, {WptInfoKey.WPT_ID: wpt_id, WptInfoKey.WPT_TYPE: PerturbType.CMD})

class ModelRVTAgent(Agent):
    r"""Agent that acts according to a model trained with RVT."""
    def __init__(
        self, 
        model_path: str, 
        device: int,
        debug_log_dir: str,
    ):
        self.model_path = model_path
        self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        self.name = PerturbType.RVT
        
        import rvt.config as default_exp_cfg
        import rvt.mvt.config as default_mvt_cfg
        model_folder = os.path.dirname(model_path)
        
        # load exp_cfg
        exp_cfg = default_exp_cfg.get_cfg_defaults()
        exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))
        # WARNING NOTE: a temporary hack to use place_with_mean in evaluation
        exp_cfg.rvt.place_with_mean = True
        exp_cfg.freeze()
        
        # load mvt_cfg
        mvt_cfg = default_mvt_cfg.get_cfg_defaults()
        mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))
        mvt_cfg.freeze()
        
        from racer_datagen.mvt.mvt import MVT
        rvt = MVT(
                renderer_device=self.device,
                **mvt_cfg,
        )
        
        import rvt.models.rvt_agent as rvt_agent
        from racer_datagen.utils.peract_utils import (
            CAMERAS,
            SCENE_BOUNDS,
            IMAGE_SIZE,
        )
        self.agent = rvt_agent.RVTAgent(
                network=rvt.to(self.device),
                image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
                add_lang=mvt_cfg.add_lang,
                scene_bounds=SCENE_BOUNDS,
                cameras=CAMERAS,
                log_dir=f"{debug_log_dir}/eval_run",
                **exp_cfg.peract,
                **exp_cfg.rvt,
            )
    
    def reset(self):
        from racer_datagen.utils.rvt_utils import load_agent as load_agent_state
        self.agent.build(training=False, device=self.device)
        load_agent_state(self.model_path, self.agent)
        self.agent.eval()
        self.agent.load_clip()
        self.agent.reset()
        print("Agent Reset. Model loaded.")
        print(self.agent)
    
    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype
    
    def _wrap_obs(self, obs: dict) -> Dict[str, torch.Tensor]:
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] for k, v in obs.items()}
        prepped_data = {k:torch.tensor(np.array([v]), device=self.device) for k, v in obs_history.items()}
        return prepped_data

    def act(self, input_obs:dict)-> RolloutAction:
        obs_tensor = self._wrap_obs(input_obs)
        act_result: ActResult = self.agent.act(step=0, observation=obs_tensor)
        return RolloutAction(act_result.action, {WptInfoKey.WPT_ID: -1})
    
    def act_on_keywpt_id(self, wpt_id):
        logger.warning("Model prediction do not support waypoint index. Please use `act` instead.")
        yield None

class TeleopAgent(Agent):
    r"""Agent that acts according to human teleoperation."""
    pass
