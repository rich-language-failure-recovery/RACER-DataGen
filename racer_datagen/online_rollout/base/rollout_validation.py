import os
import copy
import numpy as np
from multiprocessing import Process, Queue

from .utils import logger

from racer_datagen.data_aug.base.action import Action
from racer_datagen.data_aug.base.episode import Episode
from racer_datagen.data_aug.base.waypoint_and_perturbation import WayPoint, Perturbation
from racer_datagen.online_rollout.base.rollout_agent import RolloutAgent
from racer_datagen.online_rollout.base.constants import *



class RolloutValidation:
    def __init__(self, N=1, M=1, multiprocessing=False, **kwargs):
        self.N = N      # number of validated perturbations to generate for each keypoint
        self.M = M      # number of validated perturbations to store for each keypoint
        # assert(self.N >= self.M)
        self.kwargs = kwargs
        
        self.multiprocessing = multiprocessing
        if self.multiprocessing:
            self.action_queue = Queue()
            self.success_queue = Queue()
    
    def build_rollout_validator(self, action_queue, success_queue):
        print("build rollout validator...")
        kwargs = copy.deepcopy(self.kwargs)
        kwargs["save_dir"] += "_validator"
        kwargs["model_path"] = None  # pytorch model can not be shared between processes
        rollout_validator = RolloutAgent(**kwargs, debug=False)
        # check if stop_queue is empty
        while True:
            validation_args = action_queue.get()
            if validation_args is None:
                break
            success = rollout_validator.validate(**validation_args)
            success_queue.put(success)
        
        print("close rollout validator...")
        rollout_validator.close()
    
    def _run(self, func: callable, *args, **kwargs):
        if self.multiprocessing:
            p = Process(target=self.build_rollout_validator, args=(self.action_queue, self.success_queue))
            p.start()
        
        print("build rollout generator...")
        rollout_generator = RolloutAgent(**self.kwargs, debug=False)
        

        returns = func(rollout_generator, *args, **kwargs)
        print("close rollout generator...")
        rollout_generator.close()
        
        if self.multiprocessing:
            self.action_queue.put(None)
            p.join()
        
        return returns

    # rolls out expert + heuristic_agent.episode + expert
    def validate_heuristic_rollout(self, rollout_generator: RolloutAgent, is_record=True) -> Episode:
        rollout_generator.heuristic_agent.perturb_episode(N=self.N)
        episode = rollout_generator.heuristic_agent.episode
        episode.print_info()
                
        validated_episode = rollout_generator.expert_agent.episode
        keywpts = rollout_generator.keywpts[1:]
        warning_str = []

        rollout_generator.env_reset(is_record)
        for _, keywpt_id in enumerate(keywpts):
            print(f"\n\n--- validate keypoint id {keywpt_id} ---")
            validated_episode[keywpt_id].verbose = episode[keywpt_id].verbose
            # input("Press Enter to continue...")            
            # verify the perturbation one by one
            if self.M != 0:
                for perturb_idx in range(len(episode[keywpt_id].perturbations)):
                    perturb_action = episode[keywpt_id].perturbations[perturb_idx].mistake.action.to_numpy()
                    correction_action_list = []
                    if episode[keywpt_id].perturbations[perturb_idx].correction is not None:
                        correction_action_list.append(
                            episode[keywpt_id].perturbations[perturb_idx].correction.action.to_numpy())
                    correction_action_list.append(episode[keywpt_id].action.to_numpy())
                    
                    # validate perturbation
                    # prepare action queue
                    action_args = {
                        "valid_kwpt": keywpt_id, 
                        "perturb_action": perturb_action, 
                        "correct_action": correction_action_list,
                        "is_record": False # set True for bebug
                    }

                    if self.multiprocessing:
                        # wait for validation result
                        self.action_queue.put(action_args)
                        success = self.success_queue.get()
                    else:
                        success = rollout_generator.validate(**action_args)

                
                    print(f"Validation result for keypoint {keywpt_id}, perturb {perturb_idx}, success: {success}")         
                    
                    # store validated perturbations
                    if success:
                        validated_episode[keywpt_id].perturbations.append(episode[keywpt_id].perturbations[perturb_idx])
                        local_perturb_idx = len(validated_episode[keywpt_id].perturbations) - 1
                        validated_episode[keywpt_id].perturbations[local_perturb_idx].mistake.info.update({WptInfoKey.PERTURB_TYPE: PerturbType.HEURISTIC})
                        if validated_episode[keywpt_id].perturbations[local_perturb_idx].correction is not None:
                            validated_episode[keywpt_id].perturbations[local_perturb_idx].correction.info.update({WptInfoKey.PERTURB_TYPE: PerturbType.HEURISTIC})
                    else:
                        warning_str.append(f"Validation failed for keypoint {keywpt_id}, perturb {perturb_idx}. Moving on to next perturb idx.")

                    # break if we have enough perturbations for this keypoint
                    print(f"len(validated_episode[keywpt_id].perturbations): {len(validated_episode[keywpt_id].perturbations)} and {self.M}")
                    if len(validated_episode[keywpt_id].perturbations) == self.M:
                        print("Enough perturbations for this keypoint, break")
                        break

                if len(episode[keywpt_id].perturbations) > 0:
                    if len(validated_episode[keywpt_id].perturbations) < self.M:
                        warning_str.append(f"Only {len(validated_episode[keywpt_id].perturbations)} perturbations for keypoint {keywpt_id}. {self.M} required")
                        # TODO: what is the ideal way to handle this? maybe fetch more perturbations?
                        # print(f"Only {len(validated_episode[keywpt_id].perturbations)} perturbations for keypoint {keywpt_id}. {self.M} required")
                        raise ValueError(f"Only {len(validated_episode[keywpt_id].perturbations)} perturbations for keypoint {keywpt_id}. {self.M} required")
                
        logger.warning(warning_str)
        print("Finish validation for all perturbations...")

        validated_episode.update_info()
        validated_episode.save_path = os.path.join(rollout_generator.ep_save_path, 'heuristic_episode.pkl')
        return validated_episode           
                     
    def run_heuristic_rollout_validation(self, is_record=True):
        return self._run(lambda x: self.validate_heuristic_rollout(x, is_record=is_record))        
        
    def validate_model_rollout(self, rollout_generator: RolloutAgent, is_record=True) -> Episode:
        """
        start from last to first keypoint
        rollout models on a part (consecutive steps) of the trajectory to see when it fails at the first time, 
            if fails then that's the failure point
            if continue with expert keypoint can lead to success, then the expert keypoint is a correction.
        then start the above process on previous part of the trajectory iteraitvely.
        
        Update: We found that failure sometimes can not determine it's alway because of the first step of the consecutive steps.
        So we may only use model rollout a single step, then decide whether it fails or not from deviation with expert demo and the reward.
        """
        
        # N=0, no perturbation, just to get  locating alignment / general grasp / general place steps
        rollout_generator.heuristic_agent.perturb_episode(N=0) 
        episode = rollout_generator.heuristic_agent.episode
        episode.print_info()
        validated_episode = rollout_generator.expert_agent.episode
        added_perturb_info_str = []
        
        rollout_generator.model_agent.reset()
        model_name = rollout_generator.model_agent.name
        print(list(enumerate(rollout_generator.keywpts)))
        
        for idx_in_keywpts, keywpt_id in enumerate(rollout_generator.keywpts):
            if idx_in_keywpts == 0:
                continue
            if "is" not in episode[keywpt_id].verbose: # skip unimportant keypoint
                continue
            print(f"\n\n--- validate keypoint id {keywpt_id} ---")
            print("reset env...")
            rollout_generator.env_reset(is_record)
        
            print("begin with expert rollout...")
            rollout_generator._expert_rollout(
                start_kwpt=rollout_generator.keywpts[0], 
                end_kwpt=rollout_generator.keywpts[idx_in_keywpts-1], 
                is_reset=False, 
                is_record=is_record)
            
            print("switch to model rollout...")
            rollout_generator._model_rollout(
                start_idx_in_keywpts=idx_in_keywpts, 
                end_idx_in_keywpts=idx_in_keywpts, 
                is_reset=False,
                is_record=is_record)
            
            model_action = rollout_generator.prev_action
            expert_action = episode[keywpt_id].action.to_numpy()
            print(f"model action: {[round(n, 4) for n in model_action.tolist()]}")
            print(f"expert action: {[round(n, 4) for n in expert_action.tolist()]}")

            print("switch back to expert rollout...")
            rollout_generator._expert_rollout(
                start_kwpt=rollout_generator.keywpts[idx_in_keywpts+1], 
                end_kwpt=rollout_generator.keywpts[len(rollout_generator.keywpts)-1], 
                is_reset=False, 
                is_record=is_record)
                    
            rollout_success = rollout_generator.sim.is_success()
            action_args = {
                    "valid_kwpt": keywpt_id, 
                    "perturb_action": model_action, 
                    "correct_action": [expert_action],
                    "is_record": False # set True for bebug
                }
            
            if self.multiprocessing:
                self.action_queue.put(action_args)
                validate_success = self.success_queue.get()
            else:
                validate_success = rollout_generator.validate(**action_args)   
            # TODO: think about how to determine it's a catastrophic failure or recoverable failure or success
            # based on rollout_success, validate_success, diff
            # and store those information in episode
            
            if not rollout_success and validate_success:
                mistake_wpt = WayPoint(
                        id=keywpt_id, 
                        type='model', 
                        action=Action.from_numpy(model_action), 
                        info={
                            WptInfoKey.WPT_TYPE: WptType.PERTURB,
                            WptInfoKey.WPT_ID: keywpt_id,
                            WptInfoKey.PERTURB_TYPE: model_name,
                            WptInfoKey.PERTURB_IDX: 0
                        }
                    )
                validated_episode[keywpt_id].perturbations.append(Perturbation(mistake=mistake_wpt, correction=None))
                added_perturb_info_str.append(f"Add perturbation for keypoint {keywpt_id} due to false rollout_success")
            
            if rollout_success and validate_success:
                diff = find_action_difference(Action.from_numpy(model_action), Action.from_numpy(expert_action))
                # TODO: this condition needs to be improved. mostly this is added due to collision mismatch. but is expert action the right *correction*?
                # probably the best strategy is to just detect *failure* that can be corrected by expert? (the one above)
                if diff['translation'] > 0.01 or diff['rotation'] > 10 or diff['gripper'] != 0 or diff['collision'] != 0:
                    mistake_wpt = WayPoint(
                        id=keywpt_id, 
                        type='model', 
                        action=Action.from_numpy(model_action), 
                        info={
                            WptInfoKey.WPT_TYPE: WptType.PERTURB,
                            WptInfoKey.WPT_ID: keywpt_id,
                            WptInfoKey.PERTURB_TYPE: model_name,
                            WptInfoKey.PERTURB_IDX: 0
                        }
                    )
                    validated_episode[keywpt_id].perturbations.append(Perturbation(mistake=mistake_wpt, correction=None))
                    added_perturb_info_str.append({f"Add perturbation for keypoint {keywpt_id} due to diff {diff}": [model_action, expert_action]})
            
            if not rollout_success and not validate_success:
                # catastrophic failure
                # generate this separately for LLaVA
                pass
            
        
        validated_episode.update_info()
        validated_episode.save_path = os.path.join(rollout_generator.ep_save_path, f'{model_name}_episode.pkl')
        for perturb_info in added_perturb_info_str:
            print(perturb_info)     
        return validated_episode
    
    def run_model_rollout_validation(self, is_record=True) -> Episode:
        episode = self._run(lambda x: self.validate_model_rollout(x, is_record=is_record))
        return episode



def find_action_difference(pd_action: Action, gt_action: Action) -> dict:
    diff = Action.delta_action(pd_action, gt_action)
    return {
        'translation': np.linalg.norm(diff['translation']),
        'rotation': np.linalg.norm(diff['rotation']),
        'gripper': diff['gripper'],
        'collision': diff['collision']
    }