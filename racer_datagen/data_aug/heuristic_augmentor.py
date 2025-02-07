import copy
import random
from racer_datagen.data_aug.base.data_augmentor import DataAugmentor, TASK_BOUND
from racer_datagen.data_aug.base.episode import Episode
from racer_datagen.data_aug.base.waypoint_and_perturbation import WayPoint

class Heuristic(DataAugmentor):
    def __init__(self, task_name: str, cfg_path:str):
        super().__init__(task_name, cfg_path)
        self._hard_task = [] #['place_cups']
        self.task_with_circle_object = ["close_jar", "light_bulb_in"] # these tasks have a circular object, pure rotation perturbation does not help
    
    
    def heuristic_perturb(self, episode: Episode, N=100) -> Episode:
        """main perturb logics, more details check 
        https://docs.google.com/document/d/1B5zaAZ64jqweI1Ae5EZ_pCfiJT2lWu-WV5Xo8ABhAmk/edit

        1. determine general grasp & place waypoints from keypoints
        2. perturb grasp:
            2.1. find alignment steps of the grasp
            2.2. perturb the coarse first alignment step with T & R
            2.3. perturb the general grasp step with T, need intermediate waypoints for corrects
        3. perturb place:
            3.1. find alignment steps of the placing
            3.2. perturb the coarse (first) alignment steps with T & R
            3.3. perturb the fine-grained (middle) alignment steps with T & R for some tasks
            3.4. perturb the general place step, only diminish the distance of the alignment direction
        4. Do not perturb the gripper open/close action except perturbing the last step's gripper open to close
    
        Args: 
            episode: original Episode
        Returns:
            Episode: perturbed Episode (perturbation is not None for some keypoints)

        """
        flag_for_grasp = True
        flag_for_place = False
        used_wpt_idx = set()
        alignment_count = 0
        for idx, wpt in enumerate(episode.waypoints):
            if not self._is_keypoint(episode, idx) or idx == 0: 
                continue
            
            # check if the waypoint is out of the taskbound
            # for close_jar task only
            xyz = wpt.action.translation
            if not all([TASK_BOUND[i] <= xyz[i] <= TASK_BOUND[i+3] for i in range(3)]):
                # and self.task_name in ["close_jar"]:
                continue
            
            elif flag_for_grasp and self.is_general_grasp_step(episode, idx):
                flag_for_grasp = False
                flag_for_place = True
                alignment_steps = self.find_alignment_steps(episode, idx)
                
                if episode.task_name == "stack_cups" or episode.task_name == "put_item_in_drawer":
                    continue

                if len(alignment_steps) > 0:
                    first_alignment_step = alignment_steps[0]
                    if abs(first_alignment_step-idx) < 6: # too close
                        continue
                    first_alignment_wpt = episode[first_alignment_step]
                    # perturb the first alignment step
                    self._call_n_times(N)(self._add_perturbation_on_alignment_step)(
                        source_wpt=first_alignment_wpt,
                        target_wpt=wpt,
                        type="coarse")
                    used_wpt_idx.add(first_alignment_step)

                    if episode.task_name == "slide_block_to_color_target":
                        # find place step
                        for idx2, wpt2 in enumerate(episode.waypoints):
                            if not self._is_keypoint(episode, idx2) or idx2 == 0: 
                                continue
                            if self.is_general_place_step(episode, idx2):
                                place_wpt = wpt2
                                place_wpt.ignore_collision = True
                                break
                        # perturb the general place step
                        if alignment_count == 0:
                            _, next_idx = self._check_next_neighbor(episode, idx)
                            next_wpt = episode.waypoints[next_idx]
                            print("=====================================")
                            print(wpt.T)
                            print(next_wpt.T)
                            print(first_alignment_wpt.T)
                            print(place_wpt.T)
                            print("=====================================")
                            self._call_n_times(N)(self._add_perturbation_on_grasp_step2)(                        
                                alignment_wpt=first_alignment_wpt, 
                                grasp_wpt=wpt,
                                place_wpt=next_wpt,
                                episode=episode,
                                grasp_step=idx)
                            used_wpt_idx.add(idx2)
                            alignment_count += 1
                    # else:
                    # perturb the general grasp step

                    # if episode.task_name == "slide_block_to_color_target":
                    #     self._call_n_times(N)(self._add_perturbation_on_grasp_step3)(
                    #         alignment_wpt=first_alignment_wpt, 
                    #         grasp_wpt=wpt,
                    #         episode=episode,
                    #         grasp_step=idx)
                    #     used_wpt_idx.add(idx)
                    else:
                        self._call_n_times(N)(self._add_perturbation_on_grasp_step)(
                            alignment_wpt=first_alignment_wpt, 
                            grasp_wpt=wpt,
                            episode=episode,
                            grasp_step=idx)
                        used_wpt_idx.add(idx)
            
            elif flag_for_place and self.is_general_place_step(episode, idx):
                flag_for_place = False
                flag_for_grasp = True
                alignment_steps = self.find_alignment_steps(episode, idx)
                
                if episode.task_name == "insert_onto_square_peg":
                    continue

                if len(alignment_steps) > 0:
                    first_alignment_step = alignment_steps[0]
                    if abs(first_alignment_step-idx) < 6: # too close
                        continue
                    first_alignment_wpt = episode[first_alignment_step]
                    if episode.task_name == "put_item_in_drawer":
                        # perturb the first alignment step
                        print("perturb the first alignment step --------------------------")
                        
                        _, last_idx = self._check_last_neighbor(episode, first_alignment_step)
                        print(first_alignment_step, last_idx, "----------------------")
                        last_wpt = episode.waypoints[last_idx]
                        print(first_alignment_wpt.T, last_wpt.T)
                        self._call_n_times(N)(self._add_perturbation_on_alignment_step2)(
                            source_wpt=last_wpt, 
                            ref_wpt=first_alignment_wpt,
                            target_wpt=wpt,
                            type="coarse")
                        used_wpt_idx.add(first_alignment_step)
                    else:
                        # perturb the first alignment step
                        if episode.task_name != "slide_block_to_color_target":
                            self._call_n_times(N)(self._add_perturbation_on_alignment_step)(
                                source_wpt=first_alignment_wpt, 
                                target_wpt=wpt,
                                type="coarse")
                            used_wpt_idx.add(first_alignment_step)
                    
                    # perturb the fine-grained (middle) alignment steps with T & R for some tasks
                    if len(alignment_steps) > 1 and self.task_name in self._hard_task:
                        for alignment_step in alignment_steps[1:-1]:
                            alignment_wpt = episode[alignment_step]
                            self._call_n_times(N)(self._add_perturbation_on_alignment_step)(
                                source_wpt=alignment_wpt, 
                                target_wpt=wpt,
                                type="fine")
                            used_wpt_idx.add(alignment_step)
                
                    # perturb the general place step
                    self._call_n_times(N)(self._add_perturbation_on_place_step)(                        
                        alignment_wpt=first_alignment_wpt, 
                        place_wpt=wpt)
                    used_wpt_idx.add(idx)
                
                else:
                    # check if the last keypoint is gneral grasp, if so, use interpolation noise
                    # for open_drawer task only
                    keysteps = list(episode.retrieve_keypoints(start_index=idx, end_index=0))
                    if len(keysteps) >1 and episode.waypoints[keysteps[1]].verbose.get("is", None) == "general grasp":
                        self._call_n_times(N)(self._add_perturbation_on_place_step)(
                            alignment_wpt=episode[keysteps[1]],
                            place_wpt=wpt)
                        used_wpt_idx.add(idx)
                    

            # TODO: seems like this is already handled in the general place step         
            # elif idx == len(episode.waypoints)-1 and self._is_gripper_open(episode, idx):
            #     self._add_perturbation_on_last_gripper_open_step(wpt)
            #     used_wpt_idx.add(idx)
            else:
                continue
        
        return episode  
        
    @staticmethod
    def _call_n_times(n):
        def decorator(func):
            def wrapper(*args, **kwargs):
                for _ in range(n):
                    func(*args, **kwargs)
            return wrapper
        return decorator
    
    def _add_perturbation_on_alignment_step(
        self, 
        source_wpt:WayPoint, 
        target_wpt:WayPoint,
        type:str, #[coarse, fine]
    ): 
        if type == "coarse":
            cfg = self.cfg.coarse_alignment
        else:
            cfg = self.cfg.fine_alignment
        if not cfg.use_translation_noise and not cfg.use_rotation_noise:
            return
        alignment_direction = target_wpt.T - source_wpt.T
        perturbed_R = self.perturb_R_around_axis(R=source_wpt.R, rotate_axis=alignment_direction, **cfg.rotation_noise)
        if random.random() < 0.5:
            perturbed_T = self.perturb_T_in_plane(T=source_wpt.T, alignment_axis=alignment_direction, **cfg.translation_noise)
        else:
            perturbed_T = self.perturb_T_in_space(T=source_wpt.T, **cfg.translation_noise)
        action = copy.deepcopy(source_wpt.action)  
        
        if self.task_name not in self.task_with_circle_object:                  
            prob = random.random()
            if not cfg.use_translation_noise:
                action.R = perturbed_R
            elif not cfg.use_rotation_noise:
                action.T = perturbed_T
            else:
                if prob < 1/3: 
                    action.R = perturbed_R
                elif 1/3<=prob < 2/3:
                    action.T = perturbed_T
                else:
                    action.T = perturbed_T
                    action.R = perturbed_R
        else:
            action.T = perturbed_T
        action.ignore_collision = False # always consider collision
        source_wpt.add_perturbation(mistake_action=action)

    def _add_perturbation_on_alignment_step2(
        self, 
        source_wpt:WayPoint, 
        ref_wpt:WayPoint,
        target_wpt:WayPoint,
        type:str, #[coarse, fine]
    ): 
        if type == "coarse":
            cfg = self.cfg.coarse_alignment
        else:
            cfg = self.cfg.fine_alignment
        if not cfg.use_translation_noise and not cfg.use_rotation_noise:
            return
        alignment_direction = target_wpt.T - source_wpt.T
        # source_wpt.T[2] = target_wpt.T[2]
        perturbed_T_line = self.perturb_T_in_line_extrapolate(
            T_src=source_wpt.T, T_dst=ref_wpt.T, noise_type="uniform", scale=1.0, lowerbound=1.2, upperbound=1.45)
        action = copy.deepcopy(ref_wpt.action)
        action.T = perturbed_T_line
        # action.R = perturbed_R_line
        action.gripper_open = False
        # target_wpt.add_perturbation(mistake_action=action)
        # source_wpt.add_perturbation(mistake_action=action)
        ref_wpt.add_perturbation(mistake_action=action)
        

    def _add_perturbation_on_grasp_step(
        self, 
        alignment_wpt:WayPoint, 
        grasp_wpt:WayPoint,
        episode:Episode,
        grasp_step: int,
    ):
        if not self.cfg.general_grasp.use_interpolation_noise and not self.cfg.general_grasp.use_translation_noise:
            return
        perturbed_T_line = self.perturb_T_in_line(
            T_src=alignment_wpt.T, T_dst=grasp_wpt.T, **self.cfg.general_grasp.interpolation_noise)
        perturbed_T_plane = self.perturb_T_in_plane(
                T=grasp_wpt.T, alignment_axis=grasp_wpt.T - alignment_wpt.T, **self.cfg.general_grasp.translation_noise)
        if not self.cfg.general_grasp.use_translation_noise:
            perturbed_T = perturbed_T_line
        elif not self.cfg.general_grasp.use_interpolation_noise:
            perturbed_T = perturbed_T_plane
        else:
            if random.random() < 0.5:
                # add some perturbation in the perpendicular plane of the alignment direction
                perturbed_T = perturbed_T_plane + perturbed_T_line - grasp_wpt.T
            else:
                perturbed_T = perturbed_T_line
        action = copy.deepcopy(grasp_wpt.action)
        action.T = perturbed_T
        # sample an intermediate correction, and do not allow collision
        correct_action = copy.deepcopy(episode.sample_previous_intermediate_waypoints(
            keypoint_index=grasp_step, **self.cfg.intermediate
            ).action)
        # correct_action.gripper_open = True # TODO is setting this to True correct?
        correct_action.ignore_collision = False
        grasp_wpt.add_perturbation(mistake_action=action, correction_action=correct_action)

    def _add_perturbation_on_grasp_step2(
        self, 
        alignment_wpt:WayPoint, 
        grasp_wpt:WayPoint,
        place_wpt:WayPoint,
        episode:Episode,
        grasp_step: int,
    ):
        if not self.cfg.general_grasp.use_interpolation_noise and not self.cfg.general_grasp.use_translation_noise:
            return
        perturbed_T_line = self.perturb_T_in_line(
            T_src=grasp_wpt.T, T_dst=place_wpt.T, **self.cfg.general_grasp.interpolation_noise)
        perturbed_T = perturbed_T_line
        action = copy.deepcopy(grasp_wpt.action)
        action.T = perturbed_T
        action.ignore_collision = False
        grasp_wpt.add_perturbation(mistake_action=action, correction_action=None)
        grasp_wpt.ignore_collision = False

    def _add_perturbation_on_grasp_step3(
        self, 
        alignment_wpt:WayPoint, 
        grasp_wpt:WayPoint,
        episode:Episode,
        grasp_step: int,
    ):
        if not self.cfg.general_grasp.use_interpolation_noise and not self.cfg.general_grasp.use_translation_noise:
            return
        perturbed_T_line = self.perturb_T_in_line(
            T_src=alignment_wpt.T, T_dst=grasp_wpt.T, **self.cfg.general_grasp.interpolation_noise)
        perturbed_T_plane = self.perturb_T_in_plane(
                T=grasp_wpt.T, alignment_axis=grasp_wpt.T - alignment_wpt.T, **self.cfg.general_grasp.translation_noise)
        if not self.cfg.general_grasp.use_translation_noise:
            perturbed_T = perturbed_T_line
        elif not self.cfg.general_grasp.use_interpolation_noise:
            perturbed_T = perturbed_T_plane
        else:
            if random.random() < 0.5:
                # add some perturbation in the perpendicular plane of the alignment direction
                perturbed_T = perturbed_T_plane + perturbed_T_line - grasp_wpt.T
            else:
                perturbed_T = perturbed_T_line
        action = copy.deepcopy(grasp_wpt.action)
        action.T = perturbed_T
        # sample an intermediate correction, and do not allow collision
        correct_action = copy.deepcopy(episode.sample_previous_intermediate_waypoints(
            keypoint_index=grasp_step, **self.cfg.intermediate
            ).action)
        # correct_action.gripper_open = True # TODO is setting this to True correct?
        correct_action.ignore_collision = False
        grasp_wpt.add_perturbation(mistake_action=action, correction_action=None)

    
    def _add_perturbation_on_place_step(
        self, 
        alignment_wpt:WayPoint, 
        place_wpt:WayPoint,
    ):
        if not self.cfg.general_place.use_interpolation_noise:
            return
        # perturbed_T_line = self.perturb_T_in_line(
        #     T_src=alignment_wpt.T, T_dst=place_wpt.T, **self.cfg.general_place.interpolation_noise)
        perturbed_T_line, perturbed_R_line = self.perturb_TR_in_line(
            T_src=alignment_wpt.T, T_dst=place_wpt.T, 
            R_src=alignment_wpt.R, R_dst=place_wpt.R,
            **self.cfg.general_place.interpolation_noise)
        action = copy.deepcopy(place_wpt.action)
        action.T = perturbed_T_line
        action.R = perturbed_R_line
        action.gripper_open = False
        place_wpt.add_perturbation(mistake_action=action)

    # def _add_perturbation_on_last_gripper_open_step(self, wpt:WayPoint):
    #     action = copy.deepcopy(wpt.action)
    #     action.gripper_open = False # perturb gripper open to close
    #     wpt.add_perturbation(mistake_action=action)