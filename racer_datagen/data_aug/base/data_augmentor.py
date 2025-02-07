
import os
import random
from typing import Tuple
from numpy import ndarray
from rlbench.demo import Demo
from racer_datagen.utils.rvt_utils import RLBENCH_TASKS
import numpy as np
import quaternion

from .episode import Episode
from .config import get_cfg_defaults

from functools import wraps

TASK_BOUND = [-0.075, -0.455, 0.752, 0.500, 0.455, 1.477]


def check_task_bound(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        T = func(*args, **kwargs)
        if isinstance(T, tuple):
            xyz = T[0]
        else:
            xyz = T
        while not all([TASK_BOUND[i] < xyz[i] < TASK_BOUND[i+3] for i in range(3)]):
            T = func(*args, **kwargs)
            if isinstance(T, tuple):
                xyz = T[0]
            else:
                xyz = T
        return T
    return wrapper


class DataAugmentor:
    
    def __init__(self, task_name: str, cfg_path: str = ""):
        self.task_name = task_name
        assert task_name in RLBENCH_TASKS, f"Task {task_name} not found in RLBENCH_TASKS"  
        self.cfg = get_cfg_defaults()
        self.cfg_path = cfg_path
        if os.path.exists(cfg_path):
            self.cfg.merge_from_file(cfg_path)
                      
    def _is_same_T(self, T1: ndarray, T2: ndarray) -> bool:
        return np.linalg.norm(T1 - T2) < self.cfg.is_same_T_thres
    
    def _is_same_R(self, R1: ndarray, R2: ndarray) -> bool:
        return np.linalg.norm(R1 - R2) < self.cfg.is_same_R_thres
    
    def _is_in_line(self, T: ndarray, line_T1: ndarray, line_T2: ndarray) -> bool:
        """
        check if the point T is in the line fromed by T1 and T2
        """
        cross_product = np.cross(line_T2 - line_T1, T - line_T1)
        area_of_parallelogram = np.linalg.norm(cross_product)
        base_length = np.linalg.norm(line_T2 - line_T1)
        distance = area_of_parallelogram / base_length
        return abs(distance) < self.cfg.is_in_line_thres
    
    @staticmethod
    def _is_keypoint(episode: Episode, index: int) -> bool:
        """
        check if the episode[index] is a keypoint
        """
        return episode[index].type == 'keypoint'
    
    
    def retrieve_perturbed_keypoints(self, episode: Episode):
        for idx, wpt in enumerate(episode.waypoints):
            if not self._is_keypoint(episode, idx) or idx == 0: 
                continue
            else:
                if wpt.perturbations:
                    yield idx
    
    def _check_last_neighbor(self, episode: Episode, index: int):
        assert self._is_keypoint(episode, index), "The index should be a keypoint"
        if index == 0:
            # index=0 is the starting pose for Franka
            # we consider it as a special keypoint
            return index, None
        try:
            gen = episode.retrieve_keypoints(index-1, -1e5)
            last_index = next(gen)
        except StopIteration:
            assert False, "No last keypoint found"
        return index, last_index
    
    
    def _check_next_neighbor(self, episode: Episode, index: int):
        assert self._is_keypoint(episode, index), "The index should be a keypoint"
        if index == len(episode) - 1:
            return index, None
        try:
            gen = episode.retrieve_keypoints(index+1, 1e5)
            next_index = next(gen)
        except StopIteration:
            assert False, "No last keypoint found"
        return index, next_index
    
    
    def _is_screw(self, episode: Episode, index: int) -> bool:
        """
        check if the episode[index+1] is a screw/rotate action
        which means index could be a general place action
        """
        index, next_index = self._check_next_neighbor(episode, index)
        if next_index is None: return False
        if self._is_same_T(episode[index].T, episode[next_index].T) and \
            not self._is_same_R(episode[index].Rmat, episode[next_index].Rmat):
            return True
        return False
        
        
    def _is_gripper_open(self, episode: Episode, index: int) -> bool:
        """
        check if the episode[index] is a gripper start to open action
        """
        index, last_index = self._check_last_neighbor(episode, index)
        if last_index is None:
            return episode[index].gripper_open
        if episode[index].gripper_open and episode[last_index].gripper_close:
            # if index == len(episode) - 1:
            #     episode[index].verbose.update(
            #         {"is": "gripper open", "reason": "last step of the episode that opens gripper"}
            #     )
            return True
        return False
    
    def _is_gripper_close(self, episode: Episode, index: int) -> bool:
        """
        check if the episode[index] is a gripper start to close action
        """
        index, last_index = self._check_last_neighbor(episode, index)
        if last_index is None:
            return episode[index].gripper_close
        if episode[index].gripper_close and episode[last_index].gripper_open:
            return True
        return False
    
    def is_general_grasp_step(self, episode: Episode, index: int) -> bool:
        """
        check if the episode[index] is a general grasp action
        any of the following:
        1. the gripper starts to close
        """
        assert self._is_keypoint(episode, index), "The index should be a keypoint"
        if index == 0: return False
        if self._is_gripper_close(episode, index):
            episode[index].verbose.update(
                {"is": "general grasp", "reason": "gripper start to close"}
            )
            return True
        else:
            return False
            
    
    def is_general_place_step(self, episode: Episode, index: int) -> bool:
        """
        check if the episode[index] is a general place action
        any of following:
        1. the gripper starts to open
        2. the step before screw action
        3. the last step of the episode
        """
        assert self._is_keypoint(episode, index), "The index should be a keypoint"
        if index == 0: return False
        if self._is_screw(episode, index):
            episode[index].verbose.update(
                {"is": "general place", "reason": "screw action"}
            )
            return True
        elif self._is_gripper_open(episode, index):
            episode[index].verbose.update(
                {"is": "general place", "reason": "gripper start to open"}
            )
            return True
        
        elif index == len(episode) - 1:
            episode[index].verbose.update(
                {"is": "general place", "reason": "last step of the episode"}
            )
            return True
        else:
            return False

    
    def find_alignment_steps(self, episode: Episode, index: int) -> list[int]:
        # index is either the general grasp or general place step
        if index == 0: return []        
        index, first_alignment_index = self._check_last_neighbor(episode, index)
        for wpt_index in episode.retrieve_keypoints(first_alignment_index, -1e5):
            if self._is_in_line(
                T=episode[wpt_index].T, 
                line_T1=episode[first_alignment_index].T, 
                line_T2=episode[index].T
            ):
                first_alignment_index = wpt_index
            else:
                break
        
        return_list = []
        for i in list(episode.retrieve_keypoints(first_alignment_index, index))[::-1]:
            if "is" in episode[i].verbose:
                # already be used 
                break
            else:
                episode[i].verbose.update({"is": "alignment step"})
                return_list.append(i)
        return return_list[::-1]
    
    def _gen_gaussian_noise(self, size=3, scale=1, lowerbound=0, upperbound=1e5, strict_positive=False):
        assert 0 <= lowerbound < upperbound, "lowerbound should be less than upperbound"
        noise = np.random.normal(0, scale=scale, size=size)
        while any([abs(n) < lowerbound for n in noise]) or any([abs(n) >= upperbound for n in noise]):
            noise = np.random.normal(0, scale=scale, size=size)
        if strict_positive:
            noise = np.abs(noise)
        return noise
    
    
    def _gen_uniform_noise(self, size=3, scale=1.0, lowerbound=0, upperbound=1e5, strict_positive=False):
        assert 0 <= lowerbound < upperbound, "lowerbound should be less than upperbound"
        noise = np.random.uniform(-upperbound, upperbound, size=size) * scale
        while any([abs(n) < lowerbound for n in noise]):
            noise = np.random.uniform(-upperbound, upperbound, size=size) * scale
        if strict_positive:
            noise = np.abs(noise)
        return noise

    
    def _get_traslation_noise(self, noise_type, scale, lowerbound, upperbound):
        if noise_type == 'gaussian':
            return self._gen_gaussian_noise(size=3, scale=scale, lowerbound=lowerbound, upperbound=upperbound)
        elif noise_type == 'uniform':
            return self._gen_uniform_noise(size=3, scale=scale, lowerbound=lowerbound, upperbound=upperbound)
        else:
            raise ValueError(f"Unknown noise type {noise_type}")

    def _get_rotation_noise(self, noise_type, scale, lowerbound, upperbound):
        if noise_type == 'uniform':
            # let's do uniform noise for now
            return self._gen_uniform_noise(size=1, scale=scale, lowerbound=lowerbound, upperbound=upperbound)[0]
        else:
            raise ValueError(f"Unknown noise type {noise_type}")
    
    def _get_interpolation_noise(self, noise_type, scale, lowerbound, upperbound):
        if noise_type == 'uniform':
            return self._gen_uniform_noise(size=1, scale=scale, lowerbound=lowerbound, upperbound=upperbound, strict_positive=True)[0]
        elif noise_type == 'gaussian':
            return self._gen_gaussian_noise(size=1, scale=scale, lowerbound=lowerbound, upperbound=upperbound, strict_positive=True)[0]
        else:
            raise ValueError(f"Unknown noise type {noise_type}")
        
    @check_task_bound    
    def perturb_T_in_line(self, T_src: ndarray, T_dst: ndarray, *args, **kwargs) -> ndarray:
        """
        perturb the translation along the line, the direction is from T_src to T_dst
        perturb noise that is close to T_dst, noise=0 means T_dst
        """
        noise = self._get_interpolation_noise(*args, **kwargs)
        return T_dst + noise * (T_src - T_dst)
    
    @check_task_bound
    def perturb_T_in_line_extrapolate(self, T_src: ndarray, T_dst: ndarray, *args, **kwargs) -> ndarray:
        """
        perturb the translation along the line, the direction is from T_src to T_dst
        perturb noise that is close to T_src, noise=0 means T_src
        """
        noise = self._get_interpolation_noise(*args, **kwargs)
        print(f"src {T_src} | dst {T_dst} | noise {noise} | added {noise * (T_dst - T_src)} | final {T_src + noise * (T_dst - T_src)}")
        perturbed = T_src + noise * (T_dst - T_src)
        perturbed[0] = T_src[0]
        perturbed[2] = T_dst[2]
        return perturbed
    
    @check_task_bound    
    def perturb_TR_in_line(
        self, T_src: ndarray, T_dst: ndarray, 
        R_src: quaternion.quaternion, R_dst: quaternion.quaternion,  
        *args, **kwargs) -> Tuple[ndarray, quaternion.quaternion]:
        """
        perturb the translation and rotation along the line, the direction is from src pose to dst pose
        perturb noise that is close to dst pose, noise=0 means dst pose
        """
        noise = self._get_interpolation_noise(*args, **kwargs)
        # interpolate the translation
        T_new =  T_dst + noise * (T_src - T_dst)
        # interpolate the rotation
        R_new = quaternion.slerp_evaluate(R_dst, R_src, noise)
        return T_new, R_new
        
    
    @check_task_bound
    def perturb_T_in_plane(self, T: ndarray, alignment_axis: ndarray, *args, **kwargs) -> ndarray:
        """
        given the line (i.e., rotate_axis, from src point to tgt point), perturb the point that is in the perpendicular plane of the line
        add some random noise to the point
        """
        noise = self._get_traslation_noise(*args, **kwargs)
        rotate_axis_normalized = alignment_axis / np.linalg.norm(alignment_axis)
        noise = noise - np.dot(noise, rotate_axis_normalized) * rotate_axis_normalized
        return T + noise
    
    @check_task_bound
    def perturb_T_in_space(self, T: ndarray, *args, **kwargs) -> ndarray:
        """
        perturb the translation with normal noise around the point in 3D space
        """
        noise = self._get_traslation_noise(*args, **kwargs)
        return T + noise
    
    def perturb_R_around_axis(
        self,
        R: quaternion.quaternion, 
        rotate_axis: ndarray, 
        *args, **kwargs
    ) -> quaternion.quaternion:
        """
        given the rotate axis, perturb the rotation with uniform noise around 0-180 degree
        :param R: the original rotation quaternion
        :param rotate_axis: the axis to rotate the quaternion
        """
        # for circle object, this perturbation does not help
        # for non-circle object, perturb between -60~60 degree should be useful
        angle = self._get_rotation_noise(*args, **kwargs)
        if random.random() > 0.5:
            angle = -angle
        rotate_axis_normalized = rotate_axis / np.linalg.norm(rotate_axis)
        noise = quaternion.from_rotation_vector(angle * rotate_axis_normalized)
        return noise * R