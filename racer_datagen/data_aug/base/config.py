# A general parameters for all tasks
# translation_noise is for perturb_T_in_plane, perturb_T_in_space
# interpolation_noise is for perturb_T_in_line
# rotation_noise is for perturb_R_around_axis
# all noise has a lowerbound and upperbound, the sample should be in the range
# not too close and not too far

import numpy as np
from yacs.config import CfgNode as CN

_C = CN()

_C.is_same_T_thres = 0.03
_C.is_same_R_thres = 0.03
_C.is_in_line_thres = 0.03


_C.general_grasp = CN()
_C.general_grasp.use_interpolation_noise = True
_C.general_grasp.use_translation_noise = True
_C.general_grasp.interpolation_noise = CN() 
_C.general_grasp.interpolation_noise.noise_type = "uniform"
_C.general_grasp.interpolation_noise.scale = 1.0
_C.general_grasp.interpolation_noise.lowerbound = 0.2
_C.general_grasp.interpolation_noise.upperbound = 0.5
_C.general_grasp.translation_noise = CN() 
_C.general_grasp.translation_noise.noise_type = "gaussian"
_C.general_grasp.translation_noise.scale = 0.33
_C.general_grasp.translation_noise.lowerbound = 0.03
_C.general_grasp.translation_noise.upperbound = 0.05


_C.coarse_alignment = CN()
_C.coarse_alignment.use_translation_noise = True
_C.coarse_alignment.use_rotation_noise = True
_C.coarse_alignment.translation_noise = CN() 
_C.coarse_alignment.translation_noise.noise_type = "gaussian"
_C.coarse_alignment.translation_noise.scale = 0.033
_C.coarse_alignment.translation_noise.lowerbound = 0.01
_C.coarse_alignment.translation_noise.upperbound = 0.1
_C.coarse_alignment.rotation_noise = CN() 
_C.coarse_alignment.rotation_noise.noise_type = "uniform"
_C.coarse_alignment.rotation_noise.scale = 1.0
_C.coarse_alignment.rotation_noise.lowerbound = np.pi/10
_C.coarse_alignment.rotation_noise.upperbound = np.pi/3


_C.fine_alignment = CN()
_C.fine_alignment.use_translation_noise = True
_C.fine_alignment.use_rotation_noise = True
_C.fine_alignment.translation_noise = CN() 
_C.fine_alignment.translation_noise.noise_type = "gaussian"
_C.fine_alignment.translation_noise.scale = 0.02
_C.fine_alignment.translation_noise.lowerbound = 0.01
_C.fine_alignment.translation_noise.upperbound = 0.06
_C.fine_alignment.rotation_noise = CN() 
_C.fine_alignment.rotation_noise.noise_type = "uniform"
_C.fine_alignment.rotation_noise.scale = 1.0
_C.fine_alignment.rotation_noise.lowerbound = np.pi/10
_C.fine_alignment.rotation_noise.upperbound = np.pi/3



_C.general_place = CN()
_C.general_place.use_interpolation_noise = True
_C.general_place.interpolation_noise = CN() 
_C.general_place.interpolation_noise.noise_type = "uniform"
_C.general_place.interpolation_noise.scale = 1.0
_C.general_place.interpolation_noise.lowerbound = 0.2
_C.general_place.interpolation_noise.upperbound = 0.8

_C.intermediate = CN()
_C.intermediate.lowerbound = 2/4
_C.intermediate.upperbound = 3/4

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()
