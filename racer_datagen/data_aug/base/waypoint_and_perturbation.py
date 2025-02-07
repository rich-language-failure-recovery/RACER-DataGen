from dataclasses import field, dataclass
from typing import Any, Dict, List, Optional
from .action import Action

from racer_datagen.online_rollout.base.constants import *

@dataclass
class WayPoint:
    """
    :param id: unique id from original demo
    :param type: ['intermediate', 'keypoint', 'correction', 'heuristic', 'model']
    :param action: includes the action to take to get to the waypoint
    type = 'intermediate' means the non-keypoint waypoints from the original demo
    type = 'keypoint' means the keypoint waypoints extracted from the original demo
    type = 'correction' means the corrections for perturbed waypoints (usually from intermediate waypoints)
    type = 'heuristic' means the perturbed waypoints through heuristic rules
    type = 'model' means the pertubed waypoints through model prediction
    
    All the correction, perturb, predict waypoints should have the same id as the keypoint waypoints
    """
    id: int      
    type: str
    action: Action
    perturbations: List["Perturbation"] = field(default_factory=list) # this is only for the keypoint type
    dense: List["WayPoint"] = field(default_factory=list) # list for extra dense waypoints
    catastrohpies: List["WayPoint"] = field(default_factory=list) # only for model perturbations
    verbose: Dict[Any, Any] = field(default_factory=dict) # this is for debugging or other purposes
    info: Dict[Any, Any] = field(default_factory=dict)
    
    def __len__(self):
        return len(self.perturbations)

    @property
    def T(self):
        return self.action.translation
    
    @T.setter
    def T(self, value):
        self.action.translation = value
    
    @property
    def R(self):
        return self.action.rotation
    
    @R.setter
    def R(self, value):
        self.action.rotation = value
    
    @property
    def Rmat(self):
        return self.action.Rmat
    
    @property
    def gripper_open(self):
        return self.action.gripper_open
    
    @gripper_open.setter
    def gripper_open(self, value):
        self.action.gripper_open = value
    
    @property
    def gripper_close(self):
        return not self.action.gripper_open
    
    @property
    def ignore_collision(self):
        return self.action.ignore_collision
    
    @ignore_collision.setter
    def ignore_collision(self, value):
        self.action.ignore_collision = value
    
    @property
    def consider_collision(self):
        return not self.action.ignore_collision
    
    
    def add_perturbation(self, mistake_action: Action, correction_action: Optional[Action] = None, perturb_type: str = 'heuristic'):
        # if correction_action is None, that means the correction is the keypoint itself
        assert perturb_type in PERTURB_TYPE_LIST, "The type should be either heuristic or model"
        mistake_waypoint = WayPoint(self.id, perturb_type, mistake_action, info={WptInfoKey.PERTURB_TYPE: perturb_type})
        if correction_action is not None:
            correction_waypoint = WayPoint(self.id, 'correction', correction_action, info={WptInfoKey.PERTURB_TYPE: perturb_type})
        else:
            correction_waypoint = None
        self.perturbations.append(Perturbation(mistake_waypoint, correction_waypoint))

@dataclass
class Perturbation:
    mistake: WayPoint
    correction: Optional[WayPoint] # is None if the correction is the keypoint itself
    
        
    