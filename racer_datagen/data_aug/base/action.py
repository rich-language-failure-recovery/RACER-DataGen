from dataclasses import dataclass, field
from numpy import ndarray
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation


@dataclass
class Action:
    translation: ndarray    # (3,)
    rotation: quaternion.quaternion       # (4,) quaternion
    gripper_open: bool      
    ignore_collision: bool
    
    @property
    def T(self):
        return self.translation
    
    @T.setter
    def T(self, value):
        self.translation = value
    
    @property
    def R(self):
        return self.rotation
    
    @property
    def Rmat(self):
        return quaternion.as_rotation_matrix(self.rotation)
    
    @R.setter
    def R(self, value):
        self.rotation = value

    def to_numpy(self) -> ndarray:
        return np.concatenate((
            self.translation,
            self.quat_to_array(self.rotation, style='xyzw'),
            np.array([self.gripper_open, self.ignore_collision], dtype=float)
        ))
    
    @classmethod
    def from_numpy(cls, arr: ndarray):
        translation = arr[:3]
        rotation = cls.array_to_quat(arr[3:7], style='xyzw')
        gripper_open = bool(arr[7])
        ignore_collision = bool(arr[8])
        return cls(translation, rotation, gripper_open, ignore_collision)
    
    @staticmethod
    def quat_to_array(quat: quaternion.quaternion, style: str = 'xyzw'):
        # style is the output arr style
        a = quaternion.as_float_array(quat)
        if style == 'xyzw':
            return np.array([a[1], a[2], a[3], a[0]])
        elif style == 'wxyz':
            return a
        else:
            raise ValueError(f"Unknown style: {style}")
    
    @staticmethod
    def array_to_quat(arr: ndarray, style: str = 'xyzw'):
        # style is the input arr style
        if style == 'xyzw':
            return quaternion.quaternion(arr[3], arr[0], arr[1], arr[2])
        elif style == 'wxyz':
            return quaternion.quaternion(arr[0], arr[1], arr[2], arr[3])
        else:
            raise ValueError(f"Unknown style: {style}")
        
    @staticmethod
    def quat_to_euler(quat: quaternion.quaternion):
        return quaternion.as_euler_angles(quat)
    
    @staticmethod
    def delta_action(action_from: 'Action', action_to: 'Action'):
        delta_translation = action_to.T - action_from.T
        delta_rotation = Rotation.from_quat(action_from.quat_to_array(action_from.R.inverse() * action_to.R, 'xyzw')).as_euler('xyz', degrees=True)
        delta_gripper = int(action_to.gripper_open) - int(action_from.gripper_open)
        delta_ignore = int(action_to.ignore_collision) - int(action_from.ignore_collision)
        return {
            "translation": delta_translation,  
            "rotation": delta_rotation, 
            "gripper": delta_gripper,   # 0 = unchanged gripper state, 1 = open gripper, -1 = close gripper
            "collision": delta_ignore   # 0 = unchanged collision state, 1 = ignore collision, -1 = consider collision
        }

    def __str__(self):
        return_str = f"T: {self.T}\t"
        return_str += f"R: {self.quat_to_array(self.R, style='xyzw')}"
        if self.gripper_open:
            return_str += " gripper open "
        else:
            return_str += " gripper close"
        if self.ignore_collision:
            return_str += ", collision ignore"
        else:
            return_str += ", collision consider"
        return return_str
    
    
    def to_interaction(self):
        # return a string like 
        # 0.27844 -0.00816 1.47198 0.0 0.99266 -0.0 0.1209 1.0 0.0 
        # for interactive play
        return_str = ""
        for i in self.translation:
            return_str += f"{i:.5f} "
        for i in self.quat_to_array(self.rotation, style='xyzw'):
            return_str += f"{i:.5f} "
        if self.gripper_open:
            return_str += "1.0 "
        else:
            return_str += "0.0 "
        if self.ignore_collision:
            return_str += "1.0 "
        else:
            return_str += "0.0 "
        return return_str


StartAction = Action(
    translation=np.array([0.27844, -0.00816, 1.47198]),
    rotation=quaternion.quaternion(0.1209,  1.47198, 0.0, 0.99266),
    gripper_open=True,
    ignore_collision=False
)

if __name__ == "__main__":

    # Define the angle in radians
    theta = np.pi / 4  # 45 degrees

    # Create the rotation matrix
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Create the quaternion
    q = quaternion.from_rotation_matrix(R_z)
    print(q)
    print(Rotation.from_quat(Action.quat_to_array(q)).as_euler('xyz', degrees=True))

    print(Action.quat_to_euler(quaternion.quaternion(-0.956, 0.295, 0.004, 0.002, )) * 180 / np.pi)
    print(Action.quat_to_euler(quaternion.quaternion(-0.976, -0.217, 0.001, 0.0, ))  * 180 / np.pi)

    action_1 = Action(
        translation=np.array([0.319, -0.033, 0.967]),
        rotation=quaternion.quaternion(-0.956, 0.295, 0.004, 0.002), # wxyz
        gripper_open=True,
        ignore_collision=False
    )
    action_1 = Action.from_numpy(
        np.array([0.319, -0.033, 0.967, -0.956, 0.295, 0.004, 0.002, 0, 0])
    )
    action_2 = Action.from_numpy(
        np.array([0.295, -0.016, 0.967, -0.976, -0.217, 0.001, 0.0, 0, 0])
    )

    # action_2 = Action(
    #     translation=np.array([0.295, -0.016, 0.967]),
    #     rotation=quaternion.quaternion(-0.976, -0.217, 0.001, 0.0),
    #     gripper_open=False,
    #     ignore_collision=False
    # )

    delta_action = Action.delta_action(action_1, action_2)
    print(delta_action)