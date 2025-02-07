
def build_prompts(notes, robot_setup, env_setup, example):
    return \
f"""Your goal is to identify subgoals of a task from the given low-level spatial instruction and provide natural language commands specific to a task given for a robot to execute. Each timestep in a given episode in JSON provides the current robot state and scene information, along with a spatial instruction aimed at achieving the next desired state. If the robot makes a mistake and fails to reach the desired state, failure reasoning and corrective instructions are provided to guide the robot back to the desired state, but only in spatial language.
{notes if notes else ""}
{robot_setup if robot_setup else ""}
{env_setup if env_setup else ""}

You should only respond in a JSON format dict as below:
{{
    "0": "natural_language_instruction_0",
    "1": "natural_language_instruction_1",
    ...
}}
{example if example else ""}
"""

def build_input_json(task_name, lang_goal, task_description, input_json):
    return \
f"""Let's start.

The given task is {task_name}. {task_description if task_description else ""} The high-level goal is {lang_goal}.
Here is the input episode in JSON:

{input_json}
"""

EXAMPLE = \
"""
Example:
The given task is meat_off_grill and the high-level goal is "take the steak off the grill".
Input JSON:
{
    "0": {
        "wpt_id": "0",
        "initial_scene_info": {
            "objects": {
                "grill": "[0.316, -0.017, 0.983]",
                "chicken": "[0.475, 0.107, 1.067]",
                "steak": "[0.372, 0.001, 1.055]"
            }
        },
        "current_timestep": {
            "status": "init",
            "instruction_in_spatial_language": [
                "move toward me along x-axis by large distance",
                "move down along z-axis by large distance",
                "keep gripper open",
                "avoid collision"
            ],
            "robot_pose": {
                "position": "[0.278, -0.008, 1.472]",
                "orientation": "[0.0, 0.993, -0.0, 0.121]",
                "gripper_open": true,
                "ignore_collision": true
            }
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.372, 0.001, 1.149]",
                "orientation": "[-0.949, 0.316, 0.001, -0.0]",
                "gripper_open": true,
                "ignore_collision": false
            }
        }
    },
    "1": {
        "wpt_id": "61",
        "current_timestep": {
            "status": "recoverable failure",
            "failure_reasoning": [
                "moved toward me along x-axis too little",
                "moved left along y-axis too much",
                "moved down along z-axis too much"
            ],
            "robot_pose": {
                "position": "[0.344, -0.028, 1.128]",
                "orientation": "[-0.949, 0.316, 0.001, -0.0]",
                "gripper_open": true,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move toward me along x-axis by small distance",
                "move right along y-axis by small distance",
                "move up along z-axis by small distance",
                "keep gripper open"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.372, 0.001, 1.149]",
                "orientation": "[-0.949, 0.316, 0.001, -0.0]",
                "gripper_open": true,
                "ignore_collision": false
            }
        }
    },
    "2": {
        "wpt_id": "61",
        "current_timestep": {
            "status": "success",
            "robot_pose": {
                "position": "[0.372, 0.001, 1.149]",
                "orientation": "[-0.949, 0.316, 0.001, -0.0]",
                "gripper_open": true,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move down along z-axis by large distance",
                "close gripper",
                "allow collision"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.372, 0.001, 1.052]",
                "orientation": "[-0.949, 0.315, 0.001, -0.0]",
                "gripper_open": false,
                "ignore_collision": true
            }
        }
    },
    "3": {
        "wpt_id": "74",
        "current_timestep": {
            "status": "recoverable failure",
            "failure_reasoning": [
                "moved down along z-axis too little"
            ],
            "robot_pose": {
                "position": "[0.372, 0.001, 1.083]",
                "orientation": "[-0.949, 0.315, 0.001, -0.0]",
                "gripper_open": false,
                "ignore_collision": true
            },
            "instruction_in_spatial_language": [
                "move up along z-axis by small distance",
                "open gripper",
                "avoid collision"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.372, 0.001, 1.103]",
                "orientation": "[-0.949, 0.316, 0.001, -0.0]",
                "gripper_open": true,
                "ignore_collision": false
            }
        }
    },
    "4": {
        "wpt_id": "74",
        "current_timestep": {
            "status": "success",
            "robot_pose": {
                "position": "[0.372, 0.001, 1.103]",
                "orientation": "[-0.949, 0.316, 0.001, -0.0]",
                "gripper_open": true,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move down along z-axis by large distance",
                "close gripper",
                "allow collision"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.372, 0.001, 1.052]",
                "orientation": "[-0.949, 0.315, 0.001, -0.0]",
                "gripper_open": false,
                "ignore_collision": true
            }
        }
    },
    "5": {
        "wpt_id": "74",
        "current_timestep": {
            "status": "success",
            "robot_pose": {
                "position": "[0.372, 0.001, 1.052]",
                "orientation": "[-0.949, 0.315, 0.001, -0.0]",
                "gripper_open": false,
                "ignore_collision": true
            },
            "instruction_in_spatial_language": [
                "move up along z-axis by large distance",
                "keep gripper closed"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.372, 0.001, 1.148]",
                "orientation": "[-0.949, 0.315, 0.001, -0.0]",
                "gripper_open": false,
                "ignore_collision": true
            }
        }
    },
    "6": {
        "wpt_id": "86",
        "current_timestep": {
            "status": "success",
            "robot_pose": {
                "position": "[0.372, 0.001, 1.148]",
                "orientation": "[-0.949, 0.315, 0.001, -0.0]",
                "gripper_open": false,
                "ignore_collision": true
            },
            "instruction_in_spatial_language": [
                "move away from me along x-axis by large distance",
                "move left along y-axis by large distance",
                "move up along z-axis by large distance",
                "keep gripper closed",
                "avoid collision"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.232, -0.171, 1.203]",
                "orientation": "[-0.949, 0.315, 0.001, 0.0]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "moved_object": [
                "steak: [0.372, 0.001, 1.147]"
            ]
        }
    },
    "7": {
        "wpt_id": "106",
        "updated_scene_info": {
            "objects": [
                "steak: [0.372, 0.001, 1.147]"
            ]
        },
        "current_timestep": {
            "status": "recoverable failure",
            "failure_reasoning": [
                "moved away from me along x-axis too little",
                "moved left along y-axis too little",
                "moved up along z-axis too much"
            ],
            "robot_pose": {
                "position": "[0.244, -0.153, 1.218]",
                "orientation": "[-0.949, 0.315, 0.001, 0.0]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move away from me along x-axis by small distance",
                "move left along y-axis by small distance",
                "move down along z-axis by small distance",
                "keep gripper closed"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.232, -0.171, 1.203]",
                "orientation": "[-0.949, 0.315, 0.001, 0.0]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "moved_object": [
                "steak: [0.245, -0.152, 1.219]"
            ]
        }
    },
    "8": {
        "wpt_id": "106",
        "updated_scene_info": {
            "objects": [
                "steak: [0.245, -0.152, 1.219]"
            ]
        },
        "current_timestep": {
            "status": "success",
            "robot_pose": {
                "position": "[0.232, -0.171, 1.203]",
                "orientation": "[-0.949, 0.315, 0.001, 0.0]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move down along z-axis by large distance",
                "open gripper",
                "allow collision"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.231, -0.172, 1.081]",
                "orientation": "[-0.949, 0.315, 0.001, -0.0]",
                "gripper_open": true,
                "ignore_collision": true
            },
            "moved_object": [
                "steak: [0.233, -0.171, 1.206]"
            ]
        }
    },
    "9": {
        "wpt_id": "122",
        "updated_scene_info": {
            "objects": [
                "steak: [0.233, -0.171, 1.206]"
            ]
        },
        "current_timestep": {
            "status": "recoverable failure",
            "failure_reasoning": [
                "moved down along z-axis too little"
            ],
            "robot_pose": {
                "position": "[0.232, -0.172, 1.127]",
                "orientation": "[-0.949, 0.315, 0.001, -0.0]",
                "gripper_open": false,
                "ignore_collision": true
            },
            "instruction_in_spatial_language": [
                "move down along z-axis by small distance",
                "open gripper"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.231, -0.172, 1.081]",
                "orientation": "[-0.949, 0.315, 0.001, -0.0]",
                "gripper_open": true,
                "ignore_collision": true
            },
            "moved_object": [
                "steak: [0.231, -0.172, 1.13]"
            ]
        }
    },
    "10": {
        "wpt_id": "122",
        "updated_scene_info": {
            "objects": [
                "steak: [0.231, -0.172, 1.13]"
            ]
        },
        "current_timestep": {
            "status": "end of episode"
        }
    }
}

Example Response:
{
    "0": "Move the gripper right above the steak on the grill while avoiding collision.",
    "1": "You moved too much to the left and down. Align the gripper slightly to the right, up, and towards you.",
    "2": "Move the gripper down, close the gripper, and allow collision.",
    "3": "You moved too little down. Move the gripper back up slightly and open the gripper.",
    "4": "Move the gripper down and retry grasping the steak.",
    "5": "Move the gripper up and keep the gripper closed.",
    "6": "Move away from the grill to place the steak somewhere else.",
    "7": "You moved too little away from the grill and left. Move the gripper slightly away, left, and down.",
    "8": "Now move the gripper down, open the gripper, and allow collision.",
    "9": "You moved too little down. Move the gripper down a bit more and open the gripper to place the steak.",
    "10": "End of episode."
}
"""


ROBOT_SETUP = \
"""ROBOT SETUP
The robot arm is mounted on a tabletop, where objects specific to the task is placed. The robot can plan to its predicted pose, open or close its gripper, and avoid or allow collisions with the objects in the scene by path planning. 
The robot actions are specified as the pose of the gripper and contains the following:
1. position: (x, y, z)
2. orientation (quaternion): (x, y, z, w)
3. gripper_open: True or False
4. ignore_collision: True (allow collision) or False (avoid collision)
"""

# In LLaVA's perspective not robot's
ENV_SETUP = \
"""The 3D coordinate system or the world frame is as follows:
The x-axis is in the depth direction, increasing from forward to backward.
The y-axis is in the horizontal direction, increasing from left to right.
The z-axis is in the vertical direction, increasing upwards.
"""

NOTES_SHORT = \
"""Notes
1. Avoid repeating the spatial language in the given JSON and replace it with a subgoal description unless it is a failure recovery step.
2. Spatial language include overly using x-axis, y-axis, z-axis, and large/small distances. Capture the task semantics and subgoals instead.
3. Try to resolve ambiguities in the scene regarding same objects.
4. Include details about the gripper state or collision avoidance where necessary or relevant.
5. When correcting a recoverable failure, always explain why the previous action failed and how to correct it.
6. If the robot fails to align explain how it failed and how to align correctly, and if the robot fails to grasp an object explain why it failed and how to retry and grasp the object again.
7. You may also use spatial language when correcting a failure.
8. You may include the last instruction saying goal reached or end of episode.
9. For long horizon tasks with repetitive subtasks, such as picking up multiple objects of the same type, keep track of which subtask the robot is currently performing.
10. Be affirmative about failure reasoning and corrective commands. Avoid using if's or may's.
"""

NOTES_DETAILED = \
"""Notes
1. Pay attention to the changes between the current and desired robot states and how this relates to the current scene information such as object locations in the context of the task.
2. The instructions in the given episode lack semantic and contextual information, for both failure reasoning and instructions, requiring you to infer the intended actions based on the spatial language provided.
3. The spatial languages are generated based on heuristic rules which may contain ambiguities that you need to resolve.
4. You do not have to always mention details about the gripper state or collision avoidance if they are not relevant to that timestep. Only include them when you think it is relevant.
5. Avoid repeating the spatial language in the JSON.
6. When correcting a failure, always explain why the previous action failed and how to correct it in the next timestep.
7. Try to keep the instructions compact, concise, and clear, and avoid unnecessary details.
8. Use conversational language to guide the robot through the task, as if you were giving instructions to a person.
9. Unless the transition specifies that an object moved, it is likely that the robot is not grasping anything.
10. You may include the last instruction saying goal reached or end of episode.
"""