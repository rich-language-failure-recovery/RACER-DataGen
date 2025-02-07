prompt = \
    """
    Your role is to semantically translate low-level spatial actions of a robot arm into natural language instructions to command the robot or correct the robot from failure.
    The task name is {episode.task_name} and the high-level language instruction is {episode.lang_goal}.
    The task scene contains these objects and their initial positions are provided:
        lime jar0: [ 0.4121 -0.077   0.8034]
        blue jar1: [0.2722 0.1166 0.8034]
        black jar_lid0: [0.1757 0.3461 0.7605]
        initial robot pose (translation & rotation): [ 0.2785 -0.0082  1.472  -0.      0.9927 -0.      0.1209]

    The robot actions are specified as the pose of the end-effector or the gripper of the robot arm. It consists of the translation (x, y, z), rotation in quaternion (x, y, z, w), the gripper state (open/close), and whether to use motion planning to avoid collisions.

    At each timestep, the robot is given a fine-grained command based on spatial language on how to move its end-effector. 
    If the robot fails, in the next timestep, you are to explain why it failed and how to correct it as defined by the spatial language.

    The robot executes the command and the scene is updated. If the robot fails to execute the command, you are to provide a correction to the robot to recover from the failure. The robot will then execute the correction and the scene is updated again.

    Spatial language-based instruction for the current timestep:
    move toward me along x-axis by large distance
    move right along y-axis by large distance
    move down along z-axis by large distance
    rotate gripper about y-axis
    keep gripper open
    avoid collision
    
    """

prompt = \
    """
    Your role is to semantically translate low-level spatial actions of a robot arm into natural language instructions to command the robot or correct the robot from failure in the context of the task.
    The task name is "close_jar" and the high-level language instruction is "close the lime jar".
    The task scene contains these objects and their initial positions of the objects and the initial pose of the robot are provided:


    The robot actions are specified as the pose of the end-effector or the gripper of the robot arm. It consists of the translation (x, y, z), rotation in quaternion (x, y, z, w), the gripper state (open/close), and whether to use motion planning to avoid collisions.

    At each timestep, the robot is given a fine-grained command based on spatial language on how to move its end-effector. 
    If the robot fails, in the next timestep, you are to explain why it failed and how to correct it as defined by the spatial language.

    The robot executes the command and the scene is updated. If the robot fails to execute the command, you are to provide a correction to the robot to recover from the failure. The robot will then execute the correction and the scene is updated again.


    Example.
    Initial scene information:
        lime jar0: [ 0.4121 -0.077   0.8034]
        blue jar1: [0.2722 0.1166 0.8034]
        black jar_lid0: [0.1757 0.3461 0.7605]
        initial robot pose (translation & rotation): [ 0.2785 -0.0082  1.472  -0.      0.9927 -0.      0.1209]

    Spatial language-based instruction for the next timestep at the current timestep:
        move toward me along x-axis by large distance
        move right along y-axis by large distance
        move down along z-axis by large distance
        rotate gripper about y-axis
        keep gripper open
        avoid collision

    Robot pose at the next timestep: [0.1493 0.4068 0.9287 -0.2508  0.968  -0.0002 -0.0006]

    {
        natural_language: "approach and hover above the lime jar to align the gripper"
        reasoning: "the robot pose at the next timestep is now closer to the black jar_lid0"
    }
    

    Provide the semantically translated natural language instruction for the robot action for the next timestep in the following format:
    {
        natural_language: ""
    }
    """

prompt = \
    """
    Your role is to semantically translate low-level spatial actions of a robot arm into natural language instructions to command the robot or correct the robot from failure in the context of the task.
    The task name is "close_jar" and the high-level language instruction is "close the lime jar".
    The task scene contains these objects and their initial positions of the objects and the initial pose of the robot are provided:

    The robot actions are specified as the pose of the end-effector or the gripper of the robot arm. It consists of the translation (x, y, z), rotation in quaternion (x, y, z, w), the gripper state (open/close), and whether to use motion planning to avoid collisions.

    At each timestep, the robot is given a fine-grained command based on spatial language on how to move its end-effector. 
    If the robot fails, in the next timestep, you are to explain why it failed and how to correct it as defined by the spatial language.

    The robot executes the command and the scene is updated. If the robot fails to execute the command, you are to provide a correction to the robot to recover from the failure. The robot will then execute the correction and the scene is updated again.
    Semantic annotation involves rephrasing low-level, detailed robotic commands into conversational language that encapsulates the core actions and intentions in a way that feels natural for human understanding. 
    This should bridge the gap between precise robotic control commands and the kind of instructions you might naturally give to another person.

    Consider a scenario where a robot is instructed to close a jar. 
    The robot receives this command through detailed instructions like moving along specific axes, rotating certain components, and managing the gripper. These instructions are technical:
    Move toward me along x-axis by large distance.
    Rotate gripper about y-axis.
    Keep gripper open.
    In semantic annotation, these instructions would be translated to:

    "Move forward and right to position yourself above the lime jar, lower the gripper, rotate it for alignment, and make sure it’s open and ready to grip the lid without hitting anything."

    Example 1.
    Information at current timestep and the changes made to the next timestep:
    {
        initial_scene_info: {
            robot_pose: {
                position: [ 0.2785 -0.0082  1.472],
                orientation: [-0.      0.9927 -0.      0.1209]
            },
            objects: {
                jar0: [ 0.4121 -0.077   0.8034],
                jar1: [0.2722 0.1166 0.8034],
                jar_lid0: [0.1757 0.3461 0.7605],
            }
        },
        current_timestep: {
            status: "init",
            robot_pose: {
                position: [ 0.2785 -0.0082  1.472],
                orientation: [-0.      0.9927 -0.      0.1209]
            },
            failure_reasoning: null,
            spatial_language_instruction: [
                "move toward me along x-axis by large distance",
                "move right along y-axis by large distance",
                "move down along z-axis by large distance",
                "rotate gripper about y-axis",
                "keep gripper open",
                "avoid collision"
            ],
        },
        next_timestep: {
            robot_pose: {
                position: [0.1493 0.4068 0.9287],
                orientation: [-0.2508  0.968  -0.0002 -0.0006]
            },
            moved_object: null,
        }
    }

    Natural language instruction to be given at the current timestep:
    {
        natural_language: "approach and hover above the lime jar to align the gripper"
        reasoning: "the robot pose at the next timestep is now closer to the black jar_lid0"
    }

    Example 2.
    Information at current timestep and the changes made to the next timestep:
    {
        current_timestep: {
            initial_scene_info: {
                robot_pose: {
                    position: [ 0.2785 -0.0082  1.472],
                    orientation: [-0.      0.9927 -0.      0.1209]
                },
                objects: {
                    jar0: [ 0.4121 -0.077   0.8034],
                    jar1: [0.2722 0.1166 0.8034],
                    jar_lid0: [0.1757 0.3461 0.7605],
                }
            },
            robot_pose: {
                position: [0.1493 0.4068 0.9287],
                orientation: [-0.2508  0.968  -0.0002 -0.0006]
            },
            spatial_language_instruction: [
                "move toward me along x-axis by large distance",
                "move right along y-axis by large distance",
                "move down along z-axis by large distance",
                "rotate gripper about y-axis",
                "keep gripper open",
                "avoid collision"
            ]
        },
        next_timestep: {
            status: "failure",
            robot_pose: {
                position: [0.1493 0.4068 0.9287],
                orientation: [-0.2508  0.968  -0.0002 -0.0006]
            },
            moved_object: "jar0",
        }
    }
    
    
    Provide the semantically translated natural language instruction for the robot action for the next timestep like the example given above.

    Let's start.

    Current timestep:
    {
        current_timestep: {
            initial_scene_info: {
                robot_pose: {
                    position: [ 0.2785 -0.0082  1.472],
                    orientation: [-0.      0.9927 -0.      0.1209]
                },
                objects: {
                    jar0: [ 0.2791 -0.0699  0.8034]
                    jar1: [ 0.307  -0.2585  0.8034]
                    jar_lid0: [0.118  0.1714 0.7605]
                }
            },
            spatial_language_instruction: [
                "move toward me along x-axis by large distance",
                "move right along y-axis by large distance",
                "move down along z-axis by large distance",
                "rotate gripper about y-axis",
                "keep gripper open",
                "avoid collision"
            ]
        },
        next_timestep: {
            status: "success",
            robot_pose: {
                position: [0.0697 0.2164 0.8624],
                orientation: [-0.2979  0.9546  0.     -0.0005]
            },
            moved_object: null
        }
    }
    Natural language instruction to be given at the current timestep:
    """


PROMPT_TEMPLATE = \
"""
Your role is to semantically translate low-level spatial actions of a robot arm into natural language instructions to command the robot or correct the robot from failure in the context of the task.
    The task name is "close_jar" and the high-level language instruction is "close the lime jar".
    The task scene contains these objects and their initial positions of the objects and the initial pose of the robot are provided:

    The robot actions are specified as the pose of the end-effector or the gripper of the robot arm. It consists of the translation (x, y, z), rotation in quaternion (x, y, z, w), the gripper state (open/close), and whether to use motion planning to avoid collisions.

    At each timestep, the robot is given a fine-grained command based on spatial language on how to move its end-effector. 
    If the robot fails, in the next timestep, you are to explain why it failed and how to correct it as defined by the spatial language.

    The robot executes the command and the scene is updated. If the robot fails to execute the command, you are to provide a correction to the robot to recover from the failure. The robot will then execute the correction and the scene is updated again.
    Semantic annotation involves rephrasing low-level, detailed robotic commands into conversational language that encapsulates the core actions and intentions in a way that feels natural for human understanding. 
    This should bridge the gap between precise robotic control commands and the kind of instructions you might naturally give to another person.

    Consider a scenario where a robot is instructed to close a jar. 
    The robot receives this command through detailed instructions like moving along specific axes, rotating certain components, and managing the gripper. These instructions are technical:
    Move toward me along x-axis by large distance.
    Rotate gripper about y-axis.
    Keep gripper open.
    In semantic annotation, these instructions would be translated to:

    "Move forward and right to position yourself above the lime jar, lower the gripper, rotate it for alignment, and make sure it’s open and ready to grip the lid without hitting anything."

Task name: meat_off_grill
High-level language instruction: "take the steak off the grill"
{
    "0": {
        "initial_scene_info": {
            "robot_pose": {
                "position": "[0.278, -0.008, 1.472]",
                "orientation": "[0.0, 0.993, -0.0, 0.121]"
            },
            "objects": {
                "grill": "[0.316, -0.017, 0.983]",
                "chicken": "[0.475, 0.107, 1.067]",
                "steak": "[0.372, 0.001, 1.055]"
            }
        },
        "current_timestep": {
            "status": "init",
            "robot_pose": {
                "position": "[0.278, -0.008, 1.472]",
                "orientation": "[0.0, 0.993, -0.0, 0.121]",
                "gripper_open": true,
                "ignore_collision": true
            },
            "language_in_spatial_language": [
                "move toward me along x-axis by large distance",
                "move down along z-axis by large distance",
                "keep gripper open",
                "avoid collision"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.372, 0.001, 1.149]",
                "orientation": "[-0.949, 0.316, 0.001, -0.0]",
                "gripper_open": true,
                "ignore_collision": false
            },
            "moved_object": []
        }
    },
    "1": {
        "initial_scene_info": {
            "robot_pose": {
                "position": "[0.278, -0.008, 1.472]",
                "orientation": "[0.0, 0.993, -0.0, 0.121]"
            },
            "objects": {
                "grill": "[0.316, -0.017, 0.983]",
                "chicken": "[0.475, 0.107, 1.067]",
                "steak": "[0.372, 0.001, 1.055]"
            }
        },
        "current_timestep": {
            "status": "recoverable failure",
            "failure_reasoning": [
                "moved toward me along x-axis too little",
                "moved right along y-axis too much"
            ],
            "robot_pose": {
                "position": "[0.337, 0.026, 1.148]",
                "orientation": "[-0.949, 0.316, 0.001, -0.0]",
                "gripper_open": true,
                "ignore_collision": false
            },
            "language_in_spatial_language": [
                "move toward me along x-axis by small distance",
                "move left along y-axis by small distance",
                "keep gripper open"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.372, 0.001, 1.149]",
                "orientation": "[-0.949, 0.316, 0.001, -0.0]",
                "gripper_open": true,
                "ignore_collision": false
            },
            "moved_object": []
        }
    },
    "2": {
        "initial_scene_info": {
            "robot_pose": {
                "position": "[0.278, -0.008, 1.472]",
                "orientation": "[0.0, 0.993, -0.0, 0.121]"
            },
            "objects": {
                "grill": "[0.316, -0.017, 0.983]",
                "chicken": "[0.475, 0.107, 1.067]",
                "steak": "[0.372, 0.001, 1.055]"
            }
        },
        "current_timestep": {
            "status": "success",
            "failure_reasoning": null,
            "robot_pose": {
                "position": "[0.372, 0.001, 1.149]",
                "orientation": "[-0.949, 0.316, 0.001, -0.0]",
                "gripper_open": true,
                "ignore_collision": false
            },
            "language_in_spatial_language": [
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
            },
            "moved_object": []
        }
    },
    "3": {
        "initial_scene_info": {
            "robot_pose": {
                "position": "[0.278, -0.008, 1.472]",
                "orientation": "[0.0, 0.993, -0.0, 0.121]"
            },
            "objects": {
                "grill": "[0.316, -0.017, 0.983]",
                "chicken": "[0.475, 0.107, 1.067]",
                "steak": "[0.372, 0.001, 1.055]"
            }
        },
        "current_timestep": {
            "status": "recoverable failure",
            "failure_reasoning": [
                "moved down along z-axis too little"
            ],
            "robot_pose": {
                "position": "[0.372, 0.001, 1.075]",
                "orientation": "[-0.949, 0.315, 0.001, -0.0]",
                "gripper_open": false,
                "ignore_collision": true
            },
            "language_in_spatial_language": [
                "move up along z-axis by large distance",
                "open gripper",
                "avoid collision"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.372, 0.001, 1.13]",
                "orientation": "[-0.949, 0.316, 0.001, -0.0]",
                "gripper_open": true,
                "ignore_collision": false
            },
            "moved_object": []
        }
    },
    "4": {
        "initial_scene_info": {
            "robot_pose": {
                "position": "[0.278, -0.008, 1.472]",
                "orientation": "[0.0, 0.993, -0.0, 0.121]"
            },
            "objects": {
                "grill": "[0.316, -0.017, 0.983]",
                "chicken": "[0.475, 0.107, 1.067]",
                "steak": "[0.372, 0.001, 1.055]"
            }
        },
        "current_timestep": {
            "status": "success",
            "failure_reasoning": null,
            "robot_pose": {
                "position": "[0.372, 0.001, 1.13]",
                "orientation": "[-0.949, 0.316, 0.001, -0.0]",
                "gripper_open": true,
                "ignore_collision": false
            },
            "language_in_spatial_language": [
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
            },
            "moved_object": []
        }
    },
    "5": {
        "initial_scene_info": {
            "robot_pose": {
                "position": "[0.278, -0.008, 1.472]",
                "orientation": "[0.0, 0.993, -0.0, 0.121]"
            },
            "objects": {
                "grill": "[0.316, -0.017, 0.983]",
                "chicken": "[0.475, 0.107, 1.067]",
                "steak": "[0.372, 0.0, 1.055]"
            }
        },
        "current_timestep": {
            "status": "success",
            "failure_reasoning": null,
            "robot_pose": {
                "position": "[0.372, 0.001, 1.052]",
                "orientation": "[-0.949, 0.315, 0.001, -0.0]",
                "gripper_open": false,
                "ignore_collision": true
            },
            "language_in_spatial_language": [
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
            },
            "moved_object": []
        }
    },
    "6": {
        "initial_scene_info": {
            "robot_pose": {
                "position": "[0.278, -0.008, 1.472]",
                "orientation": "[0.0, 0.993, -0.0, 0.121]"
            },
            "objects": {
                "grill": "[0.316, -0.017, 0.983]",
                "chicken": "[0.475, 0.107, 1.067]",
                "steak": "[0.372, 0.001, 1.147]"
            }
        },
        "current_timestep": {
            "status": "success",
            "failure_reasoning": null,
            "robot_pose": {
                "position": "[0.372, 0.001, 1.148]",
                "orientation": "[-0.949, 0.315, 0.001, -0.0]",
                "gripper_open": false,
                "ignore_collision": true
            },
            "language_in_spatial_language": [
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
        "initial_scene_info": {
            "robot_pose": {
                "position": "[0.278, -0.008, 1.472]",
                "orientation": "[0.0, 0.993, -0.0, 0.121]"
            },
            "objects": {
                "grill": "[0.316, -0.017, 0.983]",
                "chicken": "[0.475, 0.107, 1.067]",
                "steak": "[0.147, -0.157, 1.216]"
            }
        },
        "current_timestep": {
            "status": "recoverable failure",
            "failure_reasoning": [
                "moved away from me along x-axis too much",
                "moved left along y-axis too little",
                "moved up along z-axis too much"
            ],
            "robot_pose": {
                "position": "[0.147, -0.157, 1.214]",
                "orientation": "[-0.996, -0.087, 0.004, 0.001]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "language_in_spatial_language": [
                "move toward me along x-axis by large distance",
                "move left along y-axis by small distance",
                "move down along z-axis by small distance",
                "rotate gripper about z-axis",
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
                "steak: [0.147, -0.157, 1.216]"
            ]
        }
    },
    "8": {
        "initial_scene_info": {
            "robot_pose": {
                "position": "[0.278, -0.008, 1.472]",
                "orientation": "[0.0, 0.993, -0.0, 0.121]"
            },
            "objects": {
                "grill": "[0.316, -0.017, 0.983]",
                "chicken": "[0.475, 0.107, 1.067]",
                "steak": "[0.231, -0.172, 1.205]"
            }
        },
        "current_timestep": {
            "status": "success",
            "failure_reasoning": null,
            "robot_pose": {
                "position": "[0.232, -0.171, 1.203]",
                "orientation": "[-0.949, 0.315, 0.001, 0.0]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "language_in_spatial_language": [
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
                "steak: [0.231, -0.172, 1.205]"
            ]
        }
    },
    "9": {
        "initial_scene_info": {
            "robot_pose": {
                "position": "[0.278, -0.008, 1.472]",
                "orientation": "[0.0, 0.993, -0.0, 0.121]"
            },
            "objects": {
                "grill": "[0.316, -0.017, 0.983]",
                "chicken": "[0.475, 0.107, 1.067]",
                "steak": "[0.231, -0.171, 1.179]"
            }
        },
        "current_timestep": {
            "status": "recoverable failure",
            "failure_reasoning": [
                "moved down along z-axis too little"
            ],
            "robot_pose": {
                "position": "[0.232, -0.171, 1.176]",
                "orientation": "[-0.949, 0.315, 0.001, -0.0]",
                "gripper_open": false,
                "ignore_collision": true
            },
            "language_in_spatial_language": [
                "move down along z-axis by large distance",
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
                "steak: [0.231, -0.171, 1.179]"
            ]
        }
    },
    "10": {
        "initial_scene_info": {
            "robot_pose": {
                "position": "[0.278, -0.008, 1.472]",
                "orientation": "[0.0, 0.993, -0.0, 0.121]"
            },
            "objects": {
                "grill": "[0.316, -0.017, 0.983]",
                "chicken": "[0.475, 0.107, 1.067]",
                "steak": "[0.227, -0.173, 1.074]"
            }
        },
        "current_timestep": {
            "status": "success",
            "failure_reasoning": null,
            "language_in_spatial_language": [
                "end of episode"
            ]
        }
    }
}

Provide the semantic annotation for each timestep in a list or a dict.
"""