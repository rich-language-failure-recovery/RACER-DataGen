

INSERT_ONTO_SQUARE_PEG_EXAMPLE = \
"""\
Here's an example.
INPUT:
{
    "0": {
        "initial_scene_info": {
            "objects": {
                "square_base": "[0.183, -0.069, 0.762]",
                "pillar0": "[0.296, -0.016, 0.832]",
                "pillar1": "[0.183, -0.069, 0.832]",
                "pillar2": "[0.07, -0.123, 0.832]",
                "square_ring": "[0.154, -0.279, 0.755]"
            }
        },
        "current_timestep": {
            "status": "init",
            "instruction_in_spatial_language": [
                "move forward along x-axis by large distance",
                "move left along y-axis by large distance",
                "move down along z-axis by large distance",
                "rotate gripper about y-axis",
                "keep gripper open",
                "avoid collision"
            ],
            "robot_pose": {
                "position": "[0.278, -0.008, 1.472]",
                "orientation": "[-0.0, 0.993, -0.0, 0.121]",
                "gripper_open": true,
                "ignore_collision": true
            }
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.154, -0.279, 0.855]",
                "orientation": "[-0.045, 0.999, 0.0, -0.0]",
                "gripper_open": true,
                "ignore_collision": false
            }
        }
    },
    "1": {
        "current_timestep": {
            "status": "recoverable failure",
            "failure_reasoning": "failed to align, ",
            "robot_pose": {
                "position": "[0.154, -0.279, 0.855]",
                "orientation": "[0.375, 0.927, 0.001, 0.001]",
                "gripper_open": true,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "keep gripper open"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.154, -0.279, 0.855]",
                "orientation": "[-0.045, 0.999, 0.0, -0.0]",
                "gripper_open": true,
                "ignore_collision": false
            }
        }
    },
    "2": {
        "current_timestep": {
            "status": "success",
            "robot_pose": {
                "position": "[0.154, -0.279, 0.855]",
                "orientation": "[-0.045, 0.999, 0.0, -0.0]",
                "gripper_open": true,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move down along z-axis by large distance",
                "close gripper"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.154, -0.279, 0.754]",
                "orientation": "[-0.045, 0.999, 0.0, -0.001]",
                "gripper_open": false,
                "ignore_collision": false
            }
        }
    },
    "3": {
        "current_timestep": {
            "status": "recoverable failure",
            "failure_reasoning": "failed to grasp, moved forward along x-axis too much, moved right along y-axis too much, moved down along z-axis too little",
            "robot_pose": {
                "position": "[0.114, -0.246, 0.793]",
                "orientation": "[-0.045, 0.999, 0.0, -0.001]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move backward along x-axis a bit more",
                "move left along y-axis a bit more",
                "move up along z-axis a bit more",
                "open gripper"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.154, -0.279, 0.819]",
                "orientation": "[-0.045, 0.999, 0.001, -0.001]",
                "gripper_open": true,
                "ignore_collision": false
            }
        }
    },
    "4": {
        "current_timestep": {
            "status": "success",
            "robot_pose": {
                "position": "[0.154, -0.279, 0.819]",
                "orientation": "[-0.045, 0.999, 0.001, -0.001]",
                "gripper_open": true,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move down along z-axis by large distance",
                "close gripper"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.154, -0.279, 0.754]",
                "orientation": "[-0.045, 0.999, 0.0, -0.001]",
                "gripper_open": false,
                "ignore_collision": false
            }
        }
    },
    "5": {
        "current_timestep": {
            "status": "success",
            "robot_pose": {
                "position": "[0.154, -0.279, 0.754]",
                "orientation": "[-0.045, 0.999, 0.0, -0.001]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move up along z-axis by large distance",
                "keep gripper closed",
                "allow collision"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.154, -0.279, 0.964]",
                "orientation": "[-0.045, 0.999, 0.0, -0.001]",
                "gripper_open": false,
                "ignore_collision": true
            }
        }
    },
    "6": {
        "current_timestep": {
            "status": "success",
            "robot_pose": {
                "position": "[0.154, -0.279, 0.964]",
                "orientation": "[-0.045, 0.999, 0.0, -0.001]",
                "gripper_open": false,
                "ignore_collision": true
            },
            "instruction_in_spatial_language": [
                "move backward along x-axis by large distance",
                "move right along y-axis by large distance",
                "rotate gripper about z-axis",
                "keep gripper closed"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.295, -0.016, 0.967]",
                "orientation": "[-0.976, -0.217, 0.001, 0.0]",
                "gripper_open": false,
                "ignore_collision": true
            },
            "moved_object": [
                "square_ring: [0.154, -0.278, 0.962]"
            ]
        }
    },
    "7": {
        "updated_scene_info": {
            "objects": [
                "square_ring: [0.154, -0.278, 0.962]"
            ]
        },
        "current_timestep": {
            "status": "recoverable failure",
            "failure_reasoning": "failed to align, moved backward along x-axis too much, moved right along y-axis too little",
            "robot_pose": {
                "position": "[0.319, -0.033, 0.967]",
                "orientation": "[-0.956, 0.295, 0.004, 0.002]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move forward along x-axis a bit more",
                "move right along y-axis a bit more",
                "keep gripper closed",
                "allow collision"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.295, -0.016, 0.967]",
                "orientation": "[-0.976, -0.217, 0.001, 0.0]",
                "gripper_open": false,
                "ignore_collision": true
            },
            "moved_object": [
                "square_ring: [0.318, -0.033, 0.967]"
            ]
        }
    },
    "8": {
        "updated_scene_info": {
            "objects": [
                "square_ring: [0.318, -0.033, 0.967]"
            ]
        },
        "current_timestep": {
            "status": "success",
            "robot_pose": {
                "position": "[0.295, -0.016, 0.967]",
                "orientation": "[-0.976, -0.217, 0.001, 0.0]",
                "gripper_open": false,
                "ignore_collision": true
            },
            "instruction_in_spatial_language": [
                "move down along z-axis by large distance",
                "open gripper"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.296, -0.015, 0.88]",
                "orientation": "[-0.976, -0.22, 0.001, 0.0]",
                "gripper_open": true,
                "ignore_collision": true
            },
            "moved_object": [
                "square_ring: [0.295, -0.017, 0.967]"
            ]
        }
    },
    "9": {
        "updated_scene_info": {
            "objects": [
                "square_ring: [0.295, -0.017, 0.967]"
            ]
        },
        "current_timestep": {
            "status": "recoverable failure",
            "failure_reasoning": "failed to place, moved down along z-axis too little",
            "robot_pose": {
                "position": "[0.296, -0.016, 0.912]",
                "orientation": "[-0.976, -0.219, 0.001, 0.0]",
                "gripper_open": false,
                "ignore_collision": true
            },
            "instruction_in_spatial_language": [
                "move down along z-axis a bit more",
                "open gripper"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.296, -0.015, 0.88]",
                "orientation": "[-0.976, -0.22, 0.001, 0.0]",
                "gripper_open": true,
                "ignore_collision": true
            },
            "moved_object": [
                "square_ring: [0.295, -0.016, 0.912]"
            ]
        }
    },
    "10": {
        "updated_scene_info": {
            "objects": [
                "square_ring: [0.295, -0.016, 0.912]"
            ]
        },
        "current_timestep": {
            "status": "end of episode",
            "instruction_in_spatial_language": [
                "goal reached"
            ]
        }
    }
}

"""

REACH_AND_DRAG_EXAMPLE = \
"""\
Here's an example.
INPUT:
{
    "0": {
        "initial_scene_info": {
            "objects": {
                "cube": "[0.141, -0.093, 0.79]",
                "target0": "[0.384, -0.083, 0.751]",
                "distractor1": "[0.167, -0.365, 0.751]",
                "distractor2": "[0.153, 0.135, 0.751]",
                "distractor3": "[-0.006, -0.095, 0.751]",
                "stick": "[0.107, -0.224, 0.809]"
            }
        },
        "current_timestep": {
            "status": "init",
            "instruction_in_spatial_language": [
                "move forward along x-axis by large distance",
                "move left along y-axis by large distance",
                "move down along z-axis by large distance",
                "rotate gripper about x-axis",
                "rotate gripper about z-axis",
                "keep gripper open"
            ],
            "robot_pose": {
                "position": "[0.278, -0.008, 1.472]",
                "orientation": "[-0.0, 0.993, -0.0, 0.121]",
                "gripper_open": true,
                "ignore_collision": false
            }
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.152, -0.085, 0.93]",
                "orientation": "[-0.81, -0.586, 0.0, 0.0]",
                "gripper_open": true,
                "ignore_collision": false
            }
        }
    },
    "1": {
        "current_timestep": {
            "status": "recoverable failure",
            "failure_reasoning": "failed to align, moved forward along x-axis too little, moved left along y-axis too little",
            "robot_pose": {
                "position": "[0.224, -0.072, 0.93]",
                "orientation": "[-0.91, -0.414, 0.0, 0.0]",
                "gripper_open": true,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move forward along x-axis by large distance",
                "move left along y-axis a bit more",
                "keep gripper open"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.152, -0.085, 0.93]",
                "orientation": "[-0.81, -0.586, 0.0, 0.0]",
                "gripper_open": true,
                "ignore_collision": false
            }
        }
    },
    "2": {
        "current_timestep": {
            "status": "success",
            "robot_pose": {
                "position": "[0.152, -0.085, 0.93]",
                "orientation": "[-0.81, -0.586, 0.0, 0.0]",
                "gripper_open": true,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move down along z-axis by large distance",
                "close gripper"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.152, -0.085, 0.854]",
                "orientation": "[-0.811, -0.585, 0.0, 0.0]",
                "gripper_open": false,
                "ignore_collision": false
            }
        }
    },
    "3": {
        "current_timestep": {
            "status": "recoverable failure",
            "failure_reasoning": "failed to grasp, moved forward along x-axis too much, moved left along y-axis too much",
            "robot_pose": {
                "position": "[0.008, -0.262, 0.857]",
                "orientation": "[-0.811, -0.585, 0.0, 0.0]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move backward along x-axis by large distance",
                "move right along y-axis by large distance",
                "move up along z-axis a bit more",
                "open gripper"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.152, -0.085, 0.904]",
                "orientation": "[-0.811, -0.585, 0.001, 0.001]",
                "gripper_open": true,
                "ignore_collision": false
            }
        }
    },
    "4": {
        "current_timestep": {
            "status": "success",
            "robot_pose": {
                "position": "[0.152, -0.085, 0.904]",
                "orientation": "[-0.811, -0.585, 0.001, 0.001]",
                "gripper_open": true,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move down along z-axis by large distance",
                "close gripper"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.152, -0.085, 0.854]",
                "orientation": "[-0.811, -0.585, 0.0, 0.0]",
                "gripper_open": false,
                "ignore_collision": false
            }
        }
    },
    "5": {
        "current_timestep": {
            "status": "success",
            "robot_pose": {
                "position": "[0.152, -0.085, 0.854]",
                "orientation": "[-0.811, -0.585, 0.0, 0.0]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move up along z-axis by large distance",
                "keep gripper closed",
                "allow collision"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.152, -0.085, 0.929]",
                "orientation": "[-0.811, -0.585, 0.0, 0.0]",
                "gripper_open": false,
                "ignore_collision": true
            }
        }
    },
    "6": {
        "current_timestep": {
            "status": "success",
            "robot_pose": {
                "position": "[0.152, -0.085, 0.929]",
                "orientation": "[-0.811, -0.585, 0.0, 0.0]",
                "gripper_open": false,
                "ignore_collision": true
            },
            "instruction_in_spatial_language": [
                "move forward along x-axis by large distance",
                "move right along y-axis by large distance",
                "move up along z-axis by large distance",
                "keep gripper closed",
                "avoid collision"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.065, 0.107, 0.991]",
                "orientation": "[0.701, 0.713, -0.0, -0.0]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "moved_object": [
                "stick: [0.108, -0.225, 0.883]"
            ]
        }
    },
    "7": {
        "updated_scene_info": {
            "objects": [
                "stick: [0.108, -0.225, 0.883]"
            ]
        },
        "current_timestep": {
            "status": "recoverable failure",
            "failure_reasoning": "failed to align, failed to align, moved forward along x-axis too much, moved right along y-axis too much, moved up along z-axis too much",
            "robot_pose": {
                "position": "[-0.011, 0.129, 1.035]",
                "orientation": "[0.701, 0.713, -0.0, -0.0]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move backward along x-axis by large distance",
                "move left along y-axis a bit more",
                "move down along z-axis a bit more",
                "keep gripper closed"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.065, 0.107, 0.991]",
                "orientation": "[0.701, 0.713, -0.0, -0.0]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "moved_object": [
                "stick: [-0.006, -0.017, 0.989]"
            ]
        }
    },
    "8": {
        "updated_scene_info": {
            "objects": [
                "stick: [-0.006, -0.017, 0.989]"
            ]
        },
        "current_timestep": {
            "status": "success",
            "robot_pose": {
                "position": "[0.065, 0.107, 0.991]",
                "orientation": "[0.701, 0.713, -0.0, -0.0]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "rotate gripper about y-axis",
                "keep gripper closed"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.065, 0.107, 0.994]",
                "orientation": "[0.673, 0.686, 0.194, -0.195]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "moved_object": [
                "stick: [0.068, -0.039, 0.946]"
            ]
        }
    },
    "9": {
        "updated_scene_info": {
            "objects": [
                "stick: [0.068, -0.039, 0.946]"
            ]
        },
        "current_timestep": {
            "status": "success",
            "robot_pose": {
                "position": "[0.065, 0.107, 0.994]",
                "orientation": "[0.673, 0.686, 0.194, -0.195]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move backward along x-axis by large distance",
                "keep gripper closed"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.319, 0.113, 0.991]",
                "orientation": "[0.673, 0.686, 0.194, -0.198]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "moved_object": [
                "stick: [0.07, 0.002, 0.886]"
            ]
        }
    },
    "10": {
        "updated_scene_info": {
            "objects": [
                "stick: [0.07, 0.002, 0.886]"
            ]
        },
        "current_timestep": {
            "status": "recoverable failure",
            "failure_reasoning": "failed to place, moved backward along x-axis too little",
            "robot_pose": {
                "position": "[0.225, 0.111, 0.991]",
                "orientation": "[0.69, 0.702, 0.123, -0.126]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "instruction_in_spatial_language": [
                "move backward along x-axis by large distance",
                "rotate gripper about y-axis",
                "keep gripper closed"
            ]
        },
        "desired_state_at_next_timestep": {
            "robot_pose": {
                "position": "[0.319, 0.113, 0.991]",
                "orientation": "[0.673, 0.686, 0.194, -0.198]",
                "gripper_open": false,
                "ignore_collision": false
            },
            "moved_object": [
                "cube: [0.26, -0.1, 0.797]",
                "stick: [0.222, -0.011, 0.896]"
            ]
        }
    },
    "11": {
        "updated_scene_info": {
            "objects": [
                "cube: [0.26, -0.1, 0.797]",
                "stick: [0.222, -0.011, 0.896]"
            ]
        },
        "current_timestep": {
            "status": "end of episode",
            "instruction_in_spatial_language": [
                "goal reached"
            ]
        }
    }
}

OUTPUT:
{
    "0": "Move towards the stick and prepare to grasp it.",
    "1": "The robot did not align correctly with the stick. Move towards the stick as planned previously by moving forward and left a little bit.",
    "2": "Descend towards the stick to grasp one end of the stick.",
    "3": "The attempt to grasp the stick failed due to incorrect positioning. Adjust the robot's position to be centered over the stick by moving backward and right then move up to prepare for retry.",
    "4": "Descend again and attempt to grasp the stick securely.",
    "5": "Lift the stick upwards.",
    "6": "Move right such that the other end of the stick is positioned close to the cube to prepare for dragging the cube while avoiding collision.",
    "7": "You moved right and high too much. Move left and down a bit to align the stick with the cube for dragging.",
    "8": "Rotate the gripper to adjust the stick's angle to get ready for dragging.",
    "9": "Use the stick to drag the cube to the magenta target.",
    "10": "You didn't drag the cube close enough to the target. Continue dragging the cube closer to the target.",
    "11": "Goal reached."
}

"""