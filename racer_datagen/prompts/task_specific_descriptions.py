from racer_datagen.utils.rvt_utils import RLBENCH_TASKS

INSERT_ONTO_SQUARE_PEG = \
"""\
The "insert_onto_square_peg" task has a square ring and three spokes or square sticks. \
The task is to pick up the square ring and insert it onto the spoke, specified by the color. \
The robot is likely to fail to align the gripper's orientation before grasping. \
BE SURE TO SAY THE CORRECT COLOR OF THE SPOKE TO INSERT THE RING ONTO. \
"""

SWEEP_TO_DUSTPAN_OF_SIZE = \
"""\
The "sweep_to_dustpan_of_size" task has a broom, a dustpan, and several small cubes that represent dust. \
The task is to sweep the dust to the dustpan. \
The robot has to pick up the broom and sweep the dust to the dustpan. \
It is important to emphasize that the robot needs to lower the broom enough to the ground to sweep the dust. \
"""

LIGHT_BULB_IN = \
"""\
The "light_bulb_in" task has a light bulb and a lamp. \
The task is to pick up the light bulb, place it inside the lamp, and screw the light bulb in. \
"""

PLACE_WINE_AT_RACK_LOCATION = \
"""\
The "place_wine_at_rack_location" task has a wine bottle and a wine rack. \
The task is to pick up the wine bottle and place it at the specified location on the wine rack. \
There are three locations on the wine rack: right, middle, and left. \
The right rack is the one most forward, the middle rack is in the middle, and the left rack is the one most backward. \
"""

REACH_AND_DRAG = \
"""\
The "reach_and_drag" task has a stick and a cube. \
The task is to grab the stick on one end and use it as a tool to push or drag the cube to a target location. \
The target location is indicated by square patch on the table, specified by the color. \
It is important to align the other end of the stick with the cube to push it. \
The gripper needs to rotate such that the stick is aligned with the cube. \
"""

STACK_BLOCKS = \
"""\
The "stack_blocks" task has four blocks of target color and four additional distractor blocks of different color and a place to stack. \
The task is to stack the specified number of target blocks on top of each other. \
"""

TURN_TAP = \
"""\
The "turn_tap" task has a tap or faucet object on the table. \
There are two handles or knobs on the tap. \
The task is to turn the tap by rotating one handle specified by left or right.
"""

## v1
SLIDE_BLOCK_TO_COLOR_TARGET = \
"""\
The "slide_block_to_color_target" task has a block and a target location. \
The block is placed on the table and the target location is indicated by a square patch on the table, specified by the color. \
The task is to slide the block to the target location. \
Note that the robot has to close the gripper before pushing the block. \
Remember there is no grasping involved in this task. \
The gripper closes to push the block to the target location not to grasp. \
The robot is likely to fail to align or position the gripper correctly to push the block and also fail to push the block fully to the target location. \
# Use relative_gripper_pos_to_block instead of instruction_in_spatial_language to guide your commands for the robot to specifically tell the robot which side of the block the robot should go to and push. \
# YOU MUST SPECIFY WHICH SIDE OF THE BLOCK THE ROBOT SHOULD PUSH TO. DO NOT CONFUSE THIS WITH ROBOT'S SPATIAL LANGUAGE. \
# ALL SPATIAL LANGUAGE IS IN HUMAN'S PERSPECTIVE. DO NOT TALK ANYTHING IN ROBOT'S PERSPECTIVE. \
"""

## v2
SLIDE_BLOCK_TO_COLOR_TARGET = \
"""\
The "slide_block_to_color_target" task has a block and a target location. \
The block is placed on the table and the target location is indicated by a square patch on the table, specified by the color. \
The task is to push the block to the target location. \
Use relative_gripper_pos_to_block instead of instruction_in_spatial_language to guide your commands for the robot to specifically tell the robot which side of the block the robot should go to and push. \
The robot will fail by moving to the wrong side of the block before pushing. Correct it to the correct side for the push. \
The robot also fails to push the block fully to the target location. Tell it to push more. \
YOU MUST SPECIFY WHICH SIDE OF THE BLOCK THE ROBOT SHOULD PUSH TO. DO NOT CONFUSE THIS WITH ROBOT'S SPATIAL LANGUAGE. \
ALL SPATIAL LANGUAGE IS IN HUMAN'S PERSPECTIVE. DO NOT TALK ANYTHING IN ROBOT'S PERSPECTIVE. \
THIS TASK DOES NOT INVOLVE ANY GRASPING or GRABBING, IT IS JUST CLOSING THE GRIPPER TO PUSH. IGNORE ANY GRASPING ACTION. \
LONGER TASKS REQUIRE TWO-STEP PUSHING AND THE ROBOT NEEDS TO RETRACT TO PREPARE FOR THE NEXT PUSH. \
"""

OPEN_DRAWER = \
"""\
The "open_drawer" task has a drawer with top, middle, and bottom drawers. \
The task is to open the specified drawer. \
The robot needs to approach the drawer handle which is on the left and pull the drawer to the right to open it.
"""

PUT_GROCERIES_IN_CUPBOARD = \
"""\
The "put_groceries_in_cupboard" task has several groceries on the table and a cupboard. \
The groceries include: tuna, sugar, crackers, soup, coffee, chocolate jello, strawberry jello, spam, and mustard. \
The task is to pick up a specified grocery and place it inside the cupboard.
"""

PLACE_SHAPE_IN_SHAPE_SORTER = \
"""\
The "place_shape_in_shape_sorter" task has a shape sorter toy with several holes on the top and several shapes on the table. \
The task is to pick up a specified shape, orient it correctly, and place it in the corresponding hole with the matching shape. \
The robot is likely to fail to align the gripper's orientation with the shape's orientation when grasping as well as the shape's rotation with the hole when placing at a later step. \
"""

PUSH_BUTTONS = \
"""\
The "push_buttons" task has several buttons placed over the table. \
The task is to press the specified button in the given sequence, specified by the order of the colors. \
Note that there is also no grasping involved in this task. The gripper may close in order to press the button.
"""

MEAT_OFF_GRILL = \
"""The "meat_off_grill" task has a meat object on a grill. The task is to remove the meat from the grill and place it somewhere appropriate.
"""

STACK_CUPS = \
"""\
The "stack_cups" task has three handleless cups of different colors on the table. \
The task is to stack all the other cups by picking the rim of the cups and placing on top of the cup specified by the color. \
Note that when some cups are closely placed side-by-side, it is important to pick up the cup by the rim that is not in the middle of both. \
It is important to align with the rim of the cups. \
The robot is likely to fail when aligning the orientation of the cup onto the stack right before the placing. Make sure it can rotate the gripper correctly to fix this failure. \
"""

CLOSE_JAR = \
"""\
The "close_jar" task has two jars and a single lid placed on the table. \
The task is to close the colored jar using the lid. \
The robot must first pick up the lid and then close the jar. \
When the robot fails to grasp the lid, tell it to prepare to retry by moving back up to grasp again.
"""

PUT_MONEY_IN_SAFE = \
"""\
The "put_money_in_safe task has a money object or cash on top of a safe in the shape of a bookshelf. \
The task is to pick up the money and place it inside the safe. \
There are three shelves: top, middle, and bottom.
"""

PLACE_CUPS = \
"""\
The "place_cups" task has several cups in the scene and a vertical cup holder with branches to hang using the cup handle. \
The task is to pick up the cup in correct orientation and to place it on to each hanger branch. \
The key is to align the gripper with the cup handle and to place it on the branch. \
The robot is most likely to fail to align with the cup handle with a perpendicular or misaligned rotation.
"""

PUT_ITEM_IN_DRAWER = \
"""\
The "put_item_in_drawer" task has a drawer with a single item placed on top of the drawer. \
The drawer has top, middle, and bottom drawer. \
The task is to open the drawer first, then pick up the item, and place it inside the drawer. \
Make sure that the robot opens the drawer first before picking up the item. \
The robot is not attempting to grasp the item if the next desired state of the item is not moving. \
The robot moves closer to the drawer first, then try to align with the handle, and try to grasp it. \
The robot does not move directly to the drawer to grasp it. \
Likewise, the robot takes several steps to approach the item and grasp it. The robot is not grasping until it lowers the gripper to the item. \
The robot is likely to fail when it is aligning with the opened drawer as it moves too far out of the drawer. Correct it by telling the direction to move towards before placing the item. \
"""
