import os
from racer_datagen.utils.const_utils import *

task_list = [
    "put_item_in_drawer",
    "reach_and_drag",
    "turn_tap",
    "slide_block_to_color_target",
    "open_drawer",
    "put_groceries_in_cupboard",
    "place_shape_in_shape_sorter",
    "put_money_in_safe",
    "push_buttons",
    "close_jar",
    "stack_blocks",
    "place_cups",
    "place_wine_at_rack_location",
    "light_bulb_in",
    "sweep_to_dustpan_of_size",
    "insert_onto_square_peg",
    "meat_off_grill",
    "stack_cups",
]

save_path = f"{BASE_PATH}/runs/rvt/augmented_data_heuristic/test"
txt_path = os.path.join(save_path, "missing_lang_desc_3.txt")
lang_desc_name = "language_description.json"

for task in task_list:
    for ep in range(25):
        dir = os.path.join(save_path, task, str(ep))
        # check if dir exist, if not add to txt file
        if not os.path.exists(dir):
            with open(txt_path, "a") as f:
                f.write(f"{task}, {ep} -> expert failures\n")
            continue

        # check if language_description.json exist, if not add to txt file
        if os.path.exists(dir):
            lang_desc_path = os.path.join(dir, lang_desc_name)
            if not os.path.exists(lang_desc_path):
                with open(txt_path, "a") as f:
                    f.write(f"{task}, {ep} -> missing {lang_desc_name}\n")
                continue

    # write newline into txt 
    with open(txt_path, "a") as f:
        f.write("\n")

