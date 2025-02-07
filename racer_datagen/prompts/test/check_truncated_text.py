import os
import re
import json
import clip
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

save_path = f"{BASE_PATH}/runs/rvt/augmented_data_heuristic/val"
txt_path = os.path.join(save_path, "truncated2.txt")
truncated_text = "truncated_text.txt"
language_description = "language_description.json"

for task in task_list:
    for ep in range(100):
        dir = os.path.join(save_path, task, str(ep))

        if os.path.exists(dir):
            truncated_text_path = os.path.join(dir, truncated_text)
            language_description_path = os.path.join(dir, language_description)
            if os.path.exists(truncated_text_path):
                # print(truncated_text_path)
                with open(txt_path, "a") as f:
                    f.write(f"{task}, {ep} -> {truncated_text_path}\n")
                    with open(truncated_text_path, 'r') as file:
                        # Read the contents of the file
                        text = file.read()
                        parts = text.split(', ')
                        png_part = next(part for part in parts if part.endswith('.png'))
                        key_part = png_part.split('.png')[0]
                        number_part = key_part.split('_')[0]
                        with open(language_description_path, 'r') as file2:
                            lang_desc = json.load(file2)
                            key = f"{number_part}_expert"

                            if "expert" in key_part:
                                lang = lang_desc['subgoal'][key]['gpt-lang']
                                f.write(f"{key_part}\n")
                            else:
                                lang = lang_desc['subgoal'][key]['augmentation'][key_part]['gpt-lang']
                                f.write(f"{key_part}\n")
                            # if lang_desc['subgoal'].get(key, None) is None:
                            # print(f"{task}, {ep} -> {key} not found")
                            # lang = lang_desc['subgoal'][key]['augmentation'][key_part]['gpt-lang']

                            task_goal = lang_desc["task_goal"]

                            if isinstance(lang, list):
                                lang = lang[0]
                            f.write(f"task goal: {task_goal}, current instruction: {lang}\n")
                            # tokens = clip.tokenize()

                # write newline into txt 
                with open(txt_path, "a") as f:
                    f.write("\n")
                continue

