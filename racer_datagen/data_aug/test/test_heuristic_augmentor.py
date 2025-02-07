from peract_colab.rlbench.utils import get_stored_demo

from racer_datagen.utils.const_utils import *

task_name = "open_drawer"
dataset_root=f"{BASE_PATH}/data/test/{task_name}/all_variations/episodes"
cfg_path = f"{BASE_PATH}/data_aug/configs/{task_name}.yaml"
demos = get_stored_demo(dataset_root, 0)

# 0: 0.27844 -0.00816 1.47198 0.0 0.99266 -0.0 0.1209 1.0 0.0 
# 51: 0.47308 0.25911 0.92801 0.30033 0.95384 -0.00065 -0.00072 1.0 0.0 
# 67: 0.47311 0.2594 0.75831 0.30049 0.95378 -0.0004 -0.0006 0.0 1.0 
# 81: 0.47297 0.25914 0.92605 0.30057 0.95376 -0.00077 -0.001 0.0 1.0 
# 129: 0.22672 -0.13485 0.92772 0.29813 0.95453 -5e-05 -0.00083 0.0 0.0 
# 140: 0.22594 -0.13556 0.86784 0.29951 0.95409 -8e-05 -0.00138 0.0 1.0
# 177: 0.2264 -0.13573 0.86704 -0.88425 -0.46701 0.00052 0.00028 1.0 1.0

from racer_datagen.data_aug.heuristic_augmentor import Heuristic
augmentor = Heuristic(task_name, cfg_path=cfg_path)
episode = augmentor.load_demo(demos)
print("keypoints", list(episode.retrieve_keypoints()))
episode = augmentor.heuristic_perturb(episode, N=0)
for step in episode.retrieve_keypoints():
    if step == 0:
        continue
    kwpt = episode[step]
    print(f"expert step {step}:", kwpt.action.to_interaction(), " | ", kwpt.verbose)
    for i, perturb_wpt in enumerate(kwpt.perturbations):
        print(f"\tperturb {i}:", perturb_wpt.mistake.action.to_interaction())
        if perturb_wpt.correction is not None:
            print(f"\tcorrect {i}:", perturb_wpt.correction.action.to_interaction())
        else:
            print(f"\tcorrect {i}:", None)
