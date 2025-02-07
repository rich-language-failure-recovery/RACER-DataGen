
import os
import re
import ast
import json
import clip
import pickle

from racer_datagen.data_aug.base.episode import Episode

def truncate_string_to_max_tokens(text, max_tokens=77):
    words = text.split()
    truncated_text = ""
    
    # Start with the entire text and keep reducing it
    for i in range(len(words)):
        truncated_text = " ".join(words[:i+1])
        
        try:
            tokens = clip.tokenize([truncated_text])
            if (tokens != 0).sum().item() > max_tokens:
                truncated_text = " ".join(words[:i])  # Take the previous state which was within the limit
                break
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            truncated_text = " ".join(words[:i])  # Take the previous state which was within the limit
            break
    
    return truncated_text

def count_subdirectories(path):
    return sum(os.path.isdir(os.path.join(path, entry)) for entry in os.listdir(path))

def clean_json_string(json_string):
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip()

def merge_and_sort_dicts(dict1, dict2):
    merged_dict = {**dict1, **dict2}
    sorted_dict = dict(sorted(merged_dict.items()))
    return sorted_dict

def episode_to_json_converter(ep_save_path: str, save_path: str, episode: Episode = None):
    
    episode_path = os.path.join(ep_save_path, "annotated_episode.pkl")

    if episode is None:
        with open(episode_path, "rb") as f:
            episode: Episode = pickle.load(f)

    language = {}
    language["task_goal"] = episode.lang_goal
    language["subgoal"] = {}

    number_of_perturbs = count_subdirectories(f"{ep_save_path}/output")

    for k in range(number_of_perturbs):
        # print(k)
        augmentation = {}
        prev_perturb_key = None
        
        gpt_response_path = os.path.join(ep_save_path, f"output/{k}/gpt_response.json")
        gpt_key_mapping_path = os.path.join(ep_save_path, f"output/{k}/key_mapping.json")
        language_path = os.path.join(ep_save_path, f"language_description.json")
        
        with open(gpt_response_path, "r") as f:
            gpt_response = json.load(f)
            system_prompt = gpt_response[0]
            user_prompt = gpt_response[1]
            system_response = gpt_response[2]

        system_response_dict = ast.literal_eval(clean_json_string(system_response["content"]))

        with open(gpt_key_mapping_path, "r") as f:
            key_mapping = json.load(f)
        perturb_idx = None
        for i, wpt in enumerate(episode.iterate_all_keypoints(perturb_idx=k)):
            img_path = key_mapping[str(i)]
            parts = img_path.split('_')
            keypoint_number = int(parts[0])
            if "perturb" in parts and perturb_idx is None:
                perturb_idx = int(parts[-1].split('.')[0])
            action_type = img_path.split('_')[1].split('.')[0]
            if system_response_dict.get(str(i), None) is None:
                system_response_dict[str(i)] = "Goal reached."

            try:
                tokens = clip.tokenize(f"task goal: {episode.lang_goal}, current instruction: {system_response_dict[str(i)]}")
                # keep gpt responses that exceed the token limit
                if (tokens != 0).sum().item() >= 77:
                    with open(os.path.join(ep_save_path, "truncated_text.txt"), "a") as f:
                        f.write(f"{episode.task_name}, {episode.episode_num}, {i}, perturb_idx: {k}, {img_path}, tokens {(tokens != 0).sum().item()} \n")

            except RuntimeError as e:
                print(f"RuntimeError: {e}")
            if len(system_response_dict) <= i:
                system_response_dict[str(i)] = "Goal reached."
            # truncated_text = truncate_string_to_max_tokens(system_response_dict[str(i)], max_tokens=77-(tokens != 0).sum().item())
            # print(img_path)
            if action_type == "expert":
                if language["subgoal"].get(f"{keypoint_number}_{action_type}", None) is None:
                    label = "start" if i == 0 else "success"
                    label = "end" if wpt.id == len(episode) - 1 else label
                    language["subgoal"][f"{keypoint_number}_{action_type}"] = {
                        "idx": i,
                        "label": label,
                        "heuristic-lang": ", ".join(wpt.info.get('lang', [])),
                        "gpt-lang": [system_response_dict[str(i)]],
                        "augmentation": [augmentation]
                    }
                else:
                    language["subgoal"][f"{keypoint_number}_{action_type}"]["gpt-lang"].append(system_response_dict[str(i)])
                    language["subgoal"][f"{keypoint_number}_{action_type}"]["augmentation"].append(augmentation)
                augmentation = {}

                if len(language["subgoal"][f"{keypoint_number}_{action_type}"]["augmentation"]) == number_of_perturbs:
                    merged_aug_dict = {}
                    for d in language["subgoal"][f"{keypoint_number}_{action_type}"]["augmentation"]:
                        for key, v in d.items():
                            merged_aug_dict[key] = v
                    language["subgoal"][f"{keypoint_number}_{action_type}"]["augmentation"] = merged_aug_dict
            else:
                if action_type == "perturb":
                    print(os.path.splitext(os.path.basename(img_path))[0])
                    augmentation[os.path.splitext(os.path.basename(img_path))[0]] = {
                        "idx": i,
                        "label": "recoverable_failure",
                        "heuristic-lang": ", ".join(wpt.info.get('lang', [])),
                        "gpt-lang": [system_response_dict[str(i)]],
                        "correction": {}
                    }
                    prev_perturb_key = os.path.splitext(os.path.basename(img_path))[0]
                    # print(prev_perturb_key)
                if action_type == "intermediate":
                    augmentation[prev_perturb_key]["correction"][os.path.splitext(os.path.basename(img_path))[0]] = {
                        "idx": i,
                        "label": "ongoing",
                        "heuristic-lang": ", ".join(wpt.info.get('lang', [])),
                        "gpt-lang": [system_response_dict[str(i)]],
                    }
                    prev_perturb_key = None

                # tokens = clip.tokenize(f"task goal: {episode.lang_goal}, current instruction: {system_response_dict[str(i)]}")
                # print(f"keypoint_number: {keypoint_number}, action_type: {action_type}, tokens: {(tokens != 0).sum().item()}")

        # if episode.task_name == "place_cups":
        #     if perturb_idx is None:
        #         perturb_idx = 0
        #     with open(os.path.join(ep_save_path, f"output/{perturb_idx}/place_cups_dense.json"), "r") as f:
        #         dense = json.load(f)
        #         subgoal = merge_and_sort_dicts(language["subgoal"], dense)
        #         language["subgoal"] = subgoal
        
        language["subgoal"] = dict(sorted(language["subgoal"].items(), key=lambda item: int(item[0].split('_')[0])))
            
    with open(language_path, "w") as f:
        json.dump(language, f, indent=4)