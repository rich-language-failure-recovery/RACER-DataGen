import json
from racer_datagen.utils.const_utils import *

def get_failed_episodes(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    failed_episodes = {}

    for task, episodes in data.items():
        failed_episodes[task] = [episode['episode'] for episode in episodes if episode['score'] == 0.0]

    return failed_episodes

def save_failed_episodes_to_json(failed_episodes, output_file_path):
    json_string = json.dumps(failed_episodes)
    with open(output_file_path, 'w') as outfile:
        outfile.write(json_string)

if __name__ == "__main__":
    split = "test"

    file_path = f"{BASE_PATH}/data_aug/{split}_episode_info.json"
    output_file_path = f"{split}_failed_episodes.json"
    
    failed_episodes = get_failed_episodes(file_path)
    print(failed_episodes)
    save_failed_episodes_to_json(failed_episodes, output_file_path)
    print(f"Failed episodes saved to {output_file_path}")