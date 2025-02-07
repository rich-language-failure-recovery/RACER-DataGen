from racer_datagen.utils.const_utils import BASE_PATH

# Dictionary to store the count of 'Success: False' for each task
task_failures = {}

# Open the file and process each line
with open(f"{BASE_PATH}/runs/rvt/expert_all_tasks_all_val_episode_0418/task_success.txt", 'r') as file:
    for line in file:
        # Split the line into components
        parts = line.split('|')
        
        # Extract the task name, episode number, and success status
        task = parts[0].strip().split(': ')[1]
        episode = parts[2].strip().split(': ')[1]
        success = parts[3].strip().split(': ')[1]
        
        # Check if the task is already in the dictionary
        if task not in task_failures:
            task_failures[task] = []
        
        # If the task was not successful, add the episode number to the list
        if success == 'False':
            task_failures[task].append(episode)

# Print the results
for task, episodes in task_failures.items():
    print(f"{task}: {len(episodes)} failures in ep: {', '.join(episodes)}")
total_failures = sum([len(episodes) for episodes in task_failures.values()])
print(f"Total number of failures: {total_failures}")
