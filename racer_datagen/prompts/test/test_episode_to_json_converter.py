import os
import clip
from racer_datagen.prompts.episode_to_json_converter import episode_to_json_converter



if __name__ == "__main__":
    # Example text
    text = "To handle the RuntimeError and ensure that the input string is truncated to a maximum of 77 tokens, you can use a try-except block within the truncation function. This way, if the input text is too long and causes the error, the function will catch it and attempt to truncate the text until it fits within the allowed context length. To truncate a string so that its tokenized version contains a maximum of 77 tokens, you can follow these steps."

    text = "task goal: stack the wine bottle to the left of the rack current instruction: The robot did not move backward enough and moved too far to the right, also not descending enough to align with the rack. Move backward substantially, slightly to the left, and then down to align the wine bottle correctly with the left rack location. Keep the gripper closed and allow collision."

    tokens = clip.tokenize([text])
    print(f"Number of tokens: {(tokens != 0).sum().item()}")

