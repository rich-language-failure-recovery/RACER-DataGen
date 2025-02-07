
import numpy as np
import cv2
import textwrap

def find_number_after_last_underscore(s):
    parts = s.rsplit('_', 1)
    if len(parts) > 1:
        number_str = parts[1]
        number = ''
        for char in number_str:
            if char.isdigit():
                number += char
            else:
                break
        if number:
            return int(number)
    return None

def find_number_before_first_underscore(s):
    parts = s.split('_', 1)
    if len(parts) > 1:
        number_str = parts[0]
        number = ''
        for char in number_str:
            if char.isdigit():
                number += char
            else:
                break
        if number:
            return int(number)
    return None

def sort_key(filename):
    parts = filename.split('_')
    keypoint_number = int(parts[0])
    action_type = parts[1].split('.')[0]

    # Define a custom order for the action types
    action_order = {'perturb': 0, 'intermediate': 1, 'expert': 2}
    action_priority = action_order.get(action_type, 3)  # Default to a high value if the action type is unknown

    return (keypoint_number, action_priority)


def append_text_underneath_image(image: np.ndarray, text: str):
    """Appends text underneath an image of size (height, width, channels).

    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.

    :param image: The image to appends text underneath.
    :param text: The string to display.
    :return: A new image with text appended underneath.
    """
    h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros((50,w,c), dtype=np.uint8)

    lines = text.split('\n')  # Split the text into lines based on newline characters
    wrapped_lines = []
    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]

    for line in lines:
        wrapped_lines.extend(textwrap.wrap(line, width=int(w / char_size[0])))  # Wrap each line

    y = 0
    for line in wrapped_lines:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final