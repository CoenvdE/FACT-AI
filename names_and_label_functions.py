import json
import torch


def get_encoded_labels(labels, prompt):
    """
    Get prompts that correspond with labels of given batch.
    """
    labels = labels.to(torch.int64)
    selected_encodings = prompt[labels]
    return selected_encodings


def get_ImageNet_ClassNames():
    """
    Reads and returns a list of class names from the ImageNet dataset.

    This function reads a JSON file containing mappings of ImageNet class indices
    to their respective human-readable names and returns a list of these names.

    Returns:
        list: A list of strings where each string is a class name from ImageNet.
    """
    # Path to the JSON file containing ImageNet class index and names
    text_file = '/home/scur1049/FACT/data/imagenet_class_index.json'

    # Open the JSON file and load its contents into a Python dictionary
    with open(text_file, 'r', encoding='utf-8') as f:
        class_index = json.load(f)

    # Initialize an empty list to hold the class names
    names = []

    # Iterate over the dictionary and extract class names
    for i in range(len(class_index)):
        # Append the last element (class name) of each list in the dictionary to 'names'
        name = class_index[str(i)].replace("_", " ")

        names.append(name)

    # Return the list of class names
    return names