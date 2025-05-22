import random
import os
import pickle
import config

def get_filepaths(directory):
    filepaths = []  # Initialize an empty list to store filepaths

    # Iterate over each element in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)  # Construct the filepath
            filepaths.append(filepath)  # Append the filepath to the list

    return filepaths

def shuffle_filepaths(letter, base_dir = config.base_dir):
    # Specify the directory path
    filepaths = []
    directory_path =os.path.join(base_dir, letter)
    filepaths = get_filepaths(directory_path)
    random.shuffle(filepaths)
    # save filepaths as pickle file
    output_path = os.path.join(base_dir, letter + '.pickle')
    with open(output_path, 'wb') as handle:
        pickle.dump(filepaths, handle, protocol=pickle.HIGHEST_PROTOCOL)
