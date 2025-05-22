#!/bin/bash

# Set the input and output directories
input_dir="/home/udesa_ubuntu/tesis/frontend/public/assets/stimuli/E/E8"
output_dir="/home/udesa_ubuntu/tesis/frontend/public/assets/stimuli/E/E8_norm"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Iterate through all files in the input directory
for file in "$input_dir"/*; do
    # Get the file name without the path
    filename=$(basename "$file")

    # Use FFmpeg to normalize the volume and save the result in the output directory
    ffmpeg -i "$file" -af loudnorm "$output_dir/$filename"
done
