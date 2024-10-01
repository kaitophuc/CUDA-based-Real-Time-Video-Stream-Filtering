#!/bin/bash

# Directory containing input files
DATA_DIR="./data"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory $DATA_DIR does not exist."
    exit 1
fi

# List available input files
echo "Available input files:"
ls "$DATA_DIR"

# Prompt user to select an input file
read -p "Enter the name of the input file: " INPUT_FILE

# Check if the selected file exists
if [ ! -f "$DATA_DIR/$INPUT_FILE" ]; then
    echo "Input file $DATA_DIR/$INPUT_FILE does not exist."
    exit 1
fi

# Run the project with the selected input file
echo "Running project with input file $DATA_DIR/$INPUT_FILE..."
# Replace the following line with the actual command to run your project
make all ARGS="$DATA_DIR/$INPUT_FILE"

echo "Project completed."