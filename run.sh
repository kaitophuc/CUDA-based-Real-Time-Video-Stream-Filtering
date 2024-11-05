#!/bin/bash

# Directory containing input files
DATA_DIR="./data"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory $DATA_DIR does not exist."
    exit 1
fi

# Ask the user if they want to use a live camera or a recorded video
echo "Do you want to use a live camera or a recorded video?"
echo "1. Live Camera"
echo "2. Recorded Video"
read -p "Enter your choice (1 or 2): " CHOICE

if [ "$CHOICE" -eq 1 ]; then
    # Run the project with live camera
    echo "Running project with live camera..."
    # Add your command to run the project with live camera here
    make all ARGS="0"

elif [ "$CHOICE" -eq 2 ]; then
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
else
    echo "Invalid choice. Please enter 1 or 2."
    exit 1
fi

echo "Project completed."