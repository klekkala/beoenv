#!/bin/bash

# Specify the path to your Python script
python_script="regression.py"

# Run the Python script three times
for ((i=1; i<=3; i++)); do
	    echo "Running Python script - Attempt $i"
	        python "$python_script"
	done

	echo "Script execution completed"

