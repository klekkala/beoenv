#!/bin/bash

# Argument passed to the shell script
argument="$1"

if [ -z "$argument" ]; then
    echo "Please provide an argument."
    exit 1
fi

# List of Python scripts you want to run
python_scripts=(
    "create_value.py"
    "cal_value_1.py"
    "get_dict.py"
    "trun_epi.py"
    "gen_epi_lim_comp.py"
    "gen_epi_lim_trunc.py"
)

# Loop through the Python scripts and run each with the argument
for script in "${python_scripts[@]}"; do
    echo "Running $script with argument: $argument"
    python "$script" "$argument"
done

