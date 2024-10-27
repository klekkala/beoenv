#!/bin/bash

# List of files
files=("spaceinvaders_e2e.txt" "carnival_e2e.txt" "allegheny_e2e.txt" "wall_street_e2e.txt" "union_square_e2e.txt" "south_shore_e2e.txt" "phoenix_e2e.txt" "hudson_river_e2e.txt" "demonattack_e2e.txt" "cmu_e2e.txt" "beamrider_e2e.txt")

# Loop through each file and run the Python script
for file in "${files[@]}"; do
    echo "Running python t2c_one.py $file"
    python t2c_unlabel.py "$file"
done

