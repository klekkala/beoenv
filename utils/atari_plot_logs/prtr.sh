#!/bin/bash

# List of files
files=("spaceinvaders_prtr.txt" "carnival_prtr.txt" "allegheny_prtr.txt" "wall_street_prtr.txt" "union_square_prtr.txt" "south_shore_prtr.txt" "phoenix_prtr.txt" "hudson_river_prtr.txt" "demonattack_prtr.txt" "cmu_prtr.txt" "beamrider_prtr.txt" "wall_street_same_prtr.txt" "union_square_same_prtr.txt")

# Loop through each file and run the Python script
for file in "${files[@]}"; do
    echo "Running python t2c_one.py $file"
    python t2c_unlabel.py "$file"
done

