#!/bin/bash

# List of files
files=("spaceinvaders_e2e.txt" "carnival_e2e.txt" "allegheny_e2e.txt" "wall_street_e2e.txt" "union_square_e2e.txt" "south_shore_e2e.txt" "phoenix_e2e.txt" "hudson_river_e2e.txt" "demonattack_e2e.txt" "cmu_e2e.txt" "beamrider_e2e.txt" "spaceinvaders_prtr.txt" "carnival_prtr.txt" "allegheny_prtr.txt" "wall_street_prtr.txt" "union_square_prtr.txt" "south_shore_prtr.txt" "phoenix_prtr.txt" "hudson_river_prtr.txt" "demonattack_prtr.txt" "cmu_prtr.txt" "beamrider_prtr.txt" "wall_street_same_prtr.txt" "union_square_same_prtr.txt" "wall_street_moredata.txt" "union_square_moredata.txt")

# Loop through each file and run the Python script
for file in "${files[@]}"; do
    echo "Running python t2c_one.py $file"
    python t2c_unlabel.py "$file"
done

