#!/bin/bash

# List of files
files=("wall_street_same_prtr.txt" "union_square_same_prtr.txt" "wall_street_moredata.txt" "union_square_moredata.txt")

# Loop through each file and run the Python script
for file in "${files[@]}"; do
    echo "Running python t2c_one.py $file"
    python t2c_unlabel.py "$file"
done

