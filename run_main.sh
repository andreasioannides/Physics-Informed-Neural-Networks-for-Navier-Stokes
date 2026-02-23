#!/bin/bash

txt_files=("nodes_vamv_2D.txt" "nodes_2D.txt")

for txt in "${txt_files[@]}"
do
    echo -e "\n--------------------------------------------------------- Running main.py with input: $txt ---------------------------------------------------------\n"

    if python main.py "$txt"; then
    # if python3.11 main.py "$txt"; then
    # if nohub python3.11 main.py "$txt" > run.out 2>run.err &; then 
        echo -e "\n--------------------------------------------------------- Training succeeded for $txt ---------------------------------------------------------\n"
    else
        echo -e "\n--------------------------------------------------------- Training failed for $txt ---------------------------------------------------------\n"
    fi
done

echo -e "\n--------------------------------------------------------- All trainings completed ---------------------------------------------------------\n"
done
