
#!/bin/bash

GRID_SIZE=(256 512 1024 2048 4096)
N=(256 512 1024 2048 4096)
CPU_PER_TASK=12
RUNS=5

for G in "${GRID_SIZE[@]}"; do
    for N in "${SIZES[@]}"; do
        for run in $(seq 1 $RUNS); do
            out_file="res/lenia2_${N}_${GRID_SIZE}x${GRID_SIZE}_${run}.log"
            ./run_lenia.sh "$N" "$GRID_SIZE" "$out_file" "$CPU_PER_TASK"
        done
    done
done 