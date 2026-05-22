#!/bin/bash

NUM_PROC=$1
GRID_SIZE=$2
OUT_FILE=$3
CPU_PER_TASK=$4

sbatch <<EOT
#!/bin/bash
#SBATCH --reservation=fri
#SBATCH --job-name=lenia_2_${NUM_PROC}_${GRID_SIZE}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPU_PER_TASK}
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=${OUT_FILE}
#SBATCH --hint=nomultithread

#LOAD MODULES
module load OpenMPI

#RUN
srun mpirun --mca pml ob1 -np $NUM_PROC ./lenia.out $GRID_SIZE
EOT