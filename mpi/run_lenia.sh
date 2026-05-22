#!/bin/bash

NUM_PROC=$1
GRID_SIZE=$2
OUT_FILE=$3

sbatch <<EOT
#!/bin/bash
#SBATCH --reservation=fri
#SBATCH --job-name=lenia_2_${NUM_PROC}_${GRID_SIZE}
#SBATCH --ntasks=${NUM_PROC}
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --output=${OUT_FILE}
#SBATCH --hint=nomultithread

#LOAD MODULES
module load OpenMPI

#RUN
srun mpirun --mca pml ob1 -np $NUM_PROC ./lenia.out $GRID_SIZE
EOT