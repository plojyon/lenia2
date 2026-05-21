#!/bin/bash

N=$1
OUT_FILE=$2
CPU_PER_TASK=$3

sbatch <<EOT
#!/bin/bash
#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lenia_2_${N}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPU_PER_TASK}
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=${OUT_FILE}
#SBATCH --hint=nomultithread

#LOAD MODULES
module load OpenMPI

#RUN
srun mpirun -np $N ./lenia.out
EOT