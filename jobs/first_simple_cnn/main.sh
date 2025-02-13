#!/bin/bash

#SBATCH --job-name=first_simple_CNN
#SBATCH --partition=agpu06
#SBATCH --output=hogML_main.txt
#SBATCH --error=hogML_main.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nmp002@uark.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH -qos=gpu

export OMP_NUM_THREADS=1

# load required module
module purge
module load python/anaconda-3.14

# activate venv
conda activate /home/nmp002/.conda/envs/np_env
echo $SLURM_JOB_ID

cd $SLURM_JOB_DIR || exit
# input files needed for job
files=/home/nmp002/data/Highlands_Data_for_ML

echo "Copying files..."
mkdir /scratch/$SLURM_JOB_ID/data
rsync -avq $files /scratch/$SLURM_JOB_ID/data
rsync -avq $SLURM_JOB_DIR/first_simply_CNN.py /scratch/$SLURM_JOB_ID
rsync - avq /home/nmp002/HighlandsMachineLearning/my_modules /scratch/$SLURM_JOB_ID
wait

cd /scratch/$SLURM_JOB_ID/ || EXIT

echo "Python script initiating..."
python3 first_simply_CNN.py

rsync -av -q /scratch/$SLURM_JOB_ID/ $SLURM_JOB_DIR/.rnd

# check if rsync succeeded
if [ $? -ne 0 ]; then
  echo "Error: Failed to sync files back to original directory. Check /scratch/$SLURM_JOB_ID/ for output files."
  exit 1
fi
