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
#SBATCH --qos=gpu

export OMP_NUM_THREADS=1

# load required module
module purge
module load python/anaconda-3.14

# activate venv
conda activate /home/nmp002/.conda/envs/np_env
echo $SLURM_JOB_ID

cd $SLURM_SUBMIT_DIR || exit
# input files needed for job
files=/home/nmp002/data/Highlands_Data_for_ML/newData

echo "Copying files..."
mkdir /scratch/$SLURM_JOB_ID/data
rsync -avq $files /scratch/$SLURM_JOB_ID/data
rsync -avq $SLURM_SUBMIT_DIR/first_simple_CNN.py /scratch/$SLURM_JOB_ID
mkdir /scratch/$SLURM_JOB_ID/models
rsync -avq /home/nmp002/MPM-MachineLearning/models/microscopy_cnn.py /scratch/$SLURM_JOB_ID/models
mkdir /scratch/$SLURM_JOB_ID/scripts
rsync -avq /home/nmp002/MPM-MachineLearning/scripts/dataset_loader.py /scratch/$SLURM_JOB_ID/scripts
wait

cd /scratch/$SLURM_JOB_ID/ || EXIT

echo "Python script initiating..."
python3 first_simple_CNN.py

rsync -av -q /scratch/$SLURM_JOB_ID/ $SLURM_SUBMIT_DIR/

# check if rsync succeeded
if [ $? -ne 0 ]; then
  echo "Error: Failed to sync files back to original directory. Check /scratch/$SLURM_JOB_ID/ for output files."
  exit 1
fi
