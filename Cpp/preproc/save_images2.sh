#!/bin/bash
# Above line is required!

# Name of job in SLURM queue
#SBATCH --job-name=part2_allTrain_224x224

# Output and error log locations (captures stdout and stderr respectively). Defaults to your homedir.
#SBATCH --output=/groups/CS156b/2023/Xray-diagnosis/Cpp/out/part2_allTrain_224x224.out
#SBATCH --error=/groups/CS156b/2023/Xray-diagnosis/Cpp/out/part2_allTrain_224x224.err

# Account to charge this computation time to. THIS LINE IS ESSENTIAL.
#SBATCH -A CS156b

# Estimated time this job will take. A job exceeding this time will be killed.
# Required parameter!
#SBATCH -t 03:00:00

# Total number of concurrent srun tasks. Most people will not need this.
#SBATCH --ntasks=1

# Number of CPU threads for each task as defined above. Most people will be
# using a single task, so this is the total number of threads required.
#SBATCH --cpus-per-task=1

# Total amount of system RAM for all tasks. Specify units with M and G. 
#SBATCH --mem=128G

# Request a single Tesla P100 GPU
#SBATCH --gres=gpu:1

# Send status emails to an email
#SBATCH --mail-user=jarroyoi@caltech.edu

# Enable email notifications for changes to the job state
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

python3 /groups/CS156b/2023/Xray-diagnosis/Cpp/preproc/save_images.py --start 60000 --num-images 59079 --image-dims 224 --part 2
