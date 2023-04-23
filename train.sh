#!/bin/bash
# Above line is required!

# Name of job in SLURM queue
#SBATCH --job-name=julio_train

# Output and error log locations (captures stdout and stderr respectively). Defaults to your homedir.
#SBATCH --output=/home/jarroyoi/exp1.out
#SBATCH --error=/home/jarroyoi/exp1.err

# Account to charge this computation time to. THIS LINE IS ESSENTIAL.
#SBATCH -A CS156b

# Estimated time this job will take. A job exceeding this time will be killed.
# Required parameter!
#SBATCH -t 1:30:00

# Total number of concurrent srun tasks. Most people will not need this.
#SBATCH --ntasks=1

# Number of CPU threads for each task as defined above. Most people will be
# using a single task, so this is the total number of threads required.
#SBATCH --cpus-per-task=1

# Total amount of system RAM for all tasks. Specify units with M and G. 
#SBATCH --mem=16G

# Request a single Tesla P100 GPU
#SBATCH --gres=gpu:1

# Send status emails to an email
#SBATCH --mail-user=jarroyoi@caltech.edu

# Enable email notifications for changes to the job state
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

python3 /groups/CS156b/2023/Xray-diagnosis/train.py