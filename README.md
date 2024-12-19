# HockeyAI Setup Guide

## Installation
Run `poetry install` to install all dependencies 

## Cluster Access

### Initial Login
To access from outside the Uni network:
```bash
ssh <username>@login1.tcml.uni-tuebingen.de -p 443
```

### Repository Setup

*Option 1: Clone from GitHub*
1. **Set up GitHub SSH Authentication**
    ```bash
    # Generate SSH key
    ssh-keygen -t rsa -b 4096 -C "HockeyAI"     # You want default settings so press enter bunch of times 

    # Display public key
    cat ~/.ssh/id_rsa.pub
    ```
    - Add the key to GitHub:
        1. Go to Settings > SSH and GPG keys
        2. Click "New SSH key"
        3. Paste the public key and add a title
        4. Save
    
    - Verify authentication:
    ```bash
    ssh -T git@github.com
    # should see something like: "Hi ernaz100! You've successfully authenticated, but GitHub does not provide shell access."
    ```

2. **Pull Repository**
    ```bash
    # For new setup
    git clone git@github.com:Xploeng/HockeyAI.git
    
    # If already cloned
    cd HockeyAI
    git pull origin main    # Update to latest version
    
    # Optional: Switch branches if needed
    git checkout <branch-name>
    ```

*Option 2: Direct Transfer using SCP*
```bash
# From your local machine
scp -P 443 -r /path/to/your/local/HockeyAI <username>@login1.tcml.uni-tuebingen.de:~/
```
> Note: Replace `/path/to/your/local/HockeyAI` with the actual path to your local repository and add your username. 

> Also: Using SSH keys with GitHub (Option 1) is recommended as it's faster and more convenient than copying local files after the initial setup is done once

## Docker & Dependency Management
We use Docker to create an environment for Singularity execution on the cluster. I provided a docker container with the current dependencies, **if those haven't changed skip to the Job Management section**

### Building and Publishing Docker Image
**Only needed if dependencies change:**
```bash
# Rebuild image (uses dependencies defined in pyproject.toml)
docker build --platform linux/amd64 -t your_dockerhub_username/hockeyai .

# Login to Docker Hub
docker login

# Push image
docker push your_dockerhub_username/hockeyai
```
> Note: Update the image URL in sbatch file (line 40) from `docker://rickberd/hockeyai` to `docker://your_dockerhub_username/hockeyai`

## Job Management

### Key Configurations in hockeyai.sbatch
- **Email Notifications**: Update line 37 with your email for job notification (started, failed, finished)
- **Timeout**: Default 23h59min (line 23)
  - Format options: "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes", "days-hours:minutes:seconds"
- **Partition**: Default "day" partition (line 14)
  - "test": up to 15 mins 
  - "day": up to 24 hours
  - "week": up to 7 days
  - "month": up to 30 days
  > Note: Remember to adjust both partition and timeout if you need longer runtime
- **Job Name**: Default "HockeyAI-Sigma0" (line 7)
- **GPU**: Default 4 GPUs (that is the max -> line 20)
- **Logs**: 
  - stderr: job.*jobID*.err
  - stdout: job.*jobID*.out
  - found in the HockeyAI folder
- **Script Execution**: The sbatch file will run `train.py` with the specified config in *src/configs/config.yaml* by default.
  - Ensure your config file is what you want before submitting the job. 
  - You can adjust what will be run in line 40 of the sbatch file (add args, change script or whatever).

### Job Commands
```bash
# Submit job - In HockeyAI folder:
sbatch hockeyai.sbatch

# Check job status
squeue                  # Current jobs
squeue --start         # Scheduled jobs
sinfo                  # Partition status
sacct                  # Job history
```

## Transferring Checkpoint Files
To copy files from the cluster to your local machine:
```bash
# From your local machine
scp -P 443 <username>@login1.tcml.uni-tuebingen.de:~/HockeyAI/outputs/<experiment_name>/* /path/to/local/destination/
```

> Note: Replace `<username>` with your cluster username, `<experiment_name>` with the name of the experiment defined in the config and `/path/to/local/destination/` with the folder where you want to save the checkpoints locally

> Tip: To copy a specific file, replace `<experiment_name>/*` with the path to that specific file, e.g., `test1/hydra.yaml`


## Documentation
For more details, see the [TCML Documentation](https://cogsys.cs.uni-tuebingen.de/webprojects/TCML/TCML_Documentation_2024-12-19.pdf)
