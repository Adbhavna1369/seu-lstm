#!/bin/bash
#SBATCH --job-name=seu_lstm_baseline
#SBATCH --output=logs/seu_lstm_%j.out
#SBATCH --error=logs/seu_lstm_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
# ---------------------------------------------------------------------------
# NOTE: Adjust the following for your HPC cluster:
#   --partition   : your GPU partition name (gpu, gpuq, gpu_short, etc.)
#   --gres        : gpu:1 requests one GPU; change to gpu:a100:1 if needed
#   --account     : add if your HPC requires a project/account code
#   --mail-user   : uncomment and set your email for job notifications
#   --mail-type   : BEGIN,END,FAIL
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"

# Load modules — adjust module names to match your HPC environment
# Common alternatives: module load python/3.11, cuda/12.1, pytorch/2.2
module load python/3.11 2>/dev/null || true
module load cuda/12.1   2>/dev/null || true

# Activate virtual environment (adjust path to your environment)
# Option A: venv
# source $HOME/envs/seu/bin/activate

# Option B: conda
# conda activate seu

# Option C: pip --user (no environment)
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH

# ---------------------------------------------------------------------------
# Paths — adjust $SCRATCH to your HPC scratch filesystem
# ---------------------------------------------------------------------------
PROJECT_DIR=$HOME/seu-lstm
DATA_DIR=$SCRATCH/seu_data          # fast scratch for I/O
RUN_DIR=$SCRATCH/seu_runs/baseline_$SLURM_JOB_ID

mkdir -p $DATA_DIR/processed
mkdir -p $RUN_DIR
mkdir -p $PROJECT_DIR/logs

cd $PROJECT_DIR

# ---------------------------------------------------------------------------
# Step 1: Data acquisition (only if not already cached)
# Run this block ONCE on a login node before submitting the training job.
# Comment it out after the first successful run.
# ---------------------------------------------------------------------------
# echo "=== Data acquisition ==="
# python data_acquisition.py \
#     --start 2020-01-01 \
#     --end   2023-12-31 \
#     --cache-dir $DATA_DIR/raw
#
# echo "=== Preprocessing ==="
# python preprocess.py \
#     --cache-dir   $DATA_DIR/raw \
#     --out-dir     $DATA_DIR/processed \
#     --creme96-csv $DATA_DIR/creme96_seu_rates.csv \
#     --window-size 24 \
#     --horizon     6
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 2: Training
# ---------------------------------------------------------------------------
echo "=== Training ==="
python train.py \
    --processed-dir $DATA_DIR/processed \
    --out-dir       $RUN_DIR \
    --hidden-size   128 \
    --num-layers    2 \
    --dropout       0.3 \
    --epochs        100 \
    --batch-size    128 \
    --lr            1e-3 \
    --weight-decay  1e-4 \
    --huber-delta   1.0 \
    --patience      10 \
    --num-workers   $SLURM_CPUS_PER_TASK

# ---------------------------------------------------------------------------
# Copy outputs back to project directory
# ---------------------------------------------------------------------------
echo "=== Saving outputs ==="
cp $RUN_DIR/best_model.pt   $PROJECT_DIR/checkpoints/best_model_$SLURM_JOB_ID.pt
cp $RUN_DIR/training_log.csv $PROJECT_DIR/logs/training_log_$SLURM_JOB_ID.csv
cp $RUN_DIR/test_results.txt $PROJECT_DIR/logs/test_results_$SLURM_JOB_ID.txt

echo "Job finished: $(date)"
echo "Outputs in: $RUN_DIR"
echo "Test results:"
cat $RUN_DIR/test_results.txt
