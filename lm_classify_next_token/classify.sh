#!/bin/sh
#SBATCH --job-name=lm-classify-generic-resource-type
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000152
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=12:00:00

module load conda
cd /scratch/m000152/comet/generic-resource-type/lm_classify_next_token

conda run -n comet \
    python classify.py \
    --input_file /scratch/m000152/comet/data/datacite_2024_metadata.jsonl \
    --output_file /scratch/m000152/comet/generic-resource-type/lm_classify_next_token/datacite_2024_classified_qwen_qwen3_4b.jsonl \
    --batch_size 1000
    