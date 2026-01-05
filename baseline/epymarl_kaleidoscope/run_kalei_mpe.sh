#!/bin/bash
# Run Kaleidoscope algorithm on MPE environments
# Usage: ./run_kalei_mpe.sh [ENV_NAME]

# Default environment if none provided
ENV_NAME=${1:-simple_spread_v3}

echo "========================================"
echo "Running Kaleidoscope on MPE Environment"
echo "========================================"
echo "Environment: $ENV_NAME"
echo "========================================"
echo ""

# Run with Python directly using the main entry point
python src/main.py \
    --env-config gymma \
    --env-args key=$ENV_NAME time_limit=25 common_reward=True reward_scalarisation=sum \
    --config Kalei_qmix_rnn_1R3 \
    --experiment-name Kalei_${ENV_NAME} \
    --log-dir results/mpe \
    --device cuda

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"
