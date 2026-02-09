#!/usr/bin/env python3
"""
Run script for Kaleidoscope algorithm on MPE environments.

This script runs the Kalei_qmix_rnn_1R3 algorithm configuration on
Multi-Particle Environment (MPE) tasks.

Usage:
    python run_kalei_mpe.py --env-name simple_spread_v3 --config Kalei_qmix_rnn_1R3 --env-config mpe

Example MPE environments:
    - simple_spread_v3: Cooperative navigation
    - simple_tag_v3: Predator-prey
    - simple_adversary_v3: Adversarial task
    - simple_crypto_v3: Communication
    - simple_push_v3: Cooperative pushing
    - simple_reference_v3: Cooperative reference
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import run as _run


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Kaleidoscope on MPE environments')

    # Environment configuration
    parser.add_argument('--env-name', type=str, default='simple_spread_v3',
                        help='MPE environment name (e.g., simple_spread_v3, simple_tag_v3)')
    parser.add_argument('--env-config', type=str, default='mpe',
                        help='Environment config file (default: mpe)')
    parser.add_argument('--config', type=str, default='Kalei_qmix_rnn_1R3',
                        help='Algorithm config file (default: Kalei_qmix_rnn_1R3)')

    # Training parameters
    parser.add_argument('--t-max', type=int, default=None,
                        help='Maximum number of training timesteps')
    parser.add_argument('--batch-size-run', type=int, default=None,
                        help='Batch size for collecting episodes')

    # Logging
    parser.add_argument('--log-dir', type=str, default='results/mpe',
                        help='Directory to save logs and models')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Name for this experiment run')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    # Other common parameters
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--load-ckpt', type=str, default=None,
                        help='Load checkpoint from path')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model instead of training')

    args = parser.parse_args()

    # Build sys.argv for the main function
    sys.argv = ['main.py']

    if args.env_name:
        sys.argv.extend(['--env-config', args.env_config, '--env-args', f'key={args.env_name}'])

    sys.argv.extend(['--config', args.config])

    if args.t_max is not None:
        sys.argv.extend(['--env-args', f't_max={args.t_max}'])

    if args.batch_size_run is not None:
        sys.argv.extend(['--batch-size-run', str(args.batch_size_run)])

    if args.log_dir:
        sys.argv.extend(['--log-dir', args.log_dir])

    if args.experiment_name:
        sys.argv.extend(['--experiment-name', args.experiment_name])
    else:
        # Auto-generate experiment name
        exp_name = f'{args.config}_{args.env_name}'
        sys.argv.extend(['--experiment-name', exp_name])

    if args.device:
        sys.argv.extend(['--device', args.device])

    if args.seed is not None:
        sys.argv.extend(['--seed', str(args.seed)])

    if args.load_ckpt:
        sys.argv.extend(['--load-ckpt', args.load_ckpt])

    if args.evaluate:
        sys.argv.append('--evaluate')

    return args


def main():
    """Main function."""
    args = parse_args()

    # Create log directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), 'results', 'mpe')
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 80)
    print("Kaleidoscope Algorithm on MPE Environments")
    print("=" * 80)
    print(f"Environment: {args.env_name}")
    print(f"Algorithm Config: {args.config}")
    print(f"Log Directory: {log_dir}")
    print("=" * 80)
    print()

    # Run the training/evaluation
    try:
        _run()
        print("\n" + "=" * 80)
        print("Run completed successfully!")
        print("=" * 80)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
