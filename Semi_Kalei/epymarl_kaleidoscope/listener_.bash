export CUDA_VISIBLE_DEVICES=1


python src/main.py --config=K24_nq --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-speaker-listener-v4" seed=2
python src/main.py --config=K24_nq --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-speaker-listener-v4" seed=3