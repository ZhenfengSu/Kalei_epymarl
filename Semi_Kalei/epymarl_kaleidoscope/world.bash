export CUDA_VISIBLE_DEVICES=0
# python src/main.py --config=K24_nq --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-world-comm-v3" seed=0
python src/main.py --config=K24_nq --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-world-comm-v3" seed=1

