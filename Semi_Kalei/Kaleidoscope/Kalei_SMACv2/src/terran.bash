export CUDA_VISIBLE_DEVICES=0
python src/main.py --config=K24_qmix_rnn_1R3 --env-config=sc2_gen_zerg --exp-config=terran_5v5_10M_s0 seed=0
python src/main.py --config=K24_qmix_rnn_1R3 --env-config=sc2_gen_zerg --exp-config=terran_5v5_10M_s0 seed=1
