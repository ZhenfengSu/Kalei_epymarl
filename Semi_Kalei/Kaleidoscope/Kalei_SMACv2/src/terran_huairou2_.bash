export CUDA_VISIBLE_DEVICES=1

python src/main.py --config=Kalei_qmix_rnn_1R3 --env-config=sc2_gen_zerg --exp-config=terran_5v5_10M_s0
python src/main.py --config=Kalei_qmix_rnn_1R3 --env-config=sc2_gen_zerg --exp-config=terran_5v5_10M_s0