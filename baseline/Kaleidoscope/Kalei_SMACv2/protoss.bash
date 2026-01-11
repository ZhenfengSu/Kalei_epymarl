export CUDA_VISIBLE_DEVICES=4
python src/main.py --config=qmix_rnn_1R3 --env-config=sc2_gen_zerg --exp-config=protoss_5v5_10M_s0
python src/main.py --config=qmix_rnn_1R3 --env-config=sc2_gen_zerg --exp-config=protoss_5v5_10M_s0


python src/main.py --config=nops_qmix_rnn_1R3 --env-config=sc2_gen_zerg --exp-config=protoss_5v5_10M_s0
python src/main.py --config=nops_qmix_rnn_1R3 --env-config=sc2_gen_zerg --exp-config=protoss_5v5_10M_s0

python src/main.py --config=Kalei_qmix_rnn_1R3 --env-config=sc2_gen_zerg --exp-config=protoss_5v5_10M_s0
python src/main.py --config=Kalei_qmix_rnn_1R3 --env-config=sc2_gen_zerg --exp-config=protoss_5v5_10M_s0
python src/main.py --config=Kalei_qmix_rnn_1R3 --env-config=sc2_gen_zerg --exp-config=protoss_5v5_10M_s0