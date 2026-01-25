# 在simple_spread环境上训练                                                                                                                               
python src/main.py --config=SNP_qmix_rnn_1R3 --env-config=gymma env_args.env_name=MPE.envs.simple_spread_v3                                               
                                                                                                                                                        
# 在其他MPE环境上训练                                                                                                                                     
python src/main.py --config=SNP_qmix_rnn_1R3 --env-config=gymma env_args.env_name=MPE.envs.simple_push_v3                                                 
python src/main.py --config=SNP_qmix_rnn_1R3 --env-config=gymma env_args.env_name=MPE.envs.simple_tag_v3    